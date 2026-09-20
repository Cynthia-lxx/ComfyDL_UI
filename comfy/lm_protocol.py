"""Language-model protocol layer for the dehydrated ComfyUI build (reform step 8).

ComfyUI executes a prompt inside ``torch.inference_mode()`` (``execution.py``),
so an autograd graph cannot cross a node boundary. A language model therefore
cannot be assembled as a pure TENSOR dataflow: by the time the tensors reach a
trainer the graph is gone. This module provides the *object* half of the story,
the same way ``comfy/training_protocol.py`` provides the training closure:

* :class:`Vocab` / :func:`build_vocab` - a frozen token-to-index mapping with
  ``<unk>`` handling, shared by every text node (build / encode / decode);
* :class:`EmbeddingSpec` / :class:`TransformerBlockSpec` - frozen, device-free
  blueprints of the model layers. A chain of them travels on the ``MODELSPEC``
  slot, so the node graph states the *structure* while a single build node
  materialises it;
* :class:`LanguageModel` / :func:`build_model` - the materialised
  ``nn.Module`` (GPT-style pre-LN stack with an optional sinusoidal position
  encoding and an output projection back to vocabulary size), initialised
  deterministically from a seed;
* :func:`generate_tokens` - the autoregressive next-token loop used by the
  Generate node, greedy or temperature sampled, driven by a local
  ``torch.Generator`` so the process RNG stays untouched.

The parameter naming follows the ``nn.Module`` / ``state_dict`` convention
(``embedding.weight``, ``blocks.0.attn.q_proj.weight``, ``head.weight``, ...),
so a trained model is addressable, inspectable and persisted verbatim by the
``Save Language Model`` node, which stores the spec chain and the vocabulary
in the same ``.safetensors`` file (the JSON helpers at the bottom of this
module do the value-level encoding).

Only ``torch`` and the sibling ``training_protocol`` are imported here: the
module must stay importable in the dehydrated build, which has neither
``comfy.ldm`` nor ``comfy.lora``.

Design decisions
----------------
* Pre-LN blocks (``x + attn(LN(x))``) with a final LayerNorm before the output
  projection, matching the core ``TransformerEncoderBlock`` node of reform
  step 7 and the modern (GPT-style) recipe. Post-LN needs a warm-up that a
  300-step teaching run does not have; pre-LN trains from step one.
* The attention math is spelled out explicitly (softmax(QK^T / sqrt(d) + mask))
  with the SDPA boolean-mask convention (``True`` = attend) and the finite
  dtype minimum for blocked positions - identical to the core attention nodes,
  so the numbers agree across the two paths.
* Dropout inside the module uses the *process* RNG. The trainer is responsible
  for seeding it via :func:`training_protocol.seeded_rng`, which restores the
  previous state afterwards; generation switches the module to ``eval()`` and
  never draws a random number.
* Everything a spec carries is a plain value (ints, strings, floats): a spec
  chain is safe for ComfyUI to cache and compare, and holds no tensors until
  :func:`build_model` runs.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from typing import Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.training_protocol import seeded_rng

#: The reserved unknown token; always index 0, so an unseen token never raises.
UNK_TOKEN = "<unk>"

#: Tokenisation levels offered by the vocab nodes.
LEVEL_OPTIONS: tuple[str, ...] = ("char", "word")

#: Activations offered inside a transformer block's feed-forward half.
ACTIVATION_OPTIONS: tuple[str, ...] = ("relu", "gelu")


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Language Model] {message}")


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Vocab:
    """A frozen token-to-index mapping with ``<unk>`` handling.

    What: the payload of the ``VOCAB`` slot. Index 0 is always ``<unk>``; the
          remaining tokens follow the corpus order (frequency descending, then
          alphabetically), so the same corpus always builds the same vocab.
    In:   tokens - the index-ordered vocabulary tuple, ``<unk>`` first.
          level - ``"char"`` or ``"word"``; decides how ``encode`` splits text
          and how ``decode`` joins tokens back.
    Out:  call :meth:`encode` / :meth:`decode` for text <-> index conversion;
          :attr:`size` is the vocabulary size a model spec needs.
    """

    tokens: tuple[str, ...]
    level: str = "char"

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("Vocab needs at least the <unk> token.")
        if self.tokens[0] != UNK_TOKEN:
            raise ValueError(f"Vocab must start with {UNK_TOKEN!r} at index 0.")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocab tokens must be unique.")
        object.__setattr__(
            self,
            "_token_to_idx",
            {token: index for index, token in enumerate(self.tokens)},
        )

    @property
    def size(self) -> int:
        """Number of tokens, including ``<unk>``."""
        return len(self.tokens)

    @property
    def token_to_idx(self) -> Mapping[str, int]:
        """The read-only token-to-index mapping."""
        return getattr(self, "_token_to_idx")

    def _split(self, text: str) -> list[str]:
        return list(text) if self.level != "word" else text.split()

    def _join(self, tokens: Sequence[str]) -> str:
        return "".join(tokens) if self.level != "word" else " ".join(tokens)

    def encode(self, text: str) -> list[int]:
        """Map text to token indices; unseen tokens become the ``<unk>`` index."""
        table = self.token_to_idx
        unk = table[UNK_TOKEN]
        return [table.get(token, unk) for token in self._split(str(text))]

    def decode(self, indices: Sequence[int]) -> str:
        """Map indices back to text; out-of-range indices become ``<unk>``."""
        tokens = [
            self.tokens[index] if 0 <= int(index) < self.size else UNK_TOKEN
            for index in indices
        ]
        return self._join(tokens)

    def summary(self) -> str:
        """One-line description for tooltips and log messages."""
        preview = ", ".join(repr(token) for token in self.tokens[1:6])
        more = "" if self.size <= 6 else f", +{self.size - 6} more"
        return f"{self.size} {self.level} token(s): {preview}{more}"


def _normalise_level(level: str) -> str:
    """Return a valid level, falling back to ``char`` with a warning."""
    text = "" if level is None else str(level).strip().lower()
    if text in LEVEL_OPTIONS:
        return text
    _warn(f"level {level!r} is not one of {LEVEL_OPTIONS}; using 'char'.")
    return "char"


def build_vocab(text: str, level: str = "char", min_freq: int = 1) -> Vocab:
    """Build a :class:`Vocab` from a corpus string.

    What: tokenise ``text`` at the chosen level, count the tokens, keep those
          seen at least ``min_freq`` times and order them by frequency
          (descending) then alphabetically - fully deterministic. ``<unk>``
          is always index 0.
    In:   text - the corpus; whitespace is collapsed for word level.
          level - ``"char"`` (default) or ``"word"``.
          min_freq - tokens rarer than this are dropped from the vocabulary
          (they still encode, as ``<unk>``).
    Out:  a frozen :class:`Vocab`.
    """
    normalised = _normalise_level(level)
    corpus = " ".join(str(text or "").split()) if normalised == "word" else str(text or "")
    tokens = list(corpus) if normalised == "char" else corpus.split()
    counts = Counter(token for token in tokens if token)
    min_count = max(1, int(min_freq))
    ranked = sorted(
        (token for token, count in counts.items() if count >= min_count),
        key=lambda token: (-counts[token], token),
    )
    return Vocab((UNK_TOKEN, *ranked), level=normalised)


# --------------------------------------------------------------------------- #
# Spec chain
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class EmbeddingSpec:
    """Blueprint of the input half: token embedding (+ position encoding).

    What: the first link of every spec chain. Declares the vocabulary size and
          the model width; ``include_position`` adds the fixed sinusoidal
          position encoding (no parameters) after the embedding, the standard
          recipe for a stack without recurrence.
    """

    vocab_size: int
    d_model: int
    include_position: bool = True


@dataclasses.dataclass(frozen=True)
class TransformerBlockSpec:
    """Blueprint of one pre-LN transformer block.

    What: ``x + attn(LN(x))`` followed by ``x + ffn(LN(x))``, the block shape
          used by the core ``TransformerEncoderBlock`` node and by GPT-style
          stacks. Attention is self-attention with a causal mask; the FFN is a
          two-layer MLP with ``d_ffn`` hidden units.
    """

    d_model: int
    num_heads: int
    d_ffn: int
    activation: str = "relu"
    dropout: float = 0.0


def validate_spec_chain(
    chain: Sequence,
) -> tuple[EmbeddingSpec, tuple[TransformerBlockSpec, ...]]:
    """Check a spec chain and return it as ``(embedding, blocks)``.

    In:  chain - the ``MODELSPEC`` payload: an iterable whose first element is
         an :class:`EmbeddingSpec` and whose remaining elements are
         :class:`TransformerBlockSpec` instances.
    Out: the validated ``(embedding, blocks)`` pair.

    Raises:
        ValueError: when the chain is empty, starts with the wrong spec type,
            mixes widths, uses a bad activation, or a width that the head count
            does not divide.
    """
    specs = tuple(chain or ())
    if not specs or not isinstance(specs[0], EmbeddingSpec):
        raise ValueError(
            "a spec chain must start with a LanguageModelEmbedding spec "
            "(the token-embedding link); got: "
            + ("empty chain" if not specs else type(specs[0]).__name__)
        )
    embedding = specs[0]
    blocks = tuple(specs[1:])
    for index, spec in enumerate(blocks):
        if not isinstance(spec, TransformerBlockSpec):
            raise ValueError(
                f"spec link {index + 1} is a {type(spec).__name__}; only "
                "transformer-block specs may follow the embedding spec."
            )
        if spec.d_model != embedding.d_model:
            raise ValueError(
                f"spec link {index + 1} has width {spec.d_model}, but the "
                f"embedding spec declares {embedding.d_model}; every link must "
                "share the model width."
            )
        if spec.d_model % spec.num_heads != 0:
            raise ValueError(
                f"spec link {index + 1}: width {spec.d_model} is not divisible "
                f"by num_heads {spec.num_heads}."
            )
        if spec.activation not in ACTIVATION_OPTIONS:
            raise ValueError(
                f"spec link {index + 1}: activation {spec.activation!r} is not "
                f"one of {ACTIVATION_OPTIONS}."
            )
    return embedding, blocks


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def positional_encoding(
    length: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The fixed sinusoidal position encoding, shape ``(length, width)``.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/width))`` and
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/width))`` - the classic "Attention Is
    All You Need" table, with no learnable parameters.
    """
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    fraction = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    table = torch.zeros(length, width, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * fraction)
    table[:, 1::2] = torch.cos(position * fraction[: (width + 1) // 2])
    return table.to(dtype=dtype)


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean causal mask, shape ``(length, length)``, ``True`` = attend.

    Position ``i`` may attend to positions ``<= i`` (lower triangle), the
    SDPA-boolean convention used by the core attention nodes.
    """
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


class _SelfAttention(nn.Module):
    """Explicit multi-head self-attention with the causal mask wired in."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, length, d_model = x.shape
        heads, d_head = self.num_heads, self.d_head

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(n, length, heads, d_head).transpose(1, 2)

        q = split(self.q_proj(x))
        k = split(self.k_proj(x))
        v = split(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
        # True = attend; blocked positions get the finite dtype minimum so a
        # fully masked row softmaxes uniformly instead of producing NaN.
        blocked = ~mask
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        if self.training and 0.0 < self.dropout < 1.0:
            attn = F.dropout(attn, p=self.dropout, training=True)
        merged = torch.matmul(attn, v).transpose(1, 2).reshape(n, length, d_model)
        return self.out_proj(merged)


class _TransformerBlock(nn.Module):
    """Pre-LN block: ``x + attn(LN(x))`` then ``x + ffn(LN(x))``."""

    def __init__(self, spec: TransformerBlockSpec) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.d_model)
        self.attn = _SelfAttention(spec.d_model, spec.num_heads, spec.dropout)
        self.norm2 = nn.LayerNorm(spec.d_model)
        self.ffn1 = nn.Linear(spec.d_model, spec.d_ffn)
        self.ffn2 = nn.Linear(spec.d_ffn, spec.d_model)
        self.activation = spec.activation
        self.dropout = spec.dropout
        self.ffn_activation = F.relu if spec.activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(x)
        x = x + self.attn(hidden, mask)
        hidden = self.norm2(x)
        if self.training and 0.0 < self.dropout < 1.0:
            hidden = F.dropout(hidden, p=self.dropout, training=True)
        hidden = self.ffn_activation(self.ffn1(hidden))
        if self.training and 0.0 < self.dropout < 1.0:
            hidden = F.dropout(hidden, p=self.dropout, training=True)
        return x + self.ffn2(hidden)


class LanguageModel(nn.Module):
    """A causal transformer language model materialised from a spec chain.

    What: ``embedding (+ position) -> N x pre-LN transformer block -> LayerNorm
          -> linear head back to vocabulary size``. ``forward`` maps a batch of
          token-index sequences to next-token logits; every position only sees
          earlier positions (causal mask), so one forward pass gives the loss
          of *every* next-token prediction in the sequence at once.
    In:   vocab_size / d_model / include_position from the embedding spec, one
          :class:`_TransformerBlock` per block spec.
    Out:  ``forward(ids)`` with ``ids`` of shape ``(N, L)`` returns logits of
          shape ``(N, L, vocab_size)``. Parameters follow the ``state_dict``
          naming convention (``embedding.weight``, ``blocks.0.attn.q_proj.weight``,
          ``head.weight``, ...).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        include_position: bool = True,
        block_specs: Sequence[TransformerBlockSpec] = (),
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.include_position = bool(include_position)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList(_TransformerBlock(spec) for spec in block_specs)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2 or ids.dtype not in (torch.long, torch.int64):
            raise ValueError(
                f"the model needs a (batch, seq_len) long tensor of token "
                f"indices; got shape {tuple(ids.shape)}, dtype {ids.dtype}."
            )
        length = ids.shape[1]
        x = self.embedding(ids)
        if self.include_position:
            x = x + positional_encoding(
                length, self.d_model, device=x.device, dtype=x.dtype
            )
        mask = causal_mask(length, device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def next_token_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits of the *next* token after each sequence: ``(N, vocab)``."""
        return self.forward(ids)[:, -1, :]


def build_model(chain: Sequence, seed: int = 0) -> LanguageModel:
    """Materialise a spec chain into a :class:`LanguageModel`, seeded.

    What: the bridge from the ``MODELSPEC`` slot to the ``NNMODEL`` slot. The
          initialisation runs inside :func:`training_protocol.seeded_rng`, so
          the same chain and seed always produce the same weights and the
          process RNG is restored afterwards. Linear weights use Xavier
          uniform, biases start at zero and the embedding at N(0, 0.01).
    In:   chain - the validated spec chain (see :func:`validate_spec_chain`).
          seed - the initialisation seed.
    Out:  a fresh :class:`LanguageModel` in ``train`` mode.
    """
    embedding, blocks = validate_spec_chain(chain)
    with seeded_rng(seed):
        model = LanguageModel(
            embedding.vocab_size,
            embedding.d_model,
            include_position=embedding.include_position,
            block_specs=blocks,
        )
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
        model.train()
    return model


def parameter_count(model: nn.Module) -> int:
    """Total number of trainable parameters, for tooltips and summaries."""
    return sum(parameter.numel() for parameter in model.parameters())


# --------------------------------------------------------------------------- #
# Autoregressive generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_tokens(
    model: LanguageModel,
    prefix_ids: Sequence[int],
    num_tokens: int,
    temperature: float = 1.0,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Autoregressive next-token loop on a local ``torch.Generator``.

    What: repeatedly take ``next_token_logits`` of the sequence so far, pick
          the next token and append it, ``num_tokens`` times.
    In:   model - a trained (or untrained) :class:`LanguageModel`; switched to
          ``eval()`` for the duration so dropout never fires.
          prefix_ids - the starting token indices (the prompt).
          num_tokens - how many tokens to generate.
          temperature - ``<= 0`` (or exactly 0) means greedy argmax; otherwise
          logits are divided by the temperature and sampled - ``1.0`` is
          unchanged, ``< 1`` sharpens, ``> 1`` flattens.
          seed - seed of the local generator; the same seed and inputs always
          reproduce the same continuation.
          progress - optional ``progress(done, total)`` callback invoked once
          per generated token. Injected rather than imported so this protocol
          module stays free of any UI/ComfyUI machinery.
    Out:  a 1-D long tensor: the prefix followed by the generated tokens.
    """
    ids = [int(index) for index in prefix_ids]
    count = max(0, int(num_tokens))
    if count == 0:
        return torch.tensor(ids, dtype=torch.long)

    was_training = model.training
    model.eval()
    try:
        sequence = torch.tensor([ids] if ids else [[0]], dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        for step in range(count):
            logits = model.next_token_logits(sequence)
            if temperature is None or float(temperature) <= 0.0:
                choice = int(torch.argmax(logits[0]).item())
            else:
                probabilities = torch.softmax(logits[0] / float(temperature), dim=-1)
                choice = int(
                    torch.multinomial(probabilities, 1, generator=generator).item()
                )
            sequence = torch.cat(
                [sequence, torch.tensor([[choice]], dtype=torch.long)], dim=1
            )
            if progress is not None:
                progress(step + 1, count)
        return sequence.reshape(-1)
    finally:
        model.train(was_training)


def iter_windows(
    ids: Sequence[int], window: int
) -> Iterator[tuple[list[int], int]]:
    """Yield every ``(context, next_token)`` pair of a sliding window.

    In:  ids - the token-index stream; window - the context length.
    Out: ``([ids[i], ..., ids[i+window-1]], ids[i+window])`` for every ``i``
         with a full window and a target; nothing when the stream is too short.
    """
    size = max(1, int(window))
    stream = [int(index) for index in ids]
    for start in range(0, len(stream) - size):
        yield stream[start : start + size], stream[start + size]


# --------------------------------------------------------------------------- #
# JSON persistence helpers (Save / Load Language Model)
# --------------------------------------------------------------------------- #
#: Version tag stored in the ``.safetensors`` metadata of a saved language
#: model. Bumping it is the only sanctioned way to change the layout below.
LM_FILE_FORMAT = "comfydl-lm-1"

#: JSON keys of the metadata entries written next to the weights.
LM_META_FORMAT = "format"
LM_META_SPEC = "spec"
LM_META_VOCAB = "vocab"


def spec_chain_from_model(model: LanguageModel) -> tuple:
    """Recover the spec chain of a *built* model from its module structure.

    What: the inverse of :func:`build_model` at the blueprint level - every
          value a spec carries is still readable off the module (widths off
          the linear layers, heads off the attention, activation / dropout
          off the block), so a model can be saved together with the exact
          blueprint that reproduces it, without the graph re-stating the
          chain. The result is validated by construction and always accepted
          by :func:`spec_chain_to_json`.
    In:   model - a :class:`LanguageModel` (trained or not).
    Out:  the ``(embedding, *blocks)`` spec chain of the model.
    """
    blocks = tuple(
        TransformerBlockSpec(
            d_model=block.attn.q_proj.in_features,
            num_heads=block.attn.num_heads,
            d_ffn=block.ffn1.out_features,
            activation=block.activation,
            dropout=block.dropout,
        )
        for block in model.blocks
    )
    return (
        EmbeddingSpec(model.vocab_size, model.d_model, model.include_position),
        *blocks,
    )


def vocab_to_json(vocab: Vocab) -> str:
    """Encode a :class:`Vocab` as a compact JSON string.

    What: the value half of ``VOCAB`` persistence - tokens and level only,
          everything else about a vocab is derived. The encoding is used by
          ``Save Language Model`` so a loaded model can talk text again
          without re-running ``Vocab Build``.
    In:   vocab - the vocabulary to encode.
    Out:  a JSON string ``{"tokens": [...], "level": "char"|"word"}``.
    """
    return json.dumps(
        {"tokens": list(vocab.tokens), "level": vocab.level},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def vocab_from_json(text: str) -> Vocab:
    """Rebuild a :class:`Vocab` from the JSON of :func:`vocab_to_json`.

    Raises:
        ValueError: when the text is not valid JSON, not an object, or fails
            the ``Vocab`` invariants (the ``Vocab`` constructor re-validates
            ``<unk>`` at index 0 and uniqueness, so a tampered file is caught).
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the vocab entry is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
        raise ValueError(
            "the vocab entry must be a JSON object with a 'tokens' list; got: "
            f"{type(payload).__name__}."
        )
    tokens = payload["tokens"]
    if not all(isinstance(token, str) for token in tokens):
        raise ValueError("every vocab token must be a string.")
    level = payload.get("level", "char")
    return Vocab(tuple(tokens), level=_normalise_level(level))


def spec_chain_to_json(chain: Sequence) -> str:
    """Encode a spec chain as a compact JSON string.

    What: the blueprint half of persistence - the embedding link plus every
          transformer-block link, all plain values, so a loaded model can be
          rebuilt to exactly the shape its weights expect (then filled by
          ``load_state_dict``).
    In:   chain - the ``MODELSPEC`` payload; validated first
          (:func:`validate_spec_chain`), so an invalid chain cannot be saved.
    Out:  a JSON string ``{"embedding": {...}, "blocks": [{...}, ...]}``.
    """
    embedding, blocks = validate_spec_chain(chain)
    return json.dumps(
        {
            "embedding": {
                "vocab_size": embedding.vocab_size,
                "d_model": embedding.d_model,
                "include_position": embedding.include_position,
            },
            "blocks": [
                {
                    "d_model": block.d_model,
                    "num_heads": block.num_heads,
                    "d_ffn": block.d_ffn,
                    "activation": block.activation,
                    "dropout": block.dropout,
                }
                for block in blocks
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def spec_chain_from_json(text: str) -> tuple:
    """Rebuild a spec chain from the JSON of :func:`spec_chain_to_json`.

    Out: the ``(embedding, *blocks)`` tuple, ready for :func:`build_model`.

    Raises:
        ValueError: when the text is not valid JSON, not an object, misses the
            embedding entry, or describes a chain that fails validation (mixed
            widths, bad activation, heads not dividing the width, ...).
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the spec entry is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("embedding"), dict):
        raise ValueError(
            "the spec entry must be a JSON object with an 'embedding' object; "
            f"got: {type(payload).__name__}."
        )
    head = payload["embedding"]
    blocks_payload = payload.get("blocks", [])
    if not isinstance(blocks_payload, list):
        raise ValueError("the spec 'blocks' entry must be a list.")

    embedding = EmbeddingSpec(
        vocab_size=int(head["vocab_size"]),
        d_model=int(head["d_model"]),
        include_position=bool(head.get("include_position", True)),
    )
    blocks = [
        TransformerBlockSpec(
            d_model=int(block["d_model"]),
            num_heads=int(block["num_heads"]),
            d_ffn=int(block["d_ffn"]),
            activation=str(block["activation"]),
            dropout=float(block.get("dropout", 0.0)),
        )
        for block in blocks_payload
    ]
    return (embedding, *blocks)  # validated by build_model via validate_spec_chain
