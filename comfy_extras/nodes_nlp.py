"""Core text-pipeline nodes (reform step 8): corpus -> vocabulary -> token tensors.

The text half of the language-model pipeline, operating on the ``VOCAB`` slot
type and plain ``TENSOR`` index streams:

* ``Vocab Build``        - corpus string -> frozen vocabulary (+ its size)
* ``Text Encode``        - text -> 1-D long tensor of token indices
* ``Text Decode``        - token indices -> text
* ``Sliding Window``     - index stream -> (context, next token) sample pairs

(``comfy_extras/nodes_text.py`` is a different module - the pre-existing
``Save Text`` output node; this file owns the language-model text pipeline.)

Everything here is deterministic and stateless: the vocabulary is a frozen
``{token: index}`` mapping with ``<unk>`` at index 0 (built in frequency order,
so the same corpus always yields the same vocabulary), and the token tensors
are ordinary ``torch.long`` tensors any tensor node can inspect. The model half
(``Network & Layers/Training``: Language Model Embedding / Transformer Block /
Build / Train / Forward / Generate) consumes exactly these tensors.
"""

import torch
from typing_extensions import override

from comfy import lm_protocol as mp
from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Text"

#: Default corpus - the pangram every English typing class knows. Long enough
#: to build a real vocabulary and to yield sliding-window samples at the
#: default window, short enough to train on within a teaching session.
DEFAULT_CORPUS = "the quick brown fox jumps over the lazy dog"

#: Default text handed to ``Text Encode``; every word occurs in the default
#: corpus, so the default graph runs without touching a widget.
DEFAULT_TEXT = "the quick brown"


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _as_vocab(value) -> mp.Vocab:
    """Validate a VOCAB slot payload, with a readable error for other types."""
    if isinstance(value, mp.Vocab):
        return value
    raise ValueError(
        f"this slot needs a Vocab (link it from a Vocab Build node); got "
        f"{type(value).__name__}."
    )


def _as_index_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    """Coerce a tensor to a plain 1-D long index stream, or raise."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"'{name}' needs a tensor of token indices; got {type(value).__name__}.")
    flat = value.reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"'{name}' is empty; there is nothing to process.")
    if flat.dtype.is_floating_point:
        rounded = flat.round()
        if not bool(torch.equal(flat, rounded)):
            _warn(f"'{name}' holds non-integer values; they are truncated to token indices.")
        flat = rounded
    return flat.to(dtype=torch.long)


class TextVocabBuild(io.ComfyNode):
    """Build a token vocabulary from a corpus string.

    What: tokenise ``corpus`` at the chosen level, count the tokens and keep
          the ones seen at least ``min_freq`` times, ordered by frequency
          (descending) then alphabetically - fully deterministic, so the same
          corpus always builds the same vocabulary. ``<unk>`` is reserved at
          index 0: encoding an unseen token maps to it instead of raising.
          Character level (default) keeps punctuation and spelling, the d2l
          time-machine recipe; word level gives fewer, larger tokens.
    In:   corpus (STRING) - the training text; any length, any language.
          level (COMBO) - ``char`` (default) or ``word`` tokenisation.
          min_freq (INT) - tokens rarer than this are dropped from the
          vocabulary (they still encode, as ``<unk>``).
    Out:  vocab (VOCAB) - the frozen mapping; feed it to Text Encode / Text
          Decode and to the Generate node's vocabulary slot.
          vocab_size (INT) - number of tokens including ``<unk>``; wire it
          into Language Model Embedding's ``vocab`` link.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextVocabBuild",
            display_name="Vocab Build",
            category=CATEGORY,
            description="Builds a deterministic token vocabulary (with <unk> at index 0) from a corpus string, at char or word level.",
            search_aliases=["vocab", "vocabulary", "tokenize", "tokenizer", "corpus", "text", "nlp"],
            inputs=[
                io.String.Input(
                    "corpus",
                    default=DEFAULT_CORPUS,
                    multiline=True,
                    placeholder="the training text",
                    tooltip="The corpus the vocabulary is counted from; the same corpus always builds the same vocabulary.",
                ),
                io.Combo.Input(
                    "level",
                    options=list(mp.LEVEL_OPTIONS),
                    default="char",
                    tooltip="Tokenisation level: char keeps every character, word splits on whitespace.",
                ),
                io.Int.Input(
                    "min_freq",
                    default=1,
                    min=1,
                    max=1000000,
                    step=1,
                    tooltip="Tokens rarer than this many occurrences are dropped (they encode as <unk>).",
                ),
            ],
            outputs=[
                io.Vocab.Output(display_name="vocab"),
                io.Int.Output(display_name="vocab_size"),
            ],
        )

    @classmethod
    def execute(cls, corpus: str = DEFAULT_CORPUS, level: str = "char", min_freq: int = 1) -> io.NodeOutput:
        text = "" if corpus is None else str(corpus)
        if not text.strip():
            raise ValueError(
                "the 'corpus' is empty; a vocabulary needs at least one token."
            )
        vocab = mp.build_vocab(text, level=level, min_freq=min_freq)
        return io.NodeOutput(vocab, vocab.size)


class TextEncode(io.ComfyNode):
    """Turn text into a 1-D tensor of token indices.

    What: split ``text`` exactly the way the vocabulary's level prescribes and
          map every token to its index; tokens the vocabulary has never seen
          become the ``<unk>`` index (0) instead of raising - the encoding of
          an unknown word is a first-class result, not an error.
    In:   vocab (VOCAB) - link from Vocab Build.
          text (STRING) - the text to encode; may contain unknown tokens.
    Out:  ids (TENSOR) - 1-D ``torch.long``, one entry per token; empty text
          encodes to an empty tensor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextEncode",
            display_name="Text Encode",
            category=CATEGORY,
            description="Encodes text into a 1-D long tensor of token indices through a vocabulary; unknown tokens become <unk>.",
            search_aliases=["encode", "tokenize", "ids", "indices", "text", "vocab", "nlp"],
            inputs=[
                io.Vocab.Input("vocab", tooltip="The vocabulary to encode with; link from Vocab Build."),
                io.String.Input(
                    "text",
                    default=DEFAULT_TEXT,
                    multiline=True,
                    placeholder="the text to encode",
                    tooltip="The text to encode; tokens missing from the vocabulary become <unk>.",
                ),
            ],
            outputs=[io.Tensor.Output(display_name="ids")],
        )

    @classmethod
    def execute(cls, vocab, text: str = DEFAULT_TEXT) -> io.NodeOutput:
        table = _as_vocab(vocab)
        indices = table.encode("" if text is None else str(text))
        return io.NodeOutput(torch.tensor(indices, dtype=torch.long))


class TextDecode(io.ComfyNode):
    """Turn a tensor of token indices back into text.

    What: the inverse of Text Encode - map every index to its token and join
          them the way the vocabulary's level prescribes (characters are
          concatenated directly, words joined with spaces). A 2-D tensor is
          decoded row by row and the rows are joined with newlines, so a batch
          of generated sequences prints as a list. Out-of-range indices and
          the ``<unk>`` token itself decode to ``<unk>``.
    In:   vocab (VOCAB) - link from Vocab Build.
          ids (TENSOR) - 1-D stream or 2-D ``(batch, seq_len)`` batch of token
          indices; floats are truncated, anything else must be integer-like.
    Out:  text (STRING) - the decoded text.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextDecode",
            display_name="Text Decode",
            category=CATEGORY,
            description="Decodes token indices (1-D stream or 2-D batch) back into text through a vocabulary.",
            search_aliases=["decode", "detokenize", "text", "ids", "indices", "vocab", "nlp"],
            inputs=[
                io.Vocab.Input("vocab", tooltip="The vocabulary to decode with; link from Vocab Build."),
                io.Tensor.Input(
                    "ids",
                    tooltip="Token indices: a 1-D stream or a 2-D (batch, seq_len) batch.",
                ),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, vocab, ids: torch.Tensor) -> io.NodeOutput:
        table = _as_vocab(vocab)
        if not isinstance(ids, torch.Tensor):
            raise ValueError(
                f"'ids' needs a tensor of token indices; got {type(ids).__name__}."
            )
        flat = _as_index_tensor(ids, "ids")
        if ids.dim() <= 1:
            return io.NodeOutput(table.decode(flat.tolist()))
        width = int(ids.shape[-1])
        rows = [
            flat[row * width : (row + 1) * width].tolist()
            for row in range(ids.numel() // max(1, width))
        ]
        return io.NodeOutput("\n".join(table.decode(row) for row in rows))


class TextSlidingWindow(io.ComfyNode):
    """Cut an index stream into (context, next-token) training samples.

    What: the next-token dataset of a language model. A window of ``window``
          tokens slides over the stream one position at a time; every position
          contributes one sample - the window as input ``x`` and the token
          right after it as target ``y``. With the default window 4 and the
          default corpus this yields one sample per position, exactly the
          d2l ``seq_data_iter_sequential`` dataset in its simplest (stride 1)
          form.
    In:   ids (TENSOR) - 1-D long tensor of token indices (link from Text
          Encode); floats are truncated, 2-D inputs are flattened.
          window (INT) - context length per sample (default 4).
    Out:  x (TENSOR) - ``(samples, window)`` long; the contexts.
          y (TENSOR) - ``(samples,)`` long; the next token of each context.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextSlidingWindow",
            display_name="Sliding Window",
            category=CATEGORY,
            description="Cuts a token-index stream into (context, next token) sample pairs: the next-token dataset of a language model.",
            search_aliases=["sliding window", "dataset", "next token", "samples", "language model", "ngram", "text"],
            inputs=[
                io.Tensor.Input(
                    "ids",
                    tooltip="1-D stream of token indices; link from Text Encode.",
                ),
                io.Int.Input(
                    "window",
                    default=4,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Context length per sample; must be shorter than the stream.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="x"),
                io.Tensor.Output(display_name="y"),
            ],
        )

    @classmethod
    def execute(cls, ids: torch.Tensor, window: int = 4) -> io.NodeOutput:
        flat = _as_index_tensor(ids, "ids")
        size = max(1, int(window))
        contexts: list[list[int]] = []
        targets: list[int] = []
        for context, target in mp.iter_windows(flat.tolist(), size):
            contexts.append(context)
            targets.append(target)
        if not contexts:
            raise ValueError(
                f"the stream holds {flat.numel()} token(s); a window of {size} "
                f"needs at least window + 1 = {size + 1} to cut a single sample."
            )
        return io.NodeOutput(
            torch.tensor(contexts, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


#: Every node this module registers, in node-library order.
TEXT_PIPELINE_NODES: list[type[io.ComfyNode]] = [
    TextVocabBuild,
    TextEncode,
    TextDecode,
    TextSlidingWindow,
]


class TextPipelineExtension(ComfyExtension):
    """Registers the core text-pipeline node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(TEXT_PIPELINE_NODES)


async def comfy_entrypoint() -> TextPipelineExtension:
    return TextPipelineExtension()
