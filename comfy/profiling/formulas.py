"""The DL memory formula library of ``comfy.profiling``.

Every number here is an *upper bound* of what the corresponding node keeps
alive at its peak, derived from the explicit math the nodes execute (see
``comfy_extras/nodes_lm.py`` / ``comfy/lm_protocol.py`` for the ground truth
these formulas mirror):

* training peak = parameters P + gradients G (= P) + optimizer state
  (AdamW/Adam 2P, RMSprop P, SGD 0) + an optional early-stop snapshot (P)
  + every intermediate activation autograd has to keep for backward;
* inference peak = P + the transient activations of one forward pass.

Each :class:`MemoryItem` also carries ``single_bytes`` - the largest single
allocation inside the item - because "one tensor larger than free memory"
fails even when the total would fit (the 13.3 GB k_proj output of the
profiling golden case is exactly that situation).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

from comfy.profiling.shapes import BlockInfo, FLOAT32_BYTES, INT64_BYTES

#: Optimizer -> number of state tensors per parameter (AdamW/Adam keep the
#: first and second moment, RMSprop the running average, SGD nothing).
OPTIMIZER_STATE_MULTIPLIERS = {
    "AdamW": 2,
    "Adam": 2,
    "SGD": 0,
    "RMSprop": 1,
}


@dataclasses.dataclass(frozen=True)
class MemoryItem:
    """One line of a node's memory breakdown.

    ``label`` is a human-readable English phrase (the frontend may replace
    it through ``kind``); ``single_bytes`` is the largest single allocation
    inside the item (equal to ``bytes`` for single-tensor items).
    """

    label: str
    bytes: int
    single_bytes: Optional[int] = None
    kind: str = "misc"
    approx: bool = False

    def __post_init__(self) -> None:
        if self.single_bytes is None:
            object.__setattr__(self, "single_bytes", self.bytes)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "bytes": int(self.bytes),
            "single_bytes": int(self.single_bytes or self.bytes),
            "approx": bool(self.approx),
        }


@dataclasses.dataclass(frozen=True)
class Breakdown:
    """The full memory breakdown of one node execution."""

    items: List[MemoryItem] = dataclasses.field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(item.bytes for item in self.items)

    @property
    def largest_single(self) -> Optional[MemoryItem]:
        best: Optional[MemoryItem] = None
        for item in self.items:
            if best is None or (item.single_bytes or 0) > (best.single_bytes or 0):
                best = item
        return best


def _optimiser_multiplier(optimizer: str) -> int:
    name = str(optimizer).strip().lower()
    for known, multiplier in OPTIMIZER_STATE_MULTIPLIERS.items():
        if known.lower() == name:
            return multiplier
    return 2  # unknown optimizers are assumed stateful (safe upper bound)


def lm_parameter_count(
    vocab_size: Optional[int],
    d_model: Optional[int],
    blocks: Sequence[BlockInfo],
) -> Optional[int]:
    """Exact trainable parameter count of the spec-chain language model."""
    if vocab_size is None or d_model is None:
        return None
    params = vocab_size * d_model  # embedding
    for block in blocks:
        if block.d_model is None or block.d_ffn is None:
            return None
        width, hidden = block.d_model, block.d_ffn
        params += 4 * (width * width + width)  # q/k/v/out projections
        params += 2 * (2 * width)  # norm1 + norm2
        params += width * hidden + hidden  # ffn1
        params += hidden * width + width  # ffn2
    params += 2 * d_model  # final norm
    params += d_model * vocab_size + vocab_size  # output head
    return params


def lm_training(
    batch: int,
    dataset_samples: int,
    window: int,
    vocab_size: int,
    d_model: int,
    blocks: Sequence[BlockInfo],
    optimizer: str = "AdamW",
    early_stop: bool = False,
) -> Breakdown:
    """Upper bound of the Language Model Train node's peak memory."""
    items: List[MemoryItem] = []
    b, t, e = int(batch), int(window), int(d_model)
    params = lm_parameter_count(vocab_size, d_model, blocks)
    if params:
        items.append(MemoryItem("parameters (float32)", params * 4, kind="params"))
        items.append(MemoryItem("gradients", params * 4, kind="grads"))
        multiplier = _optimiser_multiplier(optimizer)
        if multiplier:
            items.append(
                MemoryItem(
                    f"optimizer state ({optimizer}, {multiplier}x params)",
                    multiplier * params * 4,
                    params * 4,
                    kind="optimizer_state",
                )
            )
        if early_stop:
            items.append(
                MemoryItem("early-stop best-state snapshot", params * 4, kind="early_stop_snapshot")
            )
    # The full dataset stays resident for the whole run: contexts (S, T) and
    # targets (S, T) int64 plus the (S,) next-token tensor.
    samples = int(dataset_samples)
    items.append(
        MemoryItem(
            "dataset tensors (x / y / targets)",
            2 * samples * t * INT64_BYTES + samples * INT64_BYTES,
            samples * t * INT64_BYTES,
            kind="dataset",
        )
    )
    # Activation graph kept for backward - every intermediate of every block.
    items.append(
        MemoryItem(
            "embedding + position outputs", 2 * b * t * e * 4, b * t * e * 4, kind="embed_out"
        )
    )
    for index, block in enumerate(blocks, 1):
        if block.d_model is None or block.d_ffn is None:
            continue
        heads, hidden = max(1, int(block.num_heads)), int(block.d_ffn)
        scores = b * heads * t * t * 4
        items.append(
            MemoryItem(
                f"block {index}: layer-norm outputs", 2 * b * t * e * 4, b * t * e * 4, kind="ln_out"
            )
        )
        items.append(
            MemoryItem(
                f"block {index}: attention q/k/v outputs",
                3 * b * t * e * 4,
                b * t * e * 4,
                kind="attn_qkv",
            )
        )
        items.append(
            MemoryItem(f"block {index}: attention scores (BxHxTxT)", scores, scores, kind="attn_scores")
        )
        items.append(
            MemoryItem(
                f"block {index}: attention weights (softmax)", scores, scores, kind="attn_weights"
            )
        )
        items.append(
            MemoryItem(
                f"block {index}: attention merged + out_proj + residual",
                3 * b * t * e * 4,
                b * t * e * 4,
                kind="attn_out_residual",
            )
        )
        items.append(
            MemoryItem(
                f"block {index}: FFN hidden (linear + activation)",
                2 * b * t * hidden * 4,
                b * t * hidden * 4,
                kind="ffn_hidden",
            )
        )
        items.append(
            MemoryItem(
                f"block {index}: FFN output + residual",
                2 * b * t * e * 4,
                b * t * e * 4,
                kind="ffn_out_residual",
            )
        )
    items.append(MemoryItem("final norm output", b * t * e * 4, kind="ln_out"))
    items.append(
        MemoryItem(f"logits (B x T x V)", b * t * vocab_size * 4, b * t * vocab_size * 4, kind="logits")
    )
    items.append(
        MemoryItem(
            "loss buffer (log-softmax)",
            b * t * vocab_size * 4,
            b * t * vocab_size * 4,
            kind="loss_buffer",
        )
    )
    items.append(MemoryItem("batch targets (int64)", b * t * INT64_BYTES, kind="inputs"))
    return Breakdown(items)


def lm_inference(
    batch: int,
    length: int,
    vocab_size: int,
    d_model: int,
    blocks: Sequence[BlockInfo],
) -> Breakdown:
    """Upper bound of one ``no_grad`` forward pass (Forward / Generate)."""
    items: List[MemoryItem] = []
    b, t, e = int(batch), int(length), int(d_model)
    params = lm_parameter_count(vocab_size, d_model, blocks)
    if params:
        items.append(MemoryItem("parameters (float32)", params * 4, kind="params"))
    # Under no_grad only transients exist; the full backward set is a safe
    # upper bound and keeps the two paths comparable.
    items.append(MemoryItem("embedding + position outputs", 2 * b * t * e * 4, b * t * e * 4, kind="embed_out"))
    for index, block in enumerate(blocks, 1):
        if block.d_model is None or block.d_ffn is None:
            continue
        heads, hidden = max(1, int(block.num_heads)), int(block.d_ffn)
        scores = b * heads * t * t * 4
        items.append(MemoryItem(f"block {index}: attention q/k/v outputs", 3 * b * t * e * 4, b * t * e * 4, kind="attn_qkv"))
        items.append(MemoryItem(f"block {index}: attention scores (BxHxTxT)", scores, scores, kind="attn_scores"))
        items.append(MemoryItem(f"block {index}: FFN hidden", 2 * b * t * hidden * 4, b * t * hidden * 4, kind="ffn_hidden"))
    items.append(MemoryItem("logits (B x T x V)", b * t * vocab_size * 4, b * t * vocab_size * 4, kind="logits"))
    return Breakdown(items)


def parse_hidden_widths(text: Optional[str]) -> List[int]:
    """Parse the Training Loop ``hidden`` widget ("16,8" -> [16, 8])."""
    if text is None:
        return []
    widths: List[int] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            width = int(float(part))
        except ValueError:
            continue
        if width > 0:
            widths.append(width)
    return widths


def mlp_parameter_count(
    in_features: Optional[int], hidden: Sequence[int], out_features: Optional[int]
) -> Optional[int]:
    """Exact parameter count of the Training Loop's stacked-MLP."""
    if in_features is None or out_features is None:
        return None
    layers = [int(in_features)] + [int(w) for w in hidden] + [int(out_features)]
    return sum(
        layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1)
    )


def mlp_training(
    batch: int,
    dataset_samples: int,
    in_features: int,
    hidden: Sequence[int],
    out_features: int,
    optimizer: str = "AdamW",
    early_stop: bool = False,
) -> Breakdown:
    """Upper bound of the Training Loop node's peak memory."""
    items: List[MemoryItem] = []
    b = int(batch)
    params = mlp_parameter_count(in_features, hidden, out_features)
    if params:
        items.append(MemoryItem("parameters (float32)", params * 4, kind="params"))
        items.append(MemoryItem("gradients", params * 4, kind="grads"))
        multiplier = _optimiser_multiplier(optimizer)
        if multiplier:
            items.append(
                MemoryItem(
                    f"optimizer state ({optimizer}, {multiplier}x params)",
                    multiplier * params * 4,
                    params * 4,
                    kind="optimizer_state",
                )
            )
        if early_stop:
            items.append(MemoryItem("early-stop best-state snapshot", params * 4, kind="early_stop_snapshot"))
    samples = int(dataset_samples)
    items.append(
        MemoryItem(
            "dataset tensors (x / y)",
            samples * int(in_features) * 4 + samples * int(out_features) * 4,
            samples * int(in_features) * 4,
            kind="dataset",
        )
    )
    widths = [int(w) for w in hidden] + [int(out_features)]
    activation_bytes = sum(2 * b * width * 4 for width in widths)
    items.append(
        MemoryItem(
            "MLP activations (upper bound)",
            activation_bytes,
            b * max(widths) * 4 if widths else 0,
            kind="mlp_activations",
        )
    )
    items.append(MemoryItem("batch targets", b * int(out_features) * 4, kind="inputs"))
    return Breakdown(items)
