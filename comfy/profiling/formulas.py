"""The DL memory & compute formula library of ``comfy.profiling``.

Every memory number here is an *upper bound* of what the corresponding node
keeps alive at its peak, derived from the explicit math the nodes execute
(see ``comfy_extras/nodes_lm.py`` / ``comfy/lm_protocol.py`` for the ground
truth these formulas mirror):

* training peak = parameters P + gradients G (= P) + optimizer state
  (AdamW/Adam 2P, RMSprop P, SGD 0) + an optional early-stop snapshot (P)
  + every intermediate activation autograd has to keep for backward;
* inference peak = P + the transient activations of one forward pass.

Each :class:`MemoryItem` also carries ``single_bytes`` - the largest single
allocation inside the item - because "one tensor larger than free memory"
fails even when the total would fit (the 13.3 GB k_proj output of the
profiling golden case is exactly that situation).

The M2 compute ledger (:class:`FlopsItem`) counts *matmul* FLOPs only
(one multiply-accumulate = 2 FLOPs): embedding gathers, softmax,
layer-norm and other elementwise work are excluded - next to the GEMMs
they are noise, and the standard "6ND" rule (Kaplan et al. 2020) makes the
same simplification. The M3 extension adds the convolution ledger
(:func:`conv2d_flops`, kind ``conv``) with the same MAC-counting convention.
Training counts ``3x`` the forward FLOPs per step
(backward of a GEMM block costs about twice its forward). These are
*compute amounts*, not times: no throughput claim is made (that needs the
measured calibration of a later milestone).
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


# --------------------------------------------------------------------------- #
# Convolutional / visual family helpers (M3): the static algebra the tensor,
# detection, segmentation and image nodes execute.
# --------------------------------------------------------------------------- #


def conv2d_out_dim(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Spatial output size of one ``nn.Conv2d`` axis (torch convention).

    ``floor((size + 2*padding - kernel) / stride) + 1`` - the exact rule
    ``torch.nn`` uses for a stride>1 / padded convolution without dilation.
    """
    return (int(size) + 2 * int(padding) - int(kernel)) // max(1, int(stride)) + 1


def conv_param_count(
    in_channels: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    bias: bool = True,
    groups: int = 1,
) -> int:
    """Exact trainable parameter count of one ``nn.Conv2d``."""
    group = max(1, int(groups))
    weight = int(out_channels) * (int(in_channels) // group) * int(kernel_h) * int(kernel_w)
    return weight + (int(out_channels) if bias else 0)


def image_bytes(
    batch: Optional[int],
    height: Optional[int],
    width: Optional[int],
    channels: Optional[int] = 3,
    itemsize: int = FLOAT32_BYTES,
) -> Optional[int]:
    """Bytes of a ComfyUI ``IMAGE`` / ``MASK`` tensor (NHWC layout).

    Any unknown dimension propagates as ``None`` - the caller decides
    whether an assumption fills it in.
    """
    if batch is None or height is None or width is None or channels is None:
        return None
    return int(batch) * int(height) * int(width) * int(channels) * int(itemsize)


def tensor_item(
    label: str,
    nbytes: Optional[int],
    kind: str = "misc",
    approx: bool = False,
) -> MemoryItem:
    """A one-tensor memory line (``single_bytes`` equals ``bytes``).

    The factory every tensor-shaped estimator uses, so an unknown size
    degrades to a 0-byte line instead of raising.
    """
    return MemoryItem(label, int(nbytes or 0), kind=kind, approx=approx)


def rnn_param_count(
    num_inputs: int,
    num_hiddens: int,
    gates: int = 1,
    bias: bool = True,
    num_layers: int = 1,
) -> int:
    """Exact parameter count of a torch RNN/GRU *stack*.

    Layer 0 maps ``I -> H``; every further layer maps ``H -> H`` (torch's
    own layout, so a stacked GRU is not simply L times layer 0). One layer
    holds ``gates x (W_ih: In x H + W_hh: H x H)`` plus, with bias,
    ``2 x gates x H`` (``b_ih`` + ``b_hh`` - torch keeps both even when
    only one is used). ``gates``: vanilla RNN 1, GRU 3, LSTM 4.
    """
    i, h = int(num_inputs), int(num_hiddens)
    bias_count = 2 * int(gates) * h if bias else 0
    first = int(gates) * (i * h + h * h) + bias_count
    deeper = int(gates) * (h * h + h * h) + bias_count
    return first + (max(1, int(num_layers)) - 1) * deeper


def gru_param_count(num_inputs: int, num_hiddens: int, num_layers: int = 1) -> int:
    """Exact parameter count of a torch GRU stack (3 gates, both biases)."""
    return rnn_param_count(num_inputs, num_hiddens, gates=3, bias=True, num_layers=num_layers)


def dense_param_count(
    in_features: Optional[int], out_features: Optional[int], bias: bool = True
) -> Optional[int]:
    """Parameter count of one ``nn.Linear``; ``None`` while a dim is unknown.

    A ``nn.LazyLinear`` has no weights until its first forward, which is
    why the width may legitimately be missing here.
    """
    if in_features is None or out_features is None:
        return None
    return int(in_features) * int(out_features) + (int(out_features) if bias else 0)


def mha_param_count(in_features: int, num_hiddens: int, bias: bool = False) -> int:
    """The four projections (q/k/v/output) of d2l's multi-head attention.

    The head count does not change the parameter count: every head gets a
    slice of the same matrices (d2l builds no per-head weights).
    """
    return 4 * (int(in_features) * int(num_hiddens) + (int(num_hiddens) if bias else 0))


def lenet_param_count(
    num_classes: int, in_channels: int = 1, spatial: int = 28
) -> int:
    """d2l LeNet-5 with its Lazy layers resolved for one input size.

    conv1 (5x5, padding 2) keeps the size, conv2 does not, and both pools
    halve it, so the flatten width is ``16 * (((spatial // 2) - 4) // 2)^2``.
    The default (1 x 28 x 28) is the MNIST shape d2l uses.
    """
    side = ((int(spatial) // 2) - 4) // 2
    if side <= 0:
        return 0
    flat = 16 * side * side
    conv1 = int(in_channels) * 6 * 25 + 6
    conv2 = 6 * 16 * 25 + 16
    dense1 = flat * 120 + 120
    dense2 = 120 * 84 + 84
    head = 84 * int(num_classes) + int(num_classes)
    return conv1 + conv2 + dense1 + dense2 + head


def resnet18_param_count(in_channels: int, num_classes: int) -> int:
    """d2l's ResNet-18 (small stem, no max-pool) parameter count.

    One residual block is conv1 (3x3) + conv2 (3x3) + two batch-norms, plus
    an optional 1x1 shortcut; the stride changes shapes, never the counts.
    """
    def conv(cin: int, cout: int, kernel: int) -> int:
        return cin * cout * kernel * kernel + cout

    def residual(cin: int, cout: int, shortcut: bool) -> int:
        params = conv(cin, cout, 3) + conv(cout, cout, 3) + 4 * cout
        return params + conv(cin, cout, 1) if shortcut else params

    params = conv(int(in_channels), 64, 3) + 2 * 64  # stem conv + batch norm
    for cin, cout, first in ((64, 64, True), (64, 128, False),
                             (128, 256, False), (256, 512, False)):
        params += residual(cin, cout, not first) + residual(cout, cout, False)
    return params + 512 * int(num_classes) + int(num_classes)


def positional_encoding_bytes(
    max_len: int, num_hiddens: int, itemsize: int = FLOAT32_BYTES
) -> int:
    """Bytes of d2l's sine/cosine table ``(1, max_len, num_hiddens)``.

    It is a plain tensor attribute, not a registered buffer: zero trainable
    parameters, but very real memory.
    """
    return int(max_len) * int(num_hiddens) * int(itemsize)


def transformer_block_param_count(
    num_hiddens: int, ffn_num_hiddens: int, bias: bool = False
) -> int:
    """One transformer-encoder block of d2l's layout.

    Four attention projections + ReLU-FFN in/out + the two layer-norms of
    addnorm1/addnorm2, all derived exactly from the widgets.
    """
    h, f = int(num_hiddens), int(ffn_num_hiddens)
    attention = mha_param_count(h, h, bias)
    ffn = 2 * h * f + f + h  # dense1 + dense2, both biased
    norms = 2 * 2 * h  # weight + bias of each layer-norm
    return attention + ffn + norms


# --------------------------------------------------------------------------- #
# Compute ledger (M2): FLOPs of the same workloads the memory side covers.
# --------------------------------------------------------------------------- #

#: Training cost of one GEMM-dominant step = forward + backward, and the
#: backward costs about twice the forward -> the standard "3x forward" rule
#: (the 6ND estimate of Kaplan et al. 2020 uses the same factor).
TRAINING_FLOPS_MULTIPLIER = 3

#: The "6ND" empirical rule: training a dense transformer costs about
#: 6 x parameters x tokens processed. Used as a cross-check of the
#: per-layer sums in the tests, never as the primary source.
SIX_ND_COEFFICIENT = 6


@dataclasses.dataclass(frozen=True)
class FlopsItem:
    """One line of a node's compute breakdown.

    ``flops`` counts multiply-accumulate as 2 (the GEMM convention).
    ``kind`` is one of ``attn`` / ``ffn`` / ``logits`` / ``other`` - the
    frontend colours the rows by it; ``approx`` marks numbers derived
    through a rule of thumb (the 3x training factor, the no-cache
    generation bound) rather than counted per GEMM.
    """

    label: str
    flops: int
    kind: str = "other"
    approx: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "flops": int(self.flops),
            "approx": bool(self.approx),
        }


def lm_forward_flops(
    batch: int,
    seq: int,
    vocab_size: int,
    d_model: int,
    blocks: Sequence[BlockInfo],
) -> List[FlopsItem]:
    """Matmul FLOPs of one forward pass over ``(batch, seq)`` inputs.

    Per block: q/k/v/out projections are four ``E x E`` GEMMs
    (``8*B*T*E^2``); ``QK^T`` and ``A @ V`` are two ``T x T x d_head``
    GEMMs each (``4*B*T^2*E`` - independent of the head count, since
    ``d_head = E / H``); the FFN is two ``E <-> F`` GEMMs (``4*B*T*E*F``).
    The output head adds ``2*B*T*E*V``. Embedding lookups, softmax,
    layer-norm and residuals are elementwise and excluded.
    """
    b, t, e = int(batch), int(seq), int(d_model)
    items: List[FlopsItem] = []
    for index, block in enumerate(blocks, 1):
        if block.d_model is None or block.d_ffn is None:
            continue
        width, hidden = int(block.d_model), int(block.d_ffn)
        projections = 4 * 2 * b * t * width * width
        scores = 4 * b * t * t * width
        ffn = 2 * 2 * b * t * width * hidden
        items.append(
            FlopsItem(
                f"block {index}: attention (q/k/v/o + scores)", projections + scores, kind="attn"
            )
        )
        items.append(
            FlopsItem(f"block {index}: feed-forward (2 linears)", ffn, kind="ffn")
        )
    items.append(FlopsItem("output head (E -> V)", 2 * b * t * e * int(vocab_size), kind="logits"))
    return items


def lm_training_flops(
    batch: int,
    seq: int,
    vocab_size: int,
    d_model: int,
    blocks: Sequence[BlockInfo],
    steps: int,
) -> List[FlopsItem]:
    """Matmul FLOPs of a whole training run: (forward + backward) x steps.

    Every forward item is scaled by ``3 x steps`` (the backward-pass rule
    above); the optimizer update is elementwise and excluded. ``steps``
    below 1 counts as 1, mirroring the nodes' clamping.
    """
    count = max(1, int(steps))
    scale = TRAINING_FLOPS_MULTIPLIER * count
    return [
        FlopsItem(
            f"{item.label} x {scale} ({count} steps)",
            item.flops * scale,
            kind=item.kind,
            approx=True,
        )
        for item in lm_forward_flops(batch, seq, vocab_size, d_model, blocks)
    ]


def lm_generate_flops(
    prefix_length: int,
    num_tokens: int,
    vocab_size: int,
    d_model: int,
    blocks: Sequence[BlockInfo],
) -> List[FlopsItem]:
    """Matmul FLOPs of the autoregressive generation loop.

    ``lm_protocol.generate_tokens`` keeps no KV cache: every step re-runs
    the whole forward over the sequence so far, so the total is the sum of
    forwards over lengths ``L, L+1, ..., L+count-1``. For absurdly large
    token counts the loop falls back to the max-length upper bound.
    """
    count = max(0, int(num_tokens))
    start = max(1, int(prefix_length))
    if count == 0:
        return []
    if count * max(1, len(blocks)) > 1_000_000:
        return [
            FlopsItem(
                f"{item.label} x {count} (generation, bound)",
                item.flops * count,
                kind=item.kind,
                approx=True,
            )
            for item in lm_forward_flops(1, start + count - 1, vocab_size, d_model, blocks)
        ]
    acc: dict = {}
    for step in range(count):
        for item in lm_forward_flops(1, start + step, vocab_size, d_model, blocks):
            key = (item.label, item.kind)
            acc[key] = acc.get(key, 0) + item.flops
    return [
        FlopsItem(label, flops, kind=kind, approx=True)
        for (label, kind), flops in acc.items()
    ]


def mlp_forward_flops(
    batch: int,
    in_features: int,
    hidden: Sequence[int],
    out_features: int,
) -> List[FlopsItem]:
    """Matmul FLOPs of one forward pass of the Training Loop's stacked MLP."""
    layers = [int(in_features)] + [int(w) for w in hidden] + [int(out_features)]
    flops = sum(2 * int(batch) * layers[i] * layers[i + 1] for i in range(len(layers) - 1))
    return [FlopsItem("stacked-MLP linears", flops, kind="ffn")]


def mlp_training_flops(
    batch: int,
    in_features: int,
    hidden: Sequence[int],
    out_features: int,
    steps: int,
) -> List[FlopsItem]:
    """Matmul FLOPs of a whole Training Loop run (3x forward x steps)."""
    count = max(1, int(steps))
    scale = TRAINING_FLOPS_MULTIPLIER * count
    return [
        FlopsItem(
            f"{item.label} x {scale} ({count} steps)",
            item.flops * scale,
            kind=item.kind,
            approx=True,
        )
        for item in mlp_forward_flops(batch, in_features, hidden, out_features)
    ]


def conv2d_flops(
    batch: int,
    out_h: int,
    out_w: int,
    out_channels: int,
    in_channels: int,
    kernel_h: int,
    kernel_w: int,
    groups: int = 1,
) -> int:
    """FLOPs of one 2D convolution (one multiply-accumulate = 2, the GEMM
    convention the rest of the ledger uses).

    ``2 x B x Hout x Wout x Cout x (Cin / groups) x Kh x Kw``; the bias add
    and the elementwise activation around the conv are excluded, exactly the
    simplification the matmul ledger makes. Kind: ``conv``.
    """
    group = max(1, int(groups))
    return (
        2
        * int(batch)
        * int(out_h)
        * int(out_w)
        * int(out_channels)
        * (int(in_channels) // group)
        * int(kernel_h)
        * int(kernel_w)
    )


def six_nd_estimate(params, tokens) -> int:
    """The 6ND rule of thumb: ``6 x N x D`` for N parameters, D tokens."""
    return SIX_ND_COEFFICIENT * int(params or 0) * int(tokens or 0)
