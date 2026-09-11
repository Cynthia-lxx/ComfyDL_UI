"""Normalization and regularization nodes plus the train/eval state helpers.

Provides the ``Normalization`` and ``Regularization`` groups of ``Network & Layers``
- the families next to ``Basic`` and ``Activation`` - plus the two small ``Training``
nodes that tell them whether a graph is training or inferring:

* activation normalizers      - ``NormalizationBatchNorm``,
  ``NormalizationInstanceNorm``, ``NormalizationLayerNorm``,
  ``NormalizationGroupNorm``, ``NormalizationRMSNorm``
* weight re-parameterizations - ``NormalizationWeightNorm``,
  ``NormalizationSpectralNorm``
* regularizers                - ``RegularizationDropout``
* train/eval state            - ``TrainingMode``, ``TrainingRunStats``

Design notes
------------
* **One node per operation, rank adaptive.** ``F.batch_norm`` and ``F.instance_norm``
  already accept any ``(N, C, ...)`` tensor, so a single node covers the 1d/2d/3d
  flavours: the shape decides which dimensions the statistics are taken over and
  there is no rank branch to keep in sync.
* **Stateless.** ``weight`` / ``bias`` (gamma / beta) are wired in through ``TENSOR``
  slots instead of being created inside a node, nothing is cached between two
  executions, and a wired ``running_mean`` / ``running_var`` is never written to,
  because in ComfyUI the same tensor may be shared by several nodes.
* **Randomness is seeded, never implicit.** ``RegularizationDropout`` draws its mask
  from a ``torch.Generator`` built for the tensor's own device and seeded by its
  ``seed`` widget: the same seed reproduces the same mask bit for bit, and the
  process-wide RNG of the host is left untouched for the other nodes to use.
* **The train/eval switch travels through a real link.** ``TrainingMode`` publishes
  ``"train"`` / ``"eval"`` as a STRING that is wired into the ``mode`` slot of the
  normalization nodes. Only a link can do this reliably: ComfyUI derives a node's
  cache signature from its own inputs *and its ancestors' inputs*, whereas a hidden
  ``io.Hidden.prompt`` handshake is neither part of the signature nor even readable
  while the signature is computed (``execution.py`` calls ``get_input_data`` with
  ``dynprompt=None`` there, so the hidden prompt is an empty dict). A prompt-reading
  handshake would therefore silently reuse stale outputs after the dropdown is flipped.
* **Running statistics are numbers, not tensors.** ``TrainingRunStats`` keeps them in
  editable widgets, because widgets are what survives in a saved workflow, and emits
  them as 1-D tensors so they can be wired into the normalization nodes, where a
  single value is broadcast to every channel. ``BatchNorm`` and ``InstanceNorm`` also
  *export* the per-channel ``mean`` / ``var`` they normalized with, and
  ``TrainingRunStats`` accepts those through its own optional TENSOR slots: a link
  wins over the typed text, so the train-to-eval hand-off is a matter of two wires
  and the numbers never have to be copied by hand.

The nodes preserve the input dtype/device, never round-trip through host memory, and
unparsable widget text falls back to a documented default with a printed warning, so
a typo never breaks a running workflow. The file lives in ``comfy_extras``, which
registers the nodes as core nodes.
"""

import re

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

NORMALIZATION_CATEGORY = "Network & Layers/Normalization"
REGULARIZATION_CATEGORY = "Network & Layers/Regularization"
TRAINING_CATEGORY = "Network & Layers/Training"

MODE_TRAIN = "train"
MODE_EVAL = "eval"
MODE_OPTIONS = (MODE_TRAIN, MODE_EVAL)


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _split_numbers(text: str) -> list[str]:
    """Split a widget string into number tokens.

    Accepts comma, semicolon and whitespace, including the full-width variants, so a
    list typed on a Chinese IME (``"0.1，0.2"``) still works. Nothing is validated
    here: every caller decides what a token must look like.
    """
    if text is None:
        return []
    normalized = str(text).replace("，", ",").replace("；", ",").replace(";", ",")
    return [part for part in re.split(r"[,\s]+", normalized) if part]


def _parse_shape(text: str) -> tuple[int, ...] | None:
    """Parse a list of ints such as ``"8,16"``.

    Returns:
        The parsed sizes, or ``None`` when the string is empty or contains a
        non-integer entry; callers treat ``None`` as "not a usable shape".
    """
    parts = _split_numbers(text)
    if not parts:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def _parse_values(text: str) -> tuple[float, ...] | None:
    """Parse a list of floats such as ``"0.1,0.2"``.

    Returns:
        The parsed numbers, or ``None`` when the string is empty or contains a
        non-numeric entry.
    """
    parts = _split_numbers(text)
    if not parts:
        return None
    try:
        return tuple(float(part) for part in parts)
    except ValueError:
        return None


def _clamp_dim(dim: int, rank: int) -> int:
    """Clamp ``dim`` into ``[-rank, rank - 1]`` so a bad widget never crashes."""
    return max(-rank, min(rank - 1, dim))


def _normalize_mode(mode: str | None) -> str:
    """Return ``"train"`` or ``"eval"`` for a value arriving through the mode slot.

    An unparsable value (e.g. a hand-typed string linked into the slot) falls back to
    ``"train"`` - the PyTorch default - with a printed warning.
    """
    if mode is None:
        return MODE_TRAIN
    text = str(mode).strip().lower()
    if text in MODE_OPTIONS:
        return text
    _warn(f"mode={mode!r} is neither 'train' nor 'eval'; using '{MODE_TRAIN}'.")
    return MODE_TRAIN


def _resolve_normalized_shape(text: str, shape: torch.Size) -> tuple[int, ...]:
    """Return the *actual* trailing dimensions a LayerNorm/RMSNorm normalizes.

    ``"last"`` (the default) and ``"-1"`` both mean "the last dimension". A list of
    positive sizes such as ``"8,16"`` means the last two dimensions; the numbers are
    only used to decide *how many* dimensions to take, because a stale widget value
    (the tensor changed shape since it was typed) must not make the node fail.
    Anything that cannot be read as a trailing shape falls back to the last dimension
    with a printed warning.
    """
    declared = "" if text is None else str(text).strip()
    if declared == "" or declared.lower() in ("last", "-1"):
        return tuple(shape[-1:])
    dims = _parse_shape(declared)
    if dims is None or any(size < 1 for size in dims) or len(dims) > len(shape):
        _warn(f"normalized_shape={text!r} is not a valid trailing shape; normalizing the last dimension instead.")
        return tuple(shape[-1:])
    actual = tuple(shape[-len(dims):])
    if actual != dims:
        _warn(f"normalized_shape={dims} does not match the trailing dimensions {actual}; using {actual}.")
    return actual


def _broadcast_stat(
    values: torch.Tensor | None,
    channels: int,
    tensor: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    """Turn a statistics slot into a ``(channels,)`` tensor of ``tensor``'s dtype/device.

    A single value is broadcast to every channel, which is what ``TrainingRunStats``
    emits with its default widgets.

    Returns:
        The statistics to use, or ``None`` when nothing is wired in or the value does
        not fit the tensor; the caller then falls back to the batch statistics instead
        of aborting the graph.
    """
    if values is None:
        return None
    flat = values.reshape(-1).to(dtype=tensor.dtype, device=tensor.device)
    if flat.numel() == 1:
        return flat.expand(channels)
    if flat.numel() == channels:
        return flat
    _warn(f"{name} holds {flat.numel()} value(s) but the tensor has {channels} channel(s); ignoring it.")
    return None


def _stats_tensor(text: str, name: str, fallback: float) -> torch.Tensor:
    """Parse a statistics widget (``"0.1,0.2"``) into a 1-D ``float32`` tensor.

    The consumer casts the result to its own dtype/device, so the tensor is always
    built on the CPU. Unparsable text falls back to ``fallback`` with a warning.
    """
    values = _parse_values(text)
    if values is None:
        _warn(f"{name}={text!r} is not a comma separated list of numbers; using {fallback}.")
        values = (fallback,)
    return torch.tensor(values, dtype=torch.float32)


def _linked_stats(
    linked: torch.Tensor | None,
    text: str,
    name: str,
    fallback: float,
) -> torch.Tensor:
    """Prefer a linked statistics tensor, else parse the widget text.

    A link wins because it is the *measured* value of the run that produced it, while
    the widget only holds whatever was typed when the graph was saved. The linked
    tensor is flattened to 1-D and cast to ``float32``; the consumer broadcasts a
    single value to every channel and casts to its own dtype/device anyway, so the
    round-trip through the default dtype costs nothing.
    """
    if linked is not None:
        return linked.reshape(-1).to(dtype=torch.float32)
    return _stats_tensor(text, name, fallback)


def _unit_vector(values: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale ``values`` to unit length, replacing an all-zero vector by ``1/sqrt(n)``.

    The degeneracy test stays on the device (``torch.where``) so a power iteration
    never has to synchronise with the host.
    """
    lengths = torch.linalg.vector_norm(values)
    usable = lengths > eps
    lengths = torch.where(usable, lengths, torch.linalg.vector_norm(torch.ones_like(values)))
    values = torch.where(usable, values, torch.ones_like(values))
    return values / lengths.clamp_min(eps)


def _matrix_vector(provided: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    """Pick a starting u/v for the power iteration.

    ``fallback`` is the row/column sum of the weight matrix, i.e. exactly one power
    iteration away from an all-ones vector: a deterministic, data dependent start.
    Random numbers are not an option here - they would make the output unpredictable
    and defeat ComfyUI's caching. A linked vector wins whenever it is given.
    """
    if provided is None:
        return fallback
    return provided.reshape(-1).to(dtype=fallback.dtype, device=fallback.device)


_DEFAULT_MODE_TOOLTIP = "Link the mode output of a Training Mode node: 'train' normalizes with the statistics of this call, 'eval' with the statistics wired below. Unconnected means 'train'."


def _mode_input(tooltip: str = _DEFAULT_MODE_TOOLTIP) -> io.Input:
    """The ``mode`` slot shared by the normalization and regularization nodes.

    It is a plain socket (``force_input``) rather than a dropdown, because the value
    is meant to arrive through a link, and STRING is the type ``TrainingMode``
    outputs, so the two connect without any type juggling.

    Args:
        tooltip: Hover text; callers whose semantics differ from "normalizes with"
            pass their own wording, everything else keeps the default.
    """
    return io.String.Input(
        "mode",
        default=MODE_TRAIN,
        optional=True,
        force_input=True,
        tooltip=tooltip,
    )


def _normalization_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    search_aliases: list[str] | None = None,
    outputs: list | None = None,
) -> io.Schema:
    """Build the schema shared by the activation normalizers.

    They all emit ``output`` as their first TENSOR; only the inputs differ, so callers
    pass the already-built input list.

    Args:
        node_id: Globally unique, core-safe node id (``Normalization`` prefix).
        display_name: Short name shown in the node library (e.g. ``BatchNorm``).
        description: Tooltip shown when hovering over the node.
        inputs: The node's inputs, in declaration order (slots and widgets).
        search_aliases: Extra search keywords for the node library.
        outputs: The node's outputs, in declaration order. ``None`` means the single
            ``output`` TENSOR every normalizer has; BatchNorm and InstanceNorm pass
            their two extra statistics outputs explicitly.

    Returns:
        The ``io.Schema`` describing the node.
    """
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category=NORMALIZATION_CATEGORY,
        description=description,
        search_aliases=search_aliases,
        inputs=list(inputs),
        outputs=list(outputs) if outputs is not None else [io.Tensor.Output(display_name="output")],
    )


# The statistics both BatchNorm and InstanceNorm add after ``output``: the per-channel
# mean/variance this call normalized with. Shared so the two nodes cannot drift apart.
_STATS_OUTPUTS = [
    io.Tensor.Output(display_name="mean"),
    io.Tensor.Output(display_name="var"),
]


class TrainingMode(io.ComfyNode):
    """Declares whether the graph currently trains or infers, for every node linked to it.

    What: a single dropdown that fans out to the whole graph. The node computes
          nothing itself - it publishes the choice as a STRING, and the normalization
          nodes read it through their ``mode`` slot. Because the link is part of
          ComfyUI's cache signature, flipping the dropdown invalidates every consumer,
          so the graph genuinely re-runs instead of replaying cached outputs.
    In:   mode (COMBO) - ``train`` normalizes with the statistics of the current call
          and ignores the wired ``running_mean`` / ``running_var`` (the PyTorch
          default); ``eval`` normalizes with the wired statistics.
    Out:  mode (STRING) - the selected mode, to be linked into ``mode`` slots.
          Placing this node without linking anything is harmless: it is only
          evaluated when a consumer asks for it.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingMode",
            display_name="Training Mode",
            category=TRAINING_CATEGORY,
            description="Publishes train/eval to every node whose mode slot is linked to it; flip it to switch a graph between training and inference.",
            search_aliases=["train", "eval", "inference", "mode", "batchnorm", "state"],
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=list(MODE_OPTIONS),
                    default=MODE_TRAIN,
                    tooltip="train: use the statistics of the current call. eval: use the running statistics wired into the consumer.",
                ),
            ],
            outputs=[io.String.Output(display_name="mode")],
        )

    @classmethod
    def execute(cls, mode: str) -> io.NodeOutput:
        return io.NodeOutput(_normalize_mode(mode))


class TrainingRunStats(io.ComfyNode):
    """Carries ``running_mean`` / ``running_var`` across runs, as numbers or as links.

    What: the persistent half of the train/eval switch. A widget is what survives in a
          saved workflow, so both statistics can be typed in as comma separated numbers
          and are emitted as 1-D tensors; wire the outputs into the ``running_mean`` /
          ``running_var`` slots of a BatchNorm / InstanceNorm node.
          Both statistics can also be *linked* instead of typed: BatchNorm and
          InstanceNorm export the per-channel statistics they normalized with, and a
          link wins over the widget, so the numbers of a run can be handed forward
          without ever being copied by hand.
    In:   running_mean (STRING) - per-channel means, e.g. ``"0.0"`` or ``"0.1,0.2,0.3"``.
          running_var (STRING) - per-channel variances, e.g. ``"1.0"``.
          mean (TENSOR, optional) - link the ``mean`` output of a BatchNorm /
          InstanceNorm here; overrides the ``running_mean`` text above.
          var (TENSOR, optional) - link the ``var`` output here; overrides
          ``running_var``.
    Out:  running_mean (TENSOR) - 1-D tensor with the means (linked value, else parsed).
          running_var (TENSOR) - 1-D tensor with the variances.
          One value is broadcast to every channel by the consumer, a value per channel
          is used as is. Text that cannot be parsed falls back to ``0.0`` / ``1.0``
          with a warning, so a typo cannot break a workflow.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingRunStats",
            display_name="Training Run Stats",
            category=TRAINING_CATEGORY,
            description="Editable running_mean / running_var for eval mode; wire the outputs into the running stat slots of BatchNorm / InstanceNorm.",
            search_aliases=["running mean", "running var", "ema", "statistics", "state"],
            inputs=[
                io.String.Input(
                    "running_mean",
                    default="0.0",
                    placeholder="e.g. 0.1,0.2,0.3",
                    tooltip="Per-channel means as comma separated numbers; a single value is broadcast to every channel.",
                ),
                io.String.Input(
                    "running_var",
                    default="1.0",
                    placeholder="e.g. 1.0,1.0,1.0",
                    tooltip="Per-channel variances as comma separated numbers; a single value is broadcast to every channel.",
                ),
                io.Tensor.Input(
                    "mean",
                    optional=True,
                    tooltip="Optional per-channel mean linked from a BatchNorm / InstanceNorm 'mean' output; overrides the running_mean text above.",
                ),
                io.Tensor.Input(
                    "var",
                    optional=True,
                    tooltip="Optional per-channel variance linked from a BatchNorm / InstanceNorm 'var' output; overrides the running_var text above.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="running_mean"),
                io.Tensor.Output(display_name="running_var"),
            ],
        )

    @classmethod
    def execute(
        cls,
        running_mean: str = "0.0",
        running_var: str = "1.0",
        mean: torch.Tensor | None = None,
        var: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _linked_stats(mean, running_mean, "running_mean", 0.0),
            _linked_stats(var, running_var, "running_var", 1.0),
        )


class NormalizationBatchNorm(io.ComfyNode):
    """Batch normalization over the channel dimension: ``F.batch_norm``.

    What: normalizes every channel across all other dimensions - the batch and the
          spatial ones - using the statistics of the current call (``train``) or the
          statistics wired into ``running_mean`` / ``running_var`` (``eval``). One node
          covers every rank, because the shape alone decides which dimensions the
          statistics are taken over: ``(N, C)`` behaves like ``BatchNorm1d``,
          ``(N, C, L)`` like ``BatchNorm1d`` over a sequence, ``(N, C, H, W)`` like
          ``BatchNorm2d`` and ``(N, C, D, H, W)`` like ``BatchNorm3d``. The dimension
          that is *not* reduced over is always dimension 1.
          Nothing is stored inside the node: the batch statistics are not written back
          into ``running_mean`` / ``running_var``, because a tensor may be shared with
          other nodes in the same graph. To carry statistics from one run into the next
          one, wire the ``mean`` / ``var`` outputs into a Training Run Stats node.
    In:   tensor (TENSOR) - input of shape ``(N, C, ...)``; dtype/device preserved.
          weight (TENSOR, optional) - per-channel scale gamma of shape ``(C,)``.
          bias (TENSOR, optional) - per-channel shift beta of shape ``(C,)``.
          running_mean / running_var (TENSOR, optional) - per-channel statistics used in
          ``eval`` mode, shape ``(C,)``; a single value is broadcast to every channel.
          Wire them from Training Run Stats.
          eps (FLOAT) - stabiliser added to the variance.
          mode (STRING, optional) - link the ``mode`` output of a Training Mode node;
          unconnected means ``train``.
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
          mean (TENSOR) - the per-channel mean this call normalized with: the batch mean
          of the current call in ``train``, the wired ``running_mean`` in ``eval``. Shape
          ``(C,)``, so it can be linked into Training Run Stats.
          var (TENSOR) - likewise the per-channel variance.

    In ``eval`` mode without usable statistics the batch statistics are used instead
    and a warning is printed, so an incomplete graph never breaks.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationBatchNorm",
            "BatchNorm",
            "Batch normalization over the channel dimension of (N, C, ...); one node covers the 1d/2d/3d flavours.",
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of shape (N, C) or (N, C, ...)."),
                io.Tensor.Input("weight", optional=True, tooltip="Optional per-channel scale gamma, shape (C,)."),
                io.Tensor.Input("bias", optional=True, tooltip="Optional per-channel shift beta, shape (C,)."),
                io.Tensor.Input(
                    "running_mean",
                    optional=True,
                    tooltip="Per-channel mean for eval mode, shape (C,) or a single value; wire it from Training Run Stats.",
                ),
                io.Tensor.Input(
                    "running_var",
                    optional=True,
                    tooltip="Per-channel variance for eval mode, shape (C,) or a single value; wire it from Training Run Stats.",
                ),
                io.Float.Input(
                    "eps",
                    default=1e-5,
                    min=0.0,
                    max=1e-2,
                    step=1e-6,
                    tooltip="Stabiliser added to the variance.",
                ),
                _mode_input(),
            ],
            search_aliases=["bn", "batchnorm", "batch norm", "normalize batch", "running stats"],
            outputs=[io.Tensor.Output(display_name="output"), *_STATS_OUTPUTS],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        running_mean: torch.Tensor | None = None,
        running_var: torch.Tensor | None = None,
        eps: float = 1e-5,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        if tensor.dim() < 2:
            _warn("BatchNorm needs a channel dimension, i.e. rank >= 2 (N, C, ...); returning the input unchanged.")
            return io.NodeOutput(tensor, tensor.mean().reshape(1), tensor.var(unbiased=False).reshape(1))
        # Every dimension but the channel one, which is what lets one node cover the
        # 1d/2d/3d flavours. ``unbiased=False`` matches the biased variance that
        # ``F.batch_norm`` normalizes with, so the exported value is the one used.
        reduce_dims = (0,) + tuple(range(2, tensor.dim()))
        batch_mean = tensor.mean(dim=reduce_dims)
        batch_var = tensor.var(dim=reduce_dims, unbiased=False)
        if _normalize_mode(mode) == MODE_EVAL:
            channels = tensor.shape[1]
            mean = _broadcast_stat(running_mean, channels, tensor, "running_mean")
            var = _broadcast_stat(running_var, channels, tensor, "running_var")
            if mean is not None and var is not None:
                output = F.batch_norm(tensor, mean, var, weight, bias, training=False, eps=eps)
                return io.NodeOutput(output, mean.reshape(-1), var.reshape(-1))
            _warn("BatchNorm is in eval mode but no usable running statistics are wired in; using the batch statistics instead.")
        # ``training=False`` fed with the statistics computed above: the same result as
        # ``training=True``, but the reduction happens once instead of twice.
        output = F.batch_norm(tensor, batch_mean, batch_var, weight, bias, training=False, eps=eps)
        return io.NodeOutput(output, batch_mean, batch_var)


class NormalizationInstanceNorm(io.ComfyNode):
    """Instance normalization: ``F.instance_norm``.

    What: like BatchNorm, but the statistics are taken per sample *and* per channel, so
          no information is shared across the batch. Typical use is style transfer and
          any layer where each image must be normalized independently. As with BatchNorm
          a single node covers ``(N, C, L)``, ``(N, C, H, W)`` and ``(N, C, D, H, W)``;
          the spatial dimensions are whatever follows the channel dimension.
          Nothing is stored inside the node, and the wired statistics are never updated
          in place.
    In:   tensor (TENSOR) - input of shape ``(N, C, ...)`` with at least one spatial
          dimension; dtype/device preserved.
          weight (TENSOR, optional) - per-channel scale gamma of shape ``(C,)``.
          bias (TENSOR, optional) - per-channel shift beta of shape ``(C,)``.
          running_mean / running_var (TENSOR, optional) - per-channel statistics used in
          ``eval`` mode, shape ``(C,)``; a single value is broadcast to every channel.
          Wire them from Training Run Stats.
          eps (FLOAT) - stabiliser added to the variance.
          mode (STRING, optional) - link the ``mode`` output of a Training Mode node;
          unconnected means ``train`` (normalize with the statistics of this call).
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
          mean (TENSOR) - the per-channel mean this call normalized with: the statistics
          of the current call in ``train``, the wired ``running_mean`` in ``eval``. The
          per-sample means are reduced to one value per channel, so the pair means the
          same thing as BatchNorm's and can be stored in the same slots.
          var (TENSOR) - likewise the per-channel variance.

    In ``eval`` mode without usable statistics the statistics of the current call are
    used instead and a warning is printed. A tensor without a spatial dimension (rank
    below 3) is returned unchanged, because there is nothing to normalize over.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationInstanceNorm",
            "InstanceNorm",
            "Per-sample, per-channel normalization over the spatial dimensions of (N, C, ...).",
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of shape (N, C, ...) with at least one spatial dimension."),
                io.Tensor.Input("weight", optional=True, tooltip="Optional per-channel scale gamma, shape (C,)."),
                io.Tensor.Input("bias", optional=True, tooltip="Optional per-channel shift beta, shape (C,)."),
                io.Tensor.Input(
                    "running_mean",
                    optional=True,
                    tooltip="Per-channel mean for eval mode, shape (C,) or a single value; wire it from Training Run Stats.",
                ),
                io.Tensor.Input(
                    "running_var",
                    optional=True,
                    tooltip="Per-channel variance for eval mode, shape (C,) or a single value; wire it from Training Run Stats.",
                ),
                io.Float.Input(
                    "eps",
                    default=1e-5,
                    min=0.0,
                    max=1e-2,
                    step=1e-6,
                    tooltip="Stabiliser added to the variance.",
                ),
                _mode_input(),
            ],
            search_aliases=["in", "instancenorm", "instance norm", "per sample", "style transfer"],
            outputs=[io.Tensor.Output(display_name="output"), *_STATS_OUTPUTS],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        running_mean: torch.Tensor | None = None,
        running_var: torch.Tensor | None = None,
        eps: float = 1e-5,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        if tensor.dim() < 3:
            _warn("InstanceNorm needs a spatial dimension, i.e. rank >= 3 (N, C, ...); returning the input unchanged.")
            return io.NodeOutput(tensor, tensor.mean().reshape(1), tensor.var(unbiased=False).reshape(1))
        if _normalize_mode(mode) == MODE_EVAL:
            channels = tensor.shape[1]
            mean = _broadcast_stat(running_mean, channels, tensor, "running_mean")
            var = _broadcast_stat(running_var, channels, tensor, "running_var")
            if mean is not None and var is not None:
                output = F.instance_norm(tensor, mean, var, weight, bias, use_input_stats=False, eps=eps)
                return io.NodeOutput(output, mean.reshape(-1), var.reshape(-1))
            _warn("InstanceNorm is in eval mode but no usable running statistics are wired in; using the statistics of this call instead.")
        output = F.instance_norm(tensor, None, None, weight, bias, use_input_stats=True, eps=eps)
        # Instance statistics are per sample, so they are reduced to one mean/variance
        # per channel: the exported pair then means the same thing as BatchNorm's and
        # fits the same ``running_mean`` / ``running_var`` slots. The variance is the
        # two-way decomposition - the mean of the within-sample variances plus the
        # variance of the per-sample means - which is exactly the variance of the whole
        # batch, i.e. the quantity BatchNorm would have reported for this tensor.
        spatial = tuple(range(2, tensor.dim()))
        sample_mean = tensor.mean(dim=spatial)
        sample_var = tensor.var(dim=spatial, unbiased=False)
        return io.NodeOutput(
            output,
            sample_mean.mean(dim=0),
            sample_var.mean(dim=0) + sample_mean.var(dim=0, unbiased=False),
        )


class NormalizationLayerNorm(io.ComfyNode):
    """Layer normalization over the trailing dimensions: ``F.layer_norm``.

    What: normalizes each sample independently, over the dimensions named by
          ``normalized_shape`` (the last one by default), then applies gamma/beta.
          This is the transformer/MLP normalizer; unlike BatchNorm it does not reduce
          over the batch at all, which is why it has no train/eval switch - its math
          is identical in both modes.
    In:   tensor (TENSOR) - input of shape ``(..., d1, d2, ...)``; dtype/device kept.
          weight (TENSOR, optional) - scale gamma, shaped like the normalized part.
          bias (TENSOR, optional) - shift beta, shaped like the normalized part.
          normalized_shape (STRING) - ``"last"`` normalizes the last dimension; a list
          such as ``"8,16"`` normalizes the last two. The numbers are only used to
          decide how many dimensions to take: if they no longer match the tensor (a
          stale widget value after a shape change) the actual sizes win and a warning
          is printed, so a workflow is never broken by an outdated shape string.
          eps (FLOAT) - stabiliser added to the variance.
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationLayerNorm",
            "LayerNorm",
            'Layer normalization over the trailing dimensions, e.g. "last" or "8,16".',
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of shape (..., d1, d2, ...)."),
                io.Tensor.Input("weight", optional=True, tooltip="Optional scale gamma, shaped like the normalized part."),
                io.Tensor.Input("bias", optional=True, tooltip="Optional shift beta, shaped like the normalized part."),
                io.String.Input(
                    "normalized_shape",
                    default="last",
                    placeholder="e.g. 8,16",
                    tooltip='"last" normalizes the last dimension; a list such as "8,16" normalizes the last two.',
                ),
                io.Float.Input(
                    "eps",
                    default=1e-5,
                    min=0.0,
                    max=1e-2,
                    step=1e-6,
                    tooltip="Stabiliser added to the variance.",
                ),
            ],
            search_aliases=["ln", "layernorm", "layer norm", "transformer norm"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        normalized_shape: str = "last",
        eps: float = 1e-5,
    ) -> io.NodeOutput:
        if tensor.dim() < 1:
            _warn("LayerNorm received a 0-dim tensor; returning it unchanged.")
            return io.NodeOutput(tensor)
        shape = _resolve_normalized_shape(normalized_shape, tensor.shape)
        return io.NodeOutput(F.layer_norm(tensor, shape, weight, bias, eps))


class NormalizationGroupNorm(io.ComfyNode):
    """Group normalization: ``F.group_norm``.

    What: splits the channels into ``num_groups`` groups and normalizes each group
          over its channels and all remaining dimensions. It sits between LayerNorm
          (every channel in one group) and InstanceNorm (one channel per group), and
          is the usual replacement for BatchNorm when the batch is small, because it
          never reduces over the batch. ``num_groups=1`` is the friendliest default:
          it works for every channel count.
    In:   tensor (TENSOR) - input of shape ``(N, C, ...)``; dtype/device preserved.
          weight (TENSOR, optional) - per-channel scale gamma of shape ``(C,)``.
          bias (TENSOR, optional) - per-channel shift beta of shape ``(C,)``.
          num_groups (INT) - number of channel groups. If it does not divide the
          channel count (or is below 1) one group is used instead and a warning is
          printed, so a workflow is never broken by an arithmetic slip.
          eps (FLOAT) - stabiliser added to the variance.
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationGroupNorm",
            "GroupNorm",
            "Normalization over groups of channels; num_groups=1 normalizes over all channels.",
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of shape (N, C, ...)."),
                io.Tensor.Input("weight", optional=True, tooltip="Optional per-channel scale gamma, shape (C,)."),
                io.Tensor.Input("bias", optional=True, tooltip="Optional per-channel shift beta, shape (C,)."),
                io.Int.Input(
                    "num_groups",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Channel groups; must divide the channel count, otherwise 1 group is used.",
                ),
                io.Float.Input(
                    "eps",
                    default=1e-5,
                    min=0.0,
                    max=1e-2,
                    step=1e-6,
                    tooltip="Stabiliser added to the variance.",
                ),
            ],
            search_aliases=["gn", "groupnorm", "group norm", "small batch"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        num_groups: int = 1,
        eps: float = 1e-5,
    ) -> io.NodeOutput:
        if tensor.dim() < 2:
            _warn("GroupNorm needs a channel dimension, i.e. rank >= 2 (N, C, ...); returning the input unchanged.")
            return io.NodeOutput(tensor)
        channels = tensor.shape[1]
        groups = num_groups if num_groups >= 1 and channels % num_groups == 0 else 1
        if groups != num_groups:
            _warn(f"GroupNorm num_groups={num_groups} is not usable for {channels} channel(s); using 1 group instead.")
        return io.NodeOutput(F.group_norm(tensor, groups, weight, bias, eps))


class NormalizationRMSNorm(io.ComfyNode):
    """Root-mean-square normalization: ``F.rms_norm``.

    What: divides by the root mean square of the normalized dimensions and multiplies
          by an optional scale, without centering and without a bias. This is the
          normalizer used by LLaMA-style models; it is cheaper than LayerNorm (no mean
          subtraction) and has no train/eval switch, because its math is identical in
          both modes.
    In:   tensor (TENSOR) - input of shape ``(..., d1, d2, ...)``; dtype/device kept.
          weight (TENSOR, optional) - scale gamma, shaped like the normalized part.
          normalized_shape (STRING) - ``"last"`` normalizes the last dimension; a list
          such as ``"8,16"`` normalizes the last two. As in LayerNorm the numbers only
          decide how many dimensions to take and the actual sizes win when they differ.
          eps (FLOAT) - stabiliser added inside the square root; 1e-6 follows LLaMA.
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationRMSNorm",
            "RMSNorm",
            "Root-mean-square normalization over the trailing dimensions, e.g. LLaMA style.",
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of shape (..., d1, d2, ...)."),
                io.Tensor.Input("weight", optional=True, tooltip="Optional scale gamma, shaped like the normalized part."),
                io.String.Input(
                    "normalized_shape",
                    default="last",
                    placeholder="e.g. 8,16",
                    tooltip='"last" normalizes the last dimension; a list such as "8,16" normalizes the last two.',
                ),
                io.Float.Input(
                    "eps",
                    default=1e-6,
                    min=0.0,
                    max=1e-2,
                    step=1e-6,
                    tooltip="Stabiliser added inside the square root (1e-6 follows LLaMA).",
                ),
            ],
            search_aliases=["rms", "rmsnorm", "root mean square", "llama", "llm norm"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor | None = None,
        normalized_shape: str = "last",
        eps: float = 1e-6,
    ) -> io.NodeOutput:
        if tensor.dim() < 1:
            _warn("RMSNorm received a 0-dim tensor; returning it unchanged.")
            return io.NodeOutput(tensor)
        shape = _resolve_normalized_shape(normalized_shape, tensor.shape)
        return io.NodeOutput(F.rms_norm(tensor, shape, weight, eps))


class NormalizationWeightNorm(io.ComfyNode):
    """Weight re-parameterization: ``weight = g * v / ||v||``.

    What: a *re-parameterization of a weight tensor*, not a normalization of
          activations. It decouples the direction of a weight matrix from its scale:
          the norm along ``dim`` is removed and re-applied through the optional ``g``,
          which makes the scale a separate knob (the trick behind "weight
          normalization" and a common way to keep activations stable while training).
          This node is stateless - unlike ``torch.nn.utils.weight_norm`` it does not
          register anything on a module, it simply returns the re-parameterized tensor.
    In:   weight (TENSOR) - the weight to re-parameterize, e.g. ``(out, in)``.
          g (TENSOR, optional) - the scale that replaces the removed norm. Its shape is
          the weight shape without ``dim`` (e.g. ``(out, 1)`` for ``dim=0``); anything
          broadcastable against the weight works. Leave it unconnected for a purely
          direction-normalized weight (which then has unit norm along ``dim``).
          dim (INT) - dimension to take the norm over, clamped to the weight rank.
          eps (FLOAT) - lower bound applied to the norm, so a zero weight cannot divide
          by zero.
    Out:  output (TENSOR) - the re-parameterized weight, same shape as ``weight``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _normalization_schema(
            "NormalizationWeightNorm",
            "WeightNorm",
            "Re-parameterize a weight as g * v / ||v||, i.e. separate its direction from its scale.",
            inputs=[
                io.Tensor.Input("weight", tooltip="Weight tensor to re-parameterize, e.g. (out, in)."),
                io.Tensor.Input(
                    "g",
                    optional=True,
                    tooltip="Optional scale replacing the removed norm, e.g. (out, 1) for dim=0; leave unconnected for a unit-norm weight.",
                ),
                io.Int.Input(
                    "dim",
                    default=0,
                    min=-8,
                    max=7,
                    step=1,
                    tooltip="Dimension to take the norm over (clamped to the weight rank).",
                ),
                io.Float.Input(
                    "eps",
                    default=1e-12,
                    min=0.0,
                    max=1e-3,
                    step=1e-9,
                    tooltip="Lower bound applied to the norm.",
                ),
            ],
            search_aliases=["weight normalization", "reparameterization", "direction", "scale", "norm of weight"],
        )

    @classmethod
    def execute(
        cls,
        weight: torch.Tensor,
        g: torch.Tensor | None = None,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> io.NodeOutput:
        if weight.dim() < 1:
            _warn("WeightNorm received a 0-dim weight; returning it unchanged.")
            return io.NodeOutput(weight)
        norm = torch.linalg.vector_norm(weight, dim=_clamp_dim(dim, weight.dim()), keepdim=True).clamp_min(eps)
        output = weight / norm
        if g is not None:
            output = output * g
        return io.NodeOutput(output)


class NormalizationSpectralNorm(io.ComfyNode):
    """Weight re-parameterization by the spectral norm: ``weight / sigma``.

    What: divides a weight matrix by an estimate of its largest singular value, which
          bounds how much the layer can stretch its input (a Lipschitz bound; the
          "spectral normalization" of GAN discriminators). The estimate comes from a
          power iteration whose starting vectors are derived from the weight itself,
          so a run is fully reproducible and cache friendly - no random numbers and no
          hidden state, unlike ``torch.nn.utils.spectral_norm``.
    In:   weight (TENSOR) - weight of shape ``(..., m, n)``; the ``dim`` dimension
          becomes the ``m`` rows of the matrix whose singular value is estimated.
          u (TENSOR, optional) - left singular vector of length ``m``; leave it
          unconnected for the deterministic default.
          v (TENSOR, optional) - right singular vector of length ``n``; leave it
          unconnected for the deterministic default.
          n_power_iterations (INT) - how many refinement steps to run; the default 10
          sits close to the true largest singular value, while 1 already gives a usable
          estimate and skips 9 matrix-vector products.
          dim (INT) - dimension that counts as "rows", clamped to the weight rank.
          eps (FLOAT) - lower bound applied to the estimated value, so a weight that is
          numerically zero cannot divide by zero.
    Out:  output (TENSOR) - the re-parameterized weight, same shape as ``weight``.
          sigma (TENSOR) - the estimated largest singular value, so it can be inspected
          or multiplied back with a soft scale. A weight with rank below 2 is returned
          unchanged with ``sigma = 0``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NormalizationSpectralNorm",
            display_name="SpectralNorm",
            category=NORMALIZATION_CATEGORY,
            description="Divide a weight by an estimate of its largest singular value (deterministic power iteration).",
            search_aliases=["spectral normalization", "largest singular value", "power iteration", "lipschitz", "gan"],
            inputs=[
                io.Tensor.Input("weight", tooltip="Weight of shape (..., m, n)."),
                io.Tensor.Input("u", optional=True, tooltip="Optional left singular vector of length m; unconnected uses the deterministic default."),
                io.Tensor.Input("v", optional=True, tooltip="Optional right singular vector of length n; unconnected uses the deterministic default."),
                io.Int.Input(
                    "n_power_iterations",
                    default=10,
                    min=0,
                    max=20,
                    step=1,
                    tooltip="Refinement steps of the power iteration used to estimate the largest singular value.",
                ),
                io.Int.Input(
                    "dim",
                    default=0,
                    min=-8,
                    max=7,
                    step=1,
                    tooltip='Dimension that counts as the "rows" of the weight matrix (clamped to the weight rank).',
                ),
                io.Float.Input(
                    "eps",
                    default=1e-12,
                    min=0.0,
                    max=1e-3,
                    step=1e-9,
                    tooltip="Lower bound applied to the estimated singular value.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="output"),
                io.Tensor.Output(display_name="sigma"),
            ],
        )

    @classmethod
    def execute(
        cls,
        weight: torch.Tensor,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        n_power_iterations: int = 10,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> io.NodeOutput:
        if weight.dim() < 2:
            _warn("SpectralNorm needs a weight with at least two dimensions; returning it unchanged.")
            return io.NodeOutput(weight, torch.zeros((), dtype=weight.dtype, device=weight.device))
        axis = _clamp_dim(dim, weight.dim())
        matrix = weight.movedim(axis, 0).reshape(weight.shape[axis], -1)
        left = _unit_vector(_matrix_vector(u, matrix.sum(dim=1)), eps)
        right = _unit_vector(_matrix_vector(v, matrix.sum(dim=0)), eps)
        for _ in range(max(0, int(n_power_iterations))):
            left = _unit_vector(matrix @ right, eps)
            right = _unit_vector(matrix.mT @ left, eps)
        sigma = (left @ (matrix @ right)).abs()
        return io.NodeOutput(weight / sigma.clamp_min(eps), sigma)


def _dropout(tensor: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """Drop each element of ``tensor`` independently with probability ``p``.

    The mask is drawn from a generator built for the tensor's own device instead of the
    process-wide RNG, so the result depends on ``seed`` alone and the RNG stream the
    host hands to the other nodes is left untouched. Surviving elements are divided by
    ``1 - p``, which keeps ``E[output] == E[input]``.

    Args:
        tensor: Input tensor of any shape.
        p: Drop probability, strictly between 0 and 1 (the caller handles the edges).
        seed: Seed of the mask draw.

    Returns:
        A new tensor with the same shape/dtype/device as ``tensor``.
    """
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    keep = torch.rand(tensor.shape, generator=generator, device=tensor.device, dtype=torch.float32) >= p
    return torch.where(keep, tensor / (1.0 - p), tensor.new_zeros(()))


class RegularizationDropout(io.ComfyNode):
    """Element-wise dropout: ``F.dropout``.

    What: zeroes each element of the input independently with probability ``p`` and
          divides the surviving elements by ``1 - p``, so the expectation of the output
          equals the input (the ``torch.nn.Dropout`` semantics used to regularize a
          training run, and a ready source of stochastic masks in general).
          The mask is drawn element by element, so one node covers every rank -
          ``(N, C)``, ``(N, C, H, W)`` or a bare scalar - with no 1d/2d/3d flavour to
          pick. This is the plain element-wise dropout; the channel-wise variant some
          recurrent networks use is a different operation and a different node.
          The draw is seeded: the same ``seed`` and ``p`` reproduce the mask bit for
          bit, which keeps ComfyUI's caching meaningful (an unchanged graph returns the
          cached output instead of a fresh mask). The "control after generate"
          dropdown next to the seed widget is what moves the seed between runs - set it
          to "randomize" for a new mask every run, "fixed" to freeze the mask.
          Nothing is stored in the node and the input is never written to, because the
          same tensor may be shared with other nodes.
    In:   tensor (TENSOR) - input of any shape; dtype/device preserved.
          p (FLOAT) - probability that an element is dropped, 0 to 1. ``0`` passes the
          input through unchanged, ``1`` returns zeros.
          seed (INT) - seed of the mask draw; only used in ``train`` mode.
          mode (STRING, optional) - link the ``mode`` output of a Training Mode node;
          unconnected means ``train``. In ``eval`` mode the input is passed straight
          through, which makes a Dropout left in an inference graph harmless.
    Out:  output (TENSOR) - same shape/dtype/device as ``tensor``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RegularizationDropout",
            display_name="Dropout",
            category=REGULARIZATION_CATEGORY,
            description="Zero each element with probability p and rescale the rest by 1/(1-p); eval mode passes the input through.",
            search_aliases=["dropout", "drop out", "regularization", "regularizer", "dropout rate", "mask"],
            inputs=[
                io.Tensor.Input("tensor", tooltip="Input of any shape; dtype and device are preserved."),
                io.Float.Input(
                    "p",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Probability that an element is dropped. 0 passes the input through, 1 returns zeros.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed of the mask draw: the same seed reproduces the same mask. The dropdown decides whether the value changes after each run.",
                ),
                _mode_input(
                    tooltip="Link the mode output of a Training Mode node: 'train' drops elements, 'eval' passes the input through unchanged. Unconnected means 'train'.",
                ),
            ],
            outputs=[io.Tensor.Output(display_name="output")],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        p: float = 0.5,
        seed: int = 0,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        if _normalize_mode(mode) == MODE_EVAL:
            return io.NodeOutput(tensor)
        if p <= 0.0:
            return io.NodeOutput(tensor)
        if p >= 1.0:
            return io.NodeOutput(torch.zeros_like(tensor))
        return io.NodeOutput(_dropout(tensor, p, seed))


#: Every node this module registers, in node-library order. It spans three
#: ``Network & Layers`` families: Normalization, Regularization and Training.
NORMALIZATION_NODES: list[type[io.ComfyNode]] = [
    NormalizationBatchNorm,
    NormalizationInstanceNorm,
    NormalizationLayerNorm,
    NormalizationGroupNorm,
    NormalizationRMSNorm,
    NormalizationWeightNorm,
    NormalizationSpectralNorm,
    TrainingMode,
    TrainingRunStats,
    RegularizationDropout,
]


class NormalizationExtension(ComfyExtension):
    """Registers the core Normalization / Regularization families and the training-state helpers."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(NORMALIZATION_NODES)


async def comfy_entrypoint() -> NormalizationExtension:
    return NormalizationExtension()
