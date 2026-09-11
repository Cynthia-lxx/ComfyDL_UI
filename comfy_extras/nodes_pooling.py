"""Pooling nodes (reform step 4).

Provides the ``Network & Layers/Pooling`` category with **two** nodes that
together cover the whole torch pooling family:

* ``Pool``           - sliding window pooling, ``mode`` (max/avg) x ``dims`` (1/2/3)
                       dispatches to ``torch.nn.functional.max_pool{1,2,3}d`` /
                       ``avg_pool{1,2,3}d``.
* ``Adaptive Pool``  - adaptive pooling, ``mode`` (max/avg) x ``dims`` (1/2/3) plus
                       an ``output_size`` widget dispatching to
                       ``adaptive_max_pool{1,2,3}d`` / ``adaptive_avg_pool{1,2,3}d``.

Twelve torch layer types (``MaxPool1d/2d/3d``, ``AvgPool1d/2d/3d``,
``AdaptiveAvgPool1d/2d/3d``, ``AdaptiveMaxPool1d/2d/3d``) share exactly one
input/output shape, so they are collapsed into two nodes whose difference is
carried by widgets instead of by node count. Global pooling needs no node of its
own: ``Adaptive Pool`` with ``output_size=1`` *is* ``GlobalAvgPool{d}`` /
``GlobalMaxPool{d}``.

Every node is stateless (no parameters, no initialisation) and preserves the
input dtype/device.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Pooling"

#: Supported spatial ranks.
_DIMS = [1, 2, 3]

#: Supported pooling reductions.
_MODES = ["max", "avg"]


def _pooling_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by every Pooling node.

    Both Pooling nodes take one tensor and return exactly one ``TENSOR`` output
    named ``output``; only the inputs differ, so callers pass the already-built
    input list.

    Args:
        node_id: Globally unique, core-safe node id (``Pooling`` prefix).
        display_name: Short name shown in the node library (e.g. ``Pool``).
        description: Tooltip shown when hovering over the node.
        inputs: The node's inputs, in declaration order (slots and widgets).
        search_aliases: Extra search keywords for the node library.

    Returns:
        The ``io.Schema`` describing the node.
    """
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category=CATEGORY,
        description=description,
        search_aliases=search_aliases,
        inputs=list(inputs),
        outputs=[io.Tensor.Output(display_name="output")],
    )


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _dims_widget() -> io.Combo.Input:
    """Build the shared ``dims`` widget (which rank of pooling to apply)."""
    return io.Combo.Input(
        "dims",
        options=_DIMS,
        default=2,
        tooltip=(
            "Spatial rank: 1 = sequence (N, C, L), 2 = image (N, C, H, W), "
            "3 = volume (N, C, D, H, W). It also selects Pool1d/Pool2d/Pool3d."
        ),
    )


def _mode_widget(default: str, tooltip: str) -> io.Combo.Input:
    """Build the shared ``mode`` widget (max or average reduction)."""
    return io.Combo.Input(
        "mode",
        options=_MODES,
        default=default,
        tooltip=tooltip,
    )


def _ensure_min_rank(tensor: torch.Tensor, minimum: int) -> tuple[torch.Tensor, int]:
    """Unsqueeze leading dimensions until ``tensor`` has at least ``minimum`` dims.

    torch's pooling kernels accept either ``(N, C, *spatial)`` or
    ``(C, *spatial)``; a bare spatial tensor is unsqueezed so that unbatched
    tensors (the common case while learning) pool just as well as batched ones.

    Args:
        tensor: The tensor to prepare.
        minimum: Smallest rank the pooling kernel accepts.

    Returns:
        ``(prepared_tensor, added_dims)`` - pass ``added_dims`` to
        :func:`_drop_added_rank` afterwards to restore the original rank.
    """
    added = 0
    while tensor.dim() < minimum:
        tensor = tensor.unsqueeze(0)
        added += 1
    return tensor, added


def _drop_added_rank(tensor: torch.Tensor, added: int) -> torch.Tensor:
    """Remove the leading singleton dimensions added by :func:`_ensure_min_rank`."""
    for _ in range(added):
        tensor = tensor.squeeze(0)
    return tensor


def _apply_pooling(
    operation,
    prepared: torch.Tensor,
    args: tuple,
    node_name: str,
    rank: int,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    """Run one pooling call and turn a torch shape complaint into a readable error.

    Args:
        operation: The ``torch.nn.functional`` pooling function to call.
        prepared: The rank-adjusted input tensor.
        args: Positional arguments for ``operation`` (after the input).
        node_name: Display name of the node, used in the message.
        rank: Spatial rank (1/2/3) the user selected.
        original_shape: The input shape as it arrived, used in the message.

    Returns:
        The pooled tensor.

    Raises:
        RuntimeError: With a message naming the node, the input shape and the
            fix, when torch rejects the combination of shape and widgets.
    """
    try:
        result = operation(prepared, *args)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{node_name}: {rank}d pooling failed for an input of shape "
            f"{tuple(original_shape)}. Check that 'dims' matches the tensor rank "
            f"(dims=2 expects (N, C, H, W)), that kernel_size/stride/padding fit "
            f"the spatial size, and that 1d max pooling keeps padding no larger "
            f"than half the kernel. Original error: {exc}"
        ) from exc
    # adaptive_max_pool* returns (values, indices) in some torch versions.
    return result[0] if isinstance(result, tuple) else result


class PoolingSliding(io.ComfyNode):
    """Sliding window pooling (max or average) over 1, 2 or 3 spatial dimensions.

    What: the functional form of ``torch.nn.MaxPool{1,2,3}d`` and
          ``torch.nn.AvgPool{1,2,3}d``. A learnable-free downsampling step that
          slides a ``kernel_size`` window across the spatial dimensions and
          keeps the reduction (max or mean) of each window - the classic
          "shrink the feature map, keep the strongest response" operation.
          ``mode`` picks the reduction and ``dims`` picks the rank, so the twelve
          torch layer types collapse into one node.
    In:   tensor (TENSOR) - feature map, usually ``(N, C, H, W)``. A bare
          ``(C, H, W)`` or even ``(H, W)`` tensor is accepted: missing leading
          dimensions are added internally and removed again from the result, so
          unbatched tensors pool exactly like batched ones.
          dims (COMBO) - 1, 2 or 3 (default 2): selects Pool1d/2d/3d and how many
          spatial dimensions the widgets below apply to.
          mode (COMBO) - ``max`` (default) or ``avg``.
          kernel_size (INT) - window size, default 2 (halves the spatial size).
          stride (INT) - window step, default 0 meaning "same as kernel_size"
          (torch's own default, giving non-overlapping windows).
          padding (INT) - implicit zero padding added on both sides, default 0.
          dilation (INT) - spacing between window elements, default 1.
          ceil_mode (BOOLEAN) - default false; when true the last window is kept
          even if it reaches past the input edge.
          count_include_pad (BOOLEAN) - default true; when false the padded zeros
          are excluded from the average divisor.
    Out:  output (TENSOR) - pooled tensor with the same rank as the input.
          ``dims=2, kernel_size=2`` reproduces ``nn.MaxPool2d(2)`` exactly.
    Note: ``dilation`` only exists for ``mode=max`` and ``count_include_pad`` only
          for ``mode=avg`` (that is torch's API, not a limitation here); the
          unused widget is ignored and a note is printed when it is set to a
          non-default value.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _pooling_schema(
            "PoolingSliding",
            "Pool",
            "Sliding window pooling (max/avg) over 1/2/3 spatial dims - MaxPool{1,2,3}d and AvgPool{1,2,3}d in one node.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Feature map, e.g. (N, C, H, W); unbatched (C, H, W) also works.",
                ),
                _dims_widget(),
                _mode_widget(
                    "max",
                    "Reduction: 'max' keeps the strongest activation per window, 'avg' averages it.",
                ),
                io.Int.Input(
                    "kernel_size",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Window size; 2 halves the spatial size with the default stride.",
                ),
                io.Int.Input(
                    "stride",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Window step; 0 means 'use kernel_size' (torch default).",
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=32,
                    step=1,
                    tooltip="Implicit zero padding on both sides.",
                ),
                io.Int.Input(
                    "dilation",
                    default=1,
                    min=1,
                    max=16,
                    step=1,
                    tooltip="Spacing between window elements (max pooling only).",
                ),
                io.Boolean.Input(
                    "ceil_mode",
                    default=False,
                    tooltip="Keep a window that reaches past the input edge (round up instead of down).",
                ),
                io.Boolean.Input(
                    "count_include_pad",
                    default=True,
                    tooltip="Include padded zeros in the average divisor (avg pooling only).",
                ),
            ],
            search_aliases=[
                "maxpool", "avgpool", "maxpool1d", "maxpool2d", "maxpool3d",
                "avgpool1d", "avgpool2d", "avgpool3d", "max pooling",
                "average pooling", "subsampling", "downsample",
            ],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        dims: int,
        mode: str,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> io.NodeOutput:
        rank = int(dims)
        if rank not in (1, 2, 3):
            _warn(f"Pool: dims={rank} is not 1/2/3; using 2.")
            rank = 2
        kernel = max(1, int(kernel_size))
        step = int(stride) if int(stride) > 0 else None
        pad = max(0, int(padding))
        dil = max(1, int(dilation))
        if mode == "max" and not count_include_pad:
            _warn("Pool: count_include_pad is ignored for max pooling.")
        if mode != "max" and dil != 1:
            _warn("Pool: dilation is ignored for average pooling.")

        prepared, added = _ensure_min_rank(tensor, rank + 1)
        if mode == "max":
            if rank == 1:
                operation, args = F.max_pool1d, (kernel, step, pad, dil, ceil_mode)
            elif rank == 2:
                operation, args = F.max_pool2d, (kernel, step, pad, dil, ceil_mode)
            else:
                operation, args = F.max_pool3d, (kernel, step, pad, dil, ceil_mode)
        else:
            if rank == 1:
                operation, args = F.avg_pool1d, (kernel, step, pad, ceil_mode, count_include_pad)
            elif rank == 2:
                operation, args = F.avg_pool2d, (kernel, step, pad, ceil_mode, count_include_pad)
            else:
                operation, args = F.avg_pool3d, (kernel, step, pad, ceil_mode, count_include_pad)

        pooled = _apply_pooling(
            operation, prepared, args, "Pool", rank, tuple(tensor.shape)
        )
        return io.NodeOutput(_drop_added_rank(pooled, added))


class PoolingAdaptive(io.ComfyNode):
    """Adaptive pooling (max or average) that resizes to an exact output size.

    What: the functional form of ``torch.nn.AdaptiveAvgPool{1,2,3}d`` and
          ``AdaptiveMaxPool{1,2,3}d``. Instead of sliding a fixed window, each
          output cell covers the input range it needs, so the spatial size
          becomes whatever ``output_size`` asks for - independent of the input
          size. Setting ``output_size=1`` is the special case that collapses
          every spatial dimension at once, i.e. **global** average (or max)
          pooling, which is why no separate ``GlobalAvgPool`` node exists.
    In:   tensor (TENSOR) - feature map, usually ``(N, C, H, W)``. A bare
          ``(C, H, W)`` or ``(H, W)`` tensor is accepted: missing leading
          dimensions are added internally and removed again from the result.
          dims (COMBO) - 1, 2 or 3 (default 2): selects AdaptivePool1d/2d/3d.
          mode (COMBO) - ``avg`` (default) or ``max``.
          output_size (INT) - target size of every spatial dimension, default 1
          (global pooling).
    Out:  output (TENSOR) - same rank as the input, with each spatial dimension
          equal to ``output_size``. ``dims=2, output_size=1`` reproduces
          ``nn.AdaptiveAvgPool2d(1)`` and the textbook "Global Average Pooling".
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _pooling_schema(
            "PoolingAdaptive",
            "Adaptive Pool",
            "Adaptive pooling (avg/max) to an exact output_size over 1/2/3 spatial dims; output_size=1 is global pooling.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Feature map, e.g. (N, C, H, W); unbatched (C, H, W) also works.",
                ),
                _dims_widget(),
                _mode_widget(
                    "avg",
                    "Reduction: 'avg' is the usual Global Average Pooling, 'max' keeps the strongest value.",
                ),
                io.Int.Input(
                    "output_size",
                    default=1,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Target size of every spatial dimension; 1 = global pooling (GlobalAvgPool / GAP).",
                ),
            ],
            search_aliases=[
                "adaptiveavgpool", "adaptivemaxpool", "adaptiveavgpool1d",
                "adaptiveavgpool2d", "adaptiveavgpool3d", "adaptive pooling",
                "global average pooling", "global max pooling", "gap",
                "global pooling",
            ],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        dims: int,
        mode: str,
        output_size: int,
    ) -> io.NodeOutput:
        rank = int(dims)
        if rank not in (1, 2, 3):
            _warn(f"Adaptive Pool: dims={rank} is not 1/2/3; using 2.")
            rank = 2
        size = max(1, int(output_size))

        prepared, added = _ensure_min_rank(tensor, rank + 1)
        if mode == "max":
            if rank == 1:
                operation = F.adaptive_max_pool1d
            elif rank == 2:
                operation = F.adaptive_max_pool2d
            else:
                operation = F.adaptive_max_pool3d
        else:
            if rank == 1:
                operation = F.adaptive_avg_pool1d
            elif rank == 2:
                operation = F.adaptive_avg_pool2d
            else:
                operation = F.adaptive_avg_pool3d

        pooled = _apply_pooling(
            operation, prepared, (size,), "Adaptive Pool", rank, tuple(tensor.shape)
        )
        return io.NodeOutput(_drop_added_rank(pooled, added))


POOLING_NODES: list[type[io.ComfyNode]] = [
    PoolingSliding,
    PoolingAdaptive,
]


class PoolingExtension(ComfyExtension):
    """Registers the core Pooling node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(POOLING_NODES)


async def comfy_entrypoint() -> PoolingExtension:
    return PoolingExtension()
