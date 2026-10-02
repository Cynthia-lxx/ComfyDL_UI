"""Convolution nodes (reform step 4 + step 10 upsampling).

Provides the ``Network & Layers/Convolution`` category with **five** nodes:

* ``Conv``          - ``torch.nn.functional.conv{1,2,3}d``, i.e. the classic
                      parameter-sharing convolution.
* ``ConvTranspose`` - ``torch.nn.functional.conv_transpose{1,2,3}d``, the
                      "deconvolution" used to grow a feature map back to a larger
                      spatial size.
* ``Upsample``      - ``torch.nn.functional.interpolate``, the parameter-free
                      spatial enlargement (nearest / linear / bilinear /
                      bicubic / trilinear) of a feature map.
* ``PixelShuffle``  - ``torch.nn.functional.pixel_shuffle``, the sub-pixel
                      convolution's reshaping half: channels are traded for
                      spatial resolution at no compute cost.
* ``PixelUnshuffle`` - the exact inverse: spatial resolution is traded back
                      for channels.

Six torch layer types (``Conv1d/2d/3d`` and ``ConvTranspose1d/2d/3d``) share one
input/output shape, so they collapse into two nodes whose ``dims`` widget picks
the rank, exactly like the Pooling family; ``Upsample`` follows the same
``dims`` dispatch, while the two pixel-shuffle nodes are inherently 2-D
(``F.pixel_shuffle`` / ``F.pixel_unshuffle`` only exist for images).

Three ideas the nodes are meant to make visible:

* **Parameter sharing** - the same ``weight`` tensor is reused at every spatial
  position, which is why the weight shape contains only the channel counts and
  the kernel size, never the image size. That is the whole difference between a
  convolution and a dense layer over the flattened image.
* **groups / dilation** - ``groups`` splits channels into independent groups
  (``groups == in_channels`` is a depthwise convolution), ``dilation`` spreads
  the kernel out to widen the receptive field without extra weights.
* **Padding modes** - ``zeros`` / ``reflect`` / ``replicate`` / ``circular``
  control what is assumed to lie outside the image.

Following the ``Basic`` category convention, ``weight`` and ``bias`` are wired in
as tensors through input slots: the nodes are stateless, create no parameters and
initialise nothing, so one node can run any checkpoint. The three upsampling
nodes carry **no** weights at all - they are pure reshaping / interpolation.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Convolution"

#: Interpolation modes exposed on ``Upsample``, named like ``nn.Upsample``.
_INTERP_MODES = ["nearest", "linear", "bilinear", "bicubic", "trilinear"]

#: Which interpolation modes are legal per spatial rank (1/2/3), mirroring
#: ``F.interpolate``'s requirement that the mode match the tensor rank.
_MODES_PER_RANK = {
    1: ("nearest", "linear"),
    2: ("nearest", "bilinear", "bicubic"),
    3: ("nearest", "trilinear"),
}

#: Spatial ranks.
_DIMS = [1, 2, 3]

#: Padding modes exposed on ``Conv``, named like ``torch.nn.Conv*d(padding_mode=...)``.
_PADDING_MODES = ["zeros", "reflect", "replicate", "circular"]

#: Map the ``nn`` padding-mode names onto the ``torch.nn.functional.pad`` modes.
_PAD_MODE_TO_FUNCTIONAL = {
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "replicate",
    "circular": "circular",
}


def _conv_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by every Convolution node.

    Both Convolution nodes take one tensor plus a weight (and an optional bias)
    and return exactly one ``TENSOR`` output named ``output``; only the inputs
    differ, so callers pass the already-built input list.

    Args:
        node_id: Globally unique, core-safe node id (``Convolution`` prefix).
        display_name: Short name shown in the node library (e.g. ``Conv``).
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


def _weight_slots(weight_tooltip: str) -> list:
    """Build the ``tensor`` / ``weight`` / ``bias`` slots both Conv nodes share.

    Args:
        weight_tooltip: Explains the weight shape, which differs between Conv and
            ConvTranspose.

    Returns:
        The three input slots, in declaration order.
    """
    return [
        io.Tensor.Input(
            "tensor",
            tooltip="Input feature map, e.g. (N, C, H, W); unbatched (C, H, W) also works.",
        ),
        io.Tensor.Input("weight", tooltip=weight_tooltip),
        io.Tensor.Input(
            "bias",
            optional=True,
            tooltip="Optional bias of shape (out_channels,); leave unconnected for a bias-free convolution.",
        ),
    ]


def _dims_widget() -> io.Combo.Input:
    """Build the shared ``dims`` widget (which rank of convolution to apply)."""
    return io.Combo.Input(
        "dims",
        options=_DIMS,
        default=2,
        tooltip=(
            "Spatial rank: 1 = sequence (N, C, L), 2 = image (N, C, H, W), "
            "3 = volume (N, C, D, H, W). It also selects Conv1d/Conv2d/Conv3d."
        ),
    )


def _groups_widget() -> io.Int.Input:
    """Build the shared ``groups`` widget (grouped / depthwise convolution)."""
    return io.Int.Input(
        "groups",
        default=1,
        min=1,
        max=4096,
        step=1,
        tooltip="Split channels into this many independent groups; 1 = normal convolution.",
    )


def _promote(tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    """Unify the dtype of activations, weight and bias when both are floating point.

    Mirrors ``BasicLinear``: fp16 activations with fp32 weights run in fp32
    instead of raising a dtype mismatch error; integer tensors are left alone so
    torch can report a meaningful error.

    Args:
        tensor: The input activations.
        weight: The convolution kernel.
        bias: The optional bias.

    Returns:
        ``(tensor, weight, bias)`` sharing one dtype whenever that is legal.
    """
    if tensor.dtype != weight.dtype and tensor.is_floating_point() and weight.is_floating_point():
        dtype = torch.promote_types(tensor.dtype, weight.dtype)
        tensor = tensor.to(dtype=dtype)
        weight = weight.to(dtype=dtype)
        if bias is not None:
            bias = bias.to(dtype=dtype)
    return tensor, weight, bias


def _ensure_min_rank(tensor: torch.Tensor, minimum: int) -> tuple[torch.Tensor, int]:
    """Unsqueeze leading dimensions until ``tensor`` has at least ``minimum`` dims.

    torch's convolution kernels accept ``(N, C, *spatial)`` or ``(C, *spatial)``;
    a bare spatial tensor is unsqueezed so unbatched tensors convolve just like
    batched ones.

    Args:
        tensor: The tensor to prepare.
        minimum: Smallest rank the kernel accepts.

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


def _check_weight(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    groups: int,
    transpose: bool,
    node_name: str,
) -> int:
    """Validate the weight against the input and return ``out_channels``.

    Args:
        tensor: The rank-adjusted input tensor.
        weight: The convolution kernel.
        rank: Spatial rank (1/2/3).
        groups: Number of channel groups.
        transpose: ``True`` for a transposed convolution.
        node_name: Display name of the node, used in error messages.

    Returns:
        ``out_channels`` as declared by the weight.

    Raises:
        RuntimeError: With a message naming the expected weight shape, when the
            kernel rank, the channel count or the group split do not fit.
    """
    if weight.dim() != rank + 2:
        expected = "(out_channels, in_channels/groups, k...)" if not transpose else "(in_channels, out_channels/groups, k...)"
        raise RuntimeError(
            f"{node_name}: a {rank}d convolution needs a {rank + 2}-dimensional weight "
            f"{expected}, but the weight has {weight.dim()} dimensions with shape "
            f"{tuple(weight.shape)}."
        )
    in_channels = int(tensor.shape[-(rank + 1)])
    if transpose:
        weight_in = int(weight.shape[0])
        out_channels = int(weight.shape[1]) * groups
        shape_hint = "ConvTranspose weights are (in_channels, out_channels/groups, k...)"
    else:
        weight_in = int(weight.shape[1]) * groups
        out_channels = int(weight.shape[0])
        shape_hint = (
            "Conv weights are (out_channels, in_channels/groups, k...), so with "
            "groups={} the input must have weight.shape[1] * groups channels".format(groups)
        )
    if weight_in != in_channels:
        raise RuntimeError(
            f"{node_name}: the input has {in_channels} channel(s) but the weight expects "
            f"{weight_in} ({shape_hint})."
        )
    return out_channels


def _apply_padding_mode(
    tensor: torch.Tensor, pad: int, padding_mode: str, rank: int, node_name: str
) -> tuple[torch.Tensor, int]:
    """Pre-pad the input for a non-``zeros`` padding mode.

    ``torch.nn.functional.conv*d`` has no ``padding_mode`` argument (only the
    ``nn.Conv*d`` modules do), so the padding is applied explicitly first and the
    convolution is then called with ``padding=0`` to avoid double padding.

    Args:
        tensor: The rank-adjusted input.
        pad: Padding amount per side.
        padding_mode: One of ``zeros`` / ``reflect`` / ``replicate`` / ``circular``.
        rank: Spatial rank (1/2/3), which sets how many dimension pairs to pad.
        node_name: Display name of the node, used in error messages.

    Returns:
        ``(maybe_padded_tensor, remaining_padding)`` where ``remaining_padding`` is
        what should still be handed to the convolution function.

    Raises:
        RuntimeError: With a readable hint when the mode cannot pad this shape
            (``reflect``/``circular`` need the pad to be smaller than the padded
            dimension, and ``circular`` needs at least 1 spatial dimension).
    """
    if pad == 0 or padding_mode == "zeros":
        return tensor, pad
    pad_list: list[int] = []
    for _ in range(rank):
        pad_list.extend([pad, pad])
    try:
        return F.pad(tensor, pad_list, mode=_PAD_MODE_TO_FUNCTIONAL[padding_mode]), 0
    except RuntimeError as exc:
        raise RuntimeError(
            f"{node_name}: padding_mode='{padding_mode}' cannot pad an input of shape "
            f"{tuple(tensor.shape)} by {pad}. 'reflect' and 'circular' require the "
            f"padding to be smaller than the padded dimension, and 'circular' needs "
            f"at least one spatial dimension. Original error: {exc}"
        ) from exc


class ConvolutionConv(io.ComfyNode):
    """Convolution over 1, 2 or 3 spatial dimensions, with parameter sharing.

    What: ``torch.nn.functional.conv{1,2,3}d`` - slides the kernel across the
          spatial dimensions and computes a dot product at every position. The
          **same** kernel is reused everywhere, which is what "parameter sharing"
          means: the number of weights depends on the channel counts and the
          kernel size, never on the image size, so a 3x3 kernel needs 9 weights
          per input/output channel whether the image is 32x32 or 1024x1024.
          ``dims`` selects the rank; ``groups`` and ``dilation`` turn it into a
          grouped / depthwise / dilated convolution; ``padding_mode`` chooses what
          lies outside the image.
    In:   tensor (TENSOR) - input feature map, usually ``(N, C_in, H, W)``. A bare
          ``(C_in, H, W)`` or ``(H, W)`` tensor is accepted as well: missing
          leading dimensions are added internally and removed from the result.
          weight (TENSOR) - the kernel, ``(out_channels, in_channels/groups,
          kH, kW)`` (``k`` repeated ``dims`` times). ``in_channels`` must equal
          ``weight.shape[1] * groups``.
          bias (TENSOR, optional) - ``(out_channels,)``; leave unconnected for no bias.
          dims (COMBO) - 1, 2 or 3 (default 2): selects Conv1d/Conv2d/Conv3d.
          groups (INT) - default 1; ``C_in`` gives a depthwise convolution,
          anything in between a grouped convolution.
          stride (INT) - kernel step, default 1 (same spatial size when padding
          matches the kernel); values > 1 downsample.
          padding (INT) - implicit padding per side, default 1, which pairs with a
          3x3 kernel to keep the spatial size unchanged.
          padding_mode (COMBO) - ``zeros`` (default), ``reflect``, ``replicate`` or
          ``circular``. Implemented with an explicit ``F.pad`` call followed by a
          zero-padding-free convolution, because the functional API has no such
          argument.
          dilation (INT) - spacing between kernel elements, default 1; > 1 widens
          the receptive field without adding weights.
    Out:  output (TENSOR) - same rank as the input, ``out_channels`` channels and
          spatial size ``floor((size + 2 * padding - dilation * (k - 1) - 1) / stride + 1)``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _conv_schema(
            "ConvolutionConv",
            "Conv",
            "Convolution over 1/2/3 spatial dims with grouped/dilated kernels and zeros/reflect/replicate/circular padding; weight and bias are wired in.",
            inputs=[
                *_weight_slots(
                    "Kernel of shape (out_channels, in_channels/groups, kH, kW) "
                    "(k repeated 'dims' times)."
                ),
                _dims_widget(),
                _groups_widget(),
                io.Int.Input(
                    "stride",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Kernel step; > 1 downsamples the spatial size.",
                ),
                io.Int.Input(
                    "padding",
                    default=1,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Implicit padding per side; 1 with a 3x3 kernel keeps the spatial size.",
                ),
                io.Combo.Input(
                    "padding_mode",
                    options=_PADDING_MODES,
                    default="zeros",
                    tooltip="What lies outside the image: zeros, reflect, replicate or circular.",
                ),
                io.Int.Input(
                    "dilation",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Spacing between kernel elements; > 1 widens the receptive field for free.",
                ),
            ],
            search_aliases=[
                "conv", "conv1d", "conv2d", "conv3d", "convolution", "kernel",
                "filter", "parameter sharing", "grouped convolution",
                "depthwise convolution", "dilated convolution", "atrous convolution",
            ],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dims: int = 2,
        groups: int = 1,
        stride: int = 1,
        padding: int = 1,
        padding_mode: str = "zeros",
        dilation: int = 1,
    ) -> io.NodeOutput:
        rank = int(dims) if int(dims) in (1, 2, 3) else 2
        if rank != int(dims):
            _warn(f"Conv: dims={dims} is not 1/2/3; using 2.")
        group_count = max(1, int(groups))
        step = max(1, int(stride))
        pad = max(0, int(padding))
        dil = max(1, int(dilation))
        if padding_mode not in _PAD_MODE_TO_FUNCTIONAL:
            _warn(f"Conv: padding_mode={padding_mode!r} is unknown; using 'zeros'.")
            padding_mode = "zeros"

        tensor, weight, bias = _promote(tensor, weight, bias)
        prepared, added = _ensure_min_rank(tensor, rank + 1)
        out_channels = _check_weight(prepared, weight, rank, group_count, False, "Conv")
        if bias is not None and bias.numel() != out_channels:
            raise RuntimeError(
                f"Conv: bias has {bias.numel()} element(s) but the weight produces "
                f"{out_channels} channel(s)."
            )

        padded, remaining = _apply_padding_mode(prepared, pad, padding_mode, rank, "Conv")
        convolution = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
        try:
            result = convolution(
                padded, weight, bias, stride=step, padding=remaining,
                dilation=dil, groups=group_count,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Conv: {rank}d convolution failed for an input of shape "
                f"{tuple(tensor.shape)} and a weight of shape {tuple(weight.shape)}. "
                f"Check that 'dims' matches the tensor rank, that in_channels equals "
                f"weight.shape[1] * groups, and that the padding/dilation/stride "
                f"combination leaves a non-empty output. Original error: {exc}"
            ) from exc
        return io.NodeOutput(_drop_added_rank(result, added))


class ConvolutionConvTranspose(io.ComfyNode):
    """Transposed convolution over 1, 2 or 3 spatial dimensions (upsampling).

    What: ``torch.nn.functional.conv_transpose{1,2,3}d`` - the gradient of a
          convolution with respect to its input, commonly (and loosely) called
          "deconvolution". It grows the spatial size instead of shrinking it, so
          it is what decoder / generator networks use to go from a small feature
          map back to a bigger one, in place of a plain upsample followed by a
          convolution. The weight shape is **transposed** relative to ``Conv``:
          ``(in_channels, out_channels/groups, k...)``.
    In:   tensor (TENSOR) - input feature map, usually ``(N, C_in, H, W)``. A bare
          ``(C_in, H, W)`` or ``(H, W)`` tensor is accepted as well: missing
          leading dimensions are added internally and removed from the result.
          weight (TENSOR) - ``(in_channels, out_channels/groups, kH, kW)``
          (``k`` repeated ``dims`` times); note that the channel order is the
          opposite of ``Conv``. ``in_channels`` must equal ``weight.shape[0]``.
          bias (TENSOR, optional) - ``(out_channels,)``; leave unconnected for no bias.
          dims (COMBO) - 1, 2 or 3 (default 2): selects ConvTranspose1d/2d/3d.
          groups (INT) - default 1; ``C_in`` gives a depthwise transposed convolution.
          stride (INT) - output step, default 2 (doubles the spatial size).
          padding (INT) - how much padding the matching forward convolution would
          have used; default 0.
          output_padding (INT) - extra size added on one side of the output,
          default 0; needed only when ``stride > 1`` to reach an exact target size.
          dilation (INT) - spacing between kernel elements, default 1.
    Out:  output (TENSOR) - same rank as the input, with spatial size
          ``(size - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1``.
    Note: there is no ``padding_mode`` widget here: neither
          ``nn.ConvTranspose*d`` nor ``F.conv_transpose*d`` supports one.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _conv_schema(
            "ConvolutionConvTranspose",
            "ConvTranspose",
            "Transposed convolution (upsampling) over 1/2/3 spatial dims with stride/padding/output_padding/groups; weight and bias are wired in.",
            inputs=[
                *_weight_slots(
                    "Kernel of shape (in_channels, out_channels/groups, kH, kW) "
                    "(k repeated 'dims' times) - the channel order is the reverse of Conv."
                ),
                _dims_widget(),
                _groups_widget(),
                io.Int.Input(
                    "stride",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Output step; 2 doubles the spatial size.",
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Padding that the matching forward convolution would have used.",
                ),
                io.Int.Input(
                    "output_padding",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Extra size on one side of the output; use it to hit an exact target size when stride > 1.",
                ),
                io.Int.Input(
                    "dilation",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Spacing between kernel elements; > 1 widens the receptive field for free.",
                ),
            ],
            search_aliases=[
                "convtranspose", "convtranspose1d", "convtranspose2d",
                "convtranspose3d", "transposed convolution", "deconvolution",
                "deconv", "upsample", "upsampling", "fractionally strided convolution",
            ],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dims: int = 2,
        groups: int = 1,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
    ) -> io.NodeOutput:
        rank = int(dims) if int(dims) in (1, 2, 3) else 2
        if rank != int(dims):
            _warn(f"ConvTranspose: dims={dims} is not 1/2/3; using 2.")
        group_count = max(1, int(groups))
        step = max(1, int(stride))
        pad = max(0, int(padding))
        out_pad = max(0, int(output_padding))
        dil = max(1, int(dilation))
        if out_pad >= max(step, dil):
            _warn(
                f"ConvTranspose: output_padding={out_pad} must stay below "
                f"max(stride, dilation)={max(step, dil)}; torch will reject it."
            )

        tensor, weight, bias = _promote(tensor, weight, bias)
        prepared, added = _ensure_min_rank(tensor, rank + 1)
        out_channels = _check_weight(prepared, weight, rank, group_count, True, "ConvTranspose")
        if bias is not None and bias.numel() != out_channels:
            raise RuntimeError(
                f"ConvTranspose: bias has {bias.numel()} element(s) but the weight "
                f"produces {out_channels} channel(s)."
            )

        convolution = {
            1: F.conv_transpose1d,
            2: F.conv_transpose2d,
            3: F.conv_transpose3d,
        }[rank]
        try:
            result = convolution(
                prepared, weight, bias, stride=step, padding=pad,
                output_padding=out_pad, groups=group_count, dilation=dil,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"ConvTranspose: {rank}d transposed convolution failed for an input of "
                f"shape {tuple(tensor.shape)} and a weight of shape {tuple(weight.shape)}. "
                f"Check that 'dims' matches the tensor rank, that in_channels equals "
                f"weight.shape[0], and that output_padding is smaller than "
                f"max(stride, dilation). Original error: {exc}"
            ) from exc
        return io.NodeOutput(_drop_added_rank(result, added))


class ConvolutionUpsample(io.ComfyNode):
    """Upsample a feature map with interpolation - no weights, no learning.

    What: ``torch.nn.functional.interpolate`` - every spatial position is
          enlarged ``scale_factor`` times by copying (``nearest``) or by
          fitting a straight line / plane / hyperplane through the neighbours
          (``linear`` / ``bilinear`` / ``bicubic`` / ``trilinear``). It is the
          parameter-free counterpart of ``ConvTranspose``: growing a feature
          map costs zero weights here, while a learned enlargement needs the
          transposed convolution. The mode must match ``dims`` - linear for
          1-D, bilinear / bicubic for 2-D, trilinear for 3-D, nearest for any
          rank - exactly the contract ``nn.Upsample`` imposes.
    In:   tensor (TENSOR) - input feature map, e.g. ``(N, C, H, W)`` for
          ``dims=2``. A bare ``(C, H, W)`` (or one more leading dim missing)
          is accepted as well: missing leading dimensions are added internally
          and removed from the result.
          dims (COMBO) - 1, 2 or 3 (default 2): the spatial rank.
          mode (COMBO) - ``nearest`` (default), ``linear``, ``bilinear``,
          ``bicubic`` or ``trilinear``; must be legal for ``dims``.
          scale_factor (FLOAT) - spatial multiplier per axis (default 2.0);
          the output size is ``floor(input_size * scale_factor)``.
          align_corners (BOOLEAN) - if ``True``, the corner pixels of the input
          and output are aligned at the same sample point; only used by the
          interpolating modes (default ``False``).
    Out:  output (TENSOR) - same rank, channel count and dtype as the input,
          spatial size scaled by ``scale_factor`` (integer inputs are only
          legal with ``nearest``; interpolating modes cast to float32 first).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _conv_schema(
            "ConvolutionUpsample",
            "Upsample",
            "Interpolation upsampling over 1/2/3 spatial dims (nearest/linear/bilinear/bicubic/trilinear, scale_factor, align_corners); parameter-free.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input feature map, e.g. (N, C, H, W) for dims=2; unbatched inputs work too.",
                ),
                _dims_widget(),
                io.Combo.Input(
                    "mode",
                    options=_INTERP_MODES,
                    default="nearest",
                    tooltip="Interpolation mode; must match dims: 1D = nearest/linear, 2D = nearest/bilinear/bicubic, 3D = nearest/trilinear.",
                ),
                io.Float.Input(
                    "scale_factor",
                    default=2.0,
                    min=0.01,
                    max=64.0,
                    step=0.1,
                    tooltip="Spatial multiplier per axis; the output size is floor(input * scale_factor).",
                ),
                io.Boolean.Input(
                    "align_corners",
                    default=False,
                    tooltip="Align the corner pixels of input and output; only used by the interpolating (non-nearest) modes.",
                ),
            ],
            search_aliases=[
                "upsample", "interpolate", "resize", "upsampling", "nearest",
                "bilinear", "bicubic", "trilinear", "scale", "enlarge",
            ],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        dims: int = 2,
        mode: str = "nearest",
        scale_factor: float = 2.0,
        align_corners: bool = False,
    ) -> io.NodeOutput:
        rank = int(dims) if int(dims) in (1, 2, 3) else 2
        if rank != int(dims):
            _warn(f"Upsample: dims={dims} is not 1/2/3; using 2.")
        if mode not in _INTERP_MODES:
            _warn(f"Upsample: mode={mode!r} is not one of {_INTERP_MODES}; using 'nearest'.")
            mode = "nearest"
        legal = _MODES_PER_RANK[rank]
        if mode not in legal:
            raise ValueError(
                f"Upsample: mode={mode!r} cannot run at dims={rank}; the legal modes are "
                f"{list(legal)} (1D = nearest/linear, 2D = nearest/bilinear/bicubic, "
                f"3D = nearest/trilinear), exactly like nn.Upsample."
            )
        scale = float(scale_factor)
        if scale <= 0.0:
            raise ValueError(f"Upsample: scale_factor must be > 0; got {scale}.")

        # F.interpolate dispatches by the tensor rank, so a rank/dims mismatch
        # would silently reinterpret the input instead of failing - pin it.
        prepared, added = _ensure_min_rank(tensor, rank + 2)
        if prepared.dim() != rank + 2:
            raise ValueError(
                f"Upsample: dims={rank} needs a {rank + 2}-dimensional tensor (N, C, spatial...), "
                f"but the input has {tensor.dim()} dimension(s) with shape {tuple(tensor.shape)}."
            )
        if mode != "nearest" and not prepared.is_floating_point():
            prepared = prepared.to(dtype=torch.float32)
        try:
            result = F.interpolate(
                prepared,
                scale_factor=scale,
                mode=mode,
                align_corners=bool(align_corners) if mode != "nearest" else None,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Upsample: {rank}d interpolation with mode={mode!r} failed for an input of "
                f"shape {tuple(prepared.shape)}. Check that 'dims' matches the tensor rank "
                f"and that the mode is legal for it. Original error: {exc}"
            ) from exc
        return io.NodeOutput(_drop_added_rank(result, added))


class ConvolutionPixelShuffle(io.ComfyNode):
    """Trade channels for spatial resolution: the sub-pixel reshuffle.

    What: ``torch.nn.functional.pixel_shuffle`` - the cheap half of sub-pixel
          convolution (ESPCN): a conv first packs ``r * r`` sub-pixel images
          into the channel dimension, and this node then reshuffles
          ``(N, C * r^2, H, W)`` into ``(N, C, H * r, W * r)`` by moving each
          ``r x r`` block of channels out to its own spatial position. No
          multiplication happens at all - it is a pure view/permute of the
          data, which is why it is the standard way to upsample cheaply after
          a convolution. Inherently 2-D: ``F.pixel_shuffle`` only exists for
          images.
    In:   tensor (TENSOR) - ``(N, C * r^2, H, W)``; the channel count must be
          divisible by ``r^2``. A bare ``(C, r^2, ...)`` / ``(C, H, W)`` tensor
          is accepted as well: missing leading dimensions are added internally
          and removed from the result.
          r (INT) - the upscale factor per axis (default 2, i.e. ``r^2 = 4``
          channels per output pixel and a 2x2 spatial gain).
    Out:  output (TENSOR) - ``(N, C, H * r, W * r)``, same dtype as the input.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _conv_schema(
            "ConvolutionPixelShuffle",
            "Pixel Shuffle",
            "Reshuffle (N, C*r^2, H, W) into (N, C, H*r, W*r): channels become sub-pixels, the parameter-free half of sub-pixel convolution.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input of shape (N, C*r^2, H, W); the channel count must be divisible by r^2.",
                ),
                io.Int.Input(
                    "r",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Upscale factor per spatial axis; channels shrink by r^2, spatial size grows by r.",
                ),
            ],
            search_aliases=["pixel shuffle", "pixelshuffle", "sub pixel", "sub-pixel convolution", "espcn", "upsample", "reshuffle"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, r: int = 2) -> io.NodeOutput:
        factor = max(1, int(r))
        prepared, added = _ensure_min_rank(tensor, 4)
        if prepared.dim() != 4:
            raise ValueError(
                f"Pixel Shuffle: F.pixel_shuffle needs a 4-dimensional (N, C*r^2, H, W) "
                f"tensor, but the input has {tensor.dim()} dimension(s) with shape "
                f"{tuple(tensor.shape)}; it is an inherently 2-D operation."
            )
        channels = int(prepared.shape[1])
        if channels % (factor * factor) != 0:
            raise ValueError(
                f"Pixel Shuffle: the input has {channels} channel(s), which must be divisible "
                f"by r^2 = {factor * factor} (r={factor}); a conv should first pack r*r "
                f"sub-pixels into the channels."
            )
        try:
            result = F.pixel_shuffle(prepared, factor)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Pixel Shuffle: F.pixel_shuffle failed for an input of shape "
                f"{tuple(prepared.shape)} with r={factor}. Original error: {exc}"
            ) from exc
        return io.NodeOutput(_drop_added_rank(result, added))


class ConvolutionPixelUnshuffle(io.ComfyNode):
    """The inverse of Pixel Shuffle: spatial resolution back into channels.

    What: ``torch.nn.functional.pixel_unshuffle`` - the exact inverse of
          ``Pixel Shuffle``: ``(N, C, H * r, W * r)`` becomes
          ``(N, C * r^2, H, W)`` by folding every ``r x r`` spatial block back
          into the channel dimension. Wiring ``Pixel Shuffle → Pixel
          Unshuffle`` with the same ``r`` is a lossless round trip (the smoke
          tester pins this), and the pair is how an encoder undoes what its
          decoder did - or how two networks exchange data at two spatial
          scales without any interpolation.
    In:   tensor (TENSOR) - ``(N, C, H, W)``; the spatial sizes must be
          divisible by ``r``. Unbatched ``(C, H, W)`` inputs work too: missing
          leading dimensions are added internally and removed from the result.
          r (INT) - the downscale factor per axis (default 2, i.e. spatial size
          halves and channels grow by ``r^2 = 4``).
    Out:  output (TENSOR) - ``(N, C * r^2, H / r, W / r)``, same dtype as the
          input.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _conv_schema(
            "ConvolutionPixelUnshuffle",
            "Pixel Unshuffle",
            "Reshuffle (N, C, H*r, W*r) into (N, C*r^2, H, W): the exact inverse of Pixel Shuffle.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input of shape (N, C, H, W); the spatial sizes must be divisible by r.",
                ),
                io.Int.Input(
                    "r",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Downscale factor per spatial axis; spatial size shrinks by r, channels grow by r^2.",
                ),
            ],
            search_aliases=["pixel unshuffle", "pixelunshuffle", "unshuffle", "sub pixel", "downscale", "space to depth"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, r: int = 2) -> io.NodeOutput:
        factor = max(1, int(r))
        prepared, added = _ensure_min_rank(tensor, 4)
        if prepared.dim() != 4:
            raise ValueError(
                f"Pixel Unshuffle: F.pixel_unshuffle needs a 4-dimensional (N, C, H, W) "
                f"tensor, but the input has {tensor.dim()} dimension(s) with shape "
                f"{tuple(tensor.shape)}; it is an inherently 2-D operation."
            )
        spatial = tuple(prepared.shape[2:])
        if any(size % factor != 0 for size in spatial):
            raise ValueError(
                f"Pixel Unshuffle: the spatial sizes {spatial} must all be divisible by "
                f"r={factor}; the node folds every {factor}x{factor} spatial block back "
                f"into the channels."
            )
        try:
            result = F.pixel_unshuffle(prepared, factor)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Pixel Unshuffle: F.pixel_unshuffle failed for an input of shape "
                f"{tuple(prepared.shape)} with r={factor}. Original error: {exc}"
            ) from exc
        return io.NodeOutput(_drop_added_rank(result, added))


#: Every node this module registers, in node-library order.
CONVOLUTION_NODES: list[type[io.ComfyNode]] = [
    ConvolutionConv,
    ConvolutionConvTranspose,
    ConvolutionUpsample,
    ConvolutionPixelShuffle,
    ConvolutionPixelUnshuffle,
]


class ConvolutionExtension(ComfyExtension):
    """Registers the core Convolution node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(CONVOLUTION_NODES)


async def comfy_entrypoint() -> ConvolutionExtension:
    return ConvolutionExtension()
