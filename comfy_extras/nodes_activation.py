"""Core activation-function nodes (reform step 1).

Provides 14 shape-agnostic activation nodes that operate on the generic
``TENSOR`` slot type.  Every node has exactly one input slot and one output
slot, carries no learnable parameters, and exposes only the few
hyper-parameters that a user may reasonably want to tweak as node widgets.

The nodes preserve the input dtype and device, never copy through host memory,
and are registered as core nodes because this file lives in ``comfy_extras``.
"""

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY = "Activation"


def _activation_schema(
    node_id: str,
    display_name: str,
    description: str,
    extra_inputs: tuple = (),
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by every activation node.

    Each activation node takes exactly one ``TENSOR`` input named ``tensor`` and
    returns exactly one ``TENSOR`` output named ``output``; only the
    hyper-parameter widgets differ, and those are supplied via ``extra_inputs``.

    Args:
        node_id: Globally unique, core-safe node id (``Activation`` prefix).
        display_name: Short name shown in the node library (e.g. ``Sigmoid``).
        description: Tooltip shown when hovering over the node.
        extra_inputs: Hyper-parameter widget inputs specific to this node.
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
        inputs=[io.Tensor.Input("tensor"), *extra_inputs],
        outputs=[io.Tensor.Output(display_name="output")],
    )


class ActivationSigmoid(io.ComfyNode):
    """Sigmoid activation, computed element-wise as ``1 / (1 + exp(-x))``.

    What: squashes every element into the open interval (0, 1).
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, values in (0, 1).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationSigmoid",
            "Sigmoid",
            "Element-wise sigmoid: 1 / (1 + exp(-x)); output range (0, 1).",
            search_aliases=["logistic"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(torch.sigmoid(tensor))


class ActivationTanh(io.ComfyNode):
    """Hyperbolic tangent activation, ``tanh(x)``.

    What: squashes every element into the open interval (-1, 1).
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, values in (-1, 1).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationTanh",
            "Tanh",
            "Element-wise hyperbolic tangent; output range (-1, 1).",
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(torch.tanh(tensor))


class ActivationReLU(io.ComfyNode):
    """Rectified linear unit, ``max(0, x)``.

    What: zeroes negative elements, passes non-negative ones unchanged.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, all values >= 0.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationReLU",
            "ReLU",
            "Element-wise rectified linear unit: max(0, x).",
            search_aliases=["rectified"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.relu(tensor))


class ActivationLeakyReLU(io.ComfyNode):
    """Leaky ReLU, ``max(0, x) + negative_slope * min(0, x)``.

    What: like ReLU but keeps a small slope for negative elements, which avoids
          dead units.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
          negative_slope (FLOAT) - slope applied to negative values.
    Out:  output (TENSOR) - same shape/dtype/device.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationLeakyReLU",
            "Leaky ReLU",
            "ReLU variant that keeps a small slope (negative_slope) for x < 0.",
            extra_inputs=[
                io.Float.Input(
                    "negative_slope",
                    default=0.01,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Slope applied to negative values (0 gives plain ReLU).",
                )
            ],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, negative_slope: float) -> io.NodeOutput:
        return io.NodeOutput(F.leaky_relu(tensor, negative_slope=negative_slope))


class ActivationELU(io.ComfyNode):
    """Exponential linear unit, ``x`` for ``x > 0`` and ``alpha * (exp(x) - 1)`` below.

    What: smooths negative values toward ``-alpha`` while keeping positives
          unchanged.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
          alpha  (FLOAT) - saturation value for negative inputs.
    Out:  output (TENSOR) - same shape/dtype/device, values >= -alpha.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationELU",
            "ELU",
            "Exponential linear unit: x if x > 0 else alpha * (exp(x) - 1).",
            extra_inputs=[
                io.Float.Input(
                    "alpha",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Value that negative inputs saturate towards.",
                )
            ],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, alpha: float) -> io.NodeOutput:
        return io.NodeOutput(F.elu(tensor, alpha=alpha))


class ActivationSELU(io.ComfyNode):
    """Scaled exponential linear unit with the standard fixed constants.

    What: self-normalizing activation using the canonical
          ``scale=1.05070098`` / ``alpha=1.67326324`` pair; no user parameters.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationSELU",
            "SELU",
            "Self-normalizing ELU with the standard scale/alpha constants.",
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.selu(tensor))


class ActivationGELU(io.ComfyNode):
    """Gaussian error linear unit.

    What: smooth, non-monotonic activation widely used by transformers.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
          approximate (COMBO) - "none" uses the exact erf formulation,
          "tanh" uses the cheaper tanh approximation.
    Out:  output (TENSOR) - same shape/dtype/device.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationGELU",
            "GELU",
            "Gaussian error linear unit; choose the exact or tanh approximation.",
            extra_inputs=[
                io.Combo.Input(
                    "approximate",
                    options=["none", "tanh"],
                    default="none",
                    tooltip='"none" = exact erf form, "tanh" = faster approximation.',
                )
            ],
            search_aliases=["gaussian"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, approximate: str) -> io.NodeOutput:
        return io.NodeOutput(F.gelu(tensor, approximate=approximate))


class ActivationSiLU(io.ComfyNode):
    """Sigmoid linear unit (a.k.a. swish), ``x * sigmoid(x)``.

    What: smooth, non-monotonic activation with a single minimum.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, values >= ~-0.2785.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationSiLU",
            "SiLU",
            "Sigmoid linear unit (swish): x * sigmoid(x).",
            search_aliases=["swish", "silu"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.silu(tensor))


class ActivationMish(io.ComfyNode):
    """Mish activation, ``x * tanh(softplus(x))``.

    What: smooth, non-monotonic activation; unbounded above, bounded below.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationMish",
            "Mish",
            "Mish activation: x * tanh(softplus(x)).",
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        # torch.nn.functional.mish is the exact same formula; spelled out here
        # so the node works on any torch build that has softplus/tanh.
        return io.NodeOutput(tensor * torch.tanh(F.softplus(tensor)))


class ActivationSoftplus(io.ComfyNode):
    """Softplus, a smooth approximation of ReLU: ``log(1 + exp(x))``.

    What: smooth, strictly positive activation using torch defaults
          (``beta=1``, ``threshold=20``; above the threshold it is linear).
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, all values > 0.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationSoftplus",
            "Softplus",
            "Smooth ReLU approximation: log(1 + exp(x)) (beta=1, threshold=20).",
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.softplus(tensor))


class ActivationReLU6(io.ComfyNode):
    """ReLU clamped to the [0, 6] range.

    What: ``min(max(0, x), 6)``; commonly used in mobile/quantized networks.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device, values in [0, 6].
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationReLU6",
            "ReLU6",
            "ReLU clamped to [0, 6]: min(max(0, x), 6).",
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.hardtanh(tensor, 0.0, 6.0))


class ActivationHardSwish(io.ComfyNode):
    """Hard swish, the piecewise-linear approximation of SiLU.

    What: ``x * relu6(x + 3) / 6``; cheaper than SiLU and quantization friendly.
    In:   tensor (TENSOR) - arbitrary N-D tensor; dtype/device preserved.
    Out:  output (TENSOR) - same shape/dtype/device.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationHardSwish",
            "Hard Swish",
            "Piecewise-linear SiLU approximation: x * relu6(x + 3) / 6.",
            search_aliases=["hardswish"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(F.hardswish(tensor))


class ActivationIdentity(io.ComfyNode):
    """Identity activation: forwards the input unchanged.

    What: a zero-copy pass-through, useful as a placeholder or to keep a graph
          shape stable while swapping activations around.
    In:   tensor (TENSOR) - arbitrary N-D tensor.
    Out:  output (TENSOR) - the very same tensor object (no copy, no compute).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationIdentity",
            "Identity",
            "Pass-through activation; returns the input tensor unchanged.",
            search_aliases=["linear", "none"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(tensor)


class ActivationSoftmax(io.ComfyNode):
    """Softmax normalization along a chosen dimension.

    What: rescales values along ``dim`` so that they sum to 1 (a probability
          distribution over that dimension).
    In:   tensor (TENSOR) - arbitrary N-D tensor.
          dim    (INT)    - dimension to normalize over; clamped into
                            ``[-rank, rank - 1]`` so an out-of-range value
                            never breaks a workflow.
    Out:  output (TENSOR) - same shape/dtype/device, summing to 1 along ``dim``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _activation_schema(
            "ActivationSoftmax",
            "Softmax",
            "Normalize values along one dimension so that they sum to 1.",
            extra_inputs=[
                io.Int.Input(
                    "dim",
                    default=-1,
                    min=-4,
                    max=4,
                    step=1,
                    tooltip="Dimension to normalize over (clamped to the tensor rank).",
                )
            ],
            search_aliases=["normalize", "logits"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, dim: int) -> io.NodeOutput:
        rank = tensor.dim()
        if rank == 0:
            # A scalar's distribution is trivially [1.0]; softmax would raise.
            return io.NodeOutput(torch.ones_like(tensor))
        dim = max(-rank, min(rank - 1, dim))
        return io.NodeOutput(F.softmax(tensor, dim=dim))


ACTIVATION_NODES: list[type[io.ComfyNode]] = [
    ActivationSigmoid,
    ActivationTanh,
    ActivationReLU,
    ActivationLeakyReLU,
    ActivationELU,
    ActivationSELU,
    ActivationGELU,
    ActivationSiLU,
    ActivationMish,
    ActivationSoftplus,
    ActivationReLU6,
    ActivationHardSwish,
    ActivationIdentity,
    ActivationSoftmax,
]


class ActivationExtension(ComfyExtension):
    """Registers the core Activation node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(ACTIVATION_NODES)


async def comfy_entrypoint() -> ActivationExtension:
    return ActivationExtension()
