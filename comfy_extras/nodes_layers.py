"""Core network-layer nodes (reform step 2).

Provides 8 shape-agnostic "Basic" layer nodes that operate on the generic
``TENSOR`` slot type:

* affine layers      - ``Linear``, ``Embedding``
* shape manipulation - ``Flatten``, ``Reshape``, ``Broadcast``
* tensor combination - ``Concat``, ``Add``, ``Multiply``

Every node is stateless: learnable parameters (``weight`` / ``bias``) are wired
in as tensors through input slots instead of being created inside the node, so
one node can drive any checkpoint and nothing is cached between executions.
Widgets only expose the few hyper-parameters a user may reasonably want to
tweak, and every widget default already produces a valid output.

The nodes preserve the input dtype/device (``Linear`` deliberately promotes when
``tensor`` and ``weight`` disagree, see its docstring), never round-trip through
host memory, and are registered as core nodes because this file lives in
``comfy_extras``.
"""

import re

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Basic"


def _basic_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by every Basic layer node.

    All Basic nodes return exactly one ``TENSOR`` output named ``output``; only
    the inputs differ, so callers pass the already-built input list.

    Args:
        node_id: Globally unique, core-safe node id (``Basic`` prefix).
        display_name: Short name shown in the node library (e.g. ``Linear``).
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


def _parse_shape(text: str) -> tuple[int, ...] | None:
    """Parse a shape string such as ``"2,3"`` into a tuple of ints.

    Accepts comma, semicolon, whitespace and full-width separators so that a
    shape typed on a Chinese IME (``"2，3"``) still works.

    Args:
        text: The raw widget value.

    Returns:
        The parsed shape, or ``None`` when the string is empty or contains a
        non-integer entry; callers treat ``None`` as "leave the tensor alone".
    """
    if text is None:
        return None
    normalized = str(text).replace("，", ",").replace("；", ",").replace(";", ",")
    parts = [part for part in re.split(r"[,\s]+", normalized) if part]
    if not parts:
        return None
    dims: list[int] = []
    for part in parts:
        try:
            dims.append(int(part))
        except ValueError:
            return None
    return tuple(dims)


def _clamp_dim(dim: int, rank: int) -> int:
    """Clamp ``dim`` into ``[-rank, rank - 1]`` so a bad widget never crashes.

    Args:
        dim: The user supplied dimension (may be negative or out of range).
        rank: Number of dimensions of the tensor the dimension applies to.

    Returns:
        A dimension that is valid for a tensor of ``rank`` dimensions.
    """
    return max(-rank, min(rank - 1, dim))


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


class BasicLinear(io.ComfyNode):
    """Fully-connected (dense) layer: ``output = tensor @ weight.T + bias``.

    What: applies an affine transform along the last dimension, i.e.
          ``torch.nn.Linear`` in functional form. The parameters are not stored
          inside the node - ``weight`` and the optional ``bias`` are wired in as
          tensors, so the same node can run any checkpoint. A ``Linear`` followed
          by an activation node is the textbook "Dense" layer.
    In:   tensor (TENSOR) - input activations of shape ``(..., in_features)``.
          weight (TENSOR) - ``(out_features, in_features)``; the last dimension
          must match ``tensor``'s last dimension.
          bias   (TENSOR, optional) - ``(out_features,)`` (or anything broadcast
          compatible with the result); leave unconnected for a bias-free layer.
    Out:  output (TENSOR) - ``(..., out_features)``. When ``tensor`` and
          ``weight`` are both floating point but of different dtypes they are
          promoted to their common dtype first (e.g. fp16 activations with fp32
          weights run in fp32) instead of raising a dtype mismatch error.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicLinear",
            "Linear",
            "Fully-connected layer: tensor @ weight.T (+ bias); weight/bias are wired in as tensors.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input activations, shape (..., in_features).",
                ),
                io.Tensor.Input(
                    "weight",
                    tooltip="Weight matrix of shape (out_features, in_features).",
                ),
                io.Tensor.Input(
                    "bias",
                    optional=True,
                    tooltip="Optional bias of shape (out_features,); leave unconnected for no bias.",
                ),
            ],
            search_aliases=["dense", "fully connected", "fc", "affine", "projection", "matmul"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        if tensor.dtype != weight.dtype and tensor.is_floating_point() and weight.is_floating_point():
            dtype = torch.promote_types(tensor.dtype, weight.dtype)
            tensor = tensor.to(dtype=dtype)
            weight = weight.to(dtype=dtype)
            if bias is not None:
                bias = bias.to(dtype=dtype)
        return io.NodeOutput(F.linear(tensor, weight, bias))


class BasicEmbedding(io.ComfyNode):
    """Embedding lookup table: gathers rows of ``weight`` by index.

    What: ``torch.nn.Embedding`` in functional form - a pure table lookup with no
          learnable state inside the node. Useful to turn token ids (or any
          integer feature ids) into dense vectors.
    In:   tensor (TENSOR) - index tensor of any shape; cast to ``int64``
          internally, so float tensors carrying integral values are accepted.
          weight (TENSOR) - the table, ``(num_embeddings, embedding_dim)``.
    Out:  output (TENSOR) - ``tensor.shape + (embedding_dim,)``, with the dtype
          and device of ``weight``. Indices outside ``[0, num_embeddings)`` raise
          a normal torch error, because that always means the table is wrong.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicEmbedding",
            "Embedding",
            "Lookup table: gathers rows of weight by integer index; the table is wired in as a tensor.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Integer index tensor of any shape (cast to int64 internally).",
                ),
                io.Tensor.Input(
                    "weight",
                    tooltip="Lookup table of shape (num_embeddings, embedding_dim).",
                ),
            ],
            search_aliases=["lookup", "table", "word embedding", "gather", "token"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, weight: torch.Tensor) -> io.NodeOutput:
        indices = tensor.long() if tensor.dtype != torch.long else tensor
        return io.NodeOutput(F.embedding(indices, weight))


class BasicFlatten(io.ComfyNode):
    """Flattens a contiguous range of dimensions into a single dimension.

    What: ``torch.flatten`` over ``start_dim``..``end_dim``. The defaults
          (``start_dim=1``, ``end_dim=-1``) turn a batched tensor such as
          ``(N, C, H, W)`` into ``(N, C*H*W)``, which is what a following
          ``Linear`` layer expects. Like ``torch.flatten`` this returns a view
          whenever the memory layout allows it.
    In:   tensor (TENSOR) - tensor with at least 1 dimension; dtype/device kept.
          start_dim (INT) - first dimension to flatten (0..4).
          end_dim   (INT) - last dimension to flatten (-4..4); if it resolves to
                            a dimension before ``start_dim`` the two are swapped.
    Out:  output (TENSOR) - same number of elements, rank reduced to
          ``rank - (end_dim - start_dim)``. A 0-dim input is returned unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicFlatten",
            "Flatten",
            "Flatten dimensions start_dim..end_dim into one; defaults flatten (N, C, H, W) to (N, C*H*W).",
            inputs=[
                io.Tensor.Input("tensor", tooltip="Tensor to flatten."),
                io.Int.Input(
                    "start_dim",
                    default=1,
                    min=0,
                    max=4,
                    step=1,
                    tooltip="First dimension to flatten (0 keeps the batch dimension).",
                ),
                io.Int.Input(
                    "end_dim",
                    default=-1,
                    min=-4,
                    max=4,
                    step=1,
                    tooltip="Last dimension to flatten (-1 = last dimension).",
                ),
            ],
            search_aliases=["collapse", "flatten dims", "view"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        start_dim: int,
        end_dim: int,
    ) -> io.NodeOutput:
        rank = tensor.dim()
        if rank == 0:
            _warn("Flatten received a 0-dim tensor; passing it through unchanged.")
            return io.NodeOutput(tensor)
        start = start_dim if start_dim >= 0 else start_dim + rank
        end = end_dim if end_dim >= 0 else end_dim + rank
        start = max(0, min(rank - 1, start))
        end = max(0, min(rank - 1, end))
        if start > end:
            start, end = end, start
        return io.NodeOutput(torch.flatten(tensor, start_dim=start, end_dim=end))


class BasicReshape(io.ComfyNode):
    """Reshapes a tensor to an explicit target shape.

    What: ``torch.reshape`` driven by a shape string. One dimension may be
          ``-1``, meaning "infer this one from the element count". This is the
          core-node counterpart of the teaching node ``CdlReshape`` in the
          ``d2l`` group; both stay available.
    In:   tensor (TENSOR) - tensor to reshape; dtype/device kept.
          target_shape (STRING) - comma separated sizes, e.g. ``"1,-1"`` (the
          default collapses any tensor into a single row) or ``"2,3,-1"``.
    Out:  output (TENSOR) - the reshaped tensor (a view when possible). If the
          string cannot be parsed, or the shape is incompatible with the element
          count, the original tensor is returned unchanged with a printed
          warning, so a workflow is never broken by a typo.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicReshape",
            "Reshape",
            'Reshape to a comma separated shape string, e.g. "1,-1"; -1 infers one dimension.',
            inputs=[
                io.Tensor.Input("tensor", tooltip="Tensor to reshape."),
                io.String.Input(
                    "target_shape",
                    default="1,-1",
                    placeholder="e.g. 2,3,-1",
                    tooltip='Comma separated target sizes; "-1" infers one dimension.',
                ),
            ],
            search_aliases=["view", "shape", "squeeze", "unsqueeze"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, target_shape: str) -> io.NodeOutput:
        shape = _parse_shape(target_shape)
        if shape is None:
            _warn(f"Reshape could not parse target_shape={target_shape!r}; returning the input unchanged.")
            return io.NodeOutput(tensor)
        try:
            return io.NodeOutput(torch.reshape(tensor, shape))
        except RuntimeError as exc:
            _warn(f"Reshape to {shape} failed ({exc}); returning the input unchanged.")
            return io.NodeOutput(tensor)


class BasicBroadcast(io.ComfyNode):
    """Broadcasts a tensor to an explicit target shape.

    What: ``torch.broadcast_to`` driven by a shape string, i.e. the read-only
          "expand to" operation. Every target dimension must either match the
          input dimension or extend an input dimension of size 1. Because the
          result is a view, no data is copied. This is the core-node counterpart
          of the teaching node ``CdlBroadcast`` in the ``d2l`` group.
    In:   tensor (TENSOR) - tensor to broadcast; dtype/device kept.
          target_shape (STRING) - comma separated, non-negative sizes, e.g.
          ``"2,3"`` (the default) or ``"3,1,4"``.
    Out:  output (TENSOR) - a broadcast view with the requested shape. If the
          string cannot be parsed, or the input is not broadcast compatible, the
          original tensor is returned unchanged with a printed warning, so a
          workflow is never broken by a typo. Note that ``-1`` is not allowed
          here, since nothing can be inferred from a target-only shape.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicBroadcast",
            "Broadcast",
            'Broadcast (view) a tensor to a comma separated shape string, e.g. "2,3".',
            inputs=[
                io.Tensor.Input("tensor", tooltip="Tensor to broadcast."),
                io.String.Input(
                    "target_shape",
                    default="2,3",
                    placeholder="e.g. 3,1,4",
                    tooltip="Comma separated, non-negative target sizes (no -1).",
                ),
            ],
            search_aliases=["expand", "broadcast to", "repeat", "tile"],
        )

    @classmethod
    def execute(cls, tensor: torch.Tensor, target_shape: str) -> io.NodeOutput:
        shape = _parse_shape(target_shape)
        if shape is None:
            _warn(f"Broadcast could not parse target_shape={target_shape!r}; returning the input unchanged.")
            return io.NodeOutput(tensor)
        try:
            return io.NodeOutput(torch.broadcast_to(tensor, shape))
        except RuntimeError as exc:
            _warn(f"Broadcast to {shape} failed ({exc}); returning the input unchanged.")
            return io.NodeOutput(tensor)


class BasicConcat(io.ComfyNode):
    """Concatenates two tensors along one dimension.

    What: ``torch.cat`` of exactly two tensors - the building block for merging
          branches (multi-input feature fusion, encoder/decoder skips that widen
          instead of summing, ...). Longer chains are built by stacking nodes.
    In:   a, b (TENSOR) - tensors of the same rank that agree in every dimension
          except the concatenation one; dtype and device must match.
          dim (INT) - dimension to concatenate along, clamped into
          ``[-rank, rank - 1]`` so an out-of-range widget value never crashes.
    Out:  output (TENSOR) - same rank, with dimension ``dim`` equal to the sum of
          both inputs. A shape mismatch is a wiring error and is reported by
          torch, not silently swallowed.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicConcat",
            "Concat",
            "Concatenate two tensors along one dimension (torch.cat).",
            inputs=[
                io.Tensor.Input("a", tooltip="First tensor."),
                io.Tensor.Input("b", tooltip="Second tensor; same rank/dtype/device as a."),
                io.Int.Input(
                    "dim",
                    default=-1,
                    min=-4,
                    max=4,
                    step=1,
                    tooltip="Dimension to concatenate along (clamped to the tensor rank).",
                ),
            ],
            search_aliases=["concatenate", "cat", "merge", "join", "fusion"],
        )

    @classmethod
    def execute(cls, a: torch.Tensor, b: torch.Tensor, dim: int) -> io.NodeOutput:
        rank = max(a.dim(), b.dim())
        if rank == 0:
            _warn("Concat received 0-dim tensors; returning the first input unchanged.")
            return io.NodeOutput(a)
        return io.NodeOutput(torch.cat((a, b), dim=_clamp_dim(dim, rank)))


class BasicAdd(io.ComfyNode):
    """Element-wise addition of two tensors, with broadcasting.

    What: ``a + b``. This single node covers the classic "residual" and "skip
          connection" patterns (``output = input + branch(input)``): run the
          branch, then add it back to the untouched input here.
    In:   a, b (TENSOR) - tensors of the same shape, or shapes that broadcast
          against each other (e.g. ``(N, C, H, W) + (N, C, 1, 1)``). The mix of
          dtypes follows normal torch type promotion.
    Out:  output (TENSOR) - the broadcast shape of both inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicAdd",
            "Add",
            "Element-wise addition of two tensors (broadcasting); use it for residual / skip connections.",
            inputs=[
                io.Tensor.Input("a", tooltip="First tensor."),
                io.Tensor.Input("b", tooltip="Second tensor; same shape or broadcast compatible."),
            ],
            search_aliases=["sum", "plus", "residual", "skip connection", "shortcut", "bias add"],
        )

    @classmethod
    def execute(cls, a: torch.Tensor, b: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(a + b)


class BasicMultiply(io.ComfyNode):
    """Element-wise multiplication of two tensors, with broadcasting.

    What: ``a * b``. Used for gating/attention weighting (``features * mask``),
          per-channel scaling (``features * (N, C, 1, 1)``) and any multiplicative
          merge of two branches.
    In:   a, b (TENSOR) - tensors of the same shape, or shapes that broadcast
          against each other. The mix of dtypes follows normal torch type
          promotion.
    Out:  output (TENSOR) - the broadcast shape of both inputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _basic_schema(
            "BasicMultiply",
            "Multiply",
            "Element-wise multiplication of two tensors (broadcasting); useful for gating or scaling.",
            inputs=[
                io.Tensor.Input("a", tooltip="First tensor."),
                io.Tensor.Input("b", tooltip="Second tensor; same shape or broadcast compatible."),
            ],
            search_aliases=["mul", "product", "scale", "gate", "mask", "weighting"],
        )

    @classmethod
    def execute(cls, a: torch.Tensor, b: torch.Tensor) -> io.NodeOutput:
        return io.NodeOutput(a * b)


BASIC_LAYER_NODES: list[type[io.ComfyNode]] = [
    BasicLinear,
    BasicEmbedding,
    BasicFlatten,
    BasicReshape,
    BasicBroadcast,
    BasicConcat,
    BasicAdd,
    BasicMultiply,
]


class NetworkLayersExtension(ComfyExtension):
    """Registers the core Basic network-layer node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(BASIC_LAYER_NODES)


async def comfy_entrypoint() -> NetworkLayersExtension:
    return NetworkLayersExtension()
