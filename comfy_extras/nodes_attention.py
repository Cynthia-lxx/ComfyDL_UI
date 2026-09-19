"""Core attention nodes (reform step 7).

Provides 4 attention nodes that operate on the generic ``TENSOR`` slot type:

* projection attention  - ``Multihead`` (q/k/v wired separately)
* conveniences         - ``Self`` (q=k=v=x), ``Cross`` (q from x, k/v from context)
* assembled block      - ``TransformerEncoderBlock`` (LN -> MHA -> Add -> LN -> FFN -> Add)

Every node is stateless: the learnable parameters (the four projection
weight/bias sets, the two FFN matrices, the two LayerNorm affines) are wired in
as tensors through input slots instead of being created inside the node, so one
node can drive any checkpoint and nothing is cached between executions. The
train/eval switch travels through the ``mode`` link (a Training Mode node), and
the attention-weight dropout draws its mask from a local ``torch.Generator``
seeded by the ``seed`` widget - the same seed reproduces the same output bit for
bit, which keeps ComfyUI's caching meaningful.

The attention math is the standard multi-head formula, implemented explicitly
(softmax(Q K^T / sqrt(d) + mask)) rather than through
``F.scaled_dot_product_attention`` so that the mask semantics and the seeded
dropout behave identically on every backend. A boolean mask (``True`` = attend,
``False`` = blocked) is converted to an additive mask using the dtype's finite
minimum instead of ``-inf``, which avoids NaN rows when a query is fully masked.

The nodes promote floating-point inputs/weights to their common dtype (matching
``BasicLinear``), never round-trip through host memory, and are registered as
core nodes because this file lives in ``comfy_extras``.
"""

import math

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# The mode link, its normalisation and the seeded dropout mask are shared with
# the normalization family; importing them keeps the "train/eval travels through
# a link" contract in exactly one place instead of drifting between files.
# Absolute import on purpose - the extras loader names modules by file path, so a
# relative import would have no parent package (same convention as nodes_latent).
from comfy_extras.nodes_normalization import (
    MODE_TRAIN,
    _broadcast_stat,
    _dropout,
    _mode_input,
    _normalize_mode,
    _warn,
)

CATEGORY = "Network & Layers/Attention"

#: Options for the activation between the two FFN linear layers of a block.
FFN_ACTIVATION_OPTIONS = ("relu", "gelu")

_FF = 0xFFFFFFFFFFFFFFFF  #: widget max for a 64-bit seed


def _attention_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by every attention node.

    All attention nodes return exactly one ``TENSOR`` output named ``output``; only
    the inputs differ, so callers pass the already-built input list.

    Args:
        node_id: Globally unique, core-safe node id.
        display_name: Short name shown in the node library (e.g. ``Self-Attention``).
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


# --------------------------------------------------------------------------- #
# Shared slot / widget builders
# --------------------------------------------------------------------------- #
def _num_heads_input() -> io.Int:
    return io.Int.Input(
        "num_heads",
        default=4,
        min=1,
        max=64,
        step=1,
        tooltip="Number of parallel attention heads; the embedding width (read from the wired weight shapes) must be divisible by this.",
    )


def _dropout_p_input() -> io.Float:
    return io.Float.Input(
        "dropout_p",
        default=0.0,
        min=0.0,
        max=0.9,
        step=0.05,
        tooltip="Dropout probability applied to the attention weights; only active in train mode. 0 disables it.",
    )


def _seed_input() -> io.Int:
    return io.Int.Input(
        "seed",
        default=0,
        min=0,
        max=_FF,
        control_after_generate=True,
        tooltip="Seed of the attention-weight dropout draw: the same seed reproduces the same output. Only used in train mode with dropout_p > 0.",
    )


def _attn_mode_input() -> io.Input:
    return _mode_input(
        tooltip="Link the mode output of a Training Mode node: 'train' applies dropout to the attention weights, 'eval' skips it. Unconnected means 'train'.",
    )


def _weight_input(name: str, width: str) -> io.Tensor.Input:
    return io.Tensor.Input(
        name,
        tooltip=f"Projection weight of shape (out_features, {width}), wired in as a tensor.",
    )


def _bias_input(name: str, width: str) -> io.Tensor.Input:
    return io.Tensor.Input(
        name,
        optional=True,
        tooltip=f"Optional bias of shape ({width},); leave unconnected for no bias.",
    )


def _mask_input() -> io.Tensor.Input:
    return io.Tensor.Input(
        "mask",
        optional=True,
        tooltip="Optional attention mask, broadcastable to (batch, heads, query_len, key_len): boolean (True = attend, False = blocked) or additive float. Leave unconnected for full attention.",
    )


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def _multi_head_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    v_bias: torch.Tensor | None = None,
    out_bias: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    num_heads: int = 4,
    dropout_p: float = 0.0,
    training: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Standard multi-head attention with every parameter wired in.

    Computes ``softmax(Q K^T / sqrt(d) + mask) V`` per head after projecting
    ``queries`` / ``keys`` / ``values`` with the wired weights, then re-merges the
    heads and applies the output projection. The same math as
    ``torch.nn.MultiheadAttention`` (``batch_first=True``), in functional form.

    Args:
        queries: ``(..., L, E_in_q)``; leading batch dims shared with keys/values.
        keys: ``(..., S, E_in_k)``.
        values: ``(..., S, E_in_v)``.
        q_weight: ``(E, E_in_q)``; k_weight/v_weight likewise with the same
            output width ``E``.
        out_weight: ``(E_out, E)``.
        q_bias / k_bias / v_bias / out_bias: optional ``(E,)`` / ``(E_out,)``.
        mask: ``None``, or a boolean (``True`` = attend) / additive float tensor
            broadcastable to ``(N, num_heads, L, S)``.
        num_heads: Number of heads; ``E`` must be divisible by it.
        dropout_p: Attention-weight dropout probability (train mode only).
        training: Applies the dropout when true.
        seed: Seed of the dropout mask draw.

    Returns:
        ``(*batch, L, E_out)`` with the dtype/device of the promoted inputs.
    """
    # 1. dtype promotion - the BasicLinear contract: fp16 activations with fp32
    #    weights run in fp32 instead of raising a dtype mismatch error.
    parts = [queries, keys, values, q_weight, k_weight, v_weight, out_weight]
    biases = [q_bias, k_bias, v_bias, out_bias]
    present = [b for b in biases if b is not None]
    if all(t.is_floating_point() for t in parts + present):
        common = parts[0].dtype
        for t in parts[1:] + present:
            common = torch.promote_types(common, t.dtype)
        parts = [t.to(dtype=common) if t.dtype != common else t for t in parts]
        queries, keys, values, q_weight, k_weight, v_weight, out_weight = parts
        biases = [b.to(dtype=common) if b is not None and b.dtype != common else b for b in biases]
        q_bias, k_bias, v_bias, out_bias = biases

    # 2. shape checks - a wiring error must be reported, never swallowed.
    for name, tensor, weight in (
        ("queries", queries, q_weight),
        ("keys", keys, k_weight),
        ("values", values, v_weight),
    ):
        if weight.dim() != 2:
            raise ValueError(
                f"{name[0]}_weight must be 2-D (out_features, in_features); got shape {tuple(weight.shape)}."
            )
        if tensor.dim() < 2 or tensor.shape[-1] != weight.shape[1]:
            raise ValueError(
                f"{name} must end in {weight.shape[1]} feature(s) to match the input width of "
                f"{name[0]}_weight (shape {tuple(weight.shape)}); got shape {tuple(tensor.shape)}."
            )
    E = q_weight.shape[0]
    if k_weight.shape[0] != E or v_weight.shape[0] != E:
        raise ValueError(
            f"q/k/v projection weights must share the same output width, i.e. shape (E, in_features) "
            f"each; got q={E}, k={k_weight.shape[0]}, v={v_weight.shape[0]}."
        )
    if out_weight.dim() != 2 or out_weight.shape[1] != E:
        raise ValueError(
            f"out_weight must be 2-D of shape (out_features, {E}); got shape {tuple(out_weight.shape)}."
        )
    if queries.shape[:-2] != keys.shape[:-2] or queries.shape[:-2] != values.shape[:-2]:
        raise ValueError(
            "queries/keys/values must share the same leading batch dimensions; got "
            f"{tuple(queries.shape[:-2])} vs {tuple(keys.shape[:-2])} vs {tuple(values.shape[:-2])}."
        )
    if E % num_heads != 0:
        raise ValueError(
            f"embed dim ({E}) must be divisible by num_heads ({num_heads})."
        )

    # 3. project and split into heads: (..., L, E) -> (N, heads, L, d).
    batch = queries.shape[:-2]
    L, S = queries.shape[-2], keys.shape[-2]
    heads, d = num_heads, E // num_heads
    q = F.linear(queries, q_weight, q_bias).reshape(-1, L, heads, d).transpose(1, 2)
    k = F.linear(keys, k_weight, k_bias).reshape(-1, S, heads, d).transpose(1, 2)
    v = F.linear(values, v_weight, v_bias).reshape(-1, S, heads, d).transpose(1, 2)

    # 4. scores -> mask -> softmax -> (seeded) dropout -> weighted sum.
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        if mask.dtype == torch.bool:
            # True = attend; use the finite dtype minimum so a fully masked row
            # softmaxes to a uniform distribution instead of NaN.
            blocked = ~mask.to(device=scores.device)
            scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
    attn = torch.softmax(scores, dim=-1)
    if training and 0.0 < dropout_p < 1.0:
        attn = _dropout(attn, dropout_p, seed)
    out = torch.matmul(attn, v)  # (N, heads, L, d)

    # 5. merge heads and project: (N, L, E) -> (..., L, E_out).
    out = out.transpose(1, 2).reshape(-1, L, E)
    out = F.linear(out, out_weight, out_bias)
    return out.reshape(*batch, L, out.shape[-1])


def _ffn_activation(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the activation chosen between the two FFN linear layers.

    An unparsable value falls back to ``relu`` with a printed warning, so a broken
    widget never crashes the graph.

    Args:
        tensor: The hidden activations of the FFN.
        name: One of :data:`FFN_ACTIVATION_OPTIONS`.

    Returns:
        The activated tensor, same shape/dtype/device.
    """
    if name == "gelu":
        return F.gelu(tensor)
    if name != "relu":
        _warn(f"ffn_activation={name!r} is not one of {FFN_ACTIVATION_OPTIONS}; using 'relu'.")
    return F.relu(tensor)


def _block_layer_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    """LayerNorm over the last dimension, affine parameters optional.

    ``weight`` / ``bias`` unconnected means a non-affine ``F.layer_norm``; a wired
    tensor is flattened, cast to ``tensor``'s dtype/device and broadcast when it
    holds a single value. A size mismatch (stale weights from another width)
    falls back to non-affine with a printed warning, the same contract the
    normalization nodes use for their statistics slots.

    Args:
        tensor: ``(..., E)`` to normalize.
        weight: Optional affine scale of shape ``(E,)`` (or a single value).
        bias: Optional affine shift of shape ``(E,)`` (or a single value).
        name: Slot prefix used in warning messages (``ln1`` / ``ln2``).

    Returns:
        The normalized tensor, same shape/dtype/device.
    """
    channels = tensor.shape[-1]
    scale = _broadcast_stat(weight, channels, tensor, f"{name}_weight") if weight is not None else None
    shift = _broadcast_stat(bias, channels, tensor, f"{name}_bias") if bias is not None else None
    return F.layer_norm(tensor, (channels,), scale, shift)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class AttentionMultihead(io.ComfyNode):
    """Multi-head attention over wired q / k / v tensors.

    What: the full ``torch.nn.MultiheadAttention`` math in functional form -
          project ``queries`` / ``keys`` / ``values`` with the wired weights, split
          into ``num_heads`` heads, score with ``Q K^T / sqrt(d)``, apply the
          optional ``mask``, softmax, (in train mode) drop out attention weights,
          re-merge and project with ``out_weight``. The parameters are not stored
          inside the node, so one node can run any checkpoint. The optional mask
          is boolean (``True`` = attend) or additive float, broadcastable to
          ``(batch, heads, query_len, key_len)`` - a ``(L, S)`` causal or padding
          mask plugs straight in.
    In:   queries (TENSOR) - ``(..., L, E_in_q)``; shares leading batch dims with
          keys / values.
          keys (TENSOR) - ``(..., S, E_in_k)``.
          values (TENSOR) - ``(..., S, E_in_v)``.
          q_weight, k_weight, v_weight (TENSOR) - ``(E, E_in_*)`` each, wired in;
          ``E`` must be divisible by ``num_heads``.
          out_weight (TENSOR) - ``(E_out, E)``.
          q_bias, k_bias, v_bias, out_bias (TENSOR, optional) - ``(E,)`` /
          ``(E_out,)``; leave unconnected for bias-free projections.
          mask (TENSOR, optional) - boolean or additive, broadcastable to
          ``(batch, heads, L, S)``.
          num_heads (INT) - number of parallel heads (default 4).
          dropout_p (FLOAT) - attention-weight dropout probability; train mode
          only, 0 disables.
          seed (INT) - seed of the dropout draw; the same seed reproduces the
          same output bit for bit.
          mode (STRING, optional) - link the ``mode`` output of a Training Mode
          node; unconnected means ``train``.
    Out:  output (TENSOR) - ``(..., L, E_out)``. Floating-point inputs and weights
          of different dtypes are promoted to their common dtype first (e.g.
          fp16 activations with fp32 weights run in fp32).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _attention_schema(
            "AttentionMultihead",
            "Multi-Head Attention",
            "Multi-head attention over wired q/k/v tensors: projections, scaled dot-product, optional mask, softmax and output projection.",
            inputs=[
                io.Tensor.Input(
                    "queries",
                    tooltip="Queries of shape (..., query_len, in_width_q); shares leading batch dims with keys/values.",
                ),
                io.Tensor.Input(
                    "keys",
                    tooltip="Keys of shape (..., key_len, in_width_k).",
                ),
                io.Tensor.Input(
                    "values",
                    tooltip="Values of shape (..., key_len, in_width_v).",
                ),
                _weight_input("q_weight", "in_width_q"),
                _weight_input("k_weight", "in_width_k"),
                _weight_input("v_weight", "in_width_v"),
                _weight_input("out_weight", "embed width"),
                _bias_input("q_bias", "embed width"),
                _bias_input("k_bias", "embed width"),
                _bias_input("v_bias", "embed width"),
                _bias_input("out_bias", "out width"),
                _mask_input(),
                _num_heads_input(),
                _dropout_p_input(),
                _seed_input(),
                _attn_mode_input(),
            ],
            search_aliases=["attention", "mha", "scaled dot product", "sdpa", "transformer", "qkv", "self attention", "cross attention"],
        )

    @classmethod
    def execute(
        cls,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        out_weight: torch.Tensor,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        out_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_heads: int = 4,
        dropout_p: float = 0.0,
        seed: int = 0,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _multi_head_attention(
                queries, keys, values,
                q_weight, k_weight, v_weight, out_weight,
                q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias,
                mask=mask, num_heads=num_heads, dropout_p=dropout_p,
                training=_normalize_mode(mode) == MODE_TRAIN, seed=seed,
            )
        )


class AttentionSelf(io.ComfyNode):
    """Self-attention: one tensor queries, keys and values itself.

    What: the most common attention form - ``AttentionMultihead`` with
          ``queries = keys = values = tensor``, so the sequence attends to
          itself. The q/k/v projections happen inside the node from the same
          wired weight set, which removes the three duplicate wires the general
          node would need. Output shape equals input shape when ``out_weight``
          is square ``(E, E)``, the usual residual-block setup.
    In:   tensor (TENSOR) - ``(..., L, E_in)``; the sequence attending to itself.
          q_weight, k_weight, v_weight (TENSOR) - ``(E, E_in)`` each, wired in;
          ``E`` must be divisible by ``num_heads``.
          out_weight (TENSOR) - ``(E_out, E)``; square for residual use.
          q_bias, k_bias, v_bias, out_bias (TENSOR, optional) - ``(E,)`` /
          ``(E_out,)``.
          mask (TENSOR, optional) - boolean or additive, broadcastable to
          ``(batch, heads, L, L)``.
          num_heads (INT), dropout_p (FLOAT), seed (INT), mode (STRING,
          optional) - as in ``Multi-Head Attention``.
    Out:  output (TENSOR) - ``(..., L, E_out)``; dtype promotion as in
          ``BasicLinear``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _attention_schema(
            "AttentionSelf",
            "Self-Attention",
            "Self-attention on one tensor: q=k=v=the wired input, projections from the wired weight set.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input sequence of shape (..., seq_len, in_width); it queries, keys and values itself.",
                ),
                _weight_input("q_weight", "in_width"),
                _weight_input("k_weight", "in_width"),
                _weight_input("v_weight", "in_width"),
                _weight_input("out_weight", "embed width"),
                _bias_input("q_bias", "embed width"),
                _bias_input("k_bias", "embed width"),
                _bias_input("v_bias", "embed width"),
                _bias_input("out_bias", "out width"),
                _mask_input(),
                _num_heads_input(),
                _dropout_p_input(),
                _seed_input(),
                _attn_mode_input(),
            ],
            search_aliases=["self attention", "attention", "sa", "transformer", "intra attention", "mha"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        out_weight: torch.Tensor,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        out_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_heads: int = 4,
        dropout_p: float = 0.0,
        seed: int = 0,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _multi_head_attention(
                tensor, tensor, tensor,
                q_weight, k_weight, v_weight, out_weight,
                q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias,
                mask=mask, num_heads=num_heads, dropout_p=dropout_p,
                training=_normalize_mode(mode) == MODE_TRAIN, seed=seed,
            )
        )


class AttentionCross(io.ComfyNode):
    """Cross-attention: ``tensor`` queries, ``context`` keys and values.

    What: the encoder-decoder / multimodal form - queries are projected from
          ``tensor``, keys and values from ``context`` (a different sequence, a
          different width even), so one modality can attend to another. Output
          length follows ``tensor``'s sequence dimension, which is what a decoder
          block expects.
    In:   tensor (TENSOR) - ``(..., L, E_in_q)``; provides the queries.
          context (TENSOR) - ``(..., S, E_in_kv)``; provides keys and values,
          shares the leading batch dims with ``tensor``.
          q_weight (TENSOR) - ``(E, E_in_q)``; k_weight / v_weight -
          ``(E, E_in_kv)`` each, wired in; ``E`` must be divisible by
          ``num_heads``.
          out_weight (TENSOR) - ``(E_out, E)``.
          q_bias, k_bias, v_bias, out_bias (TENSOR, optional) - ``(E,)`` /
          ``(E_out,)``.
          mask (TENSOR, optional) - boolean or additive, broadcastable to
          ``(batch, heads, L, S)``.
          num_heads (INT), dropout_p (FLOAT), seed (INT), mode (STRING,
          optional) - as in ``Multi-Head Attention``.
    Out:  output (TENSOR) - ``(..., L, E_out)``; dtype promotion as in
          ``BasicLinear``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _attention_schema(
            "AttentionCross",
            "Cross-Attention",
            "Cross-attention: queries from tensor, keys/values from context; the encoder-decoder / multimodal form.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Query source of shape (..., query_len, in_width_q).",
                ),
                io.Tensor.Input(
                    "context",
                    tooltip="Key/value source of shape (..., context_len, in_width_kv); same leading batch dims as tensor.",
                ),
                _weight_input("q_weight", "in_width_q"),
                _weight_input("k_weight", "in_width_kv"),
                _weight_input("v_weight", "in_width_kv"),
                _weight_input("out_weight", "embed width"),
                _bias_input("q_bias", "embed width"),
                _bias_input("k_bias", "embed width"),
                _bias_input("v_bias", "embed width"),
                _bias_input("out_bias", "out width"),
                _mask_input(),
                _num_heads_input(),
                _dropout_p_input(),
                _seed_input(),
                _attn_mode_input(),
            ],
            search_aliases=["cross attention", "attention", "encoder decoder", "context", "multimodal", "mha"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        context: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        out_weight: torch.Tensor,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        out_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_heads: int = 4,
        dropout_p: float = 0.0,
        seed: int = 0,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _multi_head_attention(
                tensor, context, context,
                q_weight, k_weight, v_weight, out_weight,
                q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias,
                mask=mask, num_heads=num_heads, dropout_p=dropout_p,
                training=_normalize_mode(mode) == MODE_TRAIN, seed=seed,
            )
        )


class TransformerEncoderBlock(io.ComfyNode):
    """One post-LN Transformer encoder block in a single node.

    What: ``LayerNorm -> Multi-Head Attention -> Add -> LayerNorm -> FFN -> Add``
          assembled from the wired weights: the four attention projections, the
          two FFN matrices and the two LayerNorm affines all arrive through
          slots, so the block stays stateless and runs any checkpoint. It is the
          self-attention + feed-forward half of every Transformer encoder;
          stacking blocks (feed this node's output into the next one's
          ``tensor``) builds a full encoder. LayerNorm affines left unconnected
          give a non-affine ``F.layer_norm``; everything else is required.
    In:   tensor (TENSOR) - ``(..., L, E_in)``; the block's input and residual.
          q_weight, k_weight, v_weight (TENSOR) - ``(E, E_in)`` each;
          ``out_weight`` - ``(E, E)`` (square: the residual needs the same
          width); ``E`` must be divisible by ``num_heads``.
          ffn1_weight (TENSOR) - ``(ffn_dim, E)``; ``ffn2_weight`` -
          ``(E, ffn_dim)``.
          q_bias, k_bias, v_bias, out_bias, ffn1_bias, ffn2_bias (TENSOR,
          optional) - ``(E,)`` / ``(E,)`` / ``(E,)`` / ``(E,)`` / ``(ffn_dim,)``
          / ``(E,)``.
          ln1_weight, ln1_bias, ln2_weight, ln2_bias (TENSOR, optional) -
          ``(E,)`` each; unconnected = non-affine LayerNorm.
          mask (TENSOR, optional) - attention mask as in ``Multi-Head
          Attention``.
          num_heads (INT) - attention heads (default 4).
          dropout_p (FLOAT) - attention-weight dropout probability; train mode
          only, 0 disables.
          ffn_activation (COMBO) - activation between the FFN linear layers,
          ``relu`` (default) or ``gelu``.
          seed (INT) - seed of the dropout draw.
          mode (STRING, optional) - link the ``mode`` output of a Training Mode
          node; unconnected means ``train``.
    Out:  output (TENSOR) - ``(..., L, E)``; dtype promotion as in
          ``BasicLinear``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _attention_schema(
            "TransformerEncoderBlock",
            "Transformer Encoder Block",
            "Post-LN Transformer encoder block: LayerNorm -> MHA -> Add -> LayerNorm -> FFN -> Add, all weights wired in.",
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    tooltip="Input sequence of shape (..., seq_len, width); it is both the attention source and the residual.",
                ),
                _weight_input("q_weight", "width"),
                _weight_input("k_weight", "width"),
                _weight_input("v_weight", "width"),
                _weight_input("out_weight", "width"),
                _weight_input("ffn1_weight", "ffn_dim"),
                _weight_input("ffn2_weight", "width"),
                _bias_input("q_bias", "width"),
                _bias_input("k_bias", "width"),
                _bias_input("v_bias", "width"),
                _bias_input("out_bias", "width"),
                _bias_input("ffn1_bias", "ffn_dim"),
                _bias_input("ffn2_bias", "width"),
                io.Tensor.Input(
                    "ln1_weight",
                    optional=True,
                    tooltip="Optional LayerNorm scale of shape (width,) after the attention residual; unconnected = non-affine LayerNorm.",
                ),
                io.Tensor.Input(
                    "ln1_bias",
                    optional=True,
                    tooltip="Optional LayerNorm shift of shape (width,) after the attention residual; unconnected = none.",
                ),
                io.Tensor.Input(
                    "ln2_weight",
                    optional=True,
                    tooltip="Optional LayerNorm scale of shape (width,) after the FFN residual; unconnected = non-affine LayerNorm.",
                ),
                io.Tensor.Input(
                    "ln2_bias",
                    optional=True,
                    tooltip="Optional LayerNorm shift of shape (width,) after the FFN residual; unconnected = none.",
                ),
                _mask_input(),
                _num_heads_input(),
                _dropout_p_input(),
                io.Combo.Input(
                    "ffn_activation",
                    options=list(FFN_ACTIVATION_OPTIONS),
                    default="relu",
                    tooltip="Activation between the two FFN linear layers.",
                ),
                _seed_input(),
                _attn_mode_input(),
            ],
            search_aliases=["transformer", "encoder block", "post ln", "residual", "self attention", "ffn", "feed forward"],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        out_weight: torch.Tensor,
        ffn1_weight: torch.Tensor,
        ffn2_weight: torch.Tensor,
        q_bias: torch.Tensor | None = None,
        k_bias: torch.Tensor | None = None,
        v_bias: torch.Tensor | None = None,
        out_bias: torch.Tensor | None = None,
        ffn1_bias: torch.Tensor | None = None,
        ffn2_bias: torch.Tensor | None = None,
        ln1_weight: torch.Tensor | None = None,
        ln1_bias: torch.Tensor | None = None,
        ln2_weight: torch.Tensor | None = None,
        ln2_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_heads: int = 4,
        dropout_p: float = 0.0,
        ffn_activation: str = "relu",
        seed: int = 0,
        mode: str = MODE_TRAIN,
    ) -> io.NodeOutput:
        training = _normalize_mode(mode) == MODE_TRAIN
        attn = _multi_head_attention(
            tensor, tensor, tensor,
            q_weight, k_weight, v_weight, out_weight,
            q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, out_bias=out_bias,
            mask=mask, num_heads=num_heads, dropout_p=dropout_p,
            training=training, seed=seed,
        )
        hidden = _block_layer_norm(tensor + attn, ln1_weight, ln1_bias, "ln1")
        ffn = F.linear(
            _ffn_activation(F.linear(hidden, ffn1_weight, ffn1_bias), ffn_activation),
            ffn2_weight,
            ffn2_bias,
        )
        return io.NodeOutput(_block_layer_norm(hidden + ffn, ln2_weight, ln2_bias, "ln2"))


#: Every node this module registers, in node-library order.
ATTENTION_NODES: list[type[io.ComfyNode]] = [
    AttentionMultihead,
    AttentionSelf,
    AttentionCross,
    TransformerEncoderBlock,
]


class AttentionExtension(ComfyExtension):
    """Registers the core attention node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(ATTENTION_NODES)


async def comfy_entrypoint() -> AttentionExtension:
    return AttentionExtension()
