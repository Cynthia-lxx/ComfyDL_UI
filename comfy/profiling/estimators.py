"""Estimator registry of ``comfy.profiling``.

One estimator per node class type. It receives an :class:`EstimationCtx`
(the node's prompt-format inputs, with links resolved to the upstream
estimator outputs) and returns a :class:`NodeEstimate`: the values this
node publishes on its output slots plus a memory breakdown.

The registry is keyed by the ComfyUI class type (``"LanguageModelTrain"``,
...). A node type without an estimator is reported as ``unknown`` - the
engine never guesses.

Widget defaults below mirror the node definitions in ``comfy_extras/``; the
prompt normally carries every widget value anyway, the defaults only cover
a frontend that omitted one.

The estimators must stay pure: no torch import, no node import, no IO
beyond a best-effort ``os.stat`` in the Load estimator.
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from comfy.profiling import formulas
from comfy.profiling.assumptions import AssumptionSet
from comfy.profiling.shapes import (
    BlockInfo,
    EmbeddingInfo,
    EstValue,
    IntVal,
    ModelVal,
    OptimizerVal,
    SpecVal,
    TextVal,
    TensorVal,
    Unknown,
    VocabVal,
    FLOAT32_BYTES,
    INT64_BYTES,
)

ESTIMATORS: Dict[str, Any] = {}


def register(class_type: str):
    """Decorator: register an estimator function for a class type."""

    def decorate(function):
        ESTIMATORS[class_type] = function
        return function

    return decorate


@dataclasses.dataclass
class NodeEstimate:
    """What one estimator concluded about one node."""

    outputs: List[EstValue] = dataclasses.field(default_factory=list)
    items: List[formulas.MemoryItem] = dataclasses.field(default_factory=list)
    basis: Optional[dict] = None  # {"key": str, "params": {...}} - bilingual via key
    confidence: str = "exact"  # exact | approx
    status: str = "estimated"  # estimated | unknown
    reason: str = ""
    # The M2 compute ledger. ``flops_status`` is one of:
    #   "estimated" - flops_items carry the counted matmul FLOPs
    #   "zero"      - pass-through / IO node, no meaningful compute
    #   "unknown"   - the node does real work we cannot count statically
    #                 (pure-Python loops: vocab build, encode, windowing)
    flops_items: List[formulas.FlopsItem] = dataclasses.field(default_factory=list)
    flops_status: str = "zero"
    flops_reason: str = ""

    def as_unknown(self, reason: str) -> "NodeEstimate":
        self.status = "unknown"
        self.reason = reason
        self.outputs = [Unknown(reason)] * max(1, len(self.outputs))
        self.items = []
        if self.flops_status != "unknown":
            self.flops_status = "unknown"
            self.flops_reason = reason
        return self


class EstimationCtx:
    """Everything one estimator may look at.

    ``inputs`` is the node's entry of the prompt: ``{name: value}`` where a
    linked input is ``[source_node_id, output_slot]`` and a widget input is
    the literal value. ``outputs`` holds the already-estimated upstream
    outputs by node id.
    """

    def __init__(
        self,
        node_id: str,
        class_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, List[EstValue]],
        assumptions: AssumptionSet,
    ) -> None:
        self.node_id = node_id
        self.class_type = class_type
        self.inputs = inputs or {}
        self._outputs = outputs
        self.assumptions = assumptions

    # -- raw access --------------------------------------------------------
    @staticmethod
    def _is_link(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[1], int)
            and not isinstance(value, str)
        )

    def raw(self, name: str) -> Any:
        return self.inputs.get(name)

    # -- widget values ------------------------------------------------------
    def widget(self, name: str, default=None) -> Any:
        value = self.inputs.get(name, default)
        return default if value is None else value

    def str_widget(self, name: str, default: str = "") -> str:
        value = self.inputs.get(name)
        return default if value is None else str(value)

    def int_widget(self, name: str, default: Optional[int] = None) -> Optional[int]:
        value = self.inputs.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def float_widget(self, name: str, default: float = 0.0) -> float:
        value = self.inputs.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # -- linked values ------------------------------------------------------
    def est(self, name: str) -> EstValue:
        """The upstream estimator output on input ``name`` (or Unknown)."""
        value = self.inputs.get(name)
        if not self._is_link(value):
            return Unknown(f"'{name}' is not linked")
        source, slot = value[0], value[1]
        upstream = self._outputs.get(source)
        if upstream is None:
            return Unknown(f"upstream node {source} is not estimated")
        if 0 <= slot < len(upstream):
            return upstream[slot]
        return Unknown(f"upstream node {source} has no output slot {slot}")

    def tensor(self, name: str) -> EstValue:
        value = self.est(name)
        if isinstance(value, TensorVal):
            return value
        if isinstance(value, Unknown):
            return value
        return Unknown(f"'{name}' did not receive a tensor value")

    def spec(self, name: str) -> SpecVal:
        value = self.est(name)
        if isinstance(value, SpecVal):
            return value
        if isinstance(value, ModelVal) and isinstance(value.spec, SpecVal):
            return value.spec
        return SpecVal(None, ())

    def vocab_size_of(self, name: str, widget_name: str, default: int) -> Optional[int]:
        value = self.est(name)
        if isinstance(value, VocabVal):
            return value.size
        if isinstance(value, IntVal) and value.value is not None:
            return value.value
        return self.int_widget(widget_name, default)

    def assumption(self, key: str):
        return self.assumptions.get(key)


def _token_count(text: str, level: str) -> int:
    if level == "word":
        return len(str(text or "").split())
    return len(str(text or ""))


def _normalize_optimizer(value: Any) -> str:
    text = str(value or "AdamW").strip()
    canonical = {name.lower(): name for name in formulas.OPTIMIZER_STATE_MULTIPLIERS}
    return canonical.get(text.lower(), "AdamW")


def _batch_size(ctx: EstimationCtx, samples: int) -> int:
    """Mirror the nodes' batch semantics: 0 or >= samples means everything."""
    requested = ctx.int_widget("batch_size", 0) or 0
    if requested <= 0 or requested >= samples:
        return samples
    return requested


# --------------------------------------------------------------------------- #
# Text pipeline (comfy_extras/nodes_nlp.py)
# --------------------------------------------------------------------------- #
@register("TextVocabBuild")
def _estimate_vocab_build(ctx: EstimationCtx) -> NodeEstimate:
    corpus = ctx.str_widget("corpus", "")
    level = ctx.str_widget("level", "char")
    if level not in ("char", "word"):
        level = "char"
    min_freq = max(1, ctx.int_widget("min_freq", 1) or 1)
    if level == "word":
        tokens = " ".join(corpus.split()).split()
    else:
        tokens = list(corpus)
    counts = Counter(token for token in tokens if token)
    size = 1 + sum(1 for count in counts.values() if count >= min_freq)
    return NodeEstimate(
        outputs=[VocabVal(size, level), IntVal(size)],
        items=[],
        flops_status="unknown",
        flops_reason="pure-Python token loop - compute is not a matmul FLOPs model",
        basis={
            "key": "vocab_build",
            "params": {
                "chars": len(corpus),
                "tokens": len(tokens),
                "size": size,
                "level": level,
            },
        },
    )


@register("TextEncode")
def _estimate_text_encode(ctx: EstimationCtx) -> NodeEstimate:
    vocab = ctx.est("vocab")
    level = vocab.level if isinstance(vocab, VocabVal) else "char"
    text = ctx.str_widget("text", "")
    count = _token_count(text, level)
    size = count * INT64_BYTES
    return NodeEstimate(
        outputs=[TensorVal((count,), INT64_BYTES, "long")],
        items=[formulas.MemoryItem("encoded ids tensor", size, size, kind="outputs")],
        flops_status="unknown",
        flops_reason="pure-Python token loop - compute is not a matmul FLOPs model",
        basis={"key": "text_encode", "params": {"tokens": count, "level": level}},
    )


@register("TextDecode")
def _estimate_text_decode(ctx: EstimationCtx) -> NodeEstimate:
    ids = ctx.tensor("ids")
    length = ids.numel() if isinstance(ids, TensorVal) else None
    return NodeEstimate(
        outputs=[TextVal(length)],
        basis={"key": "pass_through", "params": {}},
    )


@register("TextSlidingWindow")
def _estimate_sliding_window(ctx: EstimationCtx) -> NodeEstimate:
    ids = ctx.tensor("ids")
    window = max(1, ctx.int_widget("window", 4) or 4)
    confidence = "exact"
    length = ids.dim(0) if isinstance(ids, TensorVal) else None
    if length is None:
        length, source = ctx.assumption("sequence_length")
        confidence = "approx"
    samples = max(0, length - window)
    x_bytes = samples * window * INT64_BYTES
    # The node builds Python lists-of-lists before the tensor exists; the
    # pointer overhead is real memory (a list of ~200k x 64 ints is ~100 MB).
    python_lists = samples * (56 + window * 8)
    return NodeEstimate(
        outputs=[
            TensorVal((samples, window), INT64_BYTES, "long"),
            TensorVal((samples,), INT64_BYTES, "long"),
        ],
        items=[
            formulas.MemoryItem(
                "dataset tensors (x / y)",
                x_bytes + samples * INT64_BYTES,
                x_bytes,
                kind="dataset",
            ),
            formulas.MemoryItem(
                "python window lists (approximate)", python_lists, python_lists, kind="python_lists", approx=True
            ),
        ],
        confidence=confidence,
        flops_status="unknown",
        flops_reason="pure-Python windowing loop - compute is not a matmul FLOPs model",
        basis={
            "key": "sliding_window",
            "params": {"tokens": length, "window": window, "samples": samples},
        },
    )


# --------------------------------------------------------------------------- #
# Spec chain + build (comfy_extras/nodes_lm.py)
# --------------------------------------------------------------------------- #
@register("LanguageModelEmbedding")
def _estimate_lm_embedding(ctx: EstimationCtx) -> NodeEstimate:
    vocab_size = ctx.vocab_size_of("vocab", "vocab_size", 16)
    d_model = ctx.int_widget("d_model", 32)
    include_position = bool(ctx.widget("include_position", True))
    incoming = ctx.spec("spec")
    if vocab_size is None or d_model is None:
        return NodeEstimate().as_unknown("vocab size or model width not statically known")
    embedding = EmbeddingInfo(vocab_size, d_model, include_position)
    blocks = incoming.blocks if incoming.is_known else ()
    return NodeEstimate(
        outputs=[SpecVal(embedding, blocks), IntVal(d_model)],
        basis={
            "key": "lm_spec",
            "params": {"vocab": vocab_size, "d_model": d_model, "blocks": len(blocks)},
        },
    )


@register("LanguageModelTransformerBlock")
def _estimate_lm_block(ctx: EstimationCtx) -> NodeEstimate:
    incoming = ctx.spec("spec")
    if not incoming.is_known:
        return NodeEstimate().as_unknown("incoming spec chain is unknown")
    d_model = incoming.embedding.d_model
    block = BlockInfo(
        d_model=d_model,
        num_heads=max(1, ctx.int_widget("num_heads", 4) or 4),
        d_ffn=max(2, ctx.int_widget("d_ffn", 128) or 128),
        activation=ctx.str_widget("activation", "relu"),
        dropout=ctx.float_widget("dropout", 0.0),
    )
    return NodeEstimate(
        outputs=[SpecVal(incoming.embedding, incoming.blocks + (block,)), IntVal(d_model)],
        basis={
            "key": "lm_spec",
            "params": {
                "vocab": incoming.embedding.vocab_size,
                "d_model": d_model,
                "blocks": len(incoming.blocks) + 1,
            },
        },
    )


@register("LanguageModelBuild")
def _estimate_lm_build(ctx: EstimationCtx) -> NodeEstimate:
    spec = ctx.spec("spec")
    estimate = NodeEstimate(basis={"key": "lm_build", "params": {}})
    if not spec.is_known:
        return estimate.as_unknown("spec chain is unknown")
    params = formulas.lm_parameter_count(
        spec.embedding.vocab_size, spec.embedding.d_model, spec.blocks
    )
    estimate.outputs = [ModelVal(spec), IntVal(params)]
    if params:
        estimate.items = [
            formulas.MemoryItem("parameters (float32)", params * 4, kind="params")
        ]
        estimate.basis["params"] = {"params": params, "blocks": len(spec.blocks)}
    return estimate


@register("TrainingOptimizer")
def _estimate_training_optimizer(ctx: EstimationCtx) -> NodeEstimate:
    kind = _normalize_optimizer(ctx.widget("optimizer", "AdamW"))
    return NodeEstimate(
        outputs=[OptimizerVal(kind)],
        basis={"key": "optimizer", "params": {"optimizer": kind}},
    )


def _training_common(ctx: EstimationCtx) -> Optional[dict]:
    """Shared inputs of the two trainer estimators; None when unusable."""
    model_value = ctx.est("model")
    spec = model_value.spec if isinstance(model_value, ModelVal) else None
    if spec is None and isinstance(model_value, SpecVal):
        spec = model_value
    if not (spec and spec.is_known):
        return None
    optimizer = ctx.est("optimizer")
    kind = (
        optimizer.kind
        if isinstance(optimizer, OptimizerVal)
        else _normalize_optimizer(ctx.widget("optimizer", "AdamW"))
    )
    x = ctx.tensor("x")
    if not isinstance(x, TensorVal) or x.dim(0) is None or x.dim(1) is None:
        return None
    return {"spec": spec, "optimizer": kind, "x": x}


@register("LanguageModelTrain")
def _estimate_lm_train(ctx: EstimationCtx) -> NodeEstimate:
    common = _training_common(ctx)
    estimate = NodeEstimate(basis={"key": "lm_train", "params": {}})
    if common is None:
        return estimate.as_unknown("model structure or dataset shape is not statically known")
    spec, kind, x = common["spec"], common["optimizer"], common["x"]
    embedding = spec.embedding
    samples, window = x.dim(0), x.dim(1)
    batch = _batch_size(ctx, samples)
    steps = max(1, ctx.int_widget("steps", 300) or 300)
    early_stop = (ctx.int_widget("early_stop_patience", 0) or 0) > 0
    breakdown = formulas.lm_training(
        batch=batch,
        dataset_samples=samples,
        window=window,
        vocab_size=embedding.vocab_size,
        d_model=embedding.d_model,
        blocks=spec.blocks,
        optimizer=kind,
        early_stop=early_stop,
    )
    params = formulas.lm_parameter_count(
        embedding.vocab_size, embedding.d_model, spec.blocks
    )
    estimate.outputs = [
        ModelVal(spec),
        Unknown("scalar"),
        TensorVal((steps,), FLOAT32_BYTES),
    ]
    estimate.items = breakdown.items
    estimate.flops_items = formulas.lm_training_flops(
        batch=batch,
        seq=window,
        vocab_size=embedding.vocab_size,
        d_model=embedding.d_model,
        blocks=spec.blocks,
        steps=steps,
    )
    estimate.flops_status = "estimated"
    estimate.basis["params"] = {
        "batch": batch,
        "samples": samples,
        "window": window,
        "vocab": embedding.vocab_size,
        "d_model": embedding.d_model,
        "blocks": len(spec.blocks),
        "params": params or 0,
        "steps": steps,
        "optimizer": kind,
    }
    return estimate


@register("LanguageModelForward")
def _estimate_lm_forward(ctx: EstimationCtx) -> NodeEstimate:
    model_value = ctx.est("model")
    spec = model_value.spec if isinstance(model_value, ModelVal) else None
    if spec is None and isinstance(model_value, SpecVal):
        spec = model_value
    estimate = NodeEstimate(basis={"key": "lm_forward", "params": {}})
    if not (spec and spec.is_known):
        return estimate.as_unknown("model structure is not statically known")
    ids = ctx.tensor("ids")
    confidence = "exact"
    if isinstance(ids, TensorVal) and ids.dim(0) is not None:
        if len(ids.shape) >= 2:
            batch, length = ids.dim(0), ids.dim(1)
        else:
            batch, length = 1, ids.dim(0)
    else:
        batch, length = 1, ctx.assumption("sequence_length")[0]
        confidence = "approx"
    if length is None:
        length = ctx.assumption("sequence_length")[0]
        confidence = "approx"
    breakdown = formulas.lm_inference(
        batch=batch,
        length=length,
        vocab_size=spec.embedding.vocab_size,
        d_model=spec.embedding.d_model,
        blocks=spec.blocks,
    )
    estimate.outputs = [
        TensorVal((batch, length, spec.embedding.vocab_size), FLOAT32_BYTES)
    ]
    estimate.items = breakdown.items
    estimate.flops_items = formulas.lm_forward_flops(
        batch=batch,
        seq=length,
        vocab_size=spec.embedding.vocab_size,
        d_model=spec.embedding.d_model,
        blocks=spec.blocks,
    )
    estimate.flops_status = "estimated"
    estimate.confidence = confidence
    estimate.basis["params"] = {
        "batch": batch,
        "length": length,
        "vocab": spec.embedding.vocab_size,
        "d_model": spec.embedding.d_model,
        "blocks": len(spec.blocks),
    }
    return estimate


@register("LanguageModelGenerate")
def _estimate_lm_generate(ctx: EstimationCtx) -> NodeEstimate:
    model_value = ctx.est("model")
    spec = model_value.spec if isinstance(model_value, ModelVal) else None
    if spec is None and isinstance(model_value, SpecVal):
        spec = model_value
    estimate = NodeEstimate(basis={"key": "lm_generate", "params": {}})
    if not (spec and spec.is_known):
        return estimate.as_unknown("model structure is not statically known")
    vocab = ctx.est("vocab")
    level = vocab.level if isinstance(vocab, VocabVal) else "char"
    prefix_ids = ctx.tensor("prefix_ids")
    if isinstance(prefix_ids, TensorVal) and prefix_ids.numel() is not None:
        prefix_length = prefix_ids.numel()
    else:
        prefix_length = _token_count(ctx.str_widget("prefix", ""), level)
    num_tokens = max(1, ctx.int_widget("num_tokens", 16) or 16)
    length = prefix_length + num_tokens
    # Autoregressive loop: every forward pass is over one sequence.
    breakdown = formulas.lm_inference(
        batch=1,
        length=length,
        vocab_size=spec.embedding.vocab_size,
        d_model=spec.embedding.d_model,
        blocks=spec.blocks,
    )
    estimate.outputs = [TensorVal((length,), INT64_BYTES, "long"), Unknown("text")]
    estimate.items = breakdown.items
    estimate.flops_items = formulas.lm_generate_flops(
        prefix_length=prefix_length,
        num_tokens=num_tokens,
        vocab_size=spec.embedding.vocab_size,
        d_model=spec.embedding.d_model,
        blocks=spec.blocks,
    )
    estimate.flops_status = "estimated"
    estimate.basis["params"] = {
        "prefix": prefix_length,
        "num_tokens": num_tokens,
        "length": length,
        "vocab": spec.embedding.vocab_size,
    }
    return estimate


@register("LanguageModelSave")
def _estimate_lm_save(ctx: EstimationCtx) -> NodeEstimate:
    model_value = ctx.est("model")
    spec = model_value.spec if isinstance(model_value, ModelVal) else None
    return NodeEstimate(
        outputs=[ModelVal(spec) if spec else Unknown("model"), TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("LanguageModelLoad")
def _estimate_lm_load(ctx: EstimationCtx) -> NodeEstimate:
    path = ctx.str_widget("path", "")
    estimate = NodeEstimate(basis={"key": "file_load", "params": {"path": path}})
    file_bytes = None
    for candidate in (path, os.path.join("output", path)):
        try:
            if candidate and os.path.isfile(candidate):
                file_bytes = os.path.getsize(candidate)
                break
        except OSError:
            continue
    if file_bytes is None:
        return estimate.as_unknown(
            "checkpoint not found / structure not declared in the graph"
        )
    params = file_bytes // 4
    estimate.outputs = [
        Unknown("checkpoint structure unknown"),
        VocabVal(None),
        IntVal(params),
    ]
    estimate.items = [
        formulas.MemoryItem(
            "checkpoint parameters (from file size)", file_bytes, file_bytes, kind="params"
        )
    ]
    estimate.confidence = "approx"
    estimate.basis["params"] = {"path": path, "file_bytes": file_bytes, "params": params}
    return estimate


# --------------------------------------------------------------------------- #
# Training Loop (comfy_extras/nodes_training.py)
# --------------------------------------------------------------------------- #
@register("TrainingLoop")
def _estimate_training_loop(ctx: EstimationCtx) -> NodeEstimate:
    estimate = NodeEstimate(basis={"key": "mlp_train", "params": {}})
    x = ctx.tensor("x")
    y = ctx.tensor("y")
    if not isinstance(x, TensorVal) or x.dim(0) is None or x.dim(-1) is None:
        return estimate.as_unknown("input shape is not statically known")
    samples, in_features = x.dim(0), x.dim(-1)
    out_features = 1
    if isinstance(y, TensorVal) and len(y.shape) >= 2 and y.dim(-1) is not None:
        out_features = y.dim(-1)
    hidden = formulas.parse_hidden_widths(ctx.str_widget("hidden", "8"))
    optimizer = ctx.est("optimizer")
    kind = (
        optimizer.kind
        if isinstance(optimizer, OptimizerVal)
        else _normalize_optimizer(ctx.widget("optimizer", "AdamW"))
    )
    batch = _batch_size(ctx, samples)
    steps = max(1, ctx.int_widget("steps", 200) or 200)
    early_stop = (ctx.int_widget("early_stop_patience", 0) or 0) > 0
    breakdown = formulas.mlp_training(
        batch=batch,
        dataset_samples=samples,
        in_features=in_features,
        hidden=hidden,
        out_features=out_features,
        optimizer=kind,
        early_stop=early_stop,
    )
    params = formulas.mlp_parameter_count(in_features, hidden, out_features)
    estimate.outputs = [
        Unknown("params"),
        TensorVal((), FLOAT32_BYTES),
        TensorVal((steps,), FLOAT32_BYTES),
        TensorVal((samples, out_features), FLOAT32_BYTES),
    ]
    estimate.items = breakdown.items
    estimate.flops_items = formulas.mlp_training_flops(
        batch=batch,
        in_features=in_features,
        hidden=hidden,
        out_features=out_features,
        steps=steps,
    )
    estimate.flops_status = "estimated"
    estimate.basis["params"] = {
        "batch": batch,
        "samples": samples,
        "in_features": in_features,
        "hidden": hidden,
        "out_features": out_features,
        "steps": steps,
        "optimizer": kind,
    }
    return estimate
