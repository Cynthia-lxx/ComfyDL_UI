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

import ast
import dataclasses
import math
import os
import re
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
    ModuleVal,
    OptimizerVal,
    SpecVal,
    TextVal,
    TensorVal,
    Unknown,
    VocabVal,
    WeightsVal,
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


# --------------------------------------------------------------------------- #
# Shared helpers for the M3 family estimators.
# --------------------------------------------------------------------------- #

def _parse_ints(text: Any) -> Optional[list]:
    """Comma-separated ints (``"3, 2" -> [3, 2]``); ``None`` on any bad part."""
    if text is None:
        return None
    values = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            return None
    return values or None


def _parse_floats(text: Any) -> Optional[list]:
    """Comma-separated floats; ``None`` on any bad part."""
    if text is None:
        return None
    values = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values or None


#: Where ``folder_paths.py`` looks for each model kind, newest folder first
#: (its legacy aliases included: "unet" -> diffusion_models, "clip" ->
#: text_encoders). Kept as data so the estimators stay free of ComfyUI
#: imports: only a best-effort ``os.stat`` is done, never a load.
MODEL_FOLDERS: Dict[str, tuple] = {
    "checkpoints": ("models/checkpoints",),
    "diffusion_models": ("models/diffusion_models", "models/unet"),
    "vae": ("models/vae",),
    "text_encoders": ("models/text_encoders", "models/clip"),
    "loras": ("models/loras",),
}


def _literal_shape(value: Any) -> Optional[tuple]:
    """Shape of a nested Python literal (``ast.literal_eval`` payload).

    Ragged nesting returns ``None`` - torch would fail on it too.
    """
    if isinstance(value, (int, float, bool)):
        return ()
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        first = _literal_shape(value[0])
        if first is None:
            return None
        for item in value[1:]:
            if _literal_shape(item) != first:
                return None
        return (len(value),) + first
    return None


def _flat_boxes_count(tensor: EstValue) -> Optional[int]:
    """``_ensure_2d`` semantics: ``(..., 4) -> (N, 4)`` with N the leading product."""
    if not isinstance(tensor, TensorVal) or len(tensor.shape) < 1:
        return None
    if tensor.shape[-1] != 4:
        return None
    count = 1
    for dim in tensor.shape[:-1]:
        if dim is None:
            return None
        count *= int(dim)
    return count


def _image_dims(tensor: EstValue):
    """``(B, H, W, C)`` of an IMAGE-valued TensorVal, or ``None``."""
    if isinstance(tensor, TensorVal) and len(tensor.shape) == 4:
        return tuple(tensor.shape)
    return None


def _annotated_filepath(name: str):
    """``folder_paths.get_annotated_filepath`` semantics.

    Only ``[input]`` / ``[output]`` / ``[temp]`` are real annotations
    (``folder_paths.py:257-270``); anything else is a plain - possibly
    relative - filename. Returns ``(filename, folder or "")``.
    """
    text = str(name or "").strip()
    for tag in ("input", "output", "temp"):
        suffix = " [%s]" % tag
        if text.endswith(suffix):
            return text[: -len(suffix)].strip(), tag
    return text, ""


def _file_size(folder: str, filename: str) -> Optional[int]:
    """Best-effort ``os.path.getsize`` relative to the repo root (never raises)."""
    for candidate in (os.path.join(folder, filename),):
        try:
            if filename and os.path.isfile(candidate):
                return os.path.getsize(candidate)
        except OSError:
            continue
    return None


def _model_file_size(folder_key: str, filename: str) -> Optional[int]:
    """Best-effort size of a model file, mirroring ``folder_paths``'s folders.

    The estimator stays pure: the folders ``folder_paths.py`` registers
    (including its legacy aliases) are listed here as relative paths and
    probed in the same order, first hit wins.
    """
    for folder in MODEL_FOLDERS.get(folder_key, ()):
        size = _file_size(folder, filename)
        if size is not None:
            return size
    return None


# --------------------------------------------------------------------------- #
# M3 batch 1: tensor algebra (comfydl/nodes/tensor_basic.py, tensor_ops.py)
# --------------------------------------------------------------------------- #

@register("CdlTensorToStr")
def _estimate_cdl_tensor_to_str(ctx: EstimationCtx) -> NodeEstimate:
    """Tensor pretty-printer: text output only, nothing tensor-shaped."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlStrToTensor")
def _estimate_cdl_str_to_tensor(ctx: EstimationCtx) -> NodeEstimate:
    """Literal text -> float32 tensor; the shape is parsed statically (ast).

    Parse failures follow the node's error_strategy: empty -> (0,),
    zero -> (1,), raise -> unknown.
    """
    text = ctx.str_widget("text", "")
    strategy = ctx.str_widget("error_strategy", "empty_tensor")
    cleaned = re.sub(r"\s+", "", text).replace(",]", "]").replace(",)", ")")
    shape = None
    try:
        literal = ast.literal_eval(cleaned or text)
        literal_shape = _literal_shape(literal)
        if literal_shape is not None:
            shape = (1,) if literal_shape == () else literal_shape
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        shape = None
    if shape is None:
        if strategy == "zero_tensor":
            shape = (1,)
        elif strategy == "raise_error":
            return NodeEstimate(basis={"key": "tensor_shape", "params": {}}).as_unknown(
                "text is not a parseable literal and error_strategy raises"
            )
        else:
            shape = (0,)
    value = TensorVal(tuple(shape), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("parsed tensor", value.nbytes(), kind="outputs")],
        basis={"key": "tensor_shape", "params": {"shape": list(shape)}},
    )


@register("CdlConv2d")
def _estimate_cdl_conv2d(ctx: EstimationCtx) -> NodeEstimate:
    """``torch.nn.functional.conv2d`` (no bias, scalar stride/padding).

    Inputs and kernel are padded to 4D exactly like the node does; the
    forced batch dim is squeezed away afterwards.
    """
    estimate = NodeEstimate(basis={"key": "conv2d", "params": {}})
    x = ctx.tensor("input_tensor")
    k = ctx.tensor("kernel")
    if not isinstance(x, TensorVal) or not isinstance(k, TensorVal):
        return estimate.as_unknown("input or kernel shape is not statically known")
    x4 = None
    if len(x.shape) == 2:
        x4 = (1, 1) + tuple(x.shape)
    elif len(x.shape) == 3:
        x4 = (1,) + tuple(x.shape)
    elif len(x.shape) == 4:
        x4 = tuple(x.shape)
    k4 = None
    if len(k.shape) == 2:
        k4 = (1, 1) + tuple(k.shape)
    elif len(k.shape) == 3:
        k4 = (1,) + tuple(k.shape)
    elif len(k.shape) == 4:
        k4 = tuple(k.shape)
    if x4 is None or k4 is None:
        return estimate.as_unknown("conv2d needs 2D/3D/4D operands")
    n, cin, h, w = x4
    kout, kin, kh, kw = k4
    if cin is not None and kin is not None and cin != kin:
        return estimate.as_unknown("kernel in-channels do not match the input")
    stride = max(1, ctx.int_widget("stride", 1) or 1)
    padding = max(0, ctx.int_widget("padding", 0) or 0)
    if h is None or w is None or kh is None or kw is None or kout is None:
        return estimate.as_unknown("spatial dims not fully known")
    ho = formulas.conv2d_out_dim(h, kh, stride, padding)
    wo = formulas.conv2d_out_dim(w, kw, stride, padding)
    out = (kout, ho, wo) if n == 1 else (n, kout, ho, wo)
    value = TensorVal(out, x.itemsize, x.dtype)
    estimate.outputs = [value]
    estimate.items = [formulas.tensor_item("conv output", value.nbytes(), kind="outputs")]
    cin_effective = cin if cin is not None else kin
    if n is not None and cin_effective is not None:
        estimate.flops_items = [
            formulas.FlopsItem(
                "conv2d (stride %d, pad %d)" % (stride, padding),
                formulas.conv2d_flops(n, ho, wo, kout, cin_effective, kh, kw),
                kind="conv",
            )
        ]
        estimate.flops_status = "estimated"
    else:
        estimate.flops_status = "unknown"
        estimate.flops_reason = "channel/batch counts not fully known"
    estimate.basis["params"] = {"in": list(x.shape), "kernel": list(k.shape), "out": list(out)}
    return estimate


@register("CdlTranspose")
def _estimate_cdl_transpose(ctx: EstimationCtx) -> NodeEstimate:
    """``torch.transpose``: dims swapped, dtype kept, view (no allocation)."""
    estimate = NodeEstimate(basis={"key": "tensor_shape", "params": {}})
    t = ctx.tensor("tensor")
    if not isinstance(t, TensorVal) or not t.shape:
        return estimate.as_unknown("tensor shape is not statically known")
    dim0 = ctx.int_widget("dim0", 0) or 0
    dim1 = ctx.int_widget("dim1", 1) or 1
    rank = len(t.shape)
    i0 = dim0 + rank if dim0 < 0 else dim0
    i1 = dim1 + rank if dim1 < 0 else dim1
    if not (0 <= i0 < rank and 0 <= i1 < rank):
        return estimate.as_unknown("transpose dims out of range")
    shape = list(t.shape)
    shape[i0], shape[i1] = shape[i1], shape[i0]
    return NodeEstimate(
        outputs=[TensorVal(tuple(shape), t.itemsize, t.dtype)],
        basis={"key": "tensor_shape", "params": {"shape": shape}},
    )


@register("CdlBroadcast")
def _estimate_cdl_broadcast(ctx: EstimationCtx) -> NodeEstimate:
    """``torch.broadcast_to``: view; parse/broadcast failure returns the input."""
    estimate = NodeEstimate(basis={"key": "tensor_shape", "params": {}})
    t = ctx.tensor("tensor")
    if not isinstance(t, TensorVal):
        return estimate.as_unknown("tensor shape is not statically known")
    shape = list(t.shape)
    approx = False
    target = _parse_ints(ctx.str_widget("target_shape", ""))
    if target and min(target) >= 0 and len(target) >= len(shape):
        ok = True
        for index, dim in enumerate(reversed(shape)):
            if dim is not None and dim != 1 and dim != target[-1 - index]:
                ok = False
                break
        if ok:
            shape, approx = list(target), True
    result = NodeEstimate(
        outputs=[TensorVal(tuple(shape), t.itemsize, t.dtype)],
        basis={"key": "tensor_shape", "params": {"shape": shape}},
    )
    result.confidence = "approx" if approx else "exact"
    return result


@register("CdlReshape")
def _estimate_cdl_reshape(ctx: EstimationCtx) -> NodeEstimate:
    """``torch.reshape`` with -1 inference; failure returns the input shape."""
    estimate = NodeEstimate(basis={"key": "tensor_shape", "params": {}})
    t = ctx.tensor("tensor")
    if not isinstance(t, TensorVal):
        return estimate.as_unknown("tensor shape is not statically known")
    shape = list(t.shape)
    target = _parse_ints(ctx.str_widget("target_shape", ""))
    if target and target.count(-1) <= 1 and all(d == -1 or d >= 0 for d in target):
        if -1 in target:
            numel = t.numel()
            rest = 1
            for dim in (d for d in target if d != -1):
                rest *= dim
            if numel is None or rest == 0 or numel % rest != 0:
                shape = [None if d == -1 else d for d in target]
            else:
                shape = [numel // rest if d == -1 else d for d in target]
        else:
            shape = list(target)
    value = TensorVal(tuple(shape), t.itemsize, t.dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("reshaped copy", value.nbytes(), kind="outputs")],
        basis={"key": "tensor_shape", "params": {"shape": shape}},
    )


@register("CdlActivation")
def _estimate_cdl_activation(ctx: EstimationCtx) -> NodeEstimate:
    """Elementwise activation (softmax along a dim): shape and dtype kept."""
    t = ctx.tensor("tensor")
    if not isinstance(t, TensorVal):
        return NodeEstimate(basis={"key": "pass_through", "params": {}}).as_unknown(
            "tensor shape is not statically known"
        )
    return NodeEstimate(
        outputs=[TensorVal(t.shape, t.itemsize, t.dtype)],
        items=[formulas.tensor_item("activation output", t.nbytes(), kind="outputs")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlRandomTensor")
def _estimate_cdl_random_tensor(ctx: EstimationCtx) -> NodeEstimate:
    """Random tensor from a comma-separated shape widget; randint is int64."""
    shape = _parse_ints(ctx.str_widget("shape", ""))
    if not shape or min(shape) <= 0:
        return NodeEstimate(basis={"key": "tensor_shape", "params": {}}).as_unknown(
            "shape widget is not a valid positive dim list"
        )
    dist = ctx.str_widget("dist", "normal")
    itemsize, dtype = (INT64_BYTES, "long") if dist == "randint" else (FLOAT32_BYTES, "float32")
    value = TensorVal(tuple(shape), itemsize, dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("random tensor", value.nbytes(), kind="outputs")],
        basis={"key": "tensor_shape", "params": {"shape": shape, "dist": dist}},
    )


@register("CdlLinReg")
def _estimate_cdl_linreg(ctx: EstimationCtx) -> NodeEstimate:
    """``y_hat = X @ w + b``: one GEMM plus a broadcast add (d2l linreg)."""
    estimate = NodeEstimate(basis={"key": "linreg", "params": {}})
    x = ctx.tensor("X")
    w = ctx.tensor("w")
    if not isinstance(x, TensorVal) or not isinstance(w, TensorVal):
        return estimate.as_unknown("X or w shape is not statically known")
    out = None
    flops = None
    if len(x.shape) == 2 and len(w.shape) == 2:
        out = (x.dim(0), w.dim(1))
        if x.dim(0) is not None and x.dim(1) is not None and w.dim(1) is not None:
            flops = 2 * x.dim(0) * x.dim(1) * w.dim(1)
    elif len(x.shape) == 2 and len(w.shape) == 1:
        out = (x.dim(0),)
        if x.dim(0) is not None and x.dim(1) is not None:
            flops = 2 * x.dim(0) * x.dim(1)
    elif len(x.shape) == 1 and len(w.shape) == 2:
        out = (w.dim(1),)
    else:
        return estimate.as_unknown("unsupported matmul rank combination")
    value = TensorVal(out, x.itemsize, x.dtype)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("y_hat (matmul + broadcast add)", value.nbytes(), kind="outputs")],
        basis={"key": "linreg", "params": {"x": list(x.shape), "w": list(w.shape)}},
    )
    if flops is not None:
        result.flops_items = [formulas.FlopsItem("linear regression matmul", flops, kind="ffn")]
        result.flops_status = "estimated"
    return result


@register("CdlSquaredLoss")
def _estimate_cdl_squared_loss(ctx: EstimationCtx) -> NodeEstimate:
    """``(y_hat - y)**2 / 2`` with y reshaped to y_hat: elementwise, shape kept."""
    t = ctx.tensor("y_hat")
    if not isinstance(t, TensorVal):
        return NodeEstimate(basis={"key": "pass_through", "params": {}}).as_unknown(
            "y_hat shape is not statically known"
        )
    return NodeEstimate(
        outputs=[TensorVal(t.shape, t.itemsize, t.dtype)],
        items=[formulas.tensor_item("elementwise loss", t.nbytes(), kind="outputs")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlMaskedSoftmax")
def _estimate_cdl_masked_softmax(ctx: EstimationCtx) -> NodeEstimate:
    """Masked softmax: X is cloned, masked with -1e6, softmaxed; shape kept."""
    t = ctx.tensor("X")
    if not isinstance(t, TensorVal):
        return NodeEstimate(basis={"key": "pass_through", "params": {}}).as_unknown(
            "X shape is not statically known"
        )
    return NodeEstimate(
        outputs=[TensorVal(t.shape, t.itemsize, t.dtype)],
        items=[formulas.tensor_item("masked softmax output", t.nbytes(), kind="outputs")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlSequenceMask")
def _estimate_cdl_sequence_mask(ctx: EstimationCtx) -> NodeEstimate:
    """Sequence mask (positions >= valid_len set to mask_value): shape kept."""
    t = ctx.tensor("X")
    if not isinstance(t, TensorVal):
        return NodeEstimate(basis={"key": "pass_through", "params": {}}).as_unknown(
            "X shape is not statically known"
        )
    return NodeEstimate(
        outputs=[TensorVal(t.shape, t.itemsize, t.dtype)],
        items=[formulas.tensor_item("masked sequence", t.nbytes(), kind="outputs")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlAccuracy")
def _estimate_cdl_accuracy(ctx: EstimationCtx) -> NodeEstimate:
    """Accuracy scalar (data-dependent) and the sample count."""
    y = ctx.tensor("y")
    count = y.numel() if isinstance(y, TensorVal) else None
    return NodeEstimate(
        outputs=[Unknown("data-dependent accuracy"), IntVal(count)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlSyntheticData")
def _estimate_cdl_synthetic_data(ctx: EstimationCtx) -> NodeEstimate:
    """Synthetic linear-regression data: X (n, f) and y (n, 1), float32."""
    n = max(1, ctx.int_widget("num_examples", 100) or 100)
    f = max(1, ctx.int_widget("num_features", 2) or 2)
    x = TensorVal((n, f), FLOAT32_BYTES)
    y = TensorVal((n, 1), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[x, y],
        items=[
            formulas.tensor_item("synthetic X", x.nbytes(), kind="outputs"),
            formulas.tensor_item("synthetic y", y.nbytes(), kind="outputs"),
        ],
        basis={"key": "synthetic", "params": {"examples": n, "features": f}},
    )


@register("CdlTruncatePad")
def _estimate_cdl_truncate_pad(ctx: EstimationCtx) -> NodeEstimate:
    """Truncate/pad to exactly num_steps, always int64."""
    steps = max(1, ctx.int_widget("num_steps", 64) or 64)
    value = TensorVal((steps,), INT64_BYTES, "long")
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("truncated/padded sequence", value.nbytes(), kind="outputs")],
        basis={"key": "truncate_pad", "params": {"num_steps": steps}},
    )


@register("CdlBleu")
def _estimate_cdl_bleu(ctx: EstimationCtx) -> NodeEstimate:
    """BLEU score: pure-Python n-gram counting, no tensor memory."""
    return NodeEstimate(
        outputs=[Unknown("pure-Python text metric")],
        flops_status="unknown",
        flops_reason="pure-Python n-gram loop - compute is not a matmul FLOPs model",
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlGradClipping")
def _estimate_cdl_grad_clipping(ctx: EstimationCtx) -> NodeEstimate:
    """In-place elementwise gradient rescale: no new allocations, no matmul."""
    return NodeEstimate(
        outputs=[Unknown("gradient norm")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlSgdStep")
def _estimate_cdl_sgd_step(ctx: EstimationCtx) -> NodeEstimate:
    """In-place SGD step: the model value passes through unchanged."""
    return NodeEstimate(
        outputs=[ctx.est("model")],
        basis={"key": "pass_through", "params": {}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 1: object detection (comfydl/nodes/object_detection.py)
# --------------------------------------------------------------------------- #

def _box_map_estimate(ctx: EstimationCtx, label: str) -> NodeEstimate:
    """Shared corner/centre box conversion: (..., 4) -> (N, 4), shape kept."""
    t = ctx.tensor("boxes")
    n = _flat_boxes_count(t)
    if n is None:
        return NodeEstimate(basis={"key": "boxes", "params": {}}).as_unknown(
            "box tensor shape is not statically known"
        )
    itemsize = t.itemsize if isinstance(t, TensorVal) else FLOAT32_BYTES
    dtype = t.dtype if isinstance(t, TensorVal) else "float32"
    value = TensorVal((n, 4), itemsize, dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item(label, value.nbytes(), kind="outputs")],
        basis={"key": "boxes", "params": {"boxes": n}},
    )


@register("CdlBoxCornerToCenter")
def _estimate_cdl_box_corner_to_center(ctx: EstimationCtx) -> NodeEstimate:
    """(x1, y1, x2, y2) -> (cx, cy, w, h) on a flattened (N, 4) view."""
    return _box_map_estimate(ctx, "corner-to-centre boxes")


@register("CdlBoxCenterToCorner")
def _estimate_cdl_box_center_to_corner(ctx: EstimationCtx) -> NodeEstimate:
    """(cx, cy, w, h) -> (x1, y1, x2, y2) on a flattened (N, 4) view."""
    return _box_map_estimate(ctx, "centre-to-corner boxes")


@register("CdlBoxIou")
def _estimate_cdl_box_iou(ctx: EstimationCtx) -> NodeEstimate:
    """Pairwise IoU of two flattened box sets: (N1, N2) float."""
    estimate = NodeEstimate(basis={"key": "boxes", "params": {}})
    n1 = _flat_boxes_count(ctx.tensor("boxes1"))
    n2 = _flat_boxes_count(ctx.tensor("boxes2"))
    if n1 is None or n2 is None:
        return estimate.as_unknown("box set size is not statically known")
    value = TensorVal((n1, n2), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("pairwise IoU matrix", value.nbytes(), kind="outputs")],
        basis={"key": "boxes", "params": {"boxes": "%dx%d" % (n1, n2)}},
    )


@register("CdlNms")
def _estimate_cdl_nms(ctx: EstimationCtx) -> NodeEstimate:
    """Non-maximum suppression: kept count is data-dependent (<= N), int64."""
    boxes = ctx.tensor("boxes")
    result = NodeEstimate(
        outputs=[TensorVal((None,), INT64_BYTES, "long")],
        basis={"key": "nms", "params": {"upper_bound": _flat_boxes_count(boxes)}},
    )
    result.confidence = "approx"
    result.reason = "kept-box count is data-dependent (bounded by the input)"
    return result


@register("CdlMultiboxPrior")
def _estimate_cdl_multibox_prior(ctx: EstimationCtx) -> NodeEstimate:
    """SSD anchors: (1, H*W*(n_sizes+n_ratios-1), 4) float32.

    Without a data link the node builds the 561x728 d2l example map.
    """
    estimate = NodeEstimate(basis={"key": "multibox_prior", "params": {}})
    data = ctx.tensor("data")
    approx = True
    h, w = 561, 728
    if isinstance(data, TensorVal) and len(data.shape) >= 2:
        h, w = data.dim(-2), data.dim(-1)
        approx = h is None or w is None
    if h is None or w is None:
        return estimate.as_unknown("feature-map size is not statically known")
    sizes = _parse_floats(ctx.str_widget("sizes", "")) or [0.75, 0.5, 0.25]
    ratios = _parse_floats(ctx.str_widget("ratios", "")) or [1.0, 2.0, 0.5]
    per_pixel = max(1, len(sizes) + len(ratios) - 1)
    anchors = int(h) * int(w) * per_pixel
    value = TensorVal((1, anchors, 4), FLOAT32_BYTES)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("anchor boxes (1, A, 4)", value.nbytes(), kind="outputs")],
        basis={
            "key": "multibox_prior",
            "params": {"h": h, "w": w, "per_pixel": per_pixel, "anchors": anchors},
        },
    )
    if approx:
        result.confidence = "approx"
        result.reason = "default 561x728 feature map (no data input linked)"
    return result


def _offset_estimate(ctx: EstimationCtx, label: str) -> NodeEstimate:
    """Shared offsets <-> boxes conversion: (N, 4) in, (N, 4) out."""
    anchors = ctx.tensor("anchors")
    n = _flat_boxes_count(anchors)
    if n is None:
        return NodeEstimate(basis={"key": "boxes", "params": {}}).as_unknown(
            "anchor shape is not statically known"
        )
    itemsize = anchors.itemsize if isinstance(anchors, TensorVal) else FLOAT32_BYTES
    dtype = anchors.dtype if isinstance(anchors, TensorVal) else "float32"
    value = TensorVal((n, 4), itemsize, dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item(label, value.nbytes(), kind="outputs")],
        basis={"key": "boxes", "params": {"boxes": n}},
    )


@register("CdlOffsetBoxes")
def _estimate_cdl_offset_boxes(ctx: EstimationCtx) -> NodeEstimate:
    """Ground-truth boxes -> SSD offset labels: (N, 4) elementwise."""
    return _offset_estimate(ctx, "offset labels")


@register("CdlOffsetInverse")
def _estimate_cdl_offset_inverse(ctx: EstimationCtx) -> NodeEstimate:
    """SSD offset predictions -> predicted bboxes: (N, 4) elementwise."""
    return _offset_estimate(ctx, "predicted bboxes")


@register("CdlAssignAnchorToBbox")
def _estimate_cdl_assign_anchor(ctx: EstimationCtx) -> NodeEstimate:
    """Anchor -> ground-truth map: (num_anchors,) int64 (-1 = unassigned)."""
    anchors = ctx.tensor("anchors")
    n = _flat_boxes_count(anchors)
    if n is None:
        return NodeEstimate(basis={"key": "boxes", "params": {}}).as_unknown(
            "anchor count is not statically known"
        )
    value = TensorVal((n,), INT64_BYTES, "long")
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("anchor-to-bbox map", value.nbytes(), kind="outputs")],
        basis={"key": "boxes", "params": {"boxes": n}},
    )


@register("CdlMultiboxTarget")
def _estimate_cdl_multibox_target(ctx: EstimationCtx) -> NodeEstimate:
    """SSD target encoding: bbox_offset/bbox_mask (B, 4A) + labels (B, A)."""
    estimate = NodeEstimate(basis={"key": "multibox_target", "params": {}})
    anchors = ctx.tensor("anchors")
    labels = ctx.tensor("labels")
    a = None
    if isinstance(anchors, TensorVal) and len(anchors.shape) == 3:
        a = anchors.dim(1)
    if a is None:
        a = _flat_boxes_count(anchors)
    b = labels.dim(0) if isinstance(labels, TensorVal) and len(labels.shape) == 3 else None
    if a is None or b is None:
        return estimate.as_unknown("anchor or batch count is not statically known")
    offsets = TensorVal((b, 4 * a), FLOAT32_BYTES)
    mask = TensorVal((b, 4 * a), FLOAT32_BYTES)
    classes = TensorVal((b, a), INT64_BYTES, "long")
    return NodeEstimate(
        outputs=[offsets, mask, classes],
        items=[
            formulas.tensor_item("bbox offsets (B, 4A)", offsets.nbytes(), kind="outputs"),
            formulas.tensor_item("bbox mask (B, 4A)", mask.nbytes(), kind="outputs"),
            formulas.tensor_item("class labels (B, A)", classes.nbytes(), kind="outputs"),
        ],
        basis={"key": "multibox_target", "params": {"anchors": a, "batch": b}},
    )


@register("CdlMultiboxDetection")
def _estimate_cdl_multibox_detection(ctx: EstimationCtx) -> NodeEstimate:
    """SSD detection: (B, A, 6) rows of [class, conf, x1, y1, x2, y2]."""
    estimate = NodeEstimate(basis={"key": "multibox_detection", "params": {}})
    cls = ctx.tensor("cls_probs")
    b, a = None, None
    if isinstance(cls, TensorVal) and len(cls.shape) == 3:
        b, a = cls.dim(0), cls.dim(2)
    if b is None or a is None:
        return estimate.as_unknown("class-probability shape is not statically known")
    value = TensorVal((b, a, 6), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("detections (B, A, 6)", value.nbytes(), kind="outputs")],
        basis={"key": "multibox_detection", "params": {"batch": b, "anchors": a}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 1: semantic segmentation (comfydl/nodes/semantic_segmentation.py)
# --------------------------------------------------------------------------- #

@register("CdlVocClasses")
def _estimate_cdl_voc_classes(ctx: EstimationCtx) -> NodeEstimate:
    """VOC class-name table lookup: text output only."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlVocColormap2Label")
def _estimate_cdl_voc_colormap(ctx: EstimationCtx) -> NodeEstimate:
    """The fixed VOC lookup table: (256**3,) int64 = 128 MiB, fully static."""
    value = TensorVal((256 ** 3,), INT64_BYTES, "long")
    return NodeEstimate(
        outputs=[value],
        items=[
            formulas.MemoryItem(
                "VOC colormap lookup (256^3, int64)",
                value.nbytes(),
                value.nbytes(),
                kind="outputs",
            )
        ],
        basis={"key": "colormap", "params": {}},
    )


@register("CdlVocLabelIndices")
def _estimate_cdl_voc_label_indices(ctx: EstimationCtx) -> NodeEstimate:
    """Colormap image -> per-pixel class indices: MASK (H, W) float32."""
    estimate = NodeEstimate(basis={"key": "rand_crop", "params": {}})
    cm = ctx.tensor("colormap")
    h = cm.dim(1) if isinstance(cm, TensorVal) and len(cm.shape) == 4 else None
    w = cm.dim(2) if isinstance(cm, TensorVal) and len(cm.shape) == 4 else None
    if h is None or w is None:
        return estimate.as_unknown("colormap image size is not statically known")
    value = TensorVal((h, w), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("label mask (H, W)", value.nbytes(), kind="outputs")],
        basis={"key": "label_indices", "params": {"height": h, "width": w}},
    )


@register("CdlVocRandCrop")
def _estimate_cdl_voc_rand_crop(ctx: EstimationCtx) -> NodeEstimate:
    """Random crop: both outputs (1, height, width, C) of the input's channels."""
    estimate = NodeEstimate(basis={"key": "rand_crop", "params": {}})
    feature = ctx.tensor("feature")
    c = feature.dim(3) if isinstance(feature, TensorVal) and len(feature.shape) == 4 else None
    if c is None:
        return estimate.as_unknown("feature channel count is not statically known")
    h = max(1, ctx.int_widget("height", 320) or 320)
    w = max(1, ctx.int_widget("width", 480) or 480)
    cropped = TensorVal((1, h, w, c), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[cropped, TensorVal((1, h, w, c), FLOAT32_BYTES)],
        items=[
            formulas.tensor_item("cropped feature (1, h, w, c)", cropped.nbytes(), kind="outputs"),
            formulas.tensor_item("cropped label (1, h, w, c)", cropped.nbytes(), kind="outputs"),
        ],
        basis={"key": "rand_crop", "params": {"height": h, "width": w}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 1: image tools (comfydl/nodes/image_tools.py)
# --------------------------------------------------------------------------- #

def _image_passthrough(
    ctx: EstimationCtx, label: str, key: str = "pass_through", slot: str = "image"
) -> NodeEstimate:
    estimate = NodeEstimate(basis={"key": key, "params": {}})
    t = ctx.tensor(slot)
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    value = TensorVal(t.shape, t.itemsize, t.dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item(label, value.nbytes(), kind="outputs")],
        basis={"key": key, "params": {}},
    )


@register("CdlImageNormalize")
def _estimate_cdl_image_normalize(ctx: EstimationCtx) -> NodeEstimate:
    """Per-channel z-score normalisation: (B, H, W, C) shape kept, no clamp."""
    return _image_passthrough(ctx, "normalised image (B, H, W, C)")


@register("CdlImageGrayscale")
def _estimate_cdl_image_grayscale(ctx: EstimationCtx) -> NodeEstimate:
    """RGB -> grayscale: torchvision keeps 4 dims and 3 output channels."""
    estimate = NodeEstimate(basis={"key": "pass_through", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    value = TensorVal((t.dim(0), t.dim(1), t.dim(2), 3), t.itemsize, t.dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("grayscale image (B, H, W, 3)", value.nbytes(), kind="outputs")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlImageRotate")
def _estimate_cdl_image_rotate(ctx: EstimationCtx) -> NodeEstimate:
    """Rotation: expand=False keeps the canvas, expand=True uses torchvision's
    ceil(|W cos| + |H sin|) / ceil(|H cos| + |W sin|) canvas."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    b, h, w, c = tuple(t.shape)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    old_h, old_w = h, w
    if ctx.widget("expand", False):
        angle = math.radians(ctx.float_widget("angle", 90.0))
        cos, sin = math.cos(angle), math.sin(angle)
        w = math.ceil(abs(old_w * cos) + abs(old_h * sin))
        h = math.ceil(abs(old_h * cos) + abs(old_w * sin))
    value = TensorVal((b, h, w, c), t.itemsize, t.dtype)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("rotated image", value.nbytes(), kind="outputs")],
        basis={
            "key": "image_scale",
            "params": {"from": "%dx%d" % (old_h, old_w), "to": "%dx%d" % (h, w)},
        },
    )
    result.confidence = "approx"
    return result


@register("CdlImageAdjust")
def _estimate_cdl_image_adjust(ctx: EstimationCtx) -> NodeEstimate:
    """Brightness/contrast/saturation: (B, H, W, C) shape kept."""
    return _image_passthrough(ctx, "adjusted image (B, H, W, C)")


@register("CdlImageStats")
def _estimate_cdl_image_stats(ctx: EstimationCtx) -> NodeEstimate:
    """Per-channel statistics report: text output only."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 1: built-in image scaling (comfy_extras/nodes_post_processing.py,
# nodes_images.py) and image loading (nodes.py).
# --------------------------------------------------------------------------- #

def _scaled_image(ctx: EstimationCtx, slot: str, new_h, new_w) -> NodeEstimate:
    """Shared IMAGE resize tail: keep B and C, swap in the new H/W."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor(slot)
    if not isinstance(t, TensorVal) or len(t.shape) not in (3, 4):
        return estimate.as_unknown("image/mask shape is not statically known")
    shape = list(t.shape)
    old_h, old_w = shape[-3], shape[-2]
    shape[-3], shape[-2] = new_h, new_w
    value = TensorVal(tuple(shape), t.itemsize, t.dtype)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("resized output", value.nbytes(), kind="outputs")],
        basis={
            "key": "image_scale",
            "params": {
                "from": "%sx%s" % (old_h, old_w),
                "to": "%sx%s" % (new_h, new_w),
            },
        },
    )
    result.confidence = "approx"
    return result


@register("ImageScaleToTotalPixels")
def _estimate_image_scale_total_pixels(ctx: EstimationCtx) -> NodeEstimate:
    """Scale to megapixels x 1024 x 1024 with resolution_steps rounding."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    h, w = t.dim(1), t.dim(2)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    megapixels = ctx.float_widget("megapixels", 1.0)
    steps = max(1, ctx.int_widget("resolution_steps", 1) or 1)
    scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    new_w = int(round(w * scale / steps)) * steps
    new_h = int(round(h * scale / steps)) * steps
    return _scaled_image(ctx, "image", new_h, new_w)


@register("ImageScaleToMaxDimension")
def _estimate_image_scale_max_dimension(ctx: EstimationCtx) -> NodeEstimate:
    """Scale so the larger edge equals largest_size, aspect preserved."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    h, w = t.dim(1), t.dim(2)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    size = ctx.int_widget("largest_size", 512) or 0
    if h > w:
        new_h, new_w = size, int(round(w / h * size))
    elif w > h:
        new_w, new_h = size, int(round(h / w * size))
    else:
        new_h = new_w = size
    return _scaled_image(ctx, "image", new_h, new_w)


@register("ResizeImageMaskNode")
def _estimate_resize_image_mask(ctx: EstimationCtx) -> NodeEstimate:
    """The unified resize node: nine modes, all statically computable.

    The helper mirrors each mode's own rounding in
    ``comfy_extras/nodes_post_processing.py`` (Python round, so banker's).
    """
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("input")
    if not isinstance(t, TensorVal) or len(t.shape) not in (3, 4):
        return estimate.as_unknown("image/mask shape is not statically known")
    h, w = t.dim(-3), t.dim(-2)
    if h is None or w is None:
        return estimate.as_unknown("image/mask spatial size is not statically known")
    mode = str(ctx.widget("resize_type", "scale by multiplier") or "")
    if mode == "scale by multiplier":
        m = ctx.float_widget("multiplier", 1.0)
        new_w, new_h = int(round(w * m)), int(round(h * m))
    elif mode == "scale dimensions":
        width = ctx.int_widget("width", 512) or 0
        height = ctx.int_widget("height", 512) or 0
        if width == 0 and height == 0:
            new_w, new_h = w, h
        elif width == 0:
            new_h = height
            new_w = max(1, int(round(w * height / h)))
        elif height == 0:
            new_w = width
            new_h = max(1, int(round(h * width / w)))
        else:
            new_w, new_h = width, height
    elif mode == "scale longer dimension":
        size = ctx.int_widget("longer_size", 512) or 0
        if h > w:
            new_h, new_w = size, int(round(w / h * size))
        elif w > h:
            new_w, new_h = size, int(round(h / w * size))
        else:
            new_h = new_w = size
    elif mode == "scale shorter dimension":
        size = ctx.int_widget("shorter_size", 512) or 0
        if h < w:
            new_h, new_w = size, int(round(w / h * size))
        elif w < h:
            new_w, new_h = size, int(round(h / w * size))
        else:
            new_h = new_w = size
    elif mode == "scale width":
        new_w = ctx.int_widget("width", 512) or 0
        new_h = max(1, int(round(h * new_w / w)))
    elif mode == "scale height":
        new_h = ctx.int_widget("height", 512) or 0
        new_w = max(1, int(round(w * new_h / h)))
    elif mode == "scale total pixels":
        total = int(ctx.float_widget("megapixels", 1.0) * 1024 * 1024)
        scale = math.sqrt(total / (w * h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
    elif mode == "match size":
        match = ctx.tensor("match")
        new_h = match.dim(-3) if isinstance(match, TensorVal) else None
        new_w = match.dim(-2) if isinstance(match, TensorVal) else None
        if new_h is None or new_w is None:
            return estimate.as_unknown("match reference shape is not statically known")
    elif mode == "scale to multiple":
        multiple = max(1, ctx.int_widget("multiple", 8) or 1)
        new_w = (w // multiple) * multiple
        new_h = (h // multiple) * multiple
    else:
        return estimate.as_unknown("resize_type mode is not recognised")
    return _scaled_image(ctx, "input", new_h, new_w)


def _load_image_estimate(ctx: EstimationCtx, folder: str) -> NodeEstimate:
    """Shared LoadImage tail: decode size is only known at run time.

    The compressed file (os.stat best effort) is a lower bound; the decoded
    tensors are (1, H, W, 3) IMAGE + (1, H, W) MASK with H/W unknown.
    """
    name = ctx.str_widget("image", "")
    filename, annotated_folder = _annotated_filepath(name)
    estimate = NodeEstimate(basis={"key": "load_image", "params": {"name": name}})
    estimate.outputs = [
        TensorVal((1, None, None, 3), FLOAT32_BYTES),
        TensorVal((1, None, None), FLOAT32_BYTES),
    ]
    estimate.confidence = "approx"
    estimate.reason = "pixel size, channel count and frame count follow the decode"
    file_bytes = _file_size(annotated_folder or folder, filename)
    if file_bytes is not None:
        estimate.items = [
            formulas.tensor_item(
                "image file (compressed, on disk)", file_bytes, kind="inputs", approx=True
            )
        ]
        estimate.basis["params"] = {"name": name, "file_bytes": file_bytes}
    return estimate


@register("LoadImage")
def _estimate_load_image(ctx: EstimationCtx) -> NodeEstimate:
    """Load an image from the input folder: [B, H, W, 3] + [B, H, W] mask."""
    return _load_image_estimate(ctx, "input")


@register("LoadImageMask")
def _estimate_load_image_mask(ctx: EstimationCtx) -> NodeEstimate:
    """Load one channel (alpha default) of an image as a [B, H, W] mask."""
    name = ctx.str_widget("image", "")
    filename, annotated_folder = _annotated_filepath(name)
    estimate = NodeEstimate(basis={"key": "load_image", "params": {"name": name}})
    estimate.outputs = [TensorVal((1, None, None), FLOAT32_BYTES)]
    estimate.confidence = "approx"
    estimate.reason = "pixel size follows the decode"
    file_bytes = _file_size(annotated_folder or "input", filename)
    if file_bytes is not None:
        estimate.items = [
            formulas.tensor_item(
                "image file (compressed, on disk)", file_bytes, kind="inputs", approx=True
            )
        ]
        estimate.basis["params"] = {"name": name, "file_bytes": file_bytes}
    return estimate


@register("LoadImageOutput")
def _estimate_load_image_output(ctx: EstimationCtx) -> NodeEstimate:
    """Load an image from the output folder: same decode semantics."""
    return _load_image_estimate(ctx, "output")


# --------------------------------------------------------------------------- #
# M3 batch 1: cross-correlation (comfydl/nodes/model_cv.py, tensor part)
# --------------------------------------------------------------------------- #

@register("CdlCorr2d")
def _estimate_cdl_corr2d(ctx: EstimationCtx) -> NodeEstimate:
    """d2l corr2d: a Python window loop over a 2D input, output (H-kh+1, W-kw+1)."""
    estimate = NodeEstimate(basis={"key": "corr2d", "params": {}})
    x = ctx.tensor("input_tensor")
    k = ctx.tensor("kernel")
    if not isinstance(x, TensorVal) or not isinstance(k, TensorVal):
        return estimate.as_unknown("input or kernel shape is not statically known")
    if len(x.shape) != 2 or len(k.shape) != 2:
        return estimate.as_unknown("corr2d takes 2D input and a 2D kernel")
    h, w = x.shape
    kh, kw = k.shape
    if h is None or w is None or kh is None or kw is None:
        return estimate.as_unknown("spatial dims not fully known")
    if kh > h or kw > w:
        return estimate.as_unknown("kernel is larger than the input")
    ho, wo = h - kh + 1, w - kw + 1
    value = TensorVal((ho, wo), FLOAT32_BYTES)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("cross-correlation output", value.nbytes(), kind="outputs")],
        basis={"key": "corr2d", "params": {"in": [h, w], "kernel": [kh, kw], "out": [ho, wo]}},
    )
    result.flops_items = [
        formulas.FlopsItem("corr2d window loop", 2 * ho * wo * kh * kw, kind="conv")
    ]
    result.flops_status = "estimated"
    return result


# --------------------------------------------------------------------------- #
# M3 batch 2: module constructors (cdlModel builders).
#
# A constructor publishes a ModuleVal: the parameter count it implies, the
# family it belongs to and its declared width, so downstream nodes (a head
# on top of an RNN, ModelInfo, ModelClone, ...) can be sized without
# running torch. Layers built with nn.LazyLinear have no weights before
# their first forward - those sizes are reported as None, never guessed.
# --------------------------------------------------------------------------- #

def _module_of(ctx: EstimationCtx, slot: str) -> Optional[ModuleVal]:
    return ctx.est(slot) if isinstance(ctx.est(slot), ModuleVal) else None


def _module_result(
    params: Optional[int],
    kind_hint: str,
    width: Optional[int] = None,
    extra_items: Optional[list] = None,
    basis_params: Optional[dict] = None,
    confidence: str = "exact",
    reason: str = "",
) -> NodeEstimate:
    items = []
    if params is not None:
        items.append(
            formulas.MemoryItem(
                "parameters (float32)", int(params) * 4, int(params) * 4, kind="params"
            )
        )
    items.extend(extra_items or [])
    result = NodeEstimate(
        outputs=[ModuleVal(params, kind_hint, width)],
        items=items,
        basis={"key": "module", "params": basis_params or {}},
    )
    result.confidence = confidence
    result.reason = reason
    return result


@register("CdlRNNScratch")
def _estimate_cdl_rnn_scratch(ctx: EstimationCtx) -> NodeEstimate:
    """d2l RNN from scratch: W_xh (I, H) + W_hh (H, H) + b_h (H)."""
    i = ctx.int_widget("num_inputs", 32)
    h = ctx.int_widget("num_hiddens", 64)
    if i is None or h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "rnn dimensions are not statically known"
        )
    return _module_result(
        i * h + h * h + h, "rnn_scratch", h,
        basis_params={"kind": "rnn scratch", "inputs": i, "hiddens": h,
                      "params": i * h + h * h + h},
    )


@register("CdlRNN")
def _estimate_cdl_rnn(ctx: EstimationCtx) -> NodeEstimate:
    """torch nn.RNN, single layer, both biases, tanh."""
    i = ctx.int_widget("num_inputs", 32)
    h = ctx.int_widget("num_hiddens", 64)
    if i is None or h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "rnn dimensions are not statically known"
        )
    params = formulas.rnn_param_count(i, h, gates=1)
    return _module_result(
        params, "rnn", h,
        basis_params={"kind": "nn.RNN", "inputs": i, "hiddens": h, "params": params},
    )


@register("CdlGRU")
def _estimate_cdl_gru(ctx: EstimationCtx) -> NodeEstimate:
    """torch nn.GRU: 3 gates per layer, layer 0 maps I -> H, deeper H -> H."""
    i = ctx.int_widget("num_inputs", 32)
    h = ctx.int_widget("num_hiddens", 64)
    layers = max(1, ctx.int_widget("num_layers", 1) or 1)
    if i is None or h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "gru dimensions are not statically known"
        )
    params = formulas.gru_param_count(i, h, layers)
    return _module_result(
        params, "gru", h,
        basis_params={"kind": "nn.GRU", "inputs": i, "hiddens": h,
                      "layers": layers, "params": params},
    )


@register("CdlRNNLMScratch")
def _estimate_cdl_rnnlm_scratch(ctx: EstimationCtx) -> NodeEstimate:
    """RNN language model on a scratch RNN: + output head (H, V) + bias (V).

    The node reads ``rnn.sigma`` in d2l's init_params, so only the scratch
    RNN can be plugged in here - anything else raises at construction.
    """
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    rnn = _module_of(ctx, "rnn")
    vocab = max(1, ctx.int_widget("vocab_size", 32) or 32)
    if rnn is None or rnn.param_count is None or rnn.num_hiddens is None:
        return estimate.as_unknown("rnn size is not statically known")
    if rnn.kind_hint != "rnn_scratch":
        return estimate.as_unknown(
            "RNNLMScratch needs the scratch RNN (it reads rnn.sigma)"
        )
    head = rnn.num_hiddens * vocab + vocab
    params = rnn.param_count + head
    return _module_result(
        params, "rnn_lm", rnn.num_hiddens,
        basis_params={"kind": "RNNLMScratch", "rnn_params": rnn.param_count,
                      "vocab": vocab, "head": head, "params": params},
    )


@register("CdlRNNLM")
def _estimate_cdl_rnnlm(ctx: EstimationCtx) -> NodeEstimate:
    """RNN language model on a torch RNN/GRU: head is nn.LazyLinear(V).

    The lazy head materialises (V, H) + (V,) at the first forward; before
    that it contributes nothing, which is why this one is approx.
    """
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    rnn = _module_of(ctx, "rnn")
    vocab = max(1, ctx.int_widget("vocab_size", 32) or 32)
    if rnn is None or rnn.param_count is None or rnn.num_hiddens is None:
        return estimate.as_unknown("rnn size is not statically known")
    if rnn.kind_hint not in ("rnn", "gru"):
        return estimate.as_unknown(
            "RNNLM needs a torch RNN/GRU (the lazy head wants a tensor output)"
        )
    head = rnn.num_hiddens * vocab + vocab
    params = rnn.param_count + head
    return _module_result(
        params, "rnn_lm", rnn.num_hiddens,
        basis_params={"kind": "RNNLM", "rnn_params": rnn.param_count,
                      "vocab": vocab, "head": head, "params": params},
        confidence="approx",
        reason="the output head is LazyLinear: it materialises at the first forward",
    )


@register("CdlRNNLMScratchPredict")
def _estimate_cdl_rnnlm_predict(ctx: EstimationCtx) -> NodeEstimate:
    """Autoregressive prefix continuation: one 1-step forward per token.

    ``len(prefix)`` warm-up steps plus ``num_preds`` sampled steps, no KV
    cache, so every step re-runs the whole recurrent cell.
    """
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    model = _module_of(ctx, "model")
    prefix = ctx.str_widget("prefix", "")
    steps = max(1, ctx.int_widget("num_preds", 10) or 10)
    if model is None or model.param_count is None:
        return estimate.as_unknown("model size is not statically known")
    total_steps = len(prefix) + steps
    vocab_value = ctx.est("vocab")
    vocab = vocab_value.size if isinstance(vocab_value, VocabVal) else None
    if vocab is None and isinstance(vocab_value, IntVal):
        vocab = vocab_value.value
    # The output head is the only part that scales with the vocabulary:
    # everything else is the recurrent cell itself.
    head = (
        model.num_hiddens * vocab + vocab
        if (model.num_hiddens is not None and vocab)
        else None
    )
    result = NodeEstimate(
        outputs=[TextVal(None)],
        basis={
            "key": "module",
            "params": {"kind": "predict", "params": model.param_count,
                       "steps": total_steps, "predicted": steps, "vocab": vocab},
        },
    )
    if head is None:
        result.flops_items = [
            formulas.FlopsItem(
                "forward x %d steps (no cache)" % total_steps,
                2 * model.param_count * total_steps,
                kind="other",
                approx=True,
            )
        ]
    else:
        hidden = max(0, model.param_count - head)
        result.flops_items = [
            formulas.FlopsItem(
                "recurrent cell x %d steps" % total_steps,
                2 * hidden * total_steps,
                kind="ffn",
                approx=True,
            ),
            formulas.FlopsItem(
                "output head x %d steps" % total_steps,
                2 * head * total_steps,
                kind="logits",
                approx=True,
            ),
        ]
    result.flops_status = "estimated"
    result.confidence = "approx"
    result.reason = "2 x params x tokens (one forward per step, no KV cache)"
    return result


@register("CdlSeq2SeqEncoder")
def _estimate_cdl_seq2seq_encoder(ctx: EstimationCtx) -> NodeEstimate:
    """Embedding (V, E) stacked with a GRU (E -> H, num_layers deep)."""
    vocab = ctx.int_widget("vocab_size", 32)
    embed = ctx.int_widget("embed_size", 16)
    h = ctx.int_widget("num_hiddens", 16)
    layers = max(1, ctx.int_widget("num_layers", 2) or 2)
    if vocab is None or embed is None or h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "encoder dimensions are not statically known"
        )
    params = vocab * embed + formulas.gru_param_count(embed, h, layers)
    return _module_result(
        params, "seq2seq", h,
        basis_params={"kind": "seq2seq encoder", "vocab": vocab, "embed": embed,
                      "hiddens": h, "layers": layers, "params": params},
    )


@register("CdlInitSeq2Seq")
def _estimate_cdl_init_seq2seq(ctx: EstimationCtx) -> NodeEstimate:
    """Xavier re-initialisation in place: shapes and counts unchanged."""
    return NodeEstimate(
        outputs=[ctx.est("model")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlDotProductAttention")
def _estimate_cdl_dot_product_attention(ctx: EstimationCtx) -> NodeEstimate:
    """Scaled dot-product attention: no weights at all (scaling is 1/sqrt(d))."""
    return _module_result(
        0, "attention", None,
        basis_params={"kind": "dot-product attention", "params": 0},
    )


@register("CdlAdditiveAttention")
def _estimate_cdl_additive_attention(ctx: EstimationCtx) -> NodeEstimate:
    """Additive attention: three bias-free lazy projections W_k, W_q, w_v.

    Sizes follow the first forward's query/key widths; assuming the common
    self-attention case (both equal to num_hiddens) gives H x (2H + 1).
    """
    h = ctx.int_widget("num_hiddens", 8)
    if h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "hidden width is not statically known"
        )
    return _module_result(
        h * (2 * h + 1), "attention", h,
        basis_params={"kind": "additive attention", "hiddens": h,
                      "params": h * (2 * h + 1)},
        confidence="approx",
        reason="lazy W_q/W_k assume self-attention (query/key width = num_hiddens)",
    )


@register("CdlMultiHeadAttention")
def _estimate_cdl_multihead_attention(ctx: EstimationCtx) -> NodeEstimate:
    """Four lazy projections q/k/v/output; the head count is a reshape.

    num_hiddens % num_heads != 0 makes the node raise, so it is unknown.
    """
    h = ctx.int_widget("num_hiddens", 8)
    heads = max(1, ctx.int_widget("num_heads", 4) or 4)
    bias = bool(ctx.widget("use_bias", False))
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    if h is None:
        return estimate.as_unknown("hidden width is not statically known")
    if heads > h or h % heads != 0:
        return estimate.as_unknown("num_hiddens is not divisible by num_heads")
    return _module_result(
        formulas.mha_param_count(h, h, bias), "attention", h,
        basis_params={"kind": "multi-head attention", "hiddens": h, "heads": heads,
                      "bias": bias, "params": formulas.mha_param_count(h, h, bias)},
        confidence="approx",
        reason="lazy projections assume self-attention (input width = num_hiddens)",
    )


@register("CdlPositionalEncoding")
def _estimate_cdl_positional_encoding(ctx: EstimationCtx) -> NodeEstimate:
    """No parameters, but a (1, max_len, num_hiddens) sine table in memory.

    d2l keeps it as a plain tensor attribute, so it costs bytes without
    ever showing up in a state_dict.
    """
    h = ctx.int_widget("num_hiddens", 16)
    max_len = max(1, ctx.int_widget("max_len", 1000) or 1000)
    if h is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "hidden width is not statically known"
        )
    table_bytes = formulas.positional_encoding_bytes(max_len, h)
    return _module_result(
        0, "positional", h,
        extra_items=[formulas.MemoryItem(
            "sine/cosine table (1, max_len, d)", table_bytes, table_bytes, kind="misc"
        )],
        basis_params={"kind": "positional encoding", "hiddens": h, "max_len": max_len,
                      "params": 0},
    )


@register("CdlPositionWiseFFN")
def _estimate_cdl_position_wise_ffn(ctx: EstimationCtx) -> NodeEstimate:
    """Two biased lazy linears I->F and F->O; only F and O are in the graph.

    The incoming width decides dense1's weight count, so the total stays
    unknown rather than assuming a width.
    """
    f = ctx.int_widget("ffn_num_hiddens", 64)
    o = ctx.int_widget("ffn_num_outputs", 16)
    if f is None or o is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "ffn widths are not statically known"
        )
    return _module_result(
        None, "ffn", o,
        basis_params={"kind": "position-wise ffn", "ffn_hiddens": f, "ffn_outputs": o},
        confidence="approx",
        reason="dense1 is LazyLinear: its input width is known only after a forward",
    )


@register("CdlAddNorm")
def _estimate_cdl_add_norm(ctx: EstimationCtx) -> NodeEstimate:
    """Residual + layer-norm: exactly the norm's weight and bias."""
    shape = ctx.int_widget("norm_shape", 16)
    if shape is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "norm shape is not statically known"
        )
    return _module_result(
        2 * shape, "norm", shape,
        basis_params={"kind": "add & norm", "norm_shape": shape, "params": 2 * shape},
    )


@register("CdlTransformerEncoderBlock")
def _estimate_cdl_transformer_block(ctx: EstimationCtx) -> NodeEstimate:
    """One encoder block: attention + addnorm + position-wise FFN + addnorm."""
    h = ctx.int_widget("num_hiddens", 8)
    f = ctx.int_widget("ffn_num_hiddens", 64)
    heads = max(1, ctx.int_widget("num_heads", 4) or 4)
    bias = bool(ctx.widget("use_bias", False))
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    if h is None or f is None:
        return estimate.as_unknown("block widths are not statically known")
    if heads > h or h % heads != 0:
        return estimate.as_unknown("num_hiddens is not divisible by num_heads")
    params = formulas.transformer_block_param_count(h, f, bias)
    return _module_result(
        params, "transformer", h,
        basis_params={"kind": "transformer block", "hiddens": h, "ffn": f,
                      "heads": heads, "bias": bias, "params": params},
        confidence="approx",
        reason="internal lazy layers assume the input width equals num_hiddens",
    )


@register("CdlTransformerEncoder")
def _estimate_cdl_transformer_encoder(ctx: EstimationCtx) -> NodeEstimate:
    """Embedding + positional table + num_blks encoder blocks, no head layer.

    Inside the encoder every lazy layer sees num_hiddens (the embedding
    width), so unlike the standalone blocks this one is exact.
    """
    vocab = ctx.int_widget("vocab_size", 32)
    h = ctx.int_widget("num_hiddens", 8)
    f = ctx.int_widget("ffn_num_hiddens", 64)
    heads = max(1, ctx.int_widget("num_heads", 4) or 4)
    blocks = max(1, ctx.int_widget("num_blks", 2) or 2)
    bias = bool(ctx.widget("use_bias", False))
    estimate = NodeEstimate(basis={"key": "module", "params": {}})
    if vocab is None or h is None or f is None:
        return estimate.as_unknown("encoder widths are not statically known")
    if heads > h or h % heads != 0:
        return estimate.as_unknown("num_hiddens is not divisible by num_heads")
    params = vocab * h + blocks * formulas.transformer_block_param_count(h, f, bias)
    table_bytes = formulas.positional_encoding_bytes(1000, h)  # max_len is fixed at 1000
    return _module_result(
        params, "transformer", h,
        extra_items=[formulas.MemoryItem(
            "sine/cosine table (1, 1000, d)", table_bytes, table_bytes, kind="misc"
        )],
        basis_params={"kind": "transformer encoder", "vocab": vocab, "hiddens": h,
                      "ffn": f, "heads": heads, "blocks": blocks, "bias": bias,
                      "params": params},
    )


@register("CdlLeNet")
def _estimate_cdl_lenet(ctx: EstimationCtx) -> NodeEstimate:
    """LeNet-5: every layer is Lazy, so the count follows an input shape.

    Counted for the MNIST shape d2l uses (1 x 28 x 28); any other input
    materialises different widths, hence "approx".
    """
    classes = ctx.int_widget("num_classes", 10)
    if classes is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "class count is not statically known"
        )
    return _module_result(
        formulas.lenet_param_count(classes), "conv", None,
        basis_params={"kind": "LeNet-5", "classes": classes,
                      "params": formulas.lenet_param_count(classes)},
        confidence="approx",
        reason="Lazy layers materialise at the first forward (counted for 1x28x28)",
    )


@register("CdlResNet18")
def _estimate_cdl_resnet18(ctx: EstimationCtx) -> NodeEstimate:
    """d2l ResNet-18: a non-lazy stem followed by four residual stages."""
    classes = ctx.int_widget("num_classes", 10)
    channels = ctx.int_widget("in_channels", 1)
    if classes is None or channels is None:
        return NodeEstimate(basis={"key": "module", "params": {}}).as_unknown(
            "network dimensions are not statically known"
        )
    params = formulas.resnet18_param_count(channels, classes)
    return _module_result(
        params, "conv", None,
        basis_params={"kind": "ResNet-18", "in_channels": channels,
                      "classes": classes, "params": params},
    )


def _lazy_conv_block(ctx: EstimationCtx, kind: str) -> NodeEstimate:
    """A residual/ResNeXt block: LazyConv2d + LazyBatchNorm all the way."""
    result = _module_result(
        None, "conv", None,
        basis_params={"kind": kind},
        confidence="approx",
    )
    result.reason = (
        "LazyConv2d / LazyBatchNorm: the channel counts materialise at the "
        "first forward"
    )
    return result


@register("CdlResidual")
def _estimate_cdl_residual(ctx: EstimationCtx) -> NodeEstimate:
    """Residual block: lazily sized, so the count is deferred honestly."""
    return _lazy_conv_block(ctx, "residual block")


@register("CdlResNeXtBlock")
def _estimate_cdl_resnext_block(ctx: EstimationCtx) -> NodeEstimate:
    """ResNeXt block: lazily sized, so the count is deferred honestly."""
    return _lazy_conv_block(ctx, "ResNeXt block")


@register("CdlModelInfo")
def _estimate_cdl_model_info(ctx: EstimationCtx) -> NodeEstimate:
    """Model summary text plus the two parameter counts it prints."""
    model = _module_of(ctx, "model")
    params = model.param_count if model else None
    return NodeEstimate(
        outputs=[TextVal(None), IntVal(params), IntVal(params)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlModelMode")
def _estimate_cdl_model_mode(ctx: EstimationCtx) -> NodeEstimate:
    """train()/eval() in place: nothing about the model changes."""
    return NodeEstimate(
        outputs=[ctx.est("model")],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlModelLayers")
def _estimate_cdl_model_layers(ctx: EstimationCtx) -> NodeEstimate:
    """named_modules() tree as text."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlModelParams")
def _estimate_cdl_model_params(ctx: EstimationCtx) -> NodeEstimate:
    """Parameter names/shapes as text."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlModelClone")
def _estimate_cdl_model_clone(ctx: EstimationCtx) -> NodeEstimate:
    """copy.deepcopy: a second, fully independent set of weights."""
    model = _module_of(ctx, "model")
    estimate = NodeEstimate(
        outputs=[model if model is not None else ctx.est("model")],
        basis={"key": "pass_through", "params": {}},
    )
    if model is None or model.param_count is None:
        estimate.confidence = "approx"
        estimate.reason = "clone is a deepcopy; its size follows the source model"
        return estimate
    estimate.items = [
        formulas.MemoryItem(
            "cloned weights (source + copy both live)",
            model.param_count * 4,
            model.param_count * 4,
            kind="params",
        )
    ]
    return estimate


@register("CdlModelSave")
def _estimate_cdl_model_save(ctx: EstimationCtx) -> NodeEstimate:
    """state_dict write: disk traffic, no allocation worth counting."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlModelLoad")
def _estimate_cdl_model_load(ctx: EstimationCtx) -> NodeEstimate:
    """load_state_dict in place: the architecture is unchanged."""
    return NodeEstimate(
        outputs=[ctx.est("model")],
        basis={"key": "pass_through", "params": {}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 2: NLP helpers (comfydl/nodes/nlp_utils.py)
# --------------------------------------------------------------------------- #

def _vocab_lines(text: str) -> list:
    """Non-empty, stripped lines - the shared first step of CdlVocabBuild."""
    return [line.strip() for line in str(text or "").split("\n") if line.strip()]


@register("CdlTokenize")
def _estimate_cdl_tokenize(ctx: EstimationCtx) -> NodeEstimate:
    """Line-wise tokenizer: word -> split(), char -> every character."""
    lines = _vocab_lines(ctx.str_widget("text", ""))
    mode = ctx.str_widget("token_mode", "word")
    count = sum(len(line.split()) if mode == "word" else len(line) for line in lines)
    tokens_per_line = [len(line.split()) if mode == "word" else len(line) for line in lines]
    return NodeEstimate(
        outputs=[TextVal(None)],
        flops_status="unknown",
        flops_reason="pure-Python text loop - compute is not a matmul FLOPs model",
        basis={"key": "cdl_tokenize", "params": {"mode": mode, "tokens": count,
                                                 "lines": len(lines),
                                                 "per_line": tokens_per_line}},
    )


@register("CdlGetTokensAndSegments")
def _estimate_cdl_get_tokens_and_segments(ctx: EstimationCtx) -> NodeEstimate:
    """BERT-style pair: <cls> A <sep> [B <sep>] plus the 0/1 segment ids."""
    def _count(value: str) -> int:
        return len([part for part in str(value or "").split(",") if part.strip()])

    a = _count(ctx.str_widget("tokens_a", ""))
    b_raw = ctx.str_widget("tokens_b", "")
    b = _count(b_raw) if str(b_raw or "").strip() else 0
    tokens = a + 2 + (b + 1 if b else 0)
    return NodeEstimate(
        outputs=[TextVal(None), TextVal(None)],
        flops_status="unknown",
        flops_reason="pure-Python list building - not a matmul FLOPs model",
        basis={"key": "cdl_segments", "params": {"tokens": tokens, "first": a,
                                                 "second": b}},
    )


@register("CdlVocabBuild")
def _estimate_cdl_vocab_build(ctx: EstimationCtx) -> NodeEstimate:
    """Vocabulary from a token text: <unk> + reserved + tokens at min_freq.

    Unlike TextVocabBuild this node has a reserved-token widget and sorts
    alphabetically, so the size is derived the same way the node does.
    """
    text = ctx.str_widget("tokens_text", "")
    min_freq = max(1, ctx.int_widget("min_freq", 1) or 1)
    reserved_raw = ctx.str_widget("reserved_tokens", "")
    reserved = [token.strip() for token in str(reserved_raw).split(",") if token.strip()]
    tokens: list = []
    for line in _vocab_lines(text):
        if "," in line:
            tokens.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            tokens.extend(line.split())
    counts = Counter(tokens)
    kept = [token for token, count in counts.items() if count >= min_freq]
    size = len(set(["<unk>"] + reserved + kept))
    return NodeEstimate(
        outputs=[VocabVal(size, "word"), IntVal(size)],
        flops_status="unknown",
        flops_reason="pure-Python counting loop - compute is not a matmul FLOPs model",
        basis={"key": "vocab_build", "params": {
            "chars": len(str(text or "")), "tokens": len(tokens), "size": size,
            "level": "word", "reserved": len(reserved), "min_freq": min_freq}},
    )


@register("CdlVocabEncode")
def _estimate_cdl_vocab_encode(ctx: EstimationCtx) -> NodeEstimate:
    """Tokens -> indices: (N,) int64 with N the comma-separated token count."""
    tokens = [part for part in str(ctx.str_widget("tokens", "") or "").split(",")
              if part.strip()]
    count = len(tokens)
    value = TensorVal((count,), INT64_BYTES, "long")
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("encoded indices (int64)", value.nbytes(), kind="outputs")],
        flops_status="unknown",
        flops_reason="pure-Python dict lookups - not a matmul FLOPs model",
        basis={"key": "cdl_vocab_encode", "params": {"tokens": count}},
    )


@register("CdlVocabDecode")
def _estimate_cdl_vocab_decode(ctx: EstimationCtx) -> NodeEstimate:
    """Indices -> tokens text; the length follows the tensor's element count."""
    indices = ctx.tensor("indices")
    count = indices.numel() if isinstance(indices, TensorVal) else None
    if count == 0:
        count = 1  # a scalar list still carries one index
    return NodeEstimate(
        outputs=[TextVal(None)],
        flops_status="unknown",
        flops_reason="pure-Python list lookups - not a matmul FLOPs model",
        basis={"key": "cdl_vocab_decode", "params": {"tokens": count}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 3: built-in loaders and the diffusion backbone.
#
# A loader's memory is the file: comfy.utils.load_torch_file reads the whole
# state dict into CPU RAM, and this build never calls load_models_gpu, so
# the reported bytes are CPU-resident weights. Files that are not on disk
# stay unknown rather than assumed.
# --------------------------------------------------------------------------- #

def _weights_estimate(
    file_bytes: Optional[int],
    name: str,
    folder_key: str,
    label: str,
    extra_outputs: int = 0,
) -> NodeEstimate:
    """One loader's report: the file as resident weight bytes.

    ``extra_outputs`` covers the secondary slots (a checkpoint publishes
    MODEL + CLIP + VAE) - they get an unknown share because the prefix
    bucket split is not statically known.
    """
    estimate = NodeEstimate(
        basis={"key": "weights_load", "params": {"name": name, "folder": folder_key}}
    )
    if file_bytes is None:
        return estimate.as_unknown("model file not found on this machine")
    estimate.outputs = [WeightsVal(file_bytes, name)]
    for _ in range(extra_outputs):
        estimate.outputs.append(WeightsVal(-1, name))
    estimate.items = [
        formulas.MemoryItem(label, file_bytes, file_bytes, kind="params", approx=True)
    ]
    estimate.confidence = "approx"
    estimate.reason = (
        "weights are CPU-resident at load time (no device allocation in this build)"
    )
    estimate.basis["params"] = {"name": name, "folder": folder_key,
                                "file_bytes": file_bytes}
    return estimate


@register("CheckpointLoaderSimple")
def _estimate_checkpoint_loader(ctx: EstimationCtx) -> NodeEstimate:
    """Checkpoint -> MODEL + CLIP + VAE; the prefix split is not static."""
    return _weights_estimate(
        _model_file_size("checkpoints", ctx.str_widget("ckpt_name", "")),
        ctx.str_widget("ckpt_name", ""),
        "checkpoints",
        "checkpoint weights (whole file, prefix split unknown)",
        extra_outputs=2,
    )


@register("UNETLoader")
def _estimate_unet_loader(ctx: EstimationCtx) -> NodeEstimate:
    """Diffusion model (UNet) only."""
    return _weights_estimate(
        _model_file_size("diffusion_models", ctx.str_widget("unet_name", "")),
        ctx.str_widget("unet_name", ""),
        "diffusion_models",
        "diffusion model weights",
    )


@register("VAELoader")
def _estimate_vae_loader(ctx: EstimationCtx) -> NodeEstimate:
    """VAE only."""
    return _weights_estimate(
        _model_file_size("vae", ctx.str_widget("vae_name", "")),
        ctx.str_widget("vae_name", ""),
        "vae",
        "vae weights",
    )


@register("CLIPLoader")
def _estimate_clip_loader(ctx: EstimationCtx) -> NodeEstimate:
    """Text encoder only."""
    return _weights_estimate(
        _model_file_size("text_encoders", ctx.str_widget("clip_name", "")),
        ctx.str_widget("clip_name", ""),
        "text_encoders",
        "text-encoder weights",
    )


@register("DualCLIPLoader")
def _estimate_dual_clip_loader(ctx: EstimationCtx) -> NodeEstimate:
    """Two text encoders held at once (SDXL style)."""
    estimate = NodeEstimate(basis={"key": "weights_load", "params": {}})
    sizes = []
    for widget in ("clip_name1", "clip_name2"):
        size = _model_file_size("text_encoders", ctx.str_widget(widget, ""))
        if size is None:
            return estimate.as_unknown("model file not found on this machine")
        sizes.append(size)
    total = sum(sizes)
    estimate.outputs = [WeightsVal(total, "dual clip")]
    estimate.items = [
        formulas.MemoryItem("two text encoders (both resident)", total,
                            max(sizes), kind="params", approx=True)
    ]
    estimate.confidence = "approx"
    estimate.basis["params"] = {"files": sizes, "file_bytes": total}
    return estimate


def _lora_unknown(ctx: EstimationCtx, widget: str) -> NodeEstimate:
    """LoRA application needs the model internals this build does not ship."""
    estimate = NodeEstimate(
        basis={"key": "weights_load", "params": {"name": ctx.str_widget(widget, "")}}
    )
    return estimate.as_unknown(
        "LoRA patching needs the model internals (this build's node raises)"
    )


@register("LoraLoader")
def _estimate_lora_loader(ctx: EstimationCtx) -> NodeEstimate:
    """model + clip LoRA: unmodelled here, honestly."""
    return _lora_unknown(ctx, "lora_name")


@register("LoraLoaderModelOnly")
def _estimate_lora_loader_model_only(ctx: EstimationCtx) -> NodeEstimate:
    """model-only LoRA: unmodelled here, honestly."""
    return _lora_unknown(ctx, "lora_name")


# -- the backbone: VAE encode/decode use comfy/sd.py's own memory formulas,
#    with the SD AutoencoderKL defaults recorded as assumptions.

def _vae_memory_encode(height: int, width: int, dtype_size: int = FLOAT32_BYTES) -> int:
    """``comfy/sd.py:432`` - VAE.memory_used_encode for AutoencoderKL."""
    return int(1767 * int(height) * int(width) * int(dtype_size))


def _vae_memory_decode(lat_h: int, lat_w: int, dtype_size: int = FLOAT32_BYTES) -> int:
    """``comfy/sd.py:433`` - VAE.memory_used_decode for AutoencoderKL."""
    return int(2178 * int(lat_h) * int(lat_w) * 64 * int(dtype_size))


@register("VAEDecode")
def _estimate_vae_decode(ctx: EstimationCtx) -> NodeEstimate:
    """Latent -> image: 8x upscale, 3 channels, plus the decoder's workspace.

    This build's node raises (the VAE implementation is dehydrated); the
    estimate follows the formulas and defaults kept in ``comfy/sd.py``.
    """
    estimate = NodeEstimate(basis={"key": "vae_decode", "params": {}})
    latent = ctx.tensor("samples")
    if not isinstance(latent, TensorVal) or len(latent.shape) != 4:
        return estimate.as_unknown("latent shape is not statically known")
    channels, lat_h, lat_w = latent.dim(1), latent.dim(2), latent.dim(3)
    if lat_h is None or lat_w is None:
        return estimate.as_unknown("latent spatial size is not statically known")
    scale, _ = ctx.assumption("vae_scale")
    out_channels = 3
    batch = latent.dim(0)
    value = TensorVal((batch, lat_h * scale, lat_w * scale, out_channels),
                      FLOAT32_BYTES)
    result = NodeEstimate(
        outputs=[value],
        items=[
            formulas.MemoryItem("decoded image (B, %dx, %dx, 3)" % (scale, scale),
                                value.nbytes() or 0, value.nbytes() or 0, kind="outputs"),
            formulas.MemoryItem("VAE decoder workspace (comfy/sd.py rule)",
                                _vae_memory_decode(lat_h, lat_w),
                                _vae_memory_decode(lat_h, lat_w),
                                kind="activations", approx=True),
        ],
        basis={"key": "vae_decode",
               "params": {"latent": [batch, channels, lat_h, lat_w], "scale": scale,
                          "out": list(value.shape)}},
    )
    result.confidence = "approx"
    result.reason = ("latent channels / upscale follow the SD AutoencoderKL "
                     "defaults recorded as assumptions")
    return result


@register("VAEEncode")
def _estimate_vae_encode(ctx: EstimationCtx) -> NodeEstimate:
    """Image -> latent: /8 downscale and the configured latent channels."""
    estimate = NodeEstimate(basis={"key": "vae_encode", "params": {}})
    image = ctx.tensor("pixels")
    if not isinstance(image, TensorVal) or len(image.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    h, w = image.dim(1), image.dim(2)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    scale, _ = ctx.assumption("vae_scale")
    channels, _ = ctx.assumption("latent_channels")
    batch = image.dim(0)
    value = TensorVal((batch, channels, h // scale, w // scale), FLOAT32_BYTES)
    result = NodeEstimate(
        outputs=[value],
        items=[
            formulas.MemoryItem("encoded latent (B, %d, /%d, /%d)" % (channels, scale, scale),
                                value.nbytes() or 0, value.nbytes() or 0, kind="outputs"),
            formulas.MemoryItem("VAE encoder workspace (comfy/sd.py rule)",
                                _vae_memory_encode(h, w),
                                _vae_memory_encode(h, w),
                                kind="activations", approx=True),
        ],
        basis={"key": "vae_encode",
               "params": {"image": list(image.shape), "scale": scale,
                          "channels": channels, "latent": list(value.shape)}},
    )
    result.confidence = "approx"
    result.reason = ("latent channels / downscale follow the SD AutoencoderKL "
                     "defaults recorded as assumptions")
    return result


@register("CLIPTextEncode")
def _estimate_clip_text_encode(ctx: EstimationCtx) -> NodeEstimate:
    """Text -> CONDITIONING: token x hidden, both from the assumption set.

    This build keeps no text encoder, so neither number can be measured;
    the panel surfaces both as assumptions.
    """
    text = ctx.str_widget("text", "")
    tokens, _ = ctx.assumption("clip_tokens")
    hidden, _ = ctx.assumption("clip_hidden")
    value = TensorVal((1, tokens, hidden), FLOAT32_BYTES)
    result = NodeEstimate(
        outputs=[value],
        items=[formulas.MemoryItem("conditioning (1, tokens, hidden)",
                                   value.nbytes() or 0, value.nbytes() or 0,
                                   kind="outputs", approx=True)],
        basis={"key": "clip_encode", "params": {"chars": len(str(text or "")),
                                                "tokens": tokens, "hidden": hidden}},
    )
    result.confidence = "approx"
    result.reason = "token count and hidden width are assumptions (no encoder in this build)"
    return result


@register("CLIPSetLastLayer")
def _estimate_clip_set_last_layer(ctx: EstimationCtx) -> NodeEstimate:
    """Truncates the encoder depth: the weights themselves are unchanged."""
    return NodeEstimate(
        outputs=[ctx.est("clip")],
        basis={"key": "pass_through", "params": {}},
    )


def _save_estimate(ctx: EstimationCtx, slots: tuple, kind: str) -> NodeEstimate:
    """The four *Save nodes: no output slots, one staged state dict on CPU."""
    total = 0
    known = False
    for slot in slots:
        value = ctx.est(slot)
        if isinstance(value, WeightsVal) and value.file_bytes >= 0:
            total += value.file_bytes
            known = True
    result = NodeEstimate(
        outputs=[],
        basis={"key": "weights_save", "params": {"kind": kind}},
    )
    if known:
        result.items = [
            formulas.MemoryItem("state dict staged for the write", total, total,
                                kind="misc", approx=True)
        ]
        result.confidence = "approx"
        result.reason = "the write stages one full copy of the weights on the CPU"
        result.basis["params"] = {"kind": kind, "bytes": total}
    return result


@register("CheckpointSave")
def _estimate_checkpoint_save(ctx: EstimationCtx) -> NodeEstimate:
    """model + clip + vae merged into one checkpoint file on disk."""
    return _save_estimate(ctx, ("model", "clip", "vae"), "checkpoint")


@register("ModelSave")
def _estimate_model_save(ctx: EstimationCtx) -> NodeEstimate:
    """The diffusion model's state dict, written to disk."""
    return _save_estimate(ctx, ("model",), "diffusion model")


@register("CLIPSave")
def _estimate_clip_save(ctx: EstimationCtx) -> NodeEstimate:
    """The text encoder's state dict, written to disk."""
    return _save_estimate(ctx, ("clip",), "text encoder")


@register("VAESave")
def _estimate_vae_save(ctx: EstimationCtx) -> NodeEstimate:
    """The VAE's state dict, written to disk."""
    return _save_estimate(ctx, ("vae",), "vae")


# -- the built-in image pipeline (nodes.py, the classic nodes)

@register("SaveImage")
def _estimate_save_image(ctx: EstimationCtx) -> NodeEstimate:
    """Image writer: the tensor passes through, one copy is staged per frame."""
    return _image_passthrough(ctx, "staged image copy (numpy, uint8)", "weights_save",
                              slot="images")


@register("PreviewImage")
def _estimate_preview_image(ctx: EstimationCtx) -> NodeEstimate:
    """Preview writer: same staging cost as SaveImage, into the temp folder."""
    return _image_passthrough(ctx, "staged preview copy (numpy, uint8)", "weights_save",
                              slot="images")


@register("ImageScale")
def _estimate_image_scale(ctx: EstimationCtx) -> NodeEstimate:
    """Classic ImageScale: (width, height) with 0 meaning "keep the ratio"."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    h, w = t.dim(1), t.dim(2)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    new_w = ctx.int_widget("width", 512) or 0
    new_h = ctx.int_widget("height", 512) or 0
    if new_w == 0 and new_h == 0:
        new_w, new_h = w, h
    elif new_w == 0:
        new_w = max(1, int(round(w * new_h / h)))
    elif new_h == 0:
        new_h = max(1, int(round(h * new_w / w)))
    return _scaled_image(ctx, "image", new_h, new_w)


@register("ImageScaleBy")
def _estimate_image_scale_by(ctx: EstimationCtx) -> NodeEstimate:
    """Classic ImageScaleBy: one multiplier, no cropping."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    t = ctx.tensor("image")
    if not isinstance(t, TensorVal) or len(t.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    h, w = t.dim(1), t.dim(2)
    if h is None or w is None:
        return estimate.as_unknown("image spatial size is not statically known")
    factor = ctx.float_widget("scale_by", 1.0)
    return _scaled_image(ctx, "image", int(round(h * factor)), int(round(w * factor)))


@register("ImageInvert")
def _estimate_image_invert(ctx: EstimationCtx) -> NodeEstimate:
    """1.0 - image: a fresh tensor of the same shape."""
    return _image_passthrough(ctx, "inverted image")


@register("ImageBatch")
def _estimate_image_batch(ctx: EstimationCtx) -> NodeEstimate:
    """torch.cat([image1, image2]): the batch dimension adds up."""
    estimate = NodeEstimate(basis={"key": "image_scale", "params": {}})
    first = ctx.tensor("image1")
    second = ctx.tensor("image2")
    if not isinstance(first, TensorVal) or not isinstance(second, TensorVal):
        return estimate.as_unknown("one of the images is not statically known")
    if len(first.shape) != 4 or len(second.shape) != 4:
        return estimate.as_unknown("image shape is not statically known")
    b1, b2 = first.dim(0), second.dim(0)
    if b1 is None or b2 is None:
        return estimate.as_unknown("batch size is not statically known")
    shape = (b1 + b2,) + tuple(first.shape[1:])
    value = TensorVal(shape, first.itemsize, first.dtype)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("batched images", value.nbytes(), kind="outputs")],
        basis={"key": "image_scale", "params": {"from": "%d+%d" % (b1, b2),
                                                "to": str(b1 + b2)}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 4: the remaining families (visualisation, device queries, misc,
# datasets). They allocate nothing that the two ledgers could mislead about,
# so they are reported as *estimated* instead of unknown - with the render
# canvas stated as an approximation rather than an invented tensor shape.
# --------------------------------------------------------------------------- #

#: matplotlib's own canvas defaults (rcParams figure.figsize / figure.dpi);
#: used when the node's widget is absent.
MPL_DEFAULT_FIGSIZE = (6.4, 4.8)
MPL_DEFAULT_DPI = 100


def _render_estimate(ctx: EstimationCtx, key: str, outputs: int = 1) -> NodeEstimate:
    """A matplotlib figure rendered into an IMAGE: (1, H, W, 4) float32.

    H and W come from figsize x dpi; RGBA is what the canvas produces and
    what ComfyUI converts to float32.
    """
    figsize = _parse_floats(ctx.str_widget("figsize", ""))
    if not figsize or len(figsize) < 2 or min(figsize) <= 0:
        figsize = list(MPL_DEFAULT_FIGSIZE)
    dpi = ctx.int_widget("dpi", MPL_DEFAULT_DPI) or MPL_DEFAULT_DPI
    h = int(round(figsize[1] * dpi))
    w = int(round(figsize[0] * dpi))
    value = TensorVal((1, h, w, 4), FLOAT32_BYTES, "float32")
    result = NodeEstimate(
        outputs=[value] * outputs,
        items=[formulas.tensor_item("rendered figure", value.nbytes(), kind="outputs",
                                    approx=True)],
        basis={"key": key, "params": {"figsize": list(figsize), "dpi": dpi,
                                      "canvas": [h, w]}},
    )
    result.confidence = "approx"
    result.reason = "matplotlib canvas (figsize x dpi), RGBA converted to float32"
    return result


@register("CdlShowImages")
def _estimate_cdl_show_images(ctx: EstimationCtx) -> NodeEstimate:
    """Image grid render: the canvas, plus the scaled copies of the inputs."""
    image = ctx.tensor("images")
    estimate = _render_estimate(ctx, "render")
    if isinstance(image, TensorVal) and image.nbytes():
        estimate.items.append(
            formulas.tensor_item("scaled input copies", image.nbytes(), kind="outputs",
                                 approx=True)
        )
    return estimate


@register("CdlShowHeatmaps")
def _estimate_cdl_show_heatmaps(ctx: EstimationCtx) -> NodeEstimate:
    """Heatmap render; the node downsamples to max_samples before plotting."""
    return _render_estimate(ctx, "render")


@register("CdlShowHeatmapsOutput")
def _estimate_cdl_show_heatmaps_output(ctx: EstimationCtx) -> NodeEstimate:
    """Heatmap render straight to the node output (no tensor slots)."""
    result = _render_estimate(ctx, "render", outputs=0)
    result.items = []
    return result


@register("CdlPlot")
def _estimate_cdl_plot(ctx: EstimationCtx) -> NodeEstimate:
    """Line/scatter plot render."""
    return _render_estimate(ctx, "render")


@register("CdlShowTrace2D")
def _estimate_cdl_show_trace2d(ctx: EstimationCtx) -> NodeEstimate:
    """Training-trace render."""
    return _render_estimate(ctx, "render")


@register("CdlShowBboxes")
def _estimate_cdl_show_bboxes(ctx: EstimationCtx) -> NodeEstimate:
    """Bounding-box overlay render."""
    return _render_estimate(ctx, "render")


@register("CdlHistogram")
def _estimate_cdl_histogram(ctx: EstimationCtx) -> NodeEstimate:
    """Histogram render (the tensor is aggregated into bins)."""
    return _render_estimate(ctx, "render")


@register("CdlBarChart")
def _estimate_cdl_bar_chart(ctx: EstimationCtx) -> NodeEstimate:
    """Bar-chart render."""
    return _render_estimate(ctx, "render")


@register("CdlScatter")
def _estimate_cdl_scatter(ctx: EstimationCtx) -> NodeEstimate:
    """Scatter render."""
    return _render_estimate(ctx, "render")


@register("CdlConfusionMatrix")
def _estimate_cdl_confusion_matrix(ctx: EstimationCtx) -> NodeEstimate:
    """Confusion-matrix render."""
    return _render_estimate(ctx, "render")


@register("CdlPieChart")
def _estimate_cdl_pie_chart(ctx: EstimationCtx) -> NodeEstimate:
    """Pie-chart render."""
    return _render_estimate(ctx, "render")


@register("CdlAreaChart")
def _estimate_cdl_area_chart(ctx: EstimationCtx) -> NodeEstimate:
    """Area-chart render."""
    return _render_estimate(ctx, "render")


@register("CdlHeatmapsTo3D")
def _estimate_cdl_heatmaps_to_3d(ctx: EstimationCtx) -> NodeEstimate:
    """Heatmaps exported as a 3D object file: disk artefact, no tensor out."""
    result = NodeEstimate(
        outputs=[Unknown("file path")],
        basis={"key": "pass_through", "params": {}},
    )
    result.flops_status = "zero"
    return result


@register("CdlMessageBox")
def _estimate_cdl_message_box(ctx: EstimationCtx) -> NodeEstimate:
    """Text box: no tensor, no compute."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlNoOp")
def _estimate_cdl_no_op(ctx: EstimationCtx) -> NodeEstimate:
    """Passthrough node: nothing of its own."""
    return NodeEstimate(
        outputs=[],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlTimer")
def _estimate_cdl_timer(ctx: EstimationCtx) -> NodeEstimate:
    """Timing harness: the numbers are measurements, not estimations."""
    result = NodeEstimate(
        outputs=[TextVal(None), Unknown("measured time")],
        basis={"key": "pass_through", "params": {}},
    )
    result.flops_status = "unknown"
    result.flops_reason = "a benchmark runs the operation num_iters times"
    return result


@register("CdlWhat")
def _estimate_cdl_what(ctx: EstimationCtx) -> NodeEstimate:
    """Easter egg: no outputs, no memory."""
    return NodeEstimate(
        outputs=[],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlDeviceInfo")
def _estimate_cdl_device_info(ctx: EstimationCtx) -> NodeEstimate:
    """Environment query: two integers, nothing allocated."""
    return NodeEstimate(
        outputs=[IntVal(None), IntVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlTryGpu")
def _estimate_cdl_try_gpu(ctx: EstimationCtx) -> NodeEstimate:
    """Device probe: text only."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlTryAllGpus")
def _estimate_cdl_try_all_gpus(ctx: EstimationCtx) -> NodeEstimate:
    """Device probe: text only."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("EmptyImage")
def _estimate_empty_image(ctx: EstimationCtx) -> NodeEstimate:
    """Solid-colour (batch, height, width, 3) tensor."""
    batch = max(1, ctx.int_widget("batch_size", 1) or 1)
    h = max(1, ctx.int_widget("height", 512) or 512)
    w = max(1, ctx.int_widget("width", 512) or 512)
    value = TensorVal((batch, h, w, 3), FLOAT32_BYTES)
    return NodeEstimate(
        outputs=[value],
        items=[formulas.tensor_item("empty image (B, H, W, 3)", value.nbytes(),
                                    kind="outputs")],
        basis={"key": "tensor_shape", "params": {"shape": [batch, h, w, 3]}},
    )


# --------------------------------------------------------------------------- #
# M3 batch 4 (continued): the dataset family.
#
# A dataloader is a lazy iterator: what the ledgers can honestly price is
# the materialised batch, never the corpus on disk. Sample counts of the
# downloadable datasets are not declared in the graph, so they stay unknown
# rather than being hard-coded from memory.
# --------------------------------------------------------------------------- #

@register("CdlLoadArray")
def _estimate_cdl_load_array(ctx: EstimationCtx) -> NodeEstimate:
    """TensorArray -> dataloader: one batch of the linked tensors at a time."""
    batch = max(1, ctx.int_widget("batch_size", 32) or 32)
    features = ctx.tensor("features")
    estimate = NodeEstimate(
        outputs=[IntVal(None)],
        basis={"key": "dataloader", "params": {"batch": batch}},
    )
    samples = features.dim(0) if isinstance(features, TensorVal) else None
    if isinstance(features, TensorVal):
        estimate.outputs = [IntVal(samples)]
    if isinstance(features, TensorVal) and samples is not None:
        example = features.numel() // samples if features.numel() else None
        if example:
            estimate.items = [
                formulas.tensor_item("materialised batch (x)", example * min(batch, samples) * 4,
                                     kind="dataset", approx=True)
            ]
        estimate.basis["params"] = {"batch": batch, "samples": samples,
                                    "batches": -(-samples // batch)}
    estimate.confidence = "approx"
    estimate.reason = "the dataloader keeps one batch alive, not the whole corpus"
    return estimate


@register("CdlDataLoaderInfo")
def _estimate_cdl_dataloader_info(ctx: EstimationCtx) -> NodeEstimate:
    """Loader metadata: sample count, batch size, number of batches."""
    loader = ctx.est("dataloader")
    batch = max(1, ctx.int_widget("batch_size", 32) or 32)
    samples = loader.value if isinstance(loader, IntVal) else None
    batches = -(-samples // batch) if samples else None
    return NodeEstimate(
        outputs=[IntVal(samples), IntVal(batch if samples else None), IntVal(batches)],
        basis={"key": "dataloader", "params": {"batch": batch, "samples": samples,
                                               "batches": batches}},
    )


@register("CdlDownload")
def _estimate_cdl_download(ctx: EstimationCtx) -> NodeEstimate:
    """URL download: network/disk, no tensor."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


@register("CdlDownloadExtract")
def _estimate_cdl_download_extract(ctx: EstimationCtx) -> NodeEstimate:
    """Download + unpack: network/disk, no tensor."""
    return NodeEstimate(
        outputs=[TextVal(None)],
        basis={"key": "pass_through", "params": {}},
    )


def _dataset_estimate(
    ctx: EstimationCtx,
    example_bytes: Optional[int],
    slots: int,
    note: str,
) -> NodeEstimate:
    """One downloadable dataset: priced per batch, never per corpus."""
    batch = max(1, ctx.int_widget("batch_size", 32) or 32)
    estimate = NodeEstimate(
        outputs=[IntVal(None)] * slots,
        basis={"key": "dataloader", "params": {"batch": batch}},
    )
    if example_bytes:
        estimate.items = [
            formulas.tensor_item("materialised batch (x + y)", example_bytes * batch,
                                 kind="dataset", approx=True)
        ]
        estimate.basis["params"] = {"batch": batch, "example_bytes": example_bytes}
    estimate.confidence = "approx"
    estimate.reason = note
    return estimate


@register("CdlFashionMNIST")
def _estimate_cdl_fashion_mnist(ctx: EstimationCtx) -> NodeEstimate:
    """Fashion-MNIST: (1, resize, resize) float32 per example."""
    size = max(1, ctx.int_widget("resize", 28) or 28)
    return _dataset_estimate(
        ctx, size * size * 4, 3,
        "dataset is downloaded at run time; only the materialised batch is priced",
    )


@register("CdlVOCSegmentation")
def _estimate_cdl_voc_segmentation(ctx: EstimationCtx) -> NodeEstimate:
    """VOC segmentation: a 3-channel crop plus its label map per example."""
    h = ctx.int_widget("crop_h", 480)
    w = ctx.int_widget("crop_w", 320)
    example = (3 * h * w * 4 + h * w * 4) if (h and w) else None
    return _dataset_estimate(
        ctx, example, 2,
        "dataset is downloaded at run time; only the materialised batch is priced",
    )


@register("CdlBananasDetection")
def _estimate_cdl_bananas_detection(ctx: EstimationCtx) -> NodeEstimate:
    """Banana detection: the image size is not declared, so no batch price."""
    return _dataset_estimate(
        ctx, None, 2,
        "the dataset's image size is not declared in the graph",
    )


@register("CdlDataLoaderPreview")
def _estimate_cdl_dataloader_preview(ctx: EstimationCtx) -> NodeEstimate:
    """Preview grid of a few batches: a render, priced as a canvas."""
    return _render_estimate(ctx, "render")


@register("CdlDataLoaderPreviewOutput")
def _estimate_cdl_dataloader_preview_output(ctx: EstimationCtx) -> NodeEstimate:
    """Preview grid shown on the node itself."""
    return _render_estimate(ctx, "render")


@register("CdlDataLoaderStats")
def _estimate_cdl_dataloader_stats(ctx: EstimationCtx) -> NodeEstimate:
    """Statistics text plus a render."""
    estimate = _render_estimate(ctx, "render")
    estimate.outputs = [TextVal(None)] + estimate.outputs
    return estimate
