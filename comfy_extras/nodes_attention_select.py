"""Attention backend selection nodes (dual-track M-B, SP-2).

Two MODEL->MODEL nodes, the project's first "dual-form" pair on the core
MODEL protocol layer:

* ``CdlAttentionOptimizer`` (production): benchmarks every attention backend
  this machine can actually run (import check *and* a real mini forward -
  the "import success != runs" lesson from ComfyUI-Attention-Optimizer),
  picks the fastest one whose speedup beats ``min_speedup``, and attaches it
  through the official ``ModelPatcher.set_model_optimized_attention`` hook.
  Results are cached in the signature cache (``attention_bench`` namespace)
  keyed on device/torch/shape - a repeat run pays nothing (Tier-0: the
  entire cost sits in prompt-setup, never per-step).
* ``CdlAttentionBackend`` (educational): a manual backend dropdown plus an
  explanation string - the teaching facade of the same mechanism.

Implementation is a port of ComfyUI-Attention-Optimizer (probe / validate /
benchmark / report) adapted to ComfyDL: the backend registry is the core's
own ``REGISTERED_ATTENTION_FUNCTIONS`` (only genuinely available backends
are registered there), the benchmark cache lives in the user directory via
``comfy.profiling.signature_cache`` (survives plugin updates), and both
nodes self-meter into the ``comfydl.attention_select`` switch (meta-
profiling rule).  The injection point is ``wrap_attn``'s
``optimized_attention_override`` - the pattern five surveyed plugins
converged on; ``set_model_optimized_attention`` wraps it for us.

Never patches any global: a clone of the incoming ModelPatcher carries the
override, so unrelated branches keep their backend.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from comfy.ldm.modules.attention import (
    REGISTERED_ATTENTION_FUNCTIONS,
    get_attention_function,
)
from comfy.profiling import signature_cache, switch_registry

CATEGORY = "model/perf"

#: Registry id of this feature (Tier-1, self-metered).
SWITCH_ID = "comfydl.attention_select"

switch_registry.register_switch(
    SWITCH_ID,
    default=True,
    hot_toggle=True,
    self_meter=True,
)

#: Reference backend every speedup is measured against (portable: SDPA runs
#: everywhere PyTorch runs, CPU included).
REFERENCE_BACKEND = "pytorch"

#: One-line mechanism notes for the educational facade (and the report).
_BACKEND_NOTES = {
    "pytorch": "SDPA - PyTorch's native fused kernels; the portable baseline.",
    "flash": "FlashAttention - IO-aware tiling, wins when the sequence is long.",
    "sage": "SageAttention - INT8-quantized QK^T; fastest on CUDA, fp16 only.",
    "sage3": "SageAttention 3 (Blackwell-native).",
    "xformers": "xformers memory-efficient attention.",
    "sub_quad": "sub-quadratic chunked attention (ComfyUI built-in).",
    "split": "chunked attention along the sequence (ComfyUI built-in).",
    "basic": "naive implementation - the teaching baseline, rarely fastest.",
    "comfy_kitchen_int8": "comfy-kitchen INT8 attention.",
}

#: Benchmark geometry constants (kept off the widget list on purpose: they
#: shape the cache key; seq/heads/dim are the user-facing knobs).
_BENCH_BATCH = 4
_BENCH_WARMUP = 3
_BENCH_ITERS = 10
_VALIDATION_SEQ = 128

_CACHE_NAMESPACE = "attention_bench"
_CACHE_SCHEMA_VERSION = 1

#: Per-process validation memo: ``name -> (ok, error)``, keyed by device.
_VALIDATED: Dict[str, Dict[str, Tuple[bool, str]]] = {}


# --------------------------------------------------------------------------- #
# Probing & validation (A4's two-stage check: import success is not enough)
# --------------------------------------------------------------------------- #

def registered_backends() -> List[str]:
    """Backend names the core registered - i.e. imports succeeded."""
    return list(REGISTERED_ATTENTION_FUNCTIONS)


def _probe_tensors(device: torch.device, dtype: torch.dtype, heads: int,
                   head_dim: int, seq_len: int):
    shape = (1, seq_len, heads * head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def validate_backends(device: torch.device, heads: int = 2,
                      head_dim: int = 64) -> Dict[str, Tuple[bool, str]]:
    """Run one tiny real forward per registered backend on ``device``.

    Memoized per device type (validation is about capability, not speed).
    Returns ``{name: (ok, error_string)}`` - ``ok=False`` entries must never
    be benchmarked or applied (A4: a backend that imports but crashes must
    be excluded, not merely demoted).
    """
    memo_key = device.type
    memo = _VALIDATED.get(memo_key)
    if memo is not None:
        return memo
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    result: Dict[str, Tuple[bool, str]] = {}
    for name in registered_backends():
        func = REGISTERED_ATTENTION_FUNCTIONS[name]
        try:
            q, k, v = _probe_tensors(device, dtype, heads, head_dim,
                                     _VALIDATION_SEQ)
            with torch.no_grad():
                func(q, k, v, heads, mask=None)
            result[name] = (True, "")
        except Exception as exc:  # noqa: BLE001 - capability probe
            result[name] = (False, f"{type(exc).__name__}: {exc}")
    _VALIDATED[memo_key] = result
    return result


def usable_backends(device: torch.device) -> List[str]:
    """Registered backends that also survive the real mini forward."""
    return [name for name, (ok, _err) in validate_backends(device).items() if ok]


# --------------------------------------------------------------------------- #
# Benchmark engine (A4's timer skeleton, made injectable for tests)
# --------------------------------------------------------------------------- #

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    candidates: Dict[str, Callable],
    *,
    seq_len: int,
    heads: int,
    head_dim: int,
    batch: int = _BENCH_BATCH,
    warmup: int = _BENCH_WARMUP,
    iters: int = _BENCH_ITERS,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    timer: Callable[[], float] = time.perf_counter,
) -> Dict[str, float]:
    """Median seconds per forward, per backend, on identical tensors.

    The A4 skeleton: identical synthetic tensors for every candidate,
    ``warmup`` un-timed calls (JIT/cudnn autotune/allocator warm-up), then
    ``iters`` timed calls with a device sync on each side of the timing
    window (CUDA is asynchronous - without the sync only kernel *launch*
    would be measured).  ``timer`` is injectable so tests can drive the
    engine deterministically without a GPU.
    """
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    q, k, v = _probe_tensors(device, dtype, heads, head_dim, seq_len)
    q, k, v = q[:batch], k[:batch], v[:batch]
    times: Dict[str, List[float]] = {}
    for name, func in candidates.items():
        with torch.no_grad():
            for _ in range(warmup):
                func(q, k, v, heads, mask=None)
        _sync(device)
        samples: List[float] = []
        for _ in range(iters):
            t0 = timer()
            func(q, k, v, heads, mask=None)
            _sync(device)
            samples.append(timer() - t0)
        samples.sort()
        times[name] = samples[len(samples) // 2]
    return times


def choose_backend(times: Dict[str, float],
                   min_speedup: float = 1.05) -> Tuple[str, Dict[str, float]]:
    """Pick the fastest backend that beats the reference by ``min_speedup``.

    Falls back to the reference backend itself when nothing clears the bar
    (switching backends that are not measurably faster only adds risk).
    """
    ref = times.get(REFERENCE_BACKEND)
    if ref is None:  # no reference available: pick the fastest of what exists
        best = min(times, key=times.get)
        return best, {name: ref_speed(ref_, times[best])
                      for name, ref_ in times.items() if ref_ > 0}
    speedups = {name: ref / t for name, t in times.items() if t > 0}
    best, best_speed = REFERENCE_BACKEND, 1.0
    for name, speed in speedups.items():
        if speed > best_speed and speed >= min_speedup:
            best, best_speed = name, speed
    return best, speedups


def ref_speed(t: float, best: float) -> float:
    return best / t if t > 0 else 0.0


# --------------------------------------------------------------------------- #
# Cache integration (SP-1 signature cache; code version auto-invalidates)
# --------------------------------------------------------------------------- #

def _bench_key_payload(device: torch.device, seq_len: int, heads: int,
                       head_dim: int) -> Dict[str, object]:
    return {
        "device": str(device),
        "torch": torch.__version__,
        "seq_len": seq_len,
        "heads": heads,
        "head_dim": head_dim,
        "batch": _BENCH_BATCH,
        "backends": sorted(registered_backends()),
    }


def _cache() -> signature_cache.JsonNamespaceCache:
    return signature_cache.JsonNamespaceCache(
        _CACHE_NAMESPACE, schema_version=_CACHE_SCHEMA_VERSION)


def load_cached_result(device: torch.device, seq_len: int, heads: int,
                       head_dim: int) -> Optional[Dict[str, object]]:
    key = signature_cache.digest(_CACHE_NAMESPACE,
                                 _bench_key_payload(device, seq_len, heads,
                                                    head_dim))
    return _cache().get(key)


def store_result(device: torch.device, seq_len: int, heads: int,
                 head_dim: int, result: Dict[str, object]) -> None:
    key = signature_cache.digest(_CACHE_NAMESPACE,
                                 _bench_key_payload(device, seq_len, heads,
                                                    head_dim))
    _cache().put(key, result)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

def render_report(device: torch.device, seq_len: int, heads: int,
                  head_dim: int, times: Dict[str, float], speedups: Dict[str, float],
                  chosen: str, cache_state: str) -> str:
    lines = [
        f"[attention-select] device={device.type} seq={seq_len} "
        f"heads={heads} dim={head_dim} (cache={cache_state})",
    ]
    for name, seconds in sorted(times.items(), key=lambda kv: kv[1]):
        ms = seconds * 1000.0
        speed = speedups.get(name, 0.0)
        mark = " <-- chosen" if name == chosen else ""
        if name == REFERENCE_BACKEND:
            lines.append(f"  {name:<22} {ms:9.3f} ms   (reference){mark}")
        else:
            lines.append(f"  {name:<22} {ms:9.3f} ms   x{speed:.2f}{mark}")
    note = _BACKEND_NOTES.get(chosen, "")
    if note:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Nodes (V3, MODEL protocol layer - see nodes_model_loaders for the idiom)
# --------------------------------------------------------------------------- #

def _require_model(model) -> None:
    if model is None:
        raise RuntimeError(
            "CdlAttention* nodes need a real MODEL value - wire a checkpoint "
            "or diffusion-model loader in front of this node.")


class CdlAttentionOptimizer(io.ComfyNode):
    """Benchmark every runnable attention backend and attach the fastest.

    What: clones the incoming MODEL, runs a one-shot benchmark of all
          registered attention backends that survive a real mini forward
          (synthetic tensors at the configured geometry, 3 warmup + 10 timed
          iterations, CUDA-synced), picks the fastest whose speedup over the
          SDPA reference clears ``min_speedup``, and attaches it via
          ``set_model_optimized_attention``.  The result is cached in the
          user-directory signature cache keyed on device/torch version and
          geometry - a repeated run is a cache hit and costs nothing; bump
          ``force_refresh`` to re-measure (e.g. after a driver change).
          The whole cost sits in prompt setup (Tier-0): nothing runs per
          step.  Benchmark wall time is booked into the
          ``comfydl.attention_select`` switch (meta-profiling).
    In:   model (MODEL) - the patcher to attach the chosen backend to.
          seq_len (INT, default 1024) - attention sequence length to bench.
          num_heads (INT, default 16) / head_dim (INT, default 64) - bench
          geometry; match your model (SD1.5: 8/40, SDXL: 20/64, Flux: 24/128).
          min_speedup (FLOAT, default 1.05) - do not switch unless this much
          faster than the SDPA reference.
          force_refresh (BOOLEAN) - ignore the cache and re-benchmark.
    Out:  MODEL - the clone with the chosen backend attached.
          STRING - human-readable timing table (also the teaching artifact).
    Caveats: benchmark uses an unmasked full-attention shape; backends whose
          advantage depends on masking or very long sequences may rank
          differently on real workloads.  On CPU only the portable backends
          are measurable (flash/sage validate out).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlAttentionOptimizer",
            display_name="Attention Optimizer (Auto)",
            category=CATEGORY,
            description=(
                "Benchmark all runnable attention backends once, attach the "
                "fastest to this MODEL branch; results cached until the code "
                "version or geometry changes."),
            inputs=[
                io.Model.Input("model", tooltip="MODEL to optimize."),
                io.Int.Input("seq_len", default=1024, min=16, max=131072,
                             tooltip="Attention sequence length to benchmark."),
                io.Int.Input("num_heads", default=16, min=1, max=128,
                             tooltip="Number of attention heads to benchmark."),
                io.Int.Input("head_dim", default=64, min=8, max=256,
                             tooltip="Per-head dimension to benchmark."),
                io.Float.Input("min_speedup", default=1.05, min=1.0, max=10.0,
                               step=0.01,
                               tooltip="Do not switch unless this much faster "
                                       "than the pytorch/SDPA reference."),
                io.Boolean.Input("force_refresh", default=False,
                                 tooltip="Re-benchmark even if a cached result "
                                         "exists."),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.String.Output(display_name="report"),
            ],
            search_aliases=[
                "attention", "attention optimizer", "flash attention",
                "sage attention", "benchmark", "speedup", "backend",
            ],
        )

    @classmethod
    def execute(cls, model, seq_len: int = 1024, num_heads: int = 16,
                head_dim: int = 64, min_speedup: float = 1.05,
                force_refresh: bool = False) -> io.NodeOutput:
        _require_model(model)
        patcher = model.clone()
        if not switch_registry.is_enabled(SWITCH_ID):
            return io.NodeOutput(
                patcher,
                "[attention-select] feature disabled via the switch - "
                "MODEL passed through unchanged.")

        device = torch.device("cpu") if not torch.cuda.is_available() \
            else torch.device("cuda")
        usable = usable_backends(device)
        if REFERENCE_BACKEND not in usable:
            raise RuntimeError(
                "attention-select: the pytorch/SDPA reference backend did not "
                f"validate on {device.type}; cannot benchmark.")

        cache_state = "hit"
        entry = None if force_refresh else load_cached_result(
            device, seq_len, num_heads, head_dim)
        if entry is None or not entry.get("times_ms"):
            cache_state = "miss"
            t0 = time.perf_counter()
            candidates = {name: REGISTERED_ATTENTION_FUNCTIONS[name]
                          for name in usable}
            times = run_benchmark(
                candidates, seq_len=seq_len, heads=num_heads,
                head_dim=head_dim, device=device)
            bench_s = time.perf_counter() - t0
            switch_registry.record_self_overhead(SWITCH_ID, bench_s)
            chosen, speedups = choose_backend(times, min_speedup)
            entry = {
                "times_ms": {k: round(v * 1000.0, 4) for k, v in times.items()},
                "speedup": {k: round(v, 4) for k, v in speedups.items()},
                "chosen": chosen,
                "benchmark_s": round(bench_s, 3),
            }
            store_result(device, seq_len, num_heads, head_dim, entry)
        else:
            times = {k: v / 1000.0 for k, v in entry["times_ms"].items()}
            chosen = entry["chosen"]
            speedups = entry.get("speedup", {})

        if chosen not in usable:
            # Cached choice no longer validates (driver/env changed): fall
            # back instead of attaching a crashing backend.
            chosen = REFERENCE_BACKEND
        patcher.set_model_optimized_attention(get_attention_function(chosen))
        report = render_report(device, seq_len, num_heads, head_dim,
                               times, speedups, chosen, cache_state)
        print(report)
        return io.NodeOutput(patcher, report)


class CdlAttentionBackend(io.ComfyNode):
    """Attach one specific attention backend - the manual, teaching form.

    What: the same mechanism as the optimizer node, minus the benchmark: you
          pick the backend, the node explains what it does.  This is the
          "textbook" facade of the dual-form pair - use it to learn what each
          backend is, to A/B a hunch, or to pin a backend you trust.
    In:   model (MODEL) - the patcher to attach the backend to.
          backend (COMBO) - one of the backends that validated on this
          machine (the list is probed with a real mini forward, so a backend
          that imports but crashes never appears).
    Out:  MODEL - the clone with the chosen backend attached.
          STRING - a per-backend explanation (mechanism + when it wins).
    Background: attention cost is dominated by streaming the Q/K/V tensors
          through the memory hierarchy.  FlashAttention reorders the
          computation into SRAM-sized tiles so the big activations never
          round-trip through HBM (IO-aware tiling); SageAttention goes
          further and quantizes QK^T to INT8; the chunked backends
          (sub_quad/split) trade peak speed for lower peak memory.  Which
          one wins depends on geometry - that is exactly what the Auto node
          measures.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        available = [
            name for name, (ok, _err)
            in validate_backends(_default_device()).items() if ok
        ]
        if "pytorch" not in available:
            available.insert(0, "pytorch")
        return io.Schema(
            node_id="CdlAttentionBackend",
            display_name="Attention Backend (Manual)",
            category=CATEGORY,
            description=(
                "Attach one specific attention backend to this MODEL branch; "
                "includes a per-backend mechanism explanation."),
            inputs=[
                io.Model.Input("model", tooltip="MODEL to attach the backend to."),
                io.Combo.Input(
                    "backend", options=available, default="pytorch",
                    tooltip="Backends that validated on this machine."),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.String.Output(display_name="explanation"),
            ],
            search_aliases=[
                "attention", "backend", "flash", "sage", "sdpa", "xformers",
                "teaching", "manual",
            ],
        )

    @classmethod
    def execute(cls, model, backend: str = "pytorch") -> io.NodeOutput:
        _require_model(model)
        patcher = model.clone()
        ok, err = validate_backends(_default_device()).get(
            backend, (False, "not registered"))
        if not ok:
            raise RuntimeError(
                f"attention backend {backend!r} did not validate on this "
                f"machine: {err}")
        patcher.set_model_optimized_attention(get_attention_function(backend))
        note = _BACKEND_NOTES.get(backend, "no note for this backend.")
        others = "\n".join(
            f"  {name}: {text}"
            for name, text in _BACKEND_NOTES.items() if name != backend)
        explanation = (
            f"attached backend: {backend}\n{note}\n\nother backends:\n{others}")
        return io.NodeOutput(patcher, explanation)


def _default_device() -> torch.device:
    return torch.device("cpu") if not torch.cuda.is_available() \
        else torch.device("cuda")


class AttentionSelectExtension(ComfyExtension):
    """Registers the ``model/perf`` attention-selection node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlAttentionOptimizer, CdlAttentionBackend]


async def comfy_entrypoint() -> AttentionSelectExtension:
    return AttentionSelectExtension()
