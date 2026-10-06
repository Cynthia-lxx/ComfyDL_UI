"""opgraph - the operator-level equivalent computation graph probe (M3).

What: for each node of a prompt-format workflow, run its ``execute`` once with
REAL tensors under a ``TorchDispatchMode`` recorder plus
``torch.utils.flop_counter.FlopCounterMode``.  (P0 correction, 2026-10-06: an
earlier revision of this docstring claimed a ``FakeTensorMode`` wrapper - that
import existed but was never used, so every probe to date has executed real
code on real tensors.  The safety story is therefore the explicit guardrails
below plus the dangerous-mode gate in ``app/profiling_routes.py``, NOT fake
memory.)  The probe intercepts the ATen operators the node *actually executes*
and produces, per node:

  * the op census (op name -> call count, FLOPs),
  * the probed FLOPs total (bottom-up, per-op ``2MNK`` style counting),
  * a marker of how many data-dependent scalar reads were canned.

This is the structural counterpart of the M2 formula ledger: the ledger derives
FLOPs top-down from hand-written per-family formulas, the probe derives them
bottom-up from the operators that really run.  Two independent sources that
should agree - disagreement is a bug signal.

Honest boundaries
-----------------
* Granularity is **ATen**: a ``Regression Train`` decomposes down to
  ``aten.addmm`` / ``aten.mse_loss`` level, but ``conv2d`` / ``batch_norm`` are
  single ATen ops - what happens inside a fused kernel is not Python-visible.
* Data-dependent scalar reads (``float(tensor)`` inside training loops) are
  intercepted and answered with a canned ``0.5`` so control flow stays
  deterministic; the loop then runs for exactly the widget-declared step count
  and the census reflects that contract (the report carries
  ``data_dependent_reads`` so the canned reads are visible).
* Nodes that cannot be probed (side effects like file writes on fake tensors,
  unresolved inputs, host-only dependencies) fall back to the M2 **formula**
  FLOPs and are flagged ``fallback`` - the report is always complete, only the
  per-node fidelity varies.  A fallback also cascades: downstream nodes of an
  unprobed node resolve no inputs and fall back too.
* Ops are capped (``OPS_LIMIT``); a node whose probe would explode (huge step
  counts) falls back instead of stalling the analyze request.

Inputs are resolved in topological order and the probed (real) outputs feed
downstream probes.  Results are cached per workflow hash: repeated Analyze
clicks on an unchanged graph are instant, and a changed graph simply produces
a new key.

Safety guardrails (P0)
----------------------
* Per-node string inputs above ``MAX_NODE_INPUT_BYTES`` and graphs whose
  string inputs total above ``MAX_TOTAL_INPUT_BYTES`` never reach
  ``execute`` - the node (or the whole analysis) falls back with a readable
  error instead of pinning the CPU on a megabyte of tokenization.
* ``probe_workflow`` honours a cooperative ``deadline``: once it passes, the
  remaining nodes are marked ``probe deadline exceeded`` without execution.
  The caller (the route) additionally wraps the run in a hard timeout.
* Online probing is opt-in ("dangerous mode"); the route refuses to execute
  anything without an explicit acknowledgement header.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

# Four-level verbosity (off/low/medium/high) shared with the profiling routes;
# see comfy/profiling/proflog.py (stdlib-only, safe to import here).
from comfy.profiling.proflog import log as proflog

logger = logging.getLogger(__name__)

# Version of *this* report.  The M1/M2 memory + formula report keeps its own
# REPORT_VERSION (=2) in engine.py; the two documents are independent.
OPGRAPH_REPORT_VERSION = 1

# --- P0 safety guardrails (2026-10-06) --------------------------------------
# The probe below executes REAL node code with REAL tensors (the docstring's
# old "FakeTensorMode" claim was wrong - that import was never used), so an
# unbounded input wedges the whole profiling backend: 1MB of corpus text pins
# CPU/RAM, starves the shared event loop (estimate dies with it) and Ctrl+C
# cannot stop the worker thread. These caps keep an Analyze click cheap; the
# values are tunable and sized so every shipped example workflow passes with
# room to spare. The online probe additionally requires the explicit
# "dangerous mode" acknowledgement (see app/profiling_routes.py); offline
# sampling (Profiling v2 P1) goes through the same guards deliberately.
MAX_NODE_INPUT_BYTES = 64 * 1024          # per-node string input cap
MAX_TOTAL_INPUT_BYTES = 2 * 1024 * 1024   # whole-graph string input cap
PROBE_DEADLINE_SECONDS = 60.0             # cooperative whole-analysis deadline

# A node probe recording more ATen calls than this falls back to the formula
# ledger (huge step counts would otherwise spin the dispatcher for seconds).
OPS_LIMIT = 250_000

# The canned answer for data-dependent scalar reads (``float(tensor)``).  The
# value is arbitrary; what matters is that control flow becomes deterministic
# so the loop body runs exactly ``steps`` times.
CANNED_SCALAR = 0.5

# The data-dependent ATen op the fake-tensor machinery raises on: reading a
# python scalar from a fake tensor.
_DATA_DEP_OP = "aten._local_scalar_dense.default"

_CACHE: Dict[str, dict] = {}
_CACHE_LIMIT = 8


class _ProbeTooLarge(Exception):
    """Raised by the recorder when a node exceeds :data:`OPS_LIMIT`."""


class _ProbeRecorder(TorchDispatchMode):
    """Record the ATen op census of one node probe.

    Doubles as the data-dependent-scalar shim: ``aten._local_scalar_dense``
    (a ``float(tensor)`` read) is answered with :data:`CANNED_SCALAR` instead
    of raising ``DataDependentOutputException``.  The recorder must be the
    *outermost* mode so it sees the data-dependent read before the fake-tensor
    machinery does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ops: List[str] = []
        self.data_dependent_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name == _DATA_DEP_OP:
            self.data_dependent_reads += 1
            return CANNED_SCALAR
        if len(self.ops) >= OPS_LIMIT:
            raise _ProbeTooLarge()
        self.ops.append(name)
        return func(*args, **(kwargs or {}))


def _node_registry() -> Dict[str, Any]:
    """The host registry (ComfyDL merged + core + extras), resolved lazily."""
    import nodes as host_nodes  # registry exists at request time

    return host_nodes.NODE_CLASS_MAPPINGS


def _bootstrap_registry() -> None:
    """Populate the host registry outside a running server.

    The host ``nodes`` module ships with a handful of module-level utility
    nodes, so a non-empty mapping does NOT mean the ComfyDL/extras modules
    were merged.  ``init_extra_nodes`` (an async coroutine) does the merge;
    inside the server it has already run and this is a no-op.  The smoke
    harness bootstrap uses the same reduced form (no custom nodes, no API
    nodes).
    """
    import asyncio

    import nodes as host_nodes

    if not host_nodes.NODE_CLASS_MAPPINGS:
        return
    registry = host_nodes.NODE_CLASS_MAPPINGS
    if any(key.startswith(("Cdl", "LanguageModel")) for key in registry):
        return  # comfydl modules already merged
    asyncio.run(
        host_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    )


def _resolve_inputs(node, outputs):
    """Map a prompt node's inputs onto probed fake outputs / constants."""
    resolved: Dict[str, Any] = {}
    for name, value in (node.get("inputs") or {}).items():
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], (str, int))
            and not isinstance(value[0], bool)
        ):
            upstream_id, slot = value[0], value[1]
            upstream = outputs.get(upstream_id)
            if upstream is None:
                return None, f"upstream {upstream_id!r} was not probed"
            try:
                resolved[name] = upstream[int(slot)]
            except (IndexError, TypeError):
                return None, f"upstream {upstream_id!r} has no output slot {slot!r}"
        else:
            resolved[name] = value
    return resolved, None


def _unwrap_outputs(result):
    """Per-slot output values, exactly as the host would deliver them.

    V3 nodes (``io.ComfyNode``) return a single ``io.NodeOutput`` wrapper; the
    host unwraps ``.args`` (plus ``.expand`` overrides) before feeding the
    values to downstream nodes.  Without the same unwrapping the probe hands
    downstream nodes the wrapper object itself, and their type checks fail
    with things like "this slot needs a Vocab ... got NodeOutput" (2026-10-06,
    the Language Model template probed 2/12 for exactly this reason).  V1
    nodes return plain tuples and pass through untouched.
    """
    if (
        result.__class__.__name__ == "NodeOutput"
        and hasattr(result, "args")
        and not isinstance(result, tuple)
    ):
        outputs = list(result.args)
        expand = getattr(result, "expand", None)
        if isinstance(expand, dict):
            for key, value in expand.items():
                try:
                    outputs[int(key)] = value
                except (KeyError, ValueError, IndexError):
                    continue
        return outputs
    return list(result) if isinstance(result, tuple) else [result]


def _probe_one(class_type, cls, inputs):
    """Probe a single node.  Returns ``(record, outputs)``; never raises."""
    record: Dict[str, Any] = {
        "class_type": class_type,
        "status": "probed",
        "ops": [],
        "ops_total": 0,
        "total_flops": None,
        "data_dependent_reads": 0,
        "error": None,
        "probe_ms": 0,
    }
    outputs: List[Any] = []
    started = time.perf_counter()
    try:
        instance = cls()
        execute = getattr(instance, cls.FUNCTION)
        flops_mode = FlopCounterMode(display=False)
        recorder = _ProbeRecorder()
        # The recorder must be the outermost mode: it answers data-dependent
        # scalar reads before the fake-tensor machinery raises on them, and
        # every other op flows down into the flop counter.
        with flops_mode, recorder:
            result = execute(**inputs)
        try:
            counts = flops_mode.get_flop_counts()
        except Exception:
            counts = {}
        # get_flop_counts() -> {module_name: {op_name_or_overload: flops}}.
        flops_by_op: Dict[str, int] = {}
        for _module, op_counts in counts.items():
            for op, value in op_counts.items():
                name = str(op)
                flops_by_op[name] = flops_by_op.get(name, 0) + int(value)

        census = Counter(recorder.ops)
        ops = [
            {"op": op, "count": count, "flops": flops_by_op.get(op, 0)}
            for op, count in census.items()
        ]
        ops.sort(key=lambda row: (-row["flops"], -row["count"], row["op"]))
        record["ops"] = ops
        record["ops_total"] = len(recorder.ops)
        record["total_flops"] = int(flops_mode.get_total_flops())
        record["data_dependent_reads"] = recorder.data_dependent_reads
        outputs = _unwrap_outputs(result)
    except Exception as exc:
        record["status"] = "fallback"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["probe_ms"] = int((time.perf_counter() - started) * 1000)
    return record, outputs

def _formula_flops(formula_report):
    """Extract the per-node formula FLOPs from an M2 estimate report."""
    per_node: Dict[Any, int] = {}
    total = 0
    if not isinstance(formula_report, dict):
        return per_node, total
    for record in formula_report.get("nodes") or []:
        flops = record.get("flops_total")
        if isinstance(flops, (int, float)) and flops > 0:
            per_node[record.get("id")] = int(flops)
            total += int(flops)
    return per_node, total


def _string_input_bytes(node) -> int:
    """Total UTF-8 size of a prompt node's string widget inputs.

    Link inputs arrive as ``[upstream_id, slot]`` lists and are ignored here;
    only literal strings count - they are the vector the 1MB-corpus incident
    rode in on (tokenizers and text nodes loop over every byte for real).
    """
    total = 0
    for value in (node.get("inputs") or {}).values():
        if isinstance(value, str):
            total += len(value.encode("utf-8", errors="replace"))
    return total


def probe_workflow(prompt, formula_report=None, deadline: Optional[float] = None):
    """Probe every node of ``prompt`` and return the opgraph report.

    ``formula_report`` (optional) is the M2 ``engine.estimate_workflow``
    report; its per-node FLOPs back the ``fallback`` records and the
    cross-validation totals.  ``deadline`` (optional, ``time.monotonic()``
    based) is a cooperative cap: once it passes, remaining nodes are marked
    ``probe deadline exceeded`` without being executed.

    P0 guardrails: a node whose string inputs exceed ``MAX_NODE_INPUT_BYTES``
    and a graph whose total exceeds ``MAX_TOTAL_INPUT_BYTES`` never reach
    ``execute`` - the guard fires before the probe, whatever calls this.
    """
    registry = _node_registry()
    # A fresh interpreter (unit tests, standalone probe) has only the host's
    # module-level utility nodes; bootstrap the full registry once, lazily.
    _bootstrap_registry()
    registry = _node_registry()
    formula_per_node, formula_total = _formula_flops(formula_report)

    report: Dict[str, Any] = {
        "version": OPGRAPH_REPORT_VERSION,
        "totals": {
            "nodes": 0,
            "probed": 0,
            "fallback": 0,
            "flops_formula": formula_total,
            "flops_probed": 0,
            "ratio": None,
        },
        "nodes": [],
        "error": None,
        "disclaimer_key": "opgraphDisclaimer",
    }
    if not isinstance(prompt, dict) or not prompt:
        report["error"] = "no_graph"
        return report

    total_bytes = sum(_string_input_bytes(n) for n in prompt.values())
    if total_bytes > MAX_TOTAL_INPUT_BYTES:
        report["error"] = (
            "graph string inputs too large for probe "
            f"({total_bytes // 1024} KB > {MAX_TOTAL_INPUT_BYTES // 1024} KB limit)"
        )
        return report

    from comfy.profiling.engine import _topological_order

    order, cyclic = _topological_order(prompt)
    outputs: Dict[Any, List[Any]] = {}
    probed_flops = 0
    probed_nodes = 0
    fallback_nodes = 0

    for node_id in list(order) + list(cyclic):
        node = prompt.get(node_id) or {}
        class_type = str(node.get("class_type") or "")
        formula_flops = formula_per_node.get(node_id)
        entry: Dict[str, Any] = {
            "id": node_id,
            "class_type": class_type,
            "status": "fallback",
            "ops": [],
            "ops_total": 0,
            "total_flops": formula_flops,
            "formula_flops": formula_flops,
            "data_dependent_reads": 0,
            "error": None,
            "probe_ms": 0,
        }
        cls = registry.get(class_type)
        resolved, resolve_error = _resolve_inputs(node, outputs)
        node_bytes = _string_input_bytes(node)
        if node_bytes > MAX_NODE_INPUT_BYTES:
            entry["error"] = (
                "input too large for probe "
                f"({node_bytes // 1024} KB > {MAX_NODE_INPUT_BYTES // 1024} KB limit)"
            )
        elif deadline is not None and time.monotonic() > deadline:
            entry["error"] = "probe deadline exceeded"
        elif cls is None:
            entry["error"] = "node type not registered"
        elif resolved is None:
            entry["error"] = resolve_error
        else:
            probe, node_outputs = _probe_one(class_type, cls, resolved)
            entry.update(probe)
            if probe["status"] == "probed":
                outputs[node_id] = node_outputs
                probed_nodes += 1
                if entry["total_flops"]:
                    probed_flops += entry["total_flops"]
        if entry["status"] != "probed":
            fallback_nodes += 1
        report["nodes"].append(entry)

    totals = report["totals"]
    totals["nodes"] = len(order) + len(cyclic)
    totals["probed"] = probed_nodes
    totals["fallback"] = fallback_nodes
    totals["flops_probed"] = probed_flops
    if formula_total > 0 and probed_flops > 0:
        totals["ratio"] = round(probed_flops / formula_total, 4)
    return report


def analyse_workflow(prompt, formula_report=None, deadline: Optional[float] = None):
    """Cached entry point for the ``/opgraph`` endpoint.

    The cache key is a canonical hash of the prompt (widget values included),
    so repeated Analyze clicks on an unchanged graph are instant while any
    edit produces a fresh probe.  The formula report (M2) is computed here
    when not supplied, so callers only need the prompt.  ``deadline`` is
    forwarded to :func:`probe_workflow`; reports that ended in an error
    (guardrail, deadline) are NOT cached so a retry with saner inputs or a
    raised dangerous-mode flag probes fresh.
    """
    try:
        key = hashlib.sha256(
            json.dumps(prompt, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        key = None
    proflog("high", "opgraph: cache key %s", key[:12] if key else None)
    if key is not None and key in _CACHE:
        logger.debug("opgraph cache hit (%s)", key[:12])
        proflog("medium", "opgraph: cache hit (%s) - skipping probe", key[:12])
        return _CACHE[key]

    if formula_report is None:
        from comfy.profiling.engine import estimate_workflow

        try:
            formula_report = estimate_workflow(prompt if isinstance(prompt, dict) else None)
        except Exception as exc:
            logger.warning("opgraph formula report failed: %s", exc)
            formula_report = None

    report = probe_workflow(
        prompt if isinstance(prompt, dict) else {}, formula_report, deadline=deadline)
    if key is not None and report.get("error") is None:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = report
    return report
