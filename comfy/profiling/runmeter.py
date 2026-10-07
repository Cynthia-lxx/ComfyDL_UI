"""Run-time FLOPs meter: measured per-node FLOPs as a by-product of a run (Profiling v2 P2).

While the user executes a workflow for real, one process-level
:class:`RunMeter` (a :class:`torch.utils.flop_counter.FlopCounterMode`
subclass) tallies every dispatched ATen operator and attributes it to the
currently executing node via the host's ``CurrentNodeContext`` contextvar
(``comfy_execution.utils.get_executing_context``).  The totals are persisted
after the prompt finishes (see ``app/database`` + the executor hook) and serve
two purposes: they *calibrate* the M1/M2 formula estimates (measured vs
formula ratio) and they give the user a run history.

Design contracts
----------------
* **Attribution by contextvar, never by mode lifetime.**  ``__torch_dispatch__``
  reads ``get_executing_context()`` on every call, so interleaved async node
  tasks charge their own nodes (contextvars are copied into tasks).  Operators
  observed outside any node context land in the ``unattributed`` bucket.
* **Two verbosity levels** (CLI ``--cdl-profiling-record``, default ``count``):
  ``count`` records per-node ``{flops, op_count}`` only; ``census`` also keeps
  the per-operator name histogram.  A node exceeding ``ops_limit`` dispatch
  calls is *demoted*: it keeps counting aggregate FLOPs but its census stops
  (huge training loops would otherwise balloon memory).
* **Courtesy rule**: nothing in this module raises into the execution path.
  ``__torch_dispatch__`` guards its bookkeeping; a meter bug must degrade to
  "no measurement", never break a run.
* **Accounting semantics** (documented in docs/profiling-m2-compute-and-watchdog.md):
  FLOPs are forward+backward *combined* (backward's ATen ops flow through the
  dispatcher too); fused-optimizer ops (``_foreach_*``) have no FlopCounter
  handlers and are counted as zero (under-counting, declared); operators
  spawned on threads the node created itself escape both the meter and the
  node context (same boundary as the progress attribution).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

# Imported once at module level: __torch_dispatch__ is the hot path and must
# not pay the import machinery per operator.
from comfy_execution.utils import get_executing_context  # noqa: E402

#: Recording modes for ``--cdl-profiling-record``.
MODES = ("off", "count", "census")

#: Per-node dispatch-call budget before census demotion (aggregate FLOPs keep
#: accumulating; only the per-op histogram stops).  Same spirit as the
#: offline probe's OPS_LIMIT, sized for real training loops.
DEFAULT_OPS_LIMIT = 2_000_000

#: Global dispatch-call budget per run.  Every intercepted operator pays a
#: fixed Python-dispatch toll (~60us), which on op-dense training loops adds
#: up to far more than the P2 <5% contract.  Because a node's per-step FLOPs
#: are constant, measuring a bounded SAMPLE of the first operators is
#: statistically equivalent for the calibration use - so once the budget is
#: exhausted the meter pops itself off the dispatch stack and the rest of the
#: run proceeds at full speed (snapshot is flagged "sampled").
DEFAULT_OPS_BUDGET = 500


class RunMeter(TorchDispatchMode):
    """Per-node FLOPs meter over the real execution path (P2, ADR-4).

    Single dispatch layer: the flop arithmetic is done in-class via torch's
    module-level ``flop_registry`` (the same functions
    ``FlopCounterMode._count_flops`` uses) - never nest a FlopCounterMode
    below, the doubled Python dispatch toll breaks the <5% overhead contract
    on op-dense training loops.  Attribution uses the host's
    ``CurrentNodeContext`` contextvar; enter/exit is handled by the executor
    hook, once per prompt - NOT per node (per-node enter/exit would break
    under interleaved async tasks).
    """

    def __init__(self, mode: str = "count", ops_limit: int = DEFAULT_OPS_LIMIT,
                 ops_budget: int = DEFAULT_OPS_BUDGET):
        if mode not in MODES:
            raise ValueError(f"unknown runmeter mode: {mode!r} (choose from {MODES})")
        super().__init__()
        self.mode = mode
        self.ops_limit = ops_limit
        self.ops_budget = ops_budget
        self._lock = threading.Lock()
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._unattributed: Dict[str, Any] = {"flops": 0, "op_count": 0}
        self._total_flops = 0
        self._total_ops = 0
        self._suppressed_errors = 0
        self._sampled = False  # budget exhausted -> single-hop short-circuit
        # Node-boundary gating (the zero-overhead mechanism): the meter is
        # entered/exited around EACH node by the executor, and only for the
        # FIRST sighting of a class_type - later nodes of the same type (and
        # all unmeasured-by-policy nodes) run with no meter on the stack.
        self.active = False
        self._measured_types: set = set()

    # ------------------------------------------------- node gating (P2) --
    def wants_node(self, class_type: str) -> bool:
        """Should the meter be on the stack for this node's execution?

        Policy: first sighting of a class_type only (per-step FLOPs are
        constant for a node family, so one measured sample carries the
        calibration signal), until the global op budget is spent.
        """
        if self._total_ops >= self.ops_budget:
            return False
        return class_type not in self._measured_types

    def mark_node_done(self, class_type: str) -> None:
        """Record that a class_type has been measured (skip later sightings)."""
        self._measured_types.add(class_type)

    def snapshot(self) -> Dict[str, Any]:
        """Consistent copy of everything recorded so far (for persistence)."""
        with self._lock:
            nodes = {
                node_id: {
                    "flops": data["flops"],
                    "op_count": data["op_count"],
                    "demoted": data["demoted"],
                    "ops": dict(data["ops"]) if self.census else None,
                }
                for node_id, data in self._nodes.items()
            }
            unattributed = {
                "flops": self._unattributed["flops"],
                "op_count": self._unattributed["op_count"],
            }
            return {
                "mode": self.mode,
                "nodes": nodes,
                "unattributed": unattributed,
                "total_flops": self._total_flops,
                "total_ops": self._total_ops,
                "sampled": self._sampled,
                "ops_budget": self.ops_budget,
                "suppressed_errors": self._suppressed_errors,
            }

    # ------------------------------------------------------------------ api --
    @property
    def census(self) -> bool:
        return self.mode == "census"

    # ----------------------------------------------------------- dispatch --
    def _bucket_for(self, node_id: Optional[str]) -> Dict[str, Any]:
        """Create-on-demand the per-node record (caller holds the lock)."""
        bucket = self._nodes.get(node_id)
        if bucket is None:
            bucket = {"flops": 0, "op_count": 0, "demoted": False, "ops": {}}
            self._nodes[node_id] = bucket
        return bucket

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: D102
        # During __torch_dispatch__ this mode is already popped off the stack,
        # so a direct call runs the real kernels - there is exactly ONE Python
        # dispatch hop per operator (this method).  FLOPs come from torch's
        # own flop_registry (the functions FlopCounterMode._count_flops uses).
        kwargs = kwargs or {}
        packet = getattr(func, "_overloadpacket", None)
        if self._total_ops >= self.ops_budget:
            # Sample budget spent: try to leave the dispatch stack for good
            # (single-layer meter - the pop removes THIS mode; if the host
            # refuses, we keep paying the single-hop toll, which is bounded).
            if not self._sampled:
                self._sampled = True
                try:
                    torch._C._pop_torch_dispatch_stack()
                except Exception:
                    self._suppressed_errors += 1
            # Unpriced operators also skip bookkeeping.
            if packet is None or packet not in flop_registry:
                return func(*args, **kwargs)
        out = func(*args, **kwargs)
        flops = 0
        try:
            flops = int(flop_registry[packet](*args, **kwargs, out_val=out))
        except Exception:
            flops = 0
        try:
            self._record(_context_node_id(), str(func), flops)
        except Exception:
            self._suppressed_errors += 1
        return out

    def _record(self, node_id: Optional[str], op_name: str, flops: int) -> None:
        with self._lock:
            self._total_flops += flops
            self._total_ops += 1
            if node_id is None:
                self._unattributed["flops"] += flops
                self._unattributed["op_count"] += 1
                return
            bucket = self._bucket_for(node_id)
            bucket["flops"] += flops
            bucket["op_count"] += 1
            if self.census and not bucket["demoted"]:
                if bucket["op_count"] > self.ops_limit:
                    bucket["demoted"] = True
                    bucket["ops"] = {}
                else:
                    bucket["ops"][op_name] = bucket["ops"].get(op_name, 0) + 1


def _context_node_id() -> Optional[str]:
    """The node the host is currently executing (None outside execution)."""
    try:
        return get_executing_context().node_id
    except Exception:
        return None


# ---------------------------------------------------------------- hook ----
# The executor (execution.py, comfy core layer) calls persist_run() at the end
# of each prompt; the actual database write lives in the app layer
# (app/profiling_routes.py registers itself here).  This keeps the
# comfy-profiling protocol layer free of app/database imports.
_persistence_hook: Optional[Any] = None


def set_persistence_hook(fn: Any) -> None:
    """Register the app-layer callback: ``fn(prompt_id, prompt, snapshot)``."""
    global _persistence_hook
    _persistence_hook = fn


def persist_run(prompt_id: str, prompt: Dict[str, Any], meter: "RunMeter") -> None:
    """Hand a finished run's snapshot to the registered persistence callback.

    Courtesy rule: never raises - a storage failure must not break a run.
    """
    try:
        if _persistence_hook is None or meter is None:
            return
        snapshot = meter.snapshot()
        if not snapshot.get("nodes"):
            return
        _persistence_hook(prompt_id, prompt, snapshot)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning(
            "profiling: failed to persist run measurement (%s)", exc)


def meter_from_args(args: Any = None) -> Optional[RunMeter]:
    """The meter for this launch from ``--cdl-profiling-record`` (None = off).

    Lazy on ``comfy.cli_args`` so unit tests can construct RunMeter directly
    without the flag machinery.
    """
    try:
        from comfy import cli_args

        mode = getattr(cli_args.args, "cdl_profiling_record", "count")
    except Exception:
        mode = "count"
    if mode not in MODES or mode == "off":
        return None
    return RunMeter(mode=mode)
