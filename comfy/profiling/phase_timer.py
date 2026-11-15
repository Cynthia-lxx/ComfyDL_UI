"""PhaseTimer - executor-level phase anatomy (M-D P7, ADR-D1 cost model).

What: dissects one prompt execution into the three cost positions of the
ADR-D1 model and keeps the last N executions in memory for the Profiling
panel's "Phase anatomy" card:

* ``prompt_setup``  - execute_async entry until the node loop starts
  (dynamic prompt build, cache set/lookup, GC pass);
* ``per_step``      - time spent inside per-step observability callbacks
  (KSampler step callbacks via :func:`wrap_sampler_callback`, training
  progress via :func:`step_window`) - the "每步期" ComfyDL machinery cost,
  deliberately NOT the model's own compute;
* ``run_teardown``  - from the end of node execution to executor exit
  (crash-site rescue, lifecycle notification, run persistence).

Each finished record also carries the run's share of every Tier-1 switch's
self-measured overhead (baseline delta, same contract as the runs ledger)
so the panel can show gross vs net side by side.

In:   ``begin_execution`` / ``mark_prompt_setup_done`` /
      ``mark_teardown_start`` / ``finish_execution`` from ``execution.py``;
      per-step notes from ``comfy/sample.py`` and ``comfy/training_core.py``.
Out:  read-only :func:`current` / :func:`history` snapshots (served by
      ``GET /comfydl/profiling/phases``); nothing is written to disk.
Caveats: the dehydrated host has a single prompt_worker, so the current
execution is one module-level slot under a lock - concurrent executors
would interleave.  Every hook is a fast no-op when the switch is off or
unregistered, and bookkeeping time is self-metered into
``comfydl.phasetimer`` (ADR-D1: a switch that cannot account for its own
cost does not ship).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

from comfy.profiling import switch_registry

#: Registry id of this feature (Tier-1, self-metered).
SWITCH_ID = "comfydl.phasetimer"

switch_registry.register_switch(
    SWITCH_ID,
    default=True,       # the hooks are constant-size; panels read it live
    hot_toggle=True,
    self_meter=True,
)

#: Finished executions kept in memory (the panel shows the last one).
_HISTORY_MAX = 20

_LOCK = threading.RLock()
_CURRENT: Optional[Dict[str, Any]] = None
_HISTORY: deque = deque(maxlen=_HISTORY_MAX)


def _enabled() -> bool:
    try:
        return switch_registry.is_enabled(SWITCH_ID)
    except Exception:  # noqa: BLE001 - registry is optional in bare tests
        return False


def begin_execution(prompt_id: Any) -> None:
    """Open the per-execution record at ``execute_async`` entry."""
    if not _enabled():
        return
    t0 = time.perf_counter()
    global _CURRENT
    with _LOCK:
        if _CURRENT is None:
            _CURRENT = {
                "prompt_id": str(prompt_id),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "prompt_setup_s": None,
                "per_step_s": 0.0,
                "steps": 0,
                "run_teardown_s": None,
                "gross_s": None,
                "net_s": None,
                "overheads": {},
                "status": "running",
                # private keys (stripped before serving):
                "_t_start": time.perf_counter(),
                "_t_teardown": None,
                "_own_s": 0.0,
                "_overhead_baseline": switch_registry.self_overhead_all(),
            }
    _own(t0)


def mark_prompt_setup_done(prompt_id: Any = "") -> None:
    """Close the ``prompt_setup`` phase when the node loop is about to start."""
    if not _enabled():
        return
    t0 = time.perf_counter()
    with _LOCK:
        cur = _CURRENT
        if cur is not None and cur["prompt_setup_s"] is None:
            if not prompt_id or cur["prompt_id"] == str(prompt_id):
                cur["prompt_setup_s"] = round(
                    time.perf_counter() - cur["_t_start"], 4)
    _own(t0)


def mark_teardown_start(prompt_id: Any = "") -> None:
    """Open the ``run_teardown`` phase when node execution has ended."""
    if not _enabled():
        return
    t0 = time.perf_counter()
    with _LOCK:
        cur = _CURRENT
        if cur is not None and cur["_t_teardown"] is None:
            if not prompt_id or cur["prompt_id"] == str(prompt_id):
                cur["_t_teardown"] = time.perf_counter()
    _own(t0)


def finish_execution(prompt_id: Any = "", failed: bool = False) -> None:
    """Close the record, compute gross/net, and push it into the history."""
    if not _enabled():
        return
    t0 = time.perf_counter()
    global _CURRENT
    with _LOCK:
        cur = _CURRENT
        if cur is None or (prompt_id and cur["prompt_id"] != str(prompt_id)):
            return
        _CURRENT = None
        now = time.perf_counter()
        cur["gross_s"] = round(now - cur["_t_start"], 4)
        if cur["_t_teardown"] is not None:
            cur["run_teardown_s"] = round(now - cur["_t_teardown"], 4)
        cur["status"] = "failed" if failed else "completed"
        overheads: Dict[str, float] = {}
        baseline = cur["_overhead_baseline"]
        for sid, live in switch_registry.self_overhead_all().items():
            base = baseline.get(sid)
            delta = float(live["total_s"]) - (
                float(base["total_s"]) if base else 0.0)
            if delta > 1e-6:
                overheads[sid] = round(delta, 4)
        cur["overheads"] = overheads
        cur["net_s"] = round(
            max(0.0, float(cur["gross_s"]) - sum(overheads.values())), 4)
        record = {k: v for k, v in cur.items() if not k.startswith("_")}
        _HISTORY.append(record)
        _own(t0)
        # ADR-D1: book the module's own hook time for this execution, once.
        switch_registry.record_self_overhead(
            SWITCH_ID, float(cur.get("_own_s", 0.0)))


def _own(t0: float) -> None:
    """Accumulate this hook's own cost into the current record (no lock)."""
    with _LOCK:
        cur = _CURRENT
        if cur is not None:
            cur["_own_s"] = cur.get("_own_s", 0.0) + (time.perf_counter() - t0)


def note_step(seconds: float) -> None:
    """Accumulate one per-step observability cost into the current record."""
    if not _enabled():
        return
    with _LOCK:
        cur = _CURRENT
        if cur is not None:
            cur["per_step_s"] = round(
                cur["per_step_s"] + max(0.0, float(seconds)), 4)
            cur["steps"] += 1


def wrap_sampler_callback(callback: Optional[Callable]) -> Optional[Callable]:
    """Time each KSampler step callback into ``per_step`` (P7 anatomy).

    Mirrors ``verbose_log.wrap_sampler_callback``'s contract: the original
    callable is returned untouched when the switch is off (or the callback
    is ``None``), so the OFF path is a plain passthrough.
    """
    if callback is None or not _enabled():
        return callback

    def wrapped(step, x0, x, total_steps):
        t0 = time.perf_counter()
        try:
            return callback(step, x0, x, total_steps)
        finally:
            note_step(time.perf_counter() - t0)

    return wrapped


@contextlib.contextmanager
def step_window():
    """Time one training-progress call into ``per_step`` (training loop).

    The InterruptProcessingException raised by the host's progress hook
    propagates untouched (the timing sits in ``finally``).
    """
    if not _enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        note_step(time.perf_counter() - t0)


def current() -> Optional[Dict[str, Any]]:
    """The in-flight execution record (public keys only) or ``None``."""
    with _LOCK:
        cur = _CURRENT
        if cur is None:
            return None
        return {k: v for k, v in cur.items() if not k.startswith("_")}


def history() -> list:
    """Finished execution records, oldest first (public keys only)."""
    with _LOCK:
        return [dict(rec) for rec in _HISTORY]


def _reset_for_tests() -> None:
    """Test isolation only; never call from product code."""
    global _CURRENT
    with _LOCK:
        _CURRENT = None
        _HISTORY.clear()
