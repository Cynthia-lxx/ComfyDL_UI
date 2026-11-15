#!/usr/bin/env python
"""Standalone smoke test: PhaseTimer phase anatomy (M-D P7, ADR-D1).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_phase_timer.py

Covers the ``comfy.profiling.phase_timer`` module end to end on the CPU
tier (identical logic on CUDA):

* T1 switch contract (``comfydl.phasetimer``, self-metered);
* T2 the OFF path is a full no-op (no record, callbacks pass through);
* T3 the happy-path lifecycle: begin -> steps -> prompt-setup done ->
  teardown start -> finish, with gross/net and per-phase fields;
* T4 ``wrap_sampler_callback`` times steps and preserves the return value;
* T5 ``step_window`` times a block and propagates exceptions untouched;
* T6 the failed-execution status is recorded honestly;
* T7 the HTTP route layer: ``GET /comfydl/profiling/phases`` serves the
  history + the registry's cumulative overhead meter (always 200).
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_phase_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy.profiling import phase_timer as pt  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

_ORIGINAL_ENABLED = sr.is_enabled(pt.SWITCH_ID)

# ---------------------------------------------------------------------------
# T1: switch contract

check("T1a phasetimer switch registered", pt.SWITCH_ID in sr.list_switches())
desc = sr.describe(pt.SWITCH_ID)
check("T1b describe(): default on, hot-toggleable, self-metered",
      desc.get("default") is True and desc.get("hot_toggle") is True
      and desc.get("self_meter") is True)

# ---------------------------------------------------------------------------
# T2: the OFF path is a full no-op

try:
    sr._force_state(pt.SWITCH_ID, enabled=False)
    pt.begin_execution("off-run")
    pt.note_step(0.5)
    pt.mark_prompt_setup_done("off-run")
    pt.mark_teardown_start("off-run")
    pt.finish_execution("off-run")
    check("T2a off: nothing recorded",
          pt.current() is None and pt.history() == [])
    sentinel = lambda step, x0, x, total: "passthrough"  # noqa: E731
    check("T2b off: wrap_sampler_callback is a passthrough",
          pt.wrap_sampler_callback(sentinel) is sentinel)
    with pt.step_window():
        pass
    check("T2c off: step_window is a no-op", True)
finally:
    sr._force_state(pt.SWITCH_ID, enabled=True)

# ---------------------------------------------------------------------------
# T3: happy-path lifecycle

pt.begin_execution("p1")
cur = pt.current()
check("T3a begin: in-flight record with running status",
      cur is not None and cur["prompt_id"] == "p1"
      and cur["status"] == "running"
      and cur["prompt_setup_s"] is None and cur["gross_s"] is None)
pt.note_step(0.01)
pt.note_step(0.02)
cur = pt.current()  # snapshots are copies: re-read after each mutation
check("T3b steps accumulate into per_step",
      abs(cur["per_step_s"] - 0.03) < 1e-6 and cur["steps"] == 2,
      f"per_step={cur['per_step_s']} steps={cur['steps']}")
time.sleep(0.02)
pt.mark_prompt_setup_done("p1")
cur = pt.current()
check("T3c prompt_setup closes once",
      cur["prompt_setup_s"] is not None and cur["prompt_setup_s"] >= 0)
pt.mark_teardown_start("p1")
pt.finish_execution("p1")
check("T3d finish: slot cleared, record in history",
      pt.current() is None and len(pt.history()) == 1)
rec = pt.history()[-1]
check("T3e finish: gross/net/status present and net <= gross",
      rec["gross_s"] is not None and rec["net_s"] is not None
      and rec["status"] == "completed"
      and 0.0 <= rec["net_s"] <= rec["gross_s"] + 1e-9,
      f"gross={rec['gross_s']} net={rec['net_s']}")
check("T3f finish: private keys stripped",
      all(not k.startswith("_") for k in rec))
check("T3g finish: per-phase fields populated",
      rec["prompt_setup_s"] is not None and rec["run_teardown_s"] is not None
      and rec["per_step_s"] > 0)
check("T3h ADR-D1: module self-metered its hooks",
      sr.self_overhead(pt.SWITCH_ID)["samples"] >= 1)

# ---------------------------------------------------------------------------
# T4: sampler callback timing (on path)

pt.begin_execution("p2")
seen = []
wrapped = pt.wrap_sampler_callback(
    lambda step, x0, x, total: seen.append(step) or "ok")
result = wrapped(3, None, None, 10)
check("T4a wrap: return value and args preserved",
      result == "ok" and seen == [3])
pt.finish_execution("p2")
rec2 = pt.history()[-1]
check("T4b wrap: the step landed in per_step", rec2["steps"] == 1,
      f"steps={rec2['steps']} per_step={rec2['per_step_s']}")
check("T4c wrap: None callback stays None (host contract)",
      pt.wrap_sampler_callback(None) is None)

# ---------------------------------------------------------------------------
# T5: step_window - timing + exception transparency

pt.begin_execution("p3")
raised = False
try:
    with pt.step_window():
        time.sleep(0.01)
        raise RuntimeError("host interrupt surrogate")
except RuntimeError:
    raised = True
check("T5a step_window: exception propagates untouched", raised)
pt.finish_execution("p3")
rec3 = pt.history()[-1]
check("T5b step_window: the block still landed in per_step",
      rec3["steps"] == 1 and rec3["per_step_s"] >= 0.005,
      f"steps={rec3['steps']} per_step={rec3['per_step_s']}")

# ---------------------------------------------------------------------------
# T6: failed execution recorded honestly

pt.begin_execution("p4")
pt.finish_execution("p4", failed=True)
check("T6a failed status recorded", pt.history()[-1]["status"] == "failed")

# mismatched prompt id never poisons another execution's record
pt.begin_execution("p5")
pt.finish_execution("p-other")
check("T6b mismatched finish ignored (record still in flight)",
      pt.current() is not None and pt.current()["prompt_id"] == "p5")
pt.finish_execution("p5")

# ---------------------------------------------------------------------------
# T7: HTTP route layer - GET /comfydl/profiling/phases

async def _route_checks() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app import profiling_routes

    app = web.Application()
    app.add_routes(profiling_routes.routes)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/comfydl/profiling/phases")
        body = await resp.json()
        check("T7a route: phases endpoint always 200 with contract",
              resp.status == 200 and body.get("available") is True
              and "current" in body and isinstance(body.get("history"), list)
              and isinstance(body.get("overheads"), dict))
        check("T7b route: history carries the records written above",
              any(r.get("prompt_id") == "p1" for r in body.get("history", [])))
        check("T7c route: overheads meter covers phasetimer itself",
              pt.SWITCH_ID in body.get("overheads", {}))
    finally:
        await client.close()


asyncio.run(_route_checks())

# ---------------------------------------------------------------------------
# T8: probe_session - the disable context never breaks a probe (T6-2, P7)

from comfy import compile_manager as cm  # noqa: E402

_ran = False
with cm.probe_session():
    _ran = True
check("T8a probe_session: the timed block always executes", _ran)


def _nested_ok() -> bool:
    try:
        with cm.probe_session():
            with cm.probe_session():
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


check("T8b probe_session: repeatable and nestable on this torch build",
      _nested_ok())

# restore
sr._force_state(pt.SWITCH_ID, enabled=_ORIGINAL_ENABLED)
pt._reset_for_tests()


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== phase timer test: {passed} PASS / "
          f"{len(_RESULTS) - passed} FAIL (of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
