#!/usr/bin/env python
"""Standalone smoke test: the runs comparison view (P3 Craft 3, ADR-D10).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_runs_view.py

Covers: the /comfydl/runs index + detail routes (aiohttp TestClient, same
pattern as test_profiling_history), the cdl_preview_data payload contract
for BOTH preview types - the new "runs" multi-series payload AND the
pre-existing "loss" payload (backward compatibility: same event name, same
key set, single-series semantics untouched) - and the JS syntax gate for
cdlapp.js (node --check; skipped when node is unavailable).
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_runsview_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy import loss_preview as lp  # noqa: E402
from comfy import runs_ledger as rl  # noqa: E402

# Isolate ledger storage; seed two scans' worth of runs.
_SANDBOX = Path(tempfile.mkdtemp(prefix="cdl_runsview_root_"))
rl.set_root(_SANDBOX)

_HIST_A = [[0, 1.0], [1, 0.6], [2, 0.4]]
_HIST_B = [[0, 2.0], [1, 1.9], [2, 1.7]]


def _seed() -> None:
    with rl.start_run(experiment="alpha", params={"lr": 0.01},
                      direction="min", combo_index=0, seed=1) as run_a:
        for step, value in _HIST_A:
            run_a.append_history(step, value)
        run_a.set_best(True)
    with rl.start_run(experiment="alpha", params={"lr": 0.9},
                      direction="min", combo_index=1, seed=2) as run_b:
        for step, value in _HIST_B:
            run_b.append_history(step, value)


_seed()

# ---------------------------------------------------------------------------
# T1: routes (aiohttp TestClient, test_profiling_history pattern)


def _route_checks() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app import runs_routes

    async def _checks():
        app = web.Application()
        app.add_routes(runs_routes.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/comfydl/runs")
            check("R1 index route 200", resp.status == 200, str(resp.status))
            data = await resp.json()
            check("R2 index carries both runs with ledger fields",
                  data.get("count") == 2 and len(data.get("runs") or []) == 2
                  and data["runs"][0]["experiment"] == "alpha"
                  and data["runs"][0]["combo_index"] == 0
                  and data["runs"][0]["direction"] == "min",
                  json.dumps(data)[:200])

            resp = await client.get("/comfydl/runs?limit=1")
            data = await resp.json()
            check("R3 limit keeps the newest rows",
                  data.get("count") == 1
                  and data["runs"][0]["combo_index"] == 1)

            run_id = data["runs"][0]["run_id"]
            resp = await client.get("/comfydl/runs/detail?run_id=" + run_id)
            check("R4 detail route 200", resp.status == 200, str(resp.status))
            detail = await resp.json()
            check("R5 detail carries the per-step history + params",
                  detail.get("metrics_history") == _HIST_B
                  and detail.get("params") == {"lr": 0.9}
                  and "device" in detail)

            resp = await client.get("/comfydl/runs/detail?run_id=nope")
            check("R6 unknown run -> 404", resp.status == 404, str(resp.status))
            resp = await client.get("/comfydl/runs/detail?run_id=../evil")
            check("R7 path traversal refused (path-jailed reader)",
                  resp.status == 404, str(resp.status))
        finally:
            await client.close()

    asyncio.run(_checks())


# ---------------------------------------------------------------------------
# T2: push payload contract (new "runs" type + backward-compatible "loss")


class _Capture:
    def __init__(self):
        self.events = []

    def send_sync(self, event, data):
        self.events.append((event, data))


def _capture_server():
    """Point loss_preview's lazy ``server.PromptServer`` at a recorder.

    Injects a stub module when ``server`` was never imported (the harness
    does not need it); otherwise swaps the real class's ``instance`` and
    restores it afterwards.
    """
    capture = _Capture()

    class _Stub:
        instance = capture

    mod = sys.modules.get("server")
    if mod is None:
        stub = types.ModuleType("server")
        stub.PromptServer = _Stub
        sys.modules["server"] = stub

        def _restore():
            sys.modules.pop("server", None)
    else:
        original = mod.PromptServer.instance
        mod.PromptServer.instance = capture

        def _restore():
            mod.PromptServer.instance = original
    return capture, _restore


def _payload_checks() -> None:
    capture, restore = _capture_server()
    try:
        # the new multi-series runs payload
        lp.push_runs_snapshot(
            title="alpha",
            series=[
                {"label": "#0 lr=0.01", "points": _HIST_A, "best": True},
                {"label": "#1 lr=0.9", "points": _HIST_B, "best": False},
            ],
            direction="min", best_index=0, finished=True,
        )
        check("P1 runs push emitted exactly one event",
              len(capture.events) == 1)
        event, payload = capture.events[0]
        check("P2 runs payload rides the established cdl_preview_data event",
              event == "cdl_preview_data")
        check("P3 runs payload schema: type/direction/best_index/finished",
              payload.get("type") == "runs" and payload.get("title") == "alpha"
              and payload.get("direction") == "min"
              and payload.get("best_index") == 0
              and payload.get("finished") is True)
        series = payload.get("series") or []
        check("P4 series entries carry label/points/best/final",
              len(series) == 2 and series[0]["label"] == "#0 lr=0.01"
              and series[0]["points"] == _HIST_A
              and series[0]["best"] is True and series[0]["final"] == 0.4
              and series[1]["best"] is False)

        # backward compatibility: the pre-existing single-series loss payload
        capture.events.clear()
        lp.push_loss_series(title="loss", values=[1.0, 0.5], total=10)
        event, payload = capture.events[0]
        check("P5 loss payload unchanged (same event, same key set)",
              event == "cdl_preview_data" and payload.get("type") == "loss"
              and set(payload.keys()) == {
                  "type", "node_id", "title", "step", "total", "best",
                  "points", "finished"}
              and payload["points"] == [[1, 1.0], [2, 0.5]])

        # >DATA_MAX_POINTS series get downsampled, not dropped
        capture.events.clear()
        big = [[i, float(i)] for i in range(4000)]
        lp.push_runs_snapshot(title="big", series=[
            {"label": "s", "points": big, "best": True}])
        series = capture.events[0][1]["series"]
        check("P6 oversized series downsample to the cap",
              0 < len(series[0]["points"]) <= lp.DATA_MAX_POINTS)
    finally:
        restore()


# ---------------------------------------------------------------------------
# T3: JS syntax gate (visual judgement stays with the user per convention)


def _js_checks() -> None:
    node = shutil.which("node")
    if node is None:
        print("SKIP  J1 node --check (node not on PATH)")
        return
    js = REPO_ROOT / "app" / "cdlapp_assets" / "cdlapp.js"
    result = subprocess.run(
        [node, "--check", str(js)], capture_output=True, text=True)
    check("J1 cdlapp.js parses (node --check)", result.returncode == 0,
          result.stderr[:300])
    text = js.read_text(encoding="utf-8")
    check("J2 buildRunsChart wired into renderCharts",
          "buildRunsChart(d)" in text and 'd.type === "runs"' in text)
    check("J3 runs section built in buildExtras",
          "cdlapp-sec-runs" in text and "cdlapp-runs-list" in text)
    check("J4 reduced-motion exemption present (motion discipline)",
          "prefers-reduced-motion" in text)
    css = (REPO_ROOT / "app" / "cdlapp_assets" / "cdlapp.css").read_text(
        encoding="utf-8")
    check("J5 CSS: runs grid area + palette present",
          'grid-area: runs' in css and ".cdlapp-runs-line[data-i=\"5\"]" in css)


def _report() -> int:
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL "
          f"({len(_RESULTS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    _route_checks()
    _payload_checks()
    _js_checks()
    rl.set_root(None)
    raise SystemExit(_report())
