"""End-to-end aiohttp event-loop probe for the Crash Site snapshot list.

M-D T6-1, 2026-11-13.  ``bench_crashsite_overhead.py`` measures the
in-process PromptExecutor path only; the user-reported 5~15s stall came from
the SERVER: ``GET /comfydl/crashsite/snapshots`` used to run a synchronous
library scan on the aiohttp event loop, so ``POST /prompt``, ``POST /interrupt``
and every WebSocket message queued behind it.  This probe reproduces that
sharing with a real aiohttp server and measures how long a lightweight
"prompt-like" request is blocked.

Three arms, same seeded library:
  baseline - no list request at all (the loop is idle)
  paged    - current handler (executor + P0-2 aggregates + P0-4 paging)
  inline   - a control route that calls ``store.list_snapshots()`` INLINE,
             i.e. the pre-P0-1 behaviour, to show the size of the regression
             that was fixed.  Never mount this control outside this probe.

Run (manual, NOT part of the default smoke suite):
  penv\\Scripts\\python.exe cdl_smoke_tests\\bench_crashsite_server_loop.py
  penv\\Scripts\\python.exe ... --snapshots 200 --rows 40 --duration 3

R7 statistical caliber (2026-11-14, issue 5): single-round max/max verdicts
produced garbage like "paged/baseline max = 0.12x" when the baseline's one
round happened to catch a 14ms outlier.  The probe now runs ``--rounds``
timed rounds per arm (round 0 is an UNTIMED warmup and is discarded), and
the verdict compares median-of-round-p99 values, with p50/p95/p99/max all
reported.  p99 (not p95) carries the verdict because the inline regression
is a SPIKE that only a tail percentile sees.  Three arms, same seeded
library:
  baseline - no list request at all (the loop is idle)
  paged    - current handler (executor + P0-2 aggregates + P0-4 paging)
  inline   - a control route that calls ``store.list_snapshots()`` INLINE,
             i.e. the pre-P0-1 behaviour, to show the size of the regression
             that was fixed.  Never mount this control outside this probe.

Readout: per-arm median-of-rounds ping p50/p95/p99 + max-of-rounds in ms,
plus the median list request wall time.  The acceptance bar is that
``paged``'s median p95 stays close to ``baseline`` (a few ms), while
``inline`` is expected to be much worse.  This is a server-loop
microbenchmark, NOT a substitute for the user's real-machine A/B:
it has no browser, no WebSocket fan-out and no GPU inference.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from app.crashsite import routes as cs_routes  # noqa: E402
from app.crashsite import store  # noqa: E402


def _seed_library(root: Path, n_snapshots: int, rows_per: int) -> None:
    """Create N historical snapshots with realistic manifest rows."""
    store.snapshots_root = lambda: root
    for i in range(n_snapshots):
        path = store.create_snapshot({"1": {"class_type": "Fake"}},
                                     prompt_id=f"bench-{i}")
        for j in range(rows_per):
            store.upsert_node_output(
                path, f"h-{j}", "1", j, "Fake", "Tensor", "st",
                b"\x01" * 256, "{}")
        store.mark_status(path, "completed", skipped=0)


def _make_app() -> web.Application:
    app = web.Application()

    async def ping(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def inline_list(_request: web.Request) -> web.Response:
        # Control arm ONLY: deliberately synchronous, pre-P0-1 behaviour.
        return web.json_response({"snapshots": store.list_snapshots()})

    app.router.add_get("/ping", ping)
    app.router.add_get("/comfydl/crashsite/snapshots",
                       cs_routes.snapshots_list)
    app.router.add_get("/__bench_inline", inline_list)
    return app


def _stats(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max": ordered[-1],
    }


async def _probe(client: TestClient, list_path: str,
                 duration: float) -> Dict[str, Any]:
    """Hammer a tiny request while one list request is in flight."""
    rtts: List[float] = []
    list_start = time.perf_counter()
    list_task = None

    async def do_list() -> None:
        resp = await client.get(list_path)
        await resp.read()

    if list_path:
        list_task = asyncio.create_task(do_list())
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        resp = await client.get("/ping")
        await resp.read()
        rtts.append((time.perf_counter() - t0) * 1000.0)
        await asyncio.sleep(0.01)
    if list_task is not None:
        await list_task
    out: Dict[str, Any] = _stats(rtts)
    out["list_ms"] = (time.perf_counter() - list_start) * 1000.0
    out["pings"] = len(rtts)
    out["mean"] = statistics.fmean(rtts) if rtts else 0.0
    return out


async def _run(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="cdl_e2e_loop_") as tmp:
        _seed_library(Path(tmp), args.snapshots, args.rows)
        client = TestClient(TestServer(_make_app()))
        await client.start_server()
        try:
            arms = (
                ("baseline", ""),
                ("paged  ", "/comfydl/crashsite/snapshots?limit=20"),
                ("inline ", "/__bench_inline"),
            )
            print("crash site server-loop probe")
            print(f"  snapshots={args.snapshots} rows_per={args.rows} "
                  f"duration={args.duration}s rounds={args.rounds} "
                  f"(round 0 = discarded warmup)")
            print("  arm        p50 med    p95 med    p99 med    max of rounds"
                  "  list med   pings")
            results: Dict[str, Dict[str, Any]] = {}
            for name, path in arms:
                rounds: List[Dict[str, Any]] = []
                for rep in range(args.rounds + 1):
                    r = await _probe(client, path, args.duration)
                    if rep:  # round 0 is the untimed warmup
                        rounds.append(r)
                # R7 caliber: median across rounds for every percentile;
                # max is reported but NEVER drives the verdict.
                agg = {
                    "p50": statistics.median([r["p50"] for r in rounds]),
                    "p95": statistics.median([r["p95"] for r in rounds]),
                    "p99": statistics.median([r["p99"] for r in rounds]),
                    "max": max(r["max"] for r in rounds),
                    "list_ms": statistics.median([r["list_ms"] for r in rounds]),
                    "pings": sum(r["pings"] for r in rounds),
                    "rounds": rounds,
                }
                results[name.strip()] = agg
                print(f"  {name}  {agg['p50']:8.2f}ms {agg['p95']:9.2f}ms "
                      f"{agg['p99']:9.2f}ms {agg['max']:12.2f}ms "
                      f"{agg['list_ms']:8.1f}ms {agg['pings']:6d}")
            base = results["baseline"]["p99"]
            paged = results["paged"]["p99"]
            inline = results["inline"]["p99"]
            # p99 (not p95) is the verdict statistic: the inline regression
            # is a SPIKE that only a tail percentile sees, while median
            # p95 stays clean; p99 across rounds stays robust to single-
            # round outliers that made the old max/max verdict lie.
            print(f"  VERDICT: paged/baseline median-p99 = "
                  f"{paged / max(base, 0.01):.2f}x"
                  f" | inline/baseline median-p99 = "
                  f"{inline / max(base, 0.01):.2f}x")
            print("  (verdict uses median-of-rounds p99; single-round max is")
            print("   reported above for reference only - issue 5, R7)")
            print("  SCOPE CAVEAT: no browser / WS fan-out / GPU inference;"
                  " real-machine A/B still belongs to the user.")
        finally:
            await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=50)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
