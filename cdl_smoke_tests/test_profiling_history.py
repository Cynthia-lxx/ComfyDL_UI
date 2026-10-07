"""Smoke tests for P2 measured-FLOPs persistence, history, and calibration.

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling_history.py

Uses a temporary sqlite file + ``Base.metadata.create_all`` (no Alembic, no
server): the executor hook is exercised through the app-layer callback
(``_persist_measurement``) exactly as ``runmeter.persist_run`` would invoke
it, then the /history and /estimate routes through an aiohttp TestServer.
"""

import json
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _bootstrap_db():
    """Temporary sqlite DB wired into the app's global Session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import db as app_db
    from app.database.models import Base

    path = Path(tempfile.mkdtemp(prefix="cdl_hist_")) / "test.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    app_db.Session = sessionmaker(bind=engine)
    return path


def _persist_checks() -> None:
    from app import profiling_routes
    from app.database.models import ProfilingNodeStat, ProfilingRun

    snapshot = {
        "mode": "count",
        "total_flops": 5000,
        "total_ops": 42,
        "nodes": {
            "3": {"flops": 4200, "op_count": 30, "demoted": False, "ops": None},
            "6": {"flops": 800, "op_count": 12, "demoted": False, "ops": None},
        },
        "unattributed": {"flops": 10, "op_count": 1},
        "suppressed_errors": 0,
    }
    prompt = {
        "3": {"class_type": "CdlRegressionTrain", "inputs": {}},
        "6": {"class_type": "CdlPlot", "inputs": {}},
    }
    profiling_routes._persist_measurement("pid-1234", prompt, snapshot)

    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    # read through the same sessionmaker the app global points at
    engine = create_engine("sqlite://")  # placeholder; inspect via app global
    del engine, sessionmaker, select  # keep imports minimal below
    # simplest: query through the app's own Session global
    from app.database import db as app_db

    with app_db.create_session() as session:
        run = session.query(ProfilingRun).one()
        stats = session.query(ProfilingNodeStat).all()
    check("H1 run row stored", run.prompt_id == "pid-1234"
          and run.total_flops == 5000 and run.total_ops == 42
          and run.node_count == 2 and run.mode == "count", str(run.total_flops))
    check("H2 node rows stored with class_type",
          len(stats) == 2
          and {s.class_type for s in stats} == {"CdlRegressionTrain", "CdlPlot"}
          and {s.flops for s in stats} == {4200, 800},
          str([(s.class_type, s.flops) for s in stats]))

    # H3: empty snapshot must not store anything.
    profiling_routes._persist_measurement("pid-empty", prompt,
                                          {"mode": "count", "nodes": {}})
    with app_db.create_session() as session:
        check("H3 empty snapshot stores nothing",
              session.query(ProfilingRun).count() == 1)


def _route_checks() -> None:
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app import profiling_routes

    async def _checks():
        app = web.Application()
        app.add_routes(profiling_routes.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/comfydl/profiling/history")
            check("H4 history route 200", resp.status == 200, str(resp.status))
            data = await resp.json()
            check("H5 history returns the run with node stats",
                  data.get("available") is True and len(data.get("runs") or []) == 1
                  and len(data["runs"][0]["nodes"]) == 2,
                  json.dumps(data)[:200])

            # estimate merge: a prompt with a class_type that has history now
            # carries measured_flops / measured_samples / measured_ratio.
            prompt = {"3": {"class_type": "CdlRegressionTrain", "inputs": {}}}
            resp2 = await client.post("/comfydl/profiling/estimate",
                                      json={"prompt": prompt})
            data2 = await resp2.json()
            node = [n for n in data2["nodes"] if n["class_type"] == "CdlRegressionTrain"]
            check("H6 estimate merges measured fields",
                  node and "measured_flops" in node[0]
                  and node[0].get("measured_samples", 0) >= 1,
                  json.dumps(node)[:200])
            if node and node[0].get("measured_ratio") is not None:
                check("H7 measured_ratio in (0, big]",
                      node[0]["measured_ratio"] > 0, str(node[0]["measured_ratio"]))
        finally:
            await client.close()

    asyncio.run(_checks())


def main() -> int:
    _bootstrap_db()
    _persist_checks()
    _route_checks()
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
