"""Runs ledger HTTP routes (P3 Craft 3, M-C).

Serves the runs ledger to the comparison view:

* ``GET /comfydl/runs?limit=200`` - the index layer (runs.csv rows, oldest
  first), for the App Panel's experiment list.
* ``GET /comfydl/runs/detail?run_id=<id>`` - one run's detail JSON (the
  per-step metric history the comparison chart overlays).

Read-only over :mod:`comfy.runs_ledger` (the same path-jailed readers the
node uses), no cache - the ledger is small and always current.  Mirrors the
``cdlapp_routes`` pattern; the ``/api`` twin is merged in ``server.py``.
"""

from __future__ import annotations

import logging

from aiohttp import web

from comfy import runs_ledger as rl

_routes_logger = logging.getLogger("ComfyDL.runs_routes")

routes = web.RouteTableDef()


def _ok(data) -> web.Response:
    return web.json_response(data)


@routes.get("/comfydl/runs")
async def runs_index(request: web.Request) -> web.Response:
    """Index rows (csv), newest last; ``limit`` keeps the most recent N."""
    try:
        limit = int(request.query.get("limit", "200"))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, 2000))
    try:
        rows = rl.read_index(limit=limit)
    except Exception as exc:  # noqa: BLE001 - the index is a courtesy
        _routes_logger.warning("[runs-routes] index read failed: %s", exc)
        rows = []
    return _ok({"runs": rows, "count": len(rows),
                "root": str(rl.default_root())})


@routes.get("/comfydl/runs/detail")
async def runs_detail(request: web.Request) -> web.Response:
    """One run's detail JSON; 404 when unknown (never a path traversal)."""
    run_id = request.query.get("run_id", "")
    detail = rl.read_detail(run_id)
    if detail is None:
        return web.json_response({"error": f"unknown run '{run_id}'"},
                                 status=404)
    return _ok(detail)
