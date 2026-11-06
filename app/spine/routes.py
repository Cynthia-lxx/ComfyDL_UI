"""Spine HTTP routes: state readout + rate control (dual-track P0).

Mirrors the crashsite/profiling route style (``web.RouteTableDef``
re-declared in server.py so every route also gets its ``/api`` twin).

* ``GET  /comfydl/spine``           - running state, rate, switch description
  (including the spine's own measured overhead - the meta-profiling panel).
* ``GET  /comfydl/spine/snapshot``  - the latest complete snapshot.
* ``PATCH /comfydl/spine``          - ``{"enabled": bool}`` and/or
  ``{"rate_s": number}``; ``rate_s <= 0`` stops the thread entirely.
"""

from __future__ import annotations

from typing import Any, Dict

from aiohttp import web

from app.spine import SWITCH_ID, get_spine
from comfy.profiling import switch_registry

routes = web.RouteTableDef()


def _state_payload() -> Dict[str, Any]:
    spine = get_spine()
    return {
        "switch": switch_registry.describe(SWITCH_ID),
        "running": spine.is_running(),
        "rate_s": spine.rate_s,
        "sent": spine.sent,
    }


@routes.get("/comfydl/spine")
async def spine_state(request: web.Request) -> web.Response:
    return web.json_response(_state_payload())


@routes.get("/comfydl/spine/snapshot")
async def spine_snapshot(request: web.Request) -> web.Response:
    snap = get_spine().snapshot()
    if snap is None:
        return web.json_response(
            {"schema_version": None, "ts": None, "detail": "no tick yet"},
            status=200)
    return web.json_response(snap.to_dict())


@routes.patch("/comfydl/spine")
async def spine_configure(request: web.Request) -> web.Response:
    spine = get_spine()
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "body must be an object"}, status=400)

    rate = body.get("rate_s")
    if rate is not None:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            return web.json_response(
                {"error": "rate_s must be a number"}, status=400)
        spine.set_rate(float(rate))

    enabled = body.get("enabled")
    if enabled is not None:
        if not isinstance(enabled, bool):
            return web.json_response(
                {"error": "enabled must be a boolean"}, status=400)
        if enabled and switch_registry.is_hard_disabled(SWITCH_ID):
            # hard-disabled: refuse the re-enable with 409, same contract
            # as the crashsite switch route
            return web.json_response(
                {**_state_payload(), "error": "hard-disabled"},
                status=409)
        switch_registry.set_enabled(SWITCH_ID, enabled)
        if enabled:
            spine.start()
        else:
            spine.stop()

    return web.json_response(_state_payload())


@routes.get("/comfydl/switches")
async def switches_list(request: web.Request) -> web.Response:
    """Every registered feature switch, for the Settings > ComfyDL panel."""
    return web.json_response({
        "switches": [switch_registry.describe(sid)
                     for sid in switch_registry.list_switches()]
    })


@routes.get("/comfydl/dataio/cache/stats")
async def dataio_cache_stats(request: web.Request) -> web.Response:
    """Live DataFrame-cache counters (hits/misses/entries/bytes/evictions).

    The user-visible acceptance surface for ``comfydl.dataio.cache``: run a
    workflow twice and watch ``hits`` climb; flip the switch off and watch
    ``misses`` accumulate instead.  Always 200 — ``{"available": false}`` when
    the cache module is not loaded (bare environment).
    """
    try:
        from comfydl.nodes import _dataio_cache as _dc
    except Exception:  # noqa: BLE001 - bare environment
        return web.json_response({"available": False})
    stats = _dc.stats()
    stats["enabled"] = _dc.enabled()
    stats["available"] = True
    return web.json_response(stats)


@routes.get("/comfydl/dataio/cache/entries")
async def dataio_cache_entries(request: web.Request) -> web.Response:
    """Per-entry inventory of the DataFrame cache (Profiling Data cache card).

    Returns ``{"available": bool, "enabled": bool, "stats": {...},
    "entries": [...]}`` where each entry carries ``key/reader/label/path/
    mtime_ns/file_size/params/cached_bytes/hits/stored_ts/last_hit`` (no
    payloads).  Always 200 — same bare-environment contract as the stats
    endpoint above.
    """
    try:
        from comfydl.nodes import _dataio_cache as _dc
    except Exception:  # noqa: BLE001 - bare environment
        return web.json_response({"available": False})
    return web.json_response({
        "available": True,
        "enabled": _dc.enabled(),
        "stats": _dc.stats(),
        "entries": _dc.entries(),
    })


@routes.get("/comfydl/notebook/variables")
async def notebook_variables(request: web.Request) -> web.Response:
    """Variable inspection for one notebook kernel (mechanism 4).

    ``GET /comfydl/notebook/variables?workflow=<id>`` → ``{"available": bool,
    "enabled": bool, "workflow": id, "variables": [{name, type, repr}, ...]}``.
    Reprs are truncated server-side; tensor contents never leave the process.
    Returns 409 while the ``comfydl.notebook`` switch is off (the kernels
    cannot exist with the feature disabled).
    """
    try:
        from comfydl.nodes import _notebook_kernel as _nk
    except Exception:  # noqa: BLE001 - bare environment
        return web.json_response({"available": False})
    if not _nk.enabled():
        return web.json_response({
            "available": True,
            "enabled": False,
            "error": f"the '{_nk.SWITCH_ID}' switch is off",
        }, status=409)
    workflow_id = request.query.get("workflow", "")
    if not workflow_id:
        return web.json_response({"error": "missing ?workflow=<id>"},
                                 status=400)
    return web.json_response({
        "available": True,
        "enabled": True,
        "workflow": workflow_id,
        "has_kernel": workflow_id in _nk.active_workflows(),
        "variables": _nk.describe_variables(workflow_id),
    })


@routes.post("/comfydl/notebook/free")
async def notebook_free(request: web.Request) -> web.Response:
    """Reset notebook kernel state (Jupyter's "restart kernel").

    ``POST /comfydl/notebook/free`` with ``{"workflow": "<id>"}`` drops one
    workflow's kernel; without a workflow body it drops all of them. 409
    while the ``comfydl.notebook`` switch is off.
    """
    try:
        from comfydl.nodes import _notebook_kernel as _nk
    except Exception:  # noqa: BLE001 - bare environment
        return web.json_response({"available": False})
    if not _nk.enabled():
        return web.json_response({
            "available": True,
            "enabled": False,
            "error": f"the '{_nk.SWITCH_ID}' switch is off",
        }, status=409)
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        body = {}
    workflow_id = body.get("workflow") if isinstance(body, dict) else None
    removed = _nk.clear(str(workflow_id) if workflow_id else None)
    return web.json_response({
        "available": True,
        "enabled": True,
        "freed": removed,
    })


@routes.patch("/comfydl/switches/{switch_id}")
async def switches_set(request: web.Request) -> web.Response:
    """Hot-toggle one feature switch: ``{"enabled": bool}``.

    Same contract as every other switch route: unknown id 404, re-enabling a
    hard-disabled process 409 (needs a restart without the hard flag), success
    echoes the fresh description so the panel never lies.
    """
    switch_id = request.match_info["switch_id"]
    if switch_id not in switch_registry.list_switches():
        return web.json_response({"error": "unknown switch"}, status=404)
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    enabled = body.get("enabled") if isinstance(body, dict) else None
    if not isinstance(enabled, bool):
        return web.json_response({"error": "enabled must be a boolean"},
                                 status=400)
    if enabled and switch_registry.is_hard_disabled(switch_id):
        return web.json_response(
            {"error": "hard-disabled",
             "switch": switch_registry.describe(switch_id)},
            status=409)
    switch_registry.set_enabled(switch_id, enabled)
    return web.json_response({"switch": switch_registry.describe(switch_id)})
