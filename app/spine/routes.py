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
