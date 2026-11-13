"""HTTP control + front-end ingestion for VERBOSE diagnostics (repair R0).

What: three tiny routes that let the UI (or a console command) turn the
      low-overhead timestamped log on/off and append front-end events to the
      SAME log file, so the user only has to hand back one file.
In:   ``GET /comfydl/verbose`` -> none;
      ``POST /comfydl/verbose`` -> ``{"enabled": bool, "level": str}``;
      ``POST /comfydl/verbose/fe`` -> ``{"lines": [str, ...]}``.
Out:  state dict / ``{"ok": true}``.  All handlers stay inert when OFF.
"""

from aiohttp import web

from comfy import verbose_log

routes = web.RouteTableDef()


@routes.get("/comfydl/verbose")
async def verbose_state_route(_request: web.Request) -> web.Response:
    """Report the switch and the current log path."""
    return web.json_response(verbose_log.state())


@routes.post("/comfydl/verbose")
async def verbose_set_route(request: web.Request) -> web.Response:
    """Hot-toggle the switch (never raises on a malformed body)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - diagnostics must not break a request
        body = {}
    if not isinstance(body, dict):
        body = {}
    return web.json_response(verbose_log.set_enabled(
        bool(body.get("enabled")), str(body.get("level", "verbose"))))


@routes.post("/comfydl/verbose/fe")
async def verbose_fe_route(request: web.Request) -> web.Response:
    """Append buffered front-end lines to the same log (bounded by ingest)."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    verbose_log.ingest(body.get("lines") if isinstance(body, dict) else None)
    return web.json_response({"ok": True})
