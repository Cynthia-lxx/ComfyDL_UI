"""HTTP routes for the ComfyDL Profiling panel (reform: Profiling Tools M1/M2).

POST endpoints, pure JSON:

* ``/comfydl/profiling/estimate``   - body ``{"prompt": <graphToPrompt output>,
  "budget": {...} (optional), "assumptions": {...} (optional),
  "titles": {...} (optional)}`` -> the full estimation report (M1 memory
  ledger + M2 compute ledger);
* ``/comfydl/profiling/postmortem`` - body ``{"message": <exception text>,
  "node_id", "node_type", "prompt"/"budget"/"assumptions" (optional)}`` ->
  the OOM post-mortem (attribution + suggested batch size).

GET endpoints (M2 watchdog):

* ``/comfydl/profiling/watchdog/status`` - the live sample of the execution
  monitor (current node, process CPU/RSS, system CPU);
* ``/comfydl/profiling/watchdog/log?limit=50`` - the most recent burst
  events from the persistent JSONL ledger.

The frontend asset side (``profiler.js`` / ``profiler.css``) is served from
the repository-owned ``app/profiling_assets/`` directory by a static route
mounted in ``server.py``; the loader that pulls it into the page is injected
by ``app/frontend_patch.py``. Nothing here touches the frontend package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from aiohttp import web

from comfy.profiling import analyse_oom, estimate_workflow
from comfy.profiling.engine import DeviceBudget, budget_from_environment

routes = web.RouteTableDef()

#: The M2 execution watchdog, injected by server.py after construction.
#: Kept as a module global because the route table is declared before the
#: server exists (the /api twin loop in server.py re-declares these).
_WATCHDOG = None

#: The most recent estimation report, so the watchdog can attach a
#: data-scale hint to a burst event (the engine <-> monitor handshake).
_LAST_REPORT: Dict[str, Any] = {}


def set_watchdog(watchdog) -> None:
    """Give the routes the server's watchdog instance."""
    global _WATCHDOG
    _WATCHDOG = watchdog


def scale_hint_for(node_id: str) -> Optional[dict]:
    """What the M1/M2 estimator last said about ``node_id`` (or None)."""
    for node in _LAST_REPORT.get("nodes", []):
        if node.get("id") == node_id:
            return {
                "class_type": node.get("class_type"),
                "total_bytes": node.get("total_bytes"),
                "flops_total": node.get("flops_total"),
                "basis": node.get("basis"),
            }
    return None


def _budget_from_payload(payload: Dict[str, Any]) -> Optional[DeviceBudget]:
    """The caller-supplied budget, else the live primary-device budget."""
    data = payload.get("budget")
    if isinstance(data, dict) and int(data.get("total_bytes") or 0) > 0:
        return DeviceBudget(
            name=str(data.get("name") or "device"),
            total_bytes=int(data["total_bytes"]),
            free_bytes=int(data.get("free_bytes") or data["total_bytes"]),
        )
    return budget_from_environment()


@routes.post("/comfydl/profiling/estimate")
async def estimate(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    report = estimate_workflow(
        payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {},
        _budget_from_payload(payload),
        payload.get("assumptions") if isinstance(payload.get("assumptions"), dict) else None,
        payload.get("titles") if isinstance(payload.get("titles"), dict) else None,
    )
    global _LAST_REPORT
    _LAST_REPORT = report
    return web.json_response(report)


@routes.post("/comfydl/profiling/postmortem")
async def postmortem(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    result = analyse_oom(payload, _budget_from_payload(payload))
    return web.json_response(result)


@routes.get("/comfydl/profiling/watchdog/status")
async def watchdog_status(request: web.Request) -> web.Response:
    if _WATCHDOG is None:
        return web.json_response({"available": False, "running": False}, status=503)
    return web.json_response(_WATCHDOG.snapshot())


@routes.get("/comfydl/profiling/watchdog/log")
async def watchdog_log(request: web.Request) -> web.Response:
    if _WATCHDOG is None:
        return web.json_response({"events": [], "available": False}, status=503)
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        limit = 50
    return web.json_response({"events": _WATCHDOG.recent(limit=limit)})
