"""HTTP routes for the ComfyDL Profiling panel (reform: Profiling Tools M1).

Two POST endpoints, both pure JSON:

* ``/comfydl/profiling/estimate``   - body ``{"prompt": <graphToPrompt output>,
  "budget": {...} (optional), "assumptions": {...} (optional),
  "titles": {...} (optional)}`` -> the full estimation report;
* ``/comfydl/profiling/postmortem`` - body ``{"message": <exception text>,
  "node_id", "node_type", "prompt"/"budget"/"assumptions" (optional)}`` ->
  the OOM post-mortem (attribution + suggested batch size).

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
