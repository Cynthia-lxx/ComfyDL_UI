"""Crash site HTTP routes (M1): snapshot list / resume via server re-queue.

The capture itself needs no endpoint in M1 - SnapshotProvider writes rows
incrementally as nodes finish (bigplan ADR-6).  These routes expose the
snapshot library to the panel and perform the resume: re-queueing the
stored prompt so the host's cache machinery (provider.on_lookup) skips the
finished nodes.

Route style mirrors app/profiling_routes.py (RouteTableDef re-declared in
server.py so every route also gets its /api twin).
"""

from __future__ import annotations

import logging
import time
import uuid as uuid_mod
from typing import Any, Dict

from aiohttp import web

from app.crashsite import store

_logger = logging.getLogger(__name__)

routes = web.RouteTableDef()

#: Injected by server.py at startup (mirrors profiling_routes.set_watchdog).
_SNAPSHOT_PROVIDER = None
_PROMPT_QUEUE = None
_SERVER = None


def set_snapshot_provider(provider) -> None:
    global _SNAPSHOT_PROVIDER
    _SNAPSHOT_PROVIDER = provider


def set_prompt_queue(prompt_queue, server) -> None:
    global _PROMPT_QUEUE, _SERVER
    _PROMPT_QUEUE = prompt_queue
    _SERVER = server


@routes.get("/comfydl/crashsite/snapshots")
async def snapshots_list(request: web.Request) -> web.Response:
    return web.json_response({"snapshots": store.list_snapshots()})


@routes.post("/comfydl/crashsite/resume")
async def resume(request: web.Request) -> web.Response:
    """Re-queue a snapshot's original prompt; the host skips finished nodes."""
    import traceback

    try:
        return await _resume_impl(request)
    except Exception as exc:  # noqa: BLE001 - surface a readable error
        _logger.error("crash site: resume failed\n%s", traceback.format_exc())
        return web.json_response(
            {"error": f"resume failed: {exc!r}",
             "traceback": traceback.format_tb(exc.__traceback__)}, status=500)


async def _resume_impl(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    snapshot_id = str((payload or {}).get("id") or "")
    path = store.find_snapshot_file(snapshot_id)
    if path is None:
        return web.json_response({"error": "snapshot not found"}, status=404)
    manifest = store.load_manifest(path)
    if manifest is None:
        return web.json_response({"error": "snapshot unreadable"}, status=500)
    prompt = manifest.get("prompt")
    if not isinstance(prompt, dict) or not prompt:
        return web.json_response(
            {"error": "snapshot has no stored prompt"}, status=409)
    if _PROMPT_QUEUE is None or _SERVER is None:
        return web.json_response({"error": "queue unavailable"}, status=503)

    import execution as execution_mod

    prompt_id = manifest.get("prompt_id") or str(uuid_mod.uuid4())

    # Validate exactly like POST /prompt does: a snapshot of an edited graph
    # (or a host update) failing validation must surface as a readable 409.
    valid = await execution_mod.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        return web.json_response(
            {"error": "stored prompt no longer valid",
             "node_errors": valid[3]}, status=409)

    number = float(_SERVER.number)
    _SERVER.number += 1
    extra_data: Dict[str, Any] = {"create_time": int(time.time() * 1000)}
    _PROMPT_QUEUE.put((number, prompt_id, prompt, extra_data,
                       valid[2], {}))
    _logger.info("crash site: resumed snapshot %s as prompt %s",
                 snapshot_id, prompt_id)
    return web.json_response({"resumed": True, "prompt_id": prompt_id,
                              "snapshot": manifest["id"]})
