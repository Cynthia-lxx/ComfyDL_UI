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
from comfy.profiling.proflog import log as proflog

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


@routes.get("/comfydl/crashsite/snapshots/{snapshot_id}/detail")
async def snapshot_detail_route(request: web.Request) -> web.Response:
    """Full capture detail for the scene-export summary table (git-fire M2).

    Unlike the capped list endpoint, this returns EVERY captured (node, slot)
    row annotated with export formats/candidates, plus the stored workflow
    prompt.  No payloads - metadata only.
    """
    path = store.find_snapshot_file(request.match_info["snapshot_id"])
    if path is None:
        return web.json_response({"error": "snapshot not found"}, status=404)
    detail = store.snapshot_detail(path)
    if detail is None:
        return web.json_response({"error": "snapshot unreadable"}, status=500)
    from app.crashsite import export as export_mod

    detail["rows"] = export_mod.annotate_rows(detail.get("rows") or [])
    return web.json_response(detail)


@routes.post("/comfydl/crashsite/export")
async def export_route(request: web.Request) -> web.Response:
    """Write selected captured outputs (+ optional prompt JSON) to files.

    Body: {"id": <snapshot id>, "include_prompt": bool,
           "items": [{"node_id": "...", "format": "...", "class_type": "..."}]}
    Files land in output/comfydl/scene_export/<snapshot id>/ (user decision
    2026-10-13: output directory, so exports surface in Results assets).
    Per-row failures are reported, never fatal.
    """
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    payload = payload or {}
    snapshot_id = str(payload.get("id") or "")
    path = store.find_snapshot_file(snapshot_id)
    if path is None:
        return web.json_response({"error": "snapshot not found"}, status=404)
    items = payload.get("items")
    include_prompt = bool(payload.get("include_prompt"))
    if not isinstance(items, list) or (not items and not include_prompt):
        return web.json_response(
            {"error": "no export items selected"}, status=400)
    items = [i for i in items if isinstance(i, dict) and i.get("node_id")]

    import asyncio
    from pathlib import Path as _Path

    import folder_paths
    from app.crashsite import export as export_mod

    out_dir = (_Path(folder_paths.get_output_directory()) / "comfydl"
               / "scene_export" / store.os_path_safe(snapshot_id))
    loop = asyncio.get_running_loop()
    # Deserialization + disk writes happen OFF the event loop (GB-scale
    # payloads possible - same courtesy as provider.on_store).
    result = await loop.run_in_executor(
        None, export_mod.export_selections, path, items, out_dir,
        include_prompt)
    proflog("low", "crash site: exported %d file(s) from snapshot %s to %s"
            " (%d failed row(s))", result["count"], snapshot_id,
            result["dir"], len(result["failed"]), request=request)
    return web.json_response(result)


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

    # VISIBILITY FIX (2026-10-09): without client_id the executor sets
    # server.client_id = None for the whole run, which suppresses EVERY
    # executing/executed/execution_start/success websocket event - the
    # resumed run executed silently (frontend saw nothing but our own
    # toast).  Echo the requester's client_id back so the resume behaves
    # like a run submitted from that same browser tab.
    client_id = (payload or {}).get("client_id")
    if not isinstance(client_id, str) or not client_id:
        client_id = None

    # M2 explainability: flag the next prompt start as a resume of this
    # snapshot so the provider can write the skip/re-run statistics note.
    # MUST happen BEFORE put() - the prompt worker may pick the job up and
    # fire on_prompt_start immediately.
    provider = _SNAPSHOT_PROVIDER
    if provider is not None:
        try:
            provider.mark_resume(prompt_id, snapshot_id)
        except Exception:  # noqa: BLE001 - cosmetic only
            pass

    number = float(_SERVER.number)
    _SERVER.number += 1
    extra_data: Dict[str, Any] = {"create_time": int(time.time() * 1000)}
    if client_id is not None:
        extra_data["client_id"] = client_id
    _PROMPT_QUEUE.put((number, prompt_id, prompt, extra_data,
                       valid[2], {}))
    # Same as POST /prompt: refresh the frontend's Job Queue so the resumed
    # prompt shows up as an active job instead of running invisibly.
    try:
        _SERVER.queue_updated()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass
    _logger.info("crash site: resumed snapshot %s as prompt %s",
                 snapshot_id, prompt_id)
    proflog(
        "low", "crash site: resume requested - snapshot=%s prompt=%s"
        " (%d node(s) in stored workflow)", snapshot_id, prompt_id,
        len(prompt), request=request)
    try:
        captured = store.captured_nodes(path, limit=20)
    except Exception:  # noqa: BLE001 - cosmetic only
        captured = []
    return web.json_response({
        "resumed": True, "prompt_id": prompt_id,
        "snapshot": manifest["id"],
        # M2: source snapshot's capture summary for immediate frontend
        # feedback (authoritative skip/re-run stats arrive later on the NEW
        # snapshot's note, once the resumed run finishes).
        "captured_nodes": len(captured),
        "captured": captured,
    })
