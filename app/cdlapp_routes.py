"""HTTP routes for the CDL App Panel (App-mode companion, 2026-10-30).

GET endpoints, file-oriented (the deep-learning output channel: DL runs
produce tensors/models/text - the panel surfaces them as *files* to download
and *text* to copy, not as pretty images):

* ``/comfydl/artifacts?limit=200``        - list the files under the ComfyUI
  output directory (recursive, newest first)::

      {"artifacts": [{"name": ..., "path": <output-relative posix path>,
                      "size": <bytes>, "mtime": <epoch seconds>,
                      "media_type": <mimetype>}], "root": <output dir name>}

* ``/comfydl/artifacts/download?name=<rel path>`` - stream one artifact back
  with ``Content-Disposition: attachment``.

Path jail (both endpoints): the name must be a relative path that resolves
inside the output directory root - absolute paths, ``..`` escapes, symlinked
entries and missing files are rejected with 403/404.  Only GET is registered.

The payload shape deliberately mirrors the official cloud asset schema
(``/jobs/{id}/assets``: filename/subfolder/type/media_type fields) so that if
the official App mode ever localises its jobs/assets surface, this module can
be swapped for it without touching the frontend panel (ADR-8).

The frontend asset side (``cdlapp.js`` / ``cdlapp.css``) is served from the
repository-owned ``app/cdlapp_assets/`` directory by a static route mounted in
``server.py``; the loader that pulls it into the page is injected by
``app/frontend_patch.py``.  Both route tables are merged into the ``/api``
twin by server.py (pitfall 25: every route must be re-declared there).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from aiohttp import web

routes = web.RouteTableDef()

#: Overridable output root for tests; None = resolve from folder_paths at
#: request time (the canonical ComfyUI output directory).
_OUTPUT_ROOT_OVERRIDE: Optional[Path] = None

#: Guard against pathological listing sizes (artifacts listing is bounded).
_MAX_LIST = 500


def set_output_root(path: Optional[Path]) -> None:
    """Pin the artifacts root (tests); None restores folder_paths resolution."""
    global _OUTPUT_ROOT_OVERRIDE
    _OUTPUT_ROOT_OVERRIDE = Path(path) if path else None


def _output_root() -> Path:
    if _OUTPUT_ROOT_OVERRIDE is not None:
        return _OUTPUT_ROOT_OVERRIDE
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        # Standalone / test contexts without a full ComfyUI runtime: fall back
        # to ./output next to the CWD, matching the nodes' relative convention.
        return Path(os.getcwd()) / "output"


def _resolve_in_jail(root: Path, name: str) -> Optional[Path]:
    """Resolve ``name`` inside ``root`` or return None (jail violation).

    Rejects absolute paths, ``..`` escapes and symlinks anywhere along the
    resolved path.  ``Path.resolve()`` normalises ``..`` and symlinks together,
    so the final commonpath check covers both.
    """
    if not name or name.strip() == "":
        return None
    candidate = Path(name)
    if candidate.is_absolute():
        return None
    try:
        root_res = root.resolve()
        raw = root / candidate
        probe = raw
        while probe != root_res and probe != probe.parent:
            if probe.is_symlink():
                return None
            probe = probe.parent
        full = raw.resolve()
        if os.path.commonpath([str(full), str(root_res)]) != str(root_res):
            return None
    except (OSError, ValueError):
        return None
    return full


#: Explicit media types: the platform mimetypes table is unreliable for our
#: contract (Windows maps .csv to application/vnd.ms-excel) and the payload
#: schema must stay deterministic across CPU/GPU installs.
_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".pt": "application/octet-stream",
    ".pth": "application/octet-stream",
    ".ckpt": "application/octet-stream",
    ".npy": "application/octet-stream",
    ".npz": "application/octet-stream",
}


def _media_type(name: str) -> str:
    return _MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


@routes.get("/comfydl/artifacts")
async def list_artifacts(request: web.Request) -> web.Response:
    """List files under the output directory, newest first."""
    try:
        limit = int(request.query.get("limit", "200"))
    except ValueError:
        limit = 200
    limit = max(1, min(limit, _MAX_LIST))

    root = _output_root()
    if not root.is_dir():
        return web.json_response({"artifacts": [], "root": str(root.name)})

    entries = []
    try:
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append({
                "name": path.name,
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "media_type": _media_type(path.name),
            })
    except OSError as exc:
        return web.json_response({"error": f"listing failed: {exc}"}, status=500)

    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return web.json_response({
        "artifacts": entries[:limit],
        "root": root.name,
        "generated_at": time.time(),
    })


@routes.get("/comfydl/artifacts/download")
async def download_artifact(request: web.Request) -> web.StreamResponse:
    """Stream one artifact with Content-Disposition attachment."""
    name = request.query.get("name", "")
    root = _output_root()
    full = _resolve_in_jail(root, name)
    if full is None:
        return web.json_response(
            {"error": "path escapes the output directory jail"}, status=403)
    if not full.is_file():
        return web.json_response({"error": "artifact not found"}, status=404)

    rel = full.relative_to(root.resolve()).as_posix()
    resp = web.FileResponse(full)
    resp.headers["Content-Disposition"] = (
        f'attachment; filename="{full.name}"')
    resp.headers["X-Cdl-Artifact-Path"] = rel
    return resp
