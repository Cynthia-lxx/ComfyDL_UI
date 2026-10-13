"""Scene export: turn captured snapshot rows into user-facing files (git-fire
M2, 2026-10-13).

What
----
The scene-export summary table lists every recoverable node output of a
snapshot; this module deserializes the selected rows and writes them as
ordinary files under ``output/comfydl/scene_export/<snapshot_id>/`` so they
join the normal ComfyUI output ecosystem (Results assets).  It is a READ-ONLY
consumer of the snapshot library - nothing in the library is modified.

Format matrix
-------------
Defaults follow "the format this data type is supposed to live in" (user
decision 2026-10-13); every row may override with another candidate:

=================  =====================  ==========================
serialize tag      default export         candidates
=================  =====================  ==========================
st (Tensor)        safetensors            safetensors, png*
dict-st/list-st    safetensors (+meta)    safetensors, json
json (scalar)      json                   json, txt
dataset-st         db (sqlite blob+meta)  db, json
pt (state_dict)    pt                     pt, safetensors
pickle             pt                     pt
failed             - (row disabled)       -
=================  =====================  ==========================

* png only succeeds when the tensor is image-shaped (ComfyUI IMAGE
  convention ``[B, H, W, C]`` or a bare ``[H, W, C]``); otherwise the writer
  raises and the row is reported as failed - the frontend falls back.

Reliability
-----------
Per-row isolation is the contract: one broken row must never abort the
scene - :func:`export_selections` wraps each selection in try/except and
reports failures alongside the written files.  All path components pass
through :func:`store.os_path_safe` (no traversal out of the export folder).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.crashsite import serialize, store

# Format candidates per serialize format tag (first = default).  Keys mirror
# serialize.py's format tags; "failed" is deliberately absent (not exportable).
_FORMAT_CANDIDATES: Dict[str, List[str]] = {
    "st": ["safetensors", "png"],
    "dict-st": ["safetensors", "json"],
    "list-st": ["safetensors", "json"],
    "json": ["json", "txt"],
    "dataset-st": ["db", "json"],
    "pt": ["pt", "safetensors"],
    "pickle": ["pt"],
}

#: File extensions per export format.
_EXTENSIONS: Dict[str, str] = {
    "safetensors": ".safetensors",
    "png": ".png",
    "json": ".json",
    "txt": ".txt",
    "db": ".db",
    "pt": ".pt",
}


def candidates(row_format: str) -> List[str]:
    """Export formats offered for one serialize format tag (UI dropdown)."""
    return list(_FORMAT_CANDIDATES.get(row_format, []))


def default_format(row_format: str) -> Optional[str]:
    """The preselected export format for one serialize format tag."""
    cands = _FORMAT_CANDIDATES.get(row_format)
    return cands[0] if cands else None


def annotate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach ``export_default`` / ``export_formats`` to detail rows."""
    out = []
    for row in rows:
        row = dict(row)
        row["export_default"] = default_format(row["format"])
        row["export_formats"] = candidates(row["format"])
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
#  Per-format writers: (deserialized object, destination path) -> None
# --------------------------------------------------------------------------- #

def _is_image_tensor(obj: Any) -> bool:
    """ComfyUI IMAGE convention: [B, H, W, C] with C in (3, 4), or [H, W, C]."""
    import torch

    if not isinstance(obj, torch.Tensor):
        return False
    shape = tuple(obj.shape)
    if len(shape) == 4 and shape[0] == 1 and shape[-1] in (3, 4):
        return True
    return len(shape) == 3 and shape[-1] in (3, 4)


def _write_safetensors(obj: Any, path: Path) -> None:
    """Any tensor-bearing object -> one safetensors file (meta folded in)."""
    tensors: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    serialize._flatten(obj, "v", tensors, meta)
    if not tensors:
        raise ValueError("no tensors to export as safetensors")
    path.write_bytes(serialize._st_save(tensors))
    if meta:
        side = path.with_name(path.name + ".meta.json")
        side.write_text(json.dumps(meta, default=str, indent=1),
                        encoding="utf-8")


def _write_png(obj: Any, path: Path) -> None:
    """Image-shaped tensor -> PNG (ComfyUI IMAGE convention)."""
    import torch
    from PIL import Image

    if not isinstance(obj, torch.Tensor) or not _is_image_tensor(obj):
        raise ValueError("tensor is not image-shaped [B,H,W,C] / [H,W,C]")
    frame = obj.detach().cpu().float()
    if frame.dim() == 4:
        frame = frame[0]
    arr = (frame.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr).save(path)


def _write_json(obj: Any, path: Path) -> None:
    try:
        text = json.dumps(obj, indent=1, ensure_ascii=False)
    except (TypeError, ValueError):
        text = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")


def _write_txt(obj: Any, path: Path) -> None:
    path.write_text(str(obj), encoding="utf-8")


def _write_db(obj: Any, path: Path) -> None:
    """CdlDataset -> self-describing sqlite (features/labels blobs + meta).

    Reimportable: the payload layout mirrors the snapshot store's own row
    format (safetensors blob + meta_json), so a future importer can reuse
    ``serialize.deserialize_output`` unchanged.
    """
    tensors: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    serialize._flatten(obj.features, "features", tensors, meta)
    labels = getattr(obj, "labels", None)
    if labels is not None:
        serialize._flatten(labels, "labels", tensors, meta)
    meta["feature_names"] = list(getattr(obj, "feature_names", []) or [])
    meta["target_name"] = getattr(obj, "target_name", None)
    meta["dataset_meta"] = getattr(obj, "meta", {}) or {}
    blob = serialize._st_save(tensors)
    meta_json = json.dumps(meta, default=str)

    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE dataset (payload BLOB, meta TEXT)")
        conn.execute("INSERT INTO dataset VALUES (?, ?)",
                     (sqlite3.Binary(blob), meta_json))
        conn.commit()
    finally:
        conn.close()


def _write_pt(obj: Any, path: Path) -> None:
    """state_dict (or the generic pickle fallback object) -> torch .pt."""
    import torch

    torch.save(obj, path)


_WRITERS = {
    "safetensors": _write_safetensors,
    "png": _write_png,
    "json": _write_json,
    "txt": _write_txt,
    "db": _write_db,
    "pt": _write_pt,
}


def _export_one_slot(obj: Any, fmt: str, path: Path) -> str:
    """Write one deserialized slot object; returns the format actually used.

    An unavailable choice (e.g. png on a non-image tensor) raises - the
    caller reports the row as failed rather than silently substituting.
    """
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(f"unknown export format {fmt!r}")
    writer(obj, path)
    return fmt


def export_selections(snapshot_path: Path, selections: List[Dict[str, Any]],
                      out_dir: Path, include_prompt: bool = False
                      ) -> Dict[str, Any]:
    """Export the selected nodes (+ optional prompt JSON) into ``out_dir``.

    In:
      ``snapshot_path`` — the source snapshot db;
      ``selections`` — [{node_id, format, class_type?}]; ``format`` is the
        user's dropdown choice (a candidate of the row's serialize tag);
      ``out_dir`` — destination directory (created, existing files overwritten
        - repeated exports are idempotent);
      ``include_prompt`` — also write the stored workflow prompt JSON.
    Out:
      {"dir": str, "files": [names], "failed": [{node_id, error}], "count": n}

    Per-row isolation: a failed slot/node is reported in ``failed`` and never
    aborts the remaining selections.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    failed: List[Dict[str, str]] = []

    if include_prompt:
        try:
            manifest = store.load_manifest(snapshot_path)
            prompt = (manifest or {}).get("prompt")
            if not isinstance(prompt, dict) or not prompt:
                raise ValueError("snapshot has no stored prompt")
            name = store.os_path_safe(f"{snapshot_path.stem}_prompt") + ".json"
            _write_json(prompt, out_dir / name)
            files.append(name)
        except Exception as exc:  # noqa: BLE001 - per-row isolation
            failed.append({"node_id": "prompt", "error": repr(exc)})

    for sel in selections or []:
        node_id = str((sel or {}).get("node_id") or "")
        if not node_id:
            continue
        chosen = (sel or {}).get("format")
        try:
            rows = store.get_node_rows(snapshot_path, node_id)
            if not rows:
                raise ValueError("no captured rows for this node")
            class_type = str((sel or {}).get("class_type")
                             or rows[0]["class_type"] or "node")
            for row in rows:
                fmt = chosen or default_format(row["format"])
                if fmt is None or fmt not in candidates(row["format"]):
                    # Chosen format doesn't fit this slot (multi-slot nodes
                    # with mixed types): fall back to the slot's own default.
                    fmt = default_format(row["format"])
                if fmt is None:
                    raise ValueError(
                        f"slot {row['slot']} format {row['format']!r} is not"
                        f" exportable")
                obj = serialize.deserialize_output(
                    row["format"], row["data"], row["meta_json"])
                base = store.os_path_safe(
                    f"{class_type}_{node_id}_s{row['slot']}")
                path = out_dir / (base + _EXTENSIONS[fmt])
                _export_one_slot(obj, fmt, path)
                files.append(path.name)
                # A safetensors export with structural meta also drops a
                # sidecar - count it so the UI report is exact.
                side = path.with_name(path.name + ".meta.json")
                if side.is_file():
                    files.append(side.name)
        except Exception as exc:  # noqa: BLE001 - per-row isolation
            failed.append({"node_id": node_id, "error": repr(exc)})

    return {"dir": str(out_dir), "files": files, "failed": failed,
            "count": len(files)}
