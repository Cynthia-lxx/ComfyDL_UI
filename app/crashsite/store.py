"""Crash site snapshot store: one SQLite database per execution snapshot.

Location: ``<user>/comfydl/snapshots/<snapshot_id>.db`` (ADR-3).  Each
snapshot captures ONE prompt execution: rows are appended incrementally by
SnapshotProvider.on_store (one row per node output, keyed by the input-
signature SHA256 the host computes for us), and consumed by on_lookup on a
later re-queue of the same prompt.

Standard-library sqlite3 on purpose: the schema is fixed and tiny, and the
user-facing database (app/database, SQLAlchemy) must not grow gigabyte BLOBs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id          TEXT PRIMARY KEY,
    created_at  INTEGER,
    updated_at  INTEGER,
    prompt_id   TEXT,
    prompt_json TEXT,
    note        TEXT,
    status      TEXT,
    format_version INTEGER,
    skipped_nodes  INTEGER,
    nodes          INTEGER DEFAULT 0,
    size_bytes     INTEGER DEFAULT 0,
    captured_json  TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS node_outputs (
    cache_key_hash TEXT,
    slot           INTEGER,
    node_id        TEXT,
    class_type     TEXT,
    output_type    TEXT,
    format         TEXT,
    size_bytes     INTEGER,
    sha256         TEXT,
    meta_json      TEXT,
    data           BLOB,
    PRIMARY KEY (cache_key_hash, slot)
);
CREATE INDEX IF NOT EXISTS idx_no_node ON node_outputs(node_id);
"""


def snapshots_root() -> Path:
    from folder_paths import get_user_directory

    root = Path(get_user_directory()) / "comfydl" / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


#: Snapshot on-disk format version.  Bumped when serialization semantics
#: change: on_lookup refuses snapshots whose version differs (mixed-version
#: rows would rehydrate garbage - the 2026-10-09 "got str" incident, where
#: pre-fix databases held flattened custom objects).
FORMAT_VERSION = 2

#: The snapshot library only grows.  Listing the newest K snapshots keeps the
#: cost bounded; P0-4 will expose limit/offset over HTTP once the panel has a
#: "load more" affordance.  ``nodes`` / ``size_bytes`` / ``captured_json`` are
#: maintained by ``upsert_node_output``, so the list never full-scans
#: ``node_outputs`` for databases written by this build.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 1000
_CAPTURE_LIMIT = 20
_AGG_COLUMNS = frozenset(("nodes", "size_bytes", "captured_json"))


def create_snapshot(prompt: Dict[str, Any], prompt_id: str = "",
                    note: str = "") -> Path:
    """Create a fresh snapshot database for one prompt execution."""
    snap_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = snapshots_root() / f"{snap_id}.db"
    conn = _connect(path)
    try:
        _init_schema(conn)
        now = int(time.time())
        conn.execute(
            "INSERT INTO snapshots (id, created_at, updated_at, prompt_id,"
            " prompt_json, note, status, format_version) VALUES (?,?,?,?,?,?,?,?)",
            (snap_id, now, now, prompt_id,
             json.dumps(prompt, default=str), note, "open", FORMAT_VERSION))
        conn.commit()
    finally:
        conn.close()
    return path


def format_version_of(path: Path) -> Optional[int]:
    """The stored format version, or None for a foreign/legacy file."""
    try:
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT format_version FROM snapshots WHERE id=?",
                (path.stem,)).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row and row[0] is not None else None
    except Exception:  # noqa: BLE001 - unreadable file = unusable anyway
        return None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    # Older databases may predate any of these columns; add them idempotently.
    # This only runs when a database is opened for writing - historical files
    # are NEVER migrated in bulk, so list_snapshots still needs a legacy path.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(snapshots)")}
    for column, declaration in (
        ("format_version", "INTEGER"),
        ("skipped_nodes", "INTEGER"),
        ("nodes", "INTEGER DEFAULT 0"),
        ("size_bytes", "INTEGER DEFAULT 0"),
        ("captured_json", "TEXT DEFAULT '[]'"),
    ):
        if column not in cols:
            conn.execute(
                "ALTER TABLE snapshots ADD COLUMN %s %s"
                % (column, declaration))
    conn.commit()


def upsert_node_output(path: Path, cache_key_hash: str, node_id: str,
                       slot: int, class_type: str, output_type: str,
                       format_tag: str, blob: bytes, meta_json: str) -> None:
    """Incrementally store one node output and maintain the list aggregates.

    What: append/replace one ``(cache_key_hash, slot)`` row, then keep
          ``nodes`` / ``size_bytes`` / ``captured_json`` on the single
          manifest row in lockstep, so ``list_snapshots`` needs no full scan.
    In:   the already-serialized slot payload and its input signature.
    Out:  None.  Databases from older builds lack the aggregate columns;
          those keep the historical behaviour (write the row, no aggregates)
          and are full-scanned by the list's legacy path.
    """
    sha = hashlib.sha256(blob).hexdigest()
    size = len(blob)
    conn = _connect(path)
    try:
        # One write transaction: reading the replaced row, the upsert and the
        # aggregate update must be atomic (rescue flush may run concurrently).
        conn.execute("BEGIN IMMEDIATE")
        old = conn.execute(
            "SELECT node_id, class_type, COALESCE(size_bytes, 0)"
            " FROM node_outputs WHERE cache_key_hash=? AND slot=?",
            (cache_key_hash, slot)).fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO node_outputs (cache_key_hash, node_id,"
            " slot, class_type, output_type, format, size_bytes, sha256,"
            " meta_json, data) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cache_key_hash, node_id, slot, class_type, output_type,
             format_tag, size, sha, meta_json, sqlite3.Binary(blob)))
        now = int(time.time())
        cols = {r[1] for r in conn.execute("PRAGMA table_info(snapshots)")}
        if _AGG_COLUMNS <= cols:
            row = conn.execute(
                "SELECT nodes, size_bytes, captured_json FROM snapshots"
                " WHERE id=?", (path.stem,)).fetchone() or (0, 0, "[]")
            nodes = int(row[0] or 0) + (0 if old else 1)
            total = int(row[1] or 0) + size - int(old[2] if old else 0)
            groups = _captured_groups(row[2])
            _apply_capture_delta(groups, old, node_id, class_type, size)
            conn.execute(
                "UPDATE snapshots SET nodes=?, size_bytes=?, captured_json=?,"
                " updated_at=? WHERE id=?",
                (nodes, total, json.dumps(groups), now, path.stem))
        else:
            conn.execute("UPDATE snapshots SET updated_at=? WHERE id=?",
                         (now, path.stem))
        conn.commit()
    finally:
        conn.close()


def _captured_groups(raw: Optional[str]) -> List[Dict[str, Any]]:
    """Decode the per-(node_id, class_type) aggregate kept on the manifest.

    What: tolerant decode of ``snapshots.captured_json``; a corrupt payload
          degrades to an empty list instead of breaking a capture.
    In:   raw JSON text (may be None / malformed).
    Out:  list of {node_id, class_type, slots, size_bytes}, all groups kept
          (truncation happens only when rendering the panel summary).
    """
    try:
        items = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    out: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and "node_id" in item and "class_type" in item:
            try:
                out.append({
                    "node_id": str(item["node_id"]),
                    "class_type": str(item["class_type"]),
                    "slots": int(item.get("slots") or 0),
                    "size_bytes": int(item.get("size_bytes") or 0),
                })
            except (TypeError, ValueError):
                continue
    return out


def _apply_capture_delta(groups: List[Dict[str, Any]], old: Optional[Tuple],
                         node_id: str, class_type: str, size: int) -> None:
    """Fold one upsert into the aggregate groups (INSERT OR REPLACE aware)."""
    index = {(g["node_id"], g["class_type"]): g for g in groups}
    new_key = (str(node_id), str(class_type))
    replaced_key = (str(old[0]), str(old[1])) if old else None
    if old is not None:
        prior = index.get(replaced_key)
        if prior is not None:
            prior["size_bytes"] -= int(old[2] or 0)
            if replaced_key != new_key:
                prior["slots"] -= 1
    group = index.get(new_key)
    if group is None:
        group = {"node_id": new_key[0], "class_type": new_key[1],
                 "slots": 0, "size_bytes": 0}
        groups.append(group)
        index[new_key] = group
    if replaced_key != new_key:
        group["slots"] += 1
    group["size_bytes"] += size
    groups[:] = [g for g in groups if g["slots"] > 0]


def _captured_view(groups: List[Dict[str, Any]], limit: int = 0
                   ) -> List[Dict[str, Any]]:
    """Same size-descending ordering / truncation as ``captured_nodes``."""
    ordered = sorted(groups, key=lambda d: (-d["size_bytes"], d["node_id"]))
    return ordered[:limit] if limit else ordered


def mark_status(path: Path, status: str, skipped: Optional[int] = None) -> None:
    conn = _connect(path)
    try:
        if skipped is None:
            conn.execute("UPDATE snapshots SET status=?, updated_at=? WHERE id=?",
                         (status, int(time.time()), path.stem))
        else:
            conn.execute(
                "UPDATE snapshots SET status=?, skipped_nodes=?, updated_at=?"
                " WHERE id=?", (status, int(skipped), int(time.time()),
                                path.stem))
        conn.commit()
    finally:
        conn.close()


def set_note(path: Path, note: str) -> None:
    """Overwrite the snapshot note (crash site M2 resume statistics)."""
    conn = _connect(path)
    try:
        conn.execute("UPDATE snapshots SET note=?, updated_at=? WHERE id=?",
                     (note, int(time.time()), path.stem))
        conn.commit()
    finally:
        conn.close()


def _captured_nodes_from_conn(conn: sqlite3.Connection,
                              limit: int = 0) -> List[Dict[str, Any]]:
    """Full-scan path over an already-open connection (legacy databases)."""
    rows = conn.execute(
        "SELECT node_id, class_type, COUNT(*), COALESCE(SUM(size_bytes),0)"
        " FROM node_outputs GROUP BY node_id, class_type"
        " ORDER BY 4 DESC, node_id").fetchall()
    out = [{"node_id": r[0], "class_type": r[1], "slots": r[2],
            "size_bytes": r[3]} for r in rows]
    return out[:limit] if limit else out


def captured_nodes(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    """Per-node capture summary, one row per node_id, largest output first.

    What: aggregates ``node_outputs`` by node (slots + total bytes) for the
          snapshot card's captured-node list (M2 explainability).  Kept for
          the resume/detail paths and for databases from older builds.
    In:   ``path`` — snapshot db; ``limit`` — keep only the ``limit`` largest
          (0 = all).
    Out:  list of {node_id, class_type, slots, size_bytes}, size-descending.
    """
    conn = _connect(path)
    try:
        return _captured_nodes_from_conn(conn, limit)
    finally:
        conn.close()


def _dedupe_latest_slot_rows(rows) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Latest row per (node_id, slot); rowid order = insertion order, so the
    last write of a re-executed node wins (same rule the host cache uses)."""
    latest: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        latest[(r[0], r[1])] = {
            "node_id": r[0], "slot": r[1], "class_type": r[2],
            "output_type": r[3], "format": r[4], "size_bytes": r[5] or 0,
            "cache_key_hash": r[6],
        }
    return latest


def snapshot_detail(path: Path) -> Optional[Dict[str, Any]]:
    """Manifest + the FULL captured row list for the scene-export summary.

    Unlike :func:`list_snapshots` (whose ``captured`` is capped at 20), this
    returns every (node_id, slot) row so the export UI can list all
    recoverable outputs.  Rows carry their serialize format tag; the routes
    layer annotates them with export formats/candidates.
    """
    manifest = load_manifest(path)
    if manifest is None:
        return None
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT node_id, slot, class_type, output_type, format,"
            " size_bytes, cache_key_hash FROM node_outputs ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    latest = _dedupe_latest_slot_rows(rows)
    manifest["rows"] = sorted(latest.values(),
                              key=lambda d: (d["node_id"], d["slot"]))
    return manifest


def get_node_rows(path: Path, node_id: str, with_payload: bool = True
                  ) -> List[Dict[str, Any]]:
    """All (deduped, latest-write-wins) slot rows for one node.

    ``with_payload=True`` includes the raw ``data`` blob and ``meta_json``
    (the export path); False returns metadata only (API responses must not
    ship gigabyte payloads to the frontend).
    """
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT node_id, slot, class_type, output_type, format,"
            " size_bytes, cache_key_hash, meta_json, data FROM node_outputs"
            " WHERE node_id=? ORDER BY rowid", (node_id,)).fetchall()
    finally:
        conn.close()
    latest: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        entry = {
            "node_id": r[0], "slot": r[1], "class_type": r[2],
            "output_type": r[3], "format": r[4], "size_bytes": r[5] or 0,
            "cache_key_hash": r[6],
        }
        if with_payload:
            entry["meta_json"] = r[7] or "{}"
            entry["data"] = bytes(r[8]) if r[8] is not None else b""
        latest[(r[0], r[1])] = entry
    return sorted(latest.values(), key=lambda d: d["slot"])


def get_outputs_by_hash(path: Path, cache_key_hash: str) -> List[Dict[str, Any]]:
    """All slot rows for one node execution (provider.on_lookup)."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT node_id, slot, class_type, output_type, format, meta_json,"
            " data FROM node_outputs WHERE cache_key_hash=? ORDER BY slot",
            (cache_key_hash,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "node_id": r[0], "slot": r[1], "class_type": r[2],
            "output_type": r[3], "format": r[4], "meta_json": r[5],
            "data": bytes(r[6]) if r[6] is not None else b"",
        }
        for r in rows
    ]


def list_snapshots(limit: int = DEFAULT_LIST_LIMIT,
                   offset: int = 0) -> List[Dict[str, Any]]:
    """List the newest snapshots without full-scanning ``node_outputs``.

    What: newest-first library page for the Crash Site panel.  Aggregates are
          maintained by ``upsert_node_output`` on the single manifest row, so
          one snapshot costs one connection and one row read.  Databases
          written by older builds lack those columns and fall back to the
          historical full scan on the same connection.
    In:   ``limit`` - page size (clamped to 1..``MAX_LIST_LIMIT``);
          ``offset`` - number of newest snapshots to skip.
    Out:  list of dicts with the exact same keys as before, including the
          capped ``captured`` summary (size-descending).  Broken files are
          skipped, never raised.
    """
    try:
        limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIST_LIMIT
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    paths = sorted(snapshots_root().glob("*.db"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    paths = paths[offset:offset + limit]
    out: List[Dict[str, Any]] = []
    for path in paths:
        try:
            conn = _connect(path)
            try:
                cols = {r[1] for r in
                        conn.execute("PRAGMA table_info(snapshots)")}
                names = ["id", "created_at", "prompt_id", "note", "status"]
                for optional in ("format_version", "skipped_nodes",
                                 "nodes", "size_bytes", "captured_json"):
                    if optional in cols:
                        names.append(optional)
                row = conn.execute(
                    "SELECT " + ", ".join(names)
                    + " FROM snapshots WHERE id=?", (path.stem,)).fetchone()
                if row is None:
                    continue
                d = dict(zip(names, row))
                if _AGG_COLUMNS <= cols:
                    nodes = int(d.get("nodes") or 0)
                    size_bytes = int(d.get("size_bytes") or 0)
                    # M2 explainability: 4 rows + a "+n more" expander.
                    captured = _captured_view(
                        _captured_groups(d.get("captured_json")),
                        _CAPTURE_LIMIT)
                else:
                    agg = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM"
                        " node_outputs").fetchone()
                    nodes, size_bytes = int(agg[0]), int(agg[1])
                    captured = _captured_nodes_from_conn(conn, _CAPTURE_LIMIT)
            finally:
                conn.close()
            out.append({
                "id": d["id"], "created_at": d["created_at"],
                "prompt_id": d["prompt_id"], "note": d["note"],
                "status": d["status"],
                "format_version": d.get("format_version"),
                "skipped_nodes": d.get("skipped_nodes") or 0,
                "nodes": nodes, "size_bytes": size_bytes,
                "file": path.name, "captured": captured,
            })
        except Exception:  # noqa: BLE001 - a broken db must not kill the list
            continue
    return out


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    """The snapshot header (prompt_json etc.) for the resume flow."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT id, created_at, prompt_id, prompt_json, note, status"
            " FROM snapshots WHERE id=?", (path.stem,)).fetchone()
        if row is None:
            return None
        try:
            prompt = json.loads(row[3] or "{}")
        except Exception:
            prompt = {}
        return {"id": row[0], "created_at": row[1], "prompt_id": row[2],
                "prompt": prompt, "note": row[4], "status": row[5],
                "file": path.name}
    finally:
        conn.close()


def find_snapshot_file(snapshot_id: str) -> Optional[Path]:
    """Resolve a snapshot id (or raw file name) to its db path."""
    if not snapshot_id:
        return None
    safe = os_path_safe(snapshot_id)
    path = snapshots_root() / f"{safe}.db"
    if path.is_file():
        return path
    path = snapshots_root() / safe
    if path.is_file() and path.suffix == ".db":
        return path
    return None


def os_path_safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-._")
