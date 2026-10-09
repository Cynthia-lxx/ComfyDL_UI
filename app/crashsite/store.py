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
from typing import Any, Dict, List, Optional

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
    skipped_nodes  INTEGER
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
    # Legacy databases (format_version < 1) predate the column; add it so
    # format_version_of returns a comparable value instead of raising.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(snapshots)")]
    if "format_version" not in cols:
        conn.execute("ALTER TABLE snapshots ADD COLUMN format_version INTEGER")
    if "skipped_nodes" not in cols:
        conn.execute("ALTER TABLE snapshots ADD COLUMN skipped_nodes INTEGER")
    conn.commit()


def upsert_node_output(path: Path, cache_key_hash: str, node_id: str,
                       slot: int, class_type: str, output_type: str,
                       format_tag: str, blob: bytes, meta_json: str) -> None:
    """Incrementally store one node output (provider.on_store hot path)."""
    sha = hashlib.sha256(blob).hexdigest()
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO node_outputs (cache_key_hash, node_id,"
            " slot, class_type, output_type, format, size_bytes, sha256,"
            " meta_json, data) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cache_key_hash, node_id, slot, class_type, output_type,
             format_tag, len(blob), sha, meta_json, sqlite3.Binary(blob)))
        conn.execute("UPDATE snapshots SET updated_at=? WHERE id=?",
                     (int(time.time()), path.stem))
        conn.commit()
    finally:
        conn.close()


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


def list_snapshots() -> List[Dict[str, Any]]:
    """Scan the snapshots directory; broken files are skipped, never raised."""
    out: List[Dict[str, Any]] = []
    for path in sorted(snapshots_root().glob("*.db"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            conn = _connect(path)
            try:
                row = conn.execute(
                    "SELECT id, created_at, prompt_id, note, status,"
                    " format_version, skipped_nodes FROM snapshots"
                    " WHERE id=?", (path.stem,)).fetchone()
                n_nodes = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size_bytes),0) FROM"
                    " node_outputs").fetchone()
            finally:
                conn.close()
            if row is None:
                continue
            out.append({
                "id": row[0], "created_at": row[1], "prompt_id": row[2],
                "note": row[3], "status": row[4],
                "format_version": row[5], "skipped_nodes": row[6] or 0,
                "nodes": n_nodes[0], "size_bytes": n_nodes[1],
                "file": path.name,
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
