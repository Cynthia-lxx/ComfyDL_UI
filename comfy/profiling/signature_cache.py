"""Unified signature-keyed JSON cache (dual-track P0 shared infrastructure).

Every upcoming signature-keyed cache was going to reinvent the same three
things: a key scheme that stays valid across restarts, a storage location
that survives plugin updates, and a write path that cannot leave a torn file
behind.  This module is those three things, once:

* **Key** (:func:`digest`): SHA-256 over ``(namespace, code_version,
  payload)`` with canonical JSON ordering - deterministic across sessions,
  and automatically invalidated when the code version changes (the lesson
  from the attention-benchmark ``benchmark_db.json`` version field).
* **Storage** (:class:`JsonNamespaceCache`): one JSON file per namespace
  under ``<user directory>/comfydl/cache/`` - the user directory, never the
  plugin directory, so a Manager update cannot wipe history (AMDMonitor
  lesson).  Writes are atomic (``tmp`` + ``os.replace`` - Model-Analytics
  lesson) and reads are mtime-invalidated so external processes editing the
  file are picked up.
* **Self-description**: every file carries ``schema_version``; a mismatch or
  a corrupt file is treated as empty (log + start fresh), never an error in
  the run path.

First planned consumers: the attention backend benchmark db (SP-2), opgraph
probe-result cache (P6/T6-2), compile manager cache.  Until then this module
ships as pure API + tests.

Layering: stdlib-only at import (``folder_paths`` is imported lazily inside
:func:`default_root`), mirroring the proflog rule that early imports must
stay cheap.

Public API
----------
- :func:`code_version` / :func:`set_code_version`
- :func:`digest`
- :func:`default_root` / :func:`set_root`
- :class:`JsonNamespaceCache`
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

_logger = logging.getLogger("ComfyDL.profiling.signature_cache")

_lock = threading.RLock()

#: Overridable code fingerprint folded into every digest.  Defaults to the
#: host build version; the app bootstrap may refine it (e.g. include the
#: comfydl submodule commit) via :func:`set_code_version`.
_code_version: Optional[str] = None

#: Overridable storage root (tests point this at a tempdir).
_root_override: Optional[Path] = None


def set_code_version(version: str) -> None:
    """Refine the fingerprint folded into every :func:`digest`."""
    global _code_version
    with _lock:
        _code_version = str(version)


def code_version() -> str:
    """The current code fingerprint (host version unless overridden)."""
    if _code_version is not None:
        return _code_version
    try:
        from comfyui_version import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - always present in-process
        return "dev"


def digest(namespace: str, payload: Dict[str, Any]) -> str:
    """Deterministic SHA-256 hex key for ``(namespace, code version, payload)``.

    ``payload`` must be JSON-serialisable with sortable keys (plain
    dict/list/str/int/float/bool/None).  Anything else raises ``TypeError``
    - loudly, because a key that silently degrades (``str(tensor)`` with a
    memory address in it) is a cache that never hits.
    """
    try:
        canonical = json.dumps(
            {"ns": namespace, "v": code_version(), "payload": payload},
            sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise TypeError(
            f"signature payload for namespace {namespace!r} is not "
            f"canonically JSON-serialisable: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_root() -> Path:
    """``<user directory>/comfydl/cache`` (tempdir fallback for bare tests)."""
    if _root_override is not None:
        return _root_override
    try:
        import folder_paths

        base = Path(folder_paths.get_user_directory())
    except Exception:  # pragma: no cover - headless/test contexts
        base = Path(tempfile.gettempdir()) / "comfydl"
    return base / "cache"


def set_root(path: Union[str, Path, None]) -> None:
    """Point the storage root elsewhere (test isolation only; ``None``
    restores the default user-directory location)."""
    global _root_override
    with _lock:
        _root_override = Path(path) if path is not None else None


class JsonNamespaceCache:
    """One namespaced JSON file: ``{"schema_version": N, "entries": {...}}``.

    Thread-safe within the process; external edits are picked up through
    mtime comparison on read.  A corrupt / wrong-version file logs a warning
    and behaves as empty rather than raising - caches are a courtesy, never
    a contract (same rule as proflog).
    """

    def __init__(self, namespace: str, *, schema_version: int = 1) -> None:
        if not namespace or any(c in namespace for c in "\\/:*?\"<>|"):
            raise ValueError(
                f"invalid cache namespace {namespace!r} (keep it a plain "
                f"file-name-safe token)")
        self.namespace = namespace
        self.schema_version = int(schema_version)
        self._mtime: Optional[float] = None
        self._entries: Dict[str, Any] = {}

    # -- location ----------------------------------------------------------

    @property
    def path(self) -> Path:
        return default_root() / f"{self.namespace}.json"

    # -- read ---------------------------------------------------------------

    def _load_if_stale(self) -> None:
        """(Re)read the file when its mtime moved; never raises."""
        path = self.path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            with _lock:
                self._mtime = None
                self._entries = {}
            return
        with _lock:
            if self._mtime == mtime:
                return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            version = raw.get("schema_version")
            entries = raw.get("entries")
            if version != self.schema_version or not isinstance(entries, dict):
                raise ValueError(
                    f"schema_version mismatch or bad entries "
                    f"(file={version!r}, want={self.schema_version!r})")
            with _lock:
                self._entries = entries
                self._mtime = mtime
        except Exception as exc:  # noqa: BLE001 - corrupt cache == empty cache
            _logger.warning(
                "signature cache %s unreadable (%s); starting fresh",
                path.name, exc)
            with _lock:
                self._entries = {}
                self._mtime = mtime

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """The cached value for ``key`` or ``None`` (miss / stale / corrupt)."""
        self._load_if_stale()
        with _lock:
            value = self._entries.get(key)
        return value if isinstance(value, dict) else None

    # -- write --------------------------------------------------------------

    def put(self, key: str, value: Dict[str, Any]) -> None:
        """Store ``value`` under ``key``, preserving other entries.

        Atomic write (``tmp`` in the same directory + ``os.replace``) so a
        crash mid-write can never tear the file.  ``value`` must be a dict of
        JSON-serialisable content - same fail-loudly rule as :func:`digest`.
        """
        if not isinstance(value, dict):
            raise TypeError("cache values must be dicts of JSON content")
        try:
            json.dumps(value, sort_keys=True)
        except TypeError as exc:
            raise TypeError(
                f"cache value for {key!r} is not JSON-serialisable: {exc}"
            ) from exc
        self._load_if_stale()
        path = self.path
        with _lock:
            self._entries[key] = value
            payload = {
                "schema_version": self.schema_version,
                "entries": self._entries,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{self.namespace}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        # adopt the fresh mtime so the next read does not reload needlessly
        try:
            with _lock:
                self._mtime = path.stat().st_mtime
        except OSError:  # pragma: no cover - file we just wrote
            pass

    # -- introspection -------------------------------------------------------

    def __len__(self) -> int:
        self._load_if_stale()
        with _lock:
            return len(self._entries)

    def keys(self):
        self._load_if_stale()
        with _lock:
            return list(self._entries)
