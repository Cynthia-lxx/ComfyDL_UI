"""SnapshotProvider: the crash site's CacheProvider (bigplan ADR-2/ADR-6).

Registered once at server startup; the host then drives everything:

* ``on_store``  - fired after every ``caches.outputs.set``; we serialize the
  slot output OFF the event loop (``run_in_executor``) and append it to the
  prompt's snapshot database.  Incremental capture: by the time a run fails,
  is interrupted, or completes, everything finished is already on disk - no
  rescue window left to race against.
* ``on_lookup`` - fired on a local cache miss; we look the input-signature
  hash up across snapshot databases and rehydrate the stored object, so a
  re-queued prompt skips every node whose output survived.
* ``should_cache`` - rejects what cannot round-trip (ModelPatcher,
  DataLoaders) before it ever reaches the store.
* ``on_prompt_start/end`` - opens/closes the per-prompt snapshot database.

Bridging note: the lifecycle hooks receive the prompt_id while on_store only
sees the CacheContext.  Execution is single-worker serial (main.py prompt
worker), so a single "active snapshot" pointer is sufficient and safe.

Keys are the host's input-signature SHA256 (``cache_key_hash``), NOT the
prompt_id - editing unrelated parts of the graph keeps untouched nodes
hittable, which is exactly the recovery/tamper semantics we want.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import traceback
from typing import Any, Dict, Optional

from comfy_api.latest._caching import CacheContext, CacheProvider, CacheValue

from app.crashsite import serialize, store
from comfy.profiling.proflog import log as proflog

_logger = logging.getLogger(__name__)


class SnapshotProvider(CacheProvider):
    """Incremental execution snapshotting (see module docstring)."""

    def __init__(self, enabled: bool = True, queue=None):
        self._enabled = enabled
        self._queue = queue  # PromptQueue: on_prompt_start grabs the prompt
        self._active: Optional[Any] = None  # active snapshot db path
        self._active_prompt_id: str = ""
        self._skipped = 0  # unrescuable node count for the active prompt
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------- prompt lifecycle ----
    def on_prompt_start(self, prompt_id: str) -> None:
        if not self._enabled:
            return
        try:
            prompt = self._grab_prompt(prompt_id)
            path = store.create_snapshot(prompt=prompt, prompt_id=prompt_id)
            with self._lock:
                self._active = path
                self._active_prompt_id = prompt_id
                self._skipped = 0
            _logger.info("crash site: snapshot %s opened (prompt %s)",
                         path.stem, prompt_id)
        except Exception as exc:  # noqa: BLE001 - capture must never break runs
            _logger.warning("crash site: failed to open snapshot (%s)", exc)

    def _grab_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """The prompt JSON from the running queue item (host gives us only
        the prompt_id, but the resume flow needs the original prompt)."""
        if self._queue is None:
            return {}
        try:
            for item in list(self._queue.currently_running.values()):
                if item[1] == prompt_id:
                    return item[2] if isinstance(item[2], dict) else {}
        except Exception:  # noqa: BLE001
            pass
        return {}

    def on_prompt_end(self, prompt_id: str) -> None:
        with self._lock:
            path = self._active
            self._active = None
            self._active_prompt_id = ""
            skipped = self._skipped
            self._skipped = 0
        if path is not None:
            try:
                store.mark_status(path, "completed", skipped=skipped)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("crash site: failed to close snapshot (%s)", exc)

    # ------------------------------------------------------------- gating --
    def should_cache(self, context: CacheContext,
                     value: Optional[CacheValue] = None) -> bool:
        if not self._enabled:
            return False
        if value is None:  # lookup phase: always let the db be asked
            return True
        try:
            ok = not serialize.has_exotic_leaves(value.outputs)
            if not ok:
                with self._lock:
                    self._skipped += 1
                proflog(
                    "medium", "crash site: node %s (%s) skipped - unrescuable"
                    " output, resume will re-execute it",
                    context.node_id, context.class_type, request=None)
            return ok
        except Exception:  # noqa: BLE001 - never break execution
            return False

    # ------------------------------------------------------ store/lookup ---
    async def on_store(self, context: CacheContext,
                       value: CacheValue) -> None:
        with self._lock:
            path = self._active
        if path is None:
            return  # no active snapshot (missed start / disabled): skip
        loop = asyncio.get_running_loop()
        # GB-scale serialization must happen OFF the event loop.
        await loop.run_in_executor(
            None, self._store_sync, path, context, value)

    async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
        if not self._enabled:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._lookup_sync, context)

    # ----------------------------------------------------------- internals -
    def _store_sync(self, path, context: CacheContext,
                    value: CacheValue) -> None:
        try:
            for slot, obj in enumerate(value.outputs or []):
                if serialize.has_exotic_leaves(obj):
                    continue
                format_tag, blob, meta_json = serialize.serialize_output(obj)
                store.upsert_node_output(
                    path, cache_key_hash=context.cache_key_hash,
                    node_id=context.node_id, slot=slot,
                    class_type=context.class_type,
                    output_type=type(obj).__name__,
                    format_tag=format_tag, blob=blob, meta_json=meta_json)
                proflog(
                    "medium", "crash site: captured %s (%s) slot %s -> %s,"
                    " %d bytes", context.node_id, context.class_type, slot,
                    format_tag, len(blob), request=None)
        except serialize.UnsupportedOutputError as exc:
            # Should have been caught by should_cache; belt and braces.
            _logger.warning("crash site: node %s output unrescuable (%s)",
                            context.node_id, exc)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("crash site: on_store failed for node %s (%s)",
                            context.node_id, exc)

    def _lookup_sync(self, context: CacheContext) -> Optional[CacheValue]:
        try:
            # Scan every snapshot db for this hash (M1: stateless scan; the
            # library count is small and this keeps resume flow simple).
            # Snapshots with a foreign format version are skipped: mixed-
            # version rows would rehydrate garbage (the "got str" incident).
            for path in store.snapshots_root().glob("*.db"):
                if store.format_version_of(path) != store.FORMAT_VERSION:
                    continue
                rows = store.get_outputs_by_hash(path, context.cache_key_hash)
                if not rows:
                    continue
                max_slot = max(r["slot"] for r in rows)
                outputs: list = [None] * (max_slot + 1)
                for row in rows:
                    obj = serialize.deserialize_output(
                        row["format"], row["data"], row["meta_json"])
                    outputs[row["slot"]] = obj
                proflog(
                    "medium", "crash site: rehydrated node %s (%s) from"
                    " snapshot (%d slot(s), formats=%s)",
                    context.node_id, context.class_type, len(rows),
                    [r["format"] for r in rows], request=None)
                return CacheValue(outputs=outputs, ui={})
        except Exception as exc:  # noqa: BLE001
            import traceback

            # Full trace once per failure - "list indices" style bugs are
            # impossible to place from the message alone (2026-10-09).
            _logger.warning("crash site: on_lookup failed for node %s\n%s",
                            context.node_id, traceback.format_exc())
        return None
