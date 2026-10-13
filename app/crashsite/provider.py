"""SnapshotProvider: the crash site's CacheProvider (bigplan ADR-2/ADR-6).

Registered once at server startup; the host then drives everything:

* ``on_store``  - retained for direct/manual use, but NOT driven during
  normal runs any more (lazy capture, git-fire perf fix 2026-10-13: the
  real-time serialize+write per node completion slowed runs 2-10x).
* ``on_scene_rescue`` - THE capture path now: the host calls it once per
  surviving cache entry when a run dies mid-flight (interrupt/error), so
  the scene is written while the process and cache are still alive.  A
  hard process kill is the one scenario lazy capture cannot cover.
* ``on_lookup`` - fired on a local cache miss, but only for RESUMED
  prompts (lazy gating) - normal runs never scan the snapshot library.
* ``should_cache`` - rejects what cannot round-trip (ModelPatcher,
  DataLoaders) before it ever reaches the store.
* ``on_prompt_start/end`` - opens/closes the per-prompt snapshot database.
* ``on_prompt_error`` - optional host "error" lifecycle event; marks the
  snapshot "interrupted" instead of "completed" when the run died mid-flight.
* ``on_node_cached`` - optional host notification when a node is served from
  a cache (local RAM hit or rehydration both funnel there) - the
  authoritative skip signal behind the M2 resume statistics note.

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
        self._cached_nodes: Dict[str, str] = {}  # node_id -> class_type
        self._rehydrated: set = set()  # node_ids served from snapshot dbs
        self._resume_source: Optional[str] = None  # snapshot id we resumed
        self._pending_resume: Optional[Dict[str, str]] = None
        # Lazy capture (git-fire perf fix, 2026-10-13): snapshot-db lookups
        # are only answered for RESUMED prompts - normal runs never scan the
        # library, which used to crawl first-run start-up as it grew.
        self._lookup_active = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def mark_resume(self, prompt_id: str, source_snapshot_id: str) -> None:
        """Flag the next prompt start as a resume of ``source_snapshot_id``.

        Called by the resume route BEFORE re-queueing (execution is single-
        worker serial, so a pending pointer is safe - same model as
        on_store's active pointer).
        """
        with self._lock:
            self._pending_resume = {"prompt_id": prompt_id,
                                    "source": source_snapshot_id}

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
                self._cached_nodes = {}
                self._rehydrated = set()
                pending = self._pending_resume
                self._resume_source = (
                    pending["source"]
                    if pending is not None and pending["prompt_id"] == prompt_id
                    else None)
                self._pending_resume = None
                self._lookup_active = self._resume_source is not None
            _logger.info("crash site: snapshot %s opened (prompt %s)%s",
                         path.stem, prompt_id,
                         " [resume]" if self._resume_source else "")
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
            cached_nodes = self._cached_nodes
            rehydrated = set(self._rehydrated)
            resume_source = self._resume_source
            self._cached_nodes = {}
            self._rehydrated = set()
            self._resume_source = None
            self._lookup_active = False
        if path is not None:
            try:
                store.mark_status(path, "completed", skipped=skipped)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("crash site: failed to close snapshot (%s)", exc)
        if path is not None and resume_source is not None:
            self._write_resume_note(path, resume_source, cached_nodes,
                                    rehydrated)

    def on_prompt_error(self, prompt_id: str) -> None:
        """The run died mid-flight (node error or user interrupt).

        Host-driven via the optional "error" lifecycle event (execution.py
        notifies this only when the prompt actually failed).  Marks the
        active snapshot "interrupted" instead of letting the subsequent
        on_prompt_end file it under "completed" - a crashed scene labelled
        as a finished run misled users during the 2026-10-09 resume
        debugging.  on_prompt_end afterwards becomes a no-op (path None).
        """
        with self._lock:
            if self._active is None or self._active_prompt_id != prompt_id:
                return  # not ours (missed start / late event): ignore
            path = self._active
            skipped = self._skipped
            self._active = None
            self._active_prompt_id = ""
            self._skipped = 0
            cached_nodes = self._cached_nodes
            rehydrated = set(self._rehydrated)
            resume_source = self._resume_source
            self._cached_nodes = {}
            self._rehydrated = set()
            self._resume_source = None
            self._lookup_active = False
        try:
            store.mark_status(path, "interrupted", skipped=skipped)
        except Exception as exc:  # noqa: BLE001 - never break execution
            _logger.warning("crash site: failed to mark snapshot interrupted"
                            " (%s)", exc)
        if resume_source is not None:
            self._write_resume_note(path, resume_source, cached_nodes,
                                    rehydrated)

    def on_node_cached(self, prompt_id: str, node_id: str,
                       class_type: str) -> None:
        """A node was SKIPPED this run (host cached-branch notification).

        The authoritative skip signal for resume statistics: fires for both
        local-RAM cache hits and provider-rehydrated entries, since both
        funnel through execution.py's cached branch.  Stray events for
        non-active prompts are ignored (same pattern as on_prompt_error).
        """
        with self._lock:
            if self._active is None or self._active_prompt_id != prompt_id:
                return
            self._cached_nodes[node_id] = class_type

    def _write_resume_note(self, path, source: str,
                           cached_nodes: Dict[str, str],
                           rehydrated: set) -> None:
        """Write the authoritative skip/re-run statistics onto the snapshot.

        What: summarises one resumed run - how many of the prompt's nodes
              were served from cache (split snapshot-rehydration vs local
              RAM cache) and how many actually re-executed.
        In:   ``path`` — the resumed run's snapshot db; ``source`` — the
              snapshot id it was resumed from; ``cached_nodes`` /
              ``rehydrated`` — the per-run counters.
        """
        try:
            manifest = store.load_manifest(path)
            total = 0
            if manifest is not None:
                prompt = manifest.get("prompt")
                if isinstance(prompt, dict):
                    total = len(prompt)
            skipped = len(cached_nodes)
            from_snap = len([n for n in rehydrated if n in cached_nodes])
            local = skipped - from_snap
            reran = max(total - skipped, 0)
            if total > 0:
                note = (f"resume of {source}: skipped {skipped}/{total}"
                        f" ({from_snap} snapshot + {local} local cache),"
                        f" re-executed {reran}")
            else:
                note = (f"resume of {source}: {skipped} node(s) served from"
                        f" cache")
            store.set_note(path, note)
            proflog("medium", "crash site: resume note written - %s", note,
                    request=None)
        except Exception as exc:  # noqa: BLE001 - cosmetic, never raise
            _logger.warning("crash site: failed to write resume note (%s)",
                            exc)

    # ------------------------------------------------------------- gating --
    def should_cache(self, context: CacheContext,
                     value: Optional[CacheValue] = None) -> bool:
        """Lazy-capture gate (git-fire perf fix, 2026-10-13).

        Lookup phase (``value`` None): answer only while the active prompt
        is a RESUME - normal runs never scan the snapshot library, which is
        what made every first-run node and prompt start-up crawl as the
        library grew.

        Store phase (``value`` set): NEVER store in real time.  Serializing
        every node completion (CPU clone + safetensors + sqlite) ran
        concurrently with execution and slowed runs 2-10x.  Actual capture
        now happens once, at failure time, via :meth:`on_scene_rescue`.
        The exotic-leaf walk stays only to keep the card's skipped counter
        honest (pure type checks, no tensor work).
        """
        if not self._enabled:
            return False
        if value is None:
            with self._lock:
                return self._lookup_active and self._active is not None
        try:
            if serialize.has_exotic_leaves(value.outputs):
                with self._lock:
                    self._skipped += 1
        except Exception:  # noqa: BLE001 - never break execution
            pass
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
        with self._lock:
            active = self._lookup_active and self._active is not None
        if not active:
            return None  # lazy capture: only resumed prompts scan the library
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._lookup_sync, context)

    async def on_scene_rescue(self, prompt_id: str, context: CacheContext,
                              value: CacheValue) -> None:
        """Lazy capture: serialize one surviving cache entry at failure time.

        Host-driven (BasicCache.rescue_scene) ONLY on the interrupt/error
        path, after the run died, while the output cache is still warm.
        Reuses the regular row writer; exotic slots are skipped as usual.
        Stray events for non-active prompts are ignored.
        """
        with self._lock:
            path = self._active
            mine = self._active_prompt_id == prompt_id
        if path is None or not mine:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._store_sync, path, context, value)
        proflog("medium", "crash site: rescued node %s (%s) from cache",
                context.node_id, context.class_type, request=None)

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
                with self._lock:
                    if self._active is not None:
                        self._rehydrated.add(context.node_id)
                return CacheValue(outputs=outputs, ui={})
        except Exception as exc:  # noqa: BLE001
            import traceback

            # Full trace once per failure - "list indices" style bugs are
            # impossible to place from the message alone (2026-10-09).
            _logger.warning("crash site: on_lookup failed for node %s\n%s",
                            context.node_id, traceback.format_exc())
        return None
