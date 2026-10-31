"""Execution-time resource watchdog of the ComfyDL profiling tools (M2).

Dual-track P0 (2026-10-31): the sampling thread is gone - ticks now arrive
from the shared observation spine (``app/spine``), which samples every tick
the same facts this module used to sample itself:

* the ComfyUI process's own CPU percent (psutil, *per-core scale*: one
  fully-busy core reads ~100, so ``cpu_count`` busy cores read
  ``100 * cpu_count``) and RSS;
* the system-wide CPU percent (psutil, normalized 0-100).

The burst state machine, the JSONL ledger, the snapshot/recent reads and
the ``node_id`` attribution (reading ``server.last_node_id`` in the tick
callback - the same anchor the ``executing`` WebSocket event maintains) all
survive unchanged; only the thread ownership moved, so one shared daemon
now serves every observer instead of one per feature.

When the process CPU holds at or above ``cpu_threshold`` (per-core scale)
for ``sustain_s``, a *burst event* opens; it closes (and is appended to
``user/comfydl/profiling_watchdog.jsonl``) when the load falls back below
the hysteresis factor for two consecutive ticks, or when the executing
node changes. System-wide saturation (the whole machine stalled, not just
this process) is flagged in the event - that is the "everything else
became unusable" case the watchdog was asked to document.

Design rules (mirrors the M1 philosophy):

* the watchdog may never break the server: every tick is wrapped, every
  failure is logged at most once and swallowed, stop() detaches cleanly;
* psutil missing -> the watchdog silently disables itself;
* the JSONL log is append-only and survives restarts; if it cannot be
  written, events fall back to an in-memory ring buffer and the counter
  ``events_dropped`` says so;
* a *sampled* fact is recorded, never a guess: no throughput or time
  prediction happens here (that is M3's calibrated work).

Settings can be overridden per environment: ``COMFYDL_WATCHDOG_INTERVAL``,
``COMFYDL_WATCHDOG_CPU``, ``COMFYDL_WATCHDOG_SUSTAIN``, ``COMFYDL_WATCHDOG_LOG``.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil

    PSUTIL_OK = True
except ImportError:  # dehydrated environments may lack it
    psutil = None  # type: ignore[assignment]
    PSUTIL_OK = False

DEFAULT_INTERVAL_S = 0.25
DEFAULT_CPU_THRESHOLD = 90.0
DEFAULT_SUSTAIN_S = 2.0
DEFAULT_LOG_PATH = os.path.join("user", "comfydl", "profiling_watchdog.jsonl")

#: A burst ends only after the load falls below threshold * END_FACTOR for
#: two consecutive ticks (prevents flicker at the boundary).
END_FACTOR = 0.8
END_TICKS = 2

#: In-memory fallback ring for events the JSONL log refused.
_RING_SIZE = 64

#: How much of the log tail `recent()` may read in one go (bytes).
_RECENT_TAIL_BYTES = 512 * 1024


def _env_float(name: str, default: float, low: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(low, float(raw))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a spine snapshot field (possibly an error dict) to a number."""
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return default


class ProfilingWatchdog:
    """The M2 execution-time watchdog (see the module docstring)."""

    def __init__(
        self,
        server: Any,
        *,
        interval_s: Optional[float] = None,
        cpu_threshold: Optional[float] = None,
        sustain_s: Optional[float] = None,
        log_path: Optional[str] = None,
        scale_hint_fn: Optional[Callable[[str], Optional[dict]]] = None,
    ) -> None:
        self._server = server
        self._interval = _env_float(
            "COMFYDL_WATCHDOG_INTERVAL",
            DEFAULT_INTERVAL_S if interval_s is None else interval_s,
            low=0.05,
        )
        self._threshold = _env_float(
            "COMFYDL_WATCHDOG_CPU",
            DEFAULT_CPU_THRESHOLD if cpu_threshold is None else cpu_threshold,
            low=5.0,
        )
        self._sustain = _env_float(
            "COMFYDL_WATCHDOG_SUSTAIN",
            DEFAULT_SUSTAIN_S if sustain_s is None else sustain_s,
            low=0.1,
        )
        self._log_path = log_path or os.environ.get("COMFYDL_WATCHDOG_LOG") or DEFAULT_LOG_PATH
        self._scale_hint_fn = scale_hint_fn
        self._cores = os.cpu_count() or 1
        self._process = psutil.Process() if PSUTIL_OK else None
        self._thread: Optional[threading.Thread] = None  # legacy (W3b reads it)
        self._running = False
        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._events_written = 0
        self._events_dropped = 0
        self._ring: List[dict] = []
        self._warned_log = False
        # Burst state (guarded by self._lock, mutated by _evaluate):
        self._hot_since: Optional[float] = None
        self._burst: Optional[dict] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Attach to the observation spine (idempotent; a no-op without
        psutil).  The spine switch is hot-enabled and its rate lowered to
        at most this watchdog's interval - the shared thread then ticks
        :meth:`_on_spine_tick`."""
        if not PSUTIL_OK:
            print("[ComfyDL watchdog] psutil unavailable - execution monitoring disabled")
            return
        if self._running:
            return
        from app.spine import SWITCH_ID as _spine_switch, get_spine
        from comfy.profiling import switch_registry as _switch_registry

        spine = get_spine()
        if not _switch_registry.set_enabled(_spine_switch, True):
            print("[ComfyDL watchdog] spine hard-disabled - cannot attach")
            return
        spine.add_listener(self._on_spine_tick)
        if spine.rate_s <= 0 or spine.rate_s > self._interval:
            spine.set_rate(self._interval)
        spine.start()
        self._running = True

    def stop(self, timeout_s: float = 5.0) -> None:
        """Detach from the spine (signature kept for call-site compatibility;
        ``timeout_s`` is accepted and ignored - there is no thread to join).
        The shared thread itself is stopped when this was its last listener."""
        try:
            from app.spine import get_spine

            spine = get_spine()
            spine.remove_listener(self._on_spine_tick)
            if spine.listener_count() == 0:
                spine.stop()
        except Exception:  # noqa: BLE001 - stop() must always succeed
            pass
        self._running = False

    # -- public reads -------------------------------------------------------
    def snapshot(self) -> dict:
        """The latest sample plus configuration; ``available: False`` without
        psutil, ``running: False`` when the thread is not alive."""
        if not PSUTIL_OK:
            return {"available": False, "running": False}
        with self._lock:
            latest = dict(self._latest)
        latest.setdefault("node_id", None)
        latest.setdefault("prompt_id", None)
        latest.update(
            {
                "available": True,
                "running": self._running,
                "cores": self._cores,
                "cpu_threshold": self._threshold,
                "sustain_s": self._sustain,
                "interval_s": self._interval,
                "events_written": self._events_written,
                "events_dropped": self._events_dropped,
            }
        )
        return latest

    def recent(self, limit: int = 50) -> List[dict]:
        """The last ``limit`` burst events, newest last, from the JSONL tail."""
        limit = max(1, int(limit))
        lines: List[str] = []
        try:
            size = os.path.getsize(self._log_path)
            with open(self._log_path, "r", encoding="utf-8") as handle:
                if size > _RECENT_TAIL_BYTES:
                    handle.seek(size - _RECENT_TAIL_BYTES)
                    handle.readline()  # drop the partial line
                lines = handle.readlines()
        except OSError:
            pass
        events: List[dict] = []
        for line in lines[-limit * 2 :]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        if len(events) < limit:
            with self._lock:
                events.extend(self._ring)
        return events[-limit:]

    # -- internals ----------------------------------------------------------
    def _on_spine_tick(self, snap) -> None:
        """Spine listener: one observation tick -> one attributed sample.

        Field sources mirror the old ``_tick`` exactly: process CPU (per-core
        scale) and RSS from the spine's ``proc`` section, system CPU from
        ``cpu.percent``; the node/prompt attribution is still read here, at
        tick time, from the server attributes the executor maintains.
        Provider errors degrade to 0.0 rather than raising.
        """
        try:
            now = time.monotonic()
            sample = {
                "ts_epoch": snap.ts,
                "process_cpu_percent": _as_float(snap.proc.get("cpu_percent")),
                "system_cpu_percent": _as_float(snap.cpu.get("percent")),
                "process_rss_mb": _as_float(snap.proc.get("rss_mb")),
                "node_id": getattr(self._server, "last_node_id", None),
                "prompt_id": getattr(self._server, "last_prompt_id", None),
            }
            with self._lock:
                self._latest = sample
                self._evaluate(now, sample)
        except Exception:  # never let a listener error reach the spine thread
            pass

    def _evaluate(self, now: float, sample: Dict[str, Any]) -> None:
        """The pure burst-state machine, isolated so tests can drive it."""
        hot = (
            sample["process_cpu_percent"] >= self._threshold
            or sample["system_cpu_percent"] >= self._threshold
        )
        if self._burst is None:
            if not hot:
                self._hot_since = None
                return
            if self._hot_since is None:
                self._hot_since = now
            if now - self._hot_since < self._sustain:
                return
            self._burst = {
                "started": now,
                "node_id": sample["node_id"],
                "prompt_id": sample["prompt_id"],
                "cpu_peak": sample["process_cpu_percent"],
                "cpu_sum": sample["process_cpu_percent"],
                "cpu_samples": 1,
                "system_peak": sample["system_cpu_percent"],
                "system_wide": sample["system_cpu_percent"] >= self._threshold,
                "rss_mb": sample["process_rss_mb"],
                "low_ticks": 0,
            }
            return

        node_changed = sample["node_id"] != self._burst["node_id"]
        if node_changed or not hot and self._burst["low_ticks"] >= END_TICKS:
            self._close_burst(now)
            if hot:
                self._hot_since = now
            return

        if hot:
            self._burst["low_ticks"] = 0
            self._burst["cpu_peak"] = max(
                self._burst["cpu_peak"], sample["process_cpu_percent"]
            )
            self._burst["cpu_sum"] += sample["process_cpu_percent"]
            self._burst["cpu_samples"] += 1
            self._burst["system_peak"] = max(
                self._burst["system_peak"], sample["system_cpu_percent"]
            )
            self._burst["system_wide"] = self._burst["system_wide"] or (
                sample["system_cpu_percent"] >= self._threshold
            )
            self._burst["rss_mb"] = max(self._burst["rss_mb"], sample["process_rss_mb"])
        else:
            self._burst["low_ticks"] += 1

    def _close_burst(self, now: float) -> None:
        burst, self._burst = self._burst, None
        self._hot_since = None
        if burst is None:
            return
        scale_hint = None
        if self._scale_hint_fn is not None and burst["node_id"] is not None:
            try:
                scale_hint = self._scale_hint_fn(burst["node_id"])
            except Exception:
                scale_hint = None
        event = {
            "ts": datetime.datetime.fromtimestamp(burst["started"]).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(now - burst["started"], 2),
            "node_id": burst["node_id"],
            "prompt_id": burst["prompt_id"],
            "cpu_percent_peak": burst["cpu_peak"],
            "cpu_percent_avg": round(
                burst["cpu_sum"] / max(1, burst["cpu_samples"]), 1
            ),
            "system_cpu_percent_peak": burst["system_peak"],
            "system_wide": burst["system_wide"],
            "process_rss_mb": burst["rss_mb"],
            "threshold": self._threshold,
            "cores": self._cores,
            "data_scale": scale_hint,
        }
        self._append(event)

    def _append(self, event: dict) -> None:
        try:
            directory = os.path.dirname(self._log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
            self._events_written += 1
        except OSError:
            self._events_dropped += 1
            self._ring.append(event)
            del self._ring[:-_RING_SIZE]
            if not self._warned_log:
                self._warned_log = True
                print(
                    f"[ComfyDL watchdog] cannot write {self._log_path}; "
                    "burst events stay in memory"
                )
