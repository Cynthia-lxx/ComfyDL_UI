"""Execution-time resource watchdog of the ComfyDL profiling tools (M2).

A single daemon thread samples, every ``interval_s`` (default 250 ms):

* the ComfyUI process's own CPU percent (psutil, *per-core scale*: one
  fully-busy core reads ~100, so ``cpu_count`` busy cores read
  ``100 * cpu_count``) and RSS;
* the system-wide CPU percent (psutil, normalized 0-100).

Every sample is attributed to the node currently executing by reading
``server.last_node_id`` - the same anchor the ``executing`` WebSocket event
maintains - so no extra event plumbing is needed.

When the process CPU holds at or above ``cpu_threshold`` (per-core scale)
for ``sustain_s``, a *burst event* opens; it closes (and is appended to
``user/comfydl/profiling_watchdog.jsonl``) when the load falls back below
the hysteresis factor for two consecutive ticks, or when the executing
node changes. System-wide saturation (the whole machine stalled, not just
this process) is flagged in the event - that is the "everything else
became unusable" case the watchdog was asked to document.

Design rules (mirrors the M1 philosophy):

* the watchdog may never break the server: every tick is wrapped, every
  failure is logged at most once and swallowed, stop() joins the thread;
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
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
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
        """Start the sampling thread (idempotent; a no-op without psutil)."""
        if not PSUTIL_OK:
            print("[ComfyDL watchdog] psutil unavailable - execution monitoring disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="comfydl-profiling-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Join the sampling thread; safe to call more than once."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        self._thread = None

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
                "running": self._thread is not None and self._thread.is_alive(),
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
    def _run(self) -> None:
        assert self._process is not None
        try:  # prime both counters: the first cpu_percent() call returns 0
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            pass
        while not self._stop_event.wait(self._interval):
            try:
                self._tick()
            except Exception:  # never let a sampling error kill the thread
                pass

    def _tick(self) -> None:
        now = time.monotonic()
        sample = {
            "ts_epoch": time.time(),
            "process_cpu_percent": round(float(self._process.cpu_percent(None)), 1),
            "system_cpu_percent": round(float(psutil.cpu_percent(None)), 1),
            "process_rss_mb": round(
                self._process.memory_info().rss / (1024 * 1024), 1
            ),
            "node_id": getattr(self._server, "last_node_id", None),
            "prompt_id": getattr(self._server, "last_prompt_id", None),
        }
        with self._lock:
            self._latest = sample
            self._evaluate(now, sample)

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
