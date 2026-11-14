"""ComfyDL VERBOSE diagnostics: low-overhead timestamped event log.

Repair campaign R0 (smallplans §0.90, 2026-11-13).  The user rejected
stopwatch-style manual timing and asked for a VERBOSE log whose timestamps can
be used to compute every duration instead; logs are preferred over one-off
probes.

Design
------
* OFF by default.  ``log()`` / ``ws_event()`` begin with ONE module-level bool
  read: no clock reads, no string building, no queue put, no I/O when disabled.
* A lazy daemon writer drains a bounded queue in batches, so the hot path
  never performs synchronous disk I/O.  ``set_enabled(True)`` creates it once.
* Dual clock: ``time.time()`` (wall, for matching browser events) and
  ``time.perf_counter()`` (monotonic, for durations).
* One line per event::

      [CDL-V] <epoch_seconds> <mono_ms> <level> <stage> <event> k=v ...

* Never raises: a logging bug must not break a prompt (same courtesy rule as
  :mod:`comfy.profiling.proflog`).

Public API (what / in / out)
---------------------------
``enabled()``            -> bool; the cheap gate, also usable by callers.
``set_enabled(b, level)``-> dict; hot toggle, returns :func:`state`.
``state()``              -> dict with enabled / level / log path / queue depth.
``log(stage, event, **f)`` -> None; one line.
``ws_event(event, data)``-> None; central WS hook, also correlates
                            ``executing`` -> ``executed`` into ``node_ms``.
``timed(stage, event)``  -> sync decorator emitting ``duration_ms``.
``wrap_sampler_callback(cb)`` -> callback with the SAME signature that logs
                            one line per KSampler step and forwards the
                            original (latent preview / progress preserved).
``step(stage, event, n, every_n=1, **f)`` -> None; sampled per-step logging.
``ingest(lines)``        -> None; append front-end lines to the same log file.
"""

from __future__ import annotations

import functools
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SWITCH_ID = "comfydl.verbose"
LEVELS = ("off", "low", "medium", "verbose")

_ENABLED = False
_LEVEL = "off"
_LOCK = threading.Lock()
_LOG_PATH: Optional[Path] = None
_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=20000)
_WRITER: Optional[threading.Thread] = None
_PENDING_NODES: Dict[str, float] = {}
_WS_EVENTS = frozenset((
    "execution_start", "executing", "execution_cached", "executed",
    "execution_success", "execution_error", "execution_interrupted",
    "status", "progress",
))


def enabled() -> bool:
    """One global bool read - the OFF path stays allocation-free."""
    return _ENABLED


def _log_dir() -> Path:
    try:
        from folder_paths import get_user_directory

        root = Path(get_user_directory()) / "comfydl" / "logs"
    except Exception:  # noqa: BLE001 - headless / unit-test fallback
        import tempfile

        root = Path(tempfile.gettempdir()) / "comfydl" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _writer_loop() -> None:
    batch: list = []
    while True:
        try:
            batch.append(_QUEUE.get(timeout=0.25))
            while len(batch) < 256:
                try:
                    batch.append(_QUEUE.get_nowait())
                except queue.Empty:
                    break
            if _LOG_PATH is not None:
                with open(_LOG_PATH, "a", encoding="utf-8") as handle:
                    handle.write("\n".join(batch) + "\n")
            batch.clear()
        except queue.Empty:
            continue
        except Exception:  # noqa: BLE001 - logging must never break a run
            batch.clear()


def _ensure_writer() -> None:
    global _WRITER, _LOG_PATH
    with _LOCK:
        if _LOG_PATH is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            _LOG_PATH = _log_dir() / f"verbose-{stamp}.log"
        if _WRITER is None or not _WRITER.is_alive():
            _WRITER = threading.Thread(target=_writer_loop, name="cdl-verbose",
                                       daemon=True)
            _WRITER.start()


def state() -> Dict[str, Any]:
    """What: current switch state. In: none. Out: JSON-able dict."""
    return {"enabled": _ENABLED, "level": _LEVEL,
            "log": str(_LOG_PATH) if _LOG_PATH else None,
            "queued": _QUEUE.qsize()}


def set_enabled(value: bool, level: str = "verbose") -> Dict[str, Any]:
    """What: hot toggle (mirrors into switch_registry when available).

    In: ``value`` - desired state; ``level`` - off/low/medium/verbose.
    Out: :func:`state` dict.  Never raises.
    """
    global _ENABLED, _LEVEL
    _ENABLED = bool(value)
    _LEVEL = (level if _ENABLED and level in LEVELS else
              ("verbose" if _ENABLED else "off"))
    if _ENABLED:
        try:
            _ensure_writer()
        except Exception:  # noqa: BLE001
            pass
    try:
        from comfy.profiling import switch_registry

        switch_registry.register_switch(SWITCH_ID, default=False,
                                        hot_toggle=True, self_meter=False)
        switch_registry.set_enabled(SWITCH_ID, _ENABLED)
    except Exception:  # noqa: BLE001 - registry is optional
        pass
    log("verbose", "switch", enabled=_ENABLED, level=_LEVEL)
    return state()


def _flat(value: Any) -> str:
    try:
        text = str(value)
    except Exception:  # noqa: BLE001
        return "?"
    return (text.replace("\n", "\\n").replace("\r", "\\r")
            .replace(" ", "_")[:200])


def log(stage: str, event: str, **fields: Any) -> None:
    """What: emit one VERBOSE line. In: stage/event + scalar fields.

    Out: None.  OFF path = a single bool read; all clocks/formatting/IO happen
    only after the gate.
    """
    if not _ENABLED:
        return
    try:
        now = time.time()
        mono = time.perf_counter()
        suffix = " ".join(f"{key}={_flat(val)}"
                          for key, val in fields.items())
        line = (f"[CDL-V] {now:.6f} {mono * 1000:.3f} {_LEVEL} "
                f"{stage} {event} {suffix}").rstrip()
        _QUEUE.put_nowait(line)
    except Exception:  # noqa: BLE001 - never break the caller
        pass


def ingest(lines: Any) -> None:
    """What: append already-formatted front-end lines to the same log.

    In: an iterable of strings (bounded by the caller). Out: None.
    """
    if not _ENABLED or not isinstance(lines, (list, tuple)):
        return
    for line in lines[:256]:
        try:
            _QUEUE.put_nowait("[CDL-V-FE] " + _flat(line))
        except Exception:  # noqa: BLE001
            return


def _prompt_of(data: Any) -> str:
    if isinstance(data, dict):
        return str(data.get("prompt_id", ""))
    return ""


def _node_of(data: Any) -> str:
    """Normalize the node id carried by ``executing`` / ``executed``.

    This build sends a dict ``{'node': '1', 'display_node': '1', ...}`` for
    BOTH events (queue idle is ``{'node': None, ...}``); older builds sent a
    bare id.  Stringifying the whole dict used to make the correlation key
    useless, so every ``node_ms`` came out as 0.0.
    """
    if isinstance(data, dict):
        value = data.get("node")
        if value is None:
            value = data.get("display_node", "")
        return "" if value is None else str(value)
    if data is None:
        return ""
    return str(data)


def ws_event(event: str, data: Any = None) -> None:
    """What: single central hook for ``PromptServer.send_sync``.

    In: the WS event name and payload. Out: None.  Besides timestamping the
    seven execution events it correlates ``executing`` with ``executed``
    into ``node_ms``, so R1 can compute per-node durations without touching
    ``execution.py``.  Both bare ids and dict payloads are accepted.
    """
    if not _ENABLED or event not in _WS_EVENTS:
        return
    try:
        if event == "executing":
            node = _node_of(data)
            if node:
                _PENDING_NODES[node] = time.perf_counter()
            log("ws", "executing", node=node)
            return
        if event == "executed":
            detail = data if isinstance(data, dict) else {}
            node = _node_of(detail)
            started = _PENDING_NODES.pop(node, None)
            elapsed = (time.perf_counter() - started) * 1000 if started else 0.0
            log("ws", "executed", node=node, node_ms=round(elapsed, 3),
                prompt_id=_prompt_of(detail))
            return
        if event == "execution_cached":
            detail = data if isinstance(data, dict) else {}
            log("ws", event, nodes=len(detail.get("nodes") or []),
                prompt_id=_prompt_of(detail))
            return
        log("ws", event, prompt_id=_prompt_of(data))
    except Exception:  # noqa: BLE001
        pass


def timed(stage: str, event: str) -> Callable:
    """What: decorator for a sync function, emitting ``duration_ms``.

    In: stage/event labels. Out: a wrapper that is a plain call when OFF.
    """
    def decorate(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _ENABLED:
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                log(stage, event,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3))
        return wrapper
    return decorate


def wrap_sampler_callback(callback: Optional[Callable]) -> Optional[Callable]:
    """What: log one line per KSampler step WITHOUT changing the callback.

    In: the original ``callback(step, x0, x, total_steps)`` (may be None).
    Out: a same-signature wrapper that forwards the original; the original
    object is returned unchanged when the switch is OFF.
    """
    if not _ENABLED:
        return callback

    def wrapped(step: int, x0: Any, x: Any, total_steps: int) -> Any:
        log("sampler", "step", step=step, total=total_steps)
        if callback is not None:
            return callback(step, x0, x, total_steps)
        return None

    return wrapped


def step(stage: str, event: str, n: int, every_n: int = 1,
         **fields: Any) -> None:
    """What: sampled per-step logging for high-frequency loops.

    In: stage/event, monotonically increasing ``n``, sampling period.
    Out: None; emits only when ``n % every_n == 0`` (and the switch is ON).
    """
    if not _ENABLED or (every_n > 1 and n % every_n):
        return
    log(stage, event, n=n, **fields)
