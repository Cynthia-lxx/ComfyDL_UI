"""Spine bus - the single observation thread and WS push channel.

Structure (Crystools's CMonitor thread lifecycle + XPUSYS's atomic-swap
snapshot discipline, minus the parts the survey rejected):

* **One** daemon thread per process, shared by every observer.
* ``rate_s`` is the only knob: ``> 0`` keeps the loop alive at that
  interval, ``0`` (or negative) stops it and the thread exits - "off"
  means *nothing* runs, not a sleeping loop (Crystools rate-0 lesson).
* The thread's own cost is metered into the switch registry
  (:func:`switch_registry.record_self_overhead`) per the meta-profiling
  rule - the settings panel can show exactly what the spine itself costs.
* WS push goes through ``PromptServer.instance.send_sync`` on the single
  ``cdl.spine`` channel; headless (no server) the bus still samples for
  in-process consumers like the watchdog.
* Listeners are invoked inline in the bus thread and must be cheap
  (the watchdog's in-memory burst bookkeeping is the reference workload).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, List, Optional

from app.spine.providers import default_providers
from app.spine.snapshot import CdlSnapshot
from comfy.profiling import switch_registry

_logger = logging.getLogger("ComfyDL.spine")

#: Registry id (also the Tier-1 self-metering account).
SWITCH_ID = "comfydl.spine"
#: The one and only WS message name the spine ever broadcasts.
WS_MESSAGE = "cdl.spine"

#: Floors: rates below this are clamped (a 0.05s poll would itself become
#: measurable load); anything <= 0 means "stopped".
_MIN_RATE_S = 0.05
_DEFAULT_RATE_S = 1.0

switch_registry.register_switch(
    SWITCH_ID,
    default=False,      # off until a consumer (watchdog, panel) starts it
    hot_toggle=True,
    self_meter=True,
)


class Spine:
    """Process-wide observation bus (use :func:`app.spine.get_spine`)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()  # settings changed; re-read, don't die
        self._thread: Optional[threading.Thread] = None
        self._snapshot: Optional[CdlSnapshot] = None
        self._rate_s = _DEFAULT_RATE_S
        self._providers = default_providers()
        self._listeners: List[Callable[[CdlSnapshot], None]] = []
        self._sent = 0

    # -- lifecycle -----------------------------------------------------------

    def attach_server(self, server) -> None:
        """Bind the PromptServer (or any ``send_sync`` provider) for pushes."""
        with self._lock:
            self._server = server

    @property
    def rate_s(self) -> float:
        with self._lock:
            return self._rate_s

    def set_rate(self, seconds: float) -> float:
        """Set the sampling interval; ``<= 0`` stops the thread.

        Returns the effective rate (after clamping) - 0.0 when stopped.
        """
        seconds = float(seconds)
        if seconds <= 0:
            self.stop()
            with self._lock:
                self._rate_s = 0.0
            return 0.0
        clamped = max(seconds, _MIN_RATE_S)
        with self._lock:
            self._rate_s = clamped
        self._wake.set()  # running loop re-reads the rate promptly
        return clamped

    def start(self) -> bool:
        """Start sampling (no-op when already running or switch-disabled).

        Returns whether the bus is running after the call.
        """
        if not switch_registry.is_enabled(SWITCH_ID):
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._rate_s <= 0:
                    self._rate_s = _DEFAULT_RATE_S
                return True
            self._stop.clear()
            if self._rate_s <= 0:
                self._rate_s = _DEFAULT_RATE_S
            self._thread = threading.Thread(
                target=self._loop, name="cdl-spine", daemon=True)
            self._thread.start()
            return True

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop sampling and join the thread (exits within one interval).

        Joining keeps ``is_running`` truthful the moment ``stop`` returns,
        which the control route's response contract relies on; the loop
        wakes from its interval wait immediately, so the join is short.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # -- consumption ----------------------------------------------------------

    def snapshot(self) -> Optional[CdlSnapshot]:
        """The latest complete snapshot (``None`` before the first tick)."""
        with self._lock:
            return self._snapshot

    def add_listener(self, callback: Callable[[CdlSnapshot], None]) -> None:
        """Register an inline consumer (invoked in the bus thread, be cheap)."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[CdlSnapshot], None]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def listener_count(self) -> int:
        """How many consumers are attached (the last one leaving may stop)."""
        with self._lock:
            return len(self._listeners)

    @property
    def sent(self) -> int:
        """Ticks broadcast so far (diagnostics; also asserted in tests)."""
        with self._lock:
            return self._sent

    # -- internals -------------------------------------------------------------

    _server = None  # set via attach_server

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                rate = self._rate_s
            if rate <= 0:
                return  # set_rate(0): nothing runs, thread exits
            t0 = time.perf_counter()
            try:
                snap = self._poll()
                with self._lock:
                    self._snapshot = snap
                self._push(snap)
                for cb in tuple(self._listeners):
                    try:
                        cb(snap)
                    except Exception:  # noqa: BLE001 - listeners are guests
                        _logger.exception("spine listener failed")
            except Exception:  # noqa: BLE001 - the spine must never die
                _logger.exception("spine tick failed")
            finally:
                switch_registry.record_self_overhead(
                    SWITCH_ID, time.perf_counter() - t0)
            if self._wake.wait(rate):
                self._wake.clear()  # rate changed (or stop) - re-read it
            if self._stop.is_set():
                return

    def _poll(self) -> CdlSnapshot:
        """Aggregate every provider; failures degrade one section only."""
        sections: dict = {}
        for provider in list(self._providers):
            try:
                sections[provider.fields] = provider.poll()
            except Exception as exc:  # noqa: BLE001 - XPUSYS isolation rule
                sections[provider.fields] = {"error": str(exc)}
        return CdlSnapshot.now(
            cpu=sections.get("cpu", {"error": "no provider"}),
            ram=sections.get("ram", {"error": "no provider"}),
            proc=sections.get("proc", {"error": "no provider"}),
            gpu=sections.get("gpu"),
        )

    def _push(self, snap: CdlSnapshot) -> None:
        server = self._server
        if server is None:
            # Late-bind the real PromptServer ONLY when its module is
            # already loaded - a fresh `import server` from the sampling
            # thread pulls the entire app (torch included) and stalls
            # ticks for seconds; failed imports are not cached, so every
            # tick would retry it.  server.py attaches itself explicitly,
            # and headless consumers read snapshot()/listeners directly.
            module = sys.modules.get("server")
            prompt_server = getattr(module, "PromptServer", None)
            server = getattr(prompt_server, "instance", None)
        if server is None:
            return
        try:
            server.send_sync(WS_MESSAGE, snap.to_dict())
            with self._lock:
                self._sent += 1
        except Exception:  # noqa: BLE001 - push is best-effort, never fatal
            _logger.debug("spine push failed", exc_info=True)
