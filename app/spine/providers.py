"""Spine providers - pluggable field sources polled by the bus thread.

Protocol (ported from XPUSYS-Monitor providers/base.py:67-155, adapted to
one-thread-many-providers instead of a thread per provider):

* ``fields`` - the snapshot section this provider fills (``"cpu"``,
  ``"ram"``, ``"gpu"``).
* :meth:`poll` - return the section's dict.  **Must never raise**: the bus
  wraps each poll and degrades just that section to ``{"error": ...}``
  (XPUSYS's "one broken provider must not kill the loop" rule); a provider
  that becomes healthy again simply starts returning data again.
* Polls must be cheap (the spine runs on the run path's behalf): the psutil
  CPU call is non-blocking (interval=None compares against the previous
  invocation), RAM reads ``virtual_memory`` once.

NVML/GPU arrives in P6 as another provider; no bus changes will be needed.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


class Provider:
    """Base class / duck-type documentation for spine providers."""

    fields = ""

    def poll(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class PsutilProvider(Provider):
    """CPU + RAM from psutil (first-phase provider; always available)."""

    fields = "cpu"

    def __init__(self) -> None:
        self._psutil = None
        self._per_core = False

    def poll(self) -> Dict[str, Any]:
        if self._psutil is None:
            import psutil

            self._psutil = psutil
            # per-core arrays are cheap but noisy over WS; keep them off
            # unless a consumer asks (settable for tests / future panel).
            self._per_core = getattr(self, "want_per_core", False)
        p = self._psutil
        cpu = {
            "percent": p.cpu_percent(interval=None),
            "count": p.cpu_count() or 0,
            "loadavg": None,
        }
        try:
            cpu["loadavg"] = [round(x, 3) for x in p.getloadavg()]
        except Exception:  # noqa: BLE001 - Windows < build 20252 lacks it
            cpu["loadavg"] = None
        if self._per_core:
            cpu["per_core"] = p.cpu_percent(interval=None, percpu=True)
        return cpu


class RamProvider(Provider):
    """System memory from psutil, reported in whole MB to keep WS small.

    NOTE: never call ``platform.system()`` from ``poll`` - on Windows it
    goes through a WMI query (``platform._wmi_query``) that can block the
    sampling thread for tens of seconds when the WMI service is cold.
    ``sys.platform`` carries the same information at zero cost.
    """

    fields = "ram"

    def __init__(self) -> None:
        self._psutil = None

    def poll(self) -> Dict[str, Any]:
        if self._psutil is None:
            import psutil

            self._psutil = psutil
        vm = self._psutil.virtual_memory()
        swap = self._psutil.swap_memory()
        return {
            "total_mb": int(vm.total // (1024 * 1024)),
            "used_mb": int(vm.used // (1024 * 1024)),
            "available_mb": int(vm.available // (1024 * 1024)),
            "percent": round(vm.percent, 1),
            "swap_percent": round(swap.percent, 1),
            "platform": sys.platform,
        }


class ProcessProvider(Provider):
    """This process's own CPU (per-core scale) + RSS, from psutil.

    The watchdog's primary signal: ``cpu_percent`` here is *per-core
    scaled* (one busy core reads ~100), matching the historical
    ProfilingWatchdog._tick semantics exactly.  The first poll after
    creation primes the counter and returns 0.0 - same priming contract the
    watchdog's own thread used to perform.
    """

    fields = "proc"

    def __init__(self) -> None:
        self._primed = False

    def poll(self) -> Dict[str, Any]:
        import psutil

        proc = psutil.Process()
        if not self._primed:
            proc.cpu_percent(None)
            self._primed = True
        return {
            "pid": proc.pid,
            "cpu_percent": round(float(proc.cpu_percent(None)), 1),
            "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
        }


def default_providers():
    """The provider list the bus starts with (CPU + RAM + this-process;
    system GPU joins in P6)."""
    return [PsutilProvider(), RamProvider(), ProcessProvider()]
