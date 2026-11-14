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

GPU arrives in P6 as another provider (``GpuProvider``); no bus changes were
needed - ``Spine._poll`` already routes a ``"gpu"`` section into CdlSnapshot.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from comfy import verbose_log


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
    scaled* (one busy core reads ~100; a fully busy N-core box reads
    ~100*N), matching the historical ProfilingWatchdog._tick semantics.

    R4 (issue 1.5-A, 2026-11-14): the ``Process`` object is now created
    ONCE and REUSED across polls.  ``cpu_percent(interval=None)`` measures
    the delta against the previous call ON THAT OBJECT - the previous code
    built a fresh Process wrapper every poll, so the object never had
    a previous sample and the reading was pinned at 0.0 forever (the UI's
    "Process CPU 0.0%" while the system CPU line spiked).  The first poll
    on the shared object primes the baseline and returns 0.0 exactly once -
    the same priming contract the watchdog's own thread performs.  Sampled
    raw output goes to the VERBOSE log (stage ``spine``, event
    ``proc_poll``, every 8th poll, ``obj=reused``) so numeric-caliber
    questions stay answerable from logs alone (VERBOSE-over-probe rule).
    """

    fields = "proc"

    def __init__(self) -> None:
        self._psutil = None
        self._proc = None
        self._primed = False
        self._polls = 0

    def poll(self) -> Dict[str, Any]:
        if self._psutil is None:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process()
        if not self._primed:
            # First poll on the shared object only primes the baseline.
            self._proc.cpu_percent(None)
            self._primed = True
        cpu_percent = round(float(self._proc.cpu_percent(None)), 1)
        rss_mb = round(self._proc.memory_info().rss / (1024 * 1024), 1)
        self._polls += 1
        verbose_log.step("spine", "proc_poll", self._polls, every_n=8,
                         cpu=cpu_percent, rss_mb=rss_mb, obj="reused")
        return {
            "pid": self._proc.pid,
            "cpu_percent": cpu_percent,
            "rss_mb": rss_mb,
        }


class GpuProvider(Provider):
    """Device-level VRAM snapshot for P6 (soft dependency, never raises).

    The primary signal is the project's existing torch accounting
    (``comfy.vram_manager.measure_device_usage``), which also covers HIP and
    needs no native DLL.  ``pynvml`` is an OPTIONAL enrichment for GPU
    utilisation / temperature - it is never installed at runtime and a failed
    import simply skips that enrichment.

    CPU-only machines get ``{"available": false, "reason": "no_cuda"}``: a
    value, not an exception, so the bus keeps broadcasting complete snapshots.
    """

    fields = "gpu"

    _MEASURE_KEYS = (
        "device", "total_gib", "free_gib", "used_gib",
        "torch_reserved_gib", "torch_allocated_gib", "external_gib",
    )

    def __init__(self) -> None:
        self._nvml = None
        self._nvml_tried = False

    def poll(self) -> Dict[str, Any]:
        try:
            from comfy.vram_manager import measure_device_usage

            usage = measure_device_usage() or {}
        except Exception as exc:  # noqa: BLE001 - never kill the bus loop
            return {"available": False, "reason": "measure_failed",
                    "error": str(exc)}
        if usage.get("device") is None:
            return {"available": False, "reason": "no_cuda"}
        out: Dict[str, Any] = {"available": True, "source": "torch"}
        out.update({k: usage[k] for k in self._MEASURE_KEYS if k in usage})
        nvml = self._nvml_state()
        if nvml is not None:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                rates = nvml.nvmlDeviceGetUtilizationRates(handle)
                out["nvml"] = {
                    "gpu_percent": int(rates.gpu),
                    "mem_percent": int(rates.memory),
                    "temperature_c": int(nvml.nvmlDeviceGetTemperature(
                        handle, nvml.NVML_TEMPERATURE_GPU)),
                }
            except Exception:  # noqa: BLE001 - best-effort enrichment only
                pass
        return out

    def _nvml_state(self):
        if self._nvml_tried:
            return self._nvml
        self._nvml_tried = True
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:  # noqa: BLE001 - soft dependency, no auto-install
            self._nvml = None
        return self._nvml


def default_providers():
    """The provider list the bus starts with (CPU + RAM + this-process + GPU)."""
    return [PsutilProvider(), RamProvider(), ProcessProvider(), GpuProvider()]
