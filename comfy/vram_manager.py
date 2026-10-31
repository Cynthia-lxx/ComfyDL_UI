"""VRAM manager (P1c, M-B; ADR-D7 panel-ready contract).

The last engine of the performance-red-packet milestone: a *coordinator*
over the rehydrated ``comfy.model_management`` primitives - it invents no
memory machinery of its own, it orchestrates them in the proven order and
adds the one thing upstream ComfyUI lacks: visibility of *external* VRAM
occupants (browsers, desktops, other processes) on consumer Windows cards.

Two capabilities:

* **Unified purge routine** (A13/A14 lesson, survey_A batch summary #5):
  ``gc.collect -> soft_empty_cache -> (cuda ipc_collect + synchronize)``,
  with optional full unload (``unload_all_models`` = ``free_memory(1e30)``)
  and the safe variant of Memory_Cleanup's queue flag - here expressed by
  direct orchestration, because the node is already inside a prompt (the
  worker consumes ``free_memory`` flags *between* prompts, which would
  undo the purge the user just asked for; instead we unload synchronously
  and document the difference).
* **External-VRAM correction** (DisTorch A13 pattern, de-risked): sample
  ``total - free - torch_reserved`` = memory held by everyone *except*
  this process, and optionally fold it into ``model_management.EXTRA_RESERVED_VRAM``
  so the scheduler reserves room for the browser instead of OOMing.
  Static defaults (400/600 MB) cannot know a 2 GB browser window; sampling
  can.  The correction is opt-in per node run (``apply_correction``), and
  never lowers the base value - it can only widen the reservation.

ADR-D7 contract: switch ``comfydl.vram`` (hot-toggleable, self-meter n/a -
a purge is user-triggered, not resident); ``describe()``/``status()`` for
the future Optimizer panel; estimator binding: the node's estimator
documents a zero static footprint (it *releases* memory); hard-disable:
on machines without CUDA the node module refuses to import (nodes stay out
of the library, SP-4 precedent), and the switch registers with
``hard_disable_check``.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional

import torch

from comfy.profiling import switch_registry

SWITCH_ID = "comfydl.vram"

LOG_TAG = "[vram-manager]"


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - bare-bones builds
        return False


def register_vram_switch() -> None:
    """Declare the ``comfydl.vram`` switch (idempotent)."""
    if SWITCH_ID not in switch_registry.list_switches():
        switch_registry.register_switch(
            SWITCH_ID,
            default=True,
            hot_toggle=True,
            self_meter=False,  # purge is user-triggered; nothing resident
            hard_disable_check=lambda: not _cuda_available(),
        )


register_vram_switch()


def describe() -> Dict[str, Any]:
    """ADR-D7 panel contract: static shape of this feature."""
    return {
        "id": SWITCH_ID,
        "default": True,
        "hot_toggle": True,
        "self_meter": False,
        "hard_disable_check": "cuda-available",
    }


def status() -> Dict[str, Any]:
    """ADR-D7 panel contract: live state (device usage + correction)."""
    import comfy.model_management as mm

    snap = measure_device_usage()
    return {
        "enabled": switch_registry.is_enabled(SWITCH_ID),
        "device": snap.get("device"),
        "torch_allocated_gib": snap.get("torch_allocated_gib"),
        "torch_reserved_gib": snap.get("torch_reserved_gib"),
        "external_gib": snap.get("external_gib"),
        "correction_active": _correction_active(),
        "reserved_vram_gib": round(mm.EXTRA_RESERVED_VRAM / 1024**3, 3),
    }


def _correction_active() -> bool:
    """True once the dynamic correction has replaced the static default."""
    import comfy.model_management as mm

    return getattr(mm, "_cdl_vram_correction_applied", False)


# --------------------------------------------------------------------------- #
# Measurement

def measure_device_usage() -> Dict[str, Any]:
    """Who is holding VRAM on the primary CUDA device right now?

    Returns a dict with GiB values (and ``{"device": None}`` on CPU-only
    machines - callers treat that as "no signal").  ``external_gib`` is the
    DisTorch insight: *device total used minus what torch reserved* - the
    browser, the desktop compositor, other processes.  On Windows the
    WDDM shared-VRAM illusion makes naive per-process attribution lie, so
    we use device-level accounting only.
    """
    if not _cuda_available():
        return {"device": None}
    device = torch.device("cuda", torch.cuda.current_device())
    free_b, total_b = torch.cuda.mem_get_info(device)
    used_b = total_b - free_b
    reserved_b = torch.cuda.memory_reserved(device)
    allocated_b = torch.cuda.memory_allocated(device)
    external_b = max(0, used_b - reserved_b)
    gib = 1024 ** 3
    return {
        "device": f"cuda:{device.index}",
        "total_gib": round(total_b / gib, 3),
        "free_gib": round(free_b / gib, 3),
        "used_gib": round(used_b / gib, 3),
        "torch_reserved_gib": round(reserved_b / gib, 3),
        "torch_allocated_gib": round(allocated_b / gib, 3),
        "external_gib": round(external_b / gib, 3),
    }


# --------------------------------------------------------------------------- #
# External-VRAM correction (DisTorch pattern, opt-in)

def apply_external_correction(margin_mib: int = 256) -> Optional[Dict[str, Any]]:
    """Fold the measured external usage into the scheduler's reservation.

    ``EXTRA_RESERVED_VRAM`` is consumed by ``load_models_gpu`` when deciding
    how aggressively ``free_memory`` must evict (model_management.py:867,
    920, 924) - a bigger reservation means the scheduler keeps a safety
    margin for the browser instead of trusting all "free" VRAM to torch.
    The correction only ever *widens* the reservation (never below the
    static default chosen at boot), so a quiet desktop can never make the
    scheduler more reckless.

    Returns the applied correction record (or None on CPU-only machines).
    """
    import comfy.model_management as mm

    snap = measure_device_usage()
    if snap.get("device") is None:
        return None
    external_b = int(snap["external_gib"] * 1024 ** 3)
    dynamic = max(mm.EXTRA_RESERVED_VRAM, external_b + margin_mib * 1024 ** 2)
    applied = {
        "static_gib": round(mm.EXTRA_RESERVED_VRAM / 1024 ** 3, 3),
        "external_gib": snap["external_gib"],
        "applied_gib": round(dynamic / 1024 ** 3, 3),
    }
    mm.EXTRA_RESERVED_VRAM = dynamic
    mm._cdl_vram_correction_applied = True
    logging.info(
        "%s external-VRAM correction applied: reserved %s GiB "
        "(static %s GiB, external %s GiB)",
        LOG_TAG, applied["applied_gib"], applied["static_gib"],
        applied["external_gib"])
    return applied


# --------------------------------------------------------------------------- #
# Unified purge routine (survey_A summary #5 order)

def purge(offload_models: bool = False) -> Dict[str, Any]:
    """Run the unified purge; returns a before/after ledger.

    Order (each step earns its place):
      1. snapshot usage,
      2. ``unload_all_models()`` (optional full mode: ``free_memory(1e30)``
         per device - evicts every cached model, not just dead ones),
      3. ``gc.collect()`` - drop python-side references so CUDA blocks can
         actually be released,
      4. ``soft_empty_cache()`` - return cached blocks to the driver,
      5. ``torch.cuda.ipc_collect()`` - free blocks stranded by other
         processes / IPC handles,
      6. synchronize + post snapshot.
    """
    import comfy.model_management as mm

    if not switch_registry.is_enabled(SWITCH_ID):
        return {"disabled": True}

    before = measure_device_usage()
    if offload_models:
        mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    if _cuda_available():
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()
    after = measure_device_usage()

    ledger = {
        "offloaded_models": offload_models,
        "before": before,
        "after": after,
    }
    if before.get("device") is not None:
        freed = before["torch_reserved_gib"] - after["torch_reserved_gib"]
        ledger["freed_reserved_gib"] = round(max(0.0, freed), 3)
        logging.info(
            "%s purge (offload=%s): torch reserved %s -> %s GiB "
            "(freed %s GiB), external %s GiB",
            LOG_TAG, offload_models, before["torch_reserved_gib"],
            after["torch_reserved_gib"], ledger["freed_reserved_gib"],
            after["external_gib"])
    else:
        logging.info("%s purge: no CUDA device, GC-only pass", LOG_TAG)
    return ledger


def render_ledger(ledger: Dict[str, Any]) -> str:
    """Human-readable purge report (the teaching artifact)."""
    if ledger.get("disabled"):
        return ("[vram-manager] feature disabled via the switch - "
                "MODEL state untouched.")
    if ledger["before"].get("device") is None:
        return ("[vram-manager] no CUDA device - ran a GC-only pass "
                "(gc.collect + empty cache), nothing to measure.")
    b, a = ledger["before"], ledger["after"]
    lines = [
        f"[vram-manager] purge report (offload_models={ledger['offloaded_models']})",
        f"{'':<22} {'before':>10} {'after':>10} {'unit':>5}",
        f"{'torch allocated':<22} {b['torch_allocated_gib']:>10.3f} "
        f"{a['torch_allocated_gib']:>10.3f} {'GiB':>5}",
        f"{'torch reserved':<22} {b['torch_reserved_gib']:>10.3f} "
        f"{a['torch_reserved_gib']:>10.3f} {'GiB':>5}",
        f"{'device used':<22} {b['used_gib']:>10.3f} "
        f"{a['used_gib']:>10.3f} {'GiB':>5}",
        f"{'external (non-torch)':<22} {b['external_gib']:>10.3f} "
        f"{a['external_gib']:>10.3f} {'GiB':>5}",
        f"  freed reserved: {ledger.get('freed_reserved_gib', 0):.3f} GiB",
        "  note: 'external' is device-used minus torch-reserved - the "
        "browser/desktop/other processes. Run the node with "
        "apply_correction to fold it into the scheduler's reservation.",
    ]
    return "\n".join(lines)
