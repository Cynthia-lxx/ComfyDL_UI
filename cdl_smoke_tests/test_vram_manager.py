#!/usr/bin/env python
"""Standalone smoke test: the VRAM manager (P1c, M-B; ADR-D7 contract).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_vram_manager.py

Tiers (each upgrades gracefully with hardware):

* CPU-only (P drive) - switch/describe/status contract shapes, the
  GC-only purge pass, ledger rendering on fabricated data, the estimator
  binding, and the hard-disable proof (nodes_vram must refuse to import);
* CUDA (F drive) - everything above plus a real purge on the device, the
  widen-only external-VRAM correction against model_management, and node
  registration.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_vram_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


import torch  # noqa: E402

from comfy import vram_manager as vm  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

CUDA = torch.cuda.is_available()
_ORIGINAL_ENABLED = sr.is_enabled(vm.SWITCH_ID)

# ---------------------------------------------------------------------------
# T1: switch + ADR-D7 contract shapes

check("T1a comfydl.vram switch registered", vm.SWITCH_ID in sr.list_switches())
check("T1b switch carries the hard-disable probe (cuda-available)",
      vm.describe().get("hard_disable_check") == "cuda-available")
check("T1c switch is hot-toggleable and not self-metered",
      vm.describe().get("hot_toggle") is True
      and vm.describe().get("self_meter") is False)

desc = sr.describe(vm.SWITCH_ID)
check("T1d registry description echoes the contract fields",
      all(k in desc for k in ("enabled", "default", "hot_toggle",
                              "self_meter", "self_overhead")))

st = vm.status()
check("T1e status() carries the panel fields",
      all(k in st for k in ("enabled", "device", "correction_active",
                            "reserved_vram_gib")))

# ---------------------------------------------------------------------------
# T2: measurement semantics

snap = vm.measure_device_usage()
if CUDA:
    check("T2a CUDA measurement carries all fields",
          all(k in snap for k in ("total_gib", "free_gib", "used_gib",
                                  "torch_reserved_gib",
                                  "torch_allocated_gib", "external_gib")))
    check("T2b used = total - free (device-level accounting)",
          abs(snap["used_gib"] - (snap["total_gib"] - snap["free_gib"])) < 0.01)
    check("T2c external >= 0 and <= used (no negative phantom usage)",
          0 <= snap["external_gib"] <= snap["used_gib"])
else:
    check("T2d CPU-only machine: measurement reports no device",
          snap.get("device") is None)

# ---------------------------------------------------------------------------
# T3: purge on this machine (safe on both tiers)

led = vm.purge()
if CUDA:
    check("T3a purge returns a before/after ledger",
          led.get("before", {}).get("device") is not None
          and led.get("after", {}).get("device") is not None)
    check("T3b freed_reserved_gib is non-negative",
          led.get("freed_reserved_gib", -1) >= 0)
else:
    # CPU-only: the switch's hard_disable_check auto-disables the feature -
    # the purge is a documented no-op (never a degraded partial run).
    check("T3c CPU-only: purge is hard-disabled by the switch probe",
          led.get("disabled") is True)

# ---------------------------------------------------------------------------
# T4: ledger rendering on fabricated data (deterministic, both tiers)

fake = {
    "offloaded_models": True,
    "before": {"device": "cuda:0", "total_gib": 8.0, "free_gib": 1.0,
               "used_gib": 7.0, "torch_reserved_gib": 5.0,
               "torch_allocated_gib": 4.5, "external_gib": 2.0},
    "after": {"device": "cuda:0", "total_gib": 8.0, "free_gib": 3.0,
              "used_gib": 5.0, "torch_reserved_gib": 3.0,
              "torch_allocated_gib": 0.1, "external_gib": 2.0},
    "freed_reserved_gib": 2.0,
}
text = vm.render_ledger(fake)
check("T4a ledger prints before/after and the external line",
      "torch reserved" in text and "external (non-torch)" in text)
check("T4b ledger states the freed amount and the teaching note",
      "freed reserved: 2.000 GiB" in text
      and "browser" in text and "apply_correction" in text)
check("T4c disabled ledger explains the passthrough",
      "disabled via the switch" in vm.render_ledger({"disabled": True}))
check("T4d no-CUDA ledger explains the GC-only pass",
      "GC-only pass" in vm.render_ledger(
          {"offloaded_models": False,
           "before": {"device": None}, "after": {"device": None}}))

# ---------------------------------------------------------------------------
# T5: the switch hot-disables the purge (pure passthrough, state restored)

sr.set_enabled(vm.SWITCH_ID, False)
try:
    led = vm.purge()
    check("T5 disabled switch -> purge is a documented no-op",
          led.get("disabled") is True)
finally:
    sr.set_enabled(vm.SWITCH_ID, _ORIGINAL_ENABLED)

# ---------------------------------------------------------------------------
# T6: external-VRAM correction

import comfy.model_management as mm  # noqa: E402

_ORIGINAL_RESERVED = mm.EXTRA_RESERVED_VRAM
try:
    applied = vm.apply_external_correction()
    if CUDA:
        check("T6a correction returns a record", applied is not None
              and "applied_gib" in applied)
        check("T6b correction is widen-only (never below the static base)",
              mm.EXTRA_RESERVED_VRAM >= _ORIGINAL_RESERVED)
        check("T6c applied = max(base, external + 256 MiB margin)",
              mm.EXTRA_RESERVED_VRAM == max(
                  _ORIGINAL_RESERVED,
                  int(applied["external_gib"] * 1024 ** 3) + 256 * 1024 ** 2),
              f"reserved={mm.EXTRA_RESERVED_VRAM} applied={applied}")
        check("T6d correction flagged active for the panel",
              vm._correction_active() is True
              and vm.status()["correction_active"] is True)
    else:
        check("T6e CPU-only: correction is a graceful None", applied is None)
finally:
    mm.EXTRA_RESERVED_VRAM = _ORIGINAL_RESERVED
    mm._cdl_vram_correction_applied = False

# ---------------------------------------------------------------------------
# T7: estimator binding (ADR-D7 ③) - registered and honest about zero

from comfy.profiling import estimators as est_mod  # noqa: E402


class _FakeCtx:
    def str_widget(self, _key, default=""):
        return default

    def int_widget(self, _key, default=None):
        return default

    def widget(self, _key, default=None):
        return default


check("T7a CdlPurgeVRAM estimator registered",
      "CdlPurgeVRAM" in est_mod.ESTIMATORS)
out = est_mod._estimate_vram_purge(_FakeCtx())
check("T7b estimator: exact confidence, zero items (releases memory)",
      out.confidence == "exact" and out.items == []
      and "releases memory" in out.reason)

# ---------------------------------------------------------------------------
# T8: node registration - hardware decides which side of the contract

import importlib  # noqa: E402

try:
    nodes_vram = importlib.import_module("comfy_extras.nodes_vram")
    imported = True
except ImportError:
    nodes_vram = None
    imported = False

if CUDA:
    check("T8a CUDA machine: node module imports", imported)
    if imported:
        import asyncio  # noqa: E402

        import nodes as host_nodes  # noqa: E402

        asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                                init_api_nodes=False))
        check("T8b CdlPurgeVRAM registered",
              "CdlPurgeVRAM" in host_nodes.NODE_CLASS_MAPPINGS)
else:
    check("T8c CPU-only machine: nodes_vram import is REFUSED (hard-disable)",
          not imported)
    check("T8d CPU-only: the refusal is an ImportError (loader logs it)",
          not imported)  # detail covered by T8c's exception class


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== vram manager test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
