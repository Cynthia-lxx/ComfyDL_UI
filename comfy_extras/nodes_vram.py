"""VRAM manager nodes (P1c, M-B; ADR-D7 panel-ready contract).

The production facade of ``comfy/vram_manager.py``:

* ``CdlPurgeVRAM`` (OUTPUT_NODE): runs the unified purge routine
  (``gc.collect -> soft_empty_cache -> ipc_collect -> synchronize``, with
  optional full model unload) and emits a before/after VRAM ledger STRING -
  the teaching artifact.  Optionally folds the measured *external* VRAM
  usage (browser/desktop/other processes on the same card) into the
  scheduler's ``EXTRA_RESERVED_VRAM`` so later loads keep a honest safety
  margin instead of OOMing (DisTorch pattern, widen-only).

Hard-disable contract (user ruling 2026-10-31, SP-4 precedent): on machines
without CUDA this module raises at import - the node simply stays out of
the library (nodes.py logs the reason), never a degraded error-and-continue.
The ``comfydl.vram`` switch additionally carries ``hard_disable_check`` for
the registry/panel.
"""

from __future__ import annotations

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

if not torch.cuda.is_available():
    raise ImportError(
        "The VRAM manager nodes are hard-disabled: no CUDA device on this "
        "machine - VRAM purging has nothing to act on. (They will appear "
        "automatically on a CUDA machine.)"
    )

from comfy import vram_manager  # noqa: E402  (after the CUDA gate)

CATEGORY = "model/perf"


class CdlPurgeVRAM(io.ComfyNode):
    """Purge VRAM and print a before/after ledger (production + teaching).

    What: runs the unified purge routine on the primary CUDA device -
          optional full model unload (``free_memory(1e30)`` per device),
          then ``gc.collect -> soft_empty_cache -> ipc_collect ->
          synchronize`` - and emits a human-readable ledger: torch
          allocated/reserved, device-used, and the *external* share
          (device-used minus torch-reserved: browser, desktop, other
          processes), before vs after.
    In:   offload_models (bool, default False) - full mode: unload ALL
          cached models first (next nodes will re-load from disk; use it
          between stages, not mid-generation).
          apply_correction (bool, default False) - DisTorch pattern: fold
          the measured external usage into ``EXTRA_RESERVED_VRAM``
          (widen-only) so the scheduler stops trusting browser-held VRAM
          as free; survives until process restart, re-applies on demand.
    Out:  STRING - the purge ledger (also printed to console).
    Caveats: OUTPUT_NODE - executes on every queue, no caching.  The purge
          frees cached/idle blocks; tensors still referenced by live
          models cannot be released (that is what offload_models is for).
          CPU-only machines: this node is hard-disabled (not registered).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlPurgeVRAM",
            display_name="Purge VRAM",
            category=CATEGORY,
            description=(
                "Unified VRAM purge with a before/after ledger, including "
                "the external (non-torch) share of the card."),
            inputs=[
                io.Boolean.Input(
                    "offload_models", default=False,
                    tooltip="Full mode: unload ALL cached models first "
                            "(they reload from disk on next use)."),
                io.Boolean.Input(
                    "apply_correction", default=False,
                    tooltip="Fold the measured external VRAM usage into "
                            "the scheduler's reserved margin (widen-only, "
                            "lasts until restart)."),
            ],
            outputs=[io.String.Output(display_name="ledger")],
            is_output_node=True,
            search_aliases=[
                "vram", "purge", "memory", "cleanup", "gc", "empty cache",
                "external vram", "disTorch",
            ],
        )

    @classmethod
    def execute(cls, offload_models: bool = False,
                apply_correction: bool = False) -> io.NodeOutput:
        ledger = vram_manager.purge(offload_models=offload_models)
        correction = None
        if apply_correction and not ledger.get("disabled"):
            correction = vram_manager.apply_external_correction()
        text = vram_manager.render_ledger(ledger)
        if correction is not None:
            text += (
                f"\n  correction applied: reserved {correction['static_gib']} "
                f"-> {correction['applied_gib']} GiB "
                f"(external {correction['external_gib']} GiB + 256 MiB margin)")
        print(text)
        return io.NodeOutput(text)


class VramExtension(ComfyExtension):
    """Registers the ``model/perf`` VRAM-manager node."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlPurgeVRAM]


async def comfy_entrypoint() -> VramExtension:
    return VramExtension()
