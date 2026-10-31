"""Observation spine (dual-track P0 shared infrastructure).

One sampling thread, one versioned snapshot contract, one WS channel -
serving every current and future observer (profiling watchdog today; App
Panel gauges, runs.csv hardware curves and resource monitor in P6) instead
of each growing its own daemon (the Crystools lesson: N plugins = N threads).

Design sources (see .codebuddy/memory/refs_survey + SP dossiers):
- XPUSYS-Monitor providers/base.py - immutable snapshot dataclass, atomic
  in-lock replacement, provider failures never break the loop.
- Crystools general/monitor.py - rate control with 0 == fully stopped
  (thread exits, nothing sampled); we drop its asyncio.run-in-thread hack
  and push via ``PromptServer.instance.send_sync`` instead.
- switch_registry - the spine is a Tier-1 feature and meters its own
  overhead per the meta-profiling rule (bigplan ADR-D1).

Public API: :func:`get_spine` (process singleton), ``Spine.start/stop``,
``set_rate``, ``snapshot``, ``add_listener``; control routes in
:mod:`app.spine.routes` (``GET/PATCH /comfydl/spine``).
"""

from app.spine.bus import Spine, SWITCH_ID, WS_MESSAGE
from app.spine.snapshot import CdlSnapshot, SNAPSHOT_SCHEMA_VERSION

_SPINE = None


def get_spine() -> Spine:
    """The process-wide spine singleton (created on first use)."""
    global _SPINE
    if _SPINE is None:
        _SPINE = Spine()
    return _SPINE


__all__ = [
    "Spine", "SWITCH_ID", "WS_MESSAGE",
    "CdlSnapshot", "SNAPSHOT_SCHEMA_VERSION",
    "get_spine",
]
