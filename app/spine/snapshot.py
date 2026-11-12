"""CdlSnapshot - the spine's versioned, immutable data contract.

Borrowed from XPUSYS-Monitor's GPUSnapshot (providers/base.py:21-60): a
frozen dataclass swapped wholesale under a lock, never mutated in place, so
readers (WS clients, watchdog, future App Panel gauges) either see a
complete old snapshot or a complete new one.  ``schema_version`` lets the
frontend dispatch by contract revision instead of guessing field presence
(the attention-benchmark ``benchmark_db.json`` lesson).

``gpu`` is optional: the P6 ``GpuProvider`` fills it with a dict (or
``{"available": false}`` on CPU-only machines), but hand-built snapshots and
older producers may still leave it ``None``.  Consumers must tolerate both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: Bump when the field set changes incompatibly; readers dispatch on this.
#: v2 (P6): ``gpu`` is now populated by ``GpuProvider``.  Every v1 field keeps
#: its name and type, so older consumers remain compatible.
SNAPSHOT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CdlSnapshot:
    """One complete observation tick (immutable; replaced, never mutated)."""

    ts: float
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    cpu: Dict[str, Any] = field(default_factory=dict)
    ram: Dict[str, Any] = field(default_factory=dict)
    #: This process's own CPU/RSS (the watchdog's primary signal).
    proc: Dict[str, Any] = field(default_factory=dict)
    #: Filled by ``GpuProvider`` in P6; ``None`` for hand-built/v1 snapshots.
    gpu: Optional[Dict[str, Any]] = None

    @classmethod
    def now(cls, cpu: Dict[str, Any], ram: Dict[str, Any],
            proc: Optional[Dict[str, Any]] = None,
            gpu: Optional[Dict[str, Any]] = None) -> "CdlSnapshot":
        return cls(ts=time.time(), cpu=cpu, ram=ram,
                   proc=proc if proc is not None else {}, gpu=gpu)

    def to_dict(self) -> Dict[str, Any]:
        """The exact object broadcast on the WS channel / returned by GET."""
        return {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "cpu": dict(self.cpu),
            "ram": dict(self.ram),
            "proc": dict(self.proc),
            "gpu": dict(self.gpu) if self.gpu is not None else None,
        }
