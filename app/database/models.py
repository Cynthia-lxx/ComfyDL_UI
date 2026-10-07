from typing import Any
from datetime import datetime
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

def to_dict(obj: Any, include_none: bool = False) -> dict[str, Any]:
    fields = obj.__table__.columns.keys()
    out: dict[str, Any] = {}
    for field in fields:
        val = getattr(obj, field)
        if val is None and not include_none:
            continue
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        else:
            out[field] = val
    return out

# TODO: Define models here

# ---------------------------------------------------------------------------
# Profiling v2 P2 models: measured per-node FLOPs of real runs.
#
# Two tables written once per finished prompt by the executor hook (see
# ``comfy/profiling/runmeter.py`` ``persist_run`` and the app-layer callback
# registered in ``app/profiling_routes.py``):
#
# * ``profiling_runs`` - one row per run: totals, mode, run facts;
# * ``profiling_node_stats`` - one row per node: measured FLOPs, op count,
#   census JSON (census mode), demotion flag.
#
# They feed the panel's measured-history card and the estimate calibration
# display (measured vs formula ratio per class_type).

from datetime import timezone  # noqa: E402
from typing import Optional  # noqa: E402
from sqlalchemy import JSON, DateTime, Index, Integer, String, Boolean  # noqa: E402
from sqlalchemy.orm import Mapped, mapped_column  # noqa: E402


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ProfilingRun(Base):
    """One measured execution of a prompt."""

    __tablename__ = "profiling_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prompt_id: Mapped[str] = mapped_column(String(36), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=_utc_now)
    # "count" | "census" - the recording verbosity used for this run
    mode: Mapped[str] = mapped_column(String(16), default="count")
    total_flops: Mapped[int] = mapped_column(Integer, default=0)
    total_ops: Mapped[int] = mapped_column(Integer, default=0)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    # FLOPs observed outside any node context (thread-created work etc.)
    unattributed_flops: Mapped[int] = mapped_column(Integer, default=0)
    # True when the meter hit its op budget and sampled only the head of the
    # run (per-step FLOPs are constant, so the sample carries the signal)
    sampled: Mapped[bool] = mapped_column(Boolean, default=False)
    # meters can suppress bookkeeping errors; surfaced for honesty
    suppressed_errors: Mapped[int] = mapped_column(Integer, default=0)


class ProfilingNodeStat(Base):
    """Per-node measured stats of one run."""

    __tablename__ = "profiling_node_stats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    class_type: Mapped[str] = mapped_column(String(128), index=True)
    flops: Mapped[int] = mapped_column(Integer, default=0)
    op_count: Mapped[int] = mapped_column(Integer, default=0)
    # census-mode per-op histogram {"aten.mm.default": 12, ...}; NULL in
    # count mode or after census demotion
    ops: Mapped[Optional[dict]] = mapped_column(JSON(none_as_null=True), nullable=True)
    demoted: Mapped[bool] = mapped_column(Boolean, default=False)


Index("ix_profiling_node_stats_run", ProfilingNodeStat.run_id)
