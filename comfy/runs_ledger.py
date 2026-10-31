"""Experiment run ledger (P3 Craft 1, M-C; ADR-D10 three-layer runs schema).

The experiment bench (``CdlGridSearchTrain``, Craft 2) needs a place to
remember what it tried - across restarts, comparable between runs, and
survivable when a training run takes the process down with it.  Three storage
layers, each earning its keep:

* **Index layer** - ``runs.csv``: one flat row per finished run (the
  AMDMonitor E6 pattern: easy to open in Excel, trivial for a frontend to
  list).  Carries the comparison columns (metric/direction/best) plus a
  compact ``params_json`` so a scan is readable without opening any detail.
* **Detail layer** - one JSON per run: full params, the metric history (the
  loss curve the comparison view overlays), timing and a device fingerprint.
  Written at start (``status=running``) and overwritten at finish, so a
  hard-killed run still leaves its params behind.
* **Crash layer** - ``_RunLogHandler``: mirrors the root log into the
  current run's ``.log`` file, flushing **every record** (AMDMonitor's
  crash-forensics insight: when the process aborts there is no clean
  shutdown, so anything buffered is lost; per-record flush is the whole
  point and costs nothing next to a training step).

Design contracts:

* **Location**: ``<user directory>/comfydl/runs/`` - never the plugin
  directory, so a Manager update cannot wipe history (AMDMonitor lesson,
  same rule as ``signature_cache``).  Atomic JSON writes (``tmp`` +
  ``os.replace``, Model-Analytics lesson).
* **Switch**: ``comfydl.runs`` in the central registry (dual-track P0
  contract): hot-toggleable, self-metered (ADR-D1: every Tier-1 feature
  accounts for its own overhead), no hard-disable probe (a ledger is
  hardware-independent).  When the switch is off :func:`start_run` hands
  back a no-op stand-in so callers need no branching.
* **Param identity**: :func:`param_digest` reuses the P0 signature cache's
  canonical SHA-256 - the same key scheme the attention benchmark db and
  the compile manager use, so "same params" means the same thing everywhere.
* **Layering**: stdlib-only at import (``folder_paths`` and ``torch`` are
  imported lazily), mirroring the proflog rule that early imports stay
  cheap.

Public API
----------
- :func:`default_root` / :func:`set_root`  - storage location (tests redirect)
- :func:`param_digest`                     - canonical param identity
- :func:`start_run`                        - open an :class:`ActiveRun`
- :func:`read_index` / :func:`read_detail` - query layers 1 and 2
- :func:`prune`                            - retention (newest KEEP_RUNS wins)
- :func:`describe` / :func:`status`        - ADR-D7 panel contract
- :func:`render_run_summary` / :func:`render_experiment` - teaching artifacts
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from comfy.profiling import switch_registry
from comfy.profiling.signature_cache import digest as signature_digest

SWITCH_ID = "comfydl.runs"

LOG_TAG = "[runs-ledger]"

#: Bumped on any breaking change to the detail-JSON or index schema.
SCHEMA_VERSION = 1

#: Retention: keep the newest N runs (details + logs + index rows aligned).
KEEP_RUNS = 200

#: Index-layer columns (E6's 19 trimmed to the comparison-essential 14).
INDEX_COLUMNS = (
    "run_id",       # unique id, also the detail/log file stem
    "experiment",   # caller-chosen batch name (e.g. "gridsearch-lr-sweep")
    "started_at",   # local ISO timestamp
    "duration_s",   # wall seconds, rounded
    "status",       # completed | failed
    "node_id",      # ComfyUI node unique_id that owned the run
    "combo_index",  # position within a scan batch (None for single runs)
    "direction",    # min | max - how metric_final maps to "better"
    "metric_name",  # e.g. "loss"
    "metric_final", # last recorded metric value
    "best",         # 1 when the caller flagged this run as the scan's best
    "params_digest",# param_digest(params) - canonical identity
    "params_json",  # compact params (truncated; full params in the detail)
    "detail_file",  # JSON filename, "" when the detail never got written
)

#: params_json is capped in the index; the detail layer always holds it all.
_PARAMS_JSON_MAX = 240

_logger = logging.getLogger("ComfyDL.runs")

_lock = threading.RLock()

#: Overridable storage root (tests point this at a tempdir).
_root_override: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Storage location

def default_root() -> Path:
    """``<user directory>/comfydl/runs`` (tempdir fallback for bare tests)."""
    if _root_override is not None:
        return _root_override
    try:
        import folder_paths

        base = Path(folder_paths.get_user_directory())
    except Exception:  # pragma: no cover - headless/test contexts
        base = Path(tempfile.gettempdir()) / "comfydl"
    return base / "runs"


def set_root(path: Union[str, Path, None]) -> None:
    """Point the storage root elsewhere (test isolation only; ``None``
    restores the default user-directory location)."""
    global _root_override
    with _lock:
        _root_override = Path(path) if path is not None else None


# --------------------------------------------------------------------------- #
# Switch + ADR-D7 panel contract

def register_runs_switch() -> None:
    """Declare the ``comfydl.runs`` switch (idempotent)."""
    if SWITCH_ID not in switch_registry.list_switches():
        switch_registry.register_switch(
            SWITCH_ID,
            default=True,
            hot_toggle=True,
            self_meter=True,  # ADR-D1: ledger writes are run-path work
            hard_disable_check=None,  # hardware-independent by nature
        )


register_runs_switch()


def describe() -> Dict[str, Any]:
    """ADR-D7 panel contract: static shape of this feature."""
    return {
        "id": SWITCH_ID,
        "default": True,
        "hot_toggle": True,
        "self_meter": True,
        "hard_disable_check": None,
    }


def status() -> Dict[str, Any]:
    """ADR-D7 panel contract: live state (root, row counts, overhead)."""
    root = default_root()
    rows = read_index()
    details = _detail_files()
    return {
        "enabled": switch_registry.is_enabled(SWITCH_ID),
        "root": str(root),
        "index_rows": len(rows),
        "detail_files": details,
        "last_run_id": rows[-1]["run_id"] if rows else None,
        "self_overhead": switch_registry.self_overhead(SWITCH_ID),
    }


# --------------------------------------------------------------------------- #
# Param identity (reuses the P0 signature scheme)

def param_digest(params: Dict[str, Any]) -> str:
    """Canonical SHA-256 identity for a hyper-parameter dict.

    Delegates to :func:`comfy.profiling.signature_cache.digest` with the
    ``runs`` namespace - identical params yield identical digests across
    sessions, and non-serialisable payloads (a stray tensor) raise loudly
    instead of silently poisoning the key.
    """
    return signature_digest("runs", params)


# --------------------------------------------------------------------------- #
# Crash layer: per-run log capture (AMDMonitor E6 pattern)

class _RunLogHandler(logging.Handler):
    """Mirror the root log into the current run's file, flushing every record.

    Flushing on every line is the whole point: when the process aborts there
    is no clean shutdown, so anything still buffered is lost.  The cost is
    trivial next to a training step.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._fh = None

    def open(self, path: Path) -> None:
        self.close()
        try:
            self._fh = open(path, "a", encoding="utf-8", errors="replace")
        except OSError as exc:
            _logger.warning("%s cannot open run log: %s", LOG_TAG, exc)
            self._fh = None

    def close(self) -> None:  # noqa: D102 - stdlib signature
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:  # noqa: BLE001 - logging must never raise
                pass
        self._fh = None

    def emit(self, record: logging.LogRecord) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(self.format(record) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass


_LOG_HANDLER = _RunLogHandler()
_LOG_ATTACHED = False


def _ensure_log_handler_attached() -> None:
    """Attach the per-run handler to the root logger exactly once."""
    global _LOG_ATTACHED
    if _LOG_ATTACHED:
        return
    try:
        logging.getLogger().addHandler(_LOG_HANDLER)
        _LOG_ATTACHED = True
    except Exception:  # noqa: BLE001 - forensics is a courtesy
        pass


# --------------------------------------------------------------------------- #
# The run object

class ActiveRun:
    """One recorded experiment run; use as a context manager.

    A typical consumer (Craft 2's grid-search node)::

        with start_run(experiment="lr-sweep", params={"lr": 0.01},
                       direction="min", metric_name="loss") as run:
            for step, loss in train_one_config():
                run.append_history(step, loss)
            run.set_best(is_best_of_batch)

    On normal exit the run is written with ``status=completed``; on
    exception with ``status=failed`` + the error, and the exception
    re-raises (the ledger never decides whether a failure is survivable).
    """

    def __init__(
        self,
        *,
        experiment: str,
        params: Dict[str, Any],
        direction: str = "min",
        metric_name: str = "loss",
        node_id: str = "",
        combo_index: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if direction not in ("min", "max"):
            raise ValueError(
                f"direction must be 'min' or 'max', got {direction!r} "
                "(the GridSearch direction fix starts at the ledger)")
        self.run_id = _new_run_id()
        self.root = default_root()
        self.started_monotonic = time.perf_counter()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.detail: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment": str(experiment),
            "started_at": stamp,
            "finished_at": None,
            "duration_s": None,
            "status": "running",
            "node_id": str(node_id),
            "combo_index": combo_index,
            "direction": direction,
            "metric_name": str(metric_name),
            "metric_final": None,
            "best": False,
            "params": params,
            "params_digest": param_digest(params),
            "seed": seed,
            "metrics_history": [],  # [[step, value], ...]
            "series": {},           # name -> [[k, v], ...] (extra curves)
            "device": _device_fingerprint(),
            "error": None,
        }
        self._finished = False
        _ensure_log_handler_attached()
        _LOG_HANDLER.open(self.log_path)
        # Detail layer goes down *first* with status=running: a hard-killed
        # run still leaves its params and identity behind (only the csv
        # index row, written at finish, requires a clean ending).
        self._write_detail()

    # -- paths -------------------------------------------------------------

    @property
    def detail_path(self) -> Path:
        return self.root / f"{self.run_id}.json"

    @property
    def log_path(self) -> Path:
        return self.root / f"{self.run_id}.log"

    # -- recording ----------------------------------------------------------

    def append_history(self, step: int, value: float) -> None:
        """Record one metric sample; the last one becomes ``metric_final``."""
        self.detail["metrics_history"].append(
            [int(step), float(value)])
        self.detail["metric_final"] = float(value)

    def note_series(self, name: str, points: List[List[float]]) -> None:
        """Attach a named extra curve (e.g. a validation metric)."""
        self.detail["series"][str(name)] = [
            [float(k), float(v)] for k, v in points]

    def set_best(self, best: bool) -> None:
        """Flag this run as its scan batch's best (direction already applied
        by the caller - the ledger stores the verdict, not the opinion)."""
        self.detail["best"] = bool(best)

    # -- lifecycle -----------------------------------------------------------

    def finish(self, status: str = "completed") -> None:
        """Write the detail JSON (final) and append the index row."""
        if self._finished:
            return
        self._finished = True
        _LOG_HANDLER.close()
        self.detail["status"] = status
        self.detail["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.detail["duration_s"] = round(
            time.perf_counter() - self.started_monotonic, 4)
        self._write_detail()
        self._append_index_row()

    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.detail["error"] = f"{exc_type.__name__}: {exc}"
            self.finish("failed")
            return False  # re-raise: the ledger records, it never decides
        self.finish("completed")
        return False

    # -- writing --------------------------------------------------------------

    def _write_detail(self) -> None:
        """Atomic detail write (tmp + os.replace, Model-Analytics lesson)."""
        t0 = time.perf_counter()
        path = self.detail_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{self.run_id}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.detail, fh, indent=2, default=str)
                os.replace(tmp_name, path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as exc:
            _logger.warning("%s cannot write run detail: %s", LOG_TAG, exc)
        switch_registry.record_self_overhead(
            SWITCH_ID, time.perf_counter() - t0)

    def _append_index_row(self) -> None:
        """One flat csv row per finished run (AMDMonitor pattern)."""
        t0 = time.perf_counter()
        d = self.detail
        try:
            params_json = json.dumps(
                d["params"], sort_keys=True, separators=(",", ":"),
                default=str)
        except (TypeError, ValueError):
            params_json = json.dumps({"_unserialisable": True})
        if len(params_json) > _PARAMS_JSON_MAX:
            params_json = params_json[: _PARAMS_JSON_MAX - 3] + "..."
        row = {
            "run_id": d["run_id"],
            "experiment": d["experiment"],
            "started_at": d["started_at"],
            "duration_s": d["duration_s"],
            "status": d["status"],
            "node_id": d["node_id"],
            "combo_index": d["combo_index"],
            "direction": d["direction"],
            "metric_name": d["metric_name"],
            "metric_final": d["metric_final"],
            "best": int(d["best"]),
            "params_digest": d["params_digest"],
            "params_json": params_json,
            "detail_file": self.detail_path.name,
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            index = self.root / "runs.csv"
            with _lock:
                _migrate_stale_index(index)
                # Re-check after migration: a rotated-away file means the
                # append starts a fresh index and must write the header.
                new = not index.exists() or index.stat().st_size == 0
                with open(index, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
                    if new:
                        writer.writeheader()
                    writer.writerow(row)
        except OSError as exc:
            _logger.warning("%s cannot append runs.csv: %s", LOG_TAG, exc)
        switch_registry.record_self_overhead(
            SWITCH_ID, time.perf_counter() - t0)


class _DisabledRun:
    """No-op stand-in handed out when the ``comfydl.runs`` switch is off.

    Carries the same interface as :class:`ActiveRun` so callers can use
    ``with start_run(...)`` unconditionally; every method is a no-op and
    ``run_id`` is ``None``.
    """

    run_id: Optional[str] = None

    def append_history(self, step: int, value: float) -> None: ...
    def note_series(self, name: str, points: List[List[float]]) -> None: ...
    def set_best(self, best: bool) -> None: ...
    def finish(self, status: str = "completed") -> None: ...

    def __enter__(self) -> "_DisabledRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # transparent: exceptions propagate untouched


def start_run(
    *,
    experiment: str,
    params: Dict[str, Any],
    direction: str = "min",
    metric_name: str = "loss",
    node_id: str = "",
    combo_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> Union[ActiveRun, _DisabledRun]:
    """Open a recorded run (or a no-op stand-in when the switch is off).

    See :class:`ActiveRun` for the recording interface and the failure
    semantics (``failed`` status + re-raise).
    """
    if not switch_registry.is_enabled(SWITCH_ID):
        return _DisabledRun()
    return ActiveRun(
        experiment=experiment,
        params=params,
        direction=direction,
        metric_name=metric_name,
        node_id=node_id,
        combo_index=combo_index,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Query layer

def read_index(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Index rows as dicts, oldest first (``limit`` keeps the newest N).

    A missing or corrupt csv returns ``[]`` (the index is a courtesy; the
    detail layer is the truth).
    """
    index = default_root() / "runs.csv"
    try:
        with _lock, open(index, "r", newline="", encoding="utf-8") as fh:
            rows = [
                row for row in csv.DictReader(fh)
                if row.get("run_id")
            ]
    except FileNotFoundError:
        return []
    except OSError as exc:
        _logger.warning("%s cannot read runs.csv: %s", LOG_TAG, exc)
        return []
    rows = [row for row in rows if _coerce_row(row)]
    return rows[-limit:] if limit else rows


def _coerce_row(row: Dict[str, str]) -> bool:
    """Normalise csv strings in place; drops malformed rows (returns False)."""
    try:
        for key in ("duration_s", "metric_final"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])  # type: ignore[assignment]
        for key in ("best", "combo_index"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])  # type: ignore[assignment]
    except (TypeError, ValueError):
        return False
    return True


def read_detail(run_id: str) -> Optional[Dict[str, Any]]:
    """The detail JSON for ``run_id`` (``None`` when absent/corrupt)."""
    if not run_id or any(c in run_id for c in "\\/"):
        return None  # path-jailed: run ids are our own stems, never paths
    path = default_root() / f"{Path(run_id).name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _detail_files() -> int:
    try:
        return sum(
            1 for p in default_root().iterdir() if p.suffix == ".json")
    except OSError:
        return 0


# --------------------------------------------------------------------------- #
# Retention

def prune(keep: int = KEEP_RUNS) -> int:
    """Keep the newest ``keep`` runs (by detail-file mtime); returns pruned.

    Prunes the detail JSON + log together (same stem) and rewrites the index
    without their rows, so the three layers can never disagree about what
    still exists.
    """
    root = default_root()
    try:
        details = sorted(
            (p for p in root.iterdir() if p.suffix == ".json"),
            key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return 0
    doomed_ids = {p.stem for p in details[keep:]}
    if not doomed_ids:
        return 0
    removed = 0
    for stem in doomed_ids:
        for suffix in (".json", ".log"):
            try:
                (root / f"{stem}{suffix}").unlink()
                removed += 1
            except OSError:
                pass
    _rewrite_index_without(doomed_ids)
    return removed


def _rewrite_index_without(doomed_ids: set) -> None:
    """Rewrite runs.csv dropping the given run_ids (atomic-ish)."""
    rows = read_index()
    kept = [r for r in rows if r.get("run_id") not in doomed_ids]
    if len(kept) == len(rows):
        return
    index = default_root() / "runs.csv"
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(default_root()), prefix=".runs.", suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
        os.replace(tmp_name, index)
    except OSError as exc:
        _logger.warning("%s cannot rewrite runs.csv during prune: %s",
                        LOG_TAG, exc)


def _migrate_stale_index(index: Path) -> None:
    """Rotate a pre-existing csv whose header predates INDEX_COLUMNS.

    Called (under the lock, before append) when the file exists but its
    header does not match: the old file is preserved as ``.bak-<stamp>``
    and the append starts a fresh index.  History is never destroyed.
    """
    try:
        with open(index, "r", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
    except OSError:
        return
    if header == list(INDEX_COLUMNS):
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = index.with_name(f"runs.csv.bak-{stamp}")
    _logger.warning(
        "%s runs.csv header mismatch (%s) - rotating to %s",
        LOG_TAG, header, backup.name)
    try:
        os.replace(index, backup)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Teaching artifacts (Craft 2's node output)

def render_run_summary(detail: Dict[str, Any]) -> str:
    """Human-readable one-run report (the teaching artifact)."""
    lines = [
        f"[runs-ledger] run {detail.get('run_id')} "
        f"({detail.get('status')})",
        f"  experiment : {detail.get('experiment')}",
        f"  params     : {json.dumps(detail.get('params', {}), default=str)}",
        f"  metric     : {detail.get('metric_name')}"
        f"={detail.get('metric_final')} "
        f"(direction={detail.get('direction')}"
        f"{', BEST' if detail.get('best') else ''})",
        f"  duration   : {detail.get('duration_s')} s",
    ]
    if detail.get("error"):
        lines.append(f"  error      : {detail['error']}")
    return "\n".join(lines)


def render_experiment(rows: List[Dict[str, Any]]) -> str:
    """Human-readable scan table with the best run highlighted."""
    if not rows:
        return "[runs-ledger] no runs recorded."
    direction = rows[0].get("direction", "min")
    metric_name = rows[0].get("metric_name", "metric")
    scored = [r for r in rows if r.get("metric_final") not in (None, "")]
    sign = -1.0 if direction == "max" else 1.0
    best_id = None
    if scored:
        best_id = min(
            scored, key=lambda r: sign * float(r["metric_final"]))["run_id"]
    lines = [
        f"[runs-ledger] experiment '{rows[0].get('experiment')}' - "
        f"{len(rows)} run(s), best-by-{direction} {metric_name} "
        f"marked with *",
    ]
    for r in rows:
        marker = "*" if r.get("run_id") == best_id else " "
        status = r.get("status", "?")
        metric = r.get("metric_final")
        metric = "-" if metric in (None, "") else f"{metric:.6g}"
        lines.append(
            f" {marker} #{r.get('combo_index', '-'):>3} {status:<9} "
            f"{metric:>12}  {r.get('params_json', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Internals

def _new_run_id() -> str:
    """``<YYYYmmdd-HHMMSS>-<6 hex>`` - sortable, filesystem-safe, unique."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _device_fingerprint() -> Dict[str, str]:
    """Cheap comparability stamp (benchmark E3's lesson: runs are only
    comparable when the device they ran on is recorded)."""
    fingerprint: Dict[str, str] = {}
    try:
        import torch

        fingerprint["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            fingerprint["cuda"] = torch.cuda.get_device_name(0)
        else:
            fingerprint["cuda"] = "none"
    except Exception:  # noqa: BLE001 - fingerprint is a courtesy
        fingerprint["torch"] = "unknown"
    return fingerprint
