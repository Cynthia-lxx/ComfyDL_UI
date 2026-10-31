#!/usr/bin/env python
"""Standalone smoke test: the experiment run ledger (P3 Craft 1, ADR-D10).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_runs_ledger.py

Pure stdlib behaviour, identical on every tier (CPU P drive / CUDA F drive):
switch + ADR-D7 contract shapes, param identity, the happy-path three-layer
roundtrip, failed-run semantics, per-record crash-log flushing, index
robustness (corrupt file, schema rotation), retention pruning, and the
teaching renderers.  Storage is redirected to a tempdir and restored.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_runs_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy import runs_ledger as rl  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

# Isolate storage; every check below runs against this tempdir.
_SANDBOX = Path(tempfile.mkdtemp(prefix="cdl_runs_root_"))
rl.set_root(_SANDBOX)

_ORIGINAL_ENABLED = sr.is_enabled(rl.SWITCH_ID)

# ---------------------------------------------------------------------------
# T1: switch + ADR-D7 contract shapes

check("T1a comfydl.runs switch registered", rl.SWITCH_ID in sr.list_switches())
desc = rl.describe()
check("T1b describe(): default on, hot-toggleable, self-metered, no hard probe",
      desc.get("default") is True and desc.get("hot_toggle") is True
      and desc.get("self_meter") is True
      and desc.get("hard_disable_check") is None)
check("T1c registry description echoes the contract fields",
      all(k in sr.describe(rl.SWITCH_ID)
          for k in ("enabled", "default", "hot_toggle", "self_meter",
                    "self_overhead")))
st = rl.status()
check("T1d status() carries the panel fields",
      all(k in st for k in ("enabled", "root", "index_rows", "detail_files",
                            "last_run_id", "self_overhead")))

# ---------------------------------------------------------------------------
# T2: param identity (canonical, change-sensitive, fail-loud)

check("T2a param_digest is deterministic (key-order independent)",
      rl.param_digest({"lr": 0.01, "hidden": 16})
      == rl.param_digest({"hidden": 16, "lr": 0.01}))
check("T2b param_digest changes when a value changes",
      rl.param_digest({"lr": 0.01}) != rl.param_digest({"lr": 0.02}))
check("T2c param_digest is a 64-hex SHA-256",
      len(rl.param_digest({"x": 1})) == 64
      and all(c in "0123456789abcdef" for c in rl.param_digest({"x": 1})))
try:
    rl.param_digest({"bad": object()})
    check("T2d param_digest raises TypeError on non-serialisable", False)
except TypeError:
    check("T2d param_digest raises TypeError on non-serialisable", True)

# ---------------------------------------------------------------------------
# T3: happy path - the three-layer roundtrip

with rl.start_run(experiment="t3-sweep", params={"lr": 0.01, "hidden": 16},
                  direction="min", metric_name="loss", node_id="42",
                  combo_index=0, seed=1234) as run:
    run.append_history(0, 1.0)
    run.append_history(1, 0.5)
    run.append_history(2, 0.25)
    run.note_series("val", [[0, 1.1], [1, 0.6]])
    run.set_best(True)

detail = rl.read_detail(run.run_id)
check("T3a detail JSON written and parseable", detail is not None)
check("T3b status=completed with duration and finished_at",
      detail.get("status") == "completed"
      and detail.get("finished_at") is not None
      and isinstance(detail.get("duration_s"), float))
check("T3c metric history recorded, last value becomes metric_final",
      detail.get("metrics_history") == [[0, 1.0], [1, 0.5], [2, 0.25]]
      and detail.get("metric_final") == 0.25)
check("T3d params roundtrip and digest matches param_digest()",
      detail.get("params") == {"lr": 0.01, "hidden": 16}
      and detail.get("params_digest") == rl.param_digest(
          {"lr": 0.01, "hidden": 16}))
check("T3e extra series + best flag + identity fields land in the detail",
      detail.get("series", {}).get("val") == [[0, 1.1], [1, 0.6]]
      and detail.get("best") is True and detail.get("node_id") == "42"
      and detail.get("combo_index") == 0 and detail.get("seed") == 1234)
check("T3f device fingerprint present (comparability, E3 lesson)",
      isinstance(detail.get("device"), dict) and "torch" in detail["device"])

index = rl.read_index()
check("T3g index has exactly one row with the run's identity",
      len(index) == 1 and index[0]["run_id"] == run.run_id
      and index[0]["experiment"] == "t3-sweep")
check("T3h index row: numeric columns coerced, best flagged, detail linked",
      isinstance(index[0]["metric_final"], float)
      and isinstance(index[0]["best"], int) and index[0]["best"] == 1
      and index[0]["detail_file"] == f"{run.run_id}.json")
check("T3i params_json survives (truncated) in the index layer",
      "lr" in index[0]["params_json"])

# ---------------------------------------------------------------------------
# T4: failed-run semantics - record, then re-raise

_run4 = None
try:
    with rl.start_run(experiment="t4", params={"lr": 1.0}) as run4:
        _run4 = run4
        run4.append_history(0, 2.0)
        raise RuntimeError("boom")
except RuntimeError:
    check("T4a exceptions propagate through the context manager", True)
failed = rl.read_detail(_run4.run_id)
check("T4b failed run recorded with status + error, history preserved",
      failed is not None and failed["status"] == "failed"
      and "RuntimeError" in failed.get("error", "")
      and failed.get("metrics_history") == [[0, 2.0]])
check("T4c failed run still lands in the index",
      any(r["run_id"] == _run4.run_id and r["status"] == "failed"
          for r in rl.read_index()))

# ---------------------------------------------------------------------------
# T5: direction is validated at the ledger (the GridSearch fix starts here)

try:
    rl.start_run(experiment="t5", params={}, direction="down")
    check("T5 invalid direction raises ValueError", False)
except ValueError:
    check("T5 invalid direction raises ValueError", True)

# ---------------------------------------------------------------------------
# T6: crash layer - root logs are mirrored into the run log, flushed per record
# (production main.py runs the root logger at INFO; the harness leaves the
# default WARNING, so raise it here to mirror the server condition)

import logging  # noqa: E402

logging.getLogger().setLevel(logging.INFO)

with rl.start_run(experiment="t6", params={}) as run6:
    logging.getLogger().info("cdl-run-marker-6789")
log_text = run6.log_path.read_text(encoding="utf-8")
check("T6a root-log record mirrored into the run's .log file",
      "cdl-run-marker-6789" in log_text)
check("T6b the .log file sits next to the detail under the runs root",
      run6.log_path.parent == rl.default_root())

# ---------------------------------------------------------------------------
# T7: query robustness - missing, path-jailed, corrupt

check("T7a read_detail of a missing run returns None",
      rl.read_detail("19990101-000000-deadbe") is None)
check("T7b read_detail refuses path traversal",
      rl.read_detail("..\\evil") is None and rl.read_detail("../evil") is None)
(_SANDBOX / "runs.csv").write_text("garbage header\nmore garbage\n",
                                   encoding="utf-8")
check("T7c corrupt index reads as [] (index is a courtesy, never a raise)",
      rl.read_index() == [])

# ---------------------------------------------------------------------------
# T8: schema rotation - a stale header is preserved as .bak, never destroyed

time.sleep(0.01)  # ensure the rotation backup timestamp differs
with rl.start_run(experiment="t8", params={"fresh": True}) as run8:
    run8.append_history(0, 0.1)
index = rl.read_index()
check("T8a after rotation the index carries only the new-schema row",
      len(index) == 1 and index[0]["run_id"] == run8.run_id)
check("T8b the old file is preserved as runs.csv.bak-*",
      list(_SANDBOX.glob("runs.csv.bak-*")) != [])
with open(_SANDBOX / "runs.csv", newline="", encoding="utf-8") as fh:
    import csv as _csv

    header = next(_csv.reader(fh))
check("T8c the fresh index header matches INDEX_COLUMNS exactly",
      header == list(rl.INDEX_COLUMNS))

# ---------------------------------------------------------------------------
# T9: retention - prune aligns all three layers

for i in range(5):
    time.sleep(0.01)  # mtime separation for the newest-first ordering
    with rl.start_run(experiment=f"t9-{i}", params={"i": i}) as run9:
        run9.append_history(0, float(i))
details_before = len(list(_SANDBOX.glob("*.json")))
removed = rl.prune(keep=3)
remaining = rl.read_index()
check("T9a prune removes all-but-3 runs (details+logs counted together)",
      removed == (details_before - 3) * 2 and len(remaining) == 3,
      f"details_before={details_before} removed={removed} "
      f"rows={len(remaining)}")
check("T9b the index no longer references pruned runs",
      all(r["experiment"] in ("t9-2", "t9-3", "t9-4") for r in remaining))
check("T9c detail files agree with the index after pruning",
      sorted(p.stem for p in _SANDBOX.glob("*.json"))
      == sorted(r["run_id"] for r in remaining))
check("T9d log files agree too",
      sorted(p.stem for p in _SANDBOX.glob("*.log"))
      == sorted(r["run_id"] for r in remaining))

# ---------------------------------------------------------------------------
# T10: teaching renderers

rows = rl.read_index()
table = rl.render_experiment(rows)
check("T10a experiment table names the metric, direction and marks the best",
      "best-by-min" in table and table.count("*") >= 1)
check("T10b empty table explains itself",
      "no runs recorded" in rl.render_experiment([]))
summary = rl.render_run_summary(detail)
check("T10c run summary carries params, metric and BEST flag",
      "lr" in summary and "metric" in summary and "BEST" in summary)
check("T10d failed summary carries the error line",
      "error" in rl.render_run_summary(failed).lower())

# ---------------------------------------------------------------------------
# T11: disabled mode - a no-op stand-in, zero side effects, exceptions pass

sr.set_enabled(rl.SWITCH_ID, False)
try:
    files_before = sorted(p.name for p in _SANDBOX.iterdir())
    dis = rl.start_run(experiment="t11", params={"x": 1})
    check("T11a disabled start_run hands back a no-op (run_id None)",
          dis.run_id is None)
    try:
        with dis:
            dis.append_history(0, 1.0)
            dis.set_best(True)
            raise ValueError("pass-through")
    except ValueError:
        check("T11b disabled run is exception-transparent", True)
    check("T11c disabled run writes nothing",
          sorted(p.name for p in _SANDBOX.iterdir()) == files_before)
    check("T11d status() reports the disabled state",
          rl.status()["enabled"] is False)
finally:
    sr.set_enabled(rl.SWITCH_ID, _ORIGINAL_ENABLED)

# ---------------------------------------------------------------------------
# T12: self-metering (ADR-D1) - the ledger accounts for its own writes

overhead = sr.self_overhead(rl.SWITCH_ID)
check("T12a self-overhead accumulated across the writes above",
      overhead["total_s"] > 0 and overhead["samples"] >= 10,
      f"total={overhead['total_s']:.6f}s samples={overhead['samples']}")

# restore
rl.set_root(None)


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== runs ledger test: {passed} PASS / "
          f"{len(_RESULTS) - passed} FAIL (of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
