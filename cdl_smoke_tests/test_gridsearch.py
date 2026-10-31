#!/usr/bin/env python
"""Standalone smoke test: the experiment bench (P3 Craft 2, ADR-D10).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_gridsearch.py

Covers: param_grid expansion + whitelist errors, node registration, scan
execution end-to-end (winner selection), scan determinism, the direction
min/max fix (verified against the ledger's own numbers, not assumptions),
the lr override, ledger integration (rows + detail history + best flag),
the estimator binding, disabled-switch passthrough, and the TrainingLoop
delegation regression (the moved loop must behave identically).
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_grid_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


import torch  # noqa: E402

from comfy import runs_ledger as rl  # noqa: E402
from comfy import training_protocol as tp  # noqa: E402
from comfy.profiling import estimators as est_mod  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

# Isolate ledger storage; restore at the end.
_SANDBOX = Path(tempfile.mkdtemp(prefix="cdl_grid_root_"))
rl.set_root(_SANDBOX)
_ORIGINAL_ENABLED = sr.is_enabled(rl.SWITCH_ID)

# Registration (same init path the server uses).
import nodes as host_nodes  # noqa: E402

asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                        init_api_nodes=False))
registry = host_nodes.NODE_CLASS_MAPPINGS

check("T2a CdlGridSearchTrain registered",
      "CdlGridSearchTrain" in registry)
check("T2b TrainingLoop still registered (delegation refactor)",
      "TrainingLoop" in registry)

from comfy_extras import nodes_experiment as ne  # noqa: E402
from comfy_extras.nodes_training import TrainingLoop  # noqa: E402

# ---------------------------------------------------------------------------
# T1: param_grid expansion + whitelist

combos = ne.expand_param_grid('{"hidden": ["4", "8"], "lr": [0.1, 0.01]}')
check("T1a cartesian product in insertion order",
      combos == [{"hidden": "4", "lr": 0.1}, {"hidden": "4", "lr": 0.01},
                 {"hidden": "8", "lr": 0.1}, {"hidden": "8", "lr": 0.01}])
check("T1b invalid JSON raises ValueError",
      _raises(lambda: ne.expand_param_grid("{nope"), ValueError))
check("T1c unknown key rejected (the typo'd-key trap)",
      _raises(lambda: ne.expand_param_grid('{"hidden_": ["8"]}'), ValueError))
check("T1d non-list value rejected",
      _raises(lambda: ne.expand_param_grid('{"lr": 0.01}'), ValueError))
check("T1e empty grid rejected",
      _raises(lambda: ne.expand_param_grid("{}"), ValueError))

# ---------------------------------------------------------------------------
# Shared fixture: a small linear regression problem.

torch.manual_seed(0)
x = torch.linspace(0.0, 1.0, 32).reshape(-1, 1)
y = 3.0 * x + 0.05 * torch.randn(32, 1)


def run_scan(direction, grid, experiment, steps=25, seed=11):
    out = ne.CdlGridSearchTrain.execute(
        x=x, y=y, optimizer=tp.optimizer_config(),
        steps=steps, seed=seed, early_stop_patience=0,
        param_grid=grid, direction=direction, experiment_name=experiment)
    return out.args  # (params, best_params, loss_history, summary)


# ---------------------------------------------------------------------------
# T3: happy path - the scan runs and reports a winner

args_min = run_scan("min", '{"lr": [0.02, 0.9]}', "t3-min")
check("T3a scan returns the 4-port contract",
      len(args_min) == 4 and isinstance(args_min[1], str)
      and isinstance(args_min[3], str))
check("T3b best_params is the winner's JSON",
      json.loads(args_min[1]) in ({"lr": 0.02}, {"lr": 0.9}))
check("T3c winner history is 1-D with one entry per step",
      args_min[2].dim() == 1 and args_min[2].shape[0] == 25)
check("T3d summary is the teaching table (direction + best marker)",
      "best-by-min" in args_min[3] and "*" in args_min[3])

# ---------------------------------------------------------------------------
# T4: determinism - same seed, same grid, same scan

args_min_again = run_scan("min", '{"lr": [0.02, 0.9]}', "t4-min")
check("T4a identical scan picks the identical winner",
      args_min_again[1] == args_min[1])
check("T4b identical scan reproduces the winner's history bit-for-bit",
      torch.equal(args_min_again[2], args_min[2]))

# ---------------------------------------------------------------------------
# T5: the direction fix - min/max verified against the ledger's own numbers

args_max = run_scan("max", '{"lr": [0.02, 0.9]}', "t3-max")
check("T5a min and max pick different winners (when the finals differ)",
      args_min[1] != args_max[1])


def _experiment_finals(experiment):
    finals = {}
    for row in rl.read_index():
        if row["experiment"] == experiment:
            finals[row["combo_index"]] = (
                float(row["metric_final"]), json.loads(row["params_json"]))
    return finals


finals_min = _experiment_finals("t3-min")
check("T5b the ledger recorded both combinations",
      len(finals_min) == 2)
expected_min = min(finals_min.values(), key=lambda pair: pair[0])[1]
check("T5c direction=min picked the ledger's lowest final loss",
      json.loads(args_min[1]) == expected_min)
finals_max = _experiment_finals("t3-max")
expected_max = max(finals_max.values(), key=lambda pair: pair[0])[1]
check("T5d direction=max picked the ledger's highest final loss "
      "(the EternalKernel fix)",
      json.loads(args_max[1]) == expected_max)

# ---------------------------------------------------------------------------
# T6: the lr override really reaches the optimizer

args_lo = run_scan("min", '{"lr": [0.02]}', "t6-lo")
args_hi = run_scan("min", '{"lr": [0.9]}', "t6-hi")
check("T6a different lr produces different histories",
      not torch.equal(args_lo[2], args_hi[2]))

# ---------------------------------------------------------------------------
# T7: ledger integration - rows, detail history, best flag

rows_min = [r for r in rl.read_index() if r["experiment"] == "t3-min"]
check("T7a one index row per combination",
      len(rows_min) == 2
      and sorted(r["combo_index"] for r in rows_min) == [0, 1])
check("T7b exactly one row flagged best",
      sum(int(r["best"]) for r in rows_min) == 1)
best_row = next(r for r in rows_min if int(r["best"]) == 1)
check("T7c the flagged row is the actual winner",
      json.loads(best_row["params_json"]) == json.loads(args_min[1]))
detail = rl.read_detail(best_row["run_id"])
check("T7d the winner's detail carries the per-step history",
      detail is not None and len(detail["metrics_history"]) == 25
      and abs(detail["metric_final"] - float(args_min[2][-1])) < 1e-9)
check("T7e the winner's detail is flagged too (mark_best both layers)",
      detail["best"] is True)

# ---------------------------------------------------------------------------
# T8: estimator binding

check("T8a CdlGridSearchTrain estimator registered",
      "CdlGridSearchTrain" in est_mod.ESTIMATORS)
check("T8b combo count: 2x2 grid -> 4 (exact)",
      est_mod._grid_combo_count('{"a": [1, 2], "b": [3, 4]}') == (4, True))
check("T8c combo count: unreadable grid -> approx fallback",
      est_mod._grid_combo_count("not json") == (2, False))

# ---------------------------------------------------------------------------
# T9: disabled switch - the scan runs, the ledger stays silent

sr.set_enabled(rl.SWITCH_ID, False)
try:
    before = len(rl.read_index())
    out_off = ne.CdlGridSearchTrain.execute(
        x=x, y=y, optimizer=tp.optimizer_config(), steps=5, seed=1,
        early_stop_patience=0, param_grid='{"lr": [0.02]}',
        direction="min", experiment_name="t9")
    check("T9a disabled switch: scan still completes with a winner",
          len(out_off.args) == 4 and out_off.args[1] is not None)
    check("T9b disabled switch: no ledger rows written",
          len(rl.read_index()) == before)
finally:
    sr.set_enabled(rl.SWITCH_ID, _ORIGINAL_ENABLED)

# ---------------------------------------------------------------------------
# T10: TrainingLoop delegation regression - the moved loop behaves identically

out_tl = TrainingLoop.execute(
    x=x, y=y, optimizer=tp.optimizer_config(), steps=20, seed=3)
check("T10a TrainingLoop returns the 4-port contract",
      len(out_tl.args) == 4)
history = out_tl.args[2]
check("T10b history has one entry per step",
      history.dim() == 1 and history.shape[0] == 20)
check("T10c the loss actually decreased (training still trains)",
      float(history[-1]) < float(history[0]))
out_tl_again = TrainingLoop.execute(
    x=x, y=y, optimizer=tp.optimizer_config(), steps=20, seed=3)
check("T10d TrainingLoop is still deterministic",
      torch.equal(out_tl_again.args[2], history))

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
    print(f"\n=== grid search test: {passed} PASS / "
          f"{len(_RESULTS) - passed} FAIL (of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
