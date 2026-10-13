"""Smoke tests for crash site M3: training breakpoint continuation.

Covers (2026-10-13):

* ``comfy/train_checkpoint.py``: run_key stability/sensitivity, bare
  execution degradation (no context -> no save/load), fake-context
  round-trip and the stale-prompt discard rule;
* node-level continuity through ``TrainingLoop`` with an injected
  interrupt (a global progress hook that raises
  ``InterruptProcessingException`` at step k - the exact mechanism the
  host's ``main.py`` hook uses): the interrupted run leaves a checkpoint,
  the resumed run (SAME prompt_id) continues from the breakpoint (fewer
  progress reports, loss history prefix identical to the saved one), and a
  fresh run (DIFFERENT prompt_id) discards the stale checkpoint and trains
  from scratch.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_train_checkpoint.py
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def report() -> int:
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Bootstrap (sandboxed folders, registry) - mirrors test_crashsite.
# --------------------------------------------------------------------------- #
import cdl_smoke_tests.run_smoke_test as harness  # noqa: E402

harness._bootstrap(Path(tempfile.mkdtemp(prefix="cdl_trainckpt_")))
harness._load_registry(False)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
from comfy.model_management import InterruptProcessingException  # noqa: E402
from comfy import training_protocol as tp  # noqa: E402
from comfy.train_checkpoint import (  # noqa: E402
    _path as _ckpt_path, current_prompt_id, discard as ckpt_discard,
    load as ckpt_load, run_key as ckpt_run_key, save as ckpt_save)
from comfy_execution.utils import CurrentNodeContext  # noqa: E402
from comfy_extras import nodes_training  # noqa: E402

# Redirect the checkpoint root into the sandbox BEFORE anything touches it.
_CKPT_ROOT = Path(tempfile.mkdtemp(prefix="cdl_ckpt_root_"))
import comfy.train_checkpoint as _tcm  # noqa: E402

_tcm.checkpoints_root = lambda: _CKPT_ROOT


def T(*shape):
    n = 1
    for d in shape:
        n *= d
    return torch.arange(n, dtype=torch.float32).reshape(*shape)


# ------------------------------------------------------ K1: helper identity --
def _helper_checks() -> None:
    rk1 = ckpt_run_key({"node": "X", "widgets": [1, 2.5, "a"], "t": T(3)})
    rk2 = ckpt_run_key({"t": T(3), "widgets": [1, 2.5, "a"], "node": "X"})
    rk3 = ckpt_run_key({"node": "X", "widgets": [1, 2.5, "b"], "t": T(3)})
    check("K1 run_key order-insensitive and stable", rk1 == rk2)
    check("K1 run_key sensitive to widget change", rk1 != rk3)
    check("K1 run_key sensitive to tensor payload",
          rk1 != ckpt_run_key({"node": "X", "widgets": [1, 2.5, "a"],
                               "t": T(3) * 2.0}))

    # Bare execution (no CurrentNodeContext): the whole feature is off.
    check("K2 bare: no prompt id", current_prompt_id() is None)
    check("K2 bare: save declines", ckpt_save("rk-bare", {"step": 0}) is False)
    check("K2 bare: load returns None", ckpt_load("rk-bare") is None)


# ------------------------------------------ K3: fake-context round-trip -----
def _roundtrip_checks() -> None:
    rk = ckpt_run_key({"node": "Unit", "widgets": [7]})
    payload = {"step": 3, "model": {"w": T(2, 2)}, "history": [T(1), T(1)]}
    with CurrentNodeContext("pid-a", "5"):
        check("K3 save returns True in context",
              ckpt_save(rk, payload) is True)
        back = ckpt_load(rk)
        check("K3 same-prompt load returns payload",
              back is not None and back["step"] == 3
              and torch.equal(back["model"]["w"], T(2, 2)))
    with CurrentNodeContext("pid-b", "5"):
        check("K3 foreign prompt: stale discard returns None",
              ckpt_load(rk) is None)
        check("K3 stale file removed", not _ckpt_path(rk).is_file())
    ckpt_discard(rk)


# --------------------------------- K4/K5: TrainingLoop interrupt E2E --------
def _training_loop_checks() -> None:
    STEPS = 12
    RAISE_AT = 5  # raise when the reported value reaches 5 (step index 4)
    x, y = T(8, 3), T(8, 1)
    kwargs = dict(x=x, y=y, optimizer=None, params=None, scheduler=None,
                  hidden="8", activation="relu", loss="mse", steps=STEPS,
                  batch_size=0, seed=0, early_stop_patience=0,
                  early_stop_min_delta=1e-4)
    calls: list = []

    def make_hook(raise_at):
        def hook(value, total, preview_image=None, prompt_id=None,
                 node_id=None):
            calls.append(value)
            if raise_at is not None and value >= raise_at:
                raise InterruptProcessingException()
        return hook

    try:
        # The host throttles progress reports (100ms / 0.5%); a fast toy run
        # would only ever report first+final, so the injected interrupt would
        # land on the last step instead of the middle. Disable both knobs for
        # the test - every step reports, like a slow real training does.
        _old_interval = comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL
        _old_percent = comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT
        comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL = 0.0
        comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT = 0.0

        # --- interrupted run: checkpoint appears ---------------------------
        comfy.utils.set_progress_bar_global_hook(make_hook(RAISE_AT))
        with CurrentNodeContext("pid-ck", "7"):
            try:
                nodes_training.TrainingLoop.execute(**kwargs)
                check("K4 interrupted run raises", False, "no exception")
            except InterruptProcessingException:
                check("K4 interrupted run raises", True)
            # The test cannot recompute the node's internal run_key (it
            # hashes data digests); scan the sandbox root instead.
            files = list(_CKPT_ROOT.glob("*.pt"))
            check("K4 checkpoint exists after interrupt", len(files) == 1,
                  str([f.name for f in files]))
            import torch as _t

            data = _t.load(files[0], map_location="cpu", weights_only=True)
            saved = data.get("payload") or {}
            check("K4 checkpoint holds the breakpoint step",
                  saved.get("step") == RAISE_AT - 1
                  and len(saved.get("history") or []) == RAISE_AT,
                  str(saved.get("step")))
            partial = list(saved.get("history") or [])

        # --- resumed run (SAME prompt_id): continues from the breakpoint ---
        comfy.utils.set_progress_bar_global_hook(make_hook(None))
        calls.clear()
        with CurrentNodeContext("pid-ck", "7"):
            out = nodes_training.TrainingLoop.execute(**kwargs)
        curve = out.result[2]
        check("K5 resumed run completes full history",
              len(curve) == STEPS
              and torch.equal(curve[:len(partial)],
                              torch.stack(partial)),
              f"len={len(curve)}")
        check("K5 resumed run only executes the remaining steps",
              len(calls) == STEPS - RAISE_AT,
              f"{len(calls)} report(s)")
        check("K5 completed run discards its checkpoint",
              not any(_CKPT_ROOT.glob("*.pt")))

        # --- fresh run (DIFFERENT prompt_id): stale checkpoint discarded ---
        calls.clear()
        comfy.utils.set_progress_bar_global_hook(make_hook(RAISE_AT))
        with CurrentNodeContext("pid-ck", "7"):
            try:
                nodes_training.TrainingLoop.execute(**kwargs)
            except InterruptProcessingException:
                pass
        comfy.utils.set_progress_bar_global_hook(make_hook(None))
        calls.clear()
        with CurrentNodeContext("pid-other", "7"):
            out = nodes_training.TrainingLoop.execute(**kwargs)
        curve = out.result[2]
        check("K6 fresh prompt trains from scratch",
              len(curve) == STEPS and len(calls) == STEPS,
              f"{len(calls)} report(s)")
        check("K6 stale checkpoint removed",
              not any(_CKPT_ROOT.glob("*.pt")))
    finally:
        comfy.utils.set_progress_bar_global_hook(None)
        comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL = _old_interval
        comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT = _old_percent
        for f in _CKPT_ROOT.glob("*.pt"):
            f.unlink(missing_ok=True)


def main() -> int:
    _helper_checks()
    _roundtrip_checks()
    _training_loop_checks()
    return report()


if __name__ == "__main__":
    sys.exit(main())
