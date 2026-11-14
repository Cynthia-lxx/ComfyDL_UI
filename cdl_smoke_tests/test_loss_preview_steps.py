"""Repair R2 regression: loss previews must use ABSOLUTE step numbers.

Two user-visible bugs shared one root cause (the preview layer treated
``len(values)`` as the step number):

* a 100k-step run froze both preview charts at step 4096 (max_points crop);
* after Resume (which really continued at step 18201) the curve redrew from
  step 1, making it look as if training had restarted from zero.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_loss_preview_steps.py
"""

import sys
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


from comfy import loss_preview as lp  # noqa: E402


def main() -> int:
    # 1. Cropping the window must NOT reset the x-axis.
    previewer = lp.LossCurvePreviewer(title="loss", min_interval=0.0,
                                      max_points=3)
    for step, loss in enumerate([5.0, 4.0, 3.0, 2.0, 1.0], start=10):
        previewer.record(loss, step=step)
    check("R2-a crop keeps 3 values and their absolute steps",
          len(previewer._values) == 3
          and previewer._steps == [13, 14, 15], str(previewer._steps))
    check("R2-b next step keeps counting past the crop",
          previewer._next_step == 16, str(previewer._next_step))

    # 2. Resume scenario: a fresh previewer that starts at step 18200 must
    #    show 18201, never 1 (the 4.2 "restarted from zero" illusion).
    resumed = lp.LossCurvePreviewer(title="loss", min_interval=0.0)
    resumed.record(0.5, step=18200)
    check("R2-c resumed run starts at the absolute step",
          resumed._steps == [18201], str(resumed._steps))

    # 3. Legacy callers that omit `step` still get a 1-based local counter.
    legacy = lp.LossCurvePreviewer(title="loss", min_interval=0.0)
    legacy.record(1.0)
    legacy.record(0.5)
    check("R2-d step-less callers keep the old local counter",
          legacy._steps == [1, 2], str(legacy._steps))

    # 4. reset() must drop both arrays (single-run contract).
    resumed.reset()
    check("R2-e reset clears steps and the counter",
          resumed._steps == [] and resumed._next_step == 1
          and resumed._values == [], str(resumed._steps))

    # 5. Rendering and the structured push must not raise with absolute steps.
    try:
        frame = previewer.render()
        check("R2-f render works with absolute steps",
              frame is None or isinstance(frame, tuple))
    except Exception as exc:  # noqa: BLE001
        check("R2-f render works with absolute steps", False, str(exc))
    try:
        lp.push_loss_series("loss", previewer._values,
                            total=100000, force=False,
                            steps=previewer._steps)
        lp.push_loss_series("loss", [1.0, 0.5], total=10)  # legacy signature
        check("R2-g push_loss_series accepts steps and stays backward-compatible",
              True)
    except Exception as exc:  # noqa: BLE001
        check("R2-g push_loss_series accepts steps and stays backward-compatible",
              False, str(exc))

    # 6. All three training call sites must pass the absolute step.
    training = (REPO_ROOT / "comfy_extras" / "nodes_training.py").read_text(
        encoding="utf-8")
    lm = (REPO_ROOT / "comfy_extras" / "nodes_lm.py").read_text(
        encoding="utf-8")
    regression = (REPO_ROOT / "comfydl" / "nodes" /
                  "regression_train.py").read_text(encoding="utf-8")
    check("R2-h every training loop passes its absolute step",
          "step=step" in training and "step=step" in lm
          and "step=done - 1" in regression)
    src = (REPO_ROOT / "comfy" / "loss_preview.py").read_text(encoding="utf-8")
    check("R2-i structured payload uses absolute steps",
          '"step": int(steps[-1])' in src
          and "steps=self._steps" in src
          and "steps[i]" in src
          and "axis.set_title" in src)
    return report()


if __name__ == "__main__":
    sys.exit(main())
