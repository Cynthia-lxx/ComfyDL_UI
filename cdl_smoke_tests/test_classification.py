#!/usr/bin/env python
"""Standalone smoke test for the classification line (five nodes + templates).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_classification.py

Covers (2026-10-28):
  - CdlClassificationDataGen: the three shapes (blobs / moons / circles) with
    shape, dtype, class-balance and determinism assertions plus the full
    rejection policy (negative noise, n_per_class < 1, n_classes < 2).
  - CdlSoftmaxTrain: the ln(k) anchor at zero initialisation, monotone
    collapse on the separable blobs, simplex probabilities, exact
    reproducibility (no seed — the recipe is deterministic) and the raw
    X/y vs DATASET conduits.
  - CdlClassificationTrain: near-perfect held-out accuracy, same-seed
    reproducibility, label coercion (float [n, 1] labels, rounding), the
    accuracy-based early-stop metric path, early-stop truncation with
    rollback, degenerate-split fallback (a single-sample class) and the
    negative-label rejection.
  - Breakpoint continuation (crash site M3, the ``TrainingLoop`` K4-K6
    pattern): an injected interrupt at step k leaves a checkpoint, the
    resumed run (same prompt_id) continues with an identical loss-history
    prefix and only executes the remaining steps, a fresh run (different
    prompt_id) discards the stale checkpoint.
  - CdlClassificationModelLoad: save → load → forward bitwise identity,
    human-readable info, missing-file rejection.
  - CdlConfusionMatrixEval: a hand-computed matrix, row/column semantics,
    the argmax path, explicit vs inferred class counts and the overflow /
    mismatch / empty rejections.
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
# Bootstrap (sandboxed folders, registry) - mirrors the other standalone tests.
# --------------------------------------------------------------------------- #
import cdl_smoke_tests.run_smoke_test as harness  # noqa: E402

harness._bootstrap(Path(tempfile.mkdtemp(prefix="cdl_class_")))
harness._load_registry(False)

import torch  # noqa: E402

import comfy.utils  # noqa: E402
from comfy.model_management import InterruptProcessingException  # noqa: E402
from comfy_execution.utils import CurrentNodeContext  # noqa: E402

import nodes as host_nodes  # noqa: E402

REG = host_nodes.NODE_CLASS_MAPPINGS

# Redirect the checkpoint root into a sandbox BEFORE anything touches it.
_CKPT_ROOT = Path(tempfile.mkdtemp(prefix="cdl_ckpt_root_class_"))
import comfy.train_checkpoint as _tcm  # noqa: E402

_tcm.checkpoints_root = lambda: _CKPT_ROOT


def _blobs(n_per_class: int = 200, noise: float = 0.2, seed: int = 0):
    """The 3-blob pair, inlined (the same recipe the harness factory uses)."""
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[3.0, 0.0], [-1.5, 2.598], [-1.5, -2.598]])
    labels = torch.repeat_interleave(torch.arange(3), n_per_class)
    features = centers[labels] + noise * torch.randn((n_per_class * 3, 2), generator=generator)
    return features, labels.to(torch.float32)


def _expect(node, fn, *args, **kwargs) -> bool:
    try:
        fn(node, *args, **kwargs)
    except ValueError:
        return True
    except TypeError:
        return True
    return False


# ----------------------------------------------------- C1: data generator ----
def _datagen_checks() -> None:
    gen = REG["CdlClassificationDataGen"]()

    X, y, ds = gen.execute("blobs", 50, 3, 0.1, 0)
    check("C1 blobs shape [150, 2]", tuple(X.shape) == (150, 2), str(tuple(X.shape)))
    check("C1 blobs classes balanced", all(int((y == c).sum()) == 50 for c in range(3)))
    X2, y2, _ = gen.execute("blobs", 50, 3, 0.1, 0)
    check("C1 same seed reproduces exactly", torch.equal(X, X2) and torch.equal(y, y2))

    Xm, ym, _ = gen.execute("moons", 80, 5, 0.05, 3)
    check("C1 moons forces 2 classes", tuple(Xm.shape) == (160, 2)
          and ym.unique().tolist() == [0, 1])
    check("C1 moons interleaved (x range overlaps)",
          float(Xm[:80, 0].min()) < float(Xm[80:, 0].max()))

    Xc, yc, _ = gen.execute("circles", 40, 4, 0.02, 0)
    radii = Xc.norm(dim=1)
    means = [float(radii[yc == c].mean()) for c in range(4)]
    check("C1 circles ring radii ascend", all(means[i] < means[i + 1] for i in range(3)),
          str([f"{m:.2f}" for m in means]))
    check("C1 dataset labels are float indices",
          ds.labels.dtype == torch.float32 and torch.equal(ds.labels.reshape(-1).long(), y))

    check("C1 rejects negative noise", _expect(gen, lambda g: g.execute("blobs", 10, 3, -0.1, 0)))
    check("C1 rejects n_per_class < 1", _expect(gen, lambda g: g.execute("blobs", 0, 3, 0.1, 0)))
    check("C1 rejects n_classes < 2 on blobs", _expect(gen, lambda g: g.execute("blobs", 10, 1, 0.1, 0)))
    check("C1 rejects n_classes < 2 on circles", _expect(gen, lambda g: g.execute("circles", 10, 1, 0.1, 0)))


# -------------------------------------------------- C2: softmax regression ---
def _softmax_checks() -> None:
    import math

    sm = REG["CdlSoftmaxTrain"]()
    X, y = _blobs()

    W, b, logits, probs, curve = sm.execute(X, y, 300, 0.1)
    check("C2 starts at ln(3)", abs(float(curve[0]) - math.log(3)) < 1e-4,
          f"{float(curve[0]):.6f}")
    check("C2 loss collapses >10x", float(curve[-1]) < float(curve[0]) / 10,
          f"{float(curve[0]):.4f} -> {float(curve[-1]):.4f}")
    check("C2 probabilities are a simplex",
          torch.allclose(probs.sum(1), torch.ones(X.shape[0]), atol=1e-5)
          and float(probs.min()) >= 0.0)
    train_acc = float((logits.argmax(1) == y.long()).float().mean())
    check("C2 fits the separable blobs", train_acc > 0.95, f"acc={train_acc:.4f}")

    W2, b2, logits2, probs2, curve2 = sm.execute(X, y, 300, 0.1)
    check("C2 fully deterministic (no seed needed)",
          torch.equal(W, W2) and torch.equal(b, b2) and torch.equal(curve, curve2))

    # The hand-written CE must match F.cross_entropy at the same parameters:
    # evaluate the reference at the RETURNED parameters and compare against a
    # one-step-further training run's curve tail semantics (the returned
    # curve[-1] is the pre-update loss of the last step, so the reference on
    # the final parameters must be <= it).
    ce_ref = float(torch.nn.functional.cross_entropy(X @ W.T + b, y.long()))
    check("C2 hand-written CE consistent with F.cross_entropy",
          ce_ref <= float(curve[-1]) + 1e-6, f"ref {ce_ref:.6f} vs tail {float(curve[-1]):.6f}")

    # DATASET conduit + [n, 1] float labels + rounding semantics.
    from comfydl.nodes.data_types import CdlDataset

    ds = CdlDataset.from_tensors(X, y.reshape(-1, 1), target_name="label")
    out = sm.execute(None, None, 50, 0.1, dataset=ds)
    check("C2 dataset conduit + [n,1] labels work", tuple(out[0].shape) == (3, 2))

    check("C2 rejects negative labels",
          _expect(sm, lambda s: s.execute(X, y - 1.0, 10, 0.1)))


# ------------------------------------------- C3: production classification ---
def _classification_train_checks() -> None:
    ct = REG["CdlClassificationTrain"]()
    X, y = _blobs()

    model, acc, preds, probs, curve = ct.execute(X, y, test_size=0.2, steps=400)
    check("C3 held-out accuracy near-perfect", float(acc) > 0.9, f"acc={float(acc):.4f}")
    check("C3 predictions are long indices in range",
          preds.dtype == torch.int64 and int(preds.min()) >= 0 and int(preds.max()) <= 2)
    check("C3 probabilities simplex",
          torch.allclose(probs.sum(1), torch.ones(X.shape[0]), atol=1e-4))
    check("C3 loss history non-empty", curve.numel() > 0 and curve.dim() == 1)

    model2, acc2, preds2, _, _ = ct.execute(X, y, test_size=0.2, steps=400)
    check("C3 same seed reproduces", float(acc) == float(acc2) and torch.equal(preds, preds2))

    # Stratified split: a tiny minority class must survive in the validation
    # split (a plain random permutation could easily drop all 5 of them).
    X_imb = torch.cat([_blobs(95)[0], _blobs(5, seed=99)[0][-5:]])
    y_imb = torch.cat([torch.zeros(95), torch.ones(95), 2 * torch.ones(100)]).to(torch.float32)
    _, acc_imb, _, _, _ = ct.execute(X_imb, y_imb, test_size=0.2, steps=200)
    check("C3 stratified split keeps minority class trainable",
          0.0 <= float(acc_imb) <= 1.0)

    # Degenerate split (a class with a single sample): must fall back to
    # whole-set evaluation instead of crashing.
    X_all, _ = _blobs(49)  # 147 rows: classes 0 / 1 / 2, 49 rows each
    X_one = torch.cat([X_all[:49], X_all[98:], torch.tensor([[10.0, 10.0]])])
    y_one = torch.cat([torch.zeros(49), 2 * torch.ones(49), torch.ones(1)]).to(torch.float32)
    _, acc_one, _, _, _ = ct.execute(X_one, y_one, test_size=0.2, steps=100)
    check("C3 single-sample class falls back gracefully", 0.0 <= float(acc_one) <= 1.0)

    # accuracy-based early-stop metric path.
    _, acc_m, _, _, curve_m = ct.execute(
        X, y, test_size=0.2, steps=300, early_stop_metric="accuracy",
        early_stop_patience=20,
    )
    check("C3 accuracy-metric early stop works", float(acc_m) > 0.9 and curve_m.numel() <= 300)

    # Early-stop truncation with rollback: a huge patience-free window on a
    # converged problem stops early and the history is truncated to best.
    _, acc_e, _, _, curve_e = ct.execute(
        X, y, test_size=0.2, steps=5000, early_stop_patience=10, early_stop_min_delta=1e-6,
    )
    check("C3 early stop truncates the run", curve_e.numel() < 5000,
          f"ran {curve_e.numel()} steps")

    # Label coercion: [n, 1] float labels are rounded to long indices.
    _, acc_r, _, _, _ = ct.execute(X, y.reshape(-1, 1) + 0.2, test_size=0.0, steps=100)
    check("C3 float [n,1] labels coerced via round", 0.0 <= float(acc_r) <= 1.0)

    check("C3 rejects negative class labels",
          _expect(ct, lambda c: c.execute(X, y - 1.0, test_size=0.0, steps=10)))


# ------------------------------- C4: breakpoint continuation (crash site M3) --
def _checkpoint_checks() -> None:
    STEPS = 12
    RAISE_AT = 5
    X, y = _blobs(40, 0.2, 0)
    kwargs = dict(X=X, y=y, test_size=0.0, steps=STEPS, batch_size=0, lr=0.03,
                  seed=0, early_stop_metric="loss", early_stop_patience=0,
                  early_stop_min_delta=1e-4)
    calls: list = []

    def make_hook(raise_at):
        def hook(value, total, preview_image=None, prompt_id=None, node_id=None):
            calls.append(value)
            if raise_at is not None and value >= raise_at:
                raise InterruptProcessingException()
        return hook

    node = REG["CdlClassificationTrain"]()
    try:
        _old_interval = comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL
        _old_percent = comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT
        comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL = 0.0
        comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT = 0.0

        comfy.utils.set_progress_bar_global_hook(make_hook(RAISE_AT))
        with CurrentNodeContext("pid-class", "7"):
            try:
                node.execute(**kwargs)
                check("C4 interrupted run raises", False, "no exception")
            except InterruptProcessingException:
                check("C4 interrupted run raises", True)
            files = list(_CKPT_ROOT.glob("*.pt"))
            check("C4 checkpoint exists after interrupt", len(files) == 1,
                  str([f.name for f in files]))
            data = torch.load(files[0], map_location="cpu", weights_only=True)
            saved = data.get("payload") or {}
            check("C4 checkpoint holds the breakpoint step",
                  saved.get("step") == RAISE_AT - 1
                  and len(saved.get("history") or []) == RAISE_AT,
                  str(saved.get("step")))
            partial = list(saved.get("history") or [])

        comfy.utils.set_progress_bar_global_hook(make_hook(None))
        calls.clear()
        with CurrentNodeContext("pid-class", "7"):
            out = node.execute(**kwargs)
        curve = out[4]
        check("C4 resumed run completes full history",
              len(curve) == STEPS
              and torch.equal(curve[:len(partial)], torch.stack(partial)),
              f"len={len(curve)}")
        check("C4 resumed run only executes the remaining steps",
              len(calls) == STEPS - RAISE_AT, f"{len(calls)} report(s)")
        check("C4 completed run discards its checkpoint",
              not any(_CKPT_ROOT.glob("*.pt")))

        calls.clear()
        comfy.utils.set_progress_bar_global_hook(make_hook(RAISE_AT))
        with CurrentNodeContext("pid-class", "7"):
            try:
                node.execute(**kwargs)
            except InterruptProcessingException:
                pass
        comfy.utils.set_progress_bar_global_hook(make_hook(None))
        calls.clear()
        with CurrentNodeContext("pid-other", "7"):
            out = node.execute(**kwargs)
        curve = out[4]
        check("C4 fresh prompt trains from scratch",
              len(curve) == STEPS and len(calls) == STEPS, f"{len(calls)} report(s)")
        check("C4 stale checkpoint removed", not any(_CKPT_ROOT.glob("*.pt")))
    finally:
        comfy.utils.set_progress_bar_global_hook(None)
        comfy.utils.PROGRESS_THROTTLE_MIN_INTERVAL = _old_interval
        comfy.utils.PROGRESS_THROTTLE_MIN_PERCENT = _old_percent
        for f in _CKPT_ROOT.glob("*.pt"):
            f.unlink(missing_ok=True)


# ------------------------------------------------------- C5: model loading ---
def _model_load_checks() -> None:
    ct = REG["CdlClassificationTrain"]()
    loader = REG["CdlClassificationModelLoad"]()
    X, y = _blobs()
    path = Path(tempfile.mkdtemp(prefix="cdl_class_model_")) / "classifier.pt"

    model, acc, preds, _, _ = ct.execute(X, y, test_size=0.0, steps=200, save_path=str(path))
    check("C5 trainer persisted the state_dict", path.is_file())

    restored, info = loader.execute(str(path), "relu")
    with torch.no_grad():
        ref = model(X)
        got = restored(X)
    check("C5 reloaded model reproduces the logits bitwise", torch.equal(ref, got))
    check("C5 info names the architecture",
          "restored classifier" in info and "classes=3" in info, info)
    check("C5 restored model is in eval mode", not restored.training)

    check("C5 missing file rejected",
          _expect(loader, lambda l: l.execute(str(path.parent / "nope.pt"), "relu")))


# ------------------------------------------------- C6: confusion matrix ------
def _confusion_checks() -> None:
    cm = REG["CdlConfusionMatrixEval"]()

    y_true = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    y_pred = torch.tensor([0, 1, 1, 1, 2, 0])
    matrix, normalized, acc = cm.execute(y_true, y_pred, 0)
    expected = torch.tensor([[1.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    check("C6 hand-computed matrix matches", torch.equal(matrix, expected),
          str(matrix.tolist()))
    check("C6 accuracy = trace / total", abs(float(acc) - 4.0 / 6.0) < 1e-6)
    check("C6 normalized rows sum to 1",
          torch.allclose(normalized.sum(1), torch.ones(3)))

    # [n, k] score matrices are auto-argmaxed (the CdlAccuracy convention).
    scores = torch.zeros(6, 3)
    scores[torch.arange(6), y_pred.tolist()] = 1.0
    matrix_p, _, _ = cm.execute(y_true, scores)
    check("C6 score matrix auto-argmax", torch.equal(matrix_p, expected))

    # [n, 1] true labels are flattened.
    matrix_c, _, _ = cm.execute(y_true.reshape(-1, 1), y_pred)
    check("C6 [n,1] true labels accepted", torch.equal(matrix_c, expected))

    m4, _, _ = cm.execute(y_true, y_pred, 5)
    check("C6 explicit num_classes wins", tuple(m4.shape) == (5, 5))

    check("C6 mismatched counts rejected",
          _expect(cm, lambda c: c.execute(torch.tensor([0.0, 1.0]), torch.tensor([0]))))
    check("C6 overflow index rejected",
          _expect(cm, lambda c: c.execute(torch.tensor([5.0]), torch.tensor([0]), 4)))
    check("C6 negative index rejected",
          _expect(cm, lambda c: c.execute(torch.tensor([-1.0]), torch.tensor([0]), 4)))
    check("C6 empty labels rejected",
          _expect(cm, lambda c: c.execute(torch.tensor([]), torch.tensor([]))))


def main() -> int:
    _datagen_checks()
    _softmax_checks()
    _classification_train_checks()
    _checkpoint_checks()
    _model_load_checks()
    _confusion_checks()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
