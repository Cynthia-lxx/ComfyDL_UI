#!/usr/bin/env python
"""Standalone smoke test for the linear-regression training node (batch 4).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_regression.py

Covers:
  - CdlLinRegTrain trains from raw X / y TENSORs and from a labelled DATASET,
    producing finite w / b, a decreasing loss_history and a y_hat matching the
    stateless CdlLinReg + CdlSquaredLoss nodes (the documented verification
    closure).
  - The closed loop:  CdlLinRegTrain --w,b--> CdlLinReg --y_hat--> CdlSquaredLoss
    reproduces the training loss_history[-1] up to numerical tolerance.
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _problem(seed: int = 0):
    """Deterministic ``y = 2*x0 - 3*x1 + 1`` regression problem (48 samples)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(48, 2, generator=g)
    y = 2.0 * X[:, :1] - 3.0 * X[:, 1:] + 1.0
    return X, y


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_reg_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS
    check("CdlLinRegTrain registered", "CdlLinRegTrain" in REG)
    check("CdlLinReg registered", "CdlLinReg" in REG)
    check("CdlSquaredLoss registered", "CdlSquaredLoss" in REG)
    if "CdlLinRegTrain" not in REG:
        return _report()

    X, y = _problem()

    # --- train from raw TENSORs ------------------------------------------- #
    w, b, curve, y_hat = REG["CdlLinRegTrain"]().execute(
        X, y, num_steps=200, batch_size=16, lr=0.1, seed=0
    )
    check("w shape [f,1]", tuple(w.shape) == (2, 1), str(tuple(w.shape)))
    check("b shape scalar", tuple(b.shape) in ((1,), (1, 1)), str(tuple(b.shape)))
    check("loss_history 1-D len 200", curve.dim() == 1 and curve.numel() == 200)
    check("y_hat shape [48,1]", tuple(y_hat.shape) == (48, 1), str(tuple(y_hat.shape)))
    check("w finite", torch.isfinite(w).all())
    check("b finite", torch.isfinite(b).all())
    check("loss finite", torch.isfinite(curve).all())
    check("loss decreased >10x", float(curve[-1]) < float(curve[0]) / 10.0,
          f"{float(curve[0]):.4f} -> {float(curve[-1]):.4f}")
    check("loss nearly converged", float(curve[-1]) < 1e-2,
          f"final loss {float(curve[-1]):.6f}")
    check("outputs detached", not any(t.requires_grad for t in (w, b, curve, y_hat)))

    # --- closed verification loop ----------------------------------------- #
    # CdlLinReg reproduces y_hat from w/b; CdlSquaredLoss must match the curve tail.
    y_hat2 = REG["CdlLinReg"]().execute(X, w, b)[0]
    check("CdlLinReg matches trained y_hat", torch.allclose(y_hat2, y_hat, atol=1e-5))

    sq = REG["CdlSquaredLoss"]().execute(y_hat2, y)[0]  # (y_hat - y)^2 / 2
    loop_loss = float(sq.mean())
    check("loop loss matches training tail", abs(loop_loss - float(curve[-1])) < 1e-4,
          f"loop={loop_loss:.6f} train={float(curve[-1]):.6f}")

    # --- train from a labelled DATASET ------------------------------------ #
    from comfydl.nodes.data_types import CdlDataset

    ds = CdlDataset.from_tensors(X, y, feature_names=["x0", "x1"], target_name="y")
    w2, b2, curve2, y_hat2b = REG["CdlLinRegTrain"]().execute(
        X, y, num_steps=150, batch_size=24, lr=0.1, seed=1, dataset=ds
    )
    check("dataset path: w finite", torch.isfinite(w2).all())
    check("dataset path: loss decreased", float(curve2[-1]) < float(curve2[0]) / 10.0,
          f"{float(curve2[0]):.4f} -> {float(curve2[-1]):.4f}")
    check("dataset path: converged", float(curve2[-1]) < 1e-2,
          f"final loss {float(curve2[-1]):.6f}")

    return _report()


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== regression test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
