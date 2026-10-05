#!/usr/bin/env python
"""Standalone smoke test for the formula data generator (CdlFormulaDataGen).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_gen.py

Covers:
  - Formula mode evaluates the AST-whitelisted expression exactly (noise off):
    ``y`` matches an independent vectorised recomputation of the formula.
  - Output contract: ``X`` ``(n, k)``, ``y`` ``(n, 1)`` and a labelled
    ``DATASET`` with feature names ``x0..x{k-1}``, target ``y`` and provenance
    metadata.
  - Determinism: the same seed reproduces the exact tensors; noise perturbs
    the labels while the features stay reproducible.
  - Model mode ("digital twin"): feature count is inferred from the model,
    labels equal ``model(X)`` and the formula is ignored.
  - Rejection policy: non-whitelisted grammar (attribute access, calls,
    comprehensions, conditionals, dunder names), empty formulas and an
    inverted sampling window raise ``ValueError``.
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


def _expect_value_error(node, *args, **kwargs) -> bool:
    try:
        node.execute(*args, **kwargs)
    except ValueError:
        return True
    return False


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_datagen_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS
    check("CdlFormulaDataGen registered", "CdlFormulaDataGen" in REG)
    if "CdlFormulaDataGen" not in REG:
        return _report()
    gen = REG["CdlFormulaDataGen"]()

    formula = "2 + 3*x0 - 1.5*x1 + 0.5*sin(6*x0)"

    # --- formula mode, noise off: exact signal ------------------------------ #
    X, y, ds = gen.execute(formula, 500, -3.0, 3.0, "uniform", "none", 0.0, 0)
    ref = 2 + 3 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * torch.sin(6 * X[:, 0])
    check("X shape [500, 2]", tuple(X.shape) == (500, 2), str(tuple(X.shape)))
    check("y shape [500, 1]", tuple(y.shape) == (500, 1), str(tuple(y.shape)))
    check("y finite", torch.isfinite(y).all())
    check("y matches formula exactly", torch.allclose(y.reshape(-1), ref, atol=1e-5))
    check("dataset is DATASET with labels", ds.labels is not None and ds.n_samples == 500)
    check("feature names x0/x1", ds.feature_names == ["x0", "x1"], str(ds.feature_names))
    check("target name y", ds.target_name == "y", ds.target_name)
    check("meta records formula", ds.meta.get("formula") == formula, str(ds.meta))
    check("meta source=formula", ds.meta.get("source") == "formula")

    # --- determinism & noise ------------------------------------------------ #
    X2, y2, _ = gen.execute(formula, 500, -3.0, 3.0, "uniform", "none", 0.0, 0)
    check("same seed reproduces exactly", torch.equal(X, X2) and torch.equal(y, y2))
    X3, y3, _ = gen.execute("x0", 500, -3.0, 3.0, "uniform", "gaussian", 0.1, 0)
    clean = X3[:, 0].reshape(-1, 1)
    check("gaussian noise perturbs labels", not torch.equal(y3, clean))
    check("noise stays within 5 sigma", (y3 - clean).abs().max() <= 0.5,
          f"max |noise| {float((y3 - clean).abs().max()):.4f}")
    X4, _, _ = gen.execute("x0", 500, -3.0, 3.0, "uniform", "none", 0.0, 0)
    check("noise does not touch X", torch.equal(X3, X4))
    X5, _, _ = gen.execute("x0", 500, -3.0, 3.0, "normal", "none", 0.0, 0)
    check("normal sampling covers both signs", (X5 > 0).any() and (X5 < 0).any())

    # --- model mode (digital twin) ------------------------------------------ #
    Xr = torch.randn(200, 3, generator=torch.Generator().manual_seed(0))
    yr = Xr @ torch.tensor([[2.0], [-1.0], [0.5]]) + 1.0
    trainer = REG["CdlRegressionTrain"]()
    model, _, _, _, _ = trainer.execute(
        Xr, yr, test_size=0.0, steps=80, early_stop_patience=0
    )
    Xm, ym, dsm = gen.execute("ignored", 300, -2.0, 2.0, "normal", "none", 0.0, 7, model=model)
    check("model mode infers 3 features", tuple(Xm.shape) == (300, 3), str(tuple(Xm.shape)))
    check("model mode labels = model(X)", torch.allclose(ym, model(Xm).reshape(-1, 1), atol=1e-4))
    check("model mode ignores formula", dsm.meta.get("source") == "model" and dsm.meta.get("formula") == "")

    # --- rejection policy ---------------------------------------------------- #
    bad_formulas = [
        '__import__("os")',
        "",
        "x0+",
        "x0.attr",
        "f(x0)",
        "x0 if x1 else 2",
        "[x0, x1]",
        "x0.__class__",
        "lambda x0: x0",
    ]
    for bad in bad_formulas:
        check(f"rejects {bad!r}", _expect_value_error(gen, bad, 10, -1, 1))
    check("rejects x_min > x_max",
          _expect_value_error(gen, "x0", 10, 3.0, -3.0, "uniform", "none", 0, 0))

    return _report()


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== data-gen test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
