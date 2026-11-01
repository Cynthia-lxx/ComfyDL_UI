#!/usr/bin/env python
"""Standalone smoke test for the Data List semantics family (data_list.py).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_list.py

Reuses the smoke harness bootstrap so the real node registry is loaded exactly
as the server would, then exercises the native-list channel end-to-end:
registration + contract attributes (OUTPUT_IS_LIST / INPUT_IS_LIST), the
split -> merge round-trip, drop_last truncation, shuffle determinism (local
generator, reproducible per seed), GetItem boundaries, zero-copy views for
unshuffled batches, estimator registration, and the error-message contract.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_dlist_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS

    # 1. registration + list contract ---------------------------------------
    for nid in ("CdlDatasetSplitBatches", "CdlDatasetFromList", "CdlDatasetGetItem"):
        check(f"registered:{nid}", nid in REG, "" if nid in REG else "missing")
    check("SplitBatches OUTPUT_IS_LIST",
          getattr(REG["CdlDatasetSplitBatches"], "OUTPUT_IS_LIST", None) == (True,))
    check("FromList INPUT_IS_LIST",
          getattr(REG["CdlDatasetFromList"], "INPUT_IS_LIST", None) is True)
    check("GetItem is not a list node",
          not getattr(REG["CdlDatasetGetItem"], "OUTPUT_IS_LIST", False)
          and not getattr(REG["CdlDatasetGetItem"], "INPUT_IS_LIST", False))

    # 2. split -> merge round-trip -------------------------------------------
    import torch

    from comfydl.nodes.data_types import CdlDataset

    X = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 4.0
    y = X[:, 0] * 2.0 - 1.0
    ds = CdlDataset.from_tensors(X, y, feature_names=["a", "b", "c"],
                                 target_name="y", meta={"source": "test"})

    split = REG["CdlDatasetSplitBatches"]().execute(ds, 4)[0]
    check("split emits a python list", isinstance(split, list), str(type(split)))
    check("split batch count", len(split) == 3, f"{len(split)} batches")
    check("split batch sizes", [b.n_samples for b in split] == [4, 4, 2],
          str([b.n_samples for b in split]))
    merged = REG["CdlDatasetFromList"]().execute(list(split))[0]
    check("round-trip rows", merged.n_samples == 10, str(merged.n_samples))
    check("round-trip features", torch.equal(merged.features, ds.features))
    check("round-trip labels", torch.equal(merged.labels, ds.labels))
    check("round-trip schema",
          merged.feature_names == ["a", "b", "c"] and merged.target_name == "y")
    check("round-trip meta provenance", merged.meta.get("merged", {}).get("items") == 3)

    # 3. drop_last truncation -------------------------------------------------
    split_drop = REG["CdlDatasetSplitBatches"]().execute(ds, 4, True)[0]
    check("drop_last batch count", len(split_drop) == 2, str(len(split_drop)))
    check("drop_last rows", sum(b.n_samples for b in split_drop) == 8, "expected 8 of 10")
    check("drop_last uniform sizes",
          all(b.n_samples == 4 for b in split_drop))
    # merge only what the split emitted (8 rows, in order)
    merged_drop = REG["CdlDatasetFromList"]().execute(list(split_drop))[0]
    check("drop_last merge prefix",
          torch.equal(merged_drop.features, ds.features[:8]))

    # 4. shuffle determinism --------------------------------------------------
    s1 = REG["CdlDatasetSplitBatches"]().execute(ds, 10, False, True, 7)[0]
    s2 = REG["CdlDatasetSplitBatches"]().execute(ds, 10, False, True, 7)[0]
    s3 = REG["CdlDatasetSplitBatches"]().execute(ds, 10, False, True, 8)[0]
    check("shuffle same seed reproducible", torch.equal(s1[0].features, s2[0].features))
    check("shuffle different seed differs",
          not torch.equal(s1[0].features, s3[0].features))
    check("shuffle covers all rows",
          torch.equal(
              REG["CdlDatasetFromList"]().execute(list(s1))[0].features.sort(dim=0).values,
              ds.features.sort(dim=0).values))
    check("split meta carries provenance",
          s1[0].meta.get("split", {}).get("seed") == 7
          and s1[0].meta.get("split", {}).get("shuffle") is True)

    # 5. zero-copy views (unshuffled) -----------------------------------------
    raw = REG["CdlDatasetSplitBatches"]().execute(ds, 4)[0]
    check("unshuffled batches are zero-copy views",
          raw[0].features.data_ptr() == ds.features.data_ptr())

    # 6. GetItem ----------------------------------------------------------------
    row = REG["CdlDatasetGetItem"]().execute(ds, 3)[0]
    check("getitem 1 row", row.n_samples == 1)
    check("getitem row content",
          torch.equal(row.features, ds.features[3:4]))
    check("getitem meta index", row.meta.get("row") == 3)
    check("getitem schema carried",
          row.feature_names == ["a", "b", "c"] and row.target_name == "y")
    try:
        REG["CdlDatasetGetItem"]().execute(ds, 10)
    except ValueError as exc:
        check("getitem out-of-range message", "n_samples" in str(exc), str(exc))
    else:
        check("getitem out-of-range message", False, "no error raised")
    try:
        REG["CdlDatasetGetItem"]().execute(ds, -1)
    except ValueError:
        check("getitem negative rejected", True)
    else:
        check("getitem negative rejected", False, "no error raised")

    # 7. FromList error contract ------------------------------------------------
    ds_off = CdlDataset.from_tensors(X[:, :2], y, feature_names=["a", "b"])
    try:
        REG["CdlDatasetFromList"]().execute([ds, ds_off])
    except ValueError as exc:
        check("mismatch message names columns", "'a', 'b'" in str(exc), str(exc)[:120])
    else:
        check("mismatch message names columns", False, "no error raised")
    try:
        REG["CdlDatasetFromList"]().execute([])
    except ValueError as exc:
        check("empty list message", "empty dataset list" in str(exc), str(exc)[:120])
    else:
        check("empty list message", False, "no error raised")

    # 8. estimator registration -------------------------------------------------
    from comfy.profiling.estimators import ESTIMATORS

    for nid in ("CdlDatasetSplitBatches", "CdlDatasetFromList", "CdlDatasetGetItem"):
        check(f"estimator:{nid}", nid in ESTIMATORS,
              "" if nid in ESTIMATORS else "missing")

    # report ----------------------------------------------------------------
    failed = [r for r in _RESULTS if not r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
