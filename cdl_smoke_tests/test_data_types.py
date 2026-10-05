#!/usr/bin/env python
"""Standalone smoke test for the DATASET type and the dataset adapters.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_types.py

Reuses the smoke harness bootstrap so the real node registry (including the
comfydl submodule) is loaded exactly as the server would, then exercises the
new DATASET type end-to-end: registration, tensor round-trip, pandas
DataFrame conversion, DataLoader bridging, and the teal slot-colour injection.
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
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_dt_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402
    from comfydl.nodes.data_types import CdlDataset  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS

    # 1. registration -------------------------------------------------------
    for nid in ("CdlTensorsToDataset", "CdlDatasetToTensors", "CdlDatasetToLoader"):
        check(f"registered:{nid}", nid in REG, "" if nid in REG else "missing")

    # 2. tensor round-trip through the adapters -----------------------------
    import torch

    X = torch.randn(10, 3)
    y = torch.randn(10)
    ds = REG["CdlTensorsToDataset"]().execute(X, y)[0]
    check("TensorsToDataset -> CdlDataset", isinstance(ds, CdlDataset))
    check("feature_names auto", ds.feature_names == ["x0", "x1", "x2"], str(ds.feature_names))
    X2, y2 = REG["CdlDatasetToTensors"]().execute(ds)
    check("round-trip X", torch.allclose(X, X2))
    check("round-trip y", torch.allclose(y, y2))

    # unlabelled dataset: y must come back as an empty tensor, not crash
    ds_u = REG["CdlTensorsToDataset"]().execute(X)[0]
    _, yu = REG["CdlDatasetToTensors"]().execute(ds_u)
    check("unlabelled y is empty tensor", yu.numel() == 0)

    # 3. pandas DataFrame conversion (optional dependency) -------------------
    try:
        import pandas as pd

        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "t": [0.1, 0.2, 0.3]}
        )
        d2 = CdlDataset.from_dataframe(df, target="t")
        check("from_dataframe labels", d2.labels.shape == (3,))
        check("from_dataframe features", d2.features.shape == (3, 2))
        check("from_dataframe target_name", d2.target_name == "t")
        back = d2.to_dataframe()
        check("to_dataframe columns", list(back.columns) == ["a", "b", "t"])
    except ImportError:
        check("from_dataframe (pandas)", True, "pandas absent - skipped")

    # 4. DataLoader bridging ------------------------------------------------
    loader = REG["CdlDatasetToLoader"]().execute(ds, 4, True)[0]
    batch = next(iter(loader))
    feats = batch[0] if isinstance(batch, (list, tuple)) else batch
    check("loader batch feature dim", feats.shape[1] == 3, str(tuple(feats.shape)))

    # 5. teal slot colour --------------------------------------------------
    src = (REPO_ROOT / "app" / "frontend_patch.py").read_text()
    check("color const #1ABC9C in source", '"#1ABC9C"' in src)
    check("patch fn defined", "_patch_dataset_slot_colour" in src)
    check("patch registered in list",
          '"DATASET slot colour", _patch_dataset_slot_colour' in src)

    # functional injection into a synthetic 6-palette settingStore asset
    try:
        from app import frontend_patch as fp

        assets = sandbox / "assets"
        assets.mkdir()
        (assets / "settingStore-test.js").write_text(
            "X" + 'node_slot:{"a":"#000"}' * 6 + "Y"
        )
        before = (assets / "settingStore-test.js").read_text().count(",DATASET:`#1ABC9C`")
        ok = fp._patch_dataset_slot_colour(assets)
        after = (assets / "settingStore-test.js").read_text().count(",DATASET:`#1ABC9C`")
        check("color injected into 6 palettes", after - before == 6, f"inserted {after - before}")
        check("color injection idempotent", fp._patch_dataset_slot_colour(assets) is False)
    except Exception as e:  # pragma: no cover
        check("color functional injection", False, f"{type(e).__name__}: {e}")

    # report ----------------------------------------------------------------
    failed = [r for r in _RESULTS if not r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
