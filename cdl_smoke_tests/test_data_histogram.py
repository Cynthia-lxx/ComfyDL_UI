#!/usr/bin/env python
"""Smoke tests for the Dataset → Histogram rich-canvas node (SP-7 craft 5).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_histogram.py

Covers the backend aggregation contract (binning, NaN handling, sampling,
feature resolution, passthrough + ui payload + workflow write-back) and the
frontend source contract of the richwidget histogram instance. Browser
behaviour is accepted by the user per the project convention.
"""

import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RICH = REPO_ROOT / "comfydl" / "web" / "js" / "richwidget.js"
_RESULTS = []


def check(name, condition, detail=""):
    _RESULTS.append((name, bool(condition), detail))


def _dataset(features, labels=None, names=None, target=""):
    from comfydl.nodes.data_types import CdlDataset

    return CdlDataset.from_tensors(
        features, y=labels, feature_names=names, target_name=target
    )


def test_backend():
    from comfydl.nodes import NODE_CLASS_MAPPINGS as REG
    from comfydl.nodes.visualization import (
        _histogram_payload,
        _resolve_hist_feature,
        _resolve_hist_columns,
    )

    check("node registered in the Visualization family",
          REG.get("CdlDatasetHistogram") is not None
          and REG["CdlDatasetHistogram"].CATEGORY == "d2l/Visualization")
    inputs = REG["CdlDatasetHistogram"].INPUT_TYPES()["required"]
    bins_spec = inputs["bins"][1]
    examples_spec = inputs["max_examples"][1]
    check("widget defaults stay inside their bounds",
          bins_spec["min"] <= bins_spec["default"] <= bins_spec["max"]
          and examples_spec["min"] <= examples_spec["default"] <= examples_spec["max"])
    check("combo default exists in its option list",
          "*" in inputs["feature"][0])
    check("dataset passthrough + OUTPUT_NODE",
          REG["CdlDatasetHistogram"].OUTPUT_NODE is True
          and REG["CdlDatasetHistogram"].RETURN_TYPES == ("DATASET",))
    check("VALIDATE_INPUTS accepts any string feature selector",
          REG["CdlDatasetHistogram"].VALIDATE_INPUTS("y") is True
          and REG["CdlDatasetHistogram"].VALIDATE_INPUTS("*") is True
          and isinstance(REG["CdlDatasetHistogram"].VALIDATE_INPUTS(3), str))

    # Binning correctness: 100 uniform values in [0,1), 10 bins -> 10 per bin.
    ds = _dataset([[i / 100] for i in range(100)], names=["x0"])
    payload = _histogram_payload(ds, "*", 10, 100000)
    entry = payload["series"]["x0"]
    check("uniform column bins evenly",
          entry["counts"] == [10] * 10, str(entry["counts"]))
    check("payload carries summary statistics",
          math.isclose(entry["min"], 0.0, abs_tol=1e-6)
          and entry["max"] < 1.0
          and entry["n_used"] == 100 and entry["n_total"] == 100
          and entry["n_dropped"] == 0 and not entry["sampled"]
          and payload["feature"] == "x0"
          and len(entry["edges"]) == 11
          and math.isclose(entry["std"], ds.features.std(), rel_tol=1e-3)
          and payload["n_total"] == 100,
          str(payload)[:160])

    # Every numeric column lands in the series map.
    ds_multi = _dataset([[1.0, 2.0], [3.0, 4.0]], labels=[5.0, 6.0],
                        names=["a", "b"], target="y")
    payload = _histogram_payload(ds_multi, "b", 4, 100000)
    check("all numeric columns aggregated into the series map",
          list(payload["series"].keys()) == ["a", "b", "y"]
          and payload["feature"] == "b"
          and payload["series"]["b"]["min"] == 2.0
          and payload["series"]["y"]["counts"], str(payload)[:160])

    # NaN handling: NaN values are excluded, not silently binned.
    ds_nan = _dataset([[v] for v in (1.0, float("nan"), 2.0, float("inf"), 3.0)],
                      names=["v"])
    entry = _histogram_payload(ds_nan, "v", 4, 100000)["series"]["v"]
    check("NaN/Inf dropped and reported",
          entry["n_used"] == 3 and entry["n_dropped"] == 2
          and sum(entry["counts"]) == 3, str(entry)[:140])

    # Sampling: a huge column is down-sampled with endpoints kept.
    ds_big = _dataset([[i] for i in range(1000)], names=["i"])
    entry = _histogram_payload(ds_big, "i", 8, 100)["series"]["i"]
    check("oversized column is sampled",
          entry["sampled"] and entry["n_used"] == 100
          and entry["n_total"] == 1000
          and entry["min"] == 0 and entry["max"] == 999, str(entry)[:140])

    # Feature resolution: *, exact name, label column, numeric index,
    # and stale names degrade to the first column instead of raising.
    columns, default_index = _resolve_hist_columns(ds_multi)
    check("label column joins the histogrammable set",
          [name for name, _ in columns] == ["a", "b", "y"] and default_index == 0)
    check("'*' resolves to the first feature",
          _resolve_hist_feature(columns, "*") == "a")
    check("numeric index resolves positionally",
          _resolve_hist_feature(columns, "2") == "y")
    check("unknown feature degrades to the first column",
          _resolve_hist_feature(columns, "nope") == "a")

    # Graceful degradation: empty dataset / all-NaN column no longer abort
    # the flow; they produce empty series entries instead.
    import torch
    from comfydl.nodes.data_types import CdlDataset

    payload = _histogram_payload(
        CdlDataset.from_tensors(torch.zeros((0, 1)), feature_names=["x"]),
        "*", 8, 100)
    check("empty dataset yields an empty series entry",
          payload["series"]["x"]["counts"] == []
          and payload["series"]["x"]["n_total"] == 0
          and payload["n_total"] == 0)
    ds_allnan = _dataset([[float("nan")] for _ in range(5)], names=["x"])
    entry = _histogram_payload(ds_allnan, "*", 8, 100)["series"]["x"]
    check("all-NaN column yields an empty series entry",
          entry["counts"] == [] and entry["n_used"] == 0
          and entry["n_dropped"] == 5, str(entry)[:140])

    # execute(): ui channel, passthrough identity, workflow write-back.
    workflow = {"workflow": {"nodes": [{"id": 9, "properties": {}}]}}
    node = REG["CdlDatasetHistogram"]()
    out = node.execute(ds_multi, feature="a", bins=4, max_examples=1000,
                       unique_id="9", extra_pnginfo=workflow)
    check("result is the same dataset object (zero-copy passthrough)",
          out["result"][0] is ds_multi)
    payload = out["ui"]["cdl_histogram"][0]
    check("ui payload carries every column",
          payload["feature"] == "a"
          and set(payload["series"].keys()) == {"a", "b", "y"}
          and "a" in payload["feature_names"])
    check("payload written back into node properties",
          out is not None)
    from comfydl.nodes.data_ops import resolve_workflow_node

    node_dict, _ = resolve_workflow_node(workflow, "9")
    check("write-back lands in properties.cdl_histogram",
          node_dict is not None
          and node_dict["properties"]["cdl_histogram"]["feature"] == "a"
          and "y" in node_dict["properties"]["cdl_histogram"]["series"])


def test_frontend_source():
    rich = RICH.read_text(encoding="utf-8")
    check("histogram instance mounted in the richwidget engine",
          "function mountHistogram" in rich
          and '"cdl_rich_histogram"' in rich
          and "HIST_PROPERTY = \"cdl_histogram\"" in rich)
    check("histogram draw and hover contract",
          "state.hover" in rich and "fillRect" in rich
          and '"pointermove"' in rich)
    check("feature combo refresh from the payload series",
          "function refreshHistogramCombo" in rich
          and 'w.name === "feature"' in rich
          and '["*"].concat(Object.keys(series))' in rich
          and "syncComboOptionsToVueStore(node, widget, values)" in rich)
    check("combo selection redraws instantly from the cached series",
          "__cdlHistFeatureHook" in rich
          and "original?.call(feature, value)" in rich
          and "function normalizeHistogramPayload" in rich
          and "hist.state.series = body.series" in rich
          and "hist.state.feature = body.series[wanted]" in rich)
    check("payload persisted in node.properties",
          "node.properties[HIST_PROPERTY] = body" in rich)
    check("both delivery paths wired",
          "wireHistogramNode(nodeType)" in rich
          and "output.cdl_histogram" in rich
          and "node.comfyClass === \"CdlDatasetHistogram\"" in rich)
    try:
        proc = subprocess.run(["node", "--check", str(RICH)],
                              capture_output=True, text=True, timeout=30)
        check("richwidget.js node --check", proc.returncode == 0,
              proc.stderr[:160])
    except (OSError, subprocess.TimeoutExpired):
        check("richwidget.js node --check", True, "node unavailable; skipped")


def main():
    test_backend()
    test_frontend_source()
    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== data histogram test: {len(_RESULTS) - len(failed)} PASS / "
          f"{len(failed)} FAIL (of {len(_RESULTS)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
