#!/usr/bin/env python
"""Standalone smoke test for the M3 operator-graph probe (comfy/profiling/opgraph).

Run:  penv\Scripts\python.exe cdl_smoke_tests\test_profiling_opgraph.py

Covers:
  - probe mechanics on the golden regression workflow (the Tabular Regression:
    Production template prompt): every node probed, the trainer census
    contains real compute ops (addmm / mse_loss) with positive FLOPs, and the
    data-dependent scalar reads are canned but visible.
  - FLOPs precision: a Model Forward over a known linear layer yields exactly
    2*M*N FLOPs (the FlopCounterMode 2MNK contract).
  - determinism & caching: same prompt -> same report object (cache hit);
    the fallback path (unknown class_type / no prompt) degrades gracefully.
  - route integration through a real aiohttp TestServer (pitfall 23: the
    Application must carry client_max_size=100MB) - /comfydl/profiling/opgraph
    and its /api twin both answer 200 with a valid report.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness

_RESULTS: list = []


def check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond), detail))


FORMULA = "2 + 3*x0 - 1.5*x1 + 0.5*x2"
TRAINER = {
    "test_size": 0.2, "standardize": "yes", "hidden": "", "activation": "relu",
    "loss": "mse", "steps": 60, "batch_size": 0, "lr": 0.05, "seed": 0,
    "early_stop_patience": 30, "early_stop_min_delta": 0.0, "save_path": "",
}


def _golden_regression_prompt():
    return {
        "2": {"class_type": "CdlFormulaDataGen", "inputs": {
            "formula": FORMULA, "num_examples": 800, "x_min": -3.0, "x_max": 3.0,
            "sampling": "uniform", "noise": "gaussian", "noise_std": 0.1, "seed": 0}},
        "5": {"class_type": "CdlRegressionTrain", "inputs": {
            "X": ["2", 0], "y": ["2", 1], "dataset": ["2", 2], **TRAINER}},
        "4": {"class_type": "CdlModelSave", "inputs": {
            "model": ["5", 0], "path": "output/regression_model.pt"}},
        "7": {"class_type": "CdlModelForward", "inputs": {
            "model": ["5", 0], "tensor": ["2", 0]}},
        "8": {"class_type": "CdlTensorsToDataset", "inputs": {
            "X": ["7", 0], "y": ["2", 1]}},
        "9": {"class_type": "CdlWriteCSV", "inputs": {
            "dataset": ["8", 0], "file": "opgraph_test_predictions.csv", "index": True}},
    }


def main():
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_opgraph_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    from comfy.profiling import opgraph

    check("FlopCounterMode importable",
          hasattr(torch.utils.flop_counter, "FlopCounterMode"))

    prompt = _golden_regression_prompt()
    report = opgraph.probe_workflow(prompt)
    check("report version is 1", report.get("version") == 1)
    totals = report["totals"]
    check("all nodes probed",
          totals["probed"] == totals["nodes"] and totals["nodes"] == 6,
          json.dumps(totals))
    by_id = {n["id"]: n for n in report["nodes"]}
    trainer = by_id.get("5")
    check("trainer probed",
          trainer is not None and trainer["status"] == "probed",
          str(trainer and trainer["error"]))
    if trainer and trainer["status"] == "probed":
        ops = {row["op"]: row for row in trainer["ops"]}
        check("trainer census has addmm", "aten.addmm.default" in ops,
              str(sorted(ops))[:200])
        check("trainer census has mse_loss", "aten.mse_loss.default" in ops)
        check("trainer FLOPs positive", (trainer["total_flops"] or 0) > 0,
              str(trainer["total_flops"]))
        check("trainer data-dependent reads canned",
              trainer["data_dependent_reads"] > 0, str(trainer["data_dependent_reads"]))
        check("trainer probe under op limit", trainer["ops_total"] < opgraph.OPS_LIMIT)
        check("census sorted by flops", trainer["ops"] == sorted(
            trainer["ops"], key=lambda r: (-r["flops"], -r["count"], r["op"])))

    forward = by_id.get("7")
    check("forward probed",
          forward is not None and forward["status"] == "probed",
          str(forward and forward["error"]))

    # V3 nodes (io.ComfyNode) return io.NodeOutput; the probe must unwrap it
    # the way the host does, or downstream type checks see the wrapper instead
    # of the value (2026-10-06: the Language Model template probed 2/12 because
    # TextEncode received the NodeOutput wrapper instead of the Vocab).
    vocab_prompt = {
        "1": {"class_type": "TextVocabBuild", "inputs": {
            "corpus": "the quick brown fox jumps over the lazy dog",
            "level": "char", "min_freq": 1}},
        "2": {"class_type": "TextEncode", "inputs": {
            "vocab": ["1", 0], "text": "the fox"}},
    }
    v3_report = opgraph.probe_workflow(vocab_prompt)
    v3_by_id = {n["id"]: n for n in v3_report["nodes"]}
    encode = v3_by_id.get("2")
    check("V3 NodeOutput unwrapped: TextEncode probed",
          encode is not None and encode["status"] == "probed",
          str(encode and encode["error"]))
    check("V3 chain probed 2/2",
          v3_report["totals"]["probed"] == 2, json.dumps(v3_report["totals"]))

    # FLOPs precision: Model Forward over a known linear layer = 2*M*N.
    from comfydl.nodes.regression_train import _Regressor
    model = _Regressor(10, 20, (), "relu")
    x = torch.randn(1, 10)
    expected = 2 * 1 * 10 * 20
    fcm = torch.utils.flop_counter.FlopCounterMode(display=False)
    with fcm:
        with torch.no_grad():
            model(x)
    check("2MNK contract: linear flops exact", fcm.get_total_flops() == expected,
          "%d vs %d" % (fcm.get_total_flops(), expected))

    report_a = opgraph.analyse_workflow(prompt)
    report_b = opgraph.analyse_workflow(prompt)
    check("cache hit returns the same report object", report_a is report_b)
    prompt2 = _golden_regression_prompt()
    prompt2["2"]["inputs"]["num_examples"] = 500
    report3 = opgraph.analyse_workflow(prompt2)
    check("changed graph re-probes", report3 is not report_a)

    empty = opgraph.probe_workflow({})
    check("empty prompt -> error report", empty.get("error") == "no_graph")
    junk = opgraph.probe_workflow({"1": {"class_type": "NoSuchNodeType", "inputs": {}}})
    check("unknown class_type falls back",
          junk["nodes"][0]["status"] == "fallback"
          and "not registered" in (junk["nodes"][0]["error"] or ""))
    check("fallback node has probed == 0", junk["totals"]["probed"] == 0)

    from aiohttp import test_utils, web
    import app.profiling_routes as profiling_routes

    async def _route_checks():
        app = web.Application(client_max_size=100 * 1024**2)
        app.add_routes(profiling_routes.routes)
        client = test_utils.TestClient(test_utils.TestServer(app))
        await client.start_server()
        try:
            resp = await client.post("/comfydl/profiling/opgraph",
                                     json={"prompt": prompt})
            check("route: POST /opgraph 200", resp.status == 200)
            data = await resp.json()
            check("route: report has totals + nodes",
                  "totals" in data and "nodes" in data and data.get("version") == 1)
            resp3 = await client.post("/comfydl/profiling/opgraph", data="not-json")
            check("route: bad JSON 400", resp3.status == 400)
        finally:
            await client.close()

    asyncio.run(_route_checks())

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = "  %s  %s" % (flag, name)
        if detail and not ok:
            line += "  (%s)" % detail
        print(line)
    print("\n=== opgraph test: %d PASS / %d FAIL (of %d) ==="
          % (passed, len(_RESULTS) - passed, len(_RESULTS)))
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
