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

    # P0 guardrails: oversized string inputs never reach execute().
    huge = "x" * (opgraph.MAX_NODE_INPUT_BYTES + 1)
    guard_prompt = {
        "1": {"class_type": "TextVocabBuild", "inputs": {
            "corpus": huge, "level": "char", "min_freq": 1}},
        "2": {"class_type": "TextEncode", "inputs": {
            "vocab": ["1", 0], "text": "the fox"}},
    }
    guard_report = opgraph.probe_workflow(guard_prompt)
    g_by_id = {n["id"]: n for n in guard_report["nodes"]}
    check("P0: oversized node falls back without executing",
          g_by_id["1"]["status"] == "fallback"
          and "input too large for probe" in (g_by_id["1"]["error"] or ""),
          str(g_by_id["1"]["error"]))
    check("P0: guard cascades downstream",
          "upstream" in (g_by_id["2"]["error"] or ""),
          str(g_by_id["2"]["error"]))

    whole = {
        str(i): {"class_type": "TextVocabBuild", "inputs": {
            "corpus": "y" * (opgraph.MAX_TOTAL_INPUT_BYTES // 2 + 1024),
            "level": "char", "min_freq": 1}}
        for i in range(2)
    }
    whole_report = opgraph.probe_workflow(whole)
    check("P0: whole-graph cap returns early error report",
          whole_report["error"] is not None
          and "graph string inputs too large" in whole_report["error"]
          and whole_report["totals"]["probed"] == 0,
          str(whole_report["error"]))

    dl_report = opgraph.probe_workflow(prompt, deadline=0.0)
    check("P0: cooperative deadline marks nodes without executing",
          dl_report["totals"]["probed"] == 0
          and all("probe deadline exceeded" in (n["error"] or "")
                  for n in dl_report["nodes"]),
          str(dl_report["nodes"][0]["error"] if dl_report["nodes"] else "?"))

    from aiohttp import test_utils, web
    import app.profiling_routes as profiling_routes

    async def _route_checks():
        app = web.Application(client_max_size=100 * 1024**2)
        app.add_routes(profiling_routes.routes)
        client = test_utils.TestClient(test_utils.TestServer(app))
        await client.start_server()
        try:
            # P1: without the dangerous header the route serves the assembled
            # equivalent graph - zero execution, rule census + M2 ledger.
            resp = await client.post("/comfydl/profiling/opgraph",
                                     json={"prompt": prompt})
            check("route: POST /opgraph 200", resp.status == 200)
            data = await resp.json()
            check("route: report has totals + nodes",
                  "totals" in data and "nodes" in data and data.get("version") == 1)
            check("route: assembled mode without header",
                  data.get("mode") == "assembled"
                  and isinstance(data.get("coverage"), dict)
                  and any(n["status"] == "rule" for n in data["nodes"]),
                  str(data.get("mode")))
            check("route: assembled nodes carry coverage buckets",
                  all(n.get("coverage") in
                      ("covered", "params_driven", "unknown", "missing")
                      for n in data["nodes"]))

            resp2 = await client.post("/comfydl/profiling/opgraph",
                                      json={"prompt": prompt},
                                      headers={"X-CDL-Profiling-Dangerous": "1"})
            check("route: dangerous header probes for real", resp2.status == 200)
            data2 = await resp2.json()
            check("route: dangerous run probes nodes",
                  data2.get("mode") is None and data2["totals"]["probed"] > 0,
                  json.dumps(data2.get("totals")))

            # P0: a 1MB-style oversized corpus answers instantly as a guard
            # fallback even in dangerous mode (the 2026-10-06 wedge scenario).
            big_prompt = {
                "1": {"class_type": "TextVocabBuild", "inputs": {
                    "corpus": "z" * (opgraph.MAX_NODE_INPUT_BYTES + 1),
                    "level": "char", "min_freq": 1}},
            }
            resp4 = await client.post("/comfydl/profiling/opgraph",
                                      json={"prompt": big_prompt},
                                      headers={"X-CDL-Profiling-Dangerous": "1"})
            data4 = await resp4.json()
            check("route: oversized input guarded even in dangerous mode",
                  resp4.status == 200 and data4["totals"]["probed"] == 0
                  and "input too large" in (data4["nodes"][0]["error"] or ""),
                  str(data4.get("error")))

            resp3 = await client.post("/comfydl/profiling/opgraph", data="not-json")
            check("route: bad JSON 400", resp3.status == 400)
        finally:
            await client.close()

    asyncio.run(_route_checks())

    _oprules_checks()
    _assembled_checks()

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


def _assembled_checks() -> None:
    """P1: the zero-execution assembler on the shipped example templates."""
    import json as _json
    import time

    from comfy.profiling import opgraph

    # warmup: the first call pays one-time import costs (estimators registry,
    # rules JSON) - the "fast" assertion measures the steady state.
    opgraph.assemble_workflow({})

    wf_dir = REPO_ROOT / "comfydl" / "example_workflows"
    for name, min_rules in (("language_model_train_and_chat", 5),
                            ("tabular_regression_production", 5)):
        wf = _json.loads((wf_dir / f"{name}.json").read_text(encoding="utf-8-sig"))
        prompt = {}
        for n in wf["nodes"]:
            if n["type"] in ("Note", "MarkdownNote", "PreviewImage"):
                continue
            inputs = {}
            for i in n.get("inputs", []):
                if i.get("link") is not None:
                    link = next(l for l in wf["links"] if l[0] == i["link"])
                    inputs[i["name"]] = [str(link[1]), link[2]]
            prompt[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
        t0 = time.perf_counter()
        report = opgraph.assemble_workflow(prompt)
        elapsed = time.perf_counter() - t0
        check(f"assemble {name}: mode + coverage present",
              report.get("mode") == "assembled"
              and set(report.get("coverage") or {})
              == {"covered", "params_driven", "unknown", "missing"})
        check(f"assemble {name}: rule coverage >= {min_rules}",
              report["totals"]["probed"] >= min_rules,
              _json.dumps(report["totals"]))
        check(f"assemble {name}: fast (<0.5s steady state, zero execution)",
              elapsed < 0.5, f"{elapsed * 1000:.0f}ms")
        trainer = [n for n in report["nodes"] if n["class_type"] == "CdlRegressionTrain"]
        if trainer:
            check(f"assemble {name}: trainer census from rule library",
                  trainer[0]["status"] == "rule" and trainer[0]["ops_total"] == 1429,
                  str(trainer[0]["ops_total"]))


def _oprules_checks() -> None:
    """P1: the static rules library (load / query / coverage buckets)."""
    import json as _json
    import tempfile

    from comfy.profiling import oprules

    doc = {
        "version": oprules.OPRULES_VERSION,
        "generated": "2026-10-07",
        "rules": {
            "CdlFakeNode": {
                "source": "offline_probe",
                "flops_from_shape": "estimated",
                "probes": [{
                    "input_signature": {"x": "TensorVal(shape=(1, 8))"},
                    "ops": [{"op": "aten.mm.default", "count": 1, "flops": 128}],
                    "ops_total": 1, "data_dependent_reads": 0, "probe_ms": 3,
                }],
            },
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        _json.dump(doc, fh)
        path = Path(fh.name)
    try:
        oprules.set_rules(None)  # force a fresh load from the temp file
        loaded = oprules.load_rules(path)
        check("oprules: version accepted", loaded["version"] == oprules.OPRULES_VERSION)

        entry = oprules.rule_for("CdlFakeNode")
        check("oprules: rule lookup", entry is not None
              and entry["source"] == "offline_probe")
        check("oprules: census lookup", oprules.census_for("CdlFakeNode")[0]["op"]
              == "aten.mm.default")
        check("oprules: missing rule is None (honest)", oprules.rule_for("NoSuchNode") is None
              and oprules.census_for("NoSuchNode") is None)
        check("oprules: data_dependent_reads default 0",
              oprules.data_dependent_reads_for("CdlFakeNode") == 0
              and oprules.data_dependent_reads_for("NoSuchNode") == 0)

        check("oprules: coverage covered", oprules.coverage_for("CdlFakeNode") == "covered")
        check("oprules: coverage params_driven",
              oprules.coverage_for("NoSuchNode", m2_flops_status="estimated",
                                   m2_estimated=True) == "params_driven")
        check("oprules: coverage unknown",
              oprules.coverage_for("NoSuchNode", m2_flops_status="unknown",
                                   m2_estimated=True) == "unknown")
        check("oprules: coverage missing",
              oprules.coverage_for("NoSuchNode") == "missing")
        check("oprules: summary aggregation",
              oprules.coverage_summary({"a": "covered", "b": "missing",
                                        "c": "covered"})
              == {"covered": 2, "params_driven": 0, "unknown": 0, "missing": 1})

        # Corrupt file must degrade to an empty ruleset, never raise.
        bad = Path(path.parent / "bad.json")
        bad.write_text("{not json", encoding="utf-8")
        loaded_bad = oprules.load_rules(bad)
        check("oprules: corrupt file -> empty ruleset", loaded_bad["rules"] == {})
        stats = oprules.stats()
        check("oprules: stats after empty load", stats["entries"] == 0)

        # A wrong document version is rejected the same way.
        bad2 = Path(path.parent / "badver.json")
        bad2.write_text(_json.dumps({"version": 999, "rules": {}}), encoding="utf-8")
        check("oprules: wrong version rejected",
              oprules.load_rules(bad2)["rules"] == {})
    finally:
        path.unlink(missing_ok=True)
        (path.parent / "bad.json").unlink(missing_ok=True)
        (path.parent / "badver.json").unlink(missing_ok=True)
        oprules.set_rules(None)  # restore the lazy-ship default for later checks


if __name__ == "__main__":
    raise SystemExit(main())
