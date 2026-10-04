"""Smoke tests for the memory profiling engine (``comfy/profiling/``).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling.py

Why this script exists
----------------------
The profiling engine estimates a workflow's peak memory statically (no
torch execution). These tests pin its core promises:

* the golden case - the ~1 MB-corpus OOM that started the project - is
  reproduced exactly: the attention q/k/v output alone is the 13,276,741,632
  bytes the allocator refused, and the verdict is red;
* the formula parameter counts agree with the real materialised model
  (``comfy/lm_protocol.build_model``; the only place torch runs, purely for
  cross-validation);
* unestimated node types are reported as unknown, never guessed;
* the assumption layer records what it assumed and honours overrides;
* the post-mortem parser handles the CPU and CUDA spellings and suggests a
  batch size that actually fits;
* the HTTP routes answer through a real aiohttp route layer (the lesson of
  test_templates_catalog.py T9).
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy.profiling import (  # noqa: E402
    DEFAULT_ASSUMPTIONS,
    DeviceBudget,
    analyse_oom,
    estimate_workflow,
    human_bytes,
    parse_allocation_bytes,
)
from comfy.profiling import formulas  # noqa: E402
from comfy.profiling.shapes import BlockInfo  # noqa: E402

#: Golden case: 202,651-char corpus, window 64, char vocab 256, d_model 256,
#: 2 transformer blocks (4 heads, FFN 1024), AdamW, batch_size 0.
GOLDEN_CORPUS_CHARS = 202_651
GOLDEN_WINDOW = 64
GOLDEN_SAMPLES = GOLDEN_CORPUS_CHARS - GOLDEN_WINDOW  # 202,587
GOLDEN_KPROJ_BYTES = GOLDEN_SAMPLES * 64 * 256 * 4  # 13,276,741,632

#: 16 GB total / ~2.96 GB free, the machine the real OOM happened on.
GOLDEN_BUDGET = DeviceBudget("cpu", 16 * 1024**3, 2_960_000_000)


def _golden_corpus() -> str:
    # 255 distinct characters cycling: the char vocabulary is exactly 256
    # (<unk> included), like the real Tiny Shakespeare run.
    return "".join(chr(32 + (i % 255)) for i in range(GOLDEN_CORPUS_CHARS))


def _golden_prompt() -> dict:
    corpus = _golden_corpus()
    return {
        "vocab": {"class_type": "TextVocabBuild",
                  "inputs": {"corpus": corpus, "level": "char", "min_freq": 1}},
        "enc": {"class_type": "TextEncode",
                "inputs": {"vocab": ["vocab", 0], "text": corpus}},
        "sw": {"class_type": "TextSlidingWindow",
               "inputs": {"ids": ["enc", 0], "window": GOLDEN_WINDOW}},
        "emb": {"class_type": "LanguageModelEmbedding",
                "inputs": {"vocab": ["vocab", 0], "vocab_size": 256, "d_model": 256,
                           "include_position": True}},
        "b1": {"class_type": "LanguageModelTransformerBlock",
               "inputs": {"spec": ["emb", 0], "num_heads": 4, "d_ffn": 1024,
                          "activation": "relu", "dropout": 0.0}},
        "b2": {"class_type": "LanguageModelTransformerBlock",
               "inputs": {"spec": ["b1", 0], "num_heads": 4, "d_ffn": 1024,
                          "activation": "relu", "dropout": 0.0}},
        "build": {"class_type": "LanguageModelBuild", "inputs": {"spec": ["b2", 0], "seed": 0}},
        "opt": {"class_type": "TrainingOptimizer", "inputs": {"optimizer": "AdamW"}},
        "train": {"class_type": "LanguageModelTrain",
                  "inputs": {"model": ["build", 0], "x": ["sw", 0], "y": ["sw", 1],
                             "optimizer": ["opt", 0], "steps": 300, "batch_size": 0}},
    }


def _train_of(report: dict) -> dict:
    return [n for n in report["nodes"] if n["class_type"] == "LanguageModelTrain"][0]


# ---------------------------------------------------------------------------
# T1: the golden case

rep = estimate_workflow(_golden_prompt(), GOLDEN_BUDGET)
train = _train_of(rep)
qkv = [i for i in train["items"] if i["kind"] == "attn_qkv"]
basis = train["basis"]["params"]

check("T1a golden: every node estimated",
      all(n["status"] == "estimated" for n in rep["nodes"]),
      str([(n["id"], n["status"]) for n in rep["nodes"] if n["status"] != "estimated"]))
check("T1b golden: vocab is 256", basis["vocab"] == 256, str(basis["vocab"]))
check("T1c golden: batch equals whole dataset (202,587)",
      basis["batch"] == GOLDEN_SAMPLES, str(basis["batch"]))
check("T1d golden: attention q/k/v single allocation is the exact OOM bytes",
      bool(qkv) and qkv[0]["single_bytes"] == GOLDEN_KPROJ_BYTES == 13_276_741_632,
      str(qkv[0]["single_bytes"] if qkv else None))
check("T1e golden: verdict red / single exceeds free",
      rep["verdict"] == "red" and rep["verdict_reason"] == "single_exceeds_free",
      f"{rep['verdict']}/{rep['verdict_reason']}")
check("T1f golden: peak is the trainer's upper bound",
      rep["peak_bytes"] == train["total_bytes"] and rep["peak_bytes"] > 100 * 1024**3,
      human_bytes(rep["peak_bytes"]))
check("T1g golden: FFN hidden is the largest single tensor",
      rep["largest"]["kind"] == "ffn_hidden" and rep["largest"]["single_bytes"] > GOLDEN_KPROJ_BYTES,
      rep["largest"]["kind"])

# ---------------------------------------------------------------------------
# T2: formula parameter counts equal the real model's

import torch  # noqa: E402
from comfy import lm_protocol as mp  # noqa: E402

chain = (
    mp.EmbeddingSpec(256, 256, True),
    mp.TransformerBlockSpec(d_model=256, num_heads=4, d_ffn=1024, activation="relu", dropout=0.0),
    mp.TransformerBlockSpec(d_model=256, num_heads=4, d_ffn=1024, activation="gelu", dropout=0.1),
)
model = mp.build_model(chain, seed=0)
blocks = tuple(BlockInfo(b.d_model, b.num_heads, b.d_ffn, b.activation, b.dropout)
               for b in chain[1:])
formula_params = formulas.lm_parameter_count(256, 256, blocks)
check("T2a formulas: parameter count equals the real model",
      formula_params == mp.parameter_count(model),
      f"formula={formula_params} model={mp.parameter_count(model)}")

mlp_params = formulas.mlp_parameter_count(10, [16, 8], 3)
real_mlp = sum(t.numel() for t in torch.nn.Sequential(
    torch.nn.Linear(10, 16), torch.nn.Linear(16, 8), torch.nn.Linear(8, 3),
).parameters())
check("T2b formulas: MLP parameter count exact", mlp_params == real_mlp,
      f"formula={mlp_params} real={real_mlp}")

# ---------------------------------------------------------------------------
# T3: the default teaching graph is green

default_prompt = {
    "vocab": {"class_type": "TextVocabBuild",
              "inputs": {"corpus": "the quick brown fox jumps over the lazy dog",
                         "level": "char", "min_freq": 1}},
    "enc": {"class_type": "TextEncode",
            "inputs": {"vocab": ["vocab", 0], "text": "the quick brown fox jumps over the lazy dog"}},
    "sw": {"class_type": "TextSlidingWindow", "inputs": {"ids": ["enc", 0], "window": 4}},
    "emb": {"class_type": "LanguageModelEmbedding",
            "inputs": {"vocab": ["vocab", 0], "vocab_size": 30, "d_model": 32,
                       "include_position": True}},
    "b1": {"class_type": "LanguageModelTransformerBlock",
           "inputs": {"spec": ["emb", 0], "num_heads": 4, "d_ffn": 128,
                      "activation": "relu", "dropout": 0.0}},
    "build": {"class_type": "LanguageModelBuild", "inputs": {"spec": ["b1", 0], "seed": 0}},
    "opt": {"class_type": "TrainingOptimizer", "inputs": {"optimizer": "AdamW"}},
    "train": {"class_type": "LanguageModelTrain",
              "inputs": {"model": ["build", 0], "x": ["sw", 0], "y": ["sw", 1],
                         "optimizer": ["opt", 0], "steps": 300, "batch_size": 0}},
}
rep_default = estimate_workflow(default_prompt, GOLDEN_BUDGET)
check("T3 default teaching graph: green", rep_default["verdict"] == "green",
      f"{rep_default['verdict']}/{rep_default['verdict_reason']} "
      f"peak={human_bytes(rep_default['peak_bytes'])}")

# ---------------------------------------------------------------------------
# T4: unregistered node types are unknown, never guessed

rep_unknown = estimate_workflow(
    {"note": {"class_type": "Note", "inputs": {"text": "hello"}},
     "kstack": {"class_type": "ImageScale",
                "inputs": {"upscale_method": "nearest", "width": 512, "height": 512}}},
    GOLDEN_BUDGET,
)
check("T4a unknown: node types without estimators are reported unknown",
      all(n["status"] == "unknown" for n in rep_unknown["nodes"]),
      str([(n["id"], n["status"]) for n in rep_unknown["nodes"]]))
check("T4b unknown: verdict unknown / nothing estimated",
      rep_unknown["verdict"] == "unknown" and rep_unknown["verdict_reason"] == "nothing_estimated",
      rep_unknown["verdict_reason"])

rep_assumed = estimate_workflow(
    {"sw": {"class_type": "TextSlidingWindow", "inputs": {"ids": ["ghost", 0], "window": 8}},
     "ghost": {"class_type": "MysteryLoader", "inputs": {}}},
    GOLDEN_BUDGET,
)
sw_node = [n for n in rep_assumed["nodes"] if n["class_type"] == "TextSlidingWindow"][0]
check("T4c unknown: unestimated input falls back to the assumption tier",
      sw_node["status"] == "estimated" and sw_node["confidence"] == "approx",
      f"{sw_node['status']}/{sw_node['confidence']}")
check("T4d unknown: the assumption is recorded with its source",
      {"key": "sequence_length", "value": DEFAULT_ASSUMPTIONS["sequence_length"],
       "source": "default"} in rep_assumed["assumptions_used"],
      str(rep_assumed["assumptions_used"]))

# ---------------------------------------------------------------------------
# T5: assumption overrides reach the estimate

rep_overridden = estimate_workflow(
    {"sw": {"class_type": "TextSlidingWindow", "inputs": {"ids": ["ghost", 0], "window": 8}},
     "ghost": {"class_type": "MysteryLoader", "inputs": {}}},
    GOLDEN_BUDGET,
    assumptions={"sequence_length": 1024},
)
check("T5 assumptions: override is recorded",
      any(a["key"] == "sequence_length" and a["value"] == 1024 and a["source"] == "override"
          for a in rep_overridden["assumptions_used"]),
      str(rep_overridden["assumptions_used"]))

# ---------------------------------------------------------------------------
# T6: batch_size semantics mirror the node (0 and >= samples = whole set)

prompt_batched = _golden_prompt()
prompt_batched["train"]["inputs"]["batch_size"] = 1024
rep_batched = estimate_workflow(prompt_batched, GOLDEN_BUDGET)
check("T6a batch: explicit batch_size is honoured",
      _train_of(rep_batched)["basis"]["params"]["batch"] == 1024,
      str(_train_of(rep_batched)["basis"]["params"]["batch"]))
prompt_batched["train"]["inputs"]["batch_size"] = 10_000_000
rep_clamped = estimate_workflow(prompt_batched, GOLDEN_BUDGET)
check("T6b batch: oversized batch_size falls back to the whole dataset",
      _train_of(rep_clamped)["basis"]["params"]["batch"] == GOLDEN_SAMPLES,
      str(_train_of(rep_clamped)["basis"]["params"]["batch"]))

# ---------------------------------------------------------------------------
# T7: Training Loop estimator

mlp_prompt = {
    "v": {"class_type": "TextVocabBuild", "inputs": {"corpus": "ab", "level": "char", "min_freq": 1}},
    "xsrc": {"class_type": "TextEncode", "inputs": {"vocab": ["v", 0], "text": "ab" * 5}},
    "opt": {"class_type": "TrainingOptimizer", "inputs": {"optimizer": "SGD"}},
    "loop": {"class_type": "TrainingLoop",
             "inputs": {"x": ["xsrc", 0], "y": ["xsrc", 0], "optimizer": ["opt", 0],
                        "hidden": "16,8", "steps": 100, "batch_size": 0}},
}
rep_mlp = estimate_workflow(mlp_prompt, GOLDEN_BUDGET)
loop = [n for n in rep_mlp["nodes"] if n["class_type"] == "TrainingLoop"][0]
check("T7a TrainingLoop: estimated with parsed hidden widths and SGD state",
      loop["status"] == "estimated"
      and loop["basis"]["params"]["hidden"] == [16, 8]
      and not any(i["kind"] == "optimizer_state" for i in loop["items"]),
      str(loop["basis"]["params"]))

# ---------------------------------------------------------------------------
# T8: post-mortem parsing and suggestion

check("T8a postmortem: CPU spelling (comma-grouped bytes)",
      parse_allocation_bytes("you tried to allocate 13,276,741,632 bytes.") == 13_276_741_632,
      str(parse_allocation_bytes("you tried to allocate 13,276,741,632 bytes.")))
check("T8b postmortem: CUDA binary suffix",
      parse_allocation_bytes("CUDA out of memory. Tried to allocate 2.00 GiB") == 2 * 1024**3,
      str(parse_allocation_bytes("CUDA out of memory. Tried to allocate 2.00 GiB")))
check("T8c postmortem: decimal suffix",
      parse_allocation_bytes("Tried to allocate 1.50 GB") == int(1.5 * 1000**3),
      str(parse_allocation_bytes("Tried to allocate 1.50 GB")))
check("T8d postmortem: no allocation in message -> None",
      parse_allocation_bytes("something else went wrong") is None)
check("T8e postmortem: human formatting", human_bytes(13_276_741_632) == "12.36 GB",
      str(human_bytes(13_276_741_632)))

pm = analyse_oom(
    {"message": "DefaultCPUAllocator: not enough memory: you tried to allocate "
                "13,276,741,632 bytes.",
     "node_id": "train", "node_type": "LanguageModelTrain", "prompt": _golden_prompt()},
    GOLDEN_BUDGET,
)
check("T8f postmortem: allocation extracted",
      pm["allocation_bytes"] == 13_276_741_632, str(pm["allocation_bytes"]))
check("T8g postmortem: attributed to the trainer node with an exact match",
      pm["attributed"] is not None and pm["attributed"]["node_id"] == "train"
      and pm["attributed"]["match"] == "exact",
      str(pm["attributed"]))
check("T8h postmortem: suggested batch is sensible",
      pm["suggestion"] is not None and 0 < pm["suggestion"]["batch_size"] < GOLDEN_SAMPLES,
      str(pm["suggestion"]))

suggested = pm["suggestion"]["batch_size"]
feasible_prompt = _golden_prompt()
feasible_prompt["train"]["inputs"]["batch_size"] = suggested
rep_suggested = estimate_workflow(feasible_prompt, GOLDEN_BUDGET)
check("T8i postmortem: suggested batch is green under the same budget",
      rep_suggested["verdict"] in ("green", "yellow")
      and rep_suggested["largest"]["single_bytes"] <= GOLDEN_BUDGET.free_bytes * 0.7 + 1,
      f"{rep_suggested['verdict']} largest={human_bytes(rep_suggested['largest']['single_bytes'])}")

pm_no_budget = analyse_oom({"message": "tried to allocate 1024 bytes"})
check("T8j postmortem: works without a graph / budget",
      pm_no_budget["allocation_bytes"] == 1024 and pm_no_budget["suggestion"] is None,
      str(pm_no_budget))


# ---------------------------------------------------------------------------
# T9: HTTP route layer (real aiohttp server, the T9 lesson of
# test_templates_catalog.py: dynamic routes must be tested through the
# router, never as bare handler functions).

async def _route_checks() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app import profiling_routes

    # The production app is built with client_max_size=args.max_upload_size;
    # the bare 1 MB default would reject the golden case's 1 MB-corpus
    # prompt, so mirror the production limit here.
    app = web.Application(client_max_size=100 * 1024**2)
    app.add_routes(profiling_routes.routes)
    app.router.add_static(
        "/comfydl/profiling", str(REPO_ROOT / "app" / "profiling_assets")
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        payload = {
            "prompt": _golden_prompt(),
            "budget": {"name": "cpu", "total_bytes": GOLDEN_BUDGET.total_bytes,
                       "free_bytes": GOLDEN_BUDGET.free_bytes},
        }
        resp = await client.post("/comfydl/profiling/estimate", json=payload)
        body = await resp.json()
        check("T9a route: estimate answers 200 with a red report",
              resp.status == 200 and body.get("verdict") == "red",
              f"{resp.status} {body.get('verdict')}")
        check("T9b route: report nodes carry the breakdown",
              any(n["class_type"] == "LanguageModelTrain" and n["items"]
                  for n in body.get("nodes", [])),
              str([n["class_type"] for n in body.get("nodes", [])]))

        resp = await client.post("/comfydl/profiling/estimate", data=b"not json")
        check("T9c route: malformed estimate body -> 400", resp.status == 400,
              str(resp.status))

        resp = await client.post("/comfydl/profiling/postmortem", json={
            "message": "you tried to allocate 13,276,741,632 bytes.",
            "node_id": "train", "node_type": "LanguageModelTrain",
            "prompt": _golden_prompt(),
            "budget": {"name": "cpu", "total_bytes": GOLDEN_BUDGET.total_bytes,
                       "free_bytes": GOLDEN_BUDGET.free_bytes},
        })
        body = await resp.json()
        check("T9d route: postmortem answers 200 with attribution",
              resp.status == 200 and body.get("attributed") is not None,
              f"{resp.status}")
        check("T9e route: postmortem suggests a batch size",
              body.get("suggestion") is not None and body["suggestion"]["batch_size"] > 0,
              str(body.get("suggestion")))

        # the repo-owned assets must be reachable through the same static
        # route the production server mounts (server.py)
        resp = await client.get("/comfydl/profiling/profiler.js")
        check("T9f assets: profiler.js served with JS content type",
              resp.status == 200 and "javascript" in resp.headers.get("Content-Type", ""),
              f"{resp.status} {resp.headers.get('Content-Type')}")
        resp = await client.get("/comfydl/profiling/profiler.css")
        check("T9g assets: profiler.css served",
              resp.status == 200 and "css" in resp.headers.get("Content-Type", ""),
              f"{resp.status} {resp.headers.get('Content-Type')}")
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# T10: the frontend loader patch (idempotent, temp web root)

def _frontend_checks() -> None:
    import tempfile

    from app.frontend_patch import PROFILING_LOADER_MARK, _patch_profiling_loader

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "index.html").write_text(
            '<html><head></head><body><div id="vue-app"></div></body></html>'
        )
        injected = _patch_profiling_loader(root)
        text = (root / "index.html").read_text()
        check("T10a loader: injected exactly once before </body>",
              injected and text.count(PROFILING_LOADER_MARK) == 1
              and text.index(PROFILING_LOADER_MARK) < text.index("</body>"))
        again = _patch_profiling_loader(root)
        check("T10b loader: second run is a no-op (idempotent)",
              not again and (root / "index.html").read_text().count(PROFILING_LOADER_MARK) == 1)


# ---------------------------------------------------------------------------
# Summary

def _main() -> int:
    _frontend_checks()
    asyncio.run(_route_checks())
    failures = [(name, detail) for name, ok, detail in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        mark = "PASS" if ok else "FAIL"
        suffix = f"  [{detail}]" if (detail and not ok) else ""
        print(f"{mark}  {name}{suffix}")
    print(f"\n{len(_RESULTS) - len(failures)} PASS / {len(failures)} FAIL "
          f"({len(_RESULTS)} checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
