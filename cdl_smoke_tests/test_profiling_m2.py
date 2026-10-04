"""Smoke tests for the M2 profiling additions (compute ledger + watchdog).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling_m2.py

Why this script exists
----------------------
The M2 milestone adds a second, *additive* ledger next to the M1 memory
peak: the matmul FLOPs a workflow will burn. These tests pin its promises:

* the golden case's training FLOPs match an independent, first-principles
  re-derivation of the formulas (per-block attention / FFN / logits);
* the totals cross-check against the 6ND rule of thumb (6 x params x
  tokens processed) within the tolerance the attention quadratic allows;
* generation counts the no-cache loop honestly (one forward per length);
* unknown-compute nodes (pure-Python loops: vocab build, encode, window)
  are reported unknown, pass-through nodes zero - never guessed;
* the watchdog unit layer: burst detection, attribution, jsonl round-trip.

The FLOPs ledger deliberately makes no time/throughput claims (M3).
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


from comfy.profiling import DeviceBudget, estimate_workflow  # noqa: E402
from comfy.profiling import formulas  # noqa: E402
from comfy.profiling.shapes import BlockInfo  # noqa: E402

#: The M1 golden case, reused so the two ledgers stay consistent: 202,587
#: samples, window 64, char vocab 256, d_model 256, 2 blocks, FFN 1024,
#: 300 steps, batch_size 0 (= the whole dataset).
GOLDEN_CORPUS_CHARS = 202_651
GOLDEN_WINDOW = 64
GOLDEN_SAMPLES = GOLDEN_CORPUS_CHARS - GOLDEN_WINDOW  # 202,587
GOLDEN_STEPS = 300
GOLDEN_BUDGET = DeviceBudget("cpu", 16 * 1024**3, 2_960_000_000)


def _golden_corpus() -> str:
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
                             "optimizer": ["opt", 0], "steps": GOLDEN_STEPS, "batch_size": 0}},
    }


def _node(report: dict, class_type: str) -> dict:
    return [n for n in report["nodes"] if n["class_type"] == class_type][0]


# ---------------------------------------------------------------------------
# Independent re-derivations (first principles, not the shipped formulas):
# a matmul (n, k) x (k, m) costs 2*n*k*m FLOPs.

def _attn_flops(b, t, e):
    # q/k/v/o: four (B*T, E) x (E, E) GEMMs; QK^T and A@V: two
    # (B, H, T, T)-shaped GEMMs whose head-split sums back to 4*B*T^2*E.
    return 4 * 2 * b * t * e * e + 4 * b * t * t * e


def _ffn_flops(b, t, e, f):
    return 2 * 2 * b * t * e * f


def _forward_total(b, t, e, v, f, blocks):
    return blocks * (_attn_flops(b, t, e) + _ffn_flops(b, t, e, f)) + 2 * b * t * e * v


B, T, E, V, F, L = GOLDEN_SAMPLES, GOLDEN_WINDOW, 256, 256, 1024, 2
EXPECTED_TRAIN_FLOPS = 3 * GOLDEN_STEPS * _forward_total(B, T, E, V, F, L)

rep = estimate_workflow(_golden_prompt(), GOLDEN_BUDGET)
train = _node(rep, "LanguageModelTrain")

check("F1a report: training FLOPs equal the first-principles total",
      rep["flops"]["total"] == EXPECTED_TRAIN_FLOPS == train["flops_total"],
      f"report={rep['flops']['total']} node={train['flops_total']} "
      f"expected={EXPECTED_TRAIN_FLOPS}")

expected_by_kind = {
    "attn": 3 * GOLDEN_STEPS * L * _attn_flops(B, T, E),
    "ffn": 3 * GOLDEN_STEPS * L * _ffn_flops(B, T, E, F),
    "logits": 3 * GOLDEN_STEPS * 2 * B * T * E * V,
}
check("F1b report: by-kind split matches (attn / ffn / logits)",
      all(rep["flops"]["by_kind"].get(k) == v for k, v in expected_by_kind.items())
      and rep["flops"]["by_kind"]["other"] == 0,
      str(rep["flops"]["by_kind"]))

check("F1c report: by-kind sums back to the total and the largest is real",
      sum(rep["flops"]["by_kind"].values()) == rep["flops"]["total"]
      and rep["flops"]["largest"]["node_id"] == "train"
      and rep["flops"]["largest"]["flops"] == max(
          i["flops"] for i in train["flops_items"]),
      str(rep["flops"]["largest"]))

check("F1d train: every item is approx-flagged, kind-tagged, step-scaled",
      bool(train["flops_items"])
      and all(i["approx"] and i["kind"] in ("attn", "ffn", "logits")
              for i in train["flops_items"]),
      str(len(train["flops_items"])))

check("F1e unknowns: pure-Python nodes are unknown, pass-throughs zero",
      _node(rep, "TextVocabBuild")["flops_status"] == "unknown"
      and _node(rep, "TextEncode")["flops_status"] == "unknown"
      and _node(rep, "TextSlidingWindow")["flops_status"] == "unknown"
      and _node(rep, "LanguageModelBuild")["flops_status"] == "zero"
      and _node(rep, "TrainingOptimizer")["flops_status"] == "zero"
      and all(_node(rep, c)["flops_total"] is None
              for c in ("TextVocabBuild", "LanguageModelBuild")),
      str([(_node(rep, c)["class_type"], _node(rep, c)["flops_status"])
           for c in ("TextVocabBuild", "TextEncode", "TextSlidingWindow",
                     "LanguageModelBuild", "TrainingOptimizer")]))

check("F1f report: version 2 with an additive ledger section",
      rep["version"] == 2 and rep["flops"]["any_estimated"] is True,
      str(rep["version"]))

# ---------------------------------------------------------------------------
# F2: the 6ND cross-check (training ~ 6 x params x tokens processed).
# The per-layer sum exceeds the rule of thumb by the attention quadratic
# (4*B*T^2*E per block); anything within [0.95, 1.25] means the ledger
# and the rule agree to the expected order.

blocks = tuple(
    BlockInfo(256, 4, 1024, "relu", 0.0) for _ in range(L)
)
params = formulas.lm_parameter_count(V, E, blocks)
tokens_processed = GOLDEN_STEPS * B * T
six_nd = formulas.six_nd_estimate(params, tokens_processed)
ratio = EXPECTED_TRAIN_FLOPS / six_nd
check("F2a 6ND: training total within [0.95, 1.25] of 6*params*tokens",
      0.95 <= ratio <= 1.25,
      f"ledger={EXPECTED_TRAIN_FLOPS} 6ND={six_nd} ratio={ratio:.4f} "
      f"params={params}")

check("F2b 6ND: the rule itself is exactly 6*N*D",
      six_nd == 6 * params * tokens_processed, str(six_nd))

# ---------------------------------------------------------------------------
# F3: inference / generation counted honestly

def _lm_prompt(extra: dict) -> dict:
    prompt = {
        "vocab": {"class_type": "TextVocabBuild",
                  "inputs": {"corpus": "abcde-abcde-abcde", "level": "char", "min_freq": 1}},
        "emb": {"class_type": "LanguageModelEmbedding",
                "inputs": {"vocab": ["vocab", 0], "vocab_size": 16, "d_model": 32,
                           "include_position": True}},
        "b1": {"class_type": "LanguageModelTransformerBlock",
               "inputs": {"spec": ["emb", 0], "num_heads": 4, "d_ffn": 128,
                          "activation": "relu", "dropout": 0.0}},
        "build": {"class_type": "LanguageModelBuild", "inputs": {"spec": ["b1", 0], "seed": 0}},
    }
    prompt.update(extra)
    return prompt


forward_prompt = _lm_prompt({
    "enc": {"class_type": "TextEncode",
            "inputs": {"vocab": ["vocab", 0], "text": "abcde"}},
    "fwd": {"class_type": "LanguageModelForward",
            "inputs": {"model": ["build", 0], "ids": ["enc", 0]}},
})
rep_fwd = estimate_workflow(forward_prompt, GOLDEN_BUDGET)
fwd = _node(rep_fwd, "LanguageModelForward")
# The linked TextVocabBuild wins over the widget's vocab_size: its corpus
# "abcde-abcde-abcde" has 6 distinct chars, so the char vocabulary is 7.
expected_forward = _forward_total(1, 5, 32, 7, 128, 1)
check("F3a forward: single pass counted exactly (batch 1, length 5)",
      fwd["flops_total"] == expected_forward == rep_fwd["flops"]["total"],
      f"node={fwd['flops_total']} expected={expected_forward}")

generate_prompt = _lm_prompt({
    "enc": {"class_type": "TextEncode",
            "inputs": {"vocab": ["vocab", 0], "text": "abcde"}},
    "gen": {"class_type": "LanguageModelGenerate",
            "inputs": {"model": ["build", 0], "vocab": ["vocab", 0],
                       "prefix_ids": ["enc", 0], "num_tokens": 4}},
})
rep_gen = estimate_workflow(generate_prompt, GOLDEN_BUDGET)
gen = _node(rep_gen, "LanguageModelGenerate")
# No KV cache: one full forward per length 5, 6, 7, 8 (vocab 7 as above).
expected_generate = sum(_forward_total(1, length, 32, 7, 128, 1)
                        for length in (5, 6, 7, 8))
check("F3b generate: the no-cache loop sums one forward per length",
      gen["flops_total"] == expected_generate == rep_gen["flops"]["total"],
      f"node={gen['flops_total']} expected={expected_generate}")

check("F3c generate: items carry the approx flag (rule-derived, not counted)",
      bool(gen["flops_items"]) and all(i["approx"] for i in gen["flops_items"]),
      str(gen["flops_items"][:2]))

# ---------------------------------------------------------------------------
# F4: the MLP Training Loop ledger

mlp_prompt = {
    "v": {"class_type": "TextVocabBuild", "inputs": {"corpus": "ab", "level": "char", "min_freq": 1}},
    "xsrc": {"class_type": "TextEncode", "inputs": {"vocab": ["v", 0], "text": "ab" * 5}},
    "opt": {"class_type": "TrainingOptimizer", "inputs": {"optimizer": "SGD"}},
    "loop": {"class_type": "TrainingLoop",
             "inputs": {"x": ["xsrc", 0], "y": ["xsrc", 0], "optimizer": ["opt", 0],
                        "hidden": "16,8", "steps": 100, "batch_size": 0}},
}
rep_mlp = estimate_workflow(mlp_prompt, GOLDEN_BUDGET)
loop = _node(rep_mlp, "TrainingLoop")
# in=1 (y=x is 1-D -> out_features 1? x is (10,) 1-D -> in_features 10,
# y 1-D -> out_features 1): layers 10-16-8-1.
expected_mlp = 3 * 100 * 2 * 10 * (10 * 16 + 16 * 8 + 8 * 1)
check("F4a TrainingLoop: MLP training FLOPs exact (3x forward x steps)",
      loop["flops_total"] == expected_mlp == rep_mlp["flops"]["total"],
      f"node={loop['flops_total']} expected={expected_mlp}")

# ---------------------------------------------------------------------------
# F5: unknown / empty graphs never invent FLOPs

rep_unknown = estimate_workflow(
    {"note": {"class_type": "Note", "inputs": {"text": "hello"}}}, GOLDEN_BUDGET)
check("F5a unknown: nothing estimated -> zero total, no any_estimated",
      rep_unknown["flops"]["total"] == 0
      and rep_unknown["flops"]["any_estimated"] is False
      and rep_unknown["nodes"][0]["flops_status"] == "unknown",
      str(rep_unknown["flops"]))

rep_empty = estimate_workflow(None, GOLDEN_BUDGET)
check("F5b empty: report still carries the flops section",
      rep_empty["flops"] == {"total": 0, "any_estimated": False,
                             "by_kind": {"attn": 0, "ffn": 0, "logits": 0, "other": 0},
                             "largest": None},
      str(rep_empty["flops"]))


# ---------------------------------------------------------------------------
# W: the watchdog unit layer (burst detection, attribution, jsonl)


def _sample(proc, sys_cpu, node, prompt="p1", rss=123.4):
    return {
        "ts_epoch": 0.0,
        "process_cpu_percent": proc,
        "system_cpu_percent": sys_cpu,
        "process_rss_mb": rss,
        "node_id": node,
        "prompt_id": prompt,
    }


def _watchdog_unit_checks() -> None:
    import tempfile
    import time as _time

    from app import profiling_watchdog as pw

    check("W1a watchdog: psutil importable", pw.PSUTIL_OK, "")

    class _FakeServer:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        log_path = str(Path(tmp) / "watchdog.jsonl")
        server = _FakeServer()
        server.last_node_id = None
        server.last_prompt_id = None

        # -- the pure state machine, driven with synthetic samples ----------
        dog = pw.ProfilingWatchdog(
            server,
            interval_s=0.02,
            cpu_threshold=90.0,
            sustain_s=2.0,
            log_path=log_path,
            scale_hint_fn=lambda nid: {"class_type": "TextVocabBuild", "chars": 5}
            if nid == "5"
            else None,
        )
        clock = 0.0

        def tick(sample):
            nonlocal clock
            dog._evaluate(clock, sample)
            clock += 0.5

        # 5 hot ticks: crossing at t=0, sustained 2.5s -> burst opens at t=2.5
        for _ in range(5):
            tick(_sample(95, 40, "5"))
        # 3 more hot ticks inside the burst (peak 100 at the last one)
        for proc in (98, 97, 100):
            tick(_sample(proc, 40, "5"))
        # cool-down: closes at the tick after the second low sample
        for _ in range(3):
            tick(_sample(50, 20, "5"))

        events = dog.recent(limit=10)
        check("W2a watchdog: sustained hot CPU becomes one burst event",
              len(events) == 1 and events[0]["node_id"] == "5"
              and events[0]["prompt_id"] == "p1",
              str(events))
        if events:
            event = events[0]
            check("W2b watchdog: event carries peak / avg / duration / scale hint",
                  event["cpu_percent_peak"] == 100
                  and event["cpu_percent_avg"] > 0
                  and 2.5 <= event["duration_s"] <= 3.5
                  and event["data_scale"] == {"class_type": "TextVocabBuild", "chars": 5},
                  str(event))
            check("W2c watchdog: system-wide flag false when only this process is hot",
                  event["system_wide"] is False and event["system_cpu_percent_peak"] == 40,
                  str(event))

        # system-wide saturation is flagged
        dog2 = pw.ProfilingWatchdog(
            server, interval_s=0.02, cpu_threshold=90.0, sustain_s=2.0,
            log_path=str(Path(tmp) / "wide.jsonl"),
        )
        clock = 0.0

        def tick2(sample):
            nonlocal clock
            dog2._evaluate(clock, sample)
            clock += 0.5

        for _ in range(6):
            tick2(_sample(30, 95, "7"))
        for _ in range(3):
            tick2(_sample(10, 30, "7"))
        wide = dog2.recent(limit=10)
        check("W2d watchdog: system-wide stall is captured and flagged",
              len(wide) == 1 and wide[0]["system_wide"] is True
              and wide[0]["node_id"] == "7",
              str(wide))

        # -- the real thread: start / snapshot attribution / stop ----------
        dog3 = pw.ProfilingWatchdog(
            server, interval_s=0.02, cpu_threshold=90.0, sustain_s=60.0,
            log_path=str(Path(tmp) / "thread.jsonl"),
        )
        dog3.start()
        dog3.start()  # idempotent
        server.last_node_id = "9"
        server.last_prompt_id = "p2"
        _time.sleep(0.25)
        snapshot = dog3.snapshot()
        check("W3a watchdog: live snapshot attributes to the executing node",
              snapshot.get("available") is True and snapshot.get("running") is True
              and snapshot.get("node_id") == "9" and snapshot.get("prompt_id") == "p2"
              and isinstance(snapshot.get("process_cpu_percent"), (int, float))
              and isinstance(snapshot.get("system_cpu_percent"), (int, float))
              and isinstance(snapshot.get("process_rss_mb"), (int, float)),
              str({k: snapshot.get(k) for k in
                   ("available", "running", "node_id", "process_cpu_percent")}))
        dog3.stop()
        check("W3b watchdog: stop joins the thread",
              not dog3._thread or not dog3._thread.is_alive(), "")
        check("W3c watchdog: sustained 60s means the smoke run wrote no event",
              dog3.recent(limit=5) == [] and dog3._events_written == 0,
              str(dog3._events_written))


async def _watchdog_route_checks() -> None:
    import tempfile

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app import profiling_routes, profiling_watchdog as pw

    class _FakeServer:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        server = _FakeServer()
        server.last_node_id = None
        server.last_prompt_id = None
        dog = pw.ProfilingWatchdog(
            server, interval_s=0.05, cpu_threshold=90.0, sustain_s=60.0,
            log_path=str(Path(tmp) / "route.jsonl"),
        )
        app = web.Application()
        app.add_routes(profiling_routes.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/comfydl/profiling/watchdog/status")
            check("W4a route: status without a watchdog instance -> 503",
                  resp.status == 503, str(resp.status))
            resp = await client.get("/comfydl/profiling/watchdog/log")
            check("W4b route: log without a watchdog instance -> 503",
                  resp.status == 503, str(resp.status))
        finally:
            await client.close()

        profiling_routes.set_watchdog(dog)
        app = web.Application()
        app.add_routes(profiling_routes.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/comfydl/profiling/watchdog/status")
            body = await resp.json()
            check("W4c route: status serves the live snapshot",
                  resp.status == 200 and body.get("available") is True
                  and body.get("running") is False
                  and "cpu_threshold" in body,
                  str(body)[:160])
            resp = await client.get("/comfydl/profiling/watchdog/log?limit=5")
            body = await resp.json()
            check("W4d route: log answers with the event list",
                  resp.status == 200 and body.get("events") == [],
                  str(body)[:160])
        finally:
            await client.close()
            profiling_routes.set_watchdog(None)


# ---------------------------------------------------------------------------
# Summary

def _main() -> int:
    _watchdog_unit_checks()
    asyncio.run(_watchdog_route_checks())
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
