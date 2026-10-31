#!/usr/bin/env python
"""Standalone smoke test: the attention-selection stack (SP-2, M-B).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_attention_select.py

Pins the promises of ``comfy_extras/nodes_attention_select.py`` (the engine +
the dual-form node pair) on whatever device this machine actually has:

* probing is two-stage (the A4 lesson): a backend only counts when it
  *survives a real mini forward*, and validation is memoized per device;
* the benchmark engine is deterministic and honest: identical tensors per
  candidate, warmup excluded from timing, the median taken, an injectable
  timer so tests need no GPU;
* ``choose_backend`` honours the ``min_speedup`` gate and falls back to the
  SDPA reference when nothing clears it;
* the signature-cache round-trip works (miss -> store -> hit) and is keyed on
  the bench geometry;
* both nodes execute against a real ``ModelPatcher`` and attach the chosen
  backend through ``set_model_optimized_attention`` (transformer_options
  override) on a *clone*; the switch hot-disables to a pure passthrough.
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_attn_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy_extras import nodes_attention_select as att  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

SWITCH_ID = att.SWITCH_ID
_ORIGINAL_ENABLED = sr.is_enabled(SWITCH_ID)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# T1: the backend registry

backends = att.registered_backends()
check("T1a pytorch (SDPA) is a registered reference backend",
      att.REFERENCE_BACKEND in backends, str(backends))
check("T1b every registered backend has a teaching note",
      set(backends) <= set(att._BACKEND_NOTES),
      str(sorted(set(backends) - set(att._BACKEND_NOTES))))

# ---------------------------------------------------------------------------
# T2: two-stage validation (import success != runs) + memoization

validated = att.validate_backends(DEVICE)
validated_again = att.validate_backends(DEVICE)
check("T2a validation result is memoized (same object)",
      validated is validated_again)
check("T2b every entry is an (ok, error) pair",
      all(isinstance(ok, bool) and isinstance(err, str)
          for ok, err in validated.values()))
check("T2c the reference backend validates on this machine",
      validated.get(att.REFERENCE_BACKEND, (False,))[0] is True)

usable = att.usable_backends(DEVICE)
check("T2d usable is the validated-ok subset",
      usable == [n for n, (ok, _) in validated.items() if ok])

# ---------------------------------------------------------------------------
# T3: the benchmark engine, driven by a deterministic fake timer

CALLS: list = []


def fake_timer() -> float:
    # Each timed call "takes" a deterministic step; the sequence encodes
    # per-backend timings we can predict the median of.
    CALLS.append(len(CALLS))
    return float(CALLS[-1])


candidates = {name: att.REGISTERED_ATTENTION_FUNCTIONS[name]
              for name in ("pytorch",)}
CALLS.clear()
times = att.run_benchmark(
    candidates, seq_len=32, heads=2, head_dim=16, device=torch.device("cpu"),
    dtype=torch.float32, warmup=2, iters=5, timer=fake_timer)
check("T3a benchmark returns a median per candidate",
      set(times) == {"pytorch"} and times["pytorch"] > 0, str(times))
# The timer runs twice per timed iteration (window start and end) and never
# during warmup: 5 iters -> exactly 10 timer calls.
check("T3b warmup calls are not timed (timer only inside the timed loop)",
      len(CALLS) == 10, f"timer calls={len(CALLS)}")

# The median, not the mean: give one timed window a 9-second outlier.
# warmup=3 (untimed), iters=5 -> 10 timer reads encoding durations 1,1,9,1,1.
OUTLIER_READS = iter([0.0, 1.0, 2.0, 3.0, 4.0, 13.0, 14.0, 15.0, 16.0, 17.0])


def outlier_timer() -> float:
    return next(OUTLIER_READS)


times_o = att.run_benchmark(
    candidates, seq_len=32, heads=2, head_dim=16, device=torch.device("cpu"),
    dtype=torch.float32, warmup=3, iters=5, timer=outlier_timer)
check("T3c median (not mean) is taken - the 9.0 outlier does not dominate",
      times_o["pytorch"] == 1.0, str(times_o))

# ---------------------------------------------------------------------------
# T4: choose_backend + min_speedup gate

best, speedups = att.choose_backend({"pytorch": 1.0, "flash": 0.5}, 1.05)
check("T4a the backend that clears the gate wins",
      best == "flash" and speedups["flash"] == 2.0, str((best, speedups)))
kept, _ = att.choose_backend({"pytorch": 1.0, "flash": 0.99}, 1.05)
check("T4b below min_speedup the reference is kept", kept == "pytorch")
at_gate, _ = att.choose_backend({"pytorch": 1.05, "flash": 1.0}, 1.05)
check("T4c exactly-at-gate switches (>= gate, per the ADR)",
      at_gate == "flash", str(at_gate))
fallback, speedups_f = att.choose_backend({"flash": 0.5, "split": 0.4}, 1.05)
check("T4d no reference: the fastest of what exists is picked",
      fallback == "split", str((fallback, speedups_f)))
check("T4e ref_speed edges",
      att.ref_speed(2.0, 1.0) == 0.5 and att.ref_speed(0.0, 1.0) == 0.0)

# ---------------------------------------------------------------------------
# T5: signature-cache round trip (attention_bench namespace)

att.store_result(DEVICE, 777, 4, 32, {"chosen": "pytorch", "times_ms": {}})
hit = att.load_cached_result(DEVICE, 777, 4, 32)
check("T5a stored result loads back", hit is not None
      and hit.get("chosen") == "pytorch", str(hit))
miss = att.load_cached_result(DEVICE, 778, 4, 32)
check("T5b a different geometry is a cache miss", miss is None)

# ---------------------------------------------------------------------------
# T6: report rendering

report = att.render_report(DEVICE, 777, 4, 32, {"pytorch": 0.001},
                           {"pytorch": 1.0}, "pytorch", "hit")
check("T6a report marks the chosen backend", "<-- chosen" in report)
check("T6b report labels the reference backend", "(reference)" in report)
check("T6c report records the cache state", "cache=hit" in report)

# ---------------------------------------------------------------------------
# T7: a real ModelPatcher through the auto node (benchmark runs for real)

from comfy.model_protocol import make_model_patcher  # noqa: E402

patcher = make_model_patcher({"w": torch.zeros(1)})
result = att.CdlAttentionOptimizer.execute(
    patcher, seq_len=256, num_heads=4, head_dim=32, min_speedup=1.05)
out_model, report_str = result.result[0], result.result[1]
check("T7a node returns a new patcher (clone)", out_model is not patcher)
check("T7b report carries device + geometry + choice",
      f"device={DEVICE.type}" in report_str and "seq=256" in report_str
      and "<-- chosen" in report_str, report_str.splitlines()[0] if report_str else "")
override = out_model.model_options["transformer_options"].get(
    "optimized_attention_override")
check("T7c the chosen backend is attached via the official override hook",
      callable(override))
check("T7d the incoming patcher was not mutated",
      "optimized_attention_override"
      not in patcher.model_options["transformer_options"])

# Re-run: same geometry must be a signature-cache hit (nothing re-benched).
result2 = att.CdlAttentionOptimizer.execute(
    patcher, seq_len=256, num_heads=4, head_dim=32, min_speedup=1.05)
check("T7e repeat run reports a cache hit", "cache=hit" in result2.result[1],
      result2.result[1].splitlines()[0] if result2.result else "")

# force_refresh re-measures even on a warm cache.
result3 = att.CdlAttentionOptimizer.execute(
    patcher, seq_len=256, num_heads=4, head_dim=32, force_refresh=True)
check("T7f force_refresh re-benchmarks (cache=miss)",
      "cache=miss" in result3.result[1],
      result3.result[1].splitlines()[0] if result3.result else "")

# ---------------------------------------------------------------------------
# T8: switch disabled -> pure passthrough

sr.set_enabled(SWITCH_ID, False)
result4 = att.CdlAttentionOptimizer.execute(patcher, seq_len=256)
check("T8a disabled switch passes the clone through",
      "disabled via the switch" in result4.result[1],
      result4.result[1][:80])
sr.set_enabled(SWITCH_ID, True)

# ---------------------------------------------------------------------------
# T9: the manual, teaching node

result5 = att.CdlAttentionBackend.execute(patcher, backend=att.REFERENCE_BACKEND)
out5, expl = result5.result[0], result5.result[1]
check("T9a manual node attaches the requested backend",
      callable(out5.model_options["transformer_options"].get(
          "optimized_attention_override")))
check("T9b explanation contains the backend's mechanism note",
      f"attached backend: {att.REFERENCE_BACKEND}" in expl
      and att._BACKEND_NOTES[att.REFERENCE_BACKEND].split(" - ")[0] in expl,
      expl.splitlines()[0] if expl else "")

try:
    att.CdlAttentionBackend.execute(patcher, backend="definitely_not_a_backend")
    check("T9c unknown backend raises", False, "no exception")
except RuntimeError as exc:
    check("T9c unknown backend raises", "did not validate" in str(exc), str(exc))

# ---------------------------------------------------------------------------
# T10: registration + schema sanity

import asyncio  # noqa: E402

import nodes as host_nodes  # noqa: E402

asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                        init_api_nodes=False))
registry = host_nodes.NODE_CLASS_MAPPINGS
check("T10a both attention nodes registered",
      "CdlAttentionOptimizer" in registry and "CdlAttentionBackend" in registry)

schema_b = att.CdlAttentionBackend.define_schema()
backend_input = next(i for i in schema_b.inputs if i.id == "backend")
check("T10b manual combo offers only backends that validated here",
      att.REFERENCE_BACKEND in backend_input.options
      and set(backend_input.options) <= set(usable),
      str(backend_input.options))

schema_o = att.CdlAttentionOptimizer.define_schema()
ids_o = [i.id for i in schema_o.inputs]
check("T10c optimizer widget order (report is the string output)",
      ids_o == ["model", "seq_len", "num_heads", "head_dim", "min_speedup",
                "force_refresh"], str(ids_o))

# ---------------------------------------------------------------------------
# Report

sr.set_enabled(SWITCH_ID, _ORIGINAL_ENABLED)


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== attention-select test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
