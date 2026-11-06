#!/usr/bin/env python
"""Manual benchmark: crash site / profiling master-switch overhead (SP-0 / T6-1).

Reproduces the in-process attribution method of
``.codebuddy/memory/2026-10-31-sp0-attribution.md`` so the T6-1 cost
reduction can be measured on BOTH machines before and after the fix:

    P (CPU): penv\\Scripts\\python.exe cdl_smoke_tests\\bench_crashsite_overhead.py
    F (GPU): penv_new\\Scripts\\python.exe cdl_smoke_tests\\bench_crashsite_overhead.py

NOT part of the default smoke suite: wall-clock timing is noisy (especially
on the USB CPU drive), so this is a hand-run benchmark, like
``bench_dataio_cache.py``. Each configuration gets one untimed warmup plus
five timed runs; the median is reported. Configurations are sampled in a
round-robin so machine drift cannot bias whole configuration blocks. Snapshot
databases are written to
a temporary user directory and removed on exit.

Method notes (keep these stable so the numbers stay comparable):
  * UI-format example workflow -> API prompt, widget values resolved against
    each node's INPUT_TYPES order (same parser as test_template_workflows.py).
  * Fresh ``PromptExecutor`` per rep, ``cache_type=None`` - no cache reuse.
  * Five configurations: profiling master off/on crossed with the crash site
    being absent / registered-but-disabled / fully enabled.
  * ``--cpu`` is forced through the smoke harness bootstrap on both machines,
    so P vs F isolates storage/IO differences; the GPU device-sync hypothesis
    in the SP-0 report needs a separate CUDA-tensor benchmark.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "cdl_smoke_tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_smoke_test as harness  # noqa: E402 - bootstrap helper

FRONTEND_ONLY = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
TEMPLATES = ("classification_from_scratch",
             "tabular_classification_production")
CONFIGS = (
    ("prof=F crash=off", False, "off"),
    ("prof=F crash=disabled", False, "disabled"),
    ("prof=F crash=on", False, "on"),
    ("prof=T crash=off", True, "off"),
    ("prof=T crash=on", True, "on"),
)
REPS = 5


def widget_order(node_cls):
    """Widget names in declaration order (mirrors test_template_workflows)."""
    it = node_cls.INPUT_TYPES()
    order = []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            if isinstance(spec, tuple):
                type_name = spec[0]
                if isinstance(type_name, list):
                    type_name = "COMBO"
            elif isinstance(spec, list):
                type_name = "COMBO"
            else:
                type_name = spec
            if type_name in WIDGET_TYPES:
                order.append(name)
                if type_name == "INT" and name in ("seed", "noise_seed"):
                    order.append("control_after_generate")
    return order


def ui_to_prompt(path, registry):
    """Convert a shipped UI-format workflow into an API prompt dict."""
    wf = json.loads(path.read_text(encoding="utf-8-sig"))
    links = {l[0]: (l[1], l[2]) for l in wf["links"]}
    prompt = {}
    for node in wf["nodes"]:
        ctype = node["type"]
        if ctype in FRONTEND_ONLY:
            continue
        inputs = {}
        linked = set()
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            src, sslot = links[link_id]
            inputs[inp["name"]] = [str(src), sslot]
            linked.add(inp["name"])
        wvals = node.get("widgets_values") or []
        names = [n for n in widget_order(registry[ctype]) if n not in linked]
        if len(wvals) != len(names):
            raise ValueError(
                "%s node %s (%s): %d widget values vs %d declared %s"
                % (path.name, node["id"], ctype, len(wvals), len(names), names))
        for key, value in zip(names, wvals):
            if key != "control_after_generate":
                inputs[key] = value
        prompt[str(node["id"])] = {"class_type": ctype, "inputs": inputs}
    return prompt


def main() -> int:
    import torch

    sandbox = tempfile.TemporaryDirectory(prefix="cdl-sp0-")
    try:
        harness._bootstrap(Path(sandbox.name))
        import folder_paths
        import nodes as host_nodes
        import execution as execution_mod
        from comfy_execution import cache_provider as cp
        from comfy.profiling import proflog
        from app.crashsite.provider import SnapshotProvider

        folder_paths.set_user_directory(str(Path(sandbox.name) / "user"))
        harness._load_registry(False)

        print("torch=%s cuda_available=%s" % (
            torch.__version__, torch.cuda.is_available()))
        print("SP-0/T6-1 overhead benchmark | %d warmup + %d timed reps, median"
              % (1, REPS))

        def reset_providers():
            for provider in list(cp._get_cache_providers()):
                cp.unregister_cache_provider(provider)

        def timed_run(prompt, outputs, pid, prof, crash):
            proflog.set_master_enabled(prof)
            reset_providers()
            if crash != "off":
                queue = types.SimpleNamespace(currently_running={
                    "1.0": (1.0, pid, prompt, {}, outputs, {})})
                cp.register_cache_provider(SnapshotProvider(
                    enabled=(crash == "on"), queue=queue))
            server = types.SimpleNamespace(
                send_sync=lambda *a, **k: None, client_id=None)
            executor = execution_mod.PromptExecutor(
                server, cache_type=None,
                cache_args={"ram": 0.5, "ram_inactive": 0.3})
            start = time.perf_counter()
            executor.execute(prompt, pid, {}, outputs)
            return time.perf_counter() - start

        for template in TEMPLATES:
            path = (REPO_ROOT / "comfydl" / "example_workflows"
                    / (template + ".json"))
            try:
                prompt = ui_to_prompt(path, host_nodes.NODE_CLASS_MAPPINGS)
                ok, err, outputs, node_errors = asyncio.run(
                    execution_mod.validate_prompt("sp0-validate", prompt, None))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print("SKIP %s: %s" % (template, exc))
                continue
            if not ok:
                print("SKIP %s: %s %s" % (template, err, node_errors))
                continue

            print("\n== %s (%d nodes) ==" % (template, len(prompt)))
            # Round-robin: every configuration is sampled once per round, so
            # machine drift (CPU frequency, IO contention, first-use imports)
            # is spread evenly instead of biasing whole configuration blocks.
            samples = [[] for _ in CONFIGS]
            for rep in range(REPS + 1):
                for idx, (_label, prof, crash) in enumerate(CONFIGS):
                    pid = "sp0-%s-r%d" % (crash + ("T" if prof else "F"), rep)
                    elapsed = timed_run(prompt, outputs, pid, prof, crash)
                    if rep:  # round 0 is the untimed warmup
                        samples[idx].append(elapsed)
            rows = [(CONFIGS[i][0], statistics.median(samples[i]),
                     min(samples[i]), max(samples[i]))
                    for i in range(len(CONFIGS))]
            baseline = rows[0][1]
            for label, median, lo, hi in rows:
                print("  %-22s %9.1f ms   %+7.1f%% vs all-off"
                      "   [%.0f..%.0f]"
                      % (label, median * 1000, (median / baseline - 1) * 100,
                         lo * 1000, hi * 1000))
        reset_providers()
        return 0
    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    sys.exit(main())
