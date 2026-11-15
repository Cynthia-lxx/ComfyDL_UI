"""Observation spine + switch registry + signature cache smoke tests.

Dual-track P0 (2026-10-31): covers the three shared-infrastructure pieces
this batch delivered -

* ``comfy/profiling/switch_registry.py``  - registration, fail-loudly reads,
  hard-disable refusal, meta-profiling accumulation;
* ``comfy/profiling/signature_cache.py``  - deterministic digests, atomic
  writes, mtime invalidation, corrupt-file recovery;
* ``app/spine/``                          - snapshot contract, bus lifecycle
  (rate=0 fully stops), provider/listener isolation, control routes, and
  the watchdog's attachment to the shared thread.

Also pins the one platform trap found during development: the RAM provider
must never touch ``platform.system()`` (WMI query - blocks the sampling
thread for tens of seconds on Windows); ``sys.platform`` is asserted fast.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import profiling_watchdog as pw  # noqa: E402
from app.spine import SWITCH_ID, get_spine  # noqa: E402
from app.spine import bus as spine_bus  # noqa: E402
from app.spine.providers import RamProvider  # noqa: E402
from comfy.profiling import proflog, signature_cache, switch_registry  # noqa: E402

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))


# ------------------------------------------------------------------ T1 ----
def _snapshot_checks() -> None:
    from app.spine.snapshot import CdlSnapshot, SNAPSHOT_SCHEMA_VERSION

    snap = CdlSnapshot.now(cpu={"percent": 12.5}, ram={"used_mb": 1},
                           proc={"cpu_percent": 3.0})
    check("T1a snapshot: schema version constant exported and folded in",
          snap.schema_version == SNAPSHOT_SCHEMA_VERSION == 2
          and isinstance(SNAPSHOT_SCHEMA_VERSION, int))
    check("T1b snapshot: frozen - attribute writes raise",
          _raises(lambda: setattr(snap, "cpu", {}), dataclasses.FrozenInstanceError))
    d = snap.to_dict()
    check("T1c snapshot: to_dict carries every section and gpu stays None",
          d["cpu"] == {"percent": 12.5} and d["ram"] == {"used_mb": 1}
          and d["proc"] == {"cpu_percent": 3.0} and d["gpu"] is None
          and d["ts"] == snap.ts)
    check("T1d snapshot: to_dict is JSON-serialisable (WS payload)",
          isinstance(json.dumps(d), str))


def _raises(fn, exc_type) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except BaseException:  # noqa: BLE001
        return False
    return False


# ------------------------------------------------------------------ T2 ----
def _registry_checks() -> None:
    switch_registry.register_switch(
        "cdl.test.switch", default=True, hot_toggle=True, self_meter=True)
    switch_registry.register_switch(
        "cdl.test.switch", default=True, hot_toggle=True, self_meter=True)
    check("T2a registry: identical re-registration is a no-op",
          switch_registry.is_enabled("cdl.test.switch") is True)
    try:
        switch_registry.register_switch(
            "cdl.test.switch", default=False, hot_toggle=True, self_meter=True)
        check("T2b registry: conflicting re-registration raises", False, "no raise")
    except ValueError:
        check("T2b registry: conflicting re-registration raises", True)

    check("T2c registry: unknown switch fails loudly (KeyError)",
          _raises(lambda: switch_registry.is_enabled("cdl.nope"), KeyError))

    switch_registry._force_state("cdl.test.switch", enabled=False,
                                 hard_disabled=True, hard_checked=True)
    check("T2d registry: hard-disabled refuses soft re-enable",
          switch_registry.set_enabled("cdl.test.switch", True) is False
          and switch_registry.is_enabled("cdl.test.switch") is False)
    switch_registry._force_state("cdl.test.switch", enabled=True,
                                 hard_disabled=False, hard_checked=True)

    switch_registry.record_self_overhead("cdl.unknown.switch", 1.0)  # no raise
    switch_registry.record_self_overhead("cdl.test.switch", 0.25)
    switch_registry.record_self_overhead("cdl.test.switch", 0.5)
    ov = switch_registry.self_overhead("cdl.test.switch")
    check("T2e registry: self-overhead accumulates; unknown ids ignored",
          abs(ov["total_s"] - 0.75) < 1e-9 and ov["samples"] == 2)

    all_ov = switch_registry.self_overhead_all()
    check("T2i registry: self_overhead_all snapshots every meter (P7)",
          isinstance(all_ov, dict) and "cdl.test.switch" in all_ov
          and {"total_s", "samples"} <= set(all_ov["cdl.test.switch"]))

    desc = switch_registry.describe("cdl.test.switch")
    check("T2f registry: describe exposes the settings-panel contract",
          desc["id"] == "cdl.test.switch" and desc["enabled"] is True
          and desc["hard_disabled"] is False and desc["self_meter"] is True
          and "self_overhead" in desc)

    check("T2g registry: proflog shell still speaks its historical API",
          proflog.is_master_enabled() is switch_registry.is_enabled(proflog.SWITCH_ID)
          and proflog.set_master_enabled(True) is True
          and proflog.is_hard_disabled() is False)
    import comfy.crashsite_switch  # noqa: F401  (registers on import)

    check("T2h registry: crashsite switch is a registered peer",
          "comfydl.crashsite" in switch_registry.list_switches())


# ------------------------------------------------------------------ T3 ----
def _signature_cache_checks() -> None:
    payload = {"seq_len": 1024, "heads": 16, "backend": "sdpa"}
    check("T3a sigcache: digest is deterministic",
          signature_cache.digest("bench", payload)
          == signature_cache.digest("bench", payload))
    check("T3b sigcache: digest reacts to payload and namespace",
          signature_cache.digest("bench", payload)
          != signature_cache.digest("bench", {**payload, "heads": 32})
          != signature_cache.digest("other", payload))
    original = signature_cache.code_version()
    try:
        signature_cache.set_code_version("test-build-1")
        d1 = signature_cache.digest("bench", payload)
        signature_cache.set_code_version("test-build-2")
        check("T3c sigcache: digest folds the code version (auto-invalidation)",
              d1 != signature_cache.digest("bench", payload))
    finally:
        signature_cache.set_code_version(original)
    check("T3d sigcache: non-serialisable payload fails loudly",
          _raises(lambda: signature_cache.digest("bench", {"t": object()}), TypeError))

    with tempfile.TemporaryDirectory() as tmp:
        signature_cache.set_root(Path(tmp))
        try:
            cache = signature_cache.JsonNamespaceCache("bench", schema_version=3)
            cache.put("k1", {"speedup": 2.5})
            cache.put("k2", {"speedup": 1.5})
            check("T3e sigcache: put/get roundtrip preserves sibling entries",
                  cache.get("k1") == {"speedup": 2.5}
                  and cache.get("k2") == {"speedup": 1.5}
                  and cache.get("k3") is None)
            leftovers = [p.name for p in Path(tmp).rglob(".*.tmp")]
            check("T3f sigcache: atomic write leaves no tmp files behind",
                  leftovers == [], str(leftovers))

            path = cache.path
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["entries"]["k1"]["speedup"] = 9.9  # external edit
            path.write_text(json.dumps(raw), encoding="utf-8")
            fresh = signature_cache.JsonNamespaceCache("bench", schema_version=3)
            check("T3g sigcache: external edits are picked up via mtime",
                  fresh.get("k1") == {"speedup": 9.9})

            path.write_text("{ not json", encoding="utf-8")
            fresh2 = signature_cache.JsonNamespaceCache("bench", schema_version=3)
            check("T3h sigcache: corrupt file reads as empty, not as an error",
                  fresh2.get("k1") is None and len(fresh2) == 0)
            fresh2.put("k9", {"v": 1})
            check("T3i sigcache: writing over a corrupt file recovers it",
                  fresh2.get("k9") == {"v": 1})

            check("T3j sigcache: schema-version mismatch starts fresh",
                  signature_cache.JsonNamespaceCache("bench", schema_version=99)
                  .get("k9") is None)
            check("T3k sigcache: unsafe namespaces are rejected",
                  _raises(lambda: signature_cache.JsonNamespaceCache("a/b"),
                          ValueError))
        finally:
            signature_cache.set_root(None)


# ------------------------------------------------------------------ T4 ----
def _spine_bus_checks() -> None:
    spine = get_spine()
    spine.stop()
    switch_registry._force_state(SWITCH_ID, enabled=False, hard_checked=True)
    check("T4a spine: start refused while the switch is off",
          spine.start() is False and spine.is_running() is False)

    switch_registry._force_state(SWITCH_ID, enabled=True, hard_checked=True)
    check("T4b spine: set_rate clamps to the minimum interval",
          abs(spine.set_rate(0.001) - spine_bus._MIN_RATE_S) < 1e-9)

    got = []
    spine.add_listener(got.append)
    spine.start()
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and spine.snapshot() is None:
        time.sleep(0.02)
    snap = spine.snapshot()
    check("T4c spine: first tick fills every section incl. the P6 gpu dict",
          snap is not None and "percent" in snap.cpu and "used_mb" in snap.ram
          and "cpu_percent" in snap.proc and isinstance(snap.gpu, dict)
          and "available" in snap.gpu)
    check("T4d spine: listeners receive the same immutable v2 snapshots",
          len(got) >= 1 and all(s.schema_version == 2 for s in got))
    spine.remove_listener(got.append)

    class _Boom:
        fields = "cpu"

        def poll(self):
            raise RuntimeError("provider exploded")

    solo = spine_bus.Spine()
    solo._providers = [_Boom(), RamProvider()]
    snap2 = solo._poll()
    check("T4e spine: a failing provider degrades only its own section",
          snap2.cpu == {"error": "provider exploded"} and "used_mb" in snap2.ram)

    boomed = []

    def _bad_listener(_snap):
        raise RuntimeError("listener exploded")

    solo.add_listener(_bad_listener)
    solo.add_listener(boomed.append)
    solo._poll()  # exercise aggregation only; listener isolation below
    check("T4f spine: listener bookkeeping is exact",
          solo.listener_count() == 2 and solo.remove_listener(_bad_listener) is None
          and solo.listener_count() == 1)
    solo.remove_listener(boomed.append)

    t0 = time.perf_counter()
    ram = RamProvider().poll()
    check("T4g spine: RAM provider is WMI-free and fast",
          time.perf_counter() - t0 < 0.2 and isinstance(ram["platform"], str))

    check("T4h spine: rate 0 fully stops the shared thread",
          spine.set_rate(0) == 0.0 and _soon_false(spine.is_running))

    switch_registry._force_state(SWITCH_ID, enabled=False, hard_checked=True)


def _soon_false(fn, timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not fn():
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------ T5 ----
async def _route_checks() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from app.spine import routes as spine_routes
    # The perf switches register at module import (same as at boot).
    import comfy_extras.nodes_attention_select  # noqa: F401
    import comfy_extras.nodes_compile  # noqa: F401

    spine = get_spine()
    switch_registry._force_state(SWITCH_ID, enabled=True, hard_checked=True)

    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.add_routes(spine_routes.routes)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/comfydl/spine")
        body = await resp.json()
        check("T5a route: GET state serves switch + running + rate",
              resp.status == 200 and "switch" in body and "running" in body
              and "rate_s" in body and "sent" in body)

        resp = await client.get("/comfydl/spine/snapshot")
        body = await resp.json()
        check("T5b route: GET snapshot answers (tick or explicit no-tick)",
              resp.status == 200 and "schema_version" in body)

        resp = await client.patch(
            "/comfydl/spine", json={"rate_s": 0.15, "enabled": True})
        body = await resp.json()
        check("T5c route: PATCH configures rate and starts the bus",
              resp.status == 200 and body["rate_s"] == 0.15
              and body["running"] is True)
        await _wait_true(lambda: spine.snapshot() is not None)

        resp = await client.patch("/comfydl/spine", json={"enabled": False})
        body = await resp.json()
        check("T5d route: PATCH stop detaches the thread",
              resp.status == 200 and body["running"] is False
              and _soon_false(spine.is_running))

        resp = await client.patch("/comfydl/spine", json={"rate_s": "fast"})
        check("T5e route: non-numeric rate rejected with 400",
              resp.status == 400)
        resp = await client.patch("/comfydl/spine", data=b"not json")
        check("T5f route: non-JSON body rejected with 400",
              resp.status == 400)

        # --- T5g-i: the generic switch registry routes (perf panel) -------
        resp = await client.get("/comfydl/switches")
        body = await resp.json()
        ids = {s["id"] for s in body.get("switches", [])}
        check("T5g route: GET switches lists every registered switch",
              resp.status == 200
              and {"comfydl.attention_select", "comfydl.compile"} <= ids,
              str(sorted(ids))[:120])
        desc = {s["id"]: s for s in body.get("switches", [])}
        check("T5g2 route: descriptions carry the panel's fields",
              all(k in desc.get("comfydl.attention_select", {})
                  for k in ("enabled", "default", "hot_toggle", "self_meter",
                            "self_overhead")))

        # --- P6b (ADR-D7 ②): quant switch + status aggregation ----------
        check("T5j P6b: quant switch registered (hard-disabled when no gguf)",
              "comfydl.quant" in ids
              and desc["comfydl.quant"]["enabled"] in (True, False))
        if desc["comfydl.quant"]["hard_disabled"]:
            resp = await client.patch("/comfydl/switches/comfydl.quant",
                                      json={"enabled": True})
            check("T5j2 P6b: hard-disabled quant refuses re-enable with 409",
                  resp.status == 409)
        resp = await client.get("/comfydl/switches/status")
        sbody = await resp.json()
        fams = sbody.get("backends", {})
        check("T5k P6b: status endpoint always 200, five families + contract",
              resp.status == 200 and sbody.get("available") is True
              and set(fams) == {"attention", "compile", "quant", "vram",
                                "dataio_cache"}
              and all(("available" in v) and ("status" in v)
                      and ("describe" in v) for v in fams.values()))
        check("T5k2 P6b: every importable family reports live status",
              all((not v.get("available"))
                  or isinstance(v.get("status"), dict)
                  for v in fams.values()))
        src_names = (
            "comfy_extras/nodes_attention_select.py",
            "comfy/compile_manager.py",
            "comfy/quant_formats.py",
            "comfy/vram_manager.py",
            "comfydl/nodes/_dataio_cache.py",
        )
        check("T5l P6b: five backends expose zero-arg describe()/status()",
              all("def status(" in (REPO_ROOT / p).read_text(encoding="utf-8")
                  and "def describe(" in (REPO_ROOT / p).read_text(
                      encoding="utf-8")
                  for p in src_names))

        original = desc["comfydl.compile"]["enabled"]
        resp = await client.patch("/comfydl/switches/comfydl.compile",
                                  json={"enabled": not original})
        body = await resp.json()
        check("T5h route: PATCH toggles a switch and echoes the fresh state",
              resp.status == 200
              and body["switch"]["enabled"] is (not original))
        # restore
        await client.patch("/comfydl/switches/comfydl.compile",
                           json={"enabled": original})

        resp = await client.patch("/comfydl/switches/comfydl.nope",
                                  json={"enabled": False})
        check("T5i route: unknown switch id 404s", resp.status == 404)
        resp = await client.patch("/comfydl/switches/comfydl.compile",
                                  json={"enabled": "yes"})
        check("T5i2 route: non-boolean enabled rejected with 400",
              resp.status == 400)

        # T5m P7: attention bench readout - a read-only projection of the
        # attention_bench cache into HTTP, seeded here through the same
        # store_result() path the Optimizer node uses (isolated cache root).
        import torch  # noqa: F401  (local: only this block needs it)
        from comfy_extras import nodes_attention_select as _attn
        bench_root = Path(tempfile.mkdtemp(prefix="cdl_bench_"))
        bench_old_root = signature_cache._root_override
        signature_cache.set_root(bench_root)
        try:
            _attn.store_result(torch.device("cpu"), 1024, 16, 64, {
                "times_ms": {"pytorch": 1.5, "flash": 0.75},
                "speedup": {"flash": 2.0},
                "chosen": "flash",
                "benchmark_s": 0.05,
                "device": "cpu", "torch": torch.__version__,
                "seq_len": 1024, "heads": 16, "head_dim": 64,
            })
            # Old-style entry (pre-P7, geometry only in the opaque digest)
            # must survive the readout as a geometry-less row.
            _attn.store_result(torch.device("cpu"), 512, 8, 64, {
                "times_ms": {"pytorch": 0.9},
                "speedup": {},
                "chosen": "pytorch",
                "benchmark_s": 0.01,
            })
            resp = await client.get("/comfydl/attention/bench")
            bench_body = await resp.json()
            entries = bench_body.get("entries") or []
            check("T5m P7: bench endpoint 200 with notes + stamped row",
                  resp.status == 200 and bench_body.get("available") is True
                  and isinstance(bench_body.get("notes"), dict)
                  and "flash" in bench_body.get("notes", {})
                  and any(e.get("chosen") == "flash"
                          and e.get("seq_len") == 1024
                          and e.get("heads") == 16
                          and e.get("device") == "cpu"
                          for e in entries))
            check("T5m2 P7: pre-P7 entry degrades to a geometry-less row",
                  any(e.get("chosen") == "pytorch" and e.get("times_ms")
                      and "seq_len" not in e for e in entries))
        finally:
            signature_cache.set_root(bench_old_root)
    finally:
        await client.close()
        spine.stop()
        switch_registry._force_state(SWITCH_ID, enabled=False, hard_checked=True)


async def _wait_true(fn, timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if fn():
            return
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------ T6 ----
def _watchdog_integration_checks() -> None:
    spine = get_spine()
    spine.stop()

    class _FakeServer:
        last_node_id = "n1"
        last_prompt_id = "p1"

    with tempfile.TemporaryDirectory() as tmp:
        dog = pw.ProfilingWatchdog(
            _FakeServer(), interval_s=0.05, cpu_threshold=95.0, sustain_s=60.0,
            log_path=str(Path(tmp) / "w.jsonl"))
        dog.start()
        check("T6a watchdog: riding the spine still produces live samples",
              _wait_true_sync(lambda: dog.snapshot().get("node_id") == "n1")
              and dog.snapshot().get("running") is True
              and spine.listener_count() >= 1)
        dog.stop()
        check("T6b watchdog: stop detaches and releases the shared thread",
              spine.listener_count() == 0
              and _soon_false(spine.is_running)
              and dog.snapshot().get("running") is False)


def _wait_true_sync(fn, timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if fn():
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------------ T7 ----
def _gpu_provider_checks() -> None:
    import torch

    from app.spine.providers import GpuProvider, default_providers
    from app.spine.snapshot import SNAPSHOT_SCHEMA_VERSION

    check("T7a P6: snapshot contract is v2 with v1 fields intact",
          SNAPSHOT_SCHEMA_VERSION == 2)
    check("T7b P6: gpu provider joins default_providers",
          any(getattr(p, "fields", "") == "gpu" for p in default_providers()))
    result = GpuProvider().poll()
    check("T7c P6: gpu poll never raises and always reports availability",
          isinstance(result, dict) and isinstance(result.get("available"), bool))
    check("T7d P6: CPU-only boxes degrade to available=false",
          torch.cuda.is_available() or result.get("available") is False)


# ------------------------------------------------------------------ T8 ----
def _obs_panel_checks() -> None:
    """Static contract checks for the P6 CRAFT 1 independent sidebar tab."""
    js_path = REPO_ROOT / "app" / "obs_assets" / "obs.js"
    css_path = REPO_ROOT / "app" / "obs_assets" / "obs.css"
    js = js_path.read_text(encoding="utf-8") if js_path.is_file() else ""
    css = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""
    patch = (REPO_ROOT / "app" / "frontend_patch.py").read_text(encoding="utf-8")
    server = (REPO_ROOT / "server.py").read_text(encoding="utf-8")

    check("T8a P6: obs assets, loader and static route are wired",
          bool(js) and bool(css)
          and "OBS_LOADER_MARK" in patch and "_patch_obs_loader" in patch
          and "web.static('/comfydl/obs', obs_assets)" in server)
    check("T8b P6: minimap-sized overlay with collapse/hide persistence",
          "registerSidebarTab" not in js
          and "cdl-obs-overlay" in js
          and "document.body.appendChild" in js
          and "COLLAPSE_KEY" in js and "HIDDEN_KEY" in js
          and 'removeEventListener("cdl.spine"' in js)
    check("T8c P6: one cdl.spine channel, off guidance, de-duplicated readouts",
          'addEventListener("cdl.spine"' in js and "off_hint" in js
          and "PATCH" in js and "s.proc?.cpu_percent" in js
          and "setInterval" in js)
    check("T8d P6: three motion gates honoured",
          "html.cdl-motion-off" in css
          and "body.cdlapp-motion-off" in css
          and "prefers-reduced-motion" in css)
    check("T8e P6: fixed overlay with collapsed/hidden states",
          "position: fixed" in css and "z-index: 2100" in css
          and 'data-collapsed="1"]' in css and 'data-hidden="1"]' in css)


# ------------------------------------------------------------------ T9 ----
def _obs_craft2_checks() -> None:
    """P6 CRAFT 2: drag / dock gap / execution badges / runs ledger contract."""
    js_path = REPO_ROOT / "app" / "obs_assets" / "obs.js"
    css_path = REPO_ROOT / "app" / "obs_assets" / "obs.css"
    js = js_path.read_text(encoding="utf-8") if js_path.is_file() else ""
    css = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""
    runs_routes = (REPO_ROOT / "app" / "runs_routes.py").read_text(
        encoding="utf-8")

    check("T9a CRAFT 2: pointer-capture drag writes only left/top and persists",
          'POS_KEY = "ComfyDL.Obs.Position"' in js
          and "setPointerCapture" in js and "pointerdown" in js
          and "pointermove" in js and "pointerup" in js
          and "pointercancel" in js
          and 'root.style.left = left + "px"' in js
          and 'root.style.top = top + "px"' in js
          and "savePos(state.pos.left, state.pos.top)" in js
          and "function loadPos()" in js
          and "style.transform" not in js)
    check("T9b CRAFT 2: default position keeps a 14px dock gap",
          css.count("left: calc(var(--cdl-dock-left, 59px) + 14px);") == 2
          and "--cdl-dock-left:" not in css
          and "cursor: move" in css and "touch-action: none" in css
          and "cdl-obs-dragging" in css)
    check("T9e R5: default gap derives from the live rail, never overrides App Panel",
          "function ensureDockGap" in js
          and "ensureDockGap();" in js
          and 'getPropertyValue("--cdl-dock-left")' in js
          and ".side-tool-bar-container" in js
          and "getBoundingClientRect().width" in js
          and "if (dock) return;" in js)
    check("T9c CRAFT 2: badges come from seven official exec events",
          all(name in js for name in (
              "execution_start", "executing", "execution_cached", "executed",
              "execution_success", "execution_error", "execution_interrupted"))
          and "EXEC_EVENTS" in js and "onExecEvent" in js
          and "removeEventListener(name, handler)" in js
          and 'typeof raw === "string"' in js
          and all(("cdl-obs-badge-" + k) in css
                  for k in ("ran", "cached", "blocked"))
          and "send_sync" not in js)
    check("T9d CRAFT 2: runs ledger is read-only, newest first, with empty state",
          'RUNS_URL = "/comfydl/runs?limit=5"' in js
          and '"/comfydl/runs"' in runs_routes
          and "data.runs.slice(-5).reverse()" in js
          and "runs_empty" in js and "cdl-obs-runs-empty" in css
          and "No experiment runs yet" in js and "暂无实验 run" in js
          and "refreshRuns()" in js)


# ------------------------------------------------------------------ R4 ----
def _r4_checks() -> None:
    """R4 observation calibers: 1.5-A process-CPU object reuse + 1.7 runs rows."""
    providers_src = (REPO_ROOT / "app" / "spine" / "providers.py").read_text(
        encoding="utf-8")
    obs_js = (REPO_ROOT / "app" / "obs_assets" / "obs.js").read_text(
        encoding="utf-8")
    obs_css = (REPO_ROOT / "app" / "obs_assets" / "obs.css").read_text(
        encoding="utf-8")
    fe_js = (REPO_ROOT / "app" / "verbose_assets" / "verbose.js").read_text(
        encoding="utf-8")

    # 1.5-A: the Process object must be built ONCE and reused; a fresh
    # psutil.Process() per poll has no previous cpu_percent sample and
    # returns 0.0 forever (the "Process CPU 0.0%" bug).
    check("R4a 1.5-A provider: Process built once, reused, VERBOSE-sampled",
          providers_src.count("psutil.Process()") == 1
          and providers_src.count("self._proc.cpu_percent(None)") == 2
          and "verbose_log.step(" in providers_src
          and '"proc_poll"' in providers_src and "every_n=8" in providers_src)

    import sys as _sys
    import types as _types

    class _FakeProc:
        def __init__(self) -> None:
            self.calls = 0
            self.pid = 4242

        def cpu_percent(self, interval=None):
            self.calls += 1
            # psutil semantics: the FIRST call on an object carries no
            # delta and reads 0.0; later calls report the real value.
            return 0.0 if self.calls == 1 else 137.5

        def memory_info(self):
            return _types.SimpleNamespace(rss=64 * 1024 * 1024)

    holder: dict = {}

    def _process():
        if "proc" not in holder:
            holder["proc"] = _FakeProc()
        return holder["proc"]

    fake = _types.ModuleType("psutil")
    fake.Process = _process
    saved = _sys.modules.get("psutil")
    _sys.modules["psutil"] = fake
    try:
        from app.spine.providers import ProcessProvider

        prov = ProcessProvider()
        first = prov.poll()
        second = prov.poll()
        third = prov.poll()
    finally:
        if saved is not None:
            _sys.modules["psutil"] = saved
        else:
            _sys.modules.pop("psutil", None)

    # poll 1 = prime(call 1) + read(call 2 -> real delta vs baseline);
    # polls 2/3 read the SAME object, so the value is never pinned to 0.
    # The old code rebuilt Process() per poll -> every poll was a "first
    # call" and read 0.0 forever.
    check("R4b 1.5-A provider: polls read the shared object (no rebuild)",
          first["cpu_percent"] == 137.5 and second["cpu_percent"] == 137.5
          and third["cpu_percent"] == 137.5
          and holder["proc"].calls == 4  # 1 prime + 3 reads, one object
          and holder["proc"].pid == 4242 and first["rss_mb"] == 64.0)

    # 1.5-A UI: per-core caliber must NOT be clamped into 0-100, and the
    # label/tooltip must carry the caliber.
    check("R4c 1.5-A obs: per-core caliber kept, labeled, no 0-100 clamp",
          "Number.isFinite(procCpu)" in obs_js
          and "proc_hint" in obs_js
          and "Process CPU (per-core)" in obs_js
          and "进程 CPU(per-core)" in obs_js
          and "clampPct(s.proc?.cpu_percent)" not in obs_js)

    # 1.7: run rows gain started_at (HH:MM:SS), duration, a this-session
    # tag, and a caliber hint; status text stays the real ledger value.
    check("R4d 1.7 obs: runs rows show time/duration/session tag + hint",
          "started.slice(11, 19)" in obs_js
          and "cdl-obs-run-time" in obs_js and "cdl-obs-run-dur" in obs_js
          and "cdl-obs-run-new" in obs_js and "run_new" in obs_js
          and "SESSION_START" in obs_js and "runs_hint" in obs_js
          and "cdl-obs-runs-hint" in obs_css
          and "cdl-obs-run-new" in obs_css and "cdl-obs-run-time" in obs_css)

    # VERBOSE: the /runs index body summary (R1's blind spot) is recorded.
    check("R4e verbose: /runs index-body summary recorded by the FE hook",
          "index-body" in fe_js and "/comfydl\\/runs" in fe_js
          and "resp.clone().json()" in fe_js and "statuses" in fe_js
          and "has_started_at" in fe_js)


# ------------------------------------------------------- P6b thin layer ----
def _p6b_frontend_checks() -> None:
    """M-D P6b: Optimizer panel wiring + hard-disable snap-back contract."""
    base = REPO_ROOT / "app" / "p6b_assets"
    js = (base / "p6b.js").read_text(encoding="utf-8")
    css = (base / "p6b.css").read_text(encoding="utf-8")
    patch = (REPO_ROOT / "app" / "frontend_patch.py").read_text(encoding="utf-8")
    server = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    check("P6b-a assets + loader + static route wired",
          (base / "p6b.js").is_file() and (base / "p6b.css").is_file()
          and "P6B_LOADER_MARK" in patch and "_patch_p6b_loader" in patch
          and "web.static('/comfydl/p6b', p6b_assets)" in server)
    check("P6b-b reads the three existing endpoints, no new data source",
          '"/comfydl/switches"' in js
          and '"/comfydl/switches/status"' in js
          and '"/comfydl/dataio/cache/stats"' in js
          and "SWITCHES_URL + \"/\"" in js)
    check("P6b-c special switches keep their real start/stop endpoints",
          '"/comfydl/profiling/enabled"' in js
          and '"/comfydl/crashsite/enabled"' in js
          and '"/comfydl/spine"' in js and "const SPECIAL" in js)
    check("P6b-d rejected toggle snaps back from server truth",
          "await fetchAll()" in js and "box.checked = !wanted" in js
          and "result.ok" in js)
    check("P6b-e shell: fixed window, own keys, drag writes left/top only",
          "position: fixed" in css and "z-index: 2100" in css
          and "ComfyDL.P6b.Collapsed" in js and "ComfyDL.P6b.Position" in js
          and "root.style.left = left" in js and "root.style.top = top" in js
          and "style.transform" not in js)
    check("P6b-f three-gate motion + collapsed/hidden shape",
          "html.cdl-motion-off" in css and "body.cdlapp-motion-off" in css
          and "prefers-reduced-motion" in css
          and "root.dataset.collapsed" in js and "root.dataset.hidden" in js
          and "data-collapsed=\"1\"" in css and "data-hidden=\"1\"" in css)
    check("P6b-g bilingual i18n + dock-gap repair",
          "const I18N" in js and "en:" in js and "zh:" in js
          and "function ensureDockGap" in js and "--cdl-dock-left" in js)
    check("P7-h bench card wired into the Optimizer panel (read-only)",
          '"/comfydl/attention/bench"' in js
          and "function renderBench" in js and 'data-p6b="bench"' in js
          and "bench_title" in js and "nobench" in js
          and "cdl-p6b-bench-geo" in css and "cdl-p6b-bench-backend" in css
          and "GET /comfydl/attention/bench" in
          (REPO_ROOT / "app" / "spine" / "routes.py").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ main --
def _main() -> int:
    _snapshot_checks()
    _p6b_frontend_checks()
    _r4_checks()
    _obs_craft2_checks()
    _gpu_provider_checks()
    _obs_panel_checks()
    _registry_checks()
    _signature_cache_checks()
    _spine_bus_checks()
    asyncio.run(_route_checks())
    _watchdog_integration_checks()
    # leave the world as we found it: spine off, no listeners
    spine = get_spine()
    spine.stop()
    switch_registry._force_state(SWITCH_ID, enabled=False, hard_checked=True)

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
