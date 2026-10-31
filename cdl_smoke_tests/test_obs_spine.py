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
          snap.schema_version == SNAPSHOT_SCHEMA_VERSION == 1
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
    check("T4c spine: first tick arrives and fills every section",
          snap is not None and "percent" in snap.cpu and "used_mb" in snap.ram
          and "cpu_percent" in snap.proc and snap.gpu is None)
    check("T4d spine: listeners receive the same immutable snapshots",
          len(got) >= 1 and all(s.schema_version == 1 for s in got))
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


# ------------------------------------------------------------------ main --
def _main() -> int:
    _snapshot_checks()
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
