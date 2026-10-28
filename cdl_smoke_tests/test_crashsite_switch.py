#!/usr/bin/env python
"""Standalone smoke test for the ComfyDL crash site master switch (2026-10-28).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_crashsite_switch.py

Covers:
  - T1  the ``--cdl-crashsite-disable`` CLI flag (parses, defaults off);
  - T2  the switch semantics (default on, soft off/on, hard-disable refuses
        re-enable and forces master off, profiling hard flag carry-over);
  - T3  provider detach/reattach through ``comfy_execution.cache_provider``
        (unregister == the executor's lifecycle hook short-circuits);
  - T4  ``comfy.train_checkpoint`` short-circuits while disabled (run_key
        without tensor digests, save/load/discard no-ops);
  - T5  the routes: GET/POST enabled, resume 409 while disabled, the
        hard-disabled POST rejection.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def report() -> int:
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


# ------------------------------------------------------- sandboxed fixture ---
import cdl_smoke_tests.run_smoke_test as harness  # noqa: E402

harness._bootstrap(Path(tempfile.mkdtemp(prefix="cdl_cs_switch_")))
harness._load_registry(False)

import torch  # noqa: E402

from app.crashsite import store  # noqa: E402
from app.crashsite.provider import SnapshotProvider  # noqa: E402

# Redirect the snapshot root into the sandbox BEFORE anything touches it.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="cdl_snapshots_switch_"))
store.snapshots_root = lambda: _TMP_ROOT

from comfy import crashsite_switch as switch  # noqa: E402
from comfy_execution import cache_provider as cp  # noqa: E402


def _switch_reset() -> None:
    """Restore pristine switch state between checks."""
    switch._hard_disabled = False
    switch._hard_checked = False
    switch._master_enabled = True


# ----------------------------------------------------------- T1: CLI flag ----
def _flag_checks() -> None:
    from comfy import cli_args

    args = cli_args.parser.parse_args([])
    check("T1 flag defaults off", args.cdl_crashsite_disable is False,
          str(args.cdl_crashsite_disable))
    args = cli_args.parser.parse_args(["--cdl-crashsite-disable"])
    check("T1 hard-disable flag parses", args.cdl_crashsite_disable is True)


# ------------------------------------------------------- T2: switch rules ----
def _switch_checks() -> None:
    _switch_reset()
    check("T2 defaults to enabled", switch.is_master_enabled() is True)
    check("T2 not hard-disabled by default", switch.is_hard_disabled() is False)

    check("T2 soft disable works", switch.set_master_enabled(False) is False)
    check("T2 soft-disabled state reads back", switch.is_master_enabled() is False)
    check("T2 soft re-enable works", switch.set_master_enabled(True) is True)

    # Hard-disable (as if --cdl-crashsite-disable were on the command line):
    # master is forced off and re-enabling is refused until a "restart".
    switch._hard_disabled = True
    switch._hard_checked = True
    switch._master_enabled = False
    check("T2 hard-disabled reads off", switch.is_master_enabled() is False)
    check("T2 re-enable refused while hard-disabled",
          switch.set_master_enabled(True) is False)
    check("T2 state stays off after refusal", switch.is_master_enabled() is False)

    # Profiling carry-over: the profiling hard flag also hard-disables (the
    # historical M1 wiring, see server.py).  Simulate by resetting the lazy
    # check while the profiling switch is hard-off.
    _switch_reset()
    try:
        from comfy.profiling import proflog as _prof

        _prof._hard_disabled = True
        switch._hard_checked = False  # force a re-read of the flags
        check("T2 profiling hard flag carries over", switch.is_hard_disabled() is True)
        check("T2 carry-over forces master off", switch.is_master_enabled() is False)
    finally:
        from comfy.profiling import proflog as _prof

        _prof._hard_disabled = False
    _switch_reset()


# ------------------------------------------------- T3: provider detaching ----
def _provider_checks() -> None:
    _switch_reset()
    cp._clear_cache_providers()  # pristine provider registry for the test
    try:
        provider = SnapshotProvider()
        cp.register_cache_provider(provider)
        check("T3 registered provider participates", cp._has_cache_providers() is True)

        # The executor's lifecycle hook short-circuits on the empty list, so
        # unregistering IS the "absent feature" state.
        cp.unregister_cache_provider(provider)
        check("T3 unregistered == absent", cp._has_cache_providers() is False)
        cp.register_cache_provider(provider)
        check("T3 re-registration works", cp._has_cache_providers() is True)

        # With the provider present but master-disabled, its own _enabled
        # flag keeps every on_* hook a no-op (belt and suspenders to the
        # unregister route).
        provider.set_enabled(False)
        provider.on_prompt_start("pid-x")  # must not raise / must stay inert
        check("T3 disabled provider stays inert", provider.is_enabled() is False)
        provider.set_enabled(True)
        check("T3 provider re-enabled", provider.is_enabled() is True)
    finally:
        cp._clear_cache_providers()


# ------------------------------------------------- T4: checkpoint bypass -----
def _checkpoint_checks() -> None:
    _switch_reset()
    try:
        from comfy import train_checkpoint as tc

        X = torch.randn(64, 8)  # a tensor whose digest would cost real time

        # Enabled: a real (hostless) run - run_key works, save refuses (no
        # prompt context) which is the documented bare-execution behaviour.
        key = tc.run_key({"node": "T", "X": X})
        check("T4 run_key computes while enabled", len(key) == 32, key)
        check("T4 save refuses without host context",
              tc.save(key, {"step": 1}) is False)

        # Disabled: everything short-circuits BEFORE any work.
        switch._master_enabled = False
        start = time.perf_counter()
        key_off = tc.run_key({"node": "T", "X": X})
        elapsed = time.perf_counter() - start
        check("T4 run_key is empty while disabled", key_off == "", key_off)
        check("T4 run_key skipped the digest (fast)",
              elapsed < 0.05, f"{elapsed * 1000:.1f} ms")
        check("T4 save short-circuits", tc.save("whatever", {"step": 1}) is False)
        check("T4 load short-circuits", tc.load("whatever") is None)
        tc.discard("whatever")  # must be a silent no-op
        check("T4 discard is a no-op", True)
        # The disabled run must not have written a checkpoint file for its
        # run key (the root may hold historical files - target ours only).
        check("T4 no checkpoint written for the disabled run",
              not tc._path("whatever").is_file())
    finally:
        _switch_reset()


# ------------------------------------------------------ T5: routes -----------
def _route_checks() -> None:
    try:
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
    except ImportError:
        check("R0 aiohttp available", False, "import failed")
        return

    from app.crashsite import routes

    provider = SnapshotProvider()
    provider.set_enabled(True)
    cp._clear_cache_providers()
    cp.register_cache_provider(provider)
    saved_provider = routes._SNAPSHOT_PROVIDER
    routes.set_snapshot_provider(provider)
    _switch_reset()
    try:
        async def _run():
            app = web.Application()
            app.add_routes(routes.routes)
            server = TestServer(app)
            client = TestClient(server)
            await client.start_server()
            try:
                r = await client.get("/comfydl/crashsite/enabled")
                data = await r.json()
                check("R1 GET enabled reflects the default",
                      r.status == 200 and data.get("enabled") is True
                      and data.get("hard_disabled") is False, str(data))

                r = await client.post(
                    "/comfydl/crashsite/enabled", json={"enabled": False})
                data = await r.json()
                check("R2 POST disable returns the new state",
                      r.status == 200 and data.get("enabled") is False, str(data))
                check("R2 provider unregistered from the executor",
                      cp._has_cache_providers() is False)
                check("R2 provider flag follows", provider.is_enabled() is False)

                r = await client.get("/comfydl/crashsite/enabled")
                data = await r.json()
                check("R3 GET reads back disabled", data.get("enabled") is False)

                # Resume must refuse while disabled - re-queueing into an
                # unregistered provider would run uncaptured.
                r = await client.post(
                    "/comfydl/crashsite/resume", json={"id": "whatever"})
                check("R4 resume refused while disabled", r.status == 409,
                      str(r.status))

                # The library stays browsable (no run-path interaction).
                r = await client.get("/comfydl/crashsite/snapshots")
                check("R5 library browsable while disabled", r.status == 200)

                r = await client.post(
                    "/comfydl/crashsite/enabled", json={"enabled": True})
                data = await r.json()
                check("R6 POST enable re-attaches",
                      r.status == 200 and data.get("enabled") is True, str(data))
                check("R6 provider registered again",
                      cp._has_cache_providers() is True
                      and provider.is_enabled() is True)

                # Hard-disabled: the POST must refuse to re-enable with 409.
                switch._hard_disabled = True
                switch._hard_checked = True
                switch._master_enabled = False
                r = await client.post(
                    "/comfydl/crashsite/enabled", json={"enabled": True})
                check("R7 hard-disabled POST rejected with 409", r.status == 409,
                      str(r.status))

                # Malformed bodies.
                r = await client.post("/comfydl/crashsite/enabled", data="not json")
                check("R8 invalid JSON rejected", r.status == 400)
                r = await client.post(
                    "/comfydl/crashsite/enabled", json={"enabled": "yes"})
                check("R8 non-bool enabled rejected", r.status == 400)
            finally:
                await client.close()

        import asyncio

        asyncio.run(_run())
    finally:
        cp._clear_cache_providers()
        routes.set_snapshot_provider(saved_provider)
        _switch_reset()


import time  # noqa: E402


def main() -> int:
    _flag_checks()
    _switch_checks()
    _provider_checks()
    _checkpoint_checks()
    _route_checks()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
