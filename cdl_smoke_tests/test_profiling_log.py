"""Smoke tests for the profiling four-level log system (reform: Profiling M3+).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling_log.py

Covers the 2026-10-06 diagnostics layer:

* ``--cdl-profiling-log`` CLI flag (default off, four choices, rejects junk);
* ``comfy/profiling/proflog.py`` gating (off emits nothing, header can raise
  but never lower, runtime override, never raises);
* the real routes through an aiohttp TestServer (logging stays silent with
  the default off, and a request header raises it without a restart);
* static checks of ``profiler.js``: bracket balance after stripping strings
  and comments (no node.js on this machine - memory pit 24), plus the
  renderOverlay definition that the 2026-10-06 Analyze/Expand wedge fix
  hinged on.
"""

import io
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy.profiling import proflog  # noqa: E402


# ---------------------------------------------------------------- capture --


class _Capture(logging.Handler):
    """Collect the formatted records the profiling logger emits."""

    def __init__(self):
        super().__init__()
        self.lines: list = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _capture():
    cap = _Capture()
    logger = logging.getLogger("ComfyDL.profiling")
    logger.addHandler(cap)
    logger.setLevel(logging.INFO)
    return cap, lambda: logger.removeHandler(cap)


class _FakeRequest:
    """Just enough of aiohttp's request for proflog.level_for()."""

    def __init__(self, headers=None):
        self.headers = headers or {}


# --------------------------------------------------------------- T1: flag --


def _flag_checks() -> None:
    from comfy import cli_args

    args = cli_args.parser.parse_args([])
    check("T1a default is off", args.cdl_profiling_log == "off", args.cdl_profiling_log)
    args = cli_args.parser.parse_args(["--cdl-profiling-log", "high"])
    check("T1b accepts high", args.cdl_profiling_log == "high", args.cdl_profiling_log)
    args = cli_args.parser.parse_args(["--cdl-profiling-log", "low"])
    check("T1c accepts low", args.cdl_profiling_log == "low", args.cdl_profiling_log)
    try:
        cli_args.parser.parse_args(["--cdl-profiling-log", "VERBOSE"])
        check("T1d rejects unknown level", False, "SystemExit not raised")
    except SystemExit:
        check("T1d rejects unknown level", True)


# ------------------------------------------------------------- T2: proflog --


def _proflog_checks() -> None:
    cap, drop = _capture()
    try:
        proflog.set_level(None)
        check("T2a startup default is off", proflog.get_level() == "off", proflog.get_level())

        proflog.log("low", "quiet %s", "msg")
        check("T2b off emits nothing", cap.lines == [], str(cap.lines))

        proflog.set_level("low")
        check("T2c override works", proflog.get_level() == "low")
        proflog.log("low", "loud %s", "msg")
        check("T2d low emits at low", len(cap.lines) == 1 and "loud msg" in cap.lines[0],
              str(cap.lines))

        cap.lines.clear()
        proflog.log("medium", "not yet")
        check("T2e medium gated at low", cap.lines == [], str(cap.lines))

        req = _FakeRequest({"X-CDL-Profiling-Log": "high"})
        check("T2f header raises the level", proflog.level_for(req) == "high")
        proflog.log("medium", "header raised", request=req)
        check("T2g medium emits with header", len(cap.lines) == 1, str(cap.lines))

        req_low = _FakeRequest({"X-CDL-Profiling-Log": "off"})
        check("T2h header cannot lower", proflog.level_for(req_low) == "low",
              proflog.level_for(req_low))

        req_junk = _FakeRequest({"X-CDL-Profiling-Log": "NOT-A-LEVEL"})
        check("T2i junk header ignored", proflog.level_for(req_junk) == "low")

        check("T2j is_enabled gate", proflog.is_enabled("low") is True
              and proflog.is_enabled("high") is False)
    finally:
        proflog.set_level(None)
        drop()

    # Keyword-only request: format args can never be swallowed (T2d guard).
    cap, drop = _capture()
    try:
        proflog.set_level("low")
        proflog.log("low", "sum %d", 42)
        check("T2l args reach the formatter", "sum 42" in (cap.lines[0] if cap.lines else ""),
              str(cap.lines))
    finally:
        proflog.set_level(None)
        drop()

    try:
        proflog.set_level("yelling")
        check("T2k set_level rejects junk", False, "no ValueError")
    except ValueError:
        check("T2k set_level rejects junk", True)


# -------------------------------------------------------------- T3: routes --


def _route_checks() -> None:
    try:
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
    except ImportError:
        check("T3 aiohttp available", False, "import failed")
        return

    import asyncio

    import cdl_smoke_tests.run_smoke_test as harness
    from app import profiling_routes

    # The opgraph route resolves the host node registry lazily; bootstrap the
    # reduced smoke-harness registry (same as test_profiling_opgraph).
    import tempfile

    harness._bootstrap(Path(tempfile.mkdtemp(prefix="cdl_proflog_")))
    harness._load_registry(False)

    async def _run():
        app = web.Application()
        app.add_routes(profiling_routes.routes)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            return client
        except Exception:
            await client.close()
            raise

    async def _checks():
        client = await _run()
        cap, drop = _capture()
        try:
            # 1. Default (off): requests succeed but log nothing.
            resp = await client.post("/comfydl/profiling/estimate", json={"prompt": {}})
            check("T3a estimate route 200", resp.status == 200, str(resp.status))
            check("T3b off: no log lines", cap.lines == [], str(cap.lines))

            resp = await client.post("/comfydl/profiling/opgraph", json={"prompt": {}})
            check("T3c opgraph route 200", resp.status == 200, str(resp.status))
            # P0: without the dangerous header the route answers safe-mode and
            # executes nothing. The refusal line is a "low" log, so at log
            # level off it is muted too (off = zero output, by contract).
            check("T3d off: safe mode silent", cap.lines == [], str(cap.lines))

            # 2. Header raises to high without any server restart; the analyze
            # path additionally needs the P0 dangerous-mode acknowledgement.
            headers = {"X-CDL-Profiling-Log": "high",
                       "X-CDL-Profiling-Dangerous": "1"}
            resp = await client.post("/comfydl/profiling/opgraph",
                                     json={"prompt": {}}, headers=headers)
            check("T3e opgraph route 200 (header)", resp.status == 200, str(resp.status))
            joined = "\n".join(cap.lines)
            check("T3f lifecycle lines appear", "[profiling:low]" in joined
                  and "opgraph analyze" in joined, joined[:200])
            check("T3g rank tag present", "[profiling:high]" in joined or "[profiling:low]" in joined)

            cap.lines.clear()
            resp = await client.post("/comfydl/profiling/estimate",
                                     json={"prompt": {}}, headers=headers)
            check("T3h estimate 200 (header)", resp.status == 200, str(resp.status))
            check("T3i estimate logs verdict", any("estimate:" in ln for ln in cap.lines),
                  str(cap.lines[:2]))

            cap.lines.clear()
            resp = await client.post("/comfydl/profiling/postmortem",
                                     json={"message": "CUDA out of memory"},
                                     headers={"X-CDL-Profiling-Log": "low"})
            check("T3j postmortem 200", resp.status == 200, str(resp.status))
            check("T3k postmortem logs", any("[profiling:low]" in ln for ln in cap.lines),
                  str(cap.lines[:2]))
        finally:
            drop()
            await client.close()

    asyncio.run(_checks())


# ------------------------------------------------------- T4: profiler.js --


def _strip_js(source: str) -> str:
    """Remove comments and quoted strings (naive, escape-aware).

    Regex literals in this file contain no quotes or braces, so leaving them
    in is safe for bracket balancing (memory pit 24: stripping them naively
    breaks; keeping them works for this specific file).
    """
    out = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c in "\"'":
            quote = c
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _js_checks() -> None:
    path = REPO_ROOT / "app" / "profiling_assets" / "profiler.js"
    src = path.read_text(encoding="utf-8")

    # Bracket balance (the node-less syntax fallback).
    stripped = _strip_js(src)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    balanced = True
    for ch in stripped:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                balanced = False
                break
    check("T4a profiler.js brackets balanced", balanced and not stack,
          f"stack={len(stack)}")

    # The renderOverlay fix: defined exactly once, still called everywhere.
    defined = re.findall(r"function\s+renderOverlay\s*\(", src)
    called = re.findall(r"(?<!function )(?<!function\s)renderOverlay\s*\(", src)
    check("T4b renderOverlay defined once", len(defined) == 1, str(defined))
    check("T4c renderOverlay has call sites", len(called) >= 4, str(len(called)))

    # Frontend diagnostics surface.
    for needle, name in [
        ("function plog(", "T4d plog defined"),
        ("X-CDL-Profiling-Log", "T4e request header set"),
        ("ComfyDL.Profiling.LogLevel", "T4f settings id registered"),
        ("/internal/logs/raw", "T4g server log endpoint used"),
        ("} finally {", "T4h analyze uses try/finally"),
    ]:
        check(name, needle in src)

    # Every plog rank must be a valid level.
    bad_ranks = set(re.findall(r'plog\(\s*"([a-z]+)"', src)) - set(proflog.LEVELS)
    check("T4i plog ranks valid", not bad_ranks, str(sorted(bad_ranks)))

    # Startup must never auto-open the overlay (2026-10-06 user feedback:
    # a full-screen dashboard covering the workflow on load disorients).
    check("T4j display mode starts closed", 'displayMode: "sidebar"' in src)
    check("T4k overlay mode not persisted", "cdlpDisplayMode" not in src
          and "DISPLAY_MODE_KEY" not in src)

    # P0 dangerous-probe gate (2026-10-07): opt-in setting, per-request header,
    # safe-mode hint wired into the i18n dictionaries.
    for needle, name in [
        ("ComfyDL.Profiling.DangerousProbe", "T4l dangerous setting registered"),
        ("X-CDL-Profiling-Dangerous", "T4m dangerous header sent"),
        ("opgraph_safe_mode", "T4n safe-mode hint i18n"),
        ("danger_confirm", "T4o enable confirmation i18n"),
        ("revertDangerous", "T4p declining reverts the toggle"),
    ]:
        check(name, needle in src)
    for key in ("opgraph_safe_mode", "danger_confirm"):
        check(f"T4q zh+en have {key}",
              src.count(key + ":") >= 2, str(src.count(key + ":")))


def main() -> int:
    _flag_checks()
    _proflog_checks()
    _route_checks()
    _js_checks()
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
