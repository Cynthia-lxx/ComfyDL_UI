"""Smoke tests for the VERBOSE diagnostics (repair R0, smallplans §0.90).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_verbose.py

Covers: OFF-by-default zero work, dual-timestamp line format, WS
executing/executed correlation, front-end ingest into the same log, the
timed decorator, the KSampler callback wrapper, sampled steps, hot toggle,
and the static wiring of the two central hooks.
"""

import sys
import tempfile
import time
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


from comfy import verbose_log as vl  # noqa: E402


def _static_checks() -> None:
    routes = (REPO_ROOT / "app" / "verbose_routes.py").read_text(
        encoding="utf-8")
    server = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    sample = (REPO_ROOT / "comfy" / "sample.py").read_text(encoding="utf-8")
    patch = (REPO_ROOT / "app" / "frontend_patch.py").read_text(
        encoding="utf-8")
    check("V11 verbose routes expose switch + front-end ingest",
          '/comfydl/verbose"' in routes
          and "/comfydl/verbose/fe" in routes
          and "verbose_log.state()" in routes)
    check("V12 server wires middleware + WS hook + routes + assets",
          "verbose_http" in server
          and "verbose_log.ws_event(event, data)" in server
          and "verbose_routes.routes" in server
          and "web.static('/comfydl/vassets', verbose_assets)" in server)
    check("V13 KSampler callback wrapper preserves the original",
          "wrap_sampler_callback(callback)" in sample)
    check("V14 verbose loader keeps the OBS constants untouched",
          "VERBOSE_LOADER_MARK" in patch
          and "_patch_verbose_loader" in patch
          and 'OBS_LOADER_MARK = "<!--ComfyDL_UI:obs-loader-->"' in patch
          and '/comfydl/obs/obs.js' in patch)


def main() -> int:
    # --- OFF is the default and the hot path does literally nothing -------
    vl.set_enabled(False)
    check("V1 default off", vl.enabled() is False
          and vl.state()["level"] == "off")
    before = vl.state()["queued"]
    for i in range(500):
        vl.log("test", "event", n=i)
        vl.ws_event("executing", "7")
        vl.step("test", "tick", i, every_n=3)
        vl.ingest(["should-not-appear"])
    check("V2 off path queues nothing",
          vl.state()["queued"] == before,
          str(vl.state()["queued"]))

    with tempfile.TemporaryDirectory(prefix="cdl_verbose_") as tmp:
        log_path = Path(tmp) / "verbose.log"
        vl._LOG_PATH = log_path
        state = vl.set_enabled(True, "verbose")
        check("V3 enable returns state", state["enabled"] is True
              and state["level"] == "verbose")

        vl.log("test", "hello", n=3, tag="a b")
        vl.ws_event("executing", "7")
        vl.ws_event("executed", {"node": "7", "prompt_id": "p1"})
        vl.ingest(["front-end line"])

        @vl.timed("test", "decorated")
        def work():
            return 42

        check("V7 timed decorator preserves the return value", work() == 42)

        calls = []
        wrapped = vl.wrap_sampler_callback(
            lambda s, x0, x, t: calls.append((s, t)))
        wrapped(1, None, None, 20)
        check("V8 sampler callback forwarded unchanged", calls == [(1, 20)])

        vl.step("test", "tick", 4, every_n=4, ok=1)
        vl.step("test", "tick", 5, every_n=4, ok=0)

        deadline = time.time() + 2.0
        while time.time() < deadline and not log_path.is_file():
            time.sleep(0.02)
        time.sleep(0.4)  # let the daemon writer flush its batch
        text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        lines = [line for line in text.splitlines()
                 if line.startswith("[CDL-V")]
        joined = "\n".join(lines)

        check("V4 dual timestamp + stage/event + sanitized fields",
              any("[CDL-V]" in line and "test hello" in line
                  and "n=3" in line and "tag=a_b" in line
                  for line in lines), text[:200])
        check("V5 executing/executed correlated into node_ms",
              "ws executing node=7" in joined
              and any("ws executed node=7" in line and "node_ms=" in line
                      for line in lines), joined[:300])
        check("V6 front-end lines share the same log",
              any("[CDL-V-FE]" in line and "front-end_line" in line
                  for line in lines), joined[:300])
        check("V9 timed decorator emits duration_ms",
              "test decorated" in joined and "duration_ms=" in joined)
        check("V10 sampler step + sampled tick logged",
              "sampler step step=1 total=20" in joined
              and "test tick n=4" in joined
              and "test tick n=5" not in joined, joined[-300:])

        vl.set_enabled(False)
        vl.log("test", "after-off")
        time.sleep(0.4)
        text2 = log_path.read_text(encoding="utf-8")
        check("V15 hot toggle off stops new lines",
              "test after-off" not in text2
              and vl.enabled() is False, text2[-200:])
        vl._LOG_PATH = None

    _static_checks()
    return report()


if __name__ == "__main__":
    sys.exit(main())
