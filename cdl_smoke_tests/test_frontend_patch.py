#!/usr/bin/env python
"""Standalone smoke test for the frontend default-workflow patch.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_frontend_patch.py

Covers ``app.frontend_patch._patch_default_workflow``:
  - the real installed frontend has the patch applied (the harness applies it
    first, mirroring what ``main.py`` does at startup) and re-applying is a
    no-op;
  - a synthetic ``settingStore`` chunk gets the redirected
    ``loadDefaultWorkflow`` body: it fetches the ComfyDL template through the
    ``/templates/<name>.json`` overlay channel and falls back to the upstream
    graph on any failure;
  - a chunk whose call signature drifted raises ``ValueError`` (fail loudly on
    frontend upgrades instead of silently keeping the SD demo).
"""

import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.frontend_patch import (  # noqa: E402
    DEFAULT_WORKFLOW_MARK,
    DEFAULT_WORKFLOW_TEMPLATE,
    STARTUP_WORKFLOW_MARK,
    _patch_default_workflow,
    _patch_startup_workflow,
    _upgrade_baked_template_url,
    apply_frontend_patches,
)

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


STOCK_BODY = (
    "loadDefaultWorkflow=async()=>{await Gx.loadGraphData(Gy)}"
    ",loadBlankWorkflow=async()=>{await Gx.loadGraphData(Gz)}"
)


def main() -> int:
    # --- real installed frontend -------------------------------------------- #
    web_root = Path(sys.prefix) / "Lib" / "site-packages" / "comfyui_frontend_package" / "static"
    check("real frontend package found", (web_root / "assets").is_dir())
    if (web_root / "assets").is_dir():
        apply_frontend_patches(str(web_root))
        assets = web_root / "assets"
        marked = [
            p.name
            for p in sorted(assets.glob("settingStore-*.js"))
            if DEFAULT_WORKFLOW_MARK in p.read_text(encoding="utf-8")
        ]
        check("real frontend: exactly one patched settingStore chunk", len(marked) == 1,
              str(marked))
        if marked:
            src = (assets / marked[0]).read_text(encoding="utf-8")
            check("real frontend: redirects to the ComfyDL template",
                  f"templates/{DEFAULT_WORKFLOW_TEMPLATE}.json" in src)
            check("real frontend: keeps the upstream fallback",
                  "loadGraphData(" in src and "catch(_cdlErr)" in src)
        # Startup workflow (2026-10-28 regression): the veteran-user branch of
        # resolveStartupOutcome must also go through the overlay template.
        gv_files = sorted(assets.glob("GraphView-*.js"))
        startup_srcs = [p for p in gv_files if STARTUP_WORKFLOW_MARK
                        in p.read_text(encoding="utf-8")]
        check("real frontend: exactly one patched GraphView chunk",
              len(startup_srcs) == 1, str([p.name for p in startup_srcs]))
        if startup_srcs:
            gsrc = startup_srcs[0].read_text(encoding="utf-8")
            check("real frontend: startup redirects to the ComfyDL template",
                  f"templates/{DEFAULT_WORKFLOW_TEMPLATE}.json" in gsrc
                  and "TutorialCompleted" in gsrc)
        # Generational upgrade (2026-10-30): a frontend baked with a previous
        # template name must not keep the stale URL anywhere.
        stale = [
            p.name
            for p in sorted(assets.glob("settingStore-*.js"))
            + sorted(assets.glob("GraphView-*.js"))
            if "templates/tabular_regression_production.json"
            in p.read_text(encoding="utf-8")
        ]
        check("real frontend: no stale baked template name", stale == [], str(stale))
        # Idempotency: a second apply must report nothing new.
        before = sorted(
            (p.name, p.stat().st_mtime) for p in assets.glob("settingStore-*.js")
        )
        apply_frontend_patches(str(web_root))
        after = sorted(
            (p.name, p.stat().st_mtime) for p in assets.glob("settingStore-*.js")
        )
        check("real frontend: re-apply is a no-op", before == after)

    # --- synthetic chunk: happy path ---------------------------------------- #
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_fepatch_"))
    try:
        assets = sandbox / "assets"
        assets.mkdir(parents=True)
        chunk = assets / "settingStore-TEST.js"
        chunk.write_text(STOCK_BODY, encoding="utf-8")

        check("synthetic: patch applied", _patch_default_workflow(assets) is True)
        text = chunk.read_text(encoding="utf-8")
        check("synthetic: marker present", DEFAULT_WORKFLOW_MARK in text)
        check("synthetic: fetches the overlay template",
              f"templates/{DEFAULT_WORKFLOW_TEMPLATE}.json" in text)
        check("synthetic: no-store cache busting", "no-store" in text)
        check("synthetic: upstream fallback kept", "loadGraphData(Gy)" in text)
        check("synthetic: loadBlankWorkflow untouched", "loadGraphData(Gz)" in text)
        check("synthetic: second apply is a no-op", _patch_default_workflow(assets) is False)
        # JS sanity: braces/parens stay balanced in the patched region.
        check("synthetic: balanced braces", text.count("{") == text.count("}"))
        check("synthetic: balanced parens", text.count("(") == text.count(")"))

        # --- synthetic chunk: drifted signature must fail loudly ------------ #
        assets2 = sandbox / "assets2"
        assets2.mkdir()
        drifted = assets2 / "settingStore-TEST.js"
        drifted.write_text("loadDefaultWorkflow=async()=>{await renamed()}", encoding="utf-8")
        try:
            _patch_default_workflow(assets2)
            check("drifted signature raises", False, "no ValueError")
        except ValueError:
            check("drifted signature raises", True)

        # --- synthetic chunk: startup workflow (resolveStartupOutcome) ------ #
        STOCK_STARTUP = (
            "resolveStartupOutcome=async()=>n.get(`Comfy.TutorialCompleted`)"
            "?(await X.loadGraphData(),`restored`):(await ir().loadBlankWorkflow())"
        )
        assets3 = sandbox / "assets3"
        assets3.mkdir()
        gv3 = assets3 / "GraphView-TEST.js"
        gv3.write_text(STOCK_STARTUP, encoding="utf-8")
        check("startup: patch applied", _patch_startup_workflow(assets3) is True)
        stext = gv3.read_text(encoding="utf-8")
        check("startup: marker present", STARTUP_WORKFLOW_MARK in stext)
        check("startup: fetches the overlay template",
              f"templates/{DEFAULT_WORKFLOW_TEMPLATE}.json" in stext)
        check("startup: veteran branch still returns restored",
              "`restored`" in stext)
        check("startup: upstream fallback kept",
              "await X.loadGraphData()" in stext and "catch(_cdlErr)" in stext)
        check("startup: second apply is a no-op",
              _patch_startup_workflow(assets3) is False)
        drifted3 = sandbox / "assets4"
        drifted3.mkdir()
        (drifted3 / "GraphView-TEST.js").write_text(
            "resolveStartupOutcome=async()=>renamed()", encoding="utf-8")
        try:
            _patch_startup_workflow(drifted3)
            check("startup: drifted signature raises", False, "no ValueError")
        except ValueError:
            check("startup: drifted signature raises", True)

        # --- generational upgrade helper ------------------------------------ #
        current_url = f"fetch(`/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json`"
        stale_body = "a" + "fetch(`/templates/some_old_template.json`,{cache:`no-store`})" + "b"
        upgraded, changed = _upgrade_baked_template_url(stale_body)
        check("upgrade: stale URL rewritten", changed and current_url in upgraded)
        check("upgrade: surrounding text kept",
              upgraded.startswith("a" + current_url) and upgraded.endswith("b"))
        unchanged, changed2 = _upgrade_baked_template_url(current_url + ",{cache:`no-store`})")
        check("upgrade: no-op when current", changed2 is False and unchanged == current_url + ",{cache:`no-store`})")
        official = "fetch(K.fileURL(`/templates/index_logo.json`))"
        official_text, changed3 = _upgrade_baked_template_url(official)
        check("upgrade: official fileURL fetch untouched",
              changed3 is False and official_text == official)
        # End-to-end on the patched chunks: pre-bake both targets with the old
        # name, the patch functions must upgrade in place and return True.
        assets5 = sandbox / "assets5"
        assets5.mkdir()
        store5 = assets5 / "settingStore-TEST.js"
        store5.write_text(
            "loadDefaultWorkflow=async()=>{try{const _cdlRes=await "
            "fetch(`/templates/some_old_template.json`,{cache:`no-store`});"
            "if(_cdlRes.ok){await Gx.loadGraphData(await _cdlRes.json());return}}"
            "catch(_cdlErr){}await Gx.loadGraphData(Gy)}"
            f"{DEFAULT_WORKFLOW_MARK}}}",
            encoding="utf-8",
        )
        check("upgrade e2e: default workflow upgraded",
              _patch_default_workflow(assets5) is True
              and current_url in store5.read_text(encoding="utf-8"))
        check("upgrade e2e: then idempotent", _patch_default_workflow(assets5) is False)
        gv5 = assets5 / "GraphView-TEST.js"
        gv5.write_text(
            "resolveStartupOutcome=async()=>n.get(`Comfy.TutorialCompleted`)"
            "?(await (async()=>{try{const _cdlRes=await "
            "fetch(`/templates/some_old_template.json`,{cache:`no-store`});"
            "if(_cdlRes.ok){await X.loadGraphData(await _cdlRes.json());return}}"
            "catch(_cdlErr){}})(),`restored`)"
            f"{STARTUP_WORKFLOW_MARK}:(await ir().loadBlankWorkflow())",
            encoding="utf-8",
        )
        check("upgrade e2e: startup workflow upgraded",
              _patch_startup_workflow(assets5) is True
              and current_url in gv5.read_text(encoding="utf-8"))
        check("upgrade e2e: then idempotent", _patch_startup_workflow(assets5) is False)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    return _report()


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== frontend patch test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
