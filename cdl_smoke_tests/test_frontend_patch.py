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
    _patch_default_workflow,
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
