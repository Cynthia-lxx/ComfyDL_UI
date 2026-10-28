#!/usr/bin/env python
"""Standalone smoke test for the Motion module (2026-10-28).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_motion.py

Covers the three moving parts against THROWAWAY COPIES of the live frontend:
  M1  index.html patches  - pre-hide CSS before </head>, loader before </body>,
      both idempotent.
  M2  motion.js contract  - gate classes, reduced-motion, ripple overlay,
      settings id, and the opacity-only node sweep (transform would collapse
      every node onto the canvas origin and freeze drags - regression guard).
  M3  zh bake constants   - the Motion setting translation key and the
      settingsCategories group name exist in frontend_patch.
"""
import re
import shutil
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


def main() -> int:
    from app.frontend_patch import (
        I18N_SETTINGS_CATEGORIES_ZH,
        I18N_SETTINGS_KEYS,
        MOTION_CSS_MARK,
        MOTION_LOADER_MARK,
        _patch_motion_css,
        _patch_motion_loader,
        _text,
    )

    pip_ui = (REPO_ROOT / "penv" / "Lib" / "site-packages"
              / "comfyui_frontend_package" / "static")
    if not (pip_ui / "index.html").is_file():
        check("M0 pip frontend present", False, str(pip_ui))
        return report()
    check("M0 pip frontend present", True)

    sandbox = Path(tempfile.mkdtemp(prefix="cdl_motion_"))
    shutil.copytree(pip_ui, sandbox, dirs_exist_ok=True)
    index = sandbox / "index.html"
    baseline = _text(index)

    # --- M1: index.html patches -------------------------------------------
    try:
        css_applied = _patch_motion_css(sandbox)
        loader_applied = _patch_motion_loader(sandbox)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("M1 patches apply cleanly", False, repr(exc))
        return report()
    # applied False is fine when the source index.html was already baked by a
    # live server run (the pip frontend is patched in place at startup) —
    # then the marks must already be there, which the next checks verify.
    already = MOTION_CSS_MARK in baseline or MOTION_LOADER_MARK in baseline
    check("M1 patches apply cleanly (or source already baked)",
          (css_applied and loader_applied)
          or (already and not css_applied and not loader_applied),
          f"css={css_applied} loader={loader_applied}")

    text = _text(index)
    check("M1 css mark present", MOTION_CSS_MARK in text)
    check("M1 loader mark present", MOTION_LOADER_MARK in text)
    head_at = text.find("</head>")
    css_at = text.find(MOTION_CSS_MARK)
    check("M1 css injected before </head>",
          0 <= css_at < head_at, f"css@{css_at} head@{head_at}")
    body_at = text.find("</body>")
    loader_at = text.find(MOTION_LOADER_MARK)
    check("M1 loader injected before </body>",
          0 <= loader_at < body_at, f"loader@{loader_at} body@{body_at}")

    # Idempotency: a second run must leave the file byte-identical.
    check("M1 css idempotent", _patch_motion_css(sandbox) is False)
    check("M1 loader idempotent", _patch_motion_loader(sandbox) is False)
    check("M1 no duplicate markers",
          text.count(MOTION_CSS_MARK) == 1 and text.count(MOTION_LOADER_MARK) == 1)

    # Baseline contract: the pip frontend is either pristine (both marks
    # absent, patches just applied) or fully baked by a live run (both marks
    # already present). A half-baked index.html would mean a torn patch.
    check("M1 baseline is pristine-or-fully-baked",
          ((MOTION_CSS_MARK not in baseline) and (MOTION_LOADER_MARK not in baseline))
          or ((MOTION_CSS_MARK in baseline) and (MOTION_LOADER_MARK in baseline)))

    # --- M2: motion.js contract -------------------------------------------
    motion = REPO_ROOT / "app" / "motion_assets" / "motion.js"
    src = motion.read_text(encoding="utf-8")

    check("M2 settings id", "ComfyDL.Motion.Enabled" in src)
    check("M2 ComfyDL Motion category", 'category: ["ComfyDL", "Motion"]' in src)
    check("M2 registers extension", 'name: "ComfyDL.Motion"' in src)
    for gate in ("cdl-motion-off", "cdl-entrance-done", "cdl-nodes-in", "cdl-canvas-in"):
        check(f"M2 gate class set: {gate}", f'"{gate}"' in src)
    check("M2 reduced-motion respected",
          "prefers-reduced-motion: reduce" in src and "reduceMotion()" in src)
    check("M2 ripple overlay layer", "cdl-ripple-layer" in src)
    check("M2 ripple only on left button", "e.button !== 0" in src)
    check("M2 unloadable ripple (setting off)", "uninstallRipple" in src)
    check("M2 sidebar panel expand/collapse animation",
          "sidebar-content-container" in src and "side-bar-panel" in src
          and "installSidebarPanelAnimation" in src
          and "uninstallSidebarPanelAnimation" in src)
    check("M2 shared easing", "cubic-bezier(.16,1,.3,1)" in src)
    check("M2 cdlCancelAfter contract", "cdlCancelAfter" in src)
    check("M2 populate-gated containers", "playWhenPopulated" in src)
    check("M2 zh strings present", "显存" not in src and "界面动效" in src)

    # Regression guard (homepage lesson): the node sweep animates opacity
    # ONLY. Every keyframes array applied to [data-node-id] must not carry a
    # transform; the only transforms allowed are on chrome containers
    # (topbar/bottom/right/sidebar), which Vue does not position via transform.
    sweep_idx = src.find("const sweep = () => {")
    sweep_body = src[sweep_idx:src.find("}, 150);", sweep_idx)]
    check("M2 node sweep exists", sweep_idx > 0 and "data-node-id" in sweep_body)
    node_kf = re.findall(r"animate\([^,]+,\s*\[([^\]]+)\]", sweep_body)
    check("M2 node sweep opacity-only (no transform)",
          bool(node_kf) and all("transform" not in kf for kf in node_kf),
          str(node_kf))

    # --- M3: zh bake constants ---------------------------------------------
    check("M3 Motion setting key in zh table",
          "ComfyDL_Motion_Enabled" in I18N_SETTINGS_KEYS
          and I18N_SETTINGS_KEYS["ComfyDL_Motion_Enabled"]["name"]
          == "界面动效（入场编排 + 点击涟漪）")
    check("M3 Motion group in zh categories",
          I18N_SETTINGS_CATEGORIES_ZH.get("Motion") == "动效")

    return report()


if __name__ == "__main__":
    raise SystemExit(main())
