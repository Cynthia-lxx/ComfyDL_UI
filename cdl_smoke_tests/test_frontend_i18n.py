#!/usr/bin/env python
"""Standalone smoke test for the frontend zh-locale baking patch (2026-10-28).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_frontend_i18n.py

The 0.3.14 frontend never calls the /i18n endpoint, so ``frontend_patch``
merges ``locales/zh/nodeDefs.json`` (+ settings names + settingsCategories)
directly into the frontend's own lazy-loaded zh chunks, located via content
anchors.  This test runs that patch against a THROWAWAY COPY of the homepage
baked frontend (``homepage/ui`` — the same dist layout as the pip package,
with different hashes, which is exactly the cross-build robustness the
content anchors must provide) and asserts the merged output.
"""

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
        HELP_MENU_MARK,
        I18N_MAIN_MARK,
        I18N_NODEDEFS_MARK,
        I18N_SETTINGS_KEYS,
        I18N_SETTINGS_MARK,
        _locate,
        _patch_help_menu,
        _patch_zh_locales,
        _text,
        _zh_chunk_names,
    )

    # Prefer the pip frontend (the target that ships and gets patched at
    # runtime); fall back to the homepage baked copy which shares the dist
    # layout.  Note: the homepage copy has its own baked help-menu rewrite,
    # so only the B-series (zh locale) assertions run against it reliably —
    # the help-menu H-series requires the pip frontend's pristine GraphView.
    pip_ui = (REPO_ROOT / "penv" / "Lib" / "site-packages"
              / "comfyui_frontend_package" / "static")
    src = pip_ui if (pip_ui / "assets").is_dir() else REPO_ROOT / "homepage" / "ui"
    if src == pip_ui:
        check("B0 pip frontend copy present", True, str(src))
    elif not (src / "i18n-7FfLzQir.js").is_file() and not list(src.glob("i18n-*.js")):
        check("B0 frontend copy present", False, str(src))
        return report()
    else:
        check("B0 homepage frontend copy present", True, str(src))

    sandbox = Path(tempfile.mkdtemp(prefix="cdl_i18n_bake_")) / "ui"
    shutil.copytree(src, sandbox)
    assets = (sandbox / "assets") if (sandbox / "assets").is_dir() else sandbox

    # The chunk resolver must find the zh mapping via content, not hashes.
    try:
        names = _zh_chunk_names(assets)
        check("B1 zh chunk names resolved via content anchors",
              all(k in names for k in ("main", "nodeDefs", "settings")), str(names))
    except Exception as exc:  # noqa: BLE001
        check("B1 zh chunk names resolved via content anchors", False, repr(exc))
        return report()

    nd_before = set(assets.glob("nodeDefs-*.js"))
    try:
        applied = _patch_zh_locales(assets)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check("B2 patch applies cleanly", False, repr(exc))
        return report()
    # applied False is fine when the source frontend was already baked by a
    # live server run (the pip frontend is patched in place at startup) —
    # then the markers must already be there, which B3+ verify anyway.
    check("B2 patch applies cleanly (or source already baked)",
          applied is True
          or (applied is False
              and I18N_NODEDEFS_MARK in _text(assets / names["nodeDefs"])),
          str(applied))

    nd_text = _text(assets / names["nodeDefs"])
    check("B3 nodeDefs chunk carries the marker", I18N_NODEDEFS_MARK in nd_text)
    check("B3 core node title baked (RMSNorm)", "归一化（RMSNorm）" in nd_text)
    check("B3 comfydl node title baked (Histogram)",
          "Histogram（直方图）" in nd_text)
    check("B3 nodeDefs merged as Object.assign on the default export",
          ";Object.assign(" in nd_text and I18N_NODEDEFS_MARK in nd_text)
    check("B3 nodeDefs chunk count unchanged (in-place edit)",
          set(assets.glob("nodeDefs-*.js")) == nd_before)

    st_text = _text(assets / names["settings"])
    check("B4 settings chunk carries the marker", I18N_SETTINGS_MARK in st_text)
    ok_keys = all(f"as {key}" in st_text for key in I18N_SETTINGS_KEYS)
    check("B4 all ComfyDL settings export keys present", ok_keys,
          str([k for k in I18N_SETTINGS_KEYS if f"as {k}" not in st_text]))
    check("B4 settings names in Chinese", "启用 ComfyDL 事故现场（总开关）" in st_text)

    mn_text = _text(assets / names["main"])
    check("B5 main chunk carries the marker", I18N_MAIN_MARK in mn_text)
    check("B5 settingsCategories group names baked",
          "事故现场" in mn_text and "性能分析" in mn_text)

    # Idempotency: a second run must be a no-op (markers already present).
    check("B6 second run is a no-op", _patch_zh_locales(assets) is False)
    check("B6 markers still present after re-run",
          I18N_NODEDEFS_MARK in _text(assets / names["nodeDefs"])
          and I18N_SETTINGS_MARK in _text(assets / names["settings"])
          and I18N_MAIN_MARK in _text(assets / names["main"]))

    # --- Help menu group (homepage fork merge) --------------------------------
    try:
        gv = _locate(assets, "GraphView-*.js", "Comfy.ContactSupport")
    except Exception as exc:  # noqa: BLE001
        check("H1 GraphView chunk located", False, repr(exc))
        return report()
    check("H1 GraphView chunk located", True, gv.name)

    try:
        applied_help = _patch_help_menu(assets)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check("H2 help menu patch applies cleanly", False, repr(exc))
        return report()
    # Same as B2: False means the source was already baked by a live run.
    check("H2 help menu patch applies cleanly (or source already baked)",
          applied_help is True
          or (applied_help is False and HELP_MENU_MARK in _text(gv)),
          str(applied_help))

    gv_text = _text(gv)
    check("H3 all four ComfyDL items injected",
          all(key in gv_text for key in
              ("cdl-docs", "cdl-qq", "cdl-bilibili", "cdl-repo")))
    check("H3 group labels use ComfyDL prefix",
          "ComfyDL · 项目文档" in gv_text and "ComfyDL · QQ 交流群" in gv_text
          and "ComfyDL · Bilibili" in gv_text and "ComfyDL · GitHub 仓库" in gv_text)
    check("H3 official entries preserved",
          all(key in gv_text for key in
              ("key:`feedback`", "key:`docs`", "key:`discord`", "key:`github`")))
    check("H3 injected before the manager push",
          gv_text.find("cdl-docs") < gv_text.find("return e.push({key:`manager`"))
    check("H3 chunk carries the marker", HELP_MENU_MARK in gv_text)
    check("H4 second run is a no-op", _patch_help_menu(assets) is False)

    return report()


if __name__ == "__main__":
    raise SystemExit(main())
