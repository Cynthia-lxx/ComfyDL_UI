#!/usr/bin/env python
"""Standalone smoke test: UI kit landing (U stick 1, [0.88]).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_ui_kit.py

Covers the static contracts of the osu-derived UI kit landing:

* T1  provenance: MIT licence chain + per-file headers + fresh assembly;
* T2  dist completeness: every manifest component has a dist copy;
* T3  token adaptation: the host font stack replaced the upstream fonts,
      --motion-scale still present (the reduced-motion contract);
* T4  server mounts: /comfydl/ui (kit dist) + /comfydl/ui-assets (bridge);
* T5  loader injection: exactly once, into <head>, link-only, idempotent;
* T6  bridge policy: surfaces/text/border follow official theme vars, the
      accent family is NOT mapped (hue themes keep control), the motion
      gates set --motion-scale 0;
* T7  spike page: loads the full chain and stamps machine-checkable
      data-* verdicts;
* T8  HTTP layer: the two mounts actually serve the assets (TestClient).

The real-browser spike (theme switch / motion gate / shadow DOM inside a
live page) runs separately via CDP against spike.html - see the stick-1
delivery note in the LM.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_uikit_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    # 铁律（U 批次起）：测试必须实时输出进度，不允许静默跑到最后。
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


KIT = REPO_ROOT / "ui_kit"
DIST = KIT / "dist"
UI_ASSETS = REPO_ROOT / "app" / "ui_assets"

# ---------------------------------------------------------------------------
# T1: provenance + fresh assembly

check("T1a MIT licence chain present",
      (KIT / "00-overview" / "LICENSE-MIT.txt").is_file()
      and (DIST / "LICENSE-MIT.txt").is_file()
      and (KIT / "dist" / "licence-audit.json").is_file())
button_src = (KIT / "04-components" / "ui-button" / "ui-button.js").read_text(
    encoding="utf-8")
check("T1b component files carry their provenance header",
      "SPDX-License-Identifier" in button_src or "MIT" in button_src[:400])
check("T1c dist is freshly assembled (tokens/themes/i18n/components)",
      (DIST / "tokens.css").is_file() and (DIST / "themes.css").is_file()
      and (DIST / "i18n" / "en.json").is_file()
      and (DIST / "i18n" / "zh-CN.json").is_file())

# ---------------------------------------------------------------------------
# T2: dist completeness against the manifest

import json  # noqa: E402

manifest = json.loads(
    (KIT / "00-overview" / "ui-manifest.json").read_text(encoding="utf-8-sig"))
ids = [c["id"] for c in manifest["components"]]
check("T2a manifest declares the 22-component library",
      len(ids) == 22 and "ui-button" in ids and "ui-graph" in ids,
      f"{len(ids)} components")
dist_components = {p.name for p in (DIST / "components").iterdir()
                   if p.is_dir()}
check("T2b every manifest component has a dist copy",
      dist_components == set(ids),
      f"missing={sorted(set(ids) - dist_components)}")

# ---------------------------------------------------------------------------
# T3: token adaptation (host font stack, reduced-motion contract intact)

tokens = (DIST / "tokens.css").read_text(encoding="utf-8")
check("T3a font stack adapted to the host (no upstream font primary)",
      '"Inter"' in tokens and "Torus" not in tokens.split("--font-family:")[1].split(";")[0])
check("T3b reduced-motion contract intact",
      "--motion-scale" in tokens
      and "prefers-reduced-motion" in tokens)

# ---------------------------------------------------------------------------
# T4: server mounts

server_src = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
check("T4a server mounts the kit dist and the bridge layer",
      "web.static('/comfydl/ui'" in server_src
      and "web.static('/comfydl/ui-assets'" in server_src)

# ---------------------------------------------------------------------------
# T5: loader injection

from app import frontend_patch as fp  # noqa: E402

check("T5a UIKIT loader is link-only (three stylesheets)",
      fp.UIKIT_LOADER_TAG.count("<link") == 3
      and "/comfydl/ui/tokens.css" in fp.UIKIT_LOADER_TAG
      and "/comfydl/ui/themes.css" in fp.UIKIT_LOADER_TAG
      and "/comfydl/ui-assets/bridge.css" in fp.UIKIT_LOADER_TAG
      and "<script" not in fp.UIKIT_LOADER_TAG)

web_root = Path(tempfile.mkdtemp(prefix="cdl_uikit_web_"))
index = web_root / "index.html"
index.write_text("<!doctype html><html><head><title>x</title></head>"
                 "<body><div>app</div></body></html>", encoding="utf-8")
check("T5b loader injects exactly once into <head>",
      fp._patch_uikit_loader(web_root) is True
      and index.read_text(encoding="utf-8").count(fp.UIKIT_LOADER_MARK) == 1
      and fp.UIKIT_LOADER_TAG in
      index.read_text(encoding="utf-8").split("</head>")[0])
check("T5c re-apply is a no-op",
      fp._patch_uikit_loader(web_root) is False
      and index.read_text(encoding="utf-8").count(fp.UIKIT_LOADER_MARK) == 1)

# ---------------------------------------------------------------------------
# T6: bridge policy

bridge = (UI_ASSETS / "bridge.css").read_text(encoding="utf-8")
check("T6a surfaces/text/border follow the official theme",
      "--comfy-menu-bg" in bridge and "--comfy-input-bg" in bridge
      and "--input-text" in bridge and "--border-color" in bridge)
check("T6b the accent family is NOT mapped (hue themes keep control)",
      "--ui-accent:" not in bridge)
check("T6c motion gates set the kit's scale to zero",
      "html.cdl-motion-off" in bridge and "--motion-scale: 0" in bridge)

# ---------------------------------------------------------------------------
# T7: spike page

spike = (UI_ASSETS / "spike.html").read_text(encoding="utf-8")
check("T7a spike loads the full chain",
      "/comfydl/ui/tokens.css" in spike
      and "/comfydl/ui/themes.css" in spike
      and "/comfydl/ui-assets/bridge.css" in spike
      and "/comfydl/ui/components/ui-button/ui-button.js" in spike)
check("T7b spike stamps machine-checkable verdicts",
      'document.documentElement.dataset' in spike
      and "themeSwitch" in spike and "cdl-motion-off" in spike)

# ---------------------------------------------------------------------------
# T8: HTTP layer (the two mounts actually serve)

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402


async def _route_checks() -> None:
    app = web.Application()
    app.add_routes([web.static("/comfydl/ui", DIST),
                    web.static("/comfydl/ui-assets", UI_ASSETS)])
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for path, needle in (
            ("/comfydl/ui/tokens.css", "--ui-accent"),
            ("/comfydl/ui/themes.css", "data-theme"),
            ("/comfydl/ui-assets/bridge.css", "--ui-surface"),
            ("/comfydl/ui-assets/spike.html", "spike-btn"),
            ("/comfydl/ui/components/ui-button/ui-button.js", "customElements"),
            ("/comfydl/ui/i18n/zh-CN.json", "common.name"),
        ):
            resp = await client.get(path)
            body = await resp.text()
            check(f"T8 route: {path}",
                  resp.status == 200 and needle in body,
                  f"status={resp.status} body_head={body[:120]!r}")
    finally:
        await client.close()


import asyncio  # noqa: E402

asyncio.run(_route_checks())

# ---------------------------------------------------------------------------
# T9: brand hygiene (naming-map discipline, dist artefacts)

banned = re.compile(r"(\bosu!?\b|\bppyl?\b)", re.IGNORECASE)
dirty = [p.name for p in DIST.rglob("*") if p.is_file() and banned.search(p.name)]
check("T9a no brand words in dist file names", dirty == [], str(dirty[:5]))

# ---------------------------------------------------------------------------
# T10: U stick 3 shared infrastructure (loader + B8 semantics + VFX)

loader_src = (UI_ASSETS / "kit-loader.js").read_text(encoding="utf-8")
sem_src = (UI_ASSETS / "kit_semantics.js").read_text(encoding="utf-8")
check("T10a shared loader imports every dist component by the canonical path",
      "ensureComponents" in loader_src
      and 'COMPONENT_BASE = "/comfydl/ui/components"' in loader_src
      and "${normalized}/${normalized}.js" in loader_src
      and "customElements.get" in loader_src
      and "import FAILED" in loader_src)

check("T10b B8 semantic colours have a single literal home (bridge.css)",
      "--cdl-verdict-ok" in bridge and "--cdl-verdict-warn" in bridge
      and "--cdl-verdict-danger" in bridge and "--cdl-chosen" in bridge
      and "verdictColor" in sem_src and "statusColor" in sem_src
      and "--cdl-verdict-ok" in sem_src)

check("T10c status-pulse / entrance timeline obey all three motion gates",
      "@keyframes cdl-status-pulse" in bridge
      and "@keyframes cdl-entrance-timeline" in bridge
      and "--motion-scale" in bridge
      and "html.cdl-motion-off .cdl-vfx-pulse" in bridge
      and "body.cdlapp-motion-off .cdl-vfx-entrance" in bridge
      and "prefers-reduced-motion" in bridge)

APP_ROOT = REPO_ROOT / "app"
p6b_src = (APP_ROOT / "p6b_assets" / "p6b.js").read_text(encoding="utf-8")
obs_src = (APP_ROOT / "obs_assets" / "obs.js").read_text(encoding="utf-8")
prof_src = (APP_ROOT / "profiling_assets" / "profiler.js").read_text(
    encoding="utf-8")
check("T10d all three panels use the shared loader",
      all('from "/comfydl/ui-assets/kit-loader.js"' in s
          for s in (p6b_src, obs_src, prof_src))
      and all("ensureComponents(" in s for s in (p6b_src, obs_src, prof_src)))
check("T10e natural kit surfaces land in each panel",
      '<ui-button' in p6b_src and "ui-press" in p6b_src
      and "<ui-spinner" in obs_src and "cdl-obs-hint-spinner" in obs_src
      and '<ui-textbox variant="number"' in prof_src
      and 'document.createElement("ui-tabs")' in prof_src)

# ---------------------------------------------------------------------------
# T11: post-stick-3 bug-fix contracts

prof_css = (APP_ROOT / "profiling_assets" / "profiler.css").read_text(
    encoding="utf-8")
cdlapp_src = (APP_ROOT / "cdlapp_assets" / "cdlapp.js").read_text(
    encoding="utf-8")
check("T11a kit buttons are never mistaken for a drag start",
      'closest("button, ui-button")' in p6b_src
      and 'closest("button, ui-button")' in obs_src)
check("T11b entrance replays on panel expansion",
      "replayEntrance" in p6b_src and "replayEntrance" in obs_src
      and "export function replayEntrance" in sem_src)
check("T11c Help menu pruned + dock yields while it is open",
      "pruneHelpMenu" in cdlapp_src
      and 'startsWith("ComfyDL")' in cdlapp_src
      and "!helpOpen" in cdlapp_src)
check("T11d markdown contrast, compact badge, exclusive profiler tabs",
      ".comfy-markdown-content" in bridge
      and 'button.comfyui-button[class*="cdlp-b-"]' in prof_css
      and "cdlp-tab-hidden" in prof_css
      and '"cdlp-tab-hidden", !selected' in prof_src)
check("T11e CrashSite topbar shows a glyph instead of an empty icon box",
      'content: "CS"' in (APP_ROOT / "crashsite_assets" / "crashsite.js").read_text(
          encoding="utf-8"))


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== ui kit test: {passed} PASS / "
          f"{len(_RESULTS) - passed} FAIL (of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
