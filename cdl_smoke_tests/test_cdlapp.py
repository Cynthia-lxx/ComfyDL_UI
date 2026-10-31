#!/usr/bin/env python
"""Standalone smoke test for the CDL App Panel (2026-10-30).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_cdlapp.py

Covers the panel's moving parts:
  A0  assets present, self-contained (no CDN/external URLs, read-only panel)
  A1  index.html loader patch - applied, idempotent
  A2  cdlapp.js contract  - OFFICIAL_SURFACE as the single source of official
      internals, settingsStoreReady gate, idempotent mount, soft degradation,
      z-index inside the measured band, no writes over the wire
  A3  pip frontend anchor pins - every OFFICIAL_SURFACE entry is verified
      against the INSTALLED frontend chunks.  When the pip frontend package is
      upgraded these fail loudly with migration guidance (ADR-8).
  A4  zh bake constants - the AppPanel setting translation key + category.
"""
import importlib.metadata
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []

# settings ids declared in cdlapp.js (pinned in the A2/A5 checks)
SETTING_MOTION_ID = '"ComfyDL.AppPanel.Motion"'


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
        CDLAPP_LOADER_MARK,
        CDLAPP_LOADER_TAG,
        I18N_SETTINGS_CATEGORIES_ZH,
        I18N_SETTINGS_KEYS,
        _patch_cdlapp_loader,
        _text,
    )

    repo = REPO_ROOT / "app" / "cdlapp_assets"
    js_path = repo / "cdlapp.js"
    css_path = repo / "cdlapp.css"
    if not js_path.is_file() or not css_path.is_file():
        check("A0 cdlapp assets present", False, str(repo))
        return report()
    check("A0 cdlapp assets present", True)
    js = js_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")

    pip_ui = (REPO_ROOT / "penv" / "Lib" / "site-packages"
              / "comfyui_frontend_package" / "static")

    # --- A0: self-contained, read-only panel --------------------------------
    check("A0 no external CDN/URL fetches in cdlapp.js",
          # www.w3.org is exempt: it is the SVG *namespace identifier* used
          # by createElementNS - a spec URI, never a network fetch.
          not re.search(r'https?://(?!localhost|127\.0\.0\.1|www\.w3\.org)', js))
    check("A0 panel never POSTs (read-only consumer)",
          not re.search(r'method:\s*["\']POST["\']', js))
    check("A0 z-index inside safe band (2001-9999)",
          "z-index: 2100" in css and 2001 <= 2100 <= 9999)
    check("A0 css uses official theme variables",
          "--comfy-menu-bg" in css and "--fg-color" in css)

    # --- A1: loader patch ----------------------------------------------------
    if not (pip_ui / "index.html").is_file():
        check("A1 pip frontend present", False, str(pip_ui))
        return report()
    check("A1 pip frontend present", True)
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_cdlapp_"))
    try:
        shutil.copytree(pip_ui, sandbox, dirs_exist_ok=True)
        index = sandbox / "index.html"
        # The production frontend is already baked (a real server start applies
        # the patches); simulate a fresh install by stripping the baked loader
        # from the sandbox copy, then re-apply through the real code path.
        if CDLAPP_LOADER_TAG in _text(index):
            index.write_text(_text(index).replace(CDLAPP_LOADER_TAG, ""),
                             encoding="utf-8")
        baseline = _text(index)
        try:
            first = _patch_cdlapp_loader(sandbox)
            check("A1 loader applied on fresh copy", first is True)
            check("A1 loader tag present before </body>",
                  CDLAPP_LOADER_MARK in _text(index)
                  and _text(index).index(CDLAPP_LOADER_MARK)
                  < _text(index).index("</body>"))
            check("A1 baseline otherwise untouched",
                  _text(index)
                  == baseline.replace("</body>", CDLAPP_LOADER_TAG + "</body>"),
                  "loader patch must be the only difference from the baseline")
            check("A1 loader idempotent", _patch_cdlapp_loader(sandbox) is False)
        except ValueError as exc:
            check("A1 loader applied on fresh copy", False, str(exc))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    # --- A2: cdlapp.js contract ----------------------------------------------
    check("A2 OFFICIAL_SURFACE declared once",
          len(re.findall(r"OFFICIAL_SURFACE\s*=\s*\{", js)) == 1)
    # Single source of official internals: the sensitive literals may only
    # appear inside the OFFICIAL_SURFACE block (or in the pin comments here).
    surface_end = js.find("// --------------------------------------------------------------- i18n")
    surface = js[:surface_end]
    rest = js[surface_end:]
    for literal in ("viewMode", "Comfy.ToggleLinear"):
        check(f"A2 {literal!r} only inside OFFICIAL_SURFACE",
              literal in surface and literal not in rest)
    for ev in ("progress_state", "execution_error", "execution_success",
               "execution_interrupted", "execution_cached"):
        check(f"A2 event {ev!r} only inside OFFICIAL_SURFACE",
              ev in surface and ev not in rest)
    check("A2 settingsStoreReady gate exists and is used",
          "function settingsStoreReady" in js and "settingsStoreReady(app)" in js)
    check("A2 boot degrades loudly after 30s",
          "registering without the gate" in js)
    check("A2 setting id + onChange master switch",
          SETTING := "ComfyDL.AppPanel.Enabled" in js
          and "onChange" in js and "applyEnabled" in js)
    check("A2 extension name registered",
          'name: "ComfyDL.AppPanel"' in js)
    check("A2 mount/unmount idempotent guards",
          "if (state.mounted || !state.enabled) return" in js
          and "if (!state.mounted) return" in js)
    # batch 13: the builder wizard (viewMode "builder:*") is a full-screen
    # official editor - the panel must dock ONLY in the pure "app" view and
    # unmount completely while the builder is active (user-reported buggy
    # builder: our dock CSS wrecked its layout).
    check("A2 builder wizard stays pristine (dock only in pure app view)",
          "function isPureAppMode" in js
          and ("const dock = isPureAppMode() && !drawerOpen()"
               + " && !modalDialogOpen();" in js)
          and "if (dock) mountPanel();" in js
          and "builder wizard" in js
          # the builder keeps viewMode === app but strips the DOM anchors -
          # the dock requires BOTH the mode string and the anchors
          and "comfyBody" in js
          and "layout.comfyBody" in js
          and "layout.rail" in js
          and "the builder strips these from the body" in js)
    check("A2 modal yield: dock yields while an official modal dialog is open",
          "function modalDialogOpen" in js
          and 'querySelectorAll(\'[data-state="open"]\')' in js
          and "modalOpen=" in js)
    check("A2 legacy overlay: minimap-style default collapsed window",
          "COLLAPSE_KEY" in js and 'saved !== "0") setCollapsed(true);' in js)
    check("A2 horizontal splitters: row boundaries draggable (2026-11-01)",
          "function makeHSplitter" in js and "function dragRowBoundary" in js
          and "ROWS_KEY" in js
          and "cdlapp-splitter-h" in css and "row-resize" in css
          and "--cdl-h1" in css and '"hsp1"' in js)
    check("A2 side-bar tab panels yield (help center was under-docked)",
          ".side-bar-panel .sidebar-content-container" in js)
    check("A2 builder step jump via the official setMode path",
          "workflowStoreProbe" in js and '"activeWorkflow"' in js
          and "modeField" in js and '"activeMode"' in js
          and "function setMode" in js and "function rawMode" in js
          and "enterBuilder();" in js and "rawMode() !== target" in js)
    check("A2 builder back-to-app floating button",
          "backBtnId" in js and '"cdl-builder-back"' in js
          and "function mountBuilderBack" in js
          and "function unmountBuilderBack" in js
          and "backValue" in js and '"app"' in js
          and "back_to_app" in js
          # deliberately NOT exitBuilder - its resetSelectedToWorkflow would
          # discard the selections the user just edited
          and "resetSelectedToWorkflow() would discard" in js)
    check("A2 workflow store re-probed until found (setMode dependency)",
          "!state.workflowStore) {" in js
          and "workflowStoreProbe" in js)
    check("A2 dock yields to the app-mode sidebar drawer",
          "appDrawer" in js and '".comfy-vue-side-bar-container"' in js
          and "graphCanvasContainer" in js and '".graph-canvas-container"' in js
          and "function drawerOpen" in js
          and "isPureAppMode() && !drawerOpen()" in js)
    check("A2 builder mode detected across all three wizard steps",
          "function isBuilderMode" in js
          and "const inBuilder = isBuilderMode();" in js
          and "builderPrefix" in js)
    check("A2 observer disconnect on stop",
          "clearInterval" in js and "function stopPolling" in js)
    check("A2 soft degradation warns instead of raising",
          "warnOnce" in js and "console.warn" in js)
    check("A2 builder probe via pinia (no hardcoded store symbol)",
          "enterBuilder" in js and "_s" in js
          and "builderStoreProbe" in js and "appModeStoreProbe" in js
          and "OFFICIAL_SURFACE.builder" in js)
    # 2026-10-30 real-session bug: setup() runs before the Vue app mounts, so
    # a one-shot pinia probe always missed and the panel never appeared.  The
    # poll tick must re-run probeStores() until the stores are found.
    check("A2 store probe retried from the poll tick (late-Vue-mount safe)",
          "function probeStores" in js
          and "!state.builderStore || !state.workflowStore" in js
          and "probeStores(); // first attempt" in js
          and "retried on the next tick" in js)
    # 2026-10-30 batch 5: native dock layout - instead of an overlapping
    # overlay, body.cdlapp-active CSS repositions the OFFICIAL .arrange-panel
    # (input widget panel) into the left column and our sections ride inside
    # it, so the rail, the central output area and the official inputs all
    # stay visible (user-reported cover-up bug).
    check("A2 dock: body.cdlapp-active toggled on mount/unmount",
          'document.body.classList.add("cdlapp-active")' in js
          and 'document.body.classList.remove("cdlapp-active")' in js)
    check("A2 dock: --cdl-dock-left measured from the side rail",
          "--cdl-dock-left" in js and "--cdl-dock-left" in css)
    check("A2 dock: extras appended into the official panel column",
          "function extrasColumn" in js
          and "col.appendChild(state.extras)" in js
          and "delayedRemove(extrasEl)" in js)
    check("A2 dock: unmount purge guards stale same-id surfaces",
          "function purgeStaleSurfaces" in js and "__cdlLeaveTimer" in js)
    check("A2 css: dock rule repositions the official input panel",
          "body.cdlapp-active .comfyui-body .arrange-panel" in css
          and "var(--cdl-dock-left" in css and "#cdl-app-extras" in css)
    # batch 6-7: full-bleed dock with PLANNED ZONES - the official panel
    # covers the whole remaining width (no dead empty center); inside it the
    # official params sit in a narrow left column and the CDL dashboard is a
    # named-area grid (previews hero / tall outputs column / progress /
    # errors strip), each tile scrolling inside its own box.
    check("A2 css: full-bleed dock (no empty center)",
          re.search(r"body\.cdlapp-active \.comfyui-body \.arrange-panel\s*\{"
                    r"[^}]*right:\s*0", css, re.S) is not None)
    check("A2 css: zoned layout (left params, right dashboard)",
          "var(--cdl-params-w, 340px)" in css
          and "grid-template-areas" in css
          and '"prev sp  out"' in css and '"prog sp  out"' in css
          and '"err  err err"' in css
          and "nth-of-type(1)" in css and "nth-of-type(2)" in css
          and "max-width: 880px" not in css)
    # batch 9: draggable zone splitters - 9px grid gutter tracks with handles;
    # sizes persist in localStorage (px for params col, % for previews col).
    check("A2 splitters: draggable zone boundaries with persistence",
          "function mountSplitters" in js and "function makeSplitter" in js
          and "cdlapp-splitter" in css and "col-resize" in css
          and "ComfyDL.AppPanel.ParamsW" in js
          and "ComfyDL.AppPanel.PrevW" in js
          and "mountSplitters();" in js)
    # batch 10: buttons wear the OFFICIAL tailwind recipe (declared in
    # OFFICIAL_SURFACE.buttons, attached by btnCls()); our own :where()-wrapped
    # CSS is only a zero-specificity fallback and can never override it.
    check("A2 buttons: official tailwind recipe + zero-specificity fallback",
          "OFFICIAL_SURFACE.buttons.secondary" in js
          and "hover:bg-secondary-background-hover" in js
          and "function btnCls" in js
          and ":where(.cdlapp-btn)" in css
          and ":where(.cdlapp-btn:hover)" in css)
    # batch 11: current-run scoping + bare-node-id executing + vector icons.
    check("A2 run scoping: only the active run's outputs/files are shown",
          "function beginRunIfNew" in js and "function endRun" in js
          and "function currentRunArtifacts" in js
          and "a.mtime * 1000 >= state.runStartedAt" in js
          and "state.outputs.length = 0" in js)
    check("A2 executing: bare node-id string handled (live-verified shape)",
          "BARE" in js and "node id string" in js
          and "const nodeId = typeof d === \"object\"" in js)
    check("A2 icons: inlined lucide vector paths (no emoji glyphs)",
          "function iconEl" in js
          and "download:" in js and "copy:" in js
          and 'stroke-width="2"' in js
          and "⬇" not in js and "⧉" not in js
          and ".cdlapp-icon svg" in css)
    m = re.search(r"\.cdlapp-item-name\s*\{[^}]*\}", css, re.S)
    check("A2 fonts: output rows use the official UI font (no fallback mono)",
          m is not None and "ui-monospace" not in m.group(0)
          and "font-family: inherit" in m.group(0))
    # batch 12: progressive checkmarks + status dot + graph-locate buttons.
    check("A2 progress: next-node-start completes the previous nodes",
          "r.state === \"running\") r.state = \"done\"" in js
          and "only fires for nodes with outputs" in js)
    check("A2 status dot: gray idle / teal running (class hook)",
          '"cdlapp-running"' in js and "cdlapp-running .cdlapp-title::before" in css
          and "padding-left: 4px" in css)
    check("A2 locate: jump to graph and select input/output nodes",
          "function locateOnGraph" in js
          and 'split(":")' in js
          and "canvas.selectNode(n, true)" in js
          and "canvas.centerOnNode(nodes[0])" in js
          and "log-in" in js and "log-out" in js
          and "locate_inputs" in js
          and "waitFor(() => !isAppMode()" in js)
    check("A2 previews: thumbnail grid section + official /view urls",
          "cdlapp-sec-previews" in js and "cdlapp-preview-grid" in js
          and "/view?type=output&subfolder=" in js
          and "PREVIEW_LIMIT" in js and "cdlapp-preview-grid" in css)
    check("A2 css: legacy overlay starts right of the side rail",
          re.search(r"#cdl-app-panel\s*\{[^}]*left:\s*var\(--cdl-dock-left",
                    css, re.S) is not None)
    # 2026-10-30 real-session bug: the workspace-store `setting` field has no
    # isReady - the old gate made every session time out (30s blank wait).
    check("A2 gate reads the pinia settings store (not the workspace field)",
          '_s.get("setting")' in js and "isReady === true" in js)
    check("A2 artifacts download uses the namespaced route",
          "/comfydl/artifacts/download" in js)
    check("A2 progress consumed via progress_state handler",
          re.search(r"\[E\.progressState\]\s*:\s*onProgressState", js) is not None)
    check("A2 zh dictionary present",
          "panel_title" in js and "节点图" in js and "复制" in js)

    # --- A3: pip frontend anchor pins (ADR-8) --------------------------------
    chunks: dict[str, str] = {}
    for p in pip_ui.rglob("*.js"):
        if p.stat().st_size <= 6_000_000:
            try:
                chunks[p.name] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    haystack = "\n".join(chunks.values())

    def pin(name, needle, guidance=""):
        ok = needle in haystack
        check(f"A3 pin: {name}", ok,
              "" if ok else f"MISSING {needle!r} - official surface drifted; "
              f"re-verify OFFICIAL_SURFACE in cdlapp.js. {guidance}")

    # Live-verified 2026-10-30 (agent-browser store enumeration): the pinia
    # store id is "appMode"; it carries viewMode ('graph'|'app'|'builder:*')
    # and enterBuilder()/exitBuilder().  There is NO isAppMode flag, NO
    # setMode() and the data-testid fallback strings never render as DOM
    # attributes - the earlier subagent report was wrong on all three.
    pin("appMode toolbar locale key", "linearMode.appModeToolbar",
        "Apps toolbar locale namespace - unique to the appMode area.")
    pin("appMode store field viewMode", "viewMode:",
        "String state field on the appMode store; mode flips go through "
        "Comfy.ToggleLinear (verified live).")
    pin("builder entry enterBuilder", "enterBuilder",
        "Best-effort builder entry; no args, steps cannot be requested.")
    pin("builder exit exitBuilder", "exitBuilder",
        "Companion exit action on the same store.")
    pin("view-mode-toggle testid", "view-mode-toggle",
        "Official graph/app switcher (used for manual testing only).")
    pin("builder step builder:inputs", "builder:inputs")
    pin("builder step builder:outputs", "builder:outputs")
    # batch 5 dock layout anchors: the CSS hook class must exist in the
    # official bundle or the dock rule never fires (panel falls back to the
    # legacy overlay silently).
    pin("official input panel arrange-panel", "arrange-panel")
    pin("official side rail side-tool-bar-container",
        "side-tool-bar-container")
    # batch 10: the button recipe classes must exist in the official bundle
    # (our buttons borrow them verbatim - see OFFICIAL_SURFACE.buttons).
    pin("official button tailwind recipe", "hover:bg-secondary-background-hover")
    pin("official button font recipe", "font-inter")
    pin("command Comfy.ToggleLinear", "Comfy.ToggleLinear",
        "Update OFFICIAL_SURFACE.commands.toGraph if renamed.")
    pin("workspace store registerSidebarTab", "registerSidebarTab",
        "extensionManager surface; verify command.execute exposure too.")
    check("A3 pin: workspace store command exposure",
          re.search(r"command:\{", haystack) is not None
          and "lastNodeErrors" in haystack,
          "extensionManager.command.execute not found - Node-graph button "
          "needs a new execution path (re-probe the workspace store literal)")
    for ev in ("progress_state", "executed", "execution_error",
               "execution_success", "execution_interrupted",
               "execution_cached"):
        pin(f"api event {ev}", ev)

    # frontend version pin: OFFICIAL_SURFACE.frontendVersion == installed pkg
    try:
        installed = importlib.metadata.version("comfyui_frontend_package")
        m = re.search(r'frontendVersion:\s*"([^"]+)"', js)
        declared = m.group(1) if m else None
        check("A3 frontend version declared matches installed",
              declared == installed,
              f"declared={declared} installed={installed} - upgrade "
              f"OFFICIAL_SURFACE.frontendVersion and re-run the pins")
    except importlib.metadata.PackageNotFoundError:
        check("A3 frontend version declared matches installed", False,
              "comfyui_frontend_package not found in penv")

    # --- A4: zh bake constants ------------------------------------------------
    check("A4 zh setting key present",
          "ComfyDL_AppPanel_Enabled" in I18N_SETTINGS_KEYS
          and "App 面板" in I18N_SETTINGS_KEYS["ComfyDL_AppPanel_Enabled"]["name"])
    check("A4 zh category present",
          I18N_SETTINGS_CATEGORIES_ZH.get("App") == "App")

    # --- A5: batch 16 UXD round (progress density, contrast, motion) ---------
    check("A5 progress pre-filled with the full node list at run start",
          "function prefillNodes" in js
          and "computeExecutionOrder" in js
          and "prefillNodes(state.activePrompt);" in js
          and "n.mode === 2 || n.mode === 4" in js)
    check("A5 per-node percent next to the bar + overall percent on the label",
          '"cdlapp-prog-eta"' in js
          and '{ done: done, total: total, pct: pct }' in js
          and 'bar.style.width = pct + "%";' in js
          and 'r.state === "idle"' in js)
    check("A5 sliding-window speed / ETA estimate (cheap, per-event)",
          "function noteProgressSample" in js
          and "function rateOf" in js and "function etaText" in js
          and "row.lastValue" in js and "row.maxValue" in js
          and "s.length > 6" in js)
    check("A5 progress_state payload parsed as the official nodes dict",
          # comfy_execution/progress.py sends {prompt_id, nodes: {...}};
          # the flat single-node shape is only a fallback.  NodeState enum:
          # pending | running | finished | error.
          'd.nodes && typeof d.nodes === "object"' in js
          and "n.display_node_id || n.node_id || nid" in js
          and 'n.state === "finished"' in js
          and 'n.state === "error"' in js)

    # --- A6: batch 18 vector live previews -----------------------------------
    check("A6 cdl_preview_data event wired end to end",
          "cdl_preview_data: onPreviewData" in js
          and "function onPreviewData" in js
          and "state.charts.set" in js
          and "state.charts.clear();" in js)
    check("A6 vector SVG loss chart (no external chart lib)",
          "function buildLossChart" in js
          and 'createElementNS("http://www.w3.org/2000/svg"' in js
          and '"cdlapp-chart-smooth"' in js
          and '"cdlapp-chart-raw"' in js
          and '"cdlapp-chart-best"' in js
          and 'viewBox", "0 0 240 110"' in js
          and 'cursor: crosshair' in css)
    check("A6 chart cards reused incrementally (no animation replay)",
          "function renderCharts" in js
          and "dataset.chartKey" in js
          and "existing.get(key)" in js)
    check("A6 text snapshot card rendered from structured lines",
          "function buildTextCard" in js
          and '"cdlapp-chart-card cdlapp-chart-text"' in js)
    check("A6 charts update in place with a soft pulse (no hard rebuild)",
          "card._update" in js
          and "function pulseChart" in js
          and "cdlapp-chart-pulse" in css
          # a fresh-data glow is feedback, not an entrance replay - it must
          # respect the global motion switch
          and "body.cdlapp-motion-off .cdlapp-chart-pulse" in css)
    check("A6 user-resizable chart blocks with undistorted text (batch 21)",
          # native resize handle, flex column (blocks stack, never overlap)
          "resize: both" in css and "overflow: hidden" in css
          and "max-width: 100%" in css and "min-height: 150px" in css
          # strokes stay 1px under any stretch; text lives in HTML overlay
          and "non-scaling-stroke" in js
          and "cdlapp-chart-overlay" in js and "cdlapp-chart-overlay" in css
          and "cdlapp-chart-ylab" in css and "cdlapp-chart-dot" in css)

    # --- A7: batch 20 (curve motion, fonts, run-boundary glow) ---------------
    check("A7 curve movement: eased points interpolation on persistent lines",
          "function animatePoints" in js
          and "requestAnimationFrame" in js
          and "cancelAnimationFrame(pl._raf)" in js
          and "const gGrid = svgEl" in js
          and "const rawPl = svgEl" in js
          and "const smoothPl = svgEl" in js)
    check("A7 run-boundary panel glow at got-prompt / finished moments",
          "executionStart: \"execution_start\"" in js
          and "cdl_preview_data: onPreviewData" in js
          and "flashPanel(\"cdlapp-flash-start\")" in js
          and "flashPanel(\"cdlapp-flash-done\")" in js
          and "cdlapp-flash-done-k" in css
          and "body.cdlapp-motion-off .cdlapp-flash-start" in css)
    check("A7 prose fonts for generated text + app-mode text widgets",
          # generated-text card and the docked text widgets leave monospace
          "function appendTyping" in js
          and "font-family: inherit" in css
          and "arrange-panel textarea" in css)
    check("A7 streaming text reveal + stronger curve easing (batch 22)",
          "cdlapp-typein" in js and "cdlapp-typein" in css
          and "next.startsWith(lastFull)" in js
          and "animationDelay" in js
          and "spans.length > 240" in js
          and "DUR = 550" in js and "easeOutBack" in js)
    check("A8 previews pane scrolls instead of clipping (batch 23)",
          ".cdlapp-sec-previews" in css and "overflow-y: auto" in css
          and "scrollbar-width: thin" in css
          and "min-height: 160px" in css)
    # 2026-10-31 regression: the batch 23 rule sat BEFORE the
    # .cdlapp-section { overflow: hidden } block; equal specificity, so the
    # later rule won and the previews pane silently clipped again.  The
    # scroll rule must come AFTER that block (source order decides).
    _sec_hidden = css.find("overflow: hidden; /* tiles scroll inside")
    _prev_scroll = css.find("#cdl-app-extras .cdlapp-sec-previews,")
    _runs_scroll = css.find("#cdl-app-panel .cdlapp-sec-runs {")
    check("A8 scroll rule ordered AFTER .cdlapp-section overflow hidden",
          _sec_hidden != -1 and _prev_scroll != -1 and _prev_scroll > _sec_hidden
          and _runs_scroll > _sec_hidden)
    check("A8 runs pane scrolls like previews + expand toggle (2026-11-01)",
          ".cdlapp-sec-runs" in css
          and "cdlapp-runs-item-active" in css
          and "clearRunsChart" in js and "runs_hint" in js
          and "state.runsChartName === name" in js)
    check("A8 Stop (interrupt) button wired to the official api method",
          'interruptMethod: "interrupt"' in js
          and 'square: \'<rect width="18" height="18" x="3" y="3" rx="2"/>\'' in js
          and "function buildProgressSection" in js
          and "buildProgressSection()" in js
          and 'stopBtn.style.display = state.runActive ? "" : "none";' in js
          and '"stop_tooltip"' in js)
    check("A5 dim text is contrast-derived (theme-safe, not a fixed color)",
          "--cdl-dim" in css and "color-mix(in srgb, var(--fg-color" in css
          and css.count("var(--cdl-dim") >= 4
          # the near-invisible --comfy-input-bg must be gone from text colors
          and "color: var(--comfy-input-bg" not in css)
    check("A5 motion: full animation set + global setting + reduced-motion",
          SETTING_MOTION_ID in js
          and "cdlapp-motion-off" in js
          and "function applyMotion" in js
          and "cdlapp-leaving" in js
          and "cdlapp-pop" in js
          and "prefers-reduced-motion" in css)

    # --- B: artifacts routes integration (aiohttp TestServer) ----------------
    import asyncio
    import os as _os
    import tempfile as _tmp

    try:
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
    except ImportError:
        check("B0 aiohttp available", False, "import failed")
        return report()
    check("B0 aiohttp available", True)

    from app import cdlapp_routes

    async def _run_routes() -> None:
        outdir = Path(_tmp.mkdtemp(prefix="cdl_artifacts_"))
        (outdir / "sub").mkdir()
        (outdir / "results.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (outdir / "sub" / "model.pt").write_bytes(b"\x80\x02fake")
        (outdir / "sub" / "loss.png").write_bytes(b"\x89PNG fake")
        cdlapp_routes.set_output_root(outdir)
        app = web.Application()
        app.add_routes(cdlapp_routes.routes)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            r = await client.get("/comfydl/artifacts?limit=10")
            data = await r.json()
            arts = data.get("artifacts", [])
            check("B1 listing 200 + 3 artifacts",
                  r.status == 200 and len(arts) == 3, str(data)[:120])
            check("B2 newest-first ordering",
                  all(arts[i]["mtime"] >= arts[i + 1]["mtime"]
                      for i in range(len(arts) - 1)))
            csv_e = next((a for a in arts if a["name"] == "results.csv"), None)
            check("B3 media_type csv",
                  csv_e is not None and csv_e["media_type"] == "text/csv")
            check("B4 path is output-relative posix",
                  csv_e is not None and csv_e["path"] == "results.csv")
            r = await client.get("/comfydl/artifacts/download?name=results.csv")
            body = await r.read()
            check("B5 download 200 + content",
                  r.status == 200 and b"1,2" in body)
            check("B6 Content-Disposition attachment",
                  r.headers.get("Content-Disposition", "").startswith("attachment"))
            r = await client.get(
                "/comfydl/artifacts/download?name=" + _os.path.pardir + "/x")
            check("B7 dotdot escape 403", r.status == 403)
            r = await client.get("/comfydl/artifacts/download?name="
                                 + str(outdir / "results.csv"))
            check("B8 absolute path 403", r.status == 403)
            r = await client.get("/comfydl/artifacts/download?name=missing.pt")
            check("B9 missing 404", r.status == 404)
            link = outdir / "evil.lnk"
            try:
                _os.symlink(str(outdir / "results.csv"), str(link))
                r = await client.get("/comfydl/artifacts/download?name=evil.lnk")
                check("B10 symlink 403", r.status == 403)
            except OSError:
                check("B10 symlink 403", True, "symlinks unsupported on this FS")
            check("B11 payload mirrors official asset schema keys",
                  bool(arts) and {"name", "media_type"} <= set(arts[0].keys()))
        finally:
            await client.close()
            cdlapp_routes.set_output_root(None)
            shutil.rmtree(outdir, ignore_errors=True)

    asyncio.run(_run_routes())

    return report()


if __name__ == "__main__":
    sys.exit(main())
