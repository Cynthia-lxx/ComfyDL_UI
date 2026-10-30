"""ComfyDL_UI patches for the externally installed ComfyUI frontend package.

The frontend is a pip dependency (``comfyui-frontend-package``), so it lives
outside this repository and every hand edit to it is lost the next time the
package is upgraded. Everything ComfyDL_UI changes in the UI is applied from
here instead, at startup, right after the web root has been resolved and before
anything is served.

Each patch locates its target asset by a *content marker* (never by the hashed
asset file name) and rewrites that file in place:

  * default locale is English (``Comfy.Locale``)
  * Nodes 2.0 (``Comfy.VueNodes.Enabled``) is enabled by default
  * the Settings -> About panel gets a ComfyDL_UI section plus a separate
    upstream section that keeps the original ComfyUI links
  * the ``TENSOR`` slot colour (lemon green) in the six theme palettes
  * the startup ("Load Default Workflow") graph opens the Tabular Regression:
    Production example instead of the upstream SD text-to-image demo
  * the ``<script>`` loader for the ComfyDL Profiling panel (the JS/CSS
    themselves live in this repository under ``app/profiling_assets/`` and
    are served by the ``/comfydl/profiling`` route in ``server.py``; only
    the loader tag is injected into the frontend's ``index.html``)

Patches are idempotent: one whose marker is already present does nothing. They
are also non-fatal: if a future frontend renames or rewrites a target, the patch
logs a warning and the server keeps booting with the upstream behaviour.
"""

import logging
import re
from pathlib import Path

TAG = "[ComfyDL_UI] frontend"

# Repositories referenced by the About section.
PROJECT_REPO = "https://github.com/Cynthia-lxx/ComfyDL_UI"
UPSTREAM_REPO = "https://github.com/Comfy-Org/ComfyUI"

TENSOR_SLOT_COLOUR = "#C6FF00"

# The "materialised nn.Module" type, unified between the ComfyDL inference
# nodes (formerly cdlModel) and the Network & Layers training system.
NN_MODEL_SLOT_COLOUR = "#FF8C42"
NN_MODEL_MARK = "nn_model:`" + NN_MODEL_SLOT_COLOUR + "`"

# The universal "dataset" type used by the ComfyDL data-IO and regression nodes.
# Teal, distinct from the orange nn_model and the lemon-green TENSOR slots.
DATASET_SLOT_COLOUR = "#1ABC9C"
DATASET_MARK = "DATASET:`" + DATASET_SLOT_COLOUR + "`"

# The Profiling panel loader: one <script type="module"> tag before </body>.
PROFILING_LOADER_TAG = (
    '<!--ComfyDL_UI:profiling-loader-->'
    '<script type="module" src="/comfydl/profiling/profiler.js"></script>'
)
PROFILING_LOADER_MARK = "<!--ComfyDL_UI:profiling-loader-->"

# The Crash site panel loader: same index.html injection pattern.
CRASHSITE_LOADER_TAG = (
    '<!--ComfyDL_UI:crashsite-loader-->'
    '<script type="module" src="/comfydl/crashsite/crashsite.js"></script>'
)
CRASHSITE_LOADER_MARK = "<!--ComfyDL_UI:crashsite-loader-->"

# The Motion module (staged entrance + ripple): a loader script before </body>
# plus the entrance pre-hide CSS in the <head>. The CSS must ship in the head
# (not be injected by motion.js at runtime) so it is already in the CSSOM when
# the first Vue element renders - a runtime injection would race the mount and
# flicker. The rules are gated on html classes only motion.js (or the 9s
# failsafe timer baked right next to the CSS) can set, so a failed module load
# can never leave the UI invisible.
MOTION_LOADER_TAG = (
    '<!--ComfyDL_UI:motion-loader-->'
    '<script type="module" src="/comfydl/motion/motion.js"></script>'
)
MOTION_LOADER_MARK = "<!--ComfyDL_UI:motion-loader-->"
MOTION_CSS_TAG = (
    '<!--ComfyDL_UI:motion-css-->'
    '<style id="cdl-motion-prehide">'
    ':root{--cdl-ease:cubic-bezier(.16,1,.3,1)}'
    '#graph-canvas-container canvas{transition:opacity 1.6s var(--cdl-ease)}'
    '@media not (prefers-reduced-motion: reduce){'
    'html:not(.cdl-motion-off):not(.cdl-entrance-done) .side-tool-bar-container .sidebar-item-group > *:not(.cdl-entered){opacity:0}'
    'html:not(.cdl-motion-off):not(.cdl-entrance-done) .side-tool-bar-container .mt-auto > *:not(.cdl-entered){opacity:0}'
    'html:not(.cdl-motion-off):not(.cdl-entrance-done) #comfyui-body-top:not(.cdl-entered),'
    'html:not(.cdl-motion-off):not(.cdl-entrance-done) #comfyui-body-bottom:not(.cdl-entered),'
    'html:not(.cdl-motion-off):not(.cdl-entrance-done) #comfyui-body-right:not(.cdl-entered){opacity:0}'
    'html:not(.cdl-motion-off):not(.cdl-nodes-in) [data-node-id]{opacity:0}'
    'html:not(.cdl-motion-off):not(.cdl-canvas-in) #graph-canvas-container canvas{opacity:0}'
    '}'
    '</style>'
    '<script>setTimeout(function(){var d=document.documentElement;'
    "d.classList.add('cdl-canvas-in');d.classList.add('cdl-nodes-in');"
    "d.classList.add('cdl-entrance-done');},9000);</script>"
)
MOTION_CSS_MARK = "<!--ComfyDL_UI:motion-css-->"

# Branded splash: replaces the official #splash-loader contents with the
# homepage-fork design (animated logo GIF + progress bar + live boot log).
# The official element stays in place but hidden (its id is kept so the
# official App watcher's getElementById finds a harmless hidden stub), and the
# real splash lives under #cdl-splash - fully owned by motion.js, which fades
# it out on the spinner falling edge. That removes the hard .remove() flicker
# and lets the entrance choreography start the moment the splash begins
# dissolving. Theme colors come from the official inline script via
# var(--bg-color)/var(--fg-color), so no extra theming is needed.
SPLASH_BRAND_MARK = "<!--ComfyDL_UI:splash-brand-->"

# Brand logo (2026-10-28): the user's DL_trans.png becomes the favicon and the
# top-left menu logo. The official logo is an inline SVG component
# (ComfyLogo, currentColor fill) inside .comfy-menu-button-wrapper .grid - a
# raster PNG cannot take its place through a mask/fill, so the same trick as
# the homepage fork's logo-fx applies: hide the svg and paint the grid wrapper.
BRAND_LOGO_MARK = "<!--ComfyDL_UI:brand-logo-->"
# Favicon uses the trimmed square derivative (DL_trans.png has ~50% transparent
# margins, which renders tiny and skinny at 16px); the menu logo uses it too
# for a fuller look at 42px.
FAVICON_TARGET_LINK = ('<link rel="icon" type="image/png" '
                       'href="/comfydl/motion/DL_trans_favicon.png"/>')
FAVICON_URL = "/comfydl/motion/DL_trans_favicon.png"
BRAND_LOGO_TAG = (
    BRAND_LOGO_MARK
    + '<style id="cdl-brand-logo-style">'
    + '.comfyui-logo{display:none!important}'
    + '.comfy-menu-button-wrapper .grid{width:42px;height:42px;'
    + f"background:url('{FAVICON_URL}') center/contain no-repeat}}"
    + '</style>'
)
# The official useProgressFavicon() swaps the favicon at runtime for its
# execution-progress frames (frame_9.png when idle), which silently overwrote
# the <link> baked into index.html. Redirect the idle frame to the brand PNG;
# the 0..8 progress frames keep the official artwork.
FAVICON_IDLE_OLD = "`/assets/images/favicon_progress_16x16/frame_9.png`"
# Earlier brand-URL generations accepted as upgrade sources.
FAVICON_IDLE_PREV = ("`/assets/images/favicon_progress_16x16/frame_9.png`",
                     "`/comfydl/motion/DL_trans.png`")
FAVICON_IDLE_NEW = "`/comfydl/motion/DL_trans_favicon.png`"

# 2026-10-28 regression: 0.3.14's startup path never calls loadDefaultWorkflow.
# initializeWorkflow -> resolveStartupOutcome serves *veteran* users
# (Comfy.TutorialCompleted) a bare `app.loadGraphData()` - the built-in SD3
# demo graph - which is why the old default-workflow patch stopped covering
# startup. Redirect that bare call through the same /templates overlay
# (fallback kept), with a loud failure on signature drift.
STARTUP_WORKFLOW_MARK = "/*ComfyDL_UI:startup-wf*/"
STARTUP_WORKFLOW_PATTERN = re.compile(
    r"get\(`Comfy\.TutorialCompleted`\)\?\(await ([\w$]+)\.loadGraphData\(\),`restored`\)"
)
SPLASH_BRAND_CSS = (
    '<style id="cdl-splash-style">'
    '#cdl-splash{position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;'
    'align-items:center;justify-content:center;contain:strict;'
    'background:var(--bg-color,#111318);color:var(--fg-color,#e6e6ec);'
    'transition:opacity .6s var(--cdl-ease),transform .6s var(--cdl-ease)}'
    '.cdl-splash{display:flex;flex-direction:column;align-items:center;gap:22px;width:min(560px,86vw)}'
    '.cdl-brand{display:flex;align-items:center;gap:16px}'
    '.cdl-logo{width:56px;height:56px;object-fit:contain;filter:drop-shadow(0 6px 18px rgba(26,188,156,.35))}'
    '.cdl-brand h1{margin:0;font-size:30px;font-weight:750;letter-spacing:.4px;line-height:1.1;color:var(--fg-color)}'
    '.cdl-tag{margin:4px 0 0;font-size:13.5px;color:color-mix(in srgb,var(--fg-color) 60%,transparent)}'
    '.cdl-progress{width:100%;display:flex;flex-direction:column;gap:8px}'
    '.cdl-track{position:relative;height:14px;border-radius:999px;'
    'background:color-mix(in srgb,var(--fg-color) 8%,transparent);'
    'border:1px solid color-mix(in srgb,var(--fg-color) 16%,transparent);overflow:hidden;'
    'box-shadow:inset 0 2px 6px rgba(0,0,0,.5)}'
    '.cdl-fill{height:100%;width:0%;border-radius:999px;'
    'background:linear-gradient(90deg,#1ABC9C,#4cc3a8 55%,#FF8C42);'
    'transition:width .5s var(--cdl-ease);position:relative;z-index:1}'
    '.cdl-stripes{position:absolute;inset:0;z-index:2;'
    'background:repeating-linear-gradient(-55deg,rgba(255,255,255,.10) 0 10px,transparent 10px 20px);'
    'background-size:200% 100%;animation:cdl-stripes 1.1s linear infinite;mix-blend-mode:overlay;pointer-events:none}'
    '@keyframes cdl-stripes{to{background-position:-56px 0}}'
    '.cdl-meta{display:flex;justify-content:space-between;font-size:12.5px;'
    'color:color-mix(in srgb,var(--fg-color) 60%,transparent);font-variant-numeric:tabular-nums}'
    '.cdl-pct{color:var(--fg-color);font-weight:650;font-variant-numeric:tabular-nums}'
    '.cdl-log{width:100%;height:172px;overflow-y:auto;padding:10px 12px;'
    'display:flex;flex-direction:column;gap:3px;border-radius:10px;'
    'background:color-mix(in srgb,var(--fg-color) 5%,transparent);'
    'border:1px solid color-mix(in srgb,var(--fg-color) 14%,transparent);'
    'box-shadow:inset 0 2px 8px rgba(0,0,0,.35)}'
    '.cdl-line{display:flex;align-items:baseline;gap:8px;font-size:12.5px;'
    'color:color-mix(in srgb,var(--fg-color) 80%,transparent);'
    'animation:cdl-line-in .45s var(--cdl-ease) both;font-variant-numeric:tabular-nums}'
    '.cdl-t{color:color-mix(in srgb,var(--fg-color) 42%,transparent);flex:none}'
    '.cdl-m{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}'
    '.cdl-s{flex:none;font-size:10.5px;font-weight:650;letter-spacing:.6px;padding:1px 8px;border-radius:999px}'
    '.cdl-s.ok{color:#1ABC9C;border:1px solid rgba(26,188,156,.45)}'
    '.cdl-s.run{color:#FF8C42;border:1px solid rgba(255,140,66,.45);animation:cdl-pulse 1s ease-in-out infinite}'
    '.cdl-line.run .cdl-m{color:var(--fg-color)}'
    '@keyframes cdl-line-in{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}'
    '@keyframes cdl-pulse{50%{opacity:.45}}'
    '.cdl-foot{font-size:11.5px;color:color-mix(in srgb,var(--fg-color) 42%,transparent);letter-spacing:.3px}'
    '@media (prefers-reduced-motion:reduce){.cdl-stripes,.cdl-line,.cdl-s.run{animation:none}}'
    '</style>'
)
SPLASH_BRAND_DOM = (
    '<div id="splash-loader" role="status" aria-label="Loading ComfyDL_UI" style="display:none"></div>'
    '<div id="cdl-splash" role="status" aria-label="Loading ComfyDL_UI">' + SPLASH_BRAND_CSS +
    '<div class="cdl-splash">'
    '<div class="cdl-brand"><img class="cdl-logo" src="/comfydl/motion/LoadFX1.gif" alt="">'
    '<div><h1>ComfyDL_UI</h1><p class="cdl-tag">Deep Learning is just a few clicks away!</p></div></div>'
    '<div class="cdl-progress"><div class="cdl-track"><div class="cdl-fill" id="cdlFill"></div>'
    '<div class="cdl-stripes"></div></div>'
    '<div class="cdl-meta"><span id="cdlPhase">Starting</span>'
    '<span class="cdl-pct" id="cdlPct">0%</span></div></div>'
    '<div class="cdl-log" id="cdlLog" aria-live="polite"></div>'
    '<div class="cdl-foot">ComfyDL_UI &middot; modular deep learning toolbox</div>'
    '</div></div>'
)

# Marker comments written into the patched assets: they document who changed the
# file and double as the idempotency check.
LOCALE_MARK = "/*ComfyDL_UI:locale*/"
NODES2_MARK = "/*ComfyDL_UI:nodes2*/"
TENSOR_MARK = "TENSOR:`" + TENSOR_SLOT_COLOUR + "`"

# The About injection is delimited on both sides so a later revision of the
# section can be applied by removing the previous one first.
ABOUT_BEGIN = "/*ComfyDL_UI:about:begin*/"
ABOUT_END = "/*ComfyDL_UI:about:end*/"

# The startup ("Load Default Workflow") graph: instead of the upstream
# SD text-to-image demo, open our *Classification from Scratch* example,
# fetched live from the template overlay channel (/templates/<name>.json) so
# the served workflow always matches the shipped template.
#
# Template choice (2026-10-30): classification_from_scratch is fully
# self-contained (CdlClassificationDataGen produces the data internally -
# no external CSV file), trains in seconds and renders a confusion-matrix
# heatmap, which makes it the best-looking out-of-the-box startup demo.
# Patches bake the URL in, so both workflow patches carry a generational
# upgrade that rewrites a previously baked URL to the current constant.
DEFAULT_WORKFLOW_TEMPLATE = "classification_from_scratch"
DEFAULT_WORKFLOW_MARK = "/*ComfyDL_UI:default-workflow*/"

# --- zh locale baking (2026-10-28) -------------------------------------------
#
# The 0.3.14 frontend NEVER calls the official /i18n endpoint (the custom-node
# translation funnel), so a locales/zh/*.json file alone is dead data.  This
# patch merges the repository's zh translations straight into the frontend's
# own lazy-loaded zh chunks, located VIA CONTENT: the i18n loader chunk maps
# "./zh/<name>.json" to hashed chunk files whose names differ per build, so
# nothing here may rely on a file name.
#
#   * nodeDefs.json -> Object.assign(<defaultMap>, entries)  node titles
#   * settings.json -> new vars + export keys                Settings strings
#   * main.json     -> deep-merge settingsCategories         group names
#
# Data source: locales/zh/nodeDefs.json (regenerate with gen_locales.py;
# 203 entries = 128 ComfyDL doc titles/descriptions + 75 core Network & Layers
# names from cdl_smoke_tests/core_nodes_zh.py).
I18N_NODEDEFS_MARK = "/*ComfyDL_UI:i18n-nodedefs*/"
I18N_MAIN_MARK = "/*ComfyDL_UI:i18n-main*/"
I18N_SETTINGS_MARK = "/*ComfyDL_UI:i18n-settings*/"

#: The ComfyDL settings whose names are translated into the zh settings chunk.
#: Keys follow the official convention: the setting id with "." -> "_".
I18N_SETTINGS_KEYS = {
    "ComfyDL_CrashSite_Enabled": {
        "name": "启用 ComfyDL 事故现场（总开关）",
        "tooltip": "关闭后事故现场从执行路径上完全摘除：不捕获快照、不写训练断点、恢复不可用。",
    },
    "ComfyDL_Profiling_Enabled": {
        "name": "启用 ComfyDL 性能分析（总开关）",
        "tooltip": "关闭后性能分析栈脱离运行路径：无看门狗采样、无运行计量、无前端钩子。",
    },
    "ComfyDL_Profiling_DangerousProbe": {
        "name": "ComfyDL 性能分析危险探针（执行真实节点代码）",
        "tooltip": "实测计量需要真实执行节点——仅在理解风险时开启。",
    },
    "ComfyDL_Profiling_LogLevel": {
        "name": "ComfyDL 性能分析日志级别",
        "tooltip": "后端日志详细程度：off / low / medium / high（默认 off）。",
    },
    "ComfyDL_Motion_Enabled": {
        "name": "界面动效（入场编排 + 点击涟漪）",
        "tooltip": "分阶段入场动画与点击涟漪反馈。改动即时生效；入场动画每次刷新只播放一次。",
    },
}

#: settingsCategories deep-merged into the zh main chunk (Settings group names).
I18N_SETTINGS_CATEGORIES_ZH = {
    "ComfyDL": "ComfyDL 工具包",
    "CrashSite": "事故现场",
    "Profiling": "性能分析",
    "Motion": "动效",
}


def apply_frontend_patches(web_root: str) -> None:
    """Apply every ComfyDL_UI patch to the frontend under ``web_root``.

    ``web_root`` is the already resolved frontend directory, so an alternative
    frontend selected with ``--front-end-root`` is patched just like the one
    shipped in the pip package.
    """
    assets = Path(web_root) / "assets"
    if not assets.is_dir():
        logging.warning(f"{TAG} patches skipped: no assets directory in {web_root}")
        return

    applied = []
    for name, patch in (
        ("zh locale baking", _patch_zh_locales),
        ("default locale", _patch_default_locale),
        ("Nodes 2.0 default", _patch_nodes2_default),
        ("about panel", _patch_about_panel),
        ("TENSOR slot colour", _patch_tensor_slot_colour),
        ("nn_model slot colour", _patch_nn_model_slot_colour),
        ("DATASET slot colour", _patch_dataset_slot_colour),
        ("default workflow", _patch_default_workflow),
        ("startup workflow", _patch_startup_workflow),
        ("help menu group", _patch_help_menu),
    ):
        try:
            if patch(assets):
                applied.append(name)
        except Exception as exc:
            logging.warning(f"{TAG} {name} not applied: {exc}")

    # The Profiling loader lives in index.html (the web root), not in assets/.
    try:
        if _patch_profiling_loader(Path(web_root)):
            applied.append("profiling loader")
    except Exception as exc:
        logging.warning(f"{TAG} profiling loader not applied: {exc}")

    # The Crash site loader likewise lives in index.html.
    try:
        if _patch_crashsite_loader(Path(web_root)):
            applied.append("crashsite loader")
    except Exception as exc:
        logging.warning(f"{TAG} crashsite loader not applied: {exc}")

    # The Motion module: pre-hide CSS in the head, loader script before </body>.
    try:
        motion_applied = False
        if _patch_motion_css(Path(web_root)):
            motion_applied = True
        if _patch_motion_loader(Path(web_root)):
            motion_applied = True
        if _patch_splash_brand(Path(web_root)):
            motion_applied = True
        if _patch_brand_logo(Path(web_root)):
            motion_applied = True
        if motion_applied:
            applied.append("motion loader")
    except Exception as exc:
        logging.warning(f"{TAG} motion loader not applied: {exc}")

    # One summary line per startup, so an operator can tell a fresh install (some
    # patches just applied) from a warm one (everything was already in place).
    if applied:
        logging.info(f"{TAG} applied {len(applied)} patch(es): {', '.join(applied)}")
    else:
        logging.info(f"{TAG} already up to date, nothing to apply")


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _locate(assets: Path, pattern: str, marker: str) -> Path:
    """Return the asset matching ``pattern`` that contains ``marker``."""
    for path in sorted(assets.glob(pattern)):
        if marker in _text(path):
            return path
    raise FileNotFoundError(f"no {pattern} containing {marker!r}")


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one occurrence of {old[:60]!r}, found {count}")
    return text.replace(old, new)


def _object_end(text: str, brace: int) -> int:
    """Return the index just past the ``}`` closing the object literal at ``brace``."""
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("unbalanced object literal")


def _upgrade_baked_template_url(text: str) -> tuple[str, bool]:
    """Rewrite a previously baked startup template URL to the current one.

    Both workflow patches are marker-idempotent, so changing
    ``DEFAULT_WORKFLOW_TEMPLATE`` would otherwise never reach a frontend that
    was already baked with the old name. The baked calls always look like
    ``fetch(`/templates/<name>.json`,{cache:`no-store`})`` - the official
    ``index_logo`` fetch goes through ``fileURL()`` and does not match this
    shape, so the pattern is unambiguous.
    """
    match = re.search(r"fetch\(`/templates/([\w-]+)\.json`", text)
    if match and match.group(1) != DEFAULT_WORKFLOW_TEMPLATE:
        replaced = (
            text[: match.start()]
            + f"fetch(`/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json`"
            + text[match.end() :]
        )
        return replaced, True
    return text, False


def _patch_default_workflow(assets: Path) -> bool:
    """Open the Classification from Scratch template on startup.

    Upstream's ``loadDefaultWorkflow`` loads a graph built in-bundle from the
    SD text-to-image demo. We redirect it to fetch our
    ``classification_from_scratch`` example through the template overlay
    channel (``/templates/<name>.json``) instead, so the served workflow is
    always the shipped one; any failure falls back to the upstream graph.
    A frontend already baked with a previous template name is upgraded in
    place (see ``_upgrade_baked_template_url``).

    Returns ``True`` when the asset was rewritten, ``False`` when the marker
    was already there and the URL was current.
    """
    path = _locate(assets, "settingStore-*.js", "loadDefaultWorkflow=async()=>{")
    text = _text(path)
    if DEFAULT_WORKFLOW_MARK in text:
        text, upgraded = _upgrade_baked_template_url(text)
        if upgraded:
            _write(path, text)
            logging.info(
                f"{TAG} default workflow template upgraded -> "
                f"/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json in {path.name}"
            )
            return True
        logging.debug(f"{TAG} default workflow already applied")
        return False

    match = re.search(
        r"loadDefaultWorkflow=async\(\)=>\{await "
        r"([A-Za-z_$][\w$]*)\.loadGraphData\(([A-Za-z_$][\w$]*)\)\}",
        text,
    )
    if match is None:
        raise ValueError("loadDefaultWorkflow call signature not found")

    service, stock_graph = match.group(1), match.group(2)
    patched = (
        f"loadDefaultWorkflow=async()=>{{"
        f"try{{const _cdlRes=await fetch(`/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json`,"
        f"{{cache:`no-store`}});"
        f"if(_cdlRes.ok){{await {service}.loadGraphData(await _cdlRes.json());return}}}}"
        f"catch(_cdlErr){{}}"
        f"await {service}.loadGraphData({stock_graph}){DEFAULT_WORKFLOW_MARK}}}"
    )
    text = _replace_once(text, match.group(0), patched)
    _write(path, text)
    logging.info(
        f"{TAG} default workflow -> /templates/{DEFAULT_WORKFLOW_TEMPLATE}.json "
        f"in {path.name}"
    )
    return True


# --- Help menu (2026-10-28) ---------------------------------------------------
#
# The frontend's Help dropdown is built in GraphView-*.js (HelpCenterMenuContent)
# as a literal item array.  Merged in from the homepage fork: the ComfyDL
# project links (docs / QQ group / Bilibili / repo) join the official menu as a
# VISIBLE GROUP of their own - every ComfyDL entry is labelled "ComfyDL · ..."
# and appended after the official items, so the two groups read separately
# without touching the official entries (feedback / help / docs / discord /
# github / manager / update all stay).
HELP_MENU_MARK = "/*ComfyDL_UI:help-menu*/"
HELP_MENU_INSERT_BEFORE = "];return e.push({key:`manager`"
HELP_MENU_QQ_URL = (
    "https://qun.qq.com/universal-share/share?ac=1&authKey=IbE90VoEPI2G0NeI5Gay90"
    "%2BPfMFD1o/uClv6wpNdhoEA3JqkMsn209yAaRzl7QRb&busi_data=eyJncm91cENvZGUiOiI5MjEyMTk3"
    "NDciLCJ0b2tlbiI6Ilg1V1NMWEFrZWpFTkVxcm5MN0MwWm1rWm1VTit3OVFrcklmVTA1OGRYQVA4NTFrQmIz"
    "cW9WZFlRVnpsTXVyK3giLCJ1aW4iOiIzOTI0NTIyNDIwIn0=&data=8KDOpRLkVivBv-jDrxAxAs7qx2TpE7"
    "gunwCxbzkyMrrBnU0krQ1AVU13M3HR8tOJfHxtVKbMmQiTB-xe1vadaYqHjwYOh8gtLzNlftOUlJc&svctype"
    "=5&tempid=h5_group_info"
)
HELP_MENU_BILIBILI_URL = "https://space.bilibili.com/3493290083093446"
HELP_MENU_DOCS_URL = "https://github.com/Cynthia-lxx/ComfyDL_UI/tree/master/docs"
HELP_MENU_REPO_URL = "https://github.com/Cynthia-lxx/ComfyDL_UI"


def _patch_help_menu(assets: Path) -> bool:
    """Append the ComfyDL link group to the official Help dropdown menu."""
    path = _locate(assets, "GraphView-*.js", "Comfy.ContactSupport")
    text = _text(path)
    if HELP_MENU_MARK in text:
        logging.debug(f"{TAG} help menu already applied")
        return False
    anchor_idx = text.find(HELP_MENU_INSERT_BEFORE)
    if anchor_idx < 0:
        raise ValueError("help menu array tail not found")

    def item(key: str, icon: str, label: str, url: str) -> str:
        return (
            "{key:`" + key + "`,type:`item`,icon:`" + icon + "`,label:`" + label
            + "`,showExternalIcon:!0,action:()=>{trackResourceClick(`" + key
            + "`,!0),openExternalLink(`" + url + "`)," + HELP_MENU_MARK
            + "x(`close`)}}"
        )

    group = ",".join([
        item("cdl-docs", "icon-[lucide--book-open]", "ComfyDL · 项目文档",
             HELP_MENU_DOCS_URL),
        item("cdl-qq", "icon-[comfy--qq]", "ComfyDL · QQ 交流群",
             HELP_MENU_QQ_URL),
        item("cdl-bilibili", "icon-[comfy--bilibili]", "ComfyDL · Bilibili",
             HELP_MENU_BILIBILI_URL),
        item("cdl-repo", "icon-[lucide--github]", "ComfyDL · GitHub 仓库",
             HELP_MENU_REPO_URL),
    ])
    text = (
        text[:anchor_idx]
        # The array tail ends with `}}` (last official item) — a leading comma
        # is required between it and our first injected item, otherwise the
        # chunk dies with "Unexpected token '{'" and the app never loads.
        + HELP_MENU_MARK + "," + group
        + text[anchor_idx:]
    )
    _write(path, text)
    logging.info(f"{TAG} ComfyDL help group appended in {path.name}")
    return True


def _js_str(text: str) -> str:
    """Render a Python string as a JS template-literal body (backticks)."""
    return (text.replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("${", "\\${")
                .replace("\n", "\\n"))


def _zh_chunk_names(assets: Path) -> dict:
    """Resolve the hashed zh chunk file names from the i18n loader chunk.

    The loader maps ``"./zh/<name>.json"`` to lazy chunks whose names differ
    per frontend build, so the mapping is read at patch time instead of
    hard-coding hashes.
    """
    loader = _locate(assets, "i18n-*.js", '"./zh/nodeDefs.json"')
    text = _text(loader)
    names = {}
    for kind in ("main", "nodeDefs", "settings"):
        m = re.search(
            r'"\./zh/' + kind + r'\.json":\(\)=>\w+\(\(\)=>import\(`\./([\w-]+\.js)`',
            text,
        )
        if not m:
            raise ValueError(f"zh {kind}.json chunk not mapped in {loader.name}")
        names[kind] = m.group(1)
    return names


def _load_zh_node_defs() -> dict:
    """The merged zh node-def translations (ComfyDL docs + core names)."""
    import json

    path = Path(__file__).parents[1] / "locales" / "zh" / "nodeDefs.json"
    if not path.is_file():
        raise ValueError(f"missing {path} (run cdl_smoke_tests/gen_locales.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_zh_locales(assets: Path) -> bool:
    """Bake the repository's zh translations into the frontend's zh chunks.

    Three sub-merges (each idempotent via its own marker, each non-fatal for
    the others so a partial frontend update cannot take all of them down):

    * ``nodeDefs`` chunk — ``Object.assign(<defaultExport>, {node: {...}})``
      right before ``export{``: canvas node titles/descriptions in Chinese.
    * ``settings`` chunk — the ComfyDL settings translations appended as new
      vars plus export keys (the official ``Id_With_Underscores`` convention).
    * ``main`` chunk — ``settingsCategories`` deep-merged so the Settings
      panel groups read 事故现场 / 性能分析 instead of CrashSite / Profiling.
    """
    names = _zh_chunk_names(assets)
    applied = False

    # --- nodeDefs ------------------------------------------------------------
    nd = assets / names["nodeDefs"]
    text = _text(nd)
    if I18N_NODEDEFS_MARK not in text:
        m = re.search(r"(\w+) as default", text)
        if not m:
            raise ValueError(f"no default export in {nd.name}")
        entries = ",\n".join(
            f"{json_key}:{{display_name:`{_js_str(v.get('display_name', json_key))}`"
            + (
                f",description:`{_js_str(v['description'])}`"
                if v.get("description")
                else ""
            )
            + "}"
            for json_key, v in _load_zh_node_defs().items()
        )
        insert = f";Object.assign({m.group(1)},{I18N_NODEDEFS_MARK}{{{entries}}});"
        idx = text.rfind(";export{")
        if idx < 0:
            raise ValueError(f"no export statement in {nd.name}")
        _write(nd, text[:idx] + insert + text[idx:])
        logging.info(f"{TAG} zh nodeDefs merged into {nd.name}")
        applied = True

    # --- settings ------------------------------------------------------------
    st = assets / names["settings"]
    text = _text(st)
    if I18N_SETTINGS_MARK not in text:
        vars_, exports = [], []
        for i, (key, value) in enumerate(I18N_SETTINGS_KEYS.items()):
            var = f"_cdlS{i}"
            vars_.append(
                f"{var}={{name:`{_js_str(value['name'])}`,"
                f"tooltip:`{_js_str(value['tooltip'])}`}}"
            )
            exports.append(f"{var} as {key}")
        idx = text.rfind(";export{")
        if idx < 0:
            raise ValueError(f"no export statement in {st.name}")
        text = (
            text[:idx]
            + f";{I18N_SETTINGS_MARK}var " + ",".join(vars_) + ";"
            + text[idx:idx + len(";export{")] + ",".join(exports) + ","
            + text[idx + len(";export{"):]
        )
        _write(st, text)
        logging.info(f"{TAG} zh settings merged into {st.name}")
        applied = True

    # --- main (settingsCategories) -------------------------------------------
    mn = assets / names["main"]
    text = _text(mn)
    if I18N_MAIN_MARK not in text:
        m = re.search(r"(\w+) as default", text)
        if not m:
            raise ValueError(f"no default export in {mn.name}")
        dv = m.group(1)
        cats = ",".join(
            f"{json_key}:`{_js_str(v)}`"
            for json_key, v in I18N_SETTINGS_CATEGORIES_ZH.items()
        )
        insert = (
            f";(function(){{var sc=({dv}.settingsCategories="
            f"{dv}.settingsCategories||{{}});{I18N_MAIN_MARK}"
            f"Object.assign(sc,{{{cats}}});}})();"
        )
        idx = text.rfind(";export{")
        if idx < 0:
            raise ValueError(f"no export statement in {mn.name}")
        _write(mn, text[:idx] + insert + text[idx:])
        logging.info(f"{TAG} zh settingsCategories merged into {mn.name}")
        applied = True

    return applied


def _patch_default_locale(assets: Path) -> bool:
    """Make English the locale the frontend starts with.

    ``getDefaultLocale()`` feeds both the initial vue-i18n locale and the
    ``Comfy.Locale`` setting default, so overriding it covers both paths.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "i18n-*.js", "function getDefaultLocale(){")
    text = _text(path)
    if LOCALE_MARK in text:
        logging.debug(f"{TAG} default locale already applied")
        return False

    text = _replace_once(
        text,
        "function getDefaultLocale(){return resolveSupportedLocale(navigator.languages)}",
        f"function getDefaultLocale(){{{LOCALE_MARK}return`en`}}",
    )
    _write(path, text)
    logging.info(f"{TAG} default locale set to 'en' in {path.name}")
    return True


def _patch_nodes2_default(assets: Path) -> bool:
    """Enable the modern (Vue) node rendering by default.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "GraphView-*.js", "id:`Comfy.VueNodes.Enabled`")
    text = _text(path)
    if NODES2_MARK in text:
        logging.debug(f"{TAG} Nodes 2.0 default already applied")
        return False

    setting = text.index("id:`Comfy.VueNodes.Enabled`")
    brace = text.rindex("{", 0, setting)
    end = _object_end(text, brace)

    definition = _replace_once(text[brace:end], "defaultValue:!1", f"defaultValue:!0{NODES2_MARK}")
    definition = _replace_once(
        definition,
        'defaultsByInstallVersion:{"1.41.0":!1}',
        'defaultsByInstallVersion:{"1.41.0":!0}',
    )
    _write(path, text[:brace] + definition + text[end:])
    logging.info(f"{TAG} Nodes 2.0 enabled by default in {path.name}")
    return True


def _patch_tensor_slot_colour(assets: Path) -> bool:
    """Colour the ``TENSOR`` slot in every theme palette.

    Only the six theme tables count: their ``node_slot`` value starts with a real
    key, while the palette-merge expression ``node_slot:{...t.colors.node_slot``
    starts with ``...`` and must be left alone.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "settingStore-*.js", "node_slot:{")
    text = _text(path)
    if TENSOR_MARK in text:
        logging.debug(f"{TAG} TENSOR slot colour already applied")
        return False

    marker = "node_slot:{"
    insert = f",TENSOR:`{TENSOR_SLOT_COLOUR}`"
    out = []
    pos = 0
    search = 0
    tables = 0
    while True:
        start = text.find(marker, search)
        if start < 0:
            break
        body = start + len(marker)
        search = body
        if text[body:body + 3] == "...":
            continue
        end = text.index("}", body)
        out.append(text[pos:end])
        out.append(insert)
        pos = end
        tables += 1
    out.append(text[pos:])

    if tables != 6:
        raise ValueError(f"expected 6 theme palettes, found {tables}")
    _write(path, "".join(out))
    logging.info(f"{TAG} TENSOR slot colour added to {tables} palettes in {path.name}")
    return True


def _patch_nn_model_slot_colour(assets: Path) -> bool:
    """Colour the ``nn_model`` slot in every theme palette.

    ``nn_model`` is the unified "materialised nn.Module" type used by both the
    ComfyDL inference nodes (formerly ``cdlModel``) and the ``Network & Layers``
    training system, so they share one slot colour.

    A stale ``NNMODEL`` entry left behind by the earlier cdlModel -> NNMODEL
    rename is stripped first, so only the unified ``nn_model`` colour remains.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "settingStore-*.js", "node_slot:{")
    text = _text(path)
    # Drop the stale NNMODEL entry from the earlier rename so it does not linger.
    text = text.replace(",NNMODEL:`#FF6B6B`", "")
    if NN_MODEL_MARK in text:
        logging.debug(f"{TAG} nn_model slot colour already applied")
        return False

    marker = "node_slot:{"
    insert = f",nn_model:`{NN_MODEL_SLOT_COLOUR}`"
    out = []
    pos = 0
    search = 0
    tables = 0
    while True:
        start = text.find(marker, search)
        if start < 0:
            break
        body = start + len(marker)
        search = body
        if text[body:body + 3] == "...":
            continue
        end = text.index("}", body)
        out.append(text[pos:end])
        out.append(insert)
        pos = end
        tables += 1
    out.append(text[pos:])

    if tables != 6:
        raise ValueError(f"expected 6 theme palettes, found {tables}")
    _write(path, "".join(out))
    logging.info(f"{TAG} nn_model slot colour added to {tables} palettes in {path.name}")
    return True


def _patch_dataset_slot_colour(assets: Path) -> bool:
    """Colour the ``DATASET`` slot in every theme palette (teal ``#1ABC9C``).

    ``DATASET`` is the universal dataset type used by the ComfyDL data-IO and
    regression nodes, so it gets one shared slot colour distinct from the
    orange ``nn_model`` and the lemon-green ``TENSOR`` slots.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "settingStore-*.js", "node_slot:{")
    text = _text(path)
    if DATASET_MARK in text:
        logging.debug(f"{TAG} DATASET slot colour already applied")
        return False

    marker = "node_slot:{"
    insert = f",DATASET:`{DATASET_SLOT_COLOUR}`"
    out = []
    pos = 0
    search = 0
    tables = 0
    while True:
        start = text.find(marker, search)
        if start < 0:
            break
        body = start + len(marker)
        search = body
        if text[body:body + 3] == "...":
            continue
        end = text.index("}", body)
        out.append(text[pos:end])
        out.append(insert)
        pos = end
        tables += 1
    out.append(text[pos:])

    if tables != 6:
        raise ValueError(f"expected 6 theme palettes, found {tables}")
    _write(path, "".join(out))
    logging.info(f"{TAG} DATASET slot colour added to {tables} palettes in {path.name}")
    return True


def _patch_profiling_loader(web_root: Path) -> bool:
    """Inject the Profiling panel's module loader into ``index.html``.

    The panel's JS/CSS are repository assets served by ``server.py`` from
    ``/comfydl/profiling/``; this patch only adds the one ``<script>`` tag
    that pulls the JS in, right before ``</body>`` (after the frontend's own
    module entry, so ``window.comfyAPI`` is already populated).  The marker
    comment doubles as the idempotency check, like every other patch here.

    Returns ``True`` when the file was rewritten.
    """
    path = web_root / "index.html"
    text = _text(path)
    if PROFILING_LOADER_MARK in text:
        logging.debug(f"{TAG} profiling loader already applied")
        return False

    if text.count("</body>") != 1:
        raise ValueError(f"expected exactly one </body> in {path.name}")
    _write(path, text.replace("</body>", PROFILING_LOADER_TAG + "</body>"))
    logging.info(f"{TAG} profiling panel loader injected into {path.name}")
    return True


def _patch_crashsite_loader(web_root: Path) -> bool:
    """Inject the Crash site panel's module loader into ``index.html``.

    Same pattern as :func:`_patch_profiling_loader`: the panel's JS/CSS are
    repository assets served by ``server.py`` from ``/comfydl/crashsite/``,
    and this patch only adds the one ``<script>`` tag before ``</body>``.
    """
    path = web_root / "index.html"
    text = _text(path)
    if CRASHSITE_LOADER_MARK in text:
        logging.debug(f"{TAG} crashsite loader already applied")
        return False

    if text.count("</body>") != 1:
        raise ValueError(f"expected exactly one </body> in {path.name}")
    _write(path, text.replace("</body>", CRASHSITE_LOADER_TAG + "</body>"))
    logging.info(f"{TAG} crashsite panel loader injected into {path.name}")
    return True


def _patch_motion_css(web_root: Path) -> bool:
    """Inject the entrance pre-hide CSS + failsafe timer into ``index.html``.

    The block goes right before ``</head>`` so the rules are in the CSSOM
    before the first Vue render (no flicker). Everything is gated on html
    classes (cdl-motion-off / cdl-entrance-done / cdl-nodes-in / cdl-canvas-in)
    that only :mod:`app.motion_assets.motion` sets, plus a baked 9s timer that
    force-reveals everything should the module ever fail to load. The rules
    themselves sit inside ``@media not (prefers-reduced-motion: reduce)``.

    Returns ``True`` when the file was rewritten.
    """
    path = web_root / "index.html"
    text = _text(path)
    if MOTION_CSS_MARK in text:
        logging.debug(f"{TAG} motion css already applied")
        return False

    if text.count("</head>") != 1:
        raise ValueError(f"expected exactly one </head> in {path.name}")
    _write(path, text.replace("</head>", MOTION_CSS_TAG + "</head>"))
    logging.info(f"{TAG} motion pre-hide css injected into {path.name}")
    return True


def _patch_motion_loader(web_root: Path) -> bool:
    """Inject the Motion module loader into ``index.html``.

    Same pattern as :func:`_patch_profiling_loader`: the module is served by
    ``server.py`` from ``/comfydl/motion/`` and this patch only adds the one
    ``<script type="module">`` tag before ``</body>``.

    Returns ``True`` when the file was rewritten.
    """
    path = web_root / "index.html"
    text = _text(path)
    if MOTION_LOADER_MARK in text:
        logging.debug(f"{TAG} motion loader already applied")
        return False

    if text.count("</body>") != 1:
        raise ValueError(f"expected exactly one </body> in {path.name}")
    _write(path, text.replace("</body>", MOTION_LOADER_TAG + "</body>"))
    logging.info(f"{TAG} motion module loader injected into {path.name}")
    return True


def _patch_splash_brand(web_root: Path) -> bool:
    """Replace the official splash contents with the branded boot screen.

    The official ``#splash-loader`` element (wave-animated "Comfy" SVG) is
    swapped for the homepage-fork boot screen: animated logo GIF, progress
    bar with moving stripes, and a live boot log fed by motion.js's real
    signals (resource loads, /api milestones, spinner state). The official
    element id is kept on a hidden stub so the official App watcher's
    ``getElementById('splash-loader')?.remove()`` resolves harmlessly; the
    visible splash lives under ``#cdl-splash`` and is owned end-to-end by
    motion.js (fade-out on the spinner falling edge, then removal).

    Idempotent via the marker; also re-applies cleanly when the official
    splash was already replaced (the regex simply finds nothing to replace).

    Returns ``True`` when the file was rewritten.
    """
    path = web_root / "index.html"
    text = _text(path)
    if SPLASH_BRAND_MARK in text:
        logging.debug(f"{TAG} splash brand already applied")
        return False

    # The official splash div contains only the <svg> (no nested div), so the
    # first </div> after its opening tag closes it.
    m = re.search(r'<div id="splash-loader"[^>]*>.*?</div>', text, re.DOTALL)
    if not m:
        raise ValueError("official #splash-loader element not found")
    text = text[:m.start()] + SPLASH_BRAND_MARK + SPLASH_BRAND_DOM + text[m.end():]
    _write(path, text)
    logging.info(f"{TAG} branded splash installed in {path.name}")
    return True


def _patch_brand_logo(web_root: Path) -> bool:
    """Swap the favicon and the top-left menu logo to ``DL_trans.png``.

    Three rewrite targets:
      - ``index.html``: the single favicon ``<link>`` points at the PNG (type
        fixed from image/x-icon to image/png);
      - ``index.html``: a <style> block hides the official inline ComfyLogo
        SVG and paints the ``.comfy-menu-button-wrapper .grid`` wrapper with
        the PNG (homepage fork's logo-fx technique - a raster image cannot
        inherit the SVG's currentColor fill);
      - GraphView chunk: ``useProgressFavicon()`` swaps the favicon at runtime
        for its progress frames (``frame_9.png`` when idle), silently
        overwriting the baked <link> - its idle frame is redirected to the
        brand PNG too (the 0..8 progress frames keep official artwork).

    Each target is independently idempotent; the function returns ``True``
    when anything was rewritten.
    """
    assets = web_root / "assets"
    changed = False

    path = web_root / "index.html"
    text = _text(path)
    # Favicon link: replace whatever icon link is current (official .ico or an
    # earlier brand generation) with the trimmed-derivative target.
    if FAVICON_TARGET_LINK not in text:
        if not re.search(r'<link rel="icon"', text):
            raise ValueError("official favicon link not found")
        text = re.sub(r'<link rel="icon"[^>]*/>', FAVICON_TARGET_LINK, text, count=1)
        changed = True
    # Menu-logo style block: refresh when its contents drifted (URL upgrades).
    # NOTE: the marker lives inside the tag (it must be removed together with
    # a stale style block, then re-injected) - never gate the re-injection on
    # the marker, or an upgrade deletes the style and leaves nothing behind.
    if BRAND_LOGO_TAG not in text:
        text = re.sub(r'<style id="cdl-brand-logo-style">.*?</style>', "",
                      text, flags=re.DOTALL)
        text = text.replace(BRAND_LOGO_MARK, "")
        if text.count("</head>") != 1:
            raise ValueError(f"expected exactly one </head> in {path.name}")
        text = text.replace("</head>", BRAND_LOGO_TAG + "</head>")
        changed = True
    if changed:
        _write(path, text)
        logging.info(f"{TAG} brand logo (favicon + menu logo) installed in {path.name}")
    else:
        logging.debug(f"{TAG} brand logo already applied to index.html")

    # The runtime favicon override lives in the GraphView chunk - without
    # this, useProgressFavicon() puts the official artwork back the moment
    # the app goes idle and the baked <link> never shows.
    gv = _locate(assets, "GraphView-*.js", "useProgressFavicon")
    gv_text = _text(gv)
    stale = [s for s in FAVICON_IDLE_PREV if s in gv_text]
    if stale:
        for s in stale:
            gv_text = gv_text.replace(s, FAVICON_IDLE_NEW)
        _write(gv, gv_text)
        logging.info(f"{TAG} progress-favicon idle frame -> DL_trans_favicon.png in {gv.name}")
        changed = True
    elif FAVICON_IDLE_NEW in gv_text:
        logging.debug(f"{TAG} progress-favicon idle frame already redirected")
    else:
        raise ValueError("useProgressFavicon idle frame signature not found")
    return changed


def _patch_startup_workflow(assets: Path) -> bool:
    """Redirect the veteran-user startup graph to the ComfyDL template.

    0.3.14's ``resolveStartupOutcome`` loads the built-in SD3 demo via a bare
    ``app.loadGraphData()`` for users with ``Comfy.TutorialCompleted`` set -
    the startup path never reaches ``loadDefaultWorkflow`` (patched
    separately). This rewrite sends that bare call through the
    ``/templates/<name>.json`` overlay channel, falling back to the upstream
    graph on any failure. Loud failure on signature drift, like the other
    workflow patches. A frontend already baked with a previous template name
    is upgraded in place (see ``_upgrade_baked_template_url``).

    Returns ``True`` when the chunk was rewritten, ``False`` when the marker
    was already there and the URL was current.
    """
    path = _locate(assets, "GraphView-*.js", "resolveStartupOutcome")
    text = _text(path)
    if STARTUP_WORKFLOW_MARK in text:
        text, upgraded = _upgrade_baked_template_url(text)
        if upgraded:
            _write(path, text)
            logging.info(
                f"{TAG} startup workflow template upgraded -> "
                f"/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json in {path.name}"
            )
            return True
        logging.debug(f"{TAG} startup workflow already applied")
        return False

    match = STARTUP_WORKFLOW_PATTERN.search(text)
    if match is None:
        raise ValueError("startup loadGraphData call signature not found")
    app = match.group(1)
    patched = (
        f"get(`Comfy.TutorialCompleted`)?(await (async()=>{{"
        f"try{{const _cdlRes=await fetch(`/templates/{DEFAULT_WORKFLOW_TEMPLATE}.json`,"
        f"{{cache:`no-store`}});"
        f"if(_cdlRes.ok){{await {app}.loadGraphData(await _cdlRes.json());return}}}}"
        f"catch(_cdlErr){{}}"
        f"await {app}.loadGraphData()"
        f"}})(),`restored`){STARTUP_WORKFLOW_MARK}"
    )
    text = _replace_once(text, match.group(0), patched)
    _write(path, text)
    logging.info(
        f"{TAG} startup workflow -> /templates/{DEFAULT_WORKFLOW_TEMPLATE}.json "
        f"in {path.name}"
    )
    return True


def _strip_about_section(text: str) -> str:
    """Drop a previously injected About section.

    Two markers delimit the injection, so the removal stays exact no matter how
    the section wording changes between revisions. This is what makes the patch
    re-applicable instead of append-only.
    """
    return re.sub(
        re.escape(ABOUT_BEGIN) + ".*?" + re.escape(ABOUT_END),
        "",
        text,
        flags=re.DOTALL,
    )


def _patch_about_panel(assets: Path) -> bool:
    """Add the ComfyDL_UI section and the separate upstream section to About.

    The panel is a compiled Vue chunk without any extension point, so the sections
    are injected into its render function. The minified helper names are read out
    of the chunk itself, which keeps the patch working when a frontend upgrade
    renames them.

    A previous injection is stripped first, so improving the wording only means
    editing this function: the new section replaces the old one on the next
    startup. Returns ``True`` when the asset was rewritten.
    """
    path = _locate(assets, "AboutPanel-*.js", "g.about")
    original = _text(path)
    text = _strip_about_section(original)

    heading = re.search(r"(\w+)\(`h2`,(\w+),(\w+)\((\w+)\.\$t\(`g\.about`\)\),1\),", text)
    if heading is None:
        raise ValueError("about heading not found")
    element, display, context = heading.group(1), heading.group(3), heading.group(4)

    stats = re.search(r"(\w+)\((\w+)\)\.systemStats", text)
    if stats is None:
        raise ValueError("system stats store not found")
    version = f"{stats.group(1)}({stats.group(2)}).systemStats?.system?.comfyui_version??``"

    def link(url: str, key: str) -> str:
        return (
            f"{element}(`a`,{{href:`{url}`,target:`_blank`,rel:`noopener noreferrer`,"
            f"class:`text-sm`}},{display}({context}.$t(`comfydlAbout.{key}`)),1)"
        )

    def paragraph(key: str) -> str:
        return (
            f"{element}(`div`,{{class:`text-sm opacity-70`}},"
            f"{display}({context}.$t(`comfydlAbout.{key}`)),1)"
        )

    links = (
        f"{element}(`div`,{{class:`flex flex-wrap gap-x-4`}},["
        f"{link(PROJECT_REPO, 'projectRepo')},{link(UPSTREAM_REPO, 'upstreamRepo')}"
        "])"
    )
    section = (
        f"{element}(`div`,{{class:`cdl-about space-y-2`}},["
        f"{element}(`div`,{{class:`font-medium`}},{display}(`ComfyDL_UI v`)+({version}),1),"
        f"{paragraph('summary')},{paragraph('notOriginal')},{links}"
        "]),"
    )
    upstream_heading = (
        f"{element}(`h2`,{{class:`mb-2 text-2xl font-semibold`}},"
        f"{display}({context}.$t(`comfydlAbout.upstreamHeading`)),1),"
    )
    # Both markers wrap everything that gets injected, the upstream heading
    # included: anything left outside them would be added again on every startup.
    injection = f"{ABOUT_BEGIN}{section}{upstream_heading}{ABOUT_END}"

    patched = _replace_once(text, heading.group(0), heading.group(0) + injection)
    # The first badge reports the version of the running application, which is
    # this fork, so it must not be labelled as the upstream ComfyUI release. A
    # second run finds it already renamed, which is fine.
    patched, renamed = re.subn(r"label:`ComfyUI \$\{[^}]+\}`", "label:`ComfyUI`", patched)
    if renamed == 0 and "label:`ComfyUI`" not in patched:
        raise ValueError("ComfyUI version badge not found")

    if patched == original:
        logging.debug(f"{TAG} about panel already up to date")
        return False
    _write(path, patched)
    logging.info(f"{TAG} about panel extended in {path.name}")
    return True
