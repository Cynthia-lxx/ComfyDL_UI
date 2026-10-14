"""One-shot baking script for the ComfyFront_sep static frontend package.

Imports the main repository's ``app/frontend_patch.py`` (by file path, without
touching the repository) and re-applies its content-marker patches to the
*static copy* under this directory, then applies this project's own patches on
top. All patches are idempotent content-marker rewrites — same style as the
main repo's, so they survive frontend upgrades only as long as the markers do.

Main-repo patches reused here (already in place in this copy; reported as
already-applied on every run):
  * default locale  -> English
  * Nodes 2.0       -> enabled by default (Vue markdown widget pipeline)
  * TENSOR / nn_model / DATASET slot colours in all 6 theme palettes

Own patches:
  * startup workflow -> fetch ./homepage_workflow.json (the repo landing
    canvas) before falling back to the bundled SD graph. The upstream
    ``Comfy.Workflow.Workflow`` setting is NOT read at startup in this
    frontend version, so a rewrite of ``loadDefaultWorkflow`` is the only
    reliable hook.
  * sidebar tabs: keep Queue (job history) + Node Library, drop
    Assets / Model Library / Workflows / Apps. (Templates/Help/Shortcuts/
    Console are not sidebar tabs in this frontend version — they are menu
    entries and dialogs — so there is nothing to remove there.)
  * default color palette: dark -> github.

The default-workflow / profiling-loader / crashsite-loader patches from the
main repo are intentionally NOT reused: they target the live backend.

Also generates ``extensions-manifest.json``: the URL list that a real ComfyUI
server would return from ``GET /api/extensions`` (server.py globs
``web_root/extensions/**/*.js``), plus this project's own ``home.js``.
"""

import importlib.util
import json
import re
from pathlib import Path

SEP = Path(__file__).resolve().parent
REPO_ROOT = Path(r"P:\Dev\ComfyUI_Refs\ComfyDL_UI")
FRONTEND_PATCH = REPO_ROOT / "app" / "frontend_patch.py"

HOME_EXTENSION_URL = "/extensions/comfydl_home/home.js"

# Markers doubling as idempotency checks.
HOMEPAGE_MARK = "/*ComfyFront_sep:homepage-workflow*/"
TABS_MARK = "/*ComfyFront_sep:tabs*/"
PALETTE_MARK = "/*ComfyFront_sep:palette*/"
NO_RESTORE_MARK = "/*ComfyFront_sep:no-restore*/"

STARTUP_WORKFLOW_URL = "./homepage_workflow.json"

# The five built-in tab registrations, exactly as compiled (one contiguous
# chain). Node Library is kept; the other four are dropped.
TABS_OLD = (
    "registerSidebarTab(useAssetsSidebarTab()),"
    "registerSidebarTab(useNodeLibrarySidebarTab()),"
    "registerSidebarTab(useModelLibrarySidebarTab()),"
    "registerSidebarTab(useWorkflowsSidebarTab()),"
    "registerSidebarTab(useAppsSidebarTab())"
)
TABS_NEW = "registerSidebarTab(useNodeLibrarySidebarTab())" + TABS_MARK

PALETTE_OLD = (
    "id:`Comfy.ColorPalette`,name:`The active color palette id`,"
    "type:`hidden`,defaultValue:`dark`"
)
PALETTE_NEW = (
    "id:`Comfy.ColorPalette`,name:`The active color palette id`,"
    "type:`hidden`,defaultValue:`github`" + PALETTE_MARK
)


def load_frontend_patch():
    spec = importlib.util.spec_from_file_location("cdl_frontend_patch", FRONTEND_PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- own patches
HOMEPAGE_V2_MARK = "/*ComfyFront_sep:homepage-v2*/"


def _homepage_v2_body(service: str, stock_graph: str) -> str:
    """Patched loadDefaultWorkflow with verbose startup logging."""
    return (
        f"loadDefaultWorkflow=async()=>{{"
        f"try{{"
        f"console.log(`[ComfyFront_sep] loadDefaultWorkflow: fetching {STARTUP_WORKFLOW_URL}`);"
        f"const _sepRes=await fetch(`{STARTUP_WORKFLOW_URL}`,{{cache:`no-store`}});"
        f"console.log(`[ComfyFront_sep] homepage fetch status`,_sepRes.status);"
        f"if(_sepRes.ok){{"
        f"try{{await {service}.loadGraphData(await _sepRes.json());"
        f"console.log(`[ComfyFront_sep] homepage workflow loaded OK`);"
        f"return}}"
        f"catch(_sepLoadErr){{console.error(`[ComfyFront_sep] homepage loadGraphData FAILED:`,_sepLoadErr)}}"
        f"}}"
        f"}}"
        f"catch(_cdlErr){{console.error(`[ComfyFront_sep] homepage fetch error:`,_cdlErr)}}"
        f"console.warn(`[ComfyFront_sep] FALLING BACK to bundled SD default graph`);"
        f"await {service}.loadGraphData({stock_graph})"
        f"{HOMEPAGE_MARK}{HOMEPAGE_V2_MARK}}}"
    )


def _patch_homepage_workflow(fp, assets: Path) -> bool:
    """Point the startup workflow hook at our landing canvas.

    The copy inherited the main repo's ``_patch_default_workflow`` (it fetches
    ``/templates/tabular_regression_production.json`` — a URL that does not
    exist here, so startup always falls back to the bundled SD graph). We
    retarget that fetch to ``./homepage_workflow.json`` and add our own
    idempotency marker. If the inherited patch were absent (fresh upstream
    copy), we rewrite the upstream call instead. Returns True when rewritten.
    """
    path = fp._locate(assets, "settingStore-*.js", "loadDefaultWorkflow=async()=>{")
    text = fp._text(path)
    if HOMEPAGE_V2_MARK in text:
        print(f"  homepage-workflow: already applied v2 ({path.name})")
        return False

    if HOMEPAGE_MARK in text:
        # v1 applied: replace the whole patched body with the logged v2 body.
        m = re.search(
            r"loadDefaultWorkflow=async\(\)=>\{.*?/\*ComfyFront_sep:homepage-workflow\*/\}",
            text,
        )
        if m is None:
            raise ValueError("v1 homepage patch body not found for upgrade")
        service_m = re.search(r"await ([A-Za-z_$][\w$]*)\.loadGraphData\(od\)", m.group(0))
        if service_m is None:
            raise ValueError("service reference not found in v1 homepage patch body")
        text = fp._replace_once(
            text, m.group(0), _homepage_v2_body(service_m.group(1), "od")
        )
        fp._write(path, text)
        print(f"  homepage-workflow: upgraded to v2 (logging) in {path.name}")
        return True

    if "/*ComfyDL_UI:default-workflow*/" in text:
        # Inherited main-repo patch: just retarget its template URL.
        old_url = "`/templates/tabular_regression_production.json`"
        new_url = f"`{STARTUP_WORKFLOW_URL}`"
        if old_url not in text:
            raise ValueError("inherited default-workflow patch found but URL shape changed")
        text = fp._replace_once(text, old_url, new_url)
        text = fp._replace_once(
            text,
            "/*ComfyDL_UI:default-workflow*/",
            "/*ComfyDL_UI:default-workflow*/" + HOMEPAGE_MARK,
        )
    else:
        # Fresh upstream copy: rewrite the stock loadDefaultWorkflow call.
        match = re.search(
            r"loadDefaultWorkflow=async\(\)=>\{await "
            r"([A-Za-z_$][\w$]*)\.loadGraphData\(([A-Za-z_$][\w$]*)\)\}",
            text,
        )
        if match is None:
            raise ValueError("loadDefaultWorkflow call signature not found")
        service, stock_graph = match.group(1), match.group(2)
        text = fp._replace_once(
            text, match.group(0), _homepage_v2_body(service, stock_graph)
        )
    fp._write(path, text)
    print(f"  homepage-workflow: applied -> {STARTUP_WORKFLOW_URL} in {path.name}")
    return True


def _patch_sidebar_tabs(fp, assets: Path) -> bool:
    """Drop Assets / Model Library / Workflows / Apps sidebar tabs."""
    # Locate by a needle that survives the patch itself (idempotency check
    # must come BEFORE the old-shape lookup, which disappears when applied).
    path = fp._locate(assets, "settingStore-*.js", "useNodeLibrarySidebarTab()")
    text = fp._text(path)
    if TABS_MARK in text:
        print(f"  sidebar-tabs: already applied ({path.name})")
        return False
    text = fp._replace_once(text, TABS_OLD, TABS_NEW)
    fp._write(path, text)
    print(f"  sidebar-tabs: applied (kept node-library) in {path.name}")
    return True


def _patch_default_palette(fp, assets: Path) -> bool:
    """Default color palette: dark -> github."""
    path = fp._locate(assets, "GraphView-*.js", "id:`Comfy.ColorPalette`")
    text = fp._text(path)
    if PALETTE_MARK in text:
        print(f"  default-palette: already applied ({path.name})")
        return False
    text = fp._replace_once(text, PALETTE_OLD, PALETTE_NEW)
    fp._write(path, text)
    print(f"  default-palette: applied (dark -> github) in {path.name}")
    return True


RESTORE_OLD = "let loadPreviousWorkflowFromStorage=async()=>{let e=d.getActivePath();"
RESTORE_NEW = (
    "let loadPreviousWorkflowFromStorage=async()=>{"
    + NO_RESTORE_MARK
    + "return!1;let e=d.getActivePath();"
)

# The real boot path: initializeWorkflow -> resolveStartupOutcome. When
# Comfy.TutorialCompleted is truthy it calls X.loadGraphData() with NO
# arguments, which loads the bundled SD text-to-image graph as an
# "Unsaved Workflow" tab. Redirect that branch to our homepage JSON.
RESOLVE_OLD = (
    "resolveStartupOutcome=async()=>n.get(`Comfy.TutorialCompleted`)"
    "?(await X.loadGraphData(),`restored`):"
)
RESOLVE_NEW = (
    "resolveStartupOutcome=async()=>n.get(`Comfy.TutorialCompleted`)"
    "?(await X.loadGraphData(await fetch(`./homepage_workflow.json`,"
    "{cache:`no-store`}).then(e=>e.json())),"
    "console.log(`[ComfyFront_sep] startup: homepage loaded (resolveStartupOutcome)`),"
    "`restored`):/*ComfyFront_sep:startup-homepage*/"
)


def _patch_disable_session_restore(fp, assets: Path) -> bool:
    """Disable the boot-time draft/session restore.

    The app boots by restoring the last active workflow from the client-side
    draft store (localStorage, hash-keyed payloads that our ``Comfy.*``/
    ``workflow`` cleanup cannot match). On a machine that once ran a real
    ComfyUI — or an earlier session of this demo that captured the bundled SD
    graph as a draft — the restored draft opens as a tab and the homepage
    only shows after the user closes it.

    A static homepage must always open the homepage: make
    ``loadPreviousWorkflowFromStorage`` return false so boot falls through to
    ``loadDefaultWorkflow`` (which bake_patches.py already retargets to
    ``./homepage_workflow.json``). Returns True when rewritten.
    """
    path = fp._locate(assets, "GraphView-*.js", "loadPreviousWorkflowFromStorage")
    text = fp._text(path)
    if NO_RESTORE_MARK in text:
        print(f"  session-restore: already applied ({path.name})")
        return False
    text = fp._replace_once(text, RESTORE_OLD, RESTORE_NEW)
    fp._write(path, text)
    print(f"  session-restore: applied (boot always opens homepage) in {path.name}")
    return True


def _patch_startup_resolve(fp, assets: Path) -> bool:
    """Point the TutorialCompleted startup branch at the homepage workflow.

    See RESOLVE_OLD/RESOLVE_NEW: with the tutorial flag set (needed to skip
    the onboarding popup), the boot outcome loader otherwise loads the
    bundled SD graph via a no-argument loadGraphData(). Returns True when
    rewritten.
    """
    path = fp._locate(assets, "GraphView-*.js", "resolveStartupOutcome")
    text = fp._text(path)
    if "/*ComfyFront_sep:startup-homepage*/" in text:
        print(f"  startup-resolve: already applied ({path.name})")
        return False
    text = fp._replace_once(text, RESOLVE_OLD, RESOLVE_NEW)
    fp._write(path, text)
    print(f"  startup-resolve: applied (TutorialCompleted branch -> homepage) in {path.name}")
    return True


BLANK_MARK = "/*ComfyFront_sep:blank-homepage*/"


def _patch_blank_workflow(fp, assets: Path) -> bool:
    """Point ``loadBlankWorkflow`` at the homepage.

    When the TutorialCompleted setting is not yet applied (async settings
    race at boot), ``resolveStartupOutcome`` takes its else branch and calls
    ``loadBlankWorkflow`` — which in this frontend actually loads the bundled
    SD default graph (``cd``), producing the "SD3 opens at startup" symptom.
    Make it fetch the homepage first, SD graph as fallback. Returns True when
    rewritten.
    """
    path = fp._locate(assets, "settingStore-*.js", "loadBlankWorkflow=async()=>{")
    text = fp._text(path)
    if BLANK_MARK in text:
        print(f"  blank-workflow: already applied ({path.name})")
        return False
    match = re.search(
        r"loadBlankWorkflow=async\(\)=>\{await ([A-Za-z_$][\w$]*)\.loadGraphData\(([A-Za-z_$][\w$]*)\)\}",
        text,
    )
    if match is None:
        raise ValueError("loadBlankWorkflow call signature not found")
    service, stock_graph = match.group(1), match.group(2)
    patched = (
        f"loadBlankWorkflow=async()=>{{"
        f"try{{"
        f"const _sepRes=await fetch(`{STARTUP_WORKFLOW_URL}`,{{cache:`no-store`}});"
        f"if(_sepRes.ok){{await {service}.loadGraphData(await _sepRes.json());"
        f"console.log(`[ComfyFront_sep] startup: homepage loaded (loadBlankWorkflow)`);"
        f"return}}"
        f"}}"
        f"catch(_sepErr){{console.error(`[ComfyFront_sep] loadBlankWorkflow homepage failed:`,_sepErr)}}"
        f"await {service}.loadGraphData({stock_graph})"
        f"}}{BLANK_MARK}"
    )
    text = fp._replace_once(text, match.group(0), patched)
    fp._write(path, text)
    print(f"  blank-workflow: applied (blank startup -> homepage) in {path.name}")
    return True


# ------------------------------------------------------------------ driver
def bake_patches(fp) -> dict:
    assets = SEP / "ui"  # renamed from 'assets' to defeat stale browser caches
    results = {}

    # Main-repo patches (idempotent no-ops when already in place).
    for name, fn in (
        ("default_locale", fp._patch_default_locale),
        ("nodes2_default", fp._patch_nodes2_default),
        ("tensor_slot_colour", fp._patch_tensor_slot_colour),
        ("nn_model_slot_colour", fp._patch_nn_model_slot_colour),
        ("dataset_slot_colour", fp._patch_dataset_slot_colour),
    ):
        try:
            results[name] = "already" if not fn(assets) else "applied"
        except Exception as exc:  # non-fatal, mirrors server behaviour
            results[name] = f"FAILED: {exc}"

    # Own patches.
    for name, fn in (
        ("homepage_workflow", _patch_homepage_workflow),
        ("sidebar_tabs", _patch_sidebar_tabs),
        ("default_palette", _patch_default_palette),
        ("session_restore", _patch_disable_session_restore),
        ("startup_resolve", _patch_startup_resolve),
        ("blank_workflow", _patch_blank_workflow),
        ("default_graph", lambda fp_, a_: _patch_default_graph_constant(
            fp_, a_, HOMEPAGE_MINIFIED)),
        ("help_menu", _patch_help_menu),
        ("icon_css", _patch_icon_css),
        ("side_buttons", _patch_sidebar_buttons),
        ("startup_gate", _patch_startup_gate),
    ):
        try:
            results[name] = "already" if not fn(fp, assets) else "applied"
        except Exception as exc:
            results[name] = f"FAILED: {exc}"
    return results


DEFAULT_GRAPH_FILE = "ComfyDL_UI Index Page"  # marker text inside the homepage


def _patch_default_graph_constant(fp, assets: Path, homepage_minified: str) -> bool:
    """Replace the bundled SD default graph constant with the homepage.

    ``var od={...SD3 graph...}`` in settingStore is the single source every
    "default graph" path reads from: no-argument ``loadGraphData()``, the
    ``window.comfyAPI.defaultGraph`` export, and every fallback we could not
    chase down individually. Substituting the object literal once covers all
    of them. ``sd=JSON.stringify(od)`` re-serialises at runtime, so the string
    copy follows automatically.

    Delimiting trick: the literal is immediately followed by the unique
    anchor ``,sd=JSON.stringify(od)`` — so instead of brace-matching (which
    breaks on braces inside graph strings), the span between ``var od={`` and
    that anchor is replaced wholesale. This also self-repairs any earlier
    mis-bounded attempt. Returns True when rewritten.
    """
    # Locate by a needle that survives the patch (the export block stays).
    path = fp._locate(assets, "settingStore-*.js", "window.comfyAPI.defaultGraph")
    text = fp._text(path)

    start_m = re.search(r"var od=\{", text)
    anchor = ",sd=JSON.stringify(od)"
    if start_m is None or anchor not in text:
        raise ValueError("default graph literal or anchor not found")
    # span starts AT od's opening '{' (not after it!) so the replacement
    # carries its own braces — starting one char later produced `{{`.
    span_start = text.index("{", start_m.start())
    span_end = text.index(anchor, span_start)
    current = text[span_start:span_end]

    # "Already applied" only when the span actually parses back to the
    # homepage — a truncated span from an earlier mis-bounded attempt also
    # contains the marker text, so the marker alone cannot be trusted.
    if current == homepage_minified:
        print(f"  default-graph: already applied ({path.name})")
        return False
    try:
        if json.loads(current) == json.loads(homepage_minified):
            print(f"  default-graph: already applied ({path.name})")
            return False
    except (json.JSONDecodeError, ValueError):
        print("  default-graph: repairing corrupted span from an earlier attempt")

    text = text[:span_start] + homepage_minified + text[span_end:]
    fp._write(path, text)
    print(f"  default-graph: applied (bundled SD graph -> homepage) in {path.name}")
    return True


# ------------------------------------------------- help menu & icon css
HELP_MARK = "/*ComfyFront_sep:help-menu*/"
ICON_CSS_MARK = "/*ComfyFront_sep:icon-css*/"

QQ_URL = (
    "https://qun.qq.com/universal-share/share?ac=1&authKey=IbE90VoEPI2G0NeI5Gay90"
    "%2BPfMFD1o/uClv6wpNdhoEA3JqkMsn209yAaRzl7QRb&busi_data=eyJncm91cENvZGUiOiI5MjEyMTk3"
    "NDciLCJ0b2tlbiI6Ilg1V1NMWEFrZWpFTkVxcm5MN0MwWm1rWm1VTit3OVFrcklmVTA1OGRYQVA4NTFrQmIz"
    "cW9WZFlRVnpsTXVyK3giLCJ1aW4iOiIzOTI0NTIyNDIwIn0=&data=8KDOpRLkVivBv-jDrxAxAs7qx2TpE7"
    "gunwCxbzkyMrrBnU0krQ1AVU13M3HR8tOJfHxtVKbMmQiTB-xe1vadaYqHjwYOh8gtLzNlftOUlJc&svctype"
    "=5&tempid=h5_group_info"
)
BILIBILI_URL = "https://space.bilibili.com/3493290083093446"
DOCS_URL = "https://github.com/Cynthia-lxx/ComfyDL_UI/tree/master/docs"
REPO_URL_SEP = "https://github.com/Cynthia-lxx/ComfyDL_UI"

HELP_REPLACES = [
    # 1. drop "Give Feedback"
    (
        "{key:`feedback`,type:`item`,icon:`icon-[lucide--clipboard-pen]`,"
        "label:i(`helpCenter.feedback`),showExternalIcon:!1,action:()=>{"
        "trackResourceClick(`help_feedback`,!1),p.execute(`Comfy.ContactSupport`),"
        "x(`close`)}},",
        "",
    ),
    # 2. drop "Help & Support"
    (
        "{key:`help`,type:`item`,icon:`icon-[lucide--message-circle-question]`,"
        "label:i(`helpCenter.help`),action:()=>{trackResourceClick(`help_feedback`,!1),"
        "p.execute(`Comfy.ContactSupport`),x(`close`)}},",
        "",
    ),
    # 3. Docs -> our technical docs directory on GitHub (carries the marker)
    (
        "openExternalLink(d(`/`,{includeLocale:!0}))",
        "openExternalLink(`" + DOCS_URL + "`)" + HELP_MARK,
    ),
    # 4. Discord -> QQ (penguin icon, join-group link).
    # NOTE: plain concatenation — an f-string here would eat the literal `}}`
    # as an escape sequence and break the syntax.
    (
        "{key:`discord`,type:`item`,icon:`pi pi-discord`,label:`Discord`,"
        "showExternalIcon:!0,action:()=>{trackResourceClick(`discord`,!0),"
        "openExternalLink(s.discord),x(`close`)}},",
        "{key:`qq`,type:`item`,icon:`icon-[comfy--qq]`,label:`QQ`,"
        "showExternalIcon:!0,action:()=>{trackResourceClick(`qq`,!0),openExternalLink(`"
        + QQ_URL + "`),x(`close`)}},",
    ),
    # 5. Github -> our repo, and append a Bilibili entry right after it
    (
        "openExternalLink(s.github),x(`close`)}}",
        "openExternalLink(`" + REPO_URL_SEP + "`),x(`close`)}},"
        "{key:`bilibili`,type:`item`,icon:`icon-[comfy--bilibili]`,label:`Bilibili`,"
        "showExternalIcon:!0,action:()=>{trackResourceClick(`bilibili`,!0),openExternalLink(`"
        + BILIBILI_URL + "`),x(`close`)}},",
    ),
    # 6. drop "Manager Extension"
    (
        "return e.push({key:`manager`,type:`item`,icon:uS,"
        "label:i(`helpCenter.managerExtension`),showRedDot:M.value,action:async()=>{"
        "trackResourceClick(`manager`,!1),await lu().openManager("
        "{initialTab:xu.All,showToastOnLegacyError:!1}),x(`close`)}}),ee.value&&e.push(",
        "return ee.value&&e.push(",
    ),
]

# The fully rewritten menu array span, used both as the repair target and the
# idempotency check. Built by plain concatenation for the same f-string reason.
EXPECTED_MENU_SPAN = (
    "let e=[{key:`docs`,type:`item`,icon:`icon-[lucide--book-open]`,"
    "label:i(`helpCenter.docs`),showExternalIcon:!0,action:()=>{"
    "trackResourceClick(`docs`,!0),openExternalLink(`" + DOCS_URL + "`)"
    + HELP_MARK + ",x(`close`)}},"
    "{key:`qq`,type:`item`,icon:`icon-[comfy--qq]`,label:`QQ`,"
    "showExternalIcon:!0,action:()=>{trackResourceClick(`qq`,!0),openExternalLink(`"
    + QQ_URL + "`),x(`close`)}},"
    "{key:`github`,type:`item`,icon:`icon-[lucide--github]`,"
    "label:i(`helpCenter.github`),showExternalIcon:!0,action:()=>{"
    "trackResourceClick(`github`,!0),openExternalLink(`" + REPO_URL_SEP + "`),x(`close`)}},"
    "{key:`bilibili`,type:`item`,icon:`icon-[comfy--bilibili]`,label:`Bilibili`,"
    "showExternalIcon:!0,action:()=>{trackResourceClick(`bilibili`,!0),openExternalLink(`"
    + BILIBILI_URL + "`),x(`close`)}}];return ee.value&&e.push("
)

ICON_CSS = (
    "<style>" + ICON_CSS_MARK
    + ".icon-\\[comfy--qq\\]{display:inline-block;width:1em;height:1em;line-height:1;"
      "background-color:currentColor;-webkit-mask:url('./icons/qqchat.svg') no-repeat center/contain;"
      "mask:url('./icons/qqchat.svg') no-repeat center/contain}"
    + ".icon-\\[comfy--bilibili\\]{display:inline-block;width:1em;height:1em;line-height:1;"
      "background-color:currentColor;-webkit-mask:url('./icons/bilibili.svg') no-repeat center/contain;"
      "mask:url('./icons/bilibili.svg') no-repeat center/contain}"
    + "</style>"
)


def _patch_help_menu(fp, assets: Path) -> bool:
    """Rewrite the Help dropdown menu (HelpCenterMenuContent) for the fork.

    Two entry paths:
      * fresh upstream array -> run the six content replaces;
      * any previously patched (possibly corrupted, e.g. by the f-string
        brace-escape bug) array -> wholesale-replace the span between
        ``let e=[{key:`docs``` and ``];return ee.value&&e.push(`` with
        EXPECTED_MENU_SPAN.
    """
    path = fp._locate(assets, "GraphView-*.js", "HelpCenterMenuContent")
    text = fp._text(path)

    start_m = re.search(r"let e=\[\{key:`docs`", text)
    end_anchor = "];return ee.value&&e.push("
    if start_m is None or end_anchor not in text:
        raise ValueError("help menu array span not found")
    span_start = start_m.start()
    span_end = text.index(end_anchor, span_start) + len(end_anchor)
    current = text[span_start:span_end]

    if current == EXPECTED_MENU_SPAN:
        print(f"  help-menu: already applied ({path.name})")
        return False

    if "p.execute(`Comfy.ContactSupport`)" in current:
        # Fresh upstream array: run the content replaces.
        for old, new in HELP_REPLACES:
            text = fp._replace_once(text, old, new)
        fp._write(path, text)
        print(f"  help-menu: applied (QQ/Bilibili/docs/github, removed 3 items) in {path.name}")
        return True

    # Corrupted or partially patched span: rewrite it wholesale.
    text = text[:span_start] + EXPECTED_MENU_SPAN + text[span_end:]
    fp._write(path, text)
    print(f"  help-menu: repaired (menu array rewritten) in {path.name}")
    return True


def _patch_icon_css(fp, assets: Path) -> bool:
    """Inject iconify-style CSS for our custom menu icons into index.html."""
    path = Path(assets).parent / "index.html"
    text = fp._text(path)
    if ICON_CSS_MARK in text:
        print("  icon-css: already applied (index.html)")
        return False
    if text.count("</head>") != 1:
        raise ValueError("expected exactly one </head> in index.html")
    text = fp._replace_once(text, "</head>", ICON_CSS + "</head>")
    fp._write(path, text)
    print("  icon-css: applied (qq + bilibili icon classes) in index.html")
    return True


SIDE_BUTTONS_MARK = "/*ComfyFront_sep:side-buttons*/"

# Templates button (hard-coded SidebarTemplatesButton render instance).
SIDE_TEMPLATES_OLD = "128)),A(aC)],2),l("
SIDE_TEMPLATES_NEW = "128))],2),l("
# Bottom-panel ("Console/Logs") toggle button render branch.
SIDE_CONSOLE_OLD = (
    "!P(!1)&&!e.hideWorkspaceToggles?(H(),m(XS,{key:1,"
    '"is-small":p.value},null,8,[`is-small`])):v(``,!0),'
)
SIDE_CONSOLE_NEW = SIDE_BUTTONS_MARK + "v(``,!0),"


def _patch_sidebar_buttons(fp, assets: Path) -> bool:
    """Remove the Templates button and the bottom-panel (Console) toggle.

    Both are hard-coded render calls inside the SideToolbar component (not
    registerSidebarTab entries): ``A(aC)`` renders SidebarTemplatesButton and
    the ``XS`` branch renders SidebarBottomPanelToggleButton. Returns True
    when rewritten.
    """
    # "SidebarTemplatesButton" survives: only its render call is removed.
    path = fp._locate(assets, "GraphView-*.js", "SidebarTemplatesButton")
    text = fp._text(path)
    if SIDE_BUTTONS_MARK in text:
        print(f"  side-buttons: already applied ({path.name})")
        return False
    text = fp._replace_once(text, SIDE_TEMPLATES_OLD, SIDE_TEMPLATES_NEW)
    text = fp._replace_once(text, SIDE_CONSOLE_OLD, SIDE_CONSOLE_NEW)
    fp._write(path, text)
    print(f"  side-buttons: applied (Templates + Console toggles removed) in {path.name}")
    return True


STARTUP_GATE_MARK = "/*ComfyFront_sep:startup-gate*/"


def _patch_startup_gate(fp, assets: Path) -> bool:
    """Gate the homepage loadGraphData behind the splash-fade + media-preload.

    The mock exposes ``window.__cdlHomepageGate`` (a promise resolved once the
    splash has faded and all readme_assets images are preloaded). Waiting on
    it inside resolveStartupOutcome means nodes mount exactly when the
    entrance choreography runs — markdown text and images render from cache
    in sync with the animation instead of popping in early. Returns True when
    rewritten.
    """
    path = fp._locate(assets, "GraphView-*.js", "ComfyFront_sep:startup-homepage*/")
    text = fp._text(path)
    if STARTUP_GATE_MARK in text:
        print(f"  startup-gate: already applied ({path.name})")
        return False
    old = "?(await X.loadGraphData(await fetch("
    new = ("?(await(window.__cdlHomepageGate||Promise.resolve()),"
           "await X.loadGraphData(await fetch(")
    text = fp._replace_once(text, old, new)
    text = fp._replace_once(
        text,
        "/*ComfyFront_sep:startup-homepage*/",
        "/*ComfyFront_sep:startup-homepage*/" + STARTUP_GATE_MARK,
    )
    fp._write(path, text)
    print(f"  startup-gate: applied (homepage load waits for splash+media) in {path.name}")
    return True


def build_extension_manifest() -> list:
    """Return the extension URL list exactly as server.py would build it."""
    extensions_dir = SEP / "extensions"
    urls = sorted(
        "/" + p.relative_to(SEP).as_posix()
        for p in extensions_dir.rglob("*.js")
    )
    if HOME_EXTENSION_URL not in urls:
        urls.append(HOME_EXTENSION_URL)
    return urls


def main() -> None:
    global HOMEPAGE_MINIFIED
    fp = load_frontend_patch()

    # The bundled SD graph constant is replaced by the homepage workflow,
    # embedded as a minified JSON literal (valid JS object literal).
    homepage_path = SEP / "homepage_workflow.json"
    homepage = json.loads(homepage_path.read_text(encoding="utf-8"))
    HOMEPAGE_MINIFIED = json.dumps(homepage, ensure_ascii=False, separators=(",", ":"))

    print("bake results:")
    results = bake_patches(fp)
    print(json.dumps(results, indent=2))

    manifest = build_extension_manifest()
    (SEP / "extensions-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"extensions-manifest.json: {len(manifest)} entries")


if __name__ == "__main__":
    main()
