/**
 * ComfyDL App Panel (cdlapp.js) - a DL-oriented companion overlay for the
 * official beta "App mode" (internally "linear mode").
 *
 * What it does:
 *  - Detects when the official App view is active (via the official
 *    coachmark DOM anchors) and mounts a left-side dock with three sections:
 *    OUTPUTS (downloadable files / copyable text / low-cost previews),
 *    PROGRESS (per-node progress rows driven by progress_state), and
 *    ERRORS (execution_error cards with collapsible tracebacks).
 *  - Header navigation: "Node graph" (back to the workflow), "Edit inputs" /
 *    "Edit outputs" (jump into the official builder at the matching step).
 *
 * Official-evolution containment (ADR-8): EVERY dependency on official
 * internals is declared once in OFFICIAL_SURFACE below.  Nothing else in this
 * file may hardcode official selectors, event names, command ids or store
 * access paths.  Capability probing degrades features individually - a
 * missing anchor hides one section and logs a warning, it never blanks the
 * app.  `cdl_smoke_tests/test_cdlapp.py` pins these contracts against the
 * installed frontend chunks: when the frontend package is upgraded, that test
 * fails loudly and tells you which entries to re-verify.
 *
 * Boot contract (2026-10-30 rule): waitFor comfyAPI -> settingsStoreReady
 * gate -> registerExtension.  Registering settings before the official
 * settings store finished loading aborts the store bootstrap and blanks the
 * UI (see profiler.js for the full post-mortem).
 *
 * In: nothing (script tag).  Out: DOM overlay + settings entry
 * ComfyDL.AppPanel.Enabled.  The panel is display-only: it consumes official
 * events and never emits progress or pbar calls.
 */
(function () {
  "use strict";

  const TAG = "[ComfyDL app-panel]";
  const PANEL_ID = "cdl-app-panel";
  const SETTING_ENABLED = "ComfyDL.AppPanel.Enabled";
  const SETTING_MOTION = "ComfyDL.AppPanel.Motion";
  const PREVIEW_LIMIT = 12; // thumbnail grid cap (newest first)

  // ---------------------------------------------------- OFFICIAL_SURFACE ----
  // The ONLY place allowed to know about official internals.  Each entry is
  // verified by test_cdlapp.py against the installed frontend chunks.
  const OFFICIAL_SURFACE = {
    frontendVersion: "1.51.9",
    // App-mode detection, two signals OR-ed:
    //  1. pinia appMode store flag (verbatim `isAppMode` from the official
    //     useAppMode() composable, re-exported by the `appMode` store) -
    //     primary, mode-exact.
    //  2. the official input panel's own widgets carry the verbatim testid
    //     `app-mode-widget-item` - DOM fallback for when the store probe
    //     fails.  (The official coachmark ids like `app-run-button` live in
    //     chunk constants but are NOT exposed as DOM attributes - verified
    //     2026-10-30 - so they are NOT usable as anchors.)
    detection: {
      appModeField: "viewMode", // pinia appMode store string field
      appModeValue: "app",
      builderPrefix: "builder:",
      pollMs: 400,
    },
    // Dock layout (2026-10-30 round 2, live-measured): the official app-mode
    // input panel is `.arrange-panel` (a PrimeVue splitterpanel with NATIVE
    // styling).  Instead of an overlapping overlay, body.cdlapp-active CSS
    // re-docks THAT panel into the left column (position:fixed removes it
    // from the splitter flow, so the center output area expands) and the CDL
    // sections are appended inside it - the user asked for the official
    // input area to live inside the CDL panel with the official look, and
    // this achieves both with zero Vue surgery.  When no arrange-panel
    // exists (workflow without app selections), fall back to the legacy
    // overlay panel, offset right of the sidebar rail.
    layout: {
      arrangePanel: ".arrange-panel",
      rail: ".side-tool-bar-container",
      comfyBody: ".comfyui-body", // app-mode anchor; the builder strips it
      // App-mode sidebar drawer (rail tab panels - assets, apps, ...):
      // live-verified 2026-10-31.  In app mode it renders as
      // .comfy-vue-side-bar-container inside the app splitter; the graph-mode
      // twin with the SAME class lives inside the hidden
      // .graph-canvas-container and must be ignored.
      appDrawer: ".comfy-vue-side-bar-container",
      graphCanvasContainer: ".graph-canvas-container",
      extrasId: "cdl-app-extras",
      fallbackLeft: 59, // side-tool-bar-container width on 1.51.9
    },
    // Events (api chunk, ComfyApi class).  Payload shapes are pinned in
    // test_cdlapp.py via the api chunk zod schemas.
    events: {
      progressState: "progress_state",
      executing: "executing",
      executed: "executed",
      executionError: "execution_error",
      executionSuccess: "execution_success",
      executionInterrupted: "execution_interrupted",
      executionCached: "execution_cached",
    },
    commands: {
      toggleLinear: "Comfy.ToggleLinear",
    },
    // z-index band (measured 1.51.9): official splash 9999, primevue
    // tooltip 2000, actionbar 1300.  Stay between tooltip and splash.
    zIndex: { min: 2001, max: 9999, panel: 2100 },
    // The workspace (extensionManager) store exposes command.execute -
    // the official command runner used by the hardcoded actionbar.
    // Verified in the workspaceStoreSetup object literal.
    extensionManagerKeys: ["command", "setting", "workflow"],
    // Builder entry (verified live 2026-10-30, chunk-verified 2026-10-31):
    // the appMode store itself has enterBuilder()/exitBuilder() (no args);
    // enterBuilder() lands on the arrange step when the app has outputs, or
    // the inputs step otherwise - it CANNOT be given a target step.  The
    // official useAppMode().setMode(m) is exactly `activeWorkflow.activeMode
    // = m` (read out of the minified chunk), so a specific step is reached
    // by assigning that field on the workspace store's activeWorkflow.
    // exitBuilder() hardcodes back to the GRAPH - there is no official
    // return-to-app entry, so the panel ships its own floating back button
    // (mountBuilderBack) which assigns the "app" value directly WITHOUT
    // exitBuilder()'s resetSelectedToWorkflow (that would discard the
    // selections the user just edited in the wizard).
    builder: {
      builderStoreProbe: "enterBuilder",
      appModeStoreProbe: "viewMode",
      workflowStoreProbe: "activeWorkflow",
      modeField: "activeMode",
      backValue: "app",
      backBtnId: "cdl-builder-back",
      steps: { inputs: "builder:inputs", outputs: "builder:outputs" },
    },
    // Official button recipe (verified live 2026-10-30 from the app-mode
    // header icon button): global tailwind utility classes from the frontend
    // bundle - NOT scoped, so injected DOM gets the exact official look.
    // Size classes are passed per-button (h-10 px-4 for nav, size-8 for
    // icons); our own :where(.cdlapp-btn...) CSS only acts as a zero-
    // specificity fallback if these classes ever drift.
    buttons: {
      secondary:
        "relative inline-flex items-center justify-center gap-2 cursor-pointer "
        + "touch-manipulation whitespace-nowrap appearance-none text-sm "
        + "font-medium font-inter transition-colors "
        + "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring "
        + "disabled:pointer-events-none disabled:opacity-50 rounded-lg "
        + "border border-solid border-border-default text-muted-foreground "
        + "hover:bg-secondary-background-hover hover:border-interface-stroke "
        + "hover:text-base-foreground",
    },
  };

  // --------------------------------------------------------------- i18n ----
  const I18N = {
    en: {
      panel_title: "CDL App",
      to_graph: "Node graph",
      edit_inputs: "Inputs",
      edit_outputs: "Outputs",
      collapse: "Collapse panel",
      expand: "Expand panel",
      out_title: "Outputs",
      out_empty: "Run the app to collect outputs here - files become downloads, text becomes copyable.",
      prev_title: "Previews",
      prev_empty: "Image outputs appear here as a thumbnail grid.",
      chart_idle: "Live vector charts appear here while a training / generation node streams data.",
      chart_best: "best",
      chart_step: "step",
      prev_open: "Open full size",
      splitter_hint: "Drag to resize the zones",
      locate_inputs: "Show the input widgets' nodes on the graph",
      locate_outputs: "Show the output nodes on the graph",
      out_copy: "Copy",
      out_copied: "Copied",
      out_download: "Download",
      out_image: "image",
      out_text: "text",
      prog_title: "Progress",
      prog_idle: "Idle - press Run.",
      prog_node_fmt: "{done} / {total} nodes · {pct}%",
      err_title: "Errors",
      err_none: "No errors.",
      err_copy: "Copy details",
      nav_unavailable: "Unavailable - official surface drifted (see console)",
      back_to_app: "Back to App",
      setting_enabled: "ComfyDL App Panel enabled",
      setting_enabled_tooltip: "Show the ComfyDL companion panel inside the official App mode.",
      setting_motion: "Panel animations",
      setting_motion_tooltip: "Enable panel entrance / feedback / transition animations.",
      cat_app: "App",
    },
    zh: {
      panel_title: "CDL App",
      to_graph: "节点图",
      edit_inputs: "输入",
      edit_outputs: "输出",
      collapse: "折叠面板",
      expand: "展开面板",
      out_title: "输出",
      out_empty: "运行后在收集输出：文件可下载，文本可复制。",
      prev_title: "预览",
      prev_empty: "图像输出会以缩略图网格呈现在这里。",
      chart_idle: "训练 / 生成节点上报实时数据时，矢量图表会出现在这里。",
      chart_best: "best",
      chart_step: "step",
      prev_open: "查看原图",
      splitter_hint: "拖动调整分区大小",
      locate_inputs: "在节点图中定位输入所在的节点",
      locate_outputs: "在节点图中定位输出所在的节点",
      out_copy: "复制",
      out_copied: "已复制",
      out_download: "下载",
      out_image: "图像",
      out_text: "文本",
      prog_title: "进度",
      prog_idle: "空闲——按 Run 开始。",
      prog_node_fmt: "{done} / {total} 节点 · {pct}%",
      err_title: "错误",
      err_none: "无错误。",
      err_copy: "复制详情",
      nav_unavailable: "不可用——官方界面漂移（见控制台）",
      back_to_app: "返回 App",
      setting_enabled: "启用 ComfyDL App 面板",
      setting_enabled_tooltip: "在官方 App 模式内显示 ComfyDL 伴侣面板。",
      setting_motion: "面板动效",
      setting_motion_tooltip: "启用面板的入场 / 操作反馈 / 切换动效。",
      cat_app: "App",
    },
  };

  // -------------------------------------------------------------- state ----
  const state = {
    locale: "en",
    enabled: true,
    motion: true,
    mounted: false,
    collapsed: false,
    appMode: false,
    caps: {
      anchorsSeen: false,
      commandExecute: false,
      piniaReached: false,
      builderStore: false,
      appModeStore: false,
      workflowStore: false,
    },
    warned: {},
    panel: null,
    extras: null,
    observer: null,
    handlers: null,
    builderStore: null,
    appModeStore: null,
    workflowStore: null,
    // run bookkeeping (per prompt_id)
    runs: new Map(), // prompt_id -> {nodes: Map(nodeKey -> row), order: []}
    activePrompt: null,
    // current-run scoping: only outputs/files produced by the ACTIVE run are
    // shown (the artifacts route lists the whole output directory - showing
    // it raw mixed in every historical file, which the user called out).
    runActive: false,
    runStartedAt: 0, // ms epoch of the current run's first event
    errors: [], // {node_id, node_type, title, message, etype, traceback}
    outputs: [], // {kind:'image'|'text', name, url?, text?}
    artifacts: [], // {name, path, size, mtime}
    charts: new Map(), // node_id -> live cdl_preview_data payload (batch 18)
  };

  const getApp = () => (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) || null;
  const getApi = () => (window.comfyAPI && window.comfyAPI.api && window.comfyAPI.api.api) || null;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitFor(fn, limit = 30000, step = 200) {
    const start = Date.now();
    while (Date.now() - start < limit) {
      try { if (fn()) return true; } catch (e) { /* retry */ }
      await sleep(step);
    }
    return false;
  }

  // The 2026-10-30 gate: registering settings before the official settings
  // store finished loading aborts the store bootstrap and blanks the UI.
  function settingsStoreReady(_app) {
    // Live-verified 2026-10-30: the workspace-store `setting` field is NOT
    // the pinia settings store (it has no isReady at all - the old check
    // made every session time out).  The real store hangs off the Vue app's
    // pinia instance and is created during the store bootstrap that precedes
    // the Vue mount, so this passes as soon as the app view exists.
    try {
      const root = document.getElementById("vue-app");
      const vueApp = root && root.__vue_app__;
      const pinia = vueApp && vueApp.config &&
        vueApp.config.globalProperties.$pinia;
      const st = pinia && pinia._s && pinia._s.get("setting");
      return !!st && st.isReady === true;
    } catch (e) {
      return false;
    }
  }

  function t(key, params) {
    const dict = I18N[state.locale] || I18N.en;
    let s = dict[key] !== undefined ? dict[key] : I18N.en[key] || key;
    if (params) {
      Object.keys(params).forEach((k) => {
        s = s.replace("{" + k + "}", String(params[k]));
      });
    }
    return s;
  }

  function warnOnce(key, msg) {
    if (state.warned[key]) return;
    state.warned[key] = true;
    console.warn(TAG, msg);
  }

  // ------------------------------------------------- capability probing ----
  // Everything here degrades softly: a miss only disables one feature and
  // logs - the panel itself must never break the app view.
  //
  // TIMING NOTE (2026-10-30 real-session bug): extension setup() runs BEFORE
  // the Vue app mounts, so a one-shot pinia probe always misses (no
  // #vue-app.__vue_app__ yet) and the panel silently never appeared.  The
  // store probe is therefore split into probeStores() and re-run lazily from
  // the detection poll tick until the stores are actually found - once Vue
  // mounts, the next tick (<=400ms) self-heals.
  function probeCapabilities() {
    const app = getApp();
    const em = app && app.extensionManager;
    // command runner (used by the official actionbar itself)
    state.caps.commandExecute = !!(em && em.command && typeof em.command.execute === "function");
    if (!state.caps.commandExecute) {
      warnOnce("cmd", "extensionManager.command.execute not found - Node-graph button disabled");
    }
  }

  // Pinia store probe: needs the Vue app to be mounted, so it is retried
  // lazily from the poll tick until both stores are found (see TIMING NOTE).
  function probeStores() {
    if (state.appModeStore && state.builderStore && state.workflowStore) return;
    try {
      const root = document.getElementById("vue-app");
      const vueApp = root && root.__vue_app__;
      const pinia = vueApp && vueApp.config && vueApp.config.globalProperties &&
        vueApp.config.globalProperties.$pinia;
      const wfCandidates = []; // verbose: every store carrying activeWorkflow
      if (pinia && pinia._s && typeof pinia._s.forEach === "function") {
        state.caps.piniaReached = true;
        const B = OFFICIAL_SURFACE.builder;
        pinia._s.forEach((store) => {
          if (!store) return;
          if (!state.builderStore && typeof store[B.builderStoreProbe] === "function") {
            state.builderStore = store;
          }
          if (!state.appModeStore
              && typeof store[B.appModeStoreProbe] === "string"
              && typeof store.enterBuilder === "function") {
            state.appModeStore = store;
          }
          // workspace store: the one carrying the active workflow object -
          // the raw mode lives on it (see OFFICIAL_SURFACE.builder).
          if (store[B.workflowStoreProbe]) {
            wfCandidates.push(store.$id || "?");
            if (!state.workflowStore) state.workflowStore = store;
          }
        });
        state.caps.builderStore = !!state.builderStore;
        state.caps.appModeStore = !!state.appModeStore;
        state.caps.workflowStore = !!state.workflowStore;
        if (state.caps.appModeStore && state.caps.builderStore) {
          console.info(TAG, "official stores probed OK (builder + appMode"
            + (state.caps.workflowStore ? " + workflow" : "") + ")",
            {
              builderStore: state.builderStore && state.builderStore.$id,
              appModeStore: state.appModeStore && state.appModeStore.$id,
              workflowStore: state.workflowStore && state.workflowStore.$id,
              workflowCandidates: wfCandidates,
              hasActiveWorkflow: !!(state.workflowStore
                && state.workflowStore[B.workflowStoreProbe]),
            });
        }
      }
    } catch (e) { /* degrade - retried on the next tick */ }
    if (!state.caps.builderStore) {
      warnOnce("builder",
        "builder store not reachable yet - Inputs/Outputs edit buttons disabled until then");
    }
  }

  // The raw mode string lives on the active workflow object; the official
  // useAppMode().setMode(m) is exactly `activeWorkflow.activeMode = m`
  // (chunk-verified 2026-10-31).  Reading/writing it through the workspace
  // store is the official-equivalent path - no Vue internals are touched.
  function rawMode() {
    try {
      const B = OFFICIAL_SURFACE.builder;
      const wf = state.workflowStore && state.workflowStore[B.workflowStoreProbe];
      return wf ? wf[B.modeField] : null;
    } catch (e) { return null; }
  }

  function setMode(v) {
    try {
      const B = OFFICIAL_SURFACE.builder;
      const wf = state.workflowStore && state.workflowStore[B.workflowStoreProbe];
      if (!wf) {
        console.warn(TAG, "setMode(" + v + ") skipped - no activeWorkflow on",
          state.workflowStore ? state.workflowStore.$id : "(no workflow store)");
        return false;
      }
      wf[B.modeField] = v;
      console.info(TAG, "setMode(" + v + ") -> rawMode now:", rawMode());
      return true;
    } catch (e) {
      console.warn(TAG, "setMode failed:", e);
      return false;
    }
  }

  // -------------------------------------------------- app-mode detection ----
  // Primary and only signal: the pinia appMode store's mode string field
  // (see OFFICIAL_SURFACE.detection) - values 'graph' | app | builder:*,
  // verified live 2026-10-30 (the DOM testid fallback was disproven and
  // removed).  The poll is a 400ms string read - negligible, and immune to
  // how the official view swaps DOM.
  function isAppMode() {
    if (!state.appModeStore) return false;
    try {
      const v = state.appModeStore[OFFICIAL_SURFACE.detection.appModeField];
      return v === OFFICIAL_SURFACE.detection.appModeValue
        || (typeof v === "string" && v.indexOf(OFFICIAL_SURFACE.detection.builderPrefix) === 0);
    } catch (e) { return false; }
  }

  // The builder wizard also counts as "app family", but it is a FULL-SCREEN
  // official editor with its own layout.  Live-verified 2026-10-30 (user
  // reported the buggy builder): entering it KEEPS the mode string "app" but
  // REMOVES the .comfyui-body / side tool bar anchors from the DOM - so the
  // mode string alone cannot detect it.  The dock therefore requires BOTH
  // the pure "app" mode string AND the app-mode DOM anchors; when the
  // builder takes over, the anchors vanish and the panel unmounts (returning
  // to app re-docks automatically via the poll tick).
  function isPureAppMode() {
    if (!state.appModeStore) return false;
    try {
      const v = state.appModeStore[OFFICIAL_SURFACE.detection.appModeField];
      if (v !== OFFICIAL_SURFACE.detection.appModeValue) return false;
      // DOM anchor guard: the builder strips these from the body
      return !!document.querySelector(OFFICIAL_SURFACE.layout.comfyBody)
        && !!document.querySelector(OFFICIAL_SURFACE.layout.rail);
    } catch (e) { return false; }
  }

  // Builder-wizard detection (2026-10-31 live-verified): the wizard spans
  // THREE mode strings, and the derived official mode field only reports
  // "app" for the arrange step - the inputs/outputs steps derive to the
  // graph value (isAppModeValue only accepts "app" and "builder:arrange").
  // So the previous now && !dock test went blind exactly on the two steps
  // our nav buttons jump to.  Primary signal: the raw mode string from the
  // active workflow; fallback: "app" value with the DOM anchors stripped.
  function isBuilderMode() {
    const m = rawMode();
    if (typeof m === "string") {
      return m.indexOf(OFFICIAL_SURFACE.detection.builderPrefix) === 0;
    }
    return isAppMode()
      && !(document.querySelector(OFFICIAL_SURFACE.layout.comfyBody)
        && document.querySelector(OFFICIAL_SURFACE.layout.rail));
  }

  // True while an app-mode sidebar drawer (rail tab panel) is open.  The
  // graph-mode DOM keeps a twin instance of the same class hidden inside
  // .graph-canvas-container (display:none in app mode) - only a VISIBLE
  // instance outside that container counts.  While open, the dock unmounts
  // entirely: the official splitter REMOUNTS on every drawer toggle, and
  // the drawer renders in normal flow under the fixed full-bleed dock, so
  // the only clean option is to give the space back to the official layout.
  function drawerOpen() {
    try {
      const drawers = document.querySelectorAll(OFFICIAL_SURFACE.layout.appDrawer);
      for (const d of drawers) {
        if (d.closest(OFFICIAL_SURFACE.layout.graphCanvasContainer)) continue;
        if (d.offsetWidth > 0) return true;
      }
    } catch (e) { /* drawer probe optional */ }
    return false;
  }

  let pollTimer = null;
  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      // Lazy (re-)probe: extension setup ran before the Vue app mounted, so
      // the pinia stores only become reachable on some later tick (<=400ms).
      // NOTE: the workflow store must be re-probed too - it only carries
      // activeWorkflow once a workflow is open, which can be long after the
      // appMode/builder stores appeared (2026-10-31 live bug: stopping the
      // probe at builder+appMode left the workflow store null FOREVER, so
      // setMode() silently no-op'd and the builder step jump never worked).
      if (!state.appModeStore || !state.builderStore || !state.workflowStore) {
        probeStores();
      }
      // Re-append the docked sections if a Vue re-render dropped them.
      if (state.extras && state.mounted
          && !document.getElementById(OFFICIAL_SURFACE.layout.extrasId)) {
        const col = extrasColumn();
        if (col) col.appendChild(state.extras);
      }
      // Re-mount the drag handles if the official re-render dropped them.
      if (state.extras && state.mounted) mountSplitters();
      const now = isAppMode();
      if (now) state.caps.anchorsSeen = true;
      // dock ONLY in the pure app view - the builder wizard must stay
      // pristine, and so must the official layout while a sidebar drawer is
      // open (live-verified 2026-10-31: the app-mode drawer renders UNDER
      // the fixed full-bleed dock, and the official splitter REMOUNTS on
      // every drawer toggle - so the dock gets out of the way entirely).
      const dock = isPureAppMode() && !drawerOpen();
      if (dock !== state.appMode) {
        console.info(TAG, "poll: dock " + state.appMode + " -> " + dock
          + " (rawMode=" + rawMode() + " drawerOpen=" + drawerOpen() + ")");
        state.appMode = dock;
        if (dock) mountPanel();
        else unmountPanel();
      } else if (dock && state.mounted && !state.extras && extrasColumn()) {
        // legacy overlay can upgrade to the native dock when app selections
        // appear while already in app mode (the official input panel only
        // exists once selections are configured)
        unmountPanel();
        mountPanel();
      }
      // Builder wizard active: provide the return-to-app entry the official
      // builder lacks (exitBuilder hardcodes to the graph).
      const inBuilder = isBuilderMode();
      if (inBuilder && state.workflowStore) mountBuilderBack();
      else unmountBuilderBack();
    }, OFFICIAL_SURFACE.detection.pollMs);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // ------------------------------------------------------ nav actions ----
  function goNodeGraph() {
    try {
      const em = getApp() && getApp().extensionManager;
      if (state.caps.commandExecute && em) {
        em.command.execute(OFFICIAL_SURFACE.commands.toggleLinear);
        return;
      }
    } catch (e) { /* fallthrough */ }
    warnOnce("cmd", "cannot execute " + OFFICIAL_SURFACE.commands.toggleLinear);
  }

  function goBuilderStep(step) {
    if (!state.builderStore) {
      warnOnce("builder", "builder entry unavailable");
      return;
    }
    const target = OFFICIAL_SURFACE.builder.steps[step];
    if (!target) return;
    try {
      // enterBuilder() runs the official entry (resets stale selections,
      // closes the sidebar tab) but always lands on the arrange step (app
      // with outputs) or the inputs step - so the requested step is forced
      // afterwards via the official setMode path (see OFFICIAL_SURFACE).
      console.info(TAG, "goBuilderStep(" + step + ") target=" + target
        + " rawMode(before)=" + rawMode());
      state.builderStore.enterBuilder();
      console.info(TAG, "enterBuilder() done, rawMode=" + rawMode()
        + " anchors: comfyBody=" + !!document.querySelector(
          OFFICIAL_SURFACE.layout.comfyBody)
        + " rail=" + !!document.querySelector(OFFICIAL_SURFACE.layout.rail));
      if (rawMode() !== target) setMode(target);
      console.info(TAG, "goBuilderStep(" + step + ") final rawMode=" + rawMode());
    } catch (e) {
      console.warn(TAG, "enterBuilder failed:", e);
    }
  }

  // Floating "Back to App" button, mounted ONLY while the builder wizard is
  // active (user callout 2026-10-31: the official builder offers no way back
  // to the app view - exitBuilder() hardcodes to the workflow graph).  It
  // assigns the app value directly, deliberately NOT via exitBuilder(),
  // whose resetSelectedToWorkflow() would discard the selections the user
  // just edited in the wizard.
  function mountBuilderBack() {
    if (document.getElementById(OFFICIAL_SURFACE.builder.backBtnId)) return;
    console.info(TAG, "mounting builder back-to-app button (rawMode="
      + rawMode() + ")");
    const b = el("button", btnCls("h-10 px-4", "cdlapp-builder-back"));
    b.id = OFFICIAL_SURFACE.builder.backBtnId;
    b.appendChild(iconEl("log-in"));
    b.appendChild(el("span", null, t("back_to_app")));
    b.title = t("back_to_app");
    b.addEventListener("click", () => {
      if (!setMode(OFFICIAL_SURFACE.builder.backValue)) {
        warnOnce("back", "cannot return to the app view - workflow store unreachable");
      }
    });
    document.body.appendChild(b);
  }

  function unmountBuilderBack() {
    const n = document.getElementById(OFFICIAL_SURFACE.builder.backBtnId);
    if (n) n.remove();
  }

  // Jump back to the node graph AND select the nodes that carry the app-mode
  // input widgets / output slots (user request 2026-10-30: "jump to the
  // workflow and select the slots/nodes/widgets that belong to input or
  // output").  Selection shapes (live-verified): selectedInputs entries are
  // "selectionUuid:nodeId:widgetNames", selectedOutputs are bare node ids.
  function locateOnGraph(kind) {
    const appMode = state.appModeStore;
    if (!appMode) {
      warnOnce("locate", "appMode store unavailable");
      return;
    }
    const entries = kind === "inputs"
      ? (appMode.selectedInputs || [])
      : (appMode.selectedOutputs || []);
    const ids = entries.map((s) => {
      const str = String(s);
      if (kind === "inputs") {
        const parts = str.split(":");
        return parts.length >= 2 ? parts[1] : str;
      }
      return str;
    }).filter((id) => {
      const app = getApp();
      return !!(app && app.graph && app.graph.getNodeById
        && app.graph.getNodeById(Number(id)));
    });
    if (!ids.length) {
      warnOnce("locate", "no " + kind + " nodes to locate");
      return;
    }
    goNodeGraph(); // leave app mode - the graph is where selection happens
    // wait for the graph mode switch to settle, then select + center
    waitFor(() => !isAppMode(), 3000, 100).then(() => {
      setTimeout(() => {
        try {
          const app = getApp();
          const canvas = app && app.canvas;
          const graph = app && app.graph;
          if (!canvas || !graph) return;
          const nodes = ids.map((id) => graph.getNodeById(Number(id)))
            .filter(Boolean);
          canvas.selectNodes([]); // clear any current selection first
          // selectNode takes the NODE OBJECT (an id throws - live-verified)
          nodes.forEach((n) => canvas.selectNode(n, true));
          if (nodes[0]) canvas.centerOnNode(nodes[0]);
        } catch (e) {
          console.warn(TAG, "locate failed:", e);
        }
      }, 150);
    });
  }

  // --------------------------------------------------------- panel DOM ----
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  // Official-look button: the official tailwind recipe (OFFICIAL_SURFACE,
  // ADR-8 single source) + a per-button size class.  Our :where(.cdlapp-btn)
  // CSS is only a zero-specificity fallback for if the classes drift.
  function btnCls(size, extra) {
    return "cdlapp-btn " + OFFICIAL_SURFACE.buttons.secondary + " " + size
      + (extra ? " " + extra : "");
  }

  // Vector icons (lucide paths, inlined - self-contained, no external
  // requests).  The emoji glyphs they replace looked off next to the
  // official vector UI (user callout).
  const ICONS = {
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
      + '<polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
      + '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    image: '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>'
      + '<circle cx="9" cy="9" r="2"/>'
      + '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
      + '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/>'
      + '<path d="M16 13H8"/><path d="M16 17H8"/>',
    "chevrons-left": '<path d="m11 17-5-5 5-5"/><path d="m18 17-5-5 5-5"/>',
    "log-in": '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>'
      + '<polyline points="10 17 15 12 10 7"/><line x1="15" x2="3" y1="12" y2="3"/>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>'
      + '<polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/>',
  };

  function iconEl(name, cls) {
    const s = el("span", "cdlapp-icon" + (cls ? " " + cls : ""));
    s.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
      + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
      + 'aria-hidden="true">' + (ICONS[name] || "") + "</svg>";
    return s;
  }

  function buildNav() {
    const nav = el("div", "cdlapp-nav");
    const mkBtn = (key, label, title_, fn, size, extra, icon) => {
      const b = el("button", btnCls(size || "h-10 px-4", extra));
      if (label) {
        b.textContent = label;
      } else {
        b.appendChild(iconEl(icon || "chevrons-left"));
      }
      b.title = title_;
      b.addEventListener("click", fn);
      b.dataset.nav = key;
      return b;
    };
    nav.appendChild(mkBtn("graph", t("to_graph"), t("to_graph"), goNodeGraph,
      null, "cdlapp-btn-primary"));
    nav.appendChild(mkBtn("inputs", t("edit_inputs"), t("edit_inputs"),
      () => goBuilderStep("inputs")));
    nav.appendChild(mkBtn("outputs", t("edit_outputs"), t("edit_outputs"),
      () => goBuilderStep("outputs")));
    nav.appendChild(mkBtn("locate-inputs", null, t("locate_inputs"),
      () => locateOnGraph("inputs"), "size-8", "cdlapp-btn-icon", "log-in"));
    nav.appendChild(mkBtn("locate-outputs", null, t("locate_outputs"),
      () => locateOnGraph("outputs"), "size-8", "cdlapp-btn-icon", "log-out"));
    nav.appendChild(mkBtn("collapse", null, t("collapse"),
      () => setCollapsed(true), "size-8", "cdlapp-btn-icon"));
    return nav;
  }

  function buildPanel() {
    const panel = el("div");
    panel.id = PANEL_ID;

    // header: title + nav buttons + collapse
    const head = el("div", "cdlapp-head");
    const title = el("div", "cdlapp-title", t("panel_title"));
    head.appendChild(title);
    head.appendChild(buildNav());
    panel.appendChild(head);

    // collapsed rail (hidden until collapsed): just an expand button
    const rail = el("button", "cdlapp-rail", "»");
    rail.title = t("expand");
    rail.addEventListener("click", () => setCollapsed(false));
    panel.appendChild(rail);

    // outputs section
    const secOut = el("div", "cdlapp-section cdlapp-sec-outputs");
    secOut.appendChild(el("div", "cdlapp-sec-title", t("out_title")));
    secOut.appendChild(el("div", "cdlapp-out-list"));
    secOut.appendChild(el("div", "cdlapp-empty", t("out_empty")));
    panel.appendChild(secOut);

    // progress section
    const secProg = el("div", "cdlapp-section cdlapp-sec-progress");
    secProg.appendChild(el("div", "cdlapp-sec-title", t("prog_title")));
    const totalWrap = el("div", "cdlapp-total");
    const totalBar = el("div", "cdlapp-total-bar");
    totalWrap.appendChild(totalBar);
    secProg.appendChild(totalWrap);
    secProg.appendChild(el("div", "cdlapp-total-label", t("prog_idle")));
    secProg.appendChild(el("div", "cdlapp-prog-rows"));
    panel.appendChild(secProg);

    // errors section
    const secErr = el("div", "cdlapp-section cdlapp-sec-errors");
    secErr.appendChild(el("div", "cdlapp-sec-title", t("err_title")));
    secErr.appendChild(el("div", "cdlapp-err-list"));
    secErr.appendChild(el("div", "cdlapp-empty", t("err_none")));
    panel.appendChild(secErr);

    applyCapabilities(panel);
    return panel;
  }

  function applyCapabilities(panel) {
    // Hide features whose official surface drifted (ADR-8: soft degradation).
    if (!state.caps.builderStore) {
      panel.querySelectorAll('[data-nav="inputs"],[data-nav="outputs"]').forEach((b) => {
        b.disabled = true;
        b.title = t("nav_unavailable");
        b.classList.add("cdlapp-btn-disabled");
      });
    }
  }

  function setCollapsed(v) {
    state.collapsed = v;
    if (state.panel) state.panel.classList.toggle("cdlapp-collapsed", v);
  }

  // The CDL sections that ride INSIDE the official input panel (native look
  // guaranteed - it is the official element, we only append a sibling).
  function buildExtras() {
    const extras = el("aside", "cdlapp-extras");
    extras.id = OFFICIAL_SURFACE.layout.extrasId;
    const head = el("div", "cdlapp-head");
    head.appendChild(el("div", "cdlapp-title", t("panel_title")));
    head.appendChild(buildNav());
    extras.appendChild(head);

    // outputs section
    const secOut = el("div", "cdlapp-section cdlapp-sec-outputs");
    secOut.appendChild(el("div", "cdlapp-sec-title", t("out_title")));
    secOut.appendChild(el("div", "cdlapp-out-list"));
    secOut.appendChild(el("div", "cdlapp-empty", t("out_empty")));
    extras.appendChild(secOut);

    // previews section (image artifacts / image outputs as thumbnails)
    const secPrev = el("div", "cdlapp-section cdlapp-sec-previews");
    secPrev.appendChild(el("div", "cdlapp-sec-title", t("prev_title")));
    secPrev.appendChild(el("div", "cdlapp-chart-host"));
    secPrev.appendChild(el("div", "cdlapp-preview-grid"));
    secPrev.appendChild(el("div", "cdlapp-empty", t("prev_empty")));
    extras.appendChild(secPrev);

    // progress section
    const secProg = el("div", "cdlapp-section cdlapp-sec-progress");
    secProg.appendChild(el("div", "cdlapp-sec-title", t("prog_title")));
    const totalWrap = el("div", "cdlapp-total");
    totalWrap.appendChild(el("div", "cdlapp-total-bar"));
    secProg.appendChild(totalWrap);
    secProg.appendChild(el("div", "cdlapp-total-label", t("prog_idle")));
    secProg.appendChild(el("div", "cdlapp-prog-rows"));
    extras.appendChild(secProg);

    // errors section
    const secErr = el("div", "cdlapp-section cdlapp-sec-errors");
    secErr.appendChild(el("div", "cdlapp-sec-title", t("err_title")));
    secErr.appendChild(el("div", "cdlapp-err-list"));
    secErr.appendChild(el("div", "cdlapp-empty", t("err_none")));
    extras.appendChild(secErr);

    applyCapabilities(extras);
    return extras;
  }

  function dockLeft() {
    const rail = document.querySelector(OFFICIAL_SURFACE.layout.rail);
    const w = rail ? Math.round(rail.getBoundingClientRect().width) : 0;
    const out = w || OFFICIAL_SURFACE.layout.fallbackLeft;
    if (out !== dockLeft.last) {
      dockLeft.last = out;
      console.info(TAG, "dockLeft ->", out, { rail: w });
    }
    return out;
  }

  function extrasColumn() {
    // .arrange-panel > div(flex h-full flex-col) > div(inputs scroll + run)
    const host = document.querySelector(OFFICIAL_SURFACE.layout.arrangePanel);
    if (!host) return null;
    return host.querySelector(":scope > div > div") || host;
  }

  // ------------------------------------------ draggable zone splitters ----
  // Batch 9: the zone boundaries are 9px grid gutter tracks (see cdlapp.css);
  // a splitter handle rides in each gutter and lets the user resize the two
  // columns by dragging.  Sizes persist in localStorage and are restored on
  // the next mount.  Purely cosmetic - no official state is touched.
  const SPLIT_PARAMS_KEY = "ComfyDL.AppPanel.ParamsW";
  const SPLIT_PREV_KEY = "ComfyDL.AppPanel.PrevW";

  function makeSplitter(onDrag) {
    const s = el("div", "cdlapp-splitter");
    s.title = t("splitter_hint");
    s.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      try {
        s.setPointerCapture(e.pointerId);
      } catch (err) {
        /* synthetic pointer (tests / automation) - plain listeners still work */
      }
      s.classList.add("cdlapp-dragging");
      const move = (ev) => onDrag(ev.clientX);
      const up = () => {
        s.classList.remove("cdlapp-dragging");
        s.removeEventListener("pointermove", move);
        s.removeEventListener("pointerup", up);
        s.removeEventListener("pointercancel", up);
      };
      s.addEventListener("pointermove", move);
      s.addEventListener("pointerup", up);
      s.addEventListener("pointercancel", up);
    });
    return s;
  }

  function mountSplitters() {
    if (!state.extras || !document.getElementById(OFFICIAL_SURFACE.layout.extrasId)) {
      return;
    }
    // gutter 1: official params column <-> CDL dashboard (px, 260-560)
    const bodyCol = extrasColumn();
    if (bodyCol && !bodyCol.querySelector(":scope > .cdlapp-splitter")) {
      bodyCol.style.position = "relative";
      const saved = parseInt(localStorage.getItem(SPLIT_PARAMS_KEY) || "", 10);
      if (!isNaN(saved)) {
        bodyCol.style.setProperty("--cdl-params-w",
          Math.min(560, Math.max(260, saved)) + "px");
      }
      bodyCol.appendChild(makeSplitter((x) => {
        const w = Math.round(Math.min(560, Math.max(260,
          x - bodyCol.getBoundingClientRect().left)));
        bodyCol.style.setProperty("--cdl-params-w", w + "px");
        localStorage.setItem(SPLIT_PARAMS_KEY, String(w));
      }));
    }
    // gutter 2: previews hero tile <-> outputs column (%, 30-75)
    if (!state.extras.querySelector(":scope > .cdlapp-splitter")) {
      const saved = parseFloat(localStorage.getItem(SPLIT_PREV_KEY) || "");
      if (!isNaN(saved)) {
        state.extras.style.setProperty("--cdl-prev-w",
          Math.min(75, Math.max(30, saved)).toFixed(1) + "%");
      }
      state.extras.appendChild(makeSplitter((x) => {
        const b = state.extras.getBoundingClientRect();
        const pct = Math.min(75, Math.max(30,
          ((x - b.left) / b.width) * 100));
        const v = pct.toFixed(1) + "%";
        state.extras.style.setProperty("--cdl-prev-w", v);
        localStorage.setItem(SPLIT_PREV_KEY, v);
      }));
    }
  }

  function mountPanel() {
    if (state.mounted || !state.enabled) return;
    if (!state.caps.anchorsSeen) probeCapabilities();
    document.documentElement.style.setProperty(
      "--cdl-dock-left", dockLeft() + "px");
    const col = extrasColumn();
    if (col) {
      // native dock: the official input panel is re-positioned by CSS into
      // the left column; our sections ride inside it (see body.cdlapp-active).
      state.extras = buildExtras();
      col.appendChild(state.extras);
      mountSplitters();
      state.panel = state.extras; // render* roots
      document.body.classList.add("cdlapp-active");
      console.log(TAG, "docked into the official input panel (app mode)");
    } else {
      // legacy fallback: workflow without app selections - overlay panel,
      // offset right of the sidebar rail so nothing is covered.
      state.panel = buildPanel();
      document.body.appendChild(state.panel);
      console.log(TAG, "panel mounted (legacy overlay: no app selections)");
    }
    state.mounted = true;
    renderOutputs();
    renderProgress();
    renderErrors();
    renderCharts();
  }

  function unmountPanel() {
    if (!state.mounted) return;
    const leaving = state.motion
      && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (state.extras) {
      const ex = state.extras;
      state.extras = null;
      if (leaving) {
        ex.classList.add("cdlapp-leaving");
        setTimeout(() => ex.remove(), 180);
      } else {
        ex.remove();
      }
    }
    document.body.classList.remove("cdlapp-active");
    if (state.panel && state.panel.parentNode) {
      const p = state.panel;
      state.panel = null;
      if (leaving) {
        p.classList.add("cdlapp-leaving");
        setTimeout(() => { if (p.parentNode) p.remove(); }, 180);
      } else {
        p.parentNode.removeChild(p);
      }
    }
    state.panel = null;
    state.mounted = false;
    console.log(TAG, "panel unmounted (left app mode)");
  }

  // ------------------------------------------------------ render: stubs ----
  // (filled by the progress / outputs phases below - each render function is
  // idempotent and only rebuilds its own section.)
  function renderOutputs() {
    if (!state.panel) return;
    const list = state.panel.querySelector(".cdlapp-out-list");
    const empty = state.panel.querySelector(".cdlapp-sec-outputs .cdlapp-empty");
    if (!list) return;
    renderPreviews();
    list.textContent = "";
    // Only files written by the CURRENT run (mtime >= run start; the route
    // reports seconds) join the entries the run itself emitted.
    const runArts = currentRunArtifacts();
    const items = state.outputs.length + runArts.length;
    if (empty) empty.style.display = items ? "none" : "";
    if (!items) return;
    // artifacts first (files), then executed text/image entries.
    // Text outputs that reference a written file (SaveText / CdlWriteCSV /
    // CdlModelSave return the file path as a STRING output) graduate from
    // "copyable text" to "downloadable file" via an artifact lookup.
    const byKey = new Map();
    runArts.forEach((a) => {
      byKey.set(a.path, a);
      byKey.set(a.name, a);
    });
    runArts.forEach((a) => {
      list.appendChild(artifactRow(a));
    });
    state.outputs.forEach((o) => {
      let artifact = null;
      if (o.kind === "text" && o.text) {
        const first = o.text.split("\n")[0].trim()
          .replace(/^Saved state_dict to:\s*/i, "").trim();
        artifact = byKey.get(first) || null;
      }
      list.appendChild(outputRow(o, artifact));
    });
  }

  // Artifacts written since the current run began (mtime is seconds; a small
  // slack covers the fs/server timestamp skew).  Empty when no run started
  // in this session - the raw directory listing must NOT leak in.
  function currentRunArtifacts() {
    if (!state.runStartedAt) return [];
    return state.artifacts.filter((a) => a.mtime
      && a.mtime * 1000 >= state.runStartedAt - 5000);
  }

  // Image artifacts and image outputs become a clickable thumbnail grid.
  // Artifacts resolve through the official /view endpoint (type=output,
  // subfolder split from the output-relative posix path); executed image
  // entries already carry their /view url.
  function renderPreviews() {
    if (!state.panel) return;
    const grid = state.panel.querySelector(".cdlapp-preview-grid");
    const empty = state.panel.querySelector(".cdlapp-sec-previews .cdlapp-empty");
    if (!grid) return;
    grid.textContent = "";
    const cells = [];
    currentRunArtifacts().forEach((a) => {
      if (!a.media_type || !a.media_type.startsWith("image/")) return;
      const p = a.path || a.name;
      const i = p.lastIndexOf("/");
      const sub = i >= 0 ? p.slice(0, i) : "";
      const file = i >= 0 ? p.slice(i + 1) : p;
      cells.push({
        name: a.name,
        url: "/view?type=output&subfolder=" + encodeURIComponent(sub)
          + "&filename=" + encodeURIComponent(file),
      });
    });
    state.outputs.forEach((o) => {
      if (o.kind === "image" && o.url) cells.push({ name: o.name, url: o.url });
    });
    const shown = cells.slice(0, PREVIEW_LIMIT);
    if (empty) empty.style.display = shown.length ? "none" : "";
    shown.forEach((c) => grid.appendChild(previewCell(c)));
  }

  function previewCell(c) {
    const fig = el("figure", "cdlapp-preview-item");
    const img = el("img", "cdlapp-preview-img");
    img.loading = "lazy";
    img.alt = c.name;
    img.src = c.url;
    fig.appendChild(img);
    const cap = el("figcaption", "cdlapp-preview-cap", c.name);
    cap.title = c.name;
    fig.appendChild(cap);
    fig.title = t("prev_open");
    fig.addEventListener("click", () => {
      window.open(c.url, "_blank", "noopener");
    });
    return fig;
  }

  function artifactRow(a) {
    const row = el("div", "cdlapp-item");
    const icon = iconEl("file-text", "cdlapp-item-icon");
    icon.title = a.path || a.name;
    const name = el("span", "cdlapp-item-name", a.name);
    name.title = (a.path || a.name) + " · " + humanSize(a.size);
    const size = el("span", "cdlapp-item-size", humanSize(a.size));
    const dl = el("a", btnCls("size-8", "cdlapp-btn-icon cdlapp-dl"));
    dl.appendChild(iconEl("download"));
    dl.title = t("out_download");
    dl.href = "/comfydl/artifacts/download?name=" + encodeURIComponent(a.path || a.name);
    dl.setAttribute("download", a.name);
    row.appendChild(icon);
    row.appendChild(name);
    row.appendChild(size);
    row.appendChild(dl);
    return row;
  }

  function outputRow(o, artifact) {
    const row = el("div", "cdlapp-item");
    if (o.kind === "image") {
      const icon = iconEl("image", "cdlapp-item-icon");
      icon.title = t("out_image");
      const name = el("span", "cdlapp-item-name", o.name);
      const open = el("a", btnCls("size-8", "cdlapp-btn-icon"));
      open.appendChild(iconEl("download"));
      open.title = t("out_download");
      open.href = o.url;
      open.setAttribute("download", o.name);
      row.appendChild(icon);
      row.appendChild(name);
      row.appendChild(open);
      return row;
    }
    // text: copyable (plus download when it names a real artifact)
    const icon = iconEl("file-text", "cdlapp-item-icon");
    icon.title = artifact ? (artifact.path || artifact.name) : t("out_text");
    const name = el("span", "cdlapp-item-name", (o.text || "").slice(0, 80));
    name.title = o.text || "";
    const cp = el("button", btnCls("size-8", "cdlapp-btn-icon"));
    cp.appendChild(iconEl("copy"));
    cp.title = t("out_copy");
    cp.addEventListener("click", () => {
      navigator.clipboard.writeText(o.text || "").then(() => {
        const ok = iconEl("check");
        ok.classList.add("cdlapp-pop"); // feedback pop (motion switch gated)
        cp.replaceChildren(ok);
        cp.title = t("out_copied");
        setTimeout(() => {
          cp.replaceChildren(iconEl("copy"));
          cp.title = t("out_copy");
        }, 1200);
      }).catch(() => { /* clipboard unavailable */ });
    });
    row.appendChild(icon);
    row.appendChild(name);
    row.appendChild(cp);
    if (artifact) {
      const dl = el("a", btnCls("size-8", "cdlapp-btn-icon"));
      dl.appendChild(iconEl("download"));
      dl.title = t("out_download");
      dl.href = "/comfydl/artifacts/download?name="
        + encodeURIComponent(artifact.path || artifact.name);
      dl.setAttribute("download", artifact.name);
      row.appendChild(dl);
    }
    return row;
  }

  function humanSize(n) {
    if (n === undefined || n === null || isNaN(n)) return "";
    const units = ["B", "KB", "MB", "GB"];
    let v = Number(n), i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return (i === 0 ? v : v.toFixed(1)) + " " + units[i];
  }

  function renderProgress() {
    if (!state.panel) return;
    const rowsEl = state.panel.querySelector(".cdlapp-prog-rows");
    const label = state.panel.querySelector(".cdlapp-total-label");
    const bar = state.panel.querySelector(".cdlapp-total-bar");
    if (!rowsEl) return;
    const run = state.activePrompt ? state.runs.get(state.activePrompt) : null;
    if (!run || run.order.length === 0) {
      rowsEl.textContent = "";
      if (label) label.textContent = t("prog_idle");
      if (bar) bar.style.width = "0%";
      return;
    }
    // INCREMENTAL row update (batch 17 fix, user callout: "every progress
    // tick replays the entrance animation on all rows - too glitchy"):
    // the server pushes progress on EVERY optimizer step, so rebuilding the
    // rows would replay each row's slide-in every time.  Existing rows are
    // UPDATED in place (no animation); only genuinely new rows are created
    // and thus animate once.  Rows keyed via dataset.nodeKey.
    const existing = new Map();
    Array.from(rowsEl.children).forEach((child) => {
      if (child.dataset && child.dataset.nodeKey) existing.set(child.dataset.nodeKey, child);
      else child.remove(); // stray DOM (e.g. after a Vue re-render)
    });
    const alive = new Set();
    let done = 0;
    run.order.forEach((key) => {
      const r = run.nodes.get(key);
      if (r.state === "done" || r.state === "cached") done++;
      alive.add(key);
      let row = existing.get(key);
      if (!row) {
        row = el("div", "cdlapp-prog-row");
        row.dataset.nodeKey = key;
        row.appendChild(el("span", "cdlapp-prog-name"));
        const mini = el("div", "cdlapp-prog-mini");
        mini.appendChild(el("div", "cdlapp-prog-fill"));
        row.appendChild(mini);
        row.appendChild(el("span", "cdlapp-prog-pct"));
        rowsEl.appendChild(row);
      }
      row.className = "cdlapp-prog-row cdlapp-st-" + r.state;
      const name = row.children[0];
      if (name.textContent !== r.title) {
        name.textContent = r.title;
        name.title = r.title;
      }
      const fill = row.children[1].firstChild;
      fill.style.width = (r.state === "done" || r.state === "cached") ? "100%"
        : (r.percent !== null ? r.percent + "%" : "0%");
      // right column: percent for running rows (the green bar next to it
      // carries the same value), ETA once the sliding window is long enough
      const right = row.children[2];
      let rightText;
      let eta = null;
      if (r.state === "done" || r.state === "cached") {
        rightText = "✓";
      } else if (r.state === "error") {
        rightText = "✗";
      } else if (r.state === "idle") {
        rightText = "–";
      } else {
        rightText = r.percent !== null ? r.percent + "%" : "…";
        eta = (r.percent !== null && r.percent < 100)
          ? etaText(r, r.lastValue, r.maxValue) : null;
      }
      right.textContent = rightText;
      if (eta !== null) {
        right.appendChild(el("span", "cdlapp-prog-eta", " " + eta));
      }
    });
    // drop rows that no longer belong to the active run
    existing.forEach((row, key) => {
      if (!alive.has(key)) row.remove();
    });
    const total = Math.max(run.order.length, run.totalNodes || 0);
    const pct = total ? Math.round(100 * done / total) : 0;
    if (label) label.textContent = t("prog_node_fmt",
      { done: done, total: total, pct: pct });
    if (bar) bar.style.width = pct + "%";
  }

  function renderErrors() {
    if (!state.panel) return;
    const list = state.panel.querySelector(".cdlapp-err-list");
    const empty = state.panel.querySelector(".cdlapp-sec-errors .cdlapp-empty");
    if (!list) return;
    list.textContent = "";
    if (empty) empty.style.display = state.errors.length ? "none" : "";
    state.errors.forEach((e, idx) => {
      const card = el("div", "cdlapp-err-card");
      const head = el("div", "cdlapp-err-head");
      const title = el("span", "cdlapp-err-node",
        (e.title || e.node_type || ("node " + e.node_id)));
      title.title = e.etype || "";
      const cp = el("button", btnCls("size-8", "cdlapp-btn-icon"));
      cp.appendChild(iconEl("copy"));
      cp.title = t("err_copy");
      cp.addEventListener("click", () => {
        const blob = [
          "node: " + (e.title || e.node_type) + " (#" + e.node_id + ")",
          "type: " + (e.etype || ""),
          "message: " + (e.message || ""),
          "traceback:",
          (e.traceback || []).join("\n"),
        ].join("\n");
        navigator.clipboard.writeText(blob).then(() => {
          const ok = iconEl("check");
          ok.classList.add("cdlapp-pop");
          cp.replaceChildren(ok);
          setTimeout(() => { cp.replaceChildren(iconEl("copy")); }, 1200);
        }).catch(() => { /* clipboard unavailable */ });
      });
      head.appendChild(title);
      head.appendChild(cp);
      card.appendChild(head);
      card.appendChild(el("div", "cdlapp-err-msg", e.message || ""));
      if (e.traceback && e.traceback.length) {
        const pre = el("pre", "cdlapp-err-tb", e.traceback.join("\n"));
        pre.style.display = "none";
        const toggle = el("div", "cdlapp-err-toggle", "traceback ▸");
        toggle.addEventListener("click", () => {
          const show = pre.style.display === "none";
          pre.style.display = show ? "" : "none";
          toggle.textContent = show ? "traceback ▾" : "traceback ▸";
        });
        card.appendChild(toggle);
        card.appendChild(pre);
      }
      card.dataset.errIdx = String(idx);
      list.appendChild(card);
    });
  }

  // ------------------------------------------------------ event wiring ----
  function nodeTitle(nodeId) {
    try {
      const app = getApp();
      const n = app && app.graph && app.graph.getNodeById
        ? app.graph.getNodeById(Number(nodeId))
        : null;
      if (n) return n.title || n.type || String(nodeId);
    } catch (e) { /* fallthrough */ }
    return String(nodeId);
  }

  function runFor(promptId) {
    const pid = promptId || "default";
    if (!state.runs.has(pid)) {
      state.runs.set(pid, { nodes: new Map(), order: [], totalNodes: 0 });
    }
    if (state.runs.size > 8) {
      // keep the newest handful of runs only
      const firstKey = state.runs.keys().next().value;
      state.runs.delete(firstKey);
    }
    return state.runs.get(pid);
  }

  function markNode(promptId, nodeId, patch) {
    const run = runFor(promptId);
    const key = String(nodeId);
    if (!run.nodes.has(key)) {
      run.nodes.set(key, {
        title: nodeTitle(nodeId), state: "running", percent: null,
      });
      run.order.push(key);
    }
    Object.assign(run.nodes.get(key), patch);
  }

  // Scope outputs/errors to the ACTIVE run: the artifacts route lists the
  // whole output directory, so without this every historical file piled up
  // in the panel (user callout 2026-10-30).  A run begins with its first
  // event and ends on success/interrupt/error.
  function beginRunIfNew() {
    if (state.runActive) return;
    state.runActive = true;
    state.runStartedAt = Date.now();
    state.outputs.length = 0;
    state.errors.length = 0;
    state.charts.clear();
    // Pre-fill the progress rows with the FULL node list so the user can
    // see how many nodes are ahead from the very first event (batch 16).
    prefillNodes(state.activePrompt);
    // status dot: teal while executing (CSS hook)
    if (state.extras) state.extras.classList.add("cdlapp-running");
    renderErrors();
    renderOutputs();
  }

  // Pre-create one idle progress row per executable node, in execution
  // order, straight from the litegraph graph (best effort: without it the
  // event-driven path still works, rows just appear one by one).  Skipped:
  // muted / bypassed nodes (mode 2 / 4) and the official non-executing
  // annotation types (chunk rule nodeTypeValidForApp: Note / MarkdownNote
  // never appear in the execution stream - leaving them in would pin the
  // total below 100% forever).
  function prefillNodes(promptId) {
    const run = runFor(promptId);
    if (run.order.length) return;
    try {
      const app = getApp();
      const graph = app && app.graph;
      if (!graph) return;
      let nodes = null;
      if (typeof graph.computeExecutionOrder === "function") {
        const ordered = graph.computeExecutionOrder(false, null);
        nodes = Array.isArray(ordered) ? ordered
          : (ordered && ordered.nodes) || null;
      }
      if (!nodes && Array.isArray(graph._nodes)) nodes = graph._nodes;
      if (!nodes) return;
      nodes.forEach((n) => {
        if (!n || n.id === undefined || n.id === null) return;
        if (n.mode === 2 || n.mode === 4) return; // muted / bypassed
        if (n.type === "Note" || n.type === "MarkdownNote") return;
        const key = String(n.id);
        if (run.nodes.has(key)) return;
        run.nodes.set(key, {
          title: n.title || n.type || key,
          state: "idle", percent: null,
        });
        run.order.push(key);
      });
    } catch (e) { /* best effort */ }
  }

  // Speed / ETA estimation (batch 16, user request): progress events carry
  // raw value/max; keep a tiny sliding window of (time, value) samples per
  // running row and derive the rate from its ends - O(1) per event, no
  // timers, no extrapolation beyond the observed window.
  function noteProgressSample(row, value) {
    if (value === null || value === undefined || isNaN(value)) return;
    const s = row.samples || (row.samples = []);
    s.push({ t: Date.now(), v: value });
    if (s.length > 6) s.shift();
  }

  function rateOf(row) {
    const s = row.samples;
    if (!s || s.length < 2) return null;
    const a = s[0], b = s[s.length - 1];
    const dt = (b.t - a.t) / 1000;
    if (dt < 0.4) return null; // window too young to be meaningful
    const rate = (b.v - a.v) / dt;
    return rate > 0 ? rate : null;
  }

  function etaText(row, value, max) {
    const rate = rateOf(row);
    if (!rate || max === null || value === null || value === undefined) {
      return null;
    }
    const remain = max - value;
    if (remain <= 0) return null;
    const sec = Math.max(1, Math.round(remain / rate));
    return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
  }

  function endRun() {
    state.runActive = false;
    if (state.extras) state.extras.classList.remove("cdlapp-running");
  }

  function onExecuting(ev) {
    const d = ev && ev.detail;
    if (d === null || d === undefined) return; // null = queue went idle
    if (state.activePrompt === null) state.activePrompt = "default";
    beginRunIfNew();
    // Live-verified 2026-10-30: the official executing event carries a BARE
    // node id string ("1", "2", ..., null) - NOT an object.  Handle both
    // shapes; the bare id has no prompt_id, so the current run stays active.
    const nodeId = typeof d === "object"
      ? (d.display_node || d.node || d.node_id)
      : d;
    if (nodeId !== undefined && nodeId !== null) {
      const run = runFor(state.activePrompt);
      // The official executed event only fires for nodes with outputs
      // (live-verified: 5 executing vs 1 executed), so node completion must
      // be derived: execution is serial, therefore when the next node starts
      // every OTHER still-running node has finished (this is what makes the
      // checkmarks appear one by one instead of all at once at the end).
      run.nodes.forEach((r, key) => {
        if (key !== String(nodeId) && r.state === "running") r.state = "done";
      });
      markNode(state.activePrompt, nodeId, { state: "running" });
    }
    renderProgress();
  }

  function onExecuted(ev) {
    const d = ev && ev.detail;
    if (!d) return;
    if (d.prompt_id) state.activePrompt = d.prompt_id;
    if (d.node !== undefined) {
      markNode(state.activePrompt, d.node, { state: "done", percent: 100 });
    }
    collectOutputs(d);
    renderProgress();
    renderOutputs();
  }

  function onExecutionCached(ev) {
    const d = ev && ev.detail;
    if (!d || !Array.isArray(d.nodes)) return;
    if (d.prompt_id) state.activePrompt = d.prompt_id;
    if (state.activePrompt === null) state.activePrompt = "default";
    beginRunIfNew();
    d.nodes.forEach((n) => markNode(state.activePrompt, n, { state: "cached", percent: 100 }));
    renderProgress();
  }

  function onProgressState(ev) {
    const d = ev && ev.detail;
    if (!d) return;
    state.activePrompt = d.prompt_id || state.activePrompt || "default";
    beginRunIfNew();
    // Payload shape (chunk-verified 2026-10-31 against the official
    // comfy_execution/progress.py sender):
    //   { prompt_id, nodes: { <node_id>: { value, max, state, node_id,
    //     display_node_id, parent_node_id, real_node_id } } }
    // - a DICT of per-node states (server sends the FULL non-pending set on
    //   every tick).  The old flat single-node shape is kept as a fallback.
    if (d.nodes && typeof d.nodes === "object"
        && !Array.isArray(d.nodes)) {
      const run = state.runs.get(state.activePrompt);
      Object.keys(d.nodes).forEach((nid) => {
        const n = d.nodes[nid] || {};
        const nodeId = n.display_node_id || n.node_id || nid;
        const value = n.value === undefined ? null : Number(n.value);
        const max = n.max === undefined ? null : Number(n.max);
        const pct = max ? Math.round(100 * value / max) : null;
        // NodeState enum (progress.py): pending | running | finished | error
        const st = n.state === "error" ? "error"
          : n.state === "finished" ? "done" : "running";
        markNode(state.activePrompt, nodeId, {
          percent: pct,
          state: st,
        });
        const row = run && run.nodes.get(String(nodeId));
        if (row && st === "running") {
          row.lastValue = value;
          row.maxValue = max;
          noteProgressSample(row, value);
        }
      });
      renderProgress();
      return;
    }
    // legacy flat fallback (single node fields at the top level)
    const nodeId = d.display_node_id || d.node_id;
    if (nodeId === undefined || nodeId === null) return;
    const pct = d.max ? Math.round(100 * Number(d.value) / Number(d.max)) : null;
    markNode(state.activePrompt, nodeId, {
      percent: pct,
      state: d.state === "error" ? "error" : "running",
    });
    // raw value for the sliding-window speed estimate (batch 16)
    const run = state.runs.get(state.activePrompt);
    const row = run && run.nodes.get(String(nodeId));
    if (row) {
      row.lastValue = d.value === undefined ? null : Number(d.value);
      row.maxValue = d.max === undefined ? null : Number(d.max);
      noteProgressSample(row, row.lastValue);
    }
    renderProgress();
  }

  function onExecutionSuccess() {
    if (state.activePrompt) {
      const run = state.runs.get(state.activePrompt);
      if (run) run.nodes.forEach((r) => {
        if (r.state === "running") r.state = "done";
      });
    }
    endRun();
    renderProgress();
    refreshArtifacts();
  }

  function onExecutionInterrupted() {
    if (state.activePrompt) {
      const run = state.runs.get(state.activePrompt);
      if (run) run.nodes.forEach((r) => {
        if (r.state === "running") r.state = "idle";
      });
    }
    endRun();
    renderProgress();
  }

  function onExecutionError(ev) {
    const d = (ev && ev.detail) || {};
    const entry = {
      node_id: d.node_id,
      node_type: d.node_type,
      title: d.node_id !== undefined ? nodeTitle(d.node_id) : d.node_type,
      etype: d.exception_type,
      message: d.exception_message,
      traceback: Array.isArray(d.traceback) ? d.traceback : [],
    };
    state.errors.unshift(entry);
    if (state.errors.length > 10) state.errors.pop();
    if (d.node_id !== undefined && state.activePrompt) {
      markNode(state.activePrompt, d.node_id, { state: "error" });
    }
    endRun();
    renderErrors();
    renderProgress();
  }

  // ------------------------------------------------ live charts (batch 18)
  // The backend previewer pushes structured loss series / text snapshots as
  // the "cdl_preview_data" event (comfy/loss_preview.py).  This module draws
  // them as theme-aware interactive SVG charts in the previews pane - the
  // bitmap preview frame stays only as the node-graph fallback.

  function onPreviewData(ev) {
    const d = ev && ev.detail;
    if (!d || !d.node_id) return;
    if (!state.runActive) beginRunIfNew();
    state.charts.set(String(d.node_id), d);
    renderCharts();
  }

  function svgEl(tag, cls) {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (cls) n.setAttribute("class", cls);
    return n;
  }

  function movingAvg(values, window) {
    const out = [];
    let running = 0;
    for (let i = 0; i < values.length; i++) {
      running += values[i];
      if (i >= window) running -= values[i - window];
      out.push(running / Math.min(i + 1, window));
    }
    return out;
  }

  function fmtNum(v) {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 1) return v.toFixed(3);
    return v.toPrecision(3);
  }

  // Interactive loss-curve SVG: raw series (thin, translucent) + trailing
  // moving average (accent) + best marker + crosshair readout on hover.
  // viewBox 240x110 stretched over a ~520x170 card, all theming via CSS
  // variables.  The card is created ONCE per node (batch 19: the squash fix
  // is the aspect ratio; the anti-glitch fix is that every push UPDATES the
  // skeleton in place via card._update - only the polylines' points and the
  // tick labels change, plus a soft one-shot pulse, never a card rebuild).
  function buildLossChart(d) {
    const card = el("div", "cdlapp-chart-card");
    const head = el("div", "cdlapp-chart-head");
    const title = el("span", "cdlapp-chart-title");
    const meta = el("span", "cdlapp-chart-meta");
    head.appendChild(title);
    head.appendChild(meta);
    card.appendChild(head);

    const svg = svgEl("svg", "cdlapp-chart-svg");
    svg.setAttribute("viewBox", "0 0 240 110");
    svg.setAttribute("preserveAspectRatio", "none");
    const W = 240, H = 110, L = 34, R = 8, T = 12, B = 16;
    const iw = W - L - R, ih = H - T - B;
    let pts = []; // live points, re-read by the hover handlers on every push
    let geom = null;

    // hover crosshair (created once, re-uses the latest pts/geom)
    const cross = svgEl("g", "cdlapp-chart-cross");
    cross.style.display = "none";
    const vline = svgEl("line", "cdlapp-chart-crossline");
    const dot = svgEl("circle", "cdlapp-chart-dot");
    dot.setAttribute("r", 2.4);
    const read = svgEl("text", "cdlapp-chart-read");
    cross.appendChild(vline); cross.appendChild(dot); cross.appendChild(read);
    svg.addEventListener("mousemove", (e) => {
      if (!geom || pts.length < 2) return;
      const rect = svg.getBoundingClientRect();
      const mx = (e.clientX - rect.left) / rect.width * W;
      const cx = Math.max(L, Math.min(W - R, mx));
      const idx = Math.round((cx - L) / iw * (pts.length - 1));
      const p = pts[Math.max(0, Math.min(pts.length - 1, idx))];
      const px = geom.sx(p[0]), py = geom.sy(p[1]);
      vline.setAttribute("x1", px); vline.setAttribute("x2", px);
      vline.setAttribute("y1", T); vline.setAttribute("y2", T + ih);
      dot.setAttribute("cx", px); dot.setAttribute("cy", py);
      read.setAttribute("x", Math.min(px + 4, W - R - 60));
      read.setAttribute("y", Math.max(T + 8, py - 4));
      read.textContent = "s" + p[0] + " " + fmtNum(p[1]);
      cross.style.display = "";
    });
    svg.addEventListener("mouseleave", () => { cross.style.display = "none"; });

    // in-place redraw of the dynamic layer (everything but the crosshair)
    function update(d2) {
      title.textContent = d2.title || "loss";
      meta.textContent = t("chart_step") + " " + d2.step
        + (d2.total ? "/" + d2.total : "")
        + " · " + t("chart_best") + " " + fmtNum(d2.best)
        + (d2.finished ? " · ✓" : "");
      pts = (d2.points || []).filter((p) => p && p.length >= 2);
      Array.from(svg.children).forEach((n) => { if (n !== cross) n.remove(); });
      if (pts.length < 2) return;
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      let lo = Math.min.apply(null, ys);
      let hi = Math.max.apply(null, ys);
      // the historical best can sit below the visible series - include it so
      // the dashed best marker is always drawable
      if (typeof d2.best === "number" && isFinite(d2.best)) {
        lo = Math.min(lo, d2.best);
        hi = Math.max(hi, d2.best);
      }
      if (hi <= lo) hi = lo + 1;
      const x0 = xs[0], x1 = xs[xs.length - 1];
      const sx = (x) => L + iw * (x - x0) / Math.max(1e-9, x1 - x0);
      const sy = (y) => T + ih * (1 - (y - lo) / (hi - lo));
      geom = { sx: sx, sy: sy };
      const line = (arr, cls) => {
        const pl = svgEl("polyline", cls);
        pl.setAttribute("points", arr.map((p) =>
          sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join(" "));
        svg.appendChild(pl);
      };
      for (let f = 0; f <= 2; f++) {
        const y = T + ih * f / 2;
        const gl = svgEl("line", "cdlapp-chart-grid");
        gl.setAttribute("x1", L); gl.setAttribute("x2", W - R);
        gl.setAttribute("y1", y); gl.setAttribute("y2", y);
        svg.appendChild(gl);
        const val = hi - (hi - lo) * f / 2;
        const tx = svgEl("text", "cdlapp-chart-tick");
        tx.setAttribute("x", L - 3); tx.setAttribute("y", y + 3);
        tx.setAttribute("text-anchor", "end");
        tx.textContent = fmtNum(val);
        svg.appendChild(tx);
      }
      const xl = svgEl("text", "cdlapp-chart-tick");
      xl.setAttribute("x", L); xl.setAttribute("y", H - 3);
      xl.textContent = String(x0);
      svg.appendChild(xl);
      const xr = svgEl("text", "cdlapp-chart-tick");
      xr.setAttribute("x", W - R); xr.setAttribute("y", H - 3);
      xr.setAttribute("text-anchor", "end");
      xr.textContent = String(x1);
      svg.appendChild(xr);
      const by = sy(d2.best);
      if (by >= T && by <= T + ih) {
        const bl = svgEl("line", "cdlapp-chart-best");
        bl.setAttribute("x1", L); bl.setAttribute("x2", W - R);
        bl.setAttribute("y1", by); bl.setAttribute("y2", by);
        svg.appendChild(bl);
      }
      line(pts, "cdlapp-chart-raw");
      const smooth = movingAvg(pts.map((p) => p[1]), 8);
      if (pts.length >= 8) {
        line(pts.map((p, i) => [p[0], smooth[i]]), "cdlapp-chart-smooth");
      }
      svg.appendChild(cross); // crosshair always paints on top
    }
    card._update = update;
    update(d);
    card.appendChild(svg);
    return card;
  }

  function buildTextCard(d) {
    const card = el("div", "cdlapp-chart-card cdlapp-chart-text");
    const head = el("div", "cdlapp-chart-head");
    const title = el("span", "cdlapp-chart-title");
    head.appendChild(title);
    card.appendChild(head);
    const body = el("pre", "cdlapp-chart-body");
    card.appendChild(body);
    card._update = function (d2) {
      title.textContent = d2.title || "generated";
      body.textContent = (d2.lines || []).join("\n");
      body.scrollTop = body.scrollHeight;
    };
    card._update(d);
    return card;
  }

  // One-shot soft glow on data arrival (batch 19, user request): marks the
  // frame as fresh without any hard rebuild.  Gated by the motion switch
  // and prefers-reduced-motion, ~0 cost (one CSS animation).
  function pulseChart(card) {
    if (!state.motion) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    card.classList.remove("cdlapp-chart-pulse");
    void card.offsetWidth; // restart the one-shot animation
    card.classList.add("cdlapp-chart-pulse");
  }

  function renderCharts() {
    const host = state.extras
      ? state.extras.querySelector(".cdlapp-chart-host")
      : (state.panel
        ? state.panel.querySelector(".cdlapp-chart-host")
        : null);
    if (!host) return;
    const empty = state.charts.size === 0;
    host.style.display = empty ? "none" : "";
    // incremental reuse keyed by node id - a live chart must NOT replay its
    // entrance animation on every push (same discipline as renderProgress)
    const existing = new Map();
    Array.from(host.children).forEach((child) => {
      if (child.dataset && child.dataset.chartKey) {
        existing.set(child.dataset.chartKey, child);
      } else child.remove();
    });
    const alive = new Set();
    state.charts.forEach((d, key) => {
      alive.add(key);
      let card = existing.get(key);
      if (!card) {
        card = d.type === "text" ? buildTextCard(d) : buildLossChart(d);
        card.dataset.chartKey = key;
        host.appendChild(card);
      } else {
        // batch 19: in-place data update + one-shot pulse - the entrance
        // animation only ever plays on genuinely new cards
        if (typeof card._update === "function") card._update(d);
        pulseChart(card);
      }
    });
    existing.forEach((card, key) => { if (!alive.has(key)) card.remove(); });
  }

  function collectOutputs(d) {
    const out = d.output || {};
    if (Array.isArray(out.images)) {
      out.images.forEach((img) => {
        if (!img || !img.filename) return;
        const q = "filename=" + encodeURIComponent(img.filename) +
          "&subfolder=" + encodeURIComponent(img.subfolder || "") +
          "&type=" + encodeURIComponent(img.type || "output");
        state.outputs.unshift({
          kind: "image", name: img.filename,
          url: "/view?" + q,
        });
      });
    }
    const texts = Array.isArray(out.text) ? out.text
      : typeof out.text === "string" ? [out.text] : [];
    texts.forEach((txt) => {
      if (typeof txt !== "string" || !txt) return;
      state.outputs.unshift({ kind: "text", text: txt, name: txt.slice(0, 40) });
    });
    if (state.outputs.length > 50) state.outputs.length = 50;
  }

  async function refreshArtifacts() {
    try {
      const api = getApi();
      const resp = api && api.fetchApi
        ? await api.fetchApi("/comfydl/artifacts?limit=200")
        : await fetch("/comfydl/artifacts?limit=200");
      if (!resp || !resp.ok) return;
      const data = await resp.json();
      state.artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
      renderOutputs();
    } catch (e) { /* artifacts route optional (P2) */ }
  }

  function attachEvents() {
    const api = getApi();
    if (!api || typeof api.addEventListener !== "function") {
      warnOnce("api", "api event surface unavailable");
      return;
    }
    const E = OFFICIAL_SURFACE.events;
    state.handlers = {
      [E.executing]: onExecuting,
      [E.executed]: onExecuted,
      [E.executionCached]: onExecutionCached,
      [E.progressState]: onProgressState,
      [E.executionSuccess]: onExecutionSuccess,
      [E.executionInterrupted]: onExecutionInterrupted,
      [E.executionError]: onExecutionError,
      // structured live-preview pushes from comfy/loss_preview.py (batch 18)
      cdl_preview_data: onPreviewData,
    };
    Object.keys(state.handlers).forEach((name) => {
      api.addEventListener(name, state.handlers[name]);
    });
  }

  function detachEvents() {
    const api = getApi();
    if (!api || !state.handlers) return;
    Object.keys(state.handlers).forEach((name) => {
      try { api.removeEventListener(name, state.handlers[name]); } catch (e) { /* silent */ }
    });
    state.handlers = null;
  }

  // ------------------------------------------------------ settings/boot ----
  async function init() {
    const app = getApp();
    if (!app) return;
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "/comfydl/cdlapp/cdlapp.css";
    document.head.appendChild(css);
    try {
      const s = app.extensionManager && app.extensionManager.setting;
      if (s && typeof s.get === "function") {
        const v = s.get("Comfy.Locale");
        if (typeof v === "string" && v) state.locale = v.split("-")[0];
        state.enabled = s.get(SETTING_ENABLED) !== false;
        state.motion = s.get(SETTING_MOTION) !== false;
      }
    } catch (e) { /* defaults */ }
    applyMotion(state.motion);
    if (state.locale !== "zh") state.locale = "en";

    probeCapabilities();
    probeStores(); // first attempt; retried lazily from the tick until Vue mounts
    attachEvents();
    if (state.enabled) {
      startPolling(); // mounts immediately if already in app mode
    }
  }

  function applyEnabled(enabled) {
    state.enabled = enabled !== false;
    if (state.enabled) {
      startPolling();
    } else {
      unmountPanel();
      unmountBuilderBack();
      stopPolling();
    }
  }

  // Global motion switch (batch 16): the CSS gate is body.cdlapp-motion-off
  // (animations are ON by default); prefers-reduced-motion is honoured
  // unconditionally in the stylesheet.
  function applyMotion(on) {
    state.motion = on !== false;
    document.body.classList.toggle("cdlapp-motion-off", !state.motion);
  }

  const start = (force) => {
    const app = getApp();
    if (!app || typeof app.registerExtension !== "function") return false;
    if (!force && !settingsStoreReady(app)) return false;
    // Locale best-effort before registration (init() re-detects).
    try {
      const s = app.extensionManager && app.extensionManager.setting;
      if (s && typeof s.get === "function") {
        const v = s.get("Comfy.Locale");
        if (typeof v === "string" && v) state.locale = v.split("-")[0];
      }
    } catch (e) { /* settings store not ready: init() re-detects */ }
    app.registerExtension({
      name: "ComfyDL.AppPanel",
      async setup() { await init(); },
      settings: [{
        id: SETTING_ENABLED,
        name: t("setting_enabled"),
        tooltip: t("setting_enabled_tooltip"),
        category: ["ComfyDL", "App"],
        type: "boolean",
        defaultValue: true,
        onChange: (newVal) => { applyEnabled(newVal !== false); },
      }, {
        id: SETTING_MOTION,
        name: t("setting_motion"),
        tooltip: t("setting_motion_tooltip"),
        category: ["ComfyDL", "App"],
        type: "boolean",
        defaultValue: true,
        onChange: (newVal) => { applyMotion(newVal !== false); },
      }],
      commands: [],
    });
    return true;
  };

  function boot() {
    if (!start()) {
      waitFor(() => start(false), 30000, 200).then((ok) => {
        if (!ok) {
          console.warn(TAG,
            "settings store never signalled ready in 30s; registering without the gate");
          start(true);
        }
      });
    }
  }

  boot();
})();
