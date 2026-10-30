/**
 * ComfyDL_UI motion module - staged entrance choreography + ripple feedback.
 *
 * Ported from the homepage fork (homepage/extensions/comfydl_home/home.js,
 * 2026-10-14 batch) into the main project's loader pattern, which is the same
 * as profiler.js / crashsite.js: one module script injected into index.html,
 * an IIFE that waits for window.comfyAPI, then app.registerExtension.
 *
 * What this module does:
 *   1. Staged entrance (once per page load): topbar slides down, bottom bar
 *      slides up, right bar slides in, sidebar rail + tabs stagger in, then
 *      canvas nodes fade in along a diagonal sweep and the canvas (grid +
 *      links) fades back in last. The pre-hide rules live in a <style> block
 *      baked into index.html by frontend_patch.py (_patch_motion_css); this
 *      module orchestrates the reveal and flips the gate classes.
 *   2. Ripple click feedback: a page-global, pointer-events:none overlay
 *      layer; every left pointerdown spawns a soft expanding ring. Nothing is
 *      injected into Vue-managed elements.
 *   3. Splash progress driver: the branded boot screen (#cdl-splash, baked by
 *      frontend_patch._patch_splash_brand in place of the official wave SVG)
 *      is fed with REAL signals - PerformanceObserver for chunk loads, a
 *      read-only fetch wrapper for /api milestones, and polling of
 *      app.extensionManager.spinner for the boot-sequence falling edge. On
 *      that falling edge motion.js takes over the splash (the official
 *      watcher's getElementById hits a hidden stub), fades it out and starts
 *      the entrance the same instant, so the chrome slides in "through" the
 *      dissolving splash.
 *
 * Hard rules inherited from the homepage implementation (do not regress):
 *   - Canvas node DOM animation is opacity-ONLY. The Vue node system positions
 *     each node via its own transform: translate(x, y); animating transform
 *     here would stomp it (with fill:'both' it persists) and collapse every
 *     node to the canvas origin, freezing them against drags.
 *   - Entrance animations use fill:'both' + cdlCancelAfter, so the finished
 *     animation layer is dropped and later CSS :hover/:active transforms are
 *     not shadowed.
 *   - Vue re-renders reset className; the pre-hide rules stop matching via
 *     gate classes on <html> (cdl-entrance-done / cdl-nodes-in /
 *     cdl-canvas-in / cdl-motion-off), never via per-element classes.
 *   - prefers-reduced-motion: reduce disables everything (CSS media query on
 *     the pre-hide block + explicit runtime checks here).
 *   - Failsafe: the baked <style> block carries a 9s timer that force-reveals
 *     everything, so a failed motion.js load can never leave the UI invisible.
 *
 * Settings:
 *   - ComfyDL.Motion.Enabled (default true): master switch. Turning it off
 *     mid-session removes the ripple layer and force-reveals any still-hidden
 *     elements; the entrance itself only ever plays once per page load.
 */
;(function () {
  'use strict';

  const TAG = "[ComfyDL motion]";
  const SETTING_ENABLED = "ComfyDL.Motion.Enabled";

  // --------------------------------------------------------------- i18n ----
  const I18N = {
    en: {
      setting_enabled: "UI motion (entrance choreography + click ripple)",
      setting_enabled_tooltip:
        "Staged entrance animations and the ripple click feedback. "
        + "Changing it applies immediately; the entrance itself plays once per page load.",
    },
    zh: {
      setting_enabled: "界面动效（入场编排 + 点击涟漪）",
      setting_enabled_tooltip:
        "分阶段入场动画与点击涟漪反馈。改动即时生效；入场动画每次刷新只播放一次。",
    },
  };

  let locale = "en";
  const t = (k) => (I18N[locale] && I18N[locale][k]) || I18N.en[k] || k;

  const state = {
    enabled: true,
    entranceDone: false,
    rippleInstalled: false,
    rippleLayer: null,
    sidebarObserver: null,
    // Splash progress driver (see initSplashProgress).
    splashDone: false,
    splashEl: null,
    splashScore: 0,
    splashStartedAt: 0,
    seenApi: {},
    splashDriverActive: false,
  };

  // ------------------------------------------- splash progress driver ----
  // Runs from module-evaluation time (before Vue mounts) and drives the
  // branded splash with REAL signals only:
  //   - PerformanceObserver(resource, buffered) -> chunk/css load milestones
  //   - a read-only fetch wrapper -> /api milestones (extensions,
  //     object_info, settings, ...), named after the homepage fork's table
  //   - a slow time-based crawl covers the blind JS-only stretches
  //   - polling app.extensionManager.spinner (the workspace store the App
  //     component exposes) detects the boot-sequence falling edge (C3)
  // On the falling edge motion.js takes over the splash from the official
  // App watcher: the visible layer is #cdl-splash (the official hidden
  // #splash-loader stub keeps the watcher's getElementById harmless), so we
  // fade out and remove it ourselves, and the entrance choreography starts
  // the same instant - UI elements slide in "through" the dissolving splash.
  const SPLASH_I18N = {
    en: {
      page: "Page delivered",
      chunks: "Interface chunks ({n})",
      vueMounted: "Interface mounted",
      extensions: "Extension registry",
      objectInfo: "Node definitions (canvas + library)",
      settings: "UI preferences",
      userdata: "Workspace scan",
      systemStats: "System telemetry",
      queue: "Execution queue",
      misc: "Backend API ({n})",
      modules: "ComfyDL modules ready",
      ready: "Interface ready — welcome!",
    },
    zh: {
      page: "页面已送达",
      chunks: "界面资源加载（{n}）",
      vueMounted: "界面已挂载",
      extensions: "扩展注册表",
      objectInfo: "节点定义（画布 + 节点库）",
      settings: "界面偏好",
      userdata: "工作区扫描",
      systemStats: "系统遥测",
      queue: "执行队列",
      misc: "后端接口（{n}）",
      modules: "ComfyDL 模块就绪",
      ready: "界面就绪 —— 欢迎使用！",
    },
  };
  const spl = (k, params) => {
    const dict = I18N[locale] && I18N[locale][k] ? I18N[locale] : I18N.en;
    let text = dict[k] !== undefined ? dict[k] : I18N.en[k] || k;
    if (params) {
      for (const [pk, pv] of Object.entries(params)) {
        text = text.split("{" + pk + "}").join(String(pv));
      }
    }
    return text;
  };

  const SPLASH_MAX = 100;
  const SPLASH_CRAWL_CAP = 90;   // the time-crawl never passes this
  const SPLASH_CRAWL_MS = 9000;  // crawl duration to reach the cap

  function splashLog(msg) {
    const logEl = document.getElementById("cdlLog");
    if (!logEl) return;
    const line = document.createElement("div");
    line.className = "cdl-line";
    const s = ((performance.now() - state.splashStartedAt) / 1000).toFixed(2);
    line.innerHTML = '<span class="cdl-t">[' + s + 's]</span>'
      + '<span class="cdl-m">' + String(msg).replace(/[<>&]/g, "") + "</span>"
      + '<span class="cdl-s ok">OK</span>';
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
    while (logEl.children.length > 60) logEl.removeChild(logEl.firstChild);
  }

  function splashPhase(text) {
    const el = document.getElementById("cdlPhase");
    if (el) el.textContent = text;
  }

  function splashScore(delta) {
    state.splashScore = Math.min(SPLASH_MAX, state.splashScore + delta);
    const pctEl = document.getElementById("cdlPct");
    const fillEl = document.getElementById("cdlFill");
    const pct = Math.round(state.splashScore);
    if (pctEl) pctEl.textContent = pct + "%";
    if (fillEl) fillEl.style.width = pct + "%";
  }

  // Time crawl: eases toward SPLASH_CRAWL_CAP so blind stretches (pure-JS
  // init with no observable signals) never look frozen.
  function startCrawl() {
    const tick = () => {
      if (state.splashDone) return;
      const elapsed = performance.now() - state.splashStartedAt;
      const target = Math.min(SPLASH_CRAWL_CAP, (elapsed / SPLASH_CRAWL_MS) * SPLASH_CRAWL_CAP);
      if (state.splashScore < target) splashScore(Math.min(target - state.splashScore, 2));
      setTimeout(tick, 120);
    };
    setTimeout(tick, 120);
  }

  // One-time named milestones.
  function splashMilestone(key, weight, name) {
    if (state.seenApi[key]) return;
    state.seenApi[key] = 1;
    splashLog(name);
    splashPhase(name);
    splashScore(weight);
  }

  function wrapFetch() {
    if (window.__cdlFetchWrapped) return;
    window.__cdlFetchWrapped = true;
    let miscCount = 0;
    const realFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      const url = typeof input === "string" ? input : (input && input.url) || "";
      const p = realFetch.apply(this, arguments);
      if (url.indexOf("/api/") >= 0 || url.indexOf("/api") === 0) {
        p.then(() => {
          try {
            const path = new URL(url, location.href).pathname.replace(/^.*\/api\/?/, "");
            if (path === "extensions") splashMilestone("extensions", 8, spl("extensions"));
            else if (path === "object_info") splashMilestone("objectInfo", 22, spl("objectInfo"));
            else if (path === "settings" || path === "user/settings") splashMilestone("settings", 5, spl("settings"));
            else if (path.indexOf("userdata") === 0) splashMilestone("userdata", 5, spl("userdata"));
            else if (path === "system_stats") splashMilestone("systemStats", 4, spl("systemStats"));
            else if (path === "queue" || path === "prompt" || path === "history") splashMilestone("queue", 3, spl("queue"));
            else {
              miscCount++;
              splashMilestone("misc:" + path, 2, spl("misc", { n: miscCount }));
            }
          } catch (e) { /* naming is cosmetic */ }
        }).catch(() => { /* fetch errors are the app's business */ });
      }
      return p;
    };
  }

  function initSplashProgress() {
    const el = document.getElementById("cdl-splash");
    if (!el) {
      // Splash brand patch not applied: no driver, no falling edge to wait
      // for - the entrance falls back to firing from extension setup().
      state.splashDriverActive = false;
      return;
    }
    state.splashDriverActive = true;
    state.splashEl = el;
    state.splashStartedAt = performance.now();

    splashLog(spl("page"));
    splashScore(4);
    startCrawl();
    wrapFetch();

    // Chunk/css load milestones (buffered:true replays resources that already
    // finished before this observer was attached).
    try {
      const chunkTotal = Math.max(
        1, document.querySelectorAll("link[rel=modulepreload]").length + 1);
      let chunkSeen = 0;
      const po = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const n = entry.name || "";
          if (/\.js(\?|$)/.test(n) || /\.css(\?|$)/.test(n)) chunkSeen++;
        }
        if (chunkSeen > 0) {
          splashMilestone("chunks:" + chunkSeen, 12,
            spl("chunks", { n: Math.min(chunkSeen, chunkTotal) }));
        }
      });
      po.observe({ type: "resource", buffered: true });
    } catch (e) { /* PerformanceObserver unavailable: crawl covers */ }

    // Vue mount detection: first child inside #vue-app.
    const vueApp = document.getElementById("vue-app");
    if (vueApp) {
      const ob = new MutationObserver(() => {
        if (vueApp.children.length) {
          ob.disconnect();
          splashMilestone("vueMounted", 8, spl("vueMounted"));
        }
      });
      ob.observe(vueApp, { childList: true });
    }

    // Boot-sequence falling edge (C3): poll the workspace store the App
    // component exposes. Once seen, take over the splash entirely.
    const pollSpinner = () => {
      if (state.splashDone) return;
      let spin = null;
      try {
        const app = getApp();
        const ws = app && app.extensionManager;
        if (ws && typeof ws.spinner === "boolean") spin = ws.spinner;
      } catch (e) { /* not ready yet */ }
      if (spin === false) {
        finishSplash();
        return;
      }
      setTimeout(pollSpinner, 50);
    };
    setTimeout(pollSpinner, 200);

    // Splash-own failsafe: never hold the UI hostage for more than 20s.
    setTimeout(() => { if (!state.splashDone) finishSplash(true); }, 20000);
  }

  function finishSplash(force) {
    if (state.splashDone) return;
    state.splashDone = true;
    splashScore(SPLASH_MAX - state.splashScore);
    splashLog(spl("ready"));
    splashPhase(spl("ready"));

    const done = () => {
      if (state.splashEl) {
        try { state.splashEl.remove(); } catch (e) { /* noop */ }
        state.splashEl = null;
      }
      // The entrance starts the same instant the splash begins dissolving,
      // so the chrome slides in "through" it. Disabled / reduced-motion
      // paths force-reveal instead of animating.
      if (force || reduceMotion() || !state.enabled) {
        disableAll();
        forceReveal();
      } else {
        runEntrance();
      }
    };

    const el = state.splashEl;
    if (force || reduceMotion()) { done(); return; }
    try {
      el.style.opacity = "0";
      el.style.transform = "scale(1.015)";
      setTimeout(done, 650);
    } catch (e) { done(); }
  }

  // ------------------------------------------------------------- helpers ----
  const getApp = () => (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) || null;

  function waitFor(fn, timeoutMs, intervalMs) {
    return new Promise((resolve) => {
      const t0 = Date.now();
      const poll = () => {
        let v = null;
        try { v = fn(); } catch (e) { /* not ready */ }
        if (v) return resolve(v);
        if (Date.now() - t0 > (timeoutMs || 10000)) return resolve(null);
        setTimeout(poll, intervalMs || 100);
      };
      poll();
    });
  }

  // Settings store readiness gate (2026-10-30). registerExtension() processes
  // `settings` IMMEDIATELY; if that happens before the official settings store
  // bootstrap ran, the store's load() throws "Setting values must be loaded
  // before any setting is registered" - an uncaught error that aborts the
  // bootstrap chain and leaves the whole UI blank. Wait for the store's
  // isReady signal first (see profiler.js for the full note).
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

  // Shared easing - identical to the homepage (--cdl-ease).
  const EASE = "cubic-bezier(.16,1,.3,1)";

  // WAAPI wrapper with the cdlCancelAfter contract (see module docstring).
  const animate = (el, keyframes, opts) => {
    if (!el) return false;
    try {
      const a = el.animate(keyframes, Object.assign({ easing: EASE, fill: "both" }, opts));
      if (opts && opts.cdlCancelAfter) {
        a.onfinish = () => { try { a.cancel(); } catch (e) { /* noop */ } };
      }
      return true;
    } catch (err) { return false; }
  };

  const htmlClasses = () => document.documentElement.classList;
  const reduceMotion = () =>
    !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  // Disable everything: gate class makes every pre-hide rule stop matching.
  function disableAll() {
    htmlClasses().add("cdl-motion-off");
    state.entranceDone = true;
    uninstallSidebarPanelAnimation();
  }

  // ------------------------------------------------------------- ripple ----
  function installRipple() {
    if (state.rippleInstalled || reduceMotion()) return;
    const RIPPLE_CSS = `
#cdl-ripple-layer{position:fixed;inset:0;pointer-events:none;z-index:999998;overflow:hidden}
.cdl-ripple{position:absolute;border-radius:50%;pointer-events:none;background:radial-gradient(circle,rgba(205,228,255,.95) 0%,rgba(170,205,250,0) 68%);will-change:transform,opacity}
`;
    try {
      const style = document.createElement("style");
      style.textContent = RIPPLE_CSS;
      document.head.appendChild(style);
      state.rippleLayer = document.createElement("div");
      state.rippleLayer.id = "cdl-ripple-layer";
      document.body.appendChild(state.rippleLayer);
    } catch (err) { return; }

    document.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || !state.enabled) return;
      const layer = state.rippleLayer;
      if (!layer) return;
      const size = 110;
      const r = document.createElement("span");
      r.className = "cdl-ripple";
      r.style.width = r.style.height = size + "px";
      r.style.left = (e.clientX - size / 2) + "px";
      r.style.top = (e.clientY - size / 2) + "px";
      layer.appendChild(r);
      const anim = r.animate(
        [{ transform: "scale(.25)", opacity: .5 }, { transform: "scale(1)", opacity: 0 }],
        { duration: 480, easing: "cubic-bezier(.2,.6,.3,1)" });
      anim.onfinish = () => r.remove();
      anim.oncancel = () => r.remove();
    }, { passive: true });

    state.rippleInstalled = true;
    console.log(TAG, "ripple installed");
  }

  function uninstallRipple() {
    if (state.rippleLayer) {
      try { state.rippleLayer.remove(); } catch (e) { /* noop */ }
      state.rippleLayer = null;
    }
  }

  // ------------------------------------- sidebar panel expand / collapse ----
  // The tab panel is a splitter panel (.side-bar-panel): expanding mounts a
  // .sidebar-content-container into it, collapsing removes the container and
  // flips the panel's width class. Expand = slide the content in; collapse =
  // a soft pulse on the icon rail as feedback. One MutationObserver on body
  // watches for both (childList for the mount, class attributes for the
  // width-class flip). The animations only touch opacity/transform of the
  // observer targets themselves - no class writes - so the observer can never
  // re-trigger itself.
  function installSidebarPanelAnimation() {
    if (state.sidebarObserver || reduceMotion()) return;
    try {
      const ob = new MutationObserver((muts) => {
        if (!state.enabled) return;
        for (const m of muts) {
          if (m.type === "attributes") {
            const cls = String(m.target.className || "");
            if (!cls.includes("side-bar-panel")) continue;
            if (!m.target.querySelector(".sidebar-content-container")) {
              // collapse: soft pulse on the icon rail as feedback
              const rail = document.querySelector(".side-tool-bar-container");
              if (rail) animate(rail,
                [{ opacity: 1 }, { opacity: .72 }, { opacity: 1 }],
                { duration: 260 });
            }
          } else {
            for (const n of m.addedNodes) {
              if (n.nodeType !== 1) continue;
              const c = String(n.className || "");
              if (c.includes("sidebar-content-container")) {
                animate(n,
                  [{ opacity: 0, transform: "translateX(-26px)" }, { opacity: 1, transform: "none" }],
                  { duration: 340 });
              }
            }
          }
        }
      });
      ob.observe(document.body, {
        childList: true, subtree: true, attributes: true, attributeFilter: ["class"],
      });
      state.sidebarObserver = ob;
      console.log(TAG, "sidebar panel animation armed");
    } catch (err) { /* noop */ }
  }

  function uninstallSidebarPanelAnimation() {
    if (state.sidebarObserver) {
      try { state.sidebarObserver.disconnect(); } catch (e) { /* noop */ }
      state.sidebarObserver = null;
    }
  }

  // ---------------------------------------------------- entrance choreography
  // Containers may still be empty at trigger time (content teleports in
  // later) - play their entrance only once they actually have children.
  const playWhenPopulated = (id, keyframes, opts) => {
    const el = document.getElementById(id);
    if (!el) return;
    const play = () => {
      el.classList.add("cdl-entered");
      animate(el, keyframes, opts);
    };
    if (el.children.length) play();
    else {
      const ob = new MutationObserver(() => {
        if (el.children.length) { ob.disconnect(); play(); }
      });
      ob.observe(el, { childList: true });
    }
  };

  function runEntrance() {
    if (state.entranceDone) return;
    state.entranceDone = true;
    if (reduceMotion() || !state.enabled) { disableAll(); return; }

    // Retire the pre-hide rules permanently once every animation has finished:
    // Vue re-renders of the sidebar rail reset className and would otherwise
    // wipe injected cdl-entered classes, leaving buttons invisible forever.
    setTimeout(() => htmlClasses().add("cdl-entrance-done"), 2600);

    playWhenPopulated("comfyui-body-top",
      [{ opacity: 0, transform: "translateY(-16px)" }, { opacity: 1, transform: "none" }],
      { duration: 550, cdlCancelAfter: true });
    playWhenPopulated("comfyui-body-bottom",
      [{ opacity: 0, transform: "translateY(16px)" }, { opacity: 1, transform: "none" }],
      { duration: 500, delay: 300, cdlCancelAfter: true });
    playWhenPopulated("comfyui-body-right",
      [{ opacity: 0, transform: "translateX(20px)" }, { opacity: 1, transform: "none" }],
      { duration: 500, delay: 380, cdlCancelAfter: true });

    // Sidebar rail: the container plus each tab/button staggered. The rail may
    // not exist yet at trigger time - retry.
    const trySidebar = (retries) => {
      const bar = document.querySelector(".side-tool-bar-container");
      const tabs = document.querySelectorAll(
        ".side-tool-bar-container .sidebar-item-group > *");
      if (!bar || !tabs.length) {
        if (retries > 0) setTimeout(() => trySidebar(retries - 1), 120);
        return;
      }
      bar.classList.add("cdl-entered");
      animate(bar,
        [{ opacity: 0, transform: "translateX(-24px)" }, { opacity: 1, transform: "none" }],
        { duration: 600, cdlCancelAfter: true });
      tabs.forEach((el, i) => {
        el.classList.add("cdl-entered");
        animate(el,
          [{ opacity: 0, transform: "translateX(-14px) scale(.9)" }, { opacity: 1, transform: "none" }],
          { duration: 500, delay: 150 + i * 60, cdlCancelAfter: true });
      });
      bar.querySelectorAll(".mt-auto > *").forEach((el, i) => {
        el.classList.add("cdl-entered");
        animate(el,
          [{ opacity: 0, transform: "translateX(-14px) scale(.9)" }, { opacity: 1, transform: "none" }],
          { duration: 450, delay: 480 + i * 60, cdlCancelAfter: true });
      });
      console.log(TAG, "entrance: sidebar animated (tabs:", tabs.length, ")");
    };
    trySidebar(25);

    // Canvas nodes: diagonal sweep by position, opacity ONLY (see docstring).
    setTimeout(() => {
      let tries = 0;
      const sweep = () => {
        const nodes = Array.from(document.querySelectorAll("[data-node-id]"))
          .sort((a, b) => {
            const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
            return (ra.top - rb.top) || (ra.left - rb.left);
          });
        if (!nodes.length) {
          if (++tries < 40) setTimeout(sweep, 120);
          else htmlClasses().add("cdl-nodes-in");
          return;
        }
        nodes.forEach((el, i) => animate(el,
          [{ opacity: 0 }, { opacity: 1 }],
          { duration: 450, delay: i * 40 }));
        setTimeout(() => htmlClasses().add("cdl-nodes-in"),
          600 + nodes.length * 40);
        // Canvas (grid + links) fades in mid-sweep, overlapping the rest.
        setTimeout(() => htmlClasses().add("cdl-canvas-in"),
          300 + nodes.length * 20);
        console.log(TAG, "entrance: nodes faded in (", nodes.length, ")");
      };
      sweep();
    }, 150);
    // (The baked style block's 9s failsafe is the backstop for this path.)
  }

  // Force-reveal now (settings toggled off mid-entrance).
  function forceReveal() {
    htmlClasses().add("cdl-entrance-done");
    htmlClasses().add("cdl-nodes-in");
    htmlClasses().add("cdl-canvas-in");
  }

  // -------------------------------------------------------------- boot ----
  function boot() {
    // Splash driver first: it needs no comfyAPI and must observe from the
    // earliest possible moment (this module runs before Vue mounts).
    initSplashProgress();

    const start = (force) => {
      const app = getApp();
      if (!app || !app.extensionManager) return false;
      if (!force && !settingsStoreReady(app)) return false;

      // Best-effort locale detection before settings register (same pattern
      // as profiler.js; init() below re-detects via get()).
      try {
        const s = app.extensionManager.setting;
        if (s && typeof s.get === "function") {
          const v = s.get("Comfy.Locale");
          if (typeof v === "string" && v) locale = v.split("-")[0];
        }
      } catch (e) { /* settings store not ready: setup() re-reads */ }

      app.registerExtension({
        name: "ComfyDL.Motion",
        async setup() {
          // Re-detect locale now that the settings store is live.
          try {
            const v = app.extensionManager.setting.get("Comfy.Locale");
            if (typeof v === "string" && v) locale = v.split("-")[0];
          } catch (e) { /* keep default */ }

          let enabled = true;
          try {
            enabled = app.extensionManager.setting.get(SETTING_ENABLED) !== false;
          } catch (e) { /* default true */ }
          state.enabled = enabled;

          if (!enabled || reduceMotion()) {
            disableAll();
            // The splash driver is disabled-aware: finishSplash() will skip
            // the entrance and force-reveal when the falling edge arrives.
            console.log(TAG, enabled ? "disabled (reduced motion)" : "disabled (setting)");
          } else {
            installRipple();
            installSidebarPanelAnimation();
            if (!state.splashDriverActive) runEntrance(); // fallback, no splash
          }
          // NOTE: runEntrance() is NOT called here. Extension setup runs
          // while the official splash is still up (spinner=true); the
          // entrance must start when the splash begins dissolving, which is
          // the spinner falling edge handled by the splash driver's
          // finishSplash() (see initSplashProgress).
        },
        settings: [{
          id: SETTING_ENABLED,
          name: t("setting_enabled"),
          tooltip: t("setting_enabled_tooltip"),
          category: ["ComfyDL", "Motion"],
          type: "boolean",
          defaultValue: true,
          onChange: (newVal) => {
            state.enabled = newVal !== false;
            if (!state.enabled) {
              uninstallRipple();
              uninstallSidebarPanelAnimation();
              disableAll();
              forceReveal();
              console.log(TAG, "disabled via settings");
            } else {
              installRipple();
              installSidebarPanelAnimation();
              console.log(TAG, "enabled via settings (entrance already played or will play once)");
            }
          },
        }],
      });
      return true;
    };
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
