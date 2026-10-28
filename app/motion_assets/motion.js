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
  };

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
    const start = () => {
      const app = getApp();
      if (!app || !app.extensionManager) return false;

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
            console.log(TAG, enabled ? "disabled (reduced motion)" : "disabled (setting)");
          } else {
            installRipple();
            installSidebarPanelAnimation();
            runEntrance();
          }
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
    if (!start()) waitFor(start, 30000, 200);
  }

  boot();
})();
