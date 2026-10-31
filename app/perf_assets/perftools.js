/**
 * ComfyDL_UI: perf-tools switches (dual-track M-B).
 *
 * Surfaces the switch_registry's performance features in the official
 * Settings > ComfyDL panel:
 *
 *   - ComfyDL.Perf.AttentionSelect  ->  comfydl.attention_select
 *   - ComfyDL.Perf.Compile          ->  comfydl.compile
 *
 * The settings store holds the UI value; the backend switch_registry holds
 * the truth. setup() pulls GET /comfydl/switches once and syncs the local
 * store (guarded so the sync does not loop through onChange), and every user
 * toggle PATCHes /comfydl/switches/<id> - the response description is the
 * authoritative state, so a refused toggle (404 / 409 hard-disabled) snaps
 * the UI back instead of lying.
 */

/* eslint-disable */
(function () {
  "use strict";

  const TAG = "[ComfyDL.PerfTools]";
  const SWITCHES_URL = "/comfydl/switches";

  // Setting id -> backend switch id + display strings (en/zh, same idiom
  // as motion.js: no dependency on the frontend locale chunks).
  const PERF_SWITCHES = [
    {
      settingId: "ComfyDL.Perf.AttentionSelect",
      switchId: "comfydl.attention_select",
      name: {
        en: "ComfyDL attention auto-select",
        zh: "ComfyDL attention 自动选优",
      },
      tooltip: {
        en: "Hot-disables the Attention Optimizer node: disabled, it passes the MODEL through unchanged and never benchmarks.",
        zh: "热关闭 Attention Optimizer 节点：关闭后 MODEL 原样透传，不再做基准测试。",
      },
    },
    {
      settingId: "ComfyDL.Perf.Compile",
      switchId: "comfydl.compile",
      name: {
        en: "ComfyDL torch.compile",
        zh: "ComfyDL torch.compile",
      },
      tooltip: {
        en: "Hot-disables the Compile Model node: disabled, compile requests return the original module untouched.",
        zh: "热关闭 Compile Model 节点：关闭后编译请求直接返回原模块，不做任何包装。",
      },
    },
  ];

  let locale = "en";
  const t = (table) => (table[locale] || table.en);

  function getApp() {
    return window.comfyAPI?.app?.app || null;
  }

  function getApi(app) {
    try {
      return app.extensionManager?.api || null;
    } catch (e) {
      return null;
    }
  }

  async function fetchSwitches(app) {
    const api = getApi(app);
    if (!api || typeof api.fetchApi !== "function") return null;
    try {
      const resp = await api.fetchApi(SWITCHES_URL);
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.warn(TAG, "switch list unavailable", e);
      return null;
    }
  }

  async function patchSwitch(app, switchId, enabled) {
    const api = getApi(app);
    if (!api || typeof api.fetchApi !== "function") return null;
    try {
      const resp = await api.fetchApi(`${SWITCHES_URL}/${switchId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !!enabled }),
      });
      if (!resp.ok) {
        console.warn(TAG, `PATCH ${switchId} -> ${resp.status}`);
        return null;
      }
      return await resp.json();
    } catch (e) {
      console.warn(TAG, `PATCH ${switchId} failed`, e);
      return null;
    }
  }

  // Settings store readiness gate - the exact 2026-10-30 lesson from
  // motion.js: registerExtension() processes `settings` IMMEDIATELY; if that
  // happens before the official settings store bootstrap ran, the store's
  // load() throws "Setting values must be loaded before any setting is
  // registered" - an uncaught error that aborts the bootstrap chain and
  // leaves the whole UI blank.  Never register before st.isReady.
  function settingsStoreReady(_app) {
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

  function boot() {
    const app = getApp();
    if (!app || !app.extensionManager) return false;
    if (!settingsStoreReady(app)) return false;

    // Locale before the settings definitions render (same pattern as
    // motion.js / profiler.js).
    try {
      const v = app.extensionManager.setting?.get("Comfy.Locale");
      if (typeof v === "string" && v) locale = v.split("-")[0];
    } catch (e) { /* keep default */ }

    app.registerExtension({
      name: "ComfyDL.PerfTools",
      async setup() {
        try {
          const v = app.extensionManager.setting.get("Comfy.Locale");
          if (typeof v === "string" && v) locale = v.split("-")[0];
        } catch (e) { /* keep default */ }

        // Pull the backend truth once and sync the local store. The guard
        // flag keeps this sync from re-entering onChange (which would PATCH
        // the value we just read - harmless but noisy).
        const payload = await fetchSwitches(app);
        if (!payload || !Array.isArray(payload.switches)) return;
        const store = app.extensionManager.setting;
        const states = Object.fromEntries(
          payload.switches.map((s) => [s.id, s]));
        for (const def of PERF_SWITCHES) {
          const desc = states[def.switchId];
          if (!desc) continue;
          try {
            if (store.get(def.settingId) !== desc.enabled) {
              store.set(def.settingId, desc.enabled);
            }
          } catch (e) { /* store not writable: leave the default */ }
        }
      },
      settings: PERF_SWITCHES.map((def) => ({
        id: def.settingId,
        name: t(def.name),
        tooltip: t(def.tooltip),
        category: ["ComfyDL", "Perf"],
        type: "boolean",
        defaultValue: true,
        onChange: (newVal) => {
          // Fire-and-forget: the authoritative state comes back in the
          // PATCH response; on refusal (null) snap the store back so the
          // toggle never lies.
          patchSwitch(app, def.switchId, newVal).then((body) => {
            const enabled = body?.switch?.enabled;
            if (typeof enabled === "boolean" && enabled !== newVal) {
              try {
                app.extensionManager.setting.set(def.settingId, enabled);
              } catch (e) { /* leave as-is */ }
            }
          });
        },
      })),
    });
    return true;
  }

  // Retry until the app object exists AND the settings store is ready (the
  // store may still be bootstrapping long after window.comfyAPI appears -
  // that window is exactly where the blank-UI crash lives).
  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    if (boot() || tries > 200) clearInterval(timer);
  }, 100);
})();
