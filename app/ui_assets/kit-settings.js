/**
 * ComfyDL UI kit settings (U stick 2 follow-up).
 *
 * The accent-theme selector is NOT a performance control, so it no longer
 * lives in the Optimizer panel.  This module registers it in the official
 * Settings > ComfyDL group through the public extension API; no Vue file or
 * upstream rendering code is patched.  Choosing a hue writes data-theme on
 * <html>, which re-derives every --colour-* / --ui-accent from themes.css.
 */
(() => {
  "use strict";

  const TAG = "[ComfyDL.UIKit.Settings]";
  const SETTING_ID = "ComfyDL.UIKit.Theme";
  const LEGACY_KEY = "ComfyDL.P6b.theme";
  const HOST = "host";
  const THEMES = ["blue", "aquamarine", "green", "lime", "orange",
    "pink", "plum", "purple", "red"];

  function log(...args) {
    console.log(TAG, ...args);
  }

  function applyTheme(value) {
    const theme = THEMES.includes(value) ? value : HOST;
    if (theme === HOST) {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.dataset.theme = theme;
    }
    log("theme applied:", theme,
      "html[data-theme]=" + String(document.documentElement.dataset.theme));
    return theme;
  }

  function legacyValue() {
    try {
      const raw = localStorage.getItem(LEGACY_KEY);
      return THEMES.includes(raw) ? raw : null;
    } catch (e) { return null; }
  }

  function settingsStoreReady(app) {
    try {
      const root = document.getElementById("vue-app");
      const vueApp = root && root.__vue_app__;
      const pinia = vueApp && vueApp.config && vueApp.config.globalProperties.$pinia;
      const store = pinia && pinia._s && pinia._s.get("setting");
      if (!store) return false;
      return store.isReady ? store.isReady() === true : false;
    } catch (e) { return false; }
  }

  function register(app) {
    const store = app.extensionManager?.setting;
    const migrated = legacyValue();
    const initial = (store && typeof store.get === "function"
      ? store.get(SETTING_ID) : null) || migrated || HOST;

    app.registerExtension({
      name: "ComfyDL.UIKit.Settings",
      settings: [{
        id: SETTING_ID,
        name: "UI kit accent theme",
        tooltip: "Hue used by the ComfyDL UI kit components. 'host' follows the host ComfyUI theme.",
        category: ["ComfyDL"],
        type: "combo",
        options: [HOST, ...THEMES],
        defaultValue: HOST,
        onChange: (value) => applyTheme(value),
      }],
    });

    applyTheme(initial);
    if (migrated && store && typeof store.set === "function") {
      try { store.set(SETTING_ID, migrated); } catch (e) { /* keep localStorage value */ }
    }
    log("registered " + SETTING_ID + " in Settings > ComfyDL");
  }

  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    const app = window.comfyAPI?.app?.app;
    if (app && settingsStoreReady(app)) {
      clearInterval(timer);
      try { register(app); } catch (e) { log("registration failed:", String(e)); }
      return;
    }
    if (tries > 200) {
      clearInterval(timer);
      log("settings store never became ready; theme control unavailable");
    }
  }, 100);
})();
