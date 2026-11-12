/* ComfyDL Observation panel (M-D P6, CRAFT 1).
 *
 * An INDEPENDENT sidebar tab, deliberately outside the App Panel: advanced
 * observation is opt-in and the main canvas stays lean for everyday users.
 * This craft delivers the resource monitor only; execution badges / the runs
 * ledger come in CRAFT 2.
 *
 * Data path: the existing single `cdl.spine` WS channel + the two spine REST
 * endpoints. No new thread, no new WS channel, no client-side polling timer:
 * the spine switch stays OFF by default and is started only when the user
 * opts in from this tab.
 *
 * Conventions reused from the mature panels (never invent a second UI language):
 *   - registerSidebarTab + render() cleanup   (crashsite.js:476-500)
 *   - cdlp-card / cdlp-card-title vocabulary (profiler.css)
 *   - JS-local en/zh dictionary              (official /i18n is never fetched)
 *   - three motion gates                    (profiler.css:647-660)
 *   - SVG skeleton built once, only polyline `points` updated per tick
 */
(() => {
  "use strict";

  const CSS_URL = "/comfydl/obs/obs.css";
  const TAB_ID = "comfydl-obs";
  const SPINE_URL = "/comfydl/spine";
  const SNAPSHOT_URL = "/comfydl/spine/snapshot";
  const MAX_POINTS = 60;
  const SVG_NS = "http://www.w3.org/2000/svg";

  // The official frontend only auto-loads .js files; CSS must be injected.
  if (!document.querySelector("link[data-cdl-obs-css]")) {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = CSS_URL;
    link.dataset.cdlObsCss = "1";
    document.head.appendChild(link);
  }

  const I18N = {
    en: {
      tab_title: "Observation",
      tab_tooltip: "Live CPU / RAM / VRAM monitor (ComfyDL spine)",
      card_title: "System resources",
      cpu: "CPU", ram: "RAM", vram: "VRAM",
      off_hint: "Monitoring is off. The sampling thread only starts when you enable it.",
      enable: "Enable monitoring",
      enabling: "enabling...",
      badge_on: "live", badge_off: "off",
      waiting: "Waiting for the first sample...",
      no_gpu: "No CUDA device - VRAM unavailable.",
      schema: "schema v",
    },
    zh: {
      tab_title: "观测",
      tab_tooltip: "实时 CPU / 内存 / 显存监控（ComfyDL spine）",
      card_title: "系统资源",
      cpu: "CPU", ram: "内存", vram: "显存",
      off_hint: "监控未开启。采样线程只会在你启用后启动。",
      enable: "启用监控",
      enabling: "正在启用...",
      badge_on: "运行中", badge_off: "已关闭",
      waiting: "正在等待第一个采样...",
      no_gpu: "无 CUDA 设备，显存数据不可用。",
      schema: "协议 v",
    },
  };

  let locale = "en";
  function t(key) {
    const pack = I18N[locale] || I18N.en;
    return pack[key] || I18N.en[key] || key;
  }

  function getApp() { return window.comfyAPI?.app?.app || null; }

  function getApi(app) {
    try {
      const emApi = app?.extensionManager?.api;
      if (emApi && typeof emApi.fetchApi === "function") return emApi;
    } catch (e) { /* fall through to the ComfyApp instance */ }
    try {
      const api = app?.api;
      if (api && typeof api.fetchApi === "function") return api;
    } catch (e) { /* not ready yet */ }
    return null;
  }

  function getEventApi(app) {
    const candidates = [app?.extensionManager?.api, app?.api,
                        window.comfyAPI?.api?.api];
    for (const api of candidates) {
      if (api && typeof api.addEventListener === "function") return api;
    }
    return null;
  }

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  const state = {
    panelEl: null,
    eventApi: null,
    handler: null,
    enabled: false,
    sample: null,
    hist: { cpu: [], ram: [], vram: [] },
  };

  function clampPct(value) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
  }

  function svg(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  }

  function buildSkeleton(el) {
    el.innerHTML = "";
    el.classList.add("cdlp-panel", "obs-host");
    const card = document.createElement("div");
    card.className = "cdlp-card obs-entrance";
    card.innerHTML =
      '<div class="cdlp-card-title">' +
        '<span class="cdlp-collapse-label">' + esc(t("card_title")) + "</span>" +
        '<span class="cdlp-title-right"><span class="obs-badge" data-obs="badge">' +
          esc(t("badge_off")) + "</span></span>" +
      "</div>" +
      '<div class="cdlp-card-body">' +
        '<div class="cdlp-hint obs-off" data-obs="hint">' + esc(t("off_hint")) + "</div>" +
        '<button type="button" class="obs-enable" data-obs="enable">' +
          esc(t("enable")) + "</button>" +
        '<div class="obs-rows" data-obs="rows" hidden>' +
          '<div class="obs-row"><span>' + esc(t("cpu")) +
            '</span><b data-obs="cpu">0%</b></div>' +
          '<div class="obs-row"><span>' + esc(t("ram")) +
            '</span><b data-obs="ram">0%</b></div>' +
          '<div class="obs-row"><span>' + esc(t("vram")) +
            '</span><b data-obs="vram">—</b></div>' +
          '<div class="obs-chart-wrap" data-obs="chart-wrap"></div>' +
          '<div class="cdlp-hint" data-obs="meta"></div>' +
        "</div>" +
      "</div>";
    el.appendChild(card);

    const chart = svg("svg", {
      class: "obs-chart", viewBox: "0 0 100 40",
      preserveAspectRatio: "none", "aria-hidden": "true",
    });
    for (const key of ["cpu", "ram", "vram"]) {
      chart.appendChild(svg("polyline", {
        "data-obs": key + "-line", class: "obs-line obs-line-" + key,
        points: "", fill: "none", "vector-effect": "non-scaling-stroke",
      }));
    }
    el.querySelector('[data-obs="chart-wrap"]').appendChild(chart);

    el.querySelector('[data-obs="enable"]').addEventListener("click", enableSpine);
    el.__cdlObsRefs = {
      badge: el.querySelector('[data-obs="badge"]'),
      hint: el.querySelector('[data-obs="hint"]'),
      enable: el.querySelector('[data-obs="enable"]'),
      rows: el.querySelector('[data-obs="rows"]'),
      cpu: el.querySelector('[data-obs="cpu"]'),
      ram: el.querySelector('[data-obs="ram"]'),
      vram: el.querySelector('[data-obs="vram"]'),
      meta: el.querySelector('[data-obs="meta"]'),
      lines: {
        cpu: el.querySelector('[data-obs="cpu-line"]'),
        ram: el.querySelector('[data-obs="ram-line"]'),
        vram: el.querySelector('[data-obs="vram-line"]'),
      },
    };
  }

  function linePoints(values) {
    if (!values.length) return "";
    const step = 100 / Math.max(1, MAX_POINTS - 1);
    return values.map((v, i) =>
      (i * step).toFixed(1) + "," + (40 - (v / 100) * 38 - 1).toFixed(1)
    ).join(" ");
  }

  function gpuUsedPct(gpu) {
    if (!gpu || !gpu.available || Number(gpu.total_gib) <= 0) return 0;
    return clampPct((Number(gpu.used_gib) / Number(gpu.total_gib)) * 100);
  }

  function pushSample(sample) {
    if (!sample || typeof sample !== "object") return;
    const values = {
      cpu: clampPct(sample.cpu?.percent),
      ram: clampPct(sample.ram?.percent),
      vram: gpuUsedPct(sample.gpu),
    };
    for (const key of Object.keys(values)) {
      state.hist[key].push(values[key]);
      if (state.hist[key].length > MAX_POINTS) state.hist[key].shift();
    }
    state.sample = sample;
  }

  function renderPanel() {
    const el = state.panelEl;
    if (!el || !el.__cdlObsRefs) return;
    const r = el.__cdlObsRefs;
    r.badge.textContent = state.enabled ? t("badge_on") : t("badge_off");
    r.badge.classList.toggle("obs-badge-on", state.enabled);
    if (!state.enabled) {
      r.rows.hidden = true;
      r.hint.hidden = false;
      r.hint.textContent = t("off_hint");
      r.enable.hidden = false;
      return;
    }
    r.enable.hidden = true;
    if (!state.sample) {
      r.rows.hidden = true;
      r.hint.hidden = false;
      r.hint.textContent = t("waiting");
      return;
    }
    r.hint.hidden = true;
    r.rows.hidden = false;
    const s = state.sample;
    const gpu = s.gpu;
    r.cpu.textContent = clampPct(s.cpu?.percent).toFixed(0) + "%";
    r.ram.textContent = clampPct(s.ram?.percent).toFixed(0) + "%";
    r.vram.textContent = gpu && gpu.available && Number(gpu.total_gib) > 0
      ? Number(gpu.used_gib).toFixed(1) + " / " +
        Number(gpu.total_gib).toFixed(1) + " GiB"
      : "—";
    for (const key of Object.keys(r.lines)) {
      r.lines[key].setAttribute("points", linePoints(state.hist[key]));
    }
    r.meta.textContent = t("schema") + (s.schema_version ?? "?") +
      (gpu && gpu.available ? "" : " · " + t("no_gpu"));
  }

  async function refreshState() {
    const api = getApi(getApp());
    if (!api || typeof api.fetchApi !== "function") return;
    try {
      const resp = await api.fetchApi(SPINE_URL);
      if (resp.ok) {
        const data = await resp.json();
        state.enabled = !!(
          data?.switch?.enabled === true || data?.running === true);
      }
    } catch (e) { /* keep the last known state; the tab is a courtesy */ }
    if (!state.enabled) {
      state.sample = null;
      renderPanel();
      return;
    }
    renderPanel();
    try {
      const resp = await api.fetchApi(SNAPSHOT_URL);
      if (!resp.ok) return;
      const data = await resp.json();
      // The endpoint returns the snapshot itself (older builds wrapped it).
      const snap = data?.snapshot ?? (typeof data?.schema_version === "number" ? data : null);
      if (snap) { pushSample(snap); renderPanel(); }
    } catch (e) { /* the first WS tick will fill it */ }
  }

  async function enableSpine() {
    const api = getApi(getApp());
    if (!api || typeof api.fetchApi !== "function") return;
    const btn = state.panelEl?.__cdlObsRefs?.enable;
    if (btn) { btn.disabled = true; btn.textContent = t("enabling"); }
    try {
      await api.fetchApi(SPINE_URL, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: true, rate_s: 1.0 }),
      });
    } catch (e) { /* refreshState will fall back to the off hint */ }
    await refreshState();
    if (btn && !state.enabled) {
      btn.disabled = false;
      btn.textContent = t("enable");
    }
  }

  function registerSidebar() {
    const app = getApp();
    const em = app && app.extensionManager;
    if (!em || typeof em.registerSidebarTab !== "function") return false;
    em.registerSidebarTab({
      id: TAB_ID,
      icon: "pi pi-chart-line",
      title: t("tab_title"),
      tooltip: t("tab_tooltip"),
      type: "custom",
      render: (el) => {
        state.panelEl = el;
        buildSkeleton(el);
        const eventApi = getEventApi(app);
        if (eventApi) {
          state.eventApi = eventApi;
          state.handler = (event) => {
            pushSample(event?.detail ?? event?.data ?? null);
            renderPanel();
          };
          eventApi.addEventListener("cdl.spine", state.handler);
        }
        renderPanel();
        refreshState();
        // Official contract: render() may return a cleanup function.
        return () => {
          if (state.eventApi && state.handler &&
              typeof state.eventApi.removeEventListener === "function") {
            state.eventApi.removeEventListener("cdl.spine", state.handler);
          }
          state.eventApi = null;
          state.handler = null;
          state.panelEl = null;
          state.sample = null;
          for (const key of Object.keys(state.hist)) state.hist[key] = [];
        };
      },
    });
    return true;
  }

  function boot() {
    const app = getApp();
    if (!app || typeof app.registerExtension !== "function") return false;
    const em = app.extensionManager;
    if (!em || typeof em.registerSidebarTab !== "function") return false;
    // Registering before the settings store is ready blanks the host UI.
    try {
      const setting = em.setting;
      if (setting && typeof setting.isReady === "function" &&
          !setting.isReady()) return false;
    } catch (e) { /* no readiness gate in this frontend build */ }
    app.registerExtension({
      name: "ComfyDL.Observation",
      async setup() {
        try {
          const v = em.setting.get("Comfy.Locale");
          if (typeof v === "string" && v) locale = v.split("-")[0];
        } catch (e) { /* keep the default locale */ }
        registerSidebar();
      },
    });
    return true;
  }

  const bootTimer = setInterval(() => {
    if (boot()) clearInterval(bootTimer);
  }, 250);
})();
