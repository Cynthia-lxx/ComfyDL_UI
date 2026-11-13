/* ComfyDL Observation overlay (M-D P6, 2026-11-13 shape correction).
 *
 * A minimap-sized FLOATING widget, not a sidebar tab: observation is a small
 * always-available readout and must not compete with the main canvas. It
 * supports collapse + hide, both persisted in localStorage, and defaults to
 * collapsed (same convention as the App Panel's legacy dock and the built-in
 * minimap).
 *
 * De-duplication: the Crystools status text at the bottom-left already shows
 * instantaneous CPU/RAM/VRAM percentages (and, from other sources, FPS). This
 * widget therefore shows what that text cannot: 60-point trends, absolute VRAM
 * GiB, this process's own CPU, and the spine schema / switch state.
 *
 * Data layer is unchanged from CRAFT 1: the single `cdl.spine` WS channel, the
 * two spine REST endpoints, no new thread / channel / polling timer, and the
 * switch stays OFF by default until the user opts in.
 */
(() => {
  "use strict";

  const CSS_URL = "/comfydl/obs/obs.css";
  const OVERLAY_ID = "cdl-obs-overlay";
  const REOPEN_ID = "cdl-obs-reopen";
  const SPINE_URL = "/comfydl/spine";
  const SNAPSHOT_URL = "/comfydl/spine/snapshot";
  const COLLAPSE_KEY = "ComfyDL.Obs.Collapsed";
  const HIDDEN_KEY = "ComfyDL.Obs.Hidden";
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
      title: "Observation",
      tooltip: "Live resource trends (ComfyDL spine)",
      off_hint: "Monitoring is off. The sampling thread starts only when enabled.",
      enable: "Enable monitoring",
      enabling: "enabling...",
      badge_on: "live", badge_off: "off",
      waiting: "Waiting for the first sample...",
      collapse: "Collapse", expand: "Expand", hide: "Hide",
      reopen: "Observation",
      trend: "60-sample trend",
      cpu_trend: "CPU", ram_trend: "RAM", vram_trend: "VRAM",
      proc: "Process CPU", vram: "VRAM", no_gpu: "No CUDA device",
      schema: "schema v",
      dedup: "Instantaneous values are in the bottom-left status text.",
    },
    zh: {
      title: "观测",
      tooltip: "实时资源趋势（ComfyDL spine）",
      off_hint: "监控未开启。采样线程只会在启用后启动。",
      enable: "启用监控",
      enabling: "正在启用...",
      badge_on: "运行中", badge_off: "已关闭",
      waiting: "正在等待第一个采样...",
      collapse: "收起", expand: "展开", hide: "隐藏",
      reopen: "观测",
      trend: "60 点趋势",
      cpu_trend: "CPU", ram_trend: "内存", vram_trend: "显存",
      proc: "进程 CPU", vram: "显存", no_gpu: "无 CUDA 设备",
      schema: "协议 v",
      dedup: "瞬时数值见左下角状态栏，此处显示趋势与绝对值。",
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

  function loadBool(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : raw === "1";
    } catch (e) { return fallback; }
  }

  function saveBool(key, value) {
    try { localStorage.setItem(key, value ? "1" : "0"); } catch (e) { /* private mode */ }
  }

  const state = {
    overlayEl: null,
    reopenEl: null,
    refs: null,
    eventApi: null,
    handler: null,
    enabled: false,
    sample: null,
    collapsed: loadBool(COLLAPSE_KEY, true),
    hidden: loadBool(HIDDEN_KEY, false),
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

  function buildOverlay() {
    const root = document.createElement("div");
    root.id = OVERLAY_ID;
    root.className = "cdl-obs-overlay obs-entrance";
    root.title = t("tooltip");
    root.innerHTML =
      '<div class="cdl-obs-head">' +
        '<span class="cdl-obs-title">' + esc(t("title")) + "</span>" +
        '<span class="cdl-obs-badge" data-obs="badge">' + esc(t("badge_off")) + "</span>" +
        '<button type="button" class="cdl-obs-btn" data-obs="collapse">+</button>' +
        '<button type="button" class="cdl-obs-btn" data-obs="hide">×</button>' +
      "</div>" +
      '<div class="cdl-obs-body">' +
        '<div class="cdl-obs-off" data-obs="hint">' + esc(t("off_hint")) + "</div>" +
        '<button type="button" class="cdl-obs-enable" data-obs="enable">' +
          esc(t("enable")) + "</button>" +
        '<div class="cdl-obs-rows" data-obs="rows" hidden>' +
          '<div class="cdl-obs-trend">' + esc(t("trend")) + "</div>" +
          '<div class="cdl-obs-chart-wrap" data-obs="chart-wrap"></div>' +
          '<div class="cdl-obs-legend">' +
            '<span><i class="cdl-obs-dot cdl-obs-cpu"></i>' + esc(t("cpu_trend")) + "</span>" +
            '<span><i class="cdl-obs-dot cdl-obs-ram"></i>' + esc(t("ram_trend")) + "</span>" +
            '<span><i class="cdl-obs-dot cdl-obs-vram"></i>' + esc(t("vram_trend")) + "</span>" +
          "</div>" +
          '<div class="cdl-obs-meta" data-obs="proc"></div>' +
          '<div class="cdl-obs-meta" data-obs="vram"></div>' +
          '<div class="cdl-obs-dedup">' + esc(t("dedup")) + "</div>" +
          '<div class="cdl-obs-schema" data-obs="meta"></div>' +
        "</div>" +
      "</div>";
    document.body.appendChild(root);

    const reopen = document.createElement("button");
    reopen.id = REOPEN_ID;
    reopen.type = "button";
    reopen.className = "cdl-obs-reopen";
    reopen.textContent = t("reopen");
    reopen.title = t("tooltip");
    document.body.appendChild(reopen);

    const chart = svg("svg", {
      class: "cdl-obs-chart", viewBox: "0 0 100 40",
      preserveAspectRatio: "none", "aria-hidden": "true",
    });
    for (const key of ["cpu", "ram", "vram"]) {
      chart.appendChild(svg("polyline", {
        "data-obs": key + "-line", class: "cdl-obs-line cdl-obs-" + key,
        points: "", fill: "none", "vector-effect": "non-scaling-stroke",
      }));
    }
    root.querySelector('[data-obs="chart-wrap"]').appendChild(chart);

    root.querySelector('[data-obs="collapse"]').addEventListener("click", toggleCollapse);
    root.querySelector('[data-obs="hide"]').addEventListener("click", hideOverlay);
    root.querySelector('[data-obs="enable"]').addEventListener("click", enableSpine);
    reopen.addEventListener("click", showOverlay);

    state.overlayEl = root;
    state.reopenEl = reopen;
    state.refs = {
      badge: root.querySelector('[data-obs="badge"]'),
      hint: root.querySelector('[data-obs="hint"]'),
      enable: root.querySelector('[data-obs="enable"]'),
      rows: root.querySelector('[data-obs="rows"]'),
      collapse: root.querySelector('[data-obs="collapse"]'),
      proc: root.querySelector('[data-obs="proc"]'),
      vram: root.querySelector('[data-obs="vram"]'),
      meta: root.querySelector('[data-obs="meta"]'),
      lines: {
        cpu: root.querySelector('[data-obs="cpu-line"]'),
        ram: root.querySelector('[data-obs="ram-line"]'),
        vram: root.querySelector('[data-obs="vram-line"]'),
      },
    };
    applyShape();
  }

  function applyShape() {
    const root = state.overlayEl;
    if (!root) return;
    root.dataset.collapsed = state.collapsed ? "1" : "0";
    root.dataset.hidden = state.hidden ? "1" : "0";
    if (state.reopenEl) state.reopenEl.hidden = !state.hidden;
    const btn = state.refs?.collapse;
    if (btn) {
      btn.textContent = state.collapsed ? "+" : "−";
      btn.title = state.collapsed ? t("expand") : t("collapse");
      btn.setAttribute("aria-label",
        state.collapsed ? t("expand") : t("collapse"));
    }
  }

  function toggleCollapse() {
    state.collapsed = !state.collapsed;
    saveBool(COLLAPSE_KEY, state.collapsed);
    applyShape();
  }

  function hideOverlay() {
    state.hidden = true;
    saveBool(HIDDEN_KEY, true);
    applyShape();
  }

  function showOverlay() {
    state.hidden = false;
    state.collapsed = false;
    saveBool(HIDDEN_KEY, false);
    saveBool(COLLAPSE_KEY, false);
    applyShape();
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
    if (!state.refs) return;
    const r = state.refs;
    r.badge.textContent = state.enabled ? t("badge_on") : t("badge_off");
    r.badge.classList.toggle("cdl-obs-badge-on", state.enabled);
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
    r.proc.textContent = t("proc") + "  " +
      clampPct(s.proc?.cpu_percent).toFixed(1) + "%";
    r.vram.textContent = t("vram") + "  " + (
      gpu && gpu.available && Number(gpu.total_gib) > 0
        ? Number(gpu.used_gib).toFixed(1) + " / " +
          Number(gpu.total_gib).toFixed(1) + " GiB"
        : t("no_gpu"));
    for (const key of Object.keys(r.lines)) {
      r.lines[key].setAttribute("points", linePoints(state.hist[key]));
    }
    r.meta.textContent = t("schema") + (s.schema_version ?? "?");
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
    } catch (e) { /* keep the last known state; the widget is a courtesy */ }
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
      const snap = data?.snapshot ??
        (typeof data?.schema_version === "number" ? data : null);
      if (snap) { pushSample(snap); renderPanel(); }
    } catch (e) { /* the first WS tick will fill it */ }
  }

  async function enableSpine() {
    const api = getApi(getApp());
    if (!api || typeof api.fetchApi !== "function") return;
    const btn = state.refs?.enable;
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

  function mountOverlay() {
    if (document.getElementById(OVERLAY_ID)) return true; // idempotent
    buildOverlay();
    const eventApi = getEventApi(getApp());
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
    return true;
  }

  function destroyOverlay() {
    if (state.eventApi && state.handler &&
        typeof state.eventApi.removeEventListener === "function") {
      state.eventApi.removeEventListener("cdl.spine", state.handler);
    }
    state.eventApi = null;
    state.handler = null;
    if (state.overlayEl?.parentNode) state.overlayEl.parentNode.removeChild(state.overlayEl);
    if (state.reopenEl?.parentNode) state.reopenEl.parentNode.removeChild(state.reopenEl);
    state.overlayEl = null;
    state.reopenEl = null;
    state.refs = null;
    state.sample = null;
    for (const key of Object.keys(state.hist)) state.hist[key] = [];
  }

  function boot() {
    const app = getApp();
    if (!app || typeof app.registerExtension !== "function") return false;
    // Registering before the settings store is ready blanks the host UI.
    try {
      const setting = app.extensionManager?.setting;
      if (setting && typeof setting.isReady === "function" &&
          !setting.isReady()) return false;
    } catch (e) { /* no readiness gate in this frontend build */ }
    app.registerExtension({
      name: "ComfyDL.Observation",
      async setup() {
        try {
          const v = app.extensionManager.setting.get("Comfy.Locale");
          if (typeof v === "string" && v) locale = v.split("-")[0];
        } catch (e) { /* keep the default locale */ }
        mountOverlay();
      },
    });
    return true;
  }

  const bootTimer = setInterval(() => {
    if (boot()) clearInterval(bootTimer);
  }, 250);

  // Exposed for host-side teardown / tests; the overlay normally lives for
  // the whole page lifetime.
  window.ComfyDL = window.ComfyDL || {};
  window.ComfyDL.destroyObservation = destroyOverlay;
})();
