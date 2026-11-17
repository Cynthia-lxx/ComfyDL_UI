/* ComfyDL Observation overlay (M-D P6, 2026-11-13 CRAFT 2).
 *
 * A minimap-sized FLOATING widget, draggable and collapsible/hidable. It is
 * deliberately not a sidebar tab: observation is a small always-available
 * readout and must not compete with the main canvas.
 *
 * Three data sources, all read-only, no new thread / WS channel / timer:
 *   - `cdl.spine`            : 60-point CPU/RAM/VRAM trends + VRAM GiB + proc CPU
 *   - official exec WS events: ran / cached / blocked badges
 *   - GET /comfydl/runs      : recent experiment/scan ledger rows
 *
 * De-duplication: the Crystools status text at the bottom-left already shows
 * instantaneous CPU/RAM/VRAM percentages, so this widget shows trends and
 * absolute values instead.
 */
import { ensureComponents } from "/comfydl/ui-assets/kit-loader.js";
import { markEntrance, replayEntrance, markStatus } from "/comfydl/ui-assets/kit_semantics.js";

(() => {
  "use strict";

  const CSS_URL = "/comfydl/obs/obs.css";
  const OVERLAY_ID = "cdl-obs-overlay";
  const REOPEN_ID = "cdl-obs-reopen";
  const SPINE_URL = "/comfydl/spine";
  const SNAPSHOT_URL = "/comfydl/spine/snapshot";
  const RUNS_URL = "/comfydl/runs?limit=5";
  const COLLAPSE_KEY = "ComfyDL.Obs.Collapsed";
  const HIDDEN_KEY = "ComfyDL.Obs.Hidden";
  const POS_KEY = "ComfyDL.Obs.Position";
  // R4 (1.7): page-load instant - runs started after this get the "new"
  // this-session tag in the runs list.
  const SESSION_START = Date.now();
  const MAX_POINTS = 60;
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Official execution events (execution.py), used for ran/cached/blocked.
  const EXEC_EVENTS = [
    "execution_start", "executing", "execution_cached", "executed",
    "execution_success", "execution_error", "execution_interrupted",
  ];

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
      tooltip: "Live resource trends, execution badges and experiment runs",
      off_hint: "Monitoring is off. The sampling thread starts only when enabled.",
      enable: "Enable monitoring",
      enabling: "enabling...",
      badge_on: "live", badge_off: "off",
      waiting: "Waiting for the first sample...",
      collapse: "Collapse", expand: "Expand", hide: "Hide",
      reopen: "Observation",
      trend: "60-sample trend",
      cpu_trend: "CPU", ram_trend: "RAM", vram_trend: "VRAM",
      proc: "Process CPU (per-core)", vram: "VRAM", no_gpu: "No CUDA device",
      proc_hint: "Per-core scale: one busy core reads 100%; multi-core load can exceed 100%.",
      schema: "schema v",
      dedup: "Instantaneous values are in the bottom-left status text.",
      exec_title: "Execution", runs_title: "Experiment runs",
      ran: "ran", cached: "cached", blocked: "blocked",
      run_new: "new",
      runs_hint: "Rows are written when a run finishes (completed/failed); a running run appears here only once it ends.",
      runs_empty: "No experiment runs yet - only experiment/scan nodes write here.",
    },
    zh: {
      title: "观测",
      tooltip: "实时资源趋势、执行徽章与实验 runs",
      off_hint: "监控未开启。采样线程只会在启用后启动。",
      enable: "启用监控",
      enabling: "正在启用...",
      badge_on: "运行中", badge_off: "已关闭",
      waiting: "正在等待第一个采样...",
      collapse: "收起", expand: "展开", hide: "隐藏",
      reopen: "观测",
      trend: "60 点趋势",
      cpu_trend: "CPU", ram_trend: "内存", vram_trend: "显存",
      proc: "进程 CPU(per-core)", vram: "显存", no_gpu: "无 CUDA 设备",
      proc_hint: "按单核满载=100% 计；多核满载可超过 100%。",
      schema: "协议 v",
      dedup: "瞬时数值见左下角状态栏，此处显示趋势与绝对值。",
      exec_title: "执行", runs_title: "实验 runs",
      ran: "执行", cached: "缓存", blocked: "受阻",
      run_new: "新",
      runs_hint: "索引行在 run 结束时写入（completed/failed）；进行中的 run 结束后才会出现。",
      runs_empty: "暂无实验 run——只有实验 / 扫描节点会写入。",
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

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
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

  function loadPos() {
    try {
      const raw = localStorage.getItem(POS_KEY);
      if (!raw) return null;
      const p = JSON.parse(raw);
      if (Number.isFinite(p?.left) && Number.isFinite(p?.top)) return p;
    } catch (e) { /* corrupt payload -> use the CSS default */ }
    return null;
  }

  function savePos(left, top) {
    try { localStorage.setItem(POS_KEY, JSON.stringify({ left, top })); }
    catch (e) { /* private mode */ }
  }

  const state = {
    overlayEl: null,
    reopenEl: null,
    refs: null,
    eventApi: null,
    handler: null,
    execHandlers: {},
    enabled: false,
    sample: null,
    collapsed: loadBool(COLLAPSE_KEY, true),
    hidden: loadBool(HIDDEN_KEY, false),
    pos: loadPos(),
    exec: {
      ran: 0, cached: 0, blocked: 0, active: false,
      cachedNodes: new Set(), ranNodes: new Set(),
    },
    runs: [],
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
    markEntrance(root);
    root.title = t("tooltip");
    root.innerHTML =
      '<div class="cdl-obs-head" data-obs="head">' +
        '<span class="cdl-obs-title">' + esc(t("title")) + "</span>" +
        '<span class="cdl-obs-badge" data-obs="badge">' + esc(t("badge_off")) + "</span>" +
        '<ui-button class="cdl-obs-btn" data-obs="collapse" label="+">+</ui-button>' +
        '<ui-button class="cdl-obs-btn" data-obs="hide" label="×">×</ui-button>' +
      "</div>" +
      '<div class="cdl-obs-body">' +
        '<div class="cdl-obs-off" data-obs="hint">' + esc(t("off_hint")) + "</div>" +
        // Sibling, not a child: renderPanel() rewrites hint.textContent.
        '<ui-spinner class="cdl-obs-hint-spinner" data-obs="spinner"' +
          ' variant="boxless" size="14" hidden></ui-spinner>' +
        '<ui-button class="cdl-obs-enable" data-obs="enable" label="' +
          esc(t("enable")) + '">' + esc(t("enable")) + "</ui-button>" +
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
        '<div class="cdl-obs-extra">' +
          '<div class="cdl-obs-section-title">' + esc(t("exec_title")) + "</div>" +
          '<div class="cdl-obs-badges">' +
            '<span class="cdl-obs-badge-item cdl-obs-badge-ran" data-obs="badge-ran"></span>' +
            '<span class="cdl-obs-badge-item cdl-obs-badge-cached" data-obs="badge-cached"></span>' +
            '<span class="cdl-obs-badge-item cdl-obs-badge-blocked" data-obs="badge-blocked"></span>' +
          "</div>" +
          '<div class="cdl-obs-section-title">' + esc(t("runs_title")) + "</div>" +
          '<div class="cdl-obs-runs" data-obs="runs"></div>' +
        "</div>" +
      "</div>";
    document.body.appendChild(root);

    const reopen = document.createElement("ui-button");
    reopen.id = REOPEN_ID;
    reopen.setAttribute("label", t("reopen"));
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

    root.querySelector('[data-obs="collapse"]').addEventListener("ui-press", toggleCollapse);
    root.querySelector('[data-obs="hide"]').addEventListener("ui-press", hideOverlay);
    root.querySelector('[data-obs="enable"]').addEventListener("ui-press", enableSpine);
    reopen.addEventListener("ui-press", showOverlay);
    setupDrag(root, root.querySelector('[data-obs="head"]'));

    state.overlayEl = root;
    state.reopenEl = reopen;
    ensureDockGap(); // must run before the first paint / applyPosition
    state.refs = {
      badge: root.querySelector('[data-obs="badge"]'),
      hint: root.querySelector('[data-obs="hint"]'),
      spinner: root.querySelector('[data-obs="spinner"]'),
      enable: root.querySelector('[data-obs="enable"]'),
      rows: root.querySelector('[data-obs="rows"]'),
      collapse: root.querySelector('[data-obs="collapse"]'),
      proc: root.querySelector('[data-obs="proc"]'),
      vram: root.querySelector('[data-obs="vram"]'),
      meta: root.querySelector('[data-obs="meta"]'),
      runs: root.querySelector('[data-obs="runs"]'),
      badges: {
        ran: root.querySelector('[data-obs="badge-ran"]'),
        cached: root.querySelector('[data-obs="badge-cached"]'),
        blocked: root.querySelector('[data-obs="badge-blocked"]'),
      },
      lines: {
        cpu: root.querySelector('[data-obs="cpu-line"]'),
        ram: root.querySelector('[data-obs="ram-line"]'),
        vram: root.querySelector('[data-obs="vram-line"]'),
      },
    };
    applyShape();
    applyPosition();
    renderBadges();
    renderRuns();
  }

  function setupDrag(root, handle) {
    if (!handle) return;
    handle.addEventListener("pointerdown", (event) => {
      // Kit buttons are custom elements: in a shadow-DOM event the target is
      // retargeted to <ui-button>, so a bare `button` check would start a drag
      // and preventDefault() would swallow the ui-press click.
      if (event.button !== 0 || (event.target.closest &&
          event.target.closest("button, ui-button"))) return;
      const rect = root.getBoundingClientRect();
      const start = {
        x: event.clientX, y: event.clientY, left: rect.left, top: rect.top,
      };
      handle.setPointerCapture(event.pointerId);
      root.classList.add("cdl-obs-dragging");
      event.preventDefault();

      const onMove = (ev) => {
        const width = root.offsetWidth || 240;
        const height = root.offsetHeight || 44;
        const left = clamp(start.left + ev.clientX - start.x, 0,
          Math.max(0, window.innerWidth - width));
        const top = clamp(start.top + ev.clientY - start.y, 0,
          Math.max(0, window.innerHeight - height));
        root.style.left = left + "px";
        root.style.top = top + "px";
        state.pos = { left, top };
      };
      const onUp = (ev) => {
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        handle.removeEventListener("pointercancel", onUp);
        if (handle.hasPointerCapture?.(ev.pointerId)) {
          handle.releasePointerCapture(ev.pointerId);
        }
        root.classList.remove("cdl-obs-dragging");
        if (state.pos) {
          if (state.reopenEl) {
            state.reopenEl.style.left = state.pos.left + "px";
            state.reopenEl.style.top = state.pos.top + "px";
          }
          savePos(state.pos.left, state.pos.top);
        }
      };
      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
      handle.addEventListener("pointercancel", onUp);
    });
  }

  function ensureDockGap() {
    // Repair R5 (issue 1.1): in Node Graph mode --cdl-dock-left is NEVER
    // set, so obs.css falls back to 59px + 14px.  The VERBOSE log proved it
    // (left=73px, dock=-, inline_left=-, saved=-): newer builds have a wider
    // rail, so 73px lands exactly ON its right edge and the 14px gap looks
    // gone.  Derive the variable from the LIVE rail width, but only when
    // nobody else defined it - the App Panel owns that variable when open.
    try {
      const dock = getComputedStyle(document.documentElement)
        .getPropertyValue("--cdl-dock-left").trim();
      if (dock) return;
      const rail = document.querySelector(".side-tool-bar-container");
      const width = rail ? rail.getBoundingClientRect().width : 59;
      if (Number.isFinite(width) && width > 0) {
        document.documentElement.style.setProperty(
          "--cdl-dock-left", Math.round(width) + "px");
      }
    } catch (e) { /* courtesy: keep the CSS fallback */ }
  }

  function applyPosition() {
    const root = state.overlayEl;
    if (!root || !state.pos) return;
    const left = clamp(state.pos.left, 0,
      Math.max(0, window.innerWidth - root.offsetWidth));
    const top = clamp(state.pos.top, 0,
      Math.max(0, window.innerHeight - root.offsetHeight));
    root.style.left = left + "px";
    root.style.top = top + "px";
    state.pos = { left, top }; // the viewport may have shrunk since the save
    if (state.reopenEl) {
      state.reopenEl.style.left = left + "px";
      state.reopenEl.style.top = top + "px";
    }
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
    applyPosition(); // the expanded body is taller - keep it on screen
    if (!state.collapsed) replayEntrance(state.overlayEl);
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
    applyPosition();
    replayEntrance(state.overlayEl);
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
    // U stick 3: the kit spinner replaces the bare "waiting" text as the
    // loading indicator; keep the text for accessibility / kit-off fallback.
    if (r.spinner) r.spinner.hidden = !(state.enabled && !state.sample);
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
    // R4 (1.5-A): per-core caliber - a fully busy N-core box reads 100*N,
    // so this value must NOT be clamped into 0-100 (that hid real load);
    // the label + tooltip carry the caliber instead.
    const procCpu = Number(s.proc?.cpu_percent);
    r.proc.textContent = t("proc") + "  " +
      (Number.isFinite(procCpu) ? procCpu.toFixed(1) : "0.0") + "%";
    r.proc.title = t("proc_hint");
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

  function renderBadges() {
    if (!state.refs) return;
    for (const [key, el] of Object.entries(state.refs.badges)) {
      el.textContent = t(key) + " " + state.exec[key];
      el.classList.toggle("cdl-obs-badge-zero", state.exec[key] === 0);
    }
  }

  function renderRuns() {
    if (!state.refs) return;
    const host = state.refs.runs;
    if (!state.runs.length) {
      host.innerHTML = '<div class="cdl-obs-runs-empty">' +
        esc(t("runs_empty")) + "</div>";
      return;
    }
    // R4 (1.7): the ledger's csv index layer only records FINISHED runs
    // (rows are appended in finish(); status is the real completed/failed
    // value, never hardcoded), so a bare "gridsearch completed" x5 told
    // the user nothing.  Each row now carries started_at (HH:MM:SS from
    // the ledger's local ISO stamp), the duration, and a "new" tag when
    // the run started during THIS page session.
    host.innerHTML = state.runs.map((run) => {
      const name = String(run.experiment || run.run_id || "run");
      const status = String(run.status || "");
      const cls = status === "failed"
        ? "cdl-obs-run-failed cdl-sem-failed"
        : "cdl-obs-run-completed cdl-sem-ran";
      const started = String(run.started_at || "");
      const hm = started.length >= 19 ? started.slice(11, 19) : "";
      const dur = Number(run.duration_s);
      const durTxt = Number.isFinite(dur) && dur > 0
        ? (dur >= 10 ? dur.toFixed(1) + "s"
          : dur >= 1 ? dur.toFixed(2) + "s"
          : Math.round(dur * 1000) + "ms")
        : "";
      // P7: net duration (gross minus the run's Tier-1 self-overheads);
      // shown next to the gross figure when the ledger carries it.
      const net = Number(run.duration_s_net);
      const netTxt = Number.isFinite(net) && net > 0 && net < dur
        ? (net >= 10 ? net.toFixed(1) + "s"
          : net >= 1 ? net.toFixed(2) + "s"
          : Math.round(net * 1000) + "ms")
        : "";
      const startedMs = Date.parse(started);
      const newTag = Number.isFinite(startedMs) && startedMs >= SESSION_START
        ? '<span class="cdl-obs-run-new cdl-sem-new">' + esc(t("run_new")) + "</span>"
        : "";
      return '<div class="cdl-obs-run"><span class="cdl-obs-run-name">' +
        esc(name) + '</span>' +
        (hm ? '<span class="cdl-obs-run-time">' + esc(hm) + "</span>" : "") +
        (durTxt ? '<span class="cdl-obs-run-dur">' + esc(durTxt) + "</span>" : "") +
        (netTxt ? '<span class="cdl-obs-run-dur">' + esc("\u2212 " + netTxt) + "</span>" : "") +
        newTag +
        '<span class="cdl-obs-run-status ' + cls + '">' +
        esc(status) + "</span></div>";
    }).join("") +
      '<div class="cdl-obs-runs-hint">' + esc(t("runs_hint")) + "</div>";
    // This-session runs breathe once; VFX obeys the same three motion gates.
    host.querySelectorAll(".cdl-obs-run-new").forEach((el) =>
      markStatus(el, "new", true));
  }

  function onExecEvent(name, payload) {
    // Official frontend events are CustomEvents whose `detail` is the raw WS
    // payload. `executing` carries a BARE node id string (see cdlapp.js
    // onExecuting, live-verified 2026-10-30) and null when the queue idles;
    // the other execution events carry objects.
    const raw = payload?.detail ?? payload?.data;
    const d = raw && typeof raw === "object" ? raw : {};
    const bareId = typeof raw === "string" ? raw : null;
    const ex = state.exec;
    switch (name) {
      case "execution_start":
        ex.ran = 0; ex.cached = 0; ex.blocked = 0;
        ex.cachedNodes = new Set(); ex.ranNodes = new Set();
        ex.active = true;
        break;
      case "execution_cached":
        (d.nodes || []).forEach((n) => ex.cachedNodes.add(String(n)));
        ex.cached = ex.cachedNodes.size;
        break;
      case "executing":
        if (raw === null || raw === undefined) { ex.active = false; break; }
        {
          const nodeId = bareId ?? d.display_node ?? d.node ?? d.node_id;
          if (nodeId !== null && nodeId !== undefined &&
              !ex.cachedNodes.has(String(nodeId))) {
            ex.ranNodes.add(String(nodeId));
          }
        }
        ex.ran = ex.ranNodes.size;
        break;
      case "executed":
        if (bareId !== null && !ex.cachedNodes.has(bareId)) {
          ex.ranNodes.add(bareId);
        }
        if (d.node != null && !ex.cachedNodes.has(String(d.node))) {
          ex.ranNodes.add(String(d.node));
        }
        ex.ran = ex.ranNodes.size;
        break;
      case "execution_error":
        // "ExecutionBlocked" (execution.py:528-542) and ordinary failures both
        // mean the node never produced an output -> count as blocked.
        ex.blocked += 1;
        ex.active = false;
        break;
      case "execution_interrupted":
        ex.blocked += 1;
        ex.active = false;
        break;
      case "execution_success":
        ex.active = false;
        refreshRuns();
        break;
      default:
        break;
    }
    renderBadges();
  }

  async function refreshRuns() {
    const api = getApi(getApp());
    if (!api || typeof api.fetchApi !== "function") return;
    try {
      const resp = await api.fetchApi(RUNS_URL);
      if (resp.ok) {
        const data = await resp.json();
        // read_index() already keeps the newest N, oldest first.
        state.runs = Array.isArray(data?.runs)
          ? data.runs.slice(-5).reverse() : [];
      }
    } catch (e) { state.runs = []; }
    renderRuns();
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
    if (btn) {
      btn.setAttribute("disabled", "");
      btn.textContent = t("enabling");
    }
    if (state.refs?.spinner) state.refs.spinner.hidden = false;
    try {
      await api.fetchApi(SPINE_URL, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: true, rate_s: 1.0 }),
      });
    } catch (e) { /* refreshState will fall back to the off hint */ }
    await refreshState();
    if (btn && !state.enabled) {
      btn.removeAttribute("disabled");
      btn.textContent = t("enable");
    }
    renderPanel();
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
      for (const name of EXEC_EVENTS) {
        const handler = (event) => onExecEvent(name, event);
        state.execHandlers[name] = handler;
        eventApi.addEventListener(name, handler);
      }
    }
    renderPanel();
    refreshState();
    refreshRuns();
    return true;
  }

  function destroyOverlay() {
    if (state.eventApi && state.handler &&
        typeof state.eventApi.removeEventListener === "function") {
      state.eventApi.removeEventListener("cdl.spine", state.handler);
      for (const [name, handler] of Object.entries(state.execHandlers)) {
        state.eventApi.removeEventListener(name, handler);
      }
    }
    state.eventApi = null;
    state.handler = null;
    state.execHandlers = {};
    if (state.overlayEl?.parentNode) state.overlayEl.parentNode.removeChild(state.overlayEl);
    if (state.reopenEl?.parentNode) state.reopenEl.parentNode.removeChild(state.reopenEl);
    state.overlayEl = null;
    state.reopenEl = null;
    state.refs = null;
    state.sample = null;
    state.runs = [];
    state.exec = {
      ran: 0, cached: 0, blocked: 0, active: false,
      cachedNodes: new Set(), ranNodes: new Set(),
    };
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
        await ensureComponents(["ui-button", "ui-spinner"]);
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
