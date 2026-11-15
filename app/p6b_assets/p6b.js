/* ComfyDL Optimizer panel (M-D P6b, ADR-D7): a thin rendering layer.
 *
 * What: an on-demand minimap-sized floating window that shows every
 *       registered performance switch, the five backend families' live
 *       describe()/status() readout, and the DataFrame-cache counters.
 * In:   GET /comfydl/switches, GET /comfydl/switches/status,
 *       GET /comfydl/dataio/cache/stats, GET /comfydl/attention/bench (P7,
 *       read-only cached benchmark rows); writes go through the EXISTING
 *       switch endpoints (no second switch interface is introduced).
 * Out:  DOM only - true/false state always comes back from the server;
 *       a rejected toggle (404/409) snaps the checkbox back by re-reading
 *       the registry instead of trusting the optimistic click.
 *
 * Shell reused from P6 Observation (obs.js): collapse/hide/reopen,
 * pointer drag with left/top only, position persistence, dock-gap repair,
 * three-gate motion.  IDs, CSS classes and localStorage keys are P6b-owned
 * so the two windows never share shape or position state.
 */
(() => {
  "use strict";

  const CSS_URL = "/comfydl/p6b/p6b.css";
  const OVERLAY_ID = "cdl-p6b-overlay";
  const REOPEN_ID = "cdl-p6b-reopen";
  const COLLAPSE_KEY = "ComfyDL.P6b.Collapsed";
  const HIDDEN_KEY = "ComfyDL.P6b.Hidden";
  const POS_KEY = "ComfyDL.P6b.Position";
  const SWITCHES_URL = "/comfydl/switches";
  const STATUS_URL = "/comfydl/switches/status";
  const CACHE_STATS_URL = "/comfydl/dataio/cache/stats";
  const BENCH_URL = "/comfydl/attention/bench";
  const PERF_FAMILIES = ["attention", "compile", "quant", "vram", "dataio_cache"];

  // These three switches own real start/stop side effects in their own
  // endpoints.  A generic PATCH would only change the registry state and
  // leave the watchdog / cache provider / spine thread out of sync.
  const SPECIAL = {
    "comfydl.profiling": { method: "POST", url: "/comfydl/profiling/enabled" },
    "comfydl.crashsite": { method: "POST", url: "/comfydl/crashsite/enabled" },
    "comfydl.spine": { method: "PATCH", url: "/comfydl/spine" },
  };

  const I18N = {
    en: {
      title: "Optimizer", tooltip: "ComfyDL performance control surface",
      reopen: "Optimizer", expand: "Expand", collapse: "Collapse", hide: "Hide",
      refresh: "Refresh", switches_title: "Feature switches",
      status_title: "Backend status", cache_title: "Data cache",
      hard: "hard-disabled", off: "off", on: "on",
      unavailable: "unavailable", empty: "No data yet - click Refresh.",
      hint: "Switches use the existing /comfydl/switches API; rejected toggles snap back.",
      hits: "hits", misses: "misses", entries: "entries", bytes: "cached MB",
      bench: "bench namespace",
      bench_title: "Attention bench",
      theme_title: "Accent theme",
      nobench: "No cached benchmark yet - run the Attention Optimizer node.",
      geo: (r) => "seq " + r.seq_len + " · " + r.heads + "h · d" + r.head_dim +
        (r.device ? " · " + String(r.device).replace("device(", "").replace(")", "") : ""),
    },
    zh: {
      title: "性能优化", tooltip: "ComfyDL 性能控制面",
      reopen: "优化", expand: "展开", collapse: "收起", hide: "隐藏",
      refresh: "刷新", switches_title: "功能开关",
      status_title: "后端状态", cache_title: "数据缓存",
      hard: "硬禁用", off: "关", on: "开",
      unavailable: "不可用", empty: "暂无数据——点「刷新」。",
      hint: "开关走既有 /comfydl/switches 接口；被拒的切换会自动回弹。",
      hits: "命中", misses: "未命中", entries: "条目", bytes: "缓存 MB",
      bench: "基准命名空间",
      bench_title: "Attention 实测基准",
      theme_title: "主题色",
      nobench: "暂无缓存基准——先跑一次 Attention Optimizer 节点。",
      geo: (r) => "seq " + r.seq_len + " · " + r.heads + " 头 · d" + r.head_dim +
        (r.device ? " · " + String(r.device).replace("device(", "").replace(")", "") : ""),
    },
  };

  let locale = "en";
  function t(key) {
    const dict = I18N[locale] || I18N.en;
    return dict[key] ?? I18N.en[key] ?? key;
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

  // U stick 2: string-valued settings (the kit hue theme selection).
  function loadStr(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : raw;
    } catch (e) { return fallback; }
  }
  function saveStr(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }
  const THEME_KEY = "ComfyDL.P6b.theme";
  const THEMES = ["blue", "aquamarine", "green", "lime", "orange",
    "pink", "plum", "purple", "red"];

  function loadPos() {
    try {
      const raw = localStorage.getItem(POS_KEY);
      if (!raw) return null;
      const p = JSON.parse(raw);
      if (Number.isFinite(p?.left) && Number.isFinite(p?.top)) return p;
    } catch (e) { /* corrupt payload -> CSS default */ }
    return null;
  }

  function savePos(left, top) {
    try { localStorage.setItem(POS_KEY, JSON.stringify({ left, top })); }
    catch (e) { /* private mode */ }
  }

  function getApp() {
    try { return window.comfyAPI?.app?.app || null; } catch (e) { return null; }
  }

  function getApi() {
    const app = getApp();
    try { return app?.extensionManager?.api || app?.api || window.fetch; }
    catch (e) { return window.fetch; }
  }

  async function getJson(url) {
    const api = getApi();
    let resp;
    if (api && typeof api.fetchApi === "function") {
      resp = await api.fetchApi(url);
    } else if (api && typeof api === "function") {
      resp = await api(url);
    } else {
      resp = await window.fetch(url);
    }
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return resp.json();
  }

  async function sendJson(url, method, body) {
    const resp = await window.fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({}));
    return { ok: resp.ok, status: resp.status, data };
  }

  function injectCss() {
    if (document.querySelector("link[data-cdl-p6b-css]")) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = CSS_URL;
    link.setAttribute("data-cdl-p6b-css", "1");
    document.head.appendChild(link);
  }

  const state = {
    overlayEl: null, reopenEl: null, refs: null,
    collapsed: loadBool(COLLAPSE_KEY, true),
    hidden: loadBool(HIDDEN_KEY, false),
    pos: loadPos(),
    switches: [], backends: {}, cache: null, bench: null, busy: false,
    theme: loadStr(THEME_KEY, ""),
  };

  function ensureDockGap() {
    // Mirror obs.js R5: derive --cdl-dock-left from the live rail only when
    // nobody owns the variable (App Panel owns it when open).
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
    state.pos = { left, top };
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
      btn.setAttribute("aria-label", state.collapsed ? t("expand") : t("collapse"));
    }
  }

  function toggleCollapse() {
    state.collapsed = !state.collapsed;
    saveBool(COLLAPSE_KEY, state.collapsed);
    applyShape();
    applyPosition();
    if (!state.collapsed) fetchAll();
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
    fetchAll();
  }

  function setupDrag(root, handle) {
    if (!handle) return;
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || (event.target.closest && event.target.closest("button"))) return;
      const rect = root.getBoundingClientRect();
      const start = { x: event.clientX, y: event.clientY, left: rect.left, top: rect.top };
      handle.setPointerCapture(event.pointerId);
      root.classList.add("cdl-p6b-dragging");
      event.preventDefault();
      const onMove = (ev) => {
        const width = root.offsetWidth || 280;
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
        root.classList.remove("cdl-p6b-dragging");
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

  // ---------------------------------------------------------------- data --
  async function fetchAll() {
    if (state.busy) return;
    state.busy = true;
    try {
      const [sw, st, cache, bench] = await Promise.all([
        getJson(SWITCHES_URL).catch(() => null),
        getJson(STATUS_URL).catch(() => null),
        getJson(CACHE_STATS_URL).catch(() => null),
        getJson(BENCH_URL).catch(() => null),
      ]);
      if (sw && Array.isArray(sw.switches)) state.switches = sw.switches;
      if (st && st.backends) state.backends = st.backends;
      state.cache = cache;
      state.bench = bench;
    } finally {
      state.busy = false;
    }
    renderSwitches();
    renderStatus();
    renderCache();
    renderBench();
  }

  async function toggleSwitch(id, enabled) {
    const special = SPECIAL[id];
    const url = special ? special.url : SWITCHES_URL + "/" + encodeURIComponent(id);
    const method = special ? special.method : "PATCH";
    let result;
    try {
      result = await sendJson(url, method, { enabled: !!enabled });
    } catch (e) {
      result = { ok: false, status: 0, data: {} };
    }
    // Always re-read the server truth: a 409 (hard-disabled) or 404 must
    // snap the checkbox back, and a 200 must reflect the canonical state.
    await fetchAll();
    return result.ok;
  }

  function renderSwitches() {
    const host = state.refs?.switches;
    if (!host) return;
    if (!state.switches.length) {
      host.innerHTML = '<div class="cdl-p6b-empty">' + esc(t("empty")) + "</div>";
      return;
    }
    host.innerHTML = state.switches.map((s) => {
      const id = String(s.id || "");
      const on = !!s.enabled;
      const hard = !!s.hard_disabled;
      const label = id.replace(/^comfydl\./, "");
      return '<label class="cdl-p6b-switch-row' + (hard ? " cdl-p6b-hard" : "") + '">' +
        // U stick 2: kit checkbox (checked/disabled/label attributes; the
        // hard-disabled ones stay CLICKABLE on purpose - the 409 snap-back
        // demo is part of the contract, P6b-d).
        '<ui-checkbox data-p6b-switch="' + esc(id) + '"' +
        ' label="' + esc(label) + '" title="' + esc(id) + '"' +
        (on ? " checked" : "") + ">" +
        "</ui-checkbox>" +
        '<span class="cdl-p6b-tag">' +
        (hard ? esc(t("hard")) : (on ? esc(t("on")) : esc(t("off")))) +
        "</span></label>";
    }).join("");
    host.querySelectorAll("ui-checkbox[data-p6b-switch]").forEach((box) => {
      box.addEventListener("ui-change", async (ev) => {
        const wanted = !!(ev.detail && ev.detail.checked);
        box.setAttribute("disabled", "");
        const ok = await toggleSwitch(box.dataset.p6bSwitch, wanted);
        // immediate visual snap-back: force the attribute back to server truth
        if (!ok) box.setAttribute("checked", String(!wanted));
        box.removeAttribute("disabled");
      });
    });
  }

  function statusLine(name, value) {
    return '<span class="cdl-p6b-kv"><b>' + esc(name) + "</b> " +
      esc(value) + "</span>";
  }

  function renderStatus() {
    const host = state.refs?.status;
    if (!host) return;
    const rows = PERF_FAMILIES.map((name) => {
      const entry = state.backends[name] || {};
      const live = entry.status || {};
      const desc = entry.describe || {};
      let detail = esc(t("unavailable"));
      if (entry.available) {
        const bits = [];
        bits.push((live.enabled ? t("on") : t("off")));
        if (live.hard_disabled) bits.push(t("hard"));
        if (name === "compile" && Number.isFinite(live.entries)) {
          bits.push(live.entries + " " + t("entries"));
        }
        if (name === "dataio_cache" && Number.isFinite(live.entries)) {
          bits.push(live.entries + " " + t("entries"));
        }
        if (name === "attention" && Array.isArray(live.registered_backends)) {
          bits.push(live.registered_backends.length + " backends");
        }
        if (name === "attention" && desc.cache_namespace) {
          bits.push(desc.cache_namespace + " (" + t("bench") + ")");
        }
        if (name === "quant" && typeof live.gguf_available === "boolean") {
          bits.push("gguf=" + (live.gguf_available ? "yes" : "no"));
        }
        if (name === "vram" && live.device) bits.push(String(live.device));
        detail = bits.join(" · ");
      }
      return '<div class="cdl-p6b-status-row">' +
        '<span class="cdl-p6b-family">' + esc(name) + "</span>" +
        '<span class="cdl-p6b-kv">' + detail + "</span></div>";
    });
    host.innerHTML = rows.join("");
  }

  function renderCache() {
    const host = state.refs?.cache;
    if (!host) return;
    const c = state.cache || {};
    if (!c.available) {
      host.innerHTML = '<div class="cdl-p6b-empty">' + esc(t("unavailable")) +
        "</div>";
      return;
    }
    const mb = Number.isFinite(c.cached_bytes)
      ? (c.cached_bytes / 1024 / 1024).toFixed(1) : "0.0";
    host.innerHTML =
      statusLine(t("hits"), c.hits ?? 0) +
      statusLine(t("misses"), c.misses ?? 0) +
      statusLine(t("entries"), c.entries ?? 0) +
      statusLine(t("bytes"), mb);
  }

  // P7: one cached attention_bench row per geometry - fastest backend first,
  // the chosen one highlighted.  Read-only; empty state teaches where the
  // data comes from instead of showing a blank card.
  function renderBench() {
    const host = state.refs?.bench;
    if (!host) return;
    const b = state.bench || {};
    if (!b.available) {
      host.innerHTML = '<div class="cdl-p6b-empty">' + esc(t("unavailable")) +
        "</div>";
      return;
    }
    const entries = Array.isArray(b.entries) ? b.entries : [];
    if (!entries.length) {
      host.innerHTML = '<div class="cdl-p6b-empty">' + esc(t("nobench")) +
        "</div>";
      return;
    }
    const geoFn = t("geo");
    host.innerHTML = entries.map((row) => {
      const times = row.times_ms && typeof row.times_ms === "object"
        ? row.times_ms : {};
      const speedups = row.speedup && typeof row.speedup === "object"
        ? row.speedup : {};
      const chosen = row.chosen || "";
      const backends = Object.keys(times)
        .sort((a, b2) => times[a] - times[b2])
        .map((name) => {
          const ms = Number(times[name]);
          const sp = Number(speedups[name]);
          const bits = Number.isFinite(ms) ? ms.toFixed(3) + " ms" : esc(name);
          const spTxt = Number.isFinite(sp) && name !== chosen
            ? " ×" + sp.toFixed(2) : "";
          return '<span class="cdl-p6b-bench-backend' +
            (name === chosen ? " chosen" : "") + '">' +
            "<b>" + esc(name) + "</b> " + esc(bits) + esc(spTxt) +
            (name === chosen ? " ✓" : "") + "</span>";
        })
        .join("");
      const geo = typeof geoFn === "function" && Number.isFinite(row.seq_len)
        ? esc(geoFn(row)) : esc(row.key ? String(row.key).slice(0, 10) + "…" : "");
      return '<div class="cdl-p6b-bench-row">' +
        '<div class="cdl-p6b-bench-geo">' + geo + "</div>" +
        '<div class="cdl-p6b-bench-list">' + backends + "</div></div>";
    }).join("");
  }

  // ---------------------------------------------------------------- shell --
  function buildOverlay() {
    const root = document.createElement("div");
    root.id = OVERLAY_ID;
    root.className = "cdl-p6b-overlay p6b-entrance";
    root.title = t("tooltip");
    root.innerHTML =
      '<div class="cdl-p6b-head" data-p6b="head">' +
        '<span class="cdl-p6b-title">' + esc(t("title")) + "</span>" +
        '<button type="button" class="cdl-p6b-btn" data-p6b="refresh" title="' +
          esc(t("refresh")) + '">⟳</button>' +
        '<button type="button" class="cdl-p6b-btn" data-p6b="collapse">+</button>' +
        '<button type="button" class="cdl-p6b-btn" data-p6b="hide" title="' +
          esc(t("hide")) + '">×</button>' +
      "</div>" +
      '<div class="cdl-p6b-body">' +
        '<div class="cdl-p6b-card">' +
          '<div class="cdl-p6b-card-title">' + esc(t("switches_title")) +
            "</div>" +
          '<div class="cdl-p6b-card-body" data-p6b="switches"></div>' +
        "</div>" +
        '<div class="cdl-p6b-card">' +
          '<div class="cdl-p6b-card-title">' + esc(t("status_title")) +
            "</div>" +
          '<div class="cdl-p6b-card-body" data-p6b="status"></div>' +
        "</div>" +
        '<div class="cdl-p6b-card">' +
          '<div class="cdl-p6b-card-title">' + esc(t("cache_title")) +
            "</div>" +
          '<div class="cdl-p6b-card-body" data-p6b="cache"></div>' +
        "</div>" +
        '<div class="cdl-p6b-card">' +
          '<div class="cdl-p6b-card-title">' + esc(t("bench_title")) +
            "</div>" +
          '<div class="cdl-p6b-card-body" data-p6b="bench"></div>' +
        "</div>" +
        '<div class="cdl-p6b-card">' +
          '<div class="cdl-p6b-card-title">' + esc(t("theme_title")) +
            "</div>" +
          '<div class="cdl-p6b-card-body" data-p6b="theme"></div>' +
        "</div>" +
        '<div class="cdl-p6b-hint">' + esc(t("hint")) + "</div>" +
      "</div>";
    document.body.appendChild(root);

    const reopen = document.createElement("button");
    reopen.id = REOPEN_ID;
    reopen.type = "button";
    reopen.className = "cdl-p6b-reopen";
    reopen.textContent = t("reopen");
    reopen.title = t("tooltip");
    document.body.appendChild(reopen);

    root.querySelector('[data-p6b="collapse"]').addEventListener("click", toggleCollapse);
    root.querySelector('[data-p6b="hide"]').addEventListener("click", hideOverlay);
    root.querySelector('[data-p6b="refresh"]').addEventListener("click", fetchAll);
    reopen.addEventListener("click", showOverlay);
    setupDrag(root, root.querySelector('[data-p6b="head"]'));

    state.overlayEl = root;
    state.reopenEl = reopen;
    ensureDockGap(); // must run before the first paint / applyPosition
    state.refs = {
      collapse: root.querySelector('[data-p6b="collapse"]'),
      switches: root.querySelector('[data-p6b="switches"]'),
      status: root.querySelector('[data-p6b="status"]'),
      cache: root.querySelector('[data-p6b="cache"]'),
      bench: root.querySelector('[data-p6b="bench"]'),
      theme: root.querySelector('[data-p6b="theme"]'),
    };
    applyShape();
    applyPosition();
  }

  // U stick 2: the kit hue-theme selector.  Persisted in localStorage; sets
  // data-theme on <html> so the token layer (themes.css) re-derives every
  // --colour-* / --ui-accent for ALL kit surfaces at once.
  function renderTheme() {
    const host = state.refs?.theme;
    if (!host) return;
    host.innerHTML =
      '<ui-dropdown data-p6b="theme-dd" placeholder="theme">' +
      THEMES.map((th) =>
        '<ui-option value="' + th + '" label="' + th + '"></ui-option>').join("") +
      "</ui-dropdown>";
    const dd = host.querySelector("ui-dropdown");
    if (state.theme) {
      dd.setAttribute("value", state.theme);
      document.documentElement.dataset.theme = state.theme;
    }
    dd.addEventListener("ui-select", (ev) => {
      const v = String((ev.detail && ev.detail.value) || "");
      if (!THEMES.includes(v)) return;
      state.theme = v;
      saveStr(THEME_KEY, v);
      document.documentElement.dataset.theme = v;
    });
  }

  function mountOverlay() {
    if (document.getElementById(OVERLAY_ID)) return;
    injectCss();
    buildOverlay();
    renderTheme();
  }

  function destroyOverlay() {
    if (state.overlayEl) state.overlayEl.remove();
    if (state.reopenEl) state.reopenEl.remove();
    state.overlayEl = null;
    state.reopenEl = null;
    state.refs = null;
  }

  function boot() {
    let tries = 0;
    const tick = async () => {
      const app = getApp();
      const store = app?.extensionManager?.setting;
      const ready = store?.isReady ? store.isReady() === true : !!store;
      if (ready || tries > 200) {
        try {
          locale = String(store?.get?.("Comfy.Locale") || "en").split("-")[0];
        } catch (e) { locale = "en"; }
        // U stick 2: make sure the kit components this panel uses are
        // registered before the first render; if the kit is absent the
        // panel still works (checkbox rows just render unstyled).
        try {
          await import("/comfydl/ui/components/ui-checkbox/ui-checkbox.js");
          await import("/comfydl/ui/components/ui-dropdown/ui-dropdown.js");
        } catch (e) { /* kit not served - keep going */ }
        mountOverlay();
        fetchAll(); // three cheap GETs; keep the first paint honest
        return;
      }
      tries += 1;
      setTimeout(tick, 100);
    };
    tick();
  }

  boot();
  window.ComfyDL = window.ComfyDL || {};
  window.ComfyDL.destroyOptimizer = destroyOverlay;
})();
