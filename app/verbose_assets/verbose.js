/* ComfyDL VERBOSE front-end diagnostics (repair R0, smallplans §0.90).
   INERT unless GET /comfydl/verbose reports enabled=true.  Every line is
   appended to the SAME server-side log file, so the user only hands back one
   file (no stopwatch, no manual timing).  Enable from the console with
   `ComfyDL.Verbose.enable(true)`; disable with `enable(false)`. */
(() => {
  const STATE_URL = "/comfydl/verbose";
  const FE_URL = "/comfydl/verbose/fe";
  const INTEREST = /\/comfydl\/|\/prompt|\/interrupt/;
  let enabled = false;
  let buffer = [];
  let timer = null;
  let originalFetch = null;
  const cleanups = [];

  function emit(stage, event, fields) {
    if (!enabled) return;
    const pairs = Object.entries(fields || {}).map(([k, v]) =>
      `${k}=${String(v).replace(/\s+/g, "_").slice(0, 160)}`).join(" ");
    buffer.push(`[CDL-V-FE] ${Date.now()} ${performance.now().toFixed(3)} ` +
      `FE ${stage} ${event} ${pairs}`.trimEnd());
    if (buffer.length >= 40) flush();
  }

  function flush() {
    if (!enabled || !buffer.length) return;
    const lines = buffer.slice();
    buffer = [];
    try {
      window.fetch(FE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lines }),
        keepalive: true,
      }).catch(() => {});
    } catch (e) { /* courtesy */ }
  }

  function hookFetch() {
    if (originalFetch || typeof window.fetch !== "function") return;
    originalFetch = window.fetch;
    window.fetch = function (input, init) {
      const url = typeof input === "string" ? input
        : (input && input.url) || "";
      if (!enabled || !INTEREST.test(url) || url === FE_URL) {
        return originalFetch.apply(this, arguments);
      }
      let reason = "unknown";
      try {
        reason = (new Error().stack || "").split("\n").slice(1, 4)
          .join(" | ").replace(/\s+/g, " ").slice(0, 240);
      } catch (e) { /* courtesy */ }
      emit("http", "request", {
        url, method: (init && init.method) || "GET", reason,
      });
      const t0 = performance.now();
      return originalFetch.apply(this, arguments).then((resp) => {
        emit("http", "response", {
          url, status: resp.status,
          duration_ms: (performance.now() - t0).toFixed(2),
        });
        // R4 (1.7): the /runs index body was never logged, so the
        // "all rows completed" question was unanswerable from logs alone.
        // A summary (count + status histogram + started_at presence) is
        // recorded via a clone - the UI consumer still gets the stream.
        if (/\/comfydl\/runs(\?|$)/.test(url) && resp.ok) {
          try {
            resp.clone().json().then((data) => {
              const rows = Array.isArray(data && data.runs) ? data.runs : [];
              const statuses = {};
              rows.forEach((r) => {
                const s = String((r && r.status) || "?");
                statuses[s] = (statuses[s] || 0) + 1;
              });
              emit("runs", "index-body", {
                count: rows.length,
                statuses: JSON.stringify(statuses),
                has_started_at: rows.length > 0
                  && String(rows[0].started_at || "") !== "",
              });
            }).catch(() => { /* courtesy */ });
          } catch (e) { /* courtesy */ }
        }
        return resp;
      });
    };
    cleanups.push(() => {
      window.fetch = originalFetch;
      originalFetch = null;
    });
  }

  function hookClicks() {
    const handler = (ev) => {
      const el = ev.target && ev.target.closest
        ? ev.target.closest("button,[role=button]") : null;
      emit("ui", "click", {
        label: el ? (el.textContent || el.title || el.id || "?")
          .trim().slice(0, 60) : "?",
      });
    };
    document.addEventListener("click", handler, true);
    cleanups.push(() => document.removeEventListener("click", handler, true));
  }

  function hookWs() {
    const app = window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app;
    const api = (app && app.extensionManager && app.extensionManager.api)
      || (app && app.api)
      || (window.comfyAPI && window.comfyAPI.api && window.comfyAPI.api.api);
    if (!api || typeof api.addEventListener !== "function") return;
    const events = ["execution_start", "executing", "execution_cached",
      "executed", "execution_success", "execution_error",
      "execution_interrupted"];
    const handlers = events.map((name) => {
      const fn = (ev) => emit("ws", name, {
        detail: ev && ev.detail !== undefined
          ? String(ev.detail).slice(0, 120) : "",
      });
      try { api.addEventListener(name, fn); } catch (e) { /* courtesy */ }
      return [name, fn];
    });
    cleanups.push(() => handlers.forEach(([name, fn]) => {
      try { api.removeEventListener(name, fn); } catch (e) { /* courtesy */ }
    }));
  }

  function observeOverlay() {
    // R1 needs the real left/top + CSS variable + saved position for 1.1.
    const root = document.getElementById("cdl-obs-overlay");
    if (!root) return;
    const cs = getComputedStyle(root);
    const dock = getComputedStyle(document.documentElement)
      .getPropertyValue("--cdl-dock-left");
    emit("ui", "obs-position", {
      left: cs.left, inline_left: root.style.left || "-",
      dock: (dock || "-").trim(),
      saved: (localStorage.getItem("ComfyDL.Obs.Position") || "-").slice(0, 120),
    });
  }

  function mount() {
    if (enabled) return;
    enabled = true;
    hookFetch();
    hookClicks();
    hookWs();
    observeOverlay();
    timer = setInterval(flush, 1000);
    emit("verbose", "frontend-enabled");
  }

  function unmount() {
    if (!enabled) return;
    enabled = false;
    if (timer) { clearInterval(timer); timer = null; }
    cleanups.splice(0).forEach((fn) => { try { fn(); } catch (e) { /* */ } });
    flush();
  }

  async function boot() {
    try {
      const resp = await window.fetch(STATE_URL);
      const data = await resp.json();
      if (data && data.enabled) mount();
    } catch (e) { /* courtesy */ }
  }

  window.ComfyDL = window.ComfyDL || {};
  window.ComfyDL.Verbose = {
    enable: async (on = true, level = "verbose") => {
      try {
        await window.fetch(STATE_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled: !!on, level }),
        });
      } catch (e) { /* courtesy */ }
      if (on) mount(); else unmount();
      return { enabled };
    },
    state: () => ({ enabled }),
    flush,
  };

  boot();
})();
