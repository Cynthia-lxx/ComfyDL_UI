/* ComfyDL Crash site panel (M1): execution snapshots + breakpoint recovery.
 *
 * Minimal independent sidebar tab (ADR-1: deliberately not part of the
 * profiling panel - this is a data-safety feature, not an analysis one):
 *   - a "Save scene now" hint row: capture is INCREMENTAL (the backend's
 *     CacheProvider writes every finished node output as the run goes), so
 *     there is nothing to click - this row just states it and links the
 *     current snapshot;
 *   - the snapshot library list (time / status / nodes / size / note) with
 *     a per-snapshot Resume button that re-queues the stored prompt; the
 *     host then skips every node whose output survived.
 * en/zh bilingual; host CSS variables only; no monospace fonts.
 */
(() => {
  const CSS_URL = "/comfydl/crashsite/crashsite.css";
  const SNAPSHOTS_URL = "/comfydl/crashsite/snapshots";
  const RESUME_URL = "/comfydl/crashsite/resume";
  const EXPORT_URL = "/comfydl/crashsite/export";
  const DETAIL_URL = (id) =>
    "/comfydl/crashsite/snapshots/" + encodeURIComponent(id) + "/detail";
  const TAB_ID = "comfydl-crashsite";

  const state = {
    snapshots: null, loading: false, refreshQueued: false, resuming: null,
    initialized: false, topbarButton: null,
    // Scene export (git-fire M2): the summary table for one snapshot.
    exportFor: null, exportDetail: null, exportBusy: false,
    exportSel: {}, exportFormats: {}, promptSel: true,
  };

  const I18N = {
    en: {
      tab_title: "Crash Site",
      tab_tooltip: "Execution snapshots and breakpoint recovery",
      incremental: "Capture is automatic: every finished node output is written to the snapshot library as the workflow runs. Just Resume after a crash.",
      title: "Snapshot library",
      empty: "No snapshots yet - run or interrupt a workflow to create one.",
      refresh: "Refresh",
      resume: "Resume",
      resuming: "Resuming...",
      nodes: "nodes",
      skipped_nodes: "skipped (non-serializable)",
      captured_title: "captured",
      more: "+{n} more",
      status_completed: "completed",
      status_open: "running",
      status_interrupted: "interrupted",
      resume_ok: "Re-queued - finished nodes will be skipped automatically. Skip/re-run statistics appear on the new snapshot card when the run ends.",
      export_title: "Scene export",
      export_hint: "Pick what to keep from this snapshot; files are written to output/comfydl/scene_export/.",
      export_prompt: "Workflow prompt (JSON)",
      export_run: "Export selected",
      export_busy: "Exporting...",
      export_dismiss: "Dismiss",
      export_open: "Scene export",
      export_done: "Exported {n} file(s) to {dir}",
      export_fail: "{n} row(s) failed - see server log.",
      export_fail_msg: "Export failed: ",
      export_nudge: "Workflow interrupted - scene export is ready in the Crash Site panel.",
      resume_fail: "Resume failed: ",
    },
    zh: {
      tab_title: "事故现场",
      tab_tooltip: "工作流执行快照与断点恢复",
      incremental: "自动保存：工作流运行时每个完成节点的输出都会写入快照库。崩溃后直接点恢复即可。",
      title: "快照库",
      empty: "暂无快照——运行或打断一次工作流即会生成。",
      refresh: "刷新",
      resume: "恢复",
      resuming: "恢复中...",
      nodes: "节点",
      skipped_nodes: "跳过（不可序列化）",
      captured_title: "已捕获",
      more: "另有 {n} 个",
      status_completed: "已完成",
      status_open: "进行中",
      status_interrupted: "已打断",
      resume_ok: "已重新入队——已完成的节点将自动跳过。运行结束后在新快照卡片查看跳过/重跑统计。",
      export_title: "现场导出",
      export_hint: "勾选要保留的内容，文件将写入 output/comfydl/scene_export/。",
      export_prompt: "工作流 prompt（JSON）",
      export_run: "导出所选项",
      export_busy: "导出中...",
      export_dismiss: "关闭",
      export_open: "现场导出",
      export_done: "已导出 {n} 个文件到 {dir}",
      export_fail: "{n} 行失败，详见服务端日志。",
      export_fail_msg: "导出失败：",
      export_nudge: "工作流已打断——现场导出已在「事故现场」面板就绪。",
      resume_fail: "恢复失败：",
    },
  };

  let locale = "en";
  const t = (k, vars) => {
    let s = (I18N[locale] && I18N[locale][k]) || I18N.en[k] || k;
    if (vars) Object.keys(vars).forEach((v) => { s = s.split("{" + v + "}").join(vars[v]); });
    return s;
  };

  const esc = (s) => String(s === null || s === undefined ? "" : s)
    .split("&").join("&amp;").split("<").join("&lt;")
    .split(">").join("&gt;").split('"').join("&quot;").split("'").join("&#39;");

  const fmtBytes = (n) => {
    n = Number(n) || 0;
    if (n < 1024) return n + " B";
    const units = ["KB", "MB", "GB", "TB"];
    let i = -1;
    do { n /= 1024; i++; } while (n >= 1024 && i < units.length - 1);
    return n.toFixed(1) + " " + units[i];
  };

  const fmtTime = (epoch) => {
    const d = new Date((Number(epoch) || 0) * 1000);
    return isNaN(d.getTime()) ? "-" : d.toLocaleString();
  };

  // ROOT CAUSE (2026-10-08): comfyAPI.app is a NAMESPACE - the real app
  // instance sits one level deeper (comfyAPI.app.app), same for api.api.
  // A single-level read yields an object without registerExtension, which
  // made start() pollute forever with "not ready yet" (profiler.js:457 had
  // it right all along).
  const getApp = () => (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) || null;
  const getApi = () => (window.comfyAPI && window.comfyAPI.api && window.comfyAPI.api.api) || null;

  // Temporary instrumentation (M1 acceptance debugging, 2026-10-08): trace
  // every stage so a silent failure localizes itself in the console.
  const csTrace = (msg) => { try { console.log("[CrashSite] " + msg); } catch (e) { /* */ } };
  csTrace("module evaluated");

  async function fetchSnapshots() {
    const api = getApi();
    if (!api || typeof api.fetchApi !== "function") return;
    if (state.loading) { state.refreshQueued = true; return; }
    state.loading = true;
    renderPanel();
    try {
      const resp = await api.fetchApi(SNAPSHOTS_URL);
      if (resp.ok) {
        const body = await resp.json();
        state.snapshots = body.snapshots || [];
      }
    } catch (e) { /* silent: the panel is a courtesy */ }
    state.loading = false;
    renderPanel();
    // A refresh was requested while this one was in flight - run once more
    // so the panel always reflects the latest state.
    if (state.refreshQueued) { state.refreshQueued = false; fetchSnapshots(); }
  }

  async function resume(id, btn) {
    const api = getApi();
    if (!api) return;
    state.resuming = id;
    renderPanel();
    try {
      const resp = await api.fetchApi(RESUME_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // client_id is MANDATORY for a visible resume: the executor only
        // emits websocket events (executing/executed/execution_start) when
        // extra_data carries it; without it the whole resumed run runs
        // silently (2026-10-09 "no continuation" incident).
        body: JSON.stringify({ id, client_id: api.clientId || null }),
      });
      if (resp.ok) {
        showToast(t("resume_ok"));
        await fetchSnapshots();
      } else {
        const body = await resp.json().catch(() => ({}));
        showToast(t("resume_fail") + (body.error || resp.status), true);
      }
    } catch (e) {
      showToast(t("resume_fail") + e, true);
    }
    state.resuming = null;
    renderPanel();
  }

  // ------------------------- scene export (git-fire M2) -------------------
  // Group per-slot detail rows into one row per node for the summary table.
  // Format dropdown = union of the node's slot candidates; default comes
  // from the first slot that has one.  Per-slot fallbacks happen server-side.
  function groupRows(rows) {
    const map = new Map();
    (rows || []).forEach((r) => {
      const g = map.get(r.node_id)
        || { node_id: r.node_id, class_type: r.class_type, slots: 0,
             size: 0, formats: [], defaultFormat: null };
      g.slots += 1;
      g.size += Number(r.size_bytes) || 0;
      (r.export_formats || []).forEach((f) => {
        if (!g.formats.includes(f)) g.formats.push(f);
      });
      if (!g.defaultFormat && r.export_default) g.defaultFormat = r.export_default;
      map.set(r.node_id, g);
    });
    return Array.from(map.values());
  }

  async function loadExport(id) {
    const api = getApi();
    if (!api) return;
    try {
      const resp = await api.fetchApi(DETAIL_URL(id));
      if (resp.ok) {
        state.exportDetail = await resp.json();
        state.exportFor = id;
        state.exportSel = {};
        state.exportFormats = {};
        state.promptSel = true;
        renderPanel();
        openSidebarPanel();
      } else {
        showToast(t("export_fail_msg") + resp.status, true);
      }
    } catch (e) {
      showToast(t("export_fail_msg") + e, true);
    }
  }

  async function runExport() {
    const api = getApi();
    if (!api || !state.exportDetail || state.exportBusy) return;
    state.exportBusy = true;
    renderPanel();
    const items = groupRows(state.exportDetail.rows || [])
      .filter((g) => state.exportSel[g.node_id] !== false)
      .map((g) => ({ node_id: g.node_id,
                     format: state.exportFormats[g.node_id] || g.defaultFormat,
                     class_type: g.class_type }));
    try {
      const resp = await api.fetchApi(EXPORT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: state.exportFor,
          include_prompt: state.promptSel !== false,
          items,
        }),
      });
      const body = await resp.json().catch(() => ({}));
      if (resp.ok) {
        const nFailed = (body.failed || []).length;
        showToast(t("export_done", { n: body.count || 0, dir: body.dir || "" })
          + (nFailed ? " " + t("export_fail", { n: nFailed }) : ""),
          nFailed > 0);
        state.exportDetail = null;
        state.exportFor = null;
        fetchSnapshots();
      } else {
        showToast(t("export_fail_msg") + (body.error || resp.status), true);
      }
    } catch (e) {
      showToast(t("export_fail_msg") + e, true);
    }
    state.exportBusy = false;
    renderPanel();
  }

  // Auto-nudge (user decision 2026-10-13): after a failure/interrupt the
  // panel opens itself with the newest interrupted snapshot's export table.
  async function nudgeExport() {
    await sleep(500); // let the server-side lifecycle mark the snapshot
    await fetchSnapshots();
    const target = (state.snapshots || []).find((s) => s.status === "interrupted");
    if (!target) return;
    openSidebarPanel();
    showToast(t("export_nudge"));
    await loadExport(target.id);
  }

  function renderExportSection() {
    if (!state.exportDetail) return "";
    const groups = groupRows(state.exportDetail.rows || []);
    const row = (g) => {
      const checked = state.exportSel[g.node_id] !== false;
      const fmt = state.exportFormats[g.node_id] || g.defaultFormat || "";
      const opts = g.formats.map((f) =>
        '<option value="' + esc(f) + '"' + (f === fmt ? " selected" : "") + ">"
        + esc(f) + "</option>").join("");
      return '<div class="cs-x-row">'
        + '<input type="checkbox" class="cs-x-check" data-id="'
        + esc(g.node_id) + '"' + (checked ? " checked" : "") + ">"
        + '<span class="cs-x-name">'
        + esc(String(g.class_type || "?") + " #" + String(g.node_id))
        + '</span><span class="cs-x-meta">' + esc(String(g.slots)
          + " slot · " + fmtBytes(g.size))
        + '</span><select class="cs-x-fmt" data-id="' + esc(g.node_id) + '">'
        + opts + "</select></div>";
    };
    const promptChecked = state.promptSel !== false;
    return '<div class="cs-export">'
      + '<div class="cs-title-row"><span class="cs-title">'
      + esc(t("export_title")) + '</span><button class="cs-button" id="cs-x-dismiss">'
      + esc(t("export_dismiss")) + "</button></div>"
      + '<div class="cs-hint">' + esc(t("export_hint")) + "</div>"
      + groups.map(row).join("")
      + '<div class="cs-x-row"><input type="checkbox" id="cs-x-prompt"'
      + (promptChecked ? " checked" : "") + ">"
      + '<span class="cs-x-name">' + esc(t("export_prompt"))
      + '</span><span class="cs-x-meta">JSON</span></div>'
      + '<button class="cs-button cs-x-run" id="cs-x-run"'
      + (state.exportBusy ? " disabled" : "") + ">"
      + esc(state.exportBusy ? t("export_busy") : t("export_run"))
      + "</button></div>";
  }

  function showToast(msg, isWarn) {
    try {
      const em = getApp() && getApp().extensionManager;
      if (em && em.toast) {
        em.toast.add({
          severity: isWarn ? "warn" : "info", summary: "Crash Site",
          detail: msg, life: 5000,
        });
      } else {
        console.log("[CrashSite] " + msg);
      }
    } catch (e) { /* silent */ }
  }

  function renderPanel() {
    const el = state.panelEl;
    if (!el) return;
    const html = '<div class="cs-panel">'
      + '<div class="cs-hint cs-incremental">' + esc(t("incremental")) + "</div>"
      + renderExportSection()
      + '<div class="cs-title-row"><span class="cs-title">' + esc(t("title"))
      + '</span><button class="cs-button" id="cs-refresh">' + esc(t("refresh")) + "</button></div>"
      + '<div class="cs-list">'
      + (state.loading ? '<div class="cs-hint">...</div>'
        : (!state.snapshots || !state.snapshots.length)
          ? '<div class="cs-hint">' + esc(t("empty")) + "</div>"
          : state.snapshots.map((s) => {
            const statusKey = "status_" + (s.status || "completed");
            const statusText = I18N[locale] && I18N[locale][statusKey]
              ? t(statusKey) : (s.status || "-");
            // M2: per-node captured list - first 4 rows inline, the rest
            // behind a native <details> expander so long lists can't blow
            // up the sidebar.
            const cap = Array.isArray(s.captured) ? s.captured : [];
            const capRow = (c) => '<div class="cs-cap-row">'
              + '<span class="cs-cap-name">'
              + esc(String(c.class_type || "?") + " #"
                + String(c.node_id === undefined || c.node_id === null
                  ? "?" : c.node_id))
              + '</span><span class="cs-cap-size">'
              + esc(fmtBytes(c.size_bytes)) + "</span></div>";
            let capHtml = "";
            if (cap.length) {
              capHtml = '<div class="cs-cap"><div class="cs-cap-title">'
                + esc(t("captured_title")) + " (" + esc(String(cap.length))
                + ")</div>"
                + cap.slice(0, 4).map(capRow).join("");
              if (cap.length > 4) {
                capHtml += '<details class="cs-cap-more"><summary>'
                  + esc(t("more", { n: cap.length - 4 })) + "</summary>"
                  + cap.slice(4).map(capRow).join("") + "</details>";
              }
              capHtml += "</div>";
            }
            return '<div class="cs-card">'
              + '<div class="cs-card-head"><span class="cs-time">'
              + esc(fmtTime(s.created_at)) + "</span>"
              + '<span class="cs-tag cs-tag-' + esc(s.status || "completed")
              + '">' + esc(statusText) + "</span></div>"
              + '<div class="cs-meta">' + esc(String(s.nodes || 0)) + " "
              + esc(t("nodes")) + " · " + esc(fmtBytes(s.size_bytes))
              + (Number(s.skipped_nodes) > 0
                ? " · " + esc(String(s.skipped_nodes)) + " "
                  + esc(t("skipped_nodes")) : "")
              + (s.format_version !== 2 ? " · legacy" : "") + "</div>"
              + capHtml
              + (s.note ? '<div class="cs-note">' + esc(s.note) + "</div>" : "")
              + '<button class="cs-button cs-resume" data-id="'
              + esc(s.id) + '"'
              + (state.resuming === s.id ? " disabled" : "") + ">"
              + esc(state.resuming === s.id ? t("resuming") : t("resume"))
              + "</button>"
              + (s.status === "interrupted"
                ? '<button class="cs-button cs-export-open" data-id="'
                  + esc(s.id) + '">' + esc(t("export_open")) + "</button>"
                : "")
              + "</div>";
          }).join(""))
      + "</div></div>";
    el.innerHTML = html;
    wirePanel(el);
  }

  function wirePanel(el) {
    const refresh = el.querySelector("#cs-refresh");
    if (refresh) refresh.addEventListener("click", () => fetchSnapshots());
    el.querySelectorAll(".cs-resume").forEach((btn) => {
      btn.addEventListener("click", () => resume(btn.getAttribute("data-id"), btn));
    });
    el.querySelectorAll(".cs-export-open").forEach((btn) => {
      btn.addEventListener("click", () => loadExport(btn.getAttribute("data-id")));
    });
    // Scene export controls (rendered only when a detail is loaded).
    el.querySelectorAll(".cs-x-check").forEach((box) => {
      box.addEventListener("change", () => {
        state.exportSel[box.getAttribute("data-id")] = box.checked;
      });
    });
    el.querySelectorAll(".cs-x-fmt").forEach((sel) => {
      sel.addEventListener("change", () => {
        state.exportFormats[sel.getAttribute("data-id")] = sel.value;
      });
    });
    const prompt = el.querySelector("#cs-x-prompt");
    if (prompt) prompt.addEventListener("change", () => { state.promptSel = prompt.checked; });
    const run = el.querySelector("#cs-x-run");
    if (run) run.addEventListener("click", () => runExport());
    const dismiss = el.querySelector("#cs-x-dismiss");
    if (dismiss) dismiss.addEventListener("click", () => {
      state.exportDetail = null;
      state.exportFor = null;
      renderPanel();
    });
  }

  function registerSidebar() {
    csTrace("registerSidebar called");
    const app = getApp();
    const em = app && app.extensionManager;
    if (!em || typeof em.registerSidebarTab !== "function") {
      csTrace("NO extensionManager.registerSidebarTab - aborting tab registration");
      return;
    }
    em.registerSidebarTab({
      id: TAB_ID,
      icon: "pi pi-replay",
      title: t("tab_title"),
      tooltip: t("tab_tooltip"),
      type: "custom",
      render: (el) => {
        el.innerHTML = "";
        el.classList.add("cs-host");
        state.panelEl = el;
        renderPanel();
        fetchSnapshots();
        return () => { state.panelEl = null; };
      },
    });
    csTrace("registerSidebarTab returned without error");
  }

  // Second entry point: a topbar button docked next to Run/Jobs (same
  // settingsGroup anchor and ComfyButton/ComfyButtonGroup pattern as the
  // profiler badge - user-requested so the panel is reachable without the
  // sidebar at all).
  async function mountTopbarButton() {
    const app = getApp();
    const anchor = app && app.menu && app.menu.settingsGroup
      && app.menu.settingsGroup.element;
    if (!anchor || !anchor.before) return;
    if (state.topbarButton) return;
    const capi = window.comfyAPI || {};
    let ComfyButtonGroup = capi.buttonGroup && capi.buttonGroup.ComfyButtonGroup;
    let ComfyButton = capi.button && capi.button.ComfyButton;
    if (!ComfyButtonGroup || !ComfyButton) {
      try {
        ComfyButtonGroup = (await import("/scripts/ui/components/buttonGroup.js")).ComfyButtonGroup;
        ComfyButton = (await import("/scripts/ui/components/button.js")).ComfyButton;
      } catch (e) { return; }
    }
    try {
      const button = new ComfyButton({
        icon: "pi pi-replay",
        tooltip: t("tab_tooltip"),
        action: openSidebarPanel,
      });
      const group = new ComfyButtonGroup(button.element);
      anchor.before(group.element || group);
      state.topbarButton = button;
      csTrace("topbar button mounted");
    } catch (e) {
      csTrace("topbar button failed: " + e);
    }
  }

  function openSidebarPanel() {
    try {
      const em = getApp() && getApp().extensionManager;
      if (em && typeof em.toggleSidebarTab === "function") {
        if (em.activeSidebarTabId !== TAB_ID) em.toggleSidebarTab(TAB_ID);
      }
    } catch (e) { /* silent */ }
  }

  async function init() {
    if (state.initialized) return; // setup hook + manual call: idempotent
    state.initialized = true;
    csTrace("init called");
    const app = getApp();
    if (!app) { csTrace("init: no app"); return; }
    try {
      const settings = app.extensionManager && app.extensionManager.setting;
      if (settings && typeof settings.get === "function") {
        locale = settings.get("Comfy.Locale") || "en";
        locale = String(locale).split("-")[0];
      }
    } catch (e) { /* default */ }
    if (locale !== "zh") locale = "en";
    registerSidebar();
    mountTopbarButton();
    fetchSnapshots();
    // M2: refresh the library as executions start and end (a new snapshot
    // opens as "running", the resume statistics note lands on completion)
    // so nobody has to click Refresh to see the result.
    const api = getApi();
    if (api && typeof api.addEventListener === "function") {
      const onExecEvent = (ev) => {
        fetchSnapshots();
        // git-fire M2 auto-nudge: failures/interrupts open the scene-export
        // summary automatically (user decision 2026-10-13).
        if (ev === "execution_error" || ev === "execution_interrupted") {
          nudgeExport();
        }
      };
      ["execution_start", "execution_success", "execution_error",
        "execution_interrupted"].forEach((ev) => {
        try { api.addEventListener(ev, () => onExecEvent(ev)); }
        catch (e) { /* the panel is a courtesy */ }
      });
    }
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Profiler-style polling: try/catch INSIDE the loop - the host's
  // registerExtension wraps a pinia store and THROWS while the Vue services
  // are still warming up ("Cannot read properties of undefined (reading
  // '_s')"); an unhandled throw here killed the whole retry loop before.
  async function waitFor(fn, limit = 30000, step = 200) {
    const start = Date.now();
    while (Date.now() - start < limit) {
      try { if (fn()) return true; } catch (e) { /* host not ready: retry */ }
      await sleep(step || 100);
    }
    return false;
  }

  async function boot() {
    csTrace("boot called");
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = CSS_URL;
    document.head.appendChild(css);

    const ready = await waitFor(() => {
      const app = getApp();
      return app && typeof app.registerExtension === "function";
    }, 30000, 200);
    if (!ready) { csTrace("host app never became ready (30s)"); return; }

    // registerExtension can still throw while the extension-service pinia
    // store initializes; retry until it lands.  init() is called manually
    // as well as from setup(): depending on the race, the host's setup sweep
    // may already be over when we finally register.
    let registered = false;
    let attempts = 0;
    while (!registered) {
      attempts++;
      try {
        getApp().registerExtension({
          name: "ComfyDL.CrashSite",
          settings: [],
          commands: [],
          async setup() {
            csTrace("setup called (host accepted the extension)");
            await init();
          },
        });
        registered = true;
        csTrace("registerExtension returned after " + attempts + " attempt(s)");
      } catch (e) {
        csTrace("registerExtension threw (services warming): " + e);
        await sleep(300);
      }
    }
    await init(); // setup() may or may not fire after a late registration
  }

  boot();
})();
