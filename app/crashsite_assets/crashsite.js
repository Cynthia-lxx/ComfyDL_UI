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
  const TAB_ID = "comfydl-crashsite";

  const state = { snapshots: null, loading: false, resuming: null };

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
      status_completed: "completed",
      status_open: "running",
      resume_ok: "Re-queued - finished nodes will be skipped automatically.",
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
      status_completed: "已完成",
      status_open: "进行中",
      resume_ok: "已重新入队——已完成的节点将自动跳过。",
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
    state.loading = true;
    render();
    try {
      const resp = await api.fetchApi(SNAPSHOTS_URL);
      if (resp.ok) {
        const body = await resp.json();
        state.snapshots = body.snapshots || [];
      }
    } catch (e) { /* silent: the panel is a courtesy */ }
    state.loading = false;
    render();
  }

  async function resume(id, btn) {
    const api = getApi();
    if (!api) return;
    state.resuming = id;
    render();
    try {
      const resp = await api.fetchApi(RESUME_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
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
    render();
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
            return '<div class="cs-card">'
              + '<div class="cs-card-head"><span class="cs-time">'
              + esc(fmtTime(s.created_at)) + "</span>"
              + '<span class="cs-tag cs-tag-' + esc(s.status || "completed")
              + '">' + esc(statusText) + "</span></div>"
              + '<div class="cs-meta">' + esc(String(s.nodes || 0)) + " "
              + esc(t("nodes")) + " · " + esc(fmtBytes(s.size_bytes)) + "</div>"
              + (s.note ? '<div class="cs-note">' + esc(s.note) + "</div>" : "")
              + '<button class="cs-button cs-resume" data-id="'
              + esc(s.id) + '"'
              + (state.resuming === s.id ? " disabled" : "") + ">"
              + esc(state.resuming === s.id ? t("resuming") : t("resume"))
              + "</button></div>";
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
  }

  function registerSidebar() {
    csTrace("registerSidebar called");
    const app = getApp();
    const em = app && app.extensionManager;
    if (!em || typeof em.registerSidebarTab !== "function") {
      csTrace("NO extensionManager.registerSidebarTab - aborting tab registration");
      return;
    }
    csTrace("registerSidebarTab about to be called");
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

  async function init() {
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
    fetchSnapshots();
  }

  function boot() {
    csTrace("boot called");
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = CSS_URL;
    document.head.appendChild(css);

    const start = () => {
      const app = getApp();
      if (!app || typeof app.registerExtension !== "function") {
        csTrace("start: app/registerExtension not ready yet");
        return false;
      }
      // settings/commands arrays kept explicit: the host iterates extension
      // fields and an undefined field must never be its problem here.
      app.registerExtension({
        name: "ComfyDL.CrashSite",
        settings: [],
        commands: [],
        async setup() {
          csTrace("setup called (host accepted the extension)");
          await init();
        },
      });
      csTrace("registerExtension returned");
      return true;
    };
    if (!start()) {
      const wait = () => { if (!start()) setTimeout(wait, 200); };
      wait();
    }
  }

  boot();
})();
