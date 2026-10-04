/* ComfyDL Profiling panel (reform: Profiling Tools M1).
 *
 * A repo-owned frontend extension: the server serves this file (and its CSS)
 * from /comfydl/profiling/ (see server.py), and app/frontend_patch.py
 * injects the <script> loader into index.html at startup - the frontend
 * package itself is never edited (pitfall 19).
 *
 * What it adds:
 *  - a sidebar tab (official registerSidebarTab API) with the memory report;
 *  - a topbar badge next to the settings group (ComfyUI-Manager's official
 *    ComfyButtonGroup technique, with graceful fallbacks);
 *  - a Run guard: monkey-patches app.graphToPrompt (the mandatory pass of
 *    every queue submit) and asks for confirmation when the estimate is red;
 *  - an OOM post-mortem: listens to execution_error, posts the message to
 *    /comfydl/profiling/postmortem and renders attribution + suggestions.
 *
 * Everything degrades silently: no API this extension touches may break the
 * normal workflow. i18n: zh/en dictionaries, following Comfy.Locale.
 */
(() => {
  "use strict";

  const CSS_URL = "/comfydl/profiling/profiler.css";
  const ESTIMATE_URL = "/comfydl/profiling/estimate";
  const POSTMORTEM_URL = "/comfydl/profiling/postmortem";
  const TAB_ID = "comfydl-profiling";

  // ------------------------------------------------------------------ i18n --
  const I18N = {
    en: {
      sidebar_title: "Profiling",
      sidebar_tooltip: "Memory estimation for the current workflow",
      cmd_open: "Open the Profiling panel",
      badge_tooltip: "ComfyDL memory estimate - click to open the Profiling panel",
      budget_title: "Device budget",
      budget_total: "Total",
      budget_free: "Free",
      budget_used: "Used",
      verdict_green: "SAFE",
      verdict_yellow: "MAYBE OOM",
      verdict_red: "CERTAIN OOM",
      verdict_unknown: "NO ESTIMATE",
      verdict_reason_single_exceeds_free: "a single tensor is larger than the free memory",
      verdict_reason_total_exceeds_budget: "the estimated peak exceeds 90% of the device total",
      verdict_reason_below_70_free: "the estimated peak stays under 70% of the free memory",
      verdict_reason_between: "the estimated peak is close to the budget - it may fit",
      verdict_reason_no_graph: "the workflow is empty",
      verdict_reason_nothing_estimated: "no node of this workflow can be estimated yet",
      verdict_reason_unknown: "no budget available",
      verdict_reason_empty: "nothing to estimate",
      disclaimer: "Heuristic upper bound from static analysis - it can over- or under-estimate; treat the colours, not the digits.",
      peak_label: "Estimated peak",
      largest_label: "Largest single tensor",
      assumptions_title: "Assumptions",
      assumptions_hint: "These numbers were not statically knowable; the defaults filled the gaps. Override them and the estimate updates.",
      assumption_default: "default",
      assumption_override: "override",
      table_title: "Per-node breakdown",
      table_node: "Node",
      table_bytes: "Bytes",
      table_single: "Largest single",
      unknown_node: "unknown",
      approx_tag: "approx",
      based_on: "based on {n}",
      refresh: "Refresh",
      confirm_title: "Memory warning: this run will most likely run out of memory",
      confirm_message:
        "Estimated peak {peak} vs {free} free. The largest single tensor ({largest}) alone is {single}.\n\nThis is a heuristic upper bound - you can continue anyway.",
      confirm_run: "Run anyway",
      confirm_cancel: "Cancel",
      run_cancelled: "Run cancelled: memory profiler verdict is red",
      yellow_toast: "Memory estimate: {peak} may exceed the free memory ({free}). See the Profiling panel.",
      oom_toast: "Out of memory detected - the Profiling panel has the analysis",
      postmortem_title: "Out-of-memory post-mortem",
      postmortem_alloc: "Failed allocation",
      postmortem_at: "Attributed to",
      postmortem_suggest: "Suggested fix",
      postmortem_apply: "Set batch_size = {n}",
      postmortem_apply_done: "batch_size applied to {node}",
      postmortem_no_alloc: "The error message carries no allocation size - shown as-is.",
      loading: "Estimating...",
      basis_vocab_build: "corpus: {chars} chars / {tokens} tokens, level {level}, vocab size {size}",
      basis_text_encode: "text of {tokens} tokens ({level} level)",
      basis_sliding_window: "{tokens}-token stream, window {window} -> {samples} samples (stride 1)",
      basis_lm_spec: "spec chain: vocab {vocab}, width {d_model}, {blocks} block(s)",
      basis_lm_build: "materialised model: {params} parameters",
      basis_lm_train: "training B={batch} T={window} E={d_model}, vocab {vocab}, {blocks} block(s), {params} params, {optimizer}",
      basis_lm_forward: "inference B={batch} x L={length}, vocab {vocab}, {blocks} block(s)",
      basis_lm_generate: "prefix {prefix} + {num_tokens} tokens -> length {length}, vocab {vocab}",
      basis_mlp_train: "MLP {in_features}->{hidden}->{out_features}, batch {batch} of {samples}, {optimizer}",
      basis_optimizer: "{optimizer} optimizer",
      basis_pass_through: "pass-through, no memory of its own",
      basis_file_load: "checkpoint {path}: {file_bytes} bytes (~{params} params)",
      item_params: "parameters (float32)",
      item_grads: "gradients",
      item_optimizer_state: "optimizer state",
      item_early_stop_snapshot: "early-stop best-state snapshot",
      item_dataset: "dataset tensors",
      item_embed_out: "embedding + position outputs",
      item_ln_out: "layer-norm outputs",
      item_attn_qkv: "attention q/k/v outputs",
      item_attn_scores: "attention scores (BxHxTxT)",
      item_attn_weights: "attention weights (softmax)",
      item_attn_out_residual: "attention merged + out_proj + residual",
      item_ffn_hidden: "FFN hidden (linear + activation)",
      item_ffn_out_residual: "FFN output + residual",
      item_logits: "logits (B x T x V)",
      item_loss_buffer: "loss buffer (log-softmax)",
      item_inputs: "batch inputs / targets",
      item_outputs: "output tensor",
      item_python_lists: "python window lists (approximate)",
      item_mlp_activations: "MLP activations (upper bound)",
      item_misc: "memory",
    },
    zh: {
      sidebar_title: "性能分析",
      sidebar_tooltip: "当前工作流的内存估算",
      cmd_open: "打开性能分析面板",
      badge_tooltip: "ComfyDL 内存估算 - 点击打开性能分析面板",
      budget_title: "设备预算",
      budget_total: "总量",
      budget_free: "空闲",
      budget_used: "已用",
      verdict_green: "安全",
      verdict_yellow: "可能 OOM",
      verdict_red: "一定 OOM",
      verdict_unknown: "无估算",
      verdict_reason_single_exceeds_free: "单个张量已超过空闲内存",
      verdict_reason_total_exceeds_budget: "估算峰值超过设备总量的 90%",
      verdict_reason_below_70_free: "估算峰值保持在空闲内存的 70% 以内",
      verdict_reason_between: "估算峰值接近预算 - 有可能放不下",
      verdict_reason_no_graph: "工作流为空",
      verdict_reason_nothing_estimated: "当前工作流尚无可估算的节点",
      verdict_reason_unknown: "没有可用的设备预算",
      verdict_reason_empty: "无可估算内容",
      disclaimer: "基于静态分析的启发式上界 - 可能偏高或偏低；请看颜色，别抠数字。",
      peak_label: "估算峰值",
      largest_label: "最大单张量",
      assumptions_title: "假设",
      assumptions_hint: "这些数值无法静态获知，由默认档位补齐；可手动覆盖，估算随之更新。",
      assumption_default: "默认",
      assumption_override: "覆盖",
      table_title: "按节点分解",
      table_node: "节点",
      table_bytes: "内存",
      table_single: "最大单笔",
      unknown_node: "未知",
      approx_tag: "近似",
      based_on: "基于假设 {n}",
      refresh: "刷新",
      confirm_title: "内存警告：本次运行很可能内存不足",
      confirm_message:
        "估算峰值 {peak}，空闲仅 {free}。最大单张量（{largest}）一项就占 {single}。\n\n这是启发式上界估算 - 仍可选择继续运行。",
      confirm_run: "仍然运行",
      confirm_cancel: "取消",
      run_cancelled: "已取消运行：内存分析判定为红色",
      yellow_toast: "内存估算：{peak} 可能超出空闲内存（{free}）。详见性能分析面板。",
      oom_toast: "检测到内存不足（OOM）- 性能分析面板已给出分析",
      postmortem_title: "内存不足事后分析",
      postmortem_alloc: "分配失败",
      postmortem_at: "归属",
      postmortem_suggest: "建议修复",
      postmortem_apply: "设置 batch_size = {n}",
      postmortem_apply_done: "已将 batch_size 应用于 {node}",
      postmortem_no_alloc: "报错信息中没有分配字节数 - 原样展示。",
      loading: "估算中...",
      basis_vocab_build: "语料 {chars} 字符 / {tokens} token，层级 {level}，词表大小 {size}",
      basis_text_encode: "{tokens} 个 token 的文本（{level} 层级）",
      basis_sliding_window: "{tokens} token 流，窗口 {window} -> {samples} 个样本（stride 1）",
      basis_lm_spec: "结构链：词表 {vocab}，宽度 {d_model}，{blocks} 个块",
      basis_lm_build: "物化模型：{params} 参数",
      basis_lm_train: "训练 B={batch} T={window} E={d_model}，词表 {vocab}，{blocks} 块，{params} 参数，{optimizer}",
      basis_lm_forward: "推理 B={batch} × L={length}，词表 {vocab}，{blocks} 块",
      basis_lm_generate: "前缀 {prefix} + {num_tokens} token -> 长度 {length}，词表 {vocab}",
      basis_mlp_train: "MLP {in_features}->{hidden}->{out_features}，批 {batch}/{samples}，{optimizer}",
      basis_optimizer: "{optimizer} 优化器",
      basis_pass_through: "直通，无独立内存",
      basis_file_load: "检查点 {path}：{file_bytes} 字节（约 {params} 参数）",
      item_params: "参数（float32）",
      item_grads: "梯度",
      item_optimizer_state: "优化器状态",
      item_early_stop_snapshot: "早停最佳快照",
      item_dataset: "数据集张量",
      item_embed_out: "嵌入 + 位置编码输出",
      item_ln_out: "层归一化输出",
      item_attn_qkv: "注意力 q/k/v 输出",
      item_attn_scores: "注意力分数（B×H×T×T）",
      item_attn_weights: "注意力权重（softmax）",
      item_attn_out_residual: "注意力合并 + 输出投影 + 残差",
      item_ffn_hidden: "FFN 隐层（线性 + 激活）",
      item_ffn_out_residual: "FFN 输出 + 残差",
      item_logits: "logits（B × T × V）",
      item_loss_buffer: "损失缓冲（log-softmax）",
      item_inputs: "批输入 / 目标",
      item_outputs: "输出张量",
      item_python_lists: "Python 切窗列表（近似）",
      item_mlp_activations: "MLP 激活（上界）",
      item_misc: "内存",
    },
  };

  // -------------------------------------------------------------- helpers --
  const state = {
    locale: "en",
    report: null,
    postmortem: null,
    assumptions: {},
    lastPrompt: null,
    estimating: false,
    badge: null,
    yellowToastAt: 0,
    panelEl: null,
    timer: null,
  };

  const getApp = () => (window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app) || null;
  const getApi = () => (window.comfyAPI && window.comfyAPI.api && window.comfyAPI.api.api) || null;

  function t(key, params) {
    const dict = I18N[state.locale] || I18N.en;
    let text = dict[key] !== undefined ? dict[key] : I18N.en[key] !== undefined ? I18N.en[key] : key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.split("{" + k + "}").join(String(v));
      }
    }
    return text;
  }

  function fmtBytes(n) {
    if (n === null || n === undefined || isNaN(n)) return "\u2014";
    const units = ["B", "KB", "MB", "GB", "TB", "PB"];
    let v = Number(n);
    let i = 0;
    while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return (i === 0 ? v.toFixed(0) : v.toFixed(2)) + " " + units[i];
  }

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .split("&").join("&amp;")
      .split("<").join("&lt;")
      .split(">").join("&gt;")
      .split('"').join("&quot;")
      .split("'").join("&#39;");
  }

  function itemLabel(item) {
    return t("item_" + item.kind) === "item_" + item.kind
      ? (item.label || t("item_misc"))
      : t("item_" + item.kind);
  }

  function basisText(node) {
    if (!node || !node.basis) return "";
    const key = "basis_" + node.basis.key;
    if (t(key) === key) return "";
    return t(key, node.basis.params || {});
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitFor(fn, timeout, step) {
    const limit = timeout || 30000;
    const start = Date.now();
    while (Date.now() - start < limit) {
      try { if (fn()) return true; } catch (e) { /* retry */ }
      await sleep(step || 100);
    }
    return false;
  }

  // ------------------------------------------------------- data & estimate --
  let originalGTP = null;

  async function callOriginalGTP() {
    const app = getApp();
    if (!app) return null;
    if (originalGTP) return originalGTP();
    if (typeof app.graphToPrompt === "function") return app.graphToPrompt();
    return null;
  }

  function collectTitles() {
    const titles = {};
    const graph = getApp() && getApp().graph;
    (graph && graph._nodes ? graph._nodes : []).forEach((n) => {
      titles[String(n.id)] = n.title || n.type || String(n.id);
    });
    return titles;
  }

  async function requestEstimate(force) {
    const api = getApi();
    if (!api || state.estimating) return state.report;
    state.estimating = true;
    try {
      const gtp = await callOriginalGTP();
      if (gtp && gtp.output) state.lastPrompt = gtp.output;
      const payload = {
        prompt: state.lastPrompt || {},
        assumptions: state.assumptions,
        titles: collectTitles(),
      };
      const resp = await api.fetchApi(ESTIMATE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        state.report = await resp.json();
        renderPanel();
        updateBadge();
      }
    } catch (e) { /* silent: profiling must never break the UI */ }
    state.estimating = false;
    return state.report;
  }

  function scheduleEstimate() {
    if (state.timer) clearTimeout(state.timer);
    state.timer = setTimeout(() => requestEstimate(true), 500);
  }

  // ------------------------------------------------------------ run guard --
  function patchGraphToPrompt() {
    const app = getApp();
    if (!app || typeof app.graphToPrompt !== "function" || app.__cdlpPatched) return;
    originalGTP = app.graphToPrompt.bind(app);
    app.__cdlpPatched = true;
    app.graphToPrompt = async function (...args) {
      const result = await originalGTP(...args);
      try {
        if (result && result.output) state.lastPrompt = result.output;
        await guardRun();
      } catch (err) {
        if (err && err.__cdlpCancel) throw err;
      }
      return result;
    };
  }

  async function guardRun() {
    const report = await requestEstimate(true);
    if (!report) return;
    if (report.verdict === "red") {
      const ok = await confirmRed(report);
      if (!ok) {
        const err = new Error(t("run_cancelled"));
        err.__cdlpCancel = true;
        throw err;
      }
    } else if (report.verdict === "yellow") {
      const now = Date.now();
      if (now - state.yellowToastAt > 15000) {
        state.yellowToastAt = now;
        toast("warn", t("yellow_toast", {
          peak: fmtBytes(report.peak_bytes),
          free: fmtBytes(report.budget && report.budget.free_bytes),
        }));
      }
    }
  }

  function confirmMessage(report) {
    const largest = report.largest || {};
    return t("confirm_message", {
      peak: fmtBytes(report.peak_bytes),
      free: fmtBytes(report.budget && report.budget.free_bytes),
      largest: esc(largest.label || itemLabel(largest) || ""),
      single: fmtBytes(largest.single_bytes),
    });
  }

  async function confirmRed(report) {
    const em = getApp() && getApp().extensionManager;
    if (em && em.dialog && typeof em.dialog.confirm === "function") {
      try {
        const r = await em.dialog.confirm({
          title: t("confirm_title"),
          message: confirmMessage(report),
        });
        if (r === true) return true;
        if (r === false || r === null) return false;
      } catch (e) { /* fall through */ }
    }
    try { return window.confirm(t("confirm_title") + "\n\n" + confirmMessage(report)); }
    catch (e) { return false; }
  }

  function toast(severity, summary, detail) {
    const em = getApp() && getApp().extensionManager;
    try {
      if (em && em.toast && typeof em.toast.add === "function") {
        em.toast.add({ severity, summary, detail, life: 6000 });
        return;
      }
    } catch (e) { /* fall through */ }
    try { console.log("[ComfyDL profiler]", summary, detail || ""); } catch (e) { /* ignore */ }
  }

  // ---------------------------------------------------------- post-mortem --
  async function handleExecutionError(detail) {
    const api = getApi();
    const d = detail || {};
    const message = String(d.exception_message || "");
    if (!/allocat/i.test(message)) return;
    try {
      const resp = await api.fetchApi(POSTMORTEM_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message + "\n" + String(d.exception_type || ""),
          node_id: d.node_id,
          node_type: d.node_type,
          prompt: state.lastPrompt,
          assumptions: state.assumptions,
        }),
      });
      if (resp.ok) {
        state.postmortem = await resp.json();
        renderPanel();
        toast("error", t("oom_toast"));
      }
    } catch (e) { /* silent */ }
  }

  function applySuggestion() {
    const s = state.postmortem && state.postmortem.suggestion;
    if (!s) return;
    const app = getApp();
    const graph = app && app.graph;
    const node = graph && graph._nodes
      ? graph._nodes.find((n) => String(n.id) === String(s.node_id)) : null;
    if (!node) return;
    const widget = (node.widgets || []).find((w) => w.name === "batch_size");
    if (!widget) return;
    widget.value = s.batch_size;
    try { node.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    try { graph.change(); } catch (e) { /* ignore */ }
    toast("success", t("postmortem_apply_done", { node: node.title || s.node_id }));
    scheduleEstimate();
  }

  // ---------------------------------------------------------------- badge --
  function badgeText(report) {
    if (!report) return "\u2026";
    if (report.verdict === "unknown") return "?";
    return fmtBytes(report.peak_bytes).replace(" ", "");
  }

  function badgeClass(report) {
    if (!report) return "cdlp-b-gray";
    return {
      green: "cdlp-b-green", yellow: "cdlp-b-yellow",
      red: "cdlp-b-red", unknown: "cdlp-b-gray",
    }[report.verdict] || "cdlp-b-gray";
  }

  function updateBadge() {
    const b = state.badge;
    if (!b) return;
    const text = badgeText(state.report);
    const cls = badgeClass(state.report);
    if (b.button) {
      try { b.button.content = text; } catch (e) { /* setter may differ */ }
      try {
        if (b.button.element) {
          b.button.element.classList.remove("cdlp-b-green", "cdlp-b-yellow", "cdlp-b-red", "cdlp-b-gray");
          b.button.element.classList.add(cls);
        }
      } catch (e) { /* ignore */ }
    }
    if (b.element) {
      b.element.textContent = text;
      b.element.className = "cdlp-badge-fallback " + cls;
    }
  }

  async function mountBadge() {
    const app = getApp();
    const anchor = app && app.menu && app.menu.settingsGroup && app.menu.settingsGroup.element;
    if (!anchor || !anchor.before) return;
    const capi = window.comfyAPI || {};
    let ComfyButtonGroup = capi.buttonGroup && capi.buttonGroup.ComfyButtonGroup;
    let ComfyButton = capi.button && capi.button.ComfyButton;
    if (!ComfyButtonGroup || !ComfyButton) {
      try {
        ComfyButtonGroup = (await import("/scripts/ui/components/buttonGroup.js")).ComfyButtonGroup;
        ComfyButton = (await import("/scripts/ui/components/button.js")).ComfyButton;
      } catch (e) { ComfyButtonGroup = ComfyButton = null; }
    }
    if (ComfyButtonGroup && ComfyButton) {
      try {
        const button = new ComfyButton({
          tooltip: t("badge_tooltip"),
          content: badgeText(state.report),
          action: openSidebar,
        });
        const group = new ComfyButtonGroup(button.element);
        anchor.before(group.element || group);
        state.badge = { button };
        updateBadge();
        return;
      } catch (e) { /* fall through to raw DOM */ }
    }
    try {
      const el = document.createElement("button");
      el.className = "cdlp-badge-fallback " + badgeClass(state.report);
      el.title = t("badge_tooltip");
      el.addEventListener("click", () => openSidebar());
      anchor.before(el);
      state.badge = { element: el };
      updateBadge();
    } catch (e) { /* no badge then - the sidebar stays */ }
  }

  // -------------------------------------------------------------- sidebar --
  function openSidebar() {
    const em = getApp() && getApp().extensionManager;
    const id = TAB_ID;
    const tryTab = (obj) => {
      try {
        if (obj && typeof obj.toggleSidebarTab === "function") {
          if (obj.activeSidebarTabId !== id) obj.toggleSidebarTab(id);
          return true;
        }
      } catch (e) { /* next */ }
      return false;
    };
    if (em) {
      if (em.sidebarTab && tryTab(em.sidebarTab.value ? em.sidebarTab.value : em.sidebarTab)) return;
      if (tryTab(em.sidebarTab)) return;
    }
    try {
      const el = document.querySelector('[data-toolbar-side-item="' + id + '"]');
      if (el) { el.click(); return; }
    } catch (e) { /* ignore */ }
  }

  function renderPanel() {
    const el = state.panelEl;
    if (!el) return;
    const r = state.report;
    const pm = state.postmortem;
    const budget = (r && r.budget) || null;
    const used = budget ? Math.max(0, budget.total_bytes - budget.free_bytes) : null;

    const html = [];
    html.push('<div class="cdlp-panel">');

    // Verdict header
    html.push('<div class="cdlp-verdict cdlp-v-' + ((r && r.verdict) || "unknown") + '">');
    html.push('<span class="cdlp-verdict-badge">' + esc(t("verdict_" + ((r && r.verdict) || "unknown"))) + "</span>");
    if (r) {
      html.push('<span class="cdlp-verdict-num">' + esc(fmtBytes(r.peak_bytes)) + "</span>");
      if (r.verdict_reason) {
        html.push('<span class="cdlp-verdict-reason">' + esc(t("verdict_reason_" + r.verdict_reason) !== "verdict_reason_" + r.verdict_reason ? t("verdict_reason_" + r.verdict_reason) : r.verdict_reason) + "</span>");
      }
      if (r.largest) {
        html.push('<div class="cdlp-verdict-largest">' + esc(t("largest_label")) + ": "
          + esc(itemLabel(r.largest)) + " \u00b7 " + esc(fmtBytes(r.largest.single_bytes)) + "</div>");
      }
    } else {
      html.push('<span class="cdlp-verdict-reason">' + esc(t("loading")) + "</span>");
    }
    html.push("</div>");

    // Budget card
    if (budget) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("budget_title")) + "</div>");
      html.push('<div class="cdlp-budget">');
      html.push('<div class="cdlp-budget-cell"><div class="cdlp-k">' + esc(t("budget_total")) + '</div><div class="cdlp-v">' + esc(fmtBytes(budget.total_bytes)) + "</div></div>");
      html.push('<div class="cdlp-budget-cell"><div class="cdlp-k">' + esc(t("budget_free")) + '</div><div class="cdlp-v">' + esc(fmtBytes(budget.free_bytes)) + "</div></div>");
      html.push('<div class="cdlp-budget-cell"><div class="cdlp-k">' + esc(t("budget_used")) + '</div><div class="cdlp-v">' + esc(fmtBytes(used)) + "</div></div>");
      html.push("</div>");
      const pct = budget.total_bytes ? Math.min(100, Math.round((used / budget.total_bytes) * 100)) : 0;
      html.push('<div class="cdlp-meter"><div class="cdlp-meter-fill" style="width:' + pct + '%"></div></div>');
      html.push("</div>");
    }

    // Assumptions
    const usedAssumptions = (r && r.assumptions_used) || [];
    if (usedAssumptions.length) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("assumptions_title")) + "</div>");
      html.push('<div class="cdlp-hint">' + esc(t("assumptions_hint")) + "</div>");
      usedAssumptions.forEach((a) => {
        html.push('<div class="cdlp-assumption">');
        html.push('<span class="cdlp-assumption-key">' + esc(a.key) + "</span>");
        html.push('<input type="number" min="0" step="1" value="' + esc(a.value)
          + '" data-assumption="' + esc(a.key) + '">');
        html.push('<span class="cdlp-assumption-src">' + esc(t("assumption_" + a.source)) + "</span>");
        html.push("</div>");
      });
      html.push("</div>");
    }

    // Post-mortem
    if (pm) {
      html.push('<div class="cdlp-card cdlp-pm">');
      html.push('<div class="cdlp-card-title cdlp-pm-title">' + esc(t("postmortem_title")) + "</div>");
      if (pm.allocation_human) {
        html.push('<div class="cdlp-pm-row"><span>' + esc(t("postmortem_alloc")) + '</span><b>' + esc(pm.allocation_human) + "</b></div>");
      } else {
        html.push('<div class="cdlp-hint">' + esc(t("postmortem_no_alloc")) + "</div>");
      }
      if (pm.attributed) {
        html.push('<div class="cdlp-pm-row"><span>' + esc(t("postmortem_at"))
          + '</span><b>' + esc(pm.attributed.label) + " \u00b7 " + esc(fmtBytes(pm.attributed.bytes)) + "</b></div>");
      }
      if (pm.suggestion) {
        html.push('<div class="cdlp-pm-row"><span>' + esc(t("postmortem_suggest")) + '</span><b>batch_size = ' + esc(pm.suggestion.batch_size) + "</b></div>");
        html.push('<button class="cdlp-button" id="cdlp-apply-suggestion">'
          + esc(t("postmortem_apply", { n: pm.suggestion.batch_size })) + "</button>");
      }
      if (pm.message_head) {
        html.push('<pre class="cdlp-pm-msg">' + esc(pm.message_head) + "</pre>");
      }
      html.push("</div>");
    }

    // Per-node breakdown
    html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("table_title"))
      + ' <button class="cdlp-button cdlp-refresh" id="cdlp-refresh">' + esc(t("refresh")) + "</button></div>");
    const nodes = (r && r.nodes) || [];
    nodes.forEach((n) => {
      const unknown = n.status !== "estimated";
      html.push('<details class="cdlp-node' + (unknown ? " cdlp-node-unknown" : "") + '">');
      html.push('<summary><span class="cdlp-node-title">' + esc(n.title || n.id) + "</span>");
      if (unknown) html.push('<span class="cdlp-tag">' + esc(t("unknown_node")) + "</span>");
      else html.push('<span class="cdlp-node-bytes">' + esc(fmtBytes(n.total_bytes)) + "</span>");
      html.push("</summary>");
      const basis = basisText(n);
      if (basis) html.push('<div class="cdlp-basis">' + esc(basis) + "</div>");
      if (unknown && n.reason) html.push('<div class="cdlp-basis">' + esc(n.reason) + "</div>");
      if (n.items && n.items.length) {
        html.push('<table class="cdlp-table"><tr><th>' + esc(t("table_node"))
          + "</th><th>" + esc(t("table_bytes")) + "</th><th>" + esc(t("table_single")) + "</th></tr>");
        n.items.forEach((it) => {
          html.push("<tr><td>" + esc(itemLabel(it)) + (it.approx ? ' <i>(' + esc(t("approx_tag")) + ")</i>" : "")
            + "</td><td>" + esc(fmtBytes(it.bytes)) + "</td><td>" + esc(fmtBytes(it.single_bytes)) + "</td></tr>");
        });
        html.push("</table>");
      }
      html.push("</details>");
    });
    if (!nodes.length) html.push('<div class="cdlp-hint">' + esc(t("loading")) + "</div>");
    html.push("</div>");

    html.push('<div class="cdlp-disclaimer">' + esc(t("disclaimer")) + "</div>");
    html.push("</div>");

    el.innerHTML = html.join("");

    // wire events
    el.querySelectorAll("[data-assumption]").forEach((input) => {
      input.addEventListener("change", () => {
        const key = input.getAttribute("data-assumption");
        const value = parseInt(input.value, 10);
        if (!isNaN(value) && value > 0) {
          state.assumptions[key] = value;
          requestEstimate(true);
        }
      });
    });
    const refresh = el.querySelector("#cdlp-refresh");
    if (refresh) refresh.addEventListener("click", () => requestEstimate(true));
    const apply = el.querySelector("#cdlp-apply-suggestion");
    if (apply) apply.addEventListener("click", applySuggestion);
  }

  function registerSidebar() {
    const em = getApp() && getApp().extensionManager;
    if (!em || typeof em.registerSidebarTab !== "function") return;
    em.registerSidebarTab({
      id: TAB_ID,
      icon: "pi pi-chart-bar",
      title: t("sidebar_title"),
      tooltip: t("sidebar_tooltip"),
      type: "custom",
      render: (el) => {
        el.innerHTML = "";
        el.classList.add("cdlp-host");
        state.panelEl = el;
        renderPanel();
        requestEstimate(true);
        return () => { state.panelEl = null; };
      },
    });
  }

  // ----------------------------------------------------------------- boot --
  async function detectLocale() {
    try {
      const em = getApp() && getApp().extensionManager;
      if (em && em.setting && typeof em.setting.get === "function") {
        const v = await em.setting.get("Comfy.Locale");
        if (typeof v === "string" && v) return v.split("-")[0];
      }
    } catch (e) { /* default */ }
    return "en";
  }

  async function init() {
    const app = getApp();
    if (!app) return;
    state.locale = await detectLocale();
    patchGraphToPrompt();
    const api = getApi();
    if (api && typeof api.addEventListener === "function") {
      api.addEventListener("graphChanged", scheduleEstimate);
      api.addEventListener("execution_error", (event) => {
        try { handleExecutionError(event && event.detail); } catch (e) { /* silent */ }
      });
    }
    await waitFor(() => getApp() && getApp().extensionManager);
    try { registerSidebar(); } catch (e) { /* sidebar optional */ }
    try { await mountBadge(); } catch (e) { /* badge optional */ }
    scheduleEstimate();
  }

  function boot() {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = CSS_URL;
    document.head.appendChild(css);

    const start = () => {
      const app = getApp();
      if (!app || typeof app.registerExtension !== "function") return false;
      app.registerExtension({
        name: "ComfyDL.Profiler",
        async setup() { await init(); },
        commands: [{
          id: "ComfyDL_Profiling_Open",
          icon: "pi pi-chart-bar",
          label: t("cmd_open"),
          function: openSidebar,
        }],
      });
      return true;
    };
    if (!start()) waitFor(start, 30000, 200);
  }

  boot();
})();
