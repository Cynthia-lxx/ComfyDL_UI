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
 *    /comfydl/profiling/postmortem and renders attribution + suggestions;
 *  - M2 compute ledger: the estimate report's flops section (per-node matmul
 *    FLOPs with an attention/FFN/logits split) as a summary card plus a
 *    per-node column and detail table;
 *  - M2 measured timings: the executing/executed events of every run build
 *    a per-node wall-time Top-N panel (cache hits are marked, not ranked);
 *  - M2 execution monitor: a 1 s poll of the watchdog routes (only while a
 *    run is active) drives the topbar live capsule (current node + CPU/MEM)
 *    and the persistent CPU-burst log card.
 *
 * Everything degrades silently: no API this extension touches may break the
 * normal workflow. i18n: zh/en dictionaries, following Comfy.Locale.
 */
(() => {
  "use strict";

  const CSS_URL = "/comfydl/profiling/profiler.css";
  const ESTIMATE_URL = "/comfydl/profiling/estimate";
  const OPGRAPH_URL = "/comfydl/profiling/opgraph";
  const POSTMORTEM_URL = "/comfydl/profiling/postmortem";
  const WATCHDOG_STATUS_URL = "/comfydl/profiling/watchdog/status";
  const WATCHDOG_LOG_URL = "/comfydl/profiling/watchdog/log";
  const SERVER_LOG_URL = "/internal/logs/raw";
  const TAB_ID = "comfydl-profiling";
  const SETTING_LOG_LEVEL = "ComfyDL.Profiling.LogLevel";
  // P0: the real-execution probe is opt-in. The setting alone is not enough -
  // the header is attached per-request only while this flag is true.
  const SETTING_DANGEROUS = "ComfyDL.Profiling.DangerousProbe";
  const DANGEROUS_HEADER = "X-CDL-Profiling-Dangerous";
  const LOG_RANKS = { off: 0, low: 1, medium: 2, high: 3 };

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
      basis_tensor_shape: "static shape {shape}",
      basis_conv2d: "conv2d {in} x {kernel} -> {out}",
      basis_corr2d: "cross-correlation {in} x {kernel} -> {out}",
      basis_linreg: "matmul {x} @ {w} + b",
      basis_synthetic: "synthetic data: {examples} x {features}",
      basis_truncate_pad: "sequence -> {num_steps} steps (int64)",
      basis_boxes: "{boxes} box(es)",
      basis_nms: "kept count data-dependent (<= {upper_bound})",
      basis_multibox_prior: "{h}x{w} map, {per_pixel}/pixel -> {anchors} anchors",
      basis_multibox_target: "{anchors} anchors, batch {batch}",
      basis_multibox_detection: "{batch} x {anchors} predictions",
      basis_colormap: "VOC 256^3 lookup table (int64)",
      basis_label_indices: "{height}x{width} class-index mask",
      basis_rand_crop: "crop {height}x{width} (batch 1)",
      basis_image_scale: "resize {from} -> {to}",
      basis_load_image: "decode {name} (size known only after decode)",
      basis_module: "{kind}: {params} parameters",
      basis_cdl_tokenize: "{mode} tokenisation: {tokens} tokens over {lines} line(s)",
      basis_cdl_segments: "pairs -> {tokens} tokens ({first} + {second})",
      basis_cdl_vocab_encode: "{tokens} token(s) -> int64 indices",
      basis_cdl_vocab_decode: "{tokens} token(s) looked up",
      basis_weights_load: "{folder}: {name} ({file_bytes} bytes, CPU resident)",
      basis_weights_save: "staged {bytes} bytes for the file write",
      basis_vae_decode: "latent {latent} -> image {out} ({scale}x)",
      basis_vae_encode: "image {image} -> latent {latent} ({channels}ch /{scale})",
      basis_clip_encode: "{chars} chars -> {tokens} x {hidden} conditioning (assumed)",
      basis_render: "figure canvas {canvas} px (figsize {figsize}, {dpi} dpi)",
      basis_dataloader: "batch {batch} of {samples}, {batches} batches",
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
      compute_title: "Compute estimate",
      table_flops: "FLOPs",
      flops_none: "No node of this workflow has a compute model yet (pure-Python text nodes have none by design).",
      flops_note: "Training counts 3\u00d7 the forward pass; generation runs without a KV cache (one forward per token). Amounts, not times - throughput is measured, not guessed.",
      flops_unmodelled: "flops n/a",
      kind_attn: "attention",
      kind_ffn: "feed-forward",
      kind_conv: "convolution",
      kind_logits: "output head",
      kind_other: "other",
      timing_title: "Measured node time",
      timing_total: "Total",
      timing_cached: "cache hit(s)",
      timing_running: "Collecting timings...",
      timing_empty: "Run a workflow - every node's real time lands here.",
      watchdog_title: "Execution monitor",
      watchdog_unavailable: "monitoring unavailable (psutil missing on the server)",
      watchdog_bursts: "CPU burst log",
      watchdog_systemwide: "system-wide",
      // M3: operator-graph probe
      analyze: "Analyze operators",
      analyzing: "Analyzing...",
      reanalyze: "Graph changed - re-analyze",
      opgraph_title: "Operator graph (ATen level)",
      opgraph_none: "Press Analyze to expand every node into the ATen operators it really runs - FLOPs are counted bottom-up, per operator.",
      opgraph_stale_hint: "The graph changed since this analysis - results may be stale.",
      opgraph_summary: "{probed}/{nodes} nodes probed \u00b7 formula {formula} vs probe {probed} (ratio {ratio})",
      opgraph_fallback: "formula fallback: {error}",
      opgraph_col_op: "Operator",
      opgraph_col_count: "Calls",
      opgraph_col_flops: "FLOPs",
      opgraph_canned: "{n} data-dependent read(s) answered with a constant (training loops)",
      opgraph_expand: "Expand",
      opgraph_collapse: "Collapse",
      opgraph_pick: "Select a node",
      logs_title: "Server log (profiling)",
      logs_empty: "No profiling log lines yet - press Analyze or run a workflow with the log level raised.",
      opgraph_safe_mode: "Safe mode: the probe executes REAL node code, so it is disabled by default. Enable it under Settings \u2192 ComfyDL \u2192 Profiling \u2192 Dangerous probe.",
      danger_confirm: "The profiling probe executes REAL node code with REAL tensors. Very large inputs can consume significant CPU and memory while it runs. Enable the dangerous probe?",
      opgraph_guarded: "Input guardrail blocked {n} node(s) from probing - their string inputs exceed the 64 KB per-node cap, and the probe would otherwise loop over every byte for real: {list}. Shrink the text or feed it through a data node instead.",
      opgraph_assembled: "Equivalent graph assembled from the static rule library - no node was executed.",
      opgraph_zero_flops: "Total FLOPs is 0 because the formula ledger cannot model some of these nodes (see coverage) - this is a reporting limit, not a zero-cost graph.",
      opgraph_rule_hint: "Operator sequence from the rule library (not executed); FLOPs are the formula estimate for the actual shapes.",
      coverage_line: "Coverage: {covered} rule-derived \u00b7 {driven} formula-derived \u00b7 {unknown} FLOPs unmodelled \u00b7 {missing} no info",
      export_json: "JSON",
      export_outline: "Outline",
      export_mermaid: "Mermaid",
      opgraph_leaf: "no ATen ops (pure-Python leaf)",
      opgraph_probe_ms: "probe {ms} ms",
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
      basis_tensor_shape: "静态形状 {shape}",
      basis_conv2d: "卷积 {in} × {kernel} -> {out}",
      basis_corr2d: "互相关 {in} × {kernel} -> {out}",
      basis_linreg: "矩阵乘 {x} @ {w} + b",
      basis_synthetic: "合成数据：{examples} × {features}",
      basis_truncate_pad: "序列 -> {num_steps} 步（int64）",
      basis_boxes: "{boxes} 个框",
      basis_nms: "保留数取决于数据（≤ {upper_bound}）",
      basis_multibox_prior: "{h}×{w} 特征图，每像素 {per_pixel} -> {anchors} 个锚框",
      basis_multibox_target: "{anchors} 个锚框，批 {batch}",
      basis_multibox_detection: "{batch} × {anchors} 个预测",
      basis_colormap: "VOC 256³ 查找表（int64）",
      basis_label_indices: "{height}×{width} 类别索引掩码",
      basis_rand_crop: "裁剪 {height}×{width}（批 1）",
      basis_image_scale: "缩放 {from} -> {to}",
      basis_load_image: "解码 {name}（尺寸需解码后可知）",
      basis_module: "{kind}：{params} 参数",
      basis_cdl_tokenize: "{mode} 分词：{lines} 行 -> {tokens} 个 token",
      basis_cdl_segments: "句对 -> {tokens} 个 token（{first} + {second}）",
      basis_cdl_vocab_encode: "{tokens} 个 token -> int64 索引",
      basis_cdl_vocab_decode: "回查 {tokens} 个 token",
      basis_weights_load: "{folder}：{name}（{file_bytes} 字节，CPU 常驻）",
      basis_weights_save: "为写盘暂存 {bytes} 字节",
      basis_vae_decode: "潜变量 {latent} -> 图像 {out}（{scale}×）",
      basis_vae_encode: "图像 {image} -> 潜变量 {latent}（{channels} 通道 /{scale}）",
      basis_clip_encode: "{chars} 字符 -> {tokens} × {hidden} 条件（假设值）",
      basis_render: "画布 {canvas} 像素（figsize {figsize}，{dpi} dpi）",
      basis_dataloader: "每批 {batch}/{samples}，共 {batches} 批",
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
      compute_title: "计算量估算",
      table_flops: "计算量",
      flops_none: "当前工作流尚无可估算计算量的节点（纯 Python 文本节点本就不在模型内）。",
      flops_note: "训练按前向 3 倍计；生成无 KV cache（逐 token 完整前向）。这里只给计算量，不推算时间——吞吐要靠实测。",
      flops_unmodelled: "计算量未建模",
      kind_attn: "注意力",
      kind_ffn: "前馈",
      kind_conv: "卷积",
      kind_logits: "输出投影",
      kind_other: "其他",
      timing_title: "节点实测耗时",
      timing_total: "总计",
      timing_cached: "个缓存命中",
      timing_running: "正在采集耗时...",
      timing_empty: "运行一次工作流——每个节点的真实耗时都会落在这里。",
      watchdog_title: "执行监控",
      watchdog_unavailable: "监控不可用（服务端缺少 psutil）",
      watchdog_bursts: "瞬爆案底",
      watchdog_systemwide: "整机",
      // M3：算子图探针
      analyze: "分析算子",
      analyzing: "分析中...",
      reanalyze: "图已变更——重新分析",
      opgraph_title: "算子图（ATen 级）",
      opgraph_none: "点击「分析算子」把每个节点展开成它真正执行的 ATen 算子——FLOPs 按算子自底向上统计。",
      opgraph_stale_hint: "分析后图已变更——结果可能过期。",
      opgraph_summary: "{probed}/{nodes} 个节点已分析 · 公式 {formula} vs 探针 {probed}（比值 {ratio}）",
      opgraph_fallback: "公式回退：{error}",
      opgraph_col_op: "算子",
      opgraph_col_count: "次数",
      opgraph_col_flops: "FLOPs",
      opgraph_canned: "{n} 次数据依赖读取以常量应答（训练循环）",
      opgraph_expand: "展开",
      opgraph_collapse: "收起",
      opgraph_pick: "选择节点",
      logs_title: "服务端日志（profiling）",
      logs_empty: "暂无 profiling 日志——调高档位后点一次 Analyze 或跑一次工作流。",
      opgraph_safe_mode: "安全模式：探针会真实执行节点代码，默认关闭。到 设置 \u2192 ComfyDL \u2192 Profiling \u2192 Dangerous probe 开启后再分析。",
      danger_confirm: "profiling 探针将以真实张量真实执行节点代码。输入很大时可能占用大量 CPU 与内存。确定开启危险模式？",
      opgraph_guarded: "输入护栏拦截了 {n} 个节点——字符串输入超过单节点 64KB 上限（探针会逐字节真实处理）：{list}。请缩小文本，或改用文件 / 数据节点接入。",
      opgraph_assembled: "等效计算图由规则库推定，未执行任何节点。",
      opgraph_zero_flops: "FLOPs 总量为 0 是因为公式账本无法对其中部分节点建模（见覆盖率）——这是报告能力的边界，不代表图零开销。",
      opgraph_rule_hint: "算子序列来自规则库推定（未真实执行）；FLOPs 为公式按实际 shape 的估算值。",
      coverage_line: "覆盖：{covered} 规则推定 · {driven} 公式推定 · {unknown} FLOPs 无法建模 · {missing} 无信息",
      export_json: "JSON",
      export_outline: "大纲",
      export_mermaid: "Mermaid",
      opgraph_leaf: "无 ATen 算子（纯 Python 叶子节点）",
      opgraph_probe_ms: "探针 {ms} ms",
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
    // M2: measured timings (F1)
    timings: [],
    timingPromptId: null,
    timingStart: null,
    timingEnd: null,
    executing: false,
    currentNodeTitle: null,
    // M2: execution monitor (W2)
    liveEl: null,
    liveTimer: null,
    bursts: null,
    watchdogUnavailable: false,
    // UI: collapse state for the two long sections (CPU burst log / per-node
    // breakdown). Kept in state because renderPanel() rebuilds the DOM on every
    // update and would otherwise reset any DOM-only collapsed flag.
    // Burst log starts collapsed (2026-10-07 user feedback): it is a running
    // ledger, not the first thing to read.
    collapsed: { bursts: true, nodes: false },
    // M3: operator-graph probe (manual Analyze button - never auto-run).
    opgraph: null,
    opgraphLoading: false,
    opgraphStale: false,
    selectedOpNode: null,
    // M3+: pip-style verbosity dial (off/low/medium/high), driven by the
    // ComfyDL.Profiling.LogLevel setting and sent per-request as a header.
    logLevel: "off",
    // P0: real-execution probe gate (ComfyDL.Profiling.DangerousProbe).
    dangerousProbe: false,
    serverLog: null,
    serverLogLoading: false,
    // M3: display mode - "sidebar" (quick view) vs "overlay" (full dashboard).
    // Always start closed (2026-10-06 user feedback): auto-reopening the
    // full-screen dashboard over the workflow disorients the user - Expand is
    // an explicit action, so the mode is deliberately NOT persisted.
    overlayEl: null,
    overlayWrapEl: null,
    displayMode: "sidebar",
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

  function fmtFlops(n) {
    if (n === null || n === undefined || isNaN(n)) return "\u2014";
    const units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs"];
    let v = Number(n);
    let i = 0;
    while (Math.abs(v) >= 1000 && i < units.length - 1) { v /= 1000; i++; }
    return (i === 0 ? v.toFixed(0) : v.toFixed(2)) + " " + units[i];
  }

  function fmtDuration(ms) {
    if (ms === null || ms === undefined || isNaN(ms)) return "\u2014";
    if (ms < 1000) return Math.round(ms) + " ms";
    const s = ms / 1000;
    if (s < 60) return s.toFixed(s < 10 ? 2 : 1) + " s";
    const m = Math.floor(s / 60);
    return m + "m " + Math.round(s - m * 60) + "s";
  }

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .split("&").join("&amp;")
      .split("<").join("&lt;")
      .split(">").join("&gt;")
      .split('"').join("&quot;")
      .split("'").join("&#39;");
  }

  // ------------------------------------------------- diagnostics logging --
  // pip-style dial (off/low/medium/high): state.logLevel drives both the JS
  // console lines (plog) and the per-request X-CDL-Profiling-Log header that
  // raises the backend verbosity without a server restart.
  function logRank() { return LOG_RANKS[state.logLevel] || 0; }

  function plog(rank, msg) {
    try {
      if ((LOG_RANKS[rank] || 0) > logRank()) return;
      console.log("[ComfyDL profiler][" + rank + "] " + msg);
    } catch (e) { /* logging must never break the panel */ }
  }

  function profHeaders(extra) {
    const h = extra || {};
    h["X-CDL-Profiling-Log"] = state.logLevel;
    return h;
  }

  // P0: turn the Dangerous probe toggle back off (user declined the confirm).
  async function revertDangerous() {
    state.dangerousProbe = false;
    try {
      const em = getApp() && getApp().extensionManager;
      if (em && em.setting && typeof em.setting.set === "function") {
        await em.setting.set(SETTING_DANGEROUS, false);
      }
    } catch (e) { /* private mode / unavailable: local state already off */ }
    renderPanel();
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
        headers: profHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });
      plog("high", "Estimate: server answered " + resp.status);
      if (resp.ok) {
        state.report = await resp.json();
        renderPanel();
        updateBadge();
      }
    } catch (e) {
      plog("low", "Estimate failed: " + (e && e.message ? e.message : e));
    } finally {
      state.estimating = false;
    }
    return state.report;
  }

  function scheduleEstimate() {
    if (state.timer) clearTimeout(state.timer);
    markOpgraphStale();
    state.timer = setTimeout(() => requestEstimate(true), 500);
  }

  // -------------------------------------------------- M3: operator graph --
  // Manual-only: the FakeTensor probe costs real milliseconds per node, so it
  // never runs automatically and never sits on the Run path.
  // History note (2026-10-06): the first render used to sit OUTSIDE the try
  // block and called an undefined renderOverlay() - the ReferenceError aborted
  // the function after loading=true but before the fetch, wedging the button
  // on "Analyzing" forever. Everything is inside try/finally now: the loading
  // flag always resets, whatever happens.
  async function requestOpgraph() {
    const api = getApi();
    if (!api || state.opgraphLoading) return state.opgraph;
    state.opgraphLoading = true;
    state.opgraphStale = false;
    const t0 = Date.now();
    plog("low", "Analyze: requested (prompt nodes: "
      + Object.keys(state.lastPrompt || {}).length + ")");
    try {
      renderPanel();
      renderOverlay();
      const gtp = await callOriginalGTP();
      if (gtp && gtp.output) state.lastPrompt = gtp.output;
      const opHeaders = profHeaders({ "Content-Type": "application/json" });
      // P0: the probe executes real node code - only attach the acknowledgement
      // header while the user's Dangerous probe setting is on.
      if (state.dangerousProbe) opHeaders[DANGEROUS_HEADER] = "1";
      const resp = await api.fetchApi(OPGRAPH_URL, {
        method: "POST",
        headers: opHeaders,
        body: JSON.stringify({ prompt: state.lastPrompt || {} }),
      });
      plog("medium", "Analyze: server answered " + resp.status);
      if (resp.ok) {
        state.opgraph = await resp.json();
        plog("low", "Analyze: done in " + (Date.now() - t0) + "ms (totals: "
          + JSON.stringify((state.opgraph && state.opgraph.totals) || {}) + ")");
      } else {
        plog("low", "Analyze: server error " + resp.status);
      }
    } catch (e) {
      plog("low", "Analyze failed: " + (e && e.message ? e.message : e));
    } finally {
      state.opgraphLoading = false;
      renderPanel();
      renderOverlay();
    }
    return state.opgraph;
  }

  function markOpgraphStale() {
    if (state.opgraph && !state.opgraphStale) {
      state.opgraphStale = true;
      plog("high", "opgraph marked stale (graph changed since last Analyze)");
      renderPanel();
      renderOverlay();
    }
  }

  function setDisplayMode(mode) {
    state.displayMode = mode === "overlay" ? "overlay" : "sidebar";
    plog("high", "display mode -> " + state.displayMode);
    renderOverlay();
  }

  function renderOpgraphSummary() {
    const og = state.opgraph;
    const btnLabel = esc(state.opgraphLoading ? t("analyzing")
      : state.opgraphStale ? t("reanalyze") : t("analyze"));
    const btn = '<button class="cdlp-button" id="cdlp-analyze"'
      + (state.opgraphLoading ? " disabled" : "") + ">" + btnLabel + "</button>";
    if (!og) {
      return '<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("opgraph_title")) + "</div>"
        + '<div class="cdlp-hint">' + esc(t("opgraph_none")) + "</div>" + btn + "</div>";
    }
    const totals = og.totals || {};
    const summary = t("opgraph_summary", {
      probed: totals.probed, nodes: totals.nodes,
      formula: fmtFlops(totals.flops_formula),
      probedFlops: fmtFlops(totals.flops_probed),
      ratio: totals.ratio === null || totals.ratio === undefined ? "\u2014" : totals.ratio,
    });
    let html = '<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("opgraph_title")) + "</div>";
    html += '<div class="cdlp-flops-total">' + esc(fmtFlops(totals.flops_probed)) + "</div>";
    html += '<div class="cdlp-hint">' + esc(summary) + "</div>";
    // P1: assembled mode badge + honest coverage line + exports.
    if (og.mode === "assembled") {
      html += '<div class="cdlp-hint">' + esc(t("opgraph_assembled")) + "</div>";
      const cov = og.coverage || {};
      html += '<div class="cdlp-hint">' + esc(t("coverage_line", {
        covered: cov.covered || 0, driven: cov.params_driven || 0,
        unknown: cov.unknown || 0, missing: cov.missing || 0,
      })) + "</div>";
      // The always-rendered picture (2026-10-07 user feedback): a text export
      // alone is not the "直观" default the P1 plan promised.
      html += renderEquivalentGraphSvg(og);
      html += '<div class="cdlp-hint"><span class="cdlp-title-right">'
        + '<button class="cdlp-button" id="cdlp-export-json">' + esc(t("export_json")) + "</button> "
        + '<button class="cdlp-button" id="cdlp-export-outline">' + esc(t("export_outline")) + "</button> "
        + '<button class="cdlp-button" id="cdlp-export-mermaid">' + esc(t("export_mermaid")) + "</button></span></div>";
    }
    if (state.opgraphStale) html += '<div class="cdlp-hint cdlp-stale">' + esc(t("opgraph_stale_hint")) + "</div>";
    // P0: guardrail hits must never be silent - name the blocked nodes and
    // say why (0 FLOPs on its own looks like the probe "did nothing").
    const guarded = (og.nodes || []).filter(
      (n) => n.status === "fallback" && /input too large/.test(n.error || ""));
    if (guarded.length) {
      const list = guarded.slice(0, 4)
        .map((n) => "#" + n.id + " " + n.class_type).join(", ")
        + (guarded.length > 4 ? " \u2026" : "");
      html += '<div class="cdlp-hint cdlp-stale">'
        + esc(t("opgraph_guarded", { n: guarded.length, list })) + "</div>";
    }
    // Whole-graph guardrail / other report-level errors: the report carries
    // the reason in `error` - surface it instead of showing a bare 0 FLOPs.
    if (og.error && !og.nodes.length && !guarded.length) {
      html += '<div class="cdlp-hint cdlp-stale">' + esc(String(og.error)) + "</div>";
    }
    // P1: in assembled mode a 0 FLOPs total usually means the formula ledger
    // could not model the (unknown-coverage) nodes - say so explicitly.
    if (og.mode === "assembled" && !totals.flops_probed
        && (og.coverage && og.coverage.unknown)) {
      html += '<div class="cdlp-hint cdlp-stale">'
        + esc(t("opgraph_zero_flops")) + "</div>";
    }
    // P0: safe/timeout runs carry a mode flag - say so where the user looks.
    if (og.mode === "safe") {
      html += '<div class="cdlp-hint cdlp-stale">' + esc(t("opgraph_safe_mode")) + "</div>";
    } else if (og.mode === "timeout") {
      html += '<div class="cdlp-hint cdlp-stale">' + esc(String(og.error || t("opgraph_safe_mode"))) + "</div>";
    }
    html += btn;
    html += "</div>";
    return html;
  }

  function opCensusTable(node, showFlops) {
    const rows = (node.ops || []).map((row) => {
      const name = row.op.replace(/^aten\./, "").replace(/\.default$/, "");
      const flops = showFlops ? esc(fmtFlops(row.flops)) : "\u2014";
      return "<tr><td>" + esc(name) + "</td><td>" + row.count + "</td><td>"
        + flops + "</td></tr>";
    });
    return '<table class="cdlp-op-table"><thead><tr>'
      + "<th>" + esc(t("opgraph_col_op")) + "</th><th>" + esc(t("opgraph_col_count"))
      + "</th><th>" + esc(t("opgraph_col_flops")) + "</th></tr></thead><tbody>"
      + (rows.join("") || '<tr><td colspan="3">' + esc(t("opgraph_leaf")) + "</td></tr>")
      + "</tbody></table>";
  }

  // P1 exports: the assembled report as JSON / a text outline / mermaid.
  function downloadText(filename, text) {
    try {
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    } catch (e) { plog("low", "export failed: " + e); }
  }

  // P1: hand-drawn SVG equivalent graph (zero dependencies - the host bundle
  // has no mermaid/d3 and we never add npm packages).  One box per workflow
  // node (rule-derived = teal border), its ATen op chain stacked inside,
  // workflow links drawn as dashed connectors.  Mermaid text stays available
  // as an export; this is the always-rendered picture.
  function renderEquivalentGraphSvg(og) {
    const prompt = state.lastPrompt || {};
    const nodes = og.nodes || [];
    if (!nodes.length) return "";
    const COL_W = 230, ROW_H = 310, W = 205, OP_H = 20, TITLE_H = 22, PAD = 8;
    const perRow = 5;
    const pos = {};
    const parts = [];
    let maxBottom = 40;
    const opLabel = (o) => o.op.replace(/^aten\./, "").replace(/\.default$/, "")
      + (o.count > 1 ? " \u00d7" + o.count : "");
    nodes.forEach((n, idx) => {
      const col = idx % perRow, row = Math.floor(idx / perRow);
      const ops = (n.ops || []).slice(0, 6);
      const h = TITLE_H + (ops.length ? ops.length * OP_H : OP_H) + PAD * 2;
      const x = PAD + col * COL_W;
      const y = 40 + row * ROW_H;
      pos[String(n.id)] = { x, y, w: W, h };
      maxBottom = Math.max(maxBottom, y + h);
      const stroke = n.status === "rule" ? "#1ABC9C"
        : n.status === "probed" ? "#2ECC71" : "#7f8c8d";
      parts.push('<rect x="' + x + '" y="' + y + '" width="' + W + '" height="' + h
        + '" rx="6" fill="' + (n.status === "rule" ? "#173f3a" : "#2c2f33")
        + '" stroke="' + stroke + '"/>');
      parts.push('<text x="' + (x + 8) + '" y="' + (y + 15)
        + '" class="cdlp-svg-title">' + esc(n.class_type.slice(0, 24))
        + " #" + esc(String(n.id)) + "</text>");
      if (ops.length) {
        ops.forEach((o, i) => {
          parts.push('<text x="' + (x + 10) + '" y="' + (y + TITLE_H + i * OP_H + 14)
            + '" class="cdlp-svg-op">' + esc(opLabel(o)) + "</text>");
        });
        const more = (n.ops || []).length - ops.length;
        if (more > 0) {
          parts.push('<text x="' + (x + 10) + '" y="' + (y + TITLE_H + ops.length * OP_H + 14)
            + '" class="cdlp-svg-more">+' + more + "</text>");
        }
      } else {
        parts.push('<text x="' + (x + 10) + '" y="' + (y + TITLE_H + 14)
          + '" class="cdlp-svg-more">'
          + esc(n.status === "rule" ? t("opgraph_leaf") : (n.error || "\u2014")) + "</text>");
      }
    });
    Object.entries(prompt).forEach(([target, node]) => {
      Object.values(node.inputs || {}).forEach((v) => {
        if (Array.isArray(v) && v.length === 2 && pos[String(v[0])] && pos[String(target)]) {
          const pa = pos[String(v[0])], pb = pos[String(target)];
          const x1 = pa.x + pa.w, y1 = pa.y + TITLE_H + 10;
          const x2 = pb.x, y2 = pb.y + TITLE_H + 10;
          const mx = (x1 + x2) / 2;
          parts.push('<path d="M ' + x1 + " " + y1 + " C " + mx + " " + y1 + ", "
            + mx + " " + y2 + ", " + x2 + " " + y2
            + '" fill="none" stroke="#888" stroke-dasharray="4 3"/>');
        }
      });
    });
    const svgW = PAD * 2 + Math.min(perRow, nodes.length) * COL_W;
    return '<div class="cdlp-svg-wrap"><svg width="' + svgW + '" height="'
      + (maxBottom + 10) + '" xmlns="http://www.w3.org/2000/svg">'
      + parts.join("") + "</svg></div>";
  }

  function mermaidSafe(s) {
    // split/join instead of regex-with-quotes: the bracket-balance linter
    // (T4a) strips strings/comments but not regex literals (memory pit 24).
    let out = String(s);
    out = out.split('"').join("'");
    out = out.split("{").join("(").split("}").join(")");
    return out;
  }

  function buildMermaid(og) {
    // One subgraph per workflow node, its ATen op chain inside; workflow
    // links become edges between the subgraphs' first/last operators.
    const prompt = state.lastPrompt || {};
    const byId = {};
    (og.nodes || []).forEach((n) => { byId[String(n.id)] = n; });
    const nid = (id) => "n" + String(id).replace(/\W/g, "_");
    const lines = ["flowchart LR"];
    (og.nodes || []).forEach((n) => {
      const id = nid(n.id);
      const title = mermaidSafe(n.class_type + " #" + n.id);
      const ops = (n.ops || []).slice(0, 6).map((o) => {
        const base = o.op.replace(/^aten\./, "").replace(/\.default$/, "");
        return base + (o.count > 1 ? " x" + o.count : "");
      });
      if (!ops.length) {
        lines.push('  ' + id + '("' + title + '")');
        return;
      }
      lines.push('  subgraph sg_' + id + '["' + title + '"]');
      ops.forEach((op, i) => {
        lines.push('    ' + id + "_op" + i + '["' + mermaidSafe(op) + '"]');
        if (i) lines.push("    " + id + "_op" + (i - 1) + " --> " + id + "_op" + i);
      });
      lines.push("  end");
    });
    Object.entries(prompt).forEach(([target, node]) => {
      Object.values(node.inputs || {}).forEach((v) => {
        if (Array.isArray(v) && v.length === 2 && byId[String(v[0])] && byId[String(target)]) {
          const src = byId[String(v[0])];
          const srcOps = (src.ops || []).length;
          const dstOps = ((byId[String(target)] || {}).ops || []).length;
          const from = srcOps ? nid(v[0]) + "_op" + (Math.min(srcOps, 6) - 1) : nid(v[0]);
          const to = dstOps ? nid(target) + "_op0" : nid(target);
          lines.push("  " + from + " --> " + to);
        }
      });
    });
    return lines.join("\n");
  }

  function buildOutline(og) {
    const lines = ["Equivalent computation graph (rule library, zero execution)", ""];
    (og.nodes || []).forEach((n) => {
      lines.push("#" + n.id + " " + n.class_type + "  [" + n.status
        + ", coverage=" + (n.coverage || "?") + ", FLOPs="
        + fmtFlops(n.total_flops) + "]");
      if (n.ops && n.ops.length) {
        n.ops.forEach((o) => {
          lines.push("    " + o.op.replace(/^aten\./, "").replace(/\.default$/, "")
            + "  x" + o.count);
        });
      } else {
        lines.push("    (" + t("opgraph_leaf") + ")");
      }
      if (n.error) lines.push("    ! " + n.error);
      lines.push("");
    });
    return lines.join("\n");
  }

  function exportAssembled(kind) {
    const og = state.opgraph;
    if (!og) return;
    const stamp = new Date().toISOString().slice(0, 10);
    if (kind === "json") {
      downloadText("equivalent_graph_" + stamp + ".json", JSON.stringify(og, null, 1));
    } else if (kind === "outline") {
      downloadText("equivalent_graph_" + stamp + ".txt", buildOutline(og));
    } else if (kind === "mermaid") {
      downloadText("equivalent_graph_" + stamp + ".mmd", buildMermaid(og));
    }
    plog("high", "exported assembled graph as " + kind);
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
    plog("low", "OOM post-mortem: requesting (node " + d.node_id + ")");
    try {
      const resp = await api.fetchApi(POSTMORTEM_URL, {
        method: "POST",
        headers: profHeaders({ "Content-Type": "application/json" }),
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
    } catch (e) {
      plog("low", "Post-mortem failed: " + (e && e.message ? e.message : e));
    }
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

  // ------------------------------------------------------- M2: timing + monitor --
  function nodeTitleOf(id) {
    if (id === null || id === undefined) return "?";
    const graph = getApp() && getApp().graph;
    const node = graph && graph._nodes
      ? graph._nodes.find((n) => String(n.id) === String(id)) : null;
    return node ? (node.title || node.type || String(id)) : String(id);
  }

  function displayIdOf(detail) {
    const d = detail || {};
    return (d.display_node !== undefined && d.display_node !== null) ? d.display_node : d.node;
  }

  function handleExecuting(detail) {
    const d = detail || {};
    if (d.node === null || d.node === undefined) { handleRunEnded(); return; }
    state.executing = true;
    if (state.timingPromptId !== d.prompt_id) {
      state.timingPromptId = d.prompt_id;
      state.timings = [];
      state.timingStart = Date.now();
      state.timingEnd = null;
    }
    const id = displayIdOf(d);
    state.currentNodeTitle = nodeTitleOf(id);
    state.timings.push({ id: id, title: state.currentNodeTitle, start: Date.now(), end: null });
    startLivePolling();
  }

  function handleExecuted(detail) {
    const id = displayIdOf(detail);
    for (let i = state.timings.length - 1; i >= 0; i--) {
      if (String(state.timings[i].id) === String(id) && state.timings[i].end === null) {
        state.timings[i].end = Date.now();
        break;
      }
    }
  }

  function handleRunEnded() {
    state.executing = false;
    state.currentNodeTitle = null;
    state.timingEnd = Date.now();
    stopLivePolling();
    fetchBursts();
    renderPanel();
  }

  async function pollWatchdog() {
    const api = getApi();
    if (!api) return;
    try {
      const resp = await api.fetchApi(WATCHDOG_STATUS_URL, { headers: profHeaders() });
      plog("high", "Watchdog poll: " + resp.status);
      if (resp.ok) {
        const snap = await resp.json();
        state.watchdogUnavailable = snap.available === false;
        updateLiveIndicator(snap);
      }
    } catch (e) { /* silent: the monitor is best-effort */ }
  }

  function startLivePolling() {
    if (state.liveTimer || !state.executing) return;
    pollWatchdog();
    state.liveTimer = setInterval(pollWatchdog, 1000);
  }

  function stopLivePolling() {
    if (state.liveTimer) { clearInterval(state.liveTimer); state.liveTimer = null; }
    const el = state.liveEl;
    if (el) el.classList.add("cdlp-live-hidden");
  }

  function updateLiveIndicator(snap) {
    const el = state.liveEl;
    if (!el) return;
    if (!snap || !snap.available || !state.executing || !state.currentNodeTitle) {
      el.classList.add("cdlp-live-hidden");
      return;
    }
    const cpu = Number(snap.process_cpu_percent);
    const cpuClass = isNaN(cpu) ? "cdlp-cpu-ok"
      : cpu >= 90 ? "cdlp-cpu-hot" : cpu >= 60 ? "cdlp-cpu-warn" : "cdlp-cpu-ok";
    const name = String(state.currentNodeTitle);
    el.innerHTML =
      '<span class="cdlp-live-node">' + esc(name.length > 18 ? name.slice(0, 17) + "\u2026" : name) + "</span>"
      + '<span class="cdlp-live-stat ' + cpuClass + '">CPU ' + esc(String(snap.process_cpu_percent)) + "%</span>"
      + '<span class="cdlp-live-stat">MEM ' + esc(String(snap.process_rss_mb)) + "MB</span>";
    el.classList.remove("cdlp-live-hidden");
  }

  function mountLiveIndicator(anchor) {
    try {
      const el = document.createElement("div");
      el.className = "cdlp-live cdlp-live-hidden";
      anchor.before(el);
      state.liveEl = el;
    } catch (e) { state.liveEl = null; }
  }

  async function fetchBursts() {
    const api = getApi();
    if (!api) return;
    try {
      const resp = await api.fetchApi(WATCHDOG_LOG_URL + "?limit=20", { headers: profHeaders() });
      plog("high", "Burst log fetch: " + resp.status);
      if (resp.ok) {
        const body = await resp.json();
        state.bursts = body.events || [];
        renderPanel();
      }
    } catch (e) { /* silent */ }
  }

  // M3+: pull the app-wide server log ring buffer and keep only the lines
  // this panel produced ([profiling:...] from the backend, [ComfyDL profiler]
  // fallback from toast()). Shown in the sidebar card whenever the level is
  // raised - the fastest way to see what the backend actually did.
  async function fetchServerLog() {
    const api = getApi();
    if (!api) return;
    state.serverLogLoading = true;
    renderPanel();
    try {
      const resp = await api.fetchApi(SERVER_LOG_URL, { headers: profHeaders() });
      if (resp.ok) {
        const body = await resp.json();
        const lines = (body.entries || [])
          .map((e) => String(e.m || ""))
          .filter((m) => m.indexOf("[profiling:") !== -1 || m.indexOf("[ComfyDL profiler]") !== -1);
        state.serverLog = lines.slice(-60).join("");
        plog("high", "Server log: " + lines.length + " profiling line(s) in ring buffer");
      }
    } catch (e) {
      plog("low", "Server log fetch failed: " + (e && e.message ? e.message : e));
    }
    state.serverLogLoading = false;
    renderPanel();
  }

  function flopsKindLabel(kind) {
    const key = "kind_" + kind;
    return t(key) === key ? (kind || "other") : t(key);
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
        mountLiveIndicator(anchor);
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
      mountLiveIndicator(anchor);
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
    if (!el && !state.overlayEl) return;
    const r = state.report;
    const pm = state.postmortem;
    const budget = (r && r.budget) || null;
    const used = budget ? Math.max(0, budget.total_bytes - budget.free_bytes) : null;

    const html = [];
    html.push('<div class="cdlp-panel">');
    html.push('<div class="cdlp-toolbar"><button class="cdlp-button cdlp-expand" id="cdlp-expand">'
      + esc(t("opgraph_expand")) + "</button></div>");

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

    // M3: operator-graph summary + Analyze button (manual, never auto-run)
    html.push(renderOpgraphSummary());

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

    // Compute ledger (M2)
    const flops = (r && r.flops) || null;
    if (flops && flops.any_estimated) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("compute_title")) + "</div>");
      html.push('<div class="cdlp-flops-total">' + esc(fmtFlops(flops.total)) + "</div>");
      const byKind = flops.by_kind || {};
      const flopsTotal = Math.max(1, Number(flops.total) || 0);
      html.push('<div class="cdlp-kindbar">');
      ["attn", "ffn", "conv", "logits", "other"].forEach((k) => {
        const share = ((Number(byKind[k]) || 0) / flopsTotal) * 100;
        if (share > 0) html.push('<div class="cdlp-k-' + k + '" style="width:' + share.toFixed(1) + '%"></div>');
      });
      html.push("</div>");
      const parts = ["attn", "ffn", "conv", "logits"].filter((k) => byKind[k])
        .map((k) => flopsKindLabel(k) + " " + ((Number(byKind[k]) / flopsTotal) * 100).toFixed(0) + "%");
      if (parts.length) html.push('<div class="cdlp-hint">' + esc(parts.join(" \u00b7 ")) + "</div>");
      html.push('<div class="cdlp-hint">' + esc(t("flops_note")) + "</div>");
      html.push("</div>");
    } else if (r && r.nodes && r.nodes.length) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("compute_title")) + "</div>"
        + '<div class="cdlp-hint">' + esc(t("flops_none")) + "</div></div>");
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

    // Measured node time (M2 / F1)
    if (state.timingStart) {
      const closed = (state.timings || []).filter((x) => x.end !== null);
      const totalMs = (state.timingEnd || Date.now()) - state.timingStart;
      const active = closed.filter((x) => x.end - x.start >= 5)
        .sort((a, b) => (b.end - b.start) - (a.end - a.start));
      const cachedCount = closed.length - active.length;
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("timing_title")) + "</div>");
      html.push('<div class="cdlp-timing-total"><span>' + esc(t("timing_total")) + "</span><b>"
        + esc(fmtDuration(totalMs)) + "</b></div>");
      if (active.length) {
        const maxD = Math.max.apply(null, active.map((x) => x.end - x.start));
        active.slice(0, 12).forEach((x) => {
          const d = x.end - x.start;
          const width = Math.max(3, Math.round((Math.log(Math.max(2, d)) / Math.log(maxD || 2)) * 100));
          html.push('<div class="cdlp-timing-row">'
            + '<span class="cdlp-timing-name">' + esc(x.title || x.id) + "</span>"
            + '<span class="cdlp-timing-bar"><i style="width:' + width + '%"></i></span>'
            + '<span class="cdlp-timing-val">' + esc(fmtDuration(d)) + "</span></div>");
        });
      }
      if (cachedCount) html.push('<div class="cdlp-hint">' + cachedCount + " \u00d7 " + esc(t("timing_cached")) + "</div>");
      if (!closed.length) html.push('<div class="cdlp-hint">' + esc(t("timing_running")) + "</div>");
      html.push("</div>");
    } else if (state.executing) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("timing_title")) + "</div>"
        + '<div class="cdlp-hint">' + esc(t("timing_running")) + "</div></div>");
    }

    // Execution monitor burst log (M2 / W2)
    if (state.watchdogUnavailable) {
      html.push('<div class="cdlp-card"><div class="cdlp-card-title">' + esc(t("watchdog_title")) + "</div>"
        + '<div class="cdlp-hint">' + esc(t("watchdog_unavailable")) + "</div></div>");
    } else if (state.bursts && state.bursts.length) {
      html.push('<div class="cdlp-card cdlp-wd' + (state.collapsed.bursts ? " cdlp-collapsed" : "") + '">'
        + '<div class="cdlp-card-title cdlp-wd-title cdlp-collapsible" data-collapse="bursts">'
        + '<span class="cdlp-collapse-label">' + esc(t("watchdog_bursts")) + "</span>"
        + '<i class="cdlp-chevron"></i></div>'
        + '<div class="cdlp-card-body">');
      state.bursts.slice().reverse().forEach((ev) => {
        html.push('<div class="cdlp-burst">');
        html.push('<div class="cdlp-burst-head"><b>' + esc(ev.node_id ? nodeTitleOf(ev.node_id) : "?") + "</b>"
          + (ev.system_wide ? ' <span class="cdlp-tag cdlp-tag-sys">' + esc(t("watchdog_systemwide")) + "</span>" : "")
          + '<span class="cdlp-burst-time">' + esc(fmtDuration((Number(ev.duration_s) || 0) * 1000)) + "</span></div>");
        html.push('<div class="cdlp-burst-meta">CPU ' + esc(String(ev.cpu_percent_peak)) + "% peak \u00b7 "
          + esc(String(ev.cpu_percent_avg)) + "% avg \u00b7 MEM " + esc(String(ev.process_rss_mb)) + "MB \u00b7 "
          + esc(String(ev.ts || "")) + "</div>");
        if (ev.data_scale && ev.data_scale.class_type) {
          html.push('<div class="cdlp-burst-scale">' + esc(String(ev.data_scale.class_type))
            + (ev.data_scale.flops_total ? " \u00b7 " + esc(fmtFlops(ev.data_scale.flops_total)) : "") + "</div>");
        }
        html.push("</div>");
      });
      html.push("</div></div>");
    }

    // Per-node breakdown
    html.push('<div class="cdlp-card' + (state.collapsed.nodes ? " cdlp-collapsed" : "") + '">'
      + '<div class="cdlp-card-title cdlp-collapsible" data-collapse="nodes">'
      + '<span class="cdlp-collapse-label">' + esc(t("table_title")) + "</span>"
      + '<span class="cdlp-title-right">'
      + '<button class="cdlp-button cdlp-refresh" id="cdlp-refresh">' + esc(t("refresh")) + "</button>"
      + '<i class="cdlp-chevron"></i></span></div>'
      + '<div class="cdlp-card-body">');
    const nodes = (r && r.nodes) || [];
    nodes.forEach((n) => {
      const unknown = n.status !== "estimated";
      html.push('<details class="cdlp-node' + (unknown ? " cdlp-node-unknown" : "") + '">');
      html.push('<summary><span class="cdlp-node-title">' + esc(n.title || n.id) + "</span>");
      if (unknown) html.push('<span class="cdlp-tag">' + esc(t("unknown_node")) + "</span>");
      else {
        html.push('<span class="cdlp-node-bytes">' + esc(fmtBytes(n.total_bytes)));
        if (n.flops_total !== null && n.flops_total !== undefined) {
          html.push(" \u00b7 " + esc(fmtFlops(n.flops_total)));
        } else if (n.flops_status === "unknown") {
          html.push(' <i class="cdlp-flops-na">(' + esc(t("flops_unmodelled")) + ")</i>");
        }
        html.push("</span>");
      }
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
      if (n.flops_items && n.flops_items.length) {
        html.push('<table class="cdlp-table"><tr><th>' + esc(t("table_node"))
          + "</th><th>" + esc(t("table_flops")) + "</th></tr>");
        n.flops_items.forEach((it) => {
          html.push("<tr><td>" + esc(flopsKindLabel(it.kind)) + (it.approx ? ' <i>(' + esc(t("approx_tag")) + ")</i>" : "")
            + "</td><td>" + esc(fmtFlops(it.flops)) + "</td></tr>");
        });
        html.push("</table>");
      }
      // M3/P1: the ATen census for this node (probed live or rule-derived).
      if (state.opgraph) {
        const ogNode = (state.opgraph.nodes || []).find(
          (row) => String(row.id) === String(n.id));
        if (ogNode && ogNode.status === "probed") {
          html.push('<div class="cdlp-hint">'
            + esc(t("opgraph_canned", { n: ogNode.data_dependent_reads }))
            + " \u00b7 " + esc(t("opgraph_probe_ms", { ms: ogNode.probe_ms })) + "</div>");
          html.push(opCensusTable(ogNode, true));
        } else if (ogNode && ogNode.status === "rule") {
          html.push('<div class="cdlp-hint">' + esc(t("opgraph_rule_hint")) + "</div>");
          html.push(opCensusTable(ogNode, false));
        } else if (ogNode && ogNode.status === "fallback") {
          html.push('<div class="cdlp-hint cdlp-stale">'
            + esc(t("opgraph_fallback", { error: ogNode.error || "" })) + "</div>");
        }
      }
      html.push("</details>");
    });
    if (!nodes.length) html.push('<div class="cdlp-hint">' + esc(t("loading")) + "</div>");
    html.push("</div></div>");

    // M3+: server-side profiling log tail (visible only when level != off)
    if (state.logLevel !== "off") {
      html.push('<div class="cdlp-card' + (state.collapsed.serverlog ? " cdlp-collapsed" : "") + '">'
        + '<div class="cdlp-card-title cdlp-collapsible" data-collapse="serverlog">'
        + '<span class="cdlp-collapse-label">' + esc(t("logs_title")) + "</span>"
        + '<span class="cdlp-title-right">'
        + '<button class="cdlp-button" id="cdlp-logs-refresh">' + esc(t("refresh")) + "</button>"
        + '<i class="cdlp-chevron"></i></span></div>'
        + '<div class="cdlp-card-body">');
      if (state.serverLogLoading || state.serverLog === null || !state.serverLog) {
        html.push('<div class="cdlp-hint">' + esc(t("logs_empty")) + "</div>");
      } else {
        html.push('<pre class="cdlp-pm-msg">' + esc(state.serverLog) + "</pre>");
      }
      html.push("</div></div>");
    }

    html.push('<div class="cdlp-disclaimer">' + esc(t("disclaimer")) + "</div>");
    html.push("</div>");

    const htmlString = html.join("");
    [el, state.overlayEl].forEach((target) => {
      if (!target) return;
      target.innerHTML = htmlString;
      wirePanel(target);
    });
  }

  function wirePanel(el) {
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
    // Collapsible section titles: click the title to fold / unfold the card.
    el.querySelectorAll(".cdlp-collapsible").forEach((titleEl) => {
      titleEl.addEventListener("click", (ev) => {
        // Let the embedded Refresh button act without folding the card.
        if (ev.target.closest("button")) return;
        const key = titleEl.getAttribute("data-collapse");
        const card = titleEl.closest(".cdlp-card");
        if (!key || !card) return;
        state.collapsed[key] = card.classList.toggle("cdlp-collapsed");
      });
    });
    const refresh = el.querySelector("#cdlp-refresh");
    if (refresh) refresh.addEventListener("click", () => requestEstimate(true));
    const apply = el.querySelector("#cdlp-apply-suggestion");
    if (apply) apply.addEventListener("click", applySuggestion);
    const analyze = el.querySelector("#cdlp-analyze");
    if (analyze) analyze.addEventListener("click", () => requestOpgraph());
    const expand = el.querySelector("#cdlp-expand");
    if (expand) expand.addEventListener("click", () => {
      plog("high", "Expand clicked");
      mountOverlay();
      setDisplayMode("overlay");
    });
    const logsRefresh = el.querySelector("#cdlp-logs-refresh");
    if (logsRefresh) logsRefresh.addEventListener("click", () => fetchServerLog());
    ["json", "outline", "mermaid"].forEach((kind) => {
      const btn = el.querySelector("#cdlp-export-" + kind);
      if (btn) btn.addEventListener("click", () => exportAssembled(kind));
    });
  }

  // M3: the full-dashboard overlay. Same content as the sidebar (one render
  // into both containers), just roomier - the op census deserves the space.
  function mountOverlay() {
    if (state.overlayEl) return;
    const wrap = document.createElement("div");
    wrap.className = "cdlp-overlay";
    wrap.innerHTML = '<div class="cdlp-overlay-inner"><div class="cdlp-overlay-head">'
      + '<span class="cdlp-overlay-title">' + esc(t("sidebar_title")) + " \u00b7 "
      + esc(t("opgraph_title")) + "</span>"
      + '<button class="cdlp-button" id="cdlp-collapse">' + esc(t("opgraph_collapse")) + "</button></div>"
      + '<div class="cdlp-overlay-body"></div></div>';
    document.body.appendChild(wrap);
    state.overlayEl = wrap.querySelector(".cdlp-overlay-body");
    state.overlayWrapEl = wrap;
    wrap.querySelector("#cdlp-collapse").addEventListener("click", () => setDisplayMode("sidebar"));
    wrap.addEventListener("click", (ev) => {
      if (ev.target === wrap) setDisplayMode("sidebar");
    });
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape" && state.displayMode === "overlay") setDisplayMode("sidebar");
    });
    renderPanel();
  }

  // Syncs the overlay's visibility with the display mode. This function was
  // MISSING until 2026-10-06 (four call sites -> ReferenceError), which wedged
  // Analyze on "Analyzing" and made Expand a no-op. It only toggles the open
  // class; content always flows through renderPanel() into both containers.
  function renderOverlay() {
    const wrap = state.overlayWrapEl;
    if (!wrap) return;
    wrap.classList.toggle("cdlp-overlay-open", state.displayMode === "overlay");
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
        if (state.bursts === null) fetchBursts();
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
      api.addEventListener("executing", (event) => {
        try { handleExecuting(event && event.detail); } catch (e) { /* silent */ }
      });
      api.addEventListener("executed", (event) => {
        try { handleExecuted(event && event.detail); } catch (e) { /* silent */ }
      });
      api.addEventListener("execution_error", (event) => {
        try { handleExecutionError(event && event.detail); } catch (e) { /* silent */ }
        try { handleRunEnded(); } catch (e) { /* silent */ }
      });
    }
    await waitFor(() => getApp() && getApp().extensionManager);
    try { registerSidebar(); } catch (e) { /* sidebar optional */ }
    try { await mountBadge(); } catch (e) { /* badge optional */ }
    try { mountOverlay(); } catch (e) { /* overlay optional */ }
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
        settings: [
        {
          id: SETTING_LOG_LEVEL,
          name: "ComfyDL profiling log level",
          type: "combo",
          defaultValue: "off",
          options: [
            { value: "off", text: "Off" },
            { value: "low", text: "Low" },
            { value: "medium", text: "Medium" },
            { value: "high", text: "High" },
          ],
          onChange: (newVal) => {
            state.logLevel = LOG_RANKS[newVal] !== undefined ? newVal : "off";
            plog("high", "log level -> " + state.logLevel);
            if (state.logLevel !== "off" && state.serverLog === null) fetchServerLog();
            renderPanel();
          },
        },
        {
          // P0: the real-execution probe is opt-in. Enabling asks for a
          // lightweight confirmation; declining reverts the toggle (which
          // re-enters onChange with false - benign).
          id: SETTING_DANGEROUS,
          name: "ComfyDL profiling dangerous probe (executes real node code)",
          type: "boolean",
          defaultValue: false,
          onChange: (newVal) => {
            state.dangerousProbe = newVal === true;
            plog("high", "dangerous probe -> " + state.dangerousProbe);
            if (state.dangerousProbe && !window.confirm(t("danger_confirm"))) {
              revertDangerous();
            }
            renderPanel();
          },
        },
        ],
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
