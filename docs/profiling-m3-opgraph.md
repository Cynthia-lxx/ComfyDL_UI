# Profiling M3: 等效基础算子计算图（opgraph 探针）

> 2026-10-06 交付。来源：用户想法——"在 Profiling 里做出当前工作流的等效基础算子计算图"。
> 本文档记录口径、机制、诚实边界与分期；M1（内存估算）与 M2（计算量账本 + 实测耗时 + watchdog）
> 的设计文档 §10 是本批的直接上游。

## 1. 概念与定位

**基础算子粒度 = ATen**。把当前工作流的每个节点"展开"成它真正执行的 ATen 算子序列
（`aten.addmm` / `aten.mm` / `aten._softmax` / `aten.conv2d` / ...），得到一份**算子级计算图**：

- `Regression Train` 这类档 A 节点（显式组合派）会分解到 matmul/softmax/linear 级；
- `conv2d` / `batch_norm` 这类档 B 节点在 ATen 层就是**单算子**——再往里是 cuDNN 内核，
  不是 Python 可见的（诚实边界）。

与 M2 公式账本的关系：**双口径并存交叉验证**。账本自顶向下（逐族手写公式），探针自底向上
（算子实际执行）；两者应落在容差内，偏差本身就是 bug 信号。探针失败（副作用节点等）的
节点**降级回公式值**并标记 `fallback`——报告永远完整，只有保真度分级。

## 2. 触发模型（用户可用性优先）

| 分析层 | 成本 | 触发 |
|---|---|---|
| M1 内存估算 + M2 公式账本 | 纯公式，微秒级 | 500ms 防抖自动重估（现状不变） |
| **M3 算子探针** | **真实执行**节点代码（真实张量），毫秒~秒级；P0 起默认关闭，需显式危险模式（§8） | **手动 Analyze 按钮**，异步执行，永不自动、永不进入 Run 路径 |
| W2 watchdog | 运行时旁观 | 随 Run（现状） |

Analyze 结果按 prompt 规范化 hash 缓存（进程内，上限 8 份）：图没变秒回；图变了按钮变
"重新分析"。Queue 按钮全程可用。

## 3. 探针机制（comfy/profiling/opgraph.py）

```
prompt（graphToPrompt 输出）→ engine._topological_order
→ 逐节点（拓扑序）:
    _resolve_inputs（wired 输入 ← 上游探针的真实输出; 常量原样）
    P0 护栏检查（字符串输入 > 64KB → fallback，不执行）
    FlopCounterMode + _ProbeRecorder（最外层）
      execute(**inputs)   ← 真实代码、真实张量（见 §8 事实修正）
    → ATen 调用序列 + 每 op FLOPs + 数据依赖读取计数
→ census（op/count/FLOPs 排序）+ totals（双口径 + ratio）
```

- **`_ProbeRecorder`（最外层 TorchDispatchMode）**：记录 ATen 调用名；拦截
  `aten._local_scalar_dense`（`float(tensor)` 数据依赖读取）返回常量 0.5——控制流变确定性，
  循环体恰好执行 widget 声明的步数（`data_dependent_reads` 入报告可见）。
- **FlopCounterMode**（torch 2.1+ 公开 API）：`get_total_flops()` + `get_flop_counts()`
  （**双层 dict** `{module: {op: flops}}`——解析时按 `str(op)` 归并）。
- **⚠ 事实修正（2026-10-06，P0）**：早期文档与 docstring 宣称"FakeTensorMode 包裹、
  零真实内存"——**该 import 从未被使用**，探针自交付起就是真实执行 + 真实张量。
  内存/算力安全因此完全依赖 §8 的护栏与危险模式门控（这正是 1MB corpus 事故的根因）。
- **降级**：逐节点 try/except；OP 上限 `OPS_LIMIT=250_000` 防大步数爆量；副作用节点
  （torch.save 等）自然抛错 → fallback。fallback **级联**（上游没探成，下游无输入）。
- **注册表引导**：裸解释器里宿主 `nodes` 只有模块级 utilities → `_bootstrap_registry()`
  惰性 `asyncio.run(init_extra_nodes(...))`（coroutine，勿直接调用；勿在 bootstrap 阶段
  import comfydl——会遮蔽宿主 `nodes` 模块名）。
- **重要实测教训（__import__）**：受限 eval（`__builtins__: {}`）里对 fake tensor 做
  `3 * tensor`（标量在左的 `__rmul__`）会经 torch overrides 延迟 import → 在空 builtins
  里查 `__import__` → `KeyError: '__import__'`。`data_gen._eval_formula` 的 namespace
  改为 `{"__import__": __import__}`（AST 白名单已禁调用，开口不可达，安全）。

## 4. 报告口径

```json
{
  "version": 1,
  "totals": {"nodes", "probed", "fallback", "flops_formula", "flops_probed", "ratio"},
  "nodes": [{"id", "class_type", "status": "probed|fallback",
             "ops": [{"op", "count", "flops"}], "ops_total",
             "total_flops", "formula_flops", "data_dependent_reads",
             "error", "traceback", "probe_ms"}],
  "disclaimer_key": "opgraphDisclaimer"
}
```

- engine 的 `REPORT_VERSION` 保持 **2** 不动（M1/M2 报告契约不变，测试不回退）；
  opgraph 报告自带 `version: 1`，两文档独立。
- 端点：`POST /comfydl/profiling/opgraph`（body `{prompt}`），经 server.py 既有逐条
  重声明自动获得 `/api` 孪生；探针在**专用单线程池**（`cdl-probe`）里跑，不占共享
  默认 executor。

## 5. 前端（profiler.js/css，仓库资产零补丁）

- **Analyze 按钮**（三态：分析算子 / 分析中... / 图已变更——重新分析）+ 汇总卡
  （probed/fallback、公式 vs 探针、ratio）。
- **双显示模式**：侧拉（速览）vs **全屏叠加 overlay**（完整 dashboard：verdict + budget +
  compute + op census）。同一 HTML 渲染进侧栏与 overlay 两个容器（`wirePanel` 对两容器
  各自接线）；侧栏工具条 `展开` 按钮 / overlay 内 `收起` / Esc / 点击遮罩关闭；
  模式**不再持久化**（2026-10-06 用户反馈：启动自动展开"跳脸"——恒从收起态开始）。
- **op census 表**：per-node `<details>` 内（probed 节点显示算子/次数/FLOPs 表 +
  canned reads + probe 耗时；fallback 节点显示原因）。全部 cdlp- 前缀 + ComfyUI 主题变量。
- i18n：js 内嵌 zh/en 字典（不走 gen_locales，与 M1/M2 同惯例）。

## 6. 测试与验证

- `cdl_smoke_tests/test_profiling_opgraph.py`（**20 项**）：金样例回归工作流全节点
  probed、trainer census 含 addmm/mse_loss、FLOPs>0、canned reads 可见、2MNK 精确
  （linear 2·1·10·20）、缓存同对象、换图重探、空 prompt / 未知节点降级、
  aiohttp TestServer 路由集成（`client_max_size=100MB`，坑 23；TestServer 需
  `await client.start_server()`）。
- JS 语法自查：剥字符串/注释后括号配平（坑 24 范式）。
- 全量冒烟 301/33/0 不回退（P/F 双盘）。

## 7. 诚实边界与后置项

- 粒度 = ATen，不到内核；数据依赖控制流（早停）的执行次数以 widget 声明为准（canned
  标量使早停分支确定性但不代表真实早停步数）——口径已在报告中标注。
- 后置：F4 roofline（算子强度 = FLOPs ÷ 输出字节，字节已在探针侧可采）、W3 建议库
  （未融合链检测：显式 softmax+matmul 相邻等）、算子 DAG 可视化、F5 时间预警。

## 8. Safety model（P0，2026-10-06）

> 事故：用户往 VocabBuild/TextEncode 填入 1MB+ tiny shakespeare 后点 Analyze →
> 探针真实执行 CPU/内存吃满、共享事件循环被 GIL 饿死（estimate 一并卡死）、
> Ctrl+C 无法终止工作线程、只能 taskkill。

**四道防线**（全部可独立生效）：

1. **危险模式门控（结构性主防线）**：探针是 opt-in。前端注册
   `ComfyDL.Profiling.DangerousProbe` boolean 设置（默认关，开启时 `confirm` 轻确认，
   拒绝即回拨）；仅当开启时请求附加 `X-CDL-Profiling-Dangerous: 1` 头。**无头的
   Analyze 零执行**——路由直接返回 `mode: "safe"` 报告（全节点 fallback，error=
   `probe disabled (dangerous mode off)`），前端渲染安全模式提示卡。
2. **输入规模护栏**（`comfy/profiling/opgraph.py`，常量可调）：
   `MAX_NODE_INPUT_BYTES = 64KB`（单节点字符串输入超限 → 该节点 fallback
   `input too large for probe`，不进 execute）；`MAX_TOTAL_INPUT_BYTES = 2MB`
   （全图字符串总量超限 → 提前返回错误报告）。下游节点经
   "upstream not probed" 自然级联，无需额外处理。护栏在 `probe_workflow` 内
   生效——离线采样工具（P1）同样受保护。
3. **执行器隔离 + 硬超时**：探针从共享默认 executor 移入专用单线程池
   （`ThreadPoolExecutor(max_workers=1, thread_name_prefix="cdl-probe")`）；
   路由侧 `asyncio.wait_for(..., timeout=PROBE_DEADLINE_SECONDS + 5)`（60s 初值）。
   **边界声明：Python 线程不可杀**——超时只是"放弃 future 并立即应答"，worker 会在
   后台跑完并被丢弃（结果不入缓存）；真正阻止资源占用的还是 1+2。
4. **协作 deadline**：`probe_workflow(prompt, deadline=...)` 在节点间检查
   `time.monotonic()`，超点后剩余节点标 `probe deadline exceeded` 不再执行——
   让"放弃"发生在最近的节点边界，缩小后台残留时长。

**报告形态**：safe-mode / timeout 报告同为 `version: 1` 完整结构 + 额外
`"mode": "safe" | "timeout"` 字段；带 error 的报告**不写入缓存**（改输入或开危险模式
后重试即重新探针）。

**与 P1 的关系**：规则库（离线采样 + 在线纯查表组装）上线后，危险模式仅剩
"采样复核"用途；safe-mode 报告是 P1 前的占位提示。

## 9. P1：等效计算图组装器（2026-10-07，回归原始目标）

Analyze 的**默认路径**已从探针切换为 `opgraph.assemble_workflow`——零节点执行，把两个
静态源按节点合并成等效计算图：

| 来源 | 提供 | 实现 |
|---|---|---|
| **规则库** `comfy/profiling/oprules.py` + `data/op_rules.json` | 每节点的 ATen 算子 census（结构：哪些算子、调用多少次） | 离线采样器 `comfy/profiling/sampler.py`（CLI，`python -m comfy.profiling.sampler`，复用 `_probe_one` + P0 护栏 + UI/网络黑名单 + 30s 守护超时）批量生成；可随注册表演进重跑 |
| **M2 公式账本** `engine.estimate_workflow` | shape 感知的 per-node FLOPs（纯公式，零执行） | 组装器直接读取其 nodes 的 flops_total/flops_status |

报告 `mode:"assembled"`；per-node `status:"rule"`（census 来自规则库，**未真实执行**，
FLOPs 为公式估算）或 `fallback`（无规则，error=`no rule for this node type`）；顶层
`coverage` 四档：`covered`（有 census）/ `params_driven`（M2 公式确定性给出，含
zero）/ `unknown`（M2 都无法建模）/ `missing`（无规则无 estimator）——诚实标注，
不猜。首版基线：**153 条规则 / 181 复核**；LM 模板 5 rule + 6 params_driven + 1
unknown + 0 missing（首次调用含一次性 import 成本 ~4.5s，稳态 <0.5s）；tabular 模板
5 rule。前端：assembled 徽标 + 覆盖率行 + rule census 表（FLOPs 列显示 —，数值看
M2 公式）+ 导出器 JSON / 文本大纲 / mermaid 语法文本（前端 bundle 无 mermaid
渲染器——已核实 vendor chunk 全枚举——渲染成图留 P3，绝不引入 npm 依赖）。

**采样器教训（memory 坑 28）**：全注册表真实执行必须防 UI/网络副作用节点——
`CdlMessageBox` 曾以默认 block(wait) 参数被采样执行，用户桌面被原生对话框轰炸且
采样卡死。防线 = 类名黑名单（messagebox/dialog/popup/download/upload）+ 每节点
30s 守护线程超时（超时放弃、进程继续）。V3 节点兼容：`INPUT_TYPES()` 返回 **list**
而非 tuple 且 combo 记作 `"COMBO"`+meta.options——采样器两者都吃。
