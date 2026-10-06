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
| **M3 算子探针** | FakeTensor 逐节点执行，毫秒~秒级 | **手动 Analyze 按钮**，异步执行，永不自动、永不进入 Run 路径 |
| W2 watchdog | 运行时旁观 | 随 Run（现状） |

Analyze 结果按 prompt 规范化 hash 缓存（进程内，上限 8 份）：图没变秒回；图变了按钮变
"重新分析"。Queue 按钮全程可用。

## 3. 探针机制（comfy/profiling/opgraph.py）

```
prompt（graphToPrompt 输出）→ engine._topological_order
→ 逐节点（拓扑序）:
    _resolve_inputs（wired 输入 ← 上游探针的 fake 输出缓存; 常量原样）
    FakeTensorMode 会话 + FlopCounterMode + _ProbeRecorder（最外层）
      execute(**inputs)
    → ATen 调用序列 + 每 op FLOPs + 数据依赖读取计数
→ census（op/count/FLOPs 排序）+ totals（双口径 + ratio）
```

- **`_ProbeRecorder`（最外层 TorchDispatchMode）**：记录 ATen 调用名；拦截
  `aten._local_scalar_dense`（`float(tensor)` 数据依赖读取）返回常量 0.5——控制流变确定性，
  循环体恰好执行 widget 声明的步数（`data_dependent_reads` 入报告可见）。recorder 必须
  **最外层**：在 fake 模式抛 `DataDependentOutputException` 之前接住。
- **FlopCounterMode**（torch 2.1+ 公开 API）：`get_total_flops()` + `get_flop_counts()`
  （**双层 dict** `{module: {op: flops}}`——解析时按 `str(op)` 归并）。
- **零内存**：FakeTensor 全程伪造；拓扑序上游 fake 输出直接喂下游。
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
  重声明自动获得 `/api` 孪生；探针在 `run_in_executor` 线程里跑，事件循环不卡。

## 5. 前端（profiler.js/css，仓库资产零补丁）

- **Analyze 按钮**（三态：分析算子 / 分析中... / 图已变更——重新分析）+ 汇总卡
  （probed/fallback、公式 vs 探针、ratio）。
- **双显示模式**：侧拉（速览）vs **全屏叠加 overlay**（完整 dashboard：verdict + budget +
  compute + op census）。同一 HTML 渲染进侧栏与 overlay 两个容器（`wirePanel` 对两容器
  各自接线）；侧栏工具条 `展开` 按钮 / overlay 内 `收起` / Esc / 点击遮罩关闭；
  模式记忆 `localStorage.cdlpDisplayMode`。
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
