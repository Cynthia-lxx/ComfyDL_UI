# Profiling Tools — M1：内存静态估算

> 新增于 v0.3.2（reform: Profiling Tools M1）。四层架构的内存分析工具：纯后端估算引擎
> → HTTP API → 仓库前端扩展 → 设备预算。本文是设计记录 + 人工验收清单。
> 前端注入链路见 `docs/branding-and-frontend-patch.md` 第 8 节。

## 1. 背景：13.3GB 的一次性分配

1MB Tiny Shakespeare 压测：`window=64 / vocab=256 / d_model=256 / 2×TransformerBlock(4头,
FFN 1024) / AdamW / steps=300 / batch_size=0`（= 全库一批）。第一步即

```
DefaultCPUAllocator: not enough memory: you tried to allocate 13,276,741,632 bytes
```

16GB 机器、空闲 2.96GB。反推：13,276,741,632 ÷ 4 ÷ 64 ÷ 256 ≈ B = 202,587 样本——
模型本身只有 ~1.7M 参数（~7MB），纯粹是 `batch_size=0` 把整个数据集当作了一个 batch，
k_proj 输出张量一项就是 12.36 GB。

事故的教训不是这一行代码，而是**整条流水线没有任何一层会提前告诉你放不下**。M1 补上
事前（估算+三色判定+Run 二次确认）与事后（报错归因+建议参数）两段。

## 2. 范围（用户已拍板）

- **只做 Memory Size 分析**，不建性能/算子融合等其他功能的空入口；
- **不动任何节点行为**：运行时护栏（LanguageModelTrain 的 max_batch_tokens）推迟到
  后续批次，因此本批无 FUNCTIONS/README 四件套更新；
- 红色判定 = **确认框 + 可强制继续**（教学自由优先，不硬阻断）；
- 事后分析纳入 M1；
- **中英双语文案**全覆盖（面板/徽章/确认框/事后分析）。

## 3. 架构

```
前端扩展 app/profiling_assets/profiler.js
  ├─ 侧栏页（registerSidebarTab，官方 API）
  ├─ 顶栏徽章（ComfyButtonGroup，Manager 官方手法；失败降级 raw DOM/仅侧栏）
  ├─ Run 拦截：monkey-patch app.graphToPrompt（队列提交必经点）
  ├─ 图变更：api.addEventListener("graphChanged") → 防抖 500ms
  └─ OOM 捕获：api.addEventListener("execution_error")
        ↓ fetchApi（/api 前缀孪生路由）
server.py  POST /comfydl/profiling/estimate | /postmortem（app/profiling_routes.py）
        ↓
引擎 comfy/profiling/（纯模块，零 torch 执行、零 UI 依赖）
  ├─ engine.py     拓扑序（Kahn）抽象求值 + 三色判定 + 报告聚合
  ├─ estimators.py ESTIMATORS 注册表（16 类节点；未注册 → 显式 unknown）
  ├─ formulas.py   内存公式库（上界近似 + 单张量追踪）
  ├─ shapes.py     值对象（TensorVal/SpecVal/VocabVal/Unknown…）
  ├─ assumptions.py 假设层（默认档位 + 用户覆盖 + 使用记录）
  └─ postmortem.py 报错字节数解析 + 张量归属 + 建议 batch_size（二分搜索）
        ↓
设备预算：comfy.model_management get_total/free_memory（与 /system_stats 同源）
```

估算输入是 `graphToPrompt()` 的输出（prompt 格式：`{node_id: {class_type, inputs}}`，
连线为 `[源节点, 槽位]`、控件为字面值）——名字寻址，无 widgets_values 顺序问题。

## 4. 内存公式（上界近似）

训练峰值（LanguageModelTrain / TrainingLoop）：

```
P（参数）+ G（梯度=P）+ 优化器状态（AdamW/Adam 2P；RMSprop P；SGD 0）
+ 早停快照（若开启，P）
+ 数据集常驻（contexts (S,T) + targets (S,T) int64 + y (S,)）
+ 反向传播保留的激活：逐块展开
    B×T×E 级：norm1/q/k/v/合并/残差/norm2/FFN 输出
    B×H×T×T：注意力分数 + softmax 权重
    B×T×F：FFN 隐层（线性+激活）
+ logits B×T×V + log-softmax 同级
```

`single_bytes` 与总量并行记录——「任一单张量 > 空闲内存」是比总量更硬的失败判据
（golden case 的 13,276,741,632 正是 B×T×E×4 的 q/k/v 输出）。推理（Forward/Generate）
= P + no_grad 传递的上界激活。

公式与实现逐一对照 `comfy_extras/nodes_lm.py`、`nodes_nlp.py`（`batch_size<=0 或 ≥samples`
都表示整库一批；`iter_windows` 样本数 = `len − window`）。

## 5. 三色判定

| 颜色 | 判据 | 行为 |
|---|---|---|
| 绿 SAFE | 峰值 < 空闲×0.7 | 无 |
| 黄 MAYBE OOM | 介于两者之间 | toast 提醒（15s 节流），不阻断 |
| 红 CERTAIN OOM | 峰值 ≥ 总量×0.9 **或任一单张量 > 空闲** | Run 时确认框，可「仍然运行」 |

诚实原则：未注册节点类型 → 显式 `unknown`（灰显 + 原因），绝不猜；全部未知时 verdict
= unknown。假设层只补真正的洞（如未估算上游的流长度 → 默认档位），报告里逐条列
「key / 值 / 来源(默认|覆盖)」，面板行内可改，估算即时刷新。

## 6. 事后分析（postmortem）

`execution_error` 事件携带 `exception_message`；解析 CPU（`tried to allocate N bytes`）
与 CUDA（`Tried to allocate N.NN GiB/GB`）两种拼写 → 结合最近一次提交的 prompt 重新
估算 → 归属到「哪个节点哪个张量」（字节数±1% 精确匹配）→ 对 LanguageModelTrain/
TrainingLoop 二分搜索**可行的最大 batch_size**（判定标准：峰值与最大单张量都 ≤ 空闲×0.7）
→ 面板给出「设置 batch_size = N」一键应用按钮（直改图上 widget 并触发重估）。

## 7. 测试（`cdl_smoke_tests/test_profiling.py`，37 项全绿）

- T1 金样例：vocab=256、B=202,587、q/k/v 单张量 **精确等于** 13,276,741,632、
  红/single_exceeds_free、FFN 隐层是最大单张量；
- T2 公式交叉验证：与 `lm_protocol.build_model()` 真实物化模型的参数量逐一相等
  （唯一用 torch 的地方，仅为对照）；
- T3 默认教学图绿；T4 未知节点/未知输入 → unknown + 假设层记录；T5 覆盖生效；
- T6 batch 语义镜像节点（0/超大 → 整库）；T7 TrainingLoop（SGD 无优化器状态）；
- T8 postmortem：三种拼写解析、归属精确匹配、建议值回代自检为绿；
- T9 路由层集成（真实 aiohttp TestServer）：estimate/postmortem 200、坏 JSON 400、
  资产经静态路由可达（JS/CSS Content-Type 正确）；
- T10 loader 注入幂等（临时 web root，单次注入、二次空跑）。

基线无回退：`run_smoke_test.py` 290 PASS / 23 SKIP / 0 FAIL。

## 8. 降级链（任何一环失败都不影响正常工作流）

1. 徽章：ComfyButton 不可用 → raw DOM 胶囊 → 只剩侧栏 + 命令面板
   （`ComfyDL_Profiling_Open`）；
2. 确认框：`extensionManager.dialog.confirm` → `window.confirm`；
3. graphChanged 监听失效 → 侧栏打开/刷新按钮/Run 时刻仍会重估；
4. 报错无字节数 → 原样展示不解读；引擎异常 → 该节点标 error，报告不崩。

## 9. 人工验收清单（P 盘，`python main.py` 后浏览器操作）

| # | 操作 | 期望 |
|---|---|---|
| 1 | Network 面板过滤 `profiling` | `profiler.js` / `profiler.css` 均 200 |
| 2 | 看顶栏（设置齿轮左侧） | 出现内存徽章（首次估算前 `…`，随后等宽字体数字 + 三色） |
| 3 | 打开任一含 LM 训练的工作流 | 徽章 1~2s 内变色；点徽章 → 打开 Profiling 侧栏 |
| 4 | 侧栏内容 | 三格预算卡 + verdict 徽章 + 峰值数字 + 免责小字；分解表按节点折叠，含依据行 |
| 5 | 改一个 d_model 控件 | ~600ms 后徽章/面板数字刷新（防抖 500ms） |
| 6 | 载入金样例（batch_size=0、大语料） | 红 CERTAIN OOM；点 Run → 弹醒目确认框（峰值/最大单张量/免责）；「仍然运行」继续跑，「取消」终止且报错文本可读 |
| 7 | 把 batch_size 改回小值再 Run | 绿或黄；黄只 toast 不弹框 |
| 8 | 故意制造一次 OOM（金样例原参数跑到底） | 报错 toast；Profiling 面板出现红色「事后分析」卡：分配失败字节数 + 归属张量 + 「设置 batch_size = N」按钮；点按钮 → 图上 batch_size 被改，估算刷新 |
| 9 | 界面语言切中文（Settings→Language） | 刷新后侧栏/徽章/确认框/事后分析全中文 |
| 10 | 重启服务 | 日志有 profiling loader 已应用（且第二次启动不再重复注入）；Network 无重复 profiler.js |

F 盘（GPU 环境）同步后再过一遍 1/2/3/6/8（预算来自 CUDA 设备口径）。

## 10. M2 候选

- 运行时护栏（max_batch_tokens，用户已选定后置）；
- 用真实执行校准激活系数（当前是保守上界）；符号形状完整化；
- 覆盖更多节点族（RNN scratch / 卷积族 / Embedding 训练）；
- 徽章悬停明细卡、假设档位设置页。
