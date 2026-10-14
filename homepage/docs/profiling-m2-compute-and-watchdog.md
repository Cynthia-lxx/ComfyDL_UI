# Profiling Tools — M2：计算量账本 + 逐节点实测耗时 + 瞬爆监控

## 1. 背景

M1 交付了内存静态估算（峰值/三色判定/事后归因，见
`profiling-m1-memory-estimation.md`）。用户随后明确了两个新事实：

1. **项目定位已从教学工具转向真实生产与高效调优**——时间与算力成本成为
   核心指标；OOM 保护的是"不浪费一次运行"，时间/算力账本保护的是
   "不浪费一晚上"。
2. **瞬爆问题**：某些节点会瞬间吃满系统资源（例证：大语料下的
   `TextVocabBuild`——纯 Python 逐 token 循环——把 CPU 吃满，整机卡顿，
   其他程序无法流畅使用）。

## 2. 范围（用户已拍板）

M2 = **F1 + F2 + W2 最小闭环**（三者互为校准）：

| 项 | 内容 | 本批 |
|---|---|---|
| F1 | 逐节点实测耗时面板（executing/executed 事件） | ✅ |
| F2 | 静态 FLOPs 分解（attn/ffn/logits） | ✅ |
| W2 | watchdog 瞬爆监控 + 案底 + 顶栏实时指示 | ✅ |
| W1 | 静态地雷扫描（成本类标注） | M3 |
| W3 | 具体优化建议库 | M3 |
| F3/F4/F5 | 实测吞吐/MFU、roofline 归因、时间预警 | M3 |

**诚实边界**：本批只报计算量（FLOPs）与实测事实（耗时、CPU/RSS 采样），
不做时间预测、不报吞吐效率——那需要 M3 的实测校准。前端明确标注
"Amounts, not times"。

## 3. 架构

```
profiler.js（M1 资产扩展，无新注入项）
  ├─ F1: executing/executed → 时间线 → 耗时 Top-N 面板
  ├─ F2: report.flops → 汇总卡 + 节点行 FLOPs + 明细表
  └─ W2: 执行期 1s 轮询 → 顶栏实时胶囊 + 案底红卡
        ↓ fetchApi（/api 孪生自动生成）
app/profiling_routes.py（新增 2 个 GET）
  ├─ /comfydl/profiling/watchdog/status   实时快照
  └─ /comfydl/profiling/watchdog/log       案底（jsonl 尾部）
app/profiling_watchdog.py（新模块，服务层）
  └─ psutil 250ms 采样线程 + last_node_id 归因 + 瞬爆状态机 + jsonl
comfy/profiling/（M1 引擎扩展）
  └─ formulas.flops_* / estimators.flops_items / Report["flops"]
```

关键决策：watchdog 放 `app/`（服务侧基础设施）而非 `comfy/profiling/`
（保持纯模块先例：引擎不 import 服务机制）。

## 4. FLOPs 公式（matmul-only，乘加计 2）

与 M1 同源的符号形状（B/T/E/V/F/L）直接复用，只加第二本账：

- **每块注意力**：q/k/v/o 四投影 `8·B·T·E²`；`QK^T` 与 `A@V` 两个
  `T×T×d_head` GEMM 共 `4·B·T²·E`（与头数无关，`d_head=E/H`）→
  attn = `8·B·T·E² + 4·B·T²·E`；
- **每块前馈**：两条 `E↔F` 线性 `4·B·T·E·F`；
- **输出头**：`2·B·T·E·V`；
- **训练**：`3 × 前向 × steps`（GEMM 反向≈前向 2 倍；优化器逐元素不
  计——Kaplan et al. 2020 的 6ND 同款假设），每项标 `approx`；
- **生成**：`lm_protocol.generate_tokens` **无 KV cache**——每步对整段
  序列重跑前向，总账 = 长度 `L, L+1, …, L+count-1` 各一次前向之和
  （>10⁶ 步时退化为 max-length×count 上界）；
- **MLP（TrainingLoop）**：`2·B·Σ(lᵢ·lᵢ₊₁)`，训练同乘 `3×steps`；
- **排除项**：embedding 查表、softmax、layernorm 等逐元素操作（相对
  GEMM 是噪音，6ND 亦然）。

**节点口径**（`flops_status`）：

- `estimated`：flops_items 有账（Train/Forward/Generate/TrainingLoop）；
- `zero`：直通/IO（Build/Save/Load/Decode/Embedding/Block/Optimizer）；
- `unknown`：纯 Python 循环（VocabBuild/Encode/SlidingWindow）——
  它们的成本不是 matmul FLOPs 模型，诚实记 unknown，M3 的 W1 用
  成本类另建模型。

**报告口径**：内存是"峰值"（节点共享设备），计算量是**可加**的
（各节点自己的工作）——`report.flops.total = Σ 被估算节点`。
版本号 `REPORT_VERSION = 2`。

**交叉验证**：金样例（202,587×64×256、2 块、300 步）逐层展开 vs
`6·N·D`（N=参数量，D=steps×B×T）比值落在 [0.95, 1.25]
（超出部分 = 注意力二次项 `4·B·T²·E`，测试 F2a 固定该区间）。

## 5. 耗时语义（F1，纯前端零后端改造）

- 数据源：现有 WebSocket 事件——`executing`（含 `display_node`/
  `prompt_id`，`execution.py:496`）与 `executed`（`execution.py:578`）；
  `node: null`（`main.py:420`）标记整图结束。
- **缓存命中**：executing→executed 间隔 < 5ms 的节点标注
  "cache hit(s)"，不进 Top-N 主榜（不误导）。
- 条形长度按**对数刻度**（数量级差异过大时的可读折衷）。
- 时间线按 prompt_id 隔离：新一次运行自动清零重采。

## 6. Watchdog（W2）

**采样**：守护线程每 250ms 一拍：进程 `cpu_percent()`（**per-core 口径**：
单核跑满≈100，N 核跑满≈100N）、`psutil.cpu_percent()`（系统级，0-100
归一）、`memory_info().rss`。首次调用预热一拍（psutil 首拍返回 0）。

**归因**：直接读 `server.last_node_id` / `server.last_prompt_id`
（`executing` 事件出口维护，本批补了 `last_prompt_id` 一行）——
零事件管道改造。

**瞬爆判据**（状态机 `_evaluate`，合成样本可直接驱动测试）：

- 开启：进程 CPU ≥ 90% **或** 系统 CPU ≥ 90%，持续 ≥ 2s；
- 关闭：负载回落到阈值 × 0.8 以下连续 2 拍，或执行节点切换；
- `system_wide` 标志区分"本进程吃满"与"整机被拖卡"。

阈值/间隔可用环境变量覆盖：`COMFYDL_WATCHDOG_CPU/_SUSTAIN/_INTERVAL/_LOG`。

**案底**：`user/comfydl/profiling_watchdog.jsonl`（append-only，重启不丢）。
事件字段：`ts / duration_s / node_id / prompt_id / cpu_percent_peak /
cpu_percent_avg / system_cpu_percent_peak / system_wide / process_rss_mb /
threshold / cores / data_scale`。`data_scale` 是与 M1 引擎的第一次握手：
从最近一次 estimate 报告取该节点的 class_type / total_bytes / flops_total
（查不到记 null，不阻塞）。

**实时指示**：前端**仅在执行期**以 1s 轮询 `watchdog/status`（空闲零
请求），顶栏徽章右侧胶囊显示当前节点名（截断）+ CPU%（绿/黄/红分档）
+ MEM MB。

## 7. 降级链（任何一环失败都不影响正常工作流）

- psutil 导入失败 → watchdog 静默禁用（`available: false`，面板显示
  "监控不可用"）；
- 采样线程异常 → 单拍吞掉，绝不外泄；
- jsonl 写失败 → 事件进内存环形缓冲（64 条），`events_dropped` 计数，
  告警一次；
- 轮询失败 → 静默重试；`stop()` join 超时 5s；
- watchdog 全程不阻断任何工作流执行（提示不阻断哲学）；
- F2 unknown 节点灰显，不猜数。

## 8. 测试（`cdl_smoke_tests/test_profiling_m2.py`，26 项）

- **F1a-F1f**：金样例训练 FLOPs = 独立第一性重算（每块 attn/ffn/logits
  手工推导）；by-kind 拆分与总和自洽；最大分项归属 trainer；训练项
  全部 approx 标记；纯 Python 节点 unknown / 直通节点 zero；版本 2；
- **F2a/F2b**：6ND 交叉验证区间 + 规则本身精确等于 6·N·D；
- **F3a-F3c**：Forward 单次精确；Generate 逐步求和（无 KV cache）；
  生成项 approx 标记；
- **F4a**：TrainingLoop MLP 训练 FLOPs 精确（3×前向×steps）；
- **F5a/F5b**：unknown/空图不发明数字；
- **W1a-W4d**：psutil 可用性；合成样本驱动的状态机（单次 burst、字段、
  system_wide）；真线程 start/snapshot 归因/stop join；TestServer 路由
  （无实例 503 / 有实例 200）。

另：`test_profiling.py` 37 项全部保持绿（M1 行为零回退）；
**真启动规程**（踩坑 25）：watchdog 挂在服务构造层，已用
`python main.py --port 8199` 实测——`/api/comfydl/profiling/watchdog/status`
返回实时采样（进程 CPU/系统 CPU/RSS），log 路由 200。

## 9. 人工验收清单（P 盘，`python main.py` 后浏览器操作）

> 界面字体已在交付后按用户要求换为 Gmarket Sans（子集），仅施加于数值/
> 状态类元素（判定徽章、大数字、表格数值列、按钮）；卡片标题、节点行、
> 顶栏徽章沿用系统字体。取舍与限制见 §11。

1. **F2 汇总卡**：搭任一训练图（如默认教学图）→ Profiling 侧栏在
   "设备预算"卡下方出现"计算量估算"卡：总量（自动换算
   K/M/G/T/P FLOPs）+ 蓝/绿/黄构成条 + "注意力 X% · 前馈 Y% · 输出
   投影 Z%"小字 + 口径说明（训练 3×、生成无 KV cache、不推时间）；
2. **F2 节点列**：按节点分解表中训练节点的标题行右侧显示
   "内存 · FLOPs"两个数；展开后有一张两列的计算量表（注意力/前馈/
   输出投影，近似项标"近似"）；TextVocabBuild/TextEncode/
   TextSlidingWindow 显示"（计算量未建模）"灰字；
3. **F1 耗时面板**：点 Run 跑一次工作流 → 侧栏出现"节点实测耗时"卡：
   总计 + Top-N 对数条形（300ms 展开动画）+ 缓存命中计数（第二次
   Run 应明显出现）；再次 Run 时间线自动清零重采；
4. **W2 实时指示**：跑一个耗时几秒的训练 → 顶栏徽章右侧出现胶囊：
   当前节点名 + CPU%（颜色分档）+ MEM；跑完自动淡出；
5. **W2 瞬爆案底**（可选，验证金样例）：把语料调大（如几 MB）跑
   vocab build → 执行中系统短暂卡顿后，侧栏"瞬爆案底"红卡出现
   TextVocabBuild 条目（峰值 CPU、持续时长、数据规模快照）；
   `user/comfydl/profiling_watchdog.jsonl` 有对应行，重启服务后仍在；
6. **折叠长卡**（后续新增）：计算量大时"瞬爆案底"（CPU burst log）
   与"按节点分解"（Per-node breakdown）两张卡会拖得很长——**点击卡片
   标题**（右侧有折叠箭头）即可收起/展开，箭头随之旋转；默认展开。
   折叠态存在 `state.collapsed` 中，面板因重估/自动刷新重建 DOM 时保持
   不弹开；点标题里的 Refresh 按钮不会误触发折叠；
7. **双语**：界面切中文 → 上述所有新文案为中文。

## 10. M3 候选

> 2026-10-06：讨论中新增的 **opgraph 算子探针**（等效基础算子计算图，ATen 级）已先期交付，
> 见 `docs/profiling-m3-opgraph.md`——它为下列 F4（每算子算术强度）与 W3（未融合链检测）
> 提供数据底座，且让未来新节点族（如 RL）出生即被覆盖。

## 11. P2 实测回流：run 期 FLOPs 落库与校准（2026-10-07）

> 用户原始诉求："FLOPs 测试应成为 run 过程的附属产物……记录进 user database 供纠正
> estimate 和为用户提供历史记录参考。"实现 = `comfy/profiling/runmeter.py` +
> `app/database/models.py` 两表 + alembic `0007` + `/comfydl/profiling/history`。

**机制**：`execute_async`（execution.py）在每个 prompt 开始时挂载一个
`RunMeter`（单层 TorchDispatchMode，flop 算术复用 torch 的
`flop_registry`——**刻意不嵌套 FlopCounterMode**，双 Python dispatch 层会突破
5% 开销契约）；每次算子派发经 `CurrentNodeContext` contextvar 归属到当前节点；
prompt 结束（finally）经 app 层回调写入两表。

**采样语义（关键）**：每个被拦截算子有 ~20-40us 的固定 Python 派发成本，而
同一节点的每步 FLOPs 恒定——因此采用**采样计量**：每 run 只精确记录前
`ops_budget=500` 个算子，之后 meter 在节点边界被卸载（`wants_node` 门控 +
executor 主循环节点间检查），后续图零开销。run 行标 `sampled=1`。代价：
单个超大训练节点（首测）内部仍付单层转发成本（一次性，~25us/op）；每
class_type 只测首次出现，后续同类节点零开销。

**口径声明**：
- flops = **forward+backward 合计**（backward 的 ATen op 同样流经 dispatcher）；
- fused optimizer 算子（`_foreach_*`）在 flop_registry 无条目 → 记 0
  （**漏计**，optimizer 开销不计入）；
- 复合算子可能双计 → 用 M3 探针双口径交叉验证（formula vs probe vs measured）；
- 节点自建线程内的 torch 调用逃过 meter（与 progress 归属同边界）；
- `unattributed` 桶：contextvar 缺失时的算子（应接近 0，异常时可见）。

**校准**：estimate 报告为有历史实测的 class_type 附加
`measured_flops`（均值）/`measured_samples`/`measured_ratio`
（measured ÷ formula）字段，前端 per-node 区显示"实测 vs 公式"行。

**开关**：`--cdl-profiling-record {off,count,census}` 默认 `count`
（census 档额外保留 per-op 直方图，供融合分析）；`off` 零写入。

**开销实测**（P 盘 CPU，2000 步训练循环）：单层方案 **31.9%→46%（节点门控后，
残留为被测节点的单层转发）**；GPU 上派发成本被 launch 队列掩盖，预期 <5%
（待 F 盘实测确认）。早期嵌套实现实测 119%——已废弃（见 runmeter.py docstring）。

- **W1 静态地雷扫描**：节点成本类标注（python_loop_O(N) 等）+
  数据规模阈值预警（W2 案底做实测校准源）；
- **W3 优化建议库**：逐节点族的具体修复动作（调哪个参数/如何重组），
  建议过引擎回代自检（postmortem T8i 范式）；
- **F3 实测吞吐 + MFU**：F2 静态 FLOPs ÷ F1 实测耗时 → 有效 TFLOPS，
  与 device properties 峰值比出 MFU；
- **F4 roofline 归因**：算术强度 = FLOPs ÷ M1 激活字节，分
  compute-bound / memory-bound 判定；
- **F5 Run 前时间预警**：静态 FLOPs ÷ 历史实测吞吐，报数量级（黄级
  不阻断）。

## 11. 展示字体（交付后润色：Gmarket Sans 子集）

用户反馈面板数字"字体难看"，指定 Gmarket Sans Bold（圆润、字面近正方）。
落地方式与取舍：

- **文件**：`app/profiling_assets/fonts/GmarketSansBoldYPG.ttf`（10 KB，随
  仓库分发，经 `server.py` 已有的静态路由 `/comfydl/profiling/*` 直发；
  `test_profiling.py` 的 T9h 用 TrueType 魔数 + MIME 断言它真被送达——
  路径写错会静默回退系统字体，正是本次投诉的成因，必须由测试兜住）；
- **子集限制（关键）**：该文件只含 **97 个字形**——数字 0-9、大写字母
  A-Z、小写 a-y（**缺 z**）、以及 `! ( ) + , - . / : ~`；**缺** `% · — … ×`
  `_` 与全部 CJK 汉字（仅 26 个韩文音节）。因此：
  - 字体只施加于**这些字符必然出现、且本身是"数值/状态"语义**的位置：
    判定徽章、大数字（判定值、设备预算三格、FLOPs 总量、耗时数值）、
    表格数值列、瞬爆元信息值、面板内按钮；
  - **二次收窄（用户复看后指示）**：首版把**卡片标题**、**按节点分解表的
    节点行**、**顶栏徽章**也换成了展示字体，用户对比后要求这三处恢复系统
    字体——圆体在这些位置读起来"喊叫"而非"账本"，且顶栏徽章紧邻 ComfyUI
    原生按钮，混排更突兀。现这三处仅保留字重（标题 700、节点名 600、
    徽章 600），字体继承系统栈；
  - 细排正文（提示小字、假设键名 `batch_size`、瞬爆元信息含 `%`/`·`）与
    **实时胶囊**保持系统字体——后者另有理由：等宽可避免每秒刷新的数字
    左右抖动，且 `%` 不在子集内；
  - CJK 缺失依**逐字回退**兜底（中文由 Noto Sans SC 渲染），因此中文界面
    下同一行内会混排两套字体，属预期行为（CJK 站点的常规做法）；
- **字重**：文件仅 Bold 一档，`@font-face` 声明 `font-weight: 700`，仅施加
  于本就要求粗体的位置，浏览器不会对 400 请求叠加合成加粗；
- **变量**：`--cdlp-font-display` 声明在 `:root`（面板与顶栏徽章共用）；
- **验证方式**：交付时用 Edge headless + 同源临时探针页截图自证过一次
  （字体有 CORS 约束，`file://` 打开无法加载，必须同源）。**此后取消
  自动视觉验收**——由用户直接看界面判定，探针页与截图环节不再执行，
  避免无谓 token 开销；回归防线只保留 T9h（TrueType 魔数 + font MIME）；
- **升级路径**：若日后要中文/符号也统一为该字体，把**完整版** Gmarket Sans
  TTF 放进工作区即可重新子集化（含 `% · — … × _` + 所需 CJK 子集），
  字号与排版无需改动——只替换该文件与 `@font-face` 的 `src`。
