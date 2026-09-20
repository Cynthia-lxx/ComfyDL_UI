# Reform Step 9 — Training Loop Completion (Scheduler / Loss / Metrics / Evaluate)

第九步把"训练闭环"补齐：学习率调度、损失、指标、评估从训练节点内部拆出来，成为可连线的图节点；
两个训练节点（`Training Loop` / `Language Model Train`）补上调度器连线、梯度裁剪与早停。新增 4 个核心
节点与 1 个新图数据类型（`SCHEDULER`）；协议层新增 `comfy/training_metrics.py` 纯函数层，只 import
`torch`，符合脱水构建规则。未连线、未启用任何新开关时，两个训练节点的行为与 step 8 逐字相同——
老工作流不受影响。

## 1. 动机 / Motivation

第六步交付的训练闭环里，学习率恒定、损失只有三种、评估只能靠训练器自带的 `loss` 输出，而且没有
"只看不练"的入口：想回答"这个 checkpoint 好不好"就必须再跑一遍训练。本步把这四件事全部拆成普通
节点：调度器与 `OPTIMIZER` 同构（配置走连线、超参走控件，连线天然参与缓存签名），损失 / 指标 /
评估共享一份纯函数数学，训练节点内部则补上裁剪、调度与早停三个标准 trick。

## 2. 节点与类型清单 / Node & Type Inventory

| 分组 | 节点 | 类名 | 说明 |
|------|------|------|------|
| Training（+4） | LR Scheduler | `TrainingLRScheduler` | 把 StepLR / CosineAnnealingLR / ExponentialLR / OneCycleLR / ReduceLROnPlateau 的设定发布为 `SCHEDULER` |
| | Loss | `TrainingLoss` | mse / l1 / smooth_l1 / cross_entropy / bce_with_logits / kl_div，输出标量 |
| | Metrics | `TrainingMetrics` | mae / rmse / accuracy / top_3 / top_5 / perplexity，输出标量 |
| | Evaluate | `TrainingEvaluate` | 只评估不训练：`NNMODEL`（优先）或 `PARAMS`（重建 MLP）+ x/y，输出 loss / metric / prediction |
| Training（改） | Optimizer | `TrainingOptimizer` | 新增 `grad_clip_norm` / `grad_clip_value` 控件（默认 0 = 关闭），随 `OPTIMIZER` 传递 |
| | Training Loop | `TrainingLoop` | 新增可选 `scheduler` 输入 + `early_stop_patience` / `early_stop_min_delta` 控件；loss 下拉扩到 6 种 |
| | Language Model Train | `LanguageModelTrain` | 同上（调度器 + 早停 + 按配置裁剪） |

新图数据类型 `SCHEDULER`（`comfy_api/latest/_io.py`，`Type = SchedulerConfig`），照 `PARAMS` /
`OPTIMIZER` 的 `@comfytype` 先例声明，无需前端注册。

数据流：`Optimizer(+clip) + LR Scheduler + (x, y) → Training Loop → params / loss_history`；
`params 或 model + (x, y) → Evaluate → loss / metric / prediction`；`Loss` / `Metrics` 节点接
任意 prediction / target 对，作为独立标量源。

## 3. 关键设计决策 / Design Decisions

- **协议先行、纯函数承载数学**：`SchedulerConfig`（frozen dataclass）、`scheduler_config()`、
  `build_scheduler()`、`clip_gradients()`、`snapshot_state()` / `restore_state()`、`EarlyStopTracker`
  落在 `comfy/training_protocol.py`；损失与指标数学落在新的 `comfy/training_metrics.py`
  （`LOSS_OPTIONS` / `METRIC_OPTIONS` / `compute_loss()` / `compute_metric()`，只依赖 torch）。
  训练节点、Loss / Metrics / Evaluate 四个新节点共用同一份实现，同一公式不出现第二份代码。
- **`SCHEDULER` 是配置不是活对象**：连线携带 frozen `SchedulerConfig`，训练节点用**自己的**
  optimizer 与步数构建真正的 `torch.optim.lr_scheduler`。连线上没有设备状态，可被 ComfyUI 安全缓存；
  `build_scheduler()` 返回 `(scheduler, needs_metric)`，`ReduceLROnPlateau` 走
  `scheduler.step(平滑 loss)`，其余按迭代步进。
- **总步数自动解析**：`t_max=0` 表示跟随训练节点的步数，`OneCycleLR` 的 `total_steps` 取训练节点
  的 `iterations`、`max_lr` 取 `OPTIMIZER` 的 `lr`——用户不用手填任何"总步数"。
- **梯度裁剪进 `OPTIMIZER` 载荷**（q-0 决策）：`grad_clip_norm` / `grad_clip_value` 默认 0 = 关闭，
  均为 0 时循环体零新增操作；启用后每次 `optimizer.step()` 前先全局 L2 范数、再逐元素值。
- **早停回滚到最佳点**（q-3 决策）：监控量是步 loss 的滑动平均（窗口 8），`patience` 步内改善不足
  `min_delta` 即停止，参数回滚到监控最佳点、`loss_history` 截断到该点并打印停止步与最佳 loss。
  判定只依赖 loss 序列，同 seed 同配置完全可复现。
- **训练节点不出指标输出**（q-1 决策）：指标走独立 `Metrics` / `Evaluate` 节点，训练节点的输出
  槽保持不变（老工作流的连线不破坏）。
- **Evaluate 双入口**（q-2 决策）：`NNMODEL` 优先直接前向（语言模型走逐位置交叉熵）；只给
  `PARAMS` 时按 `layer{i}.weight` 形状重建 MLP（宽度读自 weight 的 in/out，激活由控件给出），
  与 `Training Loop` 的命名约定严格一致。`loss=auto` / `metric=auto` 按目标形态（类别索引 vs
  数值）自动选择，`metric=none` 跳过指标。
- **不污染输入、不泄漏图**：裁剪、调度、快照与回滚全部作用在训练节点的副本上；任何输出保持
  `detach()` 收尾。早停提前退出后不再推进进度条（`ProgressBar.update` 同时是中断检查点）。

## 4. 默认值表 / Widget Defaults

| 控件 | 节点 | 默认 | 语义 |
|------|------|------|------|
| `grad_clip_norm` / `grad_clip_value` | Optimizer | 0.0 / 0.0 | 0 = 关闭；范围 0~1000 |
| `early_stop_patience` | 两个训练节点 | 0 | 0 = 关闭 |
| `early_stop_min_delta` | 两个训练节点 | 1e-4 | 重置耐心计数所需的最小改善 |
| `scheduler` | LR Scheduler | `CosineAnnealingLR` | 最常见、只需 T_max 且 0 = 跟随训练器 |
| `loss` | Loss / Training Loop | `mse` | 对任意浮点 `(N, C)` 形状直接可跑 |
| `metric` | Metrics | `mae` | 同上 |
| `loss` / `metric` | Evaluate | `auto` / `auto` | 按目标形态自动选择 |

所有新控件开箱可用；不接调度器、不开裁剪与早停时输出与 step 8 完全一致。

## 5. 验证结论 / Verification

- `cdl_smoke_tests/run_smoke_test.py`：**283 PASS / 23 SKIP / 0 FAIL**（306 个注册节点）。
  新增断言覆盖：
  - 调度器：五种调度的 lr 序列逐 step 对齐同名 `torch.optim.lr_scheduler` 实现；
    `t_max=0` 跟随训练器；非法调度器名回退 `CosineAnnealingLR` 并告警。
  - 损失 / 指标：数值对齐 `torch.nn.functional` 与手算值；形状错误抛可读错误。
  - 训练回归：训练输入不被污染、同 seed 逐位复现、裁剪确实改变参数更新、早停触发后
    history 截断且返回最佳参数、plateau 调度器用平滑 loss 步进。
  - Evaluate：NNMODEL 路径、PARAMS 重建路径、`auto` 损失 / 指标、`metric=none`、
    双入口优先级、两者皆空报错。
- `gen_locales.py --check` 无差异（本批不新增语言条目）。
- 注册表实测：`Network & Layers/Training` 23 个，Network & Layers 合计 68，宿主核心合计 91，
  全库 200 = 109 ComfyDL + 91 core。四份说明文件计数已按实测改写。

## 6. 已知边界 / Known Boundaries

- `SCHEDULER` 只被两个训练节点消费；`Evaluate` 不受调度影响（它不训练）。
- `ReduceLROnPlateau` 的步进依赖训练器的平滑 loss，因此单独把它连给"会提前早停"的训练器时，
  早停生效后调度器自然不再步进（与预期一致）。
- `Evaluate` 的 `PARAMS` 路径只能重建训练器命名的 MLP（`layer*.*`）；语言模型参数集请直接走
  `NNMODEL` 路径。
- 早停回滚是**参数级**回滚：`snapshot_state()` 只逐个 clone `nn.Parameter`，不覆盖 BN 运行
  统计量等缓冲区（当前两个训练器也不产生这类缓冲区）。
