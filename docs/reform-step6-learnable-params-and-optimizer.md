# Reform Step 6 — Learnable Parameters, Optimizers and a Training Loop

> **适用范围 / Scope.** 把「可学习参数」与「优化器」做成可在连线上流转的数据类型，并提供一个在
> **节点内部**完成前向 / 反向 / 参数更新的训练节点，共 **9 个节点**，全部落在
> `Network & Layers/Training` 分类。
> 新增 1 个框架侧辅助模块 `comfy/training_protocol.py`（**不改任何既有 `comfy/*.py`**）与 1 个
> `comfy_extras/` 节点文件，另在 `comfy_api/latest/_io.py` 追加 2 个类型声明（纯追加）。
> **前置条件 / Prerequisites.** 第四步（`reform-step4-pooling-convolution.md`）已把 `Basic` /
> `Conv` 层节点的「权重走插槽」约定固定下来——本步正是让这些插槽有了真正的训练产物可接。

---

## 1. 动机 / Motivation

前五步把节点库补成了「推理图」：`TENSOR` 可以流动，层节点可以计算，权重可以加载 / 合并 / 落盘。
但整个图里没有任何节点能**产生**权重——所有 `weight` / `bias` 插槽都要求用户先有现成张量。
教学场景里最核心的一件事因此缺席：**从一个随机初始化的模型出发，用数据把它训出来**。

本步补上这条闭环，使「打开节点、不改任何参数即可训出结果」成立：

```
x, y ─┐
      ├─► Training Loop ─► params ─► Parameters to Tensor ─► Linear.weight
Optimizer ─┘                 │                                  （推理图继续用）
                             ├─► loss / loss_history ─► 可视化节点
                             └─► Save / Load、Parameters to Text / Text to Parameters
```

与第一至五步一样，**不新增任何 pip 依赖**：数值只用 `torch` + 标准库，落盘复用
`comfy.utils.save_torch_file` / `load_torch_file`（safetensors，`comfy/sd.py` 已在用）。

---

## 2. 为什么训练必须发生在一个节点内部 / Why the Loop Cannot Span Nodes

这是本步全部设计的根因，也是**已核对**的事实：

| 事实 | 位置 | 后果 |
|---|---|---|
| 整轮 prompt 的执行被 `with torch.inference_mode():` 包住 | `execution.py:751` | 节点产出的张量是 inference tensor：既不能携带 autograd 图跨节点，也**不能保存用于反向**（"Inference tensors cannot be saved for backward"） |
| 节点缓存键只由**已连线输入 + 生产者控件值**决定 | `CacheKeySetInputSignature` | 任何隐藏在节点内部的 Python 状态都不进签名，跨 prompt 复用优化器状态会静默复用过期输出 |
| 上游 `TrainLoraNode` 的做法 | `ComfyUI-original/comfy_extras/nodes_train.py` | 在节点体内 `with torch.inference_mode(False):`，自己建前向、自己反向、自己 `optimizer.step()`，输入先 `.detach()` |

结论：**前向 + 反向 + `optimizer.step()` 只能同处一个节点**，循环也只能在节点内（由 `steps`
控件控制）。于是：

- 优化器不能是「活的 `torch.optim.Optimizer`」——Adam 的动量无法跨 prompt 保存（节点无处存放
  Python 状态、也无处表达），因此 `OPTIMIZER` 只承载**超参配置**；
- 可学习参数必须是**纯数据**（`nn.Parameter` 字典），才能既是训练产物、又是落盘内容、又能回流
  到既有推理节点；
- 超参（`lr`、`steps`、`hidden`…）全部留在控件上，设定与参数走连线——既符合「简单数据走
  widget、抽象数据走 slot」的项目铁律，也让缓存正确性白拿（翻转控件 = 缓存键变化）。

> 上游 `nodes_train.py` 在本脱水构建中**并不存在**（`comfy_extras/` 无该文件），它只作为**语义
> 参照**：其 LoRA / 扩散模型路线依赖 `ldm` / `weight_adapter`，本步不照搬。

---

## 3. 两个新数据类型 / The Two Graph Value Types

在 `comfy_api/latest/_io.py` 中追加，写法与既有的 `@comfytype(io_type="TENSOR")`（`:449`）和
`@comfytype(io_type="LORA_MODEL")`（`:681`）完全同构：

```python
@comfytype(io_type="PARAMS")
class Params(ComfyTypeIO):
    """A trainable parameter set: an ordered ``{name: nn.Parameter}`` mapping."""
    Type = dict[str, torch.nn.Parameter]

@comfytype(io_type="OPTIMIZER")
class Optimizer(ComfyTypeIO):
    """The hyper-parameter set of a ``torch.optim`` optimizer."""
    if TYPE_CHECKING:
        Type = OptimizerConfig
```

| 类型 | 载荷 | 为什么是这个形状 |
|---|---|---|
| `PARAMS` | 有序 `dict[str, nn.Parameter]` | 名字与 `nn.Module.state_dict()` / safetensors 词汇**逐字对齐**（`layer0.weight`、`layer0.bias`…），因此同一份载荷可以：落盘、编进控件文本、把单个条目取出来接 `Linear.weight`。相比「裸张量列表」它保留了名字；相比「整模型对象」它可以退化到单个 `weight`，也能覆盖整网络 |
| `OPTIMIZER` | `OptimizerConfig`（frozen dataclass） | 只承载超参，不绑定 device、不带优化器状态。真正的 `torch.optim.*` 在训练节点体内按配置临时构造，用完即弃，缓存永远安全 |

**为什么不去改 site-packages 里的前端配色表**：未登记的自定义类型在前端会走默认配色，可以正常
连线；写入前端包会在下一次升级时丢失（踩坑 7），收益不抵风险。

### 3.1 参数命名约定 / Naming Convention

```
layer0.weight   layer0.bias      # 第一个 nn.Linear
layer1.weight   layer1.bias      # 第二个
...
```

这条约定同时写在 `training_protocol.MLP`、节点 docstring 与本文档中。它的作用是让
`TrainingLoop(hidden=..., params=<PARAMS>)` 的热启动能按 `load_state_dict(strict=False)` 语义
对齐，并且让 `Parameters to Tensor(name="layer0.weight")` 的结果**逐字**对应
`BasicLinear` 的 `weight` 插槽语义。

---

## 4. `comfy/training_protocol.py`（670 行，新增）

框架侧辅助模块，全部公开函数都有 `What / In / Out` 三段 docstring，`__all__` 显式导出 23 个名字。
它只依赖 `torch` / `safetensors`，**不 import `comfy.sd`、不 import `comfy.lora`**（脱水构建里它们
已经残缺）；文件系统相关的路径解析刻意留在节点里，与 `nodes_model_merging` 保留 `_save_bucket`
的做法一致。

| 名字 | 作用 |
|---|---|
| `OptimizerConfig` | frozen dataclass：`name` / `lr` / `momentum` / `beta1` / `beta2` / `eps` / `weight_decay` / `amsgrad`；`describe()` 给出单行日志。`SGD` 读 `momentum`，`Adam`/`AdamW` 读两个 beta，`RMSprop` 把 `beta2` 当 `alpha` |
| `optimizer_config(...)` | 把控件值归一化成合法 `OptimizerConfig`：NaN / 非有限值 / 越界值回落到默认并告警（`lr=-1` → `DEFAULT_LR`，`beta1=5` → `< 1`），因此一个手抖的控件值不会让 `torch.optim` 在训练中途炸掉 |
| `build_optimizer(cfg, parameters)` | 按 `cfg.name` 构造真实 `torch.optim.AdamW/Adam/SGD/RMSprop`，只传该优化器认识的参数 |
| `seeded_rng(seed)` | 上下文管理器：给进程 RNG 播种（CPU + CUDA），退出时还原之前的状态。训练节点用它保证「同 `seed` 同结果」，同时不偷走图里其他节点的随机流 |
| `MLP(nn.Module)` | 全连接网络，层命名为 `layer0`、`layer1`…；`hidden` 决定隐层数量与宽度，激活只加在隐层之间，输出层恒为线性 |
| `apply_activation(name, tensor)` | `relu` / `gelu` / `tanh` / `sigmoid` / `none`，未知名称回落到 `relu` 并告警 |
| `build_mlp(sizes, activation, device, seed)` | 在 `seeded_rng` 内构造 `MLP`，给定 `seed` 时权重可复现 |
| `parameter_names(payload, limit)` | 渲染键名列表用于报错 / 告警，超长时截断 |
| `as_parameter_dict(payload)` | 把任意 `PARAMS` 载荷强制规整成 `{name: nn.Parameter}`：浮点张量包成 `nn.Parameter`；整数张量或非张量条目**丢弃并告警**（静默保留会让 `requires_grad` 在很远的地方才报错） |
| `module_parameters(module)` | 逐参数导出，键为 `layer{i}.weight` / `layer{i}.bias` |
| `load_into_module(module, params)` | 热启动：只复制**名字与形状都匹配**的条目，返回 `(missing, skipped)`，由调用方打印，绝不静默丢权重 |
| `parameters_to_tensors(payload)` | 扁平化成可保存的 CPU 张量（`.detach().contiguous()`），供 `save_torch_file` 使用 |
| `parameter_count(payload)` | 标量总数，用于日志 |
| `encode_parameters(payload)` | 先写进内存 safetensors，再 base64，前面加 `CDLPARAMS1:` 头（带版本号，将来换格式可识别而非误解析） |
| `decode_parameters(text)` | 反向解码；忽略首尾空白 / 换行，允许省略头部只粘贴 base64 正文 |

### 4.1 为什么文本通道要带版本头

`CDLPARAMS1:` 里的 `1` 是格式版本。控件文本是用户手动复制粘贴的载体，一旦未来格式变化，没有版本头
就只能在解码失败时给一个「base64 不合法」的错误；带版本头则可以直接说明「这是旧版本编码」。这与
step3 的 `running_mean` 控件文本、step5 的 `prefix_strip` 是同一种「让用户输入永远可解释」的思路。

---

## 5. 节点清单 / Node Inventory

全部 9 个节点的 `CATEGORY = "Network & Layers/Training"`（该分类常量在
`nodes_normalization.py:62` 已存在，但新文件**独立定义**，不去改既有文件，避免波及 step3 的节点）。

### 5.1 参数创建 / 合并 / 取出（3 个）

| 显示名 | 类名 | 输入 | 控件 | 行为 |
|---|---|---|---|---|
| Learnable Parameters | `TrainingParameters` | `tensor`（可选 `TENSOR`） | `name` STRING `"weight"`、`shape` STRING `"2,3"`、`init` COMBO 5 项（默认 `normal`）、`seed` INT 0 | 连入张量 → 包装成同名参数（**连线优先于 `shape` / `init`**，非浮点会升位到 float32 并告警）；未连线 → 按 `shape` + `init` 新建。`nn.Parameter` 一律在 `torch.inference_mode(False)` 内构造（inference tensor 不能作为参数） |
| Merge Parameters | `TrainingParametersMerge` | `params a`、`params b` | — | 顺序拼接两份集合；同名键**改名**（`weight` → `weight_2`）并告警，而不是覆盖 |
| Parameters to Tensor | `TrainingParametersExtract` | `params` | `name` STRING `"weight"` | 取一个条目以 `TENSOR` 输出（`.detach()`，不带 autograd）。名字不存在时抛可操作 `ValueError` 并列出全部可用名字 |

### 5.2 优化器设定（1 个）

| 显示名 | 类名 | 控件 | 行为 |
|---|---|---|---|
| Optimizer | `TrainingOptimizer` | `optimizer` COMBO AdamW/Adam/SGD/RMSprop（默认 `AdamW`）、`lr` FLOAT 0.01 (0~1)、`momentum` 0.9 (0~0.999)、`beta1` 0.9 (0~0.999)、`beta2` 0.999 (0~0.9999)、`eps` 1e-8 (0~1e-3)、`weight_decay` 0.01 (0~1)、`amsgrad` false | 只承载配置，输出 `OPTIMIZER` 走连线。所有默认值都在区间内且开箱可用；与优化器无关的控件被忽略（例如 `AdamW` 不看 `momentum`） |

### 5.3 训练器（1 个）

| 显示名 | 类名 | 输入 | 控件 | 输出 |
|---|---|---|---|---|
| Training Loop | `TrainingLoop` | `x`、`y`（`TENSOR`）、`optimizer`（`OPTIMIZER`）、`params`（可选 `PARAMS`，热启动） | `hidden` STRING `"8"`、`activation` COMBO relu/gelu/tanh/sigmoid/none（默认 `relu`）、`loss` COMBO mse/l1/cross_entropy（默认 `mse`）、`steps` INT 200 (1~100000)、`batch_size` INT 0 (0~65536；0 = 每步全批)、`seed` INT 0 | `params`（`PARAMS`）、`loss`（标量 `TENSOR`）、`loss_history`（1 维 `TENSOR`，每步一项）、`prediction`（`TENSOR`，已 detach） |

执行细节（必须照此实现，已在代码中落实）：

1. 全身包在 `with torch.inference_mode(False):` 内（对齐上游）；输入一律
   `x.detach().clone().to(device, torch.float32)`、`y` 同理，彻底脱离 inference tensor。
2. device 取 `comfy.model_management.get_torch_device()`（本机无 GPU 时自动 CPU）。
3. 结构由 `hidden` 声明：`""` → 单层线性回归；`"8,16"` → 两层隐层；
   `in_features = x.shape[-1]`、`out_features = y.shape[-1]` 自动推断；激活只加在隐层之间，
   输出层恒为线性。不可解析的 `hidden` 文本回落到 `"8"` 并告警。
4. `loss=cross_entropy` 时校验 `y` 形态（类别索引或 one-hot / 概率），不合法时给出写明「接受
   哪些形态」的 `RuntimeError`，绝不静默降级；`mse` / `l1` 下 `x`/`y` 的样本数必须一致，否则
   同样报可操作错误。
5. `seed` 同时决定初始化与批采样（`torch.manual_seed` + 局部 `torch.Generator`），保证同输入同结果。
6. 收尾：`optimizer.zero_grad(set_to_none=True)`、逐参数 `param.grad = None`，输出张量全部
   `detach()` → 不把 autograd 图带进 ComfyUI 的输出缓存。
7. `steps` 超过 `STEPS_WARN_THRESHOLD`（20000）时告警（循环是 Python 的，多打一个 0 很容易）。

### 5.4 持久化（4 个，两条通道）

| 显示名 | 类名 | 通道 | 行为 |
|---|---|---|---|
| Save Parameters | `TrainingSaveParameters` | 落盘 | `.safetensors` 写入输出目录（`filename_prefix` 默认 `comfydl/parameters`，自动加计数后缀），并把参数集**原样透传**，因此保存不中断图；另输出绝对 `path`。`is_output_node=True` |
| Load Parameters | `TrainingLoadParameters` | 落盘 | 读回文件（相对路径以输出目录为基准，也支持绝对路径）；非浮点条目丢弃、非 float32 升位，两者都报告；文件缺失 / 无可用张量时抛可操作 `ValueError`（带解析后的绝对路径） |
| Parameters to Text | `TrainingParametersToText` | 控件文本 | 编码成 `CDLPARAMS1:<base64>` 并推进节点自身的 UI 文本框，可直接复制；`is_output_node=True` |
| Text to Parameters | `TrainingTextToParameters` | 控件文本 | 把文本解码回 `PARAMS`；默认值是一段**可解码的极小常量 blob**（一个 2×3 的 `weight`），满足「不改参数即可用」 |

> 两条通道各有适用面：落盘适合成千上万参数的正式 checkpoint；控件文本适合几个参数的教学演示，
> 而且**随 `.json` 工作流一起保存**，不需要额外文件。`Parameters to Text` 在文本超过 4000 字符时
> 只打印摘要并提示改走落盘通道。

---

## 6. 健壮性 / Robustness

| 情况 | 行为 |
|---|---|
| `shape` 文本不可解析（`"abc"`、负数、全角逗号以外的杂字符） | 回落到 `"2,3"` 并告警；全角逗号 / 分号被规范化后仍可解析 |
| `init` 给了未登记的名字 | 回落到 `normal` 并告警 |
| `xavier_uniform` / `kaiming_uniform` 用于 1 维形状 | 回落到 `normal` 并告警（这两个初始化器需要 fan-in / fan-out） |
| 连入 `tensor` 是整数张量 | 升位到 float32 并告警（只有浮点张量可训练） |
| `TrainingParametersMerge` 键名冲突 | 改名（`xxx_2`）并告警，两份权重都不丢 |
| `TrainingParametersExtract` 名字不存在 | 抛 `ValueError`，消息里列出全部可用名字 |
| 控制器值越界 / NaN（`lr=-1`、`beta1=5`） | `optimizer_config` 回落到默认并告警；保证交给 `torch.optim` 的参数永远合法 |
| `hidden` 文本不可解析 | 回落到 `"8"` 并告警 |
| `hidden=""` | 合法：纯线性回归（无隐层） |
| `cross_entropy` 的 `y` 形态不合法 | 抛写明「接受哪些形态」的 `RuntimeError`，不静默降级 |
| `x` / `y` 样本数不一致 | 抛可操作 `ValueError` |
| 热启动 `params` 缺键 / 多键 / 形状不符 | 逐条报告 `missing` / `skipped`，只复制匹配项，绝不静默丢 |
| `Load Parameters` 的文件不存在 / 无浮点张量 | 抛 `ValueError`，消息带解析后的绝对路径并提示先运行 `Save Parameters` |
| `Text to Parameters` 文本不可解码 | 抛 `ValueError`，消息说明「该粘贴什么」 |
| `steps` 极大 | 超过 20000 时告警；上限 100000 由控件区间兜底 |

---

## 7. 目录与计数 / Files & Counts

| 文件 | 状态 | 规模 |
|---|---|---|
| `comfy/training_protocol.py` | 新增 | 670 行 |
| `comfy_extras/nodes_training.py` | 新增 | 1254 行 |
| `comfy_api/latest/_io.py` | 追加 2 个 `@comfytype`（+ import / `__all__`） | 纯追加 |
| `nodes.py` | 追加 1 行 | `extras_files` 白名单加入 `nodes_training.py` |
| `cdl_smoke_tests/run_smoke_test.py` | 修改 | 训练 fixture 工厂 + `PARAMS`/`OPTIMIZER` 分派 + 9 个定点断言 |

计数口径：

| 口径 | 变化 |
|---|---|
| 宿主注册表（`nodes.NODE_CLASS_MAPPINGS`） | 274 → **283**（分类 44 不变，`Network & Layers/Training` 已有） |
| 说明文件「内置节点库」 | 168 → **177** 个节点 / 32 个分类 |
| 其中核心节点（本库口径） | 59 → **68**（`Network & Layers/*` 由 36 → 45） |
| ComfyDL banner（`comfydl` 子模块） | **109 / 20 不变**（新节点属核心，不属 submodule） |
| 分类内计数 | `Network & Layers/Training` 2 → **11** |

四份说明文件同步：`FUNCTIONS.md` / `FUNCTIONS_zh.md` 更新 **§17 ComfyUI / Network & Layers（45 个
节点）** 与 **§17.4 Training（11 个节点）** 并新增 9 个节点的说明表，附录 `Network & Layers/Training`
行与总句同步；两份 README 的概述句同步。`comfydl/_update_readme.py --check` 全绿（该脚本只覆盖
ComfyDL 自身的 109 / 20 口径与类别表格，`177 / 32` 与核心节点拆解是**手改项**）。

---

## 8. 验证 / Verification

### 8.1 冒烟器扩展

`cdl_smoke_tests/run_smoke_test.py` 从宿主 `NODE_CLASS_MAPPINGS` 自动发现节点（`_load_registry:138`），
但按**输入类型**挑 fixture，因此新增：

- `_f_params` / `_f_params_alt` / `_f_optimizer` 三个工厂，注册进 `_VALUE_FACTORIES` 的
  `"PARAMS"` / `"OPTIMIZER"`；`_training_pair()` 产出成对的 `x` / `y`；
- `_INPUT_OVERRIDES` 为 `TrainingLoop` 指定 `x` / `y` / `optimizer` 配对，为
  `TrainingParametersMerge` 指定差异化的 `params_b`（触发改名分支）；
- `_seed_fixtures` 预写一个 `output/comfydl/parameters_00001_.safetensors`，让
  `Load Parameters` 的默认 `path` 真的能读到东西（真执行而非 SKIP）。

### 8.2 定点断言（`_OUTPUT_CHECKS`，实质性而非「不抛异常」）

- `TrainingParameters`：默认键为 `weight`、类型是 `requires_grad` 的 `nn.Parameter`、形状 `(2,3)`、
  同 `seed` 两次运行逐元素相等；连入张量时**连线胜过 `shape="9,9"` / `init="zeros"`**；
- `TrainingParametersMerge`：并集键为 `{weight, bias, weight_2}`，第一份集合逐键未被覆盖，
  冲突键以改名形式保留（证明没丢权重）；
- `TrainingParametersExtract`：返回的正是所点名的条目、`requires_grad=False`；名字不存在时**抛错**
  且错误消息里含可用名字；
- `TrainingOptimizer`：`OptimizerConfig` 能构造出 `torch.optim.AdamW` 等四个真实优化器，
  `lr` 与控制值一致；`lr=-1` / `beta1=5` 被夹回合法值（`< 1.0`）；
- `TrainingLoop`：参数名恰为 `layer0.weight / layer0.bias / layer1.weight / layer1.bias`；
  返回的参数 `grad is None`；`loss_history` 形状 `(200,)`、`prediction` 形状 `(48, 1)`；
  **三者的 `requires_grad` 全为 `False`**（证明 autograd 图没有泄漏进缓存）；
  `loss == loss_history[-1]` 且 **末值 < 首值的一半、且 < 0.5**（真的收敛了）；
  热启动 1 步的末值 < 首值 / 5（证明复用了连入参数而非重新初始化）；
  `x`/`y` 样本数不一致时**拒绝执行**；
- `TrainingSaveParameters` / `TrainingLoadParameters`：文件确实落在沙箱输出目录，键值与写入前
  逐字一致，且能被 `Load Parameters` 读回；路径不存在时抛错；
- `TrainingParametersToText` / `TrainingTextToParameters`：编码 → 解码后键值与形状无损往返；
  默认控件值本身可解码。

### 8.3 实测结果

```powershell
cd P:\Dev\ComfyUI_Refs\ComfyDL_UI
.\penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py
```

实测：**260 PASS / 23 SKIP / 0 FAIL**（283 个注册节点）。训练日志示例：

```
[Network & Layers] Training Loop: 2 dense layer(s) [2, 8, 1], 48 sample(s) x 2 feature(s)
-> 1 output(s), loss mse, 200 step(s), 48 sample(s) per step,
AdamW(lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False),
loss 11.2753 -> 0.0451142
```

---

## 9. 回退 / Rollback

1. 从 `nodes.py` 的 `extras_files` 白名单里去掉 `nodes_training.py` 一行；
2. 删除 `comfy_extras/nodes_training.py` 与 `comfy/training_protocol.py`；
3. 可选：移除 `comfy_api/latest/_io.py` 里的 `Params` / `Optimizer` 两个 `@comfytype` 与其
   `__all__` 条目（保留也无害，只是多了两个无人使用的类型）。

`comfy/` 框架层**零改动**（只新增一个文件，未修改 `execution.py` / `sd.py` / `utils.py`），
step1–step5 的既有 274 个节点与 `comfydl` 子模块均不受影响。回退的风险面只有「删掉这 9 个节点」。

---

## 10. 执行与实测记录 / Execution Log

| 项目 | 结果 |
|---|---|
| 新增节点 | 9（参数 3 + 优化器 1 + 训练器 1 + 持久化 4） |
| 新增模块 | `comfy/training_protocol.py`（框架侧，纯新增，670 行） |
| 新增节点文件 | `comfy_extras/nodes_training.py`（1254 行） |
| 新增类型 | `PARAMS` / `OPTIMIZER`（`comfy_api/latest/_io.py`，纯追加） |
| 实测分类 | `Network & Layers/Training = 11`（含 step3 的 2 个状态节点） |
| 全量冒烟 | **260 PASS / 23 SKIP / 0 FAIL**（283 个注册节点） |
| 文档同步 | `comfydl/_update_readme.py --check` 零差异；四份说明文件计数更新为 177 / 32 |
| 既有改动 | `comfy/` 无修改（仅新增）；`comfydl` 子模块未动 |
