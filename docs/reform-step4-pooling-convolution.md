# Reform Step 4 — `Network & Layers/Pooling` and `Network & Layers/Convolution`

> **适用范围 / Scope.** 在 `Comfy节点 → Network & Layers` 下新增 `Pooling` 与 `Convolution` 两个分类，
> 共 4 个节点：`Pool`、`Adaptive Pool`、`Conv`、`ConvTranspose`。
> 全部落位 `comfy_extras/`，由 `nodes.py` 的 `extras_files` 白名单注册（纯追加）。
> **前置条件 / Prerequisites.** step1（`TENSOR` 类型）、step2（`Network & Layers/Basic`，确立"权重走
> 连线、节点无状态"的约定）。本步不依赖 step3。

---

## 1. 动机 / Motivation

`Basic`（step2）只覆盖了"全连接 + 逐元素"这一类算子：它们把输入当成一个扁平的特征向量，没有任何空间
结构的概念。CNN 之所以是 CNN，靠的恰恰是两个空间算子——**卷积**（在空间上复用同一组权重）与**池化**
（在空间上做降采样/聚合）。缺了它们，"Network & Layers" 这一支只能搭 MLP，搭不出哪怕最小的 LeNet。

本步把这两个算子补齐，并且遵守两条已经定下来的规则：

1. **能用控件表达的语义就不要拆成多个节点。** 秩（1d/2d/3d）与归约方式（max/avg）都是少选项下拉，
   不该变成 12 个节点；`torch.nn.functional` 里同族的 `*_pool1d/2d/3d`、`conv1d/2d/3d` 本来就只差一个
   参数。
2. **可学习参数走 slot，不走控件。** 与 `Linear` 完全一致：`weight` / `bias` 是 `TENSOR` 输入，
   节点内部不初始化任何东西。这样节点保持纯函数语义，同一组权重可以喂给多个节点（权重共享可画出来），
   也才可能与未来的训练相关节点组合。

---

## 2. 节点清单 / Node List

### 2.1 `Network & Layers/Pooling`（2 个）

| 显示名 | 类名 | 覆盖的 torch 层 |
|---|---|---|
| Pool | `PoolingSliding` | `MaxPool1d/2d/3d`、`AvgPool1d/2d/3d` |
| Adaptive Pool | `PoolingAdaptive` | `AdaptiveMaxPool1d/2d/3d`、`AdaptiveAvgPool1d/2d/3d` + 全局池化 |

总共 12 种语义压缩为 2 个节点：`mode`（max/avg）× `dims`（1/2/3）就是分派表。

- `PoolingSliding` 控件：`dims` 2、`mode` `max`、`kernel_size` 2、`stride` 0、`padding` 0、
  `dilation` 1、`ceil_mode` false、`count_include_pad` true。
- `PoolingAdaptive` 控件：`dims` 2、`mode` `avg`、`output_size` 1。

**全局池化不单列节点。** `output_size=1` 就是全局池化，即
`dims=2, output_size=1` ≡ `nn.AdaptiveAvgPool2d(1)` ≡ 教科书里的 Global Average Pooling（`GAP`）；
切一下 `mode` 就是 Global Max Pooling。因此原计划的 `GlobalAvgPool` / `GlobalMaxPool` 两个节点被
吸收，改为在 docstring 与 `search_aliases` 里显式写上 `global average pooling` / `GAP` /
`global max pooling`，保证按名字搜得到。

### 2.2 `Network & Layers/Convolution`（2 个）

| 显示名 | 类名 | 覆盖的 torch 层 |
|---|---|---|
| Conv | `ConvolutionConv` | `Conv1d/2d/3d` |
| ConvTranspose | `ConvolutionConvTranspose` | `ConvTranspose1d/2d/3d` |

- `ConvolutionConv` 控件：`dims` 2、`groups` 1、`stride` 1、`padding` 1、`padding_mode` `zeros`、
  `dilation` 1。
- `ConvolutionConvTranspose` 控件：`dims` 2、`groups` 1、`stride` 2、`padding` 0、`output_padding` 0、
  `dilation` 1 —— **没有 `padding_mode`**。

**没有 `in_channels` / `out_channels` / `bias` 开关。** 通道数可以从 `weight.shape` 直接推出
（`in = weight.shape[1] * groups`），"有没有偏置"用 `bias` 这个 `optional=True` 的 slot 是否连线来表达。
多一个这样的控件就等于多一个可能与权重不一致的谎话。唯一无法从权重推出的量是 `groups`，它有自己的
语义（分组卷积会改变权重形状的解读方式），因此必须是控件。

---

## 3. 设计要点 / Design Notes

### 3.1 `padding_mode` 必须手工实现

`F.conv{1,2,3}d` **没有** `padding_mode` 参数——这个能力只存在于 `nn.Conv*d` 模块上（模块把它实现为
一次 `F.pad`）。所以四种填充模式由节点自己完成：当 `padding_mode != "zeros"` 时，先
`F.pad(x, pad, mode="reflect"|"replicate"|"circular")`，再以 `padding=0` 调用卷积，避免"手工填充 +
卷积自带填充"叠成两倍。

`F.pad` 的 `circular` 有两个硬性前置条件（输入秩 ≥ 3、每个维的填充量小于该维尺寸）。不满足时节点打印
一条可读提示并回退为补零，而不是把 torch 的底层栈直接抛给用户。

`ConvTranspose` 刻意不提供 `padding_mode` 控件：`nn.ConvTranspose*d` 与 `F.conv_transpose*d` 都没有
这个参数，摆一个永远不会生效的控件是错的。

### 3.2 模式无关的控件被忽略时要说出来

torch 的 API 本身是不对称的：`F.avg_pool*d` 没有 `dilation`，`F.max_pool*d` 没有
`count_include_pad`。既然两个模式共用一个节点，就只能把这两个控件都摆在面板上；当它们在当前模式下
没有意义、却被设成了非默认值时，节点打印提示（`[Network & Layers] ... is ignored for ... pooling.`），
而不是静默失效。这与 step2/step3 里 `_warn()` 的惯例一致。

### 3.3 无 batch 维的输入也接受

torch 的池化/卷积同时接受 `(N, C, *spatial)` 与 `(C, *spatial)`。教学场景里手搓一个 `(C, H, W)` 张量
非常常见，所以节点在入口处把张量 `unsqueeze` 到至少 `dims + 1` 维，计算完再 `squeeze` 回去，输出的秩
与输入保持一致。实现是 `_ensure_min_rank` / `_drop_added_rank` 一对 helper。

### 3.4 dtype 提升与形状校验

与 `Linear`（step2）保持一致：当输入与权重同为浮点但类型不同时（例如 fp16 激活 + fp32 权重），用
`torch.promote_types` 取更宽的类型并各自转换，而不是抛 dtype 不匹配错误——教学里混用半精度与单精度
极其常见。权重形状与 `dims` 不符时给出指名道姓的报错（说明期望 `(out, in/groups, k...)` 还是
`(in, out/groups, k...)`）；偏置的元素个数与输出通道数不符时同样报错。

### 3.5 参数共享怎么写进 docstring

`Conv` 的教学重点是"同一个卷积核在所有空间位置复用"：权重数量只与通道数和核尺寸有关，与图像尺寸无关。
这句话写在 docstring 里，并且用"3×3 的核在 32×32 与 1024×1024 的图像上都是每通道 9 个权重"来锚定。
两个节点的权重形状是镜像的（`Conv` 收 `(out, in/groups, k...)`，`ConvTranspose` 收
`(in, out/groups, k...)`），这一点也单独写明——它是最常见的踩坑点。

---

## 4. 目录与计数 / Files & Counts

| 文件 | 规模 | 说明 |
|---|---|---|
| `comfy_extras/nodes_pooling.py` | 423 行 | 2 节点 + `_pooling_schema()` + `POOLING_NODES` + `PoolingExtension` |
| `comfy_extras/nodes_convolution.py` | 583 行 | 2 节点 + `_conv_schema()` + `CONVOLUTION_NODES` + `ConvolutionExtension` |
| `nodes.py` | +2 行 | `extras_files` 白名单追加 `nodes_pooling.py`、`nodes_convolution.py`（第 697、698 行） |

分类树变化：

```
Comfy节点
├── Network & Layers
│   ├── Activation        14   (step1)
│   ├── Basic              8   (step2)
│   ├── Normalization      7   (step3)
│   ├── Regularization     1   (step3)
│   ├── Training           2   (step3)
│   ├── Pooling            2   ← 本步
│   └── Convolution        2   ← 本步
└── model                 22   (step5)
```

`Network & Layers` 分组节点数 32 → **36**；说明文件口径 141/26 → **168/32**（见 §6）。

因为两个模块都落在 `comfy_extras/`，`python_module` 首段是 `comfy_extras`，前端会把它们归入
**Comfy节点**（核心区）而不是"扩展"。前端节点库只在页面加载时取一次 `/object_info`，**后端改完必须刷新
页面**才能看到新分类。

---

## 5. 与 d2l 的关系 / Relation to the `d2l` Nodes

d2l 侧原本有 `d2l/CV Models` 里的整套 `d2l.Conv2d` / `d2l.Pool2d` 教学实现，它们与本次新增节点是
**两种不同粒度**：

- d2l 的卷积/池化节点把"超参 + 权重初始化"打包在一起，是一个"完整的层"；
- 本步的 4 个节点是**算子**：不含初始化、权重必须从外面喂进来，因此可以画出"同一组权重被两处复用"。

两者不冲突，也没有软归档：d2l 的节点仍然以 `d2l.*` 分类出现在"扩展"区，本次新增的 4 个节点在
Comfy Core 区，id 也完全不重叠。

---

## 6. 文档同步 / Documentation Sync

四份说明文件按硬规则同步：

- `comfydl/FUNCTIONS.md` / `FUNCTIONS_zh.md`：§17 标题改为 `Network & Layers (36 nodes)`，新增
  §17.6 Pooling、§17.7 Convolution；附录节点总数表新增 `Network & Layers/Pooling = 2`、
  `Network & Layers/Convolution = 2` 两行。
- `comfydl/README.md` / `README_zh.md`：概述句与"内置节点库"计数更新为 **168 个节点 / 32 个分类**
  = 109 ComfyDL + 59 核心节点，并在类别表后补一句说明该表只列 ComfyDL 提供的 20 个分类。

计数口径（三套并存，均不互相"纠正"）：

| 口径 | 数值 | 出处 |
|---|---|---|
| ComfyDL 自身 | 109 节点 / 20 分类 | `comfydl` banner、宿主注册表中 `comfydl.NODE_CLASS_MAPPINGS` |
| 宿主注册表 | 274 节点 / 44 分类 | `init_extra_nodes()` 的 `NODE_CLASS_MAPPINGS` |
| 说明文件"内置节点库" | 168 节点 / 32 分类 | 四份说明文档，= 109 ComfyDL + 59 已被采纳的核心节点 |

---

## 7. 健壮性 / Robustness

| 情况 | 行为 |
|---|---|
| `dims` 不是 1/2/3（例如工作流被手工改坏） | 警告并回退为 2，不抛异常 |
| `kernel_size` / `output_size` < 1 | 夹到 1 |
| `stride = 0` | 视为"与 `kernel_size` 相同"（torch 的默认语义），而不是报错 |
| 非 max 模式下 `dilation != 1` | 警告"被忽略"，继续执行 |
| max 模式下 `count_include_pad = false` | 同上 |
| `padding_mode != zeros` 但 `circular` 的前置条件不成立 | 打印可读提示，回退为补零 |
| 权重形状与 `dims` 不符 | 报错并写明期望形状（区分 Conv / ConvTranspose） |
| 偏置元素个数 ≠ 输出通道数 | 报错并写明两者 |
| 输入缺 batch 维 | 内部补维，输出还原为输入的秩 |
| 激活与权重 dtype 不同（同为浮点） | 用更宽的类型计算 |

全部失败路径都是"可读信息 + 明确后果"，不出现裸的 torch 栈或静默错误结果。

---

## 8. 验证 / Verification

```powershell
cd P:\Dev\ComfyUI_Refs\ComfyDL_UI
.\penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py --filter Pooling
.\penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py --filter Convolution
```

冒烟器侧新增的定点断言：

- `PoolingSliding`（`dims=2, mode=max, kernel_size=2`）的输出与 `F.max_pool2d(x, 2)` 逐元素相等；
- `PoolingAdaptive`（`output_size=1, mode=avg`）与 `F.adaptive_avg_pool2d(x, 1)` 逐元素相等
  （即全局平均池化）；
- `ConvolutionConv`（`padding_mode=reflect`）与手写的 `F.pad(mode="reflect")` + `F.conv2d(padding=0)`
  组合逐元素相等；
- `ConvolutionConvTranspose` 的 `output_padding` 会真实改变输出尺寸。

全量冒烟结果记录在 §10。

---

## 9. 回退 / Rollback

本步是**纯追加**，回退动作只有两个：

1. 从 `nodes.py` 的 `extras_files` 白名单里去掉 `nodes_pooling.py`、`nodes_convolution.py` 两行；
2. 删除这两个文件。

没有既有节点、既有工作流或 `comfydl` 子模块受到影响；四份说明文件里的对应内容同步删除即可。注意
前端会把已保存工作流里指向被删除节点的槽位标为缺失，因此回退前应先确认没有工作流在用它们。

---

## 10. 执行与实测记录 / Execution Log

| 项目 | 结果 |
|---|---|
| 新增节点 | 4（`PoolingSliding`、`PoolingAdaptive`、`ConvolutionConv`、`ConvolutionConvTranspose`） |
| 实测分类 | `Network & Layers/Pooling = 2`、`Network & Layers/Convolution = 2` |
| 全量冒烟 | **251 PASS / 23 SKIP / 0 FAIL**（274 个注册节点） |
| 文档同步 | `comfydl/_update_readme.py --check` 零差异 |
| 既有改动 | `comfy/`、`nodes.py` 与其他 `comfy_extras/*` 文件均未修改；`comfydl` 子模块未动 |
