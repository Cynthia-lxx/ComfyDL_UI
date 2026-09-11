# Reform Step 5 — The `model` Protocol Layer (`MODEL` / `CLIP` / `VAE`)

> **适用范围 / Scope.** 恢复 `MODEL`、`CLIP`、`VAE` 三种数据类型以及配套的加载 / 合并 / 保存节点，
> 共 **22 个节点**，分布在 `model/loaders`(7)、`model/merging`(11)、`model/latent`(2)、
> `model/conditioning`(2) 四个分类。
> 新增 1 个框架侧辅助模块 `comfy/model_protocol.py`（**不改任何既有 `comfy/` 文件**）与 3 个
> `comfy_extras/` 节点文件。
> **前置条件 / Prerequisites.** 脱水阶段（见 `dehydrate_manifest.md`）已移除 `ldm` / `comfy.lora` /
> 各类采样器。本步不恢复它们，而是在它们缺席的前提下把"权重搬运"这条路打通。

---

## 1. 动机 / Motivation

脱水阶段移除了 `ldm`、`comfy.lora` 与采样器，代价是 `MODEL` / `CLIP` / `VAE` 三种类型失去了生产者：
节点库里再没有任何节点能产出它们，用户也就无法在图中表达"加载一个模型 → 调整权重 → 存回去"这类
最基础的模型操作。教学节点（step1–step4）全部只吃 `TENSOR`，因此整个"权重"概念从图里消失了。

本步的目标不是恢复扩散推理，而是恢复**权重作为一等数据**这件事：让 `MODEL` / `CLIP` / `VAE` 重新
可以在节点间流转，让"权重协议层"（加载 / 合并 / 保存）真正可用。这样 ComfyDL 的下一步——可学习参数对象
与优化器——才有落点。

---

## 2. 为什么不能照搬原生实现 / Why the Native Nodes Cannot Be Reused

在"不回捞 `ldm`"的前提下，原生实现是**必然报错**的，而不是"可能报错"。三个爆点（行号已核对）：

| 爆点 | 位置 | 触发节点 |
|---|---|---|
| `model_detection.unet_prefix_from_state_dict` | `comfy/sd.py:2097` | `CheckpointLoaderSimple`、`UNETLoader` |
| `diffusers_convert` / `ldm.models.autoencoder` | `comfy/sd.py:425` | `VAELoader` |
| `comfy.lora` / `lora_convert` | `comfy/sd.py:40-44` | `LoraLoader` |
| `comfy.model_base.SDXL`、`comfy.model_sampling` / `ModelType.EPS` | `comfy_extras/nodes_model_merging.py:179`、`204-216`（原生实现） | `ModelSave`、`CheckpointSave` |

也就是说：**凡是"产生 MODEL/CLIP/VAE"的入口，执行时都会炸**，而且炸的是 `ModuleNotFoundError` ——
对用户来说这是最没有信息量的一种失败。

同时有两条事实是**可以安全依赖**的（同样已核对）：

1. `ModelPatcher.__init__`（`comfy/model_patcher.py:345`，函数体 346–404）只做属性赋值，不 import 任何
   生成模块；`clone`、`add_patches`、`get_key_patches`、`model_state_dict` 也都不引用 `comfy.lora`。
   `comfy.lora` 的全部 11 处调用点（188/194/206/921/1027/1034/1224/1231/1313/1665/1932）都在
   `LowVramPatch` / `patch_weight_to_device` / `patch_model` / `calculate_weight` 这些**应用补丁**的路径上。
2. `comfy/utils.py` 的 `load_torch_file`、`save_torch_file`、`state_dict_prefix_replace` 与
   `folder_paths.py` 的目录解析**完好无损**，是纯 IO，零 `ldm` 依赖。

于是本步的核心策略是一句话：**只走"权重搬运"，彻底避开"补丁应用"路径。**

---

## 3. `comfy/model_protocol.py`（643 行，新增）

框架侧辅助模块，全部函数都有 `What / In / Out` 三段 docstring，`__all__` 显式导出 18 个名字。
它只依赖 `torch`、`comfy.utils`、`comfy.model_patcher`、`comfy.model_management`、`folder_paths`，
**不 import `comfy.sd`，也不 import `comfy.lora`**。

| 名字 | 作用 |
|---|---|
| `StateDictModule(nn.Module)` | 扁平 state_dict 容器。覆写 `state_dict()` / `_save_to_state_dict()` / `load_state_dict()` / `_apply()`，使键集**原样往返**；张量按引用持有，不注册为 `Parameter`（因此 `.to()` 也能正常工作）。另提供 `get_sd()` 别名（对齐原生 `VAE.get_sd` / `CLIP.get_sd` 习惯）与 `key_prefixes` 属性。`state_dict()` 返回浅拷贝——`ModelPatcher.model_state_dict(filter_prefix=...)` 会 `pop` 返回的字典，不能让这些 `pop` 破坏容器自身。 |
| `group_of(key)` | 纯字符串判断 `"model"` / `"vae"` / `"clip"`；未匹配的键归入 `model`，保证不丢权重。 |
| `strip_outer_prefix(sd, mode)` | 剥离最外层容器前缀（`model.` / `state_dict.` / `module.`）。`mode` 为 `"auto"` / `"raw"` / 显式前缀。返回 `(stripped_sd, key_prefixes)`。 |
| `restore_prefixes(sd, key_prefixes)` | 反向操作：保存时把前缀原样加回去，保证"载入 → 保存"键集逐字往返。 |
| `split_checkpoint(sd, prefix_strip)` | 拆成 `(model_sd, clip_sd, vae_sd, key_prefixes)`。 |
| `make_model_patcher(sd, key_prefixes)` | 把 MODEL 组包进 `ModelPatcher`（device 取自 `model_management`，`size=total_bytes(sd)`）。 |
| `make_container(sd, key_prefixes)` | 给 CLIP / VAE 用的轻量包装（纯 `StateDictModule`，不套 `ModelPatcher`）。 |
| `container_state_dict(value)` | 从任意协议层值（`ModelPatcher` / `StateDictModule` / 裸 `dict`）取出 `(state_dict, key_prefixes)`，让合并/保存节点的连线顺序可以任意。 |
| `total_bytes(sd)` | 权重总字节数，等价于 `model_management.module_size`，但不需要先构造模块。 |
| `select_merge_keys(keys, prefix, fallback_all)` | 选出要混合的键。先按字面前缀匹配，失败则按点分段重试；两者都失败时可按 `"unprefixed"` 处理（UNET-only 文件整体就是扩散模型）。返回 `(keys, mode)`，`mode ∈ {all, exact, segment, unprefixed, none}`。 |
| `merge_state_dicts(base, other, blend_fn, …)` | 逐键 `out = w_base * base + w_other * other`，就地写入新字典。返回 `(merged, matched_count, skipped_keys)`。 |
| `prepare_for_save(sd, key_prefixes)` | 回加前缀 + 把张量变连续（safetensors 拒绝非连续视图，而 mmap 加载的权重正是文件视图）。 |
| `merge_group_state_dicts(groups)` | 把 MODEL + CLIP + VAE 三组拼成一个映射，供 `CheckpointSave` 写单文件。 |

### 3.1 为什么容器是"扁平字典"而不是重建嵌套模块

把 `a.b.c` 这样的点号键重建为嵌套 `nn.Module` 需要处理"某个键既是前缀又是叶子"的冲突，并且一旦注册为
`Parameter` 就会把 `requires_grad` 与模块语义一起带进来。扁平字典 + 覆写 `state_dict()` 更简单、更可靠：
`ModelPatcher` 只通过 `self.model.state_dict()` 读权重，兼容性已经核实，没有别的入口。

### 3.2 前缀归一化不是"结构识别"

`strip_outer_prefix` 做的是纯键字符串处理：SD1.5/SDXL 的 checkpoint 把权重套在 `model.`
（`model.diffusion_model.*`）里，这个包装恰好遮住了下游分组要用的 `diffusion_model.` 标记。
自动探测的判据是"绝大多数键共享此前缀"（`AUTO_STRIP_MIN_COVERAGE`）。前缀归一化是
`ModelMergeSimple` 能生效的**硬前提**：它按 `diffusion_model.` 取键，而 `add_patches` 要求键在
`state_dict()` 中逐字存在。

只有真正以该前缀开头的键会被缩短；`model_ema.diffusion_model.*` 这类键保持原样且不进入
`key_prefixes`，因此 `restore_prefixes` 能精确重建原键集。剥离前缀导致键名碰撞时给出警告并说明
"后者胜出"。

---

## 4. 节点清单 / Node Inventory

### 4.1 `model/loaders`（7 个）

| 显示名 | 类名 | 层 | 行为 |
|---|---|---|---|
| Load Checkpoint | `CheckpointLoaderSimple` | L1 | `split_checkpoint` 拆三组：MODEL 套 `ModelPatcher`，CLIP / VAE 套 `make_container` |
| Load Diffusion Model | `UNETLoader` | L1 | 整个文件即 MODEL |
| Load VAE | `VAELoader` | L1 | 整个文件即 VAE |
| Load CLIP | `CLIPLoader` | L1 | 整个文件即 CLIP |
| Load CLIP (Dual) | `DualCLIPLoader` | L1 | 两个文件合并为一个 CLIP 容器 |
| Load LoRA (Model and CLIP) | `LoraLoader` | L2 | 抛可操作的 `RuntimeError` |
| Load LoRA | `LoraLoaderModelOnly` | L2 | 同上 |

7 个 loader 都比原生契约多一个 `prefix_strip` 控件（默认 `"auto"`，可直接运行）。原生的
`CLIPLoader.type` 等"只为结构识别服务"的控件被**刻意省略**——它们在本路线下不会有任何作用，摆着就是
死控件。

某一组没有键时输出的是合法的**空容器**而不是 `None`，因此"只含部分权重"的 checkpoint 不会打断下游
连线。

### 4.2 `model/merging`（11 个）

合并 7 个（全部经 `merge_state_dicts` 做键对齐的张量运算，产出新容器，**不构造补丁**）：

| 显示名 | 类名 | 语义（`w_base`, `w_other`） |
|---|---|---|
| ModelMergeSimple | `ModelMergeSimple` | `(ratio, 1 - ratio)` |
| ModelMergeBlocks | `ModelMergeBlocks` | 按 `input_blocks` / `middle_block` / `output_blocks` 分段给权重，其余键取 `input` |
| ModelMergeAdd | `ModelMergeAdd` | `(1, 1)` |
| ModelMergeSubtract | `ModelMergeSubtract` | `(multiplier, -multiplier)` |
| CLIPMergeSimple | `CLIPMergeSimple` | `(ratio, 1 - ratio)`，跳过 `.position_ids` / `.logit_scale` |
| CLIPMergeAdd | `CLIPMergeAdd` | `(1, 1)` |
| CLIPMergeSubtract | `CLIPMergeSubtract` | `(multiplier, -multiplier)` |

原生 `ModelMergeSimple` 走 `clone()` + `add_patches()`，补丁真正生效发生在
`patch_weight_to_device` → `comfy.lora.calculate_weight`（`model_patcher.py:921`），**该路径必炸**。
本实现把张量运算提前做完，因此结果确定、可测、可保存。语义与原生一致。

保存 4 个（`RETURN_TYPES = ()` + `OUTPUT_NODE = True`）：

| 显示名 | 类名 | 落盘内容 | `filename_prefix` 默认值 |
|---|---|---|---|
| ModelSave | `ModelSave` | MODEL 组 | `comfydl/diffusion_models` |
| VAESave | `VAESave` | VAE 组 | `comfydl/vae` |
| CLIPSave | `CLIPSave` | CLIP 容器（单文件） | `comfydl/clip` |
| Save Checkpoint | `CheckpointSave` | 三组拼回一个 `.safetensors` | `comfydl/checkpoints` |

全部走 `comfy.utils.save_torch_file`（safetensors），绕开 `comfy.sd.save_checkpoint`，因此全程 CPU、
不触发显存分配、也不依赖任何被移除的模块。

### 4.3 `model/latent` / `model/conditioning`（4 个，L2 占位）

| 显示名 | 类名 | 分类 |
|---|---|---|
| VAE Decode | `VAEDecode` | `model/latent` |
| VAE Encode | `VAEEncode` | `model/latent` |
| CLIP Text Encode (Prompt) | `CLIPTextEncode` | `model/conditioning` |
| CLIP Set Last Layer | `CLIPSetLastLayer` | `model/conditioning` |

IO 契约与原生逐字一致（含 `stop_at_clip_layer` 的 −24~−1 范围），`execute` 抛可操作的
`RuntimeError`，docstring 顶部明确标注"协议层占位，本阶段不可执行"。

---

## 5. 两档行为 / Two Tiers

| 档位 | 行为 | 节点 |
|---|---|---|
| **L1 — 真实执行** | 真正读文件 / 拆组 / 合并 / 落盘 | 5 个 loader + 7 个合并 + 4 个保存 = **16** |
| **L2 — 已注册，不可执行** | 节点在库中、可连线、IO 契约与原生一致；执行时抛 `RuntimeError`，写明缺哪个模块、如何回捞（**绝不**是裸 `ModuleNotFoundError`） | 6 个：两个 LoRA、`VAE Decode`、`VAE Encode`、`CLIP Text Encode (Prompt)`、`CLIP Set Last Layer` |

L2 的意义是"类型与 IO 打通、工作流可搭"：用户可以今天就画出完整的文生图流程图并保存，
未来把 `ldm` / 文本编码器回捞回来后，这些节点**原地转正**，无需改工作流。

**为什么 LoRA 只能是 L2：** LoRA 的键形如 `lora_unet_...`，映射到模型键需要知道模型结构（哪一层在哪），
本质上是"结构识别"，与本步所选路线直接冲突。因此它保留契约、明确报错，而不是给一个错误的静默结果。

---

## 6. 健壮性 / Robustness

| 情况 | 行为 |
|---|---|
| 文件中某一组为空 | 输出合法的**空容器**，不是 `None`；下游连线不断 |
| 非张量条目（`epoch`、`global_step`…） | 加载时丢弃并计数（容器不记录也无法记录它们，safetensors 同样不能） |
| 自动前缀探测失灵 | `prefix_strip` 可填显式前缀；填 `"raw"` 关闭；填了不存在的 prefix 时警告并按原样保留 |
| 剥离前缀导致键名碰撞 | 警告并说明"后者胜出" |
| 合并时某一侧 dtype 不同（fp16 vs fp8） | 警告并**保留第一个模型的张量**，不做静默混合 |
| 合并时形状不同 | 同上（保留第一侧） |
| 合并时某键只存在于第二侧 | 计入 `skipped_keys` 并报告（原生行为是丢弃） |
| `ModelMerge*` 找不到 `diffusion_model.` 标记 | 按点分段重试；仍失败则在 UNET-only 场景按"整体就是扩散模型"处理并警告 |
| 保存时张量非连续（mmap 视图） | `prepare_for_save` 自动 `.contiguous()` |
| 保存的目标目录不存在 | 交由 `folder_paths` / `save_torch_file` 的正常路径处理，报错信息保留可读文本 |

---

## 7. 目录与计数 / Files & Counts

| 文件 | 状态 | 规模 |
|---|---|---|
| `comfy/model_protocol.py` | 新增 | 643 行 |
| `comfy_extras/nodes_model_loaders.py` | 新增 | 507 行 |
| `comfy_extras/nodes_model_merging.py` | 新增 | 704 行 |
| `comfy_extras/nodes_model_inference.py` | 新增 | 210 行 |
| `nodes.py` | 追加 3 行 | `extras_files` 第 699–701 行 |
| `cdl_smoke_tests/run_smoke_test.py` | 修改 | 沙箱假 checkpoint + 定点断言 + L2 SKIP |

分类计数：`model/loaders = 7`、`model/merging = 11`、`model/latent = 2`、`model/conditioning = 2`，
合计 **22**。宿主注册表中 `model/latent` 实际有 3 个节点（第三个是原生 `LatentCompositeMasked`，不属本次
改动），说明文件只统计本库提供的 2 个。

四份说明文件同步：`FUNCTIONS.md` / `FUNCTIONS_zh.md` 新增 **§19 ComfyUI / model（22 个节点）** 与
**§20 ComfyUI / 3d（1 个节点）**，附录总数表新增 7 行（6 个新分类 + `3d`），总数口径由 141/26 更新为
**168 节点 / 32 分类**（= 109 ComfyDL + 59 核心）；两份 README 的概述句同步更新。

---

## 8. 验证 / Verification

冒烟器为 L1 节点准备了**沙箱假 checkpoint**：合成一个含 `model.diffusion_model.*`、
`first_stage_model.*`、`cond_stage_model.*` 三组键的最小 `.safetensors`，并把 `folder_paths` 的
`checkpoints` / `vae` / `loras` / `text_encoders` 目录**临时指向沙箱**，让加载节点**真执行**拿到 PASS
而不是 SKIP；用例结束后清空目录并还原 `folder_paths`（沙箱产物随用例清理）。

定点断言（`_OUTPUT_CHECKS`）：

- `CheckpointLoaderSimple`：三组键**不串组**（`diffusion_model.*` 只在 MODEL、
  `first_stage_model.*` 只在 VAE、`cond_stage_model.*` 只在 CLIP），且 `key_prefixes` 覆盖全部键；
- `ModelMergeSimple(ratio=0.5)`：输出的每个共享键满足 `0.5 * a + 0.5 * b`；
- `ModelMergeSubtract(multiplier=1)`：满足 `a - b`；
- `ModelSave` / `CheckpointSave`：**确实落盘**，并且回读后键集与写入时一致（证明前缀回加无损）；
- 往返一致性：假 checkpoint 经 `CheckpointLoaderSimple` → `CheckpointSave` 后键集与原始文件逐字一致。

L2 的 6 个节点加入 `_SKIPPED_NODES`（理由形如 "needs a real LoRA file / text encoder / VAE
implementation"），SKIP 不计入失败。

```powershell
cd P:\Dev\ComfyUI_Refs\ComfyDL_UI
.\penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py --filter Model
```

实测结果：**251 PASS / 23 SKIP / 0 FAIL**（274 个注册节点）。

---

## 9. 回退 / Rollback

1. 从 `nodes.py` 的 `extras_files` 白名单里去掉 `nodes_model_loaders.py`、`nodes_model_merging.py`、
   `nodes_model_inference.py` 三行；
2. 删除这三个文件与 `comfy/model_protocol.py`。

`comfy/` 框架层**零改动**（只新增了一个文件，未修改 `sd.py` / `model_patcher.py` / `utils.py`），
既有 11 个 image 节点与 step1–step4 的 36 个节点都不受影响，`comfydl` 子模块未动。因此回退的风险面
只有"删掉这 22 个节点"。

---

## 10. 执行与实测记录 / Execution Log

| 项目 | 结果 |
|---|---|
| 新增节点 | 22（loaders 7 + merging 11 + latent 2 + conditioning 2） |
| 新增模块 | `comfy/model_protocol.py`（框架侧，纯新增） |
| 实测分类 | `model/loaders = 7`、`model/merging = 11`、`model/latent = 2`、`model/conditioning = 2` |
| 全量冒烟 | **251 PASS / 23 SKIP / 0 FAIL** |
| 文档同步 | `comfydl/_update_readme.py --check` 零差异 |
| 既有改动 | `comfy/` 无修改（仅新增）；`comfydl` 子模块未动 |
