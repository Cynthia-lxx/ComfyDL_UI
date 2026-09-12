# ComfyDL as a Built-in Module (ComfyDL 内置化)

> Migration history: developed on the experimental branch `experiment/embed-comfydl`
> (pushed to the remote as a backup), then merged into `master`.
>
> 迁移过程：先在实验分支 `experiment/embed-comfydl` 上完成（并推送远端备份），随后整体合入 `master`。

## Goal (目标)

Make the ComfyDL teaching nodes part of the ComfyUI core node registry instead of a
third-party `custom_nodes/` plugin, while keeping ComfyDL as an independent git
project (submodule) that can still be developed and committed on its own.

将 ComfyDL 教学节点从 `custom_nodes/` 第三方插件提升为 comfy core 注册的一部分，同时
保留 ComfyDL 作为独立 git 项目（submodule），仍可独立开发与提交。

## Structure Change (结构变更)

| Before (迁移前) | After (迁移后) |
| --- | --- |
| `custom_nodes/ComfyDL/` — untracked plugin copy (nested git repo, ignored by `/custom_nodes/`) | `comfydl/` at repo root — registered git **submodule** pointing to `https://github.com/Cynthia-lxx/ComfyDL.git` |
| Node loading: ComfyUI custom-nodes scanner | Node loading: core init chain (`nodes.init_extra_nodes` → `init_builtin_dl_nodes`) |

```
ComfyDL_UI/
├── .gitmodules                  # [submodule "comfydl"] url=ComfyDL.git
├── comfydl/                     # git submodule (independent ComfyDL checkout, own .git)
├── nodes.py                     # + init_builtin_dl_nodes() (core registration)
└── docs/comfydl-builtin.md      # this file
```

## Registration Mechanism (注册机制)

`nodes.py` gained `async def init_builtin_dl_nodes()`, which runs inside
`init_extra_nodes()` right after `init_builtin_extra_nodes()`:

- `import comfydl` as a **real top-level package** (repo root is on `sys.path` at
  startup), so the relative imports inside `comfydl/__init__.py` and its
  `sys.path.insert(0, _PLUGIN_ROOT)` bootstrap keep working.
- Merges `comfydl.NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` into the shared
  core registry dicts in `nodes.py` — the same dicts served by `/object_info`, so the
  frontend shows the nodes automatically under their registered categories (`d2l/*`,
  plus the core `utilities` / `image*` categories used by the merged utility nodes).
- Each node class gets `RELATIVE_PYTHON_MODULE` set to its real module
  (e.g. `comfydl.nodes.tensor_ops`).
- If the submodule is missing on a fresh clone, it logs a warning with the fix
  command (`git submodule update --init`) and continues without crashing.

`nodes.py` 新增 `init_builtin_dl_nodes()`，在 `init_extra_nodes()` 中紧随
`init_builtin_extra_nodes()` 执行。它把 `comfydl` 作为真实顶层包导入，并把其节点映射
合并进核心注册表 dict（即 `/object_info` 所读取的同一份 dict），因此前端在各自分类
（`d2l/*`，以及并入的核心分类）下自动可见；若 submodule 缺失则告警提示且不崩溃。

Verification baseline (验证基线)：`CORE_BEFORE=11` core classes →
`CORE_AFTER=127` (11 core + 102 ComfyDL, key prefix `Cdl*`, 16 categories, plus the
14 core `Activation` nodes of reform step 1 in a 17th category, no IMPORT FAILED).

Measured after reform step 1 with
`init_extra_nodes(init_api_nodes=False, init_custom_nodes=False)`: `IMPORT_FAILED []`,
`CDL 102`, `ACTIVATION 14` (the raw `NODE_CLASS_MAPPINGS` dict holds 222 classes across
30 categories in total, because it also contains every other `comfy_extras` file).

## Category Layout (分类布局)

Teaching nodes live under `d2l/*` (frontend group 扩展 → d2l). Utility nodes that the
ComfyUI core does not cover were merged into the native core categories:

| Source (old category) | New category | Nodes |
|---|---|---|
| 12 teaching categories (`ComfyDL/Datasets`, `ComfyDL/CV Models`, …) | `d2l/<same leaf>` | 93 |
| `ComfyDL/Misc` | `utilities` (core) | 4 |
| `ComfyDL/Image Tools` | `image/color` (core) | 3 |
| `ComfyDL/Image Tools` | `image/transform` (core) | 1 |
| `ComfyDL/Image Tools` | `image` (core) | 1 |

Dropped in favour of the ComfyUI core nodes: `CdlImageResize` (→ `ImageScale` /
`ResizeImageMaskNode`), `CdlImageFlip` (→ `ImageFlip`), `CdlImageBlur` (→ `ImageBlur`),
`CdlImageCrop` (→ `ImageCrop` / `ImageCropV2`). `CdlImageRotate` was kept because the
core `ImageRotate` only supports 90-degree steps.

> **Update (2026-09-11)**: 7 overlapping teaching nodes were **soft-archived** into three
> `d2l/_Legacy/*` subcategories — `CdlActivation` / `CdlReshape` / `CdlBroadcast` (→
> `d2l/_Legacy/Tensor Basic`), `CdlAddNorm` / `CdlTransformerEncoderBlock` /
> `CdlTransformerEncoder` (→ `d2l/_Legacy/NLP Models`) and `CdlModelMode` (→
> `d2l/_Legacy/Model Utils`). Node ids are unchanged and the nodes still work, but their display
> names carry a `(DEPRECATED)` suffix. The registry now reports **108 nodes in 20 categories**.
>
> **更新（2026-09-11）**：7 个功能重叠的教学节点已**软归档**进三个 `d2l/_Legacy/*` 子分类（同上）。
> 节点 id 不变、仍可正常使用，仅显示名加 `(DEPRECATED)` 后缀。注册表现为 **108 个节点 / 20 个分类**。

教学节点统一落在 `d2l/*`（前端分组：扩展 → d2l）；核心确实没有等价实现、又不属于 d2l
内容的工具节点则并入 ComfyUI 原生分类（`utilities` / `image/color` / `image/transform` /
`image`）。`CdlImageResize`、`CdlImageFlip`、`CdlImageBlur`、`CdlImageCrop` 因核心已有
等价节点而删除；`CdlImageRotate` 因核心 `ImageRotate` 仅支持 90 度步进而保留。

## Working Inside the Submodule (submodule 内开发)

ComfyDL stays an independent repo:

1. `cd comfydl` and develop as usual (its own `.git`, its own `main` tracking
   `origin/main` = github.com/Cynthia-lxx/ComfyDL).
2. `git commit` + `git push origin main` there.
3. Back in ComfyDL_UI: `git add comfydl && git commit` — records the new submodule
   pointer so the pinned revision follows the upstream fix.
4. Fresh clones must run: `git submodule update --init` (after clone, before boot).

## Smoke Verification (冒烟验证)

Run from repo root with the penv interpreter:

```
python -c "import asyncio, nodes; f=asyncio.run(nodes.init_extra_nodes(init_api_nodes=False, init_custom_nodes=False)); cdl=[k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith('Cdl')]; print('IMPORT_FAILED', f, 'CDL_COUNT', len(cdl))"
```

Expected: `IMPORT_FAILED []`, `CDL_COUNT 102`, startup banner
`[ComfyDL] 已注册 102 个节点（显示名 102 个），共 16 个分类` printed once.

A full `python main.py --cpu` boot also works and logs the banner; optional full HTTP
check: `GET /object_info/CdlAccuracy`.

## Reform Step 1: Activation + TENSOR (reform 第一步)

The host runtime gained 14 core activation nodes (`comfy_extras/nodes_activation.py`,
category `Activation`) together with a new core `TENSOR` slot type. ComfyDL's own
`cdlTensor` / `cdlBbox` names were merged into the core `TENSOR` / `BBOX` types, so
ComfyDL nodes and core nodes now share the same slots. See
[reform-step1-activation-tensor.md](./reform-step1-activation-tensor.md) for the node
table, the `TENSOR` type/colour details and the rollback recipe.

宿主新增 14 个核心激活节点（`comfy_extras/nodes_activation.py`，分类 `Activation`）以及
新的核心插槽类型 `TENSOR`；ComfyDL 原有的 `cdlTensor` / `cdlBbox` 已并入核心的
`TENSOR` / `BBOX`，两者现在共用同一插槽。节点清单、`TENSOR` 类型与配色细节、回退方式见
[reform-step1-activation-tensor.md](./reform-step1-activation-tensor.md)。

### End-to-end (端到端)

A `/prompt` graph mixing the core image nodes with the merged ComfyDL nodes was executed
against a local `python main.py --cpu` instance:

```
LoadImage -> ImageScale -> ImageFlip -> ImageBlur -> CdlImageRotate -> CdlImageGrayscale
           -> CdlImageStats / CdlImageNormalize -> CdlImageAdjust -> PreviewImage
```

Result: `status_str = success`, `completed = True`, `PreviewImage` produced an output
image. This is exactly the graph that `example_workflows/Image Processing Chain.json`
now describes (core nodes replace the dropped `CdlImage*` ones).

上述 `/prompt` 链路（核心图像节点 + 合并后的 ComfyDL 节点）在本机 `python main.py --cpu`
实例上执行成功（`success` / `completed=True`，`PreviewImage` 有输出），与示例工作流
`example_workflows/Image Processing Chain.json` 一致。

## Reform Step 2: Network & Layers (reform 第二步)

The host runtime gained 8 core basic-layer nodes (`comfy_extras/nodes_layers.py`, category
`Network & Layers/Basic`), and the 14 activation nodes were moved from the top-level
`Activation` category to `Network & Layers/Activation`, so both groups now sit under one real
`Network & Layers` parent under **Comfy nodes**. The planned-but-never-implemented empty
categories (`Network & Layers`, `ComfyDL EX`) were dropped from the plan. A lightweight
in-process smoke tester was added at `cdl_smoke_tests/run_smoke_test.py`. See
[reform-step2-network-layers.md](./reform-step2-network-layers.md) for the node table, the
deduplication decisions and the rollback recipe.

宿主新增 8 个核心基础层节点（`comfy_extras/nodes_layers.py`，分类 `Network & Layers/Basic`），
并把 14 个激活节点从顶层 `Activation` 迁到 `Network & Layers/Activation`，两组节点现收敛到
**Comfy节点** 下一个真实的 `Network & Layers` 父分类；规划过但从未落地的空分类
（`Network & Layers`、`ComfyDL EX`）已从规划中移除。同时新增轻量进程内冒烟测试器
`cdl_smoke_tests/run_smoke_test.py`。节点清单、去重决策与回退方式见
[reform-step2-network-layers.md](./reform-step2-network-layers.md)。

Measured after reform step 2: `IMPORT_FAILED []`, `CDL 102`, `ACTIVATION 14`,
`BASIC 8`; the `/object_info` tree shows `Network & Layers/Activation` (14) +
`Network & Layers/Basic` (8), all with `python_module = comfy_extras.nodes_*`; the smoke
tester reports `208 PASS / 22 SKIP / 0 FAIL` across 230 registered nodes.

reform 第二步后实测：`IMPORT_FAILED []`、`CDL 102`、`ACTIVATION 14`、`BASIC 8`；
`/object_info` 分类树为 `Network & Layers/Activation`(14) + `Network & Layers/Basic`(8)，
`python_module` 均为 `comfy_extras.nodes_*`；冒烟测试器在 230 个已注册节点上给出
`208 PASS / 22 SKIP / 0 FAIL`。

## Reform Step 3: Normalization + train/eval state (reform 第三步)

The host runtime gained 7 core normalization nodes (`comfy_extras/nodes_normalization.py`,
category `Network & Layers/Normalization`) and 2 state nodes (`Network & Layers/Training`):
`TrainingMode` publishes the train/eval choice as a STRING that is wired into the `mode` slot of
BatchNorm / InstanceNorm, and `TrainingRunStats` carries `running_mean` / `running_var` as
editable widgets that emit 1-D tensors. The switch deliberately travels through a link instead of
a hidden-prompt handshake: the cache signature (`CacheKeySetInputSignature`) covers only wired
inputs and their ancestors, and the `IS_CHANGED` / validation call sites read the prompt with
`dynprompt=None`, so a prompt-based global switch would silently replay stale cached outputs. See
[reform-step3-normalization.md](./reform-step3-normalization.md) for the node tables, the
statistics priority, the robustness rules and the rollback recipe.

宿主新增 7 个核心归一化节点（`comfy_extras/nodes_normalization.py`，分类
`Network & Layers/Normalization`）与 2 个状态节点（`Network & Layers/Training`）：`TrainingMode`
把训练/推理选择以 STRING 形式发布，连到 BatchNorm / InstanceNorm 的 `mode` 插槽；
`TrainingRunStats` 用可编辑控件承载 `running_mean` / `running_var` 并输出 1 维张量。开关刻意走
连线而不是隐藏 prompt 握手：缓存签名（`CacheKeySetInputSignature`）只覆盖已连线的输入及其祖先，
而 `IS_CHANGED` 与校验流程读取 prompt 时 `dynprompt=None`，基于 prompt 的全局开关会静默复用过期
缓存。节点清单、统计量优先级、健壮性规则与回退方式见
[reform-step3-normalization.md](./reform-step3-normalization.md)。

Measured after reform step 3: `IMPORT_FAILED []`, `CDL 102`; the tree shows
`Network & Layers/Activation` (14) + `Basic` (8) + `Normalization` (7) + `Training` (2), all with
`python_module = comfy_extras.nodes_*`; the smoke tester reports `217 PASS / 22 SKIP / 0 FAIL`
across 239 registered nodes; flipping the `Training Mode` dropdown changes the cache key of every
linked consumer.

reform 第三步后实测：`IMPORT_FAILED []`、`CDL 102`；分类树为 `Network & Layers/Activation`(14) +
`Basic`(8) + `Normalization`(7) + `Training`(2)，`python_module` 均为 `comfy_extras.nodes_*`；
冒烟测试器在 239 个已注册节点上给出 `217 PASS / 22 SKIP / 0 FAIL`；翻转 `Training Mode` 下拉框会
改变每个已连线消费者的缓存键。

## Reform Step 4: Pooling + Convolution (reform 第四步)

The host runtime gained 4 core nodes: `comfy_extras/nodes_pooling.py` (category
`Network & Layers/Pooling`, `Pool` / `Adaptive Pool`) and `comfy_extras/nodes_convolution.py`
(category `Network & Layers/Convolution`, `Conv` / `ConvTranspose`). Twelve pooling semantics
collapse into two nodes because `mode` (max/avg) × `dims` (1/2/3) is just a dispatch table;
`output_size=1` **is** global pooling, so no `GlobalAvgPool` / `GlobalMaxPool` node is shipped.
The convolutions follow the `Linear` convention exactly — `weight` / `bias` arrive through
`TENSOR` slots and nothing is initialised inside the node, so there is no `in_channels` /
`out_channels` / `bias` widget to contradict the weights. `padding_mode` is implemented by hand
(an `F.pad` pass plus `padding=0`), because `F.conv{1,2,3}d` has no such argument; `ConvTranspose`
deliberately has no `padding_mode` widget at all. See
[reform-step4-pooling-convolution.md](./reform-step4-pooling-convolution.md) for the node tables,
the mode-specific-widget rules and the rollback recipe.

宿主新增 4 个核心节点：`comfy_extras/nodes_pooling.py`（分类 `Network & Layers/Pooling`，`Pool` /
`Adaptive Pool`）与 `comfy_extras/nodes_convolution.py`（分类 `Network & Layers/Convolution`，
`Conv` / `ConvTranspose`）。十二种池化语义压缩成两个节点，因为 `mode`(max/avg) × `dims`(1/2/3)
就是一张分派表；`output_size=1` **就是**全局池化，因此不单列 `GlobalAvgPool` / `GlobalMaxPool`。
卷积与 `Linear` 完全同构——`weight` / `bias` 从 `TENSOR` 插槽传入、节点内不做初始化，因此没有
`in_channels` / `out_channels` / `bias` 开关来和权重本身矛盾。`padding_mode` 手工实现（先 `F.pad`
再以 `padding=0` 调用卷积），因为 `F.conv{1,2,3}d` 没有这个参数；`ConvTranspose` 刻意不提供
`padding_mode` 控件。节点清单、模式专属控件规则与回退方式见
[reform-step4-pooling-convolution.md](./reform-step4-pooling-convolution.md)。

Measured after reform step 4: `IMPORT_FAILED []`, `CDL 109`; the tree shows
`Network & Layers/Pooling` (2) + `Network & Layers/Convolution` (2), all with
`python_module = comfy_extras.nodes_*`.

reform 第四步后实测：`IMPORT_FAILED []`、`CDL 109`；分类树为 `Network & Layers/Pooling`(2) +
`Network & Layers/Convolution`(2)，`python_module` 均为 `comfy_extras.nodes_*`。

## Reform Step 5: The `model` protocol layer (reform 第五步)

`MODEL` / `CLIP` / `VAE` are back as first-class values: 22 nodes in `model/loaders` (7),
`model/merging` (11), `model/latent` (2) and `model/conditioning` (2), built on a new framework
helper `comfy/model_protocol.py` that treats a checkpoint as a flat *state_dict* instead of a module
tree. Nothing recognises an architecture any more — the only structure used is the documented key
prefix (`diffusion_model.` → MODEL, `first_stage_model.` → VAE,
`cond_stage_model.` / `conditioner.` / `text_encoders.` → CLIP), which is exactly why 16 of the 22
nodes really execute: they read, split, merge and write weights. The merge nodes compute
`w_base * base + w_other * other` directly on the tensors instead of building `ModelPatcher`
patches, which would have to be applied through `comfy.lora.calculate_weight` and therefore crash.
The remaining 6 nodes (the two LoRA loaders, `VAE Decode` / `VAE Encode`, `CLIP Text Encode` /
`CLIP Set Last Layer`) keep the native IO contract so graphs can be wired today, and raise an
actionable `RuntimeError` instead of a bare `ModuleNotFoundError`. `comfy/` itself is untouched —
the module is a pure addition. See [reform-step5-model-protocol.md](./reform-step5-model-protocol.md)
for the L1/L2 boundary, the `model_protocol` API and the rollback recipe.

`MODEL` / `CLIP` / `VAE` 重新成为一等数据：`model/loaders`(7)、`model/merging`(11)、
`model/latent`(2)、`model/conditioning`(2) 共 22 个节点，建立在新增的框架侧辅助模块
`comfy/model_protocol.py` 之上，把 checkpoint 当作**扁平 state_dict** 而不是模块树来处理。这里不再
做任何结构识别——用到的唯一"结构"是约定俗成的键前缀（`diffusion_model.` → MODEL、
`first_stage_model.` → VAE、`cond_stage_model.` / `conditioner.` / `text_encoders.` → CLIP），
这也正是 22 个节点里有 16 个能真正执行的原因：它们真的读文件、拆组、合并、落盘。合并节点直接对张量
计算 `w_base * base + w_other * other`，而不是构造 `ModelPatcher` 补丁——补丁要在
`comfy.lora.calculate_weight` 里应用，那条路径必然崩。其余 6 个节点（两个 LoRA 加载器、
`VAE Decode` / `VAE Encode`、`CLIP Text Encode` / `CLIP Set Last Layer`）保留原生 IO 契约以便今天就能
搭出流程图，执行时抛可操作的 `RuntimeError` 而不是裸的 `ModuleNotFoundError`。`comfy/` 框架层零改动，
模块是纯新增。L1/L2 边界、`model_protocol` API 与回退方式见
[reform-step5-model-protocol.md](./reform-step5-model-protocol.md)。

Measured after reform step 5: `IMPORT_FAILED []`, `CDL 109`; the host registry holds 274 nodes across
44 categories; the smoke tester reports `251 PASS / 23 SKIP / 0 FAIL` across those 274 registered
nodes; the documented library totals 168 nodes across 32 categories (109 ComfyDL + 59 core).

reform 第五步后实测：`IMPORT_FAILED []`、`CDL 109`；宿主注册表共 274 个节点、44 个分类；冒烟测试器
在这 274 个已注册节点上给出 `251 PASS / 23 SKIP / 0 FAIL`；说明文件口径的节点库总计 168 个节点、
32 个分类（109 个 ComfyDL + 59 个核心节点）。

## Reform Step 6: Learnable parameters, optimizers and a training loop (reform 第六步)

The host runtime gained 9 core nodes (`comfy_extras/nodes_training.py`, category
`Network & Layers/Training`) plus two graph value types declared in `comfy_api/latest/_io.py`:
`PARAMS` (an ordered `{name: nn.Parameter}` mapping) and `OPTIMIZER` (an `OptimizerConfig` dataclass,
i.e. hyper-parameters rather than a live optimizer). ComfyUI wraps an entire prompt in
`torch.inference_mode()` (`execution.py:751`), so an autograd graph cannot cross a node boundary and
an inference tensor cannot be saved for backward: `Training Loop` therefore runs forward, backward and
`optimizer.step()` itself, for `steps` iterations, inside a `with torch.inference_mode(False):`
block — the same technique upstream's `TrainLoraNode` uses — and detaches every output so no graph is
left in ComfyUI's cache. `comfy/training_protocol.py` (a pure addition, no existing `comfy/*.py` file
is touched — the same pattern as step 5's `comfy/model_protocol.py`) holds the shared pieces:
`OptimizerConfig` / `build_optimizer`, the `MLP` with its documented `layer{i}.weight` naming, the
`missing` / `skipped` warm-start bookkeeping and the safetensors-in-base64 text codec.

宿主新增 9 个核心节点（`comfy_extras/nodes_training.py`，分类 `Network & Layers/Training`）以及
`comfy_api/latest/_io.py` 中的两个图数据类型：`PARAMS`（有序的 `{name: nn.Parameter}` 映射）与
`OPTIMIZER`（`OptimizerConfig` 数据类，即超参而不是活的优化器）。ComfyUI 会把整轮 prompt 包在
`torch.inference_mode()` 里（`execution.py:751`），autograd 图无法跨越节点边界、inference tensor
也无法保存用于反向，因此 `Training Loop` 自己在 `with torch.inference_mode(False):` 块内完成
`steps` 次前向、反向与 `optimizer.step()`（与上游 `TrainLoraNode` 同一手法），并把所有输出
detach，绝不把计算图留在 ComfyUI 缓存里。`comfy/training_protocol.py` 是纯新增模块（不改任何既有
`comfy/*.py`，与第五步的 `comfy/model_protocol.py` 同构），承载共用部分：`OptimizerConfig` /
`build_optimizer`、带 `layer{i}.weight` 命名约定的 `MLP`、热启动的 `missing` / `skipped` 记账，
以及 safetensors→base64 的文本编解码。

Persistence has two channels: `Save Parameters` / `Load Parameters` write and read a normal
`.safetensors` file in the output folder (and pass the set through, so saving does not end the graph),
while `Parameters to Text` / `Text to Parameters` carry the same payload in a widget
(`CDLPARAMS1:<base64>`) so a trained set survives inside a saved `.json` workflow. `Parameters to
Tensor` pulls a single entry back out as a plain tensor and is the bridge to the stateless `Basic` /
`Conv` layer nodes, which is what closes the "train, then infer" loop. See
[reform-step6-learnable-params-and-optimizer.md](./reform-step6-learnable-params-and-optimizer.md) for
the node table, the value type declarations and the rollback recipe.

持久化有两条通道：`Save Parameters` / `Load Parameters` 在输出目录读写普通 `.safetensors` 文件
（并原样透传参数集，因此保存不会中断图）；`Parameters to Text` / `Text to Parameters` 把同样的载荷
装进控件（`CDLPARAMS1:<base64>`），让训练产物随 `.json` 工作流一起保存。`Parameters to Tensor` 把
单个条目取回成普通张量，是通往无状态 `Basic` / `Conv` 层节点的桥——「先训练、后推理」的闭环正是由
它合上。节点清单、类型声明与回退方式见
[reform-step6-learnable-params-and-optimizer.md](./reform-step6-learnable-params-and-optimizer.md)。

Measured after reform step 6: `IMPORT_FAILED []`, `CDL 109`; the host registry holds 283 nodes across
44 categories; the smoke tester reports `260 PASS / 23 SKIP / 0 FAIL` across those 283 registered
nodes; the documented library totals 177 nodes across 32 categories (109 ComfyDL + 68 core).

reform 第六步后实测：`IMPORT_FAILED []`、`CDL 109`；宿主注册表共 283 个节点、44 个分类；冒烟测试器
在这 283 个已注册节点上给出 `260 PASS / 23 SKIP / 0 FAIL`；说明文件口径的节点库总计 177 个节点、
32 个分类（109 个 ComfyDL + 68 个核心节点）。

## Rollback (回退)

The migration was developed on `experiment/embed-comfydl` and merged into `master` with
an explicit merge commit. The experimental branch is kept on the remote as a backup and
can be dropped once it is no longer needed:

```
git push origin --delete experiment/embed-comfydl   # drop remote backup
git branch -D experiment/embed-comfydl              # drop local copy
```

To undo the migration on `master`, revert the merge commit
(`git revert -m 1 <merge-sha>`) or the individual migration commits; the upstream
ComfyDL repository is unaffected either way. Deleting `comfydl/` also removes the
submodule working tree.

迁移先在 `experiment/embed-comfydl` 完成，再以显式 merge commit 合入 `master`；实验分支
保留在远端作为备份，确认无需后可删除。若需在 `master` 上撤销迁移，回滚该 merge commit
（或逐个回滚迁移提交）即可，上游 ComfyDL 仓库不受影响。

## Related Commits (相关提交)

On `experiment/embed-comfydl` (then merged into `master`):

- `feat(builtin): track ComfyDL as git submodule under comfydl/`
- `feat(builtin): register comfydl into core node registry via init_builtin_dl_nodes`
- `chore(builtin): bump comfydl submodule to adc3c81 (GBK-safe startup banner)`
- `docs(comfydl): add bilingual built-in migration note`
- `chore(submodule): bump comfydl to 71cf770 (d2l/* categories + image tool merge)`
- `docs(comfydl): record category layout, end-to-end result and merge into master`

In ComfyDL repo (`main`), after the submodule migration:

- `refactor(categories): rename ComfyDL/* categories to d2l/* and Misc to utilities`
- `refactor(image-tools): drop 4 nodes with core equivalents and retarget 5 to core image categories`
- `chore(workflows): port Image Processing Chain example to core image nodes`
- `docs: regenerate node docs for d2l categories and image tool merge`

In ComfyDL repo (`main`), backported before migration:

- `fix(import): make plugin root importable under synthetic module names`
- `fix(deps): record IPython and matplotlib-inline runtime requirements`
- `fix(print): drop emoji from startup banner for GBK console safety`
