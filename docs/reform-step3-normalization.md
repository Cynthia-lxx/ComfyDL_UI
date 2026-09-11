# Reform Step 3: Normalization + the Train/Eval Channel (reform 第三步：归一化与训练/推理态通道)

> Scope of this step: add the third core node family, `Network & Layers/Normalization`
> (7 nodes), together with the two `Network & Layers/Training` state nodes that give a graph
> an explicit train/eval switch and persistent running statistics. The step also records *why*
> that switch travels through a **link** instead of a hidden-prompt handshake: the prompt
> route was proven cache-unsafe and was dropped.
>
> 本次改动的范围：新增第三个核心节点家族 `Network & Layers/Normalization`（7 个节点），以及
> `Network & Layers/Training` 下的 2 个状态节点，为图提供显式的训练/推理开关与可持久化的
> 运行统计量。文档同时记录：该开关为什么走**连线**而非隐藏 prompt 握手——后者已被证实会击穿
> 缓存，故放弃。
>
> Prerequisite: [reform-step2-network-layers.md](./reform-step2-network-layers.md).

---

## 1. Motivation (动机)

After step 2 the core branch (`Network & Layers`) had 14 activations and 8 basic layers — a
complete feed-forward stack, but nothing that normalizes. Everything normalizing lived inside
ComfyDL's model-level helpers (`d2l/Add & Norm`, `d2l/Transformer Encoder`, `d2l/Residual
Block`), which take a whole `nn.Module`; none of them can normalize a bare `TENSOR` in a graph.
A user assembling a CNN or a transformer out of core nodes therefore had no normalizer to
place between two `BasicLinear` nodes.

第二步之后，核心分支（`Network & Layers`）已有 14 个激活 + 8 个基础层，前馈结构齐了，但**没有
任何归一化**。归一化只存在于 ComfyDL 的模型级辅助节点里（`d2l/Add & Norm`、
`d2l/Transformer Encoder`、`d2l/Residual Block`），它们接收的是整个 `nn.Module`，无法对图里一个
裸 `TENSOR` 做归一化。用核心节点搭 CNN/Transformer 的用户，找不到能放在两个 `BasicLinear` 之间
的归一化节点。

The second gap is the train/eval distinction. `BatchNorm` and `InstanceNorm` compute different
things while training (statistics of the current call) and while inferring (running statistics),
and ComfyUI's execution model has no notion of a global mode: a cached graph is re-executed only
when its inputs change. A graph therefore needs a *place* to express the mode, and the first plan
— let the nodes discover it by introspecting the prompt — turned out to be wrong (§3).

第二个缺口是训练/推理态的区分：`BatchNorm` / `InstanceNorm` 在训练时用当前批次的统计量、推理时用
运行统计量，而 ComfyUI 的执行模型没有全局"模式"概念——缓存图只在其输入变化时才重算。所以图里必须
有一个**位置**来表达这个模式；而最初的方案（让节点通过自省 prompt 自行发现模式）是错的（见 §3）。

This step closes both gaps with 9 stateless nodes and one explicit wire.

本步用 9 个无状态节点 + 一根显式连线补齐这两个缺口。

---

## 2. Node List (节点清单)

All 9 nodes live in `comfy_extras/nodes_normalization.py`, use the ComfyUI **V3 `io` API**, and
are **stateless**: `weight` / `bias` / `g` / `u` / `v` are tensors wired in through `TENSOR`
slots, never created inside a node, and nothing is ever written in place (in ComfyUI the same
tensor may be shared by several nodes, so writing into a wired `running_mean` would corrupt an
unrelated branch). All of them preserve the input dtype/device and return one `TENSOR` named
`output` — except `NormalizationSpectralNorm`, which also exposes the estimated `sigma`.

9 个节点同处一个文件、使用 V3 `io` API，且**无状态**：`weight` / `bias` / `g` / `u` / `v` 都通过
`TENSOR` 插槽由外部线入，节点内部不创建、也从不原地写入（同一个张量可能被图里多个节点共享，原地写
`running_mean` 会污染无关分支）。全部保持输入 dtype/device，输出一个名为 `output` 的 `TENSOR`；
只有 `NormalizationSpectralNorm` 额外输出估计得到的 `sigma`。

### 2.1 Normalization (7 nodes)

| Node (node_id) | Display name | Inputs (all `TENSOR` unless noted) | Widgets | Purpose |
|---|---|---|---|---|
| `NormalizationBatchNorm` | BatchNorm | `tensor`, `weight`\*, `bias`\*, `running_mean`\*, `running_var`\*, `mode`\* (STRING) | `eps` FLOAT 1e-5 (0–1e-2) | `F.batch_norm` over dimension 1 of `(N, C, ...)` |
| `NormalizationInstanceNorm` | InstanceNorm | `tensor`, `weight`\*, `bias`\*, `running_mean`\*, `running_var`\*, `mode`\* | `eps` FLOAT 1e-5 (0–1e-2) | `F.instance_norm`: statistics per sample *and* per channel; needs rank ≥ 3 |
| `NormalizationLayerNorm` | LayerNorm | `tensor`, `weight`\*, `bias`\* | `normalized_shape` STRING `"last"`, `eps` FLOAT 1e-5 | `F.layer_norm` over the trailing dimensions (`"8,16"` = the last two) |
| `NormalizationGroupNorm` | GroupNorm | `tensor`, `weight`\*, `bias`\* | `num_groups` INT 1 (1–64), `eps` FLOAT 1e-5 | `F.group_norm`; `num_groups=1` normalizes over all channels |
| `NormalizationRMSNorm` | RMSNorm | `tensor`, `weight`\* | `normalized_shape` STRING `"last"`, `eps` FLOAT 1e-6 | `F.rms_norm`; LLaMA-style (no mean subtraction, no bias) |
| `NormalizationWeightNorm` | WeightNorm | `weight`, `g`\* | `dim` INT 0 (-8–7), `eps` FLOAT 1e-12 | Weight re-parameterization `g * v / ‖v‖₂` along `dim` |
| `NormalizationSpectralNorm` | SpectralNorm | `weight`, `u`\*, `v`\* | `n_power_iterations` INT 1 (0–20), `dim` INT 0 (-8–7), `eps` FLOAT 1e-12 | Divides a weight by a deterministic power-iteration estimate of its largest singular value; extra output `sigma` |

\* optional slot. Optional slots that are left unconnected are skipped, not replaced by a
randomly initialised default — an unwired `weight` really means "no gamma".

### 2.2 Training (2 nodes)

| Node (node_id) | Display name | Inputs | Widgets | Purpose |
|---|---|---|---|---|
| `TrainingMode` | Training Mode | — | `mode` COMBO `train` / `eval` (default `train`) | Publishes `train` / `eval` as a STRING for the `mode` slots of BatchNorm / InstanceNorm |
| `TrainingRunStats` | Training Run Stats | — | `running_mean` STRING `"0.0"`, `running_var` STRING `"1.0"` | Editable `running_mean` / `running_var`, emitted as two 1-D `TENSOR`s for the statistics slots |

### 2.3 One node per operation (一函数一节点)

BatchNorm does not exist in three sizes. `F.batch_norm` and `F.instance_norm` already accept any
`(N, C, ...)` tensor, so a single node covers all ranks: the shape decides which dimensions the
statistics are taken over (`(N, C)`/`(N, C, L)` behave like `BatchNorm1d`, `(N, C, H, W)` like
`BatchNorm2d`, `(N, C, D, H, W)` like `BatchNorm3d`). There is no `dim`-style branch to keep in
sync and no `BatchNorm1d/2d/3d` triangle of near-duplicate nodes.

BatchNorm 没有做成 1d/2d/3d 三份：`F.batch_norm` / `F.instance_norm` 本就接受任意 `(N, C, ...)`，
形状自己决定统计量在哪些维度上求（`(N, C)`/`(N, C, L)` 等同 1d，`(N, C, H, W)` 等同 2d，
`(N, C, D, H, W)` 等同 3d）。既没有需要同步的分支逻辑，也避免了一组近似重复的节点。

Every widget default is usable as-is: with no parameter edits a node produces a valid output.

所有控件默认值都"开箱可用"：不改任何参数即可产出有效输出。

---

## 3. The Train/Eval Channel (训练/推理态通道)

### 3.1 Why not a hidden prompt (为什么不用隐藏 prompt)

The first design let the normalization nodes read a global mode from the prompt — a hidden
`io.Hidden.prompt` input plus a node whose name the consumer would look up. It is not
implementable on this runtime, for two independent reasons:

最初的设计是让归一化节点从 prompt 里读取全局模式（隐藏的 `io.Hidden.prompt` 输入 + 消费者按节点名
去查）。在当前运行时上这行不通，有两个彼此独立的原因：

1. **The cache key never sees it.** `CacheKeySetInputSignature`
   (`comfy_execution/caching.py`) builds a node's signature from `node["inputs"]` plus the inputs
   of *every ancestor* reachable through links. A hidden input that is not a wire is not part of
   either, so a consumer's key stays identical when a global dropdown flips → the consumer is
   served from cache and the graph silently does not re-run.
2. **It is not even readable when the key is computed.** The two places that evaluate a node
   outside a normal execution — `IS_CHANGED` and the validation pass — call
   `get_input_data(inputs, class_def, unique_id)` (`execution.py:91`, `execution.py:1083`)
   without `dynprompt`, so the hidden `prompt` is an empty dict there. Only the real execution
   path (`execution.py:493`) passes `dynprompt`.

1. **缓存键看不到它。** `CacheKeySetInputSignature`（`comfy_execution/caching.py`）用
   `node["inputs"]` 加**所有上游祖先**的 inputs 生成签名；未接线的隐藏输入两边都不在，于是全局开关
   翻转后消费者的键完全不变 → 消费者直接吃缓存，图静默不重算。
2. **算签名时它根本读不到。** 两处在正常执行之外求值节点的地方（`IS_CHANGED` 与校验流程）调用
   `get_input_data(inputs, class_def, unique_id)`（`execution.py:91`、`execution.py:1083`）时没有传
   `dynprompt`，隐藏 `prompt` 在那里是空字典；只有真正的执行路径（`execution.py:493`）才传
   `dynprompt`。

A correct version would have needed external tooling (rewriting saved workflows on save, or a
fingerprint hook), which is exactly the kind of hidden magic this reform is removing.

要做对，只能靠外部手段（保存时改写工作流、或打 fingerprint 补丁），而这正是本次重构想消除的"隐形
魔法"。因此该方案被放弃。

### 3.2 The chosen channel: a real link (选定的通道：一根真实的连线)

| Producer | Payload | Consumer slot | Behaviour |
|---|---|---|---|
| `TrainingMode` (`mode` output) | `STRING` = `"train"` / `"eval"` | `mode` on BatchNorm / InstanceNorm: `io.String.Input("mode", default="train", optional=True, force_input=True)` | Linked → the switch reaches the consumer **and** its cache key; unconnected → `"train"` (PyTorch's default) |

Details that make this work:

- **STRING, not a COMBO socket.** `validate_node_input` (`comfy_execution/validation.py`) refuses
  to connect a `STRING` output into a `[train, eval]` combo input, so a dropdown-shaped socket
  would reject the very link the design depends on. STRING ⇄ STRING connects.
- **`force_input=True`** keeps the slot a pure socket: the mode is meant to arrive through a
  link, and the value that would otherwise be typed is the default anyway.
- **No producer needed.** An unconnected `mode` slot means `train`, so BatchNorm / InstanceNorm
  work standalone in a graph that never heard of `TrainingMode`.
- **Cache correctness comes for free.** The link is part of the consumer's signature and the
  producer's widget is part of the producer's signature, which is itself an ancestor entry of the
  consumer's key — so flipping the dropdown invalidates the consumer and everything downstream.
  Verified in-process, see §8.
- **No name lookup, no prompt scan.** `TrainingMode` is only evaluated when a consumer pulls it,
  so dragging one onto the canvas without wiring it costs nothing.
- **Renaming is a canvas action.** Double-clicking the node renames it, which is the labelling
  the old plan intended to get from instance names, without any code or cache special-casing.

几点关键设计：

- **消费者侧用 STRING 而不是下拉。** `validate_node_input`（`comfy_execution/validation.py`）拒绝把
  `STRING` 接到选项为 `[train, eval]` 的 COMBO 输入上，所以做成下拉的插槽反而会拒绝本设计所依赖的
  那根连线；STRING ⇄ STRING 才能连通。
- **`force_input=True`** 让该插槽保持纯插槽形态：模式本就该由连线传入，能用手输的值就是默认值。
- **可以不接。** `mode` 不接即 `train`，BatchNorm / InstanceNorm 可以在完全不知道 `TrainingMode`
  的图里独立工作。
- **缓存正确性是白拿的。** 连线进入消费者签名，生产者控件的值进入生产者签名，而生产者签名又是消费者
  键的祖先项——翻转下拉框会同时失效消费者及其下游。已在本机实测，见 §8。
- 无节点名查表、无 prompt 扫描；拖一个没接线的 `TrainingMode` 上画布零成本（惰性求值）；双击改名
  即可获得旧方案想要的"命名"能力。

---

## 4. Running Statistics (运行统计量)

`TrainingRunStats` is the persistent half of the switch. A widget is what survives in a saved
workflow, so both statistics are typed in as comma separated numbers and emitted as 1-D tensors;
feeding the batch mean/variance of a run into these widgets is how a mean/variance carries from
one run to the next.

`TrainingRunStats` 是开关的持久化那一半。能随工作流保存下来的只有控件值，所以两个统计量以逗号分隔
的数字输入、以 1 维张量输出；把某次运行的批均值/方差填进这两个控件，就是统计量跨运行传递的方式。

Priority inside BatchNorm / InstanceNorm — the first row that applies wins:

| # | `mode` | wired `running_mean` / `running_var` | What is used |
|---|---|---|---|
| 1 | `eval` | both usable | the wired statistics (`F.batch_norm(..., training=False)` / `F.instance_norm(..., use_input_stats=False)`) |
| 2 | `train` | any | the statistics of this call; wired statistics are **ignored**, exactly as in PyTorch |
| 3 | `eval` | missing or unusable | the statistics of this call, with a printed warning |

- A single value is broadcast to every channel (that is what the default `"0.0"` / `"1.0"`
  widgets produce); a per-channel list is used as is; a list whose length is neither 1 nor `C` is
  ignored with a warning → row 3.
- The wired statistics are **never written back**. To carry statistics across runs, read them off
  the tensors and type them into a `TrainingRunStats` node.
- **Known limitation:** the nodes do not export the batch statistics they computed, so today the
  values for the widgets have to be read off the tensor by the user. Automatic export
  (a statistics output on BatchNorm / InstanceNorm) is a natural follow-up, not part of this step.

---

## 5. Robustness (健壮性)

Following the precedents of steps 1–2, every unusable parameter degrades to something valid plus
a printed warning; no user input can abort a running workflow. The warnings are prefixed
`[Network & Layers]` so they are greppable in the console.

沿用前两步的约定：任何不可用的参数都退化成有效结果并打印警告，用户输入不会中断正在跑的工作流。
警告统一带 `[Network & Layers]` 前缀，便于在控制台检索。

| Situation | Behaviour |
|---|---|
| BatchNorm on rank < 2 | input returned unchanged + warning (`(N, C)` is the minimum) |
| InstanceNorm on rank < 3 | input returned unchanged + warning (there must be a spatial dimension) |
| LayerNorm / RMSNorm on a 0-dim tensor | input returned unchanged + warning |
| WeightNorm on a 0-dim weight, SpectralNorm on rank < 2 | input returned unchanged + warning; SpectralNorm also returns `sigma = 0` |
| `num_groups` < 1 or not dividing the channel count | one group is used instead + warning |
| `normalized_shape` stale or unparsable (`"8,32"` on a `(4,8,16)` tensor, `"abc"`) | only the *count* of dimensions is taken from the widget; the actual trailing sizes win + warning |
| statistics whose element count is neither 1 nor `C` | ignored → statistics of this call + warning |
| `dim` outside `[-rank, rank - 1]` | clamped like `ActivationSoftmax` does |
| all-zero weight / all-zero σ̂ | `clamp_min(eps)`, so nothing divides by zero |
| no RNG anywhere | the power iteration starts from the weight's own row/column sums, so a rerun is bit-identical and caching stays sound |

`_split_numbers` also accepts full-width commas/semicolons (`"0.1，0.2"` typed on a Chinese IME)
and whitespace as separators, so a list typed in any of the usual ways parses.

`_split_numbers` 还接受全角逗号/分号（中文输入法下输入的 `"0.1，0.2"`）与空格分隔，常见输入方式都能
解析。

---

## 6. Category Tree & Counts (分类树与计数)

| | Before step 3 | After step 3 |
|---|---|---|
| Comfy nodes | `Network & Layers/Activation` (14) + `Basic` (8) | + `Normalization` (7) + `Training` (2) |
| Extensions | `d2l/*` (102 nodes / 16 categories) | unchanged |

```
Comfy节点
└── Network & Layers
    ├── Activation      14   (reform step 1)
    ├── Basic            8   (reform step 2)
    ├── Normalization    7   (this step)
    └── Training         2   (this step)
扩展
└── d2l + image + utilities   102  (unchanged, 16 categories)
```

The shipped library overview becomes **133 nodes across 20 categories** = 102 ComfyDL +
31 core (14 + 8 + 7 + 2). The host registry itself reports 239 nodes in 33 categories, the
difference being the remaining upstream ComfyUI core nodes that this project does not document.

随节点库说明文档一起维护的总数为 **20 个分类 133 个节点** = ComfyDL 102 + 核心 31（14+8+7+2）。
宿主注册表本身是 33 个分类 239 个节点，差额是本项目不负责记录的其余 ComfyUI 上游核心节点。

---

## 7. Relation to `d2l` (与 d2l 节点的关系)

| `d2l` node | Overlap | Resolution |
|---|---|---|
| `CdlModelMode` (`d2l/Model Utils`) | also switches train/eval, but it **mutates a `cdlModel`** (`model.train()` / `model.eval()`) and passes the module on; it has no effect on a stateless tensor graph | coexist — different payload (`cdlModel` vs a STRING that reaches `TENSOR` nodes) |
| `CdlAddNorm`, `CdlTransformerEncoder*` (`d2l/NLP Models`) | wrap residual + LayerNorm **inside a module**; they cannot normalize a bare tensor | coexist — `NormalizationLayerNorm` + `BasicAdd` compose the same math on `TENSOR`s |
| nothing | there was no tensor-level BatchNorm / InstanceNorm / GroupNorm / RMSNorm / WeightNorm / SpectralNorm | new, no duplicate |

As in step 2 the decision is **coexistence**: the `d2l` teaching nodes are untouched and the new
core nodes reimplement the semantics independently. Dropping or archiving the model-level
duplicates is a later, separate decision.

与第二步一致，本次决策是**并存**：`d2l` 教学节点原样不动，新核心节点独立实现同一语义；是否弃用
模型级重复项留待后续决定。

---

## 8. Verification (验证记录)

- **Registry**: `init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)` reports
  `IMPORT_FAILED []`; all 9 nodes are registered in `Network & Layers/Normalization` (7) and
  `Network & Layers/Training` (2), every one carrying
  `python_module = comfy_extras.nodes_normalization` (first segment hits the frontend whitelist,
  so they are classified as **Comfy nodes**).
- **Smoke test**: `cdl_smoke_tests/run_smoke_test.py` reports `217 PASS / 22 SKIP / 0 FAIL`
  across 239 registered nodes; every SKIP carries its reason
  (`--categories` prints `Network & Layers/Activation 14`, `Basic 8`, `Normalization 7`,
  `Training 2`).
- **The train/eval channel** (in-process, via `CacheKeySetInputSignature` + a `DynamicPrompt`):
  the key of a BatchNorm node linked to `TrainingMode` **changes** when the dropdown is flipped
  (`train` → `eval`), is **stable** when it is not, and an *unlinked* BatchNorm is unaffected by
  the dropdown — which is precisely the failure mode the dropped hidden-prompt design would have
  had.
- **Statistics priority**: `eval` + wired stats equals `F.batch_norm(..., training=False)` with
  the broadcast values; `train` ignores wired stats and equals `F.batch_norm(..., training=True)`;
  `eval` without usable stats falls back to the batch statistics with a warning (also when a
  statistics tensor holds the wrong number of values). The input tensor is bit-identical
  afterwards, and fp16 stays fp16.
- **Numerics**: LayerNorm `"last"` / `"8,16"` / stale `"8,32"` / typo `"abc"` all match the
  expected `F.layer_norm` shapes; RMSNorm matches `F.rms_norm`; GroupNorm matches `F.group_norm`
  for `num_groups=3` and degrades to one group for an indivisible `5`; WeightNorm yields unit norm
  along `dim=0` and applies `g`; SpectralNorm is deterministic, needs no seed, and its
  `sigma` equals `torch.linalg.matrix_norm(W, ord=2)` after 20 iterations (an output spectral norm
  of 1.0).
- **Rank guards**: InstanceNorm on rank 2 and SpectralNorm on rank 1 return the input untouched
  (with `sigma = 0`), as documented; BatchNorm on rank 2 is *valid* `(N, C)` input and normalizes
  normally.

> **Measured characteristic of the `n_power_iterations` default.** The default of 1 iteration is
> the same as `torch.nn.utils.spectral_norm`, but that module warm-starts `u`/`v` from the previous
> call, whereas this node is stateless by design. Accuracy therefore depends on the weight size:
> on a `(6, 5)` weight 1 iteration is already exact (est/true `0.996`), on `(64, 64)` it
> under-estimates (`0.706`, so the output is normalized to ≈1.4 instead of 1.0), on `(256, 128)`
> `0.783`. It converges as expected — `(64, 64)` reaches `1.000` and `(256, 128)` reaches `0.998`
> at 20 iterations — so raise the widget when a tight Lipschitz bound matters.
>
> **`n_power_iterations` 默认值的实测特性。** 默认 1 次迭代与 `torch.nn.utils.spectral_norm`
> 相同，但后者会把上次的 `u`/`v` 热启动，而本节点按设计是无状态的。因此精度取决于权重规模：
> `(6, 5)` 上 1 次迭代即精确（估计/真值 `0.996`），`(64, 64)` 上偏小（`0.706`，输出谱范数约为
> 1.4 而非 1.0），`(256, 128)` 上 `0.783`；收敛性正常——20 次迭代时分别达到 `1.000` 与 `0.998`，
> 需要严格 Lipschitz 界时把该控件调大即可。

---

## 9. Rollback (回退)

1. Delete `comfy_extras/nodes_normalization.py` and remove `"nodes_normalization.py"` from the
   `extras_files` list in `nodes.py`.
2. Optionally delete this document.
3. `git revert` the reform step 3 commit(s) in `ComfyDL_UI` (the node family and the four shipped
   doc files that describe it live in two repositories — see the commit list in
   [comfydl-builtin.md](./comfydl-builtin.md)).

Nothing in this step touches the `comfy/` registration chain or adds a slot type, so a revert is
a plain source revert with no data or state to migrate; saved workflows that use the nodes will
show the missing nodes and need the step-3 nodes re-added.

本步不触碰 `comfy/` 注册链、也不新增插槽类型，回退就是纯源码回退，没有数据或状态需要迁移；用到这些
节点的工作流会显示节点缺失，重新加入即可。
