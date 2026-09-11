# Reform Step 2: Basic Network Layers + the `Network & Layers` Category (reform 第二步：基础网络层与 `Network & Layers` 分类)

> Scope of this step: turn the plain feed-forward building blocks (affine layers, shape ops,
> tensor combination) into *core* nodes, gather them with the step-1 activations under a real
> `Network & Layers` parent category, drop the never-implemented empty categories from the
> plan, and add a lightweight node smoke tester that does not need a running server.
>
> 本次改动的范围：把最基础的神经网络层（仿射层、形状变换、张量组合）做成**核心**节点，
> 与第一步的激活节点一起收敛到真实的 `Network & Layers` 父分类下，从规划中移除从未落地的
> 空分类，并新增一套**不需要启动服务**的轻量节点冒烟测试器。
>
> Prerequisite: [reform-step1-activation-tensor.md](./reform-step1-activation-tensor.md).

---

## 1. Motivation (动机)

After step 1 the node library had an `Activation` category sitting directly under
**Comfy nodes**, plus two empty placeholders in the plan (`Network & Layers`, `ComfyDL EX`).
There was no core node that could build an actual network: no affine layer, no shape
manipulation, no tensor combination. Users had to fall back to the `d2l/` teaching nodes for
that, which put the "useful core" and the "archived teaching code" on the same level.

第一步之后，节点库里 `Comfy节点` 下只有一个 `Activation` 分类，规划中的 `Network & Layers`、
`ComfyDL EX` 都是空的。核心节点无法搭建任何真实网络：没有全连接层、没有形状变换、没有张量
组合，只能退回到 `d2l/` 教学节点，"有用的核心"和"归档的教学代码"混在同一层级。

This step fixes that: 8 stateless `Basic` nodes join the 14 activations under a two-level
`Network & Layers` branch, and the empty categories are dropped from the plan.

---

## 2. Node List (节点清单)

All 8 nodes live in `comfy_extras/nodes_layers.py`, use the ComfyUI **V3 `io` API**, sit in
`Network & Layers/Basic`, take one or two `TENSOR` inputs and return exactly one `TENSOR`
output named `output`. They are **stateless**: `weight` / `bias` are tensors wired in through
input slots, never created inside the node, so the same node can drive any checkpoint and
nothing is cached between executions.

| Node (node_id) | Display name | Inputs | Extra widget | Purpose |
|---|---|---|---|---|
| `BasicLinear` | Linear | `tensor`, `weight`, `bias` (optional) | — | `tensor @ weight.T + bias` (`F.linear`) |
| `BasicEmbedding` | Embedding | `tensor` (indices), `weight` | — | Row lookup into `weight` (`F.embedding`) |
| `BasicFlatten` | Flatten | `tensor` | `start_dim` INT 1 (0–4), `end_dim` INT -1 (-4–4) | Flattens `start_dim..end_dim` (`torch.flatten`) |
| `BasicReshape` | Reshape | `tensor` | `target_shape` STRING `"1,-1"` | `torch.reshape` from a shape string |
| `BasicBroadcast` | Broadcast | `tensor` | `target_shape` STRING `"2,3"` | `torch.broadcast_to` from a shape string |
| `BasicConcat` | Concat | `a`, `b` | `dim` INT -1 (-4–4) | `torch.cat((a, b), dim)` |
| `BasicAdd` | Add | `a`, `b` | — | Broadcast element-wise `a + b` |
| `BasicMultiply` | Multiply | `a`, `b` | — | Broadcast element-wise `a * b` |

Robustness rules, all following existing precedents:

- `dim` is clamped into `[-rank, rank - 1]` like `ActivationSoftmax` does.
- `Flatten` swaps `start_dim` / `end_dim` when they resolve to a reversed pair, and passes a
  0-dim tensor through.
- `Reshape` / `Broadcast` return the input tensor unchanged (with a printed warning) when the
  shape string cannot be parsed or is incompatible, so a typo never breaks a workflow.
- `Linear` promotes `tensor` / `weight` to their common floating dtype first (fp16 activations
  with fp32 weights run in fp32) instead of raising a dtype mismatch.

所有控件默认值都"开箱可用"：不改任何参数即可产出有效输出。

---

## 3. Category Tree (分类树)

`CATEGORY` uses the `parent/child` form, which the frontend turns into a nested menu without
any custom JS.

| | Before step 2 | After step 2 |
|---|---|---|
| Comfy nodes | `Activation` (14) | `Network & Layers/Activation` (14) |
| Comfy nodes | — (planned empty `Network & Layers`) | `Network & Layers/Basic` (8) |
| Comfy nodes | — (planned empty `ComfyDL EX`) | *removed from the plan* |
| Extensions | `d2l/*` (16 categories) | `d2l/*` (unchanged) |

```
Comfy节点
└── Network & Layers
    ├── Activation   14   (reform step 1)
    └── Basic         8   (this step)
扩展
└── d2l              102  (unchanged, 16 categories)
```

The two empty placeholders were never present in code — a full-repo search for `Network &
Layers` and `ComfyDL EX` before this step returned zero hits, and the frontend does not render
empty categories. They are dropped from the plan (and from the memory notes) rather than
deleted from code.

两个空分类**从未存在于代码中**（改动前全仓检索两个字符串均 0 命中，前端也不渲染空分类），
因此"删除"只落在规划与记忆文档中，代码侧没有删除动作。

---

## 4. Deduplication (去重)

The reform plan listed more layer names than are actually distinct. Following the user's
"if two are the same, keep only one" rule, the following were dropped:

| Planned node | Resolution |
|---|---|
| `Dense` | ≡ `Linear` (same affine transform) → `Linear` only; a "Dense" layer is `Linear` + an activation node |
| `Residual` | ≡ `Add` (broadcast element-wise sum) → `Add` only |
| `Skip Connection` | ≡ `Add` → `Add` only |

This is why `Basic` ends up with **8** nodes instead of the ~11 originally sketched.

---

## 5. Relation to `d2l` (与 `d2l` 节点的关系)

`CdlReshape` and `CdlBroadcast` (`d2l/Tensor Basic`) overlap with `BasicReshape` /
`BasicBroadcast`. Per the decision on this step they **coexist**: the `d2l` teaching nodes are
left untouched, and the new core nodes reimplement the same semantics independently (differing
only in robustness and in using the shared `TENSOR` slot). Removing or archiving the `d2l`
duplicates is a later, separate decision.

`CdlReshape`、`CdlBroadcast`（`d2l/Tensor Basic`）与新核心节点功能重叠，本次决策为**并存**：
`d2l` 教学节点原样不动，新核心节点独立实现同一语义。是否归档 `d2l` 重复项留待后续决定。

---

## 6. Smoke Tester (冒烟测试器)

`cdl_smoke_tests/run_smoke_test.py` is a **lightweight, in-process** node tester. It does not
start the ComfyUI server; it calls
`asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))`, then
synthesises dummy inputs for every registered node and executes it.

- handles both node styles: V3 (`GET_SCHEMA` / `define_schema` + classmethod `execute`) and
  legacy (`INPUT_TYPES` + instance `FUNCTION`);
- covers the core slot types (`TENSOR`, `IMAGE`, `MASK`, `LATENT`, `INT`, `FLOAT`, `STRING`,
  `BOOLEAN`, `COMBO`, `BBOX`, `ARRAY`, `DICT`, `COLOR`, …), `UNION`/`*` types and `Autogrow`;
- skips nodes that need external resources or write to disk, always printing the reason
  (`SKIP(reason)` rather than `FAIL`) and cleaning up any temporary sandbox files;
- prints a `PASS / SKIP / FAIL` summary table and exits non-zero when anything fails.

```
penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py            # all nodes
penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py --categories
penv\Scripts\python.exe cdl_smoke_tests\run_smoke_test.py --filter Basic --verbose
```

---

## 7. Verification (验证记录)

- Registry: `init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)` reports
  `IMPORT_FAILED []`; the 8 `Basic*` nodes are registered with category
  `Network & Layers/Basic`, the 14 `Activation*` nodes with `Network & Layers/Activation`, and
  all of them take/return `TENSOR`.
- Smoke test: `208 PASS / 22 SKIP / 0 FAIL` across 230 registered nodes; the `Network & Layers`
  branch is 22 PASS (14 Activation + 8 Basic), every SKIP carries a reason.
- HTTP (`--cpu` server, `/object_info`): the tree is `Network & Layers/Activation` = 14 and
  `Network & Layers/Basic` = 8; every one of the 22 has
  `python_module = comfy_extras.nodes_*` (first segment hits the frontend whitelist, so they are
  classified as **Comfy nodes**); `ComfyDL EX` does not exist.

---

## 8. Rollback (回退)

1. Delete `comfy_extras/nodes_layers.py` and remove `"nodes_layers.py"` from the `extras_files`
   list in `nodes.py`.
2. Revert `comfy_extras/nodes_activation.py`'s `CATEGORY` back to `"Activation"`.
3. Optionally delete `cdl_smoke_tests/` and this document.
4. `git revert` the reform commit(s) in `ComfyDL_UI`.

Nothing here touches the `comfy/` registration chain, so a revert is a plain source revert with
no data or state to migrate.
