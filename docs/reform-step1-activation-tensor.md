# Reform Step 1: Activation Nodes + the `TENSOR` Slot Type (reform 第一步：激活节点与 `TENSOR` 类型)

> Scope of this step: add a first batch of *core* activation nodes to the host runtime,
> introduce a generic `TENSOR` slot type for them, and merge ComfyDL's self-made slot
> names into the ComfyUI core ones so both node families can be wired together.
>
> 本次改动的范围：向宿主运行时加入第一批**核心**激活节点，为其引入通用 `TENSOR` 插槽
> 类型，并把 ComfyDL 自创的插槽名归并到 ComfyUI 核心类型上，使两套节点可以直接互连。

---

## 1. Motivation (动机)

Before this step the only activation node in the tree was ComfyDL's `CdlActivation`, and the
ComfyDL nodes exchanged tensors through a self-made `cdlTensor` type that no core node knew
about. Two problems followed:

- the activation family was a single teaching node with no per-function granularity;
- `cdlTensor` and the core slot types were disjoint, so a ComfyDL tensor output could not be
  wired into a core node without going through an adapter.

改动前，整个仓库里唯一的激活节点是 ComfyDL 的 `CdlActivation`，而 ComfyDL 节点之间通过自创的
`cdlTensor` 类型传递张量，核心节点并不认识这个类型。由此产生两个问题：激活功能只有一个教学
节点、缺少按函数拆分的粒度；`cdlTensor` 与核心插槽类型互不相通，ComfyDL 的张量输出无法直接
接到核心节点上。

This step fixes both: a new `comfy_extras/nodes_activation.py` provides 14 single-purpose
activation nodes on a new core `TENSOR` type, and the ComfyDL slot names are merged into the
core names.

---

## 2. Node List (节点清单)

All 14 nodes live in `comfy_extras/nodes_activation.py`, use the ComfyUI **V3 `io` API**, and
sat in a top-level `Activation` category when this step landed (`Comfy nodes → Activation`);
reform step 2 moved the whole group to `Network & Layers/Activation` — see
[reform-step2-network-layers.md](./reform-step2-network-layers.md). Each takes exactly one
`TENSOR` input (`tensor`), returns exactly one `TENSOR` output (`output`), preserves the input
dtype/device, and exposes no learnable parameters.

| Node (node_id) | Display name | Extra widget | Purpose |
|---|---|---|---|
| `ActivationSigmoid` | Sigmoid | — | `1 / (1 + exp(-x))`; range (0, 1) |
| `ActivationTanh` | Tanh | — | `tanh(x)`; range (-1, 1) |
| `ActivationReLU` | ReLU | — | `max(0, x)` |
| `ActivationLeakyReLU` | Leaky ReLU | `negative_slope` FLOAT `0.01` (0.0–1.0) | ReLU with a slope for `x < 0` |
| `ActivationELU` | ELU | `alpha` FLOAT `1.0` (0.0–100.0) | `x` if `x > 0`, else `alpha * (exp(x) - 1)` |
| `ActivationSELU` | SELU | — | Self-normalizing ELU (standard constants) |
| `ActivationGELU` | GELU | `approximate` COMBO `none`/`tanh` | Gaussian error linear unit |
| `ActivationSiLU` | SiLU | — | `x * sigmoid(x)` (swish) |
| `ActivationMish` | Mish | — | `x * tanh(softplus(x))` |
| `ActivationSoftplus` | Softplus | — | `log(1 + exp(x))` (beta=1, threshold=20) |
| `ActivationReLU6` | ReLU6 | — | `min(max(0, x), 6)` |
| `ActivationHardSwish` | Hard Swish | — | `x * relu6(x + 3) / 6` |
| `ActivationIdentity` | Identity | — | Zero-copy pass-through |
| `ActivationSoftmax` | Softmax | `dim` INT `-1` (-4–4) | Normalizes along `dim` (clamped to the rank) |

Every widget default is usable as-is: the node produces a valid output with no parameter
edits. `ActivationSoftmax` clamps `dim` into `[-rank, rank - 1]` (and returns an all-ones
tensor for a 0-dim input) so an out-of-range value never breaks a workflow.

所有控件的默认值都"开箱可用"：不改任何参数即可产出有效输出。`ActivationSoftmax` 会把 `dim`
钳制到 `[-rank, rank - 1]`（0 维输入返回全 1 张量），越界不会中断工作流。

---

## 3. The `TENSOR` Slot Type (`TENSOR` 插槽类型)

`TENSOR` is registered on both type systems:

| Where | Change |
|---|---|
| `comfy/comfy_types/node_typing.py` | `IO` enum gains `TENSOR = "TENSOR"` (after `VIDEO`) |
| `comfy_api/latest/_io.py` | `@comfytype(io_type="TENSOR") class Tensor(ComfyTypeIO): Type = torch.Tensor`, plus `"Tensor"` appended to the explicit `__all__` |

Notes:

- `IMAGE` / `MASK` / `LATENT` all carry a specific payload semantic, so none of them is a
  suitable slot for an arbitrary N-D tensor; `TENSOR` fills that gap.
- `_io_public.py` is `from ._io import *`, so forgetting the `__all__` entry silently hides
  `io.Tensor` from schema definitions.
- `execution.py` performs no central type whitelist check for new type strings (it only
  converts INT/FLOAT/STRING/BOOLEAN values and validates combo options), so adding `TENSOR`
  is safe.
- The ComfyDL nodes were switched over to this type (see §4), which is what makes the two
  families interoperable.

### 3.1 Slot Colour (插槽配色)

`TENSOR` uses **lemon green `#C6FF00`**, chosen because it collides with none of the
non-empty built-in slot colours (`#eacb8b #A8DADC #ad7452 #cf876f #00d78d #80a1c0 #b38ead
#a3bd8d #8978a7 #C2FFAE #DCC274 #be616b`).

The colour lives in the **frontend package**, not in this repository:

```
penv/Lib/site-packages/comfyui_frontend_package/static/assets/settingStore-<hash>.js
```

Six theme tables (`arc`, `dark`, `github`, `light`, `solarized`, `nord`) each contain a
`colors.node_slot` map; `TENSOR:\`#C6FF00\`` was appended to each of them, right after the
last entry and before the closing brace.

> ⚠ **The frontend package is not under version control.** Upgrading
> `comfyui-frontend-package` changes the file name (the hash) and drops this edit — the slot
> colour must then be re-applied to the new file.
>
> ⚠ **前端包不在版本控制内。** 升级 `comfyui-frontend-package` 后文件名（hash）会变化，该改动
> 会丢失，需要重新应用到新文件。

To re-apply, from the repo root with the venv interpreter:

```python
# Re-apply the TENSOR slot colour after a frontend package upgrade.
import re, glob
from pathlib import Path

path = glob.glob('penv/Lib/site-packages/comfyui_frontend_package/static/assets/settingStore-*.js')[0]
p = Path(path)
src = p.read_bytes().decode('utf-8')
MARK, INSERT = 'node_slot:{', ',TENSOR:`#C6FF00`'
assert 'TENSOR:' not in src, 'already applied'

out, pos, search, n = [], 0, 0, 0
while True:
    i = src.find(MARK, search)
    if i < 0:
        break
    j = i + len(MARK)
    search = j                                   # search cursor only
    if src[j:j + 3] == '...':                    # palette-merge code, not a theme table
        continue
    end = src.find('}', j)
    out.append(src[pos:end]); out.append(INSERT); pos = end; n += 1
out.append(src[pos:])
new = ''.join(out)
assert n == 6 and len(new) == len(src) + n * len(INSERT), f'unsafe: n={n}'
p.write_bytes(new.encode('utf-8'))
print('patched', n, 'theme tables')
```

Important: only the six tables whose value starts with a real key (`node_slot:{BOOLEAN:…`)
are theme palettes. The palette-merge expression `node_slot:{...t.colors.node_slot,…}`
starts with `...` and must be left untouched — a naive "insert before the closing brace"
walk corrupts it.

注意：只有值以真实键开头（`node_slot:{BOOLEAN:…`）的 6 张表才是主题调色板；调色板合并表达式
`node_slot:{...t.colors.node_slot,…}` 以 `...` 开头，必须跳过——盲目地"在右花括号前插入"
会破坏它。

---

## 4. Type Merge (类型归并)

| Old ComfyDL name | New name | Kind | Notes |
|---|---|---|---|
| `cdlTensor` | `TENSOR` | merged | ComfyUI core type introduced in this step |
| `cdlBbox` | `BBOX` | merged | core `IO.BBOX` already existed and had zero users |
| `cdlModel` | `cdlModel` | kept | no core counterpart (core `MODEL` is a diffusion model) |
| `cdlVocab` | `cdlVocab` | kept | no core counterpart |
| `cdlDataloader` | `cdlDataloader` | kept | no core counterpart |

Implementation:

- `comfydl/nodes/__init__.py` now defines `TENSOR = "TENSOR"` and `BBOX = "BBOX"` (canonical)
  and re-exports `cdlTensor = TENSOR` / `cdlBbox = BBOX` as **legacy aliases**, so any
  external code importing the old constants keeps working.
- 98 literal occurrences across 11 files under `comfydl/nodes/*.py` were rewritten
  (`("cdlTensor",)` → `("TENSOR",)`); `FUNCTIONS.md` / `FUNCTIONS_zh.md` were rewritten the
  same way (97 occurrences each).
- `comfydl/_update_nodes.py` records the rename (`"cdlTensor" → "TENSOR"`,
  `"cdlBbox" → "BBOX"`), so the migration script stays a complete, re-runnable record.
- The node files never imported the constants (they used the string literals directly), so
  the change is purely textual and has no runtime cost.

**Old workflow JSONs**: the `example_workflows/*.json` files still contain the old
`cdlTensor` type name. Loading such a workflow in the frontend shows those links as
type-mismatched; re-dragging the link fixes it. Backend execution does not validate slot
types, so graphs submitted via `/prompt` still run.

Follow-up (2026-09-11): the stored type strings were rewritten in place (`"cdlTensor"` →
`"TENSOR"`, `"cdlBbox"` → `"BBOX"`) across all `example_workflows/*.json`, and the generator
(`_gen_full_test.py`) plus the re-runnable migration record (`_update_nodes.py`) were updated so
regenerated workflows no longer write the legacy names. Node ids, link ids and widget values are
untouched.

后续（2026-09-11）：已就地重写所有 `example_workflows/*.json` 中的类型字符串
（`"cdlTensor"` → `"TENSOR"`、`"cdlBbox"` → `"BBOX"`），并同步生成脚本 `_gen_full_test.py`
与可重放迁移记录 `_update_nodes.py`，避免重新生成时写回旧名；节点 id、连线 id 与控件值均未改动。

**旧工作流 JSON**：`example_workflows/*.json` 中仍写着旧的 `cdlTensor` 类型名，前端加载这类
工作流时这些连线会显示为类型不匹配，重新拖一次连线即可；后端执行不做插槽类型校验，因此通过
`/prompt` 提交仍能正常运行。

---

## 5. Relation to `CdlActivation` (与 `CdlActivation` 的关系)

ComfyDL already ships `CdlActivation` (in `d2l/Tensor Basic`), whose activation choice is a
COMBO widget. The 14 new core nodes overlap with it functionally but differ in shape: one
function per node, one input/one output, no widget-based dispatch. `CdlActivation` was left
untouched in this step; it has since been **soft-archived**: the node id is unchanged and old
workflows still load, but its display name now carries a `(DEPRECATED)` suffix, it lives in
`d2l/_Legacy/Tensor Basic`, and its docstring points to the 14 core activation nodes.

ComfyDL 原有的 `CdlActivation`（位于 `d2l/Tensor Basic`）用 COMBO 控件选择激活函数；14 个新
核心节点与其功能重叠但形态不同（一函数一节点、单入单出、不用控件派发）。本次**未**改动；
后续已做**软归档**：节点 id 不变、旧工作流照常加载，但显示名加 `(DEPRECATED)` 后缀、归入
`d2l/_Legacy/Tensor Basic`，docstring 指向 14 个核心激活节点。

---

## 6. Verification (验证记录)

- Node level: all 14 schemas and numerics checked locally — widget defaults are inside their
  `[min, max]` ranges / option lists, outputs keep shape/dtype/device, `Identity` is a true
  zero-copy pass-through, `Softmax` matches `torch.nn.functional.softmax` and clamps
  out-of-range `dim`, and fp16 inputs stay fp16.
- Host level: `init_extra_nodes(init_api_nodes=False, init_custom_nodes=False)` reports
  `IMPORT_FAILED []`, `CDL 102`, `ACTIVATION 14`; the `Activation` category exists.
- ComfyDL level: the package still registers 102 nodes in 16 categories, 28 node classes
  return `TENSOR` and **0** still return `cdlTensor`.
- HTTP level (`/object_info` on a `--cpu` server): the 14 nodes report
  `python_module = comfy_extras.nodes_activation` — the first segment hits the frontend
  whitelist `['nodes','comfy_extras','comfy_api_nodes']`, so they are classified as
  **Comfy nodes** — with `category = "Activation"`; every node takes and returns `TENSOR`;
  the four widgets expose exactly the documented defaults/ranges; 28 ComfyDL nodes return
  `TENSOR` and no node still references `cdlTensor` on either side of the slot.
- End-to-end: the graph `CdlSyntheticData → ActivationSigmoid → ActivationSoftmax →
  CdlShowHeatmapsOutput` submitted to `/prompt` comes back with `node_errors {}` and finishes
  with `execution_success`, i.e. a ComfyDL tensor flows through the new core activation
  nodes into an output node without any adapter.
- Frontend bundle: after the package reinstall the `node_slot` patch was re-applied at byte
  level; the bundle grew from 1645581 to 1645683 chars (`6 × 17`), `node_slot:{` still occurs
  7 times (6 theme tables + 1 palette-merge expression) and the page loads normally.

---

## 7. Rollback (回退)

1. Delete `comfy_extras/nodes_activation.py` and remove `"nodes_activation.py"` from the
   `extras_files` list in `nodes.py`.
2. Remove `TENSOR` from the `IO` enum (`comfy/comfy_types/node_typing.py`) and the
   `@comfytype(io_type="TENSOR") class Tensor` + `"Tensor"` from `comfy_api/latest/_io.py`.
3. Revert the type merge by running `comfydl/_update_nodes.py` with the mapping reversed
   (`"TENSOR" → "cdlTensor"`, `"BBOX" → "cdlBbox"`), and restore the `cdlTensor` / `cdlBbox`
   definitions in `comfydl/nodes/__init__.py`.
4. Optionally restore the previous `settingStore-*.js` (or re-install
   `comfyui-frontend-package==1.51.9`) to drop the slot colour.
5. `git revert` the reform commit(s) in `ComfyDL_UI`, and the corresponding `comfydl`
   submodule commit.

Nothing in this step touches the `comfy/` registration chain beyond adding one type, so a
revert is a plain source revert with no data or state to migrate.
