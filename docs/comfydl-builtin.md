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
`CORE_AFTER=113` (11 core + 102 ComfyDL, key prefix `Cdl*`, 16 categories,
no IMPORT FAILED).

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
