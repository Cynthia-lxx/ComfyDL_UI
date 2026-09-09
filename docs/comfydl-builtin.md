# ComfyDL as a Built-in Module (ComfyDL 内置化)

> Experimental branch: `experiment/embed-comfydl`. All migration changes live on this
> branch; `master` is untouched.
>
> 实验分支：`experiment/embed-comfydl`。全部迁移改动均在该分支上，`master` 不受影响。

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
  frontend shows the nodes automatically under their existing `ComfyDL/*` categories.
- Each node class gets `RELATIVE_PYTHON_MODULE` set to its real module
  (e.g. `comfydl.nodes.tensor_ops`).
- If the submodule is missing on a fresh clone, it logs a warning with the fix
  command (`git submodule update --init`) and continues without crashing.

`nodes.py` 新增 `init_builtin_dl_nodes()`，在 `init_extra_nodes()` 中紧随
`init_builtin_extra_nodes()` 执行。它把 `comfydl` 作为真实顶层包导入，并把其节点映射
合并进核心注册表 dict（即 `/object_info` 所读取的同一份 dict），因此前端在既有
`ComfyDL/*` 分类下自动可见；若 submodule 缺失则告警提示且不崩溃。

Verification baseline (验证基线)：`CORE_BEFORE=11` core classes →
`CORE_AFTER=117` (11 core + 106 ComfyDL, key prefix `Cdl*`, 14 categories,
no IMPORT FAILED).

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

Expected: `IMPORT_FAILED []`, `CDL_COUNT 106`, startup banner
`[ComfyDL] 已注册 106 个节点（显示名 106 个），共 14 个分类` printed once.

A full `python main.py --cpu` boot also works and logs the banner; optional full HTTP
check: `GET /object_info/CdlAccuracy`.

## Rollback (回退)

All migration commits live only on `experiment/embed-comfydl`. To discard the
experiment entirely:

```
git checkout master
git branch -D experiment/embed-comfydl      # local
git push origin --delete experiment/embed-comfydl   # remote backup
```

Deleting `comfydl/` afterwards also removes the submodule working tree (the upstream
ComfyDL repo is unaffected).

## Related Commits (相关提交)

On `experiment/embed-comfydl`:

- `feat(builtin): track ComfyDL as git submodule under comfydl/`
- `feat(builtin): register comfydl into core node registry via init_builtin_dl_nodes`
- `chore(builtin): bump comfydl submodule to adc3c81 (GBK-safe startup banner)`

In ComfyDL repo (`main`), backported before migration:

- `fix(import): make plugin root importable under synthetic module names`
- `fix(deps): record IPython and matplotlib-inline runtime requirements`
- `fix(print): drop emoji from startup banner for GBK console safety`
