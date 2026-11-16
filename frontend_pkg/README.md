# comfydl-frontend-package

ComfyDL's own fork of the ComfyUI frontend static package
(base: `comfyui-frontend-package` 1.59.2, upstream dist output vendored
verbatim, source maps stripped to keep the repo lean).

## Why

- The UI overhaul (U-series) needs deeper frontend integration than
  content-marker patching can express; this fork is the "ultimate form"
  landing spot the project can adapt freely.
- `app/frontend_management.py` prefers this fork: it serves a runtime copy
  under `user/frontend_runtime/<version>/` (gitignored) so
  `apply_frontend_patches` keeps editing the runtime copy, never the repo.
- The pip package `comfyui-frontend-package` stays installed as a fallback
  and as the test harness target.

## Syncing from upstream

1. Update the pinned upstream version in `requirements.txt` and install it
   into a venv.
2. Robocopy its `static/` over `comfydl_frontend_package/static/` with
   `/XF *.map /XD __pycache__`.
3. Bump `version` in `pyproject.toml` and `__version__` in
   `comfydl_frontend_package/__init__.py` (base version + `-cdl.N` suffix),
   bump `FORK_VERSION` in `app/frontend_management.py` to match, commit.
