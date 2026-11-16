# ComfyDL fork of the ComfyUI frontend static package.
#
# Base: comfyui-frontend-package 1.59.2 (upstream dist output, vendored
# verbatim, source maps stripped).  Our UI work keeps living in
# app/frontend_patch.py (content-marker patches applied to the runtime copy
# at startup); bake changes into this fork only when a patch is genuinely
# impossible, so upstream syncs stay clean.
__version__ = "1.59.2-cdl.1"
