// Shim for scripts/ui.ts
console.warn('[ComfyUI Deprecated] Importing from "scripts/ui.js" is deprecated and will be removed in v1.34.');
export const ComfyDialog = window.comfyAPI.ui.ComfyDialog;
export const $el = window.comfyAPI.ui.$el;
export const ComfyUI = window.comfyAPI.ui.ComfyUI;
