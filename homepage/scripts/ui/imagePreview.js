// Shim for scripts/ui/imagePreview.ts
console.warn('[ComfyUI Deprecated] Importing from "scripts/ui/imagePreview.js" is deprecated and will be removed in v1.34.');
export const calculateImageGrid = window.comfyAPI.imagePreview.calculateImageGrid;
export const createImageHost = window.comfyAPI.imagePreview.createImageHost;
