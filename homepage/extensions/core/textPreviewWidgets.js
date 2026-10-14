// Shim for extensions/core/textPreviewWidgets.ts
console.warn('[ComfyUI Notice] "extensions/core/textPreviewWidgets.js" is an internal module, not part of the public API. Future updates may break this import.');
export const addTextPreviewWidgets = window.comfyAPI.textPreviewWidgets.addTextPreviewWidgets;
export const updateTextPreviewWidgets = window.comfyAPI.textPreviewWidgets.updateTextPreviewWidgets;
