// Shim for extensions/core/load3d/ModelAdapter.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/ModelAdapter.js" is an internal module, not part of the public API. Future updates may break this import.');
export const DEFAULT_MODEL_CAPABILITIES = window.comfyAPI.ModelAdapter.DEFAULT_MODEL_CAPABILITIES;
export const createAdapterRef = window.comfyAPI.ModelAdapter.createAdapterRef;
export const fetchModelData = window.comfyAPI.ModelAdapter.fetchModelData;
