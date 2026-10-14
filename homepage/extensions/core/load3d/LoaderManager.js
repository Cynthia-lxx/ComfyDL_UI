// Shim for extensions/core/load3d/LoaderManager.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/LoaderManager.js" is an internal module, not part of the public API. Future updates may break this import.');
export const LoaderManager = window.comfyAPI.LoaderManager.LoaderManager;
