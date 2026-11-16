// Shim for extensions/core/electronAdapter.ts
console.warn('[ComfyUI Notice] "extensions/core/electronAdapter.js" is an internal module, not part of the public API. Future updates may break this import.');
export const registerElectronAdapterExtension = window.comfyAPI.electronAdapter.registerElectronAdapterExtension;
