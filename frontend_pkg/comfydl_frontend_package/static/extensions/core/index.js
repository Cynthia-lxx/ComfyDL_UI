// Shim for extensions/core/index.ts
console.warn('[ComfyUI Notice] "extensions/core/index.js" is an internal module, not part of the public API. Future updates may break this import.');
export const registerCoreExtensions = window.comfyAPI.index.registerCoreExtensions;
