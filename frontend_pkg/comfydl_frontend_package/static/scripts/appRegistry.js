// Shim for scripts/appRegistry.ts
console.warn('[ComfyUI Notice] "scripts/appRegistry.js" is an internal module, not part of the public API. Future updates may break this import.');
export const registerApp = window.comfyAPI.appRegistry.registerApp;
export const useApp = window.comfyAPI.appRegistry.useApp;
