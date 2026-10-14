// Shim for scripts/changeTracker.ts
console.warn('[ComfyUI Notice] "scripts/changeTracker.js" is an internal module, not part of the public API. Future updates may break this import.');
export const ChangeTracker = window.comfyAPI.changeTracker.ChangeTracker;
