// Shim for scripts/changeTrackerAccess.ts
console.warn('[ComfyUI Notice] "scripts/changeTrackerAccess.js" is an internal module, not part of the public API. Future updates may break this import.');
export const provideChangeTracker = window.comfyAPI.changeTrackerAccess.provideChangeTracker;
