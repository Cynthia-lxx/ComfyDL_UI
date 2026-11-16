// Shim for extensions/core/touchPointerGuards.ts
console.warn('[ComfyUI Notice] "extensions/core/touchPointerGuards.js" is an internal module, not part of the public API. Future updates may break this import.');
export const touchGesture = window.comfyAPI.touchPointerGuards.touchGesture;
export const installTouchPointerGuards = window.comfyAPI.touchPointerGuards.installTouchPointerGuards;
