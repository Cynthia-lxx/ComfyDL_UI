// Shim for extensions/core/touchCanvasAccess.ts
console.warn('[ComfyUI Notice] "extensions/core/touchCanvasAccess.js" is an internal module, not part of the public API. Future updates may break this import.');
export const canvasOrReport = window.comfyAPI.touchCanvasAccess.canvasOrReport;
