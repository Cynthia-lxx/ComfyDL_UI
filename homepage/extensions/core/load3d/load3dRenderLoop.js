// Shim for extensions/core/load3d/load3dRenderLoop.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/load3dRenderLoop.js" is an internal module, not part of the public API. Future updates may break this import.');
export const startRenderLoop = window.comfyAPI.load3dRenderLoop.startRenderLoop;
