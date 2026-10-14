// Shim for extensions/core/load3d/load3dViewport.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/load3dViewport.js" is an internal module, not part of the public API. Future updates may break this import.');
export const computeLetterboxedViewport = window.comfyAPI.load3dViewport.computeLetterboxedViewport;
export const clientPointToLetterboxNdc = window.comfyAPI.load3dViewport.clientPointToLetterboxNdc;
export const computeLetterboxBars = window.comfyAPI.load3dViewport.computeLetterboxBars;
export const isLoad3dActive = window.comfyAPI.load3dViewport.isLoad3dActive;
