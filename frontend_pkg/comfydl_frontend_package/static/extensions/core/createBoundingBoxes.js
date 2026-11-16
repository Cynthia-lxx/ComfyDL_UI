// Shim for extensions/core/createBoundingBoxes.ts
console.warn('[ComfyUI Notice] "extensions/core/createBoundingBoxes.js" is an internal module, not part of the public API. Future updates may break this import.');
export const createBoundingBoxesExtension = window.comfyAPI.createBoundingBoxes.createBoundingBoxesExtension;
