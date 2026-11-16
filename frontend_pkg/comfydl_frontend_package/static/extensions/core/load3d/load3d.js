// Shim for extensions/core/load3d/load3d.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/load3d.js" is an internal module, not part of the public API. Future updates may break this import.');
export const load3dExtension = window.comfyAPI.load3d.load3dExtension;
export const preview3DExtension = window.comfyAPI.load3d.preview3DExtension;
export const preview3DAdvancedExtension = window.comfyAPI.load3d.preview3DAdvancedExtension;
export const save3DAdvancedExtension = window.comfyAPI.load3d.save3DAdvancedExtension;
