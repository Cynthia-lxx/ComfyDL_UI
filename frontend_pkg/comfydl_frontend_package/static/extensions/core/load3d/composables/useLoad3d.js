// Shim for extensions/core/load3d/composables/useLoad3d.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/composables/useLoad3d.js" is an internal module, not part of the public API. Future updates may break this import.');
export const nodeToLoad3dMap = window.comfyAPI.useLoad3d.nodeToLoad3dMap;
export const markLoad3dSceneDirty = window.comfyAPI.useLoad3d.markLoad3dSceneDirty;
export const getLoad3dSceneRevision = window.comfyAPI.useLoad3d.getLoad3dSceneRevision;
export const isLoad3dSceneDirty = window.comfyAPI.useLoad3d.isLoad3dSceneDirty;
export const getLoad3dOutputCache = window.comfyAPI.useLoad3d.getLoad3dOutputCache;
export const setLoad3dOutputCache = window.comfyAPI.useLoad3d.setLoad3dOutputCache;
export const useLoad3d = window.comfyAPI.useLoad3d.useLoad3d;
