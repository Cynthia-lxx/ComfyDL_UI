// Shim for extensions/core/load3d/load3dInstances.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/load3dInstances.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getNodeToLoad3dMapSync = window.comfyAPI.load3dInstances.getNodeToLoad3dMapSync;
export const cacheNodeToLoad3dMap = window.comfyAPI.load3dInstances.cacheNodeToLoad3dMap;
export const getNodeByLoad3d = window.comfyAPI.load3dInstances.getNodeByLoad3d;
export const removeLoad3d = window.comfyAPI.load3dInstances.removeLoad3d;
export const copyLoad3dState = window.comfyAPI.load3dInstances.copyLoad3dState;
export const handleViewportRefresh = window.comfyAPI.load3dInstances.handleViewportRefresh;
