// Shim for extensions/core/load3d/nodeTypes.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/nodeTypes.js" is an internal module, not part of the public API. Future updates may break this import.');
export const isLoad3dResultViewerNode = window.comfyAPI.nodeTypes.isLoad3dResultViewerNode;
export const isLoad3dNode = window.comfyAPI.nodeTypes.isLoad3dNode;
export const isThreeJsNode = window.comfyAPI.nodeTypes.isThreeJsNode;
