// Shim for extensions/core/load3d/cameraAngle/widgetBridge.ts
console.warn('[ComfyUI Notice] "extensions/core/load3d/cameraAngle/widgetBridge.js" is an internal module, not part of the public API. Future updates may break this import.');
export const readStateFromWidgets = window.comfyAPI.widgetBridge.readStateFromWidgets;
export const writeStateToWidgets = window.comfyAPI.widgetBridge.writeStateToWidgets;
export const isViewMode = window.comfyAPI.widgetBridge.isViewMode;
export const readViewMode = window.comfyAPI.widgetBridge.readViewMode;
export const writeViewMode = window.comfyAPI.widgetBridge.writeViewMode;
