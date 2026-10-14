// Shim for extensions/core/widgetInputs.ts
console.warn('[ComfyUI Notice] "extensions/core/widgetInputs.js" is an internal module, not part of the public API. Future updates may break this import.');
export const PrimitiveNode = window.comfyAPI.widgetInputs.PrimitiveNode;
export const getWidgetConfig = window.comfyAPI.widgetInputs.getWidgetConfig;
export const convertToInput = window.comfyAPI.widgetInputs.convertToInput;
export const setWidgetConfig = window.comfyAPI.widgetInputs.setWidgetConfig;
export const mergeIfValid = window.comfyAPI.widgetInputs.mergeIfValid;
