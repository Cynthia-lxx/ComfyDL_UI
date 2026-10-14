// Shim for scripts/widgets.ts
console.warn('[ComfyUI Notice] "scripts/widgets.js" is an internal module, not part of the public API. Future updates may break this import.');
export const updateControlWidgetLabel = window.comfyAPI.widgets.updateControlWidgetLabel;
export const addValueControlWidget = window.comfyAPI.widgets.addValueControlWidget;
export const addValueControlWidgets = window.comfyAPI.widgets.addValueControlWidgets;
export const ComfyWidgets = window.comfyAPI.widgets.ComfyWidgets;
export const isValidWidgetType = window.comfyAPI.widgets.isValidWidgetType;
