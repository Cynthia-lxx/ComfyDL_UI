// Shim for scripts/domWidget.ts
console.warn('[ComfyUI Notice] "scripts/domWidget.js" is an internal module, not part of the public API. Future updates may break this import.');
export const isDOMWidget = window.comfyAPI.domWidget.isDOMWidget;
export const isComponentWidget = window.comfyAPI.domWidget.isComponentWidget;
export const DOMWidgetImpl = window.comfyAPI.domWidget.DOMWidgetImpl;
export const ComponentWidgetImpl = window.comfyAPI.domWidget.ComponentWidgetImpl;
export const addWidget = window.comfyAPI.domWidget.addWidget;
