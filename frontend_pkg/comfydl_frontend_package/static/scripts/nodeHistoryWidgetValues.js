// Shim for scripts/nodeHistoryWidgetValues.ts
console.warn('[ComfyUI Notice] "scripts/nodeHistoryWidgetValues.js" is an internal module, not part of the public API. Future updates may break this import.');
export const collectNodeHistoryWidgets = window.comfyAPI.nodeHistoryWidgetValues.collectNodeHistoryWidgets;
export const nodeHistoryValuesChanged = window.comfyAPI.nodeHistoryWidgetValues.nodeHistoryValuesChanged;
export const withNodeHistoryValuesFrom = window.comfyAPI.nodeHistoryWidgetValues.withNodeHistoryValuesFrom;
