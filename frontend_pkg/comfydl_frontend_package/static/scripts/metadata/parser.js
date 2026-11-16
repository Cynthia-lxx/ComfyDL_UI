// Shim for scripts/metadata/parser.ts
console.warn('[ComfyUI Notice] "scripts/metadata/parser.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getWorkflowDataFromFile = window.comfyAPI.parser.getWorkflowDataFromFile;
