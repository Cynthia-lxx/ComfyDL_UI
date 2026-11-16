// Shim for scripts/defaultGraph.ts
console.warn('[ComfyUI Notice] "scripts/defaultGraph.js" is an internal module, not part of the public API. Future updates may break this import.');
export const defaultGraph = window.comfyAPI.defaultGraph.defaultGraph;
export const defaultGraphJSON = window.comfyAPI.defaultGraph.defaultGraphJSON;
export const blankGraph = window.comfyAPI.defaultGraph.blankGraph;
