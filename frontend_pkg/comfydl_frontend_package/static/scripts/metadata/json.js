// Shim for scripts/metadata/json.ts
console.warn('[ComfyUI Notice] "scripts/metadata/json.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getDataFromJSON = window.comfyAPI.json.getDataFromJSON;
