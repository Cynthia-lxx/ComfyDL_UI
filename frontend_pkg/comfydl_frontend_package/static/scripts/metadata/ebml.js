// Shim for scripts/metadata/ebml.ts
console.warn('[ComfyUI Notice] "scripts/metadata/ebml.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getFromWebmFile = window.comfyAPI.ebml.getFromWebmFile;
