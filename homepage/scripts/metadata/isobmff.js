// Shim for scripts/metadata/isobmff.ts
console.warn('[ComfyUI Notice] "scripts/metadata/isobmff.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getFromIsobmffFile = window.comfyAPI.isobmff.getFromIsobmffFile;
