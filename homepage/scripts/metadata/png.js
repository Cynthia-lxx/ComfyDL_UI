// Shim for scripts/metadata/png.ts
console.warn('[ComfyUI Notice] "scripts/metadata/png.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getFromPngBuffer = window.comfyAPI.png.getFromPngBuffer;
export const getFromPngFile = window.comfyAPI.png.getFromPngFile;
