// Shim for scripts/metadata/flac.ts
console.warn('[ComfyUI Notice] "scripts/metadata/flac.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getFromFlacBuffer = window.comfyAPI.flac.getFromFlacBuffer;
export const getFromFlacFile = window.comfyAPI.flac.getFromFlacFile;
