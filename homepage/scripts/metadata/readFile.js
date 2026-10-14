// Shim for scripts/metadata/readFile.ts
console.warn('[ComfyUI Notice] "scripts/metadata/readFile.js" is an internal module, not part of the public API. Future updates may break this import.');
export const readFileAsArrayBuffer = window.comfyAPI.readFile.readFileAsArrayBuffer;
