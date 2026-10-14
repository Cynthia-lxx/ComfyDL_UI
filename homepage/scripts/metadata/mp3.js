// Shim for scripts/metadata/mp3.ts
console.warn('[ComfyUI Notice] "scripts/metadata/mp3.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getMp3Metadata = window.comfyAPI.mp3.getMp3Metadata;
