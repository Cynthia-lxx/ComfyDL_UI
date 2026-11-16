// Shim for scripts/clipspace.ts
console.warn('[ComfyUI Notice] "scripts/clipspace.js" is an internal module, not part of the public API. Future updates may break this import.');
export const clipspace = window.comfyAPI.clipspace.clipspace;
export const copyToClipspace = window.comfyAPI.clipspace.copyToClipspace;
export const pasteFromClipspace = window.comfyAPI.clipspace.pasteFromClipspace;
