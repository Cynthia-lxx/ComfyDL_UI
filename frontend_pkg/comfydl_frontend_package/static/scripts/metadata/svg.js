// Shim for scripts/metadata/svg.ts
console.warn('[ComfyUI Notice] "scripts/metadata/svg.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getSvgMetadata = window.comfyAPI.svg.getSvgMetadata;
