// Shim for scripts/metadata/ply.ts
console.warn('[ComfyUI Notice] "scripts/metadata/ply.js" is an internal module, not part of the public API. Future updates may break this import.');
export const parseASCIIPLY = window.comfyAPI.ply.parseASCIIPLY;
export const isPLYAsciiFormat = window.comfyAPI.ply.isPLYAsciiFormat;
export const isGaussianSplatPLY = window.comfyAPI.ply.isGaussianSplatPLY;
