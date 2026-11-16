// Shim for scripts/metadata/latent.ts
console.warn('[ComfyUI Notice] "scripts/metadata/latent.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getFromLatentFile = window.comfyAPI.latent.getFromLatentFile;
