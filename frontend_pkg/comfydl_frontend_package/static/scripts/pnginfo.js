// Shim for scripts/pnginfo.ts
console.warn('[ComfyUI Notice] "scripts/pnginfo.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getAvifMetadata = window.comfyAPI.pnginfo.getAvifMetadata;
export const getFlacMetadata = window.comfyAPI.pnginfo.getFlacMetadata;
export const getLatentMetadata = window.comfyAPI.pnginfo.getLatentMetadata;
export const getPngMetadata = window.comfyAPI.pnginfo.getPngMetadata;
export const getWebpMetadata = window.comfyAPI.pnginfo.getWebpMetadata;
export const importA1111 = window.comfyAPI.pnginfo.importA1111;
