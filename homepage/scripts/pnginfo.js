// Shim for scripts/pnginfo.ts
console.warn('[ComfyUI Notice] "scripts/pnginfo.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getPngMetadata = window.comfyAPI.pnginfo.getPngMetadata;
export const getFlacMetadata = window.comfyAPI.pnginfo.getFlacMetadata;
export const getAvifMetadata = window.comfyAPI.pnginfo.getAvifMetadata;
export const getWebpMetadata = window.comfyAPI.pnginfo.getWebpMetadata;
export const getLatentMetadata = window.comfyAPI.pnginfo.getLatentMetadata;
export const importA1111 = window.comfyAPI.pnginfo.importA1111;
