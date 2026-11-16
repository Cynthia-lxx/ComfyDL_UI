// Shim for scripts/metadata/gltf.ts
console.warn('[ComfyUI Notice] "scripts/metadata/gltf.js" is an internal module, not part of the public API. Future updates may break this import.');
export const getGltfBinaryMetadata = window.comfyAPI.gltf.getGltfBinaryMetadata;
