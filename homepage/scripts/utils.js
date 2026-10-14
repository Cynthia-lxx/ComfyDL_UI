// Shim for scripts/utils.ts
console.warn('[ComfyUI Notice] "scripts/utils.js" is an internal module, not part of the public API. Future updates may break this import.');
export const clone = window.comfyAPI.utils.clone;
export const applyTextReplacements = window.comfyAPI.utils.applyTextReplacements;
export const addStylesheet = window.comfyAPI.utils.addStylesheet;
export const uploadFile = window.comfyAPI.utils.uploadFile;
export const prop = window.comfyAPI.utils.prop;
export const getStorageValue = window.comfyAPI.utils.getStorageValue;
export const setStorageValue = window.comfyAPI.utils.setStorageValue;
