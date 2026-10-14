// Shim for extensions/core/editAttention.ts
console.warn('[ComfyUI Notice] "extensions/core/editAttention.js" is an internal module, not part of the public API. Future updates may break this import.');
export const incrementWeight = window.comfyAPI.editAttention.incrementWeight;
export const findNearestEnclosure = window.comfyAPI.editAttention.findNearestEnclosure;
export const addWeightToParentheses = window.comfyAPI.editAttention.addWeightToParentheses;
