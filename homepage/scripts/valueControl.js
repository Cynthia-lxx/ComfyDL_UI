// Shim for scripts/valueControl.ts
console.warn('[ComfyUI Notice] "scripts/valueControl.js" is an internal module, not part of the public API. Future updates may break this import.');
export const nextValueForLinkedTarget = window.comfyAPI.valueControl.nextValueForLinkedTarget;
export const isValueControlWidget = window.comfyAPI.valueControl.isValueControlWidget;
export const computeNextControlledValue = window.comfyAPI.valueControl.computeNextControlledValue;
