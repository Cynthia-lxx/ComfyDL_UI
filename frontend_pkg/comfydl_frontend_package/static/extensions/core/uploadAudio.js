// Shim for extensions/core/uploadAudio.ts
console.warn('[ComfyUI Notice] "extensions/core/uploadAudio.js" is an internal module, not part of the public API. Future updates may break this import.');
export const audioWidgetExtension = window.comfyAPI.uploadAudio.audioWidgetExtension;
export const uploadAudioExtension = window.comfyAPI.uploadAudio.uploadAudioExtension;
export const recordAudioExtension = window.comfyAPI.uploadAudio.recordAudioExtension;
