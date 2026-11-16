// Shim for scripts/apiTimeouts.ts
console.warn('[ComfyUI Notice] "scripts/apiTimeouts.js" is an internal module, not part of the public API. Future updates may break this import.');
export const AUTH_INIT_TIMEOUT_MS = window.comfyAPI.apiTimeouts.AUTH_INIT_TIMEOUT_MS;
export const FETCH_RESPONSE_HEADERS_TIMEOUT_MS = window.comfyAPI.apiTimeouts.FETCH_RESPONSE_HEADERS_TIMEOUT_MS;
