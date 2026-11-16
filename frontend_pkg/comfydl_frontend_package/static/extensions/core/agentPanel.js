// Shim for extensions/core/agentPanel.ts
console.warn('[ComfyUI Notice] "extensions/core/agentPanel.js" is an internal module, not part of the public API. Future updates may break this import.');
export const registerAgentPanelExtension = window.comfyAPI.agentPanel.registerAgentPanelExtension;
