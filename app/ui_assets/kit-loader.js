/**
 * ComfyDL UI-kit shared component loader (U stick 3 [0.90]).
 *
 * Every kit component in ui_kit/dist follows the same path convention:
 *   /comfydl/ui/components/<id>/<id>.js
 *
 * ensureComponents(ids) imports each component exactly once, de-duplicates
 * by customElements registry, and logs every step to the console (VERBOSE
 * rule). A missing kit must never stop a panel from working: panels keep
 * their plain-HTML fallback when an import fails.
 */

const COMPONENT_BASE = "/comfydl/ui/components";
const TAG = "[ComfyDL kit-loader]";
const pending = new Map();

function normalizeId(raw) {
  const id = String(raw || "").trim();
  return id.startsWith("ui-") ? id : "ui-" + id;
}

export function componentUrl(id) {
  const normalized = normalizeId(id);
  return `${COMPONENT_BASE}/${normalized}/${normalized}.js`;
}

export async function ensureComponents(ids) {
  const ready = new Set();
  for (const raw of ids || []) {
    const id = normalizeId(raw);
    if (customElements.get(id)) {
      ready.add(id);
      continue;
    }
    if (!pending.has(id)) {
      const url = componentUrl(id);
      pending.set(id, (async () => {
        try {
          await import(/* @vite-ignore */ url);
          const ok = !!customElements.get(id);
          console.log(TAG, id, ok ? "registered" : "module loaded but NOT registered");
          return ok;
        } catch (e) {
          console.warn(TAG, id, "import FAILED (panel degrades):", String(e));
          return false;
        }
      })());
    }
    if (await pending.get(id)) ready.add(id);
  }
  console.log(TAG, "ready:", [...ready].join(", ") || "(none)");
  return ready;
}

export default ensureComponents;
