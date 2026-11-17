/**
 * ComfyDL UI-kit semantic colours (U stick 3 [0.90], B8 data->colour rule).
 *
 * Every verdict / run-status / highlight colour used by our panels comes
 * from ONE lookup here. JavaScript consumers call verdictColor(kind) /
 * statusColor(state) (returns a CSS var reference); CSS consumers use the
 * matching `cdl-sem-*` classes or --cdl-* variables declared in bridge.css.
 * The literal values therefore exist in exactly one place (bridge.css),
 * so a theme or palette change never leaves a stale hardcoded hex behind.
 */

const TAG = "[ComfyDL kit-semantics]";

const VERDICT_VARS = {
  ok: "--cdl-verdict-ok",
  pass: "--cdl-verdict-ok",
  green: "--cdl-verdict-ok",
  warn: "--cdl-verdict-warn",
  warning: "--cdl-verdict-warn",
  yellow: "--cdl-verdict-warn",
  danger: "--cdl-verdict-danger",
  fail: "--cdl-verdict-danger",
  failed: "--cdl-verdict-danger",
  red: "--cdl-verdict-danger",
  unknown: "--cdl-verdict-unknown",
  gray: "--cdl-verdict-unknown",
};

const STATUS_VARS = {
  active: "--cdl-status-active",
  on: "--cdl-status-active",
  ran: "--cdl-status-ran",
  completed: "--cdl-status-ran",
  cached: "--cdl-status-cached",
  blocked: "--cdl-status-blocked",
  failed: "--cdl-status-failed",
  new: "--cdl-status-new",
  chosen: "--cdl-chosen",
};

const SEMANTIC_CLASSES = {
  ok: "cdl-sem-ok", warn: "cdl-sem-warn", danger: "cdl-sem-danger",
  unknown: "cdl-sem-unknown", active: "cdl-sem-active", ran: "cdl-sem-ran",
  cached: "cdl-sem-cached", blocked: "cdl-sem-blocked",
  failed: "cdl-sem-failed", new: "cdl-sem-new", chosen: "cdl-sem-chosen",
};

/** Canonical semantic key for a verdict word (unknown words -> "unknown"). */
export function verdictKey(kind) {
  const key = String(kind || "").toLowerCase();
  return VERDICT_VARS[key] ? key : "unknown";
}

/** CSS var reference for a verdict, e.g. "var(--cdl-verdict-ok)". */
export function verdictColor(kind) {
  return `var(${VERDICT_VARS[verdictKey(kind)]})`;
}

/** CSS var reference for a run/status state. */
export function statusColor(state) {
  const key = String(state || "").toLowerCase();
  return `var(${STATUS_VARS[key] || "--cdl-status-active"})`;
}

/** Class that paints text with the matching semantic colour. */
export function semanticClass(kind) {
  const key = String(kind || "").toLowerCase();
  const canonical = VERDICT_VARS[key] || STATUS_VARS[key] ? key : "unknown";
  return SEMANTIC_CLASSES[canonical] || "cdl-sem-unknown";
}

/**
 * Ensure the --cdl-* values exist. bridge.css normally supplies them; this
 * fallback keeps semantics intact if the stylesheet was not injected (e.g.
 * a standalone panel test page).
 */
export function applySemanticVars(root = document.documentElement) {
  const current = getComputedStyle(root).getPropertyValue("--cdl-verdict-ok").trim();
  if (current) return false;
  const style = document.createElement("style");
  style.setAttribute("data-cdl-semantics", "1");
  style.textContent = `:root{
    --cdl-verdict-ok:#4ade80;--cdl-verdict-warn:#fbbf24;
    --cdl-verdict-danger:#f87171;--cdl-verdict-unknown:#9a9a9a;
    --cdl-status-active:#4a9eff;--cdl-status-ran:#3ddc97;
    --cdl-status-cached:#4a9eff;--cdl-status-blocked:#e5484d;
    --cdl-status-failed:#e5484d;--cdl-status-new:#f5a524;
    --cdl-chosen:#3fb950;
  }`;
  document.head.appendChild(style);
  console.log(TAG, "injected fallback semantic vars (bridge.css not loaded)");
  return true;
}

/** Attach the semantic colour + breathing pulse to a verdict/status badge. */
export function markStatus(el, kind, pulse = false) {
  if (!el) return el;
  el.classList.add(semanticClass(kind));
  if (pulse) el.classList.add("cdl-vfx-pulse");
  return el;
}

/** Attach the entrance-timeline class; motion gates live in bridge.css. */
export function markEntrance(el) {
  if (el) el.classList.add("cdl-vfx-entrance");
  return el;
}

applySemanticVars();
