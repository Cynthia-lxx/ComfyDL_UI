/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * JS helpers for FlashColour / FadeEdgeEffectTo / additive layers, translated from:
 *   Graphics/UserInterface/OsuAnimatedButton.cs:119, Nub.cs:64-104,
 *   Overlays/Dialog/PopupDialog.cs:294-300, Screens/Play/SkipOverlay.cs:430
 * Uses the Web Animations API; all durations respect the global motion scale.
 */

import { readCssColour, isReducedMotion } from '../particles/particles.js';

function motionScale() {
  const value = Number(getComputedStyle(document.documentElement).getPropertyValue('--motion-scale'));
  return Number.isFinite(value) ? value : 1;
}

/** FlashColour equivalent: a short colour/brightness flash on an element. */
export function flashColour(element, colour, duration = 500) {
  if (!element || isReducedMotion()) return null;
  const resolved = colour || readCssColour(element, '--colour-1');
  const base = getComputedStyle(element).backgroundColor;
  return element.animate(
    [
      { backgroundColor: resolved, offset: 0 },
      { backgroundColor: base, offset: 1 },
    ],
    {
      duration: Math.max(1, duration * motionScale()),
      easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
    },
  );
}

/** FadeEdgeEffectTo equivalent: two-phase glow (fast in, slow out). */
export function pulseGlow(element, duration = 800) {
  if (!element || isReducedMotion()) return null;
  const colour = readCssColour(element, '--colour-1');
  const strong = `0 0 16px 4px ${colour}`;
  const weak = `0 0 4px 0 ${colour}`;
  return element.animate(
    [
      { boxShadow: weak, offset: 0 },
      { boxShadow: strong, offset: 0.1 },
      { boxShadow: weak, offset: 1 },
    ],
    {
      duration: Math.max(1, duration * motionScale()),
      easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
    },
  );
}

export function setAdditive(element, enabled = true) {
  if (!element) return;
  element.classList.toggle('ui-additive', Boolean(enabled));
}

export function attachAdditiveFlash(element, duration = 300) {
  if (!element || isReducedMotion()) return null;
  const layer = document.createElement('div');
  layer.className = 'ui-additive-flash';
  layer.style.animationDuration = `${Math.max(1, duration * motionScale())}ms`;
  element.appendChild(layer);
  layer.addEventListener('animationend', () => layer.remove(), { once: true });
  return layer;
}

/** Full-screen additive white flash, matching EntranceSteps.GameWideFlash. */
export function fullSurfaceFlash(root = document.body, duration = 1000) {
  if (isReducedMotion()) return null;
  const layer = document.createElement('div');
  layer.className = 'ui-full-surface-flash';
  layer.style.animationDuration = `${duration}ms`;
  root.appendChild(layer);
  layer.addEventListener('animationend', () => layer.remove(), { once: true });
  return layer;
}

export default { flashColour, pulseGlow, setAdditive, attachAdditiveFlash, fullSurfaceFlash };
