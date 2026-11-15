/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Ported from osu.Game/Rulesets/Judgements/DrawableJudgement.cs:88-191:
 * pooled pieces, BeginAbsoluteSequence(Result.TimeAbsolute), renderer-replaceable
 * component with LatestTransformEndTime defining the lifetime, and revert/rewind.
 * No upstream code or assets copied.
 */

import { Timeline } from '../../05-motion/timeline.js';

const DEFAULT_RENDERER = {
  createPiece(type) {
    const el = document.createElement('div');
    el.className = `ui-status-pulse ui-status-pulse--${type}`;
    el.textContent = { great: '300', good: '200', ok: '100', meh: '50', miss: 'ERROR' }[type] || '';
    el.latestTransformEndTime = 600;
    return el;
  },
};

/**
 * @param {HTMLElement} container
 * @param {{poolSize?:number, renderer?:Object, clock?:Function, onResult?:Function}} [options]
 */
export function createStatusPulseLayer(container, options = {}) {
  const poolSize = options.poolSize ?? 24;
  const renderer = options.renderer || DEFAULT_RENDERER;
  const clock = options.clock || (() => performance.now());
  const pool = [];
  const active = new Map();
  let sequence = 0;
  let disposed = false;

  container.classList.add('ui-status-pulse-layer');

  const acquire = () => {
    const host = pool.pop() || document.createElement('div');
    host.className = 'ui-status-pulse-host';
    host.replaceChildren();
    container.appendChild(host);
    return host;
  };

  const release = (id) => {
    const entry = active.get(id);
    if (!entry) return;
    entry.timeline.cancel();
    entry.host.remove();
    entry.host.replaceChildren();
    if (pool.length < poolSize) pool.push(entry.host);
    active.delete(id);
  };

  /** @param {{id?:string|number, type:string, timeAbsolute?:number, x:number, y:number}} result */
  const apply = (result) => {
    if (disposed) return null;
    const id = result.id ?? ++sequence;
    release(id); // replay / rewind must produce the same state

    const host = acquire();
    const piece = renderer.createPiece(result.type);
    const duration = Math.max(1, Number(piece.latestTransformEndTime) || 600);
    host.style.left = `${result.x ?? 0}px`;
    host.style.top = `${result.y ?? 0}px`;
    host.appendChild(piece);

    const timeline = new Timeline({ duration, onComplete: () => release(id) });
    const delay = Math.max(0, (result.timeAbsolute ?? clock()) - clock());
    timeline.at(delay, () => {
      host.dataset.active = 'true';
      host.dispatchEvent(new CustomEvent('ui-status-pulse', {
        bubbles: true,
        detail: { id, type: result.type, duration },
      }));
      if (typeof options.onResult === 'function') options.onResult(result, host);
    });
    timeline.play();

    active.set(id, { host, timeline, piece });
    return id;
  };

  const revert = (id) => release(id);

  return {
    apply,
    revert,
    clear() { for (const id of [...active.keys()]) release(id); },
    setTheme(name) { if (name) container.dataset.theme = name; },
    get activeCount() { return active.size; },
    unmount() {
      disposed = true;
      for (const id of [...active.keys()]) release(id);
      pool.length = 0;
      container.classList.remove('ui-status-pulse-layer');
    },
  };
}

export default createStatusPulseLayer;
