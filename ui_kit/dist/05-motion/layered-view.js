/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Executable multi-layer reveal/collapse engine.
 *
 * Upstream model (osu.Game): each Drawable subtree owns an independent
 * TransformSequence; a fixed stack of overlay containers composites them via
 * Depth, Masking, Blending and background dimming. Enter/exit are deliberately
 * asymmetric (800/OutSine·OutQuint vs 500/InSine·InQuint) and PopOutComplete
 * (OnComplete) is a fence that delays teardown.
 *
 * Here each layer is a DOM node whose start/end state, duration, easing and
 * delay come from 05-motion/layered-view.json. Visibility is NOT gated by
 * display:none (that would remove the before-change style and skip every
 * transition); visibility + pointer-events are used instead.
 */

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
const delay = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

function finished(root, timeout = 1400) {
  const animations = root.getAnimations ? root.getAnimations({ subtree: true }) : [];
  if (!animations.length) return Promise.resolve();
  return Promise.race([
    Promise.allSettled(animations.map((animation) => animation.finished)),
    delay(timeout),
  ]);
}

function buildLayer(layer) {
  const element = document.createElement(layer.tag || 'div');
  element.className = `lv-layer ${(layer.class || '').trim()}`.trim();
  element.dataset.layer = layer.id;
  element.style.zIndex = String(layer.z ?? 0);
  element.title = `${layer.name}${layer.upstream ? ` · ${layer.upstream}` : ''}${layer.source ? ` · ${layer.source}` : ''}`;

  for (const [name, value] of Object.entries(layer.css || {})) element.style.setProperty(name, value);
  const set = (name, value) => element.style.setProperty(name, String(value));
  set('--lv-closed-transform', layer.closed?.transform ?? 'none');
  set('--lv-open-transform', layer.open?.transform ?? 'none');
  set('--lv-closed-opacity', layer.closed?.opacity ?? 1);
  set('--lv-open-opacity', layer.open?.opacity ?? 1);
  set('--lv-enter-ms', `${layer.enter?.ms ?? 800}ms`);
  set('--lv-exit-ms', `${layer.exit?.ms ?? 500}ms`);
  set('--lv-enter-ease', layer.enter?.ease ?? 'ease-out');
  set('--lv-exit-ease', layer.exit?.ease ?? 'ease-in');
  set('--lv-layer-delay', `${layer.enter?.delay ?? 0}ms`);

  if (layer.html) element.innerHTML = layer.html;
  return element;
}

/**
 * Mount a layered view into `host`.
 * @param {HTMLElement} host
 * @param {{preset: object, rate?: number, onEvent?: Function}} options
 */
export function createLayeredView(host, options = {}) {
  const preset = options.preset;
  if (!preset || !Array.isArray(preset.layers) || !preset.layers.length) {
    console.warn('[layered-view] preset with layers is required');
    return {
      open: async () => false, close: () => {}, toggle: () => {}, replay: () => {},
      seek: () => {}, setRate: () => {}, setLayerVisible: () => {}, unmount: () => {},
      get isOpen() { return false; }, get animationCount() { return 0; }, get layerCount() { return 0; },
    };
  }

  const root = document.createElement('div');
  root.className = 'lv-root';
  root.dataset.phase = 'closed';
  root.dataset.direction = 'exit';
  root.style.visibility = 'hidden';
  root.style.pointerEvents = 'none';
  root.setAttribute('role', 'dialog');
  root.setAttribute('aria-label', preset.label || 'layered view');

  const nodes = new Map();
  for (const layer of [...preset.layers].sort((a, b) => (a.z ?? 0) - (b.z ?? 0))) {
    const element = buildLayer(layer);
    nodes.set(layer.id, element);
    root.appendChild(element);
  }
  host.appendChild(root);

  let rate = options.rate || 1;
  let opened = false;
  let startedAt = 0;
  let hideToken = 0;
  let raf = 0;
  const emit = (type, detail = {}) => {
    if (typeof options.onEvent === 'function') options.onEvent(type, detail);
    root.dispatchEvent(new CustomEvent(`ui-layered-${type}`, { bubbles: true, composed: true, detail }));
  };

  const applyRate = () => {
    for (const animation of root.getAnimations({ subtree: true })) {
      try { animation.playbackRate = rate; } catch { /* finished */ }
    }
  };

  async function open() {
    if (opened) return replay();
    hideToken += 1;
    root.dataset.direction = 'enter';
    root.style.visibility = 'visible';
    root.style.pointerEvents = 'auto';
    await nextFrame();
    opened = true;
    startedAt = performance.now();
    root.dataset.phase = 'open';
    applyRate();
    emit('open', { label: preset.label, layers: nodes.size, duration: preset.duration?.enter ?? 800 });
    return true;
  }

  async function close() {
    if (!opened && root.style.visibility === 'hidden') return;
    root.dataset.direction = 'exit';
    root.dataset.phase = 'closed';
    opened = false;
    const token = ++hideToken;
    emit('close', { label: preset.label, duration: preset.duration?.exit ?? 500 });
    applyRate();
    await finished(root);
    if (token !== hideToken || opened) return;
    root.style.visibility = 'hidden';
    root.style.pointerEvents = 'none';
  }

  function replay() {
    for (const animation of root.getAnimations({ subtree: true })) animation.cancel();
    root.dataset.direction = 'enter';
    root.dataset.phase = 'closed';
    void root.offsetWidth; // force a before-change style
    opened = true;
    startedAt = performance.now();
    root.dataset.phase = 'open';
    applyRate();
    emit('step', { action: 'replay', at: 0 });
  }

  function toggle() { return opened ? (close(), false) : open(); }

  function seek(ms) {
    const animations = root.getAnimations({ subtree: true });
    for (const animation of animations) {
      try { animation.currentTime = Math.max(0, ms); } catch { /* ignore */ }
    }
    emit('step', { action: 'seek', at: ms });
  }

  function setRate(value) {
    rate = Number(value) || 1;
    applyRate();
  }

  function setLayerVisible(id, visible) {
    const element = nodes.get(id);
    if (element) element.hidden = !visible;
  }

  function elapsed() { return opened || root.dataset.phase === 'open' ? performance.now() - startedAt : 0; }

  function unmount() {
    cancelAnimationFrame(raf);
    hideToken += 1;
    for (const animation of root.getAnimations({ subtree: true })) animation.cancel();
    root.remove();
  }

  return {
    open, close, toggle, replay, seek, setRate, setLayerVisible, elapsed, unmount,
    get root() { return root; },
    get isOpen() { return opened; },
    get animationCount() { return root.getAnimations({ subtree: true }).length; },
    get layerCount() { return nodes.size; },
    get layers() { return [...nodes.keys()]; },
    get preset() { return preset; },
  };
}

export default createLayeredView;
