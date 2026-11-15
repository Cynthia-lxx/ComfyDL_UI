/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Reusable motion engine translated from the full inventory
 * (05-motion/effect-inventory.json): stagger, fan-out, wave reveal, shear slide,
 * circular reveal, parallax, beat sync, rolling numbers, ripple, glitch triangles.
 * Uses the Web Animations API + requestAnimationFrame; every effect honours
 * `--motion-scale` / prefers-reduced-motion (upstream has no such switch).
 */

export function motionScale() {
  const root = document.documentElement;
  if (root.dataset.reduceMotion === 'true') return 0;
  if (root.dataset.reduceMotion === 'false') return 1;
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return 0;
  const value = Number(getComputedStyle(root).getPropertyValue('--motion-scale'));
  return Number.isFinite(value) ? value : 1;
}

export function animate(element, keyframes, options = {}) {
  if (!element) return null;
  const scale = motionScale();
  const duration = Math.max(1, (options.duration ?? 300) * scale);
  return element.animate(keyframes, {
    duration,
    easing: options.easing || 'cubic-bezier(0.22, 1, 0.36, 1)',
    delay: scale === 0 ? 0 : (options.delay || 0),
    fill: options.fill || 'both',
    iterations: options.iterations || 1,
    direction: options.direction || 'normal',
  });
}

/** Option columns / Settings / Footer 的逐项错峰（inventory: stagger primitives）. */
export function stagger(elements, effectFn, options = {}) {
  const list = [...(elements || [])];
  const step = options.step ?? 30;
  const animations = list.map((el, index) => {
    el.style.setProperty('--ui-index', String(index));
    return effectFn(el, index, options);
  }).filter(Boolean);
  return {
    animations,
    cancel() { animations.forEach((a) => a?.cancel?.()); },
  };
}

/** 横排按钮的 Expanded/Contracted/Exploded 宽度状态机（ButtonSystem.cs）. */
export function fanOut(elements, options = {}) {
  const states = options.states || {
    expanded: 160, contracted: 0, exploded: 320,
  };
  const apply = (el, state, index = 0) => {
    if (!el) return;
    el.dataset.state = state;
    el.style.setProperty('--ui-wedge-width', `${states[state] ?? states.expanded}px`);
    el.style.setProperty('--ui-index', String(index));
    el.style.setProperty('--ui-content-alpha', String(
      Math.max(0, Math.min(1, ((states[state] ?? 160) / 160 - 0.5) / 0.3)),
    ));
  };
  return {
    apply,
    expandAll(delay = 150) {
      [...(elements || [])].forEach((el, i) => {
        if (motionScale() === 0) { apply(el, 'expanded', i); return; }
        window.setTimeout(() => apply(el, 'expanded', i), delay);
      });
    },
    contractAll(style = 0) {
      [...(elements || [])].forEach((el, i) => apply(el, style ? 'contracted-style1' : 'contracted', i));
    },
  };
}

/** WaveContainer 四层旋转揭示 + 内容相对 Y=2 双轨（WaveContainer.cs:87-192）. */
export function waveReveal(container, options = {}) {
  if (!container) return { open() {}, close() {}, destroy() {} };
  const wave = document.createElement('div');
  wave.className = 'ui-wave';
  const rotations = options.rotations || [13, -7, 4, -2];
  // 上游 FinalPosition 按全屏高度标定（-930/-560/-390/-220）；演示容器往往只有
  // 150px，直接用会把四层全推出视口，只剩静态边角。按容器高度折算成比例终值。
  const height = container.clientHeight || 400;
  const positions = options.positions || [-0.95, -0.62, -0.40, -0.22].map((r) => Math.round(r * height));
  rotations.forEach((rotation, index) => {
    const layer = document.createElement('div');
    layer.className = 'ui-wave__layer';
    layer.style.setProperty('--ui-wave-rotation', `${rotation}deg`);
    layer.style.setProperty('--ui-wave-y', `${positions[index]}px`);
    layer.style.setProperty('--ui-wave-index', String(index));
    layer.style.setProperty('--ui-wave-tint', String(index));
    wave.appendChild(layer);
  });
  const content = document.createElement('div');
  content.className = 'ui-wave__content';
  content.append(...[...container.children]);
  wave.appendChild(content);
  container.appendChild(wave);
  // 关键：CSS 选择器挂在 .ui-wave 上（子元素），状态必须写在 wave 本身，
  // 否则 content 永远停在 translateY(200%) —— 看起来"什么都没 reveal"。
  wave.dataset.open = 'false';
  container.dataset.open = 'false';

  return {
    element: wave,
    open() { wave.dataset.open = 'true'; container.dataset.open = 'true'; },
    close() { wave.dataset.open = 'false'; container.dataset.open = 'false'; },
    destroy() { wave.remove(); },
  };
}

/** ShearedOverlay / Option columns 列错峰：±700 交替 + 30ms 步进. */
export function shearSlide(elements, options = {}) {
  const distance = options.distance ?? 700;
  const step = options.step ?? 30;
  const duration = options.duration ?? 400;
  return stagger(elements, (el, index) => animate(el, [
    { opacity: 0, transform: `translateY(${index % 2 === 0 ? distance : -distance}px)` },
    { opacity: 1, transform: 'translateY(0)' },
  ], { duration, delay: index * step, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }), { step });
}

/** PopupDialog 环形揭示：scale 0.7→1 + ring 20→100. */
export function circularReveal(element, options = {}) {
  if (!element) return null;
  const ring = element.querySelector('[data-ring]');
  const animations = [animate(element, [
    { opacity: 0, transform: 'scale(0.7)' },
    { opacity: 1, transform: 'scale(1)' },
  ], { duration: options.duration ?? 750, easing: 'linear(0, 1.12 35%, 0.98 70%, 1)' })];
  if (ring) animations.push(animate(ring, [
    { width: '20px', height: '20px' }, { width: '100px', height: '100px' },
  ], { duration: options.duration ?? 750 }));
  return { animations, cancel() { animations.forEach((a) => a?.cancel()); } };
}

/** ParallaxContainer：位置 100ms / 缩放 1000ms 阻尼，amount 0.02 或 0.01. */
export function parallax(element, options = {}) {
  if (!element) return { destroy() {} };
  const amount = (options.amount ?? 0.02) * motionScale();
  let raf = 0;
  const onMove = (event) => {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      const x = (event.clientX / window.innerWidth - 0.5) * 2;
      const y = (event.clientY / window.innerHeight - 0.5) * 2;
      element.style.transform = `translate(${x * amount * 100}px, ${y * amount * 100}px) scale(${1 + Math.abs(amount)})`;
      raf = 0;
    });
  };
  window.addEventListener('pointermove', onMove);
  return { destroy() { window.removeEventListener('pointermove', onMove); cancelAnimationFrame(raf); element.style.transform = ''; } };
}

/** RollingCounter / ArgonTotalResultPanel：3000ms OutPow10 数字滚动. */
export function rollingNumber(element, from, to, options = {}) {
  if (!element) return { cancel() {} };
  const duration = Math.max(1, (options.duration ?? 3000) * motionScale());
  const format = options.format || ((v) => Math.round(v).toLocaleString());
  const start = performance.now();
  let raf = 0;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 10);
    element.textContent = format(from + (to - from) * eased);
    if (t < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return { cancel() { cancelAnimationFrame(raf); element.textContent = format(to); } };
}

export function elasticPop(element, options = {}) {
  return animate(element, [
    { transform: `scale(${options.from ?? 0.9})` },
    { transform: 'scale(1)' },
  ], { duration: options.duration ?? 750, easing: 'linear(0, 0.6 12%, 1.08 28%, 0.97 45%, 1)' });
}

/** CursorRippleVisualiser：池化圆环 scale 0.1→1 + fade 700ms. */
export function ripple(container, x, y, options = {}) {
  if (!container || motionScale() === 0) return null;
  const ring = document.createElement('div');
  Object.assign(ring.style, {
    position: 'absolute', left: `${x}px`, top: `${y}px`,
    width: '40px', height: '40px', margin: '-20px 0 0 -20px',
    border: '3px solid var(--ui-accent, #66ccff)', borderRadius: '50%',
    mixBlendMode: 'screen', pointerEvents: 'none',
  });
  container.appendChild(ring);
  const animation = animate(ring, [
    { opacity: 0.6, transform: 'scale(0.1)' },
    { opacity: 0, transform: 'scale(1)' },
  ], { duration: options.duration ?? 700 });
  animation?.finished.then(() => ring.remove()).catch(() => ring.remove());
  return ring;
}

/** BeatSyncedContainer 等价：按 BPM 触发回调，返回停止句柄. */
export function intervalSync(callback, options = {}) {
  const bpm = options.bpm ?? 120;
  const interval = 60000 / bpm;
  let index = 0;
  let timer = 0;
  let stopped = false;
  const run = () => {
    if (stopped) return;
    callback({ index, time: performance.now(), emphasis: Boolean(options.emphasis), cycleLength: interval });
    index += 1;
    timer = window.setTimeout(run, interval);
  };
  timer = window.setTimeout(run, options.early ?? 0);
  return { stop() { stopped = true; clearTimeout(timer); }, get cycleIndex() { return index; } };
}

/** Scatter shapes：每 22ms 生成随机三角，120ms 淡出. */
export function scatterShapes(container, options = {}) {
  if (!container || motionScale() === 0) return { stop() {} };
  const interval = options.interval ?? 22;
  let timer = 0;
  const spawn = () => {
    const t = document.createElement('div');
    const size = (Math.random() + 0.2) * 80;
    Object.assign(t.style, {
      position: 'absolute', left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%`,
      width: '0', height: '0',
      borderLeft: `${size / 2}px solid transparent`, borderRight: `${size / 2}px solid transparent`,
      borderBottom: `${size}px solid var(--ui-accent, #66ccff)`,
      mixBlendMode: 'screen', pointerEvents: 'none',
    });
    container.appendChild(t);
    const a = animate(t, [{ opacity: 1 }, { opacity: 0 }], { duration: options.fade ?? 120 });
    a?.finished.then(() => t.remove()).catch(() => t.remove());
    timer = window.setTimeout(spawn, interval);
  };
  timer = window.setTimeout(spawn, 0);
  return { stop() { clearTimeout(timer); container.replaceChildren(); } };
}

export default {
  motionScale, animate, stagger, fanOut, waveReveal, shearSlide, circularReveal,
  parallax, rollingNumber, elasticPop, ripple, intervalSync, scatterShapes,
};
