/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Algorithm ported from osu.Game/Rulesets/Mods/ModFlashlight.cs:172-198,256-339
 * (uniform-driven dark overlay with a soft-edged cone, intensity/break based sizing).
 * Upstream GLSL is not in this repository; this is a Canvas 2D rewrite of the
 * C# behaviour, not a copy of shader source.
 */

const INTENSITY_SCALE_100 = 0.8125; // 2.6 / 3.2
const INTENSITY_SCALE_200 = 0.625;  // 2.0 / 3.2
const EXPANDED_SCALE = 2.5;        // 8.0 / 3.2

export function getSizeForIntensity(baseSize, intensity, isBreak) {
  if (isBreak) return baseSize * EXPANDED_SCALE;
  if (intensity >= 200) return baseSize * INTENSITY_SCALE_200;
  if (intensity >= 100) return baseSize * INTENSITY_SCALE_100;
  return baseSize;
}

/**
 * Mount a soft-edged focusMask overlay on a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {{size?:number, intensity?:number, isBreak?:boolean, dim?:number, smoothness?:number}} [options]
 */
export function mountFocusMask(canvas, options = {}) {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return { unmount() {}, setTheme() {}, setIntensity() {}, setPosition() {}, setBreak() {}, available: false };
  }

  const state = {
    baseSize: options.size ?? 260,
    intensity: options.intensity ?? 0,
    isBreak: options.isBreak ?? false,
    dim: options.dim ?? 0.86,
    smoothness: options.smoothness ?? 0.45,
    x: 0.5,
    y: 0.5,
    currentSize: options.size ?? 260,
    targetSize: options.size ?? 260,
  };

  let colour = readColour(canvas, '--ui-accent-strong');
  let raf = 0;
  let disposed = false;

  function readColour(element, name) {
    const style = getComputedStyle(element);
    // `--ui-accent-strong` is itself a var() reference; canvas gradients cannot
    // parse var(), so prefer the resolved theme token and normalise via canvas.
    const raw = style.getPropertyValue('--colour-1').trim() || style.getPropertyValue(name).trim();
    const probe = document.createElement('canvas').getContext('2d');
    probe.fillStyle = '#66ccff';
    probe.fillStyle = raw || '#66ccff';
    return probe.fillStyle;
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function draw() {
    if (disposed) return;
    resize();
    const w = canvas.width;
    const h = canvas.height;
    const dpr = w / Math.max(1, canvas.clientWidth);
    const cx = state.x * w;
    const cy = state.y * h;
    const radius = Math.max(8, state.currentSize * dpr);

    ctx.clearRect(0, 0, w, h);

    // 暗场遮罩
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = `rgba(0, 0, 0, ${state.dim})`;
    ctx.fillRect(0, 0, w, h);

    // 挖出柔边光锥：径向渐变 + destination-out
    const inner = radius * (1 - state.smoothness);
    const gradient = ctx.createRadialGradient(cx, cy, inner, cx, cy, radius);
    gradient.addColorStop(0, 'rgba(0,0,0,1)');
    gradient.addColorStop(0.6, 'rgba(0,0,0,0.75)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    // 边缘微光（加法混合）
    ctx.globalCompositeOperation = 'lighter';
    const halo = ctx.createRadialGradient(cx, cy, radius * 0.82, cx, cy, radius);
    halo.addColorStop(0, 'rgba(0,0,0,0)');
    halo.addColorStop(1, colour);
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    // 阻尼插值；reduced-motion 时立即到位
    const scale = Number(getComputedStyle(document.documentElement).getPropertyValue('--motion-scale')) || 1;
    const easing = scale === 0 ? 1 : 0.12;
    state.currentSize += (state.targetSize - state.currentSize) * easing;

    raf = requestAnimationFrame(draw);
  }

  const updateTarget = () => {
    state.targetSize = getSizeForIntensity(state.baseSize, state.intensity, state.isBreak);
  };

  raf = requestAnimationFrame(draw);

  return {
    available: true,
    setIntensity(intensity) { state.intensity = Number(intensity) || 0; updateTarget(); },
    setBreak(value) { state.isBreak = Boolean(value); updateTarget(); },
    setPosition(x, y) { state.x = x; state.y = y; },
    setTheme(name) {
      if (name) canvas.dataset.theme = name;
      colour = readColour(canvas, '--ui-accent-strong');
    },
    unmount() {
      disposed = true;
      cancelAnimationFrame(raf);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
}

export default mountFocusMask;
