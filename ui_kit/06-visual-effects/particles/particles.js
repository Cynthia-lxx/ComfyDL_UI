/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Algorithm ported from osu.Game/Graphics/ParticleSpewer.cs:40-245 and
 * ParticleExplosion.cs:27-144: fixed-capacity ring buffer, time-pure position/
 * alpha/scale/angle functions, additive blending, batched draws, large-frame
 * spawn limiting, and ignoring "future" particles after a seek.
 * No upstream code or assets copied.
 */

export function readCssColour(element, name, fallback = '#66ccff') {
  const style = getComputedStyle(element);
  const raw = style.getPropertyValue(name).trim();
  const probe = document.createElement('canvas').getContext('2d');
  probe.fillStyle = fallback;
  probe.fillStyle = raw || fallback;
  return probe.fillStyle;
}

export function isReducedMotion() {
  const root = document.documentElement;
  if (root.dataset.reduceMotion === 'true') return true;
  if (root.dataset.reduceMotion === 'false') return false;
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
}

function makeParticle(now, preset, random, burst) {
  const angle = burst ? random() * Math.PI * 2 : -Math.PI / 2 + (random() - 0.5) * preset.spread;
  const distance = preset.speed * (0.5 + random());
  return {
    birth: now,
    duration: preset.duration * (0.6 + random() * 0.8),
    dirX: Math.cos(angle),
    dirY: Math.sin(angle),
    distance,
    size: preset.size * (0.5 + random()),
    seed: random(),
  };
}

/* time-pure functions: no per-frame physics integration */
function alphaAtTime(t) {
  if (t < 0 || t > 1) return 0;
  return Math.pow(1 - t, 1.5);
}
function scaleAtTime(t) { return 1 - 0.6 * t; }
function positionAtTime(p, t) {
  const eased = 1 - Math.pow(1 - t, 2);
  return [p.dirX * p.distance * eased, p.dirY * p.distance * eased + 0.35 * t * t];
}
function angleAtTime(p, t) { return p.seed * Math.PI * 2 + t * p.dirX * 2; }

/**
 * @param {HTMLCanvasElement} canvas
 * @param {Object} preset from presets.json
 */
export function createParticleField(canvas, preset = {}) {
  const ctx = canvas.getContext('2d');
  const config = {
    perSecond: preset.perSecond ?? 24,
    duration: preset.duration ?? 900,
    speed: preset.speed ?? 0.18,
    spread: preset.spread ?? 1.2,
    size: preset.size ?? 4,
    burstCount: preset.burstCount ?? 150,
    colourVar: preset.colourVar ?? '--colour-1',
    autoSpawn: preset.autoSpawn ?? true,
  };
  if (!ctx) return { unmount() {}, burst() {}, setTheme() {}, available: false };

  const capacity = Math.max(16, Math.ceil(config.perSecond * Math.ceil(config.duration / 1000)) + config.burstCount);
  const ring = new Array(capacity).fill(null);
  let cursor = 0;
  let seed = 20261113;
  let colour = readCssColour(canvas, config.colourVar);
  let raf = 0;
  let last = performance.now();
  let spawnDebt = 0;
  let disposed = false;

  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  };

  const spawn = (now, count, burst = false) => {
    for (let i = 0; i < count; i++) {
      ring[cursor] = makeParticle(now, config, random, burst);
      cursor = (cursor + 1) % capacity;
    }
  };

  const draw = (now) => {
    if (disposed) return;
    resize();
    const delta = Math.min(now - last, 100); // 大跳帧限补发
    last = now;

    if (config.autoSpawn && delta < 100 && !isReducedMotion()) {
      spawnDebt += (delta / 1000) * config.perSecond;
      if (spawnDebt >= 1) {
        const count = Math.floor(spawnDebt);
        spawnDebt -= count;
        spawn(now, count);
      }
    }

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = colour;

    // 单批次遍历：每个粒子一次 fillRect，无逐个 Drawable
    for (const p of ring) {
      if (!p || p.birth > now) continue; // 忽略回放 seek 产生的"未来粒子"
      const t = (now - p.birth) / p.duration;
      const alpha = alphaAtTime(t);
      if (alpha <= 0) continue;
      const [dx, dy] = positionAtTime(p, t);
      const scale = scaleAtTime(t);
      const size = Math.max(1, p.size * scale * (w / Math.max(1, canvas.clientWidth)));
      const x = w * 0.5 + dx * w - size / 2;
      const y = h * 0.72 + dy * h - size / 2;
      ctx.globalAlpha = alpha * 0.85;
      ctx.save();
      ctx.translate(x + size / 2, y + size / 2);
      ctx.rotate(angleAtTime(p, t));
      ctx.fillRect(-size / 2, -size / 2, size, size);
      ctx.restore();
    }

    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    raf = requestAnimationFrame(draw);
  };

  raf = requestAnimationFrame(draw);

  return {
    available: true,
    burst(count = config.burstCount) { spawn(performance.now(), count, true); },
    setTheme(name) {
      if (name) canvas.dataset.theme = name;
      colour = readCssColour(canvas, config.colourVar);
    },
    unmount() {
      disposed = true;
      cancelAnimationFrame(raf);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };
}

export function createParticleSpewer(canvas, preset) {
  return createParticleField(canvas, { ...preset, autoSpawn: true });
}

export function createParticleExplosion(canvas, preset) {
  return createParticleField(canvas, { ...preset, autoSpawn: false });
}

export default createParticleField;
