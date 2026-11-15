/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: BarGraph.cs:28-41,81-102,192-240 (0..1 normalise, 250ms InOutCubic manual
 *   interpolation, initial/final length arrays, min 1.5px, direction);
 *   LineGraph.cs:38-67,75-92,105-131 (1500ms OutQuint width reveal, min/max clamping,
 *   SmoothPath radius 1, y = (max-v)/(max-min));
 *   SegmentedGraph.cs:89-116,118-153 (tier = floor(v*tierCount), segments merged per tier,
 *   static redraw — no animation);
 *   StarCounter.cs:14-31,104-135,152-164 (80ms per star, 0.4 min scale, 100ms fade,
 *   1000ms OutElasticHalf scale, wave-like stagger).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-graph.css', import.meta.url);
const FALLBACK = `:host{display:block;height:80px;position:relative}.bars{display:flex;gap:2px;height:100%;align-items:flex-end}.bar{flex:1;min-height:1.5px;background:var(--ui-accent,#66ccff)}`;
const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class UiGraphElement extends HTMLElement {
  static get observedAttributes() { return ['values', 'variant', 'max', 'stars', 'tiers']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._root = document.createElement('div');
    this._root.className = 'root';
    this.shadowRoot.append(this._style, this._root);
    this._raf = 0;
    this._lengths = [];
  }

  connectedCallback() { this._loadStyles(); this._render(); }
  attributeChangedCallback() { if (this.isConnected) this._render(); }
  disconnectedCallback() { cancelAnimationFrame(this._raf); }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* file:// or offline: inline fallback stays */ }
  }

  _values() {
    try {
      const parsed = JSON.parse(this.getAttribute('values') || '[]');
      if (!Array.isArray(parsed)) return [];
      const maxOverride = Number(this.getAttribute('max'));
      const max = Number.isFinite(maxOverride) && maxOverride > 0
        ? maxOverride : Math.max(0, ...parsed.map(Number));
      return parsed.map((v) => (max > 0 ? Math.max(0, Number(v) / max) : 0));
    } catch { return []; }
  }

  _render() {
    const variant = this.getAttribute('variant') || 'bar';
    this.dataset.variant = variant;
    const values = this._values();
    if (variant === 'line') return this._renderLine(values);
    if (variant === 'segmented') return this._renderSegments(values);
    if (variant === 'stars') return this._renderStars();
    return this._renderBars(values);
  }

  /** BarGraph：逐帧 InOutCubic 插值，已有索引上次终值续接、新索引从 0 起。 */
  _renderBars(values) {
    cancelAnimationFrame(this._raf);
    const from = values.map((_, i) => this._lengths[i] ?? 0);
    const target = values;
    this._root.innerHTML = '';
    const bars = target.map(() => {
      const el = document.createElement('div');
      el.className = 'bar';
      el.style.height = '1.5px';
      this._root.appendChild(el);
      return el;
    });
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    const duration = Math.max(1, Math.round(250 * Math.abs(scale)));
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = easeInOutCubic(t);
      this._lengths = target.map((final, i) => from[i] + (final - from[i]) * eased);
      this._lengths.forEach((length, i) => {
        bars[i].style.height = `max(1.5px, ${(length * 100).toFixed(2)}%)`;
      });
      if (t < 1) this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  /** LineGraph：SmoothPath 一次生成，用 1500ms OutQuint 宽度揭示（非形变）。 */
  _renderLine(values) {
    cancelAnimationFrame(this._raf);
    const width = 100;
    const height = 100;
    const min = Math.min(...values, 0);
    const max = Math.max(...values, min + 1e-9);
    const points = values.map((v, i) => {
      const x = values.length > 1 ? i / (values.length - 1) * width : width;
      const y = (max - v) / (max - min || 1) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
    this._root.innerHTML = `<svg class="line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><polyline points="${points}" /></svg>`;
    const svg = this._root.querySelector('svg');
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    svg.animate([{ clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0% 0 0)' }],
      { duration: Math.max(1, Math.round(1500 * Math.abs(scale))), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
  }

  /** SegmentedGraph：tier = floor(v*tierCount)，同 tier 连续区间合并，半透明叠加。 */
  _renderSegments(values) {
    cancelAnimationFrame(this._raf);
    const tiers = Math.max(1, Number(this.getAttribute('tiers')) || 3);
    let currentTier = -1;
    let startIndex = 0;
    const segments = [];
    values.forEach((v, i) => {
      const tier = Math.min(tiers - 1, Math.floor(v * tiers));
      if (tier !== currentTier) {
        if (currentTier >= 0) segments.push({ tier: currentTier, start: startIndex, end: i });
        currentTier = tier;
        startIndex = i;
      }
    });
    if (currentTier >= 0) segments.push({ tier: currentTier, start: startIndex, end: values.length });
    this._root.innerHTML = '';
    for (const segment of segments) {
      const el = document.createElement('div');
      el.className = 'segment';
      el.style.left = `${segment.start / Math.max(1, values.length) * 100}%`;
      el.style.width = `${(segment.end - segment.start) / Math.max(1, values.length) * 100}%`;
      el.style.opacity = String(0.35 + 0.65 * (segment.tier / tiers));
      this._root.appendChild(el);
    }
  }

  /** StarCounter：小数星插值 + 80ms 波浪错峰 + OutElasticHalf 缩放。 */
  _renderStars() {
    cancelAnimationFrame(this._raf);
    const total = Math.max(1, Number(this.getAttribute('stars')) || 10);
    const value = Math.max(0, Number(this.getAttribute('value')) || 0);
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    this._root.innerHTML = '';
    for (let i = 0; i < total; i += 1) {
      const star = document.createElement('span');
      star.className = 'star';
      star.textContent = '★';
      const amount = i + 1 <= value ? 1 : Math.min(1, Math.max(0, value - i));
      const display = 0.4 + 0.6 * amount;
      star.animate(
        [{ opacity: 0.4, transform: 'scale(0.4)' }, { opacity: display, transform: `scale(${display})` }],
        { duration: Math.max(1, Math.round(1000 * Math.abs(scale))), delay: Math.round(i * 80 * Math.abs(scale)), easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)', fill: 'both' },
      );
      this._root.appendChild(star);
    }
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-graph')) {
  customElements.define('ui-graph', UiGraphElement);
}
export default UiGraphElement;
