/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: RollingCounter.cs:37-95,135-178 (DisplayedCount vs Current, TransformTo
 *   interpolation, RollingDuration/RollingEasing/IsRollingProportional, SetCountWithoutRolling,
 *   StopRolling, ResetCount, numeric font 40);
 *   ScoreCounter.cs:13-44 (1000ms Out, leading zeros, proportional |delta|, fixed width);
 *   PercentageCounter.cs:15-39 (375ms OutQuad, SetFraction divide-by-zero guard,
 *   initial 1.0, proportional |delta|*duration*100, 20px);
 *   CommaSeparatedScoreCounter.cs:11-22 (1000ms Out, N0 thousands separator).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-counter.css', import.meta.url);
const FALLBACK = `:host{display:inline-block;font-variant-numeric:tabular-nums;font-size:40px;color:var(--ui-text,#fff)}`;

const PRESETS = {
  plain: { duration: 0, easing: (t) => t, digits: 0 },
  digits: { duration: 1000, easing: (t) => 1 - (1 - t) * (1 - t), digits: 0 },
  separated: { duration: 1000, easing: (t) => 1 - (1 - t) * (1 - t), digits: 0 },
  accuracy: { duration: 375, easing: (t) => 1 - (1 - t) * (1 - t), digits: 2 },
};
const outQuad = (t) => 1 - (1 - t) * (1 - t);

export class UiCounterElement extends HTMLElement {
  static get observedAttributes() { return ['value', 'format', 'leading', 'duration']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._text = document.createElement('span');
    this.shadowRoot.append(this._style, this._text);
    this._displayed = 0;
    this._raf = 0;
    this._initialised = false;
  }

  connectedCallback() { this._loadStyles(); this._roll(); }
  attributeChangedCallback() { if (this.isConnected) this._roll(); }
  disconnectedCallback() { cancelAnimationFrame(this._raf); }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* file:// or offline: inline fallback stays */ }
  }

  get value() { return Number(this.getAttribute('value')) || 0; }
  set value(next) { this.setAttribute('value', String(next)); }

  /** 直接从 0 起（PercentageCounter 初始 1.0 除外）；首次挂载不等动画。 */
  _roll() {
    cancelAnimationFrame(this._raf);
    const target = this.value;
    if (!this._initialised) {
      this._initialised = true;
      this._displayed = this.getAttribute('format') === 'accuracy' ? 1 : target;
      this._render();
      return;
    }
    const format = this.getAttribute('format') || 'plain';
    const preset = PRESETS[format] || PRESETS.plain;
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    let duration = Number(this.getAttribute('duration')) || preset.duration;
    // 比例时长：小幅变化几乎瞬时（accuracy = |Δ|*duration*100；digits/separated = |Δ|）
    if (format === 'accuracy') duration = Math.min(duration, Math.abs(target - this._displayed) * duration * 100);
    else if (format === 'digits' || format === 'separated') duration = Math.min(duration, Math.abs(target - this._displayed));
    // fast：生产力 UI 场景——数值仍需插值动效，但要在 ~250ms 内落到准确值，
    // 不做结算界面那种 1000ms 长滚动。
    if (this.hasAttribute('fast')) duration = Math.min(duration, 250);
    duration = Math.max(0, Math.round(duration * Math.abs(scale)));
    const from = this._displayed;
    const start = performance.now();
    const step = (now) => {
      const t = duration <= 0 ? 1 : Math.min(1, (now - start) / duration);
      const eased = preset.easing ? preset.easing(t) : outQuad(t);
      this._displayed = from + (target - from) * eased;
      this._render();
      if (t < 1) this._raf = requestAnimationFrame(step);
      else this._emit('ui-counter-done');
    };
    this._raf = requestAnimationFrame(step);
  }

  setCountWithoutRolling(value) {
    cancelAnimationFrame(this._raf);
    this._displayed = Number(value) || 0;
    this._render();
  }
  stopRolling() { cancelAnimationFrame(this._raf); this._displayed = this.value; this._render(); }
  resetCount() { this.setCountWithoutRolling(0); }
  setFraction(numerator, denominator) {
    const epsilon = 1e-10;
    this.value = Math.abs(denominator) < epsilon ? 1 : numerator / denominator;
  }

  _render() {
    const format = this.getAttribute('format') || 'plain';
    const leading = Number(this.getAttribute('leading')) || 0;
    let text;
    if (format === 'accuracy') text = `${(this._displayed * 100).toFixed(2)}%`;
    else if (format === 'separated') text = Math.round(this._displayed).toLocaleString('en-US');
    else if (format === 'digits') text = String(Math.round(this._displayed)).padStart(leading, '0');
    else text = String(Math.round(this._displayed));
    this._text.textContent = text;
    this.dataset.format = format;
  }

  _emit(name) {
    this.dispatchEvent(new CustomEvent(name, { bubbles: true, composed: true, detail: { value: this.value } }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-counter')) {
  customElements.define('ui-counter', UiCounterElement);
}
export default UiCounterElement;
