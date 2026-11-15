/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: RangeSlider.cs:90,99,105,109,128-129,138-160,174-215 (minRange 0.1,
 *   KeyboardStep 0.1, dual bar mutual clamping, hit split at nub edges, 6px bar height,
 *   nub-embedded 10px labels, "0.## suffix" tooltip, single hover highlight);
 *   ShearedRangeSlider.cs:93-135,167-168,213-226 (height 30, SHEAR 0.2 label capsule,
 *   corner radius 5, nub width 30, same clamp);
 *   TimeSlider.cs:11-13 (tooltip suffix " ms");
 *   ExpandableSlider.cs:86-109 (expanded state -> labels fade / slider fade 500ms OutQuint).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-range-slider.css', import.meta.url);
const FALLBACK = `:host{display:block;height:40px;position:relative}.track{position:absolute;top:50%;left:0;right:0;height:6px;background:var(--ui-surface-sunken,#222)}.nub{position:absolute;width:15px;height:15px;border-radius:8px;background:var(--ui-accent,#66ccff);top:50%;transform:translateY(-50%);font-size:10px;text-align:center;color:#fff}`;

export class UiRangeSliderElement extends HTMLElement {
  static get observedAttributes() { return ['min', 'max', 'step', 'lower', 'upper', 'variant', 'expanded', 'label']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._track = document.createElement('div');
    this._track.className = 'track';
    this._fill = document.createElement('div');
    this._fill.className = 'fill';
    this._label = document.createElement('span');
    this._label.className = 'caption';
    this._nubLower = this._makeNub('lower');
    this._nubUpper = this._makeNub('upper');
    this.shadowRoot.append(this._style, this._label, this._track, this._fill, this._nubLower, this._nubUpper);
    this._drag = null;
    this._onDown = (event) => {
      const nub = event.composedPath().find((el) => el.classList?.contains('nub'));
      const which = nub ? nub.dataset.which : this._nearest(event);
      this._drag = which;
      this._track.parentElement.setPointerCapture(event.pointerId);
      this._apply(event, which);
    };
    this._onMove = (event) => { if (this._drag) this._apply(event, this._drag); };
    this._onUp = () => {
      if (!this._drag) return;
      this._drag = null;
      this._emit('ui-range-commit');
    };
    this._onKeyDown = (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const target = event.composedPath().find((el) => el.classList?.contains('nub'));
      const which = target?.dataset.which || 'lower';
      const delta = event.shiftKey ? this._step * 10 : this._step;
      let next = which === 'lower' ? this.lower : this.upper;
      if (event.key === 'ArrowLeft') next -= delta;
      else if (event.key === 'ArrowRight') next += delta;
      else if (event.key === 'Home') next = this._min;
      else next = this._max;
      this._set(which, next);
      this._emit('ui-range-change');
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this.shadowRoot.addEventListener('pointerdown', this._onDown);
    this.shadowRoot.addEventListener('pointermove', this._onMove);
    this.shadowRoot.addEventListener('pointerup', this._onUp);
    this.shadowRoot.addEventListener('pointercancel', this._onUp);
    this.shadowRoot.addEventListener('keydown', this._onKeyDown);
  }

  disconnectedCallback() {
    this.shadowRoot.removeEventListener('pointerdown', this._onDown);
    this.shadowRoot.removeEventListener('pointermove', this._onMove);
    this.shadowRoot.removeEventListener('pointerup', this._onUp);
    this.shadowRoot.removeEventListener('pointercancel', this._onUp);
    this.shadowRoot.removeEventListener('keydown', this._onKeyDown);
  }

  attributeChangedCallback() { if (this.isConnected) this._sync(); }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* file:// or offline: inline fallback stays */ }
  }

  _makeNub(which) {
    const el = document.createElement('div');
    el.className = 'nub';
    el.dataset.which = which;
    el.tabIndex = 0;
    el.setAttribute('role', 'slider');
    return el;
  }

  get _min() { return Number(this.getAttribute('min')) || 0; }
  get _max() { return Number(this.getAttribute('max')) || 1; }
  get _step() { return Number(this.getAttribute('step')) || 0.1; }
  get lower() { return this._clamp(Number(this.getAttribute('lower')) || 0); }
  get upper() { return this._clamp(Number(this.getAttribute('upper')) || 1); }
  _clamp(v) { return Math.min(this._max, Math.max(this._min, Number(v))); }

  _nearest(event) {
    const rect = this.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const l = (this.lower - this._min) / (this._max - this._min);
    const u = (this.upper - this._min) / (this._max - this._min);
    return Math.abs(x - l) <= Math.abs(x - u) ? 'lower' : 'upper';
  }

  _apply(event, which) {
    const rect = this.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - 15) / Math.max(1, rect.width - 30)));
    this._set(which, this._min + ratio * (this._max - this._min));
    this._emit('ui-range-change');
  }

  /** 双向钳制：lower 推 upper >= lower+minRange，upper 推 lower <= upper-minRange。 */
  _set(which, raw) {
    const minRange = Number(this.getAttribute('minrange')) || 0.1;
    let lower = this.lower;
    let upper = this.upper;
    const value = this._clamp(Math.round(raw / this._step) * this._step);
    if (which === 'lower') {
      lower = Math.min(value, upper - minRange);
      if (upper - lower < minRange) upper = Math.min(this._max, lower + minRange);
    } else {
      upper = Math.max(value, lower + minRange);
      if (upper - lower < minRange) lower = Math.max(this._min, upper - minRange);
    }
    this.setAttribute('lower', String(Number(lower.toFixed(5))));
    this.setAttribute('upper', String(Number(upper.toFixed(5))));
  }

  _sync() {
    const variant = this.getAttribute('variant') || 'plain';
    this.dataset.variant = variant;
    this.dataset.expanded = String(this.getAttribute('expanded') !== 'false');
    const span = this._max - this._min || 1;
    const l = (this.lower - this._min) / span;
    const u = (this.upper - this._min) / span;
    const suffix = variant === 'time' ? ' ms' : '';
    this._nubLower.style.left = `calc(${l * 100}% - ${l * 15}px)`;
    this._nubUpper.style.left = `calc(${u * 100}% - ${u * 15}px)`;
    this._fill.style.left = `${l * 100}%`;
    this._fill.style.width = `${(u - l) * 100}%`;
    this._nubLower.textContent = `${this.lower.toFixed(1)}${suffix}`;
    this._nubUpper.textContent = `${this.upper.toFixed(1)}${suffix}`;
    this._nubLower.title = `${this.lower.toFixed(2)}${suffix}`;
    this._nubUpper.title = `${this.upper.toFixed(2)}${suffix}`;
    for (const [nub, value] of [[this._nubLower, this.lower], [this._nubUpper, this.upper]]) {
      nub.setAttribute('aria-valuemin', String(this._min));
      nub.setAttribute('aria-valuemax', String(this._max));
      nub.setAttribute('aria-valuenow', String(value));
    }
    this._label.textContent = this.getAttribute('label') || '';
  }

  collapse() { this.setAttribute('expanded', 'false'); }
  expand() { this.setAttribute('expanded', 'true'); }

  _emit(name) {
    this.dispatchEvent(new CustomEvent(name, {
      bubbles: true, composed: true, detail: { lower: this.lower, upper: this.upper },
    }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-range-slider')) {
  customElements.define('ui-range-slider', UiRangeSliderElement);
}
export default UiRangeSliderElement;
