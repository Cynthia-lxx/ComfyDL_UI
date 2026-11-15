/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/RoundedSliderBar.cs:62-129 (5px track, accent left box vs
 *     background right box, circular mask, RangePadding = Nub.DEFAULT_EXPANDED_SIZE / 2)
 *   osu.Game/Graphics/UserInterface/RoundedSliderBar.cs:190-205 (glow on hover/drag; nub
 *     MoveToX(value, 250, Easing.OutQuint))
 *   osu.Game/Graphics/UserInterface/Nub.cs:162-178 (filled: border -> 8.5 OutElasticHalf 200ms;
 *     unfilled: width 0.75 / border 3 OutQuint) — slider nub is permanently "filled"
 *   osu.Game/Graphics/UserInterface/Nub.cs:82-105 (Glowing: 40ms flash then 800ms settle)
 *   osu.Game/Graphics/UserInterface/OsuSliderBar.cs:61-84 (notch-tick sample throttled 30ms,
 *     pitched by NormalizedValue — replicated as a throttled nub pulse)
 *   osu.Game/Graphics/UserInterface/RoundedSliderBar.cs:56-70 (double click resets to default)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-slider.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: inline-block; width: 240px; }
  .track { position: relative; height: 15px; }
  .nub { position: absolute; width: 50px; height: 15px; border-radius: 8px; background: #66ccff; }
`;

export class UiSliderElement extends HTMLElement {
  static get observedAttributes() {
    return ['min', 'max', 'step', 'value', 'disabled', 'label', 'data-i18n-key'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._default = 0;
    this._lastTickTime = 0;
    this._tickAnimation = null;
    this._dragging = false;

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._track = document.createElement('div');
    this._track.className = 'track';
    this._track.setAttribute('part', 'track');

    this._barLeft = document.createElement('div');
    this._barLeft.className = 'bar-left';
    this._barRight = document.createElement('div');
    this._barRight.className = 'bar-right';
    this._nub = document.createElement('div');
    this._nub.className = 'nub';
    this._nub.setAttribute('part', 'nub');

    this._tooltip = document.createElement('div');
    this._tooltip.className = 'tooltip';
    this._tooltip.setAttribute('role', 'presentation');

    this._track.append(this._barLeft, this._barRight, this._nub);
    this.shadowRoot.append(this._style, this._track, this._tooltip);

    this._onPointerDown = (event) => {
      if (this.hasAttribute('disabled')) return;
      this._dragging = true;
      this._track.dataset.dragging = 'true';
      this._track.setPointerCapture(event.pointerId);
      this._applyFromPointer(event);
    };
    this._onPointerMove = (event) => {
      if (!this._dragging) return;
      this._applyFromPointer(event);
    };
    this._onPointerUp = () => {
      if (!this._dragging) return;
      this._dragging = false;
      delete this._track.dataset.dragging;
      this._updateTooltip();
    };
    this._onDoubleClick = () => {
      if (this.hasAttribute('disabled')) return;
      // RoundedSliderBar.cs:56-70 — double click resets to the default bindable value.
      this.value = this._default;
      this._syncGeometry(true);
      this._pulse(true);
      this._emitChange();
    };
    this._onKeyDown = (event) => {
      if (this.hasAttribute('disabled')) return;
      const big = event.shiftKey ? 10 : 1;
      let next = null;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = this.value - this._step * big;
      else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = this.value + this._step * big;
      else if (event.key === 'PageDown') next = this.value - this._step * 10;
      else if (event.key === 'PageUp') next = this.value + this._step * 10;
      else if (event.key === 'Home') next = this._min;
      else if (event.key === 'End') next = this._max;
      else return;
      event.preventDefault();
      event.stopPropagation();
      this.value = next;
      this._syncGeometry();
      this._pulse();
      this._emitChange();
    };
    this._onFocus = () => { this.dataset.focused = 'true'; this._updateGlow(); this._updateTooltip(); };
    this._onBlur = () => { this.dataset.focused = 'false'; this._updateGlow(); this._updateTooltip(); };
    this._onEnter = () => { this._updateGlow(); this._updateTooltip(); };
    this._onLeave = () => { if (!this._dragging) { this._updateGlow(); this._updateTooltip(); } };
  }

  connectedCallback() {
    this._loadStyles();
    if (!this.hasAttribute('value')) this.setAttribute('value', String(this._default));
    this._defaultLocked = false;
    this._default = this._quantize(Number(this.getAttribute('value')) || 0);
    this._sync();
    this._track.addEventListener('pointerdown', this._onPointerDown);
    this._track.addEventListener('pointermove', this._onPointerMove);
    this._track.addEventListener('pointerup', this._onPointerUp);
    this._track.addEventListener('pointercancel', this._onPointerUp);
    this._track.addEventListener('dblclick', this._onDoubleClick);
    this._track.addEventListener('pointerenter', this._onEnter);
    this._track.addEventListener('pointerleave', this._onLeave);
    this.addEventListener('keydown', this._onKeyDown);
    this.addEventListener('focus', this._onFocus);
    this.addEventListener('blur', this._onBlur);
  }

  disconnectedCallback() {
    this._track.removeEventListener('pointerdown', this._onPointerDown);
    this._track.removeEventListener('pointermove', this._onPointerMove);
    this._track.removeEventListener('pointerup', this._onPointerUp);
    this._track.removeEventListener('pointercancel', this._onPointerUp);
    this._track.removeEventListener('dblclick', this._onDoubleClick);
    this._track.removeEventListener('pointerenter', this._onEnter);
    this._track.removeEventListener('pointerleave', this._onLeave);
    this.removeEventListener('keydown', this._onKeyDown);
    this.removeEventListener('focus', this._onFocus);
    this.removeEventListener('blur', this._onBlur);
  }

  attributeChangedCallback(name) {
    if (!this.isConnected) return;
    if (name === 'value') {
      this._sync();
    } else {
      this._sync();
      this._syncGeometry(true);
    }
  }

  /** Load the sibling CSS file into the shadow root; keep the inline fallback on failure. */
  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch {
      /* file:// or offline: inline fallback stays in place */
    }
  }

  get value() { return this._current; }
  set value(raw) {
    this._current = this._quantize(Number(raw));
    this.setAttribute('value', String(this._current));
  }

  get _min() { const v = Number(this.getAttribute('min')); return Number.isFinite(v) ? v : 0; }
  get _max() { const v = Number(this.getAttribute('max')); return Number.isFinite(v) ? Math.max(v, this._min) : 100; }
  get _step() { const v = Number(this.getAttribute('step')); return v > 0 ? v : 1; }

  _quantize(raw) {
    const step = this._step;
    const clamped = Math.min(this._max, Math.max(this._min, Number.isFinite(raw) ? raw : this._min));
    const quantized = Math.round((clamped - this._min) / step) * step + this._min;
    return Math.min(this._max, Math.max(this._min, Number(quantized.toFixed(6))));
  }

  _sync() {
    this._current = this._quantize(Number(this.getAttribute('value')));
    this.dataset.variant = this.getAttribute('variant') || 'rounded';
    this.dataset.expanded = String(this.getAttribute('expanded') !== 'false');
    if (!this._defaultLocked) {
      // 双击重置的基准 = 挂载时的初始 value；后续 value 变化不得覆盖。
      this._default = this._current;
      this._defaultLocked = true;
    }
    const disabled = this.hasAttribute('disabled');
    this._track.setAttribute('aria-disabled', String(disabled));
    this.tabIndex = disabled ? -1 : 0;
    this.setAttribute('role', 'slider');
    const key = this.getAttribute('data-i18n-key');
    const dictionary = globalThis.UIReference?.i18n;
    const label = this.getAttribute('label') ?? (key && dictionary?.[key]) ?? 'slider';
    this.setAttribute('aria-label', label);
    this.setAttribute('aria-valuemin', String(this._min));
    this.setAttribute('aria-valuemax', String(this._max));
    this.setAttribute('aria-valuenow', String(this._current));
    this._syncGeometry(true);
    this._updateGlow();
  }

  get _normalized() {
    const span = this._max - this._min;
    return span > 0 ? (this._current - this._min) / span : 0;
  }

  _applyFromPointer(event) {
    const rect = this._track.getBoundingClientRect();
    const inner = Math.max(1, rect.width - 50); // RangePadding 25 each side
    const normalized = Math.min(1, Math.max(0, (event.clientX - rect.left - 25) / inner));
    this.value = this._min + normalized * (this._max - this._min);
    this.setAttribute('aria-valuenow', String(this._current));
    this._syncGeometry();
    this._pulse();
    this._updateTooltip();
    this._emitChange();
  }

  /** Left/right boxes track the nub (RoundedSliderBar.UpdateAfterChildren); nub eases at 250ms OutQuint. */
  _syncGeometry(instant = false) {
    const width = this._track.getBoundingClientRect().width;
    const inner = Math.max(1, width - 50);
    const centerX = 25 + this._normalized * inner;
    if (instant || this._dragging) this._nub.style.transitionDuration = '0ms';
    else this._nub.style.transitionDuration = '';
    this._nub.style.left = `${Math.round(centerX - 25)}px`;
    this._barLeft.style.width = `${Math.round(Math.max(0, centerX - 2))}px`;
    this._barRight.style.width = `${Math.round(Math.max(0, width - centerX - 2))}px`;
    if (instant) requestAnimationFrame(() => { this._nub.style.transitionDuration = ''; });
  }

  /** Glowing = hovered/dragged/focused and not disabled (RoundedSliderBar.updateGlow). */
  _updateGlow() {
    const glow = !this.hasAttribute('disabled')
      && (this._track.matches(':hover') || this._dragging || this.dataset.focused === 'true');
    this.dataset.glow = String(glow);
  }

  _updateTooltip() {
    const active = this.dataset.glow === 'true';
    this._tooltip.dataset.visible = String(active);
    if (active) {
      this._tooltip.textContent = this._format(this._current);
      const width = this._track.getBoundingClientRect().width;
      const inner = Math.max(1, width - 50);
      this._tooltip.style.left = `${Math.round(25 + this._normalized * inner)}px`;
    }
  }

  _format(value) {
    const step = this._step;
    const decimals = step < 1 ? Math.min(5, Math.ceil(-Math.log10(step))) : 0;
    // TimeSlider.cs:11-13 —— tooltip 追加 " ms"
    return `${value.toFixed(decimals)}${this.dataset.variant === 'time' ? ' ms' : ''}`;
  }

  /** Notch-tick visual substitute: 30ms throttled nub pulse, stronger pitch idea dropped. */
  _pulse(elastic = false) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    if (scale <= 0) return;
    const now = performance.now();
    if (!elastic && now - this._lastTickTime < 30) return;
    this._lastTickTime = now;
    this._tickAnimation?.cancel();
    this._tickAnimation = this._nub.animate(
      [{ transform: 'scale(1)' }, { transform: `scale(${elastic ? 1.3 : 1.12})` }, { transform: 'scale(1)' }],
      { duration: Math.max(1, Math.round((elastic ? 320 : 90) * scale)), easing: elastic ? 'cubic-bezier(0.34, 1.56, 0.64, 1)' : 'ease-out' },
    );
  }

  _emitChange() {
    this.dispatchEvent(new CustomEvent('ui-change', {
      bubbles: true, composed: true, detail: { value: this._current },
    }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-slider')) {
  customElements.define('ui-slider', UiSliderElement);
}

export default UiSliderElement;
