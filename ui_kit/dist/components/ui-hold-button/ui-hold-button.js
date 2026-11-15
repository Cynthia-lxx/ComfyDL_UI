/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/Containers/HoldToConfirmContainer.cs:27 (DANGEROUS_HOLD_ACTIVATION_DELAY 500)
 *   osu.Game/Graphics/Containers/HoldToConfirmContainer.cs:91-98 (BeginConfirm: progress -> 1 over
 *     delay * (1 - current), Easing.Out)
 *   osu.Game/Graphics/Containers/HoldToConfirmContainer.cs:103-114 (AbortConfirm: hold current
 *     progress for 200ms, then decay to 0 over 200ms Easing.InSine)
 *   osu.Game/Graphics/Containers/HoldToConfirmContainer.cs:120-124 (Confirm fires the action once)
 *   osu.Game/Overlays/HoldToConfirmOverlay.cs:52-58 (visual feedback tracks Progress directly)
 * The activation delay reads the kit setting --hold-activation-delay (200ms default);
 * dangerous actions use the fixed 500ms. Ring drawn as an SVG stroke-progress circle.
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-hold-button.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: inline-block; }
  .wrap { position: relative; width: 64px; height: 64px; }
  button { width: 52px; height: 52px; margin: 6px; border-radius: 50%;
    background: var(--ui-surface-raised, #333); color: var(--ui-text, #fff);
    border: var(--border-width, 2px) solid var(--ui-border, #444); font: inherit; cursor: pointer; }
`;

const RADIUS = 29;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const HOLD_FADEOUT_DELAY = 200; // fadeout_delay
const DANGEROUS_DELAY = 500;

const easeOutQuad = (t) => 1 - (1 - t) * (1 - t);
const easeInSine = (t) => 1 - Math.cos((t * Math.PI) / 2);

export class UiHoldButtonElement extends HTMLElement {
  static get observedAttributes() {
    return ['label', 'disabled', 'dangerous'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._progress = 0;
    this._phase = 'idle'; // idle | confirming | holding | fired
    this._raf = 0;

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._wrap = document.createElement('div');
    this._wrap.className = 'wrap';

    this._button = document.createElement('button');
    this._button.setAttribute('part', 'button');
    this._button.setAttribute('type', 'button');

    this._label = document.createElement('span');
    this._label.className = 'label';
    this._button.appendChild(this._label);

    const svgNs = 'http://www.w3.org/2000/svg';
    this._svg = document.createElementNS(svgNs, 'svg');
    this._svg.setAttribute('viewBox', '0 0 64 64');
    this._svg.setAttribute('aria-hidden', 'true');
    this._svg.classList.add('ring');
    this._trackCircle = document.createElementNS(svgNs, 'circle');
    this._trackCircle.setAttribute('cx', '32');
    this._trackCircle.setAttribute('cy', '32');
    this._trackCircle.setAttribute('r', String(RADIUS));
    this._progressCircle = this._trackCircle.cloneNode();
    this._progressCircle.classList.add('progress');
    this._progressCircle.style.strokeDasharray = `${CIRCUMFERENCE}`;
    this._progressCircle.style.strokeDashoffset = `${CIRCUMFERENCE}`;
    this._svg.append(this._trackCircle, this._progressCircle);

    this._wrap.append(this._button, this._svg);
    this.shadowRoot.append(this._style, this._wrap);

    this._onDown = () => this._beginConfirm();
    this._onUp = () => this._abortConfirm();
    this._onKeyDown = (event) => {
      if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) {
        event.preventDefault();
        this._beginConfirm();
      }
    };
    this._onKeyUp = (event) => {
      if (event.key === ' ' || event.key === 'Enter') this._abortConfirm();
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this._button.addEventListener('pointerdown', this._onDown);
    this._button.addEventListener('pointerup', this._onUp);
    this._button.addEventListener('pointerleave', this._onUp);
    this._button.addEventListener('pointercancel', this._onUp);
    this._button.addEventListener('keydown', this._onKeyDown);
    this._button.addEventListener('keyup', this._onKeyUp);
  }

  disconnectedCallback() {
    this._button.removeEventListener('pointerdown', this._onDown);
    this._button.removeEventListener('pointerup', this._onUp);
    this._button.removeEventListener('pointerleave', this._onUp);
    this._button.removeEventListener('pointercancel', this._onUp);
    this._button.removeEventListener('keydown', this._onKeyDown);
    this._button.removeEventListener('keyup', this._onKeyUp);
    cancelAnimationFrame(this._raf);
  }

  attributeChangedCallback() {
    if (this.isConnected) this._sync();
  }

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

  _activationDelay() {
    if (this.hasAttribute('dangerous')) return DANGEROUS_DELAY;
    const raw = Number.parseFloat(getComputedStyle(this).getPropertyValue('--hold-activation-delay'));
    return Number.isFinite(raw) && raw > 0 ? raw : 200;
  }

  /** HoldToConfirmContainer.BeginConfirm —— 从当前进度以 Out 缓动推进到 1，时长与剩余进度成正比。 */
  _beginConfirm() {
    if (this.hasAttribute('disabled') || this._phase === 'confirming' || this._phase === 'fired') return;
    this._phase = 'confirming';
    this._button.dataset.holding = 'true';
    const frameStart = performance.now();
    const from = this._progress;
    const duration = Math.max(1, this._scaled(this._activationDelay() * (1 - from)));
    const step = (now) => {
      if (this._phase !== 'confirming') return;
      const t = Math.min(1, (now - frameStart) / duration);
      this._progress = from + (1 - from) * easeOutQuad(t);
      this._render();
      if (t >= 1) {
        this._phase = 'fired'; // Confirm(): fire once
        this._button.dataset.holding = 'false';
        this._dispatch('ui-confirm', { dangerous: this.hasAttribute('dangerous') });
        this._render();
        return;
      }
      this._raf = requestAnimationFrame(step);
    };
    this._raf = requestAnimationFrame(step);
  }

  /** HoldToConfirmContainer.AbortConfirm —— 保持当前进度 200ms，再以 InSine 衰减到 0。 */
  _abortConfirm() {
    this._button.dataset.holding = 'false';
    if (this._phase === 'confirming') {
      this._phase = 'holding';
      const frameStart = performance.now();
      const from = this._progress;
      const step = (now) => {
        if (this._phase !== 'holding') return;
        const elapsed = now - frameStart;
        if (elapsed < HOLD_FADEOUT_DELAY) {
          this._progress = from; // hold at the current value
        } else {
          const t = Math.min(1, (elapsed - HOLD_FADEOUT_DELAY) / HOLD_FADEOUT_DELAY);
          this._progress = from * (1 - easeInSine(t));
          if (t >= 1) {
            this._phase = 'idle';
            this._progress = 0;
            this._render();
            return;
          }
        }
        this._render();
        this._raf = requestAnimationFrame(step);
      };
      this._raf = requestAnimationFrame(step);
    } else if (this._phase === 'fired') {
      // 释放后重新武装（演示语义；上游由 AllowMultipleFires 外部决定）
      this._phase = 'idle';
      this._progress = 0;
      this._render();
    }
  }

  _render() {
    this._progressCircle.style.strokeDashoffset = `${CIRCUMFERENCE * (1 - this._progress)}`;
    this._label.textContent = this._phase === 'fired' ? '✓' : (this.getAttribute('label') ?? '');
  }

  _sync() {
    const disabled = this.hasAttribute('disabled');
    this._button.disabled = disabled;
    this._button.setAttribute('aria-disabled', String(disabled));
    this._button.setAttribute('role', 'button');
    const key = this.getAttribute('data-i18n-key');
    const dictionary = globalThis.UIReference?.i18n;
    const attr = this.getAttribute('label');
    const i18nText = key && dictionary && typeof dictionary[key] === 'string' ? dictionary[key] : '';
    const label = attr || i18nText || '';
    this.setAttribute('aria-label', label || 'hold to confirm');
    this._render();
  }

  _scaled(ms) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    return Math.max(1, Math.round(ms * Math.abs(scale)));
  }

  _dispatch(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { bubbles: true, composed: true, detail }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-hold-button')) {
  customElements.define('ui-hold-button', UiHoldButtonElement);
}

export default UiHoldButtonElement;
