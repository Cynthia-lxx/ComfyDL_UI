/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: ShakeContainer.cs:17,23 + DrawableExtensions.cs:19-37 (80ms default, magnitude 8,
 *   +8 40ms OutSine / -8,+8,-8 80ms InOutSine / 0 40ms InSine);
 *   ParallaxContainer.cs:20,29,63-92 (ParallaxAmount 0.02, damp 0.999, scale 1+|amount|,
 *   position 100ms OutQuint, scale 1000ms OutQuint, negative amount reverses);
 *   OsuHoverContainer.cs:16,24-38,43-84 (FADE_DURATION 500 OutQuint, IdleColour white,
 *   HoverColour yellow, Enabled gating).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-motion-container.css', import.meta.url);
const FALLBACK = `:host{display:block;position:relative}.stage{display:block;transform-origin:center}`;

export class UiMotionContainerElement extends HTMLElement {
  static get observedAttributes() { return ['variant', 'amount', 'duration']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._stage = document.createElement('div');
    this._stage.className = 'stage';
    this._stage.appendChild(document.createElement('slot'));
    this.shadowRoot.append(this._style, this._stage);
    this._onMove = (event) => {
      if (this.dataset.variant !== 'parallax') return;
      const rect = this.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      const y = (event.clientY - rect.top) / rect.height - 0.5;
      const amount = Number(this.getAttribute('amount')) || 0.02;
      this._stage.style.setProperty('--px', `${(-x * amount * 100).toFixed(2)}px`);
      this._stage.style.setProperty('--py', `${(-y * amount * 100).toFixed(2)}px`);
    };
    this._onLeave = () => {
      this._stage.style.setProperty('--px', '0px');
      this._stage.style.setProperty('--py', '0px');
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this.addEventListener('pointermove', this._onMove);
    this.addEventListener('pointerleave', this._onLeave);
  }

  disconnectedCallback() {
    this.removeEventListener('pointermove', this._onMove);
    this.removeEventListener('pointerleave', this._onLeave);
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

  _sync() {
    const variant = this.getAttribute('variant') || 'shake';
    this.dataset.variant = variant;
    const amount = Number(this.getAttribute('amount')) || 0.02;
    this._stage.style.setProperty('--parallax-scale', String(1 + Math.abs(amount)));
  }

  /** ShakeContainer：8px 四段摆动，默认 80ms；总时长 <160ms 时跳过的策略由 duration 控制。 */
  shake(duration = 80) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    const d = Math.max(1, Math.round(duration * Math.abs(scale)));
    if (d * 4 < 160) return;
    const frames = [
      { transform: 'translateX(8px)', offset: 0 },
      { transform: 'translateX(-8px)', offset: 0.125 },
      { transform: 'translateX(8px)', offset: 0.375 },
      { transform: 'translateX(-8px)', offset: 0.625 },
      { transform: 'translateX(0px)', offset: 1 },
    ];
    this._stage.animate(frames, { duration: d * 4, easing: 'ease-in-out', fill: 'none' });
    this.dispatchEvent(new CustomEvent('ui-shake', { bubbles: true, composed: true }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-motion-container')) {
  customElements.define('ui-motion-container', UiMotionContainerElement);
}
export default UiMotionContainerElement;
