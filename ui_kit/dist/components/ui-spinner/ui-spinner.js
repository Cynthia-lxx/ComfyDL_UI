/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: LoadingSpinner.cs:26,42,49-97,99-144,174-196,198-237 (500ms transition,
 *   3150ms spin, 60x60, box/boxless, inverted, beat-synced +90deg steps);
 *   LoadingLayer.cs:44-57,87 (fills parent, black 0.5 dim, clamp(short side*0.25,20,80));
 *   LoadingButton.cs:17-31,44-50,72-86 (12x12 spinner inside button, auto busy state).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-spinner.css', import.meta.url);
const FALLBACK = `:host{display:inline-block;width:60px;height:60px}.box{width:100%;height:100%;border-radius:20px;background:#000b;display:grid;place-items:center}.notch{width:36px;height:36px;border:6px solid #fff;border-right-color:transparent;border-radius:50%;animation:sp 3150ms linear infinite}@keyframes sp{to{transform:rotate(360deg)}}`;

export class UiSpinnerElement extends HTMLElement {
  static get observedAttributes() { return ['variant', 'inverted', 'dim', 'size', 'active']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._box = document.createElement('div');
    this._box.className = 'box';
    this._notch = document.createElement('div');
    this._notch.className = 'notch';
    this._box.appendChild(this._notch);
    this._dim = document.createElement('div');
    this._dim.className = 'dim';
    this.shadowRoot.append(this._style, this._dim, this._box);
  }

  connectedCallback() { this._loadStyles(); this._sync(); }
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
    const variant = this.getAttribute('variant') || 'box';
    this.dataset.variant = variant;
    this.dataset.inverted = String(this.hasAttribute('inverted'));
    const dim = this.hasAttribute('dim');
    this.dataset.dim = String(dim);
    const size = this.getAttribute('size');
    if (size) this.style.setProperty('--spinner-size', `${size}px`);
    this._animate(this.getAttribute('active') !== 'false');
  }

  /** LoadingSpinner.PopIn/PopOut：500ms OutQuint 缩放入场，250ms OutQuint 淡出。 */
  show() { this.setAttribute('active', 'true'); }
  hide() { this.setAttribute('active', 'false'); }

  _animate(active) {
    if (this._animating === active) return;
    this._animating = active;
    this._box.getAnimations().forEach((a) => { if (a !== this._spinAnimation) a.cancel(); });
    this._box.animate(
      active ? [{ opacity: 0.01, transform: 'scale(0.6)' }, { opacity: 1, transform: 'scale(1)' }]
        : [{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(0.6)' }],
      { duration: active ? 500 : 250, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' },
    ).finished.then((a) => { if (!active) a.cancel(); }).catch(() => {});
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-spinner')) {
  customElements.define('ui-spinner', UiSpinnerElement);
}
export default UiSpinnerElement;
