/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: ProgressBar.cs:24-36,57-93 (0..1 mapping, FillColour 150ms OutQuint,
 *   BackgroundColour immediate, allowSeek -> OnSeek/OnCommit, Seeking state);
 *   Bar.cs:13-59,78-116 (resize_duration 250ms InOutCubic, four BarDirection origins,
 *   clamp 0..1); ExpandingBar.cs:31-101 (Expanded 4px / Collapsed 2px, 400ms
 *   OutElastic expand / Out collapse, fade 1 <-> 0.5 at 100ms OutQuint).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-progress.css', import.meta.url);
const FALLBACK = `:host{display:block;height:6px;background:var(--ui-surface-sunken,#222);position:relative;overflow:hidden}.fill{height:100%;width:0;background:var(--ui-accent,#66ccff);transition:width 250ms cubic-bezier(0.65,0,0.35,1),height 250ms cubic-bezier(0.65,0,0.35,1)}`;

export class UiProgressElement extends HTMLElement {
  static get observedAttributes() { return ['value', 'direction', 'variant', 'seek', 'expanded']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._track = document.createElement('div');
    this._track.className = 'track';
    this._fill = document.createElement('div');
    this._fill.className = 'fill';
    this._track.appendChild(this._fill);
    this.shadowRoot.append(this._style, this._track);
    this._dragging = false;
    this._onDown = (event) => {
      if (!this.hasAttribute('seek')) return;
      this._dragging = true;
      this.dataset.seeking = 'true';
      this._track.setPointerCapture(event.pointerId);
      this._apply(event);
    };
    this._onMove = (event) => { if (this._dragging) this._apply(event); };
    this._onUp = () => {
      if (!this._dragging) return;
      this._dragging = false;
      delete this.dataset.seeking;
      this._emit('ui-progress-commit');
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this._track.addEventListener('pointerdown', this._onDown);
    this._track.addEventListener('pointermove', this._onMove);
    this._track.addEventListener('pointerup', this._onUp);
    this._track.addEventListener('pointercancel', this._onUp);
  }

  disconnectedCallback() {
    this._track.removeEventListener('pointerdown', this._onDown);
    this._track.removeEventListener('pointermove', this._onMove);
    this._track.removeEventListener('pointerup', this._onUp);
    this._track.removeEventListener('pointercancel', this._onUp);
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

  get value() { return Math.min(1, Math.max(0, Number(this.getAttribute('value')) || 0)); }
  set value(next) { this.setAttribute('value', String(Math.min(1, Math.max(0, Number(next) || 0)))); }

  _sync() {
    const direction = this.getAttribute('direction') || 'ltr';
    const variant = this.getAttribute('variant') || 'bar';
    this.dataset.direction = direction;
    this.dataset.variant = variant;
    this.dataset.expanded = String(this.getAttribute('expanded') !== 'false');
    this._fill.style.setProperty('--progress', `${this.value * 100}%`);
  }

  _apply(event) {
    const rect = this._track.getBoundingClientRect();
    const direction = this.dataset.direction;
    let ratio;
    if (direction === 'rtl') ratio = 1 - (event.clientX - rect.left) / rect.width;
    else if (direction === 'ttb') ratio = (event.clientY - rect.top) / rect.height;
    else if (direction === 'btt') ratio = 1 - (event.clientY - rect.top) / rect.height;
    else ratio = (event.clientX - rect.left) / rect.width;
    this.value = ratio;
    this._emit('ui-progress-seek');
  }

  collapse() { this.setAttribute('expanded', 'false'); }
  expand() { this.setAttribute('expanded', 'true'); }

  _emit(name) {
    this.dispatchEvent(new CustomEvent(name, {
      bubbles: true, composed: true, detail: { value: this.value, seeking: this._dragging },
    }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-progress')) {
  customElements.define('ui-progress', UiProgressElement);
}
export default UiProgressElement;
