/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Custom Element wrapper around 05-motion/layered-view.js.
 * Geometry CSS and presets are fetched from 05-motion/ so there is one source
 * of truth; the component itself only owns the host box and lifecycle.
 */

import { createLayeredView } from '../../05-motion/layered-view.js';

const GEOMETRY_URL = new URL('../../05-motion/layered-view.css', import.meta.url);
const PRESETS_URL = new URL('../../05-motion/layered-view.json', import.meta.url);
const HOST_URL = new URL('./ui-layered-view.css', import.meta.url);

const FALLBACK_CSS = `
  .lv-root { position: absolute; inset: 0; visibility: hidden; }
  .lv-layer { position: absolute; inset: 0; }
`;

export class UiLayeredViewElement extends HTMLElement {
  static get observedAttributes() { return ['preset', 'data-theme']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_CSS;
    this._stage = document.createElement('div');
    this._stage.className = 'lv-stage';
    this._stage.setAttribute('part', 'stage');
    this.shadowRoot.append(this._style, this._stage);
    this._ready = null;
    this._presets = {};
    this._view = null;
    this._rate = 1;
  }

  connectedCallback() {
    if (!this.hasAttribute('preset')) this.setAttribute('preset', 'catalogReveal');
    this._ready = this._load();
  }

  disconnectedCallback() { this._view?.unmount(); this._view = null; }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'preset' && oldValue && oldValue !== newValue && this._view) this.setPreset(newValue);
  }

  async _load() {
    const [css, presets, hostCss] = await Promise.all([
      fetch(GEOMETRY_URL).then((r) => (r.ok ? r.text() : '')).catch(() => ''),
      fetch(PRESETS_URL).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch(HOST_URL).then((r) => (r.ok ? r.text() : '')).catch(() => ''),
    ]);
    this._style.textContent = (hostCss || '') + '\n' + (css || FALLBACK_CSS);
    this._presets = presets?.presets || {};
    this._build(this.getAttribute('preset') || 'catalogReveal');
    return true;
  }

  _build(name) {
    this._view?.unmount();
    const preset = this._presets[name] || Object.values(this._presets)[0];
    if (!preset) { console.warn('[ui-layered-view] no presets available'); return; }
    this._view = createLayeredView(this._stage, {
      preset,
      rate: this._rate,
      onEvent: (type, detail) => {
        this.shadowRoot.dispatchEvent(new CustomEvent(`ui-layered-${type}`, {
          bubbles: true, composed: true, detail,
        }));
      },
    });
    this._stage.dataset.preset = name;
  }

  async setPreset(name) {
    if (this._ready) await this._ready;
    this.setAttribute('preset', name);
    this._build(name);
  }

  async open() { if (this._ready) await this._ready; return this._view?.open(); }
  close() { return this._view?.close(); }
  toggle() { return this._view?.toggle(); }
  replay() { this._view?.replay(); }
  seek(ms) { this._view?.seek(ms); }
  setRate(rate) { this._rate = rate; this._view?.setRate(rate); }
  setLayerVisible(id, visible) { this._view?.setLayerVisible(id, visible); }
  setTheme(name) { if (name) this.dataset.theme = name; return this; }

  get isOpen() { return Boolean(this._view?.isOpen); }
  get elapsed() { return this._view?.elapsed() || 0; }
  get animationCount() { return this._view?.animationCount || 0; }
  get layerCount() { return this._view?.layerCount || 0; }
  get layers() { return this._view?.layers || []; }
  get presets() { return Object.keys(this._presets); }

  unmount() { this._view?.unmount(); this._view = null; this.remove(); }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-layered-view')) {
  customElements.define('ui-layered-view', UiLayeredViewElement);
}

export default UiLayeredViewElement;
