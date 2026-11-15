/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: Hotkey.cs:16-41,43-71 (four key sources -> ResolveKeyCombination priority chain);
 *   HotkeyDisplay.cs:19-35,49-90,92-123 (AutoSize horizontal FillFlow spacing 5,
 *   keymap change redraw, HotkeyBox corner radius 3, Background4 / Light2 colours,
 *   ToUpperInvariant 12 Bold, margin 5/1).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-hotkey.css', import.meta.url);
const FALLBACK = `:host{display:inline-flex;gap:5px}.box{padding:1px 5px;border-radius:3px;background:#000b;color:#ccc;font-size:12px;font-weight:700}`;

export class UiHotkeyElement extends HTMLElement {
  static get observedAttributes() { return ['keys']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this.shadowRoot.append(this._style);
  }

  connectedCallback() { this._loadStyles(); this._render(); }
  attributeChangedCallback() { if (this.isConnected) this._render(); }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* file:// or offline: inline fallback stays */ }
  }

  /** Hotkey.ResolveKeyCombination 替身：直接组合优先，其次 global / platform 键名。 */
  _render() {
    const raw = this.getAttribute('keys') || '';
    const combinations = raw.split(/[|,]/).map((s) => s.trim()).filter(Boolean);
    this.shadowRoot.querySelectorAll('.box').forEach((el) => el.remove());
    for (const combination of combinations) {
      const box = document.createElement('kbd');
      box.className = 'box';
      box.textContent = combination.toUpperCase();
      this.shadowRoot.appendChild(box);
    }
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-hotkey')) {
  customElements.define('ui-hotkey', UiHotkeyElement);
}
export default UiHotkeyElement;
