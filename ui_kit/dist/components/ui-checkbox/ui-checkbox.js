/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/OsuCheckbox.cs:42-74 (Nub on the right, label padding-right
 *     Nub.DEFAULT_EXPANDED_SIZE + 10; disabled alpha 0.3 on label and nub)
 *   osu.Game/Graphics/UserInterface/OsuCheckbox.cs:90-100 (hover -> Nub.Glowing)
 *   osu.Game/Graphics/UserInterface/Nub.cs:162-178 (filled: width 1.0 / border 8.5 OutElasticHalf
 *     200ms; unfilled: width 0.75 / border 3 OutQuint 200ms; fill alpha 0/1)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-checkbox.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: inline-flex; align-items: center; }
  .label { padding-right: 60px; }
  .nub { width: 38px; height: 15px; border-radius: 8px; background: #66ccff; }
  :host([data-checked='true']) .nub { width: 50px; }
`;

export class UiCheckboxElement extends HTMLElement {
  static get observedAttributes() {
    return ['checked', 'disabled', 'label', 'data-i18n-key'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._label = document.createElement('span');
    this._label.className = 'label';
    this._label.setAttribute('part', 'label');

    this._nub = document.createElement('span');
    this._nub.className = 'nub';
    this._nub.setAttribute('part', 'nub');

    this.shadowRoot.append(this._style, this._label, this._nub);

    this._onClick = () => {
      if (this.hasAttribute('disabled')) return;
      this.checked = !this.checked;
      // OsuCheckbox.OnUserChange: check-on / check-off samples; visual pulse substitute.
      this._nub.animate(
        [{ transform: 'scale(1)' }, { transform: 'scale(1.08)' }, { transform: 'scale(1)' }],
        { duration: this._scaled(120), easing: 'ease-out' },
      );
      this.dispatchEvent(new CustomEvent('ui-change', {
        bubbles: true, composed: true, detail: { checked: this.checked },
      }));
    };
    this._onKeyDown = (event) => {
      if (this.hasAttribute('disabled')) return;
      if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        this._onClick();
      }
    };
    this._onI18nChange = () => this._sync();
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this.addEventListener('click', this._onClick);
    this.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('ui-i18n-change', this._onI18nChange);
  }

  disconnectedCallback() {
    this.removeEventListener('click', this._onClick);
    this.removeEventListener('keydown', this._onKeyDown);
    document.removeEventListener('ui-i18n-change', this._onI18nChange);
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

  get checked() { return this.hasAttribute('checked'); }
  set checked(value) { this.toggleAttribute('checked', Boolean(value)); }

  _sync() {
    const disabled = this.hasAttribute('disabled');
    this.dataset.checked = String(this.checked);
    this.dataset.disabled = String(disabled);
    this.tabIndex = disabled ? -1 : 0;
    this.setAttribute('role', 'switch');
    this.setAttribute('aria-checked', String(this.checked));
    const key = this.getAttribute('data-i18n-key');
    const dictionary = globalThis.UIReference?.i18n;
    const attr = this.getAttribute('label');
    const i18nText = key && dictionary && typeof dictionary[key] === 'string' ? dictionary[key] : '';
    const text = attr || i18nText || this.textContent?.trim() || '';
    this._label.textContent = text;
  }

  _scaled(ms) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    return Math.max(1, Math.round(ms * Math.abs(scale)));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-checkbox')) {
  customElements.define('ui-checkbox', UiCheckboxElement);
}

export default UiCheckboxElement;
