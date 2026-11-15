/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/OsuTabControl.cs:31,64 (HORIZONTAL_SPACING 10)
 *   osu.Game/Graphics/UserInterface/OsuTabControl.cs:47-53,88-101 (bottom 1px strip, width eases
 *     to the tab row width at 500ms OutQuint; invisible unless StripColour is set)
 *   osu.Game/Graphics/UserInterface/OsuTabControl.cs:121-146,209-213 (TRANSITION_LENGTH 500:
 *     hover/active -> Bar fade in + text white; unhover -> Bar fade out + text accent;
 *     activated -> text bold; inactive text uses AccentColour)
 *   osu.Game/Graphics/UserInterface/OsuTabControl.cs:177-184 (tab text 14px, 5px vertical margins)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-tabs.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: block; }
  .tabs { display: flex; gap: 10px; }
  .tab { padding: 5px 0; font: inherit; color: var(--ui-accent, #66ccff); background: none; border: 0; cursor: pointer; }
`;

export class UiTabsElement extends HTMLElement {
  static get observedAttributes() {
    return ['value', 'strip'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._tabs = [];

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._row = document.createElement('div');
    this._row.className = 'tabs';
    this._row.setAttribute('part', 'tabs');
    this._row.setAttribute('role', 'tablist');

    this._strip = document.createElement('span');
    this._strip.className = 'strip';

    this.shadowRoot.append(this._style, this._row, this._strip);

    this._onRowKeyDown = (event) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(event.key) || !this._tabs.length) return;
      event.preventDefault();
      const current = this._tabs.findIndex((tab) => tab.value === this.value);
      let next = current;
      if (event.key === 'ArrowLeft') next = Math.max(0, current - 1);
      else if (event.key === 'ArrowRight') next = Math.min(this._tabs.length - 1, current + 1);
      else if (event.key === 'Home') next = 0;
      else next = this._tabs.length - 1;
      this._select(next, true);
      this._tabs[next]?.button.focus();
    };
    this._onI18nChange = () => { this._readItems(); this._render(); };
  }

  connectedCallback() {
    this._loadStyles();
    this._readItems();
    this._render();
    this._row.addEventListener('keydown', this._onRowKeyDown);
    document.addEventListener('ui-i18n-change', this._onI18nChange);
  }

  disconnectedCallback() {
    this._row.removeEventListener('keydown', this._onRowKeyDown);
    document.removeEventListener('ui-i18n-change', this._onI18nChange);
  }

  attributeChangedCallback() {
    if (this.isConnected) { this._sync(); this._paintStrip(); }
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

  get value() { return this.getAttribute('value') ?? this._tabs[0]?.value ?? ''; }

  _readItems() {
    const dictionary = globalThis.UIReference?.i18n;
    this._tabs = [...this.querySelectorAll(':scope > ui-tab, :scope > [data-value]')].map((el) => {
      const value = el.getAttribute('value') ?? el.dataset.value ?? '';
      const key = el.getAttribute('data-i18n-key');
      const label = el.getAttribute('label') ?? el.textContent?.trim() ?? (key && dictionary?.[key]) ?? value;
      return { value, label };
    });
  }

  _render() {
    this._row.replaceChildren(...this._tabs.map((tab, index) => {
      const button = document.createElement('button');
      button.className = 'tab';
      button.setAttribute('role', 'tab');
      button.type = 'button';
      const text = document.createElement('span');
      text.className = 'text';
      text.textContent = tab.label;
      const bar = document.createElement('span');
      bar.className = 'bar';
      button.append(text, bar);
      button.addEventListener('click', () => this._select(index, true));
      return button;
    }));
    this._sync();
    requestAnimationFrame(() => this._paintStrip(true));
  }

  _sync() {
    this.dataset.variant = this.getAttribute('variant') || 'standard';
    const active = this.value;
    this._tabs.forEach((tab, index) => {
      const button = this._row.children[index];
      button.dataset.active = String(tab.value === active);
      button.setAttribute('aria-selected', String(tab.value === active));
      button.tabIndex = tab.value === active ? 0 : -1;
    });
  }

  _select(index, emit) {
    const tab = this._tabs[index];
    if (!tab) return;
    this.setAttribute('value', tab.value);
    this._sync();
    this._paintStrip();
    if (emit) {
      this.dispatchEvent(new CustomEvent('ui-select', {
        bubbles: true, composed: true, detail: { value: tab.value, label: tab.label },
      }));
    }
  }

  /** Strip: bottom 1px line whose width eases to the tab row width (500ms OutQuint). */
  _paintStrip(instant = false) {
    if (!this.hasAttribute('strip')) return;
    const activeButton = [...this._row.children].find((b) => b.dataset.active === 'true')
      ?? this._row.children[0];
    if (!activeButton) return;
    const rowRect = this._row.getBoundingClientRect();
    const rect = activeButton.getBoundingClientRect();
    this._strip.style.transitionDuration = instant ? '0ms' : '';
    this._strip.style.left = `${Math.round(rect.left - rowRect.left)}px`;
    this._strip.style.width = `${Math.round(rect.width)}px`;
    if (instant) requestAnimationFrame(() => { this._strip.style.transitionDuration = ''; });
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-tabs')) {
  customElements.define('ui-tabs', UiTabsElement);
}

export default UiTabsElement;
