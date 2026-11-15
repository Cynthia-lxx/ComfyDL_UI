/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:91-127 (menu open/close: FadeIn/FadeOut 300ms
 *     OutQuint + ResizeHeightTo 300ms OutQuint; items container padding 5; corner radius 5)
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:79-81 (BackgroundColour=Background5,
 *     HoverColour=Light4, SelectionColour=Background3)
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:247-254 (item hover background: fade in 100ms
 *     OutQuint / fade out 600ms OutQuint)
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:326-335 (hover chevron: slide in 400ms
 *     OutQuint from -3px, fade out 200ms)
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:346-462 (header: height 40, padding 10,
 *     truncating label, chevron-down flips via ScaleTo(1,-1) 300ms OutQuint on open;
 *     hover -> Light4, else Background5; disabled alpha 0.3)
 *   osu.Game/Graphics/UserInterface/OsuDropdown.cs:166-189 (arrow-key preselection)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-dropdown.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: inline-block; position: relative; width: 200px; }
  .header { width: 100%; height: 40px; padding: 0 10px; border: 0; border-radius: 5px;
    background: var(--ui-surface-sunken, #222); color: var(--ui-text, #fff); font: inherit;
    display: flex; align-items: center; justify-content: space-between; cursor: pointer; }
  .menu { display: none; }
`;

/** @typedef {{value: string, label: string}} UiDropdownSelection */

export class UiDropdownElement extends HTMLElement {
  static get observedAttributes() {
    return ['value', 'disabled', 'placeholder', 'data-i18n-key'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._open = false;
    this._preselect = -1;
    this._items = [];
    this._menuAnimation = null;

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._header = document.createElement('button');
    this._header.className = 'header';
    this._header.setAttribute('part', 'header');
    this._header.setAttribute('aria-haspopup', 'listbox');

    this._label = document.createElement('span');
    this._label.className = 'label';

    this._chevron = document.createElement('span');
    this._chevron.className = 'chevron';
    this._chevron.setAttribute('aria-hidden', 'true');
    this._chevron.textContent = '▾';

    this._header.append(this._label, this._chevron);

    this._menu = document.createElement('div');
    this._menu.className = 'menu';
    this._menu.setAttribute('part', 'menu');
    this._menu.setAttribute('role', 'listbox');

    this.shadowRoot.append(this._style, this._header, this._menu);

    this._onHeaderClick = () => { if (!this.hasAttribute('disabled')) this.toggle(); };
    this._onHeaderKeyDown = (event) => {
      if (this.hasAttribute('disabled')) return;
      switch (event.key) {
        case 'ArrowDown': event.preventDefault(); if (!this._open) this.open(); else this._movePreselect(1); break;
        case 'ArrowUp': event.preventDefault(); if (!this._open) this.open(); else this._movePreselect(-1); break;
        case 'Enter': case ' ':
          event.preventDefault();
          if (this._open && this._preselect >= 0) this._selectIndex(this._preselect);
          else this.open();
          break;
        case 'Escape': if (this._open) { event.stopPropagation(); this.close(); } break;
        default: break;
      }
    };
    this._onOutsideClick = (event) => {
      if (this._open && event.composedPath && !event.composedPath().includes(this)) this.close();
    };
    this._onI18nChange = () => { this._readItems(); this._sync(); };
  }

  connectedCallback() {
    this._loadStyles();
    this._readItems();
    this._renderItems();
    this._sync();
    this._header.addEventListener('click', this._onHeaderClick);
    this._header.addEventListener('keydown', this._onHeaderKeyDown);
    document.addEventListener('pointerdown', this._onOutsideClick, true);
    document.addEventListener('ui-i18n-change', this._onI18nChange);
  }

  disconnectedCallback() {
    this._header.removeEventListener('click', this._onHeaderClick);
    this._header.removeEventListener('keydown', this._onHeaderKeyDown);
    document.removeEventListener('pointerdown', this._onOutsideClick, true);
    document.removeEventListener('ui-i18n-change', this._onI18nChange);
  }

  attributeChangedCallback() {
    if (this.isConnected) { this._readItems(); this._renderItems(); this._sync(); }
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

  get value() { return this.getAttribute('value') ?? ''; }

  _readItems() {
    const dictionary = globalThis.UIReference?.i18n;
    this._items = [...this.querySelectorAll(':scope > ui-option, :scope > [data-value]')].map((el) => {
      const value = el.getAttribute('value') ?? el.dataset.value ?? '';
      const key = el.getAttribute('data-i18n-key');
      const label = el.getAttribute('label') ?? el.textContent?.trim() ?? (key && dictionary?.[key]) ?? value;
      return { value, label };
    });
  }

  _renderItems() {
    this._menu.replaceChildren(...this._items.map((item, index) => {
      const row = document.createElement('div');
      row.className = 'item';
      row.setAttribute('role', 'option');
      row.dataset.index = String(index);
      const chev = document.createElement('span');
      chev.className = 'chev';
      chev.textContent = '▸';
      const text = document.createElement('span');
      text.className = 'text';
      text.textContent = item.label;
      row.append(chev, text);
      row.addEventListener('pointerenter', () => { this._preselect = index; this._paintPreselect(); });
      row.addEventListener('click', () => this._selectIndex(index));
      return row;
    }));
  }

  _sync() {
    const disabled = this.hasAttribute('disabled');
    this._header.disabled = disabled;
    this._header.setAttribute('aria-disabled', String(disabled));
    this.dataset.disabled = String(disabled);
    const key = this.getAttribute('data-i18n-key');
    const dictionary = globalThis.UIReference?.i18n;
    const selected = this._items.find((item) => item.value === this.value);
    const placeholder = this.getAttribute('placeholder')
      ?? (key && dictionary && typeof dictionary[key] === 'string' ? dictionary[key] : '');
    this._label.textContent = selected?.label ?? placeholder ?? '';
    this._menu.setAttribute('aria-activedescendant', selected ? `opt-${selected.value}` : '');
    for (const row of this._menu.querySelectorAll('.item')) {
      const item = this._items[Number(row.dataset.index)];
      row.dataset.selected = String(item?.value === this.value);
      row.id = `opt-${item?.value ?? row.dataset.index}`;
    }
  }

  get isOpen() { return this._open; }

  /** AnimateOpen/AnimateClose + UpdateSize: fade 300ms OutQuint + height resize 300ms OutQuint. */
  open() {
    if (this._open) return;
    this._open = true;
    this.dataset.open = 'true';
    this._preselect = Math.max(0, this._items.findIndex((item) => item.value === this.value));
    this._paintPreselect();
    const target = this._menu.scrollHeight;
    this._menuAnimation?.cancel();
    this._menuAnimation = this._menu.animate([
      { opacity: 0, height: '0px' },
      { opacity: 1, height: `${target}px` },
    ], { duration: this._scaled(300), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
    this._menuAnimation.finished.then(() => {
      if (this._open) { this._menu.style.height = 'auto'; this._menuAnimation?.cancel(); }
    }).catch(() => {});
    this._dispatch('ui-dropdown-open', {});
  }

  close() {
    if (!this._open) return;
    this._open = false;
    this.dataset.open = 'false';
    const from = this._menu.getBoundingClientRect().height;
    this._menuAnimation?.cancel();
    this._menuAnimation = this._menu.animate([
      { opacity: 1, height: `${from}px` },
      { opacity: 0, height: '0px' },
    ], { duration: this._scaled(300), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
    this._menuAnimation.finished.then(() => {
      if (!this._open) this._menuAnimation?.cancel();
    }).catch(() => {});
    this._preselect = -1;
    this._paintPreselect();
    this._dispatch('ui-dropdown-close', {});
  }

  toggle() { this._open ? this.close() : this.open(); }

  _scaled(ms) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    return Math.max(1, Math.round(ms * Math.abs(scale)));
  }

  /** Item hover background: 100ms in / 600ms out is CSS; this paints keyboard preselection. */
  _paintPreselect() {
    for (const row of this._menu.querySelectorAll('.item')) {
      row.dataset.preselect = String(Number(row.dataset.index) === this._preselect);
      if (Number(row.dataset.index) === this._preselect) row.scrollIntoView({ block: 'nearest' });
    }
  }

  _movePreselect(delta) {
    if (!this._items.length) return;
    this._preselect = Math.min(this._items.length - 1, Math.max(0, this._preselect + delta));
    this._paintPreselect();
  }

  _selectIndex(index) {
    const item = this._items[index];
    if (!item) return;
    this.setAttribute('value', item.value);
    this._sync();
    this.close();
    this._dispatch('ui-select', { value: item.value, label: item.label });
  }

  _dispatch(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { bubbles: true, composed: true, detail }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-dropdown')) {
  customElements.define('ui-dropdown', UiDropdownElement);
}

export default UiDropdownElement;
