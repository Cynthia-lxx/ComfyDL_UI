/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/BreadcrumbControl.cs:29-46 (height 32, item spacing 10;
 *     items after the selected one -> Hidden, items before -> Visible with their chevron
 *     faded in at 500ms OutQuint; the selected item's own chevron stays hidden)
 *   osu.Game/Graphics/UserInterface/BreadcrumbControl.cs:74-89 (hidden state: FadeOut 500ms
 *     OutQuint + ScaleTo(0.8, 1); visible: FadeIn + scale 1 — both 500ms OutQuint)
 *   osu.Game/Graphics/UserInterface/BreadcrumbControl.cs:95-110 (text 18px, vertical margin 8,
 *     right margin 10 + chevron 10, chevron 10px centre-right, alpha 0)
 *   osu.Game/Graphics/UserInterface/OsuTabControl.cs:121-146 (active text white+bold, inactive
 *     AccentColour, hover/active bar fade 500ms OutQuint)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-breadcrumb.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: block; }
  .row { display: flex; align-items: center; }
`;

export class UiBreadcrumbElement extends HTMLElement {
  static get observedAttributes() {
    return ['value'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    // innerHTML 挂载时 attributeChangedCallback 先于 connectedCallback 触发（元素已连接），
    // _items 必须在构造器就绪，否则 _paint -> _selectedIndex 会读到 undefined。
    this._items = [];

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._row = document.createElement('div');
    this._row.className = 'row';
    this._row.setAttribute('part', 'row');
    this._row.setAttribute('role', 'menubar');

    this.shadowRoot.append(this._style, this._row);

    this._onRowKeyDown = (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const visible = this._items.map((item, index) => (index <= this._selectedIndex ? index : -1))
        .filter((index) => index >= 0);
      if (!visible.length) return;
      const currentVisible = visible.indexOf(this._selectedIndex);
      let next = currentVisible;
      if (event.key === 'ArrowLeft') next = Math.max(0, currentVisible - 1);
      else if (event.key === 'ArrowRight') next = Math.min(visible.length - 1, currentVisible + 1);
      else if (event.key === 'Home') next = 0;
      else next = visible.length - 1;
      this._select(visible[next], true);
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
    if (this.isConnected) this._paint();
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

  get value() { return this.getAttribute('value') ?? this._items[0]?.value ?? ''; }

  get _selectedIndex() {
    const index = this._items.findIndex((item) => item.value === this.value);
    return index === -1 ? 0 : index;
  }

  _readItems() {
    const dictionary = globalThis.UIReference?.i18n;
    this._items = [...this.querySelectorAll(':scope > ui-crumb, :scope > [data-value]')].map((el) => {
      const value = el.getAttribute('value') ?? el.dataset.value ?? '';
      const key = el.getAttribute('data-i18n-key');
      const label = el.getAttribute('label') ?? el.textContent?.trim() ?? (key && dictionary?.[key]) ?? value;
      return { value, label };
    });
  }

  _render() {
    this._row.replaceChildren(...this._items.map((item, index) => {
      const crumb = document.createElement('button');
      crumb.type = 'button';
      crumb.className = 'crumb';
      crumb.setAttribute('role', 'menuitem');
      const text = document.createElement('span');
      text.className = 'text';
      text.textContent = item.label;
      const chevron = document.createElement('span');
      chevron.className = 'chevron';
      chevron.textContent = '▸';
      chevron.setAttribute('aria-hidden', 'true');
      crumb.append(text, chevron);
      crumb.addEventListener('click', () => this._select(index, true));
      return crumb;
    }));
    this._paint();
  }

  _paint() {
    const selected = this._selectedIndex;
    this._items.forEach((item, index) => {
      const crumb = this._row.children[index];
      // BreadcrumbControl.cs:42-43 — after selected: hidden; before/at: visible, chevron on
      // strictly-earlier crumbs only (tIndex >= tabIndex -> chevron alpha 0).
      crumb.dataset.hidden = String(index > selected);
      crumb.querySelector('.chevron').dataset.visible = String(index < selected);
      crumb.dataset.active = String(index === selected);
      crumb.tabIndex = index === selected ? 0 : -1;
      crumb.setAttribute('aria-selected', String(index === selected));
    });
  }

  _select(index, emit) {
    const item = this._items[index];
    if (!item) return;
    this.setAttribute('value', item.value);
    this._paint();
    if (emit) {
      this.dispatchEvent(new CustomEvent('ui-select', {
        bubbles: true, composed: true, detail: { value: item.value, label: item.label },
      }));
    }
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-breadcrumb')) {
  customElements.define('ui-breadcrumb', UiBreadcrumbElement);
}

export default UiBreadcrumbElement;
