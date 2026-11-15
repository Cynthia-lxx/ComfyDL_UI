/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Ported from Screens/Menu/ButtonSystem.cs:424-528, MainMenuButton.cs:176-356,
 * ButtonArea.cs:42-132, FlowContainerWithOrigin.cs, OsuLogo.cs:291-452.
 * The "row of wedge buttons to the right of the hub" with its full interaction set:
 * 4 width states, beat-driven icon hover, additive press layer, hub tracking,
 * keyboard navigation and idle contraction. No upstream code/assets copied.
 */

import { intervalSync } from '../../05-motion/effects.js';

const STYLE_URL = new URL('./ui-action-rail.css', import.meta.url);

const DEFAULT_ITEMS = [
  { id: 'detail', label: 'Play', icon: '▶', colour: '#66ccff', min: 'primary', max: 'primary' },
  { id: 'secondary', label: 'Edit', icon: '✎', colour: '#ffcc22', min: 'primary', max: 'primary' },
  { id: 'browse', label: 'Browse', icon: '⌕', colour: '#88da20', min: 'primary', max: 'primary' },
  { id: 'hidden', label: 'Exit', icon: '×', colour: '#ff6666', min: 'primary', max: 'primary' },
];

const STATE_ORDER = { hidden: 0, initial: 1, primary: 2, detail: 3, group: 4, secondary: 5, transition: 6 };

const FALLBACK_STYLES = `
  :host { display: block; position: relative; }
  .row { display: flex; }
  .wedge { height: 100px; width: 160px; overflow: hidden; background: var(--ui-surface-raised, #333); color: var(--ui-text, #fff); }
`;

export class UiActionRailElement extends HTMLElement {
  static get observedAttributes() { return ['data-state', 'data-theme']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._hub = document.createElement('button');
    this._hub.className = 'hub';
    this._hub.type = 'button';
    this._hub.setAttribute('aria-label', 'Menu');
    this._hub.textContent = '●';

    this._row = document.createElement('div');
    this._row.className = 'row';
    this._row.setAttribute('role', 'toolbar');

    this._wrap = document.createElement('div');
    this._wrap.className = 'wrap';
    this._wrap.append(this._hub, this._row);
    this.shadowRoot.append(this._style, this._wrap);

    this._items = [];
    this._state = 'initial';
    this._hovered = null;
    this._beat = null;
    this._onLogoClick = () => this.setState(this._state === 'initial' ? 'primary' : 'initial');
    this._hub.addEventListener('click', this._onLogoClick);
  }

  connectedCallback() {
    this._loadStyles();
    if (this._items.length === 0) this.setItems(this._readLightItems() || DEFAULT_ITEMS);
    if (this.isConnected && this._state === 'initial') {
      window.setTimeout(() => this.setState('primary'), 150);
    }
    this._startBeat();
    this._wrap.addEventListener('keydown', this._onKeyDown);
  }

  disconnectedCallback() {
    this._hub.removeEventListener('click', this._onLogoClick);
    this._wrap.removeEventListener('keydown', this._onKeyDown);
    this._beat?.stop();
  }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* offline: inline fallback stays */ }
  }

  _readLightItems() {
    const nodes = [...this.querySelectorAll('[data-menu-item]')];
    if (nodes.length === 0) return null;
    return nodes.map((node) => ({
      id: node.dataset.menuItem || node.textContent.trim(),
      label: node.textContent.trim(),
      icon: node.dataset.icon || '•',
      colour: node.dataset.colour || 'var(--colour-1)',
      min: node.dataset.min || 'primary',
      max: node.dataset.max || 'primary',
    }));
  }

  /** @param {Array<{id,label,icon,colour,min,max}>} items */
  setItems(items) {
    this._items = items;
    this._row.replaceChildren();
    for (const [index, item] of items.entries()) {
      const wedge = document.createElement('button');
      wedge.type = 'button';
      wedge.className = 'wedge';
      wedge.dataset.state = 'contracted';
      wedge.dataset.id = item.id;
      wedge.style.setProperty('--ui-accent', item.colour);
      wedge.style.setProperty('--ui-index', String(index));
      wedge.innerHTML = `
        <span class="wedge__inner">
          <span class="wedge__hover"></span>
          <span class="wedge__content">
            <span class="wedge__icon" aria-hidden="true"></span>
            <span class="wedge__label"></span>
          </span>
        </span>`;
      wedge.querySelector('.wedge__icon').textContent = item.icon || '•';
      wedge.querySelector('.wedge__label').textContent = item.label || '';
      wedge.addEventListener('pointerenter', () => this._hover(wedge, item));
      wedge.addEventListener('pointerleave', () => this._unhover(wedge));
      wedge.addEventListener('pointerdown', () => { wedge.dataset.pressed = 'true'; });
      wedge.addEventListener('pointerup', () => { wedge.dataset.pressed = 'false'; });
      wedge.addEventListener('click', () => this._activate(wedge, item));
      this._row.appendChild(wedge);
    }
    this._applyState();
    return this;
  }

  _stateFor(item) {
    const state = this._state;
    if (state === 'initial' || state === 'hidden') return 'contracted';
    if (state === 'transition') return 'contracted-style1';
    const order = STATE_ORDER[state] ?? 2;
    const min = STATE_ORDER[item.min] ?? 2;
    const max = STATE_ORDER[item.max] ?? 2;
    if (order >= min && order <= max) return 'expanded';
    if (order < min) return 'contracted';
    return 'exploded';
  }

  _applyState() {
    for (const [index, item] of this._items.entries()) {
      const wedge = this._row.children[index];
      if (!wedge) continue;
      wedge.dataset.state = this._stateFor(item);
      wedge.setAttribute('aria-hidden', String(wedge.dataset.state !== 'expanded'));
      wedge.tabIndex = wedge.dataset.state === 'expanded' ? 0 : -1;
    }
  }

  setState(name) {
    if (!STATE_ORDER.hasOwnProperty(name)) return;
    this._state = name;
    this.dataset.state = name;
    this._applyState();
    this.dispatchEvent(new CustomEvent('ui-menu-state', { bubbles: true, composed: true, detail: { state: name } }));
    return this;
  }

  _hover(wedge, item) {
    if (wedge.dataset.state !== 'expanded') return;
    this._hovered = wedge;
    wedge.dataset.hovered = 'true';
    wedge.dataset.beating = 'true';
    wedge.dispatchEvent(new CustomEvent('ui-menu-hover', { bubbles: true, composed: true, detail: { id: item.id } }));
  }

  _unhover(wedge) {
    wedge.dataset.hovered = 'false';
    wedge.dataset.beating = 'false';
    if (this._hovered === wedge) this._hovered = null;
  }

  _activate(wedge, item) {
    if (wedge.dataset.state !== 'expanded') return;
    wedge.dataset.clicked = 'true';
    window.setTimeout(() => { wedge.dataset.clicked = 'false'; }, 800);
    this.dispatchEvent(new CustomEvent('ui-menu-press', {
      bubbles: true, composed: true, detail: { id: item.id, label: item.label, state: this._state },
    }));
  }

  _startBeat() {
    this._beat?.stop();
    this._beat = intervalSync(({ index, cycleLength }) => {
      const wedges = [...this._row.querySelectorAll('[data-pulsing="true"]')];
      for (const wedge of wedges) {
        wedge.style.setProperty('--ui-pulse-length', `${cycleLength}ms`);
        wedge.style.setProperty('--ui-pulse-rotation', `${index % 2 === 0 ? 8 : -8}deg`);
      }
    }, { bpm: 120 });
  }

  _onKeyDown = (event) => {
    const wedges = [...this._row.querySelectorAll('.wedge')].filter((w) => w.dataset.state === 'expanded');
    const current = wedges.indexOf(this.shadowRoot.activeElement);
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      const next = (current + (event.key === 'ArrowRight' ? 1 : -1) + wedges.length) % wedges.length;
      wedges[next]?.focus();
    } else if (event.key === 'Escape') {
      this.setState('initial');
    }
  };

  setTheme(name) { if (name) this.dataset.theme = name; return this; }

  unmount() {
    this._beat?.stop();
    this._row.replaceChildren();
    this.remove();
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-action-rail')) {
  customElements.define('ui-action-rail', UiActionRailElement);
}

export default UiActionRailElement;
