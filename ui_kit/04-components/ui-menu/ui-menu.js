/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/OsuMenu.cs:21-22 (DELAY_BEFORE_FADE_OUT 50, FADE_DURATION 280)
 *   osu.Game/Graphics/UserInterface/OsuMenu.cs:41-47 (background black 50%, corner radius 4,
 *     items padding 5; submenus open to the right, vertical)
 *   osu.Game/Graphics/UserInterface/OsuMenu.cs:68-99 (AnimateOpen FadeIn 280 OutQuint;
 *     AnimateClose delay 50 + FadeOut 280 OutQuint; ResizeHeightTo 300 OutQuint)
 *   osu.Game/Graphics/UserInterface/OsuMenu.cs:132-153 (spacer: scaleY 0.6, 2px divider)
 *   osu.Game/Graphics/UserInterface/DrawableOsuMenuItem.cs:25-28 (margins 10/4, text 17,
 *     transition 80ms), :44-45 (hover bg #172023), :58-68 (8px right chevron for submenus),
 *     :90-104 (standard white / destructive red / highlighted #ffcc22), :136 (disabled alpha 0.2),
 *     :138-147 (hover swaps normal->bold text, 80ms OutQuint, no width jump)
 *   osu.Game/Graphics/UserInterface/OsuContextMenu.cs:21-32 (radius 5, shadow black 10% r4,
 *     items vertical padding 4, MaxHeight 250, opaque ContextMenuGray background)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-menu.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { position: fixed; inset: 0; pointer-events: none; visibility: hidden; z-index: 2100; }
  .root { position: absolute; background: var(--ui-surface-raised, #333); display: none; }
`;

const FADE_DURATION = 280;
const DELAY_BEFORE_FADE_OUT = 50;

export class UiMenuElement extends HTMLElement {
  static get observedAttributes() {
    return ['for'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._open = false;

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._root = document.createElement('div');
    this._root.className = 'root';
    this._root.setAttribute('part', 'root');
    this._root.setAttribute('role', 'menu');

    this.shadowRoot.append(this._style, this._root);

    this._onTriggerClick = () => this.toggle();
    this._onOutsidePointerDown = (event) => {
      if (!this._open) return;
      // host 是全屏透明层，composedPath 里天然包含它，不能据此判断"点在菜单内"。
      // 只有命中根面板 / 子菜单 / 触发按钮才算内点。
      const path = event.composedPath();
      const inside = path.some((node) => node === this._trigger
        || node.classList?.contains('root') || node.classList?.contains('submenu'));
      if (!inside) this.close();
    };
    this._onKeyDown = (event) => {
      if (event.key === 'Escape' && this._open) {
        event.stopPropagation();
        this.close();
      }
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._build();
    this._bindTrigger();
    document.addEventListener('pointerdown', this._onOutsidePointerDown, true);
    document.addEventListener('keydown', this._onKeyDown, true);
  }

  disconnectedCallback() {
    document.removeEventListener('pointerdown', this._onOutsidePointerDown, true);
    document.removeEventListener('keydown', this._onKeyDown, true);
    this._unbindTrigger();
  }

  attributeChangedCallback() {
    if (this.isConnected) this._bindTrigger();
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

  get isOpen() { return this._open; }

  _bindTrigger() {
    if (this._trigger) this._trigger.removeEventListener('click', this._onTriggerClick);
    this._trigger = null;
    const selector = this.getAttribute('for');
    if (!selector) return;
    this._trigger = document.getElementById(selector);
    this._trigger?.addEventListener('click', this._onTriggerClick);
  }

  _unbindTrigger() {
    this._trigger?.removeEventListener('click', this._onTriggerClick);
    this._trigger = null;
  }

  /** Build the menu tree from light DOM: <ui-item label type hotkey disabled> (nested ui-item = submenu), <ui-spacer>. */
  _build() {
    this._root.replaceChildren();
    this._buildLevel(this._root, [...this.children].filter((el) => ['UI-ITEM', 'UI-SPACER'].includes(el.tagName)));
  }

  _buildLevel(container, items) {
    for (const el of items) {
      if (el.tagName === 'UI-SPACER') {
        const spacer = document.createElement('div');
        spacer.className = 'spacer';
        container.appendChild(spacer);
        continue;
      }
      const dictionary = globalThis.UIReference?.i18n;
      const key = el.getAttribute('data-i18n-key');
      const label = el.getAttribute('label') ?? (key && dictionary?.[key]) ?? el.textContent?.trim() ?? '';
      const row = document.createElement('div');
      row.className = 'item';
      row.setAttribute('role', 'menuitem');
      row.dataset.type = el.getAttribute('type') || 'standard';
      if (el.hasAttribute('disabled')) row.dataset.disabled = 'true';

      // StatefulMenuItem / ToggleMenuItem / TernaryStateMenuItem：
      // 10px 勾选槽 + 点击不关闭菜单（DrawableStatefulMenuItem.cs:15）
      const itemType = row.dataset.type;
      const stateful = itemType === 'toggle' || itemType === 'ternary';
      if (stateful) {
        const check = document.createElement('span');
        check.className = 'check';
        const states = itemType === 'toggle' ? ['false', 'true'] : ['false', 'indeterminate', 'true'];
        const icons = { false: '', indeterminate: '◉', true: '✓' };
        let state = el.getAttribute('state') || 'false';
        const paint = () => { check.textContent = icons[state] ?? ''; };
        paint();
        row.appendChild(check);
        row.addEventListener('click', () => {
          state = states[(states.indexOf(state) + 1) % states.length];
          paint();
          this._dispatch('ui-menu-state', { value: el.getAttribute('value') ?? label, state, type: itemType });
          // 关键：状态项不关闭菜单（CloseMenuOnClick = false）
        });
        container.appendChild(row);
        continue;
      }

      const text = document.createElement('span');
      text.className = 'text';
      const normal = document.createElement('span');
      normal.className = 'normal';
      normal.textContent = label;
      const bold = document.createElement('span');
      bold.className = 'bold';
      bold.textContent = label;
      text.append(normal, bold);
      row.appendChild(text);

      const hotkey = el.getAttribute('hotkey');
      if (hotkey) {
        const kbd = document.createElement('kbd');
        kbd.className = 'hotkey';
        kbd.textContent = hotkey;
        row.appendChild(kbd);
      }

      const children = [...el.children].filter((child) => child.tagName === 'UI-ITEM');
      if (children.length) {
        const chevron = document.createElement('span');
        chevron.className = 'chev';
        chevron.textContent = '▸';
        row.appendChild(chevron);
        const submenu = document.createElement('div');
        submenu.className = 'submenu';
        submenu.setAttribute('role', 'menu');
        this._buildLevel(submenu, children);
        row.appendChild(submenu);
        row.addEventListener('pointerenter', () => this._openSubmenu(row));
        // 子菜单是 row 的子节点：pointerleave 只在真正离开（含子菜单）时触发
        row.addEventListener('pointerleave', () => this._closeSubmenu(row));
      } else if (!row.dataset.disabled) {
        row.addEventListener('click', () => {
          this._dispatch('ui-select', { value: el.getAttribute('value') ?? label, label });
          this.close();
        });
      }
      container.appendChild(row);
    }
  }

  _openSubmenu(row) {
    for (const sibling of row.parentElement.children) {
      if (sibling !== row) this._closeSubmenu(sibling);
    }
    const submenu = row.querySelector(':scope > .submenu');
    if (!submenu || submenu.dataset.open === 'true') return;
    submenu.dataset.open = 'true';
    submenu.style.visibility = 'hidden';
    submenu.style.display = 'block';
    const rect = submenu.getBoundingClientRect();
    // 视口夹取：右侧放不下则翻到左侧（上游子菜单锚定 TopRight）
    if (rect.right > window.innerWidth) submenu.dataset.flip = 'true';
    submenu.style.display = '';
    submenu.style.visibility = '';
    submenu.animate([
      { opacity: 0 },
      { opacity: 1 },
    ], { duration: this._scaled(FADE_DURATION), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
  }

  _closeSubmenu(row) {
    const submenu = row.querySelector?.(':scope > .submenu');
    if (!submenu || submenu.dataset.open !== 'true') return;
    submenu.dataset.open = 'false';
    for (const child of submenu.querySelectorAll('.submenu')) child.dataset.open = 'false';
  }

  /** AnimateOpen/Close: fade 280ms OutQuint with a 50ms close delay; position below the trigger. */
  open() {
    if (this._open) return;
    const trigger = this._trigger;
    this._open = true;
    this.dataset.open = 'true';
    this.style.visibility = 'visible';
    this._root.style.display = 'block';
    this._root.style.visibility = 'hidden';
    const panelRect = this._root.getBoundingClientRect();
    let x = 0;
    let y = 0;
    if (trigger) {
      const rect = trigger.getBoundingClientRect();
      x = rect.left;
      y = rect.bottom + 6;
      if (y + panelRect.height > window.innerHeight) y = Math.max(8, rect.top - panelRect.height - 6);
      if (x + panelRect.width > window.innerWidth) x = Math.max(8, window.innerWidth - panelRect.width - 8);
    }
    this._root.style.left = `${Math.round(x)}px`;
    this._root.style.top = `${Math.round(y)}px`;
    this._root.style.visibility = '';
    this._root.animate([
      { opacity: 0 },
      { opacity: 1 },
    ], { duration: this._scaled(FADE_DURATION), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
    this._dispatch('ui-menu-open', {});
  }

  close() {
    if (!this._open) return;
    this._open = false;
    this.dataset.open = 'false';
    for (const submenu of this._root.querySelectorAll('.submenu')) submenu.dataset.open = 'false';
    const animation = this._root.animate([
      { opacity: 1 },
      { opacity: 0 },
    ], { duration: this._scaled(FADE_DURATION), delay: this._scaled(DELAY_BEFORE_FADE_OUT), easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' });
    animation.finished.then(() => {
      if (!this._open) {
        this._root.style.display = 'none';
        this.style.visibility = 'hidden';
        animation.cancel();
      }
    }).catch(() => {});
    this._dispatch('ui-menu-close', {});
  }

  toggle() { this._open ? this.close() : this.open(); }

  _scaled(ms) {
    const scale = Number(getComputedStyle(this).getPropertyValue('--motion-scale')) || 1;
    return Math.max(1, Math.round(ms * Math.abs(scale)));
  }

  _dispatch(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { bubbles: true, composed: true, detail }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-menu')) {
  customElements.define('ui-menu', UiMenuElement);
}

export default UiMenuElement;
