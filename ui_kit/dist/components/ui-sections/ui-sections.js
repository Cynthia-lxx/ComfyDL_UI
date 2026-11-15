/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Covers: SectionHeader.cs:36-82 (16 SemiBold title, 28x2 highlight circle bar,
 *   DetailsText appended in Highlight1, margin 10/5, vertical spacing 2);
 *   SectionsContainer.cs:49,163-205,208-212,249-308 (scroll_y_centre 0.1, ScrollTo
 *   with 50ms validation polling, header/footer offsets, SelectedSection resolution);
 *   OsuRearrangeableListContainer.cs:22,29-32,40-64 (DragActive broadcast, Move event,
 *   35ms throttle, OsuScrollContainer scrollbar).
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */
const STYLE_URL = new URL('./ui-sections.css', import.meta.url);
const FALLBACK = `:host{display:block;height:260px;overflow:auto;position:relative}.head{font-weight:600;font-size:16px;margin:10px 5px 2px}.head::after{content:'';display:block;width:28px;height:2px;border-radius:1px;background:var(--ui-accent,#66ccff)}`;

export class UiSectionsElement extends HTMLElement {
  static get observedAttributes() { return ['sortable']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK;
    this._scroll = document.createElement('div');
    this._scroll.className = 'scroll';
    this._scroll.setAttribute('part', 'scroll');
    this._scroll.appendChild(document.createElement('slot'));
    this.shadowRoot.append(this._style, this._scroll);
    this._dragged = null;
    this._lastSwap = 0;
    this._onDragStart = (event) => {
      const handle = event.target.closest?.('[data-drag-handle]');
      if (!handle || !this.hasAttribute('sortable')) return;
      this._dragged = handle.closest('[data-section]');
      if (this._dragged) this._dragged.dataset.dragging = 'true';
    };
    this._onDragOver = (event) => {
      if (!this._dragged) return;
      event.preventDefault();
      const target = event.target.closest?.('[data-section]');
      if (!target || target === this._dragged) return;
      const now = performance.now();
      if (now - this._lastSwap < 35) return; // Move 事件 35ms 节流
      this._lastSwap = now;
      const rect = target.getBoundingClientRect();
      const before = event.clientY < rect.top + rect.height / 2;
      this.insertBefore(this._dragged, before ? target : target.nextSibling);
      this.dispatchEvent(new CustomEvent('ui-section-move', {
        bubbles: true, composed: true, detail: { id: this._dragged.dataset.section },
      }));
    };
    this._onDragEnd = () => {
      if (this._dragged) delete this._dragged.dataset.dragging;
      this._dragged = null;
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._decorate();
    this.addEventListener('dragstart', this._onDragStart);
    this.addEventListener('dragover', this._onDragOver);
    this.addEventListener('dragend', this._onDragEnd);
  }

  disconnectedCallback() {
    this.removeEventListener('dragstart', this._onDragStart);
    this.removeEventListener('dragover', this._onDragOver);
    this.removeEventListener('dragend', this._onDragEnd);
  }

  attributeChangedCallback() { if (this.isConnected) this._decorate(); }

  async _loadStyles() {
    if (this._stylesLoaded) return;
    this._stylesLoaded = true;
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* file:// or offline: inline fallback stays */ }
  }

  /** 把轻 DOM 的 [data-section] 块加上标题（SectionHeader）与拖拽手柄。 */
  _decorate() {
    const sortable = this.hasAttribute('sortable');
    for (const section of this.querySelectorAll('[data-section]')) {
      if (section.querySelector(':scope > [data-section-head]')) continue;
      const head = document.createElement('div');
      head.className = 'head';
      head.dataset.sectionHead = '';
      const title = document.createElement('span');
      title.className = 'title';
      title.textContent = section.dataset.title || section.dataset.section || '';
      head.appendChild(title);
      const details = section.dataset.details;
      if (details) {
        const span = document.createElement('span');
        span.className = 'details';
        span.textContent = ` (${details})`;
        head.appendChild(span);
      }
      if (sortable) {
        const handle = document.createElement('span');
        handle.className = 'handle';
        handle.dataset.dragHandle = '';
        handle.draggable = true;
        handle.textContent = '⋮⋮';
        head.appendChild(handle);
      }
      section.prepend(head);
    }
  }

  /** SectionsContainer.ScrollTo：目标置于可视高度 10% 处。 */
  scrollToSection(id) {
    const section = [...this.querySelectorAll('[data-section]')].find((el) => el.dataset.section === id);
    if (!section) return;
    const target = section.offsetTop - this._scroll.clientHeight * 0.1;
    this._scroll.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    this.dispatchEvent(new CustomEvent('ui-section-select', {
      bubbles: true, composed: true, detail: { id },
    }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-sections')) {
  customElements.define('ui-sections', UiSectionsElement);
}
export default UiSectionsElement;
