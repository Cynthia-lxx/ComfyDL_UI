/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:32-33 (SCROLL_BAR_WIDTH 10, padding 3)
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:87 (MinimumDimSize = width*3 -> min thumb 30px)
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:94-110 (corner radius 5, 3px margins, masked box)
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:116-118 (Gray8 default / GrayF hover /
 *     Highlight1 drag highlight)
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:129-156 (hover fade 100ms, drag highlight,
 *     release to white 100ms)
 *   osu.Game/Graphics/Containers/OsuScrollContainer.cs:67-73 (absolute scroll: click position
 *     minus half the thumb height, clamped)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 * (Upstream uses additive blending on the scrollbar; approximated with plain translucency.)
 */

const STYLE_URL = new URL('./ui-scroll.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: block; height: 240px; }
  .viewport { height: 100%; overflow-y: auto; }
  .track { display: none; }
`;

export class UiScrollElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._viewport = document.createElement('div');
    this._viewport.className = 'viewport';
    this._viewport.setAttribute('part', 'viewport');

    this._flow = document.createElement('div');
    this._flow.className = 'flow';
    this._flow.appendChild(document.createElement('slot'));

    this._track = document.createElement('div');
    this._track.className = 'track';
    this._track.setAttribute('part', 'track');

    this._thumb = document.createElement('div');
    this._thumb.className = 'thumb';
    this._thumb.setAttribute('part', 'thumb');

    this._track.appendChild(this._thumb);
    this._viewport.appendChild(this._flow);
    this.shadowRoot.append(this._style, this._viewport, this._track);

    this._resizeObserver = new ResizeObserver(() => this._sync());
    this._onScroll = () => this._sync();
    this._onThumbDown = (event) => {
      this._track.dataset.dragging = 'true';
      this._dragOffset = event.clientY - this._thumb.getBoundingClientRect().top;
      this._track.setPointerCapture(event.pointerId);
      event.preventDefault();
    };
    this._onTrackMove = (event) => {
      if (!this._track.dataset.dragging) return;
      const rect = this._track.getBoundingClientRect();
      const ratio = this._thumb.clientHeight
        ? (event.clientY - rect.top - this._dragOffset) / Math.max(1, rect.height - this._thumb.clientHeight)
        : 0;
      this._viewport.scrollTop = ratio * (this._viewport.scrollHeight - this._viewport.clientHeight);
    };
    this._onTrackUp = () => { delete this._track.dataset.dragging; };
    // Absolute scroll: jump so the clicked point lands at the thumb centre (OsuScrollContainer:67-73).
    this._onTrackDown = (event) => {
      if (event.target === this._thumb) return;
      const rect = this._track.getBoundingClientRect();
      const target = this._viewport.clientHeight / 2
        + (event.clientY - rect.top) / rect.height * this._viewport.scrollHeight
        - this._viewport.clientHeight / 2;
      const reduce = getComputedStyle(this).getPropertyValue('--motion-scale') === '0';
      this._viewport.scrollTo({ top: target, behavior: reduce ? 'auto' : 'smooth' });
    };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this._viewport.addEventListener('scroll', this._onScroll, { passive: true });
    this._thumb.addEventListener('pointerdown', this._onThumbDown);
    this._track.addEventListener('pointermove', this._onTrackMove);
    this._track.addEventListener('pointerup', this._onTrackUp);
    this._track.addEventListener('pointercancel', this._onTrackUp);
    this._track.addEventListener('pointerdown', this._onTrackDown);
    this._resizeObserver.observe(this._viewport);
    this._resizeObserver.observe(this._flow);
  }

  disconnectedCallback() {
    this._viewport.removeEventListener('scroll', this._onScroll);
    this._thumb.removeEventListener('pointerdown', this._onThumbDown);
    this._track.removeEventListener('pointermove', this._onTrackMove);
    this._track.removeEventListener('pointerup', this._onTrackUp);
    this._track.removeEventListener('pointercancel', this._onTrackUp);
    this._track.removeEventListener('pointerdown', this._onTrackDown);
    this._resizeObserver.disconnect();
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

  _sync() {
    const { scrollTop, scrollHeight, clientHeight } = this._viewport;
    const overflow = scrollHeight - clientHeight;
    this._track.dataset.visible = String(overflow > 1);
    if (overflow <= 1) return;
    const trackHeight = this._track.clientHeight;
    const thumbHeight = Math.max(30, Math.round(clientHeight / scrollHeight * trackHeight)); // MinimumDimSize
    const thumbTop = Math.round(scrollTop / Math.max(1, overflow) * (trackHeight - thumbHeight));
    this._thumb.style.height = `${thumbHeight}px`;
    this._thumb.style.top = `${thumbTop}px`;
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-scroll')) {
  customElements.define('ui-scroll', UiScrollElement);
}

export default UiScrollElement;
