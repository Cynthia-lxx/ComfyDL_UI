/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt
 * Ported from Graphics/Containers/WaveContainer.cs:19-192,
 * Overlays/WaveOverlayContainer.cs, Overlays/Mods/ShearedOverlayContainer.cs:132-174,
 * Overlays/SettingsPanel.cs:170-204, Overlays/Dialog/PopupDialog.cs:324-354.
 *
 * Five overlay reveal geometries with real enter/exit transitions.
 *
 * Transitions are serialised through a small state queue. A variant change
 * swaps the *closed* transform (e.g. translateX(-100%) -> translateY(100%));
 * with transitions enabled the browser animates that rule change too, so the
 * following enter would start from the previous variant's axis. Changing the
 * variant therefore sets [data-switching] to disable transitions for one frame
 * while the new closed state is committed, then the queued open() runs.
 *
 * Visibility is gated by visibility/pointer-events, never display:none, so a
 * before-change style always exists. close() waits for all running animations
 * before hiding (the OnComplete fence).
 */

const STYLE_URL = new URL('./ui-reveal-overlay.css', import.meta.url);

const REVEAL_LAYERS = [
  { rotation: 13, y: -930, anchor: 'centre' },
  { rotation: -7, y: -560, anchor: 'right' },
  { rotation: 4, y: -390, anchor: 'centre' },
  { rotation: -2, y: -220, anchor: 'right' },
];

const FALLBACK_STYLES = `
  :host { position: fixed; inset: 0; display: block; z-index: 2000;
          visibility: hidden; pointer-events: none; }
  :host([data-visible='true']) { visibility: visible; pointer-events: auto; }
  .panel { position: absolute; inset: 0; background: var(--ui-surface, #2a2a2a); }
  .plate { display: none; }
`;

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
const delay = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

export class UiRevealOverlayElement extends HTMLElement {
  static get observedAttributes() { return ['variant', 'data-activation', 'data-theme']; }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._dim = document.createElement('div');
    this._dim.className = 'dim';

    this._panel = document.createElement('div');
    this._panel.className = 'panel';
    this._panel.setAttribute('part', 'panel');
    this._panel.dataset.open = 'false';

    this._plate = document.createElement('div');
    this._plate.className = 'plate';
    this._plate.setAttribute('aria-hidden', 'true');

    this._waves = document.createElement('div');
    this._waves.className = 'waves';
    this._waves.setAttribute('aria-hidden', 'true');
    for (const [index, layer] of REVEAL_LAYERS.entries()) {
      const el = document.createElement('div');
      el.className = 'wave';
      el.dataset.anchor = layer.anchor;
      el.style.setProperty('--ui-wave-rotation', `${layer.rotation}deg`);
      el.style.setProperty('--ui-wave-y', `${layer.y}px`);
      el.style.setProperty('--ui-wave-index', String(index));
      this._waves.appendChild(el);
    }

    this._content = document.createElement('div');
    this._content.className = 'content';
    this._content.setAttribute('part', 'content');
    this._content.appendChild(document.createElement('slot'));

    this._panel.append(this._plate, this._waves, this._content);
    this.shadowRoot.append(this._style, this._dim, this._panel);

    this._onDimClick = () => this.close();
    this._onKeyDown = (event) => { if (event.key === 'Escape') this.close(); };
    this._dim.addEventListener('click', this._onDimClick);

    this._stylesReady = null;
    this._hideToken = 0;
    this._queue = Promise.resolve();
    this._variantSettled = null;
    this._opened = false;
    this._phase = 'closed';
  }

  connectedCallback() {
    this._stylesReady = this._loadStyles();
    if (!this.hasAttribute('variant')) this.setAttribute('variant', 'wave');
    document.addEventListener('keydown', this._onKeyDown);
  }

  disconnectedCallback() {
    this._dim.removeEventListener('click', this._onDimClick);
    document.removeEventListener('keydown', this._onKeyDown);
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === 'variant' && oldValue && oldValue !== newValue) {
      // Queue a transition-free reset to the new variant's closed state.
      // open() awaits this, so consumers may freely use setAttribute + open().
      this._variantSettled = this._enqueue(() => this._settleVariant());
    }
  }

  _enqueue(task) {
    const next = this._queue.then(task, task);
    this._queue = next.catch(() => {});
    return next;
  }

  async _settleVariant() {
    this.dataset.switching = 'true';
    for (const animation of this.shadowRoot.getAnimations?.() || []) animation.cancel();
    this._opened = false;
    this._phase = 'closed';
    this.dataset.open = 'false';
    this._panel.dataset.open = 'false';
    void this._panel.offsetWidth;
    await nextFrame();
    delete this.dataset.switching;
  }

  async _waitForStylesAndVariant() {
    if (this._stylesReady) await this._stylesReady;
    if (this._variantSettled) {
      const pending = this._variantSettled;
      this._variantSettled = null;
      await pending;
    }
  }

  async _loadStyles() {
    try {
      const response = await fetch(STYLE_URL);
      if (response.ok) this._style.textContent = await response.text();
    } catch { /* offline: inline fallback stays */ }
  }

  get variant() { return this.getAttribute('variant') || 'wave'; }
  get activation() { return this.getAttribute('data-activation') || 'all'; }
  get isOpen() { return this._opened; }
  get phase() { return this._phase; }

  async _openTask() {
    if (this.activation === 'disabled') return false;
    await this._waitForStylesAndVariant();
    if (this._opened) return this._replayTask();

    this._hideToken += 1;
    this._phase = 'opening';
    this.dataset.visible = 'true';
    this.dataset.open = 'false';
    this._panel.dataset.open = 'false';
    void this._panel.offsetWidth;
    await nextFrame();

    this._opened = true;
    this._phase = 'open';
    this.dataset.open = 'true';
    this._panel.dataset.open = 'true';
    this.dispatchEvent(new CustomEvent('ui-overlay-open', {
      bubbles: true, composed: true, detail: { variant: this.variant },
    }));
    return true;
  }

  /** 重播入场：同帧内 open=false→true 的"重定向"会被浏览器当作回到原目标值而取消过渡。
   * 因此完全复用 open 的两段式：_settleVariant（关过渡提交关闭态）→ 逐帧 → 打开半程。 */
  async _replayTask() {
    await this._settleVariant();

    this._phase = 'opening';
    this.dataset.visible = 'true';
    this.dataset.open = 'false';
    this._panel.dataset.open = 'false';
    void this._panel.offsetWidth;
    await nextFrame();

    this._opened = true;
    this._phase = 'open';
    this.dataset.open = 'true';
    this._panel.dataset.open = 'true';
    this.dispatchEvent(new CustomEvent('ui-overlay-open', {
      bubbles: true, composed: true, detail: { variant: this.variant },
    }));
    return true;
  }

  async _closeTask() {
    if (!this._opened && this.dataset.visible !== 'true') return false;
    this._opened = false;
    this._phase = 'closing';
    this.dataset.open = 'false';
    this._panel.dataset.open = 'false';
    this.dispatchEvent(new CustomEvent('ui-overlay-close', {
      bubbles: true, composed: true, detail: { variant: this.variant },
    }));

    const token = ++this._hideToken;
    const running = this.shadowRoot.getAnimations?.() || [];
    const finished = running.length
      ? Promise.allSettled(running.map((animation) => animation.finished))
      : Promise.resolve();
    await Promise.race([finished, delay(1200)]);
    if (token === this._hideToken && !this._opened) {
      this.dataset.visible = 'false';
      this._phase = 'closed';
    }
    return true;
  }

  open() { return this._enqueue(() => this._openTask()); }
  close() { return this._enqueue(() => this._closeTask()); }
  replay() { return this._enqueue(() => this._replayTask()); }
  toggle() { return this.isOpen ? (this.close(), false) : this.open(); }

  /** Change geometry and wait until the new closed state is committed. */
  async setVariant(name) {
    if (name === this.variant) return this;
    this.setAttribute('variant', name);
    await this._waitForStylesAndVariant();
    return this;
  }

  /** Visualise the sheared content safe area (preview / debugging aid). */
  setSafeZone(visible) {
    if (visible) this.dataset.safeZone = 'true';
    else delete this.dataset.safeZone;
    return this;
  }

  /** Current transition count — used by the preview site as a regression assertion. */
  get animationCount() { return (this.shadowRoot.getAnimations?.() || []).length; }

  setTheme(name) { if (name) this.dataset.theme = name; return this; }

  unmount() {
    document.removeEventListener('keydown', this._onKeyDown);
    this.remove();
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-reveal-overlay')) {
  customElements.define('ui-reveal-overlay', UiRevealOverlayElement);
}

export default UiRevealOverlayElement;
