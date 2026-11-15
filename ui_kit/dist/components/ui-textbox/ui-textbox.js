/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Graphics/UserInterface/OsuTextBox.cs:73-93 (height/corner/padding/caret width,
 *     BackgroundUnfocused=Background5 / BackgroundFocused=Background4 / BackgroundCommit=Highlight1)
 *   osu.Game/Graphics/UserInterface/OsuTextBox.cs:258-276 (focus border 3px, blur removes)
 *   osu.Game/Graphics/UserInterface/OsuTextBox.cs:278-282 (GetDrawableCharacter -> FallingDownContainer:
 *     each typed character drops into place — replicated here as a transient overlay span)
 *   osu.Game/Graphics/UserInterface/OsuTextBox.cs:321-397 (caret beat-synced pulse; approximated
 *     by a 500ms InOutSine focus-line pulse, min beat interval 300ms ≈ 60s/200BPM)
 *   osu.Game/Graphics/UserInterface/SearchTextBox.cs:15-19 (search variant height 35)
 *   osu.Game/Graphics/UserInterface/OsuNumberBox.cs (numeric-only variant)
 * Rewritten as a native ES module Custom Element; no upstream code, assets or branding.
 */

const STYLE_URL = new URL('./ui-textbox.css', import.meta.url);

const FALLBACK_STYLES = `
  :host { display: inline-block; }
  input {
    width: 100%; height: var(--control-height, 40px); padding: 0 10px;
    border: 0 solid transparent; border-radius: var(--corner-radius, 5px);
    background: var(--ui-surface-sunken, #222); color: var(--ui-text, #fff); font: inherit;
  }
  :host([data-focused='true']) input { border-width: 3px; border-color: var(--ui-accent-strong, #66ccff); }
  input:disabled { opacity: 0.3; }
`;

/** @typedef {{value: string, committed: boolean}} UiTextDetail */

export class UiTextboxElement extends HTMLElement {
  static get observedAttributes() {
    return ['placeholder', 'value', 'disabled', 'maxlength', 'variant', 'data-i18n-key', 'caps', 'status'];
  }

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._prevValue = '';
    this._commitTimer = 0;
    this._invalidTimer = 0;

    this._style = document.createElement('style');
    this._style.textContent = FALLBACK_STYLES;

    this._box = document.createElement('div');
    this._box.className = 'box';

    this._mirror = document.createElement('span');
    this._mirror.className = 'mirror';
    this._mirror.setAttribute('aria-hidden', 'true');

    this._input = document.createElement('input');
    this._input.setAttribute('part', 'input');
    this._input.setAttribute('type', 'text');
    this._input.setAttribute('spellcheck', 'false');
    this._input.setAttribute('autocomplete', 'off');

    this._focusLine = document.createElement('span');
    this._focusLine.className = 'focus-line';
    this._focusLine.setAttribute('aria-hidden', 'true');

    this._charLayer = document.createElement('span');
    this._charLayer.className = 'char-layer';
    this._charLayer.setAttribute('aria-hidden', 'true');

    this._box.append(this._mirror, this._input, this._focusLine, this._charLayer);
    this.shadowRoot.append(this._style, this._box);

    this._history = [];
    this._historyIndex = 0;
    this._onInput = () => this._handleInput();
    this._onKeyDown = (event) => {
      if (event.key === 'Enter') {
        this._commit();
        if (this.variant === 'history' && this._input.value.trim()) {
          this._history.push(this._input.value); // HistoryTextBox.cs:83-91
          this._historyIndex = this._history.length;
        }
      } else if (this.variant === 'history' && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        // HistoryTextBox.cs:46-81 —— 无修饰键时上下翻阅，首次上翻缓存原文
        if (!this._history.length) return;
        event.preventDefault();
        if (event.key === 'ArrowUp') {
          if (this._historyIndex === this._history.length) this._originalMessage = this._input.value;
          this._historyIndex = Math.max(0, this._historyIndex - 1);
          this._input.value = this._history[this._historyIndex];
        } else {
          this._historyIndex = Math.min(this._history.length, this._historyIndex + 1);
          this._input.value = this._historyIndex === this._history.length
            ? (this._originalMessage ?? '') : this._history[this._historyIndex];
        }
        this._prevValue = this._input.value;
      }
    };
    this._onFocus = () => {
      this.dataset.focused = 'true';
      if (this.hasAttribute('select-all-on-focus')) this._input.select();
    };
    this._onBlur = () => { this.dataset.focused = 'false'; };
  }

  connectedCallback() {
    this._loadStyles();
    this._sync();
    this._input.addEventListener('input', this._onInput);
    this._input.addEventListener('keydown', this._onKeyDown);
    this._input.addEventListener('focus', this._onFocus);
    this._input.addEventListener('blur', this._onBlur);
    // 宿主不可聚焦：点击宿主时把焦点转发给内部 input。
    this._onHostClick = () => { if (!this._input.disabled) this._input.focus(); };
    this.addEventListener('mousedown', this._onHostClick);
    // 语言包切换后重解析 data-i18n-key 占位文本。
    this._onI18nChange = () => this._sync();
    document.addEventListener('ui-i18n-change', this._onI18nChange);
  }

  disconnectedCallback() {
    this.removeEventListener('mousedown', this._onHostClick);
    document.removeEventListener('ui-i18n-change', this._onI18nChange);
    this._input.removeEventListener('input', this._onInput);
    this._input.removeEventListener('keydown', this._onKeyDown);
    this._input.removeEventListener('focus', this._onFocus);
    this._input.removeEventListener('blur', this._onBlur);
  }

  attributeChangedCallback() {
    if (this.isConnected) this._sync();
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

  get value() { return this._input.value; }
  set value(text) { this._input.value = String(text ?? ''); this._prevValue = this._input.value; }

  get variant() { return this.getAttribute('variant') || 'default'; }

  _sync() {
    const key = this.getAttribute('data-i18n-key');
    const dictionary = globalThis.UIReference?.i18n;
    const i18nText = key && dictionary && typeof dictionary[key] === 'string' ? dictionary[key] : '';

    this._input.placeholder = this.getAttribute('placeholder') ?? i18nText ?? '';
    this._input.value = this.getAttribute('value') ?? this._input.value;
    this._prevValue = this._input.value;
    this._input.maxLength = Number(this.getAttribute('maxlength')) || 1000;
    this._input.disabled = this.hasAttribute('disabled');
    this._input.type = this.variant === 'password' ? 'password' : 'text';
    this._box.dataset.variant = this.variant;
    this._box.dataset.caps = String(this.hasAttribute('caps'));
    this._box.dataset.status = this.getAttribute('status') || '';
    this._focusLine.dataset.status = this.getAttribute('status') || '';
    if (this.variant === 'focused') requestAnimationFrame(() => this._input.focus());
  }

  _handleInput() {
    const before = this._prevValue;
    const after = this._input.value;

    if (this.variant === 'number' && after !== '' && !/^-?\d*(\.\d*)?$/.test(after)) {
      this._input.value = before;
      this._flashInvalid();
      this._emit('ui-text-error', { value: before });
      return;
    }

    // Locate the insertion point via common prefix/suffix, then drop the new characters in.
    let prefix = 0;
    while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix += 1;
    let suffix = 0;
    while (suffix < before.length - prefix && suffix < after.length - prefix
      && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix += 1;
    const inserted = after.slice(prefix, after.length - suffix);
    if (inserted.length >= 1 && inserted.length <= 3) this._spawnFalling(after, prefix, inserted);

    this._prevValue = after;
    this._emit('ui-text-change', { value: after, committed: false });
  }

  /** FallingDownContainer equivalent: transient spans that drop into place above the text. */
  _spawnFalling(value, startIndex, inserted) {
    [...inserted].forEach((char, index) => {
      if (this._charLayer.children.length > 6) this._charLayer.firstElementChild?.remove();
      const span = document.createElement('span');
      span.textContent = char;
      // Advance per character by measuring the mirror with one more char.
      this._mirror.textContent = value.slice(0, startIndex + index + 1);
      span.style.left = `${Math.round(this._mirror.getBoundingClientRect().width - this._measureChar(char))}px`;
      this._charLayer.appendChild(span);
      window.setTimeout(() => span.remove(), 260);
    });
    this._mirror.textContent = '';
  }

  _measureChar(char) {
    this._mirror.textContent = char;
    const width = this._mirror.getBoundingClientRect().width;
    this._mirror.textContent = '';
    return width;
  }

  _commit() {
    this.dataset.commit = 'true';
    window.clearTimeout(this._commitTimer);
    this._commitTimer = window.setTimeout(() => { this.dataset.commit = 'false'; }, 260);
    this._emit('ui-text-commit', { value: this._input.value, committed: true });
  }

  _flashInvalid() {
    this.dataset.invalid = 'true';
    window.clearTimeout(this._invalidTimer);
    this._invalidTimer = window.setTimeout(() => { this.dataset.invalid = 'false'; }, 220);
  }

  _emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, { bubbles: true, composed: true, detail }));
  }
}

if (typeof customElements !== 'undefined' && !customElements.get('ui-textbox')) {
  customElements.define('ui-textbox', UiTextboxElement);
}

export default UiTextboxElement;
