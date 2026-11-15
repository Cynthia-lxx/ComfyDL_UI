/* SPDX-License-Identifier: MIT
 * Derived from ppy/osu (MIT) — see 00-overview/LICENSE-MIT.txt and 00-overview/licence-audit.json
 * Behaviour structure from:
 *   osu.Game/Screens/Select/Panel.cs:135-205,331-385 (selection glow, X displacement)
 *   osu.Game/Graphics/Containers/BeatSyncedContainer.cs (beat-driven pulse)
 * Rewritten as a native ES module pure-DOM factory: mount / unmount / setTheme.
 * No upstream code, assets or branding.
 */

const STYLE_MARKER = 'data-ui-panel-styles';
const STYLE_URL = new URL('./ui-panel.css', import.meta.url);

function ensureStyles() {
  if (document.querySelector(`link[${STYLE_MARKER}]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = STYLE_URL.href;
  link.setAttribute(STYLE_MARKER, '');
  document.head.appendChild(link);
}

function resolveText(value, key, strings) {
  if (typeof value === 'string' && value) return value;
  const dictionary = strings || globalThis.UIReference?.i18n || {};
  if (key && typeof dictionary[key] === 'string') return dictionary[key];
  return '';
}

function toElement(target) {
  if (typeof target === 'string') {
    const found = document.querySelector(target);
    if (!found) throw new Error(`[ui-panel] mount target not found: ${target}`);
    return found;
  }
  if (target instanceof Element) return target;
  throw new TypeError('[ui-panel] mount() expects an Element or a CSS selector');
}

/**
 * Build a panel inside `target`.
 *
 * @param {Element|string} target
 * @param {Object} [options]
 * @param {string} [options.title]
 * @param {string} [options.titleKey]
 * @param {string} [options.description]
 * @param {string} [options.descriptionKey]
 * @param {boolean} [options.selected=false]
 * @param {boolean} [options.beat=false]
 * @param {string} [options.theme]
 * @param {Record<string,string>} [options.strings]
 * @returns {{element: HTMLElement, setTheme: Function, unmount: Function, update: Function, destroy: Function}}
 */
export function mount(target, options = {}) {
  const host = toElement(target);
  ensureStyles();

  const element = document.createElement('div');
  element.className = 'ui-panel';
  element.setAttribute('data-ui-panel', '');
  element.tabIndex = 0;
  element.setAttribute('role', 'button');
  element.setAttribute('aria-pressed', String(Boolean(options.selected)));

  const glow = document.createElement('div');
  glow.className = 'ui-panel__glow';
  glow.setAttribute('aria-hidden', 'true');

  const content = document.createElement('div');
  content.className = 'ui-panel__content';

  const title = document.createElement('div');
  title.className = 'ui-panel__title';
  const description = document.createElement('div');
  description.className = 'ui-panel__description';

  content.append(title, description);
  element.append(glow, content);
  host.appendChild(element);

  let disposed = false;

  const state = {
    selected: Boolean(options.selected),
    beat: Boolean(options.beat),
    strings: options.strings || null,
  };

  const apply = () => {
    title.textContent = resolveText(options.title, options.titleKey, state.strings);
    description.textContent = resolveText(options.description, options.descriptionKey, state.strings);
    element.dataset.selected = String(state.selected);
    element.dataset.beat = String(state.beat);
    element.setAttribute('aria-pressed', String(state.selected));
  };

  const toggle = () => {
    state.selected = !state.selected;
    apply();
    element.dispatchEvent(new CustomEvent('ui-select', {
      bubbles: true,
      detail: { selected: state.selected },
    }));
  };

  const onClick = (event) => {
    event.preventDefault();
    toggle();
  };

  const onKeyDown = (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    toggle();
  };

  element.addEventListener('click', onClick);
  element.addEventListener('keydown', onKeyDown);

  const setTheme = (name) => {
    if (!name || disposed) return;
    element.dataset.theme = name;
    element.dispatchEvent(new CustomEvent('ui-theme-change', {
      bubbles: true,
      detail: { theme: name },
    }));
  };

  const unmount = () => {
    if (disposed) return;
    disposed = true;
    element.removeEventListener('click', onClick);
    element.removeEventListener('keydown', onKeyDown);
    element.remove();
  };

  const update = (next = {}) => {
    if (disposed) return;
    if (typeof next.selected === 'boolean') state.selected = next.selected;
    if (typeof next.beat === 'boolean') state.beat = next.beat;
    if (next.strings) state.strings = next.strings;
    if (typeof next.title === 'string') options.title = next.title;
    if (typeof next.description === 'string') options.description = next.description;
    apply();
  };

  apply();
  if (options.theme) setTheme(options.theme);

  return {
    element,
    setTheme,
    unmount,
    update,
    destroy: unmount,
  };
}

/** Tear down a handle returned by mount(). */
export function unmount(handle) {
  if (!handle) return;
  if (typeof handle.unmount === 'function') handle.unmount();
  else if (handle.element instanceof Element) handle.element.remove();
}

/** Change the theme of a handle returned by mount(). */
export function setTheme(handle, name) {
  if (!handle || !name) return;
  if (typeof handle.setTheme === 'function') handle.setTheme(name);
  else if (handle.element instanceof Element) handle.element.dataset.theme = name;
}

export default { mount, unmount, setTheme };
