/**
 * ComfyDL homepage extension for the ComfyFront_sep static deployment.
 *
 * Loaded through the frontend's official extension channel: the mock backend
 * (mock_client.js) returns this file's URL in GET /api/extensions, and the
 * app loads it with a dynamic `import()` — hence the ES module below.
 * `../../scripts/app.js` resolves to the package's comfyAPI shim, exactly
 * like the built-in extensions under extensions/core/.
 *
 * Responsibilities:
 *   1. `WebCatalog` nodes get an iframe DOM widget (lazy-loaded) and a big
 *      default size (1200x780) — the catalog HTML is designed for half/full
 *      screen and must never render smaller than any canvas node.
 *   2. `Docs*` markdown container nodes get a comfortable default size.
 *   3. A "Docs" sidebar tab lists the available pages and opens each one in
 *      a full-screen overlay (iframe + dark chrome matching the catalog
 *      theme). More catalog pages can be added to PAGES later.
 *
 * UI strings come from i18n/en/main.json (key namespace "home.*"), with
 * built-in English fallbacks; add i18n/zh/ later by extending LANG TABLES.
 */
import { app } from "../../scripts/app.js";

// ------------------------------------------------------------ page registry
// Shared easing (module scope: overlay/panel code lives outside setup()).
const EASE = 'cubic-bezier(.16,1,.3,1)';

const PAGES = [
  {
    id: 'catalog',
    url: './catalog/network-and-layers.html',
    labelKey: 'openCatalog',
    hintKey: 'openCatalogHint',
    tabKey: 'catalogTabLabel'
  }
];

// ------------------------------------------------------------ i18n (light)
const FALLBACK = {
  sidebarTitle: 'Docs',
  sidebarTooltip: 'Interactive node catalog and documentation',
  sidebarHeading: 'ComfyDL_UI documentation',
  sidebarHint: 'Open a page in a full-screen overlay, or drag the matching node from the Node Library onto the canvas.',
  openCatalog: 'Network & Layers catalog',
  openCatalogHint: '75 nodes - features, I/O and source overview',
  close: 'Close',
  openInNewTab: 'Open in new tab',
  catalogTabLabel: 'Network & Layers'
};

let T = { ...FALLBACK };

async function loadI18n() {
  try {
    const res = await fetch('i18n/en/main.json');
    if (res.ok) {
      const data = await res.json();
      if (data && data.home) T = { ...FALLBACK, ...data.home };
    }
  } catch (err) {
    console.warn('[ComfyDL.Home] i18n table unavailable, using fallback.', err);
  }
}

// ----------------------------------------------------- fullscreen overlay
let overlay = null;
let overlayEscHandler = null;

function closeCatalogOverlay() {
  if (!overlay) return;
  document.removeEventListener('keydown', overlayEscHandler);
  const wrap = overlay;
  const panel = wrap.querySelector('.cdl-ov-panel');
  const done = () => { wrap.remove(); if (overlay === wrap) { overlay = null; overlayEscHandler = null; } };
  // Animated exit — hard removal felt jarring after an animated entrance.
  try {
    wrap.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, easing: EASE, fill: 'forwards' });
    if (panel) panel.animate(
      [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'scale(.97) translateY(8px)' }],
      { duration: 160, easing: EASE, fill: 'forwards' });
    setTimeout(done, 170);
  } catch (err) { done(); }
}

function openCatalogOverlay(page) {
  closeCatalogOverlay();

  const wrap = document.createElement('div');
  Object.assign(wrap.style, {
    position: 'fixed', inset: '0', zIndex: '10000',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: '24px', background: 'rgba(8, 9, 12, 0.62)'
  });
  wrap.addEventListener('mousedown', (e) => { if (e.target === wrap) closeCatalogOverlay(); });

  const panel = document.createElement('div');
  panel.className = 'cdl-ov-panel';
  Object.assign(panel.style, {
    display: 'flex', flexDirection: 'column',
    width: 'min(1600px, 100%)', height: '100%',
    background: '#0f1013', border: '1px solid #262832',
    borderRadius: '12px', overflow: 'hidden',
    boxShadow: '0 24px 80px rgba(0, 0, 0, 0.55)'
  });

  // -- header: tab strip + actions, catalog-dark themed
  const header = document.createElement('div');
  Object.assign(header.style, {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '8px 12px', background: '#16171c', borderBottom: '1px solid #262832',
    color: '#e6e6ec', font: '500 13px system-ui, "Segoe UI", sans-serif'
  });

  const tab = document.createElement('span');
  tab.textContent = T[page.tabKey] || page.label;
  Object.assign(tab.style, {
    padding: '4px 12px', borderRadius: '6px',
    background: '#1c1e24', border: '1px solid #262832', color: '#1ABC9C'
  });
  header.appendChild(tab);

  const spacer = document.createElement('div');
  spacer.style.flex = '1';
  header.appendChild(spacer);

  const newTabBtn = document.createElement('button');
  newTabBtn.textContent = T.openInNewTab;
  Object.assign(newTabBtn.style, {
    background: '#1c1e24', border: '1px solid #262832', borderRadius: '6px',
    color: '#9a9aa5', cursor: 'pointer', font: 'inherit', padding: '4px 10px'
  });
  newTabBtn.addEventListener('click', () => window.open(page.url, '_blank', 'noopener'));

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕ ' + T.close;
  Object.assign(closeBtn.style, {
    background: '#1c1e24', border: '1px solid #262832', borderRadius: '6px',
    color: '#9a9aa5', cursor: 'pointer', font: 'inherit', padding: '4px 10px'
  });
  closeBtn.addEventListener('click', closeCatalogOverlay);

  header.appendChild(newTabBtn);
  header.appendChild(closeBtn);
  panel.appendChild(header);

  // -- content: lazy iframe (src assigned on open, not at startup)
  const frame = document.createElement('iframe');
  Object.assign(frame.style, { flex: '1', border: '0', width: '100%', background: '#0f1013' });
  frame.title = tab.textContent;
  panel.appendChild(frame);
  wrap.appendChild(panel);
  document.body.appendChild(wrap);
  // Animated entrance: backdrop fades, panel scales up from slightly below.
  try {
    wrap.animate([{ opacity: 0 }, { opacity: 1 }],
      { duration: 180, easing: EASE, fill: 'both' });
    panel.animate(
      [{ opacity: 0, transform: 'scale(.96) translateY(16px)' }, { opacity: 1, transform: 'none' }],
      { duration: 320, delay: 60, easing: EASE, fill: 'both' });
    console.log('[ComfyDL.Home] overlay entrance animated for', page.id);
  } catch (err) {
    console.warn('[ComfyDL.Home] overlay entrance animation failed:', err);
  }
  frame.src = page.url; // assigned after insertion — keeps startup light

  overlayEscHandler = (e) => { if (e.key === 'Escape') closeCatalogOverlay(); };
  document.addEventListener('keydown', overlayEscHandler);

  overlay = wrap;
}

// --------------------------------------------------------- sidebar tab UI
function buildDocsPanel(el) {
  el.style.overflow = 'auto';

  const root = document.createElement('div');
  Object.assign(root.style, {
    padding: '16px', color: 'var(--fg-color, #e6e6ec)',
    font: '13px system-ui, "Segoe UI", sans-serif',
    display: 'flex', flexDirection: 'column', gap: '10px'
  });

  const heading = document.createElement('div');
  heading.textContent = T.sidebarHeading;
  Object.assign(heading.style, { fontSize: '15px', fontWeight: '600' });
  root.appendChild(heading);

  const hint = document.createElement('div');
  hint.textContent = T.sidebarHint;
  Object.assign(hint.style, { fontSize: '12px', opacity: '0.65', lineHeight: '1.5' });
  root.appendChild(hint);

  for (const page of PAGES) {
    const card = document.createElement('button');
    Object.assign(card.style, {
      textAlign: 'left', cursor: 'pointer', font: 'inherit',
      background: 'var(--comfy-menu-bg, #1c1e24)',
      border: '1px solid var(--border-color, #262832)', borderRadius: '8px',
      color: 'inherit', padding: '10px 12px',
      transition: 'border-color .2s var(--cdl-ease), transform .2s var(--cdl-ease), box-shadow .2s var(--cdl-ease)'
    });
    card.addEventListener('mouseenter', () => {
      card.style.borderColor = '#1ABC9C';
      card.style.transform = 'translateY(-1px)';
      card.style.boxShadow = '0 4px 14px rgba(26,188,156,.12)';
    });
    card.addEventListener('mouseleave', () => {
      card.style.borderColor = 'var(--border-color, #262832)';
      card.style.transform = 'none';
      card.style.boxShadow = 'none';
    });
    card.addEventListener('click', () => {
      // Press feedback, then open.
      try {
        card.animate([{ transform: 'scale(.97)' }, { transform: 'none' }],
          { duration: 180, easing: EASE });
      } catch (err) { /* noop */ }
      openCatalogOverlay(page);
    });

    const title = document.createElement('div');
    title.textContent = T[page.labelKey] || page.id;
    title.style.fontWeight = '600';
    card.appendChild(title);

    const sub = document.createElement('div');
    sub.textContent = T[page.hintKey] || '';
    Object.assign(sub.style, { fontSize: '11.5px', opacity: '0.6', marginTop: '2px' });
    card.appendChild(sub);

    root.appendChild(card);
  }

  el.appendChild(root);
  // Sidebar tabs may be mounted/unmounted repeatedly; nothing persistent to
  // release, but honour the cleanup-fn contract anyway.
  return () => {};
}

// --------------------------------------------------------- node behaviour
function patchNodeSize(nodeType, size) {
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
    try {
      if (!this.size || this.size[0] < size[0] || this.size[1] < size[1]) {
        this.size = [...size];
      }
    } catch (err) { /* size is advisory only */ }
    return result;
  };
}

// Docs* markdown containers: the frontend's Vue widget-type map renders any
// widget whose type is 'markdown' with the native WidgetMarkdown component
// (same one the built-in MarkdownNote uses). object_info STRING widgets come
// out as plain text/textarea types, so after node creation we simply retag
// the 'text' widget. Zero asset patching; the widget keeps serializing by
// name as usual.
function convertTextWidgetToMarkdown(node) {
  const w = (node.widgets || []).find((x) => x.name === 'text');
  if (w && w.type !== 'markdown') w.type = 'markdown';
}

function attachIframeWidget(node, nodeData) {
  const el = document.createElement('iframe');
  el.loading = 'lazy';
  Object.assign(el.style, {
    width: '100%', height: '100%', border: '0',
    borderRadius: '8px', background: '#0f1013', display: 'block'
  });

  node.addDOMWidget('page', 'IFRAME', el, { hideOnZoom: false, selectOn: ['focus', 'click'] });

  const defaultUrl =
    (nodeData.input && nodeData.input.required && nodeData.input.required.url &&
      nodeData.input.required.url[1] && nodeData.input.required.url[1].default) ||
    PAGES[0].url;

  // Lazy-load: assign src after the node has been mounted, so the homepage's
  // first paint never waits on the 919 KB catalog page.
  setTimeout(() => {
    const urlWidget = (node.widgets || []).find((w) => w.name === 'url');
    const current = (urlWidget && urlWidget.value) || defaultUrl;
    if (!el.src) el.src = current;
  }, 300);

  // Keep the frame in sync when the url widget is edited (best effort).
  const urlWidget = (node.widgets || []).find((w) => w.name === 'url');
  if (urlWidget) {
    const original = urlWidget.callback;
    urlWidget.callback = function (value) {
      try { el.src = value; } catch (err) { /* ignore */ }
      return original ? original.apply(this, arguments) : value;
    };
  }
}

// ------------------------------------------------- click feedback: ripple
// Global surface ripple: every left pointerdown spawns a soft ring at the
// click point in a dedicated overlay layer (pointer-events:none) — works on
// the canvas, dialogs, menus, panels and buttons alike. Nothing is injected
// into Vue-managed elements, so no re-render interference.
const RIPPLE_CSS = `
#cdl-ripple-layer{position:fixed;inset:0;pointer-events:none;z-index:999998;overflow:hidden}
.cdl-ripple{position:absolute;border-radius:50%;pointer-events:none;background:radial-gradient(circle,rgba(205,228,255,.95) 0%,rgba(170,205,250,0) 68%);will-change:transform,opacity}
@media (prefers-reduced-motion:reduce){#cdl-ripple-layer{display:none}}
`;
function installRipple() {
  try {
    const style = document.createElement('style');
    style.textContent = RIPPLE_CSS;
    document.head.appendChild(style);
    const layer = document.createElement('div');
    layer.id = 'cdl-ripple-layer';
    document.body.appendChild(layer);
  } catch (err) { return; }
  document.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const layer = document.getElementById('cdl-ripple-layer');
    if (!layer) return;
    const size = 110;
    const r = document.createElement('span');
    r.className = 'cdl-ripple';
    r.style.width = r.style.height = size + 'px';
    r.style.left = (e.clientX - size / 2) + 'px';
    r.style.top = (e.clientY - size / 2) + 'px';
    layer.appendChild(r);
    const anim = r.animate(
      [{ transform: 'scale(.25)', opacity: .5 }, { transform: 'scale(1)', opacity: 0 }],
      { duration: 480, easing: 'cubic-bezier(.2,.6,.3,1)' });
    anim.onfinish = () => r.remove();
    anim.oncancel = () => r.remove();
  }, { passive: true });
}

// ------------------------------------------------- cursor trail: comet streak
// A single luminous streak that eases after the pointer and fades in ~0.4s —
// thin, fast-decaying, no particles, no emoji. Disabled entirely for
// prefers-reduced-motion users. The loop parks itself when the trail dies.
function installTrail() {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const canvas = document.createElement('canvas');
  canvas.className = 'cdl-trail';
  canvas.style.cssText =
    'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:999999';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let W = 0, H = 0;
  const resize = () => {
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  window.addEventListener('resize', resize);

  const LIFE = 460;          // ms a trail point lives (ink needs a moment)
  const MAX_PTS = 24;
  const FADE = 120;          // ms global fade-out once the pointer rests
  const pts = [];            // {x, y, t} newest first
  let mx = -1e4, my = -1e4;
  let lastMove = 0;
  let running = false;

  function frame() {
    const now = performance.now();
    while (pts.length && now - pts[pts.length - 1].t > LIFE) pts.pop();
    ctx.clearRect(0, 0, W, H);
    // The gradient stroke is position-based, so it can't express "the whole
    // streak dying". Fade the entire canvas globally once the pointer stops:
    // fade-out and erase become one continuous event instead of two.
    const alpha = Math.max(0, 1 - Math.max(0, now - lastMove - 60) / FADE);
    if (pts.length >= 2 && alpha > 0) {
      const head = pts[0];
      const tail = pts[pts.length - 1];
      // One continuous path + ONE gradient stroke (head→tail): no per-segment
      // overlaps, no specks. Kept deliberately simple: a single soft blue
      // that just fades out toward the tail.
      let tx = tail.x, ty = tail.y;
      if (Math.abs(tx - head.x) + Math.abs(ty - head.y) < 1) { tx += 0.01; ty += 0.01; }
      const g = ctx.createLinearGradient(head.x, head.y, tx, ty);
      g.addColorStop(0, 'rgba(158,200,240,0.5)');
      g.addColorStop(1, 'rgba(158,200,240,0)');
      ctx.globalAlpha = alpha;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = g;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(head.x, head.y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (pts.length && alpha > 0) {
      pts.unshift({ x: mx, y: my, t: now });
      if (pts.length > MAX_PTS) pts.length = MAX_PTS;
      requestAnimationFrame(frame);
    } else {
      pts.length = 0;
      running = false; // park the loop until the pointer moves again
    }
  }

  window.addEventListener('pointermove', (e) => {
    mx = e.clientX; my = e.clientY;
    lastMove = performance.now();
    if (!running) {
      running = true;
      pts.length = 0;
      pts.push({ x: mx, y: my, t: performance.now() });
      requestAnimationFrame(frame);
    }
  }, { passive: true });
}

function installClickFeedback() {
  installRipple();
  // Comet cursor trail disabled by user preference — restore the call below
  // to bring it back (the implementation is kept intact).
  // installTrail();
  console.log('[ComfyDL.Home] click feedback installed (ripple; trail disabled)');
}

// ---------------------------------------------------------------- register
app.registerExtension({
  name: 'ComfyDL.Home',
  async setup() {
    console.log('[ComfyDL.Home] setup() entered; extensionManager =', !!app.extensionManager);
    await loadI18n();

    // ---------------------------------------- staged entrance choreography
    // Runs once the boot splash signals readiness (custom 'cdl-booted'
    // event). Uses the Web Animations API directly — immune to selector/
    // cascade issues that broke the CSS-only attempt. Note: the sidebar
    // container does NOT carry .sidebar-left (that class belongs to the
    // help-center popup); we anchor on .sidebar-item-group instead.
    let entranceDone = false;
    // opts.cdlCancelAfter: drop the persisted animation once finished, so
    // later CSS :hover/:active transforms are not shadowed by the fill layer.
    // Only safe when the final keyframe equals the element's natural state.
    const animate = (el, keyframes, opts) => {
      if (!el) return false;
      try {
        const a = el.animate(keyframes, Object.assign({ easing: EASE, fill: 'both' }, opts));
        if (opts && opts.cdlCancelAfter) a.onfinish = () => { try { a.cancel(); } catch (e) { /* noop */ } };
        return true;
      } catch (err) { return false; }
    };
    const reveal = (el) => el && el.classList.add('cdl-entered');

    // Canvas backdrop: during the entrance the canvas element is hidden and
    // the container shows a solid #020303 backdrop (deterministic on every
    // device — no reliance on render timing). DOM nodes fade in over it;
    // afterwards the canvas fades back in via the cdl-canvas-in class.

    // Containers may still be empty at trigger time (content teleports in
    // later) — play their entrance only once they actually have children.
    const playWhenPopulated = (id, keyframes, opts) => {
      const el = document.getElementById(id);
      if (!el) return;
      const play = () => { reveal(el); animate(el, keyframes, opts); };
      if (el.children.length) play();
      else {
        const ob = new MutationObserver(() => {
          if (el.children.length) { ob.disconnect(); play(); }
        });
        ob.observe(el, { childList: true });
      }
    };

    const runEntrance = () => {
      if (entranceDone) return;
      if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) {
        entranceDone = true;
        document.documentElement.classList.add('cdl-no-anim');
        document.documentElement.classList.add('cdl-entrance-done');
        return;
      }
      entranceDone = true;
      // Once every entrance animation has finished, retire the pre-hide rules
      // permanently: Vue re-renders of the sidebar rail reset className and
      // would wipe our injected `cdl-entered`, leaving buttons invisible
      // forever. The gate class makes those rules stop matching entirely.
      setTimeout(() => document.documentElement.classList.add('cdl-entrance-done'), 2600);

      playWhenPopulated('comfyui-body-top',
        [{ opacity: 0, transform: 'translateY(-16px)' }, { opacity: 1, transform: 'none' }],
        { duration: 550, cdlCancelAfter: true });
      playWhenPopulated('comfyui-body-bottom',
        [{ opacity: 0, transform: 'translateY(16px)' }, { opacity: 1, transform: 'none' }],
        { duration: 500, delay: 300, cdlCancelAfter: true });
      playWhenPopulated('comfyui-body-right',
        [{ opacity: 0, transform: 'translateX(20px)' }, { opacity: 1, transform: 'none' }],
        { duration: 500, delay: 380, cdlCancelAfter: true });

      const trySidebar = (retries) => {
        const bar = document.querySelector('.side-tool-bar-container');
        const tabs = document.querySelectorAll(
          '.side-tool-bar-container .sidebar-item-group > *');
        if (!bar || !tabs.length) {
          if (retries > 0) setTimeout(() => trySidebar(retries - 1), 120);
          return;
        }
        reveal(bar);
        animate(bar,
          [{ opacity: 0, transform: 'translateX(-24px)' }, { opacity: 1, transform: 'none' }],
          { duration: 600, cdlCancelAfter: true });
        tabs.forEach((el, i) => {
          reveal(el);
          animate(el,
            [{ opacity: 0, transform: 'translateX(-14px) scale(.9)' }, { opacity: 1, transform: 'none' }],
            { duration: 500, delay: 150 + i * 60, cdlCancelAfter: true });
        });
        const btns = bar.querySelectorAll('.mt-auto > *');
        btns.forEach((el, i) => {
          reveal(el);
          animate(el,
            [{ opacity: 0, transform: 'translateX(-14px) scale(.9)' }, { opacity: 1, transform: 'none' }],
            { duration: 450, delay: 480 + i * 60, cdlCancelAfter: true });
        });
        console.log('[ComfyDL.Home] entrance: sidebar animated (tabs:',
          tabs.length, '/ bottom buttons:', btns.length + ')');
      };
      trySidebar(25);

      // Welcome-page nodes: diagonal sweep by canvas position.
      // IMPORTANT: opacity ONLY. The Vue node system positions each node via
      // its own `transform: translate(x, y)` — animating transform here would
      // stomp that (with fill:'both' it persists) and collapse every node to
      // the canvas origin, freezing them against drags.
      // Nodes mount after the homepage gate opens — poll until they exist.
      setTimeout(() => {
        let tries = 0;
        const sweep = () => {
          const nodes = Array.from(document.querySelectorAll('[data-node-id]'))
            .sort((a, b) => {
              const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
              return (ra.top - rb.top) || (ra.left - rb.left);
            });
          if (!nodes.length) {
            if (++tries < 40) setTimeout(sweep, 120);
            else document.documentElement.classList.add('cdl-nodes-in');
            return;
          }
          nodes.forEach((el, i) => animate(el,
            [{ opacity: 0 }, { opacity: 1 }],
            { duration: 450, delay: i * 40 }));
          setTimeout(() => document.documentElement.classList.add('cdl-nodes-in'),
            600 + nodes.length * 40);
          // Fade the real canvas (grid + links) back in after the sweep.
          // Canvas fades in early (mid-sweep) and slowly, overlapping the
          // remaining node reveals.
          setTimeout(() => document.documentElement.classList.add('cdl-canvas-in'),
            300 + nodes.length * 20);
          console.log('[ComfyDL.Home] entrance: nodes faded in (', nodes.length, ')');
        };
        sweep();
      }, 150);

      // Failsafe: force-reveal everything after 9s no matter what.
      setTimeout(() => {
        document.documentElement.classList.add('cdl-canvas-in');
        document.documentElement.classList.add('cdl-nodes-in');
        document.querySelectorAll(
          '.side-tool-bar-container, #comfyui-body-top, #comfyui-body-bottom, #comfyui-body-right')
          .forEach(reveal);
        document.querySelectorAll(
          '.side-tool-bar-container .sidebar-item-group > *, .side-tool-bar-container .mt-auto > *')
          .forEach(reveal);
      }, 9000);
    };
    window.addEventListener('cdl-booted', runEntrance);
    let entranceTries = 0;
    const entrancePoll = setInterval(() => {
      if (entranceDone || ++entranceTries > 24) { clearInterval(entrancePoll); return; }
      if (document.documentElement.classList.contains('cdl-booted')) runEntrance();
    }, 250);

    // Click feedback: ripple on chrome buttons + comet cursor trail.
    installClickFeedback();

    // Full-screen docs entries (sidebar tab renders the page list; each entry
    // opens the overlay so the half/full-screen catalog always gets room).
    try {
      app.extensionManager.registerSidebarTab({
        id: 'comfydl-docs',
        icon: 'pi pi-book',
        title: T.sidebarTitle,
        tooltip: T.sidebarTooltip,
        type: 'custom',
        render: buildDocsPanel
      });
      console.log('[ComfyDL.Home] "Docs" sidebar tab registered');
      try { window.__cdlBoot && window.__cdlBoot.log('"Docs" sidebar tab registered', 'ok'); } catch (e) { /* noop */ }
      // All pre-gate boot steps are done at this point — end the splash and
      // open the homepage load gate (media preload may still hold it briefly).
      try {
        window.__cdlBoot && window.__cdlBoot.finishBoot && window.__cdlBoot.finishBoot();
      } catch (e) { /* noop */ }
    } catch (err) {
      console.warn('[ComfyDL.Home] sidebar tab registration failed:', err);
    }

    // Sidebar expand/collapse animation (final): the tab panel is a
    // splitter panel (.side-bar-panel); expanding mounts
    // .sidebar-content-container into it, collapsing removes it and flips
    // the panel's width class. Expand = slide the content in; collapse =
    // a soft pulse on the icon rail as feedback.
    try {
      const ob = new MutationObserver((muts) => {
        for (const m of muts) {
          if (m.type === 'attributes') {
            const el = m.target;
            const cls = String(el.className || '');
            if (!cls.includes('side-bar-panel')) continue;
            const hasContent = !!el.querySelector('.sidebar-content-container');
            if (!hasContent) {
              // collapse: soft pulse on the icon rail as feedback
              const rail = document.querySelector('.side-tool-bar-container');
              if (rail) animate(rail, [{ opacity: 1 }, { opacity: .72 }, { opacity: 1 }], { duration: 260 });
              console.log('[ComfyDL.Home] sidebar collapsed');
            }
          } else {
            for (const n of m.addedNodes) {
              if (n.nodeType !== 1) continue;
              const c = String(n.className || '');
              if (c.includes('sidebar-content-container')) {
                animate(n,
                  [{ opacity: 0, transform: 'translateX(-26px)' }, { opacity: 1, transform: 'none' }],
                  { duration: 340 });
                console.log('[ComfyDL.Home] sidebar panel expanded (content slide-in)');
              }
            }
          }
        }
      });
      ob.observe(document.body, {
        childList: true, subtree: true, attributes: true, attributeFilter: ['class']
      });
      console.log('[ComfyDL.Home] sidebar expand/collapse animation armed');
    } catch (err) { /* noop */ }
    },
  beforeRegisterNodeDef(nodeType, nodeData) {
    const bootLog = (m) => { try { window.__cdlBoot && window.__cdlBoot.log(m, 'ok'); } catch (e) { /* noop */ } };
    if (nodeData.name === 'WebCatalog') {
      console.log('[ComfyDL.Home] wiring WebCatalog node type (iframe widget)');
      bootLog('UI node type wired: Web Catalog');
      patchNodeSize(nodeType, [1200, 780]);
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const node = onNodeCreated ? onNodeCreated.apply(this, arguments) : this;
        console.log('[ComfyDL.Home] WebCatalog node created, attaching iframe widget');
        attachIframeWidget(node || this, nodeData);
        return node;
      };
    } else if (nodeData.name && nodeData.name.indexOf('Docs') === 0) {
      console.log('[ComfyDL.Home] wiring markdown container node type:', nodeData.name);
      bootLog('UI node type wired: ' + (nodeData.display_name || nodeData.name));
      patchNodeSize(nodeType, [520, 440]);
      const onDocsNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const node = onDocsNodeCreated ? onDocsNodeCreated.apply(this, arguments) : this;
        try { convertTextWidgetToMarkdown(node || this); } catch (err) { /* advisory */ }
        return node;
      };
    }
  }
});

console.log('[ComfyDL.Home] extension loaded.');
