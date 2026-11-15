/* SPDX-License-Identifier: MIT — Theme / typography view. */
import { escapeHtml, on, setStatus, table, buttonRow } from './views.js';

const SWATCH_VARS = [
  '--colour-0', '--colour-1', '--colour-2', '--colour-3', '--colour-4',
  '--ui-accent', '--ui-accent-strong', '--ui-text', '--ui-text-muted',
  '--ui-surface', '--ui-surface-raised', '--ui-surface-sunken', '--ui-border',
  '--pink-1', '--purple-1', '--blue-1', '--green-1', '--lime-1', '--orange-1', '--red-1',
  '--state-info', '--state-success', '--state-warning', '--state-error',
  '--state-approved', '--state-featured', '--state-verified',
  '--gray-3', '--gray-7', '--gray-b', '--gray-f',
];

const TABLE_VARS = [
  '--theme-hue', '--ui-accent', '--ui-surface', '--ui-border',
  '--corner-radius', '--corner-radius-large', '--control-height', '--footer-height',
  '--spacing-xs', '--spacing-sm', '--spacing-md', '--edge-margin', '--shear',
  '--font-family', '--font-size-title', '--font-size-body', '--font-size-caption2',
  '--duration-fast', '--duration-normal', '--duration-transition', '--duration-intro',
  '--motion-scale', '--parallax-strength', '--blur-level', '--dim-level',
];

const FONT_STEPS = [
  ['--font-size-title', 'Title 32'], ['--font-size-subtitle', 'Subtitle 28'],
  ['--font-size-heading1', 'Heading1 22'], ['--font-size-heading2', 'Heading2 18'],
  ['--font-size-body', 'Body 16'], ['--font-size-caption1', 'Caption1 14'],
  ['--font-size-caption2', 'Caption2 12'],
];

export default {
  async mount(root, ctx) {
    const [themes, settings, themesCss] = await Promise.all([
      ctx.json('themes', 'themes/index.json'),
      ctx.json('settings', 'settings.json'),
      ctx.text('themes.css'),
    ]);
    const schemes = themes?.schemes || [];

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card">
          <h2>主题方案</h2>
          <p>每个方案一个 hue，由 <code>OverlayColourProvider</code> 公式派生整套层级；点击即时切换 <code>data-theme</code>。</p>
          <div class="pv-row" id="th-schemes"></div>
          <div class="pv-row"><span class="pv-note" data-status>current: ${escapeHtml(ctx.state.theme)}</span></div>
          <h3>当前方案</h3>
          <div class="pv-code" id="th-detail">—</div>
          <p class="pv-note">themes.css 内 <code>[data-theme=...]</code> 块数：${(themesCss.match(/^\[data-theme=/gm) || []).length}</p>
        </section>
        <section class="pv-card">
          <h2>主题 token（实时计算值）</h2>
          <p>色块读取的是浏览器 computed style，主题切换后点「刷新」即可看到全部变量同步变化。</p>
          ${buttonRow([['th-refresh', '刷新色板'], ['th-cycle', '自动轮换']])}
          <div class="pv-swatches" id="th-swatches" style="margin-top:10px"></div>
        </section>
        <section class="pv-card">
          <h2>CSS 自定义属性表</h2>
          <p>tokens.css 的排版 / 动效 / 图形变量。</p>
          <div id="th-table"></div>
        </section>
        <section class="pv-card">
          <h2>排版层级</h2>
          <p>Torus / Inter 族名 + 系统回退栈（不分发字体文件）。</p>
          <div id="th-fonts"></div>
          <h3>settings 中的主题项</h3>
          <div id="th-settings"></div>
        </section>
      </div>`;

    const schemesHost = root.querySelector('#th-schemes');
    schemesHost.innerHTML = schemes.map((s) =>
      `<ui-button data-scheme="${escapeHtml(s.id)}" data-active="${s.id === ctx.state.theme}">${escapeHtml(s.id)} <span class="pv-note">${s.hue}°</span></ui-button>`).join('');

    function showScheme(id) {
      const s = schemes.find((x) => x.id === id) || {};
      root.querySelector('#th-detail').textContent = JSON.stringify({
        id: s.id, hue: s.hue,
        usedBy: themes?.usage?.[id] || '—',
        provider: 'HSL(hue, s, l) · OverlayColourProvider.cs:29-54',
      }, null, 2);
      setStatus(root, `current: ${id}`);
      for (const button of schemesHost.querySelectorAll('[data-scheme]')) {
        button.dataset.active = String(button.dataset.scheme === id);
      }
      renderSwatches();
      renderTable();
    }

    function renderSwatches() {
      const style = getComputedStyle(document.documentElement);
      root.querySelector('#th-swatches').innerHTML = SWATCH_VARS.map((name) => {
        const value = style.getPropertyValue(name).trim() || '—';
        return `<div class="pv-swatch"><i style="background:${escapeHtml(value)}"></i><span>${escapeHtml(name)}</span><b>${escapeHtml(value)}</b></div>`;
      }).join('');
    }

    function renderTable() {
      const style = getComputedStyle(document.documentElement);
      root.querySelector('#th-table').innerHTML = table(['变量', '计算值'], TABLE_VARS.map((name) => [
        `<code>${escapeHtml(name)}</code>`, `<span class="pv-note">${escapeHtml(style.getPropertyValue(name).trim() || '—')}</span>`,
      ]));
      root.querySelector('#th-fonts').innerHTML = FONT_STEPS.map(([name, label]) =>
        `<div class="pv-type-demo" style="font-size:var(${name})">${escapeHtml(label)} · Ag 中文 123</div>`).join('');
      const themeSettings = (settings?.settings || []).filter((s) => s.id.startsWith('theme.'));
      root.querySelector('#th-settings').innerHTML = table(['ID', '类型', '默认', '作用'], themeSettings.map((s) => [
        `<code>${escapeHtml(s.id)}</code>`, escapeHtml(s.type), escapeHtml(String(s.default)),
        `<span class="pv-note">${escapeHtml(s.appliesTo?.attribute || s.appliesTo?.cssVar || '')}</span>`,
      ]));
    }

    on(root, '[data-scheme]', 'click', (event) => {
      ctx.setTheme(event.currentTarget.dataset.scheme);
      showScheme(event.currentTarget.dataset.scheme);
    });
    on(root, '#th-refresh', 'click', () => { renderSwatches(); renderTable(); ctx.log('swatches refreshed'); });

    let timer = null;
    on(root, '#th-cycle', 'click', (event) => {
      if (timer) { clearInterval(timer); timer = null; event.currentTarget.dataset.active = 'false'; return; }
      event.currentTarget.dataset.active = 'true';
      let index = schemes.findIndex((s) => s.id === ctx.state.theme);
      timer = window.setInterval(() => {
        index = (index + 1) % schemes.length;
        ctx.setTheme(schemes[index].id);
        showScheme(schemes[index].id);
      }, 1200);
    });

    showScheme(ctx.state.theme);

    return () => { if (timer) clearInterval(timer); };
  },
};
