/* SPDX-License-Identifier: MIT — data / assembly view. */
import { escapeHtml, buttonRow, on, table, kv } from './views.js';
import '../components/ui-button/ui-button.js';
import '../components/ui-dropdown/ui-dropdown.js';

const ENDPOINTS = [
  'health', 'manifest', 'inventory', 'motion', 'themes', 'settings',
  'i18n/en', 'i18n/zh-CN', 'audit', 'files', 'validate',
];

export default {
  async mount(root, ctx) {
    const [manifest, settings, audit, en, zh, licence] = await Promise.all([
      ctx.json('manifest', 'manifest.json'),
      ctx.json('settings', 'settings.json'),
      ctx.json('audit', 'licence-audit.json'),
      ctx.json('i18n/en', 'i18n/en.json'),
      ctx.json('i18n/zh-CN', 'i18n/zh-CN.json'),
      ctx.text('LICENSE-MIT.txt'),
    ]);
    const keys = Object.keys(en || {});

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card">
          <h2>API 端点</h2>
          <p>由 <code>preview_server.py</code> 提供；无服务器时自动回退到 dist 内静态 JSON。</p>
          <div class="pv-row">
            <ui-dropdown id="dt-endpoint" value="health">${ENDPOINTS.map((e) => `<ui-option value="${escapeHtml(e)}" label="/api/${escapeHtml(e)}"></ui-option>`).join('')}</ui-dropdown>
            <ui-button id="dt-fetch" label="fetch"></ui-button>
          </div>
          <div class="pv-code" id="dt-result" style="max-height:280px">选择端点后点击 fetch。</div>
        </section>

        <section class="pv-card">
          <h2>装配校验 / 产物</h2>
          <p><code>/api/validate</code> 会调用 <code>assemble.py</code> 的校验逻辑（仅服务器模式）。</p>
          ${buttonRow([['dt-validate', '运行校验'], ['dt-files', 'dist 文件清单']])}
          <div class="pv-code" id="dt-validate-out" style="max-height:200px">—</div>
          <div class="pv-row"><span class="pv-note">dist 文件：${ctx.state.distFiles.length}</span></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>i18n 语言包</h2>
          <p>键名与英文 fallback 来自 MIT 源码 <code>Localisation/*Strings.cs</code>；中文为本资源包自译，未复制上游 Crowdin 译文。切换顶栏语言会更新 <code>globalThis.UIReference.i18n</code>。</p>
          <div class="pv-row">
            <span class="pv-note">示例：</span>
            <ui-button data-i18n-key="common.back"></ui-button>
            <ui-button data-i18n-key="common.next" variant="sheared"></ui-button>
            <ui-button id="dt-lang-en" label="switch to en"></ui-button>
            <ui-button id="dt-lang-zh" label="switch to zh-CN"></ui-button>
          </div>
          ${table(['key', 'en', 'zh-CN'], keys.map((k) => [
            `<code>${escapeHtml(k)}</code>`, escapeHtml(en?.[k] || ''), escapeHtml(zh?.[k] || ''),
          ]))}
        </section>

        <section class="pv-card">
          <h2>settings.json</h2>
          <p>16 项可配置 setting，标注 upstream / kit-added；<code>motion.reduce</code> 是补齐上游缺失的全局动效降级。</p>
          ${table(['ID', '类型', '默认', '作用', '来源'], (settings?.settings || []).map((s) => [
            `<code>${escapeHtml(s.id)}</code>`, escapeHtml(s.type), escapeHtml(String(s.default)),
            `<span class="pv-note">${escapeHtml(s.appliesTo?.cssVar || s.appliesTo?.attribute || '')}</span>`,
            `<span class="pv-note">${escapeHtml(s.origin)}</span>`,
          ]))}
        </section>

        <section class="pv-card">
          <h2>manifest / 四层</h2>
          ${kv(Object.entries(manifest?.layers || {}).map(([k, v]) => [k, v]))}
          <h3>装配</h3>
          <div class="pv-code">${escapeHtml(JSON.stringify(manifest?.assembly || {}, null, 2))}</div>
          <h3>资源计数</h3>
          ${kv([
            ['components', (manifest?.components || []).length],
            ['extraCss', (manifest?.extraCss || []).length],
            ['extraFiles', (manifest?.extraFiles || []).length],
            ['themes', (ctx.config.themes || []).length],
          ])}
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>许可审计</h2>
          <p>${escapeHtml(audit?.target?.compatibility || '')}</p>
          ${kv([
            ['上游许可', audit?.source?.licence || 'MIT'],
            ['目标许可', audit?.target?.licence || 'GPL-3.0-only'],
            ['扫描 .cs 文件', audit?.scan?.csFilesScanned || '—'],
            ['不同版权头', audit?.scan?.distinctFirstLineHeaders ?? '—'],
            ['第三方版权头', audit?.scan?.thirdPartyHeaders ?? '—'],
            ['可用类别', (audit?.allowed || []).length],
            ['有条件类别', (audit?.conditional || []).length],
            ['排除类别', (audit?.excluded || []).length],
          ])}
          <h3>义务</h3>
          <div class="pv-code">${escapeHtml((audit?.obligations || []).join('\n'))}</div>
          <h3>排除项</h3>
          <div class="pv-code">${escapeHtml((audit?.excluded || []).map((x) => x.item).join('\n'))}</div>
          <h3>MIT 全文（dist/LICENSE-MIT.txt）</h3>
          <div class="pv-code" style="max-height:170px">${escapeHtml(licence.slice(0, 1400))}</div>
        </section>
      </div>`;

    const result = root.querySelector('#dt-result');
    const validateOut = root.querySelector('#dt-validate-out');
    const show = (value) => {
      result.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    };
    on(root, '#dt-fetch', 'click', async () => {
      const endpoint = root.querySelector('#dt-endpoint').value;
      const fallbacks = {
        manifest: 'manifest.json', inventory: '05-motion/effect-inventory.json',
        motion: '05-motion/motion.json', themes: 'themes/index.json',
        settings: 'settings.json', audit: 'licence-audit.json',
      };
      const fallback = fallbacks[endpoint] || (endpoint.startsWith('i18n/') ? `i18n/${endpoint.split('/')[1]}.json` : null);
      const value = await ctx.json(endpoint, fallback);
      show(value || { error: `${endpoint} 需要 preview_server.py，且无静态回退` });
      ctx.log(`GET /api/${endpoint}`);
    });
    on(root, '#dt-validate', 'click', async () => {
      const value = await ctx.json('validate');
      validateOut.textContent = value
        ? JSON.stringify(value, null, 2)
        : '需要 preview_server.py：/api/validate 会调用 assemble.py 的校验并返回结果。';
      ctx.log('GET /api/validate');
    });
    on(root, '#dt-files', 'click', async () => {
      const value = await ctx.json('files');
      validateOut.textContent = (value?.files || ctx.state.distFiles || []).join('\n');
      ctx.log('GET /api/files');
    });
    on(root, '#dt-lang-en', 'click', () => ctx.setLang('en'));
    on(root, '#dt-lang-zh', 'click', () => ctx.setLang('zh-CN'));

    return () => {};
  },
};
