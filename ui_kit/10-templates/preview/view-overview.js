/* SPDX-License-Identifier: MIT — Overview dashboard. */
import { escapeHtml, kv, table, buttonRow, on } from './views.js';

export default {
  async mount(root, ctx) {
    const [manifest, inventory, settings, audit, health] = await Promise.all([
      ctx.json('manifest', 'manifest.json'),
      ctx.json('inventory', '05-motion/effect-inventory.json'),
      ctx.json('settings', 'settings.json'),
      ctx.json('audit', 'licence-audit.json'),
      ctx.json('health'),
    ]);

    const components = manifest?.components || [];
    const entries = Object.values(inventory?.entries || {}).reduce((n, list) => n + list.length, 0);
    const themes = (ctx.config.themes || []).length;

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card">
          <h2>抽离成果速览</h2>
          <p>静态资源包，无构建步骤；全部由 <code>10-templates/assemble.py</code> 装配到 <code>dist/</code>。</p>
          ${kv([
            ['dist 文件', ctx.state.distFiles.length],
            ['组件', components.length],
            ['主题方案', themes],
            ['动效档案条目', entries],
            ['settings 项', settings?.settings?.length || 0],
            ['i18n 键', Object.keys(ctx.state.i18n || {}).length],
            ['Transform 文件（上游）', inventory?.statistics?.filesWithTransforms || '—'],
            ['Easing 类型（上游）', inventory?.statistics?.easingEnumTypes || '—'],
          ])}
        </section>
        <section class="pv-card">
          <h2>服务器 / API</h2>
          <p>预览服务器提供 <code>/api/*</code> 只读端点；无服务器时页面自动回退到 dist 内的静态 JSON。</p>
          <div id="ov-health" class="pv-code">loading…</div>
          <h3>端点</h3>
          <div class="pv-code">/api/health  /api/manifest  /api/inventory  /api/motion
/api/themes  /api/settings  /api/i18n/&lt;lang&gt;  /api/audit
/api/files  /api/validate</div>
        </section>
        <section class="pv-card">
          <h2>四层结构</h2>
          <p>风格 → 组件 → 数据 → 装配，彼此以 JSON manifest 连接。</p>
          ${kv([
            ['风格/主题', 'tokens.css + themes/*.json → themes.css（9 方案）'],
            ['组件', '原生 ES module；Custom Elements / mount-unmount-setTheme'],
            ['动效', 'motion.css + timeline.js + effects.js + motion-inventory.json'],
            ['视觉效果', 'WebGL2 / Canvas 2D / CSS；shader 自写，未复制上游 GLSL'],
            ['数据', 'manifest.json / settings.json / i18n / licence-audit.json'],
            ['装配', 'assemble.py → dist/；preview_server.py 提供预览服务'],
          ])}
        </section>
        <section class="pv-card">
          <h2>许可边界</h2>
          <p>上游 MIT → 目标仓库 GPL-3.0 单向兼容；MIT 声明必须保留。</p>
          ${kv([
            ['可用', (audit?.allowed || []).length + ' 类'],
            ['有条件', (audit?.conditional || []).length + ' 类'],
            ['排除', (audit?.excluded || []).length + ' 类'],
            ['义务', (audit?.obligations || []).length + ' 条'],
            ['声明文件', 'dist/LICENSE-MIT.txt'],
          ])}
          <p class="pv-note">${escapeHtml(audit?.target?.compatibility || '')}</p>
        </section>
      </div>

      <section class="pv-card" style="margin-top:14px">
        <h2>快速跳转</h2>
        <p>数字键 1–6 可直接切换视图；<code>L</code> 打开事件日志，<code>R</code> 切换 reduced motion。</p>
        ${buttonRow([
          ['ov-go-theme', '主题 / 排版'], ['ov-go-components', '组件'], ['ov-go-motion', '动效引擎'],
          ['ov-go-layers', '图层编排 · 138 项'], ['ov-go-effects', '视觉效果'],
          ['ov-go-data', '数据 / 装配'], ['ov-classic', '旧版单页演示'],
        ])}
      </section>

      <section class="pv-card" style="margin-top:14px">
        <h2>组件清单</h2>
        ${table(['ID', '形态', '挂载点', '导出'], components.map((c) => [
          `<code>${escapeHtml(c.id)}</code>`, escapeHtml(c.kind),
          `<code>${escapeHtml(c.mount)}</code>`,
          `<span class="pv-note">${escapeHtml((c.exports || []).join(', '))}</span>`,
        ]))}
      </section>

      <section class="pv-card" style="margin-top:14px">
        <h2>装配产物（前 40）</h2>
        <div class="pv-code">${escapeHtml((ctx.state.distFiles || []).slice(0, 40).join('\n'))}</div>
      </section>`;

    root.querySelector('#ov-health').textContent = health
      ? JSON.stringify(health, null, 2)
      : 'static mode：未检测到 /api/health（直接以静态文件打开）。启动 preview_server.py 可启用全部端点。';

    const go = (id) => () => { location.hash = `#/${id}`; };
    on(root, '#ov-go-theme', 'click', go('theme'));
    on(root, '#ov-go-components', 'click', go('components'));
    on(root, '#ov-go-motion', 'click', go('motion'));
    on(root, '#ov-go-layers', 'click', go('layers'));
    on(root, '#ov-go-effects', 'click', go('effects'));
    on(root, '#ov-go-data', 'click', go('data'));
    on(root, '#ov-classic', 'click', () => window.open('index.html', '_blank'));

    return () => {};
  },
};
