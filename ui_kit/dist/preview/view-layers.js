/* SPDX-License-Identifier: MIT — 138-item expansion inventory + layered reveal lab. */
import { escapeHtml, buttonRow, on, table } from './views.js';
import '../components/ui-layered-view/ui-layered-view.js';

export default {
  async mount(root, ctx) {
    const [layerData, inventory] = await Promise.all([
      ctx.json('layers', '05-motion/layered-view.json'),
      ctx.json('expansion', '05-motion/expansion-inventory.json'),
    ]);
    const presets = layerData?.presets || {};
    const presetNames = Object.keys(presets);
    const domains = Object.keys(inventory?.byDomain || {});

    root.innerHTML = `
      <section class="pv-card" style="margin-bottom:14px">
        <h2>浏览视图展开 / 收起 · 多图层复刻</h2>
        <p>用户点名的界面：一个按钮触发 <b>12 个独立动画图层</b>同时起跑 —— 屏幕暗化、视窗裁切、4 条不同角度与终点的旋转波片、相对位移 2× 父高的内容层、固定头部、过滤、卡片流（30ms 错峰）、回顶按钮与页脚。进场 800ms / 出场 500ms，缓动 Out 系 ↔ In 系，收起用动画结束栅栏延迟清理。切换 preset 可对比另外 4 类编排。</p>
        <div class="pv-row">
          <label>编排 <ui-dropdown id="ly-preset" value="${escapeHtml(presetNames[0] || '')}">${presetNames.map((n) => `<ui-option value="${escapeHtml(n)}" label="${escapeHtml(presets[n].label || n)}"></ui-option>`).join('')}</ui-dropdown></label>
          <span class="pv-badge" id="ly-badge">未检验</span>
          <span class="pv-note" id="ly-state">phase: closed</span>
        </div>
        ${buttonRow([
          ['ly-open', '展开'], ['ly-close', '收起'], ['ly-replay', '重播'],
          ['ly-toggle', '切换'], ['ly-seek0', 'seek 400ms'], ['ly-rate', '速率 ×2'],
        ])}
        <ui-layered-view id="ly-demo" preset="${escapeHtml(presetNames[0] || 'catalogReveal')}" style="height:400px"></ui-layered-view>
        <div class="pv-row"><span>时间轴</span><input type="range" id="ly-seek" min="0" max="800" value="0" style="flex:1;accent-color:var(--pv-accent)"><output id="ly-time">0ms</output></div>
        <div class="pv-code" id="ly-diag" style="max-height:170px">点击「展开」开始检验。</div>
      </section>

      <div class="pv-grid">
        <section class="pv-card">
          <h2>图层开关</h2>
          <p>逐层关闭后可直观看到"很多图层"：每层有自己的时长、缓动与延迟，互不阻塞。</p>
          <div id="ly-toggles" style="display:grid;gap:5px"></div>
          ${buttonRow([['ly-all-on', '全部显示'], ['ly-all-off', '全部隐藏']])}
        </section>
        <section class="pv-card">
          <h2>逐图层编排表</h2>
          <p>起止状态来自 <code>layered-view.json</code>，每层独立执行。</p>
          <div id="ly-layers-table" style="max-height:330px;overflow:auto"></div>
        </section>
        <section class="pv-card" style="grid-column:1/-1">
          <h2>数学 / 插值 / 图层机制</h2>
          <p>点名界面不只是几何：还有相对位移、阻尼、深度排序、裁切高度与滚动 clamp。</p>
          <div id="ly-math"></div>
          <h3>最容易漏掉的 8 类</h3>
          <div class="pv-code">${escapeHtml((inventory?.easilyMissed || []).join('\n'))}</div>
        </section>
        <section class="pv-card" style="grid-column:1/-1">
          <h2>全仓展开 / 收起动效档案 · ${inventory?.statistics?.items || 0} 项</h2>
          <p>实测：叠层 PopIn 38 / PopOut 34、图形层 23、屏幕生命周期声明 132、可展开家族 22 个文件、序列编排 192+ 文件、Transform 调用 2580 次 / 509 文件。</p>
          <div class="pv-row">
            <label>域 <ui-dropdown id="ly-domain" value=""><ui-option value="" label="全部"></ui-option>${domains.map((d) => `<ui-option value="${escapeHtml(d)}" label="${escapeHtml(d)} (${inventory.byDomain[d]})"></ui-option>`).join('')}</ui-dropdown></label>
            <label>搜索 <input id="ly-search" class="pv-input" placeholder="名称 / 参数 / 源码" style="width:220px"></label>
            <span class="pv-note" id="ly-count"></span>
          </div>
          <div id="ly-inventory" style="max-height:420px;overflow:auto"></div>
        </section>
      </div>`;

    const view = root.querySelector('#ly-demo');
    const badge = root.querySelector('#ly-badge');
    const state = root.querySelector('#ly-state');
    const diag = root.querySelector('#ly-diag');
    const bar = root.querySelector('#ly-seek');
    const timeLabel = root.querySelector('#ly-time');
    const toggles = root.querySelector('#ly-toggles');
    const layersTable = root.querySelector('#ly-layers-table');
    const setBadge = (text, ok) => {
      badge.textContent = text;
      badge.className = `pv-badge ${ok === true ? 'ok' : ok === false ? 'bad' : ''}`;
    };

    let current = presetNames[0] || 'catalogReveal';
    let raf = 0;

    function renderPreset(name) {
      current = name;
      const preset = presets[name] || {};
      const layers = preset.layers || [];
      bar.max = String(preset.duration?.enter || 800);
      toggles.innerHTML = layers.map((layer) =>
        `<label class="pv-note" style="display:flex;gap:7px;align-items:center">
           <input type="checkbox" data-layer="${escapeHtml(layer.id)}" checked>
           <span style="min-width:26px;color:var(--pv-accent)">z${layer.z ?? 0}</span>
           <span>${escapeHtml(layer.name)}</span>
         </label>`).join('');
      layersTable.innerHTML = table(['层', '起 → 止', '进 / 出 ms', '缓动', '延迟', '源码'], layers.map((layer) => [
        `<code>${escapeHtml(layer.id)}</code>`,
        `<span class="pv-note">${escapeHtml(String(layer.closed?.transform ?? layer.closed?.opacity ?? '—'))} → ${escapeHtml(String(layer.open?.transform ?? layer.open?.opacity ?? '—'))}</span>`,
        `${layer.enter?.ms ?? '—'} / ${layer.exit?.ms ?? '—'}`,
        `<span class="pv-note">${escapeHtml(String(layer.enter?.ease || '—')).slice(0, 22)}</span>`,
        String(layer.enter?.delay ?? 0),
        `<span class="pv-note">${escapeHtml((layer.source || layer.upstream || '').slice(0, 46))}</span>`,
      ]));
      root.querySelector('#ly-math').innerHTML = table(['公式 / 机制', '表达式', '源码'], (preset.math || []).map((m) => [
        escapeHtml(m.name), `<code>${escapeHtml(m.formula)}</code>`, `<span class="pv-note">${escapeHtml(m.source || '')}</span>`,
      ]));
      on(root, '[data-layer]', 'change', (event) => {
        view.setLayerVisible(event.currentTarget.dataset.layer, event.currentTarget.checked);
        ctx.log(`layer ${event.currentTarget.dataset.layer} → ${event.currentTarget.checked}`);
      });
    }

    renderPreset(current);

    const probe = async (label) => {
      await new Promise((resolve) => window.setTimeout(resolve, 160));
      const count = view.animationCount;
      const preset = presets[current] || {};
      const lines = [
        `${label}`,
        `preset: ${current} · ${preset.label || ''}`,
        `layers: ${view.layerCount} · running transitions: ${count}`,
        `assert: ${count > 0 ? 'PASS (各图层过渡正在并行播放)' : 'FAIL (没有产生 CSS 过渡)'}`,
        `duration: enter ${preset.duration?.enter ?? '—'}ms / exit ${preset.duration?.exit ?? '—'}ms`,
        `asymmetry: ${preset.asymmetry || '进出场参数见逐层表'}`,
      ];
      diag.textContent = lines.join('\n');
      setBadge(`${label}: ${count > 0 ? 'PASS' : 'FAIL'} (${count} anim)`, count > 0);
      state.textContent = `phase: ${view.isOpen ? 'open' : 'closed'} · layers: ${view.layerCount}`;
      ctx.log(`${label}: ${count} transitions`);
    };

    on(root, '#ly-open', 'click', async () => { await view.open(); await probe('open'); });
    // Sample while the exit transitions are still running; awaiting close()
    // would wait for the OnComplete fence and report zero animations.
    on(root, '#ly-close', 'click', () => { view.close(); probe('close (退场)'); });
    on(root, '#ly-replay', 'click', async () => { view.replay(); await probe('replay'); });
    on(root, '#ly-toggle', 'click', () => { view.toggle(); probe('toggle'); });
    on(root, '#ly-seek0', 'click', () => { view.seek(400); timeLabel.textContent = '400ms'; bar.value = '400'; ctx.log('layers.seek(400)'); });
    on(root, '#ly-rate', 'click', (event) => { view.setRate(2); event.currentTarget.dataset.active = 'true'; ctx.log('layers.setRate(2)'); });
    bar.addEventListener('input', () => { timeLabel.textContent = `${bar.value}ms`; });
    bar.addEventListener('change', () => view.seek(Number(bar.value)));

    root.querySelector('#ly-preset').addEventListener('ui-select', async (event) => {
      const name = event.detail.value;
      await view.setPreset(name);
      renderPreset(name);
      ctx.log(`layered preset → ${name}`);
      setBadge('未检验', null);
      await view.open();
      await probe(`preset(${name})`);
    });

    on(root, '#ly-all-on', 'click', () => {
      for (const box of toggles.querySelectorAll('[data-layer]')) { box.checked = true; view.setLayerVisible(box.dataset.layer, true); }
    });
    on(root, '#ly-all-off', 'click', () => {
      for (const box of toggles.querySelectorAll('[data-layer]')) { box.checked = false; view.setLayerVisible(box.dataset.layer, false); }
    });

    /* ---- 138-item inventory browser ---- */
    const domainSelect = root.querySelector('#ly-domain');
    const searchInput = root.querySelector('#ly-search');
    const countLabel = root.querySelector('#ly-count');
    const invHost = root.querySelector('#ly-inventory');
    const renderInventory = () => {
      const domain = domainSelect.value;
      const query = searchInput.value.trim().toLowerCase();
      const rows = (inventory?.items || []).filter((item) => {
        if (domain && !item[2].startsWith(domain.charAt(0))) return false;
        if (query && !item.join(' ').toLowerCase().includes(query)) return false;
        return true;
      });
      countLabel.textContent = `${rows.length} / ${inventory?.items?.length || 0} 项`;
      invHost.innerHTML = table(['#', '名称', '域', '触发', '精确参数', '源码'], rows.slice(0, 70).map((item) => [
        `<code>${escapeHtml(item[0])}</code>`,
        escapeHtml(item[1]),
        `<span class="pv-note">${escapeHtml(item[2])}</span>`,
        `<span class="pv-note">${escapeHtml(item[3])}</span>`,
        escapeHtml(item[4]),
        `<span class="pv-note">${escapeHtml(item[5])}</span>`,
      ]));
    };
    domainSelect.addEventListener('ui-select', renderInventory);
    searchInput.addEventListener('input', renderInventory);
    renderInventory();

    const tick = () => {
      const duration = Number(bar.max) || 800;
      const elapsed = view.isOpen ? Math.min(duration, view.elapsed) : 0;
      if (view.isOpen && !bar.matches(':active')) bar.value = String(Math.round(elapsed));
      if (view.isOpen) timeLabel.textContent = `${Math.round(elapsed)}ms`;
      raf = requestAnimationFrame(tick);
    };
    tick();

    return () => { cancelAnimationFrame(raf); view.unmount(); };
  },
};
