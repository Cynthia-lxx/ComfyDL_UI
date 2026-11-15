/* SPDX-License-Identifier: MIT — Motion engine gallery. */
import { escapeHtml, buttonRow, on, table, setStatus } from './views.js';
import {
  motionScale, animate, stagger, fanOut, waveReveal, shearSlide, circularReveal,
  parallax, rollingNumber, elasticPop, ripple, intervalSync, scatterShapes,
} from '../05-motion/effects.js';
import { Timeline } from '../05-motion/timeline.js';

const KEYFRAME_CLASSES = [
  'ui-motion-enter', 'ui-motion-exit', 'ui-motion-slide-in', 'ui-motion-elastic',
  'ui-motion-ripple', 'ui-motion-pulse', 'ui-motion-glow', 'ui-motion-spin',
  'ui-motion-shake', 'ui-flash', 'ui-full-surface-flash', 'ui-popup',
];

export default {
  async mount(root, ctx) {
    const [motion, inventory] = await Promise.all([
      ctx.json('motion', '05-motion/motion.json'),
      ctx.json('inventory', '05-motion/effect-inventory.json'),
    ]);
    const steps = motion?.sequences?.entranceSteps?.steps || [];
    const domains = Object.keys(inventory?.entries || {});

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card" style="grid-column:1/-1">
          <h2>effects.js · 全部导出</h2>
          <p>逐个触发 <code>05-motion/effects.js</code> 的 13 个动效原语；右下角事件日志会记录每次调用。</p>
          ${buttonRow([
            ['mo-stagger', 'stagger'], ['mo-fanout', 'fanOut'], ['mo-shear', 'shearSlide'],
            ['mo-wave', 'waveReveal'], ['mo-circular', 'circularReveal'],
            ['mo-parallax', 'parallax toggle'], ['mo-roll', 'rollingNumber'],
            ['mo-elastic', 'elasticPop'], ['mo-ripple', 'ripple'],
            ['mo-beat', 'intervalSync toggle'], ['mo-glitch', 'scatterShapes toggle'],
            ['mo-animate', 'animate'], ['mo-scale', 'read motionScale'],
          ])}
          <div class="pv-row"><span class="pv-note" data-status>ready</span></div>
          <div class="pv-grid-2" style="margin-top:10px">
            <div>
              <h3>fanOut / stagger 目标</h3>
              <div class="pv-wedge-row" id="mo-wedges">
                ${[0, 1, 2, 3, 4].map((i) => `<div class="pv-wedge">item ${i + 1}</div>`).join('')}
              </div>
              <h3>shearSlide 目标</h3>
              <div class="pv-columns" id="mo-columns">${'<div></div>'.repeat(5)}</div>
            </div>
            <div>
              <h3>circularReveal / elasticPop / ripple / glitch 目标</h3>
              <div class="pv-stage" id="mo-stage">
                <div class="pv-glow-target" id="mo-target" style="display:inline-block">
                  target<div class="pv-ring" data-ring></div>
                </div>
                <div id="mo-glitch" style="position:absolute;inset:0;pointer-events:none"></div>
              </div>
              <h3>rollingNumber</h3>
              <div id="mo-roll" style="font-size:22px;font-variant-numeric:tabular-nums;color:var(--pv-accent)">0</div>
            </div>
          </div>
          <h3>waveReveal 目标（4 层旋转波浪 + content 相对位移）</h3>
          <p class="pv-note">四层角度 13° / -7° / 4° / -2°，100ms 错峰；终值按容器高度折算（上游按全屏 -930/-560/-390/-220）。打开后内容层从 200% 相对位移落回，收起反向。</p>
          ${buttonRow([['mo-wave-close', 'close()']])}
          <div class="pv-stage" id="mo-wave-host" style="height:150px"><div style="padding:12px">wave content</div></div>
        </section>

        <section class="pv-card">
          <h2>timeline.js · 绝对时间轴</h2>
          <p><code>BeginAbsoluteSequence</code> 等价实现；支持 play / pause / seek / setRate，seek 可重现同一状态。</p>
          ${buttonRow([
            ['mo-tl-play', 'detail'], ['mo-tl-pause', 'pause'], ['mo-tl-seek', 'seek 1500ms'],
            ['mo-tl-rate', 'rate ×2'], ['mo-tl-reset', 'reset'],
          ])}
          <div class="pv-bar" style="margin-top:10px"><i id="mo-tl-bar"></i></div>
          <div class="pv-row"><span class="pv-note" id="mo-tl-time">t = 0ms</span><span class="pv-note" id="mo-tl-step">step: —</span></div>
          <div class="pv-code" id="mo-tl-steps"></div>
        </section>

        <section class="pv-card">
          <h2>motion.css · 关键帧 / 工具类</h2>
          <p>点击任一 class 在示例块上重播；全部时长受 <code>--motion-scale</code> 与 reduced motion 控制。</p>
          <div class="pv-row" id="mo-classes">${KEYFRAME_CLASSES.map((c) => `<ui-button data-class="${c}" label="${c}"></ui-button>`).join('')}</div>
          <div class="pv-stage" style="margin-top:10px"><div class="pv-glow-target" id="mo-class-target">sample</div></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>motion-inventory.json · 全量动效档案</h2>
          <p>上游 <b>${inventory?.statistics?.filesWithTransforms || '—'}</b> 个文件含 Transform、<b>${inventory?.statistics?.easingEnumTypes || '—'}</b> 种 Easing、<b>${inventory?.statistics?.namedDurationConstants || '—'}</b> 处具名时长常量；本档案收录 <b>${Object.values(inventory?.entries || {}).reduce((n, l) => n + l.length, 0)}</b> 条。</p>
          <div class="pv-row">
            <label>域 <ui-dropdown id="mo-domain" value=""><ui-option value="" label="全部"></ui-option>${domains.map((d) => `<ui-option value="${escapeHtml(d)}" label="${escapeHtml(d)} (${inventory.entries[d].length})"></ui-option>`).join('')}</ui-dropdown></label>
            <label>搜索 <input id="mo-search" class="pv-input" placeholder="名称 / 源码 / 缓动" style="width:200px"></label>
            <span class="pv-note" id="mo-count"></span>
          </div>
          <div id="mo-inv-table" style="max-height:340px;overflow:auto"></div>
          <h3>缓动分布</h3>
          <div class="pv-code">${escapeHtml(Object.entries(inventory?.statistics?.easingEnumCounts || {}).map(([k, v]) => `${k}:${v}`).join('  '))}</div>
        </section>
      </div>`;

    const wedges = [...root.querySelectorAll('#mo-wedges > *')];
    const columns = [...root.querySelectorAll('#mo-columns > *')];
    const target = root.querySelector('#mo-target');
    const glitchHost = root.querySelector('#mo-glitch');
    const stage = root.querySelector('#mo-stage');
    let fan = null;
    let wave = null;
    let parallaxHandle = null;
    let beatHandle = null;
    let glitchHandle = null;
    let rollHandle = null;

    const say = (text) => { setStatus(root, text); ctx.log(text); };

    on(root, '#mo-stagger', 'click', () => {
      stagger(wedges, (el) => animate(el, [
        { opacity: 0.2, transform: 'translateY(14px)' }, { opacity: 1, transform: 'none' },
      ], { duration: 300 }), { step: 60 });
      say('stagger(wedges, 60ms)');
    });
    on(root, '#mo-fanout', 'click', () => {
      fan = fanOut(wedges, { states: { expanded: 90, contracted: 0, exploded: 180 } });
      fan.contractAll();
      window.setTimeout(() => fan.expandAll(0), 60);
      say('fanOut → contracted → expanded');
    });
    on(root, '#mo-shear', 'click', () => { shearSlide(columns, { distance: 700, step: 30 }); say('shearSlide(±700, 30ms)'); });
    on(root, '#mo-wave', 'click', () => {
      wave?.destroy();
      const host = root.querySelector('#mo-wave-host');
      host.innerHTML = '<div><b>内容层 · 相对位移 200% → 0</b><br><span class="pv-note">四层波浪在其下方依次扫入</span></div>';
      wave = waveReveal(host);
      window.setTimeout(() => wave.open(), 30);
      say('waveReveal.open()');
    });
    on(root, '#mo-wave-close', 'click', () => { wave?.close(); say('waveReveal.close()'); });
    on(root, '#mo-circular', 'click', () => { circularReveal(target); say('circularReveal()'); });
    on(root, '#mo-parallax', 'click', (event) => {
      if (parallaxHandle) { parallaxHandle.destroy(); parallaxHandle = null; event.currentTarget.dataset.active = 'false'; say('parallax off'); }
      else { parallaxHandle = parallax(target, { amount: 0.02 }); event.currentTarget.dataset.active = 'true'; say('parallax on'); }
    });
    on(root, '#mo-roll', 'click', () => { rollHandle?.cancel(); rollHandle = rollingNumber(root.querySelector('#mo-roll'), 0, 1234567, { duration: 3000 }); say('rollingNumber 0→1,234,567 / 3000ms'); });
    on(root, '#mo-elastic', 'click', () => { elasticPop(target); say('elasticPop()'); });
    on(root, '#mo-ripple', 'click', () => { ripple(stage, 120, 60); say('ripple()'); });
    on(root, '#mo-beat', 'click', (event) => {
      if (beatHandle) { beatHandle.stop(); beatHandle = null; event.currentTarget.dataset.active = 'false'; say('intervalSync off'); return; }
      event.currentTarget.dataset.active = 'true';
      beatHandle = intervalSync(({ index }) => {
        target.style.setProperty('--ui-accent', index % 2 ? 'var(--colour-1)' : 'var(--colour-0)');
      }, { bpm: 120 });
      say('intervalSync @120BPM');
    });
    on(root, '#mo-glitch', 'click', (event) => {
      if (glitchHandle) { glitchHandle.stop(); glitchHandle = null; event.currentTarget.dataset.active = 'false'; say('glitch off'); return; }
      event.currentTarget.dataset.active = 'true';
      glitchHandle = scatterShapes(glitchHost, { interval: 22, fade: 120 });
      say('scatterShapes @22ms');
    });
    on(root, '#mo-animate', 'click', () => {
      animate(target, [{ transform: 'rotate(0deg)' }, { transform: 'rotate(8deg)' }, { transform: 'rotate(0deg)' }], { duration: 500 });
      say('animate() raw WAAPI');
    });
    on(root, '#mo-scale', 'click', () => say(`motionScale() = ${motionScale()} (speed ${ctx.state.rate}×, reduce ${ctx.state.reduce})`));

    /* ---- timeline ---- */
    let timeline = null;
    let raf = 0;
    const bar = root.querySelector('#mo-tl-bar');
    const timeLabel = root.querySelector('#mo-tl-time');
    const stepLabel = root.querySelector('#mo-tl-step');
    const build = () => {
      timeline = new Timeline({ duration: 3000, onComplete: () => say('timeline complete') });
      for (const step of steps) {
        timeline.at(step.at, () => { stepLabel.textContent = `step: ${step.action} @ ${step.at}ms`; });
      }
    };
    const tickProgress = () => {
      if (!timeline) return;
      const t = timeline.currentTime;
      bar.style.width = `${Math.min(100, (t / 3000) * 100)}%`;
      timeLabel.textContent = `t = ${Math.round(t)}ms`;
      raf = requestAnimationFrame(tickProgress);
    };
    build();
    tickProgress();
    root.querySelector('#mo-tl-steps').textContent = steps.map((s) => `${String(s.at).padStart(4)}ms  ${s.action}`).join('\n');
    on(root, '#mo-tl-play', 'click', () => { build(); timeline.play(); say('timeline.play()'); });
    on(root, '#mo-tl-pause', 'click', () => { timeline?.pause(); say('timeline.pause()'); });
    on(root, '#mo-tl-seek', 'click', () => { timeline?.seek(1500); say('timeline.seek(1500)'); });
    on(root, '#mo-tl-rate', 'click', () => { timeline?.setRate(2); say('timeline.setRate(2)'); });
    on(root, '#mo-tl-reset', 'click', () => { build(); bar.style.width = '0'; stepLabel.textContent = 'step: —'; say('timeline.reset()'); });

    /* ---- keyframe gallery ---- */
    on(root, '[data-class]', 'click', (event) => {
      const cls = event.currentTarget.dataset.class;
      const el = root.querySelector('#mo-class-target');
      el.className = 'pv-glow-target';
      void el.offsetWidth;
      el.classList.add('pv-glow-target', cls);
      say(`class .${cls}`);
    });

    /* ---- inventory browser ---- */
    const domainSelect = root.querySelector('#mo-domain');
    const searchInput = root.querySelector('#mo-search');
    const countLabel = root.querySelector('#mo-count');
    const invTable = root.querySelector('#mo-inv-table');
    const renderInventory = () => {
      const domain = domainSelect.value;
      const query = searchInput.value.trim().toLowerCase();
      const rows = [];
      for (const [key, list] of Object.entries(inventory?.entries || {})) {
        if (domain && key !== domain) continue;
        for (const item of list) {
          const hay = `${item.n} ${item.a} ${item.e} ${item.s} ${item.g || ''}`.toLowerCase();
          if (query && !hay.includes(query)) continue;
          rows.push([
            `<span class="pv-note">${escapeHtml(key)}</span>`,
            escapeHtml(item.n),
            `<span class="pv-note">${escapeHtml(item.a || '')}</span>`,
            escapeHtml(String(item.ms ?? '—')),
            `<code>${escapeHtml(item.e || '')}</code>`,
            `<span class="pv-note">${escapeHtml(item.s || '')}</span>`,
          ]);
        }
      }
      countLabel.textContent = `${rows.length} 条匹配`;
      invTable.innerHTML = table(['域', '名称', '动作', 'ms', '缓动', '源码'], rows.slice(0, 60));
    };
    domainSelect.addEventListener('ui-select', renderInventory);
    searchInput.addEventListener('input', renderInventory);
    renderInventory();

    return () => {
      cancelAnimationFrame(raf);
      beatHandle?.stop();
      glitchHandle?.stop();
      parallaxHandle?.destroy();
      rollHandle?.cancel();
      wave?.destroy();
      timeline?.cancel();
    };
  },
};
