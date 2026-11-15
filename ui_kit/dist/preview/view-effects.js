/* SPDX-License-Identifier: MIT — 视觉效果 laboratory. */
import { escapeHtml, buttonRow, on, setStatus, canvasCard } from './views.js';
import { mountTriangles } from '../06-visual-effects/geometry/edge-geometry.js';
import { mountFocusMask, getSizeForIntensity } from '../06-visual-effects/geometry/focus-mask.js';
import { createParticleSpewer } from '../06-visual-effects/particles/particles.js';
import { flashColour, pulseGlow, setAdditive, attachAdditiveFlash, fullSurfaceFlash } from '../06-visual-effects/glow/glow.js';
import { createEntrancePlayer } from '../06-visual-effects/compositions/entrance-timeline.js';
import { createStatusPulseLayer } from '../06-visual-effects/compositions/status-pulse.js';

function pixels2d(canvas) {
  try {
    const context = canvas.getContext('2d');
    if (!context) return -1;
    const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let count = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) count += 1;
    return count;
  } catch { return -1; }
}
function pixelsGL(canvas) {
  const gl = canvas.getContext('webgl2');
  if (!gl) return Promise.resolve(-1);
  return new Promise((resolve) => requestAnimationFrame(() => {
    const buffer = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    let count = 0;
    for (let i = 3; i < buffer.length; i += 4) if (buffer[i] > 0) count += 1;
    resolve(count);
  }));
}

export default {
  async mount(root, ctx) {
    const [motion, presets] = await Promise.all([
      ctx.json('motion', '05-motion/motion.json'),
      ctx.json('presets', '06-visual-effects/particles/presets.json'),
    ]);
    const presetNames = Object.keys(presets?.presets || {});

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card">
          <h2>Edge geometry · WebGL2 UBO 描边</h2>
          <p>自写 GLSL + <code>BorderData{Thickness,TexelSize}</code> UBO；barycentric 描边、<code>fwidth</code> 抗锯齿、加法混合、按宽度定量补充。</p>
          ${canvasCard('vx-triangles', 'WebGL2 canvas')}
          <div class="pv-row">
            <label>density <input type="range" id="vx-t-density" min="0.2" max="4" step="0.2" value="2"><output>2</output></label>
            <label>thickness <input type="range" id="vx-t-thick" min="0.01" max="0.2" step="0.01" value="0.06"><output>0.06</output></label>
          </div>
          <div class="pv-row">
            <label>size <input type="range" id="vx-t-size" min="8" max="60" step="2" value="30"><output>30</output></label>
            <label>velocity <input type="range" id="vx-t-vel" min="0.01" max="0.3" step="0.01" value="0.09"><output>0.09</output></label>
          </div>
          <div class="pv-row"><span class="pv-note" id="vx-t-stat">painted: —</span><span class="pv-badge" id="vx-t-badge">—</span></div>
        </section>

        <section class="pv-card">
          <h2>ParticleSpewer · 时间纯函数</h2>
          <p>固定容量环形缓冲 + 单批次绘制，位置/alpha/scale/angle 全是时间函数，无物理积分。</p>
          ${canvasCard('vx-particles', 'Canvas 2D · additive')}
          <div class="pv-row">
            <label>preset <ui-dropdown id="vx-p-preset" value="${escapeHtml(presetNames[0] || '')}">${presetNames.map((n) => `<ui-option value="${escapeHtml(n)}" label="${escapeHtml(n)}"></ui-option>`).join('')}</ui-dropdown></label>
            <ui-button id="vx-p-burst" label="burst(60)"></ui-button>
          </div>
          <div class="pv-row"><span class="pv-note" id="vx-p-stat">painted: —</span></div>
        </section>

        <section class="pv-card">
          <h2>Focus mask · 柔边光锥</h2>
          <p>暗场遮罩 + <code>destination-out</code> 径向挖孔 + 加性边缘；intensity 100/200 收缩、break ×2.5。</p>
          ${canvasCard('vx-focus-mask', '移动鼠标改变光锥位置')}
          <div class="pv-row">
            <label>intensity <input type="range" id="vx-f-intensity" min="0" max="300" step="10" value="0"><output>0</output></label>
            <label>size <input type="range" id="vx-f-size" min="40" max="200" step="10" value="90"><output>90</output></label>
          </div>
          <div class="pv-row"><label><input type="checkbox" id="vx-f-break"> expanded (×2.5)</label><span class="pv-note" id="vx-f-stat">painted: —</span></div>
        </section>

        <section class="pv-card">
          <h2>glow.js · 光效工具</h2>
          <p>FlashColour / EdgeEffect 两段式 / 加性层 / 全屏白闪。</p>
          ${buttonRow([
            ['vx-g-flash', 'flashColour'], ['vx-g-pulse', 'pulseGlow'],
            ['vx-g-additive', 'setAdditive toggle'], ['vx-g-layer', 'attachAdditiveFlash'],
            ['vx-g-wide', 'fullSurfaceFlash'],
          ])}
          <div class="pv-stage" style="margin-top:10px"><div class="pv-glow-target" id="vx-g-target">glow target</div></div>
          <div class="pv-row"><span class="pv-note" data-status>ready</span></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>入场序列 · 绝对序列</h2>
          <p><code>createEntrancePlayer()</code> 兼容 motion.json v1/v2；按 Esc / 点击可中断，reduced motion 会跳过 glitch 与白闪。</p>
          ${buttonRow([
            ['vx-i-play', 'detail'], ['vx-i-seek', 'seek 1500ms'],
            ['vx-i-rate', 'rate ×2'], ['vx-i-cancel', 'cancel'],
          ])}
          <div class="pv-stage" id="vx-i-stage" style="margin-top:10px">Entrance stage</div>
          <div class="pv-row"><span class="pv-note" id="vx-i-state">step: —</span></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>状态反馈 · 池化 + 主题替换</h2>
          <p><code>createStatusPulseLayer()</code>：池化、绝对时间序列、<code>latestTransformEndTime</code> 定 lifetime、revert 倒带、可注入主题。</p>
          ${buttonRow([
            ['vx-j-success', 'SUCCESS'], ['vx-j-info', 'INFO'], ['vx-j-notice', 'NOTICE'],
            ['vx-j-warning', 'WARNING'], ['vx-j-error', 'ERROR'],
            ['vx-j-revert', 'revert last'], ['vx-j-clear', 'clear'],
            ['vx-j-renderer', 'switch renderer'], ['vx-j-theme', 'setTheme(purple)'],
          ])}
          <div class="pv-stage ui-status-pulse-layer" id="vx-j-host" style="height:150px;margin-top:10px"></div>
          <div class="pv-row"><span class="pv-note" id="vx-j-state">active: 0</span></div>
        </section>
      </div>`;

    const say = (text) => { setStatus(root, text); ctx.log(text); };
    let triangles = null;
    let particles = null;
    let focusMask = null;
    let intro = null;
    let statusPulseLayer = null;
    let lastPulseId = null;
    let customRenderer = false;
    let pollTimer = null;

    /* ---- triangles ---- */
    const tCanvas = root.querySelector('#vx-triangles');
    const tParams = () => ({
      density: Number(root.querySelector('#vx-t-density').value),
      thickness: Number(root.querySelector('#vx-t-thick').value),
      triangleSize: Number(root.querySelector('#vx-t-size').value),
      velocity: Number(root.querySelector('#vx-t-vel').value),
    });
    const buildTriangles = () => {
      triangles?.unmount();
      triangles = mountTriangles(tCanvas, tParams());
      root.querySelector('#vx-t-badge').textContent = triangles.available ? 'WebGL2 OK' : 'WebGL2 n/a';
      root.querySelector('#vx-t-badge').className = `pv-badge ${triangles.available ? 'ok' : 'bad'}`;
    };
    buildTriangles();
    for (const id of ['vx-t-density', 'vx-t-thick', 'vx-t-size', 'vx-t-vel']) {
      const input = root.querySelector(`#${id}`);
      input.addEventListener('input', () => { input.nextElementSibling.textContent = input.value; });
      input.addEventListener('change', () => { buildTriangles(); say(`triangles: ${JSON.stringify(tParams())}`); });
    }

    /* ---- particles ---- */
    const pCanvas = root.querySelector('#vx-particles');
    const pPreset = root.querySelector('#vx-p-preset');
    const buildParticles = () => {
      particles?.unmount();
      const preset = presets?.presets?.[pPreset.value] || {};
      particles = createParticleSpewer(pCanvas, { ...preset, perSecond: 60, size: 7, speed: 0.28 });
      particles.burst(50);
      say(`particles preset: ${pPreset.value}`);
    };
    buildParticles();
    pPreset.addEventListener('ui-select', buildParticles);
    on(root, '#vx-p-burst', 'click', () => { particles.burst(60); say('particles.burst(60)'); });

    /* ---- focusMask ---- */
    const fCanvas = root.querySelector('#vx-focus-mask');
    const fIntensity = root.querySelector('#vx-f-intensity');
    const fSize = root.querySelector('#vx-f-size');
    const fBreak = root.querySelector('#vx-f-break');
    const buildFocusMask = () => {
      focusMask?.unmount();
      focusMask = mountFocusMask(fCanvas, { size: Number(fSize.value) });
      focusMask.setIntensity(Number(fIntensity.value));
      focusMask.setBreak(fBreak.checked);
    };
    buildFocusMask();
    for (const [input, label] of [[fIntensity, 'intensity'], [fSize, 'size']]) {
      input.addEventListener('input', () => { input.nextElementSibling.textContent = input.value; });
    }
    fIntensity.addEventListener('change', () => {
      focusMask.setIntensity(Number(fIntensity.value));
      say(`focusMask intensity=${fIntensity.value} → size ×${getSizeForIntensity(1, Number(fIntensity.value), fBreak.checked).toFixed(3)}`);
    });
    fSize.addEventListener('change', buildFocusMask);
    fBreak.addEventListener('change', () => focusMask.setBreak(fBreak.checked));
    fCanvas.addEventListener('pointermove', (event) => {
      const rect = fCanvas.getBoundingClientRect();
      focusMask?.setPosition((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height);
    });

    /* ---- glow ---- */
    const gTarget = root.querySelector('#vx-g-target');
    let additive = false;
    on(root, '#vx-g-flash', 'click', (e) => { flashColour(gTarget); flashColour(e.currentTarget); say('flashColour()'); });
    on(root, '#vx-g-pulse', 'click', () => { pulseGlow(gTarget); say('pulseGlow() 40→800ms'); });
    on(root, '#vx-g-additive', 'click', (e) => { additive = !additive; setAdditive(gTarget, additive); e.currentTarget.dataset.active = String(additive); say(`setAdditive(${additive})`); });
    on(root, '#vx-g-layer', 'click', () => { attachAdditiveFlash(gTarget); say('attachAdditiveFlash()'); });
    on(root, '#vx-g-wide', 'click', () => { fullSurfaceFlash(document.body, 600); say('fullSurfaceFlash(600ms)'); });

    /* ---- intro ---- */
    const iStage = root.querySelector('#vx-i-stage');
    const iState = root.querySelector('#vx-i-state');
    intro = createEntrancePlayer(iStage, motion, {
      onStep: (step) => { iState.textContent = `step: ${step.action} @ ${step.at}ms`; },
    });
    const onIntroStep = (event) => { iStage.dataset.introStep = event.detail.step.action; };
    document.addEventListener('ui-entrance-step', onIntroStep);
    on(root, '#vx-i-play', 'click', () => { intro.play(); say('intro.play()'); });
    on(root, '#vx-i-seek', 'click', () => { intro.seek(1500); say('intro.seek(1500)'); });
    on(root, '#vx-i-rate', 'click', () => { intro.setRate(2); say('intro.setRate(2)'); });
    on(root, '#vx-i-cancel', 'click', () => { intro.cancel(); iState.textContent = 'step: —'; say('intro.cancel()'); });

    /* ---- status-pulse ---- */
    const jHost = root.querySelector('#vx-j-host');
    const jState = root.querySelector('#vx-j-state');
    let lastType = '—';
    const syncStatusPulse = () => {
      jState.textContent = `active: ${statusPulseLayer?.activeCount || 0} · last: ${lastType}`;
      jHost.dataset.pulsing = String(Boolean(statusPulseLayer?.activeCount));
    };
    const buildStatusPulse = () => {
      statusPulseLayer?.unmount();
      // 默认渲染器也显示类型名，让"点了有没有触发"一眼可见（上游是数字文本）
      const renderer = customRenderer ? {
        createPiece(type) {
          const el = document.createElement('div');
          el.className = `ui-status-pulse ui-status-pulse--${type}`;
          el.textContent = `CUSTOM ${type.toUpperCase()}`;
          el.latestTransformEndTime = 900;
          return el;
        },
      } : {
        createPiece(type) {
          const el = document.createElement('div');
          el.className = `ui-status-pulse ui-status-pulse--${type}`;
          el.textContent = type.toUpperCase();
          el.latestTransformEndTime = 900;
          return el;
        },
      };
      statusPulseLayer = createStatusPulseLayer(jHost, { renderer, poolSize: 12 });
      syncStatusPulse();
    };
    buildStatusPulse();
    const onStatusPulse = () => { syncStatusPulse(); window.setTimeout(syncStatusPulse, 1000); };
    document.addEventListener('ui-status-pulse', onStatusPulse);
    for (const type of ['success', 'info', 'notice', 'warning', 'error']) {
      root.querySelector(`#vx-j-${type}`).addEventListener('click', () => {
        lastType = type.toUpperCase();
        lastPulseId = statusPulseLayer.apply({ type, x: 90 + Math.random() * 240, y: 60, timeAbsolute: performance.now() });
        say(`status-pulse.apply(${type})`);
        syncStatusPulse();
      });
    }
    on(root, '#vx-j-revert', 'click', () => { statusPulseLayer.revert(lastPulseId); say('status-pulse.revert(last)'); syncStatusPulse(); });
    on(root, '#vx-j-clear', 'click', () => { statusPulseLayer.clear(); say('status-pulse.clear()'); syncStatusPulse(); });
    on(root, '#vx-j-renderer', 'click', (event) => { customRenderer = !customRenderer; event.currentTarget.dataset.active = String(customRenderer); buildStatusPulse(); say(`renderer: ${customRenderer ? 'custom' : 'default'}`); });
    on(root, '#vx-j-theme', 'click', () => { statusPulseLayer.setTheme('purple'); say('statusPulseLayer.setTheme(purple)'); });

    /* ---- pixel telemetry ---- */
    const poll = async () => {
      const t = await pixelsGL(tCanvas);
      root.querySelector('#vx-t-stat').textContent = `painted: ${t < 0 ? 'n/a' : t + ' px'}`;
      root.querySelector('#vx-p-stat').textContent = `painted: ${pixels2d(pCanvas)} px`;
      root.querySelector('#vx-f-stat').textContent = `painted: ${pixels2d(fCanvas)} px`;
    };
    pollTimer = window.setInterval(poll, 900);
    window.setTimeout(poll, 400);

    const onTheme = () => { triangles?.setTheme(ctx.state.theme); particles?.setTheme(ctx.state.theme); focusMask?.setTheme(ctx.state.theme); statusPulseLayer?.setTheme(ctx.state.theme); };
    window.addEventListener('pv-theme-change', onTheme);

    return () => {
      clearInterval(pollTimer);
      window.removeEventListener('pv-theme-change', onTheme);
      document.removeEventListener('ui-entrance-step', onIntroStep);
      document.removeEventListener('ui-status-pulse', onStatusPulse);
      triangles?.unmount();
      particles?.unmount();
      focusMask?.unmount();
      statusPulseLayer?.unmount();
      intro?.unmount();
    };
  },
};
