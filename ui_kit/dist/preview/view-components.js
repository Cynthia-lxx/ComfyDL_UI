/* SPDX-License-Identifier: MIT — Component gallery with live assertions. */
import { escapeHtml, buttonRow, on, setStatus, table } from './views.js';
import '../components/ui-button/ui-button.js';
import '../components/ui-textbox/ui-textbox.js';
import '../components/ui-slider/ui-slider.js';
import '../components/ui-dropdown/ui-dropdown.js';
import '../components/ui-tabs/ui-tabs.js';
import '../components/ui-checkbox/ui-checkbox.js';
import '../components/ui-scroll/ui-scroll.js';
import '../components/ui-breadcrumb/ui-breadcrumb.js';
import '../components/ui-menu/ui-menu.js';
import '../components/ui-hold-button/ui-hold-button.js';
import '../components/ui-spinner/ui-spinner.js';
import '../components/ui-progress/ui-progress.js';
import '../components/ui-counter/ui-counter.js';
import '../components/ui-graph/ui-graph.js';
import '../components/ui-range-slider/ui-range-slider.js';
import '../components/ui-motion-container/ui-motion-container.js';
import '../components/ui-sections/ui-sections.js';
import '../components/ui-hotkey/ui-hotkey.js';
import { mount as mountPanel } from '../components/ui-panel/ui-panel.js';
import '../04-components/ui-action-rail/ui-action-rail.js';
import '../04-components/ui-reveal-overlay/ui-reveal-overlay.js';

const VARIANTS = [
  ['wave', 'Wave 四层旋转揭示'], ['sheared', 'Sheared 斜切下滑'],
  ['slide-x', 'Slide-X 侧栏'], ['slide-y', 'Slide-Y 底部'], ['pop', 'Pop 环形弹性'],
];

const MENU_ITEMS = [
  ['settings', '⚙', 'primary', 'primary', '#88da20', 'Settings'],
  ['detail', '▶', 'primary', 'primary', '#66ccff', 'Play'],
  ['secondary', '✎', 'primary', 'primary', '#ffcc22', 'Edit'],
  ['browse', '⌕', 'primary', 'primary', '#8c66ff', 'Browse'],
  ['solo', '●', 'detail', 'detail', '#66ccff', 'Solo'],
  ['group', '●●', 'detail', 'detail', '#ff66ab', 'Multi'],
  ['edit-map', '▤', 'secondary', 'secondary', '#ffcc22', 'Modify item'],
  ['edit-renderer', '◈', 'secondary', 'secondary', '#ff9966', 'Modify theme'],
];

export default {
  async mount(root, ctx) {
    const manifest = await ctx.json('manifest', 'manifest.json');
    const components = manifest?.components || [];

    root.innerHTML = `
      <div class="pv-grid">
        <section class="pv-card">
          <h2>ui-button</h2>
          <p>Custom Element + Shadow DOM；四态、sheared/danger 变体、i18n 标签、点击 <code>data-flash</code> 闪光。</p>
          <div class="pv-row">
            <ui-button label="Default"></ui-button>
            <ui-button variant="sheared" label="Sheared"></ui-button>
            <ui-button variant="danger" label="Danger"></ui-button>
            <ui-button label="Disabled" disabled></ui-button>
            <ui-button data-i18n-key="common.back"></ui-button>
          </div>
          <div class="pv-row"><span class="pv-note" data-status>presses: 0</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-textbox</h2>
          <p>聚焦 3px 描边、逐字坠落入场、光标节拍脉冲、Enter 提交闪烁、非法输入红闪；<code>search</code> / <code>number</code> 变体。</p>
          <div class="pv-row">
            <ui-textbox id="cp-tb-main" placeholder="type here…"></ui-textbox>
            <ui-textbox variant="search" data-i18n-key="ui.user_search"></ui-textbox>
            <ui-textbox variant="number" value="42"></ui-textbox>
            <ui-textbox placeholder="disabled" disabled></ui-textbox>
          </div>
          <div class="pv-row"><span class="pv-note" id="cp-tb-status">change 0 · commit 0 · error 0</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-slider</h2>
          <p>nub 250ms OutQuint 跟随、hover/drag 辉光、30ms 节流 tick 脉冲、双击恢复默认、方向键 / Shift×10 / PgUp·PgDn / Home·End。</p>
          <div class="pv-row">
            <ui-slider id="cp-sl-main" min="0" max="100" step="1" value="70"></ui-slider>
            <output id="cp-sl-out">70</output>
          </div>
          <div class="pv-row">
            <ui-slider min="0" max="1" step="0.05" value="0.5"></ui-slider>
            <ui-slider min="0" max="100" value="30" disabled></ui-slider>
          </div>
          <div class="pv-row"><span class="pv-note" id="cp-sl-status">changes: 0</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-dropdown</h2>
          <p>菜单 300ms OutQuint 淡入+高度过渡、悬停条 100ms 入 / 600ms 出、悬停 chevron 滑入、头部 chevron 翻转、方向键预选 / Enter 选择 / Esc 关闭、长标签截断省略。</p>
          <div class="pv-row">
            <ui-dropdown id="cp-dd-main" placeholder="choose…">
              <ui-option value="wave" label="Wave 四层旋转"></ui-option>
              <ui-option value="sheared" label="Sheared 斜切"></ui-option>
              <ui-option value="slide-x" label="Slide-X 侧栏"></ui-option>
              <ui-option value="pop" label="Pop 环形弹性揭示"></ui-option>
            </ui-dropdown>
            <span class="pv-badge" id="cp-dd-value">value: —</span>
            <ui-dropdown disabled placeholder="disabled"></ui-dropdown>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-tabs</h2>
          <p>悬停/激活底栏 1px 淡入 500ms OutQuint、激活加粗白字 / 未激活 accent 字、<code>strip</code> 属性启用底线跟随、方向键 / Home / End。</p>
          <div class="pv-row">
            <ui-tabs id="cp-tabs-main" value="all" strip>
              <ui-tab value="all" label="All"></ui-tab>
              <ui-tab value="recent" label="Recently Played"></ui-tab>
              <ui-tab value="favourite" label="Favourites"></ui-tab>
              <ui-tab value="group" label="Group"></ui-tab>
            </ui-tabs>
          </div>
          <div class="pv-row"><span class="pv-badge" id="cp-tabs-value">value: all</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-checkbox</h2>
          <p>nub 靠右、选中 OutElasticHalf 弹出（width 38→50 / 内环 3→8.5）、未选 OutQuint 收回、悬停辉光、空格 / 回车切换。</p>
          <div class="pv-row">
            <ui-checkbox id="cp-cb-main" label="Parallax" checked></ui-checkbox>
            <ui-checkbox label="Interface voices"></ui-checkbox>
            <ui-checkbox label="Disabled" disabled checked></ui-checkbox>
            <ui-checkbox data-i18n-key="ui.rotate_cursor"></ui-checkbox>
          </div>
          <div class="pv-row"><span class="pv-badge" id="cp-cb-value">checked: true</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-scroll</h2>
          <p>自绘滚动条（宽 10、圆角 5、最小拇指 30）：悬停 GrayF、拖拽 accent 高亮、轨道点击绝对滚动、原生滚动条隐藏。</p>
          <ui-scroll id="cp-scroll-main" style="height:180px">
            <div class="pv-code" style="max-height:none">scroll me —
              ${Array.from({ length: 14 }, (_, i) => `\nline ${String(i + 1).padStart(2, '0')} — The quick brown fox jumps over the lazy dog.`).join('')}
            </div>
          </ui-scroll>
        </section>

        <section class="pv-card">
          <h2>ui-breadcrumb</h2>
          <p>选中之后的项 500ms OutQuint 淡出+横向缩至 0.8、之前的项带分隔 chevron、激活白字加粗+底栏。</p>
          <div class="pv-row">
            <ui-breadcrumb id="cp-crumb-main" value="all">
              <ui-crumb value="all" label="All"></ui-crumb>
              <ui-crumb value="favourite" label="Favourites"></ui-crumb>
              <ui-crumb value="recent" label="Recently Played"></ui-crumb>
              <ui-crumb value="group" label="Group"></ui-crumb>
            </ui-breadcrumb>
          </div>
          <div class="pv-row"><span class="pv-badge" id="cp-crumb-value">value: all</span></div>
        </section>

        <section class="pv-card" id="cp-menu-card">
          <h2>ui-menu</h2>
          <p>280ms OutQuint 淡入开 / 50ms 延迟淡出关、子菜单右上锚定（放不下自动翻面）、悬停 80ms 常规↔加粗互换、destructive/highlighted 文字色、disabled 0.2、分隔条、Esc / 点击外部关闭。</p>
          <div class="pv-row">
            <ui-button id="cp-menu-open" label="open menu" animated></ui-button>
            <ui-menu for="cp-menu-open">
              <ui-item value="copy" label="Copy" hotkey="Ctrl+C"></ui-item>
              <ui-item value="rename" label="Rename" hotkey="F2"></ui-item>
              <ui-item value="theme" label="Theme">
                <ui-item value="dark" label="Dark"></ui-item>
                <ui-item value="light" label="Light"></ui-item>
                <ui-item value="nightcore" label="Nightcore" type="highlighted"></ui-item>
              </ui-item>
              <ui-spacer></ui-spacer>
              <ui-item value="disabled-demo" label="Unavailable" disabled></ui-item>
              <ui-item value="delete" label="Delete" type="destructive"></ui-item>
            </ui-menu>
          </div>
          <div class="pv-row"><span class="pv-badge" id="cp-menu-value">selected: —</span></div>
        </section>

        <section class="pv-card">
          <h2>ui-spinner · 加载族</h2>
          <p>3150ms 旋转、500ms OutQuint 入场 / 250ms 离场；box / plain / inverted；dim 铺满父级并 clamp(20–80px)。按钮 <code>loading</code> 与 show-more 自动忙碌态。</p>
          <div class="pv-row">
            <ui-spinner id="cp-sp-main"></ui-spinner>
            <ui-spinner variant="plain" inverted size="44"></ui-spinner>
            <ui-button id="cp-sp-toggle" label="show / hide"></ui-button>
            <ui-button label="submit" variant="show-more"></ui-button>
          </div>
          <div class="pv-row">
            <div style="position:relative;width:180px;height:72px;border:1px solid var(--pv-line);border-radius:6px;display:grid;place-items:center">
              <span class="pv-note">layer content</span>
              <ui-spinner id="cp-sp-dim" dim active="false"></ui-spinner>
            </div>
            <ui-button id="cp-sp-dim-toggle" label="toggle dim"></ui-button>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-progress · 进度族</h2>
          <p>四向生长（250ms InOutCubic）、seek 拖动发 seek/commit、FillColour 150ms OutQuint；expanding 4px↔2px 400ms OutElastic。</p>
          <div class="pv-stack">
            <ui-progress value="0.6" seek></ui-progress>
            <ui-progress value="0.35" direction="rtl"></ui-progress>
            <ui-progress value="0.8" direction="ttb" style="height:40px"></ui-progress>
            <ui-progress value="0.5" direction="btt" style="height:40px"></ui-progress>
            <div class="pv-row">
              <ui-progress variant="expanding" style="flex:1"></ui-progress>
              <ui-button id="cp-pg-expand" label="expand"></ui-button>
              <ui-button id="cp-pg-collapse" label="collapse"></ui-button>
            </div>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-counter · 滚动计数族</h2>
          <p>DisplayedCount / Current 双值插值、比例时长、前导零、千分位、精度格式；<code>fast</code> 属性把动效压到 ~250ms 以适配生产力 UI（去掉即上游结算界面的 1000ms）。</p>
          <div class="pv-row">
            <ui-counter class="cp-ct" fast format="plain" value="42"></ui-counter>
            <ui-counter class="cp-ct" fast format="digits" value="123456" leading="8"></ui-counter>
            <ui-counter class="cp-ct" fast format="separated" value="987654"></ui-counter>
            <ui-counter class="cp-ct" format="accuracy" value="0.9876"></ui-counter>
          </div>
          <div class="pv-row"><ui-button id="cp-ct-bump" label="roll to random"></ui-button></div>
        </section>

        <section class="pv-card">
          <h2>ui-graph · 图表族</h2>
          <p>bar 250ms InOutCubic 逐帧插值、line 1500ms OutQuint 宽度揭示、segmented tier 分层、stars 80ms 错峰 OutElasticHalf。</p>
          <div class="pv-stack">
            <ui-graph class="cp-gr" variant="bar" values="[1,3,2,5,4,2,6,3]"></ui-graph>
            <ui-graph class="cp-gr" variant="line" values="[1,3,2,5,4,2,6,3]" style="height:50px"></ui-graph>
            <ui-graph class="cp-gr" variant="segmented" values="[1,3,2,5,4,2,6,3]" tiers="3" style="height:16px"></ui-graph>
            <ui-graph class="cp-gr" variant="stars" value="6.5" stars="10"></ui-graph>
          </div>
          <div class="pv-row"><ui-button id="cp-gr-random" label="randomise"></ui-button></div>
        </section>

        <section class="pv-card">
          <h2>范围滑块 / 变体族</h2>
          <p>双端 minRange=0.1 钳制、nub 内嵌标签；plain / sheared(30px 斜切) / time(ms) / expandable；单滑块 sheared / time / expandable。</p>
          <div class="pv-stack">
            <ui-range-slider lower="0.2" upper="0.8" label="range"></ui-range-slider>
            <ui-range-slider variant="sheared" lower="0.1" upper="0.9" label="sheared"></ui-range-slider>
            <ui-range-slider variant="time" min="0" max="500" step="10" lower="100" upper="400" label="time"></ui-range-slider>
            <ui-slider variant="sheared" min="0" max="100" value="60"></ui-slider>
            <ui-slider variant="time" min="0" max="2000" step="10" value="500"></ui-slider>
            <ui-slider variant="expandable" min="0" max="100" value="30"></ui-slider>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-motion-container / ui-sections</h2>
          <p>shake ±8px 四段、parallax 指针跟随、hover 白→黄 500ms；分段标题 + 10% 视高滚动定位 + 拖拽重排（35ms 节流）。</p>
          <div class="pv-row">
            <ui-motion-container id="cp-mc-shake" variant="shake"><ui-button label="shake()"></ui-button></ui-motion-container>
            <ui-motion-container variant="parallax" style="width:120px;height:60px"><div class="pv-badge">parallax</div></ui-motion-container>
            <ui-motion-container variant="hover"><span class="pv-note">hover me</span></ui-motion-container>
          </div>
          <ui-sections id="cp-sec-main" sortable>
            <div data-section="a" data-title="General" data-details="3 items"><p class="pv-note">general section content</p></div>
            <div data-section="b" data-title="Interface"><p class="pv-note">interface section content</p></div>
            <div data-section="c" data-title="Input"><p class="pv-note">input section content</p></div>
          </ui-sections>
        </section>

        <section class="pv-card">
          <h2>ui-hotkey / 按钮变体族</h2>
          <p>HotkeyDisplay badge 链；ShearedButton / TwoLayerButton / DangerousRoundedButton / GrayButton / DialogButton / IconButton 全部映射为 <code>variant</code>。</p>
          <div class="pv-row">
            <ui-hotkey keys="Ctrl+C, Ctrl+Shift+V"></ui-hotkey>
            <ui-hotkey keys="F2"></ui-hotkey>
          </div>
          <div class="pv-row">
            <ui-button label="sheared" variant="sheared-button"></ui-button>
            <ui-button label="layers" variant="two-layer"></ui-button>
            <ui-button label="danger" variant="dangerous"></ui-button>
            <ui-button label="◆" variant="gray"></ui-button>
            <ui-button label="◇" variant="icon"></ui-button>
            <ui-button label="dialog" variant="dialog"></ui-button>
          </div>
        </section>

        <section class="pv-card">
          <h2>输入框 / 标签变体族</h2>
          <p>password 圆点 + caps 警告、history 上下翻阅、basic-search 右侧图标、sheared-search / sheared-filter 斜切框与状态行；tabs page / gradient-line / checkbox。</p>
          <div class="pv-stack">
            <div class="pv-row">
              <ui-textbox variant="password" placeholder="password" caps></ui-textbox>
              <ui-textbox variant="history" placeholder="history: enter then ↑"></ui-textbox>
              <ui-textbox variant="basic-search" placeholder="search"></ui-textbox>
            </div>
            <div class="pv-row">
              <ui-textbox variant="sheared-search" placeholder="sheared search"></ui-textbox>
              <ui-textbox variant="sheared-filter" status="12 matches" placeholder="filter"></ui-textbox>
            </div>
            <ui-tabs variant="page" value="a"><ui-tab value="a" label="Page A"></ui-tab><ui-tab value="b" label="Page B"></ui-tab></ui-tabs>
            <ui-tabs variant="gradient-line" value="a"><ui-tab value="a" label="Line A"></ui-tab><ui-tab value="b" label="Line B"></ui-tab></ui-tabs>
            <ui-tabs variant="checkbox" value="a"><ui-tab value="a" label="Checked"></ui-tab><ui-tab value="b" label="Unchecked"></ui-tab></ui-tabs>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-menu · 状态项族</h2>
          <p>toggle（bool ↔ ✓）/ ternary（False → Indeterminate ◉ → True ✓），点击不关闭菜单（CloseMenuOnClick = false）。</p>
          <div class="pv-row">
            <ui-button id="cp-state-open" label="open state menu"></ui-button>
            <ui-menu for="cp-state-open">
              <ui-item type="toggle" state="false" value="parallax" label="Parallax"></ui-item>
              <ui-item type="toggle" state="true" value="voices" label="Interface voices"></ui-item>
              <ui-spacer></ui-spacer>
              <ui-item type="ternary" state="indeterminate" value="selection" label="Mixed selection"></ui-item>
            </ui-menu>
            <span class="pv-badge" id="cp-state-value">state: —</span>
          </div>
        </section>

        <section class="pv-card">
          <h2>组合示例 · 设置面板</h2>
          <p>tabs + scroll + textbox + dropdown + slider + checkbox + button + hold-button 拼装；全部吃同一套 token，切主题整体联动。open menu 按钮带 <code>animated</code>（按住缓慢收缩、松开弹性回弹）。</p>
          <div class="pv-settings">
            <div class="pv-settings-head">
              <ui-tabs id="cp-set-tabs" value="general">
                <ui-tab value="general" label="General"></ui-tab>
                <ui-tab value="interface" label="Interface"></ui-tab>
              </ui-tabs>
            </div>
            <ui-scroll style="height:190px">
              <div class="pv-stack" style="padding-right:10px">
                <div class="pv-row"><label style="min-width:110px">name</label><ui-textbox variant="search" placeholder="type here…"></ui-textbox></div>
                <div class="pv-row"><label style="min-width:110px">background</label>
                  <ui-dropdown value="seasonal">
                    <ui-option value="always" label="Always"></ui-option>
                    <ui-option value="seasonal" label="Seasonal"></ui-option>
                    <ui-option value="never" label="Never"></ui-option>
                  </ui-dropdown>
                </div>
                <div class="pv-row"><label style="min-width:110px">cursor size</label><ui-slider min="0" max="2" step="0.05" value="1"></ui-slider></div>
                <div class="pv-row"><ui-checkbox label="Parallax" checked></ui-checkbox></div>
                <div class="pv-row"><ui-checkbox data-i18n-key="ui.rotate_cursor"></ui-checkbox></div>
              </div>
            </ui-scroll>
            <div class="pv-row" style="justify-content:flex-end">
              <ui-button id="cp-set-apply" label="Apply"></ui-button>
              <ui-hold-button id="cp-set-hold" label="reset" dangerous></ui-hold-button>
            </div>
          </div>
        </section>

        <section class="pv-card">
          <h2>ui-panel</h2>
          <p>纯 DOM 工厂，导出 <code>mount / unmount / setTheme</code>；选中发光与节拍脉冲。</p>
          ${buttonRow([
            ['cp-panel-mount', 'mount'], ['cp-panel-unmount', 'unmount'],
            ['cp-panel-selected', 'toggle selected'], ['cp-panel-beat', 'toggle beat'],
            ['cp-panel-theme', 'setTheme(pink)'],
          ])}
          <div id="cp-panel-host" style="margin-top:10px"></div>
          <div class="pv-row"><span class="pv-note" id="cp-panel-state">handle: —</span></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>ui-action-rail</h2>
          <p>锚点右侧横排楔形按钮：4 种宽度状态、<code>-20px</code> 负间距、<code>Shear 0.2 / -0.2</code> 反切、节拍驱动 hover、加性按压层、键盘导航、hub 追踪。</p>
          <ui-action-rail id="cp-menu">
            ${MENU_ITEMS.map(([id, icon, min, max, colour, label]) =>
              `<span data-menu-item="${id}" data-icon="${icon}" data-min="${min}" data-max="${max}" data-colour="${colour}">${escapeHtml(label)}</span>`).join('')}
          </ui-action-rail>
          ${buttonRow([
            ['cp-state-initial', 'Initial'], ['cp-state-primary', 'Primary'], ['cp-state-detail', 'Play'],
            ['cp-state-group', 'Multi'], ['cp-state-secondary', 'Edit'],
            ['cp-state-transition', 'Transition'], ['cp-state-hidden', 'Exit'],
          ])}
          <div class="pv-row"><strong class="pv-note" id="cp-menu-state">state: initial</strong><span class="pv-note">hover 楔形按钮查看 160→240px OutElastic 与图标节拍心跳</span></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>ui-reveal-overlay · 入/出场动效检验</h2>
          <p>5 种几何 variant。<b>关键回归断言</b>：打开后读取 <code>shadowRoot.getAnimations()</code>，必须 &gt; 0 才算真的播放了过渡；关闭后等动画结束才隐藏。</p>
          ${buttonRow(VARIANTS.map(([id, label]) => [`cp-ov-${id}`, label]))}
          ${buttonRow([
            ['cp-ov-close', 'Close（退场）'], ['cp-ov-replay', 'Replay（重播入场）'],
            ['cp-ov-cycle', '自动轮换'], ['cp-ov-disabled', 'activation=disabled'],
            ['cp-ov-axis', '连续切换方向检验'],
          ])}
          <div class="pv-row">
            <span class="pv-badge" id="cp-ov-badge">not tested</span>
            <span class="pv-note" id="cp-ov-state">overlay: closed</span>
            <label class="pv-note"><input type="checkbox" id="cp-ov-safe"> 显示斜切内容安全区</label>
          </div>
          <div class="pv-code" id="cp-ov-diag">点击任一 variant 开始检验。</div>
          <div id="cp-ov-axis-out" style="margin-top:8px"></div>
        </section>

        <section class="pv-card" style="grid-column:1/-1">
          <h2>清单</h2>
          ${table(['组件', '形态', '挂载点', '导出', '来源'], components.map((c) => [
            `<code>${escapeHtml(c.id)}</code>`, escapeHtml(c.kind),
            `<code>${escapeHtml(c.mount)}</code>`,
            `<span class="pv-note">${escapeHtml((c.exports || []).join(', '))}</span>`,
            `<span class="pv-note">${escapeHtml((c.source || [])[0] || '')}</span>`,
          ]))}
        </section>
      </div>

      <ui-reveal-overlay id="cp-overlay" variant="wave">
        <h2>Overlay content</h2>
        <p>当前 variant：<strong id="cp-ov-variant">wave</strong>。按 Esc 或点击遮罩关闭。</p>
        <p class="pv-note">Wave 四层 rotation 13/-7/4/-2、finalY -930/-560/-390/-220；content relative Y 200%→0。</p>
        <ui-button id="cp-ov-inner-close" label="Close"></ui-button>
      </ui-reveal-overlay>`;

    /* ---- ui-button ---- */
    let presses = 0;
    const onPress = () => { presses += 1; setStatus(root, `presses: ${presses}`); };
    root.addEventListener('ui-press', onPress);

    /* ---- ui-textbox ---- */
    const tbStatus = root.querySelector('#cp-tb-status');
    const tbEvents = { change: 0, commit: 0, error: 0 };
    const tbSync = () => { tbStatus.textContent = `change ${tbEvents.change} · commit ${tbEvents.commit} · error ${tbEvents.error}`; };
    for (const name of ['ui-text-change', 'ui-text-commit', 'ui-text-error']) {
      const kind = name.split('-')[2];
      root.addEventListener(name, (event) => {
        tbEvents[kind] += 1;
        tbSync();
        ctx.log(`textbox ${name}: ${event.detail.value}`);
      });
    }

    /* ---- ui-slider ---- */
    const slider = root.querySelector('#cp-sl-main');
    const sliderOut = root.querySelector('#cp-sl-out');
    const sliderStatus = root.querySelector('#cp-sl-status');
    let sliderChanges = 0;
    slider.addEventListener('ui-change', (event) => {
      sliderChanges += 1;
      sliderOut.textContent = String(event.detail.value);
      const nubAnims = slider.shadowRoot.querySelector('.nub').getAnimations().length;
      sliderStatus.textContent = `changes: ${sliderChanges} · nub anims: ${nubAnims}`;
      ctx.log(`slider → ${event.detail.value}`);
    });

    /* ---- ui-dropdown ---- */
    const dropdown = root.querySelector('#cp-dd-main');
    const ddValue = root.querySelector('#cp-dd-value');
    dropdown.addEventListener('ui-select', (event) => {
      ddValue.textContent = `value: ${event.detail.value}`;
      ddValue.className = 'pv-badge ok';
      ctx.log(`dropdown select: ${event.detail.value}`);
    });

    /* ---- ui-tabs / ui-checkbox ---- */
    const tabs = root.querySelector('#cp-tabs-main');
    const tabsValue = root.querySelector('#cp-tabs-value');
    tabs.addEventListener('ui-select', (event) => {
      tabsValue.textContent = `value: ${event.detail.value}`;
      ctx.log(`tabs select: ${event.detail.value}`);
    });
    const checkbox = root.querySelector('#cp-cb-main');
    const cbValue = root.querySelector('#cp-cb-value');
    checkbox.addEventListener('ui-change', (event) => {
      cbValue.textContent = `checked: ${event.detail.checked}`;
      ctx.log(`checkbox → ${event.detail.checked}`);
    });

    /* ---- ui-breadcrumb ---- */
    const breadcrumb = root.querySelector('#cp-crumb-main');
    const crumbValue = root.querySelector('#cp-crumb-value');
    breadcrumb.addEventListener('ui-select', (event) => {
      crumbValue.textContent = `value: ${event.detail.value}`;
      ctx.log(`breadcrumb select: ${event.detail.value}`);
    });

    /* ---- ui-menu ---- */
    const contextMenu = root.querySelector('ui-menu');
    const ctxMenuValue = root.querySelector('#cp-menu-value');
    contextMenu.addEventListener('ui-select', (event) => {
      ctxMenuValue.textContent = `selected: ${event.detail.value}`;
      ctx.log(`menu select: ${event.detail.value}`);
    });

    /* ---- 组合示例 / ui-hold-button ---- */
    const holdButton = root.querySelector('#cp-set-hold');
    holdButton.addEventListener('ui-confirm', (event) => {
      ctx.log(`hold-confirm (${event.detail.dangerous ? 'dangerous' : 'normal'})`);
    });
    const applyButton = root.querySelector('#cp-set-apply');
    applyButton?.addEventListener('ui-press', () => ctx.log('settings apply'));

    /* ---- 加载族 ---- */
    const spinner = root.querySelector('#cp-sp-main');
    root.querySelector('#cp-sp-toggle')?.addEventListener('click', () => {
      const active = spinner.getAttribute('active') === 'true';
      active ? spinner.hide() : spinner.show();
      ctx.log(`spinner → ${active ? 'hidden' : 'visible'}`);
    });
    const dimSpinner = root.querySelector('#cp-sp-dim');
    root.querySelector('#cp-sp-dim-toggle')?.addEventListener('click', () => {
      const on = dimSpinner.hasAttribute('active') && dimSpinner.getAttribute('active') === 'true';
      if (on) { dimSpinner.setAttribute('active', 'false'); dimSpinner.removeAttribute('dim'); }
      else { dimSpinner.setAttribute('dim', ''); dimSpinner.show(); }
      ctx.log(`loading layer → ${on ? 'off' : 'on'}`);
    });

    /* ---- 进度 / 计数 / 图表族 ---- */
    const expanding = root.querySelector('ui-progress[variant="expanding"]');
    root.querySelector('#cp-pg-expand')?.addEventListener('click', () => expanding?.expand());
    root.querySelector('#cp-pg-collapse')?.addEventListener('click', () => expanding?.collapse());
    root.querySelector('ui-progress[seek]')?.addEventListener('ui-progress-commit', (event) => {
      ctx.log(`progress commit: ${event.detail.value.toFixed(2)}`);
    });
    root.querySelector('#cp-ct-bump')?.addEventListener('click', () => {
      for (const counter of root.querySelectorAll('.cp-ct')) {
        const format = counter.getAttribute('format');
        counter.value = format === 'accuracy' ? (0.8 + Math.random() * 0.2).toFixed(4)
          : Math.round(Math.random() * 900000);
      }
      ctx.log('counters rolled');
    });
    root.querySelector('#cp-gr-random')?.addEventListener('click', () => {
      const values = Array.from({ length: 8 }, () => Math.round(Math.random() * 6 + 1));
      for (const graph of root.querySelectorAll('.cp-gr')) {
        if (graph.getAttribute('variant') === 'stars') graph.setAttribute('value', String((Math.random() * 10).toFixed(1)));
        else graph.setAttribute('values', JSON.stringify(values));
      }
      ctx.log('graphs redrawn');
    });

    /* ---- 容器族 ---- */
    const shakeBox = root.querySelector('#cp-mc-shake');
    shakeBox?.addEventListener('click', () => shakeBox.shake());
    root.querySelector('#cp-sec-main')?.addEventListener('ui-section-move', (event) => {
      ctx.log(`section moved: ${event.detail.id}`);
    });

    /* ---- 状态菜单项 ---- */
    const stateValue = root.querySelector('#cp-state-value');
    root.querySelector('ui-menu[for="cp-state-open"]')?.addEventListener('ui-menu-state', (event) => {
      stateValue.textContent = `state: ${event.detail.value}=${event.detail.state}`;
      ctx.log(`menu state: ${event.detail.value} → ${event.detail.state}`);
    });

    /* ---- ui-panel ---- */
    let panel = null;
    const panelHost = root.querySelector('#cp-panel-host');
    const panelState = root.querySelector('#cp-panel-state');
    const syncPanel = () => { panelState.textContent = `handle: ${panel ? 'mounted' : '—'}`; };
    const mountHandler = () => {
      if (panel) panel.unmount();
      panel = mountPanel(panelHost, {
        title: 'Panel component', description: 'mount / unmount / setTheme',
        selected: true, strings: ctx.state.i18n,
      });
      panel.element.addEventListener('ui-select', () => ctx.log('panel clicked'));
      syncPanel();
    };
    mountHandler();
    on(root, '#cp-panel-mount', 'click', mountHandler);
    on(root, '#cp-panel-unmount', 'click', () => { panel?.unmount(); panel = null; syncPanel(); });
    on(root, '#cp-panel-selected', 'click', () => {
      if (!panel) return;
      panel.update({ selected: panel.element.dataset.selected !== 'true' });
    });
    on(root, '#cp-panel-beat', 'click', () => {
      if (!panel) return;
      panel.update({ beat: panel.element.dataset.beat !== 'true' });
    });
    on(root, '#cp-panel-theme', 'click', () => panel?.setTheme('pink'));

    /* ---- ui-action-rail ---- */
    const menu = root.querySelector('#cp-menu');
    const menuState = root.querySelector('#cp-menu-state');
    for (const button of root.querySelectorAll('[id^="cp-state-"]')) {
      button.addEventListener('click', () => menu.setState(button.id.replace('cp-state-', '')));
    }
    const onMenuState = (event) => { menuState.textContent = `state: ${event.detail.state}`; };
    const onMenuPress = (event) => { menuState.textContent = `pressed: ${event.detail.id} @ ${event.detail.state}`; };
    menu.addEventListener('ui-menu-state', onMenuState);
    menu.addEventListener('ui-menu-press', onMenuPress);
    window.setTimeout(() => menu.setState('primary'), 150);

    /* ---- ui-reveal-overlay ---- */
    const overlay = root.querySelector('#cp-overlay');
    const badge = root.querySelector('#cp-ov-badge');
    const diag = root.querySelector('#cp-ov-diag');
    const ovState = root.querySelector('#cp-ov-state');
    const setBadge = (text, ok) => {
      badge.textContent = text;
      badge.className = `pv-badge ${ok === true ? 'ok' : ok === false ? 'bad' : ''}`;
    };
    const probe = async (label, waitMs = 140) => {
      await new Promise((resolve) => window.setTimeout(resolve, waitMs));
      const count = overlay.animationCount;
      const visible = getComputedStyle(overlay).visibility;
      const open = overlay.isOpen;
      diag.textContent = `${label}\nrunning transitions: ${count}\nvisibility: ${visible}\ndata-open: ${open}\nvariant: ${overlay.variant}\nassert: ${count > 0 ? 'PASS (transition is running)' : 'FAIL (no CSS transition produced)'}`;
      setBadge(`${label}: ${count > 0 ? 'PASS' : 'FAIL'} (${count} anim)`, count > 0);
      ovState.textContent = `overlay: ${open ? overlay.variant + ' / open' : 'closed'}`;
      ctx.log(`${label}: ${count} transitions`);
    };

    const showVariant = async (id) => {
      ctx.setReduce(false);
      overlay.removeAttribute('data-activation');
      await overlay.setVariant(id);
      root.querySelector('#cp-ov-variant').textContent = id;
      await overlay.open();
    };

    for (const [id] of VARIANTS) {
      root.querySelector(`#cp-ov-${id}`).addEventListener('click', async () => {
        await showVariant(id);
        await probe(`open(${id})`);
      });
    }
    on(root, '#cp-ov-close', 'click', async () => {
      overlay.close();
      await probe('close (exit)', 120);
    });
    on(root, '#cp-ov-inner-close', 'click', () => overlay.close());
    on(root, '#cp-ov-replay', 'click', async () => {
      if (!overlay.isOpen) await overlay.open(); else overlay.replay();
      await probe('replay', 120);
    });
    on(root, '#cp-ov-disabled', 'click', async () => {
      overlay.setAttribute('data-activation', 'disabled');
      const ok = await overlay.open();
      diag.textContent = `activation=disabled → open() returned ${ok}；组件被 OverlayActivation 门控，符合上游 OsuFocusedOverlayContainer 行为。`;
      setBadge('gated', null);
      overlay.removeAttribute('data-activation');
    });
    on(root, '#cp-ov-safe', 'change', (event) => {
      overlay.setSafeZone(event.currentTarget.checked);
      ctx.log(`sheared safe zone → ${event.currentTarget.checked}`);
    });

    /* Regress the reported bug: switching variants must never reuse the
     * previous geometry's axis for the first enter. */
    const EXPECTED_AXIS = { wave: 'contentY', sheared: 'plateY', 'slide-x': 'X', 'slide-y': 'Y', pop: 'scale' };
    let axisBusy = false;
    on(root, '#cp-ov-axis', 'click', async (event) => {
      if (axisBusy) return;
      axisBusy = true;
      event.currentTarget.dataset.active = 'true';
      const out = root.querySelector('#cp-ov-axis-out');
      out.innerHTML = '<span class="pv-note">依次切换 5 个 variant 并采样入场方向…</span>';
      const rows = [];
      for (const [id] of VARIANTS) {
        await showVariant(id);
        await new Promise((resolve) => window.setTimeout(resolve, 130));
        const shadow = overlay.shadowRoot;
        const panel = new DOMMatrix(getComputedStyle(shadow.querySelector('.panel')).transform);
        const plate = new DOMMatrix(getComputedStyle(shadow.querySelector('.plate')).transform);
        const content = new DOMMatrix(getComputedStyle(shadow.querySelector('.content')).transform);
        let actual = 'none';
        if (id === 'pop') actual = panel.a < 0.99 ? 'scale' : 'none';
        else if (id === 'wave') actual = Math.abs(content.f) > 20 ? 'contentY' : 'none';
        else if (id === 'sheared') actual = Math.abs(plate.f) > 20 ? 'plateY' : 'none';
        else actual = Math.abs(panel.e) > Math.abs(panel.f) ? 'X' : 'Y';
        rows.push([
          `<code>${escapeHtml(id)}</code>`,
          `<span class="pv-note">${EXPECTED_AXIS[id]}</span>`,
          escapeHtml(actual),
          `${Math.round(panel.e)} / ${Math.round(panel.f)}`,
          `<span class="pv-badge ${actual === EXPECTED_AXIS[id] ? 'ok' : 'bad'}">${actual === EXPECTED_AXIS[id] ? 'PASS' : 'FAIL'}</span>`,
        ]);
        await overlay.close();
      }
      out.innerHTML = table(['variant', '期望轴', '实测轴', 'tx / ty', '结论'], rows);
      const passed = rows.filter((row) => row[4].includes('PASS')).length;
      setBadge(`方向检验 ${passed}/${VARIANTS.length} PASS`, passed === VARIANTS.length);
      ctx.log(`variant axis regression: ${passed}/${VARIANTS.length} PASS`);
      axisBusy = false;
      event.currentTarget.dataset.active = 'false';
    });

    let timer = null;
    on(root, '#cp-ov-cycle', 'click', (event) => {
      if (timer) { clearInterval(timer); timer = null; event.currentTarget.dataset.active = 'false'; return; }
      event.currentTarget.dataset.active = 'true';
      let index = 0;
      let busy = false;
      const step = async () => {
        if (busy) return;
        busy = true;
        const [id] = VARIANTS[index % VARIANTS.length];
        await showVariant(id);
        await probe(`cycle(${id})`, 120);
        index += 1;
        busy = false;
      };
      step();
      timer = window.setInterval(step, 2600);
    });

    const onTheme = (event) => {
      panel?.setTheme(event.detail.theme);
      menu.setTheme(event.detail.theme);
      overlay.setTheme(event.detail.theme);
    };
    window.addEventListener('pv-theme-change', onTheme);
    document.addEventListener('ui-overlay-close', () => { ovState.textContent = 'overlay: closed'; });

    return () => {
      if (timer) clearInterval(timer);
      window.removeEventListener('pv-theme-change', onTheme);
      root.removeEventListener('ui-press', onPress);
      menu.removeEventListener('ui-menu-state', onMenuState);
      menu.removeEventListener('ui-menu-press', onMenuPress);
      panel?.unmount();
      overlay.unmount();
    };
  },
};
