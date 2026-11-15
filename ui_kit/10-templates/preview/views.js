/* SPDX-License-Identifier: MIT
 * View registry + tiny DOM helpers shared by every preview view.
 */

import overview from './view-overview.js';
import theme from './view-theme.js';
import components from './view-components.js';
import motion from './view-motion.js';
import effects from './view-effects.js';
import layers from './view-layers.js';
import data from './view-data.js';

export const VIEW_ORDER = ['overview', 'theme', 'components', 'motion', 'layers', 'effects', 'data'];

export const VIEW_META = {
  overview: { title: '总览', desc: '抽离成果全貌、许可边界、服务器状态与快速跳转', enter: 'rise' },
  theme: { title: '主题 / 排版', desc: '9 套主题、CSS 自定义属性色板、字号层级与 settings 项', enter: 'fade' },
  components: { title: '组件', desc: '5 个组件的全部状态、变体、事件与入/出场动效', enter: 'up' },
  motion: { title: '动效引擎', desc: 'effects.js 全部导出、timeline.js 时间轴、motion.css 关键帧、140 条档案', enter: 'x' },
  layers: { title: '图层编排', desc: '138 项展开/收起全量档案 + 12 层并行揭示的可交互复刻', enter: 'scale' },
  effects: { title: '视觉效果', desc: 'WebGL 三角描边、粒子、光锥、辉光、入场序列、状态反馈', enter: 'elastic' },
  data: { title: '数据 / 装配', desc: 'manifest、i18n、settings、许可审计、API 与装配校验', enter: 'shear' },
};

export const VIEWS = { overview, theme, components, motion, layers, effects, data };

/* ---------------- helpers ---------------- */
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

export function card(title, description, body) {
  return `<section class="pv-card"><h2>${escapeHtml(title)}</h2><p>${description || ''}</p>${body || ''}</section>`;
}

export function on(root, selector, event, handler) {
  for (const element of root.querySelectorAll(selector)) element.addEventListener(event, handler);
}

export function setStatus(root, text) {
  for (const element of root.querySelectorAll('[data-status]')) element.textContent = String(text);
}

export function kv(entries) {
  return `<dl class="pv-kv">${entries.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('')}</dl>`;
}

export function table(headers, rows) {
  return `<table class="pv-table"><thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead><tbody>${
    rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join('')}</tr>`).join('')
  }</tbody></table>`;
}

export function canvasCard(id, caption, height = 150) {
  return `<figure style="margin:0"><figcaption class="pv-note" style="margin-bottom:5px">${escapeHtml(caption)}</figcaption>
    <canvas id="${id}" class="pv-canvas" style="height:${height}px"></canvas></figure>`;
}

export function buttonRow(buttons) {
  return `<div class="pv-row">${buttons.map(([id, label]) => `<ui-button id="${id}" label="${escapeHtml(label)}"></ui-button>`).join('')}</div>`;
}
