# Profiling 诊断日志（四档详细度）

> 2026-10-06 引入。Profiling 面板（Templates 系列文档见
> `profiling-m1-memory-estimation.md` / `profiling-m2-compute-and-watchdog.md` /
> `profiling-m3-opgraph.md`）此前是黑盒：Analyze 卡住或按钮无反应时没有任何
> 输出可看。本机制给整条 profiling 链路一个仿 pip `-v/-vv/-vvv` 的详细度旋钮。

## 档位语义

| 档 | 内容 |
|----|------|
| `off`（默认） | 一行不输出，零开销 |
| `low` | 请求生命周期：estimate / opgraph / postmortem 的开始与结束（含耗时）、汇总数字（probed/fallback/verdict）、客户端 4xx、worker 崩溃 |
| `medium` | 逐节点探针结果（节点 id / class_type / status / ops / flops / probe_ms / error）、opgraph 缓存命中 |
| `high` | 其余细节：缓存键、前端状态变迁（displayMode、stale 标记）、watchdog 轮询、各 fetch 的状态码 |

## 配置通道（三者取最大，互不冲突）

1. **启动参数**：`--cdl-profiling-log {off,low,medium,high}`（默认 `off`）。
   定义于 `comfy/cli_args.py`，由 `comfy/profiling/proflog.py` 惰性读取
   （该模块纯 stdlib，绝不引入 torch）。
2. **前端设置**：Settings → ComfyDL → Profiling → *ComfyDL profiling log
   level*（官方 settings API 注册的 combo，改动即时生效，无需重启）。
3. **请求头**：前端每次调用 profiling 端点都会带
   `X-CDL-Profiling-Log: <level>`；后端 `proflog.level_for()` 取
   max(启动档, 头档)——**头只能升不能降**（排障时不必重启服务）。

## 日志去向

- 后端：标准 `logging.getLogger("ComfyDL.profiling")`，格式
  `[profiling:<rank>] <message>`，INFO 级 → 自动进入 `app/logger.py` 的
  300 条环形缓冲，并可经 `GET /internal/logs/raw` 拉取。
- 前端：`console.log("[ComfyDL profiler][<rank>] ...")`（profiler.js 的
  `plog()`）。
- 面板内：档位非 `off` 时，侧栏出现「Server log (profiling)」卡——按需拉
  `/internal/logs/raw` 并只显示本组件的行（后端 `[profiling:` + 前端 toast
  兜底行），带 Refresh 按钮。

## 使用范式（排障 SOP）

1. Settings 里把档位调到 **high**（或启动时加 `--cdl-profiling-log high`）；
2. 复现问题（点 Analyze / 跑工作流）；
3. 收集：浏览器 DevTools console（`[ComfyDL profiler]` 前缀过滤）+ 面板内
   「Server log」卡（或 `curl http://127.0.0.1:8188/internal/logs/raw`）；
4. 诊断完成后调回 off。

## 同日修复（本轮日志系统引入的契机）

`profiler.js` 的 `renderOverlay()` 曾被 4 处调用但**从未定义**：

- **Expand 无反应**：`mountOverlay()` 在启动时已建好 overlay 而早退，
  `setDisplayMode → renderOverlay()` 抛 ReferenceError，`cdlp-overlay-open`
  class 永远加不上；
- **Analyze 永久卡 "Analyzing"**：`requestOpgraph()` 的首次渲染在 **try 块
  外**，`renderOverlay()` 抛错后函数中止——loading 已置位、UI 已重绘，但
  fetch 从未发出、重置代码永不执行。

修复：补上 `renderOverlay()`（仅切换 overlay 可见性，内容仍由
`renderPanel()` 双容器渲染）；`requestOpgraph()` 全面包进
`try/finally`（loading 必然复位）。静态断言（T4b/c）已入
`cdl_smoke_tests/test_profiling_log.py` 防回归。

## Master switch（2026-10-07）

**动机**：多次实测证明 profiling 的存在必然使工作流变慢（前端每次 Run 一次
estimate 往返 + graphToPrompt 劫持、后端 watchdog 250ms 恒采样、per-run meter
记账）。给用户一个完全停用 / 一键恢复的开关。

**软开关（热切换，无需重启）**：Settings → ComfyDL → Profiling → Enabled
（默认 on）。

- 停用（前端）：恢复被劫持的 `graphToPrompt`（保守恢复——仅当当前引用仍是自家
  patched 函数，防破坏其他扩展的包装链）、移除 graphChanged/executing/executed/
  execution_error 四个监听、清 auto-estimate 定时器、隐藏顶栏徽章与 watchdog
  胶囊；侧栏 tab 与 Expand/overlay 切换保留，两面板变为居中空态（图标 +
  "Profiling currently disabled" + Enable 按钮，风格参考宿主 Assets 空态）。
- 停用（后端，经 `POST /comfydl/profiling/enabled`）：`watchdog.stop()`；
  `comfy/profiling/proflog.py` 的 master flag 置 off → `meter_from_args()` 返回
  None、`persist_run()` 早退（executor 零改动）。
- 恢复：同一开关切回 on（面板空态的 Enable 按钮走同一 Settings 通道），
  watchdog 重启、劫持重挂，全部幂等可重入。

**硬关（启动参数）**：`--cdl-profiling-disable`——不启动 watchdog、flag 恒 off
且 `POST /enabled` 拒绝改回（409），前端空态显示"需去除参数重启"说明而无
Enable 按钮。`GET /comfydl/profiling/enabled` 返回 `{enabled, hard_disabled}`
供前端区分。

**分层**：master flag 落 comfy 层（`proflog.py`，stdlib-only）——app 层写、comfy
层（runmeter）读，符合"comfy 层不得 import app 层"铁律。手动路由
（estimate/opgraph/history）停用后保留，但空态不发请求。

**测试**：`test_profiling_log.py` T1e-f / T5a-p / T4aq-bb；真启动
`--cpu --quick-test-for-ci --cdl-profiling-disable` EXIT=0（坑 25）。
