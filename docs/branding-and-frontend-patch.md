# 品牌化与前端补丁层 / Branding and frontend patch layer

> 适用范围：ComfyDL_UI v0.3.1 起的身份统一（名称/版本/版权）与 `app/frontend_patch.py`
> 前端补丁层、仓库级语言包注册。本文同时记录回退方式与验证判据。

## 1. 背景与目标

ComfyDL_UI 是 ComfyUI 的脱水 fork，前端来自第三方 pip 包
`comfyui-frontend-package`（安装位置 `penv/Lib/site-packages/comfyui_frontend_package/static/`），
**不在版本控制内**。此前为了给 `TENSOR` 插槽上色，是直接手工改这个包里的
`settingStore-<hash>.js`，一旦升级前端包（文件名 hash 变化）改动就会丢失，别人克隆仓库也复现不出
来。本次改造把这类改动收编为一个**随仓库走的启动期补丁层**，并顺带完成品牌、版权与多语言的统一。

目标：**克隆 + 配置依赖 + 启动 = 与本机一致的界面效果**，无需任何手工改包操作。

## 2. 身份与版权统一

| 位置 | 改动 |
|---|---|
| `pyproject.toml` | `name = "ComfyDL_UI"`，`version = "0.3.1"`，新增 description / authors，`[project.urls]` 指向本项目并保留 `Upstream` 指向原版仓库 |
| `comfyui_version.py` | `__version__ = "0.3.1"`（启动时由 `/system_stats` 上报，同时驱动「关于」面板里的版本号） |
| `main.py` | 启动日志 `ComfyDL_UI version: ...` |
| `LICENSE` | 保留 GPL-3.0 全文，末尾追加本项目版权声明；原版 ComfyUI 的归属单独列出 |
| `README.md` | 顶部改为 ComfyDL_UI 自述（是什么 / 不是什么 / 原版去哪下载），原版徽章、官网、社区链接移入独立的「Upstream ComfyUI (original project)」分区；其后全部上游文档按原样保留并加归属说明 |
| `README_zh.md` | 新增，中文同口径自述 |
| `comfydl/`（子模块） | `LICENSE` MIT → GPL-3.0；`pyproject.toml` license → `GPL-3.0-or-later`；`README.md` / `README_zh.md` 增加「ComfyDL 有自己的 GUI 版本 ComfyDL_UI」说明，安装章节改为「方式一：GUI 版本 / 方式二：原经典四步」；`FUNCTIONS.md` / `FUNCTIONS_zh.md` 同步许可证与 GUI 口径（节点数量口径不变：177 节点 / 32 类别） |

## 3. 前端补丁层

`app/frontend_patch.py` 对外只有一个入口：

```python
apply_frontend_patches(web_root: str) -> None
```

调用点在 `server.py` 的 `PromptServer.__init__`：前端根路径解析完成之后、任何静态资源被服务之前。

```python
self.web_root = (
    FrontendManager.init_frontend(args.front_end_version)
    if args.front_end_root is None
    else args.front_end_root
)
logging.info(f"[Prompt Server] web root: {self.web_root}")
# Re-apply the ComfyDL_UI UI changes before the static files are served.
apply_frontend_patches(self.web_root)
```

### 3.1 设计约束

- **按内容标记定位**，不依赖带 hash 的文件名：前端包升级后补丁会自动重新应用到新文件。
- **幂等**：每个补丁在自己的目标文件里写一个标记注释（`/*ComfyDL_UI:xxx*/`）；
  标记已存在则直接跳过，重复启动零成本。
- **不阻塞启动**：每个补丁独立 `try/except`，失败只打印 ASCII 告警（例如「标记没找到」），
  服务照常以原版行为启动。
- **尊重 `--front-end-root`**：补丁目标是解析后的 web root，而不是写死 `penv/...`。

### 3.2 四个补丁

| 补丁 | 目标资产 | 内容 |
|---|---|---|
| 默认语言 | `i18n-*.js` | `getDefaultLocale()` 返回 `en`。它同时喂给 vue-i18n 的初始 locale 与 `Comfy.Locale` 的设置默认值，因此一处即可覆盖两条路径 |
| Nodes 2.0 默认 | `GraphView-*.js` | 在 `Comfy.VueNodes.Enabled` 设置定义对象内把 `defaultValue` 与 `defaultsByInstallVersion` 由 `!1` 改为 `!0` |
| 关于面板 | `AboutPanel-*.js` | 注入 ComfyDL_UI 自述区块（名称+版本、定位说明、「这不是完整原版 ComfyUI」、本项目仓库、原版官方仓库），并在原版链接列表之前加一个「Upstream ComfyUI (original project)」小标题；原版徽章里的 `ComfyUI <版本>` 改为 `ComfyUI`（版本号属于本 fork，展示在自述区块） |
| TENSOR 插槽配色 | `settingStore-*.js` | 六张主题调色板（`arc`/`dark`/`github`/`light`/`solarized`/`nord`）的 `colors.node_slot` 各插入 `TENSOR:`#C6FF00``，跳过 `node_slot:{...t.colors.node_slot,…}` 这种调色板合并表达式；找到的表数不为 6 则拒绝写入 |

关于面板注入的 Vue 渲染片段所用的压缩变量名（`createElementVNode`、`toDisplayString`、
systemStats store 等）是**从被改文件里用正则抓出来的**，因此前端升级重新压缩也不会错位。

## 4. 语言包注册

原版前端内置多种语言，但「我们加入的部分」原先只有英文。现在：

- 语言包放在仓库根 `locales/<lang>/{main,nodeDefs}.json`：
  - `main.json`：本项目的界面文案（目前是「关于」面板自述区块的 `comfydlAbout.*` 命名空间）。
  - `nodeDefs.json`：节点显示名与说明的译文（**只有中文包**；英文即注册表本身，缺键时前端自动回退）。
- 注册链路复用 ComfyUI 的官方机制：`app/custom_node_manager.py` 的
  `CustomNodeManager.build_translations()` 增加了仓库级 `locales/` 来源
  （`REPO_LOCALES_DIR` + 抽出的 `load_locales_dir()`），`/i18n` 因此会返回我们的条目，
  前端启动时通过 `mergeCustomNodesI18n` 合并进 vue-i18n。
- 节点译文由 `cdl_smoke_tests/gen_locales.py` 生成：`display_name` 取
  `comfydl/FUNCTIONS_zh.md` 中该节点的中文小节标题（标题仍是英文术语的节点不生成该字段），
  `description` 取同一小节的 `- **功能**：` 一行。`--check` 可用于校验语言包是否与文档同步。

## 5. 验证判据

1. 启动日志出现 `ComfyDL_UI version: 0.3.1`，且无 `frontend patch ... not applied` 告警；
   `[ComfyDL_UI] frontend` 行确认四项补丁各自已应用或已是最新。
2. 浏览器打开界面：设置面板中 Nodes 2.0 为开启、语言为 English；画布上 `TENSOR` 插槽为
   `#C6FF00`；设置 → 关于 显示 ComfyDL_UI 自述区块，其下方是独立的 Upstream 分区。
3. 切到中文后，关于面板的自述文案与节点菜单中的中文节点名/说明生效。
4. 重启一次，补丁层全部走「已是最新」分支，资产字节数不变（幂等）。
5. `python cdl_smoke_tests/run_smoke_test.py` 全绿（0 FAIL）。
6. `python cdl_smoke_tests/gen_locales.py --check` 返回 0。

## 6. 回退

- 前端补丁：删除 `server.py` 中的 `apply_frontend_patches(...)` 调用（或在
  `app/frontend_patch.py` 里注释掉对应子补丁）后，重装/升级前端包即可回到原版行为。
- 语言包：删除仓库根 `locales/` 即回到「只有英文」；`/i18n` 只是少一个来源，不影响其它功能。
- 版本与版权：`comfyui_version.py` / `pyproject.toml` 是可单独回退的文本改动，
  与前端补丁层互不依赖。
