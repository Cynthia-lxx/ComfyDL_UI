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

## 7. 模板目录补丁层（Templates 面板）

> 新增于 v0.3.1：`app/template_catalog.py`。与第 3 节的启动期 JS 补丁不同，这一层是
> **请求期的服务端改写**——不改 `site-packages` 里的任何文件（`comfyui-workflow-templates`
> 系列包同受第 3 节"绝不直改包"约束保护），包升级后逻辑自动对 newIndex 重算，幂等。

### 7.1 背景

- 前端 Templates 面板的数据来自 `/templates/index.json`（及本地化 `index.<locale>.json`、
  机器侧 `index.mcp.json`），文件在 PyPI 包 `comfyui-workflow-templates`（数据实体为
  `comfyui_workflow_templates_json`，当前 0.1.61）。
- 本仓库是脱水构建：后端注册表只保留 302 个节点。**对 521 个官方模板逐一核对（含内嵌
  subgraph 定义与前端内置类型 Note/MarkdownNote/PrimitiveNode/Reroute 的白名单）后，519 个
  模板引用了已删除节点**；仅 `basic_mask_operations_and_compositing` 与
  `utility_image_stitch` 两个 Node Basics 模板幸存。
- 另一个实情：`server.py` 按 `use_legacy_templates`（安装版本 < 0.3.0 时为真）走
  `web.static('/templates', legacy_templates_path())`，但**元包内根本没有 `templates/`
  目录**——即在此修复之前，面板对所有 `/templates/*` 请求都是 404（面板完全空白）。

### 7.2 实现与调用链

```
server.py  PromptServer.__init__
  ├─ use_legacy_templates 且元包 templates/ 目录真实存在 → 原版 web.static（不动）
  └─ 否则（本仓库现状）→ FrontendManager.template_asset_handler()
       └─ app/template_catalog.build_handler(assets, legacy_dir)
            1. resolve_overlay_asset()      仓库自有资产（ComfyDL 示例工作流 + 封面）
            2. curated_index_payload()      index*.json 请求期改写（缓存）
            3. packaged asset map           官方包逐文件服务（template_asset_map()，1254 项）
            4. legacy 目录兜底（带路径穿越防护）
```

- **死模板过滤**：懒加载扫描全部打包工作流 JSON，对照 `nodes.NODE_CLASS_MAPPINGS`；
  引用缺失节点类型即剔除，结果进程内缓存。注册表未初始化时 fail-open（不改写）。
- **分类重组**：剔除死模板后清空的全部分类一并移除；在首位注入
  `ComfyDL Examples` 分类（`moduleName: default`——**必须**是 `"default"`，见下条
  「moduleName 硬约束」；`isEssential: true`，zh 标题「ComfyDL 示例」），侧边栏最终形态：
  `All Templates / Popular`（前端硬编码）+ `ComfyDL Examples` + `Node Basics`。
- **本地化**：`index.<locale>.json` 同样过滤，注入分类应用 `_CATEGORY_TITLE_OVERRIDES` /
  `_CATEGORY_DESCRIPTION_OVERRIDES`（目前只有 zh）；`index.mcp.json` 只过滤不注入；
  `index_logo.json` / `index.schema.json` 原样透传。
- **命名规范（硬约束）**：模板 `name`（=文件名主干）必须匹配官方 schema 模式
  `^[a-zA-Z0-9._-]+$`，**禁止空格与非 ASCII**。这不是仅 CI 洁癖——2026-09-20 实测故障：
  早期版本用带空格文件名（`Language Model - Train and Chat.json`），前端
  `fetchTemplateJson` 直接拼 `/templates/${name}.json` 不做编码，浏览器把空格转义为
  `%20` 发出；而 aiohttp 动态路由 `{path:.*}` 的 match info 只解码 `%2F`/`%25`
  （`web_urldispatcher._unquote_path_safe`），handler 拿到带字面 `%20` 的字符串 →
  白名单匹配失败 → 404 → 前端 `.json()` 解析失败静默 return false，**点击卡片无任何反应**。
  修复双管齐下：文件重命名为官方风格（`language_model_train_and_chat.json`，封面直接命名
  `<name>-1.jpg` 命中前端缩略图 URL 模式），handler 入口对 rel_path 做
  `urllib.parse.unquote` 兜底（防未来任何非 ASCII 名）。
- **moduleName 硬约束（2026-10-04 修复「点击卡片无反应」）**：前端把分类的
  `moduleName` 原样拷进每张卡片的 `sourceModule`，且
  `fetchTemplateJson` / `getTemplateThumbnailUrl` **只认 `moduleName === "default"`**
  走核心通道 `/templates/<name>.json`（缩略图 `/templates/<name>-1.jpg`）；任何其他值
  都被路由到自定义节点通道 `/api/workflow_templates/<moduleName>/<name>.json`——该通道
  只扫 `custom_nodes/` 目录，内置 overlay 不在其中 → 404 → 前端 `.json()` 抛异常后
  **静默 return false，点击毫无反应**（缩略图同因失效）。这正是 2026-10-02 在案的
  「Train and Chat / Load and Chat 入口坏了」的根因：注入分类当时用了
  `moduleName: "ComfyDL"`。修复 = 改回 `"default"`（与全部官方分类一致），
  JSON 与缩略图两条 URL 一并回到已验证的 `/templates/` overlay 通道。
  回归测试：`test_templates_catalog.py` T3b2。

### 7.3 如何新增示例工作流

1. 把工作流 JSON 放进 `comfydl/example_workflows/`，**文件名主干必须匹配
   `^[a-zA-Z0-9._-]+$`**（小写下划线风格，参照官方如 `basic_mask_operations_and_compositing`）；
2. 封面命名为 `<name>-1.<mediaSubtype>`（直接命中前端缩略图 URL 模式）；若复用其他
   模板的封面，在 `_THUMBNAIL_FILENAMES` 指过去；
3. 在 `app/template_catalog.py` 的 `_COMFYDL_CATEGORY["templates"]` 加一条元数据
   （`name` = 文件名主干；`mediaType` ∈ image/video/audio/3d，`mediaSubtype`
   为封面扩展名）；
4. 需要中文标题/描述时补 `_CATEGORY_TITLE_OVERRIDES` / `_CATEGORY_DESCRIPTION_OVERRIDES`；
5. `penv\Scripts\python.exe cdl_smoke_tests\test_templates_catalog.py` 全绿即可
   （T3c/T5/T8 会自动覆盖新条目）。

### 7.4 验证判据

1. 启动后打开 Templates 面板：侧边栏为 `All Templates / Popular / ComfyDL Examples /
   Node Basics`；`ComfyDL Examples` 下两张工作流（Train and Chat / Load and Chat）带封面；
2. 任一官方死模板不再出现；`Node Basics` 下恰为两个幸存模板且可正常加载；
3. 界面语言切中文后，`ComfyDL 示例` 分类与中文描述生效；
4. `penv\Scripts\python.exe cdl_smoke_tests\test_templates_catalog.py` 全绿（42 项，
   含 T9 路由层集成：真实 aiohttp 服务验证编码 URL / 穿越拒绝 / 官方资产对照）；
   全量 `run_smoke_test.py` 0 FAIL。

### 7.5 回退

- 删除 `server.py` legacy 分支中的 handler 回退调用即恢复原版 `web.static` 行为
  （即回到「面板 404」的现状）；新分支（包版本 ≥ 0.3.0）中 `template_asset_handler`
  的 overlay 行为同样在 `app/template_catalog.py` 内可整体旁路——
  把 `build_handler` 换回纯 `assets.get()` 查表即回到原版逐文件服务。
