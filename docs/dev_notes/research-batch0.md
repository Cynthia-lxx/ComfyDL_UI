# 批次 0 调研笔记 — ComfyDL_UI 自定义节点 & 回归/数据集节点开发

> 目的：为批次 1–5（及后续交接给 Hy3 模型执行）提供权威参考，防止实现跑偏。
> 本文所有结论来自只读探查（本地参考库 + 代码），未改动任何文件。
> 生成日期：2026-10-05。

## 1. 关键路径

- 工作仓库：`P:\Dev\ComfyUI_Refs\ComfyDL_UI`（父仓库 master + 子模块 `comfydl` main）
- 本地参考库（用户指定必读）：
  - `P:\Dev\ComfyUI_Refs\ComfyUI_Docs_Work\` — Comfy 官方 Dev Docs 镜像（`development/` `custom-nodes/` `basic-concepts/`）
  - `P:\Dev\ComfyUI_Refs\Comfy_CustomNode_Docs_md\custom-nodes\` — 33 个 `.md` 自定义节点文档
  - `P:\Dev\ComfyUI_Refs\ComfyUI-original\` — Comfy 原版源码（`nodes.py` / `execution.py` / `comfy_api/`）

## 2. V1 节点定义契约（comfydl 沿用）

范本 `ComfyUI-original/nodes.py:56-77`（`CLIPTextEncode`）：
- `INPUT_TYPES`（classmethod）返回 `{"required": {...}, "optional": {...}, "hidden": {...}}`，每槽 `(类型字符串, {选项dict})`
- `RETURN_TYPES`：字符串元组；`RETURN_NAMES`：可选显示名；`FUNCTION`：方法名；`CATEGORY`：菜单路径（支持 `/`）
- comfydl 现有节点即此风格（`comfydl/nodes/tensor_ops.py`、`model_cv.py`、`visualization.py` 等），类型常量集中在 `comfydl/nodes/__init__.py:40-59`

## 3. 自定义类型字符串 DATASET（最关键，防跑偏）

**结论：后端完全接受任意类型字符串，无需注册；对象按引用透传（零拷贝）；唯一必须做的是给「输入」槽加 `{"forceInput": True}`，否则前端把该槽渲染成空控件。**

证据：
- 类型校验无白名单，纯字符串比较：`comfy_execution/validation.py:4-58` 的 `validate_node_input`，只要输出/输入类型字符串一致即过；`execution.py:938` 调用它。
- 值透传：`execution.py:187-190` `input_data_all[x] = cached.outputs[output_index]`，无拷贝/转换。
- 文档佐证：`Comfy_CustomNode_Docs_md/custom-nodes/backend/more_on_inputs.md:53-70` —— “自定义类型几乎只是起个大写唯一字符串名，放进 INPUT_TYPES/RETURN_TYPES；客户端只允许同名互连；可以是任意 Python 对象”，并明确输入须 `{"forceInput": True}`（或 `defaultInput`）。

故新增 `DATASET`：
- 输出节点：`RETURN_TYPES = ("DATASET",)`，`RETURN_NAMES = ("dataset",)`
- 中间/输入节点：`"required": {"dataset": ("DATASET", {"forceInput": True})}`
- 常量（可选，便于复用）：在 `comfydl/nodes/__init__.py` 加 `DATASET = "DATASET"`（照 `cdlVocab`/`cdlDataloader` 先例）

## 4. 插槽配色注入（teal #1ABC9C）

机制：往 6 套主题 palette 的 `node_slot:{...}` 对象追加条目 `,TYPE:`#hex``；前端按类型名查 `node_slot[type]`。
证据：`app/frontend_patch.py`：
- `TENSOR_SLOT_COLOUR = "#C6FF00"`（37），`NN_MODEL_SLOT_COLOUR = "#FF8C42"`（41）；注入串 `,nn_model:`#FF8C42``（42, 259）
- `_patch_nn_model_slot_colour`（237-283）：定位含 `"node_slot:{"` 的 `settingStore-*.js`，向每套 palette 插入（跳过 `...t.colors.node_slot` 合并表达式，270-271），`assert tables == 6`（279-280），用 marker 幂等去重，并清旧 NNMODEL 残留（253）
- patch 清单：`apply_frontend_patches` 的列表（76-82）

**DATASET 照此新增**：`DATASET_SLOT_COLOUR = "#1ABC9C"`，`_patch_dataset_slot_colour(asset)` 复刻 237-283 仅改标记/颜色，挂进 76-82 列表。未知类型即便不注入也会以默认色渲染，注入只为换青绿。

## 5. V1 与 V3 共存

- `comfydl/nodes/*` = V1；`comfy_extras/nodes_training.py` 的训练栈 = V3（`io.ComfyNode` + `define_schema`）。两者共存无碍。
- 决策：DATASET 及所有新 I/O/回归节点继续用 V1（与现有 `cdlVocab`/`cdlDataloader` 一致，最低成本，无需 V3 schema 改造）。V3 训练栈仅作为粗粒度节点内部「被组合」的组件。

## 6. 性能要点（效率至上，生产大负荷）

- 张量返回**保持 batch 维**，勿 squeeze（`tensors.md:34-35` 警告）
- 张量真值判断用 `is not None` 而非 `if a:`（`tensors.md:96-116`）
- **惰性输入**：可能不使用的输入加 `"lazy": True`（`lazy_evaluation.md:9-18` 强烈建议，几乎零成本，可跳过上游子树），配合 `check_lazy_status`
- **对象透传零拷贝**：DATASET dataclass 在节点间引用传递，仅在会 mutate 时才 `.detach().clone()`（参照 `nodes_training.py:523,549`）
- pandas 读取：`usecols`/`dtype`/C 引擎按需加载；`DataFrame→Tensor` 用 `torch.from_numpy`（零拷贝，要求 contiguous + 正确 dtype）；**禁 iterrows**，列级向量化
- 导出大文件：分块写

## 7. 训练栈 API（TrainingLoop 等，供批次 5 组合）

文件 `comfy_extras/nodes_training.py`（全部 V3 `io.ComfyNode`）。

### TrainingLoop（:809）
- 输入：`x`/`y`（TENSOR）、`optimizer`（OPTIMIZER，必填）、`params`（PARAMS，可选 warm start）、`scheduler`（SCHEDULER，可选）、`hidden`（str）、`activation`（combo，默认 relu）、`loss`（combo，默认 mse）、`steps`（int，默认 200）、`batch_size`（int，默认 0=整批）、`seed`（int，默认 0）、`early_stop_patience`（int，默认 0=关）、`early_stop_min_delta`（float，默认 1e-4）
- 输出：`params`（PARAMS）、`loss`（TENSOR 标量）、`loss_history`（TENSOR 1-D）、`prediction`（TENSOR）
- **不吃 DATASET** → 粗粒度节点须先解包 DATASET→x/y TENSOR

### TrainingEvaluate（:1704）
- 输入：`model`（nn_model，可选，优先级高于 params）、`params`（PARAMS，可选）、`x`/`y`（TENSOR）、`activation`（默认 relu）、`loss`（combo，AUTO 默认）、`metric`（combo，EVALUATE_METRIC_OPTIONS，AUTO 默认）
- 输出：`loss`（TENSOR）、`metric`（TENSOR）、`prediction`（TENSOR）；forward-only

### TrainingSaveParameters（:1141）/ TrainingLoadParameters（:1206）
- 写/读 `.safetensors` 到 output 目录；`params`（PARAMS） ↔ `path`（STRING）

### TrainingParameters（显示名 "Learnable Parameters"，:437）
- 输入：`tensor`（TENSOR，可选）、`name`、`shape`、`init`、`seed` → 输出 `params`（PARAMS）

### 损失/指标选项（已核实，可直接映射）
- `comfy/training_metrics.py:24-31` `LOSS_OPTIONS = ("mse","l1","smooth_l1","cross_entropy",...)`
- `:34-39` `METRIC_OPTIONS = ("mae","rmse","accuracy",...)`
- `:57` `EVALUATE_METRIC_OPTIONS = ("auto",) + METRIC_OPTIONS + ("none",)`
- **映射**（粗粒度节点 combo 暴露给用户）：`mse`→`"mse"`、`mae`→`"l1"`、`huber`→`"smooth_l1"`；指标 `mae`/`rmse` 直接可用。
- 结论：粗粒度节点可选的 loss=[mse, mae, huber]、metric=[mae, rmse] 全部被训练栈原生支持，**无需扩展 TrainingLoop**。

## 8. 数据集 I/O 设计要点（批次 1–3）

- `DATASET` 载体（`comfydl/nodes/data_types.py`）：`@dataclass CdlDataset{features: Tensor, labels: Optional[Tensor], feature_names: list[str], target_name: str, meta: dict}`；构造统一经 `from_dataframe(df, target=...)`（零拷贝）或 `from_tensors(X, y, ...)`。
- 读取节点（`comfydl/nodes/data_io.py`）：`CdlReadText` / `CdlReadString` / `CdlReadCSV` / `CdlReadJSON` / `CdlReadXLSX` / `CdlReadDB`（sqlite 经已装 SQLAlchemy）/ `CdlReadAccDB`（pyodbc，惰性 import） → `RETURN_TYPES = ("DATASET",)`。
- 导出节点：`CdlWriteCSV` / `CdlWriteJSON` / `CdlWriteXLSX` / `CdlWriteDB` + `CdlDatasetPreview`（STRING/IMAGE）。
- **依赖**：`requirements.txt` 加 `pandas` + `openpyxl`；pyodbc 仅 accdb 可选，`execute` 内 `try: import pyodbc` 失败抛可读 "pip install pyodbc" 错误（禁止模块加载即崩）。
- 适配器（`comfydl/nodes/dataset_adapters.py`）：`CdlTensorsToDataset`（X,y→DATASET）、`CdlDatasetToTensors`（DATASET→X,y）、`CdlDatasetToLoader`（DATASET→cdlDataloader，复用 `comfydl/nodes/datasets.py:CdlLoadArray` 能力）。
- 类目：`d2l/Datasets`、`d2l/Training`。

## 9. 防跑偏检查单（每批次收尾执行）

1. 新模块在 `comfydl/nodes/__init__.py` 完成 import 注册（否则节点不生效）
2. DATASET 输入槽必须 `{"forceInput": True}`
3. 配色：`app/frontend_patch.py` 新增 `_patch_dataset_slot_colour` 并挂进 patch 清单
4. 五文档更新：`FUNCTIONS.md` / `FUNCTIONS_zh.md` / `README.md` / `README_zh.md` / `comfydl/FUNCTIONS.md`
5. 冒烟：`cdl_smoke_tests/run_smoke_test.py` 保持 291 PASS / 24 SKIP / 0 FAIL 不回退；为新增节点补 `test_data_types.py` / `test_data_io.py` / `test_regression.py`
6. 提交推送：先子模块 `comfydl` 后父仓库；F 盘 `fetch + reset --hard origin/master + submodule update` + GPU 冒烟

## 10. 开放项已全部消解

- 损失/指标映射见 §7，训练栈原生支持 → 批次 5 无需扩展 TrainingLoop。
- DATASET 无需后端注册 → 批次 1 仅加常量 + 配色 + 适配器。
