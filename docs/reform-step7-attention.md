# Reform Step 7 — Attention & the Transformer Encoder Block

第七步补齐核心注意力族：`comfy_extras/nodes_attention.py` 新增 4 个节点（分类
`Network & Layers/Attention`），与 `BasicLinear` 完全同构的无状态约定——四组投影权重
（q / k / v / out 的 weight 必连、bias 可选）全部走 `TENSOR` 插槽，节点内部不做任何初始化；
训练/推理开关经 `Training Mode` 的 `mode` 连线传递；注意力权重的 dropout 由 `seed` 控件
播种的本地 `torch.Generator` 生成。旧 `CdlMultiHeadAttention`（cdlModel 构建器）按弃用
三件套软归档。

## 1. 动机 / Motivation

前六步把 CV 侧（卷积、池化）与训练侧（参数、优化器）都搬到了 bare `TENSOR`，唯独 NLP 的核心
——注意力——还只能在 `cdlModel` 管线里构建。搭一个 Transformer 编码器块需要
`CdlMultiHeadAttention` + `CdlAddNorm` ×2 + `CdlPositionWiseFFN` + …共约 7 个节点，且都在
module 世界里。本步之后：多头注意力、自注意力、交叉注意力与整个 post-LN 编码器块
（`LN → MHA → Add → LN → FFN → Add`）都成为单节点操作，权重照旧从插槽接线。

## 2. 节点清单 / Node Inventory

| 节点 | 类名 | 说明 |
|------|------|------|
| Multi-Head Attention | `AttentionMultihead` | q/k/v 三个 TENSOR + 四组投影权重 + 可选 mask；`nn.MultiheadAttention` 的函数式实现 |
| Self-Attention | `AttentionSelf` | `q = k = v = tensor`，最常用形态 |
| Cross-Attention | `AttentionCross` | q 来自 `tensor`，k/v 来自 `context`；encoder-decoder / 多模态 |
| Transformer Encoder Block | `TransformerEncoderBlock` | 单节点 post-LN 块：注意力权重 + FFN 两矩阵必连，LN 仿射可选（不连 = 无仿射 LayerNorm），FFN 激活 relu/gelu 可选 |

三者共享模块级 `_multi_head_attention()`：投影 → 按 `num_heads` reshape 分头 →
`softmax(QK^T/√d + mask)` →（train 态）带种子 dropout → 输出投影。`E` 从权重形状读出，
不做 channel 控件（沿第四步卷积的决策）。

## 3. 关键设计决策 / Design Decisions

- **mask 语义**：布尔掩码遵循 `F.scaled_dot_product_attention` 的现代约定 —— `True` = 可注意。
  这与 `torch.nn.MultiheadAttention`（`True` = 屏蔽）**相反**，文档与 tooltip 均已写明；浮点
  additive 掩码两侧语义一致（直接加到分数上）。冒烟断言中对照 `nn.MultiheadAttention` 时取反。
- **全屏蔽行不产生 NaN**：被屏蔽位置填 dtype 的有限最小值（`torch.finfo(dtype).min`）而非
  `-inf`，整行被屏蔽的查询 softmax 后是均匀分布。
- **不用 fused SDPA 内核**：为保证带种子的 dropout 与 mask 语义在所有后端逐位一致，显式实现
  softmax 路径（数值上与 SDPA 一致，冒烟断言 pin 到 `nn.MultiheadAttention`）。
- **弃用范围**：仅 `CdlMultiHeadAttention` 软归档到 `d2l/_Legacy/NLP Models`（id 不变，显示名加
  `(DEPRECATED)`）。`CdlDotProductAttention` / `CdlAdditiveAttention` 数学独立（缩放点积 /
  加性注意力），在 cdlModel 管线内无核心替代，保留。
- **跨模块导入**：`nodes_attention` 从 `comfy_extras.nodes_normalization` 绝对导入
  `MODE_TRAIN` / `_mode_input` / `_normalize_mode` / `_dropout` / `_broadcast_stat` /
  `_warn`（extras 加载器按文件路径命名模块，相对导入无父包；与 `nodes_latent` 的先例一致），
  保证 mode 连线契约与 dropout 种子行为只有一份实现。
- **TransformerEncoderBlock 的 LN 仿射可选**：`ln*_weight` / `ln*_bias` 不连 = 无仿射
  `F.layer_norm`；接入时经 `_broadcast_stat` 广播/降级，与归一化家族的统计插槽同一契约。

## 4. 冒烟器扩展 / Smoke Extensions

- `_INPUT_OVERRIDES`：4 个节点各配成套形状匹配工厂（`q/k/v = (2,5,8)` / `(2,7,8)`、投影权重
  `(8,8)`、FFN `(32,8)` / `(8,32)`）——通用 `(2,3)` dummy 无法与权重大矩阵相乘。
- `_OUTPUT_CHECKS`（用 `_rerun_v3` 对照 torch 参考实现）：
  - `AttentionMultihead` ≡ `nn.MultiheadAttention`（batch_first、打包 `in_proj_weight`）；
    bias / additive mask / 布尔 mask（取反对照）分支各验一次；`eval` 态 dropout 恒等；同 seed
    逐位一致、异 seed 不同。
  - `AttentionSelf` / `AttentionCross` ≡ `AttentionMultihead` 的退化情形（q=k=v / k=v=context）。
  - `TransformerEncoderBlock` ≡ 手工 `F.layer_norm` + MHA + `F.linear` 组合；affine 与 gelu
    分支各验一次。

## 5. 目录与计数 / Files & Counts

| 文件 | 变化 |
|------|------|
| `comfy_extras/nodes_attention.py` | 新增（4 节点 + helper，约 600 行含 docstring） |
| `nodes.py` | `extras_files` 白名单加 `nodes_attention.py` |
| `cdl_smoke_tests/run_smoke_test.py` | 6 个工厂 + 4 组 overrides + 4 个检查函数 |
| `comfydl/nodes/model_attention.py` | `CdlMultiHeadAttention` 弃用三件套 |
| `comfydl/FUNCTIONS.md` / `FUNCTIONS_zh.md` | §17 → 49 节点、新增 17.8 Attention、弃用标记与计数 |
| `comfydl/README.md` / `README_zh.md` | 类别表与总句 → 181 / 33 = 109 + 72 |
| `comfydl/_update_readme.py` | `_CAT_DESC` 的 _Legacy/NLP Models 行加 Multi-Head Attention |
| `locales/zh/nodeDefs.json` | 重生成（109 条不变，显示名更新） |
| 本文档 | 新增 |

- 宿主注册表：283 → **287** 个节点、44 → **45** 个分类。
- 冒烟基线：260 → **264 PASS** / 23 SKIP / **0 FAIL**。
- 说明文件口径：**181 个节点 / 33 个分类** = 109 ComfyDL + **72** core（N&L 45→49，
  `_Legacy/NLP Models` 3→4，`NLP Models` 13→12）。

## 6. 回退 / Rollback

单步回退：主仓删除 `nodes_attention.py` 与 `nodes.py` 白名单行、还原冒烟器改动；子仓把
`CdlMultiHeadAttention` 的 `DEPRECATED` / `CATEGORY` / 显示名还原并重跑
`_update_readme.py` 与 `gen_locales.py`。所有 id 均未改变，旧工作流不受影响。

## 7. 执行与实测记录 / Execution Log

- 快速数值自检（对照 `nn.MultiheadAttention` / `F.scaled_dot_product_attention`）：默认、bias、
  additive mask、布尔 mask、dropout 确定性、unbatched、非方阵 out、块组合（含 affine / gelu）、
  dtype 提升全部一致（最大偏差 1.9e-6）。期间定位到 torch `nn.MultiheadAttention` 布尔掩码为
  `True`=屏蔽 的旧语义（与 SDPA 相反），本节点采用 SDPA 语义并写入文档。
- 冒烟：`--filter Attention` 6 PASS / 0 FAIL；`--filter TransformerEncoderBlock` 2 PASS / 0 FAIL；
  全量 **264 PASS / 23 SKIP / 0 FAIL（287 个节点）**。
- `comfydl/_update_readme.py --check`：四份文档与注册表一致。
- `cdl_smoke_tests/gen_locales.py --check`：语言包最新（109 条）。
