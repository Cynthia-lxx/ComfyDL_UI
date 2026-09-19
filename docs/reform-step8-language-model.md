# Reform Step 8 — Language Model Pipeline (Text → Vocab → Train → Generate)

第八步补齐语言模型的端到端闭环：语料进、训练好的 Transformer 出、模型开口续写。新增 13 个核心
节点（3 个掩码 / 位置工具 + 4 个文本节点 + 6 个语言模型节点）、3 个新图数据类型（`VOCAB` /
`MODELSPEC` / `NNMODEL`）与 1 个新分类（`Network & Layers/Text`）；协议层 `comfy/lm_protocol.py`
只 import `torch` 与 `training_protocol`，符合脱水构建规则。需求与草稿清单来自一次外部 gap 分析
（"给一批语料词汇，训练一个语言模型并和它对话"），本步对其去粗取精后实现。

## 1. 动机 / Motivation

第七步之后，注意力已经是无状态核心节点，但搭一个语言模型仍然缺整条流水线：没有分词器、没有下一词
数据集、没有因果掩码生成器，而 `TrainingLoop` 是硬编码 MLP（`_flatten_features` 把输入拍平，
序列维留不住），梯度又无法跨越节点边界（`execution.py:751` 整轮 prompt 包在
`torch.inference_mode()` 里）。因此本步的**通用训练器**只能走"模型对象"路线：结构以冻结 spec 链
声明、单节点体内训练——这是 `inference_mode` 约束下唯一可行的通用形态。

## 2. 节点清单 / Node Inventory

| 分组 | 节点 | 类名 | 说明 |
|------|------|------|------|
| Attention（+3） | Causal Mask | `AttentionCausalMask` | 下三角布尔 `(L, L)` 掩码（含对角线），SDPA 语义 True=可注意 |
| | Padding Mask | `AttentionPaddingMask` | 由有效长度向量生成 `(N, 1, 1, S)` 填充掩码 |
| | Positional Encoding | `AttentionPositionalEncoding` | 固定正弦位置表 `(L, E)`；接 `tensor` 时输出 `tensor + 编码` |
| Text（新分类，4） | Vocab Build | `TextVocabBuild` | 语料 → 冻结词表（词频降序 + 字典序，`<unk>` 固定索引 0） |
| | Text Encode | `TextEncode` | 文本 → 1 维 `torch.long` 索引；未见 token → `<unk>` |
| | Text Decode | `TextDecode` | 索引 → 文本；2 维批逐行解码换行拼接；越界 → `<unk>` |
| | Sliding Window | `TextSlidingWindow` | token 流 → `(x, y)` 下一词预测样本对 |
| Training（+6） | Language Model Embedding | `LanguageModelEmbedding` | spec 链首链接点：词表大小 / 宽度 / 可选内置 PE |
| | Language Model Transformer Block | `LanguageModelTransformerBlock` | 向链追加 pre-LN 块；宽度从链上读取，错配接不进来 |
| | Language Model Build | `LanguageModelBuild` | spec 链 → 带种子物化的 `nn.Module`（Xavier / 零 bias / N(0,0.01) 嵌入） |
| | Language Model Train | `LanguageModelTrain` | 深拷贝上 `steps` ×（前向 + 反向 + `optimizer.step()`），复用 `OPTIMIZER` 槽 |
| | Language Model Forward | `LanguageModelForward` | eval 态纯前向，输出 `(N, L, vocab)` logits |
| | Language Model Generate | `LanguageModelGenerate` | 自回归续写：贪心 / 温度采样，本地 `torch.Generator`，输出 ids + 文本 |

数据流：`Vocab Build → Text Encode → Sliding Window` 产出 `(x, y)`；`Language Model Embedding →
Transformer Block ×N → Build` 产出模型；`TrainingOptimizer + 模型 + (x, y) → Train → Generate`。

## 3. 关键设计决策 / Design Decisions

- **架构路线：模型对象 + spec 链，而非纯 TENSOR 拼装**。`inference_mode` 约束（见动机）使"接收外部
  拼好的模型"的 trainer 到手时计算图已断。结构声明采用 spec 链：每个 spec 节点 optional `spec` 输入
  接前一节、输出累积链，普通槽位即可表达任意深度。哲学与 step6 一致：**结构走连线、超参走 widget、
  参数由 trainer 拥有**。
- **spec 链是冻结 dataclass 元组**（`EmbeddingSpec` 首、`TransformerBlockSpec` 后）：纯值、无张量、
  无设备状态，交给 ComfyUI 缓存安全；`LanguageModelEmbedding` 接到已有链时**替换嵌入链接点、保留
  块**（文档语义与实现一致）；块节点从链读宽度，宽度错配在图上就接不进来。
- **类型命名避开占用**：核心已有 `MODEL`（ModelPatcher），故模型类型叫 `NNMODEL`；`VOCAB` /
  `MODELSPEC` 核对后未占用。三类型照 `PARAMS` 的 `@comfytype` 先例声明于 `_io.py`。
- **trainer 复用 `OPTIMIZER` 槽**（`tp.build_optimizer`），与 step6 训练族无缝互通，省一个节点。
- **缓存正确性**：`LanguageModelTrain` 在 `deepcopy` 上训练，返回**新的** eval 态模型；输入模型的
  缓存值不被污染（冒烟断言 pin 住）。同 seed 完全复现（`seeded_rng` 保存 / 还原进程 RNG）。
- **生成节点文本无关**：输出 token 索引 TENSOR + 解码文本双输出，`vocab` 可选（不接时文本为空串），
  可喂任意 `VOCAB`；贪心（`temperature` 0）与采样（本地 `torch.Generator`）都确定性。
- **pre-LN 块 + 因果掩码 + 逐位置交叉熵**：GPT 式配方。损失对**每个位置**计算（每个 token 预测其后继），
  一次前向 = 全序列监督，300 步教学级训练即收敛；dropout 只在 train 态生效且由 trainer 的种子控制。
- **位置编码两条路径**：spec 链接点 `include_position`（默认开）与独立 `AttentionPositionalEncoding`
  节点（编码器式手动注入）共用 `lm_protocol.positional_encoding` 一张表。
- **文件命名教训**：`comfy_extras/nodes_text.py` 已是上游 Save Text 节点（实现时一度覆盖，git 已
  恢复），文本四件套落在新文件 `nodes_nlp.py`；同理 `comfy/model_protocol.py` 与上游文件撞名，改名
  `comfy/lm_protocol.py`。新增节点文件两个（`nodes_nlp.py`、`nodes_lm.py`）均已加入 `nodes.py`
  `extras_files` 白名单。

## 4. 冒烟器扩展 / Smoke Extensions

- `_VALUE_FACTORIES`：`VOCAB`（"abcabcabd" 5-token 词表）、`MODELSPEC`（16 词 / 宽 8 / 1 块链）、
  `NNMODEL`（seed 0 物化）全局工厂。
- `_INPUT_OVERRIDES`：11 个新节点各配成套输入——掩码节点给 lengths、文本节点给同一词表与 10-token
  流、LM 节点给匹配的 spec / model 与 `(6, 4)` 上下文批 `(6,)` 目标批（同一 `_lm_pair()` 保证一致，
  规避 `_rerun_v3` 未显式传入的张量重新随机的坑）。
- `_OUTPUT_CHECKS`：11 个数值断言——
  - 因果掩码 ≡ `tril`；填充掩码 ≡ 逐位比较 + `max_len` 优先级；PE ≡ `positional_encoding` 表 +
    `tensor + encoding`。
  - 词表确定性 / 词频序 / `<unk>` 首；编码 roundtrip（词表内文本）；未知 token → 0；解码越界 →
    `<unk>`；滑窗每个窗口配对下一 token。
  - Embedding：VOCAB 覆盖控件、接链替换嵌入保留块；Block：追加一链、宽度继承、块首链报错；
    Build：同 seed 同权重 / 异 seed 异权重 / 前向形状；Train：loss 下降、输入模型未被污染、同 seed
    复现、y 形状错配报错；Forward：`(1, L, vocab)` 有限值；Generate：前缀保留、贪心与采样确定性、
    `prefix_ids` 覆盖、无 vocab 时空文本。

## 5. 实测 / Measurements

- 冒烟：`277 PASS / 23 SKIP / 0 FAIL`（300 个已注册节点，46 个分类）；`IMPORT_FAILED []`。
- 功能链路（临时脚本，已删）：默认语料 50 步 loss 4.38 → 0.035；贪心 / 采样生成可用且确定性；
  训练不污染源模型。
- 计数三口径：宿主注册表 287/45 → **300/46**；说明文件 181/33 → **194/34**（core 72 → 85）；
  ComfyDL banner 109/20 不变。

## 6. 回退 / Rollback

本步纯新增（13 节点、3 类型、2 个 extras 文件、1 个协议层文件），无既有节点改动。回退 = 删
`comfy/lm_protocol.py`、`comfy_extras/nodes_nlp.py`、`comfy_extras/nodes_lm.py`，还原
`nodes_attention.py` / `nodes_training.py`（若被触碰）/ `_io.py` / `nodes.py` 白名单，并同步还原
四份说明文件计数。与 step7 同在实验分支续作，合并策略与其一致。
