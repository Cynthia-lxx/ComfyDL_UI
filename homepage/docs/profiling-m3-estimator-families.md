# Profiling M3：估算器规则库扩展（按族分批）

M2 交付的 Profiling 面板只认识 14 个 LM/d2l 教学节点，其余一律 `unknown`。
M3 把两本账（M1 内存静态估算 + M2 FLOPs 计算量）的规则库扩到全库：

| 指标 | M2 结束 | M3 结束 |
|---|---|---|
| 已注册估算器 | 14 | **142** |
| ComfyDL 自有节点（`comfydl/nodes/` 109 个） | 0 | **100**（92%） |
| 内置节点（加载族 / 主干 / 图像 IO） | 0 | **28** |

用户拍板的三条边界：**内置只做加载族 + 主干通路**；**按族分批交付、每批独立验收**；
**精度口径维持诚实**——形状可推导就给 exact/approx，权重按 `os.stat` 实测，
推不出来照旧 `unknown` 并写明理由，宁缺毋假。

---

## 1. 批次划分与覆盖清单

### 批 1：张量 / 检测 / 分割 / 图像（约 34 个）

`tensor_basic`（8）、`tensor_ops`（10）、`object_detection`（10）、
`semantic_segmentation`（4）、`image_tools`（5）、`model_cv` 的 `CdlCorr2d`、
内置 `ImageScaleToTotalPixels` / `ResizeImageMaskNode` / `ImageScaleToMaxDimension` /
`LoadImage` / `LoadImageMask` / `LoadImageOutput`。

两个内存大头是静态可算的典型：

* `CdlVocColormap2Label` 恒为 `(256**3,)` int64 = **128 MiB**（无输入、纯常量）；
* `CdlMultiboxPrior` 默认 561×728 特征图、每像素 5 个锚框 → **2,042,040 个锚框 =
  32.7 MB**，面板上一眼能看到"锚框才是内存主体"。

形状公式全部**逐行对齐节点实现**：`CdlConv2d` 的 2D/3D/4D 补齐与 `squeeze(0)`、
`CdlBroadcast`/`CdlReshape` 解析失败时**静默原样返回**、`CdlImageGrayscale` 保持
4 维且通道数 3、`CdlImageRotate(expand=True)` 用 torchvision 的
`ceil(|W·cos| + |H·sin|)`、`ImageScaleToTotalPixels` 的
`round(W·scale/steps)·steps`（Python 银行家舍入）。

### 批 2：模块构造器（约 24 个）

`model_nlp`（6）、`model_seq2seq`（2）、`model_attention`（8）、`model_cv`
（LeNet / ResNet18 / Residual / ResNeXt）、`model_utils`（8）、`nlp_utils`（5）。

参数量一律按 **torch/d2l 源码**手算，不是经验值：

* `nn.RNN`：`H·(I + H + 2)`；`nn.GRU`：`3H·(I + H + 2)`，且**多层时第 2 层起输入
  变成 H**（`formulas.rnn_param_count` 按 torch 真实分层布局，不是"单层×层数"）；
* `RNNScratch` 是三个裸 Parameter：`I·H + H² + H`；
* `RNNLMScratch` 只能接 scratch RNN（它读 `rnn.sigma`），`RNNLM` 只能接
  torch RNN/GRU（lazy head 需要张量输出）——配对错误直接 `unknown` 并说明；
* 多头注意力的**头数不改变参数量**（同一组矩阵按头切片），四个投影含 `W_o`；
* `TransformerEncoder` 每块 `4H² + 2HF + F + 5H + 4H·bias`，编码器内部 lazy 宽度
  可由 embedding 宽度确定，因此可给 **exact**；单独使用的 lazy 模块（Additive /
  MultiHead / FFN / Residual / ResNeXt）给 **approx 并写明假设**；
* `CdlPositionalEncoding` **0 参数但持有 `max_len × H × 4` 字节的普通张量**
  （不是注册 buffer，不进 state_dict）——单独计价。

### 批 3：内置加载族 + 主干（约 28 个）

7 个 loader（`CheckpointLoaderSimple` / `UNETLoader` / `VAELoader` / `CLIPLoader` /
`DualCLIPLoader` / `LoraLoader` / `LoraLoaderModelOnly`）、VAE 编解码、CLIP 编码、
4 个 Save、以及 `SaveImage` / `PreviewImage` / `ImageScale` / `ImageScaleBy` /
`ImageInvert` / `ImageBatch` / `EmptyImage`。

* **权重 = 文件**：`folder_paths` 的目录（含 `unet`→`diffusion_models`、
  `clip`→`text_encoders` 旧别名）以纯数据形式落在 `MODEL_FOLDERS`，最佳努力
  `os.stat`；文件不在 → `unknown`（绝不猜大小）。本 build 不调用
  `load_models_gpu`，因此记的是 **CPU 常驻**。
* **Checkpoint 的分桶不可静态知**：MODEL 桶收未匹配前缀的键，所以只有 MODEL 槽
  拿到字节，CLIP/VAE 槽是 `WeightsVal(-1)`（未知份额），绝不三倍计数。
* **LoRA 不给数**：本 build 的 `LoraLoader` 直接 raise，与其编造 patch 大小，
  不如 `unknown` 并写理由。
* **VAE 用 comfy/sd.py 自己的公式**：`1767·H·W·dtype`（encode）、
  `2178·H·W·64·dtype`（decode，`:432-433`）；8× 与 4 通道是同文件 `:434-438`
  的 SD AutoencoderKL 默认值，但因为 `comfy/ldm` 已脱水、VAE 不可实例化，
  它们登记为**可覆盖假设**（`latent_channels` / `vae_scale`），面板会显示
  "based on assumption"。
* **CLIP 条件同理**：本 build 无文本编码器，77 token × 768 维是假设
  （`clip_tokens` / `clip_hidden`），可在面板覆盖。

### 批 4：收尾降 unknown（约 30 个）

可视化 12 个 + `CdlHeatmapsTo3D` + `misc` 4 + `device_utils` 3 + 数据集 10。

* 可视化统一按 **matplotlib 画布**计价：`figsize × dpi` → `(1, H, W, 4)` RGBA
  float32，缺 widget 时用 rcParams 默认（6.4×4.8 @ 100 dpi），标 approx。
  它们从"unknown"变成"estimated（渲染画布）"，面板不再误报。
* 数据集只计**物化批次**，语料在磁盘且需下载，样本数不硬编码（硬编码就是编造）。
* `CdlLoadArray` → 批次字节；`CdlDataLoaderInfo` → 样本数/批数沿数据流推导。

### 明确不建模（保持 unknown 并写明理由）

| 节点 | 理由 |
|---|---|
| `CdlModelForward` | 模块输出形状不可从模型值推导（部分模块返回 tuple 或需多参） |
| `CdlLoraModelToTensor` / `CdlTensorToLoraModel` / `CdlLossMapToTensor` / `CdlTensorToLossMap` / `CdlValueToTensor` / `CdlTensorToValue` | pack/unpack 的大小取决于运行时字典 |
| `CdlUpdateD` / `CdlUpdateG` | GAN 更新是训练副作用，两个网络均为黑盒 |

---

## 2. 公式层与值对象

`comfy/profiling/formulas.py` 新增（全部有手算金样例）：

```python
conv2d_flops(batch, out_h, out_w, out_c, in_c, kh, kw, groups=1)   # kind "conv"
conv2d_out_dim(size, kernel, stride=1, padding=0)                   # torch floor 规则
conv_param_count(in_c, out_c, kh, kw, bias=True, groups=1)
image_bytes(batch, height, width, channels=3, itemsize=4)           # NHWC
tensor_item(label, nbytes, kind="misc", approx=False)               # 单行内存工厂
rnn_param_count(num_inputs, num_hiddens, gates=1, bias=True, num_layers=1)
gru_param_count(num_inputs, num_hiddens, num_layers=1)
dense_param_count(in_features, out_features, bias=True)             # lazy → None
mha_param_count(in_features, num_hiddens, bias=False)
positional_encoding_bytes(max_len, num_hiddens)
transformer_block_param_count(num_hiddens, ffn_num_hiddens, bias=False)
lenet_param_count(num_classes, in_channels=1, spatial=28)
resnet18_param_count(in_channels, num_classes)
```

`comfy/profiling/shapes.py` 新增两个 EstValue：

* `ModuleVal(param_count, kind_hint, num_hiddens)`：非 LM 模块的输出，携带参数量、
  族别与声明宽度，使"在 RNN 上接输出头"这类下游节点可推导；
* `WeightsVal(file_bytes, source_path)`：加载族输出，`-1` 表示"文件未找到/份额未知"。

`engine.py` 的 `flops.by_kind` 增加 `conv`，`profiler.js` 的构成条与
`profiler.css` 的 `.cdlp-k-conv`（紫）同步；assumption 表新增
`latent_channels=4` / `vae_scale=8` / `clip_tokens=77` / `clip_hidden=768`。

面板**零结构改动**：新增估算器会自动把节点从 unknown 渲染为 estimated，
只需补 zh/en 的 basis key（新增约 25 条）。

---

## 3. 测试

`cdl_smoke_tests/test_profiling_m3.py`，**107 项，全绿**，按族分组：

* **F0 地基**：conv 输出/参数/FLOPs、图像字节、RNN 栈（含"多层不是单层×层数"
  这条 torch 布局）、模块助手——全部纸上手算对照；
* **F1 批 1**：LeNet conv1 → `(6,24,24)` + 172,800 FLOPs；corr2d；字面量解析与
  失败策略；reshape -1；linreg GEMM；multibox 锚框 32,672,640 字节；
  VOC 128 MiB 查找表；内置缩放的 1182×887 / 512×384；LoadImage 的诚实降级；
* **F2 批 2**：RNN/GRU/scratch/seq2seq/Transformer 参数量逐个对照 d2l 源码；
  RNNLM 配对错误 → unknown；ResNet-18 = 11,175,818；LeNet = 61,706；
* **F3 批 3**：真文件 `os.stat`（临时 fixture，用完即删）、缺失文件 → unknown、
  VAE 双向（512² ↔ 64² 潜变量 + comfy/sd.py 工作内存）、假设被记录且可覆盖、
  LoRA 不给数、经典图像管线（batch 叠加、缩放比例推导）；
* **F4 批 4**：渲染画布、misc/设备不再 unknown、数据集按批次计价、
  故意不建模的仍 unknown 且带理由、注册表规模 ≥ 100。

基线不回退：`test_profiling.py` 38 / `test_profiling_m2.py` 26 /
全量冒烟 **290 PASS, 23 SKIP, 0 FAIL（313）**。

---

## 4. 人工验收清单

1. 搭一条含 `CdlMultiboxPrior` → `CdlMultiboxTarget` 的检测链：面板应显示锚框
   占了几十 MB，而不是 unknown；
2. 搭 `CdlRNNScratch` → `CdlRNNLMScratch`：参数量应为 8,288（= 6,208 + 2,080）；
   改成 `CdlRNN` → `CdlRNNLMScratch` 应显示 unknown 并说明需要 scratch RNN；
3. 放一个真实 checkpoint 到 `models/checkpoints/`，用 `CheckpointLoaderSimple`：
   应显示文件字节数（CPU 常驻）；改名让文件找不到 → unknown 而非 0；
4. `EmptyImage` → `VAEEncode` → `VAEDecode`：面板显示潜变量 4 通道 /8，
   并提示这些数字来自假设（可在面板改）；
5. 任意可视化节点：不再是 unknown，显示"画布近似"；
6. 中英切换：新增 basis 文案双语齐全。
