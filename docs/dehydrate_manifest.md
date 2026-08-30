# ComfyUI Dehydrate 脱水清单（适配 ComfyDL）

> 版本：ComfyUI v0.34.0
> 目标目录：`ComfyDL_UI/`（改装操作目录）
> 参照副本：`ComfyUI-original/`（只读权威回捞源）
> 生成依据：`.codebuddy/scripts/import_scan.py` 全库 AST 静态分析 + code-explorer 人工依赖核查
> 数据快照：`.codebuddy/scripts/import_report.json`（797 模块，闭合集 357 / 删除候选 439）

---

## 1. 项目背景与目标

将 ComfyUI 从"AI 图像生成引擎"脱水瘦身为"ComfyDL 深度学习教学节点的宿主运行时"：

- **保留**：带颜色的数据槽类型系统（L0）、数据结构协议（L1）、节点注册机制、执行引擎、服务层骨架、基础 IO 图像节点。
- **删除**：扩散模型生成实现层（L2，ldm/text_encoders/k_diffusion/采样器/云端节点等）。
- **收益**：源码体积大幅缩减、启动时间与内存占用下降、工程聚焦教学场景。
- **安全网**：所有删除项均可从 `ComfyUI-original/` 按清单 10 秒回捞；`compare_dups.py` 提供哈希校验。

## 2. 双目录策略与回捞安全网

| 目录 | 角色 | 规则 |
|---|---|---|
| `ComfyDL_UI/` | 改装操作目录 | 所有改动在此进行；已 git init（初始提交 `3c394c3`） |
| `ComfyUI-original/` | 原始参照 | 只读，绝不动；回捞唯一权威源 |

- 删除采用"先登记清单 → 再删除 → git commit 检查点"三步走。
- 回捞命令范式：`copy ComfyUI-original\comfy\xxx.py ComfyDL_UI\comfy\xxx.py`。
- 校验工具：`.codebuddy/scripts/compare_dups.py`（哈希对比两目录）。

## 3. 三层保留策略（L0 / L1 / L2）

| 层 | 内容 | 处置 |
|---|---|---|
| **L0 类型系统** | `comfy/comfy_types/`：`IO(StrEnum)` 全部约 30 个数据槽类型（IMAGE/MASK/LATENT/MODEL/CLIP/VAE/CONDITIONING/CONTROL_NET/CLIP_VISION/STYLE_MODEL/GLIGEN/UPSCALE_MODEL/AUDIO/ANY 等）+ `InputTypeDict` + `ComfyNodeABC` + `FileLocator` | **全量保留**（零成本，前端连线颜色体系依赖类型名） |
| **L1 数据结构** | `model_patcher.py`（MODEL 结构）、`sd.py` 拆分后的 CLIP/VAE 类骨架、LATENT/CONDITIONING 数据约定、`conds.py` | **保留最小集**（类骨架 + 数据约定，与生成实现解耦） |
| **L2 生成实现** | ldm/text_encoders/k_diffusion/采样器/模型加载/VAE 编解码实现 | **删除**，回捞安全网兜底 |

## 4. 保留清单（目标态）

### 4.1 入口与服务层（全量保留）
```
main.py  server.py  execution.py  nodes.py  folder_paths.py  node_helpers.py
latent_preview.py(改造)  protocol.py  comfyui_version.py  cuda_malloc.py
extra_model_paths.yaml.example  hook_breaker_ac10a0.py
comfy_execution/ (9)   comfy_api/ (36，含 internal/version_list/latest，schema 核心)
app/ (36)  api_server/ (6)  middleware/ (2)  utils/ (5)  comfy_config/ (2)
alembic_db/ alembic.ini  custom_nodes/  input/  output/  models/(空)
```

### 4.2 comfy/ 核心子集（A 档，改造后保留）
```
comfy/__init__.py  comfy_types/  (L0 全量)
cli_args.py  options.py  internal_logging.py  deploy_environment.py  comfy_api_env.py
model_management.py  memory_management.py  system_memory.py  model_prefetch.py
model_patcher.py  (L1，移除 lora import 后保留)   hooks.py  (改造)   patcher_extension.py
float.py  ops.py  quant_ops.py  pinned_memory.py  utils.py  multigpu.py
sd.py  (L1 拆分后仅类骨架)
conds.py  (CONDITIONING 数据约定，若拆分后无生成依赖则保留)
nested_tensor.py  (待核查：仅被生成层引用则可删)
```

### 4.3 comfy_extras 保留名单（依赖已核实干净，见 §6.3 判定）
```
nodes_mask.py       # MASK 类型链：ImageToMask / MaskToImage / InvertMask 等
nodes_images.py     # IMAGE 链：SaveImageWebsocket 等
nodes_post_processing.py  nodes_compositing.py  nodes_rebatch.py  nodes_resolution.py
nodes_color.py      nodes_logic.py  nodes_math.py  nodes_primitive.py  nodes_number_convert.py
nodes_nop.py  nodes_curve.py  nodes_string.py  nodes_text.py
nodes_latent.py     # LATENT 数据结构操作（L1 层，不依赖生成模型）
nodes_toolkit.py
```
> 其余 extras 文件（约 120+）在任务 5 执行时按"依赖干净 + IMAGE/MASK/LATENT 链"逐文件复核，未过审即列入删除清单。

## 5. 改造清单（锚点，必须先于删除执行）

本版本生成层经 4 处锚点**强制进入入口 import 链**，不改造则无法删除。顺序：先 5.5 → 再 5.1~5.4。

### 5.1 `nodes.py`（任务 3）
- 移除顶层生成 import（第 23-34 行附近）：
  `comfy.diffusers_load` / `comfy.samplers` / `comfy.sample` / `comfy.sd` / `comfy.controlnet` / `comfy.clip_vision`
- 删除生成节点类及映射（约 60 个）：KSampler、KSamplerAdvanced、CheckpointLoaderSimple、CLIPTextEncode、CLIPSetLastLayer、VAE 系（Decode/Encode/EncodeForInpaint/Loader/DecodeTiled/EncodeTiled）、EmptyLatentImage、Latent 系、Conditioning 系、LoraLoader、CLIPLoader、UNETLoader、DualCLIPLoader、CLIPVision*、StyleModel*、ControlNet*、DiffControlNetLoader、GLIGEN*、unCLIP*、DiffusersLoader、LoadLatent/SaveLatent、InpaintModelConditioning 等。
- **保留**：LoadImage、LoadImageMask、LoadImageOutput、SaveImage、PreviewImage、ImageScale、ImageScaleBy、ImageInvert、ImageBatch、EmptyImage、ImagePadForOutpaint（待定，看依赖）。
- 保留 `load_custom_node` / `init_extra_nodes` / `init_builtin_api_nodes` 机制不变。

### 5.2 `model_patcher.py`（任务 2）
- 第 34 行 `import comfy.lora`：改为延迟注入。`calculate_weight` 相关逻辑提供 stub，注册 `lora.calculate_weight` 为可替换钩子；无 lora 时返回原始权重。

### 5.3 `hooks.py`（任务 2）
- 第 14 行 `import comfy.lora`：同样延迟化或删除（hooks 本身被 model_patcher/samplers 使用；samplers 删除后 hooks 仅服务 model_patcher）。

### 5.4 `latent_preview.py`（任务 2）
- 第 4 行 `from comfy.taesd.taesd import TAESD`、第 5 行 `from comfy.sd import VAE`：改为惰性导入，preview 方法缺 VAE/TAESD 时降级为"无预览"或仅保存图像。

### 5.5 `comfy/sd.py` 拆分（任务 2）
- 保留：`CLIP`、`VAE`、`ModelPatcher`（实际在 model_patcher.py）的**类定义与数据约定**（encode/decode 签名不变）。
- 所有 ldm/text_encoders import 改为函数内 lazy import；模型构造入口在缺少模块时抛出明确错误（如 `ComfyDL 脱水模式：SD 生成功能未内置`）。
- 拆分完成后，`comfy.sd` 不再触发 ldm/text_encoders/k_diffusion 加载。

## 6. 删除清单（C 档）

> 回捞源一律为 `ComfyUI-original\` 对应路径。目录删除后文件数已计入，单文件删除逐条列出。

### 6.1 comfy/ 目录级（整目录删除）
| 目录 | 文件数 | 职责 | 回捞源 |
|---|---|---|---|
| `comfy/ldm/` | 180 py | 扩散模型实现（SD/FLUX/Wan/Hunyuan/MoGe/Supir 等） | `ComfyUI-original\comfy\ldm\` |
| `comfy/text_encoders/` | 51 py + 25 json + 1 model | 文本编码器（CLIP/T5/LLaMA/Gemma） | `ComfyUI-original\comfy\text_encoders\` |
| `comfy/k_diffusion/` | 5 py | k-diffusion 采样 | `ComfyUI-original\comfy\k_diffusion\` |
| `comfy/sd1_tokenizer/` | 若干 | SD1 tokenizer | `ComfyUI-original\comfy\sd1_tokenizer\` |
| `comfy/audio_encoders/` | 3 py | 音频编码（Wav2Vec2/Whisper） | `ComfyUI-original\comfy\audio_encoders\` |
| `comfy/image_encoders/` | 若干 | 图像编码（CLIP Vision 实现） | `ComfyUI-original\comfy\image_encoders\` |
| `comfy/extra_samplers/` | 1 py | uni_pc 采样器 | `ComfyUI-original\comfy\extra_samplers\` |
| `comfy/taesd/` | 若干 | TinyAE 预览解码 | `ComfyUI-original\comfy\taesd\` |
| `comfy/cldm/` | 若干 | ControlNet 实现 | `ComfyUI-original\comfy\cldm\` |
| `comfy/t2i_adapter/` | 若干 | T2I Adapter | `ComfyUI-original\comfy\t2i_adapter\` |
| `comfy/weight_adapter/` | 9 py | 权重适配（LoRA/ControlNet 共用） | `ComfyUI-original\comfy\weight_adapter\` |
| `comfy/background_removal/` | 若干 | 抠图模型 | `ComfyUI-original\comfy\background_removal\` |

### 6.2 comfy/ 单文件级
**A. 锚点改造后可删的生成单文件（任务 4 执行）：**
```
diffusers_load.py  diffusers_convert.py  samplers.py  sample.py  sampler_helpers.py
controlnet.py  clip_vision.py  clip_model.py  sd1_clip.py  sdxl_clip.py
lora.py  lora_convert.py  model_base.py  model_detection.py  model_sampling.py
latent_formats.py  conds.py  context_windows.py  supported_models.py  supported_models_base.py
pixel_space_convert.py  gligen.py  bg_removal_model.py  rmsnorm.py  (nested_tensor.py 待核查)
```
**B. 配置文件（随生成链删除）：**
```
clip_config_bigg.json  clip_vision_config_g/h/vitl_336/vitl_336_llava/siglip_384/siglip_512/
siglip2_base_naflex.json  sd1_clip_config.json  (sdxl_clip_config*.json 若存在)
```
**C. 静态确认零引用的 41 个模块（import_scan 已核对，任意阶段可删）：**
```
comfy.audio_encoders.audio_encoders / wav2vec2 / whisper
comfy.diffusers_convert
comfy.extra_samplers.uni_pc
comfy.gligen
comfy.k_diffusion.deis / sa_solver / sampling / utils
comfy.ldm.anima.lllite / colormap / depth_anything_3.preprocess /
  hunyuan_video.upsampler / hunyuan_video.vae_refiner / lightricks.duration_head /
  lightricks.latent_upsampler / lumina.controlnet / mmaudio.vae.activations /
  modules.temporal_ae / moge.geometry / moge.model / moge.modules / moge.panorama /
  sam3d_body.* (8) / seedvr.color_fix / supir / supir.supir_modules / supir.supir_patch /
  triposplat.preview / wan.model_multitalk / wan.uni3c
comfy.sdxl_clip
comfy.comfy_types.examples.example_nodes
```

### 6.3 comfy_extras/（159 py，任务 5 执行）
- **保留**：§4.3 名单（16 个）。
- **删除**：其余全部（约 120+），含 nodes_custom_sampler / nodes_controlnet / nodes_sd3 / nodes_flux / nodes_wan / nodes_train / nodes_audio_encoder 等生成链节点。删除清单以最终复核为准，逐文件登记后删除。
- 注：白名单（`nodes.py:2400-2534` extras_files）中未过审的文件需同步从白名单移除，或依赖 `load_custom_node` 的异常捕获自动跳过（warning 不崩）。**推荐同步移除白名单条目**，避免启动日志刷 warning。

### 6.4 comfy_api_nodes/（86 py，整目录删除）
云端服务商节点（BFL/Gemini/Kling/OpenAI/Wan/Veo/Qwen 等 40 家 API），与本地教学运行无关。同时评估 `nodes.py` 中 `init_builtin_api_nodes` 调用是否保留空逻辑。

### 6.5 其他（整目录删除）
```
blueprints/         (89 json 工作流蓝图 + glsl 子目录)
tests/              (22 py)
tests-unit/         (103 py)
script_examples/    (3)
custom_nodes/websocket_image_save.py   (演示节点，可选删)
```

## 7. 验证标准（每阶段 + 最终）

| # | 验证项 | 方法 |
|---|---|---|
| 1 | 启动成功 | `penv\Scripts\python main.py`，日志无 `ModuleNotFoundError` |
| 2 | 节点注册 | GET `/object_info` 或启动日志，ComfyDL 约 106 节点完整 |
| 3 | 基础工作流 | LoadImage → ComfyDL 节点 → PreviewImage 可执行 |
| 4 | 无残留 import 报错 | 启动日志无 `IMPORT FAILED`（或仅限已登记条目） |
| 5 | 启动耗时 | 对比脱水前后 `main.py` 启动耗时，应显著下降 |
| 6 | 回捞可逆 | `compare_dups.py` 对删除文件哈希，确保 original 完好 |

## 8. 阶段检查点（git commit）

```
[1] baseline  初始状态（3c394c3，已完成）         → git tag baseline
[2] split-sd  sd.py 拆分 + model_patcher/hooks/latent_preview 改造
[3] nodes     nodes.py 生成节点移除
[4] delete-1  comfy/ 生成目录与单文件删除
[5] delete-2  comfy_extras/comfy_api_nodes/blueprints/tests 删除
[6] verify    最终验证 + 文档更新
```
每阶段完成后 `git add -A && git commit`，异常时可 `git checkout` 回退上一检查点。
