<div align="center">

# ComfyDL_UI

**Deep Learning is just a few clicks away!**

[ComfyDL](https://github.com/Cynthia-lxx/ComfyDL) 的图形界面版本 —— 一个为深度学习特化改装的
ComfyUI 分支。

[English](README.md)

</div>

## 这是什么

`ComfyDL_UI` 让你**用连线代替写代码**来搭建深度学习工作流——从 CNN 到语言模型。它托管
[ComfyDL](https://github.com/Cynthia-lxx/ComfyDL) 节点包并以内置节点形式注册（张量、网络层、
可学习参数、优化器、训练循环、NLP 流水线、数据工具与可视化），运行在精简后的 ComfyUI 运行时之上。

它**不是完整的原版 ComfyUI**：本仓库是脱水（dehydrated）分支，只保留节点图引擎、服务层与基础
图像 IO，删去了扩散模型生成栈；脱水清单见 [`docs/dehydrate_manifest.md`](docs/dehydrate_manifest.md)。
若想使用原版 ComfyUI，请走官方仓库下载：<https://github.com/comfyanonymous/ComfyUI>。

## 效果预览

在图内端到端训练一个小语言模型——Vocab Build → Text Encode → Sliding Window → Language Model
流水线 → Generate → Save。`Language Model Train` 节点在训练的同时会把 cross-entropy 曲线
**实时刷在自己节点下方**（节点下方那张预览卡片）：

![ComfyDL_UI 中的语言模型训练工作流](assets/languange_model_train_workflow.png)

<p align="center">
  <img src="assets/language_model_train_node_focus.png" alt="Language Model Train 节点与其实时 loss 曲线预览" width="440" />
</p>

> 更多示例与完整节点参考见 [`comfydl/`](comfydl/) 子模块说明
> （[English](comfydl/README.md)、[FUNCTIONS_zh.md](comfydl/FUNCTIONS_zh.md)）。

## 特性

- 可视化节点图，无需写代码即可搭建、复用深度学习工作流。
- 完整内置 [ComfyDL](https://github.com/Cynthia-lxx/ComfyDL) 节点包：张量运算、网络层、可学习
  参数、优化器、带实时 loss 曲线预览的训练循环、语言模型训练与生成、数据集与可视化节点。
- 高效本地执行：异步队列 + 部分重执行——只有变化过的子图会重新运行。
- 智能显存/内存管理；无任何 GPU 后端时自动回退 CPU。
- 运行中节点下方的实时进度与预览卡片（loss 曲线、生成文本快照），经内置 WebSocket 通道推送。
- 内置内存分析：峰值估算、三色判定、入队前的超限警告，以及分配失败后的事后归因。
- 工作流以 JSON 保存与加载。
- 完全离线运行：除非你主动要求，核心不会下载任何东西。
- 仍可通过 `custom_nodes/` 兼容第三方自定义节点包。
- 可通过 [`extra_model_paths.yaml`](extra_model_paths.yaml.example) 配置额外的模型目录。

### 内存分析

任何东西入队之前，ComfyDL_UI 会根据当前图的参数与数据流估算峰值内存，与设备预算比对并给出
绿 / 黄 / 红三色判定，放不下的图会在入队前先警告。

![Profiling 侧栏：设备预算、CERTAIN OOM 判定与按节点分解](assets/profiling_memory_verdict.png)

- **边改边估**：500ms 防抖后重新估算；侧栏给出设备预算、估算峰值、最大单张量，以及按节点分解（参数/激活/优化器状态）。
- **三色判定**：绿=放得下；黄=吃紧（只弹 toast）；红=一定 OOM，入队前先警告。
- **事后分析**：分配失败会把字节数反查回对应张量，并给出可行的取值建议。
- **对未知诚实**：未覆盖的节点类型标为 `unknown`。
- **双语**：面板、徽章、确认框与事后分析卡片均跟随界面语言。

红色图按 Run 会先弹警告，并允许覆盖：

<p align="center">
  <img src="assets/profiling_memory_warning.png" alt="对判定为一定 OOM 的图入队时弹出的内存警告框" width="620" />
</p>

公式、阈值与验收清单见 [`docs/profiling-m1-memory-estimation.md`](docs/profiling-m1-memory-estimation.md)。

## 安装

需要 Python 3.12+（推荐 3.13），任何有可用 PyTorch 版本的系统均可运行。

建议先创建虚拟环境以隔离依赖——避免与你机器上其他位置安装的包发生冲突：

```bash
git clone https://github.com/Cynthia-lxx/ComfyDL_UI
cd ComfyDL_UI
python -m venv .venv

.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux / macOS
```

然后安装依赖并启动：

```bash
pip install -r requirements.txt
python main.py
```

打开提示的地址即可：ComfyDL 节点已作为内置节点注册，无需再放进 `custom_nodes`。

### GPU 加速

纯 CPU 机器开箱即用。若要 GPU 加速，请在同一环境中安装匹配硬件的 PyTorch 版本：

- **NVIDIA**：`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130`
- **AMD（Linux，ROCm）**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2`
- **Intel Arc**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`
- **Apple 芯片**：参见 Apple 官方指南 [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)。

若报 "Torch not compiled with CUDA enabled"，先 `pip uninstall torch`，再按上面的对应命令重装。

## 运行

```bash
python main.py
```

常用参数（完整列表见 `python main.py --help`）：

- `--cpu` —— 强制 CPU 执行。
- `--port 8188` —— 修改监听端口。
- `--tls-keyfile key.pem --tls-certfile cert.pem` —— 以 HTTPS 而非 HTTP 提供服务。

AMD 显卡未被 ROCm 正式支持时，可尝试 `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
（RDNA2 及更早）或 `HSA_OVERRIDE_GFX_VERSION=11.0.0`（RDNA3）。

### 说明

- 只有输出所需输入齐全的子图才会执行；重复提交相同工作流时，只有变化过的部分及其下游会重跑。
- 把生成的 PNG 拖回页面，即可载入生成它的工作流（含随机种子）。

## 键盘快捷键

| 按键组合                              | 功能                                                                                                              |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | 将当前图加入生成队列                                                                                                |
| `Ctrl` + `Shift` + `Enter`              | 将当前图插队为第一个执行                                                                                             |
| `Ctrl` + `Alt` + `Enter`                | 取消当前生成                                                                                                       |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | 撤销/重做                                                                                                          |
| `Ctrl` + `S`                          | 保存工作流                                                                                                         |
| `Ctrl` + `O`                          | 加载工作流                                                                                                         |
| `Ctrl` + `A`                          | 全选节点                                                                                                           |
| `Alt `+ `C`                           | 折叠/展开所选节点                                                                                                   |
| `Ctrl` + `M`                          | 静音/取消静音所选节点                                                                                               |
| `Ctrl` + `B`                           | 旁路所选节点（等效于移除该节点并把连线直接接通）                                                                      |
| `Delete`/`Backspace`                   | 删除所选节点                                                                                                       |
| `Ctrl` + `Backspace`                   | 删除当前图                                                                                                         |
| `Space`                              | 按住并移动光标以平移画布                                                                                             |
| `Ctrl`/`Shift` + `Click`                 | 将点击的节点加入选区                                                                                                 |
| `Ctrl` + `C`/`Ctrl` + `V`                  | 复制粘贴所选节点（不保留与未选节点输出端口的连线）                                                                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | 复制粘贴所选节点（保留未选节点输出端口到粘贴节点输入端口的连线）                                                        |
| `Shift` + `Drag`                       | 同时移动多个所选节点                                                                                                 |
| `Ctrl` + `D`                           | 载入默认图                                                                                                         |
| `Alt` + `+`                          | 画布放大                                                                                                           |
| `Alt` + `-`                          | 画布缩小                                                                                                           |
| `Ctrl` + `Shift` + LMB + 垂直拖动 | 画布缩放                                                                                                           |
| `P`                                  | 固定/取消固定所选节点                                                                                                |
| `Ctrl` + `G`                           | 将所选节点编组                                                                                                      |
| `Q`                                 | 切换队列面板可见性                                                                                                   |
| `H`                                  | 切换历史面板可见性                                                                                                   |
| `R`                                  | 刷新图                                                                                                             |
| `F`                                  | 显示/隐藏菜单                                                                                                       |
| `.`                                  | 视图适配所选内容（未选中时适配整图）                                                                                    |
| 双击左键                            | 打开节点快速搜索面板                                                                                                 |
| `Shift` + `Drag`                       | 同时移动多根连线                                                                                                    |
| `Ctrl` + `Alt` + LMB                   | 断开点击端口的所有连线                                                                                               |

macOS 用户可用 `Cmd` 代替 `Ctrl`。

## 文档索引

- [`comfydl/`](comfydl/) 子模块：节点包说明 `README.md` / `README_zh.md` 与完整节点参考
  `FUNCTIONS.md` / `FUNCTIONS_zh.md`。
- [`docs/`](docs/)：脱水清单、品牌化与前端补丁、ComfyDL 内置化、内存分析及各阶段 reform 设计文档。
- [`AGENTS.md`](AGENTS.md)：本仓库的工作约定。
- [English README](README.md)

## 版本与致谢

- 版本：`v0.3.1`（脱水基线 ComfyUI `v0.34.0`）。
- 维护者：[Cynthia-lxx](https://github.com/Cynthia-lxx)。
- 版权：本项目与 `comfydl/` 子模块均以 GPL-3.0 发布，声明见 [`LICENSE`](LICENSE) 与
  [`comfydl/LICENSE`](comfydl/LICENSE)。

本项目派生自 comfyanonymous 及贡献者的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。
上游 ComfyUI 保留其自身版权与维护者，相关声明见 [`LICENSE`](LICENSE) 末尾；上游的官网、文档与
社区位于 <https://www.comfy.org/>，与 ComfyDL_UI 分别维护。

界面默认设置为**英文**与**开启 Nodes 2.0**；两者都可在「设置」面板中随时改回。设置面板底部的
「关于」会说明本项目与上游原版 ComfyUI 的关系。
