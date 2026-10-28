# ComfyDL_UI

![Cover](assets/cover_v2_HD.png)

## 这是什么？

**由 AI 绘画软件硬核爆改而成的模块化深度学习工具箱。**

`ComfyDL_UI` 提供了一个节点式界面来搭建各种深度学习工作流——从回归运算、CNN、RLHF 到风格迁移、语言模型。

> 这**不是**完整的原版 ComfyUI，而是脱水（dehydrated）分支
> （见 [`docs/dehydrate_manifest.md`](docs/dehydrate_manifest.md)），只保留节点图引擎、
> 服务层与基础图像 IO，删去了Stable Diffusion栈。若想使用原版 ComfyUI，请走官方仓库：
> <https://github.com/comfyanonymous/ComfyUI>。

## 长什么样？

![ComfyDL_UI 中的表格回归工作流](assets/tabular_regression_workflow.png)

![Language Model Train 在节点下方实时铺开交叉熵曲线](assets/language_model_train_running.png)

> 更多示例与完整节点参考见 [`comfydl/`](comfydl/) 子模块说明（[English](comfydl/README.md)、[FUNCTIONS_zh.md](comfydl/FUNCTIONS_zh.md)）。

## 特性

`ComfyDL_UI` 的几大核心能力在下方各占一节——**可视化编辑器**、**深度学习算子包**、实时进度、**性能调优**，以及**翻车现场保存与恢复**。基础特性包括：

- 异步队列 + 部分计算图重执行（只重跑变化过的部分）。
- 智能显存/内存管理；无 GPU 后端时自动回退 CPU。
- 完全离线运行（除非你主动要求联网下载）。
- 兼容第三方自定义节点。
- 通过 [`extra_model_paths.yaml`](extra_model_paths.yaml.example) 配置额外模型目录。

### 可视化节点图编辑器

用连线代替写代码来搭建深度学习工作流：拖拽、连线、分组、静音、旁路、折叠；只重跑变化过
的部分；工作流以纯 JSON 保存与分享（把生成的 PNG 拖回页面即可还原产出它的工作流——含
种子）。

### 深度学习算子

你大概懂深度学习（不是吗？）——在这里，大部分内容都做成了开箱即用的节点。张量与形状操作、
linear / conv / norm / attention 层、损失函数、带梯度裁剪的优化器、LR 调度器、带早停的
训练循环、数据集读写（CSV / Excel / DB / JSON / TXT）、分词器与语言模型块、目标检测、
分割与可视化——133 个节点、21 个分类，连线代替手写代码。

目录一览：

![Network & Layers：75 个节点按门类分组](assets/network_layers_catalog.svg)

**交互式目录**——点击任意节点即可查看描述与源码——位于 [`docs/network-and-layers.html`](docs/network-and-layers.html)，可以通过 [htmlpreview](https://htmlpreview.github.io/?https://github.com/Cynthia-lxx/ComfyDL_UI/blob/master/docs/network-and-layers.html)
查看渲染效果，或在本地直接打开该文件。

### 实时进度与预览

每个长耗时节点都通过内置 WebSocket 通道逐步上报进度——语言模型节点边生成边在节点下方
显示文本，顶栏同步跟踪每次运行的开销。

### 性能调优（内存、算力与算子）

`ComfyDL_UI` 对一张图运行前后的全部开销做剖析。一个面板装下整个故事——三色内存判定、
设备预算、算力估算、逐节点拆解、实测历史与 ATen 算子图：

<p align="center">
  <img src="assets/profiling_operator_panel.png" alt="展开的 Profiling 面板：SAFE 内存判定、设备预算、18.90 TFLOPs 算力估算、逐节点拆解、实测历史与 CPU 突发日志" width="720" />
</p>

- **内存** — 基于参数与数据流的峰值估算，并按设备预算给出内存情况判定，入队前即警告。拆解深入到张量族——参数、梯度、注意力分数、FFN、logits——
  每一族都给出字节数与最大单张量，就贴在它分析的节点旁边：
  
  <p align="center">
    <img src="assets/profiling_memory_breakdown_table.png" alt="逐张量族的内存拆解表，与它所分析的 Language Model Train 节点合影" width="620" />
  </p>
  <p align="center">
    <img src="assets/profiling_oom_warning.png" alt="内存警告对话框：估算峰值 2.94 TB 对比剩余 2.16 GB，带取消与确认" width="620" />
  </p>

- **算力负载** — 按节点族（attention / FFN / logits / conv…）维护 FLOPs 账本，并统计逐
  节点实际耗时；运行时顶栏实时指示。
- **算子拆解** — 每个节点都被展开成它真正执行的 ATen 算子（`aten.mm`、`aten._softmax`、
  `aten.conv2d`…）；算子级账本与公式估算交叉验证，偏差会被标记而不是隐藏。
- **看门狗** — 采样到的 CPU / RSS 事实归属到正在执行的节点，抓住运行中独占机器的节点。

设计文档：[`docs/profiling-m1-memory-estimation.md`](docs/profiling-m1-memory-estimation.md)、
[`docs/profiling-m2-compute-and-watchdog.md`](docs/profiling-m2-compute-and-watchdog.md)、
[`docs/profiling-m3-estimator-families.md`](docs/profiling-m3-estimator-families.md)、
[`docs/profiling-m3-opgraph.md`](docs/profiling-m3-opgraph.md)。

### 翻车现场（Crash Site）保存与恢复

- **惰性捕获** — 工作流全速运行时不写任何东西；被打断或失败的那一刻，缓存中幸存的输出
  立即写入快照库。
- **一键恢复** — 重新入队存储的提示词并跳过所有已完成节点；跳过 / 重跑统计会在运行结束
  后出现在新快照卡片上。
- **现场导出** — 逐节点勾选要保留的内容，并按行选择格式（safetensors / png / json / db /
  pt），外加工作流提示词；文件落入 output 目录并进入 Results assets。
- **断点续训** — 训练节点（Regression Train、Training Loop、Language Model Train）在被打
  断的瞬间保存完整状态；恢复时从断点步继续，而不是把万步训练从头再来。
- **总开关** — *设置 > ComfyDL crash site enabled (master switch)* 可随时把整个功能从执行
  路径上摘下来：快照 Provider 被反注册，执行器生命周期钩子与缓存层不再调用它，训练 checkpoint
  在触碰张量之前就短路，恢复按钮给出可读的拒绝提示。停用后执行路径的行为与"没有装过这个功能"
  一致：面板显示醒目警示横幅、快照库不再加载（零请求）、顶栏快捷按钮变为警示样式，重新开启后
  自动恢复。`--cdl-crashsite-disable` 硬停用（重启才能恢复）；
  `--cdl-profiling-disable` 也会连带硬停用事故现场（历史遗留接线）。

侧栏中的快照库——每次运行都会连同捕获的节点一起归档；被打断的现场提供现场导出：

<p align="center">
  <img src="assets/crashsite_snapshot_library.png" alt="Crash Site 侧栏：附带捕获节点与现场导出入口的快照库" width="440" />
</p>
<p align="center">
  <img src="assets/crashsite_scene_export.png" alt="Crash Site 现场导出：逐节点勾选框、格式下拉与工作流提示词行" width="520" />
</p>

### 实时 loss 图表

每个训练器都在节点正下方的卡片上实时绘制 loss map——原始步损失与平滑曲线并列，随运行
推进刷新——收敛还是发散一眼可见，无需打开任何东西：

<p align="center">
  <img src="assets/loss_map_focus.png" alt="训练节点下方的实时 loss map 预览卡，同时展示原始与平滑损失曲线" width="520" />
</p>

### 界面动效——分阶段入场与点击涟漪

移植自 homepage 分叉：顶栏、底栏、侧栏与右栏按编排依次滑入，画布节点沿对角线依次
淡入、网格最后回归，每次左键点击在页全局覆盖层上泛起一圈涟漪。一个设置总控——
*设置 > ComfyDL > 动效 > 界面动效*——另有 `prefers-reduced-motion` 硬豁免与 9 秒
保险丝（模块加载失败也不会让界面卡在隐形状态）。画布节点只用 opacity 做动画：
节点位置由 `transform` 决定，动效模块绝不触碰。

## 安装

需要 Python 3.12+（推荐 3.13）。任何有受支持 PyTorch 构建的操作系统均可。

先创建虚拟环境以隔离依赖——避免与机器上其他包冲突：

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

打开它打印的地址。ComfyDL 节点已注册为内置——无需往 `custom_nodes` 放任何东西。

### GPU 加速

纯 CPU 机器开箱即用。如需 GPU 加速，请为同一环境安装与硬件匹配的 PyTorch 构建：

- **NVIDIA**：`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130`
- **AMD（Linux, ROCm）**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2`
- **Intel Arc**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`
- **Apple Silicon**：参考 [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) 指南。

若出现 "Torch not compiled with CUDA enabled"，`pip uninstall torch` 后按上述对应命令重装。

## 运行

```bash
python main.py
```

常用参数（全部见 `python main.py --help`）：

- `--cpu` — 强制 CPU 执行。
- `--port 8188` — 更换监听端口。
- `--tls-keyfile key.pem --tls-certfile cert.pem` — 以 HTTPS 代替 HTTP 提供服务。

未被 ROCm 官方覆盖的 AMD 卡可尝试 `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`（RDNA2 及更早）或 `HSA_OVERRIDE_GFX_VERSION=11.0.0`（RDNA3）。

### 说明

- 只有输出所需输入齐全的节点才会被执行；重复提交未变化的工作流只会重跑变化过的部分。
- 把生成的 PNG 拖到页面上，即可加载产出它的工作流（含种子）。

## 键盘快捷键

| 按键 | 说明 |
|------------------------------------|----------------------------------------------------|
| `Ctrl` + `Enter` | 将当前工作流加入计算队列 |
| `Ctrl` + `Shift` + `Enter` | 将当前工作流作为队首加入计算队列 |
| `Ctrl` + `Alt` + `Enter` | 取消当前计算 |
| `Ctrl` + `Z` / `Ctrl` + `Y` | 撤销 / 重做 |
| `Ctrl` + `S` | 保存工作流 |
| `Ctrl` + `O` | 加载工作流 |
| `Ctrl` + `A` | 全选节点 |
| `Alt` + `C` | 折叠 / 展开选中节点 |
| `Ctrl` + `M` | 静音 / 取消静音选中节点 |
| `Ctrl` + `B` | 绕过选中节点 |
| `Delete` / `Backspace` | 删除选中节点 |
| `Ctrl` + `Backspace` | 删除当前工作流 |
| `Space` | 按住并移动鼠标以平移画布 |
| `Ctrl` / `Shift` + `Click` | 将点击的节点加入选中 |
| `Ctrl` + `C` / `Ctrl` + `V` | 复制粘贴选中节点（不保持与未选中节点输出的连接） |
| `Ctrl` + `C` / `Ctrl` + `Shift` + `V` | 复制粘贴选中节点（保持选中节点输出到粘贴节点输入的连接） |
| `Shift` + `Drag` | 同时移动多个选中节点 |
| `Ctrl` + `D` | 加载默认工作流 |
| `Alt` + `+` | 画布放大 |
| `Alt` + `-` | 画布缩小 |
| `Ctrl` + `Shift` + 鼠标左键 + 纵向拖动 | 画布缩放 |
| `P` | 钉住 / 取消钉住选中节点 |
| `Ctrl` + `G` | 为选中节点创建分组 |
| `Q` | 显示 / 隐藏队列 |
| `H` | 显示 / 隐藏历史 |
| `R` | 刷新图 |
| `F` | 显示 / 隐藏菜单 |
| `.` | 视图适配选中项（未选中时适配整个工作流） |
| 双击鼠标左键 | 打开节点快速搜索面板 |
| `Shift` + `Drag` | 同时拖动多条连线 |
| `Ctrl` + `Alt` + 鼠标左键 | 断开点击槽位上的所有连线 |

`macOS` 用户可将 `Ctrl` 替换为 `Cmd`。

## 文档

- [`comfydl/`](comfydl/) — 节点包：`README.md` / `README_zh.md` 与完整节点参考 `FUNCTIONS.md` / `FUNCTIONS_zh.md`。
- [`docs/`](docs/) — 分支专属说明：脱水清单、品牌、内置节点包、Profiling 与分步改造设计
  文档。
- [`AGENTS.md`](AGENTS.md) — 本仓库的工作方式。
- [English](README.md)

## 版本与致谢

- **版本** — `v0.3.1`，自 ComfyUI `v0.34.0` 脱水而来。
- **维护者** — [Cynthia-lxx](https://github.com/Cynthia-lxx)。
- **许可** — GPL-3.0，适用于本仓库与 `comfydl/` 子模块
  （[`LICENSE`](LICENSE)、[`comfydl/LICENSE`](comfydl/LICENSE)）。

本项目派生自 comfyanonymous 及贡献者的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。上游 ComfyUI 保留其版权与维护者；两份声明均列于 [`LICENSE`](LICENSE) 末尾。上游项目的网站、文档与社区位于 <https://www.comfy.org/>，独立于 ComfyDL_UI 维护。

<p align="center">
  <img src="assets/dev_2333.png" alt="橘雪莉牛逼" />
</p>
