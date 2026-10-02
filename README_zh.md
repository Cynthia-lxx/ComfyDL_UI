<div align="center">

# ComfyDL_UI

**ComfyDL 的图形界面版本 —— 一个为深度学习特化改装的 ComfyUI 分支。**

</div>

## 这是什么

`ComfyDL_UI` 是 [ComfyDL](https://github.com/Cynthia-lxx/ComfyDL) 的图形界面（GUI）运行时：
它托管 ComfyDL 节点包，并运行该节点包在 ComfyUI 运行时之上新增的张量、网络层、训练节点与数据工具。

## 效果预览

在图内端到端训练一个小语言模型——Vocab Build → Text Encode → Sliding Window → Language Model
流水线 → Generate → Save。`Language Model Train` 节点在训练的同时会把 cross-entropy 曲线
**实时刷在自己节点下方**（节点下方那张预览卡片）：

![ComfyDL_UI 中的语言模型训练工作流](comfydl/assets/languange_model_train_workflow.png)

<p align="center">
  <img src="comfydl/assets/language_model_train_node_focus.png" alt="Language Model Train 节点与其实时 loss 曲线预览" width="440" />
</p>

> 更多示例与完整节点参考见 [`comfydl/`](comfydl/) 子模块说明
> （[English](comfydl/README.md)、[FUNCTIONS_zh.md](comfydl/FUNCTIONS_zh.md)）。

## 它不是什么

- **它不是完整的原版 ComfyUI**：本仓库是脱水（dehydrated）分支，只保留 ComfyDL 节点所需的
  运行时，脱水清单见 [`docs/dehydrate_manifest.md`](docs/dehydrate_manifest.md)。
- 若想使用**原版 ComfyUI**，请走官方仓库下载：<https://github.com/comfyanonymous/ComfyUI>。
  官方的安装器、安装包与云服务均由上游项目维护，与本项目无关，相关入口统一列在下面的
  「上游原版 ComfyUI」一节。

## 版本与维护

- 版本：`v0.3.1`（脱水基线 ComfyUI `v0.34.0`）。
- 维护者：[Cynthia-lxx](https://github.com/Cynthia-lxx)。
- 版权：本项目与 `comfydl/` 子模块均以 GPL-3.0 发布，声明见 [`LICENSE`](LICENSE) 与
  [`comfydl/LICENSE`](comfydl/LICENSE)；上游 ComfyUI 保留其自身版权与维护者。

## 快速上手

1. 使用仓库自带的虚拟环境 `penv\`（Embedded Python 3.14，依赖已配置）。
2. 启动：`penv\Scripts\python.exe main.py`（可选 `--cpu`、`--port` 等参数，见 `main.py --help`）。
3. 打开提示的地址即可：ComfyDL 节点包已作为内置节点注册，无需再放进 `custom_nodes`。

界面默认设置为**英文**与**开启 Nodes 2.0**；两者都可在「设置」面板中随时改回。设置面板底部的
「关于」会说明本项目与上游原版 ComfyUI 的关系。

## 文档索引

- [`README.md`](README.md)：英文自述；其下半部分是**上游原版 ComfyUI 文档**（安装、运行、排错），
  按原样保留并已明确标注归属。
- [`docs/`](docs/)：脱水清单、品牌化与前端补丁、ComfyDL 内置化等记录。
- [`AGENTS.md`](AGENTS.md)：本仓库的工作约定。
- `comfydl/` 子模块：节点包文档 `README.md`、`README_zh.md`、`FUNCTIONS.md`、`FUNCTIONS_zh.md`。

## 上游原版 ComfyUI

本项目派生自上游 ComfyUI，但**不是**完整的原版包。原版项目的徽章、官网、社区与下载入口如下，
它们与原版项目本身一样，都不属于 ComfyDL_UI 的维护范围：

- 官方仓库：<https://github.com/comfyanonymous/ComfyUI>
- 官网与下载：<https://www.comfy.org/>
- 官方文档：<https://docs.comfy.org/>
- Discord 社区：<https://discord.com/invite/comfyorg>

上游的安装说明请见 [`README.md`](README.md) 的「Upstream documentation」一节。
