<div align="center">

# ComfyDL_UI

**Deep Learning is just a few clicks away!**

</div>

![Cover](assets/cover_v2_HD.png)

## What's This?

**Hardcore-overhauled by AI art software, this is a modular Deep Learning Toolbox GUI, API
and backend with a graph/nodes interface.**

`ComfyDL_UI` lets you build deep learning workflows — from CNNs to language models — by
connecting nodes in a graph, not by writing code. It hosts the
[ComfyDL](https://github.com/Cynthia-lxx/ComfyDL) node pack as a built-in set of nodes
(tensors, layers, optimizers, training loops, NLP pipelines, data utilities and
visualizations) on top of a trimmed ComfyUI runtime.

> It is **not** the complete original ComfyUI: this repository is a dehydrated fork
> (see [`docs/dehydrate_manifest.md`](docs/dehydrate_manifest.md)) that keeps the node-graph
> engine, server and basic image IO, and drops the diffusion-generation stack. If you want the
> original ComfyUI, get it from <https://github.com/comfyanonymous/ComfyUI>.

## What it looks like

A production-grade tabular regression workflow looks like:

![Tabular regression workflow in ComfyDL_UI](assets/tabular_regression_workflow.png)

Language models tell the same story: 

![Language Model Train streaming its cross-entropy curve under the node](assets/language_model_train_running.png)

> More examples and the complete node reference live in the [`comfydl/`](comfydl/) submodule
> README ([中文](comfydl/README_zh.md), [FUNCTIONS.md](comfydl/FUNCTIONS.md)).

## Features

`ComfyDL_UI`'s headline capabilities each get their own section below — the visual editor, the
built-in node pack, live progress, full profiling, and the Crash Site save & resume net. The
everyday basics:

- Efficient local execution with asynchronous queueing and partial graph re-execution (only
  the parts of the graph that changed are re-run).
- Smart VRAM/RAM management with automatic CPU fallback when no GPU backend is available.
- Runs fully offline, downloads nothing unless you ask it to.
- Compatible with third-party custom nodes.
- Configure additional model locations with [`extra_model_paths.yaml`](extra_model_paths.yaml.example).

### Visual node-graph editor

Build deep-learning workflows by connecting nodes instead of writing code: drag, wire, group,
mute, bypass and collapse; re-run only the parts that changed; save and share the graph as
plain JSON (a generated PNG reloads the workflow — seeds included — that produced it).

### Most deep-learning operators, built in

You already know deep learning (don't you?) — here, most of it is a node away. Tensors and shape ops,
linear / conv / norm / attention layers, losses, optimizers with gradient clipping, LR
schedulers, training loops with early stopping, dataset readers and writers (CSV / Excel /
DB / JSON / TXT), tokenizers and language-model blocks, object detection, segmentation and
visualization — 133 nodes across 21 categories, wired instead of written.

The catalog at a glance:

![Network & Layers: all 75 nodes grouped by category](assets/network_layers_catalog.svg)

An **interactive version** — click any node for its description and source — ships at [`docs/network-and-layers.html`](docs/network-and-layers.html). View it rendered via [htmlpreview](https://htmlpreview.github.io/?https://github.com/Cynthia-lxx/ComfyDL_UI/blob/master/docs/network-and-layers.html) or open the file locally.

### Live progress & previews

Every long-running node reports per-step progress over the built-in WebSocket channel —
generated text appears under the language-model nodes as it is produced, and the top bar
tracks each run's cost while it executes.

### Profiling — memory, compute & operators

ComfyDL_UI profiles everything a graph costs, before and during the run. One panel holds the
whole story — the traffic-light memory verdict, the device budget, the compute estimate, the
per-node breakdown, the measured history and the ATen operator graph:

<p align="center">
  <img src="assets/profiling_operator_panel.png" alt="Expanded profiling panel: SAFE memory verdict, device budget, 18.90 TFLOPs compute estimate, per-node breakdown, measured history and CPU burst log" width="720" />
</p>

- **Memory** — peak estimation from parameters and data flow, graded green / amber / red
  against the device budget; red is **certain OOM** and warns before queueing. The breakdown
  goes down to tensor families — parameters, gradients, attention scores, FFN, logits — each
  with its byte count and largest single tensor, right next to the node it analyzes:

  <p align="center">
    <img src="assets/profiling_memory_breakdown_table.png" alt="Per-tensor-family memory breakdown table beside the Language Model Train node it analyzes" width="620" />
  </p>

  Pressing Run on a red graph warns with the numbers behind the verdict — and can be
  overridden:

  <p align="center">
    <img src="assets/profiling_oom_warning.png" alt="Memory warning dialog: estimated peak 2.94 TB vs 2.16 GB free, with Cancel and Confirm" width="620" />
  </p>

- **Compute load** — a FLOPs ledger per node family (attention / FFN / logits / conv …) and
  per-node wall time, with a live top-bar indicator while the run executes.
- **Operator breakdown** — each node is expanded into the ATen operators it actually executes
  (`aten.mm`, `aten._softmax`, `aten.conv2d`, …); the op-level ledger cross-checks the formula
  estimates, and mismatches are flagged, not hidden.
- **Watchdog** — sampled CPU / RSS facts attributed to the executing node catch the nodes that
  monopolize the machine mid-run.
- **Honest about unknowns** — uncovered node types report `unknown`; failed probes fall back
  to formula values with an explicit `fallback` grade. Panels, badges and dialogs follow the
  UI language.

Design notes: [`docs/profiling-m1-memory-estimation.md`](docs/profiling-m1-memory-estimation.md),
[`docs/profiling-m2-compute-and-watchdog.md`](docs/profiling-m2-compute-and-watchdog.md),
[`docs/profiling-m3-estimator-families.md`](docs/profiling-m3-estimator-families.md),
[`docs/profiling-m3-opgraph.md`](docs/profiling-m3-opgraph.md).

### Crash Site — save & resume

A safety net for interrupted runs, with **zero overhead while everything works**:

- **Lazy capture** — nothing is written while the workflow runs at full speed; the moment it
  is interrupted or fails, surviving outputs are written to the snapshot library.
- **One-click resume** — re-queues the stored prompt and skips every finished node; the
  skip / re-run statistics appear on the new snapshot card when the run ends.
- **Scene export** — pick exactly what to keep, per node, with a format per row
  (safetensors / png / json / db / pt) plus the workflow prompt; files land in the output
  folder and surface in Results assets.
- **Breakpoint continuation** — the trainers (Regression Train, Training Loop, Language Model
  Train) save their complete state at the interrupt moment; a resume continues from the
  breakpoint step instead of starting a 10k-step training over.
- **Master switch** — *Settings > ComfyDL crash site enabled (master switch)* detaches the
  whole feature from the run path on demand: the snapshot provider is unregistered, so the
  executor's lifecycle hook and the caching layer stop calling into it entirely, training
  checkpoints short-circuit before touching the tensors, and Resume refuses with a readable
  error. Disabled, the run path behaves as if the feature were absent: the panel shows a
  warning banner, the snapshot library is not loaded (zero requests) and the topbar shortcut
  is restyled as a warning until re-enabled. `--cdl-crashsite-disable` hard-disables until restart;
  `--cdl-profiling-disable` also hard-disables the crash site (historical wiring).

The snapshot library in the sidebar — every run is filed with its captured nodes; interrupted
scenes offer Scene export:

<p align="center">
  <img src="assets/crashsite_snapshot_library.png" alt="Crash Site sidebar: snapshot library with captured nodes and Scene export entries" width="440" />
</p>

The scene-export table — choose what to keep and in which format:

<p align="center">
  <img src="assets/crashsite_scene_export.png" alt="Crash Site scene export: per-node checkboxes with format dropdowns and the workflow prompt row" width="520" />
</p>

### Live loss map

Every trainer draws its loss map live on a card right under the node — the raw step loss and
the smoothed curve side by side, refreshed as the run progresses — so convergence or
divergence is visible at a glance, without opening anything:

<p align="center">
  <img src="assets/loss_map_focus.png" alt="Live loss map preview card under a training node, showing the raw and smoothed loss curves" width="520" />
</p>

### CDL App Panel — the App mode, rebuilt for deep learning

The official beta App mode (workflow → app) is designed for image outputs; DL
runs produce tensors, models, text and files. ComfyDL_UI rebuilds the whole
App screen as one native CDLApp UI with planned zones — no dead empty space,
no vertical cramming:

- **Zoned layout** — the official params column (real Vue widgets + Run
  button) docks as a narrow left column; the CDL dashboard takes the rest:
  previews as the hero tile, the outputs list as a tall column, progress and
  errors filling the remaining quadrants — every tile scrolls inside its own
  box. The zone boundaries are draggable (sizes persist), so nothing is
  covered and nothing is wasted.
- **Outputs first** — files from the output directory with one-click downloads
  (CSV / JSON / txt / `.pt` / PNG...), text outputs as copyable entries, and
  low-cost previews from the existing renderers. Outputs carry equal weight
  with inputs.
- **Inline previews** — image outputs render as a clickable thumbnail grid in
  the hero tile (official `/view` channel, newest first); one click opens the
  full image.
- **Detailed progress** — a per-node progress list (title / percent /
  running-done-cached-error) plus a total bar, driven purely by the official
  WebSocket events.
- **Errors that make sense** — failed nodes render as cards with the node
  title, the exception message and the full collapsible traceback, one click
  to copy.
- **Navigation** — jump back to the node graph, or straight into the official
  builder at the Inputs/Outputs step, from the panel header.

The dock mounts only while App mode is active and releases cleanly on exit;
one setting (*Settings > ComfyDL > App*) governs it. Design notes and the
official-evolution containment strategy:
[`docs/cdlapp-mode.md`](docs/cdlapp-mode.md). Try it with **Templates →
ComfyDL Examples → Formula Playground: App Mode**.

### UI motion — branded splash, staged entrance & click ripple

Ported from the homepage fork: the official wave-SVG splash is replaced by the
branded boot screen (animated logo GIF, progress bar with moving stripes and a
live boot log) fed with *real* signals — PerformanceObserver for chunk loads,
a read-only fetch wrapper for `/api` milestones, and polling of the workspace
store's spinner for the boot-sequence falling edge. On that edge motion.js
takes over the splash: it fades out (no hard DOM removal) and the entrance
starts the same instant, so chrome elements (topbar, bottom bar, sidebar rail,
right panel) slide in *through* the dissolving splash; canvas nodes fade in
along a diagonal sweep and the canvas grid returns last. Every left click
spawns a soft ripple on a page-global overlay. The user's `DL_trans.png` also
replaces the favicon and the top-left menu logo. One setting governs it all —
*Settings > ComfyDL > Motion > 界面动效* — plus a hard `prefers-reduced-motion`
exemption, a 9s pre-hide failsafe and a 20s splash failsafe. Canvas nodes are
animated with opacity only: their position lives in `transform`, which the
motion module never touches.

### Simplified Chinese UI (i18n)

Switching the ComfyUI language setting to 简体中文 localizes everything ComfyDL_UI adds —
sidebar tabs, the Profiling and Crash Site panels, node titles (both the built-in comfydl
nodes and the core Network & Layers nodes such as RMSNorm), the ComfyDL settings entries
and their panel groups, and the Help menu's ComfyDL link group (docs / QQ group / Bilibili /
repo, appended as a labelled group after the official entries).

This works by *baking* rather than serving: the 0.3.x frontend never calls the `/i18n`
endpoint for custom nodes, so at startup `app/frontend_patch.py` merges
`locales/zh/nodeDefs.json` (regenerate with `cdl_smoke_tests/gen_locales.py`) plus the
settings translations directly into the frontend's own lazy-loaded zh chunks, located by
content anchors — robust across frontend builds. Verify with
`cdl_smoke_tests/test_frontend_i18n.py` (23 checks: chunk resolution, merge placement,
idempotency, help-menu grouping).

## Installation

Requires Python 3.12+ (3.13 recommended). Any OS with a supported PyTorch build works.

First create a virtual environment to keep the dependencies isolated — this avoids conflicts
with packages installed elsewhere on your machine:

```bash
git clone https://github.com/Cynthia-lxx/ComfyDL_UI
cd ComfyDL_UI
python -m venv .venv

.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux / macOS
```

Then install the dependencies and launch:

```bash
pip install -r requirements.txt
python main.py
```

Open the address it prints. The ComfyDL nodes are already registered as built-ins — nothing
goes into `custom_nodes`.

### GPU acceleration

A CPU-only machine works out of the box. For GPU acceleration, install the PyTorch build that
matches your hardware into the same environment:

- **NVIDIA**: `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130`
- **AMD (Linux, ROCm)**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2`
- **Intel Arc**: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu`
- **Apple Silicon**: follow the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) guide.

If you get "Torch not compiled with CUDA enabled", `pip uninstall torch` and reinstall with the
matching command above.

## Running

```bash
python main.py
```

Useful flags (see `python main.py --help` for all of them):

- `--cpu` — force CPU execution.
- `--port 8188` — change the listening port.
- `--tls-keyfile key.pem --tls-certfile cert.pem` — serve over HTTPS instead of HTTP.

On AMD cards not officially covered by ROCm, try `HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py`
(RDNA2 and older) or `HSA_OVERRIDE_GFX_VERSION=11.0.0` (RDNA3).

### Notes

- Only parts of the graph whose outputs have all their inputs are executed, and re-submitting
  an unchanged graph re-runs only the parts that changed.
- Dragging a generated PNG onto the page loads the workflow (including seeds) that produced it.

## Keyboard shortcuts

| Keybind                            | Explanation                                                                                                        |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `Ctrl` + `Enter`                      | Queue up current graph for generation                                                                              |
| `Ctrl` + `Shift` + `Enter`              | Queue up current graph as first for generation                                                                     |
| `Ctrl` + `Alt` + `Enter`                | Cancel current generation                                                                                          |
| `Ctrl` + `Z`/`Ctrl` + `Y`                 | Undo/Redo                                                                                                          |
| `Ctrl` + `S`                          | Save workflow                                                                                                      |
| `Ctrl` + `O`                          | Load workflow                                                                                                      |
| `Ctrl` + `A`                          | Select all nodes                                                                                                   |
| `Alt `+ `C`                           | Collapse/uncollapse selected nodes                                                                                 |
| `Ctrl` + `M`                          | Mute/unmute selected nodes                                                                                         |
| `Ctrl` + `B`                           | Bypass selected nodes (acts like the node was removed from the graph and the wires reconnected through)            |
| `Delete`/`Backspace`                   | Delete selected nodes                                                                                              |
| `Ctrl` + `Backspace`                   | Delete the current graph                                                                                           |
| `Space`                              | Move the canvas around when held and moving the cursor                                                             |
| `Ctrl`/`Shift` + `Click`                 | Add clicked node to selection                                                                                      |
| `Ctrl` + `C`/`Ctrl` + `V`                  | Copy and paste selected nodes (without maintaining connections to outputs of unselected nodes)                     |
| `Ctrl` + `C`/`Ctrl` + `Shift` + `V`          | Copy and paste selected nodes (maintaining connections from outputs of selected nodes to inputs of pasted nodes)   |
| `Shift` + `Drag`                       | Move multiple selected nodes at the same time                                                                      |
| `Ctrl` + `D`                           | Load default graph                                                                                                 |
| `Alt` + `+`                          | Canvas Zoom in                                                                                                     |
| `Alt` + `-`                          | Canvas Zoom out                                                                                                    |
| `Ctrl` + `Shift` + LMB + Vertical drag | Canvas Zoom in/out                                                                                                 |
| `P`                                  | Pin/Unpin selected nodes                                                                                           |
| `Ctrl` + `G`                           | Group selected nodes                                                                                               |
| `Q`                                 | Toggle visibility of the queue                                                                                     |
| `H`                                  | Toggle visibility of history                                                                                       |
| `R`                                  | Refresh graph                                                                                                      |
| `F`                                  | Show/Hide menu                                                                                                      |
| `.`                                  | Fit view to selection (Whole graph when nothing is selected)                                                        |
| Double-Click LMB                   | Open node quick search palette                                                                                     |
| `Shift` + Drag                       | Move multiple wires at once                                                                                        |
| `Ctrl` + `Alt` + LMB                   | Disconnect all wires from clicked slot                                                                             |

`Ctrl` can also be replaced with `Cmd` instead for macOS users.

## Documentation

- [`comfydl/`](comfydl/) — the node pack: `README.md` / `README_zh.md` and the complete node
  references `FUNCTIONS.md` / `FUNCTIONS_zh.md`.
- [`docs/`](docs/) — fork-specific notes: dehydration manifest, branding, built-in node pack,
  memory profiling, and the step-by-step reform design documents.
- [`AGENTS.md`](AGENTS.md) — how this repository is worked on.
- [中文说明](README_zh.md)

## Version and credits

- **Version** — `v0.3.1`, dehydrated from ComfyUI `v0.34.0`.
- **Maintainer** — [Cynthia-lxx](https://github.com/Cynthia-lxx).
- **License** — GPL-3.0, for this repository and the `comfydl/` submodule
  ([`LICENSE`](LICENSE), [`comfydl/LICENSE`](comfydl/LICENSE)).

This project is derived from [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by
comfyanonymous and contributors. Upstream ComfyUI keeps its own copyright and maintainers; both
notices are listed at the end of [`LICENSE`](LICENSE). The upstream project's website, docs and
community live at <https://www.comfy.org/> and are maintained separately from ComfyDL_UI.

<p align="center">
  <img src="assets/dev_2333.png" alt="" />
</p>
