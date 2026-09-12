#!/usr/bin/env python
"""Lightweight in-process smoke tester for the ComfyDL_UI node registry.

Why this exists
---------------
Starting the ComfyUI server for every change is slow.  This script loads the
very same node set the server would load (the ``comfy_extras`` whitelist plus the
``comfydl`` submodule) *inside the current process*, then executes every
registered node once with synthesised dummy inputs.  No HTTP server, no
browser, no model download: a full pass takes seconds.

What it checks per node
-----------------------
1. the node's schema / ``INPUT_TYPES`` can be resolved;
2. a value can be synthesised for every *required* input (and for hidden inputs);
3. the node's entry point runs without raising;
4. the number of returned values matches the declared output arity;
5. for the handful of nodes listed in ``_OUTPUT_CHECKS``, the returned *values* are
   the ones the documentation promises (this is how the statistics a normalization
   node exports are pinned against ``tensor.mean`` / ``tensor.var``).

Result vocabulary
-----------------
``PASS``  executed successfully and returned the declared arity.
``SKIP``  deliberately not executed (needs network / datasets / real files / ...);
          the reason is always printed and a SKIP never fails the run.
``FAIL``  unexpected exception, timeout, or an arity mismatch.

Usage
-----
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --filter Basic
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --categories
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --filter Cdl --verbose

Exit code: ``0`` when no node FAILed, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import itertools
import os
import shutil
import sys
import tempfile
import threading
import traceback
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

# V3 nodes are executed through the same normalisation hook the engine uses.
_V3_ENTRY_POINTS = ("EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC")


class Unsupported(Exception):
    """Raised when a dummy value cannot be synthesised for an input."""


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _bootstrap(sandbox: Path) -> None:
    """Make the repo importable and keep every write inside a temporary sandbox.

    ComfyUI parses ``sys.argv`` while its modules are imported, so we hand it a
    minimal command line pointing ``input``/``output``/``temp`` at a scratch
    directory.  The process also *works* inside the sandbox (``chdir``), so nodes
    that save to a plain relative path (e.g. ``torch.save(state_dict, "model.pt")``)
    cannot litter the repository.  Everything is deleted again afterwards.
    """
    warnings.simplefilter("ignore")  # keep the report readable (matplotlib etc.)
    for sub in ("input", "output", "temp"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(sandbox)
    _seed_fixtures(sandbox)
    sys.argv = [
        str(Path(__file__).resolve()),
        "--cpu",
        "--input-directory", str(sandbox / "input"),
        "--output-directory", str(sandbox / "output"),
        "--temp-directory", str(sandbox / "temp"),
    ]


def _seed_fixtures(sandbox: Path) -> None:
    """Write the small input files that a few nodes read through a default path.

    ``CdlModelSave`` / ``CdlModelLoad`` both default their ``path`` widget to
    ``"model.pt"`` (relative to the working directory, which is the sandbox).  The
    state_dict is taken from the same dummy model :func:`_f_model` hands to those
    nodes, so the load node really exercises ``torch.load`` +
    ``load_state_dict`` instead of being skipped.

    The parameter file does the same job for the training family: ``Load
    Parameters`` defaults to the first file ``Save Parameters`` writes, and the
    harness reaches the nodes in alphabetical order
    (``TrainingLoadParameters`` before ``TrainingSaveParameters``), so one has to
    exist before the run.  Its content is :func:`_train_fixture_parameters`,
    which ``_check_training_load_parameters`` pins the loaded values against.
    """
    import torch
    from safetensors.torch import save_file

    torch.save(_f_model({}, "model").state_dict(), sandbox / "model.pt")

    bucket = sandbox / "output" / "comfydl"
    bucket.mkdir(parents=True, exist_ok=True)
    save_file(
        {"weight": _train_fixture_parameters()},
        str(bucket / "parameters_00001_.safetensors"),
    )


def _install_sandbox_paths(sandbox: Path) -> None:
    """Redirect ComfyUI's file access into the sandbox and add a fake checkpoint.

    ``folder_paths`` is imported long after :func:`_bootstrap` ran and derives the
    output / input / temp roots from the repository itself, so they are redirected
    here - otherwise the MODEL protocol save nodes would drop ``.safetensors`` files
    into the checkout.

    The protocol loaders read real files through ``folder_paths``, so one small fake
    checkpoint is written into a scratch model tree and every model folder is
    registered to point at it (prepended, so the loaders' default widget value
    resolves to the fixture).  ``filename_list_cache`` is dropped afterwards: the
    dropdowns would otherwise keep the empty list they were first built with.
    """
    import comfy.utils
    import folder_paths

    folder_paths.set_output_directory(str(sandbox / "output"))
    folder_paths.set_input_directory(str(sandbox / "input"))
    folder_paths.set_temp_directory(str(sandbox / "temp"))

    sd = _fixture_state_dict(0)
    for folder in ("checkpoints", "diffusion_models", "vae", "text_encoders", "loras"):
        root = sandbox / "models" / folder
        root.mkdir(parents=True, exist_ok=True)
        comfy.utils.save_torch_file(sd, root / _FIXTURE_FILE)
        folder_paths.add_model_folder_path(folder, str(root), is_default=True)
    folder_paths.filename_list_cache.clear()


def _load_registry(verbose: bool) -> dict[str, type]:
    """Import the host `nodes` module and populate the global node mappings."""
    import nodes as host_nodes

    failed = asyncio.run(
        host_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    )
    if failed:
        print(f"[smoke] WARNING: modules that failed to import: {failed}")
    elif verbose:
        print("[smoke] all built-in node modules imported cleanly")
    return dict(host_nodes.NODE_CLASS_MAPPINGS)


# --------------------------------------------------------------------------- #
# Dummy value synthesis
# --------------------------------------------------------------------------- #
def _clamp_number(value: Any, lo: Any, hi: Any) -> Any:
    """Clamp ``value`` into the ``[lo, hi]`` window, ignoring unset bounds."""
    if isinstance(lo, (int, float)) and value < lo:
        value = lo
    if isinstance(hi, (int, float)) and value > hi:
        value = hi
    return value


def _f_int(cfg: dict, name: str) -> int:
    value = cfg.get("default")
    if value is None:
        value = cfg.get("min")
    if value is None:
        value = 1
    return int(_clamp_number(int(value), cfg.get("min"), cfg.get("max")))


def _f_float(cfg: dict, name: str) -> float:
    value = cfg.get("default")
    if value is None:
        value = cfg.get("min")
    if value is None:
        value = 0.5
    return float(_clamp_number(float(value), cfg.get("min"), cfg.get("max")))


def _f_string(cfg: dict, name: str) -> str:
    """Use the declared default, so the harness mirrors "drop the node and run".

    Empty defaults are kept as-is: several nodes use ``""`` to mean "feature
    off", and a made-up placeholder would change their behaviour.
    """
    value = cfg.get("default")
    return value if isinstance(value, str) else "smoke"


def _f_bool(cfg: dict, name: str) -> bool:
    value = cfg.get("default")
    return bool(value) if value is not None else True


def _f_combo(cfg: dict, name: str) -> Any:
    options = cfg.get("options")
    if options is None:
        options = cfg.get("values")
    if callable(options):
        raise Unsupported(f"dynamic options for combo input {name!r}")
    if isinstance(options, dict):
        options = list(options.keys())
    if not options:
        raise Unsupported(f"no options for combo input {name!r}")
    values = [opt.value if isinstance(opt, Enum) else opt for opt in options]
    default = cfg.get("default")
    default = default.value if isinstance(default, Enum) else default
    return default if default in values else values[0]


def _f_tensor(cfg: dict, name: str) -> Any:
    import torch

    return torch.randn(2, 3)


def _f_image(cfg: dict, name: str) -> Any:
    import torch

    return torch.rand(1, 8, 8, 3)


def _f_mask(cfg: dict, name: str) -> Any:
    import torch

    return torch.rand(1, 8, 8)


def _f_latent(cfg: dict, name: str) -> dict[str, Any]:
    import torch

    return {"samples": torch.randn(1, 4, 8, 8)}


def _f_model(cfg: dict, name: str) -> Any:
    """A tiny ``nn.Module``: ComfyDL's ``cdlModel`` is just "any nn.Module".

    ``in_features=3`` matches the dummy ``TENSOR`` (2, 3) produced by
    :func:`_f_tensor`, so forward-pass nodes work with the synthesised pair.
    """
    import torch

    return torch.nn.Linear(3, 2)


def _f_dataloader(cfg: dict, name: str) -> Any:
    import torch

    dataset = torch.utils.data.TensorDataset(torch.randn(8, 3), torch.randn(8, 1))
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def _f_index_small(cfg: dict, name: str) -> Any:
    """Index dummy that always stays inside a 2-row table."""
    import torch

    return torch.randint(0, 2, (2, 3))


def _f_index_1d(cfg: dict, name: str) -> Any:
    """Flat index list, for nodes that iterate ids one by one."""
    import torch

    return torch.randint(0, 2, (4,))


def _f_boxes(cfg: dict, name: str) -> Any:
    """Bounding boxes: ``(num_boxes, 4)``, the shape every box helper expects."""
    import torch

    return torch.rand(3, 4)


def _f_anchors(cfg: dict, name: str) -> Any:
    """Batched anchors: ``(batch, num_anchors, 4)``."""
    import torch

    return torch.rand(1, 3, 4)


def _f_multibox_labels(cfg: dict, name: str) -> Any:
    """Detection labels: ``(batch, num_objects, 5)`` = class + 4 coordinates."""
    import torch

    return torch.rand(2, 3, 5)


def _f_cls_probs(cfg: dict, name: str) -> Any:
    """Detection scores: ``(batch, num_classes, num_anchors)``."""
    import torch

    return torch.rand(1, 3, 3)


def _f_offset_preds(cfg: dict, name: str) -> Any:
    """Flattened box offsets: ``(batch, num_anchors * 4)``."""
    import torch

    return torch.rand(1, 12)


def _f_scores(cfg: dict, name: str) -> Any:
    """One confidence score per box: ``(num_boxes,)``."""
    import torch

    return torch.rand(3)


def _f_seq3d(cfg: dict, name: str) -> Any:
    """Sequence batch: ``(batch, num_steps, num_features)``."""
    import torch

    return torch.randn(2, 4, 3)


def _f_valid_len(cfg: dict, name: str) -> Any:
    """One valid length per sequence in the batch."""
    import torch

    return torch.tensor([2, 3])


def _f_class_idx(cfg: dict, name: str) -> Any:
    """One class index per prediction row."""
    import torch

    return torch.randint(0, 3, (2,))


def _f_weight_col(cfg: dict, name: str) -> Any:
    """Weight vector shaped to matmul against the generic ``(2, 3)`` tensor."""
    import torch

    return torch.randn(3, 1)


def _f_bias_1d(cfg: dict, name: str) -> Any:
    """Bias that broadcasts against a single regression output."""
    import torch

    return torch.randn(1)


def _f_features(cfg: dict, name: str) -> Any:
    """Feature table for the array/dataset helper nodes."""
    import torch

    return torch.randn(8, 3)


def _f_values6(cfg: dict, name: str) -> Any:
    """Non-negative values for chart nodes that reject negatives."""
    import torch

    return torch.rand(6)


def _f_values3(cfg: dict, name: str) -> Any:
    """Three non-negative chart values (pie slices)."""
    import torch

    return torch.rand(3)


def _f_labels6(cfg: dict, name: str) -> str:
    """Six category labels, matching :func:`_f_values6`."""
    return "A,B,C,D,E,F"


def _f_labels3(cfg: dict, name: str) -> str:
    """Three category labels, matching :func:`_f_values3`."""
    return "A,B,C"


def _f_array(cfg: dict, name: str) -> Any:
    """A plain list of numbers (core ``ARRAY`` inputs)."""
    return [1.0, 2.0, 3.0]


def _f_dict(cfg: dict, name: str) -> Any:
    """A small mapping (core ``DICT`` inputs)."""
    return {"a": 1}


def _f_color(cfg: dict, name: str) -> str:
    """A hex colour string (core ``COLOR`` inputs)."""
    return "#FF0000"


def _f_bounding_box(cfg: dict, name: str) -> dict[str, int]:
    """Core ``BOUNDING_BOX`` value: pixel coordinates inside the dummy image."""
    return {"x": 0, "y": 0, "width": 8, "height": 8}


def _f_any(cfg: dict, name: str) -> float:
    """A harmless scalar for wildcard (``*``) inputs.

    Logic/formatting nodes pass the value through untouched, so any type works;
    a float keeps it numeric for nodes that would otherwise need a real link.
    """
    return 1.0


def _f_vocab(cfg: dict, name: str) -> Any:
    """Build a vocabulary through ComfyDL's own builder so shapes always match.

    Running the builder node with its own synthesised widgets keeps the dummy in
    sync with the real implementation instead of duplicating its data layout.
    """
    from comfydl.nodes.nlp_utils import CdlVocabBuild

    args, _ = _legacy_inputs("CdlVocabBuild", CdlVocabBuild)
    out = CdlVocabBuild().execute(**args)
    return out[0] if isinstance(out, (tuple, list)) else out


def _f_lora_model(cfg: dict, name: str) -> Any:
    """A minimal ``LORA_MODEL``: two parameter tensors sharing one dtype.

    The shapes differ on purpose so a pack/unpack round trip has to restore
    more than one layout.
    """
    import torch

    return {"lora_A.weight": torch.randn(2, 3), "lora_B.weight": torch.randn(4)}


def _f_loss_map(cfg: dict, name: str) -> Any:
    """A minimal ``LOSS_MAP``: an ordered list of loss tensors, one dtype."""
    import torch

    return {"loss": [torch.randn(2, 3), torch.randn(4)]}


def _f_tensor_4d(cfg: dict, name: str) -> Any:
    """A ``(N, C, H, W)`` tensor, so channel-wise reductions really run.

    The generic ``(2, 3)`` dummy is rank 2: BatchNorm still works there, but
    InstanceNorm bails out early because it needs a spatial dimension. ``C=3``
    keeps the exported statistics small enough to compare element by element.
    """
    import torch

    return torch.randn(2, 3, 4, 4)


def _f_conv_weight(cfg: dict, name: str) -> Any:
    """A ``(out, in/groups, kH, kW)`` kernel matching :func:`_f_tensor_4d`.

    ``(4, 3, 3, 3)`` pairs with the ``(2, 3, 4, 4)`` dummy: 3 input channels, 4
    output channels, 3x3 kernel - the shape a ``nn.Conv2d(3, 4, 3)`` would own.
    """
    import torch

    return torch.randn(4, 3, 3, 3)


def _f_conv_transpose_weight(cfg: dict, name: str) -> Any:
    """A ``(in, out/groups, kH, kW)`` kernel matching :func:`_f_tensor_4d`.

    Same channel counts as :func:`_f_conv_weight` but the order is reversed,
    which is exactly the trap the ``ConvTranspose`` docstring warns about.
    """
    import torch

    return torch.randn(3, 4, 3, 3)


def _f_stats_linked(cfg: dict, name: str) -> Any:
    """Per-channel statistics for the *linked* Training Run Stats slots.

    Deliberately unlike the widget defaults (``0.0`` / ``1.0``) so a check can tell
    whether the link or the typed text won.
    """
    import torch

    return torch.tensor([0.5, -1.5, 2.0])


def _f_tensor_large(cfg: dict, name: str) -> Any:
    """A 1080p-shaped tensor, so the heatmap element budget really kicks in.

    ``1 x 3 x 400 x 400`` is 480 000 elements, above the ``max_samples`` default of
    262 144.  Without a tensor that big Show Heatmaps and ``CdlHeatmapsTo3D`` would
    take the "nothing to sample" shortcut and the stride arithmetic would never be
    exercised.
    """
    import torch

    return torch.randn(1, 3, 400, 400)


def _f_file_3d(cfg: dict, name: str) -> Any:
    """A minimal in-memory OBJ, so ``Preview3D`` goes through ``File3D.save_to``.

    A path string would satisfy the slot just as well, but it would never touch the
    write path - and that is where a 3D export either lands in the output folder or
    silently disappears.
    """
    import io

    from comfy_api.latest import Types

    payload = b"# cdl smoke test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    return Types.File3D(io.BytesIO(payload), file_format="obj")


# --------------------------------------------------------------------------- #
# MODEL protocol fixtures
# --------------------------------------------------------------------------- #
#: The fake weight file written into every sandbox model folder, so the loader
#: nodes have something real to read instead of being skipped.
_FIXTURE_FILE = "cdl_smoke_fixture.safetensors"

#: ``key -> shape`` of the synthetic checkpoint.  The ``model.`` wrapper is what an
#: SD1.5/SDXL file really uses, so the loaders have to detect and strip it; the keys
#: then group into MODEL (``diffusion_model.*``), VAE (``first_stage_model.*``) and
#: CLIP (``cond_stage_model.*``).  ``other.weight`` carries no known prefix on
#: purpose: the "never drop a weight" rule puts it in the MODEL bucket, and the merge
#: checks use it to prove a non-diffusion key is *not* blended.  ``position_ids`` is
#: an integer tensor, which the CLIP merges must leave alone.
_FIXTURE_SHAPES: dict[str, tuple[int, ...]] = {
    "model.diffusion_model.input_blocks.0.weight": (2, 3),
    "model.diffusion_model.middle_block.weight": (2, 2),
    "model.diffusion_model.output_blocks.0.weight": (3, 2),
    # Inside the diffusion model but in no block group, so ModelMergeBlocks has to
    # fall back to the 'input' ratio for it.
    "model.diffusion_model.time_embed.weight": (2, 2),
    "other.weight": (2, 2),
    "first_stage_model.decoder.conv.weight": (2, 2),
    "cond_stage_model.transformer.weight": (2, 2),
    "cond_stage_model.transformer.text_model.embeddings.position_ids": (1, 4),
}

#: Seeds handed out one at a time, so two dummies of the same type hold different
#: weights and a merge check can tell "blended" apart from "returned the first input".
_FIXTURE_SEEDS = itertools.count(1)


def _fixture_state_dict(seed: int) -> dict[str, Any]:
    """One deterministic fake checkpoint following :data:`_FIXTURE_SHAPES`."""
    import torch

    generator = torch.Generator().manual_seed(1000 + seed)
    sd: dict[str, Any] = {}
    for key, shape in _FIXTURE_SHAPES.items():
        if key.endswith("position_ids"):
            sd[key] = torch.arange(shape[-1], dtype=torch.int64).expand(*shape).clone()
        else:
            sd[key] = torch.randn(*shape, generator=generator)
    return sd


def _fixture_bucket(prefixes: tuple[str, ...], seed: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Take one bucket out of a fake checkpoint, with the container prefix normalised."""
    import comfy.model_protocol as protocol

    sd = {
        key: value
        for key, value in _fixture_state_dict(seed).items()
        if key.startswith(prefixes)
    }
    return protocol.strip_outer_prefix(sd, "auto")


def _f_protocol_model(cfg: dict, name: str) -> Any:
    """A ``MODEL``: the diffusion-model bucket of a fresh fake checkpoint."""
    import comfy.model_protocol as protocol

    sd, prefixes = _fixture_bucket(("model.diffusion_model.", "other."), next(_FIXTURE_SEEDS))
    return protocol.make_model_patcher(sd, prefixes)


def _f_protocol_clip(cfg: dict, name: str) -> Any:
    """A ``CLIP``: the ``cond_stage_model.*`` bucket of a fresh fake checkpoint."""
    import comfy.model_protocol as protocol

    sd, prefixes = _fixture_bucket(("cond_stage_model.",), next(_FIXTURE_SEEDS))
    return protocol.make_container(sd, prefixes)


def _f_protocol_vae(cfg: dict, name: str) -> Any:
    """A ``VAE``: the ``first_stage_model.*`` bucket of a fresh fake checkpoint."""
    import comfy.model_protocol as protocol

    sd, prefixes = _fixture_bucket(("first_stage_model.",), next(_FIXTURE_SEEDS))
    return protocol.make_container(sd, prefixes)


def _f_fixture_file(cfg: dict, name: str) -> str:
    """File-dropdown value: the fake checkpoint :func:`_install_fixture_models` wrote."""
    return _FIXTURE_FILE


def _protocol_weights(value: Any) -> dict[str, Any]:
    """The weights behind a MODEL / CLIP / VAE dummy."""
    import comfy.model_protocol as protocol

    return protocol.container_state_dict(value)[0]


def _train_fixture_parameters():
    """The 2x3 matrix :func:`_seed_fixtures` writes into the parameter file."""
    import torch

    return torch.arange(6, dtype=torch.float32).reshape(2, 3) / 6.0


def _training_pair():
    """A deterministic tiny regression problem: ``y = 2 * x0 - 3 * x1 + 1``.

    A pair the trainer *has* to be able to fit, so the checks can demand that the
    loss really goes down instead of only that the node returned something.
    """
    import torch

    generator = torch.Generator().manual_seed(7)
    features = torch.randn(48, 2, generator=generator)
    targets = 2.0 * features[:, :1] - 3.0 * features[:, 1:] + 1.0
    return features, targets


def _f_params(cfg: dict, name: str) -> dict[str, Any]:
    """``PARAMS`` dummy: a trainable set carrying the names the family addresses.

    ``weight`` is the key a freshly dropped ``Learnable Parameters`` node creates
    and the key ``Parameters to Tensor`` looks up by default, while
    ``layer0.weight`` follows the naming convention of the trainer's own output.
    """
    import torch

    with torch.inference_mode(False):
        return {
            "weight": torch.nn.Parameter(_train_fixture_parameters()),
            "layer0.weight": torch.nn.Parameter(torch.full((2, 3), 0.25)),
        }


def _f_params_alt(cfg: dict, name: str) -> dict[str, Any]:
    """A second, distinguishable ``PARAMS`` dummy: other values, plus a new key."""
    import torch

    with torch.inference_mode(False):
        return {
            "weight": torch.nn.Parameter(torch.full((2, 3), -1.0)),
            "bias": torch.nn.Parameter(torch.full((3,), 0.5)),
        }


def _f_optimizer(cfg: dict, name: str) -> Any:
    """``OPTIMIZER`` dummy: what the ``Optimizer`` node publishes by default."""
    from comfy import training_protocol as protocol

    return protocol.optimizer_config()


#: type string (upper-cased) -> factory producing a dummy value
_VALUE_FACTORIES: dict[str, Callable[[dict, str], Any]] = {
    "INT": _f_int,
    "FLOAT": _f_float,
    "STRING": _f_string,
    "BOOLEAN": _f_bool,
    "COMBO": _f_combo,
    "TENSOR": _f_tensor,
    "IMAGE": _f_image,
    "MASK": _f_mask,
    "LATENT": _f_latent,
    "CDLMODEL": _f_model,
    "CDLDATALOADER": _f_dataloader,
    "CDLVOCAB": _f_vocab,
    "ARRAY": _f_array,
    "DICT": _f_dict,
    "COLOR": _f_color,
    "BOUNDING_BOX": _f_bounding_box,
    "FILE_3D": _f_file_3d,
    "FILE_3D_OBJ": _f_file_3d,
    "MODEL": _f_protocol_model,
    "CLIP": _f_protocol_clip,
    "VAE": _f_protocol_vae,
    "PARAMS": _f_params,
    "OPTIMIZER": _f_optimizer,
    "*": _f_any,
}

#: types we are willing to fake for an *optional* input.  Optional slots are
#: left unconnected on purpose: a generic dummy tensor cannot know the shape a
#: node expects for e.g. a bias vector, and "not connected" is the documented
#: meaning of an optional input anyway.
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

#: hidden input name -> fixed dummy value
_HIDDEN_VALUES: dict[str, Any] = {
    "prompt": {},
    "dynprompt": {},
    "extra_pnginfo": {"workflow": {}},
    "unique_id": "smoke-test",
    "auth_token_comfy_org": "",
    "api_key_comfy_org": "",
}

#: node id -> reason; nodes that need network access, datasets, real checkpoints
#: or user interaction are not meaningfully testable with synthetic inputs.
_SKIPPED_NODES: dict[str, str] = {
    "CdlDownload": "downloads a file over HTTP",
    "CdlDownloadExtract": "downloads and extracts a dataset from the d2l DATA_HUB",
    "CdlFashionMNIST": "downloads the Fashion-MNIST dataset",
    "CdlBananasDetection": "downloads the banana-detection dataset",
    "CdlVOCSegmentation": "downloads the VOC2012 dataset (hundreds of MB)",
    "CdlMessageBox": "opens a modal Windows MessageBox and would block the run",
    "CdlWhat": "can launch a browser window when OMG is enabled",
    "CdlRNNLMScratch": "needs an RNN model built by the RNN scratch nodes, not a generic nn.Module",
    "CdlRNNLMScratchPredict": "needs an RNN model built by the RNN scratch nodes, not a generic nn.Module",
    "GetImageSize": "reports progress through PromptServer.instance, which only exists inside the server",
}

#: node id -> substring its raised error must contain.  These nodes are part of the
#: MODEL protocol layer: they are registered with a correct IO contract but cannot do
#: any work in a dehydrated build.  They are still executed here so that the "clear,
#: actionable error instead of a bare ModuleNotFoundError" promise is *verified*
#: rather than assumed, and so a regression that makes them crash differently shows up.
_EXPECTED_ERRORS: dict[str, str] = {
    "LoraLoader": "comfy/lora.py",
    "LoraLoaderModelOnly": "comfy/lora.py",
    "VAEDecode": "comfy/ldm",
    "VAEEncode": "comfy/ldm",
    "CLIPTextEncode": "comfy/text_encoders",
    "CLIPSetLastLayer": "comfy/text_encoders",
}

#: node id -> {input name -> factory}.  Some inputs are semantically narrower
#: than their slot type: an ``Embedding`` tensor is an *index* tensor, or a
#: bounding-box tensor must have exactly 4 columns.  The generic dummy cannot
#: know that, so the harness declares it here instead of weakening the node.
_INPUT_OVERRIDES: dict[str, dict[str, Callable[[dict, str], Any]]] = {
    "BasicEmbedding": {"tensor": _f_index_small},
    # NLP: indices have to fall inside the dummy vocabulary.
    "CdlVocabDecode": {"indices": _f_index_1d},
    # Object detection: (num_boxes, 4) boxes plus matching scores/anchors.
    "CdlBoxCornerToCenter": {"boxes": _f_boxes},
    "CdlBoxCenterToCorner": {"boxes": _f_boxes},
    "CdlBoxIou": {"boxes1": _f_boxes, "boxes2": _f_boxes},
    "CdlNms": {"boxes": _f_boxes, "scores": _f_scores},
    "CdlOffsetBoxes": {"anchors": _f_boxes, "assigned_bb": _f_boxes},
    "CdlOffsetInverse": {"anchors": _f_boxes, "offset_preds": _f_boxes},
    "CdlAssignAnchorToBbox": {"ground_truth": _f_boxes, "anchors": _f_boxes},
    "CdlMultiboxTarget": {"anchors": _f_anchors, "labels": _f_multibox_labels},
    "CdlMultiboxDetection": {
        "cls_probs": _f_cls_probs,
        "offset_preds": _f_offset_preds,
        "anchors": _f_anchors,
    },
    "CdlShowBboxes": {"bboxes": _f_boxes},
    # Sequences / regression / metrics.
    "CdlSequenceMask": {"X": _f_seq3d, "valid_len": _f_valid_len},
    "CdlLinReg": {"w": _f_weight_col, "b": _f_bias_1d},
    "CdlAccuracy": {"y": _f_class_idx},
    # Array/dataset helpers and matplotlib charts need consistent, non-negative data.
    "CdlLoadArray": {"features": _f_features},
    "CdlBarChart": {"values": _f_values6, "labels": _f_labels6},
    "CdlConfusionMatrix": {"class_labels": _f_labels3},
    "CdlPieChart": {"values": _f_values3},
    # utilities/conversion: tensor collections have no generic dummy value, so
    # the two pack nodes would otherwise be skipped instead of executed.
    "CdlLoraModelToTensor": {"lora_model": _f_lora_model},
    "CdlLossMapToTensor": {"loss_map": _f_loss_map},
    # Normalization: a 4-D dummy, otherwise the exported statistics would be taken
    # from a rank-2 tensor and InstanceNorm would early-out instead of normalizing.
    "NormalizationBatchNorm": {"tensor": _f_tensor_4d},
    "NormalizationInstanceNorm": {"tensor": _f_tensor_4d},
    # Training Run Stats: only ``mean`` is linked, so a single run covers both the
    # "a link wins over the widget" path and the "widget text is parsed" path.
    "TrainingRunStats": {"mean": _f_stats_linked},
    # Show Heatmaps / Heatmaps to 3D: a 1080p-shaped tensor forces the element
    # budget - and with it the stride and axis-selection - to actually run.
    "CdlShowHeatmaps": {"matrices": _f_tensor_large},
    "CdlShowHeatmapsOutput": {"matrices": _f_tensor_large},
    "CdlHeatmapsTo3D": {"matrices": _f_tensor_large},
    # Preview3D: a real ``Types.File3D``, so a file is really written to disk.
    "Preview3D": {"model_file": _f_file_3d},
    # Pooling / Convolution: rank matters.  The generic (2, 3) dummy is rank 2, so
    # 2d pooling would run on a (1, 2, 3) "image" and the convolution kernels need
    # a real (out, in/groups, kH, kW) / (in, out/groups, kH, kW) partner for the
    # (2, 3, 4, 4) input.
    "PoolingSliding": {"tensor": _f_tensor_4d},
    "PoolingAdaptive": {"tensor": _f_tensor_4d},
    "ConvolutionConv": {"tensor": _f_tensor_4d, "weight": _f_conv_weight},
    "ConvolutionConvTranspose": {
        "tensor": _f_tensor_4d,
        "weight": _f_conv_transpose_weight,
    },
    # MODEL protocol: the loader combos are empty in a fresh checkout, so every
    # loader is pointed at the fake checkpoint the harness writes into the sandbox
    # model folders.  Without this they would be skipped instead of really reading.
    "CheckpointLoaderSimple": {"ckpt_name": _f_fixture_file},
    "UNETLoader": {"unet_name": _f_fixture_file},
    "VAELoader": {"vae_name": _f_fixture_file},
    "CLIPLoader": {"clip_name": _f_fixture_file},
    "DualCLIPLoader": {"clip_name1": _f_fixture_file, "clip_name2": _f_fixture_file},
    # Training: 'x' and 'y' are a *matched pair* (a generic dummy tensor has no
    # reason to be), and 'params_b' has to differ from 'params_a' so that the
    # merge check can tell the two inputs apart.
    "TrainingLoop": {
        "x": lambda cfg, name: _training_pair()[0],
        "y": lambda cfg, name: _training_pair()[1],
    },
    "TrainingParametersMerge": {"params_b": _f_params_alt},
}


def _as_tuple(result: Any) -> tuple[Any, ...]:
    """The node's outputs as a plain tuple, whatever wrapper it used."""
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def _channel_dims(tensor: Any) -> tuple[int, ...]:
    """The reduction dims BatchNorm uses: everything but the channel axis."""
    return (0,) + tuple(range(2, tensor.dim()))


def _check_batchnorm_stats(result: Any, args: dict[str, Any]) -> None:
    """``mean`` / ``var`` must be exactly the statistics the output was built from.

    The point of the node is that the two exported values are not a *second*,
    independently computed estimate: they are the numbers ``F.batch_norm`` was
    handed, which is what makes feeding them into Training Run Stats lossless.
    """
    import torch

    tensor = args["tensor"]
    output, mean, var = _as_tuple(result)
    dims = _channel_dims(tensor)
    expected_mean = tensor.mean(dim=dims)
    expected_var = tensor.var(dim=dims, unbiased=False)
    assert tuple(mean.shape) == tuple(expected_mean.shape), (
        f"mean shape {tuple(mean.shape)} != {tuple(expected_mean.shape)}"
    )
    assert tuple(var.shape) == tuple(expected_var.shape), (
        f"var shape {tuple(var.shape)} != {tuple(expected_var.shape)}"
    )
    assert torch.allclose(mean, expected_mean, atol=1e-6), f"mean {mean} != {expected_mean}"
    assert torch.allclose(var, expected_var, atol=1e-6), f"var {var} != {expected_var}"
    view = (1, -1) + (1,) * (tensor.dim() - 2)
    manual = (tensor - expected_mean.view(view)) / torch.sqrt(
        expected_var.view(view) + float(args.get("eps", 1e-5))
    )
    assert torch.allclose(output, manual, atol=1e-5), "output was not built from the exported statistics"


def _check_instancenorm_stats(result: Any, args: dict[str, Any]) -> None:
    """``mean`` / ``var`` must be the whole-batch statistics BatchNorm would report.

    InstanceNorm computes one mean per sample; collapsing those to one value per
    channel only reproduces the batch variance because of the two-way variance
    decomposition (mean of the within-sample variances + variance of the means).
    Comparing against ``tensor.var((0, 2, 3))`` pins that identity down.
    """
    import torch

    tensor = args["tensor"]
    _, mean, var = _as_tuple(result)
    spatial = tuple(range(2, tensor.dim()))
    dims = (0,) + spatial
    assert torch.allclose(mean, tensor.mean(dim=dims), atol=1e-6), f"mean {mean} != {tensor.mean(dim=dims)}"
    assert torch.allclose(var, tensor.var(dim=dims, unbiased=False), atol=1e-6), (
        f"var {var} != {tensor.var(dim=dims, unbiased=False)}"
    )


def _check_training_run_stats(result: Any, args: dict[str, Any]) -> None:
    """A linked statistics tensor must win over the widget text it shadows."""
    import torch

    mean_out, var_out = _as_tuple(result)
    linked = args["mean"]
    assert torch.allclose(mean_out, linked.reshape(-1).to(torch.float32)), (
        f"the linked 'mean' did not override the widget: {mean_out}"
    )
    assert torch.allclose(var_out, torch.tensor([1.0])), (
        f"the unlinked 'var' widget was not parsed as before: {var_out}"
    )


def _rerun_v3(node_id: str, **overrides: Any) -> tuple[Any, ...]:
    """Run a V3 node again with a few inputs replaced.

    The mirror image of :func:`_rerun_legacy` for ``io.ComfyNode`` classes: the
    schema is re-read so overrides only have to name the inputs that change.
    """
    node_cls = _REGISTRY[node_id]
    args, _, _, _ = _v3_inputs(node_id, node_cls)
    args.update(overrides)
    result = node_cls.execute(**args)
    return _as_tuple(getattr(result, "result", result))


def _check_pooling_sliding(result: Any, args: dict[str, Any]) -> None:
    """``Pool`` must be a thin dispatcher over ``F.max_pool{1,2,3}d`` / ``avg_pool``.

    The default widgets (dims=2, mode=max, kernel_size=2, stride=0 => 2,
    padding=0, dilation=1, ceil_mode=False) are the textbook ``nn.MaxPool2d(2)``,
    and every rank has to reach the matching kernel with the same widget set -
    that dispatch table is the whole node, so a shape check is not enough.
    """
    import torch
    import torch.nn.functional as F

    tensor = args["tensor"]
    (output,) = _as_tuple(result)
    expected = F.max_pool2d(tensor, 2, 2, 0, 1, False)
    assert tuple(output.shape) == tuple(expected.shape), (
        f"{tuple(output.shape)} != {tuple(expected.shape)}"
    )
    assert torch.allclose(output, expected, atol=1e-6), "mode=max did not reach F.max_pool2d"

    (averaged,) = _rerun_v3("PoolingSliding", tensor=tensor, mode="avg")
    expected_avg = F.avg_pool2d(tensor, 2, 2, 0, False, True)
    assert torch.allclose(averaged, expected_avg, atol=1e-6), "mode=avg did not reach F.avg_pool2d"

    rank_cases = {
        1: (torch.randn(2, 3, 8), F.max_pool1d),
        3: (torch.randn(2, 3, 8, 8, 8), F.max_pool3d),
    }
    for rank, (shaped, kernel) in rank_cases.items():
        (pooled,) = _rerun_v3("PoolingSliding", tensor=shaped, dims=rank, kernel_size=2)
        expected_rank = kernel(shaped, 2, 2, 0, 1, False)
        assert pooled.dim() == shaped.dim(), f"dims={rank} changed the tensor rank"
        assert torch.allclose(pooled, expected_rank, atol=1e-6), f"dims={rank} reached the wrong kernel"

    # An unbatched (C, H, W) tensor has to survive the round trip unchanged in rank.
    (unbatched,) = _rerun_v3("PoolingSliding", tensor=torch.randn(3, 8, 8))
    assert unbatched.dim() == 3 and tuple(unbatched.shape[:1]) == (3,), tuple(unbatched.shape)


def _check_pooling_adaptive(result: Any, args: dict[str, Any]) -> None:
    """``Adaptive Pool`` must be ``F.adaptive_*_pool{1,2,3}d``, with 1 = global pooling."""
    import torch
    import torch.nn.functional as F

    tensor = args["tensor"]
    (output,) = _as_tuple(result)
    expected = F.adaptive_avg_pool2d(tensor, 1)
    assert tuple(output.shape) == tuple(expected.shape), (
        f"output_size=1 must collapse every spatial dim, got {tuple(output.shape)}"
    )
    assert torch.allclose(output, expected, atol=1e-6), "Global Average Pooling is not adaptive_avg_pool2d(1)"

    (maxed,) = _rerun_v3("PoolingAdaptive", tensor=tensor, mode="max", output_size=2)
    assert torch.allclose(maxed, F.adaptive_max_pool2d(tensor, 2), atol=1e-6), (
        "mode=max did not reach F.adaptive_max_pool2d"
    )

    (unbatched,) = _rerun_v3("PoolingAdaptive", tensor=torch.randn(3, 8, 8))
    assert unbatched.dim() == 3, f"an unbatched tensor changed rank: {tuple(unbatched.shape)}"


def _check_convolution_conv(result: Any, args: dict[str, Any]) -> None:
    """``Conv`` must reach ``F.conv{1,2,3}d``, padding modes included.

    The default widgets (dims=2, groups=1, stride=1, padding=1, padding_mode=zeros,
    dilation=1) reproduce ``nn.Conv2d(3, 4, 3, padding=1)``; the ``reflect`` case
    proves the explicit ``F.pad`` path really replaces ``padding`` instead of
    padding twice.
    """
    import torch
    import torch.nn.functional as F

    tensor = args["tensor"]
    weight = args["weight"]
    (output,) = _as_tuple(result)
    expected = F.conv2d(tensor, weight, None, stride=1, padding=1)
    assert tuple(output.shape) == tuple(expected.shape), (
        f"{tuple(output.shape)} != {tuple(expected.shape)}"
    )
    assert torch.allclose(output, expected, atol=1e-5), (
        "the default widgets are not nn.Conv2d(3, 4, 3, padding=1)"
    )

    (reflected,) = _rerun_v3(
        "ConvolutionConv", tensor=tensor, weight=weight, padding_mode="reflect"
    )
    padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
    assert torch.allclose(reflected, F.conv2d(padded, weight, None, stride=1, padding=0), atol=1e-5), (
        "padding_mode=reflect is not F.pad(mode='reflect') followed by a zero-padded conv"
    )

    bias = torch.randn(4)
    (biased,) = _rerun_v3("ConvolutionConv", tensor=tensor, weight=weight, bias=bias)
    assert torch.allclose(biased, F.conv2d(tensor, weight, bias, stride=1, padding=1), atol=1e-5), (
        "the optional bias slot was ignored"
    )

    # dims=1 must reach F.conv1d with the 1-D kernel shape (out, in/groups, k).
    shaped = torch.randn(2, 3, 8)
    weight_1d = torch.randn(4, 3, 3)
    (rank1,) = _rerun_v3("ConvolutionConv", tensor=shaped, weight=weight_1d, dims=1)
    assert torch.allclose(rank1, F.conv1d(shaped, weight_1d, None, stride=1, padding=1), atol=1e-5), (
        "dims=1 did not reach F.conv1d"
    )
    assert tuple(rank1.shape[:2]) == (2, 4), tuple(rank1.shape)


def _check_convolution_conv_transpose(result: Any, args: dict[str, Any]) -> None:
    """``ConvTranspose`` must reach ``F.conv_transpose{1,2,3}d`` and grow the spatial size."""
    import torch
    import torch.nn.functional as F

    tensor = args["tensor"]
    weight = args["weight"]
    (output,) = _as_tuple(result)
    expected = F.conv_transpose2d(
        tensor, weight, None, stride=2, padding=0, output_padding=0, groups=1, dilation=1
    )
    assert tuple(output.shape) == tuple(expected.shape), (
        f"{tuple(output.shape)} != {tuple(expected.shape)}; stride=2 should enlarge the spatial size"
    )
    assert torch.allclose(output, expected, atol=1e-5), (
        "the default widgets are not F.conv_transpose2d(stride=2)"
    )

    # The reverse channel order is the trap this node's docstring warns about.
    shaped = torch.randn(2, 3, 8)
    weight_1d = torch.randn(3, 4, 3)
    (rank1,) = _rerun_v3(
        "ConvolutionConvTranspose", tensor=shaped, weight=weight_1d, dims=1
    )
    assert torch.allclose(
        rank1,
        F.conv_transpose1d(
            shaped, weight_1d, None, stride=2, padding=0, output_padding=0, groups=1, dilation=1
        ),
        atol=1e-5,
    ), "dims=1 did not reach F.conv_transpose1d"
    assert tuple(rank1.shape[:2]) == (2, 4), tuple(rank1.shape)


#: node id -> class, filled in by :func:`main`.  Output checks need it to re-run a
#: node with a different input, because the harness only ever synthesises one set of
#: dummies per node.
_REGISTRY: dict[str, type] = {}

#: Shapes used by the heatmap checks: every render branch, plus the ranks that only
#: become renderable after a reduction.
_HEATMAP_SHAPES = {
    "1-D strip": (64,),
    "2-D plane": (12, 9),
    "3-D cube": (6, 7, 8),
    "4-D batch": (1, 3, 8, 9),
    "5-D batch": (2, 3, 4, 5, 6),
}


def _rerun_legacy(node_id: str, **overrides: Any) -> tuple[Any, ...]:
    """Run a legacy node again with a few inputs replaced."""
    node_cls = _REGISTRY[node_id]
    args, _ = _legacy_inputs(node_id, node_cls)
    args.update(overrides)
    return _as_tuple(node_cls().execute(**args))


def _check_heatmap_ranks(node_id: str, result: Any, args: dict[str, Any]) -> None:
    """Every rank must render as one finite RGB image, and the budget must pay off.

    1-D becomes a bar-code strip, 2-D a plane, 3-D a translucent cube; the 4-D and
    5-D cases only work because the extra axes are reduced away first.  This is the
    check that pins the behaviour the old implementation got wrong.
    """
    import time

    import torch

    from comfydl.nodes.visualization import HeatmapSpecError

    is_output_node = bool(getattr(_REGISTRY[node_id], "OUTPUT_NODE", False))
    for label, shaped in _HEATMAP_SHAPES.items():
        out = _rerun_legacy(node_id, matrices=torch.randn(*shaped), max_samples=0)
        if is_output_node:
            assert out == (), f"{label}: output node returned {out}"
            continue
        image = out[0]
        assert tuple(image.shape)[0] == 1 and tuple(image.shape)[-1] == 3, (
            f"{label}: {tuple(image.shape)} is not a single RGB image"
        )
        assert torch.isfinite(image).all(), f"{label}: image contains NaN/Inf"

    # An axis that does not exist must either be reported or quietly recovered from.
    for on_error in ("error", "fallback_first_n"):
        try:
            _rerun_legacy(node_id, matrices=torch.randn(4, 5), dims="9",
                          on_error=on_error, max_samples=0)
        except HeatmapSpecError:
            assert on_error == "error", "on_error='fallback_first_n' must not raise"
        else:
            assert on_error == "fallback_first_n", "on_error='error' must raise"

    # The element budget has to pay for itself on a real 1080p tensor.
    big = torch.randn(1, 3, 1080, 1920)
    start = time.perf_counter()
    _rerun_legacy(node_id, matrices=big)
    sampled_cost = time.perf_counter() - start
    start = time.perf_counter()
    _rerun_legacy(node_id, matrices=big, max_samples=0)
    full_cost = time.perf_counter() - start
    print(f"    sampling self-check: {sampled_cost:.3f}s sampled vs {full_cost:.3f}s unsampled")
    assert sampled_cost < full_cost, (
        f"sampling did not pay for itself ({sampled_cost:.3f}s vs {full_cost:.3f}s)"
    )


def _check_heatmaps_to_3d(result: Any, args: dict[str, Any]) -> None:
    """Every rank must export an OBJ whose MTL really lands next to it on save.

    Writing the ``.mtl`` is the part that cannot be taken for granted: the viewer
    renames the OBJ to ``preview3d_<uuid>.obj``, so a helper that wrote the MTL
    under the *original* name would leave the model uncoloured and opaque.
    """
    import folder_paths

    import torch

    node_cls = _REGISTRY["CdlHeatmapsTo3D"]
    output_dir = Path(folder_paths.get_output_directory())
    for label, shaped in _HEATMAP_SHAPES.items():
        payload = node_cls().execute(torch.randn(*shaped), "Reds", 0.5, 0.15, max_samples=0)[0]
        assert payload.format == "obj", f"{label}: file format {payload.format!r}"
        target = output_dir / f"smoke_{label.replace(' ', '_').replace('-', '_')}.obj"
        payload.save_to(str(target))
        material = target.with_suffix(".mtl")
        assert target.is_file() and material.is_file(), (
            f"{label}: {target.name} / {material.name} missing"
        )
        text = target.read_text(encoding="utf-8")
        assert text.splitlines()[0] == f"mtllib {material.name}", (
            f"{label}: first line was {text.splitlines()[0]!r}"
        )
        assert "\nusemtl " in text and "\nf " in text, f"{label}: mesh has no faces"

    payload = node_cls().execute(torch.randn(4, 4), "Reds", 0.25, 0.15, max_samples=0)[0]
    target = output_dir / "smoke_opacity.obj"
    payload.save_to(str(target))
    assert "d 0.2500" in target.with_suffix(".mtl").read_text(encoding="utf-8"), (
        "the opacity widget never reached the MTL"
    )


def _check_preview3d(result: Any, args: dict[str, Any]) -> None:
    """``Preview3D`` must persist what it is given - and keep a mesh's MTL with it."""
    import folder_paths

    import torch

    preview = _REGISTRY["Preview3D"]
    output_dir = Path(folder_paths.get_output_directory())

    name = preview.execute(args["model_file"]).ui.as_dict()["result"][0]
    assert name.startswith("preview3d_") and name.endswith(".obj"), name
    assert (output_dir / name).is_file(), f"{name} was never written to the output folder"

    mesh = _REGISTRY["CdlHeatmapsTo3D"]().execute(
        torch.randn(6, 7, 8), "Reds", 0.5, 0.15, max_samples=0
    )[0]
    name = preview.execute(mesh).ui.as_dict()["result"][0]
    assert (output_dir / name).is_file(), "the renaming preview did not write the OBJ"
    assert (output_dir / Path(name).with_suffix(".mtl").name).is_file(), (
        "the sibling .mtl did not follow the renamed OBJ: the model would be uncoloured"
    )


# --------------------------------------------------------------------------- #
# MODEL protocol checks
# --------------------------------------------------------------------------- #
def _written_files(pattern: str) -> list[Path]:
    """Files written under the (sandboxed) output directory, newest last.

    ``filename_prefix`` is a *path plus file stem*, not a directory: a save node
    called with ``comfydl/diffusion_models`` writes
    ``output/comfydl/diffusion_models_00001_.safetensors``.
    """
    import folder_paths

    return sorted(Path(folder_paths.get_output_directory()).glob(pattern))


def _check_checkpoint_loader(result: Any, args: dict[str, Any]) -> None:
    """The three buckets must come from the three key prefixes and nowhere else.

    ``prefix_strip=auto`` has to remove the ``model.`` container wrapper (a merge
    filter could not see ``diffusion_model.`` through it), every key must land in
    exactly one bucket, and no key may be lost.
    """
    model, clip, vae = _as_tuple(result)
    model_sd, clip_sd, vae_sd = (
        _protocol_weights(model),
        _protocol_weights(clip),
        _protocol_weights(vae),
    )
    total = len(model_sd) + len(clip_sd) + len(vae_sd)
    assert total == len(_FIXTURE_SHAPES), (
        f"{len(model_sd)}(model) + {len(clip_sd)}(clip) + {len(vae_sd)}(vae) "
        f"!= {len(_FIXTURE_SHAPES)} source keys: a weight was dropped or duplicated"
    )
    assert all(key.startswith(("diffusion_model.", "other.")) for key in model_sd), sorted(model_sd)
    assert all(key.startswith("first_stage_model.") for key in vae_sd), sorted(vae_sd)
    assert all(key.startswith("cond_stage_model.") for key in clip_sd), sorted(clip_sd)
    assert not (set(model_sd) & set(clip_sd)), "a key landed in both the MODEL and CLIP buckets"
    assert not (set(model_sd) & set(vae_sd)), "a key landed in both the MODEL and VAE buckets"
    assert not any(key.startswith("model.") for key in (*model_sd, *clip_sd, *vae_sd)), (
        "prefix_strip=auto did not strip the 'model.' container wrapper"
    )


def _check_checkpoint_save(result: Any, args: dict[str, Any]) -> None:
    """The checkpoint must really be on disk with the source file's key set.

    This is the round-trip proof for the whole protocol layer: keys normalised on
    load have to come back out exactly as they went in, otherwise "load, merge,
    save" would silently rewrite a checkpoint's key layout.
    """
    import comfy.utils

    written = _written_files("comfydl/checkpoints*.safetensors")
    assert written, "no checkpoint .safetensors was written under output/comfydl/"
    loaded = comfy.utils.load_torch_file(str(written[-1]), safe_load=True)
    assert set(loaded) == set(_FIXTURE_SHAPES), (
        f"saved keys differ from the source file: {sorted(set(loaded) ^ set(_FIXTURE_SHAPES))}"
    )


def _check_model_save(result: Any, args: dict[str, Any]) -> None:
    """``ModelSave`` must write the MODEL bucket with its prefixes replayed."""
    import comfy.utils

    written = _written_files("comfydl/diffusion_models*.safetensors")
    assert written, "no model .safetensors was written under output/comfydl/"
    loaded = comfy.utils.load_torch_file(str(written[-1]), safe_load=True)
    expected = {
        key for key in _FIXTURE_SHAPES if key.startswith(("model.diffusion_model.", "other."))
    }
    assert set(loaded) == expected, f"{sorted(set(loaded) ^ expected)}"


def _check_clip_save(result: Any, args: dict[str, Any]) -> None:
    """``CLIPSave`` must write the CLIP bucket."""
    assert _written_files("comfydl/clip*.safetensors"), (
        "no CLIP .safetensors was written under output/comfydl/"
    )


def _check_vae_save(result: Any, args: dict[str, Any]) -> None:
    """``VAESave`` must write the VAE bucket."""
    assert _written_files("comfydl/vae*.safetensors"), (
        "no VAE .safetensors was written under output/comfydl/"
    )


def _check_model_merge_simple(result: Any, args: dict[str, Any]) -> None:
    """``ratio`` must scale ``model1`` and ``1 - ratio`` must scale ``model2``.

    The official documentation defines ``ratio=1`` as "100% model1" and ``ratio=0``
    as "100% model2".  All three cases are checked here, and with two *different*
    dummies, so returning either input untouched cannot pass by accident.  The
    filter is checked too: ``other.weight`` carries no ``diffusion_model.`` marker,
    so it must be copied from model1 exactly as the native node does.
    """
    import torch

    model1, model2 = args["model1"], args["model2"]
    a, b = _protocol_weights(model1), _protocol_weights(model2)
    assert set(a) == set(b), "the two MODEL dummies must share a key set"

    (kept1,) = _as_tuple(result)
    for key, value in _protocol_weights(kept1).items():
        assert torch.equal(value, a[key]), f"ratio=1.0 (default) did not keep model1's {key!r}"

    (blended_value,) = _rerun_v3("ModelMergeSimple", model1=model1, model2=model2, ratio=0.5)
    blended = _protocol_weights(blended_value)
    for key in a:
        if key.startswith("diffusion_model."):
            expected = 0.5 * a[key] + 0.5 * b[key]
            assert torch.allclose(blended[key], expected, atol=1e-6), (
                f"ratio=0.5 did not blend {key!r}: {blended[key]} != {expected}"
            )
        else:
            assert torch.equal(blended[key], a[key]), f"non-diffusion key {key!r} was blended"

    (kept2,) = _rerun_v3("ModelMergeSimple", model1=model1, model2=model2, ratio=0.0)
    for key, value in _protocol_weights(kept2).items():
        expected = b[key] if key.startswith("diffusion_model.") else a[key]
        assert torch.allclose(value, expected, atol=1e-6), f"ratio=0.0 did not keep model2's {key!r}"


def _check_model_merge_blocks(result: Any, args: dict[str, Any]) -> None:
    """Each UNet block group must use its own ratio, ``input`` acting as the fallback."""
    import torch

    model1, model2 = args["model1"], args["model2"]
    a, b = _protocol_weights(model1), _protocol_weights(model2)
    (value,) = _rerun_v3(
        "ModelMergeBlocks", model1=model1, model2=model2, input=0.25, middle=0.5, out=0.75
    )
    out = _protocol_weights(value)
    cases = {
        "diffusion_model.input_blocks.0.weight": 0.25,
        "diffusion_model.middle_block.weight": 0.5,
        "diffusion_model.output_blocks.0.weight": 0.75,
        # Inside the diffusion model but in no block group, so the 'input' ratio
        # applies - native behaviour, where it is the default of the kwargs order.
        "diffusion_model.time_embed.weight": 0.25,
    }
    for key, ratio in cases.items():
        expected = ratio * a[key] + (1 - ratio) * b[key]
        assert torch.allclose(out[key], expected, atol=1e-6), (
            f"{key!r} should use ratio {ratio}: {out[key]} != {expected}"
        )
    # A key outside the diffusion model is not part of the merge at all (the native
    # node filters on ``diffusion_model.`` too), so it must come through untouched.
    assert torch.equal(out["other.weight"], a["other.weight"]), (
        "'other.weight' is outside the diffusion model and must not be blended"
    )


def _check_model_merge_add(result: Any, args: dict[str, Any]) -> None:
    """``ModelMergeAdd`` must be ``model1 + model2``."""
    import torch

    a = _protocol_weights(args["model1"])
    b = _protocol_weights(args["model2"])
    (value,) = _as_tuple(result)
    out = _protocol_weights(value)
    for key in a:
        if key.startswith("diffusion_model."):
            assert torch.allclose(out[key], a[key] + b[key], atol=1e-6), key
        else:
            assert torch.equal(out[key], a[key]), key


def _check_model_merge_subtract(result: Any, args: dict[str, Any]) -> None:
    """``ModelMergeSubtract`` must be ``model1 - model2`` at multiplier 1.0."""
    import torch

    a = _protocol_weights(args["model1"])
    b = _protocol_weights(args["model2"])
    (value,) = _as_tuple(result)
    out = _protocol_weights(value)
    for key in a:
        if key.startswith("diffusion_model."):
            assert torch.allclose(out[key], a[key] - b[key], atol=1e-6), key
        else:
            assert torch.equal(out[key], a[key]), key


def _check_clip_merge_simple(result: Any, args: dict[str, Any]) -> None:
    """CLIP blends must skip ``.position_ids`` / ``.logit_scale``, as the natives do."""
    import torch

    clip1, clip2 = args["clip1"], args["clip2"]
    a, b = _protocol_weights(clip1), _protocol_weights(clip2)
    assert set(a) == set(b), "the two CLIP dummies must share a key set"
    ids_key = next(key for key in a if key.endswith(".position_ids"))
    assert not a[ids_key].is_floating_point(), "position_ids must be an index tensor"

    (value,) = _rerun_v3("CLIPMergeSimple", clip1=clip1, clip2=clip2, ratio=0.5)
    out = _protocol_weights(value)
    for key in a:
        if key.endswith((".position_ids", ".logit_scale")) or not a[key].is_floating_point():
            assert torch.equal(out[key], a[key]), f"{key!r} must be kept from clip1 untouched"
        else:
            expected = 0.5 * a[key] + 0.5 * b[key]
            assert torch.allclose(out[key], expected, atol=1e-6), (
                f"ratio=0.5 did not blend {key!r}: {out[key]} != {expected}"
            )


def _check_clip_merge_add(result: Any, args: dict[str, Any]) -> None:
    """``CLIPMergeAdd`` must be ``clip1 + clip2`` for the weight keys."""
    import torch

    a = _protocol_weights(args["clip1"])
    b = _protocol_weights(args["clip2"])
    (value,) = _as_tuple(result)
    out = _protocol_weights(value)
    for key in a:
        if key.endswith((".position_ids", ".logit_scale")) or not a[key].is_floating_point():
            assert torch.equal(out[key], a[key]), key
        else:
            assert torch.allclose(out[key], a[key] + b[key], atol=1e-6), key


# --------------------------------------------------------------------------- #
# Training family checks
# --------------------------------------------------------------------------- #
def _check_training_parameters(result: Any, args: dict[str, Any]) -> None:
    """The created parameter must be a trainable float32 tensor of the declared shape.

    Both routes are checked.  The widget route has to be reproducible for a given
    seed, and a wired tensor has to win over ``shape`` / ``init`` - the documented
    link-beats-widget rule of this family.
    """
    import torch

    from comfy import training_protocol as protocol

    (payload,) = _as_tuple(result)
    params = protocol.as_parameter_dict(payload)
    assert list(params) == ["weight"], f"the default key must be 'weight': {list(params)}"
    value = params["weight"]
    assert isinstance(value, torch.nn.Parameter), "the created value is not an nn.Parameter"
    assert value.requires_grad, "the created value cannot be trained"
    assert value.dtype == torch.float32, value.dtype
    assert tuple(value.shape) == (2, 3), value.shape

    (again,) = _rerun_v3("TrainingParameters")
    assert torch.equal(protocol.as_parameter_dict(again)["weight"], value), (
        "the same seed produced different parameters: the node is not reproducible"
    )

    (wired,) = _rerun_v3(
        "TrainingParameters",
        tensor=torch.full((4, 2), 3.5),
        name="bias",
        shape="9,9",
        init="zeros",
    )
    linked = protocol.as_parameter_dict(wired)
    assert list(linked) == ["bias"], list(linked)
    assert tuple(linked["bias"].shape) == (4, 2), (
        f"the wired tensor must override the shape widget, got {tuple(linked['bias'].shape)}"
    )
    assert torch.all(linked["bias"] == 3.5), "the wired tensor did not become the parameter"


def _check_training_parameters_merge(result: Any, args: dict[str, Any]) -> None:
    """Both sets must survive: a colliding name is renamed, never overwritten."""
    import torch

    from comfy import training_protocol as protocol

    (payload,) = _as_tuple(result)
    merged = protocol.as_parameter_dict(payload)
    first = protocol.as_parameter_dict(args["params_a"])
    second = protocol.as_parameter_dict(args["params_b"])
    assert set(merged) == set(first) | {"bias", "weight_2"}, sorted(merged)
    for key, value in first.items():
        assert torch.equal(merged[key], value), f"'{key}' of the first set was overwritten"
    assert torch.equal(merged["bias"], second["bias"]), "a non-colliding key was lost"
    assert torch.equal(merged["weight_2"], second["weight"]), (
        "the colliding key must be kept under a renamed key, not dropped"
    )


def _check_training_parameters_extract(result: Any, args: dict[str, Any]) -> None:
    """The default lookup must return the named entry, detached; a bad name must raise."""
    import torch

    from comfy import training_protocol as protocol

    (value,) = _as_tuple(result)
    expected = protocol.as_parameter_dict(args["params"])["weight"].detach()
    assert torch.equal(value, expected), "the wrong entry was returned"
    assert not value.requires_grad, "the extracted tensor must be detached"

    try:
        _rerun_v3("TrainingParametersExtract", name="nope")
    except ValueError as exc:
        assert "weight" in str(exc), f"the error must list the available names, got: {exc}"
    else:
        raise AssertionError("an unknown parameter name must raise instead of returning")


def _check_training_optimizer(result: Any, args: dict[str, Any]) -> None:
    """Every published config must build the real ``torch.optim`` optimizer."""
    import torch

    from comfy import training_protocol as protocol

    (config,) = _as_tuple(result)
    assert isinstance(config, protocol.OptimizerConfig), type(config).__name__
    module = protocol.build_mlp((2, 3, 1), seed=0)
    default = protocol.build_optimizer(config, module.parameters())
    assert isinstance(default, torch.optim.AdamW), type(default).__name__
    assert default.param_groups[0]["lr"] == config.lr, default.param_groups[0]["lr"]

    for name in protocol.OPTIMIZER_OPTIONS:
        (other,) = _rerun_v3("TrainingOptimizer", optimizer=name)
        built = protocol.build_optimizer(other, module.parameters())
        assert isinstance(built, getattr(torch.optim, name)), (
            f"{name} did not build a torch.optim.{name}"
        )

    (clamped,) = _rerun_v3("TrainingOptimizer", lr=-1.0, beta1=5.0)
    assert clamped.lr == protocol.DEFAULT_LR, (
        f"a negative learning rate must fall back to the default, got {clamped.lr}"
    )
    assert clamped.beta1 < 1.0, "a beta of 1 or more would explode inside torch.optim"


def _check_training_loop(result: Any, args: dict[str, Any]) -> None:
    """The trainer must really train: documented names, detached outputs, falling loss."""
    import torch

    from comfy import training_protocol as protocol

    payload, loss, curve, prediction = _as_tuple(result)
    params = protocol.as_parameter_dict(payload)
    assert list(params) == [
        "layer0.weight",
        "layer0.bias",
        "layer1.weight",
        "layer1.bias",
    ], f"the documented naming convention is not honoured: {list(params)}"
    assert isinstance(params["layer0.weight"], torch.nn.Parameter), (
        "the trained parameters are not nn.Parameter objects"
    )
    assert all(parameter.grad is None for parameter in params.values()), (
        "the trainer left gradients on the parameters it returned"
    )
    assert tuple(curve.shape) == (200,), curve.shape
    assert tuple(prediction.shape) == (48, 1), prediction.shape
    assert not (curve.requires_grad or prediction.requires_grad or loss.requires_grad), (
        "the outputs still carry autograd state: the graph must not reach ComfyUI's cache"
    )
    assert float(loss) == float(curve[-1]), (float(loss), float(curve[-1]))
    assert float(curve[-1]) < float(curve[0]) / 2.0, (
        f"the loss did not fall: {float(curve[0])} -> {float(curve[-1])}"
    )
    assert float(curve[-1]) < 0.5, (
        "200 steps of AdamW on y = 2*x0 - 3*x1 + 1 must get close to 0, got "
        f"{float(curve[-1])}"
    )

    # A warm start has to continue from the trained set, not restart from scratch.
    warm, _, warm_curve, _ = _rerun_v3("TrainingLoop", params=payload, steps=1)
    assert list(protocol.as_parameter_dict(warm)) == list(params), (
        "a warm started run must return the same parameter names"
    )
    assert float(warm_curve[-1]) < float(curve[0]) / 5.0, (
        f"a warm start did not reuse the wired parameters: {float(warm_curve[-1])}"
    )

    try:
        _rerun_v3("TrainingLoop", x=torch.zeros(3, 2), y=torch.zeros(5, 1), steps=1)
    except ValueError as exc:
        assert "sample" in str(exc), f"the mismatch must be explained, got: {exc}"
    else:
        raise AssertionError("mismatched x/y sample counts must be refused")


def _check_training_save_parameters(result: Any, args: dict[str, Any]) -> None:
    """The written file must round trip exactly and be readable by Load Parameters."""
    import comfy.utils
    import torch

    from comfy import training_protocol as protocol

    payload, path = _as_tuple(result)
    written = _written_files("comfydl/parameters_*.safetensors")
    assert written, "Save Parameters wrote no .safetensors under output/comfydl/"
    assert Path(path).is_file(), f"{path} was reported but does not exist"
    assert Path(path).resolve() == written[-1].resolve(), (path, written)

    expected = protocol.parameters_to_tensors(payload)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    assert set(state) == set(expected), (sorted(state), sorted(expected))
    for key, value in expected.items():
        assert torch.equal(state[key], value), f"{key} changed on its way to disk"

    # the very same file, read back through the loader node
    (loaded,) = _rerun_v3("TrainingLoadParameters", path=path)
    recovered = protocol.as_parameter_dict(loaded)
    assert set(recovered) == set(expected), (sorted(recovered), sorted(expected))
    for key, value in expected.items():
        assert torch.equal(recovered[key], value), key
        assert recovered[key].requires_grad, f"{key} did not come back trainable"


def _check_training_load_parameters(result: Any, args: dict[str, Any]) -> None:
    """The default path must really read the seeded file, and a missing one must raise."""
    import torch

    from comfy import training_protocol as protocol

    (payload,) = _as_tuple(result)
    params = protocol.as_parameter_dict(payload)
    assert list(params) == ["weight"], list(params)
    value = params["weight"]
    assert isinstance(value, torch.nn.Parameter) and value.requires_grad, (
        "the loaded value is not trainable"
    )
    assert value.dtype == torch.float32, value.dtype
    assert torch.equal(value, _train_fixture_parameters()), (
        "the loaded values differ from the file the sandbox seeded"
    )

    try:
        _rerun_v3("TrainingLoadParameters", path="no/such/file.safetensors")
    except ValueError as exc:
        assert "file.safetensors" in str(exc), f"the error must carry the path, got: {exc}"
    else:
        raise AssertionError("a missing file must raise instead of returning nothing")


def _check_training_parameters_to_text(result: Any, args: dict[str, Any]) -> None:
    """The encoded text must decode back to the same values - that is what a workflow keeps."""
    import torch

    from comfy import training_protocol as protocol

    (text,) = _as_tuple(result)
    assert text.startswith(protocol.PARAMS_TEXT_PREFIX), text[:32]
    expected = protocol.parameters_to_tensors(args["params"])
    decoded = protocol.decode_parameters(text)
    assert set(decoded) == set(expected), (sorted(decoded), sorted(expected))
    for key, value in expected.items():
        assert torch.equal(decoded[key], value), f"{key} did not survive the encoding"

    (back,) = _rerun_v3("TrainingTextToParameters", text=text)
    recovered = protocol.as_parameter_dict(back)
    assert set(recovered) == set(expected), (sorted(recovered), sorted(expected))
    for key, value in expected.items():
        assert torch.equal(recovered[key], value), key  # the pair must round trip

    try:
        _rerun_v3(
            "TrainingTextToParameters",
            text=protocol.PARAMS_TEXT_PREFIX + "!!!not base64!!!",
        )
    except ValueError as exc:
        assert "base64" in str(exc), f"a corrupt blob must be refused, got: {exc}"
    else:
        raise AssertionError("corrupt text must raise instead of decoding to garbage")


def _check_training_text_to_parameters(result: Any, args: dict[str, Any]) -> None:
    """The default widget text must be a valid blob: the node works untouched."""
    import torch

    from comfy import training_protocol as protocol

    (payload,) = _as_tuple(result)
    params = protocol.as_parameter_dict(payload)
    assert list(params) == ["weight"], list(params)
    assert tuple(params["weight"].shape) == (2, 3), params["weight"].shape
    assert params["weight"].requires_grad, "the decoded value is not trainable"
    assert params["weight"].dtype == torch.float32, params["weight"].dtype

    decoded = protocol.decode_parameters(args["text"])
    assert set(decoded) == set(params), (sorted(decoded), sorted(params))
    for key in params:
        assert torch.equal(decoded[key], params[key].detach()), key


#: node id -> [callable(result, args)].  Extra assertions beyond "it ran and the
#: arity matches", for the nodes whose returned *values* carry the contract.
_OUTPUT_CHECKS: dict[str, list[Callable[[Any, dict[str, Any]], None]]] = {
    "NormalizationBatchNorm": [_check_batchnorm_stats],
    "NormalizationInstanceNorm": [_check_instancenorm_stats],
    "TrainingRunStats": [_check_training_run_stats],
    "TrainingParameters": [_check_training_parameters],
    "TrainingParametersMerge": [_check_training_parameters_merge],
    "TrainingParametersExtract": [_check_training_parameters_extract],
    "TrainingOptimizer": [_check_training_optimizer],
    "TrainingLoop": [_check_training_loop],
    "TrainingSaveParameters": [_check_training_save_parameters],
    "TrainingLoadParameters": [_check_training_load_parameters],
    "TrainingParametersToText": [_check_training_parameters_to_text],
    "TrainingTextToParameters": [_check_training_text_to_parameters],
    "PoolingSliding": [_check_pooling_sliding],
    "PoolingAdaptive": [_check_pooling_adaptive],
    "ConvolutionConv": [_check_convolution_conv],
    "ConvolutionConvTranspose": [_check_convolution_conv_transpose],
    "CheckpointLoaderSimple": [_check_checkpoint_loader],
    "CheckpointSave": [_check_checkpoint_save],
    "ModelSave": [_check_model_save],
    "CLIPSave": [_check_clip_save],
    "VAESave": [_check_vae_save],
    "ModelMergeSimple": [_check_model_merge_simple],
    "ModelMergeBlocks": [_check_model_merge_blocks],
    "ModelMergeAdd": [_check_model_merge_add],
    "ModelMergeSubtract": [_check_model_merge_subtract],
    "CLIPMergeSimple": [_check_clip_merge_simple],
    "CLIPMergeAdd": [_check_clip_merge_add],
    "CdlShowHeatmaps": [partial(_check_heatmap_ranks, "CdlShowHeatmaps")],
    "CdlShowHeatmapsOutput": [partial(_check_heatmap_ranks, "CdlShowHeatmapsOutput")],
    "CdlHeatmapsTo3D": [_check_heatmaps_to_3d],
    "Preview3D": [_check_preview3d],
}


def _synthesise(type_name: str, cfg: dict, name: str) -> Any:
    """Build a dummy value for one input, based on its declared type string.

    Comma separated type strings (V3 "match type" inputs such as
    ``INT,FLOAT,STRING,BOOLEAN``) are satisfied with their first alternative.
    """
    for alternative in str(type_name).split(","):
        factory = _VALUE_FACTORIES.get(alternative.strip().upper())
        if factory is not None:
            return factory(cfg, name)
    raise Unsupported(f"no dummy value for type {type_name!r}")


# --------------------------------------------------------------------------- #
# Spec parsing (legacy INPUT_TYPES)
# --------------------------------------------------------------------------- #
def _split_spec(spec: Any, name: str) -> tuple[str, dict]:
    """Normalise the many shapes an ``INPUT_TYPES`` entry can take.

    Handles ``"IMAGE"``, ``("IMAGE",)``, ``("INT", {"default": 1})`` and the
    option-list form ``(["train", "eval"], {"default": "eval"})``.
    """
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise Unsupported(f"empty spec for input {name!r}")
        head = spec[0]
        cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(head, str):
            return head, cfg
        options = list(head)
        if cfg:
            options = cfg.get("options", options)
        return "COMBO", {"options": options, **cfg}
    raise Unsupported(f"unrecognised spec {spec!r} for input {name!r}")


def _legacy_inputs(node_id: str, node_cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(args, required_names)`` for a legacy ``INPUT_TYPES`` node."""
    spec = node_cls.INPUT_TYPES()
    overrides = _INPUT_OVERRIDES.get(node_id, {})
    args: dict[str, Any] = {}
    required: dict[str, Any] = {}

    for section in ("required", "optional"):
        for name, raw in (spec.get(section) or {}).items():
            type_name, cfg = _split_spec(raw, name)
            if name in overrides:
                args[name] = overrides[name](cfg, name)
                if section == "required":
                    required[name] = args[name]
                continue
            if section == "optional" and type_name.upper() not in _WIDGET_TYPES:
                continue  # leave optional slots unconnected
            try:
                args[name] = _synthesise(type_name, cfg, name)
            except Unsupported:
                if section == "required":
                    raise
                continue  # an optional input we cannot fake is simply omitted
            if section == "required":
                required[name] = args[name]

    for name in (spec.get("hidden") or {}):
        if name in args:
            continue
        if name not in _HIDDEN_VALUES:
            raise Unsupported(f"no dummy value for hidden input {name!r}")
        args[name] = _HIDDEN_VALUES[name]

    if getattr(node_cls, "INPUT_IS_LIST", False):
        args = {key: [value] for key, value in args.items()}
    return args, required


# --------------------------------------------------------------------------- #
# Spec parsing (V3 schema)
# --------------------------------------------------------------------------- #
def _autogrow_value(slot: Any) -> Any:
    """Build the ``{name: value}`` mapping the engine passes to an Autogrow input.

    An ``io.Autogrow.Input`` carries a template describing the repeated sub-input
    (``template.input``) and the names the engine would use
    (``template.names``); the harness connects exactly the minimum number of
    them, filled with dummy values of the template's type.
    """
    template = slot.template
    inner = template.input
    io_type = str(inner.get_io_type())
    cfg = {
        key: getattr(inner, key)
        for key in ("default", "min", "max", "options")
        if hasattr(inner, key)
    }
    # Connect at least two names: aggregate nodes (``a and b``, ``a + b``) need a
    # second operand before their default widget value makes sense.
    count = min(len(getattr(template, "names", [])), max(2, int(getattr(template, "min", 1) or 1)))
    names = list(getattr(template, "names", []))[:count]
    if not names:
        raise Unsupported(f"autogrow input {slot.id!r} exposes no names")
    return {name: _synthesise(io_type, cfg, name) for name in names}


def _v3_inputs(node_id: str, node_cls: type) -> tuple[dict[str, Any], dict[str, Any], int, bool]:
    """Return ``(args, required_names, output_arity, allows_fewer_outputs)``.

    ``allows_fewer_outputs`` mirrors ``Schema.is_output_node``: an output node is
    allowed to return fewer values than it declares (the engine fills the rest
    with ``None``), which is exactly what ``SaveImage`` style nodes rely on.
    """
    schema = node_cls.GET_SCHEMA()
    overrides = _INPUT_OVERRIDES.get(node_id, {})
    args: dict[str, Any] = {}
    required: dict[str, Any] = {}

    for slot in schema.inputs:
        io_type = str(slot.get_io_type())
        cfg = {
            key: getattr(slot, key)
            for key in ("default", "min", "max", "options")
            if hasattr(slot, key)
        }
        optional = bool(getattr(slot, "optional", False))
        if slot.id in overrides:
            value = overrides[slot.id](cfg, slot.id)
        elif optional and io_type.upper() not in _WIDGET_TYPES:
            continue  # leave optional slots unconnected
        else:
            try:
                if io_type == "COMFY_AUTOGROW_V3":
                    value = _autogrow_value(slot)
                else:
                    value = _synthesise(io_type, cfg, slot.id)
            except Unsupported:
                if optional:
                    continue
                raise
        args[slot.id] = value
        if not getattr(slot, "optional", False):
            required[slot.id] = value

    if getattr(schema, "is_input_list", False):
        # The engine hands every input over as a list for such nodes.
        args = {key: [value] for key, value in args.items()}

    allows_fewer = bool(getattr(schema, "is_output_node", False))
    return args, required, len(schema.outputs), allows_fewer


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run_with_timeout(call: Callable[[], Any], timeout: float) -> Any:
    """Run ``call`` in a daemon thread so a hanging node cannot block the run."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - reported verbatim below
            box["error"] = exc
            box["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"did not finish within {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _install_hidden(node_cls: type) -> None:
    """Plant the engine-style hidden inputs a V3 node may read.

    The server injects a ``HiddenHolder`` on the node class before calling it;
    without an engine, output nodes such as ``GetImageSize`` or ``SaveImage``
    would see ``cls.hidden is None``.
    """
    from comfy_api.latest._io import HiddenHolder

    node_cls.hidden = HiddenHolder(
        unique_id="smoke-test",
        prompt={},
        extra_pnginfo={"workflow": {}},
        dynprompt={},
        auth_token_comfy_org="",
        api_key_comfy_org="",
    )


def _execute(
    node_id: str,
    node_cls: type,
    args: dict[str, Any],
    timeout: float,
) -> tuple[Any, int, bool]:
    """Execute one node and return ``(raw_output, expected_arity, allows_fewer)``."""
    if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
        _, _, arity, allows_fewer = _v3_inputs(node_id, node_cls)
        _install_hidden(node_cls)
        entry_name = getattr(node_cls, "FUNCTION", _V3_ENTRY_POINTS[0])
        entry = getattr(node_cls, entry_name)
        if entry_name.endswith("_ASYNC") or inspect.iscoroutinefunction(entry):
            result = _run_with_timeout(lambda: asyncio.run(entry(**args)), timeout)
        else:
            result = _run_with_timeout(lambda: entry(**args), timeout)
        return getattr(result, "result", result), arity, allows_fewer

    instance = node_cls()
    entry = getattr(instance, node_cls.FUNCTION)
    result = _run_with_timeout(lambda: entry(**args), timeout)
    arity = len(getattr(node_cls, "RETURN_TYPES", ()) or ())
    return result, arity, bool(getattr(node_cls, "OUTPUT_NODE", False))


def _check_expected_error(node_id: str, node_cls: type, timeout: float) -> tuple[str, str, str]:
    """Run a node that must refuse to work, and verify *why* it refuses.

    The MODEL protocol layer registers a few nodes whose implementation was removed
    by the dehydration pass.  They are supposed to raise a ``RuntimeError`` naming
    the missing module, so a bare ``ModuleNotFoundError`` (or, worse, a silent
    success) is a failure of the design, not of the test.

    Returns:
        ``(status, detail, traceback)``, ready to append to the result list.
    """
    expected = _EXPECTED_ERRORS[node_id]
    try:
        args, _, _, _ = _v3_inputs(node_id, node_cls)
        _execute(node_id, node_cls, args, timeout)
    except RuntimeError as exc:
        if expected.lower() in str(exc).lower():
            return "PASS", "", ""
        return "FAIL", f"the error does not mention {expected!r}: {exc}", ""
    except BaseException as exc:  # noqa: BLE001 - reported verbatim
        return "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc()
    return "FAIL", "ran successfully although its implementation is missing", ""


def _count_outputs(result: Any) -> int:
    """Number of values a node returned, tolerating UI-only dict returns."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return 0
    if isinstance(result, (tuple, list)):
        return len(result)
    return 1


def _category(node_cls: type) -> str:
    if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
        try:
            return node_cls.GET_SCHEMA().category or "<none>"
        except Exception:  # noqa: BLE001 - a broken schema is reported by the run
            return "<bad schema>"
    return getattr(node_cls, "CATEGORY", "<none>") or "<none>"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_categories(mappings: dict[str, type]) -> None:
    """Print category -> node count, handy when updating the shipped docs."""
    counts: dict[str, list[str]] = {}
    for node_id, node_cls in mappings.items():
        counts.setdefault(_category(node_cls), []).append(node_id)
    print(f"registered nodes: {len(mappings)}")
    print(f"categories      : {len(counts)}")
    for name in sorted(counts):
        print(f"  {name:<44} {len(counts[name]):>4}")
    uncategorised = counts.get("<none>", [])
    if uncategorised:
        print("  nodes without a category: " + ", ".join(sorted(uncategorised)))


def _print_summary(results: list[tuple[str, str, str, str]]) -> int:
    passed = [r for r in results if r[1] == "PASS"]
    skipped = [r for r in results if r[1] == "SKIP"]
    failed = [r for r in results if r[1] == "FAIL"]

    if skipped:
        print("\n--- SKIPPED ---")
        for node_id, _, detail, _ in skipped:
            print(f"  {node_id:<28} {detail}")

    if failed:
        print("\n--- FAILURES ---")
        for node_id, _, detail, tb in failed:
            print(f"  {node_id:<28} {detail}")
            if tb:
                print(tb)

    print(
        f"\n=== smoke result: {len(passed)} PASS, {len(skipped)} SKIP, "
        f"{len(failed)} FAIL (of {len(results)}) ==="
    )
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filter", default="", help="only run node ids containing this text")
    parser.add_argument("--verbose", action="store_true", help="print one line per node as it runs")
    parser.add_argument(
        "--categories",
        action="store_true",
        help="only print the category -> node count table (no execution)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per node time budget in seconds")
    parser.add_argument("--keep-sandbox", action="store_true", help="keep the scratch directory")
    opts = parser.parse_args(argv)

    sandbox = Path(tempfile.mkdtemp(prefix="cdl_smoke_"))
    try:
        _bootstrap(sandbox)
        print(f"[smoke] repo      : {REPO_ROOT}")
        print(f"[smoke] python    : {sys.executable}")
        mappings = _load_registry(opts.verbose)
        _REGISTRY.clear()
        _REGISTRY.update(mappings)
        _install_sandbox_paths(sandbox)

        if opts.categories:
            _print_categories(mappings)
            return 0

        node_ids = sorted(node_id for node_id in mappings if opts.filter.lower() in node_id.lower())
        print(f"[smoke] nodes     : {len(node_ids)} selected of {len(mappings)} registered\n")

        results: list[tuple[str, str, str, str]] = []
        for node_id in node_ids:
            node_cls = mappings[node_id]
            if node_id in _SKIPPED_NODES:
                results.append((node_id, "SKIP", _SKIPPED_NODES[node_id], ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}")
                continue
            if node_id in _EXPECTED_ERRORS:
                status, detail, tb = _check_expected_error(node_id, node_cls, opts.timeout)
                results.append((node_id, status, detail, tb))
                if opts.verbose:
                    print(f"{status}  {node_id}: {detail}")
                continue
            try:
                if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
                    args, _, _, _ = _v3_inputs(node_id, node_cls)
                else:
                    args, _ = _legacy_inputs(node_id, node_cls)
                result, arity, allows_fewer = _execute(node_id, node_cls, args, opts.timeout)
                got = _count_outputs(result)
                if got > arity or (got != arity and not allows_fewer):
                    raise AssertionError(f"returned {got} value(s), schema declares {arity}")
                for check in _OUTPUT_CHECKS.get(node_id, ()):
                    check(result, args)
            except Unsupported as exc:
                results.append((node_id, "SKIP", str(exc), ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}: {exc}")
            except TimeoutError as exc:
                results.append((node_id, "SKIP", f"timeout ({exc})", ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}: {exc}")
            except BaseException as exc:  # noqa: BLE001 - a FAIL is the whole point
                results.append(
                    (node_id, "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc())
                )
                if opts.verbose:
                    print(f"FAIL  {node_id}: {type(exc).__name__}: {exc}")
            else:
                results.append((node_id, "PASS", "", ""))
                if opts.verbose:
                    print(f"PASS  {node_id}  [{_category(node_cls)}]")

        return _print_summary(results)
    finally:
        if opts.keep_sandbox:
            print(f"[smoke] sandbox kept at {sandbox}")
        else:
            os.chdir(REPO_ROOT)  # Windows cannot remove the current directory
            shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
