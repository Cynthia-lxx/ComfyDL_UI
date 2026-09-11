"""Checkpoint / model loader nodes (reform step 5: MODEL protocol layer).

The dehydrated build has no architecture code, so nothing here can build a
*runnable* diffusion model. What it can do - and what these nodes do - is turn a
weight file into the ``MODEL`` / ``CLIP`` / ``VAE`` **values** the rest of the
node graph expects, grouped purely by key prefix (no architecture detection):

* ``Load Checkpoint``  - one file, split into ``diffusion_model.*`` (MODEL),
  ``first_stage_model.*`` (VAE) and ``cond_stage_model.*`` (CLIP).
* ``Load UNET``        - ``diffusion_models`` folder, whole file is one MODEL.
* ``Load VAE``         - ``vae`` folder, whole file is one VAE.
* ``Load CLIP``        - ``text_encoders`` folder, whole file is one CLIP.
* ``Load Dual CLIP``   - two text-encoder files merged into one CLIP container.

All of the above really read the file from disk, and their outputs can really be
merged and saved again by the nodes in :mod:`nodes_model_merging`, so a workflow
built from them produces a real ``.safetensors`` file.

Two nodes are registered but deliberately not executable in this build:

* ``Load LoRA`` / ``Load LoRA (Model Only)`` keep the native IO contract so a
  workflow can be wired up, but applying a LoRA requires mapping LoRA keys onto
  model keys - i.e. exactly the architecture recognition that was dehydrated.
  They raise a ``RuntimeError`` that says so and names the module to restore.

The ``prefix_strip`` widget controls how the outermost container prefix of a
file is handled. ``auto`` detects and strips a wrapper such as ``model.`` (so an
SD1.5 checkpoint's ``model.diffusion_model.*`` keys become ``diffusion_model.*``
and can be grouped and merged); ``raw`` strips nothing; any other value is used
as a literal prefix. Whatever is stripped is remembered per key and replayed on
save, so "load then save" round-trips the key set verbatim.
"""

from __future__ import annotations

from typing import Mapping

import torch
from typing_extensions import override

import comfy.model_protocol as protocol
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

CATEGORY = "model/loaders"

#: Widget tooltip shared by the five loader nodes.
_PREFIX_STRIP_TOOLTIP = (
    "How to handle the outer container prefix of the file: 'auto' detects and "
    "strips a wrapper (e.g. 'model.'), 'raw' keeps every key as-is, or type an "
    "explicit prefix such as 'model.'."
)

#: The LoRA nodes are registered with the native contract but cannot run here.
_LORA_HINT = (
    "This build is a dehydrated ComfyUI: applying a LoRA requires mapping LoRA "
    "keys onto model keys, i.e. model-structure recognition, and the module that "
    "implements it (comfy/lora.py, plus the model classes it patches) was removed. "
    "The node is registered so a workflow can be wired up; restore comfy/lora.py "
    "from the read-only reference checkout 'ComfyUI-original/' (see "
    "docs/dehydrate_manifest.md) to make it executable again."
)


def _loader_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    outputs: list,
    search_aliases: list[str] | None = None,
) -> io.Schema:
    """Build the schema shared by the loader nodes.

    Args:
        node_id: Globally unique, core-safe node id (native id, so a workflow
            written here stays compatible with upstream ComfyUI).
        display_name: Name shown in the node library.
        description: Tooltip shown when hovering over the node.
        inputs: The node's inputs, in declaration order (slots and widgets).
        outputs: The node's outputs, in declaration order.
        search_aliases: Extra search keywords for the node library.

    Returns:
        The ``io.Schema`` describing the node.
    """
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category=CATEGORY,
        description=description,
        search_aliases=search_aliases,
        inputs=list(inputs),
        outputs=list(outputs),
    )


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[model/loaders] {message}")


def _file_widget(folder: str, widget_id: str, tooltip: str) -> io.Combo.Input:
    """Build a file dropdown over one of ComfyUI's model folders.

    The options are read from ``folder_paths`` when the schema is built, which is
    what the server does every time it serves ``/object_info``; the default is the
    first entry so the widget is usable without touching it. An empty folder
    yields an empty option list and an empty default - the node is then simply
    unusable until a file is added, which is the documented behaviour.

    Args:
        folder: ``folder_paths`` folder name (``"checkpoints"``, ``"loras"``, ...).
        widget_id: Widget id, e.g. ``"ckpt_name"``.
        tooltip: Tooltip text.

    Returns:
        The configured ``io.Combo.Input``.
    """
    names = list(folder_paths.get_filename_list(folder))
    return io.Combo.Input(
        widget_id,
        options=names,
        default=names[0] if names else "",
        tooltip=tooltip,
    )


def _prefix_strip_widget() -> io.String.Input:
    """Build the shared ``prefix_strip`` widget."""
    return io.String.Input("prefix_strip", default="auto", tooltip=_PREFIX_STRIP_TOOLTIP)


def _load(filename: str, folder: str) -> dict[str, torch.Tensor]:
    """Read one weight file from a ``folder_paths`` folder.

    Args:
        filename: Relative name as shown in the dropdown.
        folder: ``folder_paths`` folder name.

    Returns:
        The raw, flat ``{key: tensor}`` mapping (non tensor entries are dropped
        later by :func:`comfy.model_protocol.split_checkpoint`).

    Raises:
        FileNotFoundError: When the file is gone (ComfyUI renders this as a
            normal node error naming the folder and file).
    """
    path = folder_paths.get_full_path_or_raise(folder, filename)
    return comfy.utils.load_torch_file(path, safe_load=True)


def _subset_prefixes(
    key_prefixes: Mapping[str, str], sd: Mapping[str, torch.Tensor]
) -> dict[str, str]:
    """Keep only the prefix entries that belong to one bucket.

    Args:
        key_prefixes: The full map returned by ``split_checkpoint``.
        sd: The bucket whose prefixes are wanted.

    Returns:
        A map covering exactly the keys of ``sd``.
    """
    return {key: key_prefixes[key] for key in sd if key in key_prefixes}


class CheckpointLoaderSimple(io.ComfyNode):
    """Load a checkpoint and split it into MODEL, CLIP and VAE by key prefix.

    What: reads a ``.safetensors`` / ``.ckpt`` file and splits its keys into the
          three buckets a checkpoint normally holds, using documented key
          prefixes only (``diffusion_model.`` -> MODEL, ``first_stage_model.``
          -> VAE, ``cond_stage_model.`` / ``conditioner.`` / ``text_encoders.``
          -> CLIP). No architecture detection happens: the buckets are complete
          but are not, on their own, a runnable model. The MODEL bucket is
          wrapped in a ``ModelPatcher`` so it is a proper ``MODEL`` value, and
          the weights stay in memory by reference (no copy).
          A file with no key for a bucket still produces an empty container of
          that type, so downstream links never break; a one-line summary of each
          bucket is printed to the console.
    In:   ckpt_name (COMBO) - file from the ``models/checkpoints`` folder.
          prefix_strip (STRING) - default ``auto``; see the module docstring.
    Out: MODEL - the ``diffusion_model.*`` (plus every unmatched key) bucket.
         CLIP  - the ``cond_stage_model.*`` / ``conditioner.*`` bucket.
         VAE   - the ``first_stage_model.*`` bucket.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "CheckpointLoaderSimple",
            "Load Checkpoint",
            "Load a checkpoint file and split it into MODEL / CLIP / VAE by weight key prefix (state_dict level, no architecture detection).",
            inputs=[
                _file_widget("checkpoints", "ckpt_name", "Checkpoint file from models/checkpoints."),
                _prefix_strip_widget(),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.Clip.Output(display_name="CLIP"),
                io.Vae.Output(display_name="VAE"),
            ],
            search_aliases=[
                "load checkpoint", "checkpoint", "ckpt", "model loader",
                "safetensors", "load model",
            ],
        )

    @classmethod
    def execute(
        cls, ckpt_name: str, prefix_strip: str = "auto"
    ) -> io.NodeOutput:
        sd = _load(ckpt_name, "checkpoints")
        model_sd, clip_sd, vae_sd, prefixes = protocol.split_checkpoint(sd, prefix_strip)
        model = protocol.make_model_patcher(model_sd, _subset_prefixes(prefixes, model_sd))
        clip = protocol.make_container(clip_sd, _subset_prefixes(prefixes, clip_sd))
        vae = protocol.make_container(vae_sd, _subset_prefixes(prefixes, vae_sd))
        print(
            f"[model/loaders] Load Checkpoint {ckpt_name!r}: MODEL {len(model_sd)} key(s), "
            f"CLIP {len(clip_sd)} key(s), VAE {len(vae_sd)} key(s)."
        )
        return io.NodeOutput(model, clip, vae)


class UNETLoader(io.ComfyNode):
    """Load a diffusion-model-only weight file as a MODEL.

    What: reads a file from the ``models/unet`` / ``models/diffusion_models``
          folders and treats the whole file as the diffusion model. Files like
          this usually carry no ``diffusion_model.`` prefix at all (their keys
          start at ``double_blocks.`` or ``input_blocks.``), which is exactly why
          the MODEL bucket is "the whole file" rather than a prefix filter.
    In:   unet_name (COMBO) - file from ``models/unet`` or ``models/diffusion_models``.
          prefix_strip (STRING) - default ``auto``; see the module docstring.
    Out: MODEL - the file's weights wrapped in a ``ModelPatcher``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "UNETLoader",
            "Load Diffusion Model",
            "Load a diffusion-model-only weight file (models/unet or models/diffusion_models) as a MODEL value.",
            inputs=[
                _file_widget("diffusion_models", "unet_name", "Weight file from models/unet or models/diffusion_models."),
                _prefix_strip_widget(),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=[
                "load unet", "unet", "diffusion model", "load diffusion model",
                "dit", "safetensors",
            ],
        )

    @classmethod
    def execute(cls, unet_name: str, prefix_strip: str = "auto") -> io.NodeOutput:
        sd = _load(unet_name, "diffusion_models")
        stripped, prefixes = protocol.strip_outer_prefix(sd, prefix_strip)
        model = protocol.make_model_patcher(stripped, prefixes)
        print(f"[model/loaders] Load UNET {unet_name!r}: {len(stripped)} key(s).")
        return io.NodeOutput(model)


class VAELoader(io.ComfyNode):
    """Load a VAE weight file as a VAE value.

    What: reads a file from the ``models/vae`` folder and keeps the whole file as
          one VAE container; keys are preserved verbatim (after the optional
          container-prefix strip) so the file can be written back unchanged.
    In:   vae_name (COMBO) - file from ``models/vae``.
          prefix_strip (STRING) - default ``auto``; see the module docstring.
    Out: VAE - the file's weights.
    Note: this node only carries the weights. Encoding/decoding an image needs
          the autoencoder architecture code, which was dehydrated, so the
          ``VAE Encode`` / ``VAE Decode`` nodes report that instead of running.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "VAELoader",
            "Load VAE",
            "Load a VAE weight file from models/vae as a VAE value (state_dict level only; no architecture detection).",
            inputs=[
                _file_widget("vae", "vae_name", "VAE weight file from models/vae."),
                _prefix_strip_widget(),
            ],
            outputs=[io.Vae.Output(display_name="VAE")],
            search_aliases=["load vae", "vae", "first stage model", "autoencoder"],
        )

    @classmethod
    def execute(cls, vae_name: str, prefix_strip: str = "auto") -> io.NodeOutput:
        sd = _load(vae_name, "vae")
        stripped, prefixes = protocol.strip_outer_prefix(sd, prefix_strip)
        vae = protocol.make_container(stripped, prefixes)
        print(f"[model/loaders] Load VAE {vae_name!r}: {len(stripped)} key(s).")
        return io.NodeOutput(vae)


class CLIPLoader(io.ComfyNode):
    """Load a text-encoder weight file as a CLIP value.

    What: reads a file from ``models/text_encoders`` (the legacy ``models/clip``
          folder is searched too) and keeps the whole file as one CLIP container.
    In:   clip_name (COMBO) - file from ``models/text_encoders``.
          prefix_strip (STRING) - default ``auto``; see the module docstring.
    Out: CLIP - the file's weights.
    Note: text encoding itself needs the encoder architecture, which was
          dehydrated, so ``CLIP Text Encode`` reports that instead of running.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "CLIPLoader",
            "Load CLIP",
            "Load a text encoder weight file from models/text_encoders as a CLIP value (state_dict level only).",
            inputs=[
                _file_widget("text_encoders", "clip_name", "Text encoder file from models/text_encoders."),
                _prefix_strip_widget(),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
            search_aliases=[
                "load clip", "clip", "text encoder", "text encoders", "conditioner",
                "t5", "clip-l", "clip-g",
            ],
        )

    @classmethod
    def execute(cls, clip_name: str, prefix_strip: str = "auto") -> io.NodeOutput:
        sd = _load(clip_name, "text_encoders")
        stripped, prefixes = protocol.strip_outer_prefix(sd, prefix_strip)
        clip = protocol.make_container(stripped, prefixes)
        print(f"[model/loaders] Load CLIP {clip_name!r}: {len(stripped)} key(s).")
        return io.NodeOutput(clip)


class DualCLIPLoader(io.ComfyNode):
    """Load two text encoders and merge them into a single CLIP value.

    What: the two-tower setup used by SDXL / SD3 / Flux style models, where a
          second encoder (CLIP-G, T5, ...) is loaded alongside the first. The two
          files are read, prefix-normalised and then joined into one container so
          downstream nodes only see a single CLIP. Duplicate keys cannot normally
          happen (a key would have to exist in both files); when it does, the
          second file wins and a warning is printed.
    In:   clip_name1 (COMBO) - first encoder, ``models/text_encoders``.
          clip_name2 (COMBO) - second encoder, ``models/text_encoders``.
          prefix_strip (STRING) - default ``auto``; see the module docstring.
    Out: CLIP - the union of both files' weights.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "DualCLIPLoader",
            "Load CLIP (Dual)",
            "Load two text encoders (e.g. CLIP-L + T5) and merge them into one CLIP value.",
            inputs=[
                _file_widget("text_encoders", "clip_name1", "First text encoder from models/text_encoders."),
                _file_widget("text_encoders", "clip_name2", "Second text encoder from models/text_encoders."),
                _prefix_strip_widget(),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
            search_aliases=[
                "dual clip", "two text encoders", "sdxl clip", "t5 + clip",
                "clip-l", "clip-g", "load two clip",
            ],
        )

    @classmethod
    def execute(
        cls, clip_name1: str, clip_name2: str, prefix_strip: str = "auto"
    ) -> io.NodeOutput:
        first, first_prefixes = protocol.strip_outer_prefix(
            _load(clip_name1, "text_encoders"), prefix_strip
        )
        second, second_prefixes = protocol.strip_outer_prefix(
            _load(clip_name2, "text_encoders"), prefix_strip
        )
        collisions = sorted(set(first) & set(second))
        if collisions:
            _warn(
                f"Load Dual CLIP: {len(collisions)} key(s) exist in both files "
                f"(first: {collisions[0]!r}); the second file wins."
            )
        merged = {**first, **second}
        prefixes = {**first_prefixes, **second_prefixes}
        clip = protocol.make_container(merged, prefixes)
        print(
            f"[model/loaders] Load Dual CLIP {clip_name1!r} + {clip_name2!r}: "
            f"{len(first)} + {len(second)} key(s) -> {len(merged)}."
        )
        return io.NodeOutput(clip)


class LoraLoader(io.ComfyNode):
    """Load a LoRA and apply it to a MODEL and CLIP (registered, not executable).

    What: keeps the native input/output contract so a workflow can be wired up,
          but cannot run in this build. Applying a LoRA means matching LoRA keys
          against model keys - model-structure recognition - and depends on
          ``comfy/lora.py`` and the model classes it patches, all removed by the
          dehydration pass. Executing therefore raises a ``RuntimeError`` that
          names the missing module and how to restore it, instead of failing
          with a bare ``ModuleNotFoundError`` deep inside the engine.
    In:   model (MODEL), clip (CLIP) - the values the LoRA would be added to.
          lora_name (COMBO) - file from ``models/loras``.
          strength_model (FLOAT) - default 1.0.
          strength_clip (FLOAT) - default 1.0.
    Out: MODEL, CLIP - the patched values (in a build that has ``comfy/lora.py``).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "LoraLoader",
            "Load LoRA (Model and CLIP)",
            "Apply a LoRA to a MODEL and CLIP. Registered for wiring; requires comfy/lora.py, which this dehydrated build does not include.",
            inputs=[
                io.Model.Input("model", tooltip="MODEL the LoRA would be added to."),
                io.Clip.Input("clip", tooltip="CLIP the LoRA would be added to."),
                _file_widget("loras", "lora_name", "LoRA file from models/loras."),
                io.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="LoRA strength for the MODEL; 1.0 is the file's own scale.",
                ),
                io.Float.Input(
                    "strength_clip", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="LoRA strength for the CLIP; 1.0 is the file's own scale.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.Clip.Output(display_name="CLIP"),
            ],
            search_aliases=["lora", "load lora", "apply lora", "lora loader"],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
    ) -> io.NodeOutput:
        raise RuntimeError(f"Load LoRA: {_LORA_HINT}")


class LoraLoaderModelOnly(io.ComfyNode):
    """Load a LoRA and apply it to a MODEL only (registered, not executable).

    What: the MODEL-only variant of :class:`LoraLoader`; see its docstring for
          why this build cannot execute it.
    In:   model (MODEL), lora_name (COMBO), strength_model (FLOAT, default 1.0).
    Out: MODEL - the patched model (in a build that has ``comfy/lora.py``).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _loader_schema(
            "LoraLoaderModelOnly",
            "Load LoRA",
            "Apply a LoRA to a MODEL only. Registered for wiring; requires comfy/lora.py, which this dehydrated build does not include.",
            inputs=[
                io.Model.Input("model", tooltip="MODEL the LoRA would be added to."),
                _file_widget("loras", "lora_name", "LoRA file from models/loras."),
                io.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="LoRA strength for the MODEL; 1.0 is the file's own scale.",
                ),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=["lora", "load lora", "lora model only", "unet lora"],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength_model: float = 1.0) -> io.NodeOutput:
        raise RuntimeError(f"Load LoRA (Model Only): {_LORA_HINT}")


LOADER_NODES: list[type[io.ComfyNode]] = [
    CheckpointLoaderSimple,
    UNETLoader,
    VAELoader,
    CLIPLoader,
    DualCLIPLoader,
    LoraLoader,
    LoraLoaderModelOnly,
]


class ModelLoaderExtension(ComfyExtension):
    """Registers the ``model/loaders`` node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(LOADER_NODES)


async def comfy_entrypoint() -> ModelLoaderExtension:
    return ModelLoaderExtension()
