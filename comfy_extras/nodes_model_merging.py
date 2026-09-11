"""Model / CLIP merge and save nodes (reform step 5: MODEL protocol layer).

Eleven nodes in the native ``model/merging`` category, rebuilt at the
**state_dict** level:

Merge (7):
* ``ModelMergeSimple`` / ``ModelMergeBlocks`` / ``ModelMergeAdd`` /
  ``ModelMergeSubtract`` - blend two ``MODEL`` values key by key.
* ``CLIPMergeSimple`` / ``CLIPMergeAdd`` / ``CLIPMergeSubtract`` - the same for
  two ``CLIP`` values, skipping ``.position_ids`` / ``.logit_scale`` exactly
  like the native nodes do.

Save (4):
* ``CheckpointSave`` - MODEL + CLIP + VAE into one ``.safetensors`` file.
* ``ModelSave`` / ``CLIPSave`` / ``VAESave`` - one bucket each.

Why the arithmetic is done here instead of with ``ModelPatcher.add_patches``
-------------------------------------------------------------------------
The native merge nodes clone the first model and attach *patches*; those patches
are only materialised during inference by ``comfy.lora.calculate_weight``, which
this dehydrated build does not ship. Rather than depend on that path, the merge
here computes the blended tensor immediately:

    out[key] = w_base * model1[key] + w_other * model2[key]

which is exactly what a patch with ``(strength_patch=w_other,
strength_model=w_base)`` would have produced. The mapping between the node's
widget and those two weights follows the official documentation: for
``ModelMergeSimple`` ``ratio=1`` keeps 100% of ``model1`` and ``ratio=0`` keeps
100% of ``model2``, i.e. ``(w_base, w_other) = (ratio, 1 - ratio)``.

Keys the second model does not have are copied from the first model untouched
and reported, so a merge between two different architectures degrades to "keep
model1's extra keys" instead of silently dropping weights. The blended weights
are real tensors, which is what makes ``ModelSave`` produce a real file.
"""

from __future__ import annotations

import json
import os

import torch
from typing_extensions import override

import comfy.model_protocol as protocol
import comfy.utils
import folder_paths
from comfy.cli_args import args as cli_args
from comfy_api.latest import ComfyExtension, io

CATEGORY = "model/merging"

#: Native filter prefix for MODEL merges. Keys inside a checkpoint's model bucket
#: carry it; a bare UNET file does not, which ``select_merge_keys`` copes with.
MODEL_PREFIX = "diffusion_model."

#: Suffixes the native CLIP merges leave alone (they are not weights).
CLIP_EXCLUDE_SUFFIXES = (".position_ids", ".logit_scale")

#: Block names ``ModelMergeBlocks`` can weight separately, in native order.
_BLOCK_NAMES = ("input", "middle", "out")


def _merge_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    outputs: list,
    search_aliases: list[str] | None = None,
    is_output_node: bool = False,
) -> io.Schema:
    """Build the schema shared by the merge and save nodes.

    Args:
        node_id: Globally unique node id (the native id, for upstream alignment).
        display_name: Name shown in the node library.
        description: Tooltip shown when hovering over the node.
        inputs: The node's inputs, in declaration order (slots and widgets).
        outputs: The node's outputs, in declaration order (empty for save nodes).
        search_aliases: Extra search keywords for the node library.
        is_output_node: ``True`` for the save nodes, which have side effects.

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
        is_output_node=is_output_node,
    )


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[model/merging] {message}")


def _blend(
    base,
    other,
    blend_fn,
    node_name: str,
    prefix: str,
    exclude_suffixes=(),
    wrap=None,
):
    """Blend two protocol-layer values key by key and wrap the result.

    Args:
        base: The first value (``model1`` / ``clip1``); its keys define the
            result's key set.
        other: The second value (``model2`` / ``clip2``).
        blend_fn: ``key -> (w_base, w_other)`` - the per-key blend weights.
        node_name: Display name used in warnings.
        prefix: Key prefix the merge should be limited to; ``""`` means every key.
        exclude_suffixes: Key suffixes to leave untouched (CLIP merges use this).
        wrap: ``protocol.make_model_patcher`` or ``protocol.make_container``.

    Returns:
        The merged value, of the same kind as ``base``.
    """
    base_sd, base_prefixes = protocol.container_state_dict(base)
    other_sd, _ = protocol.container_state_dict(other)
    keys, mode = protocol.select_merge_keys(
        base_sd.keys(), prefix, fallback_all=bool(prefix)
    )
    if mode == "none":
        _warn(
            f"{node_name}: no key carries the {prefix!r} marker and the bucket has no "
            f"keys to fall back on, so there is nothing to blend; the first input is "
            f"returned unchanged."
        )
        return base
    if mode == "unprefixed":
        _warn(
            f"{node_name}: no key carries the {prefix!r} marker (a bare UNET file), so "
            f"the whole bucket is treated as the diffusion model and blended in full. "
            f"Set prefix_strip='raw' on the loader to keep the original wrapper instead."
        )
    merged, matched, skipped = protocol.merge_state_dicts(
        base_sd, other_sd, blend_fn, keys=keys, exclude_suffixes=exclude_suffixes
    )
    if skipped:
        _warn(
            f"{node_name}: {len(skipped)} key(s) exist in the first input only "
            f"(e.g. {skipped[0]!r}) and were kept as-is."
        )
    print(
        f"[model/merging] {node_name}: blended {matched} key(s) "
        f"of {len(base_sd)} (prefix {prefix!r}, matched as {mode!r})."
    )
    return wrap(merged, base_prefixes)


def _linear_blend(w_base: float, w_other: float):
    """Return a ``blend_fn`` with constant weights for every key."""

    def blend(_key: str) -> tuple[float, float]:
        return float(w_base), float(w_other)

    return blend


def _strip_model_prefix(key: str) -> str:
    """Drop everything up to and including the ``diffusion_model.`` marker."""
    marker = MODEL_PREFIX
    index = key.find(marker)
    return key[index + len(marker):] if index >= 0 else key


def _blocks_blend(input_ratio: float, middle_ratio: float, out_ratio: float):
    """Build the per-key blend function of ``ModelMergeBlocks``.

    Mirrors the native node: a key is assigned the ratio of the *longest*
    matching block name, and ``input`` acts as the default for keys belonging to
    no block at all (``time_embed.*`` and friends).
    """
    ratios = {"input": input_ratio, "middle": middle_ratio, "out": out_ratio}
    default_ratio = input_ratio

    def blend(key: str) -> tuple[float, float]:
        name = _strip_model_prefix(key)
        ratio = default_ratio
        best = 0
        for block in _BLOCK_NAMES:
            if name.startswith(block) and best < len(block):
                ratio = ratios[block]
                best = len(block)
        return float(ratio), 1.0 - float(ratio)

    return blend


def _metadata_for_save(extra: dict | None = None) -> dict[str, str]:
    """Collect the ``safetensors`` metadata a save node should attach.

    ``--disable-metadata`` is honoured exactly like the native save nodes.
    """
    metadata: dict[str, str] = {"format": "pt"}
    if cli_args.disable_metadata:
        return metadata
    if extra:
        metadata.update({key: str(value) for key, value in extra.items()})
    return metadata


def _save_bucket(state_dict, key_prefixes, filename_prefix: str, metadata=None) -> str:
    """Write one bucket to a uniquely named ``.safetensors`` file in ``output/``.

    Args:
        state_dict: Normalised weights.
        key_prefixes: Prefix map from load time, replayed so the file's key set
            matches the source file (or the merge inputs).
        filename_prefix: Widget value; may contain a subfolder and ``%date%``
            style placeholders, resolved by ``folder_paths.get_save_image_path``.
        metadata: Optional ``safetensors`` metadata.

    Returns:
        The absolute path that was written.
    """
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, _subfolder, _prefix = (
        folder_paths.get_save_image_path(filename_prefix, output_dir)
    )
    os.makedirs(full_output_folder, exist_ok=True)
    output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
    comfy.utils.save_torch_file(
        protocol.prepare_for_save(state_dict, key_prefixes), output_path, metadata=metadata
    )
    print(
        f"[model/merging] saved {len(state_dict)} tensor(s) to {output_path}"
    )
    return output_path


class ModelMergeSimple(io.ComfyNode):
    """Blend two models linearly by a single ratio.

    What: the state_dict counterpart of the native patch-based merge, i.e. the
          "A x ratio + B x (1 - ratio)" formula checkpoint fusers are built on.
          Only the diffusion-model keys are blended; keys that exist in one input
          only are kept as they are and reported.
    In:   model1 (MODEL) - the base model; its key set becomes the result's.
          model2 (MODEL) - the model blended into it.
          ratio (FLOAT, default 1.0, 0..1) - 1.0 keeps 100% of model1, 0.0 keeps
          100% of model2, 0.5 splits it evenly.
    Out: MODEL - the blended weights.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "ModelMergeSimple",
            "ModelMergeSimple",
            "Blend two models with a single ratio: ratio=1 keeps model1, ratio=0 keeps model2.",
            inputs=[
                io.Model.Input("model1", tooltip="Base model; its keys define the result."),
                io.Model.Input("model2", tooltip="Model blended into model1."),
                io.Float.Input(
                    "ratio", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 = 100% model1, 0.0 = 100% model2.",
                ),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=["merge models", "model merge", "checkpoint merge", "blend models"],
        )

    @classmethod
    def execute(cls, model1, model2, ratio: float = 1.0) -> io.NodeOutput:
        merged = _blend(
            model1, model2, _linear_blend(ratio, 1.0 - ratio),
            "ModelMergeSimple", MODEL_PREFIX, wrap=protocol.make_model_patcher,
        )
        return io.NodeOutput(merged)


class ModelMergeBlocks(io.ComfyNode):
    """Blend two models with a separate ratio per UNet block group.

    What: like :class:`ModelMergeSimple`, but the ratio comes from the block the
          key belongs to - ``input_blocks``, ``middle_block``, ``output_blocks``.
          Keys outside those groups (``time_embed.*``, ...) use the ``input``
          ratio, matching the native node.
    In:   model1 (MODEL), model2 (MODEL) - as in ``ModelMergeSimple``.
          input (FLOAT, default 1.0) - ratio for ``input_blocks`` (and the default).
          middle (FLOAT, default 1.0) - ratio for ``middle_block``.
          out (FLOAT, default 1.0) - ratio for ``output_blocks``.
    Out: MODEL - the blended weights.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "ModelMergeBlocks",
            "ModelMergeBlocks",
            "Blend two models with one ratio per UNet block group (input / middle / out).",
            inputs=[
                io.Model.Input("model1", tooltip="Base model; its keys define the result."),
                io.Model.Input("model2", tooltip="Model blended into model1."),
                io.Float.Input(
                    "input", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Ratio for input_blocks (also the default for keys in no block).",
                ),
                io.Float.Input(
                    "middle", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Ratio for middle_block.",
                ),
                io.Float.Input(
                    "out", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Ratio for output_blocks.",
                ),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=["merge blocks", "block merge", "unet block merge", "layer merge"],
        )

    @classmethod
    def execute(
        cls,
        model1,
        model2,
        input: float = 1.0,
        middle: float = 1.0,
        out: float = 1.0,
    ) -> io.NodeOutput:
        merged = _blend(
            model1, model2, _blocks_blend(input, middle, out),
            "ModelMergeBlocks", MODEL_PREFIX, wrap=protocol.make_model_patcher,
        )
        return io.NodeOutput(merged)


class ModelMergeAdd(io.ComfyNode):
    """Add two models together (``model1 + model2``).

    What: straight tensor addition of every diffusion-model key - the operation
          behind "model arithmetic" experiments. Add the same model twice to
          double its weights, or add a difference model to shift a base model.
    In:   model1 (MODEL), model2 (MODEL).
    Out: MODEL - ``model1 + model2``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "ModelMergeAdd",
            "ModelMergeAdd",
            "Add two models key by key (model1 + model2).",
            inputs=[
                io.Model.Input("model1", tooltip="Base model; its keys define the result."),
                io.Model.Input("model2", tooltip="Model added to model1."),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=["add models", "model add", "sum models", "model arithmetic"],
        )

    @classmethod
    def execute(cls, model1, model2) -> io.NodeOutput:
        merged = _blend(
            model1, model2, _linear_blend(1.0, 1.0),
            "ModelMergeAdd", MODEL_PREFIX, wrap=protocol.make_model_patcher,
        )
        return io.NodeOutput(merged)


class ModelMergeSubtract(io.ComfyNode):
    """Subtract one model from another (``model1 - multiplier * model2``).

    What: the inverse of :class:`ModelMergeAdd`, used to remove a concept from a
          model (the classic "base - (finetuned - base)" arithmetic).
    In:   model1 (MODEL), model2 (MODEL).
          multiplier (FLOAT, default 1.0, -10..10) - how much of model2 to take
          away, and negative values add it back.
    Out: MODEL - ``model1 - multiplier * model2``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "ModelMergeSubtract",
            "ModelMergeSubtract",
            "Subtract one model from another key by key (model1 - multiplier * model2).",
            inputs=[
                io.Model.Input("model1", tooltip="Base model; its keys define the result."),
                io.Model.Input("model2", tooltip="Model subtracted from model1."),
                io.Float.Input(
                    "multiplier", default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip="How much of model2 to subtract; negative values add it back.",
                ),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=["subtract models", "model difference", "model arithmetic", "negate model"],
        )

    @classmethod
    def execute(cls, model1, model2, multiplier: float = 1.0) -> io.NodeOutput:
        merged = _blend(
            model1, model2, _linear_blend(multiplier, -multiplier),
            "ModelMergeSubtract", MODEL_PREFIX, wrap=protocol.make_model_patcher,
        )
        return io.NodeOutput(merged)


class CLIPMergeSimple(io.ComfyNode):
    """Blend two text encoders linearly by a single ratio.

    What: the CLIP counterpart of :class:`ModelMergeSimple`. Every key is blended
          except the non-weight entries ``.position_ids`` and ``.logit_scale``,
          which are copied from ``clip1`` - the native nodes skip them for the
          same reason (they are indices / scalars, not weights).
    In:   clip1 (CLIP) - the base; its key set becomes the result's.
          clip2 (CLIP) - the encoder blended into it.
          ratio (FLOAT, default 1.0, 0..1) - 1.0 keeps clip1, 0.0 keeps clip2.
    Out: CLIP - the blended weights.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "CLIPMergeSimple",
            "CLIPMergeSimple",
            "Blend two text encoders with a single ratio; .position_ids / .logit_scale are kept from clip1.",
            inputs=[
                io.Clip.Input("clip1", tooltip="Base text encoder; its keys define the result."),
                io.Clip.Input("clip2", tooltip="Text encoder blended into clip1."),
                io.Float.Input(
                    "ratio", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 = 100% clip1, 0.0 = 100% clip2.",
                ),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
            search_aliases=["merge clip", "clip merge", "text encoder merge", "blend clip"],
        )

    @classmethod
    def execute(cls, clip1, clip2, ratio: float = 1.0) -> io.NodeOutput:
        merged = _blend(
            clip1, clip2, _linear_blend(ratio, 1.0 - ratio),
            "CLIPMergeSimple", "", exclude_suffixes=CLIP_EXCLUDE_SUFFIXES,
            wrap=protocol.make_container,
        )
        return io.NodeOutput(merged)


class CLIPMergeAdd(io.ComfyNode):
    """Add two text encoders together (``clip1 + clip2``).

    What: straight addition of every weight key, skipping ``.position_ids`` /
          ``.logit_scale`` like the native node.
    In:   clip1 (CLIP), clip2 (CLIP).
    Out: CLIP - ``clip1 + clip2``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "CLIPMergeAdd",
            "CLIPMergeAdd",
            "Add two text encoders key by key (clip1 + clip2).",
            inputs=[
                io.Clip.Input("clip1", tooltip="Base text encoder; its keys define the result."),
                io.Clip.Input("clip2", tooltip="Text encoder added to clip1."),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
            search_aliases=["add clip", "clip add", "combine clip", "text encoder add"],
        )

    @classmethod
    def execute(cls, clip1, clip2) -> io.NodeOutput:
        merged = _blend(
            clip1, clip2, _linear_blend(1.0, 1.0),
            "CLIPMergeAdd", "", exclude_suffixes=CLIP_EXCLUDE_SUFFIXES,
            wrap=protocol.make_container,
        )
        return io.NodeOutput(merged)


class CLIPMergeSubtract(io.ComfyNode):
    """Subtract one text encoder from another (``clip1 - multiplier * clip2``).

    What: the inverse of :class:`CLIPMergeAdd`, skipping ``.position_ids`` /
          ``.logit_scale`` like the native node.
    In:   clip1 (CLIP), clip2 (CLIP).
          multiplier (FLOAT, default 1.0, -10..10) - how much of clip2 to remove.
    Out: CLIP - ``clip1 - multiplier * clip2``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "CLIPMergeSubtract",
            "CLIPMergeSubtract",
            "Subtract one text encoder from another (clip1 - multiplier * clip2).",
            inputs=[
                io.Clip.Input("clip1", tooltip="Base text encoder; its keys define the result."),
                io.Clip.Input("clip2", tooltip="Text encoder subtracted from clip1."),
                io.Float.Input(
                    "multiplier", default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip="How much of clip2 to subtract; negative values add it back.",
                ),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
            search_aliases=["subtract clip", "clip difference", "text encoder subtract"],
        )

    @classmethod
    def execute(cls, clip1, clip2, multiplier: float = 1.0) -> io.NodeOutput:
        merged = _blend(
            clip1, clip2, _linear_blend(multiplier, -multiplier),
            "CLIPMergeSubtract", "", exclude_suffixes=CLIP_EXCLUDE_SUFFIXES,
            wrap=protocol.make_container,
        )
        return io.NodeOutput(merged)


class CheckpointSave(io.ComfyNode):
    """Write MODEL + CLIP + VAE into one checkpoint file.

    What: the natural companion of the merge nodes - after blending, the result
          is written to ``output/`` as a ``.safetensors`` checkpoint holding all
          three buckets with their original key prefixes replayed, so the file
          has the same key layout as the checkpoint the workflow started from.
          Nothing is written when the same node runs twice: the filename counter
          is derived from the files already in the folder, like the native node.
    In:   model (MODEL), clip (CLIP), vae (VAE) - the buckets to save.
          filename_prefix (STRING, default ``comfydl/checkpoints``) - output
          subfolder / file stem; supports ``%date%`` style placeholders.
    Out: (none) - the file itself is the result.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "CheckpointSave",
            "Save Checkpoint",
            "Save MODEL + CLIP + VAE as one .safetensors checkpoint in the output folder.",
            inputs=[
                io.Model.Input("model", tooltip="MODEL bucket to save."),
                io.Clip.Input("clip", tooltip="CLIP bucket to save."),
                io.Vae.Input("vae", tooltip="VAE bucket to save."),
                io.String.Input(
                    "filename_prefix", default="comfydl/checkpoints",
                    tooltip="Output subfolder and file stem; %date% style placeholders are supported.",
                ),
            ],
            outputs=[],
            is_output_node=True,
            search_aliases=["save checkpoint", "export checkpoint", "merge save", "save model"],
        )

    @classmethod
    def execute(cls, model, clip, vae, filename_prefix: str = "comfydl/checkpoints") -> io.NodeOutput:
        model_sd, model_prefixes = protocol.container_state_dict(model)
        clip_sd, clip_prefixes = protocol.container_state_dict(clip)
        vae_sd, vae_prefixes = protocol.container_state_dict(vae)
        combined = protocol.merge_group_state_dicts([
            (model_sd, model_prefixes),
            (clip_sd, clip_prefixes),
            (vae_sd, vae_prefixes),
        ])
        _save_bucket(combined, {}, filename_prefix, _metadata_for_save())
        return io.NodeOutput()


class ModelSave(io.ComfyNode):
    """Write a MODEL value to ``.safetensors``.

    What: saves the diffusion-model bucket only, keyed exactly like the model it
          came from. Use it after a merge to keep just the UNet, or to export a
          UNET file read by ``Load UNET``.
    In:   model (MODEL) - the weights to save.
          filename_prefix (STRING, default ``comfydl/diffusion_models``).
    Out: (none) - the file itself is the result.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "ModelSave",
            "ModelSave",
            "Save a MODEL value as a .safetensors file in the output folder.",
            inputs=[
                io.Model.Input("model", tooltip="MODEL weights to save."),
                io.String.Input(
                    "filename_prefix", default="comfydl/diffusion_models",
                    tooltip="Output subfolder and file stem; %date% style placeholders are supported.",
                ),
            ],
            outputs=[],
            is_output_node=True,
            search_aliases=["save model", "export model", "save unet", "checkpoint save"],
        )

    @classmethod
    def execute(cls, model, filename_prefix: str = "comfydl/diffusion_models") -> io.NodeOutput:
        state_dict, prefixes = protocol.container_state_dict(model)
        _save_bucket(state_dict, prefixes, filename_prefix, _metadata_for_save())
        return io.NodeOutput()


class CLIPSave(io.ComfyNode):
    """Write a CLIP value to ``.safetensors``.

    What: saves a text-encoder bucket as a single file. The native node splits a
          multi-encoder CLIP into one file per encoder family (``clip_l``,
          ``t5xxl``, ...); this protocol layer keeps the container as one file,
          which is what ``Load CLIP`` / ``Load Dual CLIP`` read back.
    In:   clip (CLIP) - the weights to save.
          filename_prefix (STRING, default ``comfydl/clip``).
    Out: (none) - the file itself is the result.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "CLIPSave",
            "CLIPSave",
            "Save a CLIP value as a single .safetensors file in the output folder.",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP weights to save."),
                io.String.Input(
                    "filename_prefix", default="comfydl/clip",
                    tooltip="Output subfolder and file stem; %date% style placeholders are supported.",
                ),
            ],
            outputs=[],
            is_output_node=True,
            search_aliases=["save clip", "export text encoder", "save text encoder"],
        )

    @classmethod
    def execute(cls, clip, filename_prefix: str = "comfydl/clip") -> io.NodeOutput:
        state_dict, prefixes = protocol.container_state_dict(clip)
        _save_bucket(state_dict, prefixes, filename_prefix, _metadata_for_save())
        return io.NodeOutput()


class VAESave(io.ComfyNode):
    """Write a VAE value to ``.safetensors``.

    What: saves the VAE bucket, which is exactly the file layout ``Load VAE``
          expects back.
    In:   vae (VAE) - the weights to save.
          filename_prefix (STRING, default ``comfydl/vae``).
    Out: (none) - the file itself is the result.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _merge_schema(
            "VAESave",
            "VAESave",
            "Save a VAE value as a .safetensors file in the output folder.",
            inputs=[
                io.Vae.Input("vae", tooltip="VAE weights to save."),
                io.String.Input(
                    "filename_prefix", default="comfydl/vae",
                    tooltip="Output subfolder and file stem; %date% style placeholders are supported.",
                ),
            ],
            outputs=[],
            is_output_node=True,
            search_aliases=["save vae", "export vae", "save autoencoder"],
        )

    @classmethod
    def execute(cls, vae, filename_prefix: str = "comfydl/vae") -> io.NodeOutput:
        state_dict, prefixes = protocol.container_state_dict(vae)
        _save_bucket(state_dict, prefixes, filename_prefix, _metadata_for_save())
        return io.NodeOutput()


MERGE_NODES: list[type[io.ComfyNode]] = [
    ModelMergeSimple,
    ModelMergeBlocks,
    ModelMergeAdd,
    ModelMergeSubtract,
    CLIPMergeSimple,
    CLIPMergeAdd,
    CLIPMergeSubtract,
    CheckpointSave,
    ModelSave,
    CLIPSave,
    VAESave,
]


class ModelMergingExtension(ComfyExtension):
    """Registers the ``model/merging`` node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(MERGE_NODES)


async def comfy_entrypoint() -> ModelMergingExtension:
    return ModelMergingExtension()
