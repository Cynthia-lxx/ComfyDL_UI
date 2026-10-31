"""GGUF quantized-model nodes (SP-4, M-B; ADR-D3).

The quantization family's dual-form pair, on the MODEL protocol layer:

* ``CdlLoadGGUF`` (production): loads a GGUF-quantized diffusion model from
  ``models/unet_gguf`` as a MODEL whose weights stay on disk-compressed
  GGML blocks and dequantize layer-by-layer right before each matmul
  (``comfy.quant_formats``; a port of ComfyUI-GGUF, city96, Apache-2.0).
  VRAM scales with the *file*, not the fp16 footprint.
* ``CdlQuantizationInspector`` (teaching): reads a GGUF (or safetensors)
  file's metadata without loading weights and renders the per-layer byte
  ledger - layer name, quant type, compressed vs fp bytes, compression
  ratio, effective bits - the first brick of the Quantization Lab.

ADR-D3 hard binding: the profiling estimators consume the same size table
(``quant_formats.GGML_QTYPE_SIZES``), so a GGUF model's memory verdict is
computed from the *quantized* bytes-per-element, not an fp16 assumption.
Both nodes self-meter nothing at runtime (load cost is prompt-setup, the
inspector is Tier-2-style user-triggered).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

import comfy.sd
import folder_paths

from comfy import quant_formats as qf

CATEGORY = "model/perf"

_DEQUANT_DTYPE_OPTIONS = ["default", "target", "float32", "float16", "bfloat16"]


def _register_unet_gguf_folder() -> None:
    """Register the ``unet_gguf`` folder key (idempotent), mirroring the
    ``unet``/``diffusion_models`` base paths, extension ``.gguf``."""
    base = folder_paths.folder_names_and_paths.get("unet_gguf", ([], {}))
    base_paths = base[0] if isinstance(base[0], (list, set, tuple)) else []
    if base_paths:
        return
    target = next((key for key in ("diffusion_models", "unet")
                   if key in folder_paths.folder_names_and_paths),
                  "diffusion_models")
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths["unet_gguf"] = (list(orig) or [], {".gguf"})


_register_unet_gguf_folder()


def _resolve_dtype(option: str) -> Optional[torch.dtype]:
    if option in ("default", None):
        return None
    if option == "target":
        return "target"
    return getattr(torch, option)


class CdlLoadGGUF(io.ComfyNode):
    """Load a GGUF-quantized diffusion model as a MODEL (lazy dequant).

    What: reads a ``.gguf`` file from ``models/unet_gguf`` (mmapped, nothing
          expanded), wraps every tensor in a ``GGMLTensor`` that carries the
          raw quantized blocks plus the logical shape/type, detects the
          architecture from the state dict and builds a runnable MODEL whose
          Linear/Conv/Embedding/Norm layers dequantize their weight right
          before each forward (``GGMLOps``).  VRAM scales with the *file* -
          a Q4_K_M UNet costs ~4.5 bits/weight instead of 16.
    In:   unet_name (COMBO) - file from the ``unet_gguf`` folder.
          dequant_dtype (COMBO, default ``default``) - dtype the blocks are
          decompressed to; ``target`` uses the layer's compute dtype.  fp32
          is the quality-safe choice, fp16 the fast default.
          patch_dtype (COMBO, default ``default``) - dtype used to apply
          LoRA patches on top of the dequantized weight.
    Out:  MODEL - the quantized model wrapped in a quant-aware ModelPatcher.
    Caveats: needs the ``gguf`` package (pip install gguf>=0.13.0,<0.14).
          Image-model architectures only (SD1/SDXL/SD3/Flux/...; text-encoder
          GGUFs are out of scope for this loader).  Combining GGUF with
          ``CdlCompileModel`` (torch.compile) is untested - GGMLTensor is a
          tensor subclass and trace behaviour is not guaranteed; prefer not
          stacking them, or use ``aot_eager`` if you must.  LoRA patches are
          applied at dequant time (patch dtype selectable); exotic patch
          target keys that alias quantized weights are applied to the
          logical weight, matching upstream behaviour.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        unet_names = folder_paths.get_filename_list("unet_gguf")
        return io.Schema(
            node_id="CdlLoadGGUF",
            display_name="Load Diffusion Model (GGUF)",
            category=CATEGORY,
            description=(
                "Load a GGUF-quantized diffusion model; weights stay "
                "compressed on the file's dtype and dequantize on the fly "
                "per layer, so VRAM scales with the file."),
            inputs=[
                io.Combo.Input(
                    "unet_name", options=unet_names,
                    tooltip="GGUF file from models/unet_gguf."),
                io.Combo.Input(
                    "dequant_dtype", options=_DEQUANT_DTYPE_OPTIONS,
                    default="default",
                    tooltip="Dtype to decompress blocks to ('target' = the "
                            "layer's compute dtype)."),
                io.Combo.Input(
                    "patch_dtype", options=_DEQUANT_DTYPE_OPTIONS,
                    default="default",
                    tooltip="Dtype used when applying LoRA patches on top "
                            "of the dequantized weight."),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=[
                "gguf", "quantized", "quantization", "low vram", "q4", "q8",
                "load model", "diffusion model",
            ],
        )

    @classmethod
    def execute(cls, unet_name: str, dequant_dtype: str = "default",
                patch_dtype: str = "default") -> io.NodeOutput:
        gguf = qf.require_gguf()
        ops = qf.GGMLOps()
        ops.Linear.dequant_dtype = _resolve_dtype(dequant_dtype)
        ops.Linear.patch_dtype = _resolve_dtype(patch_dtype)

        unet_path = folder_paths.get_full_path_or_raise("unet_gguf", unet_name)
        sd, extra = qf.gguf_sd_loader(unet_path)

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops},
            metadata=extra.get("metadata", {}),
        )
        if model is None:
            raise RuntimeError(
                f"ERROR: Could not detect model type of: {unet_path}")
        model = qf.GGUFModelPatcher.clone(model)
        logging.info(
            f"[quant/gguf] loaded {unet_name!r} "
            f"(arch={extra.get('arch_str')}) as a lazy-dequant MODEL.")
        return io.NodeOutput(model)


class CdlQuantizationInspector(io.ComfyNode):
    """Read a quantized model file's metadata and render the byte ledger.

    What: opens a .gguf (or .safetensors) file *without loading weights into
          RAM* and produces a per-layer ledger STRING: layer name, quant
          type, compressed bytes, hypothetical fp16/fp32 bytes, compression
          ratio and the effective bits-per-weight - sorted by size, with a
          totals footer.  This is the "what am I actually saving" teaching
          artifact of the Quantization Lab: for K-quants the effective bits
          are fractional (Q4_K = 144 bytes / 256 elements = 4.5 bits), which
          is exactly why the family is named Q4 but stores more than 4 bits.
    In:   model_path (COMBO) - file from the unet_gguf folder (first release
          scope: GGUF files; safetensors inspect stays a backlog item).
    Out:  STRING - the human-readable ledger.
    Caveats: read-only metadata inspection; the file is never loaded as
          weights.  Needs the ``gguf`` package for .gguf files.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        _register_unet_gguf_folder()
        names = folder_paths.get_filename_list("unet_gguf")
        return io.Schema(
            node_id="CdlQuantizationInspector",
            display_name="Quantization Inspector",
            category=CATEGORY,
            description=(
                "Per-layer byte ledger of a GGUF file: quant type, "
                "compressed vs fp bytes, compression ratio and effective "
                "bits per weight."),
            inputs=[
                io.Combo.Input(
                    "model_path", options=names,
                    tooltip="GGUF file from models/unet_gguf (metadata-only "
                            "read, weights are not loaded)."),
            ],
            outputs=[io.String.Output(display_name="ledger")],
            search_aliases=[
                "quantization", "gguf", "bits", "byte", "inspect",
                "compression", "q4_k", "teaching",
            ],
        )

    @classmethod
    def execute(cls, model_path: str) -> io.NodeOutput:
        qf.require_gguf()
        path = folder_paths.get_full_path_or_raise("unet_gguf", model_path)
        sd, extra = qf.gguf_sd_loader(path)
        ledger = cls._render(path, sd, extra)
        print(ledger)
        return io.NodeOutput(ledger)

    @staticmethod
    def _render(path: str, sd: dict, extra: dict) -> str:
        summary = qf.summarize_state_dict_bytes(sd)
        lines = [
            f"[quant-inspector] {os.path.basename(path)} "
            f"(arch={extra.get('arch_str')})",
            f"{'layer':<58} {'qtype':<10} {'bytes':>12} {'fp bytes':>12} "
            f"{'ratio':>7} {'bits/w':>7}",
        ]
        for entry in summary["layers"]:
            ratio = (entry["fp_bytes"] / entry["bytes"]) if entry["bytes"] else 0.0
            bits = entry["bytes"] * 8.0 / entry["elements"] if entry["elements"] else 0.0
            lines.append(
                f"{entry['key'][:57]:<58} {entry['qtype']:<10} "
                f"{entry['bytes']:>12,} {entry['fp_bytes']:>12,} "
                f"{ratio:>6.2f}x {bits:>7.3f}")
        total = summary["total_bytes"]
        fp = summary["fp_bytes"]
        lines.append(
            f"{'TOTAL':<58} {'':<10} {total:>12,} {fp:>12,} "
            f"{(fp / total) if total else 0:>6.2f}x")
        lines.append(
            f"  saved {(1 - total / fp) * 100 if fp else 0:.1f}% of the fp "
            f"footprint ({total / 1024**3:.2f} GiB vs {fp / 1024**3:.2f} GiB). "
            f"K-quants carry block scales, so e.g. Q4_K is 4.5 bits/weight, "
            f"not 4.")
        return "\n".join(lines)


class QuantExtension(ComfyExtension):
    """Registers the ``model/perf`` quantization node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlLoadGGUF, CdlQuantizationInspector]


async def comfy_entrypoint() -> QuantExtension:
    return QuantExtension()
