"""Smoke tests for the generation-layer rehydration (Load Checkpoint + KSampler).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_rehydration.py

Why this script exists
----------------------
The dehydration pass replaced the diffusion generation layer with a
protocol-level stand-in: ``CheckpointLoaderSimple`` returned weight
containers instead of runnable models, and ``KSampler`` did not exist at
all. Rehydrating means two things that can silently regress:

* the restored ``comfy/`` generation modules import again (they are a large,
  interlinked tree - one missing file breaks the whole chain), and
* the nodes are registered and wired to the *real* code path rather than to
  the container helper.

This test pins both, plus the profiling rules that cover the new nodes. It
needs ``transformers`` and ``torchsde`` (see docs/dehydrate_manifest.md).
"""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


MODULES = (
    "comfy.sd",
    "comfy.model_detection",
    "comfy.supported_models",
    "comfy.model_base",
    "comfy.latent_formats",
    "comfy.conds",
    "comfy.clip_model",
    "comfy.sd1_clip",
    "comfy.samplers",
    "comfy.sample",
    "comfy.k_diffusion.sampling",
    "comfy.lora",
    "comfy.ldm.models.autoencoder",
    "comfy.ldm.modules.diffusionmodules.model",
    "latent_preview",
)


def _module_checks() -> None:
    for name in MODULES:
        try:
            importlib.import_module(name)
            check("R1 import %s" % name, True)
        except Exception as exc:  # noqa: BLE001 - every failure must surface
            check("R1 import %s" % name, False, "%s: %s" % (type(exc).__name__, exc))


async def _load_nodes() -> None:
    import nodes

    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)


def _node_checks() -> None:
    import nodes

    asyncio.run(_load_nodes())
    mapping = nodes.NODE_CLASS_MAPPINGS
    for node_id in ("CheckpointLoaderSimple", "KSampler", "EmptyLatentImage",
                    "VAEDecode", "VAEEncode", "CLIPTextEncode"):
        check("R2 registered: %s" % node_id, node_id in mapping,
              "missing from NODE_CLASS_MAPPINGS")

    # The loader must call the real detection/construction path, not the
    # container helper the dehydrated build used.
    import comfy_extras.nodes_model_loaders as loaders

    source = inspect.getsource(loaders.CheckpointLoaderSimple)
    check("R3 loader: uses load_checkpoint_guess_config",
          "load_checkpoint_guess_config" in source, "container path still wired")
    check("R4 loader: no longer builds containers",
          "make_model_patcher" not in source and "make_container" not in source)

    decode_source = inspect.getsource(
        importlib.import_module("comfy_extras.nodes_model_inference").VAEDecode
    )
    check("R5 VAE Decode: real decode instead of a raise",
          "vae.decode" in decode_source and "raise _unsupported" not in decode_source)

    encode_source = inspect.getsource(
        importlib.import_module("comfy_extras.nodes_model_inference").CLIPTextEncode
    )
    check("R6 CLIP Text Encode: real encoding instead of a raise",
          "encode_from_tokens_scheduled" in encode_source
          and "raise _unsupported" not in encode_source)


def _profiling_checks() -> None:
    from comfy.profiling import estimators as est
    from comfy.profiling import estimate_workflow

    for class_type in ("CheckpointLoaderSimple", "EmptyLatentImage", "KSampler",
                       "VAEDecode", "CLIPTextEncode"):
        check("R7 estimator registered: %s" % class_type, class_type in est.ESTIMATORS)

    # EmptyLatentImage -> VAEDecode: the latent shape flows through the chain
    report = estimate_workflow({
        "lat": {"class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "dec": {"class_type": "VAEDecode",
                "inputs": {"samples": ["lat", 0], "vae": ["na", 0]}},
    })
    lat = _node(report, "EmptyLatentImage")
    check("R8 EmptyLatentImage: (1, 4, 64, 64) latent by assumption",
          lat["status"] == "estimated" and lat["total_bytes"] == 1 * 4 * 64 * 64 * 4,
          str(lat["total_bytes"]))
    dec = _node(report, "VAEDecode")
    check("R9 VAEDecode: 64x64 latent -> 512x512 image (8x)",
          dec["status"] == "estimated"
          and dec["items"][0]["bytes"] == 1 * 512 * 512 * 3 * 4,
          str(dec["items"]))
    check("R10 assumptions recorded: latent channels and vae scale",
          {entry["key"] for entry in report["assumptions_used"]}
          >= {"latent_channels", "vae_scale"},
          str(report["assumptions_used"]))


def _node(report: dict, class_type: str) -> dict:
    return [n for n in report["nodes"] if n["class_type"] == class_type][0]


def main() -> int:
    _module_checks()
    _node_checks()
    _profiling_checks()
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
        print()
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
