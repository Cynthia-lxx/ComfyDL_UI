"""Generation nodes restored by the rehydration pass (Load Checkpoint + KSampler).

These two nodes complete the diffusion chain that the dehydration pass had
reduced to the protocol layer:

* ``Empty Latent Image`` builds the (B, 4, H/8, W/8) latent a sampler starts
  from, which is what ``KSampler`` consumes and what ``VAE Decode`` turns
  back into an image;
* ``KSampler`` denoises that latent with the model, the positive/negative
  conditioning and the sampler/scheduler pair, using the restored
  ``comfy.sample`` / ``comfy.samplers`` / ``comfy.k_diffusion`` stack.

Both are the native implementations; only the schema plumbing follows this
build's v3 (``io.Schema``) style, like the other ``comfy_extras`` nodes.
"""

from __future__ import annotations

import latent_preview
import torch
from typing_extensions import override

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_api.latest import ComfyExtension, io

CATEGORY_LATENT = "model/latent"
CATEGORY_SAMPLING = "model/sampling"

MAX_RESOLUTION = 16384


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    """The native sampling helper shared by KSampler and KSamplerAdvanced."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    if disable_noise:
        noise = comfy.sample.prepare_empty_noise(latent_image)
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step,
        last_step=last_step, force_full_denoise=force_full_denoise, noise_mask=noise_mask,
        callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


class EmptyLatentImage(io.ComfyNode):
    """Create a batch of empty latents to be denoised by a sampler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyLatentImage",
            display_name="Empty Latent Image",
            category=CATEGORY_LATENT,
            description="Create a new batch of empty latent images to be denoised via sampling.",
            search_aliases=["empty", "empty latent", "new latent", "create latent", "blank latent"],
            inputs=[
                io.Int.Input("width", default=512, min=16, max=MAX_RESOLUTION, step=8,
                             tooltip="The width of the latent images in pixels."),
                io.Int.Input("height", default=512, min=16, max=MAX_RESOLUTION, step=8,
                             tooltip="The height of the latent images in pixels."),
                io.Int.Input("batch_size", default=1, min=1, max=4096,
                             tooltip="The number of latent images in the batch."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def execute(cls, width: int, height: int, batch_size: int = 1) -> io.NodeOutput:
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 8})


class KSampler(io.ComfyNode):
    """Denoise a latent with a model and positive/negative conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KSampler",
            display_name="KSampler",
            category=CATEGORY_SAMPLING,
            description="Uses the provided model, positive and negative conditioning to denoise the latent image.",
            search_aliases=["sampler", "sample", "generate", "denoise", "txt2img", "img2img"],
            inputs=[
                io.Model.Input("model", tooltip="The model used for denoising the input latent."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True,
                             tooltip="The random seed used for creating the noise."),
                io.Int.Input("steps", default=20, min=1, max=10000,
                             tooltip="The number of steps used in the denoising process."),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1,
                               tooltip="The Classifier-Free Guidance scale."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                               tooltip="The algorithm used when sampling."),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                               tooltip="The scheduler controls how noise is gradually removed."),
                io.Conditioning.Input("positive",
                                      tooltip="The conditioning describing the attributes you want."),
                io.Conditioning.Input("negative",
                                      tooltip="The conditioning describing the attributes to avoid."),
                io.Latent.Input("latent_image", tooltip="The latent image to denoise."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="The amount of denoising applied."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def execute(
        cls,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ) -> io.NodeOutput:
        out = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                              positive, negative, latent_image, denoise=denoise)
        return io.NodeOutput(out)


GENERATION_NODES: list = [EmptyLatentImage, KSampler]


class GenerationExtension(ComfyExtension):
    """Registers the rehydrated generation nodes."""

    @override
    async def get_node_list(self) -> list:
        return list(GENERATION_NODES)


async def comfy_entrypoint() -> GenerationExtension:
    return GenerationExtension()
