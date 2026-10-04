"""Inference nodes for the diffusion pipeline (rehydrated from the original build).

These nodes close the gap the dehydration pass opened in the generation
pipeline. The ``comfy/ldm`` autoencoder, the ``text_encoders`` and the sampler
stack (``comfy.sample`` / ``comfy.samplers`` / ``comfy.k_diffusion``) have been
restored, so these nodes now call the native code path:

* ``VAE Decode`` / ``VAE Encode`` use the restored autoencoder architecture to
  turn latents into pixels and back,
* ``CLIP Text Encode`` runs the restored text-encoder stack,
* ``CLIP Set Last Layer`` reads the encoder's layer count.

They need the optional dependencies ``torchsde`` (sampler stack) and
``transformers`` (text encoders); see ``docs/dehydrate_manifest.md``.
"""

from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY_LATENT = "model/latent"
CATEGORY_CONDITIONING = "model/conditioning"


def _unsupported(node_name: str, feature: str, modules: str) -> RuntimeError:
    """Build the error raised by every placeholder node.

    Args:
        node_name: Display name used at the start of the message.
        feature: What the node would do, in one short phrase.
        modules: The module(s) that must be restored.

    Returns:
        The ``RuntimeError`` to raise.
    """
    return RuntimeError(
        f"{node_name}: {feature} is not available in this dehydrated ComfyUI build. "
        f"It needs {modules}, which the dehydration pass removed; the node is "
        f"registered so a workflow can be wired up, but it cannot run until that code "
        f"is restored from the read-only reference checkout 'ComfyUI-original/' (see "
        f"docs/dehydrate_manifest.md). Saving, merging and loading weights does work - "
        f"only the image / text side of the pipeline is affected."
    )


class VAEDecode(io.ComfyNode):
    """Decode latents into an image (protocol placeholder, not executable here).

    What: the native VAE decode. Registered with the native IO contract so a
          workflow validates, but it needs the autoencoder architecture that the
          dehydration pass removed, and raises a ``RuntimeError`` saying so.
    In:   samples (LATENT), vae (VAE).
    Out: IMAGE - the decoded image (in a build that has ``comfy/ldm``).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VAEDecode",
            display_name="VAE Decode",
            category=CATEGORY_LATENT,
            description="Decode latents to an image. Registered for wiring; requires comfy/ldm, which this dehydrated build does not include.",
            search_aliases=["vae decode", "decode latent", "latent to image"],
            inputs=[
                io.Latent.Input("samples", tooltip="Latents to decode."),
                io.Vae.Input("vae", tooltip="VAE used for decoding."),
            ],
            outputs=[io.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        # Rehydrated: the native decode (comfy/ldm autoencoder is restored).
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:  # combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)


class VAEEncode(io.ComfyNode):
    """Encode an image into latents (protocol placeholder, not executable here).

    What: the native VAE encode. Registered with the native IO contract; needs
          the autoencoder architecture that the dehydration pass removed.
    In:   pixels (IMAGE), vae (VAE).
    Out: LATENT - the encoded latents (in a build that has ``comfy/ldm``).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VAEEncode",
            display_name="VAE Encode",
            category=CATEGORY_LATENT,
            description="Encode an image into latents. Registered for wiring; requires comfy/ldm, which this dehydrated build does not include.",
            search_aliases=["vae encode", "encode image", "image to latent"],
            inputs=[
                io.Image.Input("pixels", tooltip="Image to encode."),
                io.Vae.Input("vae", tooltip="VAE used for encoding."),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def execute(cls, pixels, vae) -> io.NodeOutput:
        # Rehydrated: the native encode (comfy/ldm autoencoder is restored).
        t = vae.encode(pixels)
        return io.NodeOutput({"samples": t})


class CLIPTextEncode(io.ComfyNode):
    """Turn a prompt into conditioning (protocol placeholder, not executable here).

    What: the native text encoding step. Registered with the native IO contract so
          a prompt can be wired to a sampler later; needs a text-encoder
          architecture, which the dehydration pass removed.
    In:   text (STRING, multiline) - the prompt.
          clip (CLIP) - the text encoder read by ``Load CLIP``.
    Out: CONDITIONING - the encoded prompt (in a build that has the encoders).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncode",
            display_name="CLIP Text Encode (Prompt)",
            category=CATEGORY_CONDITIONING,
            description="Encode a prompt with CLIP. Registered for wiring; requires a text-encoder architecture, which this dehydrated build does not include.",
            search_aliases=["clip text encode", "prompt", "text encode", "conditioning", "positive prompt"],
            inputs=[
                io.String.Input(
                    "text", multiline=True, default="a photo of a cat",
                    tooltip="Prompt text. Use the node's title to label positive / negative.",
                ),
                io.Clip.Input("clip", tooltip="Text encoder to use."),
            ],
            outputs=[io.Conditioning.Output(display_name="CONDITIONING")],
        )

    @classmethod
    def execute(cls, text: str, clip) -> io.NodeOutput:
        # Rehydrated: the native prompt encoding (comfy/text_encoders restored).
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a "
                "checkpoint loader node your checkpoint does not contain a valid "
                "clip or text encoder model."
            )
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))


class CLIPSetLastLayer(io.ComfyNode):
    """Clamp how many text-encoder layers are used (protocol placeholder).

    What: the native node that truncates a text encoder for the
          "clip skip" trick (``stop_at_clip_layer = -2`` is the usual choice).
          Registered with the native IO contract; needs a text-encoder
          architecture, which the dehydration pass removed.
    In:   clip (CLIP) - the encoder to truncate.
          stop_at_clip_layer (INT, default -1) - which layer to stop at; ``-1``
          means "use every layer".
    Out: CLIP - the truncated encoder (in a build that has the encoders).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPSetLastLayer",
            display_name="CLIP Set Last Layer",
            category=CATEGORY_CONDITIONING,
            description="Stop a text encoder at a given layer (clip skip). Registered for wiring; requires a text-encoder architecture, which this dehydrated build does not include.",
            search_aliases=["clip skip", "set last layer", "clip layer", "clip set last layer"],
            inputs=[
                io.Clip.Input("clip", tooltip="Text encoder to truncate."),
                io.Int.Input(
                    "stop_at_clip_layer", default=-1, min=-24, max=-1, step=1,
                    tooltip="-1 uses every layer; -2 is the common 'clip skip' setting.",
                ),
            ],
            outputs=[io.Clip.Output(display_name="CLIP")],
        )

    @classmethod
    def execute(cls, clip, stop_at_clip_layer: int = -1) -> io.NodeOutput:
        # Rehydrated: the native "clip skip" truncation.
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return io.NodeOutput(clip)


INFERENCE_NODES: list[type[io.ComfyNode]] = [
    VAEDecode,
    VAEEncode,
    CLIPTextEncode,
    CLIPSetLastLayer,
]


class ModelInferenceExtension(ComfyExtension):
    """Registers the L2 placeholder inference nodes."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(INFERENCE_NODES)


async def comfy_entrypoint() -> ModelInferenceExtension:
    return ModelInferenceExtension()
