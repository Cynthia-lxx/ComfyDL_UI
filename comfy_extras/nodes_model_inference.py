"""Placeholder inference nodes (reform step 5: MODEL protocol layer).

These four nodes exist so that a workflow can be *wired up* end to end - the
``MODEL`` / ``CLIP`` / ``VAE`` types and the sockets all match the native nodes,
so the graph validates and the node library looks complete - but they cannot do
any work in this build, because that work is exactly what the dehydration pass
removed:

* ``VAE Decode`` / ``VAE Encode`` need the autoencoder architecture
  (``comfy/ldm/models/autoencoder.py``) to turn latents into pixels,
* ``CLIP Text Encode`` needs a text-encoder architecture,
* ``CLIP Set Last Layer`` needs the same encoder to know how many layers it has.

Rather than let the user hit a bare ``ModuleNotFoundError`` from deep inside the
engine, ``execute`` raises a ``RuntimeError`` that names the missing module and
how to restore it. Everything they need to be *registered and connectable* is
already correct, so when ``ldm`` / ``text_encoders`` are restored these nodes
become functional in place.
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
        raise _unsupported(
            "VAEDecode", "decoding latents into an image",
            "comfy/ldm (the autoencoder architecture)",
        )


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
        raise _unsupported(
            "VAEEncode", "encoding an image into latents",
            "comfy/ldm (the autoencoder architecture)",
        )


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
        raise _unsupported(
            "CLIPTextEncode", "encoding text into conditioning",
            "comfy/text_encoders (a text-encoder architecture)",
        )


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
        raise _unsupported(
            "CLIPSetLastLayer", "truncating a text encoder",
            "comfy/text_encoders (a text-encoder architecture)",
        )


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
