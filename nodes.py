import torch


import os
import sys
import json
import glob
import hashlib
import inspect

import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from comfy_api.internal import register_versions, ComfyAPIWithVersion
from comfy_api.version_list import supported_versions
from comfy_api.latest import io, ComfyExtension, InputImpl

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

if args.enable_manager:
    import comfyui_manager

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=16384

class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."
                })
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."
    SEARCH_ALIASES = ["save", "save image", "export image", "output image", "write image", "download"]

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results }, "result" : (images,) }

class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    SEARCH_ALIASES = ["preview", "preview image", "show image", "view image", "display image", "image viewer"]
    DESCRIPTION = "Preview the images without saving them to the ComfyUI output directory."

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"
    ESSENTIALS_CATEGORY = "Basics"
    SEARCH_ALIASES = ["load image", "open image", "import image", "image input", "upload image", "read image", "image loader"]

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()

        components = InputImpl.VideoFromFile(image_path).get_components()
        if components.images.shape[0] > 0:
            return (components.images.to(device=device, dtype=dtype), (1.0 - components.alpha[..., -1]).to(device=device, dtype=dtype) if components.alpha is not None else torch.zeros((components.images.shape[0], 64, 64), dtype=dtype, device=device))

        # This code is left here to handle animated webp which pyav does not support loading
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image.to(dtype=dtype))
            output_masks.append(mask.unsqueeze(0).to(dtype=dtype))

        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)

        return (output_image.to(device=device, dtype=dtype), output_mask.to(device=device, dtype=dtype))

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class LoadImageMask(LoadImage):
    ESSENTIALS_CATEGORY = "Image Tools"
    SEARCH_ALIASES = ["import mask", "alpha mask", "channel mask"]

    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        return {
            "required": {
                **types["required"],
                "channel": (s._color_channels, )
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image_mask"

    def load_image_mask(self, image, channel):
        image_tensor, mask_tensor = super().load_image(image)
        c = channel[0].upper()

        if c == 'A':
            return (mask_tensor,)

        channel_idx = {'R': 0, 'G': 1, 'B': 2}.get(c, 0)

        if channel_idx < image_tensor.shape[-1]:
            return (image_tensor[..., channel_idx].clone(),)
        else:
            empty_mask = torch.zeros(
                image_tensor.shape[:-1],
                dtype=image_tensor.dtype,
                device=image_tensor.device
            )
            return (empty_mask,)

    @classmethod
    def IS_CHANGED(s, image, channel):
        return super().IS_CHANGED(image)


class LoadImageOutput(LoadImage):
    SEARCH_ALIASES = ["output image", "previous generation"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("COMBO", {
                    "image_upload": True,
                    "image_folder": "output",
                    "remote": {
                        "route": "/internal/files/output",
                        "refresh_button": True,
                        "control_after_refresh": "first",
                    },
                }),
            }
        }

    DESCRIPTION = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration."
    EXPERIMENTAL = True
    FUNCTION = "load_image"


class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    ESSENTIALS_CATEGORY = "Image Tools"
    SEARCH_ALIASES = ["resize", "resize image", "scale image", "image resize", "zoom", "zoom in", "change size"]

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)

class ImageScaleBy:
    ESSENTIALS_CATEGORY = "Image Tools"
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        samples = image.movedim(-1,1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

class ImageInvert:
    SEARCH_ALIASES = ["reverse colors"]
    ESSENTIALS_CATEGORY = "Image Tools"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"

    CATEGORY = "image/color"

    def invert(self, image):
        s = 1.0 - image
        return (s,)

class ImageBatch:
    SEARCH_ALIASES = ["combine images", "merge images", "stack images"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image1": ("IMAGE",), "image2": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "image/batch"
    DEPRECATED = True

    def batch(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            if image1.shape[-1] > image2.shape[-1]:
                image2 = torch.nn.functional.pad(image2, (0,1), mode='constant', value=1.0)
            else:
                image1 = torch.nn.functional.pad(image1, (0,1), mode='constant', value=1.0)
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1,1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1,-1)
        s = torch.cat((image1, image2), dim=0)
        return (s,)

class EmptyImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF, device=device, dtype=dtype)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF, device=device, dtype=dtype)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF, device=device, dtype=dtype)
        return (torch.cat((r, g, b), dim=-1), )

class ImagePadForOutpaint:
    SEARCH_ALIASES = ["extend canvas", "expand image"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1, "advanced": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image/transform"

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask.unsqueeze(0))


NODE_CLASS_MAPPINGS = {
    # Dehydrate: 仅保留基础 IO 图像节点（生成节点全部移除，见 docs/dehydrate_manifest.md）
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "LoadImage": LoadImage,
    "LoadImageMask": LoadImageMask,
    "LoadImageOutput": LoadImageOutput,
    "ImageScale": ImageScale,
    "ImageScaleBy": ImageScaleBy,
    "ImageInvert": ImageInvert,
    "ImageBatch": ImageBatch,
    "ImagePadForOutpaint": ImagePadForOutpaint,
    "EmptyImage": EmptyImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image（保留 IO 节点与 comfy_extras 图像链节点的显示名）
    "EmptyImage": "Empty Image",
    "SaveImage": "Save Image",
    "PreviewImage": "Preview Image",
    "LoadImage": "Load Image",
    "LoadImageMask": "Load Image (as Mask)",
    "LoadImageOutput": "Load Image (from Outputs)",
    "ImageScale": "Upscale Image",
    "ImageScaleBy": "Upscale Image By",
    "ImageInvert": "Invert Image Colors",
    "ImagePadForOutpaint": "Pad Image for Outpainting",
    "ImageBatch": "Batch Images (DEPRECATED)",
    "ImageCrop": "Crop Image",
    "ImageStitch": "Stitch Images",
    "ImageBlend": "Blend Images",
    "ImageBlur": "Blur Image",
    "ImageQuantize": "Quantize Image",
    "ImageSharpen": "Sharpen Image",
    "ImageScaleToTotalPixels": "Scale Image to Total Pixels",
    "GetImageSize": "Get Image Size",
}

EXTENSION_WEB_DIRS = {}

# Dictionary of successfully loaded module names and associated directories.
LOADED_MODULE_DIRS = {}


def get_module_name(module_path: str) -> str:
    """
    Returns the module name based on the given module path.
    Examples:
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.disabled") -> "custom_nodes
    Args:
        module_path (str): The path of the module.
    Returns:
        str: The module name.
    """
    base_path = os.path.basename(module_path)
    if os.path.isfile(module_path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


async def load_custom_node(module_path: str, ignore=set(), module_parent="custom_nodes") -> bool:
    module_name = get_module_name(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
        sys_module_name = module_name
    elif os.path.isdir(module_path):
        sys_module_name = module_path.replace(".", "_x_")

    try:
        logging.debug("Trying to load custom node {}".format(module_path))
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(sys_module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(sys_module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[sys_module_name] = module
        module_spec.loader.exec_module(module)

        LOADED_MODULE_DIRS[module_name] = os.path.abspath(module_dir)

        try:
            from comfy_config import config_parser

            project_config = config_parser.extract_node_configuration(module_path)

            web_dir_name = project_config.tool_comfy.web

            if web_dir_name:
                web_dir_path = os.path.join(module_path, web_dir_name)

                if os.path.isdir(web_dir_path):
                    project_name = project_config.project.name

                    EXTENSION_WEB_DIRS[project_name] = web_dir_path

                    logging.info("Automatically register web folder {} for {}".format(web_dir_name, project_name))
        except Exception as e:
            logging.warning(f"Unable to parse pyproject.toml due to lack dependency pydantic-settings, please run 'pip install -r requirements.txt': {e}")

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                EXTENSION_WEB_DIRS[module_name] = web_dir

        # V1 node definition
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = node_cls
                    node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        # V3 Extension Definition
        elif hasattr(module, "comfy_entrypoint"):
            entrypoint = getattr(module, "comfy_entrypoint")
            if not callable(entrypoint):
                logging.warning(f"comfy_entrypoint in {module_path} is not callable, skipping.")
                return False
            try:
                if inspect.iscoroutinefunction(entrypoint):
                    extension = await entrypoint()
                else:
                    extension = entrypoint()
                if not isinstance(extension, ComfyExtension):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a ComfyExtension, skipping.")
                    return False
                await extension.on_load()
                node_list = await extension.get_node_list()
                if not isinstance(node_list, list):
                    logging.warning(f"comfy_entrypoint in {module_path} did not return a list of nodes, skipping.")
                    return False
                for node_cls in node_list:
                    node_cls: io.ComfyNode
                    schema = node_cls.GET_SCHEMA()
                    if schema.node_id not in ignore:
                        NODE_CLASS_MAPPINGS[schema.node_id] = node_cls
                        node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
                    if schema.display_name is not None:
                        NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
                return True
            except Exception as e:
                logging.warning(f"Error while calling comfy_entrypoint in {module_path}: {e}")
                return False
        else:
            logging.warning(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS or comfy_entrypoint (need one).")
            return False
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(f"Cannot import {module_path} module for custom nodes: {e}")
        return False

async def init_external_custom_nodes():
    """
    Initializes the external custom nodes.

    This function loads custom nodes from the specified folder paths and imports them into the application.
    It measures the import times for each custom node and logs the results.

    Returns:
        None
    """
    # TODO: remove at some point when custom nodes don't break.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue
            if module_path.endswith(".disabled"):
                continue
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logging.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue

            if args.enable_manager:
                if comfyui_manager.should_be_disabled(module_path):
                    logging.info(f"Blocked by policy: {module_path}")
                    continue

            time_before = time.perf_counter()
            success = await load_custom_node(module_path, base_node_names, module_parent="custom_nodes")
            node_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_import_times) > 0:
        logging.info("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

async def init_builtin_extra_nodes():
    """
    Initializes the built-in extra nodes in ComfyUI.

    This function loads the extra node files located in the "comfy_extras" directory and imports them into ComfyUI.
    If any of the extra node files fail to import, a warning message is logged.

    Returns:
        None
    """
    extras_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_extras")
    # Dehydrate: 仅保留 IMAGE/MASK/LATENT 链与基础工具节点（生成链 extras 全部删除）
    extras_files = [
        "nodes_latent.py",
        "nodes_post_processing.py",
        "nodes_mask.py",
        "nodes_compositing.py",
        "nodes_rebatch.py",
        "nodes_images.py",
        "nodes_primitive.py",
        "nodes_string.py",
        "nodes_logic.py",
        "nodes_resolution.py",
        "nodes_nop.py",
        "nodes_color.py",
        "nodes_toolkit.py",
        "nodes_math.py",
        "nodes_number_convert.py",
        "nodes_curve.py",
        "nodes_text.py",
    ]

    import_failed = []
    for node_file in extras_files:
        if not await load_custom_node(os.path.join(extras_dir, node_file), module_parent="comfy_extras"):
            import_failed.append(node_file)

    return import_failed


async def init_builtin_api_nodes():
    api_nodes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_api_nodes")
    api_nodes_files = sorted(glob.glob(os.path.join(api_nodes_dir, "nodes_*.py")))

    import_failed = []
    for node_file in api_nodes_files:
        if not await load_custom_node(node_file, module_parent="comfy_api_nodes"):
            import_failed.append(os.path.basename(node_file))

    return import_failed


async def init_builtin_dl_nodes():
    """Register the ComfyDL teaching nodes as built-in core nodes.

    ComfyDL lives in the ``./comfydl`` git submodule (github.com/Cynthia-lxx/ComfyDL)
    and is imported here as a real top-level package so that its relative imports
    and the sys.path bootstrap inside ``comfydl/__init__.py`` keep working. Node
    mappings are merged into the core registry exactly like any built-in module;
    the frontend picks them up automatically through the shared global dicts.

    On a fresh clone where the submodule was not checked out yet, this degrades
    gracefully to a warning with an actionable ``git submodule update --init``.
    """
    import_failed = []
    try:
        import comfydl
    except Exception:
        import_failed.append("comfydl")
        logging.warning("WARNING: comfydl (ComfyDL built-in nodes) failed to import.")
        logging.warning("comfydl is a git submodule; if it is missing, run: git submodule update --init")
        logging.warning("  repo: https://github.com/Cynthia-lxx/ComfyDL.git")
        logging.warning(traceback.format_exc())
        return import_failed

    dl_class_mappings = getattr(comfydl, "NODE_CLASS_MAPPINGS", {}) or {}
    if not dl_class_mappings:
        import_failed.append("comfydl")
        logging.warning("WARNING: comfydl registered 0 nodes (expected ~102 'Cdl*' nodes);")
        logging.warning("this may indicate an import/aggregation issue inside the comfydl submodule.")

    for name, node_cls in dl_class_mappings.items():
        if name in NODE_CLASS_MAPPINGS:
            logging.warning("ComfyDL node '{}' collides with an existing core node; skipping.".format(name))
            continue
        NODE_CLASS_MAPPINGS[name] = node_cls
        node_cls.RELATIVE_PYTHON_MODULE = getattr(node_cls, "__module__", None) or "comfydl"
    dl_display_mappings = getattr(comfydl, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
    NODE_DISPLAY_NAME_MAPPINGS.update(dl_display_mappings)

    return import_failed


async def init_public_apis():
    register_versions([
        ComfyAPIWithVersion(
            version=getattr(v, "VERSION"),
            api_class=v
        ) for v in supported_versions
    ])

async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    await init_public_apis()

    import_failed = await init_builtin_extra_nodes()
    import_failed += await init_builtin_dl_nodes()

    import_failed_api = []
    if init_api_nodes:
        import_failed_api = await init_builtin_api_nodes()

    if init_custom_nodes:
        await init_external_custom_nodes()
    else:
        logging.info("Skipping loading of custom nodes")

    if len(import_failed_api) > 0:
        logging.warning("WARNING: some comfy_api_nodes/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed_api:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    if len(import_failed) > 0:
        logging.warning("WARNING: some comfy_extras/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")

    return import_failed
