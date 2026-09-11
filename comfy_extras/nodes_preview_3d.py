"""Preview 3D - the stock ``Preview3D`` node, on its own.

Upstream ComfyUI ships this node inside ``comfy_extras/nodes_load_3d.py`` together
with the ``Load3D`` / Gaussian-splat family. The dehydrated build keeps neither,
but the frontend still binds its 3D canvas to the hard-coded node id
``Preview3D`` (see ``web/assets/load3d-*.js``), which means a custom node can
never render 3D itself - it can only produce a file and hand it over. That is why
this thin node exists: it is the receiving end for ``CdlHeatmapsTo3D``.

Only ``Preview3D`` is registered here. It accepts anything the loader understands
(obj / glb / gltf / fbx / stl / usdz) as a path string or as a ``Types.File3D``
object, drops object payloads into the output folder and asks the frontend to
display the result.
"""

import os
import uuid

from typing_extensions import override

import folder_paths
from comfy_api.latest import IO, UI, ComfyExtension, Types


class Preview3D(IO.ComfyNode):
    """Show a 3D file (obj/glb/gltf/fbx/stl/usdz) in the node's 3D canvas.

    Inputs:
        model_file (STRING | Types.File3D): a path below the ComfyUI input/output
            directory, or a 3D file object such as the one ``CdlHeatmapsTo3D``
            returns. Objects are saved to the output directory under a generated
            name; path strings are passed through untouched.
        camera_info (LOAD3D_CAMERA, optional): camera state to restore.
        bg_image (IMAGE, optional): image shown behind the model.

    Outputs:
        none. The preview travels through the ``ui`` payload, which the frontend
        routes to this node's canvas widget.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Preview3D",
            search_aliases=["view mesh", "3d preview"],
            display_name="Preview 3D",
            category="3d",
            description="Preview a 3D model file without saving it to the ComfyUI output directory.",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    IO.String.Input("model_file", default="", multiline=False),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DFBX,
                        IO.File3DOBJ,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DAny,
                    ],
                    tooltip="3D model file or path string",
                ),
                IO.Load3DCamera.Input("camera_info", optional=True, advanced=True),
                IO.Image.Input("bg_image", optional=True, advanced=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, model_file: str | Types.File3D, **kwargs) -> IO.NodeOutput:
        if isinstance(model_file, Types.File3D):
            filename = f"preview3d_{uuid.uuid4().hex}.{model_file.format}"
            model_file.save_to(os.path.join(folder_paths.get_output_directory(), filename))
        else:
            filename = model_file

        camera_info = kwargs.get("camera_info", None)
        bg_image = kwargs.get("bg_image", None)
        return IO.NodeOutput(ui=UI.PreviewUI3D(filename, camera_info, bg_image=bg_image))

    process = execute  # TODO: remove


class Preview3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Preview3D]


async def comfy_entrypoint() -> Preview3DExtension:
    return Preview3DExtension()
