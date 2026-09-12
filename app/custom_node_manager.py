import os
import folder_paths
import glob
from aiohttp import web
import json
import logging
from functools import lru_cache

from utils.json_util import merge_json_recursive


# Extra locale files to load into main.json
EXTRA_LOCALE_FILES = [
    "nodeDefs.json",
    "commands.json",
    "settings.json",
]

# ComfyDL_UI ships its translations with the host runtime instead of a
# custom_nodes package, so the repository level locale folder is scanned as well.
REPO_LOCALES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locales"
)


def safe_load_json_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error loading {file_path}")
        return {}


def load_locales_dir(locales_dir: str, translations: dict) -> None:
    """Merge the ``<lang>/*.json`` files of a locales folder into ``translations``."""
    if not os.path.exists(locales_dir):
        return

    for lang_dir in sorted(glob.glob(os.path.join(locales_dir, "*/"))):
        lang_code = os.path.basename(os.path.dirname(lang_dir))
        if lang_code not in translations:
            translations[lang_code] = {}

        # Load main.json
        node_translations = safe_load_json_file(os.path.join(lang_dir, "main.json"))

        # Load extra locale files
        for extra_file in EXTRA_LOCALE_FILES:
            json_data = safe_load_json_file(os.path.join(lang_dir, extra_file))
            if json_data:
                node_translations[extra_file.split(".")[0]] = json_data

        if node_translations:
            translations[lang_code] = merge_json_recursive(
                translations[lang_code], node_translations
            )


class CustomNodeManager:
    @lru_cache(maxsize=1)
    def build_translations(self):
        """Load all custom nodes translations during initialization. Translations are
        expected to be loaded from `locales/` folder.

        The folder structure is expected to be the following:
        - custom_nodes/
            - custom_node_1/
                - locales/
                    - en/
                        - main.json
                        - commands.json
                        - settings.json

        ComfyDL_UI's own translations live in the repository level `locales/` folder
        (same layout) and are merged on top of the custom node ones.

        returned translations are expected to be in the following format:
        {
            "en": {
                "nodeDefs": {...},
                "commands": {...},
                "settings": {...},
                ...{other main.json keys}
            }
        }
        """

        translations = {}

        for folder in folder_paths.get_folder_paths("custom_nodes"):
            # Sort glob results for deterministic ordering
            for custom_node_dir in sorted(glob.glob(os.path.join(folder, "*/"))):
                load_locales_dir(os.path.join(custom_node_dir, "locales"), translations)

        load_locales_dir(REPO_LOCALES_DIR, translations)

        return translations

    def add_routes(self, routes, webapp, loadedModules):

        example_workflow_folder_names = ["example_workflows", "example", "examples", "workflow", "workflows"]

        @routes.get("/workflow_templates")
        async def get_workflow_templates(request):
            """Returns a web response that contains the map of custom_nodes names and their associated workflow templates. The ones without templates are omitted."""

            files = []

            for folder in folder_paths.get_folder_paths("custom_nodes"):
                for folder_name in example_workflow_folder_names:
                    pattern = os.path.join(folder, f"*/{folder_name}/*.json")
                    matched_files = glob.glob(pattern)
                    files.extend(matched_files)

            workflow_templates_dict = (
                {}
            )  # custom_nodes folder name -> example workflow names
            for file in files:
                custom_nodes_name = os.path.basename(
                    os.path.dirname(os.path.dirname(file))
                )
                workflow_name = os.path.splitext(os.path.basename(file))[0]
                workflow_templates_dict.setdefault(custom_nodes_name, []).append(
                    workflow_name
                )
            return web.json_response(workflow_templates_dict)

        # Serve workflow templates from custom nodes.
        for module_name, module_dir in loadedModules:
            for folder_name in example_workflow_folder_names:
                workflows_dir = os.path.join(module_dir, folder_name)

                if os.path.exists(workflows_dir):
                    if folder_name != "example_workflows":
                        logging.debug(
                            "Found example workflow folder '%s' for custom node '%s', consider renaming it to 'example_workflows'",
                            folder_name, module_name)

                    webapp.add_routes(
                        [
                            web.static(
                                "/api/workflow_templates/" + module_name, workflows_dir
                            )
                        ]
                    )

        @routes.get("/i18n")
        async def get_i18n(request):
            """Returns translations from all custom nodes' locales folders."""
            return web.json_response(self.build_translations())
