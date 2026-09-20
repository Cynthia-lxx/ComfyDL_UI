"""ComfyDL overlay for the workflow-template catalog served at ``/templates``.

What this module does
---------------------
The ComfyUI frontend fetches ``/templates/index.json`` (or a localized
``index.<locale>.json``) that ships inside the ``comfyui-workflow-templates``
PyPI package and renders the Templates sidebar from it.  The ComfyDL
dehydrated build keeps only a small subset of the upstream node registry, so
most packaged templates reference node classes that no longer exist and would
load as broken workflows.  Instead of editing files inside ``site-packages``
(lost on every package upgrade), this module rewrites the catalog **at request
time**:

* Templates that reference node types missing from
  ``nodes.NODE_CLASS_MAPPINGS`` are dropped.  Frontend-only types
  (``Note``/``MarkdownNote``/``PrimitiveNode``/``Reroute``) and node types
  that resolve to subgraph definitions embedded in the same workflow file do
  not count as missing.
* Categories left without a single surviving template are dropped.
* A "ComfyDL Examples" category is injected at the top, serving the example
  workflows that live in ``comfydl/example_workflows/`` straight from this
  repository (workflow JSON plus thumbnail JPG), so newly added example
  workflows only need an entry in ``_COMFYDL_CATEGORY`` below.

Everything is computed lazily on the first ``/templates/index*.json`` request
and cached for the lifetime of the process; the scan is idempotent against
package upgrades (a refreshed ``comfyui-workflow-templates`` simply gets
re-scanned on the next server start).
"""

from __future__ import annotations

import importlib.resources
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple
from urllib.parse import unquote

__all__ = [
    "build_handler",
    "build_curated_index",
    "compute_dead_templates",
    "resolve_overlay_asset",
    "curated_index_payload",
    "packaged_templates_dir",
]

# ---------------------------------------------------------------------------
# Constants

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_WORKFLOWS_DIR = _REPO_ROOT / "comfydl" / "example_workflows"

# Node types implemented by the frontend itself; they never resolve through
# the backend registry and must not count as "missing".
_FRONTEND_ONLY_TYPES = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "Reroute"})

# Locales the frontend may request via ``index.<locale>.json``.
_KNOWN_LOCALES = frozenset({"zh", "zh-TW", "fr", "ko", "ru", "es", "ja", "ar"})

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Thumbnails actually shipped in ``comfydl/example_workflows``.  By default the
# frontend requests ``<name>-1.<mediaSubtype>``, so cover art is *renamed to
# that URL form* (e.g. ``language_model_train_and_chat-1.jpg``) and this
# mapping is only needed when a template reuses another template's art.
# "language_model_load_and_chat" reuses the training workflow's screenshot
# until a dedicated one is captured.
_THUMBNAIL_FILENAMES: Dict[str, str] = {
    "language_model_load_and_chat": "language_model_train_and_chat-1.jpg",
}

# The injected category.  ``name`` values MUST equal the workflow filename
# stems inside ``comfydl/example_workflows`` AND match the official index
# schema pattern ``^[a-zA-Z0-9._-]+$`` (no spaces, ASCII only).  This is not
# just CI pedantry: the frontend builds raw fetch URLs
# ``/templates/${name}.json`` / ``/templates/${name}-1.${mediaSubtype}``
# without encoding, and while browsers percent-encode spaces to ``%20``,
# aiohttp's dynamic-route match info only unquotes ``%2F``/``%25``
# (``web_urldispatcher._unquote_path_safe``) - a literal ``%20`` then fails
# every lookup and the template silently fails to load (see the handler,
# which additionally unquotes defensively).
_COMFYDL_CATEGORY: Dict[str, Any] = {
    "moduleName": "ComfyDL",
    "title": "ComfyDL Examples",
    "isEssential": True,
    "icon": "icon-[lucide--graduation-cap]",
    "templates": [
        {
            "name": "language_model_train_and_chat",
            "title": "Language Model: Train and Chat",
            "description": (
                "Train a small transformer language model from scratch on raw "
                "text, then chat with it - an end-to-end NLP teaching workflow "
                "built entirely from ComfyDL nodes."
            ),
            "mediaType": "image",
            "mediaSubtype": "jpg",
            "tags": ["language model", "transformer", "training", "nlp"],
            "date": "2026-09-20",
            "openSource": True,
        },
        {
            "name": "language_model_load_and_chat",
            "title": "Language Model: Load and Chat",
            "description": (
                "Load a trained ComfyDL language model checkpoint and generate "
                "text with it - the lightweight companion to the training "
                "workflow."
            ),
            "mediaType": "image",
            "mediaSubtype": "jpg",
            "tags": ["language model", "inference", "nlp"],
            "date": "2026-09-20",
            "openSource": True,
        },
    ],
}

# Per-locale overrides for the injected category (title only; entries keep
# English descriptions outside zh until more translations are contributed).
_CATEGORY_TITLE_OVERRIDES: Dict[str, str] = {
    "zh": "ComfyDL 示例",
}
_CATEGORY_DESCRIPTION_OVERRIDES: Dict[str, Dict[str, str]] = {
    "zh": {
        "language_model_train_and_chat": (
            "用 ComfyDL 节点从零训练一个小型 Transformer 语言模型，并与它对话"
            "——端到端的 NLP 教学工作流。"
        ),
        "language_model_load_and_chat": (
            "加载训练好的 ComfyDL 语言模型权重并生成文本——训练工作流的轻量配套。"
        ),
    },
}

# ---------------------------------------------------------------------------
# Cached state (process-wide)

_LOCK = threading.Lock()
_DEAD_TEMPLATES: Optional[FrozenSet[str]] = None
_CACHED_INDEXES: Dict[str, List[Any]] = {}

# ---------------------------------------------------------------------------
# Packaged catalog discovery


def packaged_templates_dir() -> Optional[Path]:
    """Locate the installed ``templates`` data directory of the PyPI package.

    Tries the JSON data package first, then falls back to the meta package's
    asset API.  Returns ``None`` when neither is importable (the overlay then
    serves only repository-owned entries and the caller decides the fallback).
    """
    try:
        import comfyui_workflow_templates_json

        candidate = Path(
            str(importlib.resources.files(comfyui_workflow_templates_json))
        ) / "templates"
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    try:
        from comfyui_workflow_templates import get_asset_path, iter_templates

        for entry in iter_templates():
            for asset in entry.assets:
                if asset.filename == "index.json":
                    return Path(get_asset_path(entry.template_id, "index.json")).parent
    except Exception:
        pass
    return None


def _node_registry() -> FrozenSet[str]:
    """Return the backend node registry keys, or an empty set if unavailable."""
    try:
        import nodes

        return frozenset(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Dead-template detection


def _collect_referenced_types(workflow: Mapping[str, Any]) -> Tuple[set, set]:
    """Return (real node types, subgraph definition ids) referenced by a workflow.

    Handles both the UI format (``nodes`` list plus embedded ``definitions``
    or ``extra.definitions`` subgraphs) and the API format (top-level mapping
    whose values carry ``class_type``).
    """
    types: set = set()
    subgraph_ids: set = set()

    node_lists: List[list] = []
    if isinstance(workflow.get("nodes"), list):
        node_lists.append(workflow["nodes"])
    definitions = workflow.get("definitions")
    if not isinstance(definitions, Mapping):
        extra = workflow.get("extra")
        definitions = extra.get("definitions") if isinstance(extra, Mapping) else None
    if isinstance(definitions, Mapping) and isinstance(
        definitions.get("subgraphs"), list
    ):
        for subgraph in definitions["subgraphs"]:
            if isinstance(subgraph, Mapping):
                if subgraph.get("id"):
                    subgraph_ids.add(subgraph["id"])
                if isinstance(subgraph.get("nodes"), list):
                    node_lists.append(subgraph["nodes"])

    for node_list in node_lists:
        for node in node_list:
            if isinstance(node, Mapping) and node.get("type"):
                types.add(node["type"])

    if not node_lists:  # API format fallback
        for value in workflow.values():
            if isinstance(value, Mapping) and value.get("class_type"):
                types.add(value["class_type"])

    return types, subgraph_ids


def _missing_node_types(
    workflow: Mapping[str, Any], registry: FrozenSet[str]
) -> set:
    """Node types referenced by ``workflow`` that the backend cannot provide."""
    types, subgraph_ids = _collect_referenced_types(workflow)
    missing = set()
    for node_type in types:
        if node_type in registry or node_type in _FRONTEND_ONLY_TYPES:
            continue
        if _UUID_RE.match(node_type):
            # Subgraph instance: fine when its definition ships in the same
            # file, broken when the definition is absent.
            if node_type not in subgraph_ids:
                missing.add(node_type)
            continue
        missing.add(node_type)
    return missing


def compute_dead_templates(
    index_entries: List[Mapping[str, Any]], templates_dir: Path
) -> FrozenSet[str]:
    """Names of packaged templates that cannot run against this registry.

    A template is dead when its workflow file is missing/unreadable or when it
    references at least one node type absent from the registry.  Returns an
    empty set when the registry has not been initialized yet (fail-open).
    """
    registry = _node_registry()
    if not registry:
        return frozenset()

    dead: set = set()
    for category in index_entries:
        if not isinstance(category, Mapping):
            continue
        for template in category.get("templates", []):
            if not isinstance(template, Mapping):
                continue
            name = template.get("name")
            if not name:
                continue
            workflow_path = templates_dir / f"{name}.json"
            if not workflow_path.is_file():
                dead.add(name)
                continue
            try:
                data = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
            except Exception:
                dead.add(name)
                continue
            if not isinstance(data, Mapping):
                dead.add(name)
                continue
            if _missing_node_types(data, registry):
                dead.add(name)
    return frozenset(dead)


def _dead_templates() -> FrozenSet[str]:
    """Lazily computed, process-cached set of dead packaged template names."""
    global _DEAD_TEMPLATES
    with _LOCK:
        if _DEAD_TEMPLATES is not None:
            return _DEAD_TEMPLATES
        templates_dir = packaged_templates_dir()
        if templates_dir is None:
            _DEAD_TEMPLATES = frozenset()
            return _DEAD_TEMPLATES
        try:
            index_data = json.loads(
                (templates_dir / "index.json").read_text(encoding="utf-8-sig")
            )
        except Exception:
            _DEAD_TEMPLATES = frozenset()
            return _DEAD_TEMPLATES
        _DEAD_TEMPLATES = compute_dead_templates(index_data, templates_dir)
        return _DEAD_TEMPLATES


# ---------------------------------------------------------------------------
# Index rewriting


def _comfydl_category(locale: Optional[str]) -> Dict[str, Any]:
    """Deep-copied injected category with per-locale title/description applied."""
    import copy

    category = copy.deepcopy(_COMFYDL_CATEGORY)
    if locale:
        title = _CATEGORY_TITLE_OVERRIDES.get(locale)
        if title:
            category["title"] = title
        descriptions = _CATEGORY_DESCRIPTION_OVERRIDES.get(locale, {})
        for template in category["templates"]:
            description = descriptions.get(template["name"])
            if description:
                template["description"] = description
    return category


def build_curated_index(
    index_data: List[Mapping[str, Any]],
    locale: Optional[str] = None,
    inject_comfydl: bool = True,
) -> List[Mapping[str, Any]]:
    """Rewrite a packaged template index for the dehydrated registry.

    Drops dead templates, drops categories emptied by the filter and (unless
    ``inject_comfydl`` is False, used for the machine-facing MCP index) inserts
    the ComfyDL Examples category first.
    """
    dead = _dead_templates()
    curated: List[Mapping[str, Any]] = []
    for category in index_data:
        if not isinstance(category, Mapping):
            continue
        surviving = [
            template
            for template in category.get("templates", [])
            if isinstance(template, Mapping)
            and template.get("name") not in dead
        ]
        if not surviving:
            continue
        rewritten = dict(category)
        rewritten["templates"] = surviving
        curated.append(rewritten)
    if inject_comfydl:
        curated.insert(0, _comfydl_category(locale))
    return curated


def _index_request_locale(rel_path: str) -> str:
    """Classify an ``index*.json`` request filename.

    Returns:
        ``"skip"``  -- not an index payload (logo/schema files, etc.);
        ``"default"`` -- the English ``index.json``;
        ``"mcp"``   -- ``index.mcp.json``;
        a locale string for ``index.<locale>.json``.
    """
    if rel_path == "index.json":
        return "default"
    if rel_path == "index.mcp.json":
        return "mcp"
    match = re.fullmatch(r"index\.([A-Za-z][A-Za-z-]*)\.json", rel_path)
    if match and match.group(1) in _KNOWN_LOCALES:
        return match.group(1)
    return "skip"


def curated_index_payload(
    rel_path: str, read_source: Callable[[str], Optional[str]]
) -> Optional[str]:
    """Return the JSON text served for an ``index*.json`` request, if any.

    ``read_source`` must return the packaged file content for the given
    relative path (or ``None`` when the file does not exist).  Returns
    ``None`` for non-index paths so the caller falls through to plain asset
    serving / 404.  Results are cached per request path.
    """
    locale = _index_request_locale(rel_path)
    if locale == "skip":
        return None
    with _LOCK:
        if rel_path in _CACHED_INDEXES:
            return json.dumps(_CACHED_INDEXES[rel_path])
    source = read_source(rel_path)
    if source is None:
        return None
    try:
        index_data = json.loads(source)
    except Exception:
        return None
    if locale == "mcp":
        curated = build_curated_index(index_data, inject_comfydl=False)
    elif locale == "default":
        curated = build_curated_index(index_data, locale=None)
    else:
        curated = build_curated_index(index_data, locale=locale)
    with _LOCK:
        _CACHED_INDEXES[rel_path] = curated
    return json.dumps(curated)


# ---------------------------------------------------------------------------
# Repository-owned asset resolution


def resolve_overlay_asset(rel_path: str) -> Optional[Path]:
    """Resolve a repository-owned template asset (workflow JSON or thumbnail).

    Only exact names registered through the injected category are served;
    anything else (including ``_archived/`` content and path traversal
    attempts) returns ``None``.
    """
    known_names = {template["name"] for template in _COMFYDL_CATEGORY["templates"]}
    # Workflow JSON: "<name>.json"
    stem, dot, ext = rel_path.rpartition(".")
    if dot and ext == "json" and stem in known_names:
        candidate = _EXAMPLE_WORKFLOWS_DIR / rel_path
        if candidate.is_file():
            return candidate
    # Thumbnail: "<name>-1.<ext>" (frontend default module URL scheme) mapped
    # onto the actual image file shipped in the repository.
    match = re.fullmatch(r"(.*[^-])-1\.([A-Za-z0-9]+)", rel_path)
    if match:
        template_name = match.group(1)
        if template_name in known_names:
            source = _THUMBNAIL_FILENAMES.get(template_name)
            if source:
                candidate = _EXAMPLE_WORKFLOWS_DIR / source
                if candidate.is_file():
                    return candidate
            # Fall back to a literal "<name>-1.<ext>" file when present.
            candidate = _EXAMPLE_WORKFLOWS_DIR / rel_path
            if candidate.is_file():
                return candidate
    return None


# ---------------------------------------------------------------------------
# aiohttp handler


def build_handler(
    assets: Optional[Mapping[str, str]],
    legacy_dir: Optional[str] = None,
) -> Optional[Callable]:
    """Build the ``/templates/{path:.*}`` request handler.

    Layering (first match wins):
      1. repository-owned overlay assets (ComfyDL example workflows + thumbs);
      2. request-time-rewritten ``index*.json`` payloads;
      3. packaged asset map (``assets``, filename -> absolute path);
      4. legacy static directory fallback (``legacy_dir``), traversal-guarded.

    Returns ``None`` when no data source is available at all.
    """
    if assets is None and legacy_dir is None:
        return None

    legacy_root = Path(legacy_dir).resolve() if legacy_dir else None
    templates_dir = packaged_templates_dir()

    def _read_source(rel_path: str) -> Optional[str]:
        source_path: Optional[str] = None
        if assets is not None:
            source_path = assets.get(rel_path)
        if source_path is None and legacy_root is not None:
            candidate = (legacy_root / rel_path).resolve()
            if candidate.is_relative_to(legacy_root) and candidate.is_file():
                source_path = str(candidate)
        if source_path is None and templates_dir is not None:
            candidate = templates_dir / rel_path
            if candidate.is_file():
                source_path = str(candidate)
        if source_path is None:
            return None
        try:
            return Path(source_path).read_text(encoding="utf-8-sig")
        except Exception:
            return None

    async def serve_template(request):  # noqa: ANN001 - aiohttp handler
        from aiohttp import web

        # aiohttp's dynamic-route match info only unquotes %2F and %25
        # (web_urldispatcher._unquote_path_safe), so any other percent escape
        # (space -> %20, non-ASCII, ...) arrives still encoded.  Fully decode
        # before any lookup; traversal protection below is path-based
        # (resolve + is_relative_to), not string-based, so decoding is safe.
        rel_path = unquote(request.match_info.get("path", ""))
        overlay_target = resolve_overlay_asset(rel_path)
        if overlay_target is not None:
            return web.FileResponse(overlay_target)
        payload = curated_index_payload(rel_path, _read_source)
        if payload is not None:
            return web.Response(
                text=payload, content_type="application/json", charset="utf-8"
            )
        target = assets.get(rel_path) if assets is not None else None
        if target is None and legacy_root is not None:
            candidate = (legacy_root / rel_path).resolve()
            if candidate.is_relative_to(legacy_root) and candidate.is_file():
                target = str(candidate)
        if target is None:
            raise web.HTTPNotFound()
        return web.FileResponse(target)

    return serve_template
