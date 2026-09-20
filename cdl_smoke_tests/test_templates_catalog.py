"""Smoke tests for the ``/templates`` catalog overlay (``app/template_catalog.py``).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_templates_catalog.py

Why this script exists
----------------------
The dehydrated ComfyDL build keeps only a small subset of the upstream node
registry, so most packaged workflow templates reference node classes that no
longer exist.  ``app/template_catalog.py`` rewrites the packaged index at
request time (drops dead templates and emptied categories, injects the
repository-owned "ComfyDL Examples" category).  These tests pin that behaviour
against the *currently installed* ``comfyui-workflow-templates`` package:

* the dead-template scan finds the expected survivor set;
* the curated index is schema-shaped and contains the injected category;
* overlay asset resolution works for workflows/thumbnails and rejects
  traversal / archived content;
* the localized and MCP index variants are transformed correctly;
* the two shipped example workflows reference only registered node types.

Note: the survivor set depends on the installed package version; after
upgrading ``comfyui-workflow-templates`` this script may legitimately need its
expectations refreshed.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


# ---------------------------------------------------------------------------
# Initialize the node registry exactly like the dehydrated host does.

import nodes  # noqa: E402

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
REGISTRY = set(nodes.NODE_CLASS_MAPPINGS)

from app import template_catalog as tc  # noqa: E402
from app.frontend_management import FrontendManager  # noqa: E402

# ---------------------------------------------------------------------------
# T1: packaged catalog discovery

templates_dir = tc.packaged_templates_dir()
check(
    "T1 packaged templates dir discovered",
    templates_dir is not None and (templates_dir / "index.json").is_file(),
    str(templates_dir),
)

raw_index = json.loads((templates_dir / "index.json").read_text(encoding="utf-8-sig"))
all_names = [
    t["name"] for cat in raw_index for t in cat.get("templates", []) if t.get("name")
]
KNOWN_SURVIVORS = {
    "basic_mask_operations_and_compositing",
    "utility_image_stitch",
}

# ---------------------------------------------------------------------------
# T2: dead-template scan

dead = tc.compute_dead_templates(raw_index, templates_dir)
survivors = set(all_names) - dead
check(
    "T2a dead scan: known survivors survive",
    KNOWN_SURVIVORS <= survivors,
    f"missing {sorted(KNOWN_SURVIVORS - survivors)}",
)
check(
    "T2b dead scan: vast majority of packaged templates are dead",
    len(dead) > 0.8 * len(all_names),
    f"dead={len(dead)} total={len(all_names)}",
)
check(
    "T2c dead scan: survivors reference no missing node types",
    all(
        not tc._missing_node_types(
            json.loads((templates_dir / f"{name}.json").read_text(encoding="utf-8-sig")),
            frozenset(REGISTRY),
        )
        for name in survivors
    ),
    f"checked {len(survivors)} survivors",
)

# ---------------------------------------------------------------------------
# T3: curated index structure (English default)

curated = tc.build_curated_index(raw_index)
check("T3a curated: non-empty", len(curated) > 0)
check(
    "T3b curated: ComfyDL category injected first",
    curated
    and curated[0].get("moduleName") == "ComfyDL"
    and curated[0].get("isEssential") is True,
    str(curated[0].get("title") if curated else None),
)
comfydl_names = {t["name"] for t in curated[0]["templates"]}
check(
    "T3c curated: both example workflows listed",
    comfydl_names
    == {"language_model_train_and_chat", "language_model_load_and_chat"},
    str(sorted(comfydl_names)),
)
check(
    "T3c2 curated: names follow the official schema pattern (no spaces)",
    all(re.fullmatch(r"[a-zA-Z0-9._-]+", t["name"]) for t in curated[0]["templates"]),
    "official schema requires ^[a-zA-Z0-9._-]+$",
)
other_titles = [c.get("title") for c in curated[1:]]
check(
    "T3d curated: only expected upstream category survives (Node Basics)",
    other_titles == ["Node Basics"],
    str(other_titles),
)
check(
    "T3e curated: no category emptied by filtering",
    all(c.get("templates") for c in curated),
)
check(
    "T3f curated: no dead template leaks into the catalog",
    all(t["name"] not in dead for c in curated for t in c["templates"]),
)

# ---------------------------------------------------------------------------
# T4: schema-shape validation (subset of index.schema.json)

for cat in curated:
    ok_keys = (
        re.fullmatch(r"[a-zA-Z0-9_-]+", cat.get("moduleName", ""))
        and isinstance(cat.get("title"), str)
        and isinstance(cat.get("templates"), list)
    )
    if not ok_keys:
        check("T4 category shape", False, json.dumps(cat)[:200])
        break
    for t in cat["templates"]:
        ok_t = (
            isinstance(t.get("name"), str)
            and t.get("name") + ".json"  # names double as filename stems
            and t.get("mediaType") in {"image", "video", "audio", "3d"}
            and re.fullmatch(r"[a-zA-Z0-9]+", t.get("mediaSubtype", ""))
            and isinstance(t.get("description"), str)
            and t.get("description")
        )
        if not ok_t:
            check("T4 template shape", False, json.dumps(t)[:200])
            break
    else:
        continue
    break
else:
    check("T4 schema shape (categories + templates)", True)

# ---------------------------------------------------------------------------
# T5: overlay asset resolution

for name in sorted(comfydl_names):
    check(
        f"T5a overlay workflow resolvable: {name}",
        (tc.resolve_overlay_asset(f"{name}.json") or Path()).is_file(),
    )
    thumb = tc.resolve_overlay_asset(f"{name}-1.jpg")
    check(
        f"T5b overlay thumbnail resolvable: {name}-1.jpg",
        (thumb or Path()).is_file()
        and thumb.is_relative_to(tc._EXAMPLE_WORKFLOWS_DIR.resolve()),
        str(thumb),
    )
check("T5c overlay: index.json is not an overlay asset",
      tc.resolve_overlay_asset("index.json") is None)
check("T5d overlay: traversal rejected",
      tc.resolve_overlay_asset("../app/server.py") is None
      and tc.resolve_overlay_asset("..\\app\\server.py") is None)
check("T5e overlay: archived workflows unreachable",
      tc.resolve_overlay_asset("_archived/anything.json") is None)

# ---------------------------------------------------------------------------
# T6: index payload serving (default / localized / MCP / passthrough)

asset_map = FrontendManager.template_asset_map()


def read_source(rel_path: str):
    path = asset_map.get(rel_path) if asset_map else None
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8-sig")


payload = tc.curated_index_payload("index.json", read_source)
check("T6a payload: index.json rewritten as JSON", isinstance(payload, str))
if payload:
    served = json.loads(payload)
    check(
        "T6b payload: served index starts with ComfyDL category",
        served and served[0].get("moduleName") == "ComfyDL",
    )

zh_path = templates_dir / "index.zh.json"
if zh_path.is_file():
    zh_payload = tc.curated_index_payload("index.zh.json", read_source)
    zh_served = json.loads(zh_payload) if zh_payload else None
    check(
        "T6c payload: zh locale keeps packaged translations + zh ComfyDL title",
        zh_served is not None and zh_served[0].get("title") == "ComfyDL 示例",
        str(zh_served[0].get("title")) if zh_served else "no payload",
    )
else:
    check("T6c payload: zh locale file absent (skipped)", True)

mcp_payload = tc.curated_index_payload("index.mcp.json", read_source)
check("T6d payload: index.mcp.json rewritten", isinstance(mcp_payload, str))
if mcp_payload:
    mcp_served = json.loads(mcp_payload)
    check(
        "T6e payload: MCP index filtered without ComfyDL injection",
        all(c.get("moduleName") != "ComfyDL" for c in mcp_served)
        and all(
            t.get("name") not in dead for c in mcp_served for t in c.get("templates", [])
        ),
    )

check(
    "T6f payload: logo/schema indexes pass through untouched",
    tc.curated_index_payload("index_logo.json", read_source) is None
    and tc.curated_index_payload("index.schema.json", read_source) is None,
)

# ---------------------------------------------------------------------------
# T7: handler construction

handler = FrontendManager.template_asset_handler()
check("T7a handler: built from asset map", handler is not None)
check("T7b handler: None when no data source",
      tc.build_handler(None, None) is None)

# ---------------------------------------------------------------------------
# T8: shipped example workflows only reference registered node types

for name in sorted(comfydl_names):
    wf_path = tc._EXAMPLE_WORKFLOWS_DIR / f"{name}.json"
    workflow = json.loads(wf_path.read_text(encoding="utf-8-sig"))
    missing = tc._missing_node_types(workflow, frozenset(REGISTRY))
    check(f"T8 example workflow runs on this registry: {name}",
          not missing, f"missing {sorted(missing)}")

# ---------------------------------------------------------------------------
# T9: full HTTP route-layer integration (this is what actually broke the
# Templates panel before: the frontend fetches raw URLs, browsers encode
# spaces as %20, and aiohttp's {path:.*} match info only unquotes %2F/%25,
# so the handler must unquote defensively - see build_handler).

async def _route_checks() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    handler = FrontendManager.template_asset_handler()
    app = web.Application()
    app.router.add_get("/templates/{path:.*}", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        async def hit(url, expect_status):
            resp = await client.get(url)
            body = await resp.read()
            return resp.status == expect_status, (
                f"{url} -> {resp.status} (expect {expect_status}), "
                f"ct={resp.headers.get('Content-Type', '')}, {len(body)}B"
            ), body

        ok, info, body = await hit(
            "/templates/language_model_train_and_chat.json", 200)
        check("T9a route: our workflow JSON serves 200", ok, info)
        if ok:
            check("T9a2 route: served body is a UI-format workflow",
                  b'"nodes"' in body and b'"links"' in body)

        # Underscore sent percent-encoded: aiohttp leaves %5F untouched, the
        # handler's unquote must decode it back for the lookup to succeed.
        ok, info, _ = await hit(
            "/templates/language%5Fmodel%5Ftrain%5Fand%5Fchat.json", 200)
        check("T9b route: handler unquotes percent escapes (%5F)", ok, info)

        ok, info, _ = await hit(
            "/templates/language_model_load_and_chat-1.jpg", 200)
        check("T9c route: load-and-chat thumbnail 200 (mapped art)", ok, info)

        ok, info, _ = await hit(
            "/templates/language_model_train_and_chat-1.jpg", 200)
        check("T9d route: train-and-chat thumbnail 200", ok, info)

        ok, info, body = await hit("/templates/index.json", 200)
        check("T9e route: rewritten index.json 200", ok, info)
        if ok:
            served = json.loads(body)
            check("T9e2 route: index starts with ComfyDL category",
                  served and served[0].get("moduleName") == "ComfyDL")

        ok, info, _ = await hit(
            "/templates/basic_mask_operations_and_compositing.json", 200)
        check("T9f route: official survivor serves 200 via asset map", ok, info)

        ok, info, _ = await hit(
            "/templates/language%20model_train_and_chat.json", 404)
        check("T9g route: wrongly-encoded (space) name 404s cleanly", ok, info)

        ok, info, _ = await hit("/templates/unknown_name.json", 404)
        check("T9h route: unknown template 404s", ok, info)

        ok, info, _ = await hit("/templates/..%2Fapp%2Fserver.py", 404)
        check("T9i route: traversal attempt rejected", ok, info)
    finally:
        await client.close()


asyncio.run(_route_checks())

# ---------------------------------------------------------------------------
# T10: graph integrity of the shipped workflows (ids/links/inputs/outputs
# cross-references must be consistent - a stale hand-edit here is what
# "template opens as a broken graph" looks like)

for name in sorted(comfydl_names):
    d = json.loads(
        (tc._EXAMPLE_WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8-sig")
    )
    link_ids = {l[0] for l in d["links"]}
    node_ids = {n["id"] for n in d["nodes"]}
    problems = []
    for l in d["links"]:
        if l[1] not in node_ids or l[3] not in node_ids:
            problems.append(f"link {l[0]} endpoint not a node")
    for n in d["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("link") is not None and inp["link"] not in link_ids:
                problems.append(f"node {n['id']} input dangling link")
        for out in n.get("outputs", []):
            for lid in out.get("links") or []:
                if lid not in link_ids:
                    problems.append(f"node {n['id']} output dangling link")
        for key in list(n) + list(n.get("properties", {})):
            if key.startswith("__"):
                problems.append(f"node {n['id']} dunder key {key}")
    check(f"T10 graph integrity: {name}", not problems,
          "; ".join(sorted(set(problems))[:4]))

# ---------------------------------------------------------------------------
# Report

failed = [(n, d) for n, ok, d in _RESULTS if not ok]
for name, ok, detail in _RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
if failed:
    sys.exit(1)
