"""ComfyDL_UI patches for the externally installed ComfyUI frontend package.

The frontend is a pip dependency (``comfyui-frontend-package``), so it lives
outside this repository and every hand edit to it is lost the next time the
package is upgraded. Everything ComfyDL_UI changes in the UI is applied from
here instead, at startup, right after the web root has been resolved and before
anything is served.

Each patch locates its target asset by a *content marker* (never by the hashed
asset file name) and rewrites that file in place:

  * default locale is English (``Comfy.Locale``)
  * Nodes 2.0 (``Comfy.VueNodes.Enabled``) is enabled by default
  * the Settings -> About panel gets a ComfyDL_UI section plus a separate
    upstream section that keeps the original ComfyUI links
  * the ``TENSOR`` slot colour (lemon green) in the six theme palettes

Patches are idempotent: one whose marker is already present does nothing. They
are also non-fatal: if a future frontend renames or rewrites a target, the patch
logs a warning and the server keeps booting with the upstream behaviour.
"""

import logging
import re
from pathlib import Path

TAG = "[ComfyDL_UI] frontend"

# Repositories referenced by the About section.
PROJECT_REPO = "https://github.com/Cynthia-lxx/ComfyDL_UI"
UPSTREAM_REPO = "https://github.com/Comfy-Org/ComfyUI"

TENSOR_SLOT_COLOUR = "#C6FF00"

# Marker comments written into the patched assets: they document who changed the
# file and double as the idempotency check.
LOCALE_MARK = "/*ComfyDL_UI:locale*/"
NODES2_MARK = "/*ComfyDL_UI:nodes2*/"
TENSOR_MARK = "TENSOR:`" + TENSOR_SLOT_COLOUR + "`"

# The About injection is delimited on both sides so a later revision of the
# section can be applied by removing the previous one first.
ABOUT_BEGIN = "/*ComfyDL_UI:about:begin*/"
ABOUT_END = "/*ComfyDL_UI:about:end*/"


def apply_frontend_patches(web_root: str) -> None:
    """Apply every ComfyDL_UI patch to the frontend under ``web_root``.

    ``web_root`` is the already resolved frontend directory, so an alternative
    frontend selected with ``--front-end-root`` is patched just like the one
    shipped in the pip package.
    """
    assets = Path(web_root) / "assets"
    if not assets.is_dir():
        logging.warning(f"{TAG} patches skipped: no assets directory in {web_root}")
        return

    applied = []
    for name, patch in (
        ("default locale", _patch_default_locale),
        ("Nodes 2.0 default", _patch_nodes2_default),
        ("about panel", _patch_about_panel),
        ("TENSOR slot colour", _patch_tensor_slot_colour),
    ):
        try:
            if patch(assets):
                applied.append(name)
        except Exception as exc:
            logging.warning(f"{TAG} {name} not applied: {exc}")

    # One summary line per startup, so an operator can tell a fresh install (some
    # patches just applied) from a warm one (everything was already in place).
    if applied:
        logging.info(f"{TAG} applied {len(applied)} patch(es): {', '.join(applied)}")
    else:
        logging.info(f"{TAG} already up to date, nothing to apply")


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _locate(assets: Path, pattern: str, marker: str) -> Path:
    """Return the asset matching ``pattern`` that contains ``marker``."""
    for path in sorted(assets.glob(pattern)):
        if marker in _text(path):
            return path
    raise FileNotFoundError(f"no {pattern} containing {marker!r}")


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one occurrence of {old[:60]!r}, found {count}")
    return text.replace(old, new)


def _object_end(text: str, brace: int) -> int:
    """Return the index just past the ``}`` closing the object literal at ``brace``."""
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("unbalanced object literal")


def _patch_default_locale(assets: Path) -> bool:
    """Make English the locale the frontend starts with.

    ``getDefaultLocale()`` feeds both the initial vue-i18n locale and the
    ``Comfy.Locale`` setting default, so overriding it covers both paths.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "i18n-*.js", "function getDefaultLocale(){")
    text = _text(path)
    if LOCALE_MARK in text:
        logging.debug(f"{TAG} default locale already applied")
        return False

    text = _replace_once(
        text,
        "function getDefaultLocale(){return resolveSupportedLocale(navigator.languages)}",
        f"function getDefaultLocale(){{{LOCALE_MARK}return`en`}}",
    )
    _write(path, text)
    logging.info(f"{TAG} default locale set to 'en' in {path.name}")
    return True


def _patch_nodes2_default(assets: Path) -> bool:
    """Enable the modern (Vue) node rendering by default.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "GraphView-*.js", "id:`Comfy.VueNodes.Enabled`")
    text = _text(path)
    if NODES2_MARK in text:
        logging.debug(f"{TAG} Nodes 2.0 default already applied")
        return False

    setting = text.index("id:`Comfy.VueNodes.Enabled`")
    brace = text.rindex("{", 0, setting)
    end = _object_end(text, brace)

    definition = _replace_once(text[brace:end], "defaultValue:!1", f"defaultValue:!0{NODES2_MARK}")
    definition = _replace_once(
        definition,
        'defaultsByInstallVersion:{"1.41.0":!1}',
        'defaultsByInstallVersion:{"1.41.0":!0}',
    )
    _write(path, text[:brace] + definition + text[end:])
    logging.info(f"{TAG} Nodes 2.0 enabled by default in {path.name}")
    return True


def _patch_tensor_slot_colour(assets: Path) -> bool:
    """Colour the ``TENSOR`` slot in every theme palette.

    Only the six theme tables count: their ``node_slot`` value starts with a real
    key, while the palette-merge expression ``node_slot:{...t.colors.node_slot``
    starts with ``...`` and must be left alone.

    Returns ``True`` when the asset was rewritten, ``False`` when the marker was
    already there.
    """
    path = _locate(assets, "settingStore-*.js", "node_slot:{")
    text = _text(path)
    if TENSOR_MARK in text:
        logging.debug(f"{TAG} TENSOR slot colour already applied")
        return False

    marker = "node_slot:{"
    insert = f",TENSOR:`{TENSOR_SLOT_COLOUR}`"
    out = []
    pos = 0
    search = 0
    tables = 0
    while True:
        start = text.find(marker, search)
        if start < 0:
            break
        body = start + len(marker)
        search = body
        if text[body:body + 3] == "...":
            continue
        end = text.index("}", body)
        out.append(text[pos:end])
        out.append(insert)
        pos = end
        tables += 1
    out.append(text[pos:])

    if tables != 6:
        raise ValueError(f"expected 6 theme palettes, found {tables}")
    _write(path, "".join(out))
    logging.info(f"{TAG} TENSOR slot colour added to {tables} palettes in {path.name}")
    return True


def _strip_about_section(text: str) -> str:
    """Drop a previously injected About section.

    Two markers delimit the injection, so the removal stays exact no matter how
    the section wording changes between revisions. This is what makes the patch
    re-applicable instead of append-only.
    """
    return re.sub(
        re.escape(ABOUT_BEGIN) + ".*?" + re.escape(ABOUT_END),
        "",
        text,
        flags=re.DOTALL,
    )


def _patch_about_panel(assets: Path) -> bool:
    """Add the ComfyDL_UI section and the separate upstream section to About.

    The panel is a compiled Vue chunk without any extension point, so the sections
    are injected into its render function. The minified helper names are read out
    of the chunk itself, which keeps the patch working when a frontend upgrade
    renames them.

    A previous injection is stripped first, so improving the wording only means
    editing this function: the new section replaces the old one on the next
    startup. Returns ``True`` when the asset was rewritten.
    """
    path = _locate(assets, "AboutPanel-*.js", "g.about")
    original = _text(path)
    text = _strip_about_section(original)

    heading = re.search(r"(\w+)\(`h2`,(\w+),(\w+)\((\w+)\.\$t\(`g\.about`\)\),1\),", text)
    if heading is None:
        raise ValueError("about heading not found")
    element, display, context = heading.group(1), heading.group(3), heading.group(4)

    stats = re.search(r"(\w+)\((\w+)\)\.systemStats", text)
    if stats is None:
        raise ValueError("system stats store not found")
    version = f"{stats.group(1)}({stats.group(2)}).systemStats?.system?.comfyui_version??``"

    def link(url: str, key: str) -> str:
        return (
            f"{element}(`a`,{{href:`{url}`,target:`_blank`,rel:`noopener noreferrer`,"
            f"class:`text-sm`}},{display}({context}.$t(`comfydlAbout.{key}`)),1)"
        )

    def paragraph(key: str) -> str:
        return (
            f"{element}(`div`,{{class:`text-sm opacity-70`}},"
            f"{display}({context}.$t(`comfydlAbout.{key}`)),1)"
        )

    links = (
        f"{element}(`div`,{{class:`flex flex-wrap gap-x-4`}},["
        f"{link(PROJECT_REPO, 'projectRepo')},{link(UPSTREAM_REPO, 'upstreamRepo')}"
        "])"
    )
    section = (
        f"{element}(`div`,{{class:`cdl-about space-y-2`}},["
        f"{element}(`div`,{{class:`font-medium`}},{display}(`ComfyDL_UI v`)+({version}),1),"
        f"{paragraph('summary')},{paragraph('notOriginal')},{links}"
        "]),"
    )
    upstream_heading = (
        f"{element}(`h2`,{{class:`mb-2 text-2xl font-semibold`}},"
        f"{display}({context}.$t(`comfydlAbout.upstreamHeading`)),1),"
    )
    # Both markers wrap everything that gets injected, the upstream heading
    # included: anything left outside them would be added again on every startup.
    injection = f"{ABOUT_BEGIN}{section}{upstream_heading}{ABOUT_END}"

    patched = _replace_once(text, heading.group(0), heading.group(0) + injection)
    # The first badge reports the version of the running application, which is
    # this fork, so it must not be labelled as the upstream ComfyUI release. A
    # second run finds it already renamed, which is fine.
    patched, renamed = re.subn(r"label:`ComfyUI \$\{[^}]+\}`", "label:`ComfyUI`", patched)
    if renamed == 0 and "label:`ComfyUI`" not in patched:
        raise ValueError("ComfyUI version badge not found")

    if patched == original:
        logging.debug(f"{TAG} about panel already up to date")
        return False
    _write(path, patched)
    logging.info(f"{TAG} about panel extended in {path.name}")
    return True
