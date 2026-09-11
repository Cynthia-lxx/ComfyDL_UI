#!/usr/bin/env python
"""Lightweight in-process smoke tester for the ComfyDL_UI node registry.

Why this exists
---------------
Starting the ComfyUI server for every change is slow.  This script loads the
very same node set the server would load (the ``comfy_extras`` whitelist plus the
``comfydl`` submodule) *inside the current process*, then executes every
registered node once with synthesised dummy inputs.  No HTTP server, no
browser, no model download: a full pass takes seconds.

What it checks per node
-----------------------
1. the node's schema / ``INPUT_TYPES`` can be resolved;
2. a value can be synthesised for every *required* input (and for hidden inputs);
3. the node's entry point runs without raising;
4. the number of returned values matches the declared output arity.

Result vocabulary
-----------------
``PASS``  executed successfully and returned the declared arity.
``SKIP``  deliberately not executed (needs network / datasets / real files / ...);
          the reason is always printed and a SKIP never fails the run.
``FAIL``  unexpected exception, timeout, or an arity mismatch.

Usage
-----
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --filter Basic
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --categories
    penv\\Scripts\\python.exe cdl_smoke_tests\\run_smoke_test.py --filter Cdl --verbose

Exit code: ``0`` when no node FAILed, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import shutil
import sys
import tempfile
import threading
import traceback
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

# V3 nodes are executed through the same normalisation hook the engine uses.
_V3_ENTRY_POINTS = ("EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC")


class Unsupported(Exception):
    """Raised when a dummy value cannot be synthesised for an input."""


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _bootstrap(sandbox: Path) -> None:
    """Make the repo importable and keep every write inside a temporary sandbox.

    ComfyUI parses ``sys.argv`` while its modules are imported, so we hand it a
    minimal command line pointing ``input``/``output``/``temp`` at a scratch
    directory.  The process also *works* inside the sandbox (``chdir``), so nodes
    that save to a plain relative path (e.g. ``torch.save(state_dict, "model.pt")``)
    cannot litter the repository.  Everything is deleted again afterwards.
    """
    warnings.simplefilter("ignore")  # keep the report readable (matplotlib etc.)
    for sub in ("input", "output", "temp"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(sandbox)
    _seed_fixtures(sandbox)
    sys.argv = [
        str(Path(__file__).resolve()),
        "--cpu",
        "--input-directory", str(sandbox / "input"),
        "--output-directory", str(sandbox / "output"),
        "--temp-directory", str(sandbox / "temp"),
    ]


def _seed_fixtures(sandbox: Path) -> None:
    """Write the small input files that a few nodes read through a default path.

    ``CdlModelSave`` / ``CdlModelLoad`` both default their ``path`` widget to
    ``"model.pt"`` (relative to the working directory, which is the sandbox).  The
    state_dict is taken from the same dummy model :func:`_f_model` hands to those
    nodes, so the load node really exercises ``torch.load`` +
    ``load_state_dict`` instead of being skipped.
    """
    import torch

    torch.save(_f_model({}, "model").state_dict(), sandbox / "model.pt")


def _load_registry(verbose: bool) -> dict[str, type]:
    """Import the host `nodes` module and populate the global node mappings."""
    import nodes as host_nodes

    failed = asyncio.run(
        host_nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    )
    if failed:
        print(f"[smoke] WARNING: modules that failed to import: {failed}")
    elif verbose:
        print("[smoke] all built-in node modules imported cleanly")
    return dict(host_nodes.NODE_CLASS_MAPPINGS)


# --------------------------------------------------------------------------- #
# Dummy value synthesis
# --------------------------------------------------------------------------- #
def _clamp_number(value: Any, lo: Any, hi: Any) -> Any:
    """Clamp ``value`` into the ``[lo, hi]`` window, ignoring unset bounds."""
    if isinstance(lo, (int, float)) and value < lo:
        value = lo
    if isinstance(hi, (int, float)) and value > hi:
        value = hi
    return value


def _f_int(cfg: dict, name: str) -> int:
    value = cfg.get("default")
    if value is None:
        value = cfg.get("min")
    if value is None:
        value = 1
    return int(_clamp_number(int(value), cfg.get("min"), cfg.get("max")))


def _f_float(cfg: dict, name: str) -> float:
    value = cfg.get("default")
    if value is None:
        value = cfg.get("min")
    if value is None:
        value = 0.5
    return float(_clamp_number(float(value), cfg.get("min"), cfg.get("max")))


def _f_string(cfg: dict, name: str) -> str:
    """Use the declared default, so the harness mirrors "drop the node and run".

    Empty defaults are kept as-is: several nodes use ``""`` to mean "feature
    off", and a made-up placeholder would change their behaviour.
    """
    value = cfg.get("default")
    return value if isinstance(value, str) else "smoke"


def _f_bool(cfg: dict, name: str) -> bool:
    value = cfg.get("default")
    return bool(value) if value is not None else True


def _f_combo(cfg: dict, name: str) -> Any:
    options = cfg.get("options")
    if options is None:
        options = cfg.get("values")
    if callable(options):
        raise Unsupported(f"dynamic options for combo input {name!r}")
    if isinstance(options, dict):
        options = list(options.keys())
    if not options:
        raise Unsupported(f"no options for combo input {name!r}")
    values = [opt.value if isinstance(opt, Enum) else opt for opt in options]
    default = cfg.get("default")
    default = default.value if isinstance(default, Enum) else default
    return default if default in values else values[0]


def _f_tensor(cfg: dict, name: str) -> Any:
    import torch

    return torch.randn(2, 3)


def _f_image(cfg: dict, name: str) -> Any:
    import torch

    return torch.rand(1, 8, 8, 3)


def _f_mask(cfg: dict, name: str) -> Any:
    import torch

    return torch.rand(1, 8, 8)


def _f_latent(cfg: dict, name: str) -> dict[str, Any]:
    import torch

    return {"samples": torch.randn(1, 4, 8, 8)}


def _f_model(cfg: dict, name: str) -> Any:
    """A tiny ``nn.Module``: ComfyDL's ``cdlModel`` is just "any nn.Module".

    ``in_features=3`` matches the dummy ``TENSOR`` (2, 3) produced by
    :func:`_f_tensor`, so forward-pass nodes work with the synthesised pair.
    """
    import torch

    return torch.nn.Linear(3, 2)


def _f_dataloader(cfg: dict, name: str) -> Any:
    import torch

    dataset = torch.utils.data.TensorDataset(torch.randn(8, 3), torch.randn(8, 1))
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def _f_index_small(cfg: dict, name: str) -> Any:
    """Index dummy that always stays inside a 2-row table."""
    import torch

    return torch.randint(0, 2, (2, 3))


def _f_index_1d(cfg: dict, name: str) -> Any:
    """Flat index list, for nodes that iterate ids one by one."""
    import torch

    return torch.randint(0, 2, (4,))


def _f_boxes(cfg: dict, name: str) -> Any:
    """Bounding boxes: ``(num_boxes, 4)``, the shape every box helper expects."""
    import torch

    return torch.rand(3, 4)


def _f_anchors(cfg: dict, name: str) -> Any:
    """Batched anchors: ``(batch, num_anchors, 4)``."""
    import torch

    return torch.rand(1, 3, 4)


def _f_multibox_labels(cfg: dict, name: str) -> Any:
    """Detection labels: ``(batch, num_objects, 5)`` = class + 4 coordinates."""
    import torch

    return torch.rand(2, 3, 5)


def _f_cls_probs(cfg: dict, name: str) -> Any:
    """Detection scores: ``(batch, num_classes, num_anchors)``."""
    import torch

    return torch.rand(1, 3, 3)


def _f_offset_preds(cfg: dict, name: str) -> Any:
    """Flattened box offsets: ``(batch, num_anchors * 4)``."""
    import torch

    return torch.rand(1, 12)


def _f_scores(cfg: dict, name: str) -> Any:
    """One confidence score per box: ``(num_boxes,)``."""
    import torch

    return torch.rand(3)


def _f_seq3d(cfg: dict, name: str) -> Any:
    """Sequence batch: ``(batch, num_steps, num_features)``."""
    import torch

    return torch.randn(2, 4, 3)


def _f_valid_len(cfg: dict, name: str) -> Any:
    """One valid length per sequence in the batch."""
    import torch

    return torch.tensor([2, 3])


def _f_class_idx(cfg: dict, name: str) -> Any:
    """One class index per prediction row."""
    import torch

    return torch.randint(0, 3, (2,))


def _f_weight_col(cfg: dict, name: str) -> Any:
    """Weight vector shaped to matmul against the generic ``(2, 3)`` tensor."""
    import torch

    return torch.randn(3, 1)


def _f_bias_1d(cfg: dict, name: str) -> Any:
    """Bias that broadcasts against a single regression output."""
    import torch

    return torch.randn(1)


def _f_features(cfg: dict, name: str) -> Any:
    """Feature table for the array/dataset helper nodes."""
    import torch

    return torch.randn(8, 3)


def _f_values6(cfg: dict, name: str) -> Any:
    """Non-negative values for chart nodes that reject negatives."""
    import torch

    return torch.rand(6)


def _f_values3(cfg: dict, name: str) -> Any:
    """Three non-negative chart values (pie slices)."""
    import torch

    return torch.rand(3)


def _f_labels6(cfg: dict, name: str) -> str:
    """Six category labels, matching :func:`_f_values6`."""
    return "A,B,C,D,E,F"


def _f_labels3(cfg: dict, name: str) -> str:
    """Three category labels, matching :func:`_f_values3`."""
    return "A,B,C"


def _f_array(cfg: dict, name: str) -> Any:
    """A plain list of numbers (core ``ARRAY`` inputs)."""
    return [1.0, 2.0, 3.0]


def _f_dict(cfg: dict, name: str) -> Any:
    """A small mapping (core ``DICT`` inputs)."""
    return {"a": 1}


def _f_color(cfg: dict, name: str) -> str:
    """A hex colour string (core ``COLOR`` inputs)."""
    return "#FF0000"


def _f_bounding_box(cfg: dict, name: str) -> dict[str, int]:
    """Core ``BOUNDING_BOX`` value: pixel coordinates inside the dummy image."""
    return {"x": 0, "y": 0, "width": 8, "height": 8}


def _f_any(cfg: dict, name: str) -> float:
    """A harmless scalar for wildcard (``*``) inputs.

    Logic/formatting nodes pass the value through untouched, so any type works;
    a float keeps it numeric for nodes that would otherwise need a real link.
    """
    return 1.0


def _f_vocab(cfg: dict, name: str) -> Any:
    """Build a vocabulary through ComfyDL's own builder so shapes always match.

    Running the builder node with its own synthesised widgets keeps the dummy in
    sync with the real implementation instead of duplicating its data layout.
    """
    from comfydl.nodes.nlp_utils import CdlVocabBuild

    args, _ = _legacy_inputs("CdlVocabBuild", CdlVocabBuild)
    out = CdlVocabBuild().execute(**args)
    return out[0] if isinstance(out, (tuple, list)) else out


#: type string (upper-cased) -> factory producing a dummy value
_VALUE_FACTORIES: dict[str, Callable[[dict, str], Any]] = {
    "INT": _f_int,
    "FLOAT": _f_float,
    "STRING": _f_string,
    "BOOLEAN": _f_bool,
    "COMBO": _f_combo,
    "TENSOR": _f_tensor,
    "IMAGE": _f_image,
    "MASK": _f_mask,
    "LATENT": _f_latent,
    "CDLMODEL": _f_model,
    "CDLDATALOADER": _f_dataloader,
    "CDLVOCAB": _f_vocab,
    "ARRAY": _f_array,
    "DICT": _f_dict,
    "COLOR": _f_color,
    "BOUNDING_BOX": _f_bounding_box,
    "*": _f_any,
}

#: types we are willing to fake for an *optional* input.  Optional slots are
#: left unconnected on purpose: a generic dummy tensor cannot know the shape a
#: node expects for e.g. a bias vector, and "not connected" is the documented
#: meaning of an optional input anyway.
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

#: hidden input name -> fixed dummy value
_HIDDEN_VALUES: dict[str, Any] = {
    "prompt": {},
    "dynprompt": {},
    "extra_pnginfo": {"workflow": {}},
    "unique_id": "smoke-test",
    "auth_token_comfy_org": "",
    "api_key_comfy_org": "",
}

#: node id -> reason; nodes that need network access, datasets, real checkpoints
#: or user interaction are not meaningfully testable with synthetic inputs.
_SKIPPED_NODES: dict[str, str] = {
    "CdlDownload": "downloads a file over HTTP",
    "CdlDownloadExtract": "downloads and extracts a dataset from the d2l DATA_HUB",
    "CdlFashionMNIST": "downloads the Fashion-MNIST dataset",
    "CdlBananasDetection": "downloads the banana-detection dataset",
    "CdlVOCSegmentation": "downloads the VOC2012 dataset (hundreds of MB)",
    "CdlMessageBox": "opens a modal Windows MessageBox and would block the run",
    "CdlWhat": "can launch a browser window when OMG is enabled",
    "CdlRNNLMScratch": "needs an RNN model built by the RNN scratch nodes, not a generic nn.Module",
    "CdlRNNLMScratchPredict": "needs an RNN model built by the RNN scratch nodes, not a generic nn.Module",
    "GetImageSize": "reports progress through PromptServer.instance, which only exists inside the server",
}

#: node id -> {input name -> factory}.  Some inputs are semantically narrower
#: than their slot type: an ``Embedding`` tensor is an *index* tensor, or a
#: bounding-box tensor must have exactly 4 columns.  The generic dummy cannot
#: know that, so the harness declares it here instead of weakening the node.
_INPUT_OVERRIDES: dict[str, dict[str, Callable[[dict, str], Any]]] = {
    "BasicEmbedding": {"tensor": _f_index_small},
    # NLP: indices have to fall inside the dummy vocabulary.
    "CdlVocabDecode": {"indices": _f_index_1d},
    # Object detection: (num_boxes, 4) boxes plus matching scores/anchors.
    "CdlBoxCornerToCenter": {"boxes": _f_boxes},
    "CdlBoxCenterToCorner": {"boxes": _f_boxes},
    "CdlBoxIou": {"boxes1": _f_boxes, "boxes2": _f_boxes},
    "CdlNms": {"boxes": _f_boxes, "scores": _f_scores},
    "CdlOffsetBoxes": {"anchors": _f_boxes, "assigned_bb": _f_boxes},
    "CdlOffsetInverse": {"anchors": _f_boxes, "offset_preds": _f_boxes},
    "CdlAssignAnchorToBbox": {"ground_truth": _f_boxes, "anchors": _f_boxes},
    "CdlMultiboxTarget": {"anchors": _f_anchors, "labels": _f_multibox_labels},
    "CdlMultiboxDetection": {
        "cls_probs": _f_cls_probs,
        "offset_preds": _f_offset_preds,
        "anchors": _f_anchors,
    },
    "CdlShowBboxes": {"bboxes": _f_boxes},
    # Sequences / regression / metrics.
    "CdlSequenceMask": {"X": _f_seq3d, "valid_len": _f_valid_len},
    "CdlLinReg": {"w": _f_weight_col, "b": _f_bias_1d},
    "CdlAccuracy": {"y": _f_class_idx},
    # Array/dataset helpers and matplotlib charts need consistent, non-negative data.
    "CdlLoadArray": {"features": _f_features},
    "CdlBarChart": {"values": _f_values6, "labels": _f_labels6},
    "CdlConfusionMatrix": {"class_labels": _f_labels3},
    "CdlPieChart": {"values": _f_values3},
}


def _synthesise(type_name: str, cfg: dict, name: str) -> Any:
    """Build a dummy value for one input, based on its declared type string.

    Comma separated type strings (V3 "match type" inputs such as
    ``INT,FLOAT,STRING,BOOLEAN``) are satisfied with their first alternative.
    """
    for alternative in str(type_name).split(","):
        factory = _VALUE_FACTORIES.get(alternative.strip().upper())
        if factory is not None:
            return factory(cfg, name)
    raise Unsupported(f"no dummy value for type {type_name!r}")


# --------------------------------------------------------------------------- #
# Spec parsing (legacy INPUT_TYPES)
# --------------------------------------------------------------------------- #
def _split_spec(spec: Any, name: str) -> tuple[str, dict]:
    """Normalise the many shapes an ``INPUT_TYPES`` entry can take.

    Handles ``"IMAGE"``, ``("IMAGE",)``, ``("INT", {"default": 1})`` and the
    option-list form ``(["train", "eval"], {"default": "eval"})``.
    """
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise Unsupported(f"empty spec for input {name!r}")
        head = spec[0]
        cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(head, str):
            return head, cfg
        options = list(head)
        if cfg:
            options = cfg.get("options", options)
        return "COMBO", {"options": options, **cfg}
    raise Unsupported(f"unrecognised spec {spec!r} for input {name!r}")


def _legacy_inputs(node_id: str, node_cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(args, required_names)`` for a legacy ``INPUT_TYPES`` node."""
    spec = node_cls.INPUT_TYPES()
    overrides = _INPUT_OVERRIDES.get(node_id, {})
    args: dict[str, Any] = {}
    required: dict[str, Any] = {}

    for section in ("required", "optional"):
        for name, raw in (spec.get(section) or {}).items():
            type_name, cfg = _split_spec(raw, name)
            if name in overrides:
                args[name] = overrides[name](cfg, name)
                if section == "required":
                    required[name] = args[name]
                continue
            if section == "optional" and type_name.upper() not in _WIDGET_TYPES:
                continue  # leave optional slots unconnected
            try:
                args[name] = _synthesise(type_name, cfg, name)
            except Unsupported:
                if section == "required":
                    raise
                continue  # an optional input we cannot fake is simply omitted
            if section == "required":
                required[name] = args[name]

    for name in (spec.get("hidden") or {}):
        if name in args:
            continue
        if name not in _HIDDEN_VALUES:
            raise Unsupported(f"no dummy value for hidden input {name!r}")
        args[name] = _HIDDEN_VALUES[name]

    if getattr(node_cls, "INPUT_IS_LIST", False):
        args = {key: [value] for key, value in args.items()}
    return args, required


# --------------------------------------------------------------------------- #
# Spec parsing (V3 schema)
# --------------------------------------------------------------------------- #
def _autogrow_value(slot: Any) -> Any:
    """Build the ``{name: value}`` mapping the engine passes to an Autogrow input.

    An ``io.Autogrow.Input`` carries a template describing the repeated sub-input
    (``template.input``) and the names the engine would use
    (``template.names``); the harness connects exactly the minimum number of
    them, filled with dummy values of the template's type.
    """
    template = slot.template
    inner = template.input
    io_type = str(inner.get_io_type())
    cfg = {
        key: getattr(inner, key)
        for key in ("default", "min", "max", "options")
        if hasattr(inner, key)
    }
    # Connect at least two names: aggregate nodes (``a and b``, ``a + b``) need a
    # second operand before their default widget value makes sense.
    count = min(len(getattr(template, "names", [])), max(2, int(getattr(template, "min", 1) or 1)))
    names = list(getattr(template, "names", []))[:count]
    if not names:
        raise Unsupported(f"autogrow input {slot.id!r} exposes no names")
    return {name: _synthesise(io_type, cfg, name) for name in names}


def _v3_inputs(node_id: str, node_cls: type) -> tuple[dict[str, Any], dict[str, Any], int, bool]:
    """Return ``(args, required_names, output_arity, allows_fewer_outputs)``.

    ``allows_fewer_outputs`` mirrors ``Schema.is_output_node``: an output node is
    allowed to return fewer values than it declares (the engine fills the rest
    with ``None``), which is exactly what ``SaveImage`` style nodes rely on.
    """
    schema = node_cls.GET_SCHEMA()
    overrides = _INPUT_OVERRIDES.get(node_id, {})
    args: dict[str, Any] = {}
    required: dict[str, Any] = {}

    for slot in schema.inputs:
        io_type = str(slot.get_io_type())
        cfg = {
            key: getattr(slot, key)
            for key in ("default", "min", "max", "options")
            if hasattr(slot, key)
        }
        optional = bool(getattr(slot, "optional", False))
        if slot.id in overrides:
            value = overrides[slot.id](cfg, slot.id)
        elif optional and io_type.upper() not in _WIDGET_TYPES:
            continue  # leave optional slots unconnected
        else:
            try:
                if io_type == "COMFY_AUTOGROW_V3":
                    value = _autogrow_value(slot)
                else:
                    value = _synthesise(io_type, cfg, slot.id)
            except Unsupported:
                if optional:
                    continue
                raise
        args[slot.id] = value
        if not getattr(slot, "optional", False):
            required[slot.id] = value

    if getattr(schema, "is_input_list", False):
        # The engine hands every input over as a list for such nodes.
        args = {key: [value] for key, value in args.items()}

    allows_fewer = bool(getattr(schema, "is_output_node", False))
    return args, required, len(schema.outputs), allows_fewer


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run_with_timeout(call: Callable[[], Any], timeout: float) -> Any:
    """Run ``call`` in a daemon thread so a hanging node cannot block the run."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - reported verbatim below
            box["error"] = exc
            box["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"did not finish within {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _install_hidden(node_cls: type) -> None:
    """Plant the engine-style hidden inputs a V3 node may read.

    The server injects a ``HiddenHolder`` on the node class before calling it;
    without an engine, output nodes such as ``GetImageSize`` or ``SaveImage``
    would see ``cls.hidden is None``.
    """
    from comfy_api.latest._io import HiddenHolder

    node_cls.hidden = HiddenHolder(
        unique_id="smoke-test",
        prompt={},
        extra_pnginfo={"workflow": {}},
        dynprompt={},
        auth_token_comfy_org="",
        api_key_comfy_org="",
    )


def _execute(
    node_id: str,
    node_cls: type,
    args: dict[str, Any],
    timeout: float,
) -> tuple[Any, int, bool]:
    """Execute one node and return ``(raw_output, expected_arity, allows_fewer)``."""
    if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
        _, _, arity, allows_fewer = _v3_inputs(node_id, node_cls)
        _install_hidden(node_cls)
        entry_name = getattr(node_cls, "FUNCTION", _V3_ENTRY_POINTS[0])
        entry = getattr(node_cls, entry_name)
        if entry_name.endswith("_ASYNC") or inspect.iscoroutinefunction(entry):
            result = _run_with_timeout(lambda: asyncio.run(entry(**args)), timeout)
        else:
            result = _run_with_timeout(lambda: entry(**args), timeout)
        return getattr(result, "result", result), arity, allows_fewer

    instance = node_cls()
    entry = getattr(instance, node_cls.FUNCTION)
    result = _run_with_timeout(lambda: entry(**args), timeout)
    arity = len(getattr(node_cls, "RETURN_TYPES", ()) or ())
    return result, arity, bool(getattr(node_cls, "OUTPUT_NODE", False))


def _count_outputs(result: Any) -> int:
    """Number of values a node returned, tolerating UI-only dict returns."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return 0
    if isinstance(result, (tuple, list)):
        return len(result)
    return 1


def _category(node_cls: type) -> str:
    if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
        try:
            return node_cls.GET_SCHEMA().category or "<none>"
        except Exception:  # noqa: BLE001 - a broken schema is reported by the run
            return "<bad schema>"
    return getattr(node_cls, "CATEGORY", "<none>") or "<none>"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_categories(mappings: dict[str, type]) -> None:
    """Print category -> node count, handy when updating the shipped docs."""
    counts: dict[str, list[str]] = {}
    for node_id, node_cls in mappings.items():
        counts.setdefault(_category(node_cls), []).append(node_id)
    print(f"registered nodes: {len(mappings)}")
    print(f"categories      : {len(counts)}")
    for name in sorted(counts):
        print(f"  {name:<44} {len(counts[name]):>4}")
    uncategorised = counts.get("<none>", [])
    if uncategorised:
        print("  nodes without a category: " + ", ".join(sorted(uncategorised)))


def _print_summary(results: list[tuple[str, str, str, str]]) -> int:
    passed = [r for r in results if r[1] == "PASS"]
    skipped = [r for r in results if r[1] == "SKIP"]
    failed = [r for r in results if r[1] == "FAIL"]

    if skipped:
        print("\n--- SKIPPED ---")
        for node_id, _, detail, _ in skipped:
            print(f"  {node_id:<28} {detail}")

    if failed:
        print("\n--- FAILURES ---")
        for node_id, _, detail, tb in failed:
            print(f"  {node_id:<28} {detail}")
            if tb:
                print(tb)

    print(
        f"\n=== smoke result: {len(passed)} PASS, {len(skipped)} SKIP, "
        f"{len(failed)} FAIL (of {len(results)}) ==="
    )
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filter", default="", help="only run node ids containing this text")
    parser.add_argument("--verbose", action="store_true", help="print one line per node as it runs")
    parser.add_argument(
        "--categories",
        action="store_true",
        help="only print the category -> node count table (no execution)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per node time budget in seconds")
    parser.add_argument("--keep-sandbox", action="store_true", help="keep the scratch directory")
    opts = parser.parse_args(argv)

    sandbox = Path(tempfile.mkdtemp(prefix="cdl_smoke_"))
    try:
        _bootstrap(sandbox)
        print(f"[smoke] repo      : {REPO_ROOT}")
        print(f"[smoke] python    : {sys.executable}")
        mappings = _load_registry(opts.verbose)

        if opts.categories:
            _print_categories(mappings)
            return 0

        node_ids = sorted(node_id for node_id in mappings if opts.filter.lower() in node_id.lower())
        print(f"[smoke] nodes     : {len(node_ids)} selected of {len(mappings)} registered\n")

        results: list[tuple[str, str, str, str]] = []
        for node_id in node_ids:
            node_cls = mappings[node_id]
            if node_id in _SKIPPED_NODES:
                results.append((node_id, "SKIP", _SKIPPED_NODES[node_id], ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}")
                continue
            try:
                if hasattr(node_cls, "GET_SCHEMA") and hasattr(node_cls, "define_schema"):
                    args, _, _, _ = _v3_inputs(node_id, node_cls)
                else:
                    args, _ = _legacy_inputs(node_id, node_cls)
                result, arity, allows_fewer = _execute(node_id, node_cls, args, opts.timeout)
                got = _count_outputs(result)
                if got > arity or (got != arity and not allows_fewer):
                    raise AssertionError(f"returned {got} value(s), schema declares {arity}")
            except Unsupported as exc:
                results.append((node_id, "SKIP", str(exc), ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}: {exc}")
            except TimeoutError as exc:
                results.append((node_id, "SKIP", f"timeout ({exc})", ""))
                if opts.verbose:
                    print(f"SKIP  {node_id}: {exc}")
            except BaseException as exc:  # noqa: BLE001 - a FAIL is the whole point
                results.append(
                    (node_id, "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc())
                )
                if opts.verbose:
                    print(f"FAIL  {node_id}: {type(exc).__name__}: {exc}")
            else:
                results.append((node_id, "PASS", "", ""))
                if opts.verbose:
                    print(f"PASS  {node_id}  [{_category(node_cls)}]")

        return _print_summary(results)
    finally:
        if opts.keep_sandbox:
            print(f"[smoke] sandbox kept at {sandbox}")
        else:
            os.chdir(REPO_ROOT)  # Windows cannot remove the current directory
            shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
