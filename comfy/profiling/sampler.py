"""Offline ATen op-census sampler -> static rules for the equivalent graph (Profiling v2 P1).

Runs the M3 probe (``opgraph._probe_one`` - real execution, P0 guardrails
apply) once per registered node class on standard tiny inputs, then freezes
the resulting ATen op census into the shipped rules JSON
(``comfy/profiling/data/op_rules.json``).  At Analyze time the assembler
serves those censuses from this file instead of executing anything - the
whole point of Profiling v2's P1.

Usage (offline, one-off or CI)::

    python -m comfy.profiling.sampler                     # full registry
    python -m comfy.profiling.sampler --filter Cdl        # name substring
    python -m comfy.profiling.sampler --out my_rules.json

Honest boundaries
-----------------
* Inputs are guessed: widget values come from ``INPUT_TYPES`` defaults (or
  small synthesized values), tensor inputs try a few standard shapes until
  one probes.  A class that probes under none of the candidates, raises, or
  cannot be described at all lands in the **review list** next to the rules
  file - the rules library is allowed to be incomplete and the coverage
  report says so.
* Sampled FLOPs are those of the sample inputs; only the census structure is
  frozen (see oprules.py's boundary notes).
* Side-effecting nodes (file writers, ...) run for real: the sampler chdirs
  into a temp sandbox for the whole run and restores the cwd afterwards.
* Being offline, the sampler executes on the MAIN thread - Ctrl+C works.

Requires the host registry: bootstraps it lazily exactly like the Analyze
endpoint does (``_bootstrap_registry``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from comfy.profiling import oprules, opgraph

#: Candidate tensor shapes tried in order until a probe succeeds.  Small on
#: purpose: the probe executes for real, and OPS_LIMIT caps runaway census.
_TENSOR_CANDIDATES: Tuple[tuple, ...] = (
    (1, 8),          # sequence-like (embeddings, losses)
    (1, 1, 8, 8),    # grayscale-ish image
    (1, 3, 8, 8),    # RGB-like image
    (1, 16, 16, 1),  # NHWC-ish image
    (2, 4),          # batched sequence
)

#: Widget fallback values when INPUT_TYPES has no default.
_WIDGET_FALLBACKS: Dict[str, Any] = {
    "INT": 1,
    "FLOAT": 1.0,
    "STRING": "the quick brown fox",
    "BOOLEAN": True,
    "SEED": 0,
    "COMBO": None,  # filled from the options list
}

#: Types the sampler cannot synthesize (socket-only inputs).  A class whose
#: required inputs contain one of these and has no default goes to review.
_UNSYNTHESIZABLE_PREFIXES = (
    "TENSOR", "IMAGE", "LATENT", "MASK", "MODEL", "CLIP", "VAE",
    "VOCAB", "PARAMS", "OPTIMIZER", "SCHEDULER", "DATASET", "nn_model",
    "SIGMAS", "CONTROL_NET", "UPSCALE_MODEL", "LORA_MODEL", "LOSS_MAP",
    "AUDIO", "VIDEO", "MESH", "GAUSSIAN", "MODELSPEC", "WEIGHTS",
)

#: UI / network side-effect classes must NEVER be probed: the probe executes
#: for real, and e.g. CdlMessageBox pops a native blocking dialog on the
#: user's desktop (2026-10-07 incident: the whole full-registry sweep got
#: wedged on MB_OK dialogs).  Matched case-insensitively on the class name.
_SIDE_EFFECT_BLACKLIST = (
    "messagebox", "msgbox", "dialog", "popup", "download", "upload",
)

#: Per-node wall-clock budget.  The probe runs on a daemon thread; a node that
#: ignores this budget is abandoned as REVIEW (the thread may linger but the
#: sweep continues - a blocking dialog can no longer wedge the whole run).
_NODE_TIMEOUT_SECONDS = 30.0


def _preset_inputs(class_type: str, cls: Any) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """Split required inputs into (widget values, tensor slot names).

    Widget values come from ``INPUT_TYPES`` defaults or small synthesized
    values; ``TENSOR`` sockets are collected for the shape-candidate probe.
    ``None`` = this class cannot be preset at all (a required socket of a
    high-level type we cannot fabricate, or a broken INPUT_TYPES) - it goes
    to the review list honestly.
    """
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return None
    widgets: Dict[str, Any] = {}
    tensor_slots: List[str] = []
    for name, raw in (spec.get("required") or {}).items():
        # V1 nodes use tuples ("STRING", {...}); V3 (io.ComfyNode) compatibility
        # INPUT_TYPES returns LISTS ["STRING", {...}] and names combos "COMBO"
        # with the options inside the meta dict - handle both shapes.
        if isinstance(raw, (tuple, list)):
            type_name = raw[0]
            meta = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        else:
            type_name, meta = raw, {}
        if isinstance(type_name, list):  # V1-style combo options
            widgets[name] = type_name[0]
            continue
        type_name = str(type_name)
        if type_name == "COMBO" or isinstance(meta.get("options"), list):
            options = meta.get("options") or []
            default = meta.get("default")
            if default in options:
                widgets[name] = default
            elif options:
                widgets[name] = options[0]
            else:
                return None
            continue
        if type_name == "TENSOR":
            tensor_slots.append(name)
            continue
        if type_name in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            widgets[name] = meta.get("default", _WIDGET_FALLBACKS.get(type_name))
            if widgets[name] is None:
                return None
            continue
        if type_name == "SEED" or name.endswith("seed"):
            widgets[name] = _WIDGET_FALLBACKS["SEED"]
            continue
        if type_name.startswith(_UNSYNTHESIZABLE_PREFIXES):
            return None  # a required socket we cannot fabricate
        return None  # unknown widget type: honest review, no guessing
    return widgets, tensor_slots


def _tensor_shape_candidates(n_slots: int) -> List[List[tuple]]:
    """Shape assignment candidates for ``n_slots`` tensor inputs.

    Ordered most-likely-first; the probe executes for real, so candidates are
    few and small.  A node that rejects every candidate lands in the review
    list - no guessing beyond this closed set.
    """
    seq = (1, 8)
    img = (1, 1, 8, 8)
    sq = (8, 8)
    if n_slots <= 1:
        return [[seq], [img], [(1, 1)]]
    if n_slots == 2:
        # matmul-ish: data x weights; then all-same fallbacks
        return [
            [seq, sq],
            [seq, seq],
            [img, img],
            [(1, 3, 8, 8), (1, 1, 8, 8)],
        ]
    if n_slots == 3:
        # linear-ish: X, w, b(bias shares output width)
        return [
            [seq, (8, 8), (8,)],
            [seq, sq, sq],
            [seq, seq, seq],
            [img, img, img],
        ]
    return [[seq] * n_slots, [img] * n_slots, [(8, 8)] * n_slots]


def _try_probe(class_type: str, cls: Any, widgets: Dict[str, Any],
               tensor_slots: List[str]) -> Tuple[Optional[dict], Optional[str]]:
    """Probe one class over the tensor-shape candidates.

    Returns ``(record, input_signature)`` on success, else ``(None, error)``.
    """
    import torch

    last_error = "no tensor-shape candidate succeeded"
    for shapes in _tensor_shape_candidates(len(tensor_slots)):
        inputs: Dict[str, Any] = dict(widgets)
        signature: Dict[str, Any] = dict(widgets)
        for slot, shape in zip(tensor_slots, shapes):
            inputs[slot] = torch.zeros(shape)
            signature[slot] = f"TensorVal(shape={list(shape)})"
        try:
            record, _outputs = opgraph._probe_one(class_type, cls, inputs)
        except Exception as exc:  # noqa: BLE001 - candidate shapes may disagree
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if record.get("status") == "probed":
            return record, signature
        last_error = record.get("error") or "probe failed"
    return None, last_error


def _run_with_timeout(fn, timeout: float) -> Tuple[Optional[Any], Optional[str]]:
    """Run ``fn`` on a daemon thread with a wall-clock budget.

    Returns ``(value, None)`` or ``(None, reason)``.  A timed-out thread
    cannot be killed (Python), but as a daemon it no longer blocks the sweep
    or the process exit.
    """
    import threading

    box: Dict[str, Any] = {}

    def _runner():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None, f"probe timed out after {timeout:.0f}s (possible UI block or hang)"
    if "error" in box:
        raise box["error"]
    return box.get("value"), None


def sample_registry(filter_substring: str = "", limit: int = 0,
                    include_legacy: bool = True) -> Tuple[Dict[str, Any], List[dict]]:
    """Sample every registered class; returns (rules, review list)."""
    opgraph._bootstrap_registry()
    import nodes as host_nodes

    registry = host_nodes.NODE_CLASS_MAPPINGS
    rules: Dict[str, Any] = {}
    review: List[dict] = []
    done = 0
    for class_type in sorted(registry):
        if filter_substring and filter_substring.lower() not in class_type.lower():
            continue
        if not include_legacy and class_type.startswith(("DEPRECATED_",)):
            continue
        if limit and done >= limit:
            break
        lowered = class_type.lower()
        if any(token in lowered for token in _SIDE_EFFECT_BLACKLIST):
            review.append({
                "class_type": class_type,
                "reason": "skipped: UI/network side-effect class (blacklist)",
            })
            done += 1
            print(f"[sampler] {done:>4} {class_type:<40} "
                  "REVIEW (blacklisted: UI/network side effects)", flush=True)
            continue
        entry: Dict[str, Any] = {
            "source": "offline_probe",
            "flops_from_shape": "estimated",
            "probes": [],
        }
        cls = registry[class_type]
        preset = _preset_inputs(class_type, cls)
        record = signature = None
        error = None
        if preset is None:
            error = "required inputs not synthesizable from INPUT_TYPES"
        else:
            widgets, tensor_slots = preset
            try:
                (record, signature), timeout_error = _run_with_timeout(
                    lambda: _try_probe(class_type, cls, widgets, tensor_slots),
                    _NODE_TIMEOUT_SECONDS,
                )
                if timeout_error:
                    record, error = None, timeout_error
                else:
                    error = None if record is not None else (signature or "probe failed")
            except Exception as exc:  # noqa: BLE001 - a crashing node is a review entry
                record, error = None, f"{type(exc).__name__}: {exc}"
        if record is None:
            review.append({
                "class_type": class_type,
                "reason": error or "no successful probe candidate",
            })
        else:
            entry["probes"].append({
                "input_signature": signature,
                "ops": record.get("ops") or [],
                "ops_total": record.get("ops_total") or 0,
                "data_dependent_reads": record.get("data_dependent_reads") or 0,
                "probe_ms": record.get("probe_ms") or 0,
            })
            if not entry["probes"][0]["ops"]:
                # probed with zero ATen calls (pure-python node): still a valid
                # "empty census" rule - the graph shows a leaf, not a gap.
                entry["flops_from_shape"] = "unknown"
            rules[class_type] = entry
        done += 1
        print(f"[sampler] {done:>4} {class_type:<40} "
              + ("rule" if record is not None else f"REVIEW ({error})"),
              flush=True)
    return rules, review


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample the node registry into static ATen op-census rules.")
    parser.add_argument("--out", default=str(oprules._DATA_PATH),
                        help="output rules JSON (default: the shipped path)")
    parser.add_argument("--review", default=None,
                        help="output review list path (default: <out>.review.txt)")
    parser.add_argument("--filter", default="", help="class_type substring filter")
    parser.add_argument("--limit", type=int, default=0, help="max classes to sample")
    parser.add_argument("--include-legacy", action="store_true",
                        help="also sample DEPRECATED_/legacy classes")
    args = parser.parse_args(argv)

    sandbox = tempfile.mkdtemp(prefix="cdl_sampler_")
    cwd = os.getcwd()
    os.chdir(sandbox)  # side-effecting nodes (file writers) write here
    started = time.monotonic()
    try:
        rules, review = sample_registry(args.filter, args.limit, args.include_legacy)
    finally:
        os.chdir(cwd)

    document = {
        "version": oprules.OPRULES_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "comfy.profiling.sampler",
        "rules": rules,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    review_path = Path(args.review) if args.review else out_path.with_suffix(".review.txt")
    review_path.write_text(
        "\n".join(f"{r['class_type']}: {r['reason']}" for r in review) or "(none)",
        encoding="utf-8")
    print(f"[sampler] rules: {len(rules)} -> {out_path}")
    print(f"[sampler] review: {len(review)} -> {review_path}")
    print(f"[sampler] done in {time.monotonic() - started:.1f}s (sandbox: {sandbox})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
