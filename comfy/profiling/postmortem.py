"""OOM post-mortem analysis of ``comfy.profiling``.

When an out-of-memory error has already happened, the exception text still
carries the one number that matters: how many bytes the allocator tried to
hand out. This module extracts it, attributes it to the workflow node (and
tensor kind) whose estimated largest allocation matches, and - when the
failing node is a trainer with a ``batch_size`` widget - searches the largest
batch size that would fit into 70% of the free budget.

The 13,276,741,632-byte k_proj failure of the profiling golden case is the
canonical example: attributed to Language Model Train's attention q/k/v
output, with a suggested batch far below the 202,587-sample default.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Optional

from comfy.profiling.engine import (
    GREEN_FRACTION_OF_FREE,
    DeviceBudget,
    TUNABLE_TRAINERS,
    estimate_workflow,
)

# "you tried to allocate 13,276,741,632 bytes" / "can't allocate N bytes"
_BYTE_PATTERNS = (
    re.compile(r"allocat\w*\s*:?\s*([0-9][0-9,.]*)\s*bytes", re.IGNORECASE),
)
# "Tried to allocate 13.30 GiB" (CUDA style, binary suffix) ...
_BINARY_SUFFIX = re.compile(r"allocat\w*\s*:?\s*([0-9][0-9.,]*)\s*([KMGTPE])iB", re.IGNORECASE)
# ... and the decimal "13.30 GB" spelling.
_DECIMAL_SUFFIX = re.compile(r"allocat\w*\s*:?\s*([0-9][0-9.,]*)\s*([KMGTPE])B", re.IGNORECASE)

_SCALE_BINARY = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
_SCALE_DECIMAL = {"K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4, "P": 1000**5}

_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def human_bytes(size: Optional[int]) -> Optional[str]:
    """Format a byte count the way the memory panels do."""
    if size is None:
        return None
    value = float(size)
    for unit in _UNITS:
        if abs(value) < 1024.0 or unit == _UNITS[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return None


def parse_allocation_bytes(message: str) -> Optional[int]:
    """Extract the failed allocation size from an OOM exception message."""
    text = str(message or "")
    for pattern, scale in ((_BINARY_SUFFIX, _SCALE_BINARY), (_DECIMAL_SUFFIX, _SCALE_DECIMAL)):
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            return int(value * scale[match.group(2).upper()])
    for pattern in _BYTE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1).replace(",", "").rstrip("."))
            except ValueError:
                continue
    return None


def _attribute(report: dict, allocation: int) -> Optional[dict]:
    """Find the estimated tensor whose single allocation matches best."""
    best: Optional[dict] = None
    best_ratio: Optional[float] = None
    for node in report.get("nodes", []):
        for item in node.get("items", []):
            single = item.get("single_bytes", 0)
            if not single:
                continue
            ratio = abs(single - allocation) / max(1.0, float(allocation))
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best = {
                    "node_id": node["id"],
                    "class_type": node["class_type"],
                    "label": item["label"],
                    "kind": item["kind"],
                    "bytes": single,
                    "bytes_human": human_bytes(single),
                    "match": "exact" if ratio < 0.01 else ("close" if ratio < 0.1 else "nearest"),
                }
    return best


def _feasible_batch(
    prompt: dict, node_id: Any, samples: int, budget: DeviceBudget, assumptions: Optional[dict]
) -> Optional[int]:
    """Largest ``batch_size`` of ``node_id`` fitting into 70% of free memory."""

    def feasible(batch: int) -> bool:
        candidate = copy.deepcopy(prompt)
        candidate[node_id]["inputs"]["batch_size"] = batch
        report = estimate_workflow(candidate, budget, assumptions)
        limit = budget.free_bytes * GREEN_FRACTION_OF_FREE
        largest = report.get("largest") or {}
        return report["peak_bytes"] <= limit and largest.get("single_bytes", 0) <= limit

    if not feasible(1):
        return None
    low, high = 1, max(1, int(samples))
    while low < high:
        middle = (low + high + 1) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle - 1
    return low


def analyse_oom(payload: dict, budget: Optional[DeviceBudget] = None) -> dict:
    """Post-mortem for one ``execution_error`` event.

    ``payload``: ``{"message": str, "node_id": ..., "node_type": str,
    "prompt": {...} (optional), "assumptions": {...} (optional)}``.
    """
    message = str(payload.get("message") or "")
    allocation = parse_allocation_bytes(message)
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), dict) else None
    assumptions = payload.get("assumptions")

    result: Dict[str, Any] = {
        "allocation_bytes": allocation,
        "allocation_human": human_bytes(allocation),
        "message_head": message.splitlines()[0][:300] if message else "",
        "node_id": payload.get("node_id"),
        "node_type": payload.get("node_type"),
        "attributed": None,
        "suggestion": None,
        "note_key": "postmortem_note" if allocation else "postmortem_no_alloc",
    }

    report: Optional[dict] = None
    if prompt:
        try:
            report = estimate_workflow(prompt, budget, assumptions)
        except Exception:
            report = None

    if report and allocation:
        result["attributed"] = _attribute(report, allocation)

    if report and prompt and allocation and budget and budget.free_bytes > 0:
        for node in report.get("nodes", []):
            if node.get("class_type") not in TUNABLE_TRAINERS:
                continue
            basis = (node.get("basis") or {}).get("params") or {}
            samples = basis.get("samples")
            if not isinstance(samples, int) or samples <= 1:
                continue
            previous = (prompt.get(node["id"]) or {}).get("inputs", {}).get("batch_size", 0)
            suggestion = _feasible_batch(prompt, node["id"], samples, budget, assumptions)
            if suggestion is not None:
                result["suggestion"] = {
                    "node_id": node["id"],
                    "class_type": node["class_type"],
                    "batch_size": suggestion,
                    "previous_batch_size": previous,
                    "samples": samples,
                }
                break

    return result
