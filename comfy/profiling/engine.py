"""The topological estimator of ``comfy.profiling``.

Walks a prompt-format workflow (``{node_id: {"class_type": ..., "inputs":
{...}}}`` - exactly what the frontend's ``graphToPrompt`` produces) in
dependency order, runs the estimator registered for each class type,
propagates the estimated output values along the links, and aggregates a
report: per-node breakdowns, the peak over all nodes, the largest single
allocation, and the three-colour verdict against the device budget.

Verdict rules (upper-bound heuristics, see docs/profiling-m1-*.md):

* red   - the estimated peak exceeds 90% of the device total, **or** any
  single tensor exceeds the free budget (that allocation cannot succeed);
* green - the peak stays under 70% of the free budget;
* yellow - everything in between ("may OOM");

The M2 compute ledger is aggregated alongside: per-node ``flops_items``
(matmul + convolution FLOPs, kind attn/ffn/conv/logits) and a report-level
``flops`` section
(total / by-kind / largest). Unlike memory - where nodes share the device
and the report tracks the *peak* - compute is *additive*: the workflow
total is the sum over the nodes that were counted. Time is deliberately
not derived from it (that needs the measured calibration of M3).

The engine never raises on a malformed graph: the report carries the error.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from comfy.profiling import formulas
from comfy.profiling.assumptions import AssumptionSet
from comfy.profiling.estimators import ESTIMATORS, EstimationCtx, NodeEstimate
from comfy.profiling.shapes import Unknown

#: Verdict thresholds (fractions of the device budget).
GREEN_FRACTION_OF_FREE = 0.7
RED_FRACTION_OF_TOTAL = 0.9

#: Report layout version; bumped whenever the JSON shape changes.
#: v2 adds the per-node flops fields and the report-level "flops" section.
REPORT_VERSION = 2

#: Trainable node classes the post-mortem batch-size search may tune.
TUNABLE_TRAINERS = ("LanguageModelTrain", "TrainingLoop")


@dataclasses.dataclass(frozen=True)
class DeviceBudget:
    """The memory the estimated workload has to fit into."""

    name: str = "cpu"
    total_bytes: int = 0
    free_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "total_bytes": int(self.total_bytes),
            "free_bytes": int(self.free_bytes),
        }


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and not isinstance(value, str)
    )


def _dependencies(node: dict) -> List[Any]:
    return [
        value[0]
        for value in (node.get("inputs") or {}).values()
        if _is_link(value)
    ]


def _topological_order(prompt: dict) -> tuple:
    """Kahn's algorithm over the link graph; leftover nodes are cycles."""
    remaining = dict(prompt)
    order: List[Any] = []
    while remaining:
        ready = [
            node_id
            for node_id, node in remaining.items()
            if not any(dep in remaining for dep in _dependencies(node))
        ]
        if not ready:
            break  # dependency cycle - the leftovers are reported as such
        for node_id in ready:
            order.append(node_id)
            del remaining[node_id]
    return order, list(remaining)


def _verdict(peak: int, largest_single, budget: Optional[DeviceBudget]) -> tuple:
    if budget is None or budget.total_bytes <= 0:
        return "unknown", "no_budget"
    single = largest_single["single_bytes"] if largest_single else 0
    if single > budget.free_bytes:
        return "red", "single_exceeds_free"
    if peak >= budget.total_bytes * RED_FRACTION_OF_TOTAL:
        return "red", "total_exceeds_budget"
    if peak < budget.free_bytes * GREEN_FRACTION_OF_FREE:
        return "green", "below_70_free"
    return "yellow", "between"


def estimate_workflow(
    prompt: Optional[dict],
    budget: Optional[DeviceBudget] = None,
    assumptions: Optional[dict] = None,
    titles: Optional[dict] = None,
) -> dict:
    """Estimate a prompt-format workflow and return the report dict."""
    report: Dict[str, Any] = {
        "version": REPORT_VERSION,
        "budget": budget.as_dict() if budget else None,
        "verdict": "unknown",
        "verdict_reason": "empty",
        "peak_bytes": 0,
        "largest": None,
        "nodes": [],
        "assumptions_used": [],
        "error": None,
        "disclaimer_key": "disclaimer",
        # The M2 compute ledger: matmul FLOPs summed over the counted nodes.
        # ``total`` is 0 and ``any_estimated`` False when nothing was counted
        # (pass-through graphs, or unknown-only) - not a claim of free work.
        "flops": {
            "total": 0,
            "any_estimated": False,
            "by_kind": {"attn": 0, "ffn": 0, "conv": 0, "logits": 0, "other": 0},
            "largest": None,  # {"node_id", "class_type", "kind", "label", "flops"}
        },
    }
    if not isinstance(prompt, dict) or not prompt:
        report["verdict_reason"] = "no_graph"
        return report

    assumption_set = AssumptionSet(overrides=dict(assumptions or {}))
    order, cyclic = _topological_order(prompt)
    outputs: Dict[Any, List] = {}
    node_records: List[dict] = []
    peak = 0
    largest: Optional[dict] = None
    any_estimated = False
    flops_total = 0
    flops_any = False
    flops_by_kind: Dict[str, int] = {"attn": 0, "ffn": 0, "conv": 0, "logits": 0, "other": 0}
    flops_largest: Optional[dict] = None

    for node_id in order:
        node = prompt.get(node_id) or {}
        class_type = str(node.get("class_type") or "")
        record = {
            "id": node_id,
            "class_type": class_type,
            "title": (titles or {}).get(node_id, class_type),
            "status": "estimated",
            "confidence": "exact",
            "items": [],
            "total_bytes": 0,
            "basis": None,
            "reason": "",
            "flops_total": None,
            "flops_status": "unknown",
            "flops_items": [],
            "flops_reason": "",
        }
        estimator = ESTIMATORS.get(class_type)
        if estimator is None:
            record["status"] = "unknown"
            record["reason"] = "no estimator registered"
            outputs[node_id] = [Unknown("node type not estimated")]
        else:
            ctx = EstimationCtx(
                node_id, class_type, node.get("inputs") or {}, outputs, assumption_set
            )
            try:
                estimate: NodeEstimate = estimator(ctx)
            except Exception as exc:  # defensive: one bad node never kills the report
                estimate = NodeEstimate().as_unknown(f"estimator error: {exc}")
            outputs[node_id] = estimate.outputs
            record["status"] = estimate.status
            record["confidence"] = estimate.confidence
            record["reason"] = estimate.reason
            record["items"] = [item.as_dict() for item in estimate.items]
            record["total_bytes"] = sum(item["bytes"] for item in record["items"])
            record["basis"] = estimate.basis
            record["flops_status"] = estimate.flops_status
            record["flops_items"] = [item.as_dict() for item in estimate.flops_items]
            record["flops_reason"] = estimate.flops_reason
            if estimate.flops_status == "estimated" and estimate.flops_items:
                node_flops = sum(item.flops for item in estimate.flops_items)
                record["flops_total"] = node_flops
                flops_any = True
                flops_total += node_flops
                for item in estimate.flops_items:
                    flops_by_kind[item.kind] = flops_by_kind.get(item.kind, 0) + item.flops
                    if flops_largest is None or item.flops > flops_largest["flops"]:
                        flops_largest = {
                            "node_id": node_id,
                            "class_type": class_type,
                            "kind": item.kind,
                            "label": item.label,
                            "flops": item.flops,
                        }
            if estimate.status == "estimated":
                any_estimated = True
                peak = max(peak, record["total_bytes"])
                for item in record["items"]:
                    if largest is None or item["single_bytes"] > largest["single_bytes"]:
                        largest = dict(item, node_id=node_id, class_type=class_type)
        node_records.append(record)

    for node_id in cyclic:
        node = prompt.get(node_id) or {}
        node_records.append(
            {
                "id": node_id,
                "class_type": str(node.get("class_type") or ""),
                "title": (titles or {}).get(node_id, str(node.get("class_type") or "")),
                "status": "unknown",
                "confidence": "exact",
                "items": [],
                "total_bytes": 0,
                "basis": None,
                "reason": "dependency cycle",
                "flops_total": None,
                "flops_status": "unknown",
                "flops_items": [],
                "flops_reason": "dependency cycle",
            }
        )

    report["nodes"] = node_records
    report["peak_bytes"] = peak
    report["largest"] = largest
    report["flops"] = {
        "total": flops_total,
        "any_estimated": flops_any,
        "by_kind": flops_by_kind,
        "largest": flops_largest,
    }
    report["assumptions_used"] = assumption_set.used_entries()
    verdict, reason = _verdict(peak, largest, budget)
    if not any_estimated:
        verdict, reason = "unknown", "nothing_estimated"
    report["verdict"] = verdict
    report["verdict_reason"] = reason
    return report


def budget_from_environment() -> Optional[DeviceBudget]:
    """The primary torch device's budget, /system_stats' exact source."""
    try:
        import comfy.model_management as model_management

        device = model_management.get_torch_device()
        return DeviceBudget(
            name=str(device),
            total_bytes=int(model_management.get_total_memory(device)),
            free_bytes=int(model_management.get_free_memory(device)),
        )
    except Exception:
        return None
