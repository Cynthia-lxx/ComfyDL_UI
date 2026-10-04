"""Static memory profiling for ComfyDL workflows (reform: Profiling Tools M1).

A pure-Python estimation engine: given the prompt-format serialization of a
workflow (what ``app.graphToPrompt()`` produces on the frontend) it walks the
graph topologically, runs the estimator registered for every node class and
reports an upper bound of the peak memory plus a three-colour verdict against
the device budget. Nodes without an estimator are reported as unknown - the
engine never guesses.

The package deliberately mirrors the other ``comfy/`` protocol modules: no
torch execution, no UI machinery, importable and testable on its own.

Public surface::

    from comfy.profiling import (
        DeviceBudget, AssumptionSet, DEFAULT_ASSUMPTIONS,
        estimate_workflow, analyse_oom, human_bytes, parse_allocation_bytes,
    )
"""

from comfy.profiling.assumptions import DEFAULT_ASSUMPTIONS, AssumptionSet
from comfy.profiling.engine import (
    DeviceBudget,
    budget_from_environment,
    estimate_workflow,
)
from comfy.profiling.postmortem import (
    analyse_oom,
    human_bytes,
    parse_allocation_bytes,
)

__all__ = [
    "DEFAULT_ASSUMPTIONS",
    "AssumptionSet",
    "DeviceBudget",
    "analyse_oom",
    "budget_from_environment",
    "estimate_workflow",
    "human_bytes",
    "parse_allocation_bytes",
]
