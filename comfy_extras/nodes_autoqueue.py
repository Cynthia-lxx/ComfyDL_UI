"""Auto-queue nodes (P3 Craft 4, M-C).

``CdlBatchLoop`` re-queues the same prompt N times (the workflow-level batch
scan; the machinery lives in :mod:`comfy.autoqueue` and the worker hook in
``main.py:prompt_worker``).  ``CdlOutputGate`` wraps the official
ExecutionBlocker: feed it the loop's ``is_final`` output and a Save/Image
node downstream only produces output on the last round - the "save only the
final result of a batch" pattern.

Both are control nodes: zero compute, exact estimator confidence.
"""

from __future__ import annotations

from typing_extensions import override

from comfy import autoqueue
from comfy_api.latest import ComfyExtension, io

CATEGORY = "utilities"


class CdlBatchLoop(io.ComfyNode):
    """Re-queue this workflow N times (workflow-level batch scan).

    What: an output-node loop driver.  Each execution books one iteration
          against ``chain_id``; while rounds remain it asks the server to
          re-queue the same prompt after the run finishes, so the whole
          graph - and every node in it - runs again with the next index.
    In:   iterations (INT) - total rounds, 1 disables re-queueing;
          chain_id (STRING) - names the counter; editing the chain id or
          the iterations starts a fresh chain at 1/iterations.
    Out:  status (STRING) - ``"3/10"`` position readout;
          index (INT) - 0-based round number (use as a seed offset!);
          is_final (BOOLEAN) - True on the last round; feed it into
          ``CdlOutputGate`` to save only the final round's outputs.
    Caveats: counters live in server memory (a restart resets them);
          interrupted or failed runs are never re-queued; the re-queued
          prompt is byte-identical, so any node that wants variation must
          derive it from ``index`` (e.g. seed + index).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlBatchLoop",
            display_name="Batch Loop",
            category=CATEGORY,
            description=(
                "Re-queues the same prompt N times for workflow-level batch "
                "runs; outputs the round position and whether it is the "
                "final round."),
            search_aliases=[
                "auto queue", "batch", "loop", "repeat", "requeue",
                "iterate", "batch scan",
            ],
            is_output_node=True,
            inputs=[
                io.Int.Input(
                    "iterations",
                    default=2,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Total rounds. 1 runs once and never re-queues.",
                ),
                io.String.Input(
                    "chain_id",
                    default="default",
                    tooltip="Names this loop's counter; a new id (or changed iterations) restarts at 1.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="status"),
                io.Int.Output(display_name="index"),
                io.Boolean.Output(display_name="is_final"),
            ],
        )

    @classmethod
    def execute(cls, iterations: int = 2, chain_id: str = "default"):
        verdict = autoqueue.register_iteration(chain_id, iterations)
        if not verdict["is_final"]:
            autoqueue.note_requeue(chain_id)
        print(
            f"[ComfyDL] Batch Loop '{chain_id}': round "
            f"{verdict['iteration']}/{verdict['iterations']}"
            + ("" if verdict["is_final"] else " - will re-queue")
        )
        return io.NodeOutput(
            verdict["status"], verdict["index"], verdict["is_final"])


class CdlOutputGate(io.ComfyNode):
    """Pass a value through only when the gate is open (ExecutionBlocker).

    What: the "save only the last round of a batch" switch.  Wire
          ``CdlBatchLoop.is_final`` (or any BOOLEAN) into ``gate`` and the
          gated node downstream into ``value``; a closed gate blocks the
          *downstream node* with the official ExecutionBlocker mechanism
          (its tile shows as skipped), an open gate passes ``value``.
    In:   gate (BOOLEAN) - False blocks, True passes;
          value (ANY) - anything (images, tensors, strings...).
    Out:  value (ANY) - unchanged when the gate is open.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlOutputGate",
            display_name="Output Gate",
            category=CATEGORY,
            description=(
                "Blocks the downstream node unless the gate input is true - "
                "combine with Batch Loop's is_final to save only the last "
                "round of a batch."),
            search_aliases=[
                "gate", "conditional", "save last", "only final",
                "execution blocker", "skip",
            ],
            inputs=[
                io.Boolean.Input(
                    "gate",
                    default=True,
                    tooltip="False blocks the downstream node; True passes value through.",
                ),
                io.AnyType.Input(
                    "value",
                    tooltip="The value to pass through when the gate is open.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def execute(cls, gate: bool = True, value=None):
        if not gate:
            from comfy_execution.graph import ExecutionBlocker

            return io.NodeOutput(ExecutionBlocker("output gate closed"))
        return io.NodeOutput(value)


class AutoQueueExtension(ComfyExtension):
    """Registers the auto-queue nodes."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlBatchLoop, CdlOutputGate]


async def comfy_entrypoint() -> AutoQueueExtension:
    return AutoQueueExtension()
