#!/usr/bin/env python
"""Standalone smoke test: the auto-queue primitive (P3 Craft 4, ADR-D10).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_autoqueue.py

Covers: iteration bookkeeping (1..N + over-run), the config-signature reset,
chain reset, the requeue intent lifecycle (note -> take -> idempotence), the
CdlBatchLoop node contract (status/index/is_final + intent only while rounds
remain), the CdlOutputGate ExecutionBlocker semantics, node registration +
estimator binding, and the describe() snapshot shape.  The worker hook in
main.py is a 4-line consumer of take_requeue(); its semantics are pinned
here by the intent lifecycle (consume == clear, failures never re-queue).
"""

import asyncio
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_autoq_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy import autoqueue  # noqa: E402
from comfy.profiling import estimators as est_mod  # noqa: E402

# fresh module state for this test process
autoqueue._counters.clear()
autoqueue._pending_requeue = None

# ---------------------------------------------------------------------------
# T1: iteration bookkeeping 1..N (and the over-run stays final)

v1 = autoqueue.register_iteration("t1", 3)
v2 = autoqueue.register_iteration("t1", 3)
v3 = autoqueue.register_iteration("t1", 3)
v4 = autoqueue.register_iteration("t1", 3)
check("A1 rounds count up from 1/3",
      (v1["index"], v1["status"]) == (0, "1/3"))
check("A2 only the last round is final",
      v1["is_final"] is False and v2["is_final"] is False
      and v3["is_final"] is True)
check("A3 an over-run round is still final (no negative remainder)",
      v4["is_final"] is True and v4["index"] == 3)

# ---------------------------------------------------------------------------
# T2: the config signature resets the chain transparently

v_cfg = autoqueue.register_iteration("t2", 5)
v_reset = autoqueue.register_iteration("t2", 7)
check("A4 changing iterations restarts the chain at 1/7",
      v_cfg["status"] == "1/5" and v_reset["status"] == "1/7")
autoqueue.register_iteration("other", 5)
check("A5 a different chain_id counts independently",
      autoqueue.register_iteration("t2", 7)["status"] == "2/7")

# ---------------------------------------------------------------------------
# T3: reset()

autoqueue.reset("t1")
check("A6 reset clears the counter",
      autoqueue.register_iteration("t1", 3)["status"] == "1/3")

# ---------------------------------------------------------------------------
# T4: the requeue intent lifecycle (worker contract: consume == clear)

autoqueue.note_requeue("t1")
check("A7 one note -> one take", autoqueue.take_requeue() == "t1")
check("A8 take is idempotent (nothing pending afterwards)",
      autoqueue.take_requeue() is None)

# ---------------------------------------------------------------------------
# T5: the CdlBatchLoop node contract

from comfy_extras.nodes_autoqueue import CdlBatchLoop, CdlOutputGate  # noqa: E402

autoqueue._counters.clear()
out1 = CdlBatchLoop.execute(iterations=2, chain_id="node")
check("B1 first round: 1/2, index 0, not final, intent recorded",
      out1.args[0] == "1/2" and out1.args[1] == 0 and out1.args[2] is False
      and autoqueue.take_requeue() == "node")
out2 = CdlBatchLoop.execute(iterations=2, chain_id="node")
check("B2 final round: 2/2, index 1, final, NO intent",
      out2.args[0] == "2/2" and out2.args[1] == 1 and out2.args[2] is True
      and autoqueue.take_requeue() is None)
out_single = CdlBatchLoop.execute(iterations=1, chain_id="solo")
check("B3 iterations=1 runs once and never re-queues",
      out_single.args[2] is True and autoqueue.take_requeue() is None)

# ---------------------------------------------------------------------------
# T6: the CdlOutputGate ExecutionBlocker semantics

from comfy_execution.graph import ExecutionBlocker  # noqa: E402

out_open = CdlOutputGate.execute(gate=True, value="payload")
check("C1 open gate passes the value through", out_open.args[0] == "payload")
out_closed = CdlOutputGate.execute(gate=False, value="payload")
check("C2 closed gate returns an ExecutionBlocker placeholder",
      isinstance(out_closed.args[0], ExecutionBlocker))

# ---------------------------------------------------------------------------
# T7: registration + estimator binding

import nodes as host_nodes  # noqa: E402

asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                        init_api_nodes=False))
registry = host_nodes.NODE_CLASS_MAPPINGS
check("D1 CdlBatchLoop registered", "CdlBatchLoop" in registry)
check("D2 CdlOutputGate registered", "CdlOutputGate" in registry)
check("D3 estimators bound for both control nodes",
      "CdlBatchLoop" in est_mod.ESTIMATORS
      and "CdlOutputGate" in est_mod.ESTIMATORS)

# ---------------------------------------------------------------------------
# T8: describe() snapshot shape

snap = autoqueue.describe()
check("E1 describe() carries chains + pending slot",
      "chains" in snap and "pending_requeue" in snap
      and snap["pending_requeue"] is None
      and "node" in snap["chains"])


def _report() -> int:
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL "
          f"({len(_RESULTS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_report())
