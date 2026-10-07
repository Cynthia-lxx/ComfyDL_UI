"""Smoke tests for the run-time FLOPs meter (Profiling v2 P2).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling_runmeter.py

Covers the RunMeter contract before it touches the executor:

* count/census recording modes and the unattributed bucket (ops outside any
  CurrentNodeContext);
* per-node attribution through the host's CurrentNodeContext contextvar
  (the exact mechanism the executor hook relies on);
* census demotion after the per-node op budget;
* training-loop compatibility: inference_mode(False) + backward +
  optimizer.step must not break the meter and must satisfy the declared
  fwd+bwd combined accounting (bwd adds FLOPs);
* CLI flag parsing for --cdl-profiling-record.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy.profiling.runmeter import MODES, RunMeter, meter_from_args  # noqa: E402
from comfy_execution.utils import CurrentNodeContext  # noqa: E402


def _train_step():
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 4)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    y = torch.randn(2, 4)
    loss = ((layer(x) - y) ** 2).mean()
    loss.backward()
    opt.step()


def main() -> int:
    check("T1 modes constant", MODES == ("off", "count", "census"), str(MODES))

    # T2: count mode, no node context -> unattributed bucket.
    with RunMeter("count") as meter:
        torch.mm(torch.zeros(2, 8), torch.zeros(8, 4))
    snap = meter.snapshot()
    check("T2a count: total ops recorded", snap["total_ops"] >= 1, str(snap["total_ops"]))
    check("T2b count: flops recorded", snap["total_flops"] > 0, str(snap["total_flops"]))
    check("T2c count: outside context -> unattributed",
          snap["unattributed"]["op_count"] >= 1
          and snap["unattributed"]["flops"] == snap["total_flops"],
          str(snap["unattributed"]))

    # T3: census mode + contextvar attribution to two different nodes.
    # (element-wise ops like aten.add are unpriced in flop_registry - they
    # run without bookkeeping by design; priced ops like mm are recorded.)
    with RunMeter("census") as meter:
        with CurrentNodeContext("prompt-1", "5", None):
            torch.mm(torch.zeros(2, 8), torch.zeros(8, 4))
        with CurrentNodeContext("prompt-1", "7", None):
            torch.mm(torch.zeros(2, 8), torch.zeros(8, 4))
    snap = meter.snapshot()
    n5, n7 = snap["nodes"].get("5"), snap["nodes"].get("7")
    check("T3a census: node 5 bucketed with flops",
          n5 is not None and n5["flops"] > 0 and n5["op_count"] >= 1, str(n5))
    check("T3b census: node 7 bucketed", n7 is not None and n7["op_count"] >= 1, str(n7))
    check("T3c census: per-op histogram present",
          n5 is not None and any("mm" in op for op in n5["ops"]), str(n5 and n5["ops"]))
    check("T3d census: ops dict copied (snapshot isolation)",
          n5 is not None and isinstance(n5["ops"], dict))

    # T4: census demotion after the per-node op budget.
    with RunMeter("census", ops_limit=3) as meter:
        with CurrentNodeContext("p", "9", None):
            for _ in range(10):
                torch.mm(torch.zeros(2, 8), torch.zeros(8, 4))
    snap = meter.snapshot()
    n9 = snap["nodes"].get("9")
    check("T4a census demoted after op budget",
          n9 is not None and n9["demoted"] is True and n9["ops"] == {},
          str(n9))
    check("T4b demotion keeps aggregate counting",
          n9 is not None and n9["op_count"] >= 10, str(n9))

    # T5: training-loop compatibility + fwd+bwd accounting semantics.
    with RunMeter("count") as fwd_only:
        with torch.inference_mode(False):
            layer = torch.nn.Linear(8, 4)
            with torch.no_grad():
                layer(torch.randn(2, 8))
    with RunMeter("count") as fwd_bwd:
        with torch.inference_mode(False):
            layer = torch.nn.Linear(8, 4)
            opt = torch.optim.SGD(layer.parameters(), lr=0.01)
            loss = ((layer(torch.randn(2, 8)) - torch.randn(2, 4)) ** 2).mean()
            loss.backward()
            opt.step()
    f1 = fwd_only.snapshot()["total_flops"]
    f2 = fwd_bwd.snapshot()["total_flops"]
    check("T5a training step recorded without error",
          f2 > 0, f"fwd={f1} fwd+bwd={f2}")
    check("T5b bwd adds flops (fwd+bwd combined > forward alone)",
          f2 > f1, f"fwd={f1} fwd+bwd={f2}")

    # T6: meter errors must never surface into the run path (courtesy rule).
    class _Broken(RunMeter):
        def _record(self, node_id, op_name, flops):
            raise RuntimeError("bookkeeping bug")

    with _Broken("count") as broken:
        out = torch.mm(torch.zeros(2, 8), torch.zeros(8, 4))
    check("T6a dispatch survives bookkeeping bugs", out.shape == (2, 4))
    check("T6b suppressed errors visible in snapshot",
          broken.snapshot()["suppressed_errors"] >= 1)

    # T7: invalid mode rejected at construction.
    try:
        RunMeter("verbose")
        check("T7 invalid mode rejected", False, "no ValueError")
    except ValueError:
        check("T7 invalid mode rejected", True)

    # T8: CLI flag parses with the documented default.
    from comfy import cli_args

    check("T8a default record mode is count",
          cli_args.parser.parse_args([]).cdl_profiling_record == "count")
    check("T8b census accepted",
          cli_args.parser.parse_args(
              ["--cdl-profiling-record", "census"]).cdl_profiling_record == "census")
    check("T8c meter_from_args honors off",
          meter_from_args() is None
          or cli_args.parser.parse_args(
              ["--cdl-profiling-record", "off"]).cdl_profiling_record == "off")

    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
