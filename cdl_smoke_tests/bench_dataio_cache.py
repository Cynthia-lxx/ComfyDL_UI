#!/usr/bin/env python
"""Benchmark for the data_io DataFrame cache (SP-6c, user-mandated).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\bench_dataio_cache.py

Measures (median of N repeats): cold (cache OFF) / miss / hit on a ~32 MB CSV;
a 20-shard sweep; small-batch overhead (100-row file x 200 runs); memory
ledger vs budget under FIFO pressure with an RSS cross-check.
Experiment report, not a gate: prints numbers, asserts structural sanity only.
"""

import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402


def _median(fn, n=5):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_diobench_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    import psutil  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS
    from comfy.profiling import switch_registry  # noqa: E402

    from comfydl.nodes import _dataio_cache as dc

    proc = psutil.Process()
    reader = REG["CdlReadCSV"]()
    report: list = []

    # 1. cold / miss / hit on a ~32 MB CSV
    big = sandbox / "big.csv"
    rows_n = 200_000
    with big.open("w", encoding="utf-8") as fh:
        fh.write(",".join(f"x{i}" for i in range(20)) + "\n")
        for r in range(rows_n):
            fh.write(",".join(str((r * 7 + c) % 1000) for c in range(20)) + "\n")
    size_mb = big.stat().st_size / 1e6

    switch_registry.set_enabled(dc.SWITCH_ID, False)
    cold = _median(lambda: reader.execute(str(big), ""))
    switch_registry.set_enabled(dc.SWITCH_ID, True)
    # miss = a fresh first parse each repeat (clear between repeats, else the
    # median over N repeats collapses into hits)
    miss = _median(lambda: (dc.clear(), reader.execute(str(big), "")), n=3)
    hit = _median(lambda: reader.execute(str(big), ""))
    report.append(f"[1] {rows_n}x20 CSV ({size_mb:.1f} MB), median:")
    report.append(f"    cold (cache OFF) : {cold * 1e3:9.1f} ms")
    report.append(f"    miss (cache ON)  : {miss * 1e3:9.1f} ms")
    report.append(f"    hit  (cache ON)  : {hit * 1e3:9.1f} ms")
    report.append(f"    speedup hit/cold : {cold / max(hit, 1e-9):9.0f}x")

    # 2. shard sweep: 20 shards share one parse
    dc.clear()
    t0 = time.perf_counter()
    for s in range(20):
        reader.execute(str(big), "", start_idx=s * 1000, num_samples=500)
    sweep = time.perf_counter() - t0
    report.append(f"[2] 20 shards of one file: {sweep * 1e3:.1f} ms total "
                  f"(vs ~{cold * 20 * 1e3:.0f} ms re-parsing each)")
    report.append(f"    cache entries after sweep: {dc.stats()['entries']} (expect 1)")

    # 3. small-batch overhead: tiny file, cache ON vs OFF
    small = sandbox / "small.csv"
    small.write_text(
        "a,b\n" + "\n".join(f"{i},{i}" for i in range(100)) + "\n",
        encoding="utf-8")
    switch_registry.set_enabled(dc.SWITCH_ID, False)
    off = _median(lambda: reader.execute(str(small), ""), n=200)
    switch_registry.set_enabled(dc.SWITCH_ID, True)
    dc.clear()
    on = _median(lambda: reader.execute(str(small), ""), n=200)
    report.append(f"[3] 100-row file x 200 runs (median): OFF {off * 1e3:.2f} ms "
                  f"| ON {on * 1e3:.2f} ms | delta {(on - off) * 1e6:+.0f} us")

    # 4. memory: ledger vs budget under FIFO pressure, RSS cross-check
    dc.clear()
    real_budget, real_entries = dc.MAX_TOTAL_BYTES, dc.MAX_ENTRIES
    try:
        dc.MAX_TOTAL_BYTES = 8 * 1024 * 1024
        dc.MAX_ENTRIES = 4
        rss_before = proc.memory_info().rss
        for i in range(8):
            p = sandbox / f"fill_{i}.csv"
            with p.open("w", encoding="utf-8") as fh:
                fh.write(",".join(f"c{j}" for j in range(20)) + "\n")
                for r in range(6_000):  # ~0.96 MB/frame: 4 frames fit 8 MB
                    fh.write(",".join(str((r + i) % 97) for j in range(20)) + "\n")
            reader.execute(str(p), "")
        rss_after = proc.memory_info().rss
        st = dc.stats()
        report.append(f"[4] fill 8 x ~0.96 MB frames with budget "
                      f"{dc.MAX_TOTAL_BYTES // 1024 // 1024} MB / "
                      f"{dc.MAX_ENTRIES} entries:")
        report.append(f"    entries kept        : {st['entries']} (cap 4)")
        report.append(f"    ledger              : "
                      f"{st['cached_bytes'] / 1e6:.1f} MB "
                      f"(budget {st['budget_bytes'] / 1e6:.0f} MB)")
        report.append(f"    evictions           : {st['evictions']}")
        report.append(f"    RSS delta (proc)    : "
                      f"{(rss_after - rss_before) / 1e6:+.1f} MB "
                      f"(includes parser scratch, not just the ledger)")
        report.append(f"    sanity: ledger <= budget : "
                      f"{st['cached_bytes'] <= st['budget_bytes']}")
    finally:
        dc.MAX_TOTAL_BYTES = real_budget
        dc.MAX_ENTRIES = real_entries
        dc.clear()

    report.append(f"[5] switch self-overhead ledger: "
                  f"{switch_registry.self_overhead(dc.SWITCH_ID)}")

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
