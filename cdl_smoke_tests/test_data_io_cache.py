#!/usr/bin/env python
"""Standalone smoke test for the data_io sharding + DataFrame cache (SP-6c).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_io_cache.py

Covers:
  * the ``start_idx``/``num_samples`` shard window on the file readers
    (all-rows sentinel, partial shard, boundary/empty-value errors),
  * the DataFrame cache invariants: parse-parameter keys, file-change keys,
    shard-change immunity (the NNT defect fixed *by construction*),
  * the ``comfydl.dataio.cache`` switch (registry entry, hot toggle,
    self-metering ledger),
  * the byte budget / FIFO eviction / too-large rejection memory discipline.
"""

import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _expect_error(name, fn, needle):
    try:
        fn()
        check(name, False, "no error raised")
    except ValueError as exc:
        check(name, needle in str(exc), str(exc)[:110])


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_diocache_"))
    harness._bootstrap(sandbox)
    harness._install_sandbox_paths(sandbox)  # redirect folder_paths input root
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS

    import pandas as pd  # noqa: E402

    from comfydl.nodes import _dataio_cache as dc

    # ------------------------------------------------------------------ #
    # fixture: a 20-row CSV
    csv_path = sandbox / "fixture.csv"
    rows = ["a,b,y"] + [f"{i},{i * 10},{i / 100.0}" for i in range(20)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dc.clear()

    reader = REG["CdlReadCSV"]()

    # 1. shard windows ------------------------------------------------------
    ds_all = reader.execute(str(csv_path), "y")[0]
    check("shard default reads all", ds_all.n_samples == 20)
    ds_part = reader.execute(str(csv_path), "y", start_idx=5, num_samples=7)[0]
    check("shard partial size", ds_part.n_samples == 7)
    check("shard partial content",
          float(ds_part.features[0, 0]) == 5.0
          and float(ds_part.features[-1, 0]) == 11.0)
    ds_tail = reader.execute(str(csv_path), "y", start_idx=15, num_samples=99)[0]
    check("shard clamps to table end", ds_tail.n_samples == 5)

    _expect_error("shard start beyond table",
                  lambda: reader.execute(str(csv_path), "y", start_idx=20, num_samples=1),
                  "out of range")
    _expect_error("shard num_samples=0 ambiguous",
                  lambda: reader.execute(str(csv_path), "y", num_samples=0),
                  "ambiguous")
    _expect_error("shard negative start",
                  lambda: reader.execute(str(csv_path), "y", start_idx=-3, num_samples=1),
                  "must be >= 0")

    empty_path = sandbox / "empty.csv"
    empty_path.write_text("a,b\n", encoding="utf-8")
    _expect_error("shard empty source",
                  lambda: reader.execute(str(empty_path), ""),
                  "0 rows")

    # 2. cache: hit / miss / key coverage ------------------------------------
    dc.clear()
    before = dc.stats()
    reader.execute(str(csv_path), "y")
    after_miss = dc.stats()
    check("first read is a miss",
          after_miss["misses"] == before["misses"] + 1
          and after_miss["entries"] == 1)
    reader.execute(str(csv_path), "y")
    after_hit = dc.stats()
    check("second read is a hit", after_hit["hits"] == before["hits"] + 1)

    # same file, different delimiter -> different parse -> different key.
    # (";" merges the header into one column, so from_dataframe may reject the
    # schema afterwards — the cache miss is what we assert, store ran first.)
    try:
        reader.execute(str(csv_path), "y", delimiter=";")
    except (ValueError, KeyError):
        pass
    check("delimiter change is a miss",
          dc.stats()["misses"] == after_miss["misses"] + 1,
          str(dc.stats()))

    # shard changes must NOT multiply cache entries (structural immunity)
    entries_before = dc.stats()["entries"]
    for start, num in ((0, 3), (3, 4), (7, 2), (10, 10)):
        reader.execute(str(csv_path), "y", start_idx=start, num_samples=num)
    check("shard changes share one cached parse",
          dc.stats()["entries"] == entries_before,
          f"entries {entries_before} -> {dc.stats()['entries']}")

    # file content change (mtime/size) -> new key -> miss.
    # Reset the ledger first: the ";" attempt above owns its own entry.
    dc.clear()
    reader.execute(str(csv_path), "y")
    base_entries = dc.stats()["entries"]
    time.sleep(0.02)  # mtime_ns resolution safety on coarse filesystems
    csv_path.write_text(
        "\n".join(rows + ["100,1000,1.0"]) + "\n", encoding="utf-8")
    reader.execute(str(csv_path), "y")
    check("file change invalidates the cache",
          dc.stats()["entries"] == base_entries + 1,
          f"base={base_entries} now={dc.stats()['entries']}")

    # 3. switch: registry entry + hot toggle + self-metering ------------------
    from comfy.profiling import switch_registry

    state = switch_registry.describe(dc.SWITCH_ID)
    check("switch registered",
          bool(state) and state.get("self_meter") is True
          and state.get("hot_toggle") is True,
          str(sorted(switch_registry.list_switches()))[:80])
    prev = switch_registry.is_enabled(dc.SWITCH_ID)
    try:
        switch_registry.set_enabled(dc.SWITCH_ID, False)
        dc.clear()
        entries_before = dc.stats()["entries"]
        reader.execute(str(csv_path), "y")
        check("switch off -> no caching",
              dc.stats()["entries"] == entries_before)
    finally:
        switch_registry.set_enabled(dc.SWITCH_ID, prev)
    ov = switch_registry.self_overhead(dc.SWITCH_ID)
    check("self-metering ledger fed",
          ov.get("samples", 0) > 0 and ov.get("total_s", 1.0) >= 0.0,
          str(ov))

    # 4. memory discipline: byte budget + FIFO eviction -----------------------
    dc.clear()
    real_budget = dc.MAX_TOTAL_BYTES
    real_entries = dc.MAX_ENTRIES
    try:
        dc.MAX_TOTAL_BYTES = 4096   # tiny budget: one 100-row frame evicts
        dc.MAX_ENTRIES = 3
        for i in range(6):
            p = sandbox / f"evict_{i}.csv"
            p.write_text(
                "a,b,c\n" + "\n".join(
                    f"{j},{j},{j}" for j in range(100)) + "\n",
                encoding="utf-8")
            reader.execute(str(p), "")
        st = dc.stats()
        check("entry cap respected", st["entries"] <= 3, str(st))
        check("byte ledger within budget",
              st["cached_bytes"] <= dc.MAX_TOTAL_BYTES, str(st))
        check("evictions happened", st["evictions"] >= 3, str(st))
        check("ledger visible in stats",
              st["budget_bytes"] == 4096 and st["max_entries"] == 3)
    finally:
        dc.MAX_TOTAL_BYTES = real_budget
        dc.MAX_ENTRIES = real_entries
        dc.clear()

    # too-large rejection: one frame bigger than the whole budget
    dc.MAX_TOTAL_BYTES = 64  # bytes: any real frame exceeds this
    try:
        dc.store("big", pd.DataFrame({"a": range(10)}))
        st = dc.stats()
        check("oversized frame rejected",
              st["entries"] == 0 and st["rejected_too_large"] >= 1, str(st))
    finally:
        dc.MAX_TOTAL_BYTES = real_budget
        dc.clear()

    # 5. other readers shard too (JSON + inline STRING) ------------------------
    json_path = sandbox / "fixture.json"
    pd.DataFrame({"a": range(12), "b": range(12, 24)}).to_json(
        json_path, orient="records")
    ds_j = REG["CdlReadJSON"]().execute(
        str(json_path), "", start_idx=4, num_samples=3)[0]
    check("json shard", ds_j.n_samples == 3 and float(ds_j.features[0, 0]) == 4.0)

    text_data = "\n".join(
        "\t".join(str(v) for v in range(4)) for _ in range(10))
    ds_s = REG["CdlReadString"]().execute(
        text_data, "", header=False, start_idx=2, num_samples=4)[0]
    check("string shard", ds_s.n_samples == 4)

    # 5b. input-directory path resolution (ComfyUI convention) ------------------
    # The harness chdirs into the sandbox and redirects folder_paths' input
    # root to sandbox/input; a bare filename must resolve from there first,
    # then the CWD (where the seed fixtures live), and absolute paths must
    # stay untouched.
    from comfydl.nodes.data_io import _resolve_input_path

    (sandbox / "input" / "in_dir.csv").write_text("p,q\n1,2\n", encoding="utf-8")
    ds_in = reader.execute("in_dir.csv", "q")[0]
    check("bare filename resolves from the input directory", ds_in.n_samples == 1)
    check("absolute path passes through",
          _resolve_input_path(str(csv_path)) == str(csv_path))
    check("cwd-relative still resolves (legacy fallback)",
          Path(_resolve_input_path("cdl_dataset.csv")).is_file())

    def _expect_filenf(name, fn):
        try:
            fn()
            check(name, False, "no error raised")
        except FileNotFoundError as exc:
            check(name, "tried:" in str(exc), str(exc)[:120])

    _expect_filenf(
        "missing file names every candidate",
        lambda: reader.execute("nope_nowhere.csv", ""))

    # 5c. single-line giant file guard (the executor-wedge hotfix) ------------
    # A >1MB file with no line break must fail FAST with a readable error
    # instead of wedging the executor in the C parser for hours.
    giant = sandbox / "single_line.csv"
    giant.write_text(",".join(str(i % 100) for i in range(400_000)),
                     encoding="utf-8")  # ~1 MB of numeric CSV, zero newlines
    t0 = time.perf_counter()
    _expect_error("single-line giant file fails fast",
                  lambda: reader.execute(str(giant), ""),
                  "no line break")
    check("guard fails immediately",
          time.perf_counter() - t0 < 1.0,
          f"{(time.perf_counter() - t0):.2f}s")
    # opt-out: the Advanced toggle bypasses the guard (robustness is a
    # choice, not an unconditional gate) — numeric single-line data parses
    ds_raw = reader.execute(str(giant), "", header=False,
                            single_line_guard=False)[0]
    check("guard opt-out parses anyway", ds_raw.n_samples == 1)
    spec_adv = REG["CdlReadCSV"].INPUT_TYPES()["optional"]["single_line_guard"]
    check("guard toggle sits in the Advanced section",
          spec_adv[1].get("advanced") is True)

    # 6. Preview ShowText contract (SP-6c) --------------------------------------
    from comfydl.nodes.data_io_writers import (
        PREVIEW_WRITEBACK_MAX_CHARS,
        PREVIEW_WRITEBACK_MAX_LINES,
    )
    from comfydl.nodes.data_types import CdlDataset

    prev_cls = REG["CdlDatasetPreview"]
    check("preview is OUTPUT_NODE", bool(getattr(prev_cls, "OUTPUT_NODE", False)))
    prev = prev_cls()
    wide = pd.DataFrame({f"c{i}": range(500) for i in range(10)})
    wf = {"workflow": {"nodes": [
        {"id": 31, "inputs": [], "widgets_values": [5, ""]}]}}

    out = prev.execute(
        CdlDataset.from_tensors(wide.values.astype("float32"), None,
                                [f"c{i}" for i in range(10)]),
        60, 31, wf)
    check("preview returns ui dict",
          isinstance(out, dict) and out["ui"]["text"]
          and isinstance(out["result"], tuple))
    wv = wf["workflow"]["nodes"][0]["widgets_values"]
    check("writeback widget order [head, text]",
          wv[0] == 60 and isinstance(wv[1], str) and "c0" in wv[1])
    lines = wv[1].splitlines()
    check("writeback clipped to line cap",
          len(lines) == PREVIEW_WRITEBACK_MAX_LINES + 1
          and "truncated" in lines[-1], f"{len(lines)} lines")
    check("writeback clipped per line",
          all(len(l) <= PREVIEW_WRITEBACK_MAX_CHARS for l in lines))
    check("result string unclipped",
          out["result"][0].count("\n") == 60)  # 60 rows + header line

    # no workflow payload -> degrade gracefully (no write, no crash)
    out2 = prev.execute(
        CdlDataset.from_tensors(wide.values[:5].astype("float32"), None,
                                [f"c{i}" for i in range(10)]), 5, 31, None)
    check("preview without workflow payload",
          isinstance(out2, dict) and out2["result"][0].count("\n") == 5)

    # nested subgraph reflection reaches the preview node too
    nested = {"workflow": {
        "nodes": [{"id": 19, "type": "sg-x", "inputs": []}],
        "definitions": {"subgraphs": [
            {"id": "sg-x", "nodes": [
                {"id": 7, "inputs": [], "widgets_values": [5, ""]}]}]}}}
    prev.execute(
        CdlDataset.from_tensors(wide.values[:5].astype("float32"), None,
                                [f"c{i}" for i in range(10)]),
        5, "19:7", nested)
    inner = nested["workflow"]["definitions"]["subgraphs"][0]["nodes"][0]
    check("preview writeback inside subgraph",
          inner["widgets_values"][1].startswith("    c0") or "c0" in inner["widgets_values"][1])

    # 7. frontend contract: show_text.js + WEB_DIRECTORY wiring ----------------
    js_path = REPO_ROOT / "comfydl" / "web" / "js" / "show_text.js"
    check("show_text.js exists", js_path.is_file())
    if js_path.is_file():
        js = js_path.read_text(encoding="utf-8")
        check("js official imports",
              '"/scripts/app.js"' in js and '"/scripts/widgets.js"' in js)
        check("js targets preview node", '"CdlDatasetPreview"' in js)
        check("js onExecuted + onConfigure",
              "onExecuted" in js and "onConfigure" in js)
        check("js widgets_values[1:] restore", "v.slice(1)" in js)
        check("js readonly display", "readOnly" in js)
        check("js readonly survives re-render",
              'setAttribute("readonly"' in js and "requestAnimationFrame" in js)
        check("js no eval", "eval(" not in js and "new Function" not in js)
        import subprocess
        try:
            proc = subprocess.run(
                ["node", "--check", str(js_path)],
                capture_output=True, text=True, timeout=30)
            check("js node --check", proc.returncode == 0, proc.stderr[:160])
        except (OSError, subprocess.TimeoutExpired):
            check("js node --check", True, "node unavailable; skipped")

    init_src = (REPO_ROOT / "comfydl" / "__init__.py").read_text(encoding="utf-8")
    check("WEB_DIRECTORY declared", 'WEB_DIRECTORY = "./web"' in init_src)
    nodes_src = (REPO_ROOT / "nodes.py").read_text(encoding="utf-8")
    check("web dir registered in nodes.py",
          'EXTENSION_WEB_DIRS["comfydl"]' in nodes_src)

    # stats acceptance surface: route + panel logging
    routes_src = (REPO_ROOT / "app" / "spine" / "routes.py").read_text(
        encoding="utf-8")
    check("cache stats route",
          '"/comfydl/dataio/cache/stats"' in routes_src
          and '"available": False' in routes_src)
    perf_src = (REPO_ROOT / "app" / "perf_assets" / "perftools.js").read_text(
        encoding="utf-8")
    check("panel logs cache stats",
          "logCacheStats" in perf_src
          and "/comfydl/dataio/cache/stats" in perf_src)

    # the route handlers themselves (spine app, no torch needed)
    try:
        from aiohttp import web as _webmod

        # seed one entry so the inventory endpoint has something to list
        dc.clear()
        dc.store("probe_key", pd.DataFrame({"a": range(10)}),
                 meta=dc.describe("csv", "probe_fixture.csv", str(csv_path),
                                  "sep=',' header=on"))

        async def _probe(path):
            from aiohttp.test_utils import TestClient, TestServer

            app = _webmod.Application()
            import importlib
            routes_mod = importlib.import_module("app.spine.routes")
            app.add_routes(routes_mod.routes)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get(path)
                body = await resp.json()
                return resp.status, body
            finally:
                await client.close()

        import asyncio
        _loop = asyncio.new_event_loop()
        status, body = _loop.run_until_complete(
            _probe("/comfydl/dataio/cache/stats"))
        check("stats route responds",
              status == 200 and body.get("available") is True
              and {"hits", "misses", "entries", "cached_bytes"} <= set(body),
              str(body)[:110])
        status, body = _loop.run_until_complete(
            _probe("/comfydl/dataio/cache/entries"))
        check("entries route responds",
              status == 200 and body.get("available") is True
              and body.get("enabled") is not None
              and isinstance(body.get("stats"), dict)
              and isinstance(body.get("entries"), list)
              and len(body["entries"]) == 1
              and body["entries"][0]["reader"] == "csv"
              and body["entries"][0]["label"] == "probe_fixture.csv"
              and body["entries"][0]["hits"] == 0,
              str(body)[:160])
        _loop.close()
        dc.clear()
    except Exception as exc:  # noqa: BLE001
        check("stats route responds", False, str(exc)[:110])
        check("entries route responds", False, str(exc)[:110])

    # 8. per-entry metadata (Data cache card backend) --------------------------
    import math

    dc.clear()
    reader.execute(str(csv_path), "y")
    ents = dc.entries()
    check("meta recorded on store",
          len(ents) == 1 and ents[0]["reader"] == "csv"
          and ents[0]["label"] == "fixture.csv"
          and ents[0]["cached_bytes"] > 0
          and ents[0]["hits"] == 0 and ents[0]["last_hit"] is None,
          str(ents)[:130])
    check("meta stamp matches the file",
          bool(ents) and ents[0]["file_size"] == csv_path.stat().st_size
          and ents[0]["mtime_ns"] == csv_path.stat().st_mtime_ns
          and Path(ents[0]["path"]).is_file())
    check("meta params recorded",
          bool(ents) and "sep=','" in ents[0]["params"],
          str(ents[0]["params"])[:60] if ents else "no entries")
    reader.execute(str(csv_path), "y")
    ents = dc.entries()
    check("meta hit ledger updates",
          len(ents) == 1 and ents[0]["hits"] == 1
          and ents[0]["last_hit"] is not None, str(ents)[:130])

    # eviction pops the side table in lockstep
    dc.MAX_TOTAL_BYTES = 64
    try:
        dc.store("meta_evict", pd.DataFrame({"a": range(10)}),
                 meta=dc.describe("csv", "evict_me.csv", None))
        check("eviction drops meta too",
              all(e["label"] != "evict_me.csv" for e in dc.entries()),
              str([e["label"] for e in dc.entries()])[:100])
    finally:
        dc.MAX_TOTAL_BYTES = real_budget
        dc.clear()

    # IS_CHANGED: the official execution-cache fingerprint (V1 contract:
    # same-args-as-FUNCTION, compared with !=, never a bool)
    csv_cls = REG["CdlReadCSV"]
    fp = csv_cls.IS_CHANGED(file=str(csv_path))
    check("IS_CHANGED returns the file stamp",
          isinstance(fp, tuple) and len(fp) == 3
          and fp[1] == csv_path.stat().st_mtime_ns
          and fp[2] == csv_path.stat().st_size, str(fp)[:110])
    fp_missing = csv_cls.IS_CHANGED(file="nope_nowhere.csv")
    check("IS_CHANGED missing file -> NaN (forces re-run)",
          isinstance(fp_missing, float) and math.isnan(fp_missing),
          str(fp_missing))
    check("IS_CHANGED never returns bool",
          not isinstance(fp, bool) and not isinstance(fp_missing, bool))
    db_fp = REG["CdlReadDB"].IS_CHANGED(db_path=str(csv_path))
    check("IS_CHANGED honours the db_path parameter",
          isinstance(db_fp, tuple) and db_fp == fp, str(db_fp)[:110])
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    check("IS_CHANGED tracks file change",
          csv_cls.IS_CHANGED(file=str(csv_path)) != fp)
    dc.clear()

    # 9. frontend contract: Data cache card in the Profiling panel -------------
    prof_src = (REPO_ROOT / "app" / "profiling_assets" / "profiler.js").read_text(
        encoding="utf-8")
    check("profiler data cache card",
          "fetchCacheEntries" in prof_src
          and "/comfydl/dataio/cache/entries" in prof_src
          and "datacache_title" in prof_src
          and "cdlp-cache-refresh" in prof_src)
    css_src = (REPO_ROOT / "app" / "profiling_assets" / "profiler.css").read_text(
        encoding="utf-8")
    check("profiler data cache css",
          "cdlp-cache-item" in css_src and "cdlp-cache-path" in css_src)
    import subprocess
    try:
        proc = subprocess.run(
            ["node", "--check",
             str(REPO_ROOT / "app" / "profiling_assets" / "profiler.js")],
            capture_output=True, text=True, timeout=30)
        check("profiler.js node --check", proc.returncode == 0,
              proc.stderr[:160])
    except (OSError, subprocess.TimeoutExpired):
        check("profiler.js node --check", True, "node unavailable; skipped")

    _fail = [r for r in _RESULTS if not r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== data_io cache test: "
          f"{len(_RESULTS) - len(_fail)} PASS / {len(_fail)} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
