#!/usr/bin/env python
"""Standalone smoke test for the Dataset Pack/Unpack family (data_ops.py).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_data_ops.py

Exercises the column-to-port dynamic binding machinery:
  * wildcard toolkit (AnyType / ByPassTypeTuple / ContainsDynamicDict),
  * workflow reflection incl. nested-subgraph execution-ID chains,
  * CdlDatasetPack contract (schema from port names, target detection,
    provenance metadata, error contract).

The Unpack half lives in the same file and is exercised by sections marked
"Unpack" below.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


PACK_CLASS_NAME = "CdlDatasetPack"


def _brackets_balanced(src: str) -> bool:
    """Bracket balance fallback when node is unavailable (comments/strings naively stripped)."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list = []
    for ch in src:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_dops_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)
    import nodes as host_nodes  # noqa: E402

    REG = host_nodes.NODE_CLASS_MAPPINGS

    import torch  # noqa: E402

    from comfydl.nodes import data_ops  # noqa: E402
    from comfydl.nodes.data_ops import (
        ANY,
        COL_PREFIX,
        AnyType,
        ByPassTypeTuple,
        CdlDatasetPack,
        ContainsDynamicDict,
        resolve_workflow_node,
    )
    from comfydl.nodes.data_types import CdlDataset  # noqa: E402

    # 1. registration + contract --------------------------------------------
    for nid in ("CdlDatasetPack",):
        check(f"registered:{nid}", nid in REG, "" if nid in REG else "missing")
    check("Pack RETURN_TYPES", REG["CdlDatasetPack"].RETURN_TYPES == ("DATASET",))
    check("Pack CATEGORY", REG["CdlDatasetPack"].CATEGORY == "d2l/DataOps")
    hidden = REG["CdlDatasetPack"].INPUT_TYPES().get("hidden", {})
    check("Pack hidden unique_id", hidden.get("unique_id") == "UNIQUE_ID")
    check("Pack hidden extra_pnginfo",
          hidden.get("extra_pnginfo") == "EXTRA_PNGINFO")

    # 2. wildcard toolkit -----------------------------------------------------
    check("ANY is AnyType", isinstance(ANY, AnyType))
    check("ANY ne TENSOR", not (ANY != "TENSOR"))
    check("ANY ne ANY", not (ANY != ANY))
    # Reflected comparison: because AnyType is a str subclass overriding
    # __ne__, Python routes "TENSOR" != ANY through AnyType.__ne__ too —
    # the wildcard must pass validation in BOTH directions.
    check("TENSOR ne ANY (reflected)", not ("TENSOR" != ANY))
    bypass = ByPassTypeTuple((ANY,))
    check("ByPass len 1", len(bypass) == 1)
    check("ByPass [0] wildcard", isinstance(bypass[0], AnyType))
    check("ByPass [7] wildcard", isinstance(bypass[7], AnyType))
    check("ByPass [-1] passthrough", isinstance(bypass[-1], AnyType))

    dyn = ContainsDynamicDict({
        "col_0": ("TENSOR", {"_dynamic": "number"}),
    })
    check("dyn seed contains", "col_0" in dyn)
    check("dyn grown contains", "col_5" in dyn)
    check("dyn grown getitem", dyn["col_5"] == dyn["col_0"])
    check("dyn prefix guard", "column_5" not in dyn and "col_x" not in dyn)
    check("dyn unrelated miss", "target" not in dyn)

    check("COL_PREFIX", COL_PREFIX == "col_")

    # 3. workflow reflection --------------------------------------------------
    root_workflow = {
        "nodes": [
            {"id": 7, "inputs": [
                {"name": "col_0", "label": "x0"},
                {"name": "col_1", "label": "target"},
                {"name": "col_2"},
            ]},
            {"id": 8, "inputs": []},
        ]
    }
    nested_workflow = {
        "nodes": [{"id": 19, "type": "sg-a", "inputs": []}],
        "definitions": {"subgraphs": [
            {"id": "sg-a", "nodes": [
                {"id": 84, "inputs": [{"name": "col_0", "label": "depth"}]},
            ]},
        ]},
    }

    node, graph = resolve_workflow_node({"workflow": root_workflow}, 7)
    check("reflect root node", bool(node) and node["id"] == 7)
    check("reflect root graph is workflow", graph is root_workflow)
    node, graph = resolve_workflow_node(
        [{"workflow": nested_workflow}], "19:84")
    check("reflect nested subgraph node",
          bool(node) and node["id"] == 84 and node["inputs"][0]["label"] == "depth")
    check("reflect nested graph is subgraph def",
          bool(graph) and graph.get("id") == "sg-a")
    check("reflect missing workflow",
          resolve_workflow_node(None, 7) == (None, None))
    check("reflect broken chain",
          resolve_workflow_node({"workflow": root_workflow}, "7:99") == (None, None))
    check("reflect unknown root id",
          resolve_workflow_node({"workflow": root_workflow}, 999) == (None, None))

    names = data_ops.reflect_input_names(
        {"workflow": root_workflow}, 7, ["col_0", "col_1"])
    check("reflect label wins", names == ["x0", "target"])
    names = data_ops.reflect_input_names(None, None, ["col_0", "col_1"])
    check("reflect fallback", names == ["col_0", "col_1"])

    # 4. Pack: schema from port names + target detection ----------------------
    N = 6
    packer = REG["CdlDatasetPack"]()
    wf = {"workflow": root_workflow}
    cols = {
        "col_0": torch.arange(N, dtype=torch.float32),
        "col_1": torch.linspace(0, 1, N),
    }
    ds = packer.execute("target", 7, wf, **cols)[0]
    # col_1's label is "target" -> it becomes the labels column, leaving a
    # single feature column
    check("pack features shape", tuple(ds.features.shape) == (N, 1))
    check("pack feature names from reflection", ds.feature_names == ["x0"])
    # col_1's label is "target" -> becomes the labels column
    check("pack target detected", ds.labels is not None)
    check("pack target_name", ds.target_name == "target")
    check("pack labels value",
          torch.equal(ds.labels.reshape(-1), cols["col_1"]))
    check("pack single feature remains",
          torch.equal(ds.features[:, 0], cols["col_0"]))
    check("pack meta provenance",
          ds.meta["pack"]["columns"] == ["x0"] and
          ds.meta["pack"]["target"] == "target" and
          ds.meta["pack"]["rows"] == N)

    # 4b. Pack without workflow metadata: raw port names, unlabelled ----------
    ds2 = packer.execute("", None, None,
                         col_0=torch.arange(4, dtype=torch.float32),
                         col_1=torch.ones(4))[0]
    check("pack fallback names", ds2.feature_names == ["col_0", "col_1"])
    check("pack no target", ds2.labels is None and ds2.target_name == "")

    # 4c. Pack unlabelled even with target_name set (no port matches) ---------
    ds3 = packer.execute("y", None, None,
                         col_0=torch.arange(4, dtype=torch.float32))[0]
    check("pack unmatched target_name -> unlabelled",
          ds3.labels is None and ds3.target_name == "")

    # 4d. Pack (n,1) columns and mixed width coercion --------------------------
    ds4 = packer.execute("", None, None,
                         col_0=torch.arange(4, dtype=torch.float32).unsqueeze(1),
                         col_1=torch.ones(4))[0]
    check("pack (n,1) accepted", tuple(ds4.features.shape) == (4, 2))

    # 5. Pack error contract ---------------------------------------------------
    def expect_error(name, fn, needle):
        try:
            fn()
            check(name, False, "no error raised")
        except ValueError as exc:
            check(name, needle in str(exc), str(exc)[:90])

    expect_error(
        "pack no columns", lambda: packer.execute("target", None, None),
        "no columns connected")
    expect_error(
        "pack row mismatch",
        lambda: packer.execute("", None, None,
                               col_0=torch.arange(4.0),
                               col_1=torch.arange(5.0)),
        "must agree on the row count")
    # duplicate names need reflection (fallback names are distinct by design)
    dup_wf = {"workflow": {"nodes": [
        {"id": 10, "inputs": [
            {"name": "col_0", "label": "dup"},
            {"name": "col_1", "label": "dup"},
        ]}]}}
    expect_error(
        "pack duplicate names",
        lambda: packer.execute("", 10, dup_wf,
                               col_0=torch.arange(4.0),
                               col_1=torch.arange(4.0)),
        "duplicate column name")
    expect_error(
        "pack multi-column tensor",
        lambda: packer.execute("", None, None,
                               col_0=torch.arange(8.0).reshape(4, 2)),
        "exactly ONE column")
    expect_error(
        "pack non-tensor",
        lambda: packer.execute("", None, None, col_0=[1, 2, 3]),
        "must be a TENSOR")
    # only-target: workflow says the single port IS the target
    only_target_wf = {"workflow": {"nodes": [
        {"id": 9, "inputs": [{"name": "col_0", "label": "target"}]}]}}
    expect_error(
        "pack only target",
        lambda: packer.execute("target", 9, only_target_wf,
                               col_0=torch.arange(4.0)),
        "at least one feature column")

    # 6. registry sanity: the module is aggregated by the host loader ----------
    check("data_ops aggregated", "CdlDatasetPack" in REG and "CdlDatasetUnpack" in REG)

    # ================= Unpack =================================================
    unpacker = REG["CdlDatasetUnpack"]()
    check("registered:Unpack", "CdlDatasetUnpack" in REG)
    check("Unpack CATEGORY", REG["CdlDatasetUnpack"].CATEGORY == "d2l/DataOps")
    check("Unpack ByPassTypeTuple",
          isinstance(REG["CdlDatasetUnpack"].RETURN_TYPES, tuple) and
          isinstance(REG["CdlDatasetUnpack"].RETURN_TYPES[0], AnyType))

    # 7. Unpack: schema + zero-copy views ---------------------------------------
    X = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    y = X[:, 0] * 2.0
    src = CdlDataset.from_tensors(X, y, feature_names=["a", "b", "c"],
                                  target_name="y", meta={"source": "unp"})
    out = unpacker.execute(src)
    check("unpack returns ui dict",
          isinstance(out, dict) and "ui" in out and "result" in out)
    check("unpack ui schema",
          [c["name"] for c in out["ui"]["columns"]] == ["a", "b", "c", "y"] and
          all(c["type"] == "TENSOR" for c in out["ui"]["columns"]))
    cols = out["result"]
    check("unpack port count", len(cols) == 4)
    check("unpack col a", torch.equal(cols[0], X[:, 0]))
    check("unpack col y", torch.equal(cols[3], y))
    check("unpack zero-copy feature",
          cols[0].untyped_storage().data_ptr() ==
          src.features.untyped_storage().data_ptr())
    check("unpack zero-copy target",
          cols[3].untyped_storage().data_ptr() ==
          src.labels.untyped_storage().data_ptr())

    # 8. Unpack -> Pack round-trip (reflection supplies the port names) ---------
    upk = unpacker.execute(src)["result"]
    rt_wf = {"workflow": {"nodes": [
        {"id": 21, "inputs": [
            {"name": "col_0", "label": "a"},
            {"name": "col_1", "label": "b"},
            {"name": "col_2", "label": "c"},
            {"name": "col_3", "label": "y"},
        ]}]}}
    back = packer.execute("y", 21, rt_wf,
                          col_0=upk[0], col_1=upk[1],
                          col_2=upk[2], col_3=upk[3])[0]
    check("roundtrip features", torch.equal(back.features, src.features))
    check("roundtrip labels", torch.equal(back.labels, src.labels))
    check("roundtrip schema",
          back.feature_names == src.feature_names and
          back.target_name == src.target_name)

    # 8b. unlabelled dataset unpack ---------------------------------------------
    out2 = unpacker.execute(CdlDataset.from_tensors(X[:, :2], None, ["a", "b"]))
    check("unpack unlabelled count", len(out2["result"]) == 2)
    check("unpack unlabelled schema",
          [c["name"] for c in out2["ui"]["columns"]] == ["a", "b"])

    def expect_error2(name, fn, needle):
        try:
            fn()
            check(name, False, "no error raised")
        except ValueError as exc:
            check(name, needle in str(exc), str(exc)[:90])

    expect_error2(
        "unpack empty dataset",
        lambda: unpacker.execute(CdlDataset(
            features=torch.zeros(0, 0), labels=None, feature_names=[])),
        "nothing to unpack")

    # ================= Frontend contracts (dataops.js) ========================
    js_path = REPO_ROOT / "app" / "dataops_assets" / "dataops.js"
    check("dataops.js exists", js_path.is_file())
    if js_path.is_file():
        js = js_path.read_text(encoding="utf-8")
        check("js references Pack class", PACK_CLASS_NAME in js)
        check("js references Unpack class", "CdlDatasetUnpack" in js)
        check("js dataset io type", '"DATASET"' in js)
        check("js configuringGraph guard", "configuringGraph" in js)
        check("js onExecuted channel", "onExecuted" in js)
        check("js boot race guard",
              "registerExtension too early" in js and "return false" in js)
        check("js no eval", "eval(" not in js and "new Function" not in js)

        # Syntax: node --check when available, bracket balance as fallback.
        import subprocess
        node_ok = False
        try:
            proc = subprocess.run(
                ["node", "--check", str(js_path)],
                capture_output=True, text=True, timeout=30,
            )
            node_ok = proc.returncode == 0
            if not node_ok:
                check("js node --check", False, proc.stderr[:160])
        except (OSError, subprocess.TimeoutExpired):
            pass
        if node_ok:
            check("js node --check", True)
        else:
            check("js bracket balance", _brackets_balanced(js))

    # loader wiring: constants + patch fn + apply_frontend_patches entry
    fp = (REPO_ROOT / "app" / "frontend_patch.py").read_text(encoding="utf-8")
    check("loader mark constant", "DATAOPS_LOADER_MARK" in fp)
    check("loader tag points at route", "/comfydl/dataops/dataops.js" in fp)
    check("loader fn defined", "def _patch_dataops_loader(" in fp)
    check("loader fn registered",
          "if _patch_dataops_loader(Path(web_root))" in fp)
    check("loader idempotent fn", fp.count("DATAOPS_LOADER_MARK in text") == 1)
    srv = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    check("static route mounted", "'/comfydl/dataops'" in srv)

    # loader idempotency: second application must be a no-op
    try:
        import shutil

        import app.frontend_patch as fpmod

        with tempfile.TemporaryDirectory(prefix="cdl_dops_fe_") as tmp:
            web_root = Path(tmp)
            (web_root / "index.html").write_text(
                "<html><body><div id=graph-canvas></div></body></html>",
                encoding="utf-8")
            first = fpmod._patch_dataops_loader(web_root)
            second = fpmod._patch_dataops_loader(web_root)
            html = (web_root / "index.html").read_text(encoding="utf-8")
            check("loader first apply", first is True)
            check("loader second apply noop", second is False)
            check("loader injected once", html.count("dataops-loader") == 1)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the run
        check("loader idempotency runnable", False, str(exc)[:120])

    _fail = [r for r in _RESULTS if not r[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== data_ops test: "
          f"{len(_RESULTS) - len(_fail)} PASS / {len(_fail)} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
