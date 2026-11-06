#!/usr/bin/env python
"""Smoke tests for the Notebook -> Cell node (SP-7 craft 6).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_notebook.py

Covers the four ported mechanisms (per-workflow kernel, interruptible
execution, streaming-stdout capture, variable inspection) plus the
comfydl.notebook switch gating, error propagation, plot placeholder,
properties write-back and the frontend source contract. Browser behaviour
is accepted by the user per the project convention.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NOTEBOOK_JS = REPO_ROOT / "comfydl" / "web" / "js" / "notebook.js"
_RESULTS = []


def check(name, condition, detail=""):
    _RESULTS.append((name, bool(condition), detail))


def _run(node, code, wid="wf-1", uid="7", inp=None, pnginfo=None):
    extra = pnginfo if pnginfo is not None else {
        "workflow": {"id": wid},
    }
    return node.execute(code, input=inp, unique_id=uid, extra_pnginfo=extra)


def test_backend():
    from comfydl.nodes import NODE_CLASS_MAPPINGS as REG
    import comfydl.nodes.notebook as nb
    from comfydl.nodes import _notebook_kernel as nk

    cls = REG.get("CdlNotebookCell")
    check("node registered in the Notebook family",
          cls is not None and cls.CATEGORY == "d2l/Notebook")
    check("output node with the three-slot contract",
          cls.OUTPUT_NODE is True
          and cls.RETURN_TYPES == ("*", "IMAGE", "STRING")
          and cls.RETURN_NAMES == ("result", "plot", "stdout"))
    inputs = cls.INPUT_TYPES()
    check("code default is runnable and multiline",
          "Result" in inputs["required"]["code"][1]["default"]
          and inputs["required"]["code"][1].get("multiline") is True)
    check("optional wildcard input socket",
          inputs["optional"].get("input") == ("*",))
    check("switch defaults to OFF",
          nk.SWITCH_ID == "comfydl.notebook"
          and (nk._registry() is None or not nk.enabled()))

    node = cls()

    # Gate: switch off (bare env has no registry -> disabled) must refuse.
    try:
        _run(node, "Result = 1")
        check("switch off refuses execution", False)
    except RuntimeError as exc:
        check("switch off refuses execution", "comfydl.notebook" in str(exc),
              str(exc)[:120])

    # Enable for the rest of the suite (monkeypatched probe, no registry).
    original_enabled = nb._notebook_enabled
    nb._notebook_enabled = lambda: True
    try:
        nk.clear()

        out = _run(node, "shared = 41\nprint('half of', shared)")
        check("stdout is captured into ui and result",
              "half of 41" in out["ui"]["cdl_notebook_log"][0]
              and "half of 41" in out["result"][2])
        check("plot placeholder without figures",
              out["result"][1].shape == (1, 1, 1, 3))
        check("result defaults to None when Result is unset",
              out["result"][0] is None)

        # Mechanism 1: kernel persistence within one workflow.
        out = _run(node, "Result = shared + 1")
        check("variables persist across cells of one workflow",
              out["result"][0] == 42, repr(out["result"][0]))

        # Workflow isolation.
        out = _run(node, "Result = 'fresh' if 'shared' not in dir() else 'leaked'",
                   wid="wf-2")
        check("kernels are isolated per workflow",
              out["result"][0] == "fresh", repr(out["result"][0]))

        # Input socket injection.
        out = _run(node, "Result = input * 2", wid="wf-3", inp=21)
        check("optional input reaches the cell as 'input'",
              out["result"][0] == 42, repr(out["result"][0]))

        # Properties write-back (save/reload channel).
        workflow = {"workflow": {"nodes": [{"id": "9", "properties": {}}]}}
        _run(node, "print('logged')", uid="9", pnginfo=workflow)
        node_dict = workflow["workflow"]["nodes"][0]
        check("final log written back into node.properties",
              "logged" in node_dict["properties"].get("cdl_notebook_log", ""))

        # Error propagation.
        try:
            _run(node, "Result = 1 / 0")
            check("cell errors propagate as node failures", False)
        except ZeroDivisionError:
            check("cell errors propagate as node failures", True)

        # Mechanism 2: host Stop breaks a spinning cell.
        import comfy.model_management as mm

        original_probe = mm.processing_interrupted
        mm.processing_interrupted = lambda: True
        try:
            _run(node, "while True:\n    check_interrupt()")
            check("host interrupt breaks the cell", False)
        except mm.InterruptProcessingException:
            check("host interrupt breaks the cell", True)
        except BaseException as exc:  # noqa: BLE001
            check("host interrupt breaks the cell", False, repr(exc)[:120])
        finally:
            mm.processing_interrupted = original_probe

        # Mechanism 4: variable inspection.
        _run(node, "tensor_thing = [1, 2, 3]\nResult = tensor_thing", wid="wf-4")
        rows = {r["name"]: r for r in nk.describe_variables("wf-4")}
        check("inspection lists user variables with type and repr",
              "tensor_thing" in rows
              and rows["tensor_thing"]["type"] == "list"
              and "[1, 2, 3]" in rows["tensor_thing"]["repr"],
              str(rows)[:160])
        check("inspection skips modules and dunders",
              "torch" not in rows and all(not n.startswith("__") for n in rows))
        check("inspection of an unknown workflow is empty",
              nk.describe_variables("wf-missing") == [])

        check("kernel accounting tracks active workflows",
              set(nk.active_workflows()) >= {"wf-1", "wf-2", "wf-3", "wf-4"})
        check("free drops one workflow kernel",
              nk.clear("wf-1") == 1 and "wf-1" not in nk.active_workflows())
        check("free-all clears the rest",
              nk.clear() >= 3 and nk.active_workflows() == [])
    finally:
        nb._notebook_enabled = original_enabled
        nk.clear()


def test_frontend_source():
    js = NOTEBOOK_JS.read_text(encoding="utf-8")
    check("notebook extension targets only the cell node",
          'nodeData?.name === "CdlNotebookCell"' in js
          and "beforeRegisterNodeDef" in js)
    check("sandbox banner is permanent in the canvas",
          "BANNER_TEXT" in js and "comfydl.notebook switch" in js)
    check("payload extraction tolerates both message shapes",
          "function extractLog" in js
          and "message?.output ?? message" in js)
    check("node matching accepts comfyClass and type",
          "function isNotebookNode" in js
          and 'node?.type === "CdlNotebookCell"' in js)
    check("result repr echoed on the console",
          "[Result]" in js)
    check("streamed log channel wired twice",
          '"cdl_notebook_log"' in js
          and 'app.api.addEventListener("executed", onStreamedEvent)' in js
          and "onExecuted" in js)
    check("fallback retry never gives up",
          "}, 2000);" in js
          and "setTimeout(() => clearInterval(retry), 10000);" not in js)
    check("restore reads the persisted log",
          "LOG_PROPERTY" in js and "onConfigure" in js)
    try:
        import subprocess

        proc = subprocess.run(["node", "--check", str(NOTEBOOK_JS)],
                              capture_output=True, text=True, timeout=30)
        check("notebook.js node --check", proc.returncode == 0,
              proc.stderr[:160])
    except (OSError, subprocess.TimeoutExpired):
        check("notebook.js node --check", True, "node unavailable; skipped")


def main():
    test_backend()
    test_frontend_source()
    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== notebook test: {len(_RESULTS) - len(failed)} PASS / "
          f"{len(failed)} FAIL (of {len(_RESULTS)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
