#!/usr/bin/env python
"""Standalone smoke test: execute the shipped example workflows as authored.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_template_workflows.py

Loads each ``comfydl/example_workflows/*.json`` (UI format), resolves the
``widgets_values`` arrays against every node's ``INPUT_TYPES`` widget order
(this is exactly where hand-authored templates go wrong), then executes the
graph in topological order and verifies the out-of-the-box behaviour the
Templates panel promises:

  - linear_regression_from_scratch: the textbook closure converges (loss falls
    >10x) and the stateless Linear Regression + Squared Loss verification
    reproduces the training tail; the Plot node renders an IMAGE.
  - tabular_regression_production: the one-box trainer with the template's
    widget values reaches the noise floor (mae < 0.15, rmse < 0.2 on the
    5000-row formula dataset), predictions are exported to CSV and the table
    preview is produced.

Frontend-only node types (Note / MarkdownNote / PrimitiveNode / Reroute) are
skipped, mirroring the dead-template scan.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_smoke_test as harness  # noqa: E402

_RESULTS: list = []

_FRONTEND_ONLY = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def _widget_order(node_cls):
    """Widget names in declaration order (required then optional)."""
    it = node_cls.INPUT_TYPES()
    order = []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            if isinstance(spec, tuple):
                type_name = spec[0]
                if isinstance(type_name, list):  # combo: (["a", "b"], {...})
                    type_name = "COMBO"
            elif isinstance(spec, list):
                type_name = "COMBO"
            else:
                type_name = spec
            if type_name in _WIDGET_TYPES:
                order.append(name)
                # The frontend appends a `control_after_generate` widget after
                # INT widgets named exactly `seed` / `noise_seed` — its value
                # occupies a slot in the serialized widgets_values array.
                if type_name == "INT" and name in ("seed", "noise_seed"):
                    order.append("control_after_generate")
    return order


def _run_workflow(path: Path):
    """Execute a UI-format workflow dict; return {node_id: output tuple}."""
    import nodes as host_nodes

    registry = host_nodes.NODE_CLASS_MAPPINGS
    wf = json.loads(path.read_text(encoding="utf-8-sig"))
    links = {l[0]: (l[1], l[2], l[3], l[4]) for l in wf["links"]}
    node_out = {}
    for node in sorted(wf["nodes"], key=lambda n: n.get("order", 0)):
        if node["type"] in _FRONTEND_ONLY:
            continue
        cls = registry[node["type"]]
        args = {}
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            src, sslot, dst, _dslot = links[link_id]
            assert dst == node["id"], f"link {link_id} target mismatch"
            args[inp["name"]] = node_out[src][sslot]
        wvals = node.get("widgets_values") or []
        # Widget-converted-to-input names (e.g. SaveText.text) drop out of
        # widgets_values once they carry a link.
        linked = {i["name"] for i in node.get("inputs", []) if i.get("link") is not None}
        order_names = [n for n in _widget_order(cls) if n not in linked]
        assert len(wvals) == len(order_names), (
            f"node {node['id']} ({node['type']}): {len(wvals)} widget values "
            f"but {len(order_names)} widgets declared {order_names}"
        )
        args.update(
            {k: v for k, v in zip(order_names, wvals) if k != "control_after_generate"}
        )
        instance = cls()
        # Mirror the host: execution.py runs the whole prompt inside
        # torch.inference_mode(), so every node here must survive that too.
        with torch.inference_mode():
            out = getattr(instance, cls.FUNCTION)(**args)
        node_out[node["id"]] = out if isinstance(out, tuple) else (out,)
    return node_out


def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="cdl_tpl_"))
    harness._bootstrap(sandbox)
    harness._load_registry(False)

    workflows_dir = REPO_ROOT / "comfydl" / "example_workflows"

    for name in (
        "linear_regression_from_scratch",
        "tabular_regression_production",
        "regression_model_reuse",
    ):
        wf = json.loads((workflows_dir / f"{name}.json").read_text(encoding="utf-8-sig"))
        types = {n["type"] for n in wf["nodes"]}
        has_wired_preview = any(
            n["type"] == "PreviewImage" and n.get("inputs") and n["inputs"][0].get("link") is not None
            for n in wf["nodes"]
        )
        # SaveText is the OUTPUT_NODE of the inference template; the other two
        # templates end in a wired PreviewImage.
        has_output_node = "SaveText" in types or has_wired_preview
        check(f"{name}: output node present (PreviewImage wired / SaveText)",
              has_output_node,
              "a template without an OUTPUT_NODE fails to queue ('workflow has no outputs')")

    # --- Template A: linear_regression_from_scratch -------------------------- #
    out = _run_workflow(workflows_dir / "linear_regression_from_scratch.json")
    w, b, curve, y_hat = out[3]
    check("A: loss curve fell >10x",
          float(curve[-1]) < float(curve[0]) / 10,
          f"{float(curve[0]):.4f} -> {float(curve[-1]):.4f}")
    verified = float(out[5][0].mean())
    check("A: verified loss ~= training tail",
          abs(verified - float(curve[-1])) < max(1e-3, float(curve[-1])),
          f"verified {verified:.5f} vs tail {float(curve[-1]):.5f}")
    check("A: plot produced IMAGE", out[6][0].dim() == 4)

    # --- Template B: tabular_regression_production ---------------------------- #
    out = _run_workflow(workflows_dir / "tabular_regression_production.json")
    model, mae, rmse, preds, hist = out[5]
    check("B: loss curve fell >10x",
          float(hist[-1]) < float(hist[0]) / 10,
          f"{float(hist[0]):.4f} -> {float(hist[-1]):.4f}")
    check("B: mae at noise floor", float(mae) < 0.15, f"mae={float(mae):.4f}")
    check("B: rmse at noise floor", float(rmse) < 0.2, f"rmse={float(rmse):.4f}")
    check("B: predictions [5000, 1]", tuple(preds.shape) == (5000, 1),
          str(tuple(preds.shape)))
    check("B: preview text written", isinstance(out[3][0], str) and "x0" in out[3][0])
    csv_path = Path(out[8][0])
    check("B: CSV exported",
          csv_path.exists() and csv_path.stat().st_size > 1000, str(csv_path))
    check("B: plot produced IMAGE", out[6][0].dim() == 4)
    check("B: model saved for reuse", Path("output/regression_model.pt").exists(),
          "Model Save must persist the trained weights for the Load & Predict template")

    # --- Template C: regression_model_reuse (pure inference) ------------------- #
    # Loads the file B just saved — this is exactly the intended user flow:
    # run Production once, then Load & Predict in the same or a later session.
    out = _run_workflow(workflows_dir / "regression_model_reuse.json")
    preds, truth = out[4][0], out[3][1]
    err = float((preds - truth).abs().mean())
    check("C: reloaded model tracks fresh data", err < 0.15, f"mae={err:.4f}")
    csv_path = Path(out[6][0])
    check("C: predictions CSV exported",
          csv_path.exists() and csv_path.stat().st_size > 1000, str(csv_path))
    check("C: preview text written", isinstance(out[7][0], str) and "x0" in out[7][0])

    return _report()


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== template workflow test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
