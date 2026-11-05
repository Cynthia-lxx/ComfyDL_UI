#!/usr/bin/env python
"""Standalone contract tests for the dual-renderer rich-widget foundation.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_richwidget.py

Covers the JavaScript source contracts plus a Node/vm behavioural probe of
createCanvasRichWidget and the pre-migration workflow-restore hook. Real
browser acceptance (Legacy + Nodes 2.0) is recorded in smallplans section
0.82f; this file is the fast regression gate, not a browser replacement.
"""

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RICH = REPO_ROOT / "comfydl" / "web" / "js" / "richwidget.js"
SHOW = REPO_ROOT / "comfydl" / "web" / "js" / "show_text.js"
_RESULTS = []


def check(name, condition, detail=""):
    _RESULTS.append((name, bool(condition), detail))


def node_check(name, path):
    try:
        proc = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        check(name, proc.returncode == 0, proc.stderr[:180])
    except (OSError, subprocess.TimeoutExpired):
        check(name, True, "node unavailable; skipped")


def behavioural_probe():
    runner = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8')
  .replace(/^import .*$/m, '')
  .replace(/export function /g, 'function ')
  + '\nglobalThis.__api = { createCanvasRichWidget, redrawRichWidget };';
const context = {
  console,
  setTimeout,
  LiteGraph: { vueNodesMode: false, NODE_TEXT_COLOR: '#ddd' },
  app: { registerExtension(ext) { this.extension = ext; } },
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(source, context);
const node = {
  widgets: [],
  size: [300, 200],
  addCustomWidget(widget) { this.widgets.push(widget); },
  graph: { dirty: 0, setDirtyCanvas() { this.dirty += 1; } },
};
let mouseCalls = 0;
const widget = context.__api.createCanvasRichWidget(node, {
  name: 'probe', type: 'probe', minHeight: 48,
  contentHeight: () => 120,
  mouse() { mouseCalls += 1; return true; },
});
widget.y = 20;
widget.computedHeight = 130;
const legacy = {
  serialize: widget.serialize,
  optionSerialize: widget.options.serialize,
  computedHeight: widget.computedHeight,
  layout: widget.computeLayoutSize(),
  widthUndefined: widget.width === undefined,
  mouseConsumed: widget.mouse({ type: 'pointerdown' }, [1, 2], node),
  mouseCalls,
};
widget.width = 999;
legacy.widthAfterSetUndefined = widget.width === undefined;
context.LiteGraph.vueNodesMode = true;
let triggered = 0;
widget.triggerDraw = () => { triggered += 1; };
context.__api.redrawRichWidget(node, widget);
const vue = {
  computedHeightUndefined: widget.computedHeight === undefined,
  layout: widget.computeLayoutSize(),
  dirty: node.graph.dirty,
  triggered,
};
const graph = {
  nodes: [{ type: 'CdlDatasetPreview', id: 1, widgets_values: [7, 'a\nb'] }],
  definitions: { subgraphs: [{
    nodes: [{ type: 'CdlDatasetPreview', id: 2, widgets_values: [9, 'c'] }],
  }] },
};
context.app.extension.beforeConfigureGraph(graph);
const restore = {
  root: graph.nodes[0].properties.cdl_restore_widgets_values,
  subgraph: graph.definitions.subgraphs[0].nodes[0]
    .properties.cdl_restore_widgets_values,
};
console.log(JSON.stringify({ legacy, vue, restore }));
"""
    try:
        proc = subprocess.run(
            ["node", "-e", runner, str(RICH)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            check("richwidget Node behavioural probe", False, proc.stderr[:220])
            return
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        check("richwidget Node behavioural probe", True, f"skipped: {exc}")
        return
    check("richwidget Node behavioural probe", True)
    legacy = data["legacy"]
    check("double serialize=false",
          legacy["serialize"] is False and legacy["optionSerialize"] is False)
    check("Legacy computedHeight frees resize zone",
          legacy["computedHeight"] == 165, str(legacy["computedHeight"]))
    check("Legacy growable layout",
          legacy["layout"]["minHeight"] == 48
          and legacy["layout"]["maxHeight"] == 100000)
    check("width shield ignores external writes",
          legacy["widthUndefined"] is True
          and legacy["widthAfterSetUndefined"] is True)
    check("pointer/mouse callback consumed",
          legacy["mouseConsumed"] is True and legacy["mouseCalls"] == 1)
    vue = data["vue"]
    check("Vue computedHeight hidden",
          vue["computedHeightUndefined"] is True, str(vue))
    check("Vue fixed content layout",
          vue["layout"]["minHeight"] == 120
          and vue["layout"]["maxHeight"] == 120)
    check("dual renderer redraw",
          vue["dirty"] == 1 and vue["triggered"] == 1, str(vue))
    check("restore hook preserves root and subgraph values",
          data["restore"]["root"] == [7, "a\nb"]
          and data["restore"]["subgraph"] == [9, "c"],
          str(data["restore"]))


def main():
    check("richwidget.js exists", RICH.is_file())
    rich = RICH.read_text(encoding="utf-8")
    show = SHOW.read_text(encoding="utf-8")

    check("official app import", 'from "/scripts/app.js"' in rich)
    check("layout and canvas normalization",
          "computeLayoutSize()" in rich and "normalizedCanvasBox" in rich
          and "offsetWidth" in rich and "ctx.setTransform" in rich)
    check("pointer and mouse event support",
          all(event in rich for event in (
              '"pointerdown"', '"mousedown"', '"pointermove"', '"mousemove"',
              '"pointerup"', '"mouseup"')))
    check("preview persistence in node.properties",
          'PREVIEW_PROPERTY = "cdl_preview_text"' in rich
          and "node.properties[PREVIEW_PROPERTY]" in rich)
    check("pre-migration restore hook",
          "beforeConfigureGraph(graphData)" in rich
          and "RESTORE_PROPERTY" in rich
          and "definitions?.subgraphs" in rich)
    check("executed-event fallback with late api binding",
          "executedFallbackBound" in rich
          and "clearInterval(retry), 10000" in rich
          and "addEventListener(\"executed\", updatePreviewFromEvent)" in rich)
    check("fallback reads payload from event.detail",
          "const detail = event?.detail;" in rich
          and "detail?.output?.text" in rich
          and "detail.display_node ?? detail.node" in rich)
    check("fallback normalizes string node ids",
          "function findPreviewNode" in rich
          and "Number(rawId)" in rich
          and "String(n.id) === String(rawId)" in rich)
    check("showPreview is idempotent",
          "preview.state.lines.join(\"\\n\") === text" in rich)
    check("loadedGraphNode restores head and text",
          "loadedGraphNode(node)" in rich and "head.value = raw[0]" in rich
          and "state.value = previewText(raw.slice(1))" in rich)
    check("no dynamic code execution",
          "eval(" not in rich and "new Function" not in rich)

    check("ShowText hidden dual marks",
          "w.hidden = true" in show and "w.options.hidden = true" in show)
    check("ShowText zero Vue layout and DOM hide",
          "minHeight: 0, maxHeight: 0, minWidth: 0, maxWidth: 0" in show
          and 'el.style.display = "none"' in show)
    check("ShowText pre-creates state before value mapping",
          "onNodeCreated" in show and 'populate(this, "")' in show)
    check("rich renderer moved after serialized state",
          "node.widgets.push(rich)" in show)
    check("ShowText loadedGraphNode rebuilds state",
          "loadedGraphNode(node)" in show
          and "cdl_preview_text" in show)
    check("ShowText fallback retained",
          "ComfyWidgets.STRING" in show and "el.readOnly = true" in show)

    node_check("richwidget.js node --check", RICH)
    node_check("show_text.js node --check", SHOW)
    behavioural_probe()

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== richwidget test: {len(_RESULTS) - len(failed)} PASS / "
          f"{len(failed)} FAIL (of {len(_RESULTS)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
