#!/usr/bin/env python
"""Smoke tests for the To Text conversion node (SP-7 follow-up).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_to_text.py

Covers the wildcard->STRING casting rules (None -> "", tensor summaries,
list summaries, str passthrough) and the frontend-free backend contract.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_RESULTS = []


def check(name, condition, detail=""):
    _RESULTS.append((name, bool(condition), detail))


def main():
    import torch

    from comfydl.nodes import NODE_CLASS_MAPPINGS as REG
    from comfydl.nodes.text_convert import _to_text

    cls = REG.get("CdlToText")
    check("node registered in utilities/conversion",
          cls is not None and cls.CATEGORY == "utilities/conversion")
    check("contract: wildcard in, STRING out, no OUTPUT_NODE",
          cls.RETURN_TYPES == ("STRING",)
          and cls.INPUT_TYPES()["required"]["value"] == ("*", {"forceInput": True})
          and cls.OUTPUT_NODE is False)
    check("VALIDATE_INPUTS accepts the wildcard",
          cls.VALIDATE_INPUTS({"*": ("*",)}) is True)

    check("int renders as str",
          _to_text(90) == "90")
    check("float renders as str",
          _to_text(0.5) == "0.5")
    check("str passes through",
          _to_text("hello") == "hello")
    check("None renders as empty string (not 'None')",
          _to_text(None) == "")
    check("short list renders like Python",
          _to_text([1, 2, 3]) == "[1, 2, 3]")
    check("long list renders as a summary",
          _to_text(list(range(40))) == "list(len=40)",
          _to_text(list(range(40))))
    t = torch.randn(4, 3)
    rendered = _to_text(t)
    check("tensor renders as a compact summary",
          rendered.startswith("Tensor(shape=(4, 3), dtype=torch.float32)")
          and "values=" in rendered, rendered[:80])
    check("single-element tensor renders as its value",
          _to_text(torch.tensor(2.5)) == "2.5")
    big = torch.randn(64)
    check("large tensor renders as shape-only summary",
          _to_text(big) == f"Tensor(shape=(64,), dtype=torch.float32)",
          _to_text(big)[:80])

    node = cls()
    out = node.execute(90, unique_id="1", extra_pnginfo=None)
    check("execute returns the STRING tuple",
          out == ("90",))

    failed = [row for row in _RESULTS if not row[1]]
    for name, ok, detail in _RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n=== to_text test: {len(_RESULTS) - len(failed)} PASS / "
          f"{len(failed)} FAIL (of {len(_RESULTS)}) ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
