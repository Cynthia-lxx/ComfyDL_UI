#!/usr/bin/env python
"""Smoke tests for the universal Convert node (CdlConvert).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_convert.py

Covers the registration contract, the "match link" passthrough, the full
casting matrix (STRING / INT / FLOAT / BOOLEAN / TENSOR / IMAGE / MASK /
LATENT) and the readable-error behaviour for impossible casts.
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
    from comfydl.nodes.convert import TARGET_TYPES, _convert

    cls = REG.get("CdlConvert")
    check("node registered in utilities/conversion",
          cls is not None and cls.CATEGORY == "utilities/conversion")
    check("display name is Convert",
          cls is not None and REG is not None)
    check("contract: wildcard in, wildcard out, no OUTPUT_NODE",
          cls.RETURN_TYPES == ("*",)
          and cls.RETURN_NAMES == ("value",)
          and cls.INPUT_TYPES()["optional"]["value"] == ("*",)
          and cls.OUTPUT_NODE is False)
    check("VALIDATE_INPUTS accepts any wildcard combination",
          cls.VALIDATE_INPUTS({"value": ("*",)}) is True)
    check("combo default is 'match link' and it is part of the options",
          cls.INPUT_TYPES()["required"]["target_type"][1]["default"]
          == "match link"
          and "match link" in cls.INPUT_TYPES()["required"]["target_type"][0])
    check("frontend contract: 8 concrete targets beyond 'match link'",
          TARGET_TYPES[0] == "match link" and len(TARGET_TYPES) == 9)

    node = cls()

    # --- match link passthrough ------------------------------------------------
    passthroughs = [None, 3, "abc", torch.randn(2, 2), {"samples": torch.ones(1)}]
    for i, payload in enumerate(passthroughs):
        out = node.execute(value=payload)
        check(f"match link passthrough #{i} returns the payload untouched",
              len(out) == 1 and out[0] is payload)

    # --- STRING ---------------------------------------------------------------
    check("STRING: int via str()",
          node.execute(target_type="STRING", value=90)[0] == "90")
    check("STRING: None via To-Text semantics (empty string)",
          node.execute(target_type="STRING", value=None)[0] == "")
    check("STRING: single-element tensor renders its value",
          node.execute(target_type="STRING", value=torch.tensor(2.5))[0] == "2.5")
    check("STRING: multi-element tensor renders a compact summary",
          node.execute(target_type="STRING", value=torch.zeros(4, 3))[0]
          .startswith("Tensor(shape=(4, 3)"))
    check("STRING: list renders as a summary",
          node.execute(target_type="STRING", value=list(range(40)))[0]
          == "list(len=40)")

    # --- INT / FLOAT / BOOLEAN from scalars -------------------------------------
    check("INT: bool True -> 1",
          node.execute(target_type="INT", value=True)[0] == 1)
    check("INT: float truncates",
          node.execute(target_type="INT", value=2.7)[0] == 2)
    check("INT: numeric string parses",
          node.execute(target_type="INT", value="7")[0] == 7)
    check("FLOAT: int widens",
          node.execute(target_type="FLOAT", value=3)[0] == 3.0)
    check("FLOAT: bool widens",
          node.execute(target_type="FLOAT", value=False)[0] == 0.0)
    check("BOOLEAN: 0 -> False",
          node.execute(target_type="BOOLEAN", value=0)[0] is False)
    check("BOOLEAN: non-empty string -> True",
          node.execute(target_type="BOOLEAN", value="x")[0] is True)

    # --- INT / FLOAT / BOOLEAN from tensors and latents --------------------------
    check("INT: single-element tensor",
          node.execute(target_type="INT", value=torch.tensor([4.9]))[0] == 4)
    check("FLOAT: latent dict unwrap",
          node.execute(target_type="FLOAT",
                       value={"samples": torch.tensor(1.25)})[0] == 1.25)
    check("BOOLEAN: single-element tensor",
          node.execute(target_type="BOOLEAN", value=torch.tensor([0]))[0] is False)
    try:
        node.execute(target_type="INT", value=torch.zeros(2, 3))
        check("INT: multi-element tensor raises", False, "no exception")
    except ValueError as exc:
        check("INT: multi-element tensor raises a readable error",
              "element" in str(exc) and "INT" in str(exc), str(exc))

    # --- TENSOR ------------------------------------------------------------------
    t = torch.randn(3, 3)
    out_t = node.execute(target_type="TENSOR", value=t)[0]
    check("TENSOR: tensor passthrough is the same object", out_t is t)
    check("TENSOR: scalar wrapped",
          torch.is_tensor(node.execute(target_type="TENSOR", value=5)[0]))
    check("TENSOR: list materialized",
          torch.equal(node.execute(target_type="TENSOR", value=[1, 2, 3])[0],
                      torch.tensor([1, 2, 3])))
    check("TENSOR: latent dict unwrapped to samples",
          torch.is_tensor(node.execute(
              target_type="TENSOR", value={"samples": t})[0]))
    check("TENSOR: bool wrapped",
          torch.is_tensor(node.execute(target_type="TENSOR", value=True)[0]))
    check("TENSOR: numeric string parsed",
          torch.equal(node.execute(target_type="TENSOR", value="2.5")[0],
                      torch.tensor(2.5)))
    try:
        node.execute(target_type="TENSOR", value="abc")
        check("TENSOR: non-numeric string raises", False, "no exception")
    except ValueError as exc:
        check("TENSOR: non-numeric string raises a readable error",
              "numeric" in str(exc), str(exc))

    # --- IMAGE / MASK / LATENT dict convention ------------------------------------
    img_t = torch.randn(1, 8, 8, 3)
    for target in ("IMAGE", "MASK", "LATENT"):
        wrapped = node.execute(target_type=target, value=img_t)[0]
        check(f"{target}: tensor wrapped into the dict convention",
              isinstance(wrapped, dict) and wrapped["samples"] is img_t)
        passthrough = node.execute(
            target_type=target, value={"samples": img_t})[0]
        check(f"{target}: matching dict passthrough is the same object",
              isinstance(passthrough, dict) and passthrough["samples"] is img_t)
    check("IMAGE: dict re-wrapped keeps the samples tensor",
          node.execute(target_type="IMAGE",
                       value={"samples": img_t})[0]["samples"] is img_t)
    check("LATENT: extra keys passthrough untouched",
          node.execute(target_type="LATENT",
                       value={"samples": img_t, "extra": 1})[0].get("extra") == 1)

    # --- impossible casts raise readable errors -----------------------------------
    bad = [
        ("None to FLOAT", None, "FLOAT", "None"),
        ("dict without samples to INT", {"a": 1}, "INT", "samples"),
        ("list to FLOAT (sequences only cast to TENSOR)", [1, 2], "FLOAT",
         "sequences"),
        ("str to INT", "not-a-number", "INT", "not-a-number"),
        ("unsupported payload to TENSOR", object(), "TENSOR", "unsupported"),
    ]
    for label, payload, target, needle in bad:
        try:
            node.execute(target_type=target, value=payload)
            check(f"error: {label} raises", False, "no exception")
        except ValueError as exc:
            check(f"error: {label} raises a readable error",
                  needle in str(exc), str(exc)[:80])

    # --- report -------------------------------------------------------------------
    failed = [(n, d) for n, ok, d in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(("PASS" if ok else "FAIL") + f"  {name}"
              + (f"  [{detail}]" if detail and not ok else ""))
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed, "
          f"{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
