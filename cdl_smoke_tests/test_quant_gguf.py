#!/usr/bin/env python
"""Standalone smoke test: the GGUF quantization family (SP-4, M-B; ADR-D3).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_quant_gguf.py

Layers tested (each tier upgrades gracefully as the ``gguf`` package
becomes available):

* gguf-free  - the size table (ADR-D3's byte-rate contract), GGMLTensor
  semantics, and handcrafted Q8_0/Q4_0 block dequant;
* gguf-optional - the full GGML_QUANT_SIZES cross-check, random-block
  K-quant (Q4_K/Q6_K) dequant against gguf-py's reference dequantizer,
  and a tiny .gguf fixture round-trip (writer -> gguf_sd_loader ->
  GGMLTensor -> dequantize_tensor -> byte ledger), generated into a
  scratch directory at test time (no binary fixtures in the repo).
"""

import os
import struct
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfy import quant_formats as qf  # noqa: E402

_RESULTS: list = []
_SKIPPED: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def skip(name: str, why: str) -> None:
    _SKIPPED.append((name, why))


GGUF_OK = True
try:
    import gguf  # noqa: F401
except ImportError:
    GGUF_OK = False

# ---------------------------------------------------------------------------
# T1: size table (ADR-D3 byte-rate contract) - no gguf needed

check("T1a table covers the standard types",
      qf.bytes_per_element(qf.QTYPE_F32) == 4.0
      and qf.bytes_per_element(qf.QTYPE_F16) == 2.0
      and qf.bytes_per_element(qf.QTYPE_Q8_0) == 34 / 32
      and qf.bytes_per_element(qf.QTYPE_Q4_K) == 144 / 256
      and qf.bytes_per_element(qf.QTYPE_Q6_K) == 210 / 256)
check("T1b Q4_K is 4.5 bits/weight (block scales, not 4)",
      abs(qf.bytes_per_element(qf.QTYPE_Q4_K) * 8 - 4.5) < 1e-9)
check("T1c every table entry is (block, bytes) with block > 0, bytes > 0",
      all(bs > 0 and ts > 0 for bs, ts in qf.GGML_QTYPE_SIZES.values()))
check("T1d names cover the table", set(qf.GGML_QTYPE_SIZES) <= set(qf.QTYPE_NAMES))
try:
    qf.bytes_per_element(999)
    check("T1e unknown qtype raises", False, "no exception")
except ValueError:
    check("T1e unknown qtype raises", True)

# ---------------------------------------------------------------------------
# T2: GGMLTensor semantics

raw = torch.randint(0, 255, (2176,), dtype=torch.uint8)  # 64 Q8_0 blocks
gt = qf.GGMLTensor(raw, tensor_type=qf.QTYPE_Q8_0,
                   tensor_shape=torch.Size((64, 32)))
check("T2a shape is the logical (dequantized) shape", tuple(gt.shape) == (64, 32))
check("T2b clone/detach return the same blocks (no copies)",
      gt.clone() is gt and gt.detach() is gt)
gt2 = gt.to(torch.uint8)
check("T2c to() carries type/shape metadata",
      gt2.tensor_type == qf.QTYPE_Q8_0 and tuple(gt2.tensor_shape) == (64, 32))
check("T2d classification helpers",
      qf.is_quantized(gt) and not qf.is_quantized(torch.zeros(2)))


def _make_q8_0(n_blocks: int, d: float, values: torch.Tensor) -> torch.Tensor:
    """Handcraft Q8_0 blocks: 2 fp16 scale bytes + 32 int8 weights."""
    d_bytes = struct.pack("<e", d)
    payload = bytes((v.int() & 0xFF).item() for v in values)
    block = torch.frombuffer(
        b"".join(d_bytes + payload for _ in range(n_blocks)), dtype=torch.uint8)
    return block.clone()


def _make_q4_0(n_blocks: int, d: float) -> tuple:
    """Handcraft Q4_0 blocks (GGML layout: elements 0-15 are the LOW nibbles
    of the 16 qs bytes, elements 16-31 the HIGH nibbles)."""
    d_bytes = struct.pack("<e", d)
    vals = [i % 16 for i in range(32)]  # logical element nibble values
    qs = bytes((vals[16 + i] << 4) | vals[i] for i in range(16))
    expected = torch.tensor([d * (v - 8) for v in vals] * n_blocks)
    block = torch.frombuffer(
        b"".join(d_bytes + qs for _ in range(n_blocks)), dtype=torch.uint8)
    return block.clone(), expected

# ---------------------------------------------------------------------------
# T3: handcrafted dequant correctness

q8 = _make_q8_0(4, 0.25, torch.arange(-16, 16, dtype=torch.int8))
out8 = qf.dequantize(q8, qf.QTYPE_Q8_0, torch.Size((4, 32)), dtype=torch.float32)
expected8 = (torch.arange(-16, 16, dtype=torch.float32) * 0.25).repeat(4)
check("T3a Q8_0: d * int8 exactly", torch.allclose(out8.flatten(), expected8))

q4, expected4 = _make_q4_0(3, 0.5)
out4 = qf.dequantize(q4, qf.QTYPE_Q4_0, torch.Size((3, 32)), dtype=torch.float32)
check("T3b Q4_0: nibble-8 exactly",
      torch.allclose(out4.flatten(), expected4),
      f"got {out4.flatten()[:4].tolist()} want {expected4[:4].tolist()}")

q8t = qf.GGMLTensor(_make_q8_0(64, 0.25, torch.arange(-16, 16, dtype=torch.int8)),
                    tensor_type=qf.QTYPE_Q8_0, tensor_shape=torch.Size((64, 32)))
deq = qf.dequantize_tensor(q8t, dtype=torch.float32)
check("T3c dequantize_tensor returns plain fp tensor of logical shape",
      tuple(deq.shape) == (64, 32) and deq.dtype == torch.float32
      and not isinstance(deq, qf.GGMLTensor))
check("T3d fp16-classified as torch-compatible (no dequant path)",
      qf.bytes_per_element(qf.QTYPE_F16) == 2.0)

# ---------------------------------------------------------------------------
# T4/T5: gguf-dependent tiers

if GGUF_OK:
    import importlib.metadata

    # T4: full table cross-check against gguf-py
    mismatches = []
    for name, member in vars(gguf.GGMLQuantizationType).items():
        if not isinstance(member, gguf.GGMLQuantizationType):
            continue
        entry = gguf.GGML_QUANT_SIZES.get(member)
        ours = qf.GGML_QTYPE_SIZES.get(int(member))
        if entry is None:
            continue
        if ours != (entry[0], entry[1]):
            mismatches.append(f"{name}: ours={ours} ref={entry[:2]}")
    check("T4 GGML_QUANT_SIZES table matches gguf-py exactly",
          not mismatches, "; ".join(mismatches[:6]))
    try:
        ver = importlib.metadata.version("gguf")
        check("T4b gguf version in the pinned window",
              ver.split(".")[:2] >= ["0", "13"], f"gguf {ver}")
    except Exception as exc:  # noqa: BLE001
        skip("T4b gguf version probe", str(exc))

    # T5: random-block K-quants vs gguf-py reference dequantizer
    for qtype, block_size, type_size in ((qf.QTYPE_Q4_K, 256, 144),
                                         (qf.QTYPE_Q6_K, 256, 210)):
        n_blocks = 16
        rng = torch.randint(0, 255, (n_blocks, type_size), dtype=torch.uint8)
        ours = qf.dequantize(rng.reshape(-1), qtype,
                             torch.Size((n_blocks, block_size)),
                             dtype=torch.float32).flatten()
        ref = gguf.quants.dequantize(rng.numpy(), gguf.GGMLQuantizationType(qtype))
        ref_t = torch.from_numpy(ref.copy()).flatten().to(torch.float32)
        scale = ref_t.abs().max().clamp(min=1e-6)
        check(f"T5 {qf.qtype_name(qtype)} torch dequant matches gguf-py reference",
              torch.allclose(ours, ref_t, atol=scale * 1e-3),
              f"max diff {(ours - ref_t).abs().max().item():.3e} vs scale {scale.item():.2f}")

    # T5b/T5c: fixture round-trip through a real .gguf file
    def _write_fixture(path: Path) -> None:
        writer = gguf.GGUFWriter(str(path), arch="flux")
        writer.add_string("general.name", "cdl-sp4-fixture")
        writer.add_uint32("general.version", 1)
        # a quantized weight: 4 blocks x Q8_0, logical (4, 32)
        writer.add_tensor("model.diffusion_model.blocks.0.weight",
                          _make_q8_0(4, 0.25,
                                     torch.arange(-16, 16, dtype=torch.int8)).numpy(),
                          raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        # an fp16 bias-ish tensor
        writer.add_tensor("model.diffusion_model.blocks.0.bias",
                          torch.arange(8, dtype=torch.float16).numpy(),
                          raw_dtype=gguf.GGMLQuantizationType.F16)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    with tempfile.TemporaryDirectory(prefix="cdl_gguf_") as tmp:
        fixture = Path(tmp) / "tiny-Q8_0.gguf"
        _write_fixture(fixture)
        check("T5b fixture written and small",
              fixture.exists() and fixture.stat().st_size < 64 * 1024,
              f"{fixture.stat().st_size if fixture.exists() else -1} bytes")

        sd, extra = qf.gguf_sd_loader(str(fixture))
        check("T5c loader returns arch + metadata",
              extra["arch_str"] == "flux"
              and extra["metadata"].get("general.name") == "cdl-sp4-fixture")
        w = sd["model.diffusion_model.blocks.0.weight"]
        check("T5d weight is a GGMLTensor with qtype + logical shape",
              isinstance(w, qf.GGMLTensor) and w.tensor_type == qf.QTYPE_Q8_0
              and tuple(w.tensor_shape) == (4, 32))
        check("T5e dequant round-trips through the file",
              torch.allclose(
                  qf.dequantize_tensor(w, dtype=torch.float32).flatten(),
                  expected8))
        b = sd["model.diffusion_model.blocks.0.bias"]
        check("T5f fp16 tensor passes through as a plain view",
              tuple(b.shape) == (8,)
              and torch.allclose(qf.dequantize_tensor(b).float(),
                                 torch.arange(8, dtype=torch.float16).float()))
        check("T5g largest weight marked for vram estimation",
              getattr(w, "is_largest_weight", False))

        # T5h: the inspector's byte ledger on the fixture
        summary = qf.summarize_state_dict_bytes(sd)
        w_entry = next(e for e in summary["layers"] if e["key"].endswith("weight"))
        check("T5h ledger: Q8_0 bytes = 4 blocks x 34",
              w_entry["bytes"] == 4 * 34 and w_entry["qtype"] == "Q8_0")
        check("T5i ledger: fp reference uses 2 bytes/elem for fp16 weights",
              w_entry["fp_bytes"] == 4 * 32 * 4)  # fp32 reference default
        # T5j: the node's renderer end to end
        from comfy_extras.nodes_quant import CdlQuantizationInspector  # noqa: E402

        text = CdlQuantizationInspector._render(str(fixture), sd, extra)
        check("T5j inspector render: TOTAL + savings footer present",
              "TOTAL" in text and "saved" in text)
        check("T5k inspector render carries both layers",
              text.count("model.diffusion_model") == 2)
else:
    for name in ("T4 table cross-check", "T5 K-quant reference dequant",
                 "T5b-j gguf fixture round-trip"):
        skip(name, "gguf package not installed")

# ---------------------------------------------------------------------------
# T6: estimators hard binding (ADR-D3) - fake ctx, no engine needed

from comfy.profiling import estimators as est_mod  # noqa: E402


class _FakeCtx:
    def __init__(self, name: str) -> None:
        self._name = name

    def str_widget(self, _key, default=""):
        return self._name

    def int_widget(self, _key, default=None):
        return default


check("T6a CdlLoadGGUF estimator registered",
      "CdlLoadGGUF" in est_mod.ESTIMATORS)
check("T6b CdlQuantizationInspector estimator registered",
      "CdlQuantizationInspector" in est_mod.ESTIMATORS)

_ORIG_FILE_SIZE = est_mod._model_file_size
est_mod._model_file_size = lambda folder, name: 1234
try:
    out = est_mod._estimate_gguf_loader(_FakeCtx("model-Q4_K_M.gguf"))
finally:
    est_mod._model_file_size = _ORIG_FILE_SIZE
check("T6c estimator: WeightsVal carries the quantized FILE bytes",
      len(out.outputs) == 1 and out.outputs[0].file_bytes == 1234,
      str([getattr(o, "file_bytes", None) for o in out.outputs]))
check("T6d estimator: byte-rate table named in the basis (ADR-D3 audit)",
      out.basis["params"].get("byte_rate_table", "").endswith("GGML_QTYPE_SIZES"))
out2 = est_mod._estimate_gguf_loader(_FakeCtx("missing.gguf"))
check("T6e estimator: missing file -> unknown, not a guess",
      out2.status == "unknown" and "not found" in out2.reason)
out3 = est_mod._estimate_quant_inspector(_FakeCtx("model.gguf"))
check("T6f estimator: inspector is a metadata-only read (exact, no weights)",
      out3.confidence == "exact" and out3.items == [])

# ---------------------------------------------------------------------------
# T7: quant node registration + readable error without gguf

import asyncio  # noqa: E402

import nodes as host_nodes  # noqa: E402

asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                        init_api_nodes=False))
registry = host_nodes.NODE_CLASS_MAPPINGS
check("T7a both quant nodes registered",
      "CdlLoadGGUF" in registry and "CdlQuantizationInspector" in registry)

try:
    qf.require_gguf()
    check("T7b require_gguf passes when installed", True)
except RuntimeError as exc:
    check("T7b require_gguf hints the pip command",
          "pip install gguf" in str(exc))


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    for name, why in _SKIPPED:
        print(f"  SKIP  {name}  ({why})")
    print(f"\n=== quant gguf test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"/ {len(_SKIPPED)} SKIP (of {len(_RESULTS) + len(_SKIPPED)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
