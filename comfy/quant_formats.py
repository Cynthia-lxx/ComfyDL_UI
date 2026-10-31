"""Quantized weight formats for the MODEL protocol layer (SP-4, M-B; ADR-D3).

The single home of "quantized layout" knowledge:

* **GGML/GGUF** (first implementation): a ``GGMLTensor`` torch.Tensor subclass
  that carries the raw quantized blocks (mmapped, never expanded) plus the
  original dtype/shape, a pure-torch block dequantizer, and ``GGMLOps`` -
  a ``comfy.ops.manual_cast`` derivative whose Linear/Conv/Embedding/Norm
  layers dequantize their weights on the fly inside ``cast_bias_weight``.
  Memory stays proportional to the *file* (a Q4_K_M 7B UNet costs ~4.5 GB,
  not ~13 GB); each layer is decompressed just before its matmul.

Attribution: the tensor subclass, the lazy-dequant ops and the pure-torch
dequant kernels are a port of ComfyUI-GGUF (city96, Apache-2.0), adapted to
ComfyDL (gguf import confined to the file reader; types referenced by their
stable GGML integer codes so this module imports without the ``gguf`` package).

Layering rule (ADR-D3): the **estimators** consume only the size table below
(``GGML_QTYPE_SIZES`` / :func:`bytes_per_element`) - never the ``gguf``
package and never torch ops - so the profiling layer stays import-light.

Public API
----------
- :func:`bytes_per_element` / ``GGML_QTYPE_SIZES``  - the size table (ADR-D3)
- :class:`GGMLTensor` / :func:`is_quantized`        - the tensor subclass
- :func:`dequantize_tensor`                         - block -> float dequant
- :class:`GGMLOps`                                  - ops for model construction
- :class:`GGUFModelPatcher`                         - quant-aware ModelPatcher
- :func:`gguf_sd_loader`                            - .gguf file -> fake state dict
- :func:`require_gguf`                              - readable missing-pkg error
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# GGML type codes + size table (gguf-py GGMLQuantizationType / GGML_QUANT_SIZES)
# --------------------------------------------------------------------------- #

#: GGML quantization type integer codes - MIRRORED from gguf-py 0.13.0's
#: ``GGMLQuantizationType`` (dumped via the T4 cross-check in
#: ``cdl_smoke_tests/test_quant_gguf.py``, which fails loudly on drift).
#: Do NOT hand-edit: re-dump from the installed package instead.
QTYPE_F32 = 0
QTYPE_F16 = 1
QTYPE_Q4_0 = 2
QTYPE_Q4_1 = 3
QTYPE_Q5_0 = 6
QTYPE_Q5_1 = 7
QTYPE_Q8_0 = 8
QTYPE_Q8_1 = 9
QTYPE_Q2_K = 10
QTYPE_Q3_K = 11
QTYPE_Q4_K = 12
QTYPE_Q5_K = 13
QTYPE_Q6_K = 14
QTYPE_Q8_K = 15
QTYPE_IQ2_XXS = 16
QTYPE_IQ2_XS = 17
QTYPE_IQ3_XXS = 18
QTYPE_IQ1_S = 19
QTYPE_IQ4_NL = 20
QTYPE_IQ3_S = 21
QTYPE_IQ2_S = 22
QTYPE_IQ4_XS = 23
QTYPE_I8 = 24
QTYPE_I16 = 25
QTYPE_I32 = 26
QTYPE_I64 = 27
QTYPE_F64 = 28
QTYPE_IQ1_M = 29
QTYPE_BF16 = 30
QTYPE_TQ1_0 = 34
QTYPE_TQ2_0 = 35

QTYPE_NAMES: Dict[int, str] = {
    QTYPE_F32: "F32", QTYPE_F16: "F16", QTYPE_Q4_0: "Q4_0", QTYPE_Q4_1: "Q4_1",
    QTYPE_Q5_0: "Q5_0", QTYPE_Q5_1: "Q5_1", QTYPE_Q8_0: "Q8_0",
    QTYPE_Q8_1: "Q8_1", QTYPE_Q2_K: "Q2_K", QTYPE_Q3_K: "Q3_K",
    QTYPE_Q4_K: "Q4_K", QTYPE_Q5_K: "Q5_K", QTYPE_Q6_K: "Q6_K",
    QTYPE_Q8_K: "Q8_K", QTYPE_IQ2_XXS: "IQ2_XXS", QTYPE_IQ2_XS: "IQ2_XS",
    QTYPE_IQ3_XXS: "IQ3_XXS", QTYPE_IQ1_S: "IQ1_S", QTYPE_IQ4_NL: "IQ4_NL",
    QTYPE_IQ3_S: "IQ3_S", QTYPE_IQ2_S: "IQ2_S", QTYPE_IQ4_XS: "IQ4_XS",
    QTYPE_I8: "I8", QTYPE_I16: "I16", QTYPE_I32: "I32", QTYPE_I64: "I64",
    QTYPE_F64: "F64", QTYPE_IQ1_M: "IQ1_M", QTYPE_BF16: "BF16",
    QTYPE_TQ1_0: "TQ1_0", QTYPE_TQ2_0: "TQ2_0",
}

#: ``{qtype: (block_size, type_size)}`` - mirrored from gguf-py 0.13.0's
#: ``GGML_QUANT_SIZES`` (GGML spec).  ``type_size`` is the bytes each
#: ``block_size``-element block occupies on disk; bytes/element is the ratio
#: (e.g. Q4_K: 144/256 = 0.5625 B/elem = 4.5 bits).  Same drift guard as above.
GGML_QTYPE_SIZES: Dict[int, Tuple[int, int]] = {
    QTYPE_F32: (1, 4), QTYPE_F16: (1, 2), QTYPE_Q4_0: (32, 18),
    QTYPE_Q4_1: (32, 20), QTYPE_Q5_0: (32, 22), QTYPE_Q5_1: (32, 24),
    QTYPE_Q8_0: (32, 34), QTYPE_Q8_1: (32, 40), QTYPE_Q2_K: (256, 84),
    QTYPE_Q3_K: (256, 110), QTYPE_Q4_K: (256, 144), QTYPE_Q5_K: (256, 176),
    QTYPE_Q6_K: (256, 210), QTYPE_Q8_K: (256, 292), QTYPE_IQ2_XXS: (256, 66),
    QTYPE_IQ2_XS: (256, 74), QTYPE_IQ3_XXS: (256, 98), QTYPE_IQ1_S: (256, 50),
    QTYPE_IQ4_NL: (32, 18), QTYPE_IQ3_S: (256, 110), QTYPE_IQ2_S: (256, 82),
    QTYPE_IQ4_XS: (256, 136), QTYPE_I8: (1, 1), QTYPE_I16: (1, 2),
    QTYPE_I32: (1, 4), QTYPE_I64: (1, 8), QTYPE_F64: (1, 8),
    QTYPE_IQ1_M: (256, 56), QTYPE_BF16: (1, 2), QTYPE_TQ1_0: (256, 54),
    QTYPE_TQ2_0: (256, 66),
}

#: Types torch handles natively (weight used as-is, no dequant pass).
TORCH_COMPATIBLE_QTYPES = frozenset({None, QTYPE_F32, QTYPE_F16})


def qtype_name(qtype: Optional[int]) -> str:
    return QTYPE_NAMES.get(qtype, repr(qtype))


def bytes_per_element(qtype: Optional[int]) -> float:
    """Disk bytes per *logical* element for a GGML type (the ADR-D3 hook).

    K-quants are fractional: Q4_K stores 256 elements in 144 bytes
    (0.5625 B/elem = 4.5 bits).  Unknown types raise - callers decide whether
    that is fatal or whether to assume fp16.
    """
    if qtype is None:
        return 4.0
    try:
        block_size, type_size = GGML_QTYPE_SIZES[qtype]
    except KeyError:
        raise ValueError(f"unknown GGML qtype {qtype!r} - no size table entry")
    return type_size / block_size


def is_torch_compatible(tensor) -> bool:
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


def is_quantized(tensor) -> bool:
    return not is_torch_compatible(tensor)


def require_gguf():
    """Return the ``gguf`` module or raise a readable install hint."""
    try:
        import gguf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The GGUF nodes need the 'gguf' package - install it with "
            "`pip install gguf>=0.13.0,<0.14` (see requirements.txt)."
        ) from exc
    import gguf
    return gguf


# --------------------------------------------------------------------------- #
# GGMLTensor (city96 ops.py, types by int code)
# --------------------------------------------------------------------------- #

class GGMLTensor(torch.Tensor):
    """A quantized weight: raw blocks on disk dtype + logical shape/qtype.

    The underlying storage is the mmapped uint8 block data; ``tensor_shape``
    is the *logical* (dequantized) shape and ``tensor_type`` the GGML type
    code.  ``clone``/``detach`` return self (the blocks are shared, never
    copied) so ComfyUI-style weight plumbing never doubles memory.
    """

    def __new__(cls, *args, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - the original swallows too
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy(),
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


# --------------------------------------------------------------------------- #
# Pure-torch dequant kernels (city96 dequant.py)
# --------------------------------------------------------------------------- #

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    """``GGMLTensor`` (or plain tensor) -> float tensor of ``dtype``."""
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in _DEQUANT_FUNCTIONS:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        # numpy fallback covers the long tail of IQ types; "incredibly slow".
        gguf = require_gguf()
        logging.info(
            f"[quant-formats] numpy dequant fallback for qtype {qtype_name(qtype)}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(),
                                     gguf.GGMLQuantizationType(qtype))
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)


def dequantize(data, qtype: int, oshape, dtype=None):
    """Dequantize raw ``uint8`` block data back to ``oshape`` floats.

    Returns a *plain* tensor: ``torch.Tensor`` subclasses propagate through
    ops by default, so without the final ``as_subclass`` the freshly
    dequantized weights would come back wrapped in ``GGMLTensor``.
    """
    try:
        block_size, type_size = GGML_QTYPE_SIZES[qtype]
    except KeyError:
        raise RuntimeError(
            f"no pure-torch dequant path for qtype {qtype_name(qtype)}; "
            "install the 'gguf' package for the numpy fallback.")
    dequantize_blocks = _DEQUANT_FUNCTIONS[qtype]

    data = data.as_subclass(torch.Tensor)  # plain view, zero copy
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape).as_subclass(torch.Tensor)


def _to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def _split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# --- full weights -----------------------------------------------------------

def _deq_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# --- legacy quants ----------------------------------------------------------

def _deq_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _deq_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device,
                                                   dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = qh & 1
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh.to(torch.uint8) << 4)
    return (d * qs) + m


def _deq_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device,
                                                 dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = qh & 1
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh.to(torch.uint8) << 4)).to(torch.int8) - 16
    return d * qs


def _deq_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _deq_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# --- K quants ---------------------------------------------------------------

QK_K = 256
K_SCALE_SIZE = 12


def _get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min_ = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), min_.reshape((n_blocks, 8))


def _deq_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = qh & 0x03
    q = (ql | (qh.reshape((n_blocks, -1, 32)) << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _deq_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _deq_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _deq_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))


def _deq_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device,
                         dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


# --- IQ quants ---------------------------------------------------------------

_KVALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8)


def _deq_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)
    kvalues = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return d * qs


def _deq_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = _to_uint16(scales_h)
    shift_a = torch.tensor([0, 4], device=d.device,
                           dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device,
                           dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03
    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F
    kvalues = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    return (dl * qs).reshape((n_blocks, -1))


_DEQUANT_FUNCTIONS = {
    QTYPE_BF16: _deq_BF16,
    QTYPE_Q8_0: _deq_Q8_0,
    QTYPE_Q5_1: _deq_Q5_1,
    QTYPE_Q5_0: _deq_Q5_0,
    QTYPE_Q4_1: _deq_Q4_1,
    QTYPE_Q4_0: _deq_Q4_0,
    QTYPE_Q6_K: _deq_Q6_K,
    QTYPE_Q5_K: _deq_Q5_K,
    QTYPE_Q4_K: _deq_Q4_K,
    QTYPE_Q3_K: _deq_Q3_K,
    QTYPE_Q2_K: _deq_Q2_K,
    QTYPE_IQ4_NL: _deq_IQ4_NL,
    QTYPE_IQ4_XS: _deq_IQ4_XS,
}


# --------------------------------------------------------------------------- #
# GGMLOps: lazy-dequant layers (city96 ops.py, adapted)
# --------------------------------------------------------------------------- #

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    return item


def _torch_compiler_disable():
    """``torch.compiler.disable`` where torch.compile would choke on
    GGMLTensor subclasses (torch < 2.8); a no-op on modern torch (ours are
    2.11/2.13) - kept for fidelity with the upstream port."""
    def noop(x):
        return x

    try:
        version_parts = torch.__version__.split("+")[0].split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
    except (ValueError, IndexError):
        return noop
    if (major, minor) >= (2, 8):
        return noop
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable
    return noop


# NOTE: ``comfy.lora`` is deliberately NOT imported here - it sits on
# ``_lazy_modules.LAZY_SUBMODULES`` (dehydrate list), and eagerly importing it
# pulls ``model_base`` -> the whole ldm tree, which materializes
# ``comfy.text_encoders`` as a real module and evicts the lazy proxy that
# ``sd.load_text_encoder_state_dicts`` relies on at runtime (CLIPLoader
# breaks).  Import it lazily where ``calculate_weight`` is used.
import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402


class GGMLLayer(torch.nn.Module):
    """Mixin: dequantize on the fly inside cast_bias_weight."""

    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = state_dict.get(f"{prefix}weight")
        bias = state_dict.get(f"{prefix}bias")
        # NOTE: load Linear always through the ggml path - the layer is not
        # initialized with a real weight at creation (see GGMLOps.Linear).
        if self.is_ggml_quantized(weight=weight, bias=bias) \
                or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch.
        if isinstance(self, torch.nn.Embedding) and self.weight is not None \
                and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if getattr(self, "weight", None) is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")

        # for vram estimation
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype \
                and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        # apply patches
        if len(patch_list) > 0:
            import comfy.lora  # lazy: on the dehydrate list (see module note)
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @_torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device,
                                     non_blocking=non_blocking, copy=False)

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device,
                                   non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


class GGMLOps(comfy.ops.manual_cast):
    """Dequantize weights on the fly before doing the compute."""

    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # No torch.empty here: on Windows the allocator would reserve the
            # full layer memory up front (pagefile spike - city96 issue).
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv1d(GGMLLayer, comfy.ops.manual_cast.Conv1d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Conv3d(GGMLLayer, comfy.ops.manual_cast.Conv3d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def __init__(self, *args, device=None, dtype=None, **kwargs):
            torch.nn.Module.__init__(self)
            self.num_embeddings = args[0] if len(args) > 0 else kwargs.get("num_embeddings")
            self.embedding_dim = args[1] if len(args) > 1 else kwargs.get("embedding_dim")
            for attr, default in (
                    ("padding_idx", None), ("max_norm", None), ("norm_type", 2.0),
                    ("scale_grad_by_freq", False), ("sparse", False)):
                setattr(self, attr, kwargs.get(attr, default))
            self.weight = None

        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight is not None and self.weight.dtype in (torch.float16,
                                                                 torch.bfloat16):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device,
                                                  dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm, self.norm_type,
                self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps)


# --------------------------------------------------------------------------- #
# GGUFModelPatcher: quant-aware weight patching (city96 nodes.py)
# --------------------------------------------------------------------------- #

import collections  # noqa: E402

import comfy.float  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.utils  # noqa: E402


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher whose weight patches ride on the GGMLTensor instead of
    dequantizing first (patches are applied to the dequantized weight inside
    ``GGMLLayer.get_weight`` at forward time)."""

    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(
                patches, self.load_device if self.patch_on_device
                else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches?
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple(
                    "Dimension", ["weight", "inplace_update"])(
                    weight.to(device=self.offload_device, copy=inplace_update),
                    inplace_update)

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(
                    weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            import comfy.lora  # lazy: on the dehydrate list (see module note)
            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: find another way to not unload after patches
        return super().unpatch_model(device_to=device_to,
                                     unpatch_weights=unpatch_weights)

    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        return super().load(*args, force_patch_weights=True, **kwargs)

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0  # force recalc
        return n


# --------------------------------------------------------------------------- #
# .gguf file reader (city96 loader.py, image-models subset)
# --------------------------------------------------------------------------- #

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos",
                 "ltxv", "hyvid", "wan", "lumina2", "qwen_image"}


def _gguf_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != 8:  # GGUFValueType.STRING
            raise TypeError(
                f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in (int, float, bool):
        return field_type(field.parts[field.data[-1]].item())
    raise TypeError(f"Unknown field type {field_type}")


def _gguf_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != 9 \
            or field.types[1] not in (4, 10):
        # expected ARRAY(9) of INT32(4) or INT64(10) - writers differ
        raise TypeError(f"Bad original shape metadata for {field_key}: {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def _gguf_metadata(reader):
    """All simple scalar metadata fields, safetensors-style."""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == 8:    # STRING
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == 4:  # INT32
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == 7:  # FLOAT32
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == 6:  # BOOL
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            continue
    return metadata


def gguf_sd_loader(path, handle_prefix="model.diffusion_model."):
    """Read a .gguf diffusion-model file as a state dict of GGMLTensor fakes.

    The tensors are mmapped views of the file (uint8 block data); nothing is
    dequantized here.  Returns ``(state_dict, extra)`` where ``extra`` carries
    ``arch_str`` and the file's scalar ``metadata``.
    """
    gguf = require_gguf()
    import warnings

    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    arch_str = _gguf_field(reader, "general.architecture", str)
    if arch_str is None:
        arch_str = "sd.cpp"
    if arch_str not in IMG_ARCH_LIST:
        raise ValueError(
            f"Unexpected architecture type in GGUF file: {arch_str!r} "
            f"(supported: {sorted(IMG_ARCH_LIST)})")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = _gguf_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32,
                                  gguf.GGMLQuantizationType.F16):
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor,
                                        tensor_type=int(tensor.tensor_type),
                                        tensor_shape=shape)

        # 1D tensors shouldn't be quantized; fix for BF16
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key],
                                                   dtype=torch.float32)

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    logging.info("[quant-formats] gguf qtypes: "
                 + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    extra = {"arch_str": arch_str, "metadata": _gguf_metadata(reader)}
    return state_dict, extra


# --------------------------------------------------------------------------- #
# teaching helpers (CdlQuantizationInspector groundwork)
# --------------------------------------------------------------------------- #

_QUANT_SUFFIX_RE = re.compile(r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$", re.IGNORECASE)


def strip_quant_suffix(name: str) -> str:
    """Drop the trailing ``-Q4_K_M`` style quant tag from a file stem."""
    match = _QUANT_SUFFIX_RE.search(name)
    if match:
        name = name[:match.start()]
    return name


def summarize_state_dict_bytes(state_dict) -> Dict[str, object]:
    """Layer-level byte ledger for a (possibly GGML) state dict.

    Returns ``{"layers": [...], "total_bytes": int, "fp_bytes": int}`` where
    each layer entry is ``{"key", "qtype", "bytes", "fp_bytes", "elements"}``.
    Unquantized tensors count at their natural itemsize.
    """
    layers = []
    total = 0
    fp_total = 0
    for key, tensor in state_dict.items():
        qtype = getattr(tensor, "tensor_type", None)
        logical_shape = getattr(tensor, "tensor_shape", None) or tuple(tensor.shape)
        numel = 1
        for dim in logical_shape:
            numel *= int(dim)
        if is_quantized(tensor):
            bpe = bytes_per_element(qtype)
            nbytes = int(numel * bpe)
            fp_bytes = numel * 2 if qtype in (QTYPE_F16, QTYPE_BF16) else numel * 4
            name = qtype_name(qtype)
        else:
            nbytes = numel * tensor.element_size()
            fp_bytes = nbytes
            name = str(getattr(tensor, "dtype", "float32")).replace("torch.", "")
        layers.append({"key": key, "qtype": name, "bytes": nbytes,
                       "fp_bytes": fp_bytes, "elements": numel})
        total += nbytes
        fp_total += fp_bytes
    layers.sort(key=lambda entry: entry["bytes"], reverse=True)
    return {"layers": layers, "total_bytes": total, "fp_bytes": fp_total}
