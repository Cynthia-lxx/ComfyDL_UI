"""Crash site output serialization: ComfyUI node outputs <-> SQLite BLOBs.

Type -> format mapping (bigplan ADR-4, see docs/crashsite.md):

======================  ====================  =========
output                  format tag            fidelity
======================  ====================  =========
torch.Tensor            "st"                  full
dict (LATENT/AUDIO/..)  "dict-st"             full*     tensors flattened, scalars -> meta
list/tuple (COND.)      "list-st"             degraded  flattened + structural meta
str/int/float/bool/None "json"                full
CdlDataset (duck-typed) "dataset-st"          full      features/labels + names/meta
nn.Module (nn_model)    "pt"                  degraded  state_dict only
ModelPatcher            unsupported           --        rejected in should_cache
anything else           "pickle" (fallback)   degraded  torch.save; restored with
                                                        weights_only=False (own DB)
======================  ====================  =========

BLOB payload is ``safetensors.torch.save(...)`` bytes (training_protocol.py:949
precedent) whenever tensors are involved; meta (scalars, structure, key order)
travels as a JSON string in a separate column.  Tensors are always moved to
CPU with ``detach().cpu().clone()`` before serialization - the executor runs
inside ``torch.inference_mode`` and outputs may live on the GPU.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional, Tuple

import torch


class UnsupportedOutputError(Exception):
    """Raised for outputs we deliberately refuse to snapshot (MODEL etc.)."""


def _is_model_patcher(obj: Any) -> bool:
    try:
        from comfy.model_patcher import ModelPatcher

        return isinstance(obj, ModelPatcher)
    except Exception:
        return type(obj).__name__ in ("ModelPatcher", "ModelPatcherDynamic")


def is_unsupported(obj: Any) -> bool:
    """True when should_cache must reject the output outright (ADR-4)."""
    if _is_model_patcher(obj):
        return True
    # A DataLoader cannot round-trip; its node is marked unrescuable in the
    # UI and rebuilt from its DATASET upstream instead.
    return type(obj).__name__ in ("DataLoader", "cdlDataloader")


def _cpu(t: torch.Tensor) -> torch.Tensor:
    # clone() escapes inference_mode views; cpu() survives the GPU going away.
    return t.detach().cpu().contiguous().clone()


def _flatten(value: Any, prefix: str, tensors: Dict[str, Any],
             meta: Dict[str, Any]) -> None:
    """Split a structure into safetensors entries + JSON-able meta."""
    if isinstance(value, torch.Tensor):
        tensors[prefix] = _cpu(value)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        meta[prefix] = value
    elif isinstance(value, dict):
        meta[prefix + "/__keys__"] = [str(k) for k in value.keys()]
        for k, v in value.items():
            _flatten(v, f"{prefix}/{k}", tensors, meta)
    elif isinstance(value, (list, tuple)):
        meta[prefix + "/__type__"] = type(value).__name__
        for i, item in enumerate(value):
            _flatten(item, f"{prefix}/{i}", tensors, meta)
    else:  # exotic leaf: lossy repr, flagged
        meta[prefix + "/__repr__"] = repr(value)[:2000]


def _st_save(tensors: Dict[str, Any]) -> bytes:
    import safetensors.torch

    return safetensors.torch.save(tensors)


def _st_load(blob: bytes) -> Dict[str, Any]:
    import safetensors.torch

    return safetensors.torch.load(blob)


def _torch_save_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def serialize_output(obj: Any) -> Tuple[str, bytes, str]:
    """Serialize one node output object -> (format_tag, blob, meta_json).

    Never raises for ordinary failures: a ``"failed"`` marker is returned so
    one weird output cannot take down capture (courtesy rule).
    """
    try:
        if isinstance(obj, torch.Tensor):
            tensors, meta = {}, {}
            _flatten(obj, "v", tensors, meta)
            return "st", _st_save(tensors), json.dumps(meta)

        if isinstance(obj, (dict, list, tuple)):
            tensors, meta = {}, {}
            _flatten(obj, "root", tensors, meta)
            if tensors:
                tag = "dict-st" if isinstance(obj, dict) else "list-st"
                return tag, _st_save(tensors), json.dumps(meta)
            # Wrap like every other json row - deserialize_output unpacks
            # {"v": ...}; a bare row broke list outputs with "list indices
            # must be integers" on lookup (2026-10-09).
            return "json", json.dumps({"v": obj}, default=str).encode("utf-8"), "{}"

        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return "json", json.dumps({"v": obj}).encode("utf-8"), "{}"

        # ComfyDL CdlDataset (duck-typed: avoids importing the node package).
        if all(hasattr(obj, a) for a in ("features", "feature_names")):
            tensors, meta = {}, {}
            _flatten(obj.features, "features", tensors, meta)
            if getattr(obj, "labels", None) is not None:
                _flatten(obj.labels, "labels", tensors, meta)
            meta["feature_names"] = list(obj.feature_names)
            meta["target_name"] = getattr(obj, "target_name", None)
            meta["dataset_meta"] = _jsonable(getattr(obj, "meta", {}))
            return ("dataset-st", _st_save(tensors),
                    json.dumps(meta, default=str))

        if isinstance(obj, torch.nn.Module):
            return "pt", _torch_save_bytes(obj.state_dict()), "{}"

        # Generic fallback for unknown small objects.
        return "pickle", _torch_save_bytes(obj), "{}"
    except UnsupportedOutputError:
        raise
    except Exception as exc:  # noqa: BLE001
        return "failed", json.dumps({"error": repr(exc)}).encode("utf-8"), "{}"


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)[:2000]


def _unflatten(prefix: str, tensors: Dict[str, Any], meta: Dict[str, Any]) -> Any:
    """Inverse of :func:`_flatten` for the subtree rooted at ``prefix``."""
    if prefix in tensors:
        return tensors[prefix]
    if prefix in meta:
        return meta[prefix]
    if prefix + "/__repr__" in meta:
        return None  # lossy leaf: cannot restore, surface as None
    type_name = meta.get(prefix + "/__type__")
    keys = meta.get(prefix + "/__keys__")
    if keys is not None:  # dict subtree
        out = {}
        for k in keys:
            out[k] = _unflatten(f"{prefix}/{k}", tensors, meta)
        return out
    if type_name is not None:  # list/tuple subtree
        items = []
        i = 0
        while f"{prefix}/{i}" in tensors or f"{prefix}/{i}" in meta \
                or f"{prefix}/{i}/__keys__" in meta \
                or f"{prefix}/{i}/__type__" in meta \
                or f"{prefix}/{i}/__repr__" in meta:
            items.append(_unflatten(f"{prefix}/{i}", tensors, meta))
            i += 1
        return tuple(items) if type_name == "tuple" else items
    return None


def deserialize_output(format_tag: str, blob: bytes,
                       meta_json: str) -> Optional[Any]:
    """Inverse of :func:`serialize_output`. Returns the restored object."""
    if format_tag == "failed":
        return None
    meta = json.loads(meta_json or "{}")

    if format_tag == "json":
        data = json.loads(blob.decode("utf-8"))
        # {"v": ...} is the wrapped form; legacy rows written before the
        # 2026-10-09 fix were bare - accept both.
        if isinstance(data, dict) and set(data.keys()) == {"v"}:
            return data["v"]
        return data

    if format_tag == "st":
        return _unflatten("v", _st_load(blob), meta)

    if format_tag == "dataset-st":
        from comfydl.nodes.data_types import CdlDataset

        return CdlDataset(
            features=_unflatten("features", _st_load(blob), meta),
            labels=_unflatten("labels", _st_load(blob), meta),
            feature_names=list(meta.get("feature_names") or []),
            target_name=meta.get("target_name"),
            meta=meta.get("dataset_meta") or {},
        )

    if format_tag in ("dict-st", "list-st"):
        root = _unflatten("root", _st_load(blob), meta)
        return root

    if format_tag == "pt":
        return torch.load(io.BytesIO(blob), map_location="cpu",
                          weights_only=True)

    if format_tag == "pickle":
        # Own database, own writer: the plain-pickle fallback is restored
        # with the restriction lifted deliberately (documented).
        return torch.load(io.BytesIO(blob), map_location="cpu",
                          weights_only=False)
    return None
