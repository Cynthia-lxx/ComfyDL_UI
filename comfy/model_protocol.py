"""Weight-level MODEL protocol layer for the dehydrated ComfyUI build.

The dehydrated build removed ``comfy/ldm``, ``comfy/model_detection``,
``comfy/supported_models*``, ``comfy/model_base``, ``comfy/text_encoders`` and
the LoRA implementation, so nothing in the repository can turn a checkpoint into
a runnable diffusion model any more. This module rebuilds only the part of that
pipeline that never needed architecture knowledge in the first place:

* reading a weight file,
* grouping its keys into "diffusion model" / "text encoder" / "VAE" buckets
  (a pure string operation on key prefixes - **no** architecture detection),
* packing a bucket into a :class:`~comfy.model_patcher.ModelPatcher` so it can
  travel through the node graph as a ``MODEL`` / ``CLIP`` / ``VAE`` value,
* blending two buckets key by key (the state_dict equivalent of a merge), and
* writing a bucket back to disk.

Nothing here imports :mod:`comfy.sd`, :mod:`comfy.lora`, :mod:`comfy.hooks`
extension points or any other module that was dehydrated, so the whole module
works on a vanilla ``torch`` install. Inference (sampling, VAE encode/decode,
text encoding) is deliberately out of scope: it needs the deleted architecture
code and is reported as such by the nodes built on top of this layer.

Key design decisions
--------------------
* :class:`StateDictModule` keeps the weights in a **flat** dictionary and only
  overrides the ``state_dict`` family, instead of rebuilding a nested module
  tree from the dotted keys. Dotted keys such as ``a.b.c`` may address both a
  leaf (``a.b``) and a container (``a.b.c``) in the same file, which a nested
  tree cannot represent; a flat dictionary round-trips every key verbatim.
* Tensors are held **by reference** (no copy, no ``nn.Parameter`` wrapping), so
  loading a ``SAFETENSORS`` file costs exactly one file's worth of memory.
* Key names are **never rewritten**. Stripping the outer container prefix (e.g.
  ``model.`` in an SD1.5 checkpoint) records the removed prefix per key in a
  small map, which is replayed verbatim on save. "Load then save" is therefore
  byte-for-byte key-identical, and merges stay predictable because the filter
  prefix is a documented, user-visible widget.
"""

from __future__ import annotations

import collections
import logging
from typing import Callable, Iterable, Mapping, Sequence

import torch

import comfy.model_management
import comfy.model_patcher

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Key-prefix vocabulary
# --------------------------------------------------------------------------- #

#: Prefixes whose keys belong to the diffusion model bucket.
MODEL_GROUP_PREFIXES: tuple[str, ...] = (
    "diffusion_model.",
    "unet.",
    "denoiser.",
)

#: Prefixes whose keys belong to the VAE bucket.
VAE_GROUP_PREFIXES: tuple[str, ...] = (
    "first_stage_model.",
    "vae.",
    "autoencoder.",
    "taesd.",
)

#: Prefixes whose keys belong to the text-encoder bucket.
CLIP_GROUP_PREFIXES: tuple[str, ...] = (
    "cond_stage_model.",
    "conditioner.",
    "text_encoders.",
    "text_encoder.",
    "clip.",
)

#: Outermost wrappers seen in the wild around a whole checkpoint
#: (``model.diffusion_model.*`` in an SD1.5/SDXL file, ``module.``/``state_dict.``
#: in files exported from a training script). ``model_ema.`` is deliberately NOT
#: listed: stripping it would collide with the plain ``model.`` keys.
CONTAINER_PREFIXES: tuple[str, ...] = (
    "model.",
    "module.",
    "state_dict.",
)

#: ``strip_outer_prefix(mode="auto")`` strips a candidate only when it covers at
#: least this fraction of the file's tensor keys - this keeps a small, genuine
#: ``model.`` submodule inside an unrelated file from being mistaken for a
#: container wrapper.
AUTO_STRIP_MIN_COVERAGE = 0.30

_IncompatibleKeys = collections.namedtuple(
    "_IncompatibleKeys", ["missing_keys", "unexpected_keys"]
)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #


class StateDictModule(torch.nn.Module):
    """Weight container backed by a flat ``state_dict``.

    What: holds plain tensors in a flat dictionary and overrides the ``state_dict``
          family so the keys round-trip verbatim. This is the object a
          ``ModelPatcher`` wraps in the protocol layer: ``ModelPatcher`` reads
          weights through ``self.model.state_dict()``, so overriding that single
          method is enough to make ``MODEL`` travel through the graph without
          any architecture knowledge.
    In:   sd (Mapping[str, torch.Tensor]) - the weights; non tensor entries are
          dropped (a checkpoint may carry ``epoch``/``global_step`` scalars).
          key_prefixes (Mapping[str, str], optional) - the outer container prefix
          that was stripped from each key on load (``"model."`` for the keys that
          came from ``model.diffusion_model.*``). Replayed by
          :func:`restore_prefixes` when the weights are written back.
    Out:  behaves like a parameter-less ``nn.Module`` whose ``state_dict()`` is a
          shallow copy of the flat dictionary. ``get_sd()`` is an alias matching
          the native ``VAE.get_sd`` / ``CLIP.get_sd`` habit.
    """

    def __init__(
        self,
        sd: Mapping[str, torch.Tensor],
        key_prefixes: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._sd: dict[str, torch.Tensor] = {
            key: value for key, value in sd.items() if isinstance(value, torch.Tensor)
        }
        self._key_prefixes: dict[str, str] = dict(key_prefixes or {})
        # ModelPatcher assigns these when it wraps us; provide sane defaults so
        # the container is usable stand-alone too.
        self.device = torch.device("cpu")
        self.model_type = None
        self.dropped_entries = len(sd) - len(self._sd)

    # -- state_dict protocol ------------------------------------------------ #

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Return a shallow copy of the flat dictionary (``nn.Module`` compatible).

        A copy is essential: ``ModelPatcher.model_state_dict(filter_prefix=...)``
        pops keys from whatever this returns, and those pops must not damage the
        stored weights.
        """
        if destination is None:
            return dict(self._sd)
        for key, value in self._sd.items():
            destination[prefix + key if prefix else key] = value
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for key, value in self._sd.items():
            destination[prefix + key] = value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Replace the weights with ``state_dict`` (flat, non tensor entries dropped)."""
        incoming = {
            key: value
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
        missing = [key for key in self._sd if key not in incoming]
        unexpected = [key for key in incoming if key not in self._sd]
        self._sd = incoming
        if strict and (missing or unexpected):
            raise RuntimeError(
                "StateDictModule.load_state_dict: {} missing and {} unexpected key(s); "
                "first missing={!r} first unexpected={!r}".format(
                    len(missing), len(unexpected), missing[:1], unexpected[:1]
                )
            )
        return _IncompatibleKeys(missing, unexpected)

    def _apply(self, fn, recurse=True):
        # Route .to()/.cpu()/.cuda()/.float() through the flat dictionary, since
        # there are no registered parameters or buffers to walk.
        self._sd = {key: fn(value) for key, value in self._sd.items()}
        return self

    # -- convenience -------------------------------------------------------- #

    def get_sd(self) -> dict[str, torch.Tensor]:
        """Alias of ``state_dict()`` - native ``VAE.get_sd`` / ``CLIP.get_sd`` habit."""
        return dict(self._sd)

    @property
    def key_prefixes(self) -> dict[str, str]:
        """Map ``current key -> outer container prefix stripped on load``."""
        return dict(self._key_prefixes)

    def keys(self):
        return self._sd.keys()

    def values(self):
        return self._sd.values()

    def items(self):
        return self._sd.items()

    def __len__(self) -> int:
        return len(self._sd)

    def __contains__(self, key: object) -> bool:
        return key in self._sd

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._sd[key]

    def __repr__(self) -> str:
        return "StateDictModule(keys={}, bytes={}, stripped_prefixes={})".format(
            len(self._sd), total_bytes(self._sd), len(self._key_prefixes)
        )


# --------------------------------------------------------------------------- #
# Introspection helpers
# --------------------------------------------------------------------------- #


def total_bytes(sd: Mapping[str, torch.Tensor]) -> int:
    """Sum the storage size of every tensor in ``sd``.

    In:  sd - a flat weight mapping; non tensor entries (``epoch``,
         ``global_step``, ...) are ignored.
    Out: the size in bytes (the same number ``comfy.model_management.module_size``
         reports for a container, computed without materialising a module).
    """
    return sum(
        int(value.nbytes) for value in sd.values() if isinstance(value, torch.Tensor)
    )


def group_of(key: str) -> str:
    """Classify a weight key into ``"model"``, ``"vae"`` or ``"clip"``.

    Purely a string test on documented key prefixes; architecture detection is
    not involved. Unknown keys are reported as ``"model"`` so that nothing is
    ever dropped by the loader.

    In:  key - one weight key, e.g. ``"first_stage_model.decoder.conv.weight"``.
    Out: one of ``"model"``, ``"vae"``, ``"clip"``.
    """
    for prefix in VAE_GROUP_PREFIXES:
        if key.startswith(prefix):
            return "vae"
    for prefix in CLIP_GROUP_PREFIXES:
        if key.startswith(prefix):
            return "clip"
    return "model"


# --------------------------------------------------------------------------- #
# Prefix normalisation
# --------------------------------------------------------------------------- #


def strip_outer_prefix(
    sd: Mapping[str, torch.Tensor], mode: str = "auto"
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Strip the outermost container prefix from a checkpoint's keys.

    What: SD1.5/SDXL checkpoints wrap their weights in ``model.``
          (``model.diffusion_model.*``), which hides the ``diffusion_model.``
          marker the rest of this module groups on. This removes that wrapper.
          It is a pure key-string operation, not architecture detection.
    In:   sd   - the flat weight mapping (unmodified).
          mode - ``"auto"`` (detect using :data:`CONTAINER_PREFIXES` and
                 :data:`AUTO_STRIP_MIN_COVERAGE`), ``"raw"`` (never strip), or an
                 explicit prefix string such as ``"model."``.
    Out:  ``(stripped_sd, key_prefixes)``. Only keys that really start with the
          prefix are shortened; every other key (``model_ema.diffusion_model.*``
          in an SD checkpoint, for instance) is kept verbatim and simply has no
          entry in ``key_prefixes``. :func:`restore_prefixes` can therefore rebuild
          the original key set exactly.
    """
    if not sd:
        return {}, {}
    mode = str(mode or "auto").strip()
    if mode.lower() == "raw":
        return dict(sd), {}

    if mode.lower() == "auto":
        prefix = _detect_container_prefix(sd)
    else:
        prefix = mode if mode.endswith(".") else mode + "."
        if not any(key.startswith(prefix) for key in sd):
            LOGGER.warning(
                "model_protocol: prefix_strip=%r was requested but no key starts with "
                "it; the keys are kept as-is.", prefix
            )
            return dict(sd), {}

    if not prefix:
        return dict(sd), {}

    stripped: dict[str, torch.Tensor] = {}
    key_prefixes: dict[str, str] = {}
    collisions = 0
    for key, value in sd.items():
        short = key[len(prefix):] if key.startswith(prefix) else key
        if short in stripped and short not in (key,):
            collisions += 1
        stripped[short] = value
        if short != key:
            key_prefixes[short] = prefix
    if collisions:
        LOGGER.warning(
            "model_protocol: stripping %r collapsed %d key(s) onto an existing name; "
            "the later value wins.", prefix, collisions,
        )
    return stripped, key_prefixes


def _detect_container_prefix(sd: Mapping[str, torch.Tensor]) -> str:
    """Return the container prefix to strip, or ``""`` when there is none."""
    total = len(sd)
    for candidate in CONTAINER_PREFIXES:
        covered = sum(1 for key in sd if key.startswith(candidate))
        if covered and covered / total >= AUTO_STRIP_MIN_COVERAGE:
            LOGGER.info(
                "model_protocol: auto-detected container prefix %r covering %d/%d keys.",
                candidate, covered, total,
            )
            return candidate
    return ""


def restore_prefixes(
    sd: Mapping[str, torch.Tensor], key_prefixes: Mapping[str, str] | None
) -> dict[str, torch.Tensor]:
    """Undo :func:`strip_outer_prefix` by prepending the recorded prefixes.

    In:  sd - a (possibly merged) flat weight mapping with normalised keys.
         key_prefixes - the map produced on load; missing keys are left alone.
    Out: a new mapping whose keys match the original file's key set.
    """
    if not key_prefixes:
        return dict(sd)
    return {
        key_prefixes.get(key, "") + key: value for key, value in sd.items()
    }


def split_checkpoint(
    sd: Mapping[str, torch.Tensor], prefix_strip: str = "auto"
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, str],
]:
    """Split a checkpoint into ``(model, clip, vae, key_prefixes)`` by key prefix.

    What: the protocol-layer replacement for ``comfy.sd.load_checkpoint_guess_config``.
          Grouping uses :func:`group_of`, i.e. documented key prefixes only; no
          architecture detection is performed, so the buckets are only guaranteed
          to be *complete*, not to be a runnable model.
    In:   sd - the flat weight mapping of a checkpoint file.
          prefix_strip - forwarded to :func:`strip_outer_prefix`.
    Out: ``(model_sd, clip_sd, vae_sd, key_prefixes)``. Keys that match no group
         land in ``model_sd`` so no weight is ever lost, and ``key_prefixes``
         covers every key of all three buckets. Non tensor entries (``epoch``,
         ``global_step``) are dropped, because neither a container nor
         ``safetensors`` can carry them.
    """
    stripped, key_prefixes = strip_outer_prefix(sd, prefix_strip)
    model_sd: dict[str, torch.Tensor] = {}
    clip_sd: dict[str, torch.Tensor] = {}
    vae_sd: dict[str, torch.Tensor] = {}
    buckets = {"model": model_sd, "clip": clip_sd, "vae": vae_sd}
    for key, value in stripped.items():
        if not isinstance(value, torch.Tensor):
            key_prefixes.pop(key, None)
            continue
        buckets[group_of(key)][key] = value
    return model_sd, clip_sd, vae_sd, key_prefixes


# --------------------------------------------------------------------------- #
# ModelPatcher packing
# --------------------------------------------------------------------------- #


def make_model_patcher(
    sd: Mapping[str, torch.Tensor],
    key_prefixes: Mapping[str, str] | None = None,
) -> comfy.model_patcher.ModelPatcher:
    """Wrap weights in a :class:`ModelPatcher` so they can flow as ``MODEL``.

    What: the protocol-layer stand-in for ``comfy.sd.load_checkpoint_guess_config``'s
          first return value. The patcher is created with no patches applied and
          with devices coming from ``comfy.model_management``, so downstream
          nodes can read ``patcher.model_state_dict()`` and save it again.
    In:   sd - the model bucket.
          key_prefixes - prefix map from :func:`split_checkpoint`.
    Out: a ``ModelPatcher`` whose ``.model`` is a :class:`StateDictModule`.
    """
    container = StateDictModule(sd, key_prefixes)
    return comfy.model_patcher.ModelPatcher(
        container,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
        size=total_bytes(sd),
    )


def make_container(
    sd: Mapping[str, torch.Tensor],
    key_prefixes: Mapping[str, str] | None = None,
) -> StateDictModule:
    """Wrap weights in a plain :class:`StateDictModule` (used for ``CLIP``/``VAE``)."""
    return StateDictModule(sd, key_prefixes)


def container_state_dict(value) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read ``(weights, key_prefixes)`` out of any protocol-layer value.

    Accepts a ``ModelPatcher`` (unwraps ``.model``), a :class:`StateDictModule`,
    or a bare ``dict`` so that the save/merge nodes can be wired in any order.

    In:  value - a MODEL / CLIP / VAE protocol-layer object.
    Out: ``(state_dict, key_prefixes)``; ``key_prefixes`` is ``{}`` when unknown.
    """
    model = getattr(value, "model", None)
    if model is not None and hasattr(model, "state_dict"):
        # ModelPatcher.model_state_dict() keeps patches out of the picture
        # (they are baked at merge time in this layer), so read it directly.
        inner = model
    else:
        inner = value
    if isinstance(inner, Mapping):
        return dict(inner), {}
    sd = inner.state_dict()
    prefixes = getattr(inner, "key_prefixes", None)
    return dict(sd), dict(prefixes or {})


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def select_merge_keys(
    keys: Iterable[str], prefix: str, fallback_all: bool = False
) -> tuple[list[str], str]:
    """Select the keys a merge node should blend.

    What: mirrors the native merge nodes, which filter on ``diffusion_model.``.
          When that literal prefix matches nothing (for example a UNET-only file
          whose keys are ``double_blocks.*``, or a checkpoint loaded with
          ``prefix_strip="raw"`` so its keys read ``model.diffusion_model.*``),
          the prefix is retried at a dotted-segment boundary so the node still
          does what the user asked for instead of silently merging nothing.
    In:  keys - the candidate keys (usually the first model's).
         prefix - the user supplied filter; ``""`` means "every key".
         fallback_all - when the marker appears nowhere at all, report the whole
         key set as ``"unprefixed"`` instead of ``"none"``. A bucket loaded from
         a bare UNET file *is* the diffusion model, so blending all of it is the
         correct reading; the caller decides whether to use that or warn.
    Out: ``(selected_keys, mode)`` with ``mode`` one of ``"all"``, ``"exact"``,
         ``"segment"``, ``"unprefixed"`` or ``"none"``.
    """
    key_list = list(keys)
    prefix = str(prefix or "").strip()
    if not prefix:
        return key_list, "all"
    exact = [key for key in key_list if key.startswith(prefix)]
    if exact:
        return exact, "exact"
    marker = "." + prefix if not prefix.startswith(".") else prefix
    segmented = [key for key in key_list if marker in key]
    if segmented:
        return segmented, "segment"
    if fallback_all:
        return key_list, "unprefixed"
    return [], "none"


def merge_state_dicts(
    base_sd: Mapping[str, torch.Tensor],
    other_sd: Mapping[str, torch.Tensor],
    blend_fn: Callable[[str], tuple[float, float] | None],
    *,
    keys: Sequence[str] | None = None,
    exclude_suffixes: Sequence[str] = (),
    dtype_unify: bool = True,
) -> tuple[dict[str, torch.Tensor], int, list[str]]:
    """Blend two weight mappings key by key (the state_dict equivalent of a merge).

    What: the native merge nodes build ``ModelPatcher`` patches that are only
          applied during inference, which is impossible in this build. This
          computes ``out[k] = w_base * a[k] + w_other * b[k]`` for every selected
          key straight away, so the merged weights are real, saveable tensors.
          The weight pair matches the native ``add_patches({k: kp2[k]},
          strength_model, strength_patch)`` semantics, where ``strength_model``
          scales the first model and ``strength_patch`` the second
          (``ModelMergeSimple(ratio)`` uses ``(ratio, 1 - ratio)`` so that
          ``ratio=1`` keeps 100% of the first model, ``ModelMergeAdd`` uses
          ``(1, 1)`` and ``ModelMergeSubtract(m)`` uses ``(m, -m)``).
    In:   base_sd - weights of the first model; every key of it appears in the
          result (unmatched keys are copied through untouched).
          other_sd - weights of the second model.
          blend_fn - ``key -> (w_base, w_other)``; return ``None`` to leave the
          key alone.
          keys - the keys to consider; defaults to every key of ``base_sd``.
          exclude_suffixes - keys ending with any of these are skipped (the CLIP
          merges use this for ``.position_ids`` / ``.logit_scale``).
          dtype_unify - when ``True`` (default) a dtype mismatch between the two
          sides is reported and the key is left untouched, because silently
          blending fp16 with fp8 tends to produce garbage.
    Out: ``(merged_sd, matched_count, skipped_keys)``. ``skipped_keys`` holds the
         keys that exist in ``other_sd`` only (native behaviour drops those, so
         they are reported rather than silently blended).
    """
    merged = dict(base_sd)
    considered = list(keys) if keys is not None else list(base_sd.keys())
    skipped: list[str] = []
    matched = 0
    for key in considered:
        if any(key.endswith(suffix) for suffix in exclude_suffixes):
            continue
        if key not in other_sd:
            skipped.append(key)
            continue
        weights = blend_fn(key)
        if weights is None:
            continue
        base = base_sd[key]
        other = other_sd[key]
        merged[key] = _blend_tensor(base, other, weights[0], weights[1], dtype_unify, key)
        matched += 1
    return merged, matched, skipped


def _blend_tensor(
    base: torch.Tensor,
    other: torch.Tensor,
    w_base: float,
    w_other: float,
    dtype_unify: bool,
    key: str,
) -> torch.Tensor:
    """Blend one pair of tensors, leaving non float/shape-mismatched pairs alone."""
    if base.shape != other.shape:
        LOGGER.warning(
            "model_protocol: %r has shape %s on one side and %s on the other; "
            "keeping the first model's tensor.", key, tuple(base.shape), tuple(other.shape),
        )
        return base
    if not base.is_floating_point() or not other.is_floating_point():
        return base
    if dtype_unify and base.dtype != other.dtype:
        LOGGER.warning(
            "model_protocol: %r is %s on one side and %s on the other; keeping the "
            "first model's tensor.", key, base.dtype, other.dtype,
        )
        return base
    if w_base == 1.0 and w_other == 0.0:
        return base
    if w_base == 0.0 and w_other == 1.0:
        return other
    base_f = base if base.dtype == torch.float32 else base.to(torch.float32)
    other_f = other if other.dtype == torch.float32 else other.to(torch.float32)
    if w_base == 1.0:
        blended = base_f.clone()
        blended.add_(other_f, alpha=float(w_other))
    elif w_other == 1.0:
        blended = other_f.clone()
        blended.add_(base_f, alpha=float(w_base))
    else:
        # mul() always allocates, so base_f / other_f are never mutated.
        blended = base_f.mul(float(w_base)).add_(other_f, alpha=float(w_other))
    return blended if blended.dtype == base.dtype else blended.to(base.dtype)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


def prepare_for_save(
    sd: Mapping[str, torch.Tensor], key_prefixes: Mapping[str, str] | None = None
) -> dict[str, torch.Tensor]:
    """Build the dictionary handed to ``safetensors`` for one container.

    What: replays the stripped container prefix, makes every tensor contiguous
          (``safetensors`` refuses non-contiguous views, and mmap-loaded weights
          are views into the file) and passes non tensor entries through.
    In:   sd - normalised weights; key_prefixes - the map from load time.
    Out: a flat mapping keyed exactly like the source file.
    """
    output: dict[str, torch.Tensor] = {}
    for key, value in restore_prefixes(sd, key_prefixes).items():
        if isinstance(value, torch.Tensor) and not value.is_contiguous():
            value = value.contiguous()
        output[key] = value
    return output


def merge_group_state_dicts(
    groups: Sequence[tuple[Mapping[str, torch.Tensor], Mapping[str, str] | None]]
) -> dict[str, torch.Tensor]:
    """Concatenate several ``(state_dict, key_prefixes)`` pairs into one mapping.

    Used by ``CheckpointSave`` to write the MODEL + CLIP + VAE buckets of a
    checkpoint back into a single file.

    In:  groups - the buckets, in output order (later buckets win on collision).
    Out: one flat mapping ready for :func:`prepare_for_save`-style contiguity
         handling and ``comfy.utils.save_torch_file``.
    """
    output: dict[str, torch.Tensor] = {}
    for sd, prefixes in groups:
        output.update(prepare_for_save(sd, prefixes))
    return output


__all__ = [
    "StateDictModule",
    "AUTO_STRIP_MIN_COVERAGE",
    "CLIP_GROUP_PREFIXES",
    "CONTAINER_PREFIXES",
    "MODEL_GROUP_PREFIXES",
    "VAE_GROUP_PREFIXES",
    "container_state_dict",
    "group_of",
    "make_container",
    "make_model_patcher",
    "merge_group_state_dicts",
    "merge_state_dicts",
    "prepare_for_save",
    "restore_prefixes",
    "select_merge_keys",
    "split_checkpoint",
    "strip_outer_prefix",
    "total_bytes",
]
