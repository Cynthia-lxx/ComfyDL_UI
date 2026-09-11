"""Round-trip tests for the ``utilities/conversion`` nodes.

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_conversion_roundtrip.py

Why this script exists
----------------------
``run_smoke_test.py`` synthesises only the *first* alternative of a union input,
which for ``CdlValueToTensor`` is always IMAGE.  Every other branch (MASK,
LATENT, AUDIO, SIGMAS) and every metadata socket would therefore never be
exercised.  This script drives both directions explicitly and asserts that a
collect -> dispatch round trip rebuilds the original object key for key.

The node module is loaded straight from disk rather than through ``comfydl``:
importing the package would pull in every other ComfyDL node module, and the
registration path is already covered by the smoke test.
"""

import importlib.util
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MODULE_PATH = REPO_ROOT / "comfydl" / "nodes" / "conversion.py"
_spec = importlib.util.spec_from_file_location("cdl_conversion_under_test", _MODULE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"cannot load {_MODULE_PATH}")
conversion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conversion)

Collect = conversion.CdlValueToTensor
Dispatch = conversion.CdlTensorToValue
PackLora = conversion.CdlLoraModelToTensor
UnpackLora = conversion.CdlTensorToLoraModel
PackLoss = conversion.CdlLossMapToTensor
UnpackLoss = conversion.CdlTensorToLossMap

_RESULTS: list[tuple[str, bool, str]] = []


def _unwrap(node_output):
    """Return the tuple a ``io.NodeOutput`` wraps."""
    return tuple(getattr(node_output, "result", node_output))


def collect(value):
    """Run the collect node, returning its six outputs."""
    return _unwrap(Collect.execute(value))


def dispatch(tensor, **metadata):
    """Run the dispatch node.

    Only the sockets the caller names are passed, which is exactly what the
    engine does for unconnected optional inputs.
    """
    return _unwrap(Dispatch.execute(tensor=tensor, **metadata))


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one assertion result."""
    _RESULTS.append((name, bool(condition), detail))


def expect_raises(name: str, exc_type, call) -> None:
    """Record a result that passes only when ``call`` raises ``exc_type``."""
    try:
        call()
    except exc_type as exc:
        _RESULTS.append((name, True, str(exc)))
        return
    except Exception as exc:  # noqa: BLE001 - report the unexpected type
        _RESULTS.append((name, False, f"raised {type(exc).__name__}: {exc}"))
        return
    _RESULTS.append((name, False, "did not raise"))


# --------------------------------------------------------------------------- #
# Generic collect / dispatch
# --------------------------------------------------------------------------- #

def test_passthrough_tensors() -> None:
    """IMAGE / MASK / SIGMAS are plain tensors and must pass through untouched."""
    for label, tensor, slot in (
        ("IMAGE", torch.randn(1, 8, 8, 3), 0),
        ("MASK", torch.randn(8, 8), 1),
        ("SIGMAS", torch.randn(5), 4),
    ):
        sample, origin, _, _, _, _ = collect(tensor)
        check(f"{label}: collect returns the same tensor object", sample is tensor)
        check(f"{label} round trip: origin label is {label}", origin == label, f"got {origin!r}")

        outputs = dispatch(sample, origin=origin)
        check(
            f"{label} round trip: value survives on its own socket",
            outputs[slot] is tensor,
        )
        # A 1-D sigma tensor cannot be a 4-D latent or a 2-D/3-D audio waveform.
        if label != "IMAGE":
            check(f"{label} round trip: LATENT socket is None", outputs[2] is None)
        if label == "SIGMAS":
            check("SIGMAS round trip: AUDIO socket is None", outputs[3] is None)


def test_latent_full_roundtrip() -> None:
    """A fully populated LATENT must come back key for key."""
    original = {
        "samples": torch.randn(2, 4, 8, 8),
        "noise_mask": torch.randn(2, 1, 8, 8),
        "batch_index": [0, 3],
        "type": "hunyuan3dv2",
    }
    sample, origin, noise_mask, batch_index, latent_type, _ = collect(original)
    check("LATENT collect: samples is extracted", sample is original["samples"])
    check("LATENT collect: origin label is LATENT", origin == "LATENT", f"got {origin!r}")
    check("LATENT collect: noise_mask is forwarded", noise_mask is original["noise_mask"])
    check("LATENT collect: batch_index is forwarded", batch_index == [0, 3], f"got {batch_index!r}")
    check("LATENT collect: type is forwarded", latent_type == "hunyuan3dv2", f"got {latent_type!r}")

    restored = dispatch(
        sample,
        origin=origin,
        noise_mask=noise_mask,
        batch_index=batch_index,
        latent_type=latent_type,
    )[2]
    check("LATENT round trip: LATENT socket is populated", restored is not None)
    if restored is None:
        return
    check(
        "LATENT round trip: key set is identical",
        set(restored) == set(original),
        f"{sorted(restored)} vs {sorted(original)}",
    )
    check("LATENT round trip: samples is identical", restored.get("samples") is original["samples"])
    check(
        "LATENT round trip: noise_mask is identical",
        restored.get("noise_mask") is original["noise_mask"],
    )
    check("LATENT round trip: batch_index is equal", restored.get("batch_index") == [0, 3])
    check("LATENT round trip: type is equal", restored.get("type") == "hunyuan3dv2")


def test_latent_minimal_roundtrip() -> None:
    """Unconnected metadata must not add keys the original never had."""
    original = {"samples": torch.randn(1, 4, 8, 8)}
    sample, origin, noise_mask, batch_index, latent_type, _ = collect(original)
    check("LATENT minimal collect: no noise_mask", noise_mask is None)
    check("LATENT minimal collect: no batch_index", batch_index is None)
    check("LATENT minimal collect: empty type", latent_type == "", f"got {latent_type!r}")

    restored = dispatch(sample, origin=origin)[2]
    check("LATENT minimal round trip: only 'samples' survives", set(restored or {}) == {"samples"},
          f"got {sorted(restored or {})}")


def test_audio_roundtrip() -> None:
    """AUDIO carries its sample rate on a parallel socket."""
    original = {"waveform": torch.randn(1, 2, 1000), "sampler_rate": 32000}
    sample, origin, _, _, _, sampler_rate = collect(original)
    check("AUDIO collect: waveform is extracted", sample is original["waveform"])
    check("AUDIO collect: origin label is AUDIO", origin == "AUDIO", f"got {origin!r}")
    check("AUDIO collect: sampler_rate is forwarded", sampler_rate == 32000, f"got {sampler_rate!r}")

    restored = dispatch(sample, origin=origin, sampler_rate=sampler_rate)[3]
    check("AUDIO round trip: AUDIO socket is populated", restored is not None)
    check(
        "AUDIO round trip: waveform is identical",
        (restored or {}).get("waveform") is original["waveform"],
    )
    check(
        "AUDIO round trip: sampler_rate is equal",
        (restored or {}).get("sampler_rate") == 32000,
    )
    check(
        "AUDIO round trip: key set is identical",
        set(restored or {}) == set(original),
        f"got {sorted(restored or {})}",
    )


def test_audio_default_rate() -> None:
    """A missing or unusable sample rate falls back to the documented default."""
    for supplied in (None, 0, -1, "44100"):
        restored = dispatch(torch.randn(1, 2, 64), sampler_rate=supplied)[3]
        check(
            f"AUDIO fallback: sampler_rate={supplied!r} uses {conversion.DEFAULT_SAMPLER_RATE}",
            (restored or {}).get("sampler_rate") == conversion.DEFAULT_SAMPLER_RATE,
            f"got {(restored or {}).get('sampler_rate')!r}",
        )


def test_shape_guard_returns_none() -> None:
    """A tensor that cannot be a latent/audio yields ``None`` on those sockets."""
    # 1-D can be neither a 4-D latent nor an audio waveform ([C, T] / [B, C, T]).
    outputs = dispatch(torch.randn(5), origin="SIGMAS")
    check("shape guard: 1-D tensor is not a LATENT", outputs[2] is None)
    check("shape guard: 1-D tensor is not an AUDIO", outputs[3] is None)
    check("shape guard: IMAGE socket still passes through", outputs[0].shape == (5,))

    # 2-D is deliberately accepted as audio: a mask-shaped tensor and a
    # channel-major waveform are indistinguishable, so rejecting would misfire.
    outputs_2d = dispatch(torch.randn(3, 3), origin="MASK")
    check("shape guard: 2-D tensor is not a LATENT", outputs_2d[2] is None)
    check("shape guard: 2-D tensor is accepted as AUDIO", outputs_2d[3] is not None)


def test_unsupported_value_raises() -> None:
    """Unsupported inputs fail with a readable message instead of a shape error."""
    expect_raises("collect: unsupported type raises TypeError", TypeError, lambda: collect(42))
    expect_raises(
        "collect: unknown mapping raises TypeError",
        TypeError,
        lambda: collect({"nope": 1}),
    )


# --------------------------------------------------------------------------- #
# LORA_MODEL / LOSS_MAP pack / unpack (reserved)
# --------------------------------------------------------------------------- #

def test_lora_roundtrip() -> None:
    """Pack then unpack must restore keys, shapes, dtypes and values."""
    original = {
        "lora_A.weight": torch.randn(3, 4),
        "lora_B.weight": torch.randn(4),
        "scalar": torch.tensor(2.5),
    }
    flat, keys, shapes, dtypes = _unwrap(PackLora.execute(original))
    check("LORA pack: keys keep insertion order", keys == list(original), f"got {keys!r}")
    check("LORA pack: flat tensor holds every element",
          isinstance(flat, torch.Tensor) and flat.numel() == 3 * 4 + 4 + 1,
          f"got {getattr(flat, 'numel', lambda: '?')()}")
    check("LORA pack: 0-D scalar encodes as an empty shape", shapes[-1] == "", f"got {shapes[-1]!r}")

    restored = _unwrap(UnpackLora.execute(flat, keys, shapes, dtypes))[0]
    check("LORA round trip: key set is identical", set(restored) == set(original),
          f"got {sorted(restored)}")
    for key, value in original.items():
        got = restored.get(key)
        check(f"LORA round trip: {key} dtype matches", getattr(got, "dtype", None) == value.dtype)
        check(f"LORA round trip: {key} shape matches", getattr(got, "shape", None) == value.shape)
        check(f"LORA round trip: {key} values match", torch.equal(got, value))


def test_lora_mixed_dtype_refused() -> None:
    """Silently promoting mixed dtypes would make the round trip lossy."""
    expect_raises(
        "LORA pack: mixed dtypes raise ValueError",
        ValueError,
        lambda: PackLora.execute({"a": torch.randn(2), "b": torch.randn(2).double()}),
    )


def test_lora_empty_and_malformed() -> None:
    """Degenerate metadata must not abort a workflow."""
    flat, keys, shapes, dtypes = _unwrap(PackLora.execute({}))
    check("LORA pack: empty model yields an empty tensor", flat.numel() == 0)

    restored = _unwrap(UnpackLora.execute(torch.randn(6), ["bad"], ["not-a-shape"], ["x"]))[0]
    check(
        "LORA unpack: malformed shape degrades to a scalar",
        set(restored) == {"bad"} and restored["bad"].shape == (),
        f"got {restored!r}",
    )

    # A flat tensor that travelled through generic ops may come back
    # multi-dimensional; slicing it along axis 0 would mis-index the layout.
    reshaped = _unwrap(
        UnpackLora.execute(torch.randn(2, 3), ["a"], ["2,3"], ["torch.float32"])
    )[0]
    check(
        "LORA unpack: multi-dimensional input is flattened first",
        list(reshaped) == ["a"] and reshaped["a"].shape == (2, 3),
        f"got { {k: v.shape for k, v in reshaped.items()} }",
    )

    truncated = _unwrap(UnpackLora.execute(torch.randn(2), ["a", "b"], ["4", "4"], ["float32"]))[0]
    check("LORA unpack: truncated tensor keeps only what fits", len(truncated) == 0,
          f"got {sorted(truncated)}")


def test_loss_map_roundtrip() -> None:
    """The LOSS_MAP pair must restore an ordered list of tensors."""
    original = {"loss": [torch.randn(2, 3), torch.randn(5), torch.tensor(1.5)]}
    flat, shapes, dtypes = _unwrap(PackLoss.execute(original))
    restored = _unwrap(UnpackLoss.execute(flat, shapes, dtypes))[0]

    check("LOSS_MAP round trip: 'loss' key is preserved", set(restored) == {"loss"},
          f"got {sorted(restored)}")
    losses = restored.get("loss", [])
    check("LOSS_MAP round trip: list length matches", len(losses) == len(original["loss"]))
    for index, value in enumerate(original["loss"]):
        got = losses[index] if index < len(losses) else None
        check(f"LOSS_MAP round trip: loss[{index}] shape matches",
              getattr(got, "shape", None) == value.shape)
        check(f"LOSS_MAP round trip: loss[{index}] values match", torch.equal(got, value))


def test_loss_map_shapes() -> None:
    """A bare tensor and an explicit single-element list are both accepted."""
    for label, supplied in (("bare tensor", torch.randn(4)), ("single-item list", [torch.randn(4)])):
        flat, shapes, dtypes = _unwrap(PackLoss.execute(supplied))
        check(f"LOSS_MAP pack ({label}): one tensor is packed", len(shapes) == 1, f"got {shapes!r}")
        restored = _unwrap(UnpackLoss.execute(flat, shapes, dtypes))[0]
        check(f"LOSS_MAP pack ({label}): round trip keeps one loss", len(restored["loss"]) == 1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

_TESTS = (
    test_passthrough_tensors,
    test_latent_full_roundtrip,
    test_latent_minimal_roundtrip,
    test_audio_roundtrip,
    test_audio_default_rate,
    test_shape_guard_returns_none,
    test_unsupported_value_raises,
    test_lora_roundtrip,
    test_lora_mixed_dtype_refused,
    test_lora_empty_and_malformed,
    test_loss_map_roundtrip,
    test_loss_map_shapes,
)


def main() -> int:
    for test in _TESTS:
        try:
            test()
        except Exception:  # noqa: BLE001 - a crash is a failure of that test
            _RESULTS.append((test.__name__, False, traceback.format_exc(limit=3)))

    failures = [(name, detail) for name, ok, detail in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAIL  {name}")
            if detail:
                print(f"      {detail.strip().splitlines()[-1]}")

    passed = len(_RESULTS) - len(failures)
    print(f"\n=== conversion round trip: {passed} passed, {len(failures)} failed "
          f"({len(_TESTS)} test cases) ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
