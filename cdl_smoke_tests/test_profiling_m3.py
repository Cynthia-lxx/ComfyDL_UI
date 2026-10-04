"""Smoke tests for the M3 estimator-library expansion (family batches).

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_profiling_m3.py

Why this script exists
----------------------
M3 grows the two profiling ledgers from the 14 LM teaching nodes to the
ComfyDL node families and the built-in loaders. The file grows batch by
batch; this foundation layer pins the shared algebra every batch builds
on, so a wrong helper fails once here instead of quietly skewing ~85
estimators:

* conv output-size / parameter-count / FLOPs against hand-computed
  golden values (the LeNet first conv, the d2l RNN example, a 720p image);
* the torch floor rule for strided convolutions;
* the new EstValue carriers (ModuleVal / WeightsVal) and the tensor memory
  factory degrading unknown sizes to 0-byte lines instead of raising;
* the engine's compute ledger exposing the new "conv" kind.

Every number below is derived independently on paper, never by calling the
shipped formula under test.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy.profiling import estimate_workflow  # noqa: E402
from comfy.profiling import formulas  # noqa: E402
from comfy.profiling.shapes import ModuleVal, Unknown, WeightsVal  # noqa: E402


# ---------------------------------------------------------------------------
# F0: the shared helpers, pinned to hand-computed golden values.

def _foundation_checks() -> None:
    # torch's conv output rule, on paper:
    #   28x28 MNIST input, 5x5 kernel, no pad -> (28-5)+1 = 24
    check("F0a conv2d_out_dim: plain 5x5 on 28",
          formulas.conv2d_out_dim(28, 5) == 24, str(formulas.conv2d_out_dim(28, 5)))
    #   same conv with padding 2 keeps the size (LeNet's second block does
    #   this on 14x14)
    check("F0b conv2d_out_dim: padding 2 preserves size",
          formulas.conv2d_out_dim(14, 5, 1, 2) == 14, str(formulas.conv2d_out_dim(14, 5, 1, 2)))
    #   stride 2: floor((34 - 5) / 2) + 1 = 15 (floor, like torch)
    check("F0c conv2d_out_dim: stride 2 floors",
          formulas.conv2d_out_dim(34, 5, 2) == 15, str(formulas.conv2d_out_dim(34, 5, 2)))

    # LeNet's first conv on paper: 6 out-channels, 1 in-channel, 5x5, bias
    #   -> 6 * 1 * 25 + 6 = 156 parameters
    check("F0d conv_param_count: LeNet conv1",
          formulas.conv_param_count(1, 6, 5, 5) == 156, str(formulas.conv_param_count(1, 6, 5, 5)))
    #   bias-free variant: 150
    check("F0e conv_param_count: no bias",
          formulas.conv_param_count(1, 6, 5, 5, bias=False) == 150)
    #   grouped: in 4, out 6, k 3x3, groups 2 -> 6 * 2 * 9 + 6 = 114
    check("F0f conv_param_count: groups divide",
          formulas.conv_param_count(4, 6, 3, 3, groups=2) == 114,
          str(formulas.conv_param_count(4, 6, 3, 3, groups=2)))

    # The MAC count of that first LeNet conv over one 24x24 output:
    #   2 * 1 * 24 * 24 * 6 * 1 * 5 * 5 = 172,800
    check("F0g conv2d_flops: LeNet conv1 forward",
          formulas.conv2d_flops(1, 24, 24, 6, 1, 5, 5) == 172_800,
          str(formulas.conv2d_flops(1, 24, 24, 6, 1, 5, 5)))
    #   grouped variant keeps the /groups on the input side only
    check("F0h conv2d_flops: groups halve the inner product",
          formulas.conv2d_flops(1, 8, 8, 6, 4, 3, 3, groups=2)
          == 2 * 1 * 8 * 8 * 6 * 2 * 9)

    # A 720p RGB image, NHWC float32: 720 * 1280 * 3 * 4 = 11,059,200 bytes
    check("F0i image_bytes: 720p RGB",
          formulas.image_bytes(1, 720, 1280, 3) == 11_059_200,
          str(formulas.image_bytes(1, 720, 1280, 3)))
    #   unknown dims propagate as None, never as a guess
    check("F0j image_bytes: unknown width stays unknown",
          formulas.image_bytes(1, 720, None, 3) is None)

    # The d2l RNN example (sec 8): inputs 4, hiddens 256, both biases:
    #   4*256 + 256*256 + 2*256 = 67,072
    check("F0k rnn_param_count: d2l RNN example",
          formulas.rnn_param_count(4, 256) == 67_072, str(formulas.rnn_param_count(4, 256)))
    #   GRU: 3 gates -> 3 * (4*256 + 256*256) + 3 * 2 * 256 = 201,216
    check("F0l rnn_param_count: GRU (3 gates)",
          formulas.rnn_param_count(4, 256, gates=3) == 201_216,
          str(formulas.rnn_param_count(4, 256, gates=3)))
    check("F0m gru_param_count: matches gates=3",
          formulas.gru_param_count(4, 256) == formulas.rnn_param_count(4, 256, gates=3))
    #   stacked layers: layer 0 maps I -> H, layer 1+ maps H -> H, so two
    #   layers are NOT simply twice layer 0
    #     layer0  = 512*256 + 256*256 + 2*256          = 197,120
    #     layer1  = 256*256 + 256*256 + 2*256          = 131,584
    check("F0n rnn_param_count: torch's H->H deep layers",
          formulas.rnn_param_count(512, 256, num_layers=2) == 197_120 + 131_584,
          str(formulas.rnn_param_count(512, 256, num_layers=2)))

    # The one-tensor memory factory: bytes == single_bytes, None -> 0
    item = formulas.tensor_item("x", 4000, kind="inputs")
    check("F0o tensor_item: single_bytes equals bytes",
          item.bytes == 4000 and item.single_bytes == 4000 and item.kind == "inputs")
    empty = formulas.tensor_item("x", None, kind="misc")
    check("F0p tensor_item: unknown size degrades to 0", empty.bytes == 0)

    # The new EstValue carriers
    module = ModuleVal(param_count=201_216, kind_hint="rnn")
    check("F0q ModuleVal: describe carries the count",
          "201,216" in module.describe() and "rnn" in module.describe(), module.describe())
    missing = WeightsVal(file_bytes=-1, source_path="checkpoints/sd.safetensors")
    check("F0r WeightsVal: not-found describe is honest",
          "not found" in missing.describe(), missing.describe())
    loaded = WeightsVal(file_bytes=2_000_000_000, source_path="x")
    check("F0s WeightsVal: size in describe", "2,000,000,000" in loaded.describe())
    frozen_ok = False
    try:
        module.param_count = 1  # type: ignore[misc]
    except Exception:
        frozen_ok = True
    check("F0t ModuleVal: frozen dataclass", frozen_ok)
    check("F0u Unknown still importable alongside", isinstance(Unknown("x"), Unknown))

    # Module-family helpers, hand-computed from the d2l source:
    #   d2l MultiHeadAttention = four LazyLinear(H), no bias by default
    check("F0x mha_param_count: four projections, head count irrelevant",
          formulas.mha_param_count(8, 8) == 256 and formulas.mha_param_count(8, 8, bias=True) == 288,
          str(formulas.mha_param_count(8, 8)))
    #   one encoder block = 4H^2 (attn) + 2HF+F+H (ffn) + 4H (two norms)
    check("F0y transformer_block_param_count: 8/64 block = 1384",
          formulas.transformer_block_param_count(8, 64) == 1384,
          str(formulas.transformer_block_param_count(8, 64)))
    #   the positional table is a plain tensor, not a parameter
    check("F0z positional_encoding_bytes: 1000x16 float32 = 64,000",
          formulas.positional_encoding_bytes(1000, 16) == 64_000)
    #   a lazy layer has no weights yet -> None, never a guess
    check("F0za dense_param_count: unknown width stays unknown",
          formulas.dense_param_count(None, 32) is None
          and formulas.dense_param_count(16, 32) == 16 * 32 + 32)


def _engine_checks() -> None:
    report = estimate_workflow({})
    kinds = report["flops"]["by_kind"]
    check("F0v engine: by_kind exposes the conv bucket",
          "conv" in kinds and kinds["conv"] == 0, str(kinds))
    check("F0w engine: empty graph stays unknown-verdict",
          report["verdict"] == "unknown", str(report.get("verdict")))


# ---------------------------------------------------------------------------
# F1: batch 1 - the tensor / detection / segmentation / image estimators.
# Every expectation below was derived on paper from the node sources.

def _batch1_checks() -> None:
    # -- conv chain: random 28x28 input, 5x5 literal kernel, stride 1, pad 0
    prompt = {
        "rand": {"class_type": "CdlRandomTensor",
                 "inputs": {"shape": "1,1,28,28", "dist": "normal", "mean": 0.0,
                            "std": 1.0, "low": 0.0, "high": 1.0, "seed": -1}},
        # LeNet conv1: 6 output channels, 1 input channel, 5x5 kernel
        "k": {"class_type": "CdlRandomTensor", "inputs": {"shape": "6,1,5,5"}},
        "conv": {"class_type": "CdlConv2d",
                 "inputs": {"input_tensor": ["rand", 0], "kernel": ["k", 0],
                            "stride": 1, "padding": 0}},
    }
    report = estimate_workflow(prompt)
    conv = _node(report, "CdlConv2d")
    check("F1a conv2d: LeNet conv1 -> (6, 24, 24) = 13,824 bytes",
          conv["status"] == "estimated" and conv["total_bytes"] == 6 * 24 * 24 * 4,
          f"{conv['status']} {conv['total_bytes']}")
    check("F1b conv2d: conv-kind FLOPs 2*B*H*W*Co*Ci*k*k",
          conv["flops_status"] == "estimated" and conv["flops_total"] == 172_800,
          f"{conv['flops_status']} {conv['flops_total']}")
    check("F1c engine: conv FLOPs land in the conv bucket",
          report["flops"]["by_kind"]["conv"] >= 172_800, str(report["flops"]["by_kind"]))

    # -- d2l corr2d: 8x8 input, 2x2 kernel -> (7, 7), 392 FLOPs
    prompt = {
        "x": {"class_type": "CdlRandomTensor", "inputs": {"shape": "8,8"}},
        "k": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,2"}},
        "corr": {"class_type": "CdlCorr2d",
                 "inputs": {"input_tensor": ["x", 0], "kernel": ["k", 0]}},
    }
    corr = _node(estimate_workflow(prompt), "CdlCorr2d")
    check("F1d corr2d: (8,8) x (2,2) -> (7,7), 392 MACs",
          corr["status"] == "estimated" and corr["flops_total"] == 392
          and corr["total_bytes"] == 7 * 7 * 4,
          f"{corr['status']} {corr['flops_total']} {corr['total_bytes']}")

    # -- literal parsing and shape algebra
    lit = _node(estimate_workflow({"n": {"class_type": "CdlStrToTensor",
             "inputs": {"text": "[[1, 2, 3], [4, 5, 6]]", "error_strategy": "empty_tensor"}}}),
        "CdlStrToTensor")
    check("F1e str->tensor: nested literal -> (2, 3)", lit["total_bytes"] == 24,
          str(lit["total_bytes"]))
    bad = _node(estimate_workflow({"n": {"class_type": "CdlStrToTensor",
             "inputs": {"text": "not a literal", "error_strategy": "empty_tensor"}}}),
        "CdlStrToTensor")
    check("F1f str->tensor: unparseable + empty strategy -> (0,)",
          bad["status"] == "estimated" and bad["total_bytes"] == 0, str(bad["total_bytes"]))

    shape_prompt = {
        "x": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,3"}},
        "t": {"class_type": "CdlTranspose",
              "inputs": {"tensor": ["x", 0], "dim0": 0, "dim1": 1}},
        "r": {"class_type": "CdlReshape",
              "inputs": {"tensor": ["x", 0], "target_shape": "3,2"}},
        "inf": {"class_type": "CdlReshape",
                "inputs": {"tensor": ["x", 0], "target_shape": "-1"}},
    }
    report = estimate_workflow(shape_prompt)
    check("F1g transpose: (2,3) -> (3,2)",
          _node(report, "CdlTranspose")["status"] == "estimated")
    check("F1h reshape: 6 elements -> (3,2)",
          _node(report, "CdlReshape")["status"] == "estimated")

    # -- synthetic data, linreg, truncate/pad
    prompt = {
        "sd": {"class_type": "CdlSyntheticData",
               "inputs": {"num_features": 2, "num_examples": 100, "noise_std": 0.01, "seed": 0}},
        "w": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,1"}},
        "b": {"class_type": "CdlRandomTensor", "inputs": {"shape": "1"}},
        "lr": {"class_type": "CdlLinReg",
               "inputs": {"X": ["sd", 0], "w": ["w", 0], "b": ["b", 0]}},
    }
    lr = _node(estimate_workflow(prompt), "CdlLinReg")
    check("F1i linreg: (100,2)@(2,1) -> 400 FLOPs (ffn kind)",
          lr["flops_status"] == "estimated" and lr["flops_total"] == 400
          and lr["flops_items"][0]["kind"] == "ffn",
          f"{lr['flops_status']} {lr['flops_total']}")
    tp = _node(estimate_workflow({"n": {"class_type": "CdlTruncatePad",
            "inputs": {"num_steps": 64, "padding_token": 0}}}), "CdlTruncatePad")
    check("F1j truncate/pad: (64,) int64 = 512 bytes", tp["total_bytes"] == 512,
          str(tp["total_bytes"]))

    # -- detection family: default 561x728 map -> 2,042,040 anchors
    prompt = {
        "img": {"class_type": "CdlRandomTensor", "inputs": {"shape": "1,3,561,728"}},
        "prior": {"class_type": "CdlMultiboxPrior",
                  "inputs": {"sizes": "0.75,0.5,0.25", "ratios": "1,2,0.5",
                             "data": ["img", 0]}},
        "labels": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,3,5"}},
        "target": {"class_type": "CdlMultiboxTarget",
                   "inputs": {"anchors": ["prior", 0], "labels": ["labels", 0]}},
        "cls": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,3,10"}},
        "det": {"class_type": "CdlMultiboxDetection",
                "inputs": {"cls_probs": ["cls", 0], "offset_preds": ["cls", 0],
                           "anchors": ["prior", 0], "nms_threshold": 0.5,
                           "pos_threshold": 0.01}},
        "nms": {"class_type": "CdlNms",
                "inputs": {"boxes": ["cls", 0], "scores": ["cls", 0], "iou_threshold": 0.5}},
    }
    report = estimate_workflow(prompt)
    prior = _node(report, "CdlMultiboxPrior")
    check("F1k multibox prior: 561x728 map, 5/pixel -> 32,672,640 bytes",
          prior["total_bytes"] == 2_042_040 * 4 * 4, str(prior["total_bytes"]))
    check("F1l multibox prior: exact confidence with a linked feature map",
          prior["confidence"] == "exact", str(prior["confidence"]))
    target = _node(report, "CdlMultiboxTarget")
    check("F1m multibox target: (B, 4A) + (B, 4A) + (B, A) outputs",
          target["status"] == "estimated" and len(target["items"]) == 3
          and target["items"][0]["bytes"] == 2 * 4 * 2_042_040 * 4,
          str(target["items"]))
    det = _node(report, "CdlMultiboxDetection")
    check("F1n multibox detection: (2, 10, 6) = 480 bytes",
          det["total_bytes"] == 2 * 10 * 6 * 4, str(det["total_bytes"]))
    nms = _node(report, "CdlNms")
    check("F1o nms: approx, kept count honestly unknown",
          nms["status"] == "estimated" and nms["confidence"] == "approx"
          and nms["reason"] != "", str(nms["reason"]))

    # -- VOC family
    prompt = {
        "cm": {"class_type": "CdlVocColormap2Label", "inputs": {}},
        "feat": {"class_type": "CdlRandomTensor", "inputs": {"shape": "1,600,800,3"}},
        "crop": {"class_type": "CdlVocRandCrop",
                 "inputs": {"feature": ["feat", 0], "label": ["feat", 0],
                            "height": 320, "width": 480, "seed": 0}},
    }
    report = estimate_workflow(prompt)
    cm = _node(report, "CdlVocColormap2Label")
    check("F1p VOC colormap: 256^3 int64 = 134,217,728 bytes",
          cm["total_bytes"] == 256 ** 3 * 8, str(cm["total_bytes"]))
    crop = _node(report, "CdlVocRandCrop")
    check("F1q VOC crop: two (1,320,480,3) outputs = 3,686,400 bytes",
          crop["total_bytes"] == 2 * 320 * 480 * 3 * 4, str(crop["total_bytes"]))

    # -- built-in resizers: replicate their own rounding on paper
    prompt = {
        "img": {"class_type": "CdlRandomTensor", "inputs": {"shape": "2,1024,768,3"}},
        "mp": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["img", 0], "upscale_method": "bilinear",
                          "megapixels": 1.0, "resolution_steps": 1}},
        "max": {"class_type": "ImageScaleToMaxDimension",
                "inputs": {"image": ["img", 0], "upscale_method": "area",
                           "largest_size": 512}},
    }
    report = estimate_workflow(prompt)
    mp = _node(report, "ImageScaleToTotalPixels")
    import math as _math
    scale = _math.sqrt(1.0 * 1024 * 1024 / (768 * 1024))
    check("F1r scale-to-pixels: 1024x768 @ 1MP -> 1182x887",
          mp["status"] == "estimated"
          and mp["total_bytes"] == 2 * int(round(1024 * scale)) * int(round(768 * scale)) * 3 * 4,
          str(mp["total_bytes"]))
    mx = _node(report, "ImageScaleToMaxDimension")
    check("F1s scale-to-max-dim: 1024x768 -> 512x384",
          mx["status"] == "estimated"
          and mx["total_bytes"] == 2 * 512 * int(round(768 / 1024 * 512)) * 3 * 4,
          str(mx["total_bytes"]))

    prompt = {
        "img": {"class_type": "CdlRandomTensor", "inputs": {"shape": "1,64,64,3"}},
        "half": {"class_type": "ResizeImageMaskNode",
                 "inputs": {"input": ["img", 0], "resize_type": "scale by multiplier",
                            "multiplier": 0.5, "scale_method": "area"}},
        "mult": {"class_type": "ResizeImageMaskNode",
                 "inputs": {"input": ["img", 0], "resize_type": "scale to multiple",
                            "multiple": 16, "scale_method": "area"}},
    }
    report = estimate_workflow(prompt)
    half = _node(report, "ResizeImageMaskNode")
    check("F1t resize: multiplier 0.5 -> (1, 32, 32, 3)",
          half["status"] == "estimated" and half["total_bytes"] == 32 * 32 * 3 * 4,
          str(half["total_bytes"]))
    mult = [n for n in report["nodes"] if n["class_type"] == "ResizeImageMaskNode"
            and n["id"] == "mult"][0]
    check("F1u resize: multiple 16 -> (1, 64, 64, 3) unchanged",
          mult["total_bytes"] == 64 * 64 * 3 * 4, str(mult["total_bytes"]))

    # -- image loaders: honest approx without a real file
    prompt = {"load": {"class_type": "LoadImage", "inputs": {"image": "no_such_file.png"}}}
    load = _node(estimate_workflow(prompt), "LoadImage")
    check("F1v LoadImage: estimated-approx, size honestly deferred to the decode",
          load["status"] == "estimated" and load["confidence"] == "approx"
          and load["total_bytes"] == 0 and load["reason"] != "", str(load["reason"]))

    # -- unknown propagation across the new family: grayscale after LoadImage
    prompt = {
        "load": {"class_type": "LoadImage", "inputs": {"image": "no_such_file.png"}},
        "gray": {"class_type": "CdlImageGrayscale", "inputs": {"image": ["load", 0]}},
    }
    gray = _node(estimate_workflow(prompt), "CdlImageGrayscale")
    check("F1w grayscale: passes the unknown spatial dims through",
          gray["status"] == "estimated" and gray["total_bytes"] == 0,
          f"{gray['status']} {gray['total_bytes']}")

    # -- flops honesty: pure-python and pass-through nodes stay zero/unknown
    bleu = _node(estimate_workflow({"n": {"class_type": "CdlBleu",
            "inputs": {"pred_seq": "a b", "label_seq": "a b c", "max_n": 4}}}), "CdlBleu")
    check("F1x bleu: pure-Python loop -> flops unknown",
          bleu["flops_status"] == "unknown" and bleu["status"] == "estimated",
          bleu["flops_status"])
    acc = _node(estimate_workflow({
        "sd": {"class_type": "CdlSyntheticData", "inputs": {"num_features": 2,
                                                           "num_examples": 100}},
        "acc": {"class_type": "CdlAccuracy",
                "inputs": {"y_hat": ["sd", 0], "y": ["sd", 1]}}}), "CdlAccuracy")
    check("F1y accuracy: zero flops, count derivable", acc["flops_status"] == "zero",
          acc["flops_status"])


def _batch2_checks() -> None:
    # -- d2l RNN family, defaults 32 inputs / 64 hiddens
    prompt = {
        "scratch": {"class_type": "CdlRNNScratch",
                    "inputs": {"num_inputs": 32, "num_hiddens": 64, "sigma": 0.01}},
        "rnn": {"class_type": "CdlRNN", "inputs": {"num_inputs": 32, "num_hiddens": 64}},
        "gru": {"class_type": "CdlGRU",
                "inputs": {"num_inputs": 32, "num_hiddens": 64, "num_layers": 1,
                           "dropout": 0.0}},
        "gru2": {"class_type": "CdlGRU",
                 "inputs": {"num_inputs": 32, "num_hiddens": 64, "num_layers": 3,
                            "dropout": 0.0}},
    }
    report = estimate_workflow(prompt)
    # on paper: W_xh (32x64) + W_hh (64x64) + b_h (64) = 2048 + 4096 + 64
    scratch = _node(report, "CdlRNNScratch")
    check("F2a RNNScratch: I*H + H^2 + H = 6208 params",
          scratch["total_bytes"] == 6208 * 4, str(scratch["total_bytes"]))
    # torch nn.RNN: weight_ih (64,32) + weight_hh (64,64) + two biases of 64
    rnn = _node(report, "CdlRNN")
    check("F2b nn.RNN: H*(I+H+2) = 6272 params",
          rnn["total_bytes"] == 6272 * 4, str(rnn["total_bytes"]))
    # torch nn.GRU: 3 gates -> every weight triple-width
    gru = _by_id(report, "gru")
    check("F2c nn.GRU (1 layer): 3H*(I+H+2) = 18816 params",
          gru["total_bytes"] == 18816 * 4, str(gru["total_bytes"]))
    gru2 = _by_id(report, "gru2")
    check("F2d nn.GRU (3 layers): deep layers take H inputs, not I",
          gru2["total_bytes"] == (18816 + 2 * (3 * (64 * 64 + 64 * 64) + 6 * 64)) * 4,
          str(gru2["total_bytes"]))

    # -- language models built on top of an RNN: head (H, V) + bias (V)
    prompt = {
        "scratch": {"class_type": "CdlRNNScratch",
                    "inputs": {"num_inputs": 32, "num_hiddens": 64, "sigma": 0.01}},
        "rnn": {"class_type": "CdlRNN", "inputs": {"num_inputs": 32, "num_hiddens": 64}},
        "lm": {"class_type": "CdlRNNLMScratch",
               "inputs": {"rnn": ["scratch", 0], "vocab_size": 32, "lr": 0.01}},
        "lm2": {"class_type": "CdlRNNLM",
                "inputs": {"rnn": ["rnn", 0], "vocab_size": 32, "lr": 0.01}},
        "bad": {"class_type": "CdlRNNLMScratch",
                "inputs": {"rnn": ["rnn", 0], "vocab_size": 32, "lr": 0.01}},
    }
    report = estimate_workflow(prompt)
    lm = _node(report, "CdlRNNLMScratch")
    check("F2e RNNLMScratch: rnn + H*V + V = 8288 params",
          lm["total_bytes"] == 8288 * 4, str(lm["total_bytes"]))
    lm2 = _by_id(report, "lm2")
    check("F2f RNNLM: lazy head materialises to 6272 + 64*32 + 32",
          lm2["total_bytes"] == (6272 + 2048 + 32) * 4
          and lm2["confidence"] == "approx", str(lm2["total_bytes"]))
    bad = _by_id(report, "bad")
    check("F2g RNNLMScratch: refuses a non-scratch rnn instead of guessing",
          bad["status"] == "unknown" and "scratch" in bad["reason"], bad["reason"])

    # -- seq2seq encoder: embedding (32,16) + GRU 16->16 x2 layers
    enc = _node(estimate_workflow({"n": {"class_type": "CdlSeq2SeqEncoder",
            "inputs": {"vocab_size": 32, "embed_size": 16, "num_hiddens": 16,
                       "num_layers": 2, "dropout": 0.0}}}), "CdlSeq2SeqEncoder")
    check("F2h seq2seq encoder: V*E + GRU stack = 3776 params",
          enc["total_bytes"] == 3776 * 4, str(enc["total_bytes"]))

    # -- attention family
    prompt = {
        "dot": {"class_type": "CdlDotProductAttention", "inputs": {"dropout": 0.0}},
        "add": {"class_type": "CdlAdditiveAttention",
                "inputs": {"num_hiddens": 8, "dropout": 0.0}},
        "mha": {"class_type": "CdlMultiHeadAttention",
                "inputs": {"num_hiddens": 8, "num_heads": 4, "dropout": 0.0,
                           "use_bias": False}},
        "mha_bad": {"class_type": "CdlMultiHeadAttention",
                    "inputs": {"num_hiddens": 8, "num_heads": 3, "dropout": 0.0,
                               "use_bias": False}},
        "pe": {"class_type": "CdlPositionalEncoding",
               "inputs": {"num_hiddens": 16, "dropout": 0.0, "max_len": 1000}},
        "ffn": {"class_type": "CdlPositionWiseFFN",
                "inputs": {"ffn_num_hiddens": 64, "ffn_num_outputs": 16}},
        "blk": {"class_type": "CdlTransformerEncoderBlock",
                "inputs": {"num_hiddens": 8, "ffn_num_hiddens": 64, "num_heads": 4,
                           "dropout": 0.0, "use_bias": False}},
        "enc": {"class_type": "CdlTransformerEncoder",
                "inputs": {"vocab_size": 32, "num_hiddens": 8, "ffn_num_hiddens": 64,
                           "num_heads": 4, "num_blks": 2, "dropout": 0.0,
                           "use_bias": False}},
    }
    report = estimate_workflow(prompt)
    check("F2i attention (dot-product): zero weights, still estimated",
          _node(report, "CdlDotProductAttention")["status"] == "estimated")
    check("F2j attention (additive): H*(2H+1) = 136 params, approx",
          _node(report, "CdlAdditiveAttention")["total_bytes"] == 136 * 4
          and _node(report, "CdlAdditiveAttention")["confidence"] == "approx",
          str(_node(report, "CdlAdditiveAttention")["total_bytes"]))
    check("F2k attention (multi-head): four projections = 256 params",
          _by_id(report, "mha")["total_bytes"] == 256 * 4,
          str(_by_id(report, "mha")["total_bytes"]))
    check("F2l attention (multi-head): indivisible heads -> unknown",
          _by_id(report, "mha_bad")["status"] == "unknown", str(_by_id(report, "mha_bad")["reason"]))
    check("F2m positional encoding: 0 params but a 64,000 B table",
          _node(report, "CdlPositionalEncoding")["total_bytes"] == 64_000,
          str(_node(report, "CdlPositionalEncoding")["total_bytes"]))
    ffn = _node(report, "CdlPositionWiseFFN")
    check("F2n position-wise FFN: lazy dense1 -> size honestly unknown",
          ffn["status"] == "estimated" and ffn["confidence"] == "approx"
          and ffn["total_bytes"] == 0, ffn["reason"])
    check("F2o transformer block: 4H^2 + 2HF + F + 5H = 1384 params",
          _node(report, "CdlTransformerEncoderBlock")["total_bytes"] == 1384 * 4,
          str(_node(report, "CdlTransformerEncoderBlock")["total_bytes"]))
    enc = _node(report, "CdlTransformerEncoder")
    check("F2p transformer encoder: V*H + 2 blocks + table = 3024 params + 32,000 B",
          enc["total_bytes"] == 3024 * 4 + 32_000, str(enc["total_bytes"]))

    # -- CV constructors
    prompt = {
        "lenet": {"class_type": "CdlLeNet", "inputs": {"num_classes": 10, "lr": 0.1}},
        "res": {"class_type": "CdlResNet18",
                "inputs": {"num_classes": 10, "in_channels": 1}},
        "blk": {"class_type": "CdlResidual",
                "inputs": {"num_channels": 64, "use_1x1conv": False, "strides": 1}},
    }
    report = estimate_workflow(prompt)
    # conv1 156 + conv2 2416 + 400*120+120 + 120*84+84 + 84*10+10 = 61,706
    lenet = _node(report, "CdlLeNet")
    check("F2x LeNet: 61,706 params for the d2l MNIST shape",
          lenet["total_bytes"] == 61_706 * 4 and lenet["confidence"] == "approx",
          str(lenet["total_bytes"]))
    # stem 768 + 148,224 + 525,952 + 2,100,480 + 8,395,264 + fc 5,130
    res = _node(report, "CdlResNet18")
    check("F2y ResNet-18: four stages summed = 11,175,818 params",
          res["total_bytes"] == 11_175_818 * 4 and res["confidence"] == "exact",
          str(res["total_bytes"]))
    check("F2z residual block: lazy sizes stay unknown, no invention",
          _node(report, "CdlResidual")["status"] == "estimated"
          and _node(report, "CdlResidual")["total_bytes"] == 0,
          _node(report, "CdlResidual")["reason"])

    # -- model utils: clone doubles the resident weights
    prompt = {
        "rnn": {"class_type": "CdlRNNScratch",
                "inputs": {"num_inputs": 32, "num_hiddens": 64, "sigma": 0.01}},
        "clone": {"class_type": "CdlModelClone", "inputs": {"model": ["rnn", 0]}},
        "info": {"class_type": "CdlModelInfo", "inputs": {"model": ["rnn", 0]}},
        "mode": {"class_type": "CdlModelMode", "inputs": {"model": ["rnn", 0],
                                                          "mode": "eval"}},
    }
    report = estimate_workflow(prompt)
    check("F2q ModelClone: a second full copy of the weights",
          _node(report, "CdlModelClone")["total_bytes"] == 6208 * 4,
          str(_node(report, "CdlModelClone")["total_bytes"]))
    check("F2r ModelInfo: reports the same parameter count",
          _node(report, "CdlModelInfo")["status"] == "estimated")

    # -- nlp_utils vocab family
    prompt = {
        "vb": {"class_type": "CdlVocabBuild",
               "inputs": {"tokens_text": "the quick\nbrown fox\nthe lazy dog",
                          "min_freq": 1, "reserved_tokens": "<pad>,<bos>,<eos>"}},
        "enc": {"class_type": "CdlVocabEncode",
                "inputs": {"vocab": ["vb", 0], "tokens": "the,quick,brown"}},
    }
    report = estimate_workflow(prompt)
    vb = _node(report, "CdlVocabBuild")
    # 6 distinct tokens + <unk> + 3 reserved
    check("F2s VocabBuild: <unk> + 3 reserved + 6 distinct = 10",
          vb["status"] == "estimated"
          and vb["basis"]["params"]["size"] == 10, str(vb["basis"]))
    enc_node = _node(report, "CdlVocabEncode")
    check("F2t VocabEncode: 3 tokens -> (3,) int64 = 24 bytes",
          enc_node["total_bytes"] == 24, str(enc_node["total_bytes"]))

    tok = _node(estimate_workflow({"n": {"class_type": "CdlTokenize",
            "inputs": {"text": "the quick brown fox\njumps over", "token_mode": "word"}}}),
        "CdlTokenize")
    check("F2u Tokenize (word): 4 + 2 = 6 tokens",
          tok["basis"]["params"]["tokens"] == 6, str(tok["basis"]))
    seg = _node(estimate_workflow({"n": {"class_type": "CdlGetTokensAndSegments",
            "inputs": {"tokens_a": "a,b,c", "tokens_b": "d,e"}}}), "CdlGetTokensAndSegments")
    check("F2v segments: <cls> + 3 + <sep> + 2 + <sep> = 8 tokens",
          seg["basis"]["params"]["tokens"] == 8, str(seg["basis"]))

    # -- the predict loop: one forward per token, no cache
    prompt = {
        "vb": {"class_type": "CdlVocabBuild",
               "inputs": {"tokens_text": "a b c", "min_freq": 1,
                          "reserved_tokens": "<pad>,<bos>,<eos>"}},
        "rnn": {"class_type": "CdlRNN", "inputs": {"num_inputs": 32, "num_hiddens": 64}},
        "lm": {"class_type": "CdlRNNLM",
               "inputs": {"rnn": ["rnn", 0], "vocab_size": 32, "lr": 0.01}},
        "gen": {"class_type": "CdlRNNLMScratchPredict",
                "inputs": {"model": ["lm", 0], "vocab": ["vb", 0], "prefix": "abc",
                           "num_preds": 10}},
    }
    gen = _node(estimate_workflow(prompt), "CdlRNNLMScratchPredict")
    # 13 steps: 6272 rnn params + 2080 head params, 2 FLOPs each
    check("F2w predict: 2 x params x (prefix+num_preds) = 217,152 FLOPs",
          gen["flops_status"] == "estimated"
          and gen["flops_total"] == 2 * 8352 * 13,
          f"{gen['flops_status']} {gen['flops_total']}")


def _batch3_checks() -> None:
    import tempfile

    # -- loaders: os.stat a real file, then delete it again
    staging = Path(REPO_ROOT) / "models" / "checkpoints"
    staging.mkdir(parents=True, exist_ok=True)
    fixture = staging / "cdl_m3_fixture.safetensors"
    fixture.write_bytes(b"\0" * 4096)
    try:
        prompt = {
            "ckpt": {"class_type": "CheckpointLoaderSimple",
                     "inputs": {"ckpt_name": "cdl_m3_fixture.safetensors",
                                "prefix_strip": "auto"}},
            "miss": {"class_type": "UNETLoader",
                     "inputs": {"unet_name": "no_such_model.safetensors",
                                "prefix_strip": "auto"}},
            "save": {"class_type": "ModelSave",
                     "inputs": {"model": ["ckpt", 0],
                                "filename_prefix": "comfydl/diffusion_models"}},
        }
        report = estimate_workflow(prompt)
        ckpt = _node(report, "CheckpointLoaderSimple")
        check("F3a loader: file size is the resident weight cost",
              ckpt["status"] == "estimated" and ckpt["total_bytes"] == 4096,
              f"{ckpt['status']} {ckpt['total_bytes']}")
        check("F3b loader: secondary buckets stay unknown (prefix split)",
              len(ckpt["items"]) == 1 and ckpt["confidence"] == "approx",
              str(ckpt["items"]))
        missing = _node(report, "UNETLoader")
        check("F3c loader: a missing file is unknown, never a guess",
              missing["status"] == "unknown" and missing["total_bytes"] == 0,
              missing["reason"])
        saved = _node(report, "ModelSave")
        check("F3d ModelSave: stages the linked weight bytes for the write",
              saved["total_bytes"] == 4096, str(saved["total_bytes"]))
    finally:
        try:
            fixture.unlink()
        except OSError:
            pass

    # -- VAE: comfy/sd.py's own formulas, 8x and 4 channels by assumption
    prompt = {
        "img": {"class_type": "EmptyImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1, "color": 0}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["img", 0],
                                                      "vae": ["ckpt", 2]}},
        "dec": {"class_type": "VAEDecode",
                "inputs": {"samples": ["enc", 0], "vae": ["ckpt", 2]}},
    }
    report = estimate_workflow(prompt)
    enc = _node(report, "VAEEncode")
    # 512x512 -> (1, 4, 64, 64) latent + 1767*512*512*4 working bytes
    check("F3e VAEEncode: /8 downscale and 4 latent channels by assumption",
          enc["status"] == "estimated"
          and enc["items"][0]["bytes"] == 1 * 4 * 64 * 64 * 4
          and enc["items"][1]["bytes"] == 1767 * 512 * 512 * 4,
          str(enc["items"]))
    check("F3f VAEEncode: the assumption is recorded in the report",
          any(entry["key"] == "latent_channels" for entry in report["assumptions_used"]),
          str(report["assumptions_used"]))
    dec = _node(report, "VAEDecode")
    # 64x64 latent -> 512x512x3 image + 2178*64*64*64*4 working bytes
    check("F3g VAEDecode: 8x upscale and comfy/sd.py's decoder workspace",
          dec["status"] == "estimated"
          and dec["items"][0]["bytes"] == 1 * 512 * 512 * 3 * 4
          and dec["items"][1]["bytes"] == 2178 * 64 * 64 * 64 * 4,
          str(dec["items"]))

    # -- CLIP: pure assumption, surfaced as such
    clip = _node(estimate_workflow({"n": {"class_type": "CLIPTextEncode",
            "inputs": {"text": "a photo of a cat", "clip": ["na", 0]}}}),
        "CLIPTextEncode")
    check("F3h CLIPTextEncode: 77x768 conditioning, labelled an assumption",
          clip["status"] == "estimated" and clip["confidence"] == "approx"
          and clip["total_bytes"] == 77 * 768 * 4,
          f"{clip['status']} {clip['total_bytes']}")
    check("F3i CLIPTextEncode: the hidden width is recorded as assumed",
          any(entry["key"] == "clip_hidden" for entry in
              estimate_workflow({"n": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "x", "clip": ["na", 0]}}})["assumptions_used"]))

    # -- LoRA: this build's node raises, so the estimator refuses to invent
    lora = _node(estimate_workflow({"n": {"class_type": "LoraLoader",
            "inputs": {"model": ["na", 0], "clip": ["na", 0],
                       "lora_name": "x.safetensors", "strength_model": 1.0,
                       "strength_clip": 1.0}}}), "LoraLoader")
    check("F3j LoraLoader: unknown with a reason, not a fabricated number",
          lora["status"] == "unknown" and lora["reason"] != "", lora["reason"])

    # -- the classic image pipeline
    prompt = {
        "a": {"class_type": "EmptyImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1, "color": 0}},
        "b": {"class_type": "EmptyImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 2, "color": 0}},
        "batch": {"class_type": "ImageBatch",
                  "inputs": {"image1": ["a", 0], "image2": ["b", 0]}},
        "by": {"class_type": "ImageScaleBy",
               "inputs": {"image": ["a", 0], "upscale_method": "bilinear",
                          "scale_by": 0.5}},
        "scale": {"class_type": "ImageScale",
                  "inputs": {"image": ["a", 0], "upscale_method": "bilinear",
                             "width": 0, "height": 256, "crop": "disabled"}},
        "inv": {"class_type": "ImageInvert", "inputs": {"image": ["a", 0]}},
        "prev": {"class_type": "PreviewImage", "inputs": {"images": ["a", 0]}},
    }
    report = estimate_workflow(prompt)
    check("F3k ImageBatch: batch dims add up (1 + 2)",
          _node(report, "ImageBatch")["total_bytes"] == 3 * 512 * 512 * 3 * 4,
          str(_node(report, "ImageBatch")["total_bytes"]))
    check("F3l ImageScaleBy: 0.5 -> 256x256",
          _node(report, "ImageScaleBy")["total_bytes"] == 256 * 256 * 3 * 4,
          str(_node(report, "ImageScaleBy")["total_bytes"]))
    check("F3m ImageScale: height only -> width follows the ratio",
          _node(report, "ImageScale")["total_bytes"] == 256 * 256 * 3 * 4,
          str(_node(report, "ImageScale")["total_bytes"]))
    check("F3n ImageInvert: same shape, fresh allocation",
          _node(report, "ImageInvert")["total_bytes"] == 512 * 512 * 3 * 4,
          str(_node(report, "ImageInvert")["total_bytes"]))
    check("F3o PreviewImage: estimated, not unknown",
          _node(report, "PreviewImage")["status"] == "estimated",
          str(_node(report, "PreviewImage")["status"]))
    check("F3p EmptyImage: (1, 512, 512, 3)",
          _by_id(report, "a")["total_bytes"] == 512 * 512 * 3 * 4,
          str(_by_id(report, "a")["total_bytes"]))

    # -- an override replaces the assumption (the panel's own knob)
    report = estimate_workflow(
        {"n": {"class_type": "CLIPTextEncode",
               "inputs": {"text": "x", "clip": ["na", 0]}}},
        None, {"clip_hidden": 1280})
    check("F3q assumption override: user values win and are recorded",
          _node(report, "CLIPTextEncode")["total_bytes"] == 77 * 1280 * 4
          and any(entry["key"] == "clip_hidden" and entry["source"] == "override"
                  for entry in report["assumptions_used"]),
          str(report["assumptions_used"]))

    del tempfile


def _batch4_checks() -> None:
    # -- visualization: a matplotlib canvas, stated as a canvas, not a tensor
    prompt = {
        "x": {"class_type": "CdlRandomTensor", "inputs": {"shape": "10,2"}},
        "plot": {"class_type": "CdlPlot",
                 "inputs": {"X": ["x", 0], "Y": ["x", 0], "xlabel": "x",
                            "ylabel": "y", "xscale": "linear", "yscale": "linear",
                            "figsize": "6,4"}},
        "hist": {"class_type": "CdlHistogram",
                 "inputs": {"tensor": ["x", 0], "bins": 10, "density": False}},
        "images": {"class_type": "CdlShowImages",
                   "inputs": {"images": ["img", 0], "rows": 1, "cols": 1,
                              "scale": 1.5, "titles": ""}},
        "img": {"class_type": "EmptyImage",
                "inputs": {"width": 64, "height": 64, "batch_size": 1, "color": 0}},
    }
    report = estimate_workflow(prompt)
    plot = _node(report, "CdlPlot")
    check("F4a render: figsize 6x4 at 100 dpi -> 400x600 RGBA canvas",
          plot["status"] == "estimated"
          and plot["total_bytes"] == 1 * 400 * 600 * 4 * 4,
          f"{plot['status']} {plot['total_bytes']}")
    check("F4b render: honestly marked approx (a canvas is not a tensor)",
          plot["confidence"] == "approx" and plot["flops_status"] == "zero",
          f"{plot['confidence']} {plot['flops_status']}")
    hist = _node(report, "CdlHistogram")
    check("F4c render: matplotlib defaults when the widget is absent",
          hist["status"] == "estimated"
          and hist["total_bytes"] == 1 * int(round(4.8 * 100)) * int(round(6.4 * 100)) * 4 * 4,
          str(hist["total_bytes"]))
    check("F4d ShowImages: canvas plus the scaled copies of its input",
          len(_node(report, "CdlShowImages")["items"]) == 2,
          str(_node(report, "CdlShowImages")["items"]))

    # -- misc / device queries: estimated instead of unknown, zero cost
    prompt = {
        "box": {"class_type": "CdlMessageBox",
                "inputs": {"title": "t", "text": "hello"}},
        "dev": {"class_type": "CdlDeviceInfo", "inputs": {}},
        "gpu": {"class_type": "CdlTryGpu", "inputs": {"gpu_index": 0}},
    }
    report = estimate_workflow(prompt)
    check("F4e misc/device: no longer reported as unknown",
          all(node["status"] == "estimated" for node in report["nodes"]),
          str([(n["class_type"], n["status"]) for n in report["nodes"]]))

    # -- datasets: priced per batch, corpus size honestly unknown
    prompt = {
        "sd": {"class_type": "CdlSyntheticData",
               "inputs": {"num_features": 2, "num_examples": 100, "noise_std": 0.01,
                          "seed": 0}},
        "arr": {"class_type": "CdlLoadArray",
                "inputs": {"features": ["sd", 0], "labels": ["sd", 1],
                           "batch_size": 32, "shuffle": True}},
        "info": {"class_type": "CdlDataLoaderInfo", "inputs": {"dataloader": ["arr", 0]}},
        "fm": {"class_type": "CdlFashionMNIST",
               "inputs": {"batch_size": 32, "resize": 28}},
        "dl": {"class_type": "CdlDownload",
               "inputs": {"url": "http://example.invalid/x.zip", "save_dir": "data",
                          "sha1_hash": ""}},
    }
    report = estimate_workflow(prompt)
    arr = _node(report, "CdlLoadArray")
    # 100 samples of 2 features, batches of 32 -> 64 floats per batch
    check("F4f LoadArray: one materialised batch, not the corpus",
          arr["status"] == "estimated" and arr["total_bytes"] == 32 * 2 * 4,
          str(arr["total_bytes"]))
    info = _node(report, "CdlDataLoaderInfo")
    check("F4g DataLoaderInfo: 100 samples / batch 32 -> 4 batches",
          info["status"] == "estimated"
          and info["basis"]["params"]["batches"] == 4
          and info["basis"]["params"]["samples"] == 100,
          str(info["basis"]))
    fm = _node(report, "CdlFashionMNIST")
    check("F4h FashionMNIST: batch priced, corpus size not invented",
          fm["status"] == "estimated" and fm["total_bytes"] == 32 * 28 * 28 * 4,
          str(fm["total_bytes"]))
    check("F4i CdlDownload: estimated, no fake memory",
          _node(report, "CdlDownload")["status"] == "estimated"
          and _node(report, "CdlDownload")["total_bytes"] == 0)

    # -- the deliberately unmodelled ones stay unknown, with a reason
    prompt = {
        "n1": {"class_type": "CdlModelForward",
               "inputs": {"model": ["m", 0], "tensor": ["t", 0]}},
        "m": {"class_type": "CdlRNNScratch",
              "inputs": {"num_inputs": 32, "num_hiddens": 64, "sigma": 0.01}},
        "t": {"class_type": "CdlRandomTensor", "inputs": {"shape": "1,32"}},
        "n2": {"class_type": "CdlLoraModelToTensor",
               "inputs": {"lora_model": ["m", 0]}},
        "n3": {"class_type": "CdlUpdateD",
               "inputs": {"X": ["t", 0], "Z": ["t", 0], "net_D": ["m", 0],
                          "net_G": ["m", 0]}},
    }
    report = estimate_workflow(prompt)
    forward = _node(report, "CdlModelForward")
    check("F4j ModelForward: a module's output shape is not inferable",
          forward["status"] == "unknown" and forward["reason"] != "",
          forward["reason"])
    check("F4k unmodelled families: unknown, never fabricated",
          _by_id(report, "n2")["status"] == "unknown"
          and _by_id(report, "n3")["status"] == "unknown",
          str([(_by_id(report, "n2")["reason"]), _by_id(report, "n3")["reason"]]))

    # -- the whole-library view: how many node types the registry now knows
    from comfy.profiling import estimators as _est
    check("F4l registry: the rule library grew well past the 14 LM nodes",
          len(_est.ESTIMATORS) >= 100, str(len(_est.ESTIMATORS)))
    registry = set(_est.ESTIMATORS)
    cdl_nodes = set()
    for path in sorted((REPO_ROOT / "comfydl" / "nodes").glob("*.py")):
        cdl_nodes |= set(re.findall(r'NODE_CLASS_MAPPINGS\["([A-Za-z0-9_]+)"\]',
                                    path.read_text(encoding="utf-8")))
    lm_family = {
        "TextVocabBuild", "TextEncode", "TextDecode", "TextSlidingWindow",
        "LanguageModelEmbedding", "LanguageModelTransformerBlock",
        "LanguageModelBuild", "TrainingOptimizer", "LanguageModelTrain",
        "LanguageModelForward", "LanguageModelGenerate", "LanguageModelSave",
        "LanguageModelLoad", "TrainingLoop",
    }
    print("registry: %d estimators | ComfyDL %d/%d | built-in %d" % (
        len(registry), len(cdl_nodes & registry), len(cdl_nodes),
        len(registry - cdl_nodes - lm_family)))
    print("ComfyDL types still unmodelled: %s" % ", ".join(sorted(cdl_nodes - registry)))
    print("built-in types modelled: %s" % ", ".join(sorted(registry - cdl_nodes - lm_family)))


def _by_id(report: dict, node_id: str) -> dict:
    return [n for n in report["nodes"] if n["id"] == node_id][0]


def _node(report: dict, class_type: str) -> dict:
    return [n for n in report["nodes"] if n["class_type"] == class_type][0]


def main() -> int:
    _foundation_checks()
    _engine_checks()
    _batch1_checks()
    _batch2_checks()
    _batch3_checks()
    _batch4_checks()
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if not ok and detail else ""))
        print()
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
