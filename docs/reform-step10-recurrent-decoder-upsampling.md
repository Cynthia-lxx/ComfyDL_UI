# Reform Step 10: Recurrent / Decoder / Upsampling / reform 第十步：循环、解码器与上采样

Three node families land at once, all of them following the existing core-layer conventions
(stateless, weights and data on slots, `TENSOR` flow):

1. **Recurrent family** — a new category `Network & Layers/Recurrent` with three nodes:
   `RecurrentRNN` / `RecurrentLSTM` / `RecurrentGRU`.
2. **Transformer decoder block** — `TransformerDecoderBlock`, category
   `Network & Layers/Attention`.
3. **Upsampling family** — `ConvolutionUpsample` / `ConvolutionPixelShuffle` /
   `ConvolutionPixelUnshuffle`, category `Network & Layers/Convolution`.

Plus the soft-archive of the three superseded d2l builders.

## Motivation (动机)

* The core layer family covered MLPs (steps 1–4), attention / Transformers (steps 7–8) and
  convnets (step 4) but had **no sequential layer at all**: the classic recurrence family
  (RNN → LSTM → GRU) that every DL course teaches first was only available as cdlModel
  builders on the d2l side. The core `TENSOR` graphs could not express them.
* The attention family had the *encoder* block but not the *decoder* block — the second half of
  the original Transformer. All the pieces existed (causal mask, cross-attention, FFN), only
  the assembly was missing.
* `ConvTranspose` was the only way to grow a feature map, and it carries weights. The two
  parameter-free routes — interpolation and sub-pixel reshuffling — were missing.

## Nodes (节点清单)

### 1. Recurrent family — `comfy_extras/nodes_recurrent.py` (new file, new category)

| Node | Class | Cell | Gate rows | States |
|---|---|---|---|---|
| RNN | `RecurrentRNN` | `h = tanh(W_ih·x + b_ih + W_hh·h + b_hh)` | 1 | h |
| LSTM | `RecurrentLSTM` | `[i, f, g, o]` rows; `c = f·c + i·g`, `h = o·tanh(c)` | 4 | h + c |
| GRU | `RecurrentGRU` | `[r, z, n]` rows; reset scales the hidden side only; `h = (1-z)·n + z·h` | 3 | h |

Shared contract of all three:

* `x` is batch-first `(B, T, I)`; a bare `(T, I)` runs as a batch of one, extra leading dims
  are flattened into the batch and restored on `y`.
* Weights / biases / initial states are **optional slots** in the exact `nn.RNNBase` layout
  (`weight_ih_l0` etc. plug straight in; `bias_ih + bias_hh` add, as in torch). Unconnected
  means zero — a freshly placed node with only `x` wired already runs.
* Single layer, no `num_layers` / `bidirectional` / `dropout` widgets: stacking is the previous
  node's `y` into the next node's `x`, and `hn` / `cn` round-trip into `h0` / `c0`.
* Only widget: `hidden_size` INT (default 16). Outputs `y (B, T, H)` + `hn (B, H)` (+ `cn`
  for the LSTM), all `detach()`ed. Dtype promotion as in `BasicLinear`; non-float input runs
  in float32.
* The cell math is written out **explicitly** (input and hidden sides computed separately with
  `F.linear`, because the GRU's reset gate scales only the hidden side) — the same
  "show every step" stance as the attention family's hand-rolled softmax, not a wrapper over
  `nn.RNN`.

### 2. TransformerDecoderBlock — `comfy_extras/nodes_attention.py`

Post-LN, one sub-block more than the encoder:

```
Self-Attn → Add → LN → Cross-Attn (k/v from context) → Add → LN → FFN → Add → LN
```

* Two full weight sets on slots (self-attention 4 + cross-attention 4 + FFN 2), three LN
  affine pairs (optional; unconnected = non-affine), `self_mask` / `cross_mask` optional
  (SDPA boolean semantics, `True` = attend). Wire `AttentionCausalMask` into `self_mask` for
  the autoregressive property.
* Cross-attention `k` / `v` weights are `(E, E_kv)` — the context width may differ from the
  decoder width; `cross_out` stays square `(E, E)` for the residual.
* Reuses `_multi_head_attention`, `_normalize_mode`, the local-`Generator` dropout and the
  dtype promotion of the existing family verbatim.

### 3. Upsampling family — `comfy_extras/nodes_convolution.py`

| Node | Class | Function | Contract |
|---|---|---|---|
| Upsample | `ConvolutionUpsample` | `F.interpolate` | `dims` 1/2/3 dispatch; mode must match the rank (1D = nearest/linear, 2D = nearest/bilinear/bicubic, 3D = nearest/trilinear) — the `nn.Upsample` contract; `scale_factor` FLOAT 2.0; `align_corners` only for interpolating modes; rank is pinned exactly (a mismatch raises instead of being silently reinterpreted by `F.interpolate`) |
| Pixel Shuffle | `ConvolutionPixelShuffle` | `F.pixel_shuffle` | `(N, C·r², H, W) → (N, C, H·r, W·r)`, the parameter-free half of sub-pixel convolution; `r` INT 2 |
| Pixel Unshuffle | `ConvolutionPixelUnshuffle` | `F.pixel_unshuffle` | the exact inverse, `(N, C, H·r, W·r) → (N, C·r², H, W)`; the pair at the same `r` is a lossless round trip |

No weights at all — pure interpolation / reshaping, the parameter-free counterparts of
`ConvTranspose`.

## Soft archive (软归档)

`CdlRNNScratch` / `CdlRNN` / `CdlGRU` in `comfydl/nodes/model_nlp.py`:

* `DEPRECATED = True`, display names gain ` (DEPRECATED)`, category moves to
  `d2l/_Legacy/NLP Models`, **node ids unchanged** — the standard recipe.
* The `RNNLM*` wrapper family stays where it is: it does not duplicate the core nodes and
  still works (it can even wrap the archived builders — the wiring stays valid).

## Design decisions (设计决策)

* **Explicit cells, not `nn` wrappers.** Teaching visibility and bit-exact determinism over
  convenience; the weight layout still matches torch so checkpoints plug straight in, and the
  smoke tester pins every cell against `nn.RNN` / `nn.LSTM` / `nn.GRU` numerically.
* **Zero-on-unconnected instead of in-node random init.** A stateless node must not invent
  parameters; "out of the box" means *it runs*, not *it invents weights*. Real weights come
  from `Learnable Parameters` / a checkpoint.
* **`(B, H)` states, not `(1, B, H)`.** The node's own output feeds its own input without a
  squeeze; the torch layout `(1, B, H)` is accepted too (any tensor holding `B·H` values
  reshapes).
* **Decoder block in the attention family, upsampling in the convolution family** — no new
  categories beyond `Recurrent`, which needs one because the nodes are neither attention nor
  convolution.
* **`Upsample` pins the rank exactly.** `F.interpolate` dispatches by tensor rank, so a
  `dims`/rank mismatch would silently reinterpret the input; the node raises instead.

## Widget defaults (控件默认值)

| Node | Widget | Default | Why |
|---|---|---|---|
| RNN / LSTM / GRU | `hidden_size` | 16 | matches the smoke fixture and typical toy configs; weights validate against it |
| Upsample | `dims` / `mode` / `scale_factor` / `align_corners` | 2 / nearest / 2.0 / false | the textbook nearest ×2 upsample on an image map |
| Pixel Shuffle / Unshuffle | `r` | 2 | the common sub-pixel factor (×2 per axis) |

## Verification (验证)

* Smoke coverage: `_INPUT_OVERRIDES` provide one matched recurrent world (`x` (2, 5, 6),
  `hidden_size=16` → weights (16/48/64, 6) / (16/48/64, 16)); the decoder block reuses the
  (2, 5, 8) queries / (2, 7, 8) context fixtures; the pixel nodes get divisibility-matched
  maps.
* `_OUTPUT_CHECKS`:
  * RNN / LSTM / GRU vs `nn.RNN` / `nn.LSTM` / `nn.GRU` with the same wired weights
    (biases added the torch way), plus the all-unconnected branch which must produce exactly
    zero states;
  * decoder block vs a hand composition of `AttentionMultihead` re-runs + `F.layer_norm` /
    `F.linear`, plus the causal-mask autoregressive property (perturbing the last input
    position must not change any earlier output);
  * `Upsample` vs `F.interpolate` (nearest + bilinear), `Pixel Shuffle` vs `F.pixel_shuffle`
    and the lossless `Shuffle → Unshuffle` round trip.
* Full run after step 10: **290 PASS / 23 SKIP / 0 FAIL across 313 registered nodes**
  (47 categories; `IMPORT_FAILED []`, `CDL 109`).
* Documented library: **207 nodes across 35 categories = 109 ComfyDL + 98 core**
  (`_update_readme.py --check` green; `gen_locales.py --check` green after re-generating
  `locales/zh/nodeDefs.json` for the three renamed display names).
