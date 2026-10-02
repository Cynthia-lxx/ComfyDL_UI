# Reform Step 11: Live Preview / reform 第十一步：实时预览

No new nodes. The three long-running nodes that already drive ComfyUI's progress bar gained
**live preview images** on the same official side channel, and the mechanism research behind it
is recorded here so the next batch (video / 3D previews, external progress handlers) does not
have to rediscover it.

## The channel (通道分析)

How a preview reaches the screen in stock ComfyUI, verified against the ComfyUI-original source:

1. **A node calls** `comfy.utils.ProgressBar.update_absolute(value, total, preview)` with
   `preview = ("JPEG", PIL.Image, max_size)` — the triple format `server.py`
   (`send_image_with_metadata`) expects. A preview-bearing call **always** reaches the hook
   immediately (the 100 ms / 0.5 % throttle of `comfy/utils.py` is skipped whenever a preview
   is attached), which is why the *drawing* has to be rate-limited on the node side.
2. **The global hook** installed by `main.py:hijack_progress()` resolves the *currently
   executing node* from `comfy_execution.utils.get_executing_context` — the node never needs to
   know its own id — and forwards to `get_progress_state().update_progress`.
3. **The server** pushes the numbers as the `progress` websocket event and the image as a
   binary event (`PREVIEW_IMAGE_WITH_METADATA` for new clients, `UNENCODED_PREVIEW_IMAGE` for
   old ones, scaled to `MAX_PREVIEW_RESOLUTION` and JPEG-encoded on the way).
4. **The frontend** binds both to `node_id` and renders the image under the running node's
   progress bar — exactly where a KSampler shows its latent preview.

This is KSampler's own path (`latent_preview.prepare_callback` builds the callback,
`samplers.py` fires it per step with `x0`); custom nodes reuse it with **zero** changes to the
server, the frontend or the execution layer. The alternatives considered and rejected:
`PromptServer.instance.send_sync` (needs a manually wired `unique_id`, duplicates the hook),
the `ui` field of a node's return value (static, only after the node finishes), and new binary
event types (only worth it for genuinely new media such as video).

## What was added (新增内容)

### `comfy/loss_preview.py` (new, pure)

| Piece | Role |
|---|---|
| `LossCurvePreviewer` | one `record(loss, force=False)` per optimizer step; renders the raw curve + an 8-step moving average + best value, rate-limited to ~1 frame / 0.5 s (`MIN_RENDER_INTERVAL`), `force=True` always renders |
| `render_text_snapshot` | the generated-so-far text on a small wrapped card (`generated k/n` header) |
| rendering | matplotlib first (Agg backend, `Figure` created and closed locally, lazy import), PIL `ImageDraw` polyline fallback, then `None` — a preview can never raise into a training loop |

### Wiring (接入)

| Node | Preview | Drive |
|---|---|---|
| Training Loop | live loss curve | the step's existing progress call; `force` on the last step |
| Language Model Train | live cross-entropy curve | same pattern |
| Language Model Generate | generated-text card (raw token ids without a `vocab`) | `on_token` callback — one call per token, `force` on the last |

`comfy/lm_protocol.generate_tokens` gained an optional `on_token(token_id)` callback (sibling of
the existing `progress(done, total)`, same injection-not-import stance); the node drives the bar
from `on_token` alone now, which keeps the **one `ProgressBar` call per step / per token**
contract exact — a frame rides on the call the step was already making, never an extra one. The
headless behaviour is unchanged: without a hook, `ProgressBar` is a no-op and the previewer's
frames are simply discarded (the smoke tester exercises exactly this path through its
`Recorder`).

## Contract test (契约测试)

`cdl_smoke_tests/test_progress_reporting.py` grew from 11 to **19 PASS**:

* T1/T2/T3 additionally assert `previews >= 1` per run (the Recorder now counts preview-bearing
  updates) while keeping the exact `updates == steps` / `updates == num_tokens` counts — the
  preview must not change the progress semantics.
* T3 now runs with a `Vocab` linked, so the text card path (encode prefix → incremental decode)
  is the one under test.
* T7 pins `comfy/loss_preview.py` itself: empty render is `None`, the triple format
  (`"JPEG"`, `PIL.Image`, max size), throttling (`min_interval=60` swallows the second frame)
  and the text snapshot's tolerance of empty input.

## Verification (验证)

* `test_progress_reporting.py`: `19 PASS / 0 FAIL`.
* Full smoke: `290 PASS / 23 SKIP / 0 FAIL` across 313 registered nodes — identical to the
  step-10 baseline, as expected for a node-count-neutral change.
* Registry: unchanged, 313 nodes / 47 categories; documented library unchanged at 207 nodes /
  35 categories (109 ComfyDL + 98 core).
* Browser-level verification deliberately skipped (project convention: bottom-layer tests only);
  the UI rendering itself is stock ComfyUI behaviour exercised daily by KSampler previews.

## Follow-ups (可能的后续)

* Other heavy nodes could adopt the same pattern (the previewer is generic over any scalar
  history); nothing else currently runs long enough to need it.
* A progress/preview *handler* registered via `comfy_execution/progress.py` could mirror loss
  curves into logs or an external dashboard without touching the nodes again.
* New media (video previews for generative animation nodes) would need the "modify the base"
  route: a new `BinaryEventTypes` entry plus frontend support — out of scope here.
