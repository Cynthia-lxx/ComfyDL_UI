# Crash site — execution snapshots & breakpoint recovery (M1)

## What it is

A breakpoint-recovery net for interrupted/failed runs.  **Lazy capture**
(git-fire perf fix, 2026-10-13): normal runs pay ZERO crash-site overhead —
nothing is serialized while the workflow executes.  The moment a run is
interrupted or a node fails, the host sweeps whatever survived in the output
cache into a per-prompt SQLite database
(`user/comfydl/snapshots/<id>.db`); the panel can then re-queue the stored
prompt and the host skips every node whose output survived — only the
remaining part actually executes.

Trade-off: a HARD process death (kill/segfault/power loss) loses the scene,
because capture happens at failure time from the still-warm cache.  The
earlier always-on incremental capture covered that case but slowed runs
2-10x — judged the wrong trade.

Independent of the profiling stack (the profiling master switch does not
affect it); a single hard-disable flag, `--cdl-profiling-disable`, turns it
off too (data safety is treated as part of the same stack for that flag).

## How it works

1. `app/crashsite/provider.py` registers a **CacheProvider** with the host
   (`comfy_api.latest._caching`).  Its `should_cache` declines ALL real-time
   stores (lazy capture) and gates snapshot lookups to resumed prompts only.
2. When a run dies mid-flight, the executor's failure path calls
   `BasicCache.rescue_scene(prompt_id)` (comfy_execution/caching.py), which
   walks the surviving output-cache entries and hands each to
   `SnapshotProvider.on_scene_rescue` — serialization happens in a thread
   pool (never on the event loop) and rows land in the snapshot DB keyed by
   the host's input-signature SHA256.
3. `POST /comfydl/crashsite/resume {"id": ..., "client_id": ...}` re-validates
   and re-queues the stored prompt; on the fresh run, each node's local cache
   miss is answered by `on_lookup` (the stored output is rehydrated), so the
   host marks those nodes cached and skips them.  The `client_id` from the
   requesting browser tab is echoed into `extra_data` — **mandatory for a
   visible resume**: without it the executor sets `server.client_id = None`
   and suppresses every `executing`/`executed`/`execution_*` websocket event,
   so the resumed run executed silently (the 2026-10-09 "no continuation"
   incident).  A `queue_updated` broadcast follows the re-queue so the
   resumed prompt shows up as an active job.
3. The sidebar tab "Crash Site" lists the snapshot library (time / status /
   node count / size) with a per-snapshot Resume button.

## M2 explainability (2026-10-09)

The panel now answers "what did resume actually do":

* **Per-node captured list** — every snapshot card lists the nodes whose
  outputs survived (node id + class_type + size), first four inline, the
  rest behind a `+n more` expander.  The snapshot/resume HTTP responses
  carry the same `captured` summary.
* **Authoritative skip/re-run statistics** — after a resumed run ends
  (completed *or* interrupted), the NEW snapshot card's note reads
  `resume of <source>: skipped X/Y (a snapshot + b local cache), re-executed c`.
  The numbers come from the executor's cached branch (`execution.py` →
  `notify_node_cached` → `SnapshotProvider.on_node_cached`), the single
  funnel where both local-RAM hits and snapshot rehydrations land — NOT
  from `on_lookup` counts, which miss same-process local cache hits.
* **Auto refresh** — the library refreshes on `execution_start/success/
  error/interrupted`, so the statistics appear without clicking Refresh.

The resume toast also points at the new snapshot card for the statistics.

## Scene export (git-fire M2, 2026-10-13)

After an interrupt (or any time, via the per-card button on interrupted
snapshots), the panel shows a **Scene export** summary table: every
recoverable node output of the snapshot (class_type #node_id · slots · size)
plus a workflow-prompt JSON row.  Each row is pre-checked with a sensible
default export format and a dropdown to override:

| serialize tag | default export | candidates |
|---|---|---|
| st (Tensor) | .safetensors | safetensors, png (image-shaped only) |
| dict-st / list-st | .safetensors (+meta sidecar) | safetensors, json |
| json (scalar) | .json | json, txt |
| dataset-st | .db (self-describing sqlite) | db, json |
| pt / pickle | .pt | pt, safetensors |

Selected files are written to `output/comfydl/scene_export/<snapshot id>/`
so they surface in Results assets; the prompt row becomes `*_prompt.json`.
Per-row failures never abort the scene - they are reported in the response
and counted in the toast.  A chosen format that doesn't fit a slot falls
back to that slot's own default.  Endpoints: `GET .../snapshots/{id}/detail`
(full annotated capture rows), `POST .../export`.

Auto-nudge: on `execution_error`/`execution_interrupted` the panel opens
itself with the newest interrupted snapshot's export table and a toast.

Resume semantics (M1): **node-level memoization, not mid-node continuation** —
with one M3 exception: the three trainers (Regression Train, Training Loop,
Language Model Train) checkpoint their full state at the interrupt moment
(model/optimizer/scheduler/loss history/early-stop tracker/RNG) into
`user/comfydl/checkpoints/`, and a resume re-executing them CONTINUES from
the breakpoint step.  The freshness gate is the prompt_id: a checkpoint only
continues under the prompt that was interrupted — a fresh Run carries a new
prompt_id, the stale checkpoint is discarded and training starts over.
Finished non-training nodes are skipped; edits made after the interrupt do
not apply to a resume (they affect the next normal Run).  Note that a
trainer with early stopping may legitimately finish with a short
`loss_history` (e.g. `early_stop_patience` default 30 on a plateauing
curve) — check the skip statistics before assuming a resume was
"truncated".

## Type mapping & honest boundaries

| Type | Format | Fidelity |
|---|---|---|
| TENSOR / LATENT / IMAGE / MASK / AUDIO / SIGMAS | safetensors BLOB | full |
| scalars, cdlVocab | JSON | full |
| CdlDataset | safetensors + JSON sidecar | full |
| CONDITIONING | flattened safetensors + JSON | degraded |
| CLIP / VAE / CLIP_VISION / CONTROL_NET | state_dict + JSON | degraded |
| MODEL (ModelPatcher) | — | **rejected** (would produce a broken object) |
| nn_model | torch.save (state_dict) | degraded (needs node spec to rebuild) |
| cdlDataloader | — | **unsupported** (rebuild from DATASET upstream) |

Not covered in M1: `--cache-none` mode (no output cache to hook), subgraph
internal nodes (only the expand node's own output is recoverable), training
node internal state (checkpoint integration is M3).

Snapshot status: `running` while the prompt executes, `completed` when it
finishes, `interrupted` when the run dies mid-flight (the executor notifies
the provider via the optional `error` lifecycle event — node exceptions and
user interrupts both land here).

## Tests

`cdl_smoke_tests/test_crashsite.py` — serialization round-trips, store
CRUD, the provider lifecycle (start → store → end → lookup, plus the
error→interrupted filing), lazy capture (real-time store declined,
resume-gated lookups, and the real-executor L1/L2 pair: successful run
stores nothing, failed run rescues its surviving outputs), scene export
(format matrix, png/db writers, fallbacks, per-row isolation), and the
routes through a real TestServer with a crashed 3-node scene (client_id
passthrough, capture summaries, detail/export endpoints); plus a
real-startup smoke (`--quick-test-for-ci`).
