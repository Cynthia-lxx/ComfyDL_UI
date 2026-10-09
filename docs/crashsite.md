# Crash site — execution snapshots & breakpoint recovery (M1)

## What it is

Every workflow execution is snapshotted incrementally: as each node finishes,
its output is serialized and written into a per-prompt SQLite database
(`user/comfydl/snapshots/<id>.db`).  If the run later fails, is interrupted,
or you simply want to re-run it, the panel can re-queue the stored prompt and
the host skips every node whose output survived — only the remaining part
actually executes.

Independent of the profiling stack (the profiling master switch does not
affect it); a single hard-disable flag, `--cdl-profiling-disable`, turns it
off too (data safety is treated as part of the same stack for that flag).

## How it works

1. `app/crashsite/provider.py` registers a **CacheProvider** with the host
   (`comfy_api.latest._caching`): `on_store` fires after every node output
   enters the host output cache — serialization happens in a thread pool
   (never on the event loop) and the row lands in the snapshot DB keyed by
   the host's input-signature SHA256.
2. `POST /comfydl/crashsite/resume {"id": ..., "client_id": ...}` re-validates
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

Resume semantics (M1): **node-level memoization, not mid-node continuation.**
Finished nodes are skipped; the interrupted node re-executes from scratch
with the ORIGINAL stored widget values (edits made after the interrupt do
not apply — they affect the next normal Run).  Mid-node continuation for
long training nodes is the M3 checkpoint integration.  Note that a trainer
with early stopping may legitimately finish with a short `loss_history`
(e.g. `early_stop_patience` default 30 on a plateauing curve) — check the
skip statistics before assuming a resume was "truncated".

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
error→interrupted filing), and the routes through a real TestServer with a
crashed 3-node scene (including the client_id passthrough regression);
plus a real-startup smoke (`--quick-test-for-ci`).
