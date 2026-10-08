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
2. `POST /comfydl/crashsite/resume {"id": ...}` re-validates and re-queues
   the stored prompt; on the fresh run, each node's local cache miss is
   answered by `on_lookup` (the stored output is rehydrated), so the host
   marks those nodes cached and skips them.
3. The sidebar tab "Crash Site" lists the snapshot library (time / status /
   node count / size) with a per-snapshot Resume button.

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

## Tests

`cdl_smoke_tests/test_crashsite.py` — serialization round-trips, store
CRUD, the provider lifecycle (start → store → end → lookup), and the routes
through a real TestServer with a crashed 3-node scene; plus a real-startup
smoke (`--quick-test-for-ci`).
