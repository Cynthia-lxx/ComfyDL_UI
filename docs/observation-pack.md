# The Observation & Performance Pack (M-D, v0.5.1 → v0.6.0)

Design notes and usage reference for the subsystems delivered by the
observation-milestone sticks (M-D-0 → P6 → T6-1 → P6b → P7). Companion docs:
[`crashsite.md`](crashsite.md), [`cdlapp-mode.md`](cdlapp-mode.md),
[`profiling-logging.md`](profiling-logging.md).

## 1. The switch registry (`comfy.profiling.switch_registry`)

Every optional performance feature registers `{id, default, hot_toggle,
self_meter}`. The meta-profiling rule (ADR-D1): **a feature that cannot
account for its own cost does not ship** — each Tier-1 feature measures its
own bookkeeping into the registry, and any consumer can subtract that cost
from a measured duration.

| Switch id | Default | Hot toggle | Self-meter | Governs |
|---|---|---|---|---|
| `comfydl.profiling` | on | yes | yes | the whole profiling stack (watchdog sampling, run metering, frontend hooks) |
| `comfydl.crashsite` | on | yes | yes | snapshot capture + resume + training checkpoints |
| `comfydl.spine` | on | yes | yes | the background observation-sampling daemon |
| `comfydl.attention_select` | on | yes | yes | attention backend benchmarking & injection |
| `comfydl.compile` | on | yes | yes | global torch.compile hot-disable (the node itself is opt-in per graph) |
| `comfydl.quant` | on | yes | yes | GGUF lazy-dequant loading (family absent entirely without the `gguf` package) |
| `comfydl.vram` | on | yes | yes | VRAM hygiene (Purge VRAM) |
| `comfydl.dataio.cache` | on | yes | yes | the DataFrame cache behind the dataset readers |
| `comfydl.runs` | on | yes | yes | the experiment runs ledger |
| `comfydl.phasetimer` | on | yes | yes | the per-execution phase anatomy |
| `comfydl.verbose` | off | yes | no | per-step VERBOSE logging to disk |
| `comfydl.notebook` | off | yes | yes | the in-graph Notebook cell (Tier 2, off by default) |

**Hard-disable flags** (read once at startup, re-enabling refused with 409
until a restart):

- `--cdl-profiling-disable` — kills the whole analysis stack; historically it
  also hard-disables the crash site.
- `--cdl-crashsite-disable` — kills snapshot capture, resume endpoints and
  training checkpoints.

**Family-absence degradation**: a feature whose backing package is missing
(e.g. the quant family without `gguf`) does not degrade to error-continue —
it is absent from the node library entirely (hard-disable-by-import contract),
and its switch reports `hard_disabled` over HTTP.

**Per-run attribution**: `self_overhead()` / `self_overhead_all()` expose
`{total_s, samples}`. A *per-run* overhead is the baseline delta between a
snapshot taken at run start and one taken at finish — resident threads (the
spine) accumulate across runs, so absolute totals must never be attributed to
a single run.

## 2. The observation spine (`app/spine/`)

One daemon sampling thread owning all periodic observation (GPU/CPU facts,
watchdog attribution) behind a versioned snapshot contract; the watchdog
migrated onto it in M-D-0. Sample rate 0 = fully stopped (the Crystools
start/stop model). The spine switch (`comfydl.spine`) gates the daemon; its
HTTP face is `GET /comfydl/spine` (describe), `GET /comfydl/spine/snapshot`
(latest sample) and `PATCH /comfydl/spine` (hot configure — this endpoint,
not the generic switch PATCH, owns the daemon's start/stop side effects).

## 3. The runs ledger (`comfy.runs_ledger`, `comfydl.runs`)

Every experiment run is filed in three layers:

1. **Detail JSON** — one file per run: params, per-step loss history, timing,
   status, and (since v0.6.0) the net-duration fields below.
2. **`runs.csv` index** — one row per run (`run_id`, experiment, metric,
   `started_at`, `duration_s`, `duration_s_net`, status). Header changes
   rotate the old csv into a timestamped backup (`_migrate_stale_index`).
3. **Experiment grouping** — the `experiment` name groups runs for the
   comparison charts in the CDL App Panel.

**Net duration** (v0.6.0): `duration_s_net = max(0, duration_s − Σ
self_overheads)` where the Σ is the run's baseline-delta share of every
Tier-1 switch's self-measured overhead; the per-switch breakdown lands in
`detail.self_overheads`. The floor at zero is deliberate: a mis-attributed
overhead must never render a negative duration.

Ledger identity: `run_id = YYYYMMDD-HHMMSS-xxxxxx`; monotonic-clock timing;
index rows are coerced (`duration_s_net` comes back as a float).

## 4. PhaseTimer (`comfy.profiling.phase_timer`, `comfydl.phasetimer`)

Dissects one prompt execution into the ADR-D1 cost positions:

- **prompt_setup** — `execute_async` entry until the node loop starts
  (dynamic prompt build, cache set/lookup, GC pass);
- **per_step** — time spent inside per-step observability callbacks (KSampler
  step callbacks via `wrap_sampler_callback`, training progress via
  `step_window`). Deliberately *not* the model's own compute, and deliberately
  **not** hooked on `PROGRESS_BAR_HOOK` (100 ms throttling would swallow
  steps) or spine ticks (1 s sampling phase).
- **run_teardown** — from the end of node execution to executor exit
  (crash-site rescue, lifecycle notification, run persistence).

Hooks are constant-size no-ops when the switch is off; bookkeeping is
self-metered in one batch at `finish_execution`. The last 20 executions stay
in memory, served read-only by `GET /comfydl/profiling/phases` (which also
carries the registry's cumulative overhead meter) and rendered as the
**Phase anatomy** card in the Profiling panel (gross / setup / per-step ×N /
teardown / net, bilingual).

## 5. The attention bench cache (`comfy_extras/nodes_attention_select.py`)

**Attention Optimizer (Auto)** benchmarks every registered backend that
*actually runs* on this machine (a mini forward must succeed — imports are
never trusted), attaches the fastest and caches the verdict per geometry in
the signature cache (`attention_bench` namespace). **Attention Backend
(Manual)** pins one by hand and explains it.

- Cache key payload: device, torch version, seq_len, heads, head_dim, batch,
  sorted backend list, **and the global compile switch state** (`compile_enabled`,
  since v0.6.0 — a compiled attention path times differently; per-node widgets
  vary per graph and would shatter the cache, so only the stable global toggle
  rides in the key).
- Cache values carry the stamped geometry (`device/torch/seq_len/heads/
  head_dim`) since v0.6.0 — the digest key itself is opaque. Pre-stamping
  entries survive the readout as geometry-less rows (additive contract,
  pinned by tests).
- **Probe sessions** (`compile_manager.probe_session()`) wrap any timing
  session in `torch.compiler.disable` so tracing/recompilation never lands
  inside the timing window. Some torch builds (observed on 2.13) forbid the
  context-manager usage of `dynamo.optimize`; the context degrades to a plain
  session there rather than breaking the measurement.

## 6. HTTP API surface (the always-200 degradation contract)

Read-only status endpoints NEVER fail: a bare environment or hard-disabled
family yields 200 + `{"available": false}` instead of an error status, so
panels render empty states instead of breaking. Writes (PATCH/POST) do return
real status codes (409 = hard-disabled, 400 = bad payload).

| Endpoint | Purpose |
|---|---|
| `GET /comfydl/switches` | the registry |
| `GET /comfydl/switches/status` | per-family live `describe()` + `status()` for all performance families |
| `PATCH /comfydl/switches/{id}` | soft hot-toggle (generic families) |
| `GET/POST /comfydl/profiling/enabled` | profiling master switch (+ hard-disable) |
| `GET/POST /comfydl/crashsite/enabled` | crash-site master switch (+ hard-disable) |
| `GET/PATCH /comfydl/spine` | spine describe / hot configure |
| `GET /comfydl/spine/snapshot` | latest observation sample |
| `GET /comfydl/dataio/cache/stats` · `/entries` | DataFrame cache stats / per-table cards |
| `GET /comfydl/attention/bench` | cached benchmark rows + backend mechanism notes |
| `GET /comfydl/profiling/phases` | PhaseTimer anatomy + overhead meter |
| `GET /comfydl/profiling/history` | measured-FLOPs run history |
| `GET /comfydl/runs` · `/detail` | runs ledger index / per-run detail |
| `GET /comfydl/notebook/variables` · `POST /free` | Notebook cell kernel |

## 7. Panels

Two floating panels share the same shell mechanics (minimap-sized, collapse /
hide / drag with persisted position, dock-gap repair, three-gate reduced-motion
exemption, bilingual):

- **Optimizer panel** (P6b) — the registry's runtime face: per-family live
  truth, cache stats, the attention bench card; toggles ride the switch
  endpoints and snap back to server truth on 409/404.
- **Observation panel** (P6) — the run monitor over the runs ledger.

Both mount through `frontend_patch.py` loader injection (idempotent, single
`</body>` assertion) and clean up on unload.
