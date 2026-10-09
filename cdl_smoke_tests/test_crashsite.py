"""Smoke tests for the Crash site M1 core loop (execution snapshots + resume).

Covers (2026-10-08):
* serialize.py: tensor / LATENT-style dict / CONDITIONING-style list / JSON /
  CdlDataset round-trips; unsupported-type detection;
* store.py: create / upsert / get-by-hash / list / manifest / status on a
  temp root;
* SnapshotProvider: the exact host-driven sequence on_prompt_start ->
  on_store(s) -> on_prompt_end -> later on_lookup, including a miss;
* routes through a real aiohttp TestServer: list / resume with a live
  PromptQueue mock and a prompt that passes validate_prompt (the 2026-10-07
  spike's 3-node chain, promoted to a permanent regression);
* M2 explainability (2026-10-09): on_node_cached accounting, the resume
  statistics note (completed + interrupted endings), and the capture
  summary on the list/resume responses.

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_crashsite.py
"""

import asyncio
import json
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


def report() -> int:
    failed = 0
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + (f"  [{detail}]" if not ok and detail else ""))
        if not ok:
            failed += 1
    print(f"{len(_RESULTS) - failed} PASS / {failed} FAIL ({len(_RESULTS)} checks)")
    return 1 if failed else 0


# ----------------------------------------------------- sandboxed fixture ---
import cdl_smoke_tests.run_smoke_test as harness  # noqa: E402

harness._bootstrap(Path(tempfile.mkdtemp(prefix="cdl_crashsite_")))
harness._load_registry(False)

import torch  # noqa: E402

from app.crashsite import serialize, store  # noqa: E402
from app.crashsite.provider import SnapshotProvider  # noqa: E402

# Redirect the snapshot root into the sandbox BEFORE anything touches it.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="cdl_snapshots_"))
store.snapshots_root = lambda: _TMP_ROOT


def T(*shape):
    n = 1
    for d in shape:
        n *= d
    return torch.arange(n).reshape(*shape).float()


# ------------------------------------------------------ T1: serialization --
def _serialize_checks() -> None:
    t = T(2, 3)
    tag, blob, meta = serialize.serialize_output(t)
    back = serialize.deserialize_output(tag, blob, meta)
    check("S1 tensor round-trip", tag == "st" and torch.equal(back, t)
          and back.shape == t.shape and back.dtype == t.dtype)

    latent = {"samples": T(1, 4, 4, 8), "batch_index": 3}
    tag, blob, meta = serialize.serialize_output(latent)
    back = serialize.deserialize_output(tag, blob, meta)
    check("S2 LATENT-style dict round-trip", tag == "dict-st"
          and torch.equal(back["samples"], latent["samples"])
          and back["batch_index"] == 3 and back["samples"].is_cpu)

    cond = [[T(1, 77, 768), {"pooled_output": T(1, 768), "weight": 1.5}]]
    tag, blob, meta = serialize.serialize_output(cond)
    back = serialize.deserialize_output(tag, blob, meta)
    check("S3 CONDITIONING-style list round-trip", tag == "list-st"
          and torch.equal(back[0][0], cond[0][0])
          and torch.equal(back[0][1]["pooled_output"],
                          cond[0][1]["pooled_output"])
          and back[0][1]["weight"] == 1.5)

    tag, blob, meta = serialize.serialize_output("hello")
    check("S4 scalar round-trip", tag == "json"
          and serialize.deserialize_output(tag, blob, meta) == "hello")

    # 2026-10-09 fix: tensor-less lists/dicts used to serialize as BARE json
    # while deserialize unpacked {"v": ...} - lists crashed on_lookup with
    # "list indices must be integers or slices, not str".
    tag, blob, meta = serialize.serialize_output(["a", "b"])
    check("S4b tensor-less list round-trip", tag == "json"
          and serialize.deserialize_output(tag, blob, meta) == ["a", "b"])
    tag, blob, meta = serialize.serialize_output({"k": "val"})
    check("S4c tensor-less dict round-trip", tag == "json"
          and serialize.deserialize_output(tag, blob, meta) == {"k": "val"})

    try:
        from comfydl.nodes.data_types import CdlDataset

        ds = CdlDataset(features=T(10, 4), labels=T(10),
                        feature_names=["a", "b", "c", "d"],
                        target_name="y", meta={"source": "unit"})
        tag, blob, meta = serialize.serialize_output(ds)
        back = serialize.deserialize_output(tag, blob, meta)
        check("S5 DATASET round-trip", tag == "dataset-st"
              and torch.equal(back.features, ds.features)
              and back.feature_names == ds.feature_names
              and back.meta.get("source") == "unit")
    except ImportError:
        check("S5 DATASET round-trip", True, "skipped: CdlDataset unavailable")

    class DataLoader:  # duck: the class NAME is what matters
        pass

    check("S6 DataLoader unsupported", serialize.is_unsupported(DataLoader()))
    import comfy.model_patcher as mp

    check("S7 ModelPatcher unsupported",
          serialize.is_unsupported(mp.ModelPatcher.__new__(mp.ModelPatcher)))


# ------------------------------------------------------------ T2: store ----
def _store_checks() -> None:
    path = store.create_snapshot({"1": {"class_type": "X"}}, prompt_id="pid-x",
                                 note="unit")
    store.upsert_node_output(path, "hash-a", "1", 0, "X", "Tensor",
                             "st", b"\x01\x02\x03", "{}")
    store.upsert_node_output(path, "hash-b", "2", 0, "Y", "dict",
                             "dict-st", b"\x04", "{}")
    rows = store.get_outputs_by_hash(path, "hash-a")
    check("S8 upsert/get by hash", len(rows) == 1 and rows[0]["node_id"] == "1"
          and rows[0]["data"] == b"\x01\x02\x03")
    check("S9 miss returns empty",
          store.get_outputs_by_hash(path, "nope") == [])
    store.mark_status(path, "completed")
    listed = store.list_snapshots()
    check("S10 list finds snapshot", len(listed) == 1
          and listed[0]["nodes"] == 2 and listed[0]["status"] == "completed")
    manifest = store.load_manifest(path)
    check("S11 manifest carries prompt",
          manifest is not None and manifest["prompt_id"] == "pid-x"
          and manifest["prompt"].get("1", {}).get("class_type") == "X")


# --------------------------------------------------------- T3: provider ----
def _provider_checks() -> None:
    provider = SnapshotProvider()
    provider.on_prompt_start("pid-p")
    ctx = types.SimpleNamespace(node_id="1", class_type="SpikeSource",
                                cache_key_hash="h-111")
    check("P1 should_cache accepts tensor output",
          provider.should_cache(ctx, types.SimpleNamespace(outputs=[T(2, 2)])))
    asyncio.run(provider.on_store(ctx, types.SimpleNamespace(
        outputs=[{"samples": T(1, 2, 2, 2)}])))
    provider.on_prompt_end("pid-p")
    miss = asyncio.run(provider.on_lookup(types.SimpleNamespace(
        node_id="1", class_type="SpikeSource", cache_key_hash="h-none")))
    check("P2 lookup miss returns None", miss is None)
    hit = asyncio.run(provider.on_lookup(types.SimpleNamespace(
        node_id="1", class_type="SpikeSource", cache_key_hash="h-111")))
    check("P3 lookup hit rehydrates output",
          hit is not None and hit.outputs is not None
          and isinstance(hit.outputs[0], dict)
          and torch.equal(hit.outputs[0]["samples"], T(1, 2, 2, 2)))
    listed = store.list_snapshots()
    mine = [s for s in listed if s["prompt_id"] == "pid-p"]
    check("P4 prompt lifecycle recorded", len(mine) == 1
          and mine[0]["status"] == "completed"
          and mine[0]["nodes"] == 1)

    # 2026-10-09: a run that dies mid-flight must be filed "interrupted",
    # not "completed" (the executor notifies on_prompt_error before the
    # finally-block's on_prompt_end).
    provider.on_prompt_start("pid-err")
    provider.on_prompt_error("pid-err")
    provider.on_prompt_end("pid-err")
    listed = store.list_snapshots()
    mine = [s for s in listed if s["prompt_id"] == "pid-err"]
    check("P5 failed run marked interrupted", len(mine) == 1
          and mine[0]["status"] == "interrupted",
          json.dumps(mine)[:120])
    # A stray error for an unknown prompt must be a harmless no-op.
    provider.on_prompt_error("pid-nobody")
    check("P6 stray on_prompt_error ignored", True)

    # ---- M2: node-cached accounting + the resume statistics note ---------
    prompt3 = {
        "1": {"class_type": "A"},
        "2": {"class_type": "B"},
        "3": {"class_type": "C"},
    }

    class _Q3:
        currently_running = {"1.0": (1.0, "pid-r", prompt3, {}, ["3"], {})}

    rprovider = SnapshotProvider(queue=_Q3())
    rprovider.mark_resume("pid-r", "src-snap")
    rprovider.on_prompt_start("pid-r")
    rprovider._rehydrated.add("1")  # as if on_lookup rehydrated node 1
    rprovider.on_node_cached("pid-r", "1", "A")
    rprovider.on_node_cached("pid-r", "1", "A")  # duplicate id: deduped
    rprovider.on_node_cached("pid-r", "2", "B")
    rprovider.on_node_cached("pid-other", "9", "X")  # stray: ignored
    rprovider.on_prompt_end("pid-r")
    listed = store.list_snapshots()
    mine = [s for s in listed if s["prompt_id"] == "pid-r"]
    note = mine[0]["note"] if mine else ""
    check("P7 resume note: skip/re-run split", len(mine) == 1
          and "skipped 2/3" in note
          and "1 snapshot + 1 local cache" in note
          and "re-executed 1" in note,
          note)

    # An interrupted resume writes the same statistics on its note.
    class _Q4:
        currently_running = {"1.0": (1.0, "pid-ri", prompt3, {}, ["3"], {})}

    riprovider = SnapshotProvider(queue=_Q4())
    riprovider.mark_resume("pid-ri", "src2")
    riprovider.on_prompt_start("pid-ri")
    riprovider.on_node_cached("pid-ri", "1", "A")
    riprovider.on_prompt_error("pid-ri")
    riprovider.on_prompt_end("pid-ri")  # no-op: already closed
    listed = store.list_snapshots()
    mine = [s for s in listed if s["prompt_id"] == "pid-ri"]
    note = mine[0]["note"] if mine else ""
    check("P8 interrupted resume note", len(mine) == 1
          and mine[0]["status"] == "interrupted"
          and "resume of src2" in note
          and "skipped 1/3" in note,
          note)


# ------------------------------------------------- T4: routes integration --
def _route_checks() -> None:
    try:
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
    except ImportError:
        check("R0 aiohttp available", False, "import failed")
        return

    import nodes as host_nodes
    from app.crashsite import routes

    # The spike's 3-node chain, promoted to a permanent fixture: node 3's
    # first execution fails, so the snapshot holds a REAL crashed scene.
    class SpikeSource:
        RETURN_TYPES = ("SPIKE",)
        FUNCTION = "run"
        CATEGORY = "spike"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def run(self):
            return ({"v": 7},)

    class SpikePass:
        RETURN_TYPES = ("SPIKE",)
        FUNCTION = "run"
        CATEGORY = "spike"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("SPIKE",)}}

        def run(self, x):
            return (x,)

    class SpikeBoom:
        RETURN_TYPES = ("SPIKE",)
        FUNCTION = "run"
        CATEGORY = "spike"
        OUTPUT_NODE = True  # validate_prompt requires at least one output node
        boom = {"on": True}

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("SPIKE",)}}

        def run(self, x):
            if SpikeBoom.boom["on"]:
                raise RuntimeError("spike boom")
            return (x,)

    for cls in (SpikeSource, SpikePass, SpikeBoom):
        host_nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls

    PROMPT = {
        "1": {"class_type": "SpikeSource", "inputs": {}},
        "2": {"class_type": "SpikePass", "inputs": {"x": ["1", 0]}},
        "3": {"class_type": "SpikeBoom", "inputs": {"x": ["2", 0]}},
    }

    # A real executor produces a REAL crashed scene (spike promoted).
    import execution as execution_mod

    mock_server = types.SimpleNamespace(
        send_sync=lambda *a, **k: None, client_id=None)
    executor = execution_mod.PromptExecutor(
        mock_server, cache_type=None,
        cache_args={"ram": 0.5, "ram_inactive": 0.3})
    executor.execute(PROMPT, "pid-crash", {}, ["3"])

    class MockQ:  # lets on_prompt_start grab the original prompt JSON
        currently_running = {"1.0": (1.0, "pid-crash", PROMPT, {}, ["3"], {})}

    provider = SnapshotProvider(queue=MockQ())
    provider.on_prompt_start("pid-crash")
    # Replay the executor's surviving cache entries through the provider (the
    # host does this via on_store during the run; the executor here was built
    # with cache_type=None which does not fire providers, so we drive it).
    for node_id in ("1", "2"):
        entry = asyncio.run(executor.caches.outputs.get(node_id))
        asyncio.run(provider.on_store(
            types.SimpleNamespace(
                node_id=node_id,
                class_type=PROMPT[node_id]["class_type"],
                cache_key_hash=f"sig-{node_id}"),
            types.SimpleNamespace(outputs=list(entry.outputs or []),
                                  ui=entry.ui or {})))
    provider.on_prompt_end("pid-crash")

    # A queue mock that records put() items.
    queued = []

    class MockQueue:
        def put(self, item):
            queued.append(item)

    class MockServer:
        number = 100.0

        def queue_updated(self):
            MockServer.queue_refreshed = True

    async def _run():
        app = web.Application()
        app.add_routes(routes.routes)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            r = await client.get("/comfydl/crashsite/snapshots")
            body = await r.json()
            snaps = body.get("snapshots") or []
            crashed = [s for s in snaps if s["prompt_id"] == "pid-crash"]
            check("R1 list shows crashed snapshot",
                  r.status == 200 and len(crashed) == 1
                  and crashed[0]["nodes"] == 2,
                  json.dumps(snaps)[:120])
            r = await client.post("/comfydl/crashsite/resume",
                                  json={"id": "does-not-exist"})
            check("R2 resume 404 on unknown id", r.status == 404)
            r = await client.post("/comfydl/crashsite/resume",
                                  json={"id": crashed[0]["id"]})
            if r.status != 200:
                print("DIAG resume body:", (await r.text())[:800])
            body = await r.json()
            check("R3 resume re-queues prompt", r.status == 200
                  and body.get("resumed") is True
                  and body.get("prompt_id") == "pid-crash",
                  json.dumps(body)[:120])
            check("R4 queue received the stored prompt",
                  len(queued) == 1 and queued[0][1] == "pid-crash"
                  and queued[0][2] == PROMPT and queued[0][4] == ["3"])
            # 2026-10-09: client_id must reach extra_data - without it the
            # executor suppresses every websocket event and the resumed run
            # is invisible to the frontend (the "no continuation" bug).
            check("R9 list includes captured summary",
                  isinstance(crashed[0].get("captured"), list)
                  and len(crashed[0]["captured"]) == 2
                  and crashed[0]["captured"][0]["size_bytes"]
                  >= crashed[0]["captured"][1]["size_bytes"],
                  json.dumps(crashed[0].get("captured"))[:120])
            r = await client.post("/comfydl/crashsite/resume",
                                  json={"id": crashed[0]["id"],
                                        "client_id": "cid-test"})
            body = await r.json()
            check("R5 resume 200 with client_id", r.status == 200
                  and body.get("resumed") is True,
                  json.dumps(body)[:120])
            check("R6 client_id passed to executor extra_data",
                  len(queued) == 2
                  and queued[1][3].get("client_id") == "cid-test")
            check("R7 queue refresh broadcast", MockServer.queue_refreshed
                  is True)
            check("R8 resume response carries capture summary",
                  body.get("captured_nodes") == 2
                  and isinstance(body.get("captured"), list)
                  and len(body["captured"]) == 2,
                  json.dumps(body)[:160])
            check("R10 mark_resume handed to provider",
                  len(mock_provider.resume_calls) == 2
                  and mock_provider.resume_calls[-1]
                  == ("pid-crash", crashed[0]["id"]),
                  str(mock_provider.resume_calls))
        finally:
            await client.close()

    # M2: a mock provider records the resume flag so the route test can
    # assert the mark_resume handoff (the real provider consumes it at
    # on_prompt_start on the worker side).
    class MockProvider:
        def __init__(self):
            self.resume_calls = []

        def mark_resume(self, prompt_id, source):
            self.resume_calls.append((prompt_id, source))

    mock_provider = MockProvider()
    routes.set_snapshot_provider(mock_provider)
    routes.set_prompt_queue(MockQueue(), MockServer())
    asyncio.new_event_loop().run_until_complete(_run())


def main() -> int:
    _serialize_checks()
    _store_checks()
    _provider_checks()
    _route_checks()
    return report()


if __name__ == "__main__":
    sys.exit(main())
