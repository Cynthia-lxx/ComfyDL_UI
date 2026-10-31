#!/usr/bin/env python
"""Standalone smoke test: the torch.compile stack (SP-3, M-B).

Run:  penv\\Scripts\\python.exe cdl_smoke_tests\\test_compile.py

Pins the promises of ``comfy/compile_manager.py`` (the central compile cache)
and the ``CdlCompileModel`` node that drives it:

* a disabled switch (or ``disable=True``) is a pure passthrough - the module
  comes back untouched and the cache stays empty;
* one compilation per ``(module, settings)`` pair: identical repeat requests
  are cache hits returning the same wrapper object, while a changed setting
  or a different module compiles separately;
* the cache holds weak references - dropping the compiled wrapper evicts the
  entry - with a strong-ref fallback for results that refuse weakrefs;
* compilation wall time is booked into the ``comfydl.compile`` switch
  (meta-profiling rule);
* the real ``torch.compile`` (eager backend, CPU) produces a callable wrapper
  whose forward matches the original module;
* the node attaches the compiled wrapper via ``add_object_patch`` on a *clone*
  (the WaveSpeed pattern): the incoming patcher is never mutated.
"""

import gc
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep every write (user-dir caches etc.) inside a scratch sandbox.
from run_smoke_test import _bootstrap as harness_bootstrap  # noqa: E402

harness_bootstrap(Path(tempfile.mkdtemp(prefix="cdl_compile_")))

_RESULTS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


from comfy import compile_manager as cm  # noqa: E402
from comfy.profiling import switch_registry as sr  # noqa: E402

SWITCH_ID = cm.SWITCH_ID
_ORIGINAL_ENABLED = sr.is_enabled(SWITCH_ID)
cm._reset_for_tests()


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Wrapper:
    """Minimal compiled-wrapper stand-in."""

    def __init__(self, inner):
        self.inner = inner


class _Recorder:
    """Stand-in for torch.compile: counts calls, returns a fresh wrapper."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_kwargs = {}

    def __call__(self, module, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return _Wrapper(module)


# ---------------------------------------------------------------------------
# T1: switch registration

check("T1 comfydl.compile switch registered", SWITCH_ID in sr.list_switches())

# ---------------------------------------------------------------------------
# T2: disabled switch is a passthrough

module_a = _Tiny()
sr.set_enabled(SWITCH_ID, False)
out = cm.compile_module(module_a)
check("T2a disabled switch returns the module untouched", out is module_a)
check("T2b disabled switch leaves the cache empty", cm.stats()["entries"] == 0,
      str(cm.stats()))

# ---------------------------------------------------------------------------
# T3: disable=True bypasses even an enabled switch

sr.set_enabled(SWITCH_ID, True)
out = cm.compile_module(module_a, disable=True)
check("T3 disable=True returns the module untouched", out is module_a)

# ---------------------------------------------------------------------------
# T4/T5: cache miss -> hit; identical settings return the same wrapper

rec = _Recorder()
cm._compile_fn = rec
wrapper1 = cm.compile_module(module_a)
wrapper2 = cm.compile_module(module_a)
check("T4a compiled once for repeated identical requests", rec.calls == 1,
      f"calls={rec.calls}")
check("T4b cache hit returns the same wrapper object", wrapper1 is wrapper2)
stats = cm.stats()
check("T5 cache counters: 1 miss + 1 hit",
      stats["misses"] == 1 and stats["hits"] == 1, str(stats))

# ---------------------------------------------------------------------------
# T6: changed settings compile separately

rec_b = _Recorder()
cm._compile_fn = rec_b
wrapper3 = cm.compile_module(module_a, fullgraph=True)
check("T6a changed settings trigger a new compilation", rec_b.calls == 1)
check("T6b settings live in the cache key", wrapper3 is not wrapper1)
check("T6c compile kwargs forwarded (fullgraph)",
      rec_b.last_kwargs.get("fullgraph") is True
      and rec_b.last_kwargs.get("backend") == "inductor",
      str(rec_b.last_kwargs))

# ---------------------------------------------------------------------------
# T7: module identity is part of the key

cm.compile_module(_Tiny())
check("T7 a different module compiles separately",
      cm.stats()["misses"] == 3, str(cm.stats()))

# ---------------------------------------------------------------------------
# T8: weakref eviction

rec_c = _Recorder()
cm._compile_fn = rec_c
holder = cm.compile_module(module_a, dynamic=True)
del holder
gc.collect()
# The dead weakref is purged lazily: the next request for the same key finds
# it dead and compiles again (this is also the id-reuse guard).
rec_c2 = _Recorder()
cm._compile_fn = rec_c2
fresh = cm.compile_module(module_a, dynamic=True)
check("T8a a dead entry is recompiled, not resurrected", rec_c2.calls == 1,
      f"calls={rec_c2.calls}")
check("T8b the fresh wrapper is a new object", isinstance(fresh, _Wrapper))
del fresh
gc.collect()

# ---------------------------------------------------------------------------
# T9: strong-ref fallback for non-weakref-able results

cm._compile_fn = lambda module, **kwargs: (module,)  # tuples refuse weakrefs
out_tuple = cm.compile_module(module_a, mode="reduce-overhead")
check("T9a tuple result cached via _StrongRef", isinstance(out_tuple, tuple))
check("T9b tuple cache hit returns the same object",
      cm.compile_module(module_a, mode="reduce-overhead") is out_tuple)

# ---------------------------------------------------------------------------
# T10: self-metering

overhead = sr.self_overhead(SWITCH_ID)
check("T10 compile wall time booked into the switch",
      overhead.get("samples", 0) >= 1, str(overhead))

# ---------------------------------------------------------------------------
# T11: the real torch.compile (eager backend) is callable and correct

cm._reset_for_tests()  # drops the fake _compile_fn too
tiny = _Tiny()
compiled = cm.compile_module(tiny, backend="eager")
x = torch.randn(2, 4)
with torch.no_grad():
    match = torch.allclose(compiled(x), tiny(x), atol=1e-5)
check("T11a real compile returns a wrapper", compiled is not tiny)
check("T11b compiled forward matches the original", match)

# ---------------------------------------------------------------------------
# T12: node registration + schema sanity

import asyncio  # noqa: E402

import nodes as host_nodes  # noqa: E402

asyncio.run(host_nodes.init_extra_nodes(init_custom_nodes=False,
                                        init_api_nodes=False))
registry = host_nodes.NODE_CLASS_MAPPINGS
check("T12a CdlCompileModel registered", "CdlCompileModel" in registry)

from comfy_extras.nodes_compile import CdlCompileModel  # noqa: E402

schema = CdlCompileModel.define_schema()
combo_inputs = {i.id: i for i in schema.inputs
                if getattr(i, "options", None) is not None}
check("T12b backend combo contains its default",
      "inductor" in combo_inputs["backend"].options)
check("T12c mode combo contains its default",
      "default" in combo_inputs["mode"].options)

# ---------------------------------------------------------------------------
# T13: node execute - add_object_patch on a clone, original untouched

from comfy.model_patcher import ModelPatcher  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = _Tiny()


net = _Net()
patcher = ModelPatcher(net, load_device=torch.device("cpu"),
                       offload_device=torch.device("cpu"))
rec_d = _Recorder()
cm._compile_fn = rec_d
result = CdlCompileModel.execute(patcher, backend="eager", mode="default")
out_patcher = result.result[0]
check("T13a node returns a new patcher (clone)", out_patcher is not patcher)
check("T13b compiled wrapper attached as object patch",
      "diffusion_model" in out_patcher.object_patches
      and isinstance(out_patcher.object_patches["diffusion_model"], _Wrapper))
check("T13c the incoming patcher was not mutated",
      "diffusion_model" not in patcher.object_patches)
check("T13d mode 'default' mapped to mode=None",
      rec_d.last_kwargs.get("mode") is None, str(rec_d.last_kwargs))

# ---------------------------------------------------------------------------
# T14: test hook leaves a clean state

cm._reset_for_tests()
sr.set_enabled(SWITCH_ID, _ORIGINAL_ENABLED)
check("T14 reset clears cache and counters",
      cm.stats() == {"entries": 0, "hits": 0, "misses": 0}, str(cm.stats()))


def _report() -> int:
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        line = f"  {flag}  {name}"
        if detail and not ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n=== compile stack test: {passed} PASS / {len(_RESULTS) - passed} FAIL "
          f"(of {len(_RESULTS)}) ===")
    return 1 if (len(_RESULTS) - passed) else 0


if __name__ == "__main__":
    raise SystemExit(_report())
