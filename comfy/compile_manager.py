"""Central torch.compile manager (dual-track M-B, SP-3; bigplan ADR-D4).

Every "compile this module" request in the codebase goes through here so
that:

* **compilation happens once** per ``(module, compile settings)`` pair for
  the lifetime of the compiled wrapper - the cache is keyed on the module
  identity plus the full compile-argument set, and holds *weak* references,
  so a compiled wrapper dies exactly when its last holder (a ModelPatcher's
  ``object_patches``) dies and is transparently recompiled afterwards;
* **no in-place rewrites happen** - the caller receives a compiled wrapper
  and is expected to attach it via ``ModelPatcher.add_object_patch`` (the
  WaveSpeed pattern), never ``setattr`` on a shared module (the
  ComfyUI-Torch-Compile defect this manager exists to avoid);
* **the feature meters itself** (meta-profiling rule, bigplan ADR-D1): every
  compilation's wall time is booked into the ``comfydl.compile`` switch via
  :func:`switch_registry.record_self_overhead`, and the switch hot-disables
  the manager (disabled -> the original module is returned untouched).

Known blind spot (documented, by design): compiled regions bypass
``TorchDispatchMode``, so the profiling run-meter cannot see individual ATen
calls inside them - attribution degrades to node granularity.  Opgraph
probe sessions should wrap their work in ``torch.compiler.disable`` (planned
for T6-2).

Layering: stdlib-only at import (``torch`` is imported lazily inside the
compile call), mirroring the proflog rule.

Public API
----------
- :func:`compile_module`   - the one entry point
- :func:`stats`            - cache diagnostics (entries, hits, misses)
- :func:`_reset_for_tests` - test isolation
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import Any, Dict, Optional, Tuple

from comfy.profiling import switch_registry

#: Registry id of the compile feature (Tier-1, self-metered).
SWITCH_ID = "comfydl.compile"

switch_registry.register_switch(
    SWITCH_ID,
    default=True,       # the node is opt-in per graph; the switch only
    hot_toggle=True,    # hot-disables compilation globally
    self_meter=True,
)

_LOCK = threading.RLock()


class _StrongRef:
    """Fallback for compile results that do not support weakrefs (the real
    ``torch.compile`` returns an ``OptimizedModule`` and always does)."""

    __slots__ = ("_obj",)

    def __init__(self, obj) -> None:
        self._obj = obj

    def __call__(self):
        return self._obj


#: ``(key) -> weakref.ref(compiled wrapper)``.
_CACHE: Dict[Tuple, Any] = {}
_HITS = 0
_MISSES = 0

#: The compile callable, indirected for tests (monkeypatch this).
_compile_fn = None


def _get_compile_fn():
    """``torch.compile``, imported lazily (and monkeypatchable in tests)."""
    global _compile_fn
    if _compile_fn is None:
        import torch

        _compile_fn = torch.compile
    return _compile_fn


def _cache_key(module, backend: str, mode: Optional[str], fullgraph: bool,
               dynamic: bool, disable: bool) -> Tuple:
    # id(module): identity, not equality - two structurally identical modules
    # must compile separately (weights differ); a dead module's id may be
    # reused, which the weakref lookup already guards against.
    return (id(module), backend, mode, fullgraph, dynamic, disable)


def compile_module(
    module: Any,
    *,
    backend: str = "inductor",
    mode: Optional[str] = None,
    fullgraph: bool = False,
    dynamic: bool = False,
    disable: bool = False,
) -> Any:
    """Compile ``module`` once per (module, settings); return the wrapper.

    Never raises for "compilation is off" - a disabled switch or an unknown
    backend degrades to returning ``module`` unchanged (the graph must keep
    running; a compile failure is the caller's error to surface).  Compile
    wall time is booked into the ``comfydl.compile`` switch.
    """
    global _HITS, _MISSES
    if not switch_registry.is_enabled(SWITCH_ID) or disable:
        return module
    key = _cache_key(module, backend, mode, fullgraph, dynamic, disable)
    with _LOCK:
        ref = _CACHE.get(key)
        cached = ref() if ref is not None else None
        if cached is not None:
            _HITS += 1
            return cached
    t0 = time.perf_counter()
    compiled = _get_compile_fn()(
        module, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    elapsed = time.perf_counter() - t0
    switch_registry.record_self_overhead(SWITCH_ID, elapsed)
    with _LOCK:
        _MISSES += 1
        try:
            _CACHE[key] = weakref.ref(compiled)
        except TypeError:  # pragma: no cover - exotic compile results
            _CACHE[key] = _StrongRef(compiled)
    return compiled


def stats() -> Dict[str, Any]:
    """``{"entries", "hits", "misses"}`` - diagnostics for panels/tests."""
    with _LOCK:
        return {
            "entries": len(_CACHE),
            "hits": _HITS,
            "misses": _MISSES,
        }


def _reset_for_tests() -> None:
    """Drop cache and counters (test isolation only; never call in product)."""
    global _HITS, _MISSES, _compile_fn
    with _LOCK:
        _CACHE.clear()
        _HITS = 0
        _MISSES = 0
        _compile_fn = None
