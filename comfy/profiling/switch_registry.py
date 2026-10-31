"""Central switch registry for ComfyDL run-path features (dual-track P0).

Historically every run-path feature grew its own ad-hoc master switch
(``comfy/profiling/proflog.py`` for the profiling stack, 2026-10-07;
``comfy/crashsite_switch.py`` for the crash site, 2026-10-28) - same two-layer
semantics (lazy one-time hard disable from CLI flags + hot soft toggle),
duplicated code, no common place to see them all.  This module is the
generalisation those two pioneered:

* One registry entry per feature: ``{id, default, hot_toggle, self_meter}``.
* Two-layer semantics, byte-compatible with the pioneers:
  - ``hard_disable_check`` is a lazy, one-time probe (usually reading
    ``comfy.cli_args``); once it fires the switch is off for the process
    lifetime and :func:`set_enabled` refuses to re-enable.
  - the soft toggle is the hot path: no restart, no executor surgery.
* **Meta-profiling rule** (bigplan ADR-D1): every Tier-1 feature must measure
  its own overhead via :func:`record_self_overhead`; the settings panel reads
  it back with :func:`self_overhead`.  A switch that cannot account for its
  own cost does not ship.

Layering: stdlib-only (like proflog), because the comfy layer consults
switches while the app layer writes them - the rule forbids the reverse.

Public API
----------
- :func:`register_switch`  - declare a feature switch (idempotent)
- :func:`is_enabled` / :func:`set_enabled` / :func:`is_hard_disabled`
- :func:`record_self_overhead` / :func:`self_overhead`
- :func:`describe` / :func:`list_switches` - settings-panel facing
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

_LOCK = threading.RLock()


class _SwitchState:
    """Mutable state behind one registered switch (registry-internal)."""

    __slots__ = (
        "switch_id", "default", "hot_toggle", "self_meter",
        "hard_disable_check", "hard_disabled", "hard_checked", "enabled",
        "overhead_total_s", "overhead_samples",
    )

    def __init__(
        self,
        switch_id: str,
        default: bool,
        hot_toggle: bool,
        self_meter: bool,
        hard_disable_check: Optional[Callable[[], bool]],
    ) -> None:
        self.switch_id = switch_id
        self.default = default
        self.hot_toggle = hot_toggle
        self.self_meter = self_meter
        self.hard_disable_check = hard_disable_check
        self.hard_disabled = False
        self.hard_checked = False
        self.enabled = default
        self.overhead_total_s = 0.0
        self.overhead_samples = 0


_REGISTRY: Dict[str, _SwitchState] = {}


def register_switch(
    switch_id: str,
    *,
    default: bool = True,
    hot_toggle: bool = True,
    self_meter: bool = False,
    hard_disable_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Declare a feature switch.

    Idempotent: re-registering the exact same signature is a no-op (safe under
    module re-import / hot reload); a conflicting re-registration raises so a
    typo'd duplicate cannot silently shadow an existing switch.

    ``hard_disable_check`` is probed lazily exactly once, at the first
    ``is_enabled`` / ``set_enabled`` / ``is_hard_disabled`` call - never at
    import time (mirrors the pioneers' no-import-cost rule).
    """
    with _LOCK:
        existing = _REGISTRY.get(switch_id)
        if existing is not None:
            same = (
                existing.default == default
                and existing.hot_toggle == hot_toggle
                and existing.self_meter == self_meter
                and existing.hard_disable_check == hard_disable_check
            )
            if not same:
                raise ValueError(
                    f"switch {switch_id!r} already registered with a "
                    f"different signature")
            return
        _REGISTRY[switch_id] = _SwitchState(
            switch_id, default, hot_toggle, self_meter, hard_disable_check)


def _get(switch_id: str) -> _SwitchState:
    state = _REGISTRY.get(switch_id)
    if state is None:
        raise KeyError(
            f"unknown switch {switch_id!r} - register it first "
            f"(registered: {sorted(_REGISTRY)})")
    return state


def _ensure_hard_checked(state: _SwitchState) -> None:
    """One-time lazy hard-disable probe (never raises, never imports early)."""
    if state.hard_checked:
        return
    state.hard_checked = True
    if state.hard_disable_check is None:
        return
    try:
        if bool(state.hard_disable_check()):
            state.hard_disabled = True
            state.enabled = False
    except Exception:  # noqa: BLE001 - a broken probe must not kill the run path
        state.hard_disabled = False


def is_enabled(switch_id: str) -> bool:
    """Whether the feature participates in the run path at all."""
    with _LOCK:
        state = _get(switch_id)
        _ensure_hard_checked(state)
        return state.enabled


def set_enabled(switch_id: str, enabled: bool) -> bool:
    """Soft enable/disable (hot path). Returns the resulting state.

    Refuses to re-enable a hard-disabled process - that needs a restart
    without the hard flag (same contract as the pioneer switches).
    """
    with _LOCK:
        state = _get(switch_id)
        _ensure_hard_checked(state)
        if state.hard_disabled:
            return False
        state.enabled = bool(enabled)
        return state.enabled


def is_hard_disabled(switch_id: str) -> bool:
    """True only when the switch's hard-disable probe fired for this process."""
    with _LOCK:
        state = _get(switch_id)
        _ensure_hard_checked(state)
        return state.hard_disabled


def record_self_overhead(switch_id: str, seconds: float) -> None:
    """Accumulate the wall time this feature's own hooks consumed (ADR-D1).

    Courtesy rule: never raises - bookkeeping must not break a run.  Unknown
    ids are ignored (they cannot happen for a registered feature, and a
    stray measurement from a torn-down test must not explode).
    """
    with _LOCK:
        state = _REGISTRY.get(switch_id)
        if state is None:
            return
        try:
            state.overhead_total_s += float(seconds)
            state.overhead_samples += 1
        except (TypeError, ValueError):
            return


def self_overhead(switch_id: str) -> Dict[str, Any]:
    """``{"total_s": float, "samples": int}`` - the settings panel's meter."""
    with _LOCK:
        state = _get(switch_id)
        return {
            "total_s": state.overhead_total_s,
            "samples": state.overhead_samples,
        }


def describe(switch_id: str) -> Dict[str, Any]:
    """Read-only snapshot for the settings panel / diagnostics endpoints."""
    with _LOCK:
        state = _get(switch_id)
        _ensure_hard_checked(state)
        return {
            "id": state.switch_id,
            "default": state.default,
            "hot_toggle": state.hot_toggle,
            "self_meter": state.self_meter,
            "enabled": state.enabled,
            "hard_disabled": state.hard_disabled,
            "self_overhead": {
                "total_s": state.overhead_total_s,
                "samples": state.overhead_samples,
            },
        }


def list_switches() -> List[str]:
    """Registered switch ids in registration order."""
    with _LOCK:
        return list(_REGISTRY)


def _force_state(
    switch_id: str,
    *,
    enabled: Optional[bool] = None,
    hard_disabled: Optional[bool] = None,
    hard_checked: Optional[bool] = None,
) -> None:
    """Test hook: set switch state directly, bypassing every rule.

    Product code must use :func:`set_enabled` (and let the hard-disable probe
    do its job); this exists so test suites can simulate CLI hard-disable
    flags without parsing argv inside a live process - the pioneer switch
    tests used to patch the module globals the same way.  Unknown ids raise
    ``KeyError`` (same contract as :func:`is_enabled`).
    """
    with _LOCK:
        state = _get(switch_id)
        if enabled is not None:
            state.enabled = bool(enabled)
        if hard_disabled is not None:
            state.hard_disabled = bool(hard_disabled)
        if hard_checked is not None:
            state.hard_checked = bool(hard_checked)


def _reset_for_tests() -> None:
    """Drop every entry (test isolation only; never call from product code)."""
    with _LOCK:
        _REGISTRY.clear()
