"""Four-level verbosity for the ComfyDL profiling tools (reform: Profiling).

The profiling panel used to be a black box: when Analyze spun forever or the
expand button did nothing, there was nothing to look at. This module gives the
whole profiling stack a pip-style verbosity dial — ``off`` (default, zero
output) / ``low`` / ``medium`` / ``high`` — so a user can run once at ``high``
and paste the log for diagnosis.

Levels
------
* ``off``    - the shipped default: not a single line, zero overhead;
* ``low``    - request lifecycle: start/done + duration per endpoint,
               aggregates (probed/fallback counts), client errors;
* ``medium`` - per-node probe results (status / FLOPs / probe_ms / error) and
               opgraph cache hits - enough to see *which* node misbehaves;
* ``high``   - everything else: payload sizes, cache keys, frontend state
               transitions, watchdog polls.

Configuration (three channels, take the loudest)
------------------------------------------------
1. Startup flag: ``--cdl-profiling-log {off,low,medium,high}`` (default off),
   defined in ``comfy/cli_args.py`` and read lazily via ``comfy.cli_args.args``
   - this module deliberately stays stdlib-only (importing it must never drag
   in torch: ``main.py`` forbids importing torch before its own setup).
2. Runtime override: :func:`set_level` (tests, embedders).
3. Per-request header: ``X-CDL-Profiling-Log`` sent by the frontend panel
   (settings-driven, no restart needed) - can only *raise* the level.

Output goes through the standard ``logging`` machinery (logger
``ComfyDL.profiling`` at INFO), so lines automatically land in the app-wide
ring buffer (``app/logger.py``) and are visible through
``GET /internal/logs/raw`` - the panel's own "server log" card reads that.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

#: Ordered verbosity ranks; index = loudness.
LEVELS = ("off", "low", "medium", "high")
_RANK = {name: idx for idx, name in enumerate(LEVELS)}

#: The per-request header the frontend sends to raise the backend verbosity.
HEADER = "X-CDL-Profiling-Log"

_logger = logging.getLogger("ComfyDL.profiling")
_lock = threading.Lock()
_override: Optional[str] = None  # set_level() runtime override (tests/embedders)


def _startup_level() -> str:
    """The level requested on the command line (lazy: no torch, no import cost)."""
    try:
        from comfy import cli_args

        level = getattr(cli_args.args, "cdl_profiling_log", "off")
    except Exception:  # pragma: no cover - cli_args always exists in-process
        return "off"
    return level if level in _RANK else "off"


def set_level(level: Optional[str]) -> None:
    """Runtime override; ``None`` clears it back to the startup value."""
    global _override
    if level is None:
        with _lock:
            _override = None
        return
    if level not in _RANK:
        raise ValueError(f"unknown profiling log level: {level!r} (choose from {LEVELS})")
    with _lock:
        _override = level


def get_level() -> str:
    """The effective base level: runtime override, else the startup flag."""
    if _override is not None:
        return _override
    return _startup_level()


def level_for(request: Any = None) -> str:
    """The level for one request: max(base level, ``X-CDL-Profiling-Log``).

    The header can only *raise* the verbosity (a user diagnosing from the
    browser should not need a server restart; a quiet deployment should not
    get noisy because one client sends a bogus header).
    """
    base = get_level()
    if request is not None:
        try:
            asked = request.headers.get(HEADER, "")
        except Exception:
            asked = ""
        if asked in _RANK and _RANK[asked] > _RANK[base]:
            return asked
    return base


def is_enabled(rank: str, request: Any = None) -> bool:
    """Cheap gate for callers that build expensive payloads before logging."""
    want = _RANK.get(rank)
    if want is None:
        return False
    return _RANK[level_for(request)] >= want


# --- master switch (Profiling v2 post-plan, 2026-10-07) ---------------------
#
# Repeated measurements showed profiling's mere presence slows every run
# (frontend graphToPrompt hijack + auto-estimate, backend watchdog sampling,
# per-run meter bookkeeping), so the user gets a single switch that detaches
# the whole stack from the run path and re-attaches it on demand. This lives
# in the comfy layer (stdlib-only) because runmeter.py must consult it while
# the app layer must write it - the layering rule forbids the reverse.
#
# * ``--cdl-profiling-disable`` hard-disables: the flag is set once at import
#   time from the CLI args and :func:`set_master_enabled` then refuses to
#   re-enable (only a restart without the flag can).
# * Soft disable/enable is the hot path used by the Settings toggle: no
#   restart, executor untouched - meter_from_args()/persist_run() just short-
#   circuit.

_hard_disabled = False
_master_enabled = True
_hard_checked = False


def _check_hard_disabled() -> None:
    """One-time lazy read of ``--cdl-profiling-disable`` (no torch import)."""
    global _hard_disabled, _hard_checked
    if _hard_checked:
        return
    _hard_checked = True
    try:
        from comfy import cli_args

        _hard_disabled = bool(getattr(cli_args.args, "cdl_profiling_disable", False))
    except Exception:  # pragma: no cover - cli_args always exists in-process
        _hard_disabled = False
    if _hard_disabled:
        _master_enabled = False


def is_master_enabled() -> bool:
    """Whether profiling participates in the run path at all."""
    _check_hard_disabled()
    return _master_enabled


def is_hard_disabled() -> bool:
    """True only when ``--cdl-profiling-disable`` is on the command line."""
    _check_hard_disabled()
    return _hard_disabled


def set_master_enabled(enabled: bool) -> bool:
    """Soft enable/disable (hot path). Returns the resulting state.

    Refuses to re-enable a hard-disabled process - that needs a restart
    without ``--cdl-profiling-disable``.
    """
    global _master_enabled
    _check_hard_disabled()
    if _hard_disabled:
        return False
    with _lock:
        _master_enabled = bool(enabled)
    return _master_enabled


def log(rank: str, message: str, *args: Any, request: Any = None) -> None:
    """Emit ``message % args`` tagged with ``rank`` when verbosity allows.

    ``request`` is keyword-only so call-site format arguments can never be
    silently swallowed by it. Never raises: a logging bug must not break a
    profiling request (same courtesy rule as the panel itself).
    """
    try:
        if not is_enabled(rank, request):
            return
        text = message % args if args else message
        _logger.info("[profiling:%s] %s", rank, text)
    except Exception:  # noqa: BLE001 - logging is a courtesy, not a contract
        pass
