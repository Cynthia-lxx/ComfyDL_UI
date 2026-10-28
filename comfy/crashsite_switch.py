"""ComfyDL Crash site master switch (2026-10-28).

The crash site hooks every executed prompt (snapshot capture, resume
accounting, training checkpoints).  That is the point of the feature, but a
user who wants *maximum production throughput* may want the whole stack to
step out of the run path entirely - the same argument that produced
profiling's master switch (``comfy/profiling/proflog.py``, 2026-10-07).  This
module is the crash site's counterpart.

Two layers, mirroring the profiling switch exactly:

* ``--cdl-crashsite-disable`` hard-disables: the flag is read once (lazily)
  from the CLI args and :func:`set_master_enabled` then refuses to re-enable -
  only a restart without the flag can bring the crash site back.  For
  historical continuity the profiling hard flag (``--cdl-profiling-disable``)
  ALSO hard-disables the crash site (server.py's M1 wiring treated the two as
  one "analysis off" switch; data-safety capture followed the analysis stack).
* The soft switch is the hot path used by the Settings toggle: no restart.
  Disabling UNREGISTERS the snapshot provider from
  ``comfy_execution.cache_provider``, so the executor's lifecycle hook
  (``_notify_prompt_lifecycle``) short-circuits on its empty-provider check
  and the caching layer never calls ``on_store`` / ``on_lookup`` again - from
  the run path's point of view the crash site is simply not there.

This lives in the comfy layer (stdlib-only) because ``train_checkpoint.py``
must consult it, and the layering rule forbids the comfy layer from
importing ``app``.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_hard_disabled = False
_hard_checked = False
_master_enabled = True


def _check_hard_disabled() -> None:
    """One-time lazy read of the hard-disable flags (no torch import)."""
    global _hard_disabled, _hard_checked
    if _hard_checked:
        return
    _hard_checked = True
    try:
        from comfy import cli_args

        _hard_disabled = bool(
            getattr(cli_args.args, "cdl_crashsite_disable", False))
    except Exception:  # pragma: no cover - cli_args always exists in-process
        _hard_disabled = False
    if not _hard_disabled:
        # Historical continuity: --cdl-profiling-disable hard-disables the
        # whole analysis stack AND the crash site's capture with it (server.py
        # M1 behaviour - see server.py's provider registration block).
        try:
            from comfy.profiling.proflog import is_hard_disabled as _prof_off

            _hard_disabled = bool(_prof_off())
        except Exception:  # noqa: BLE001 - profiling may be absent
            pass
    if _hard_disabled:
        global _master_enabled
        _master_enabled = False


def is_hard_disabled() -> bool:
    """True when ``--cdl-crashsite-disable`` (or profiling's hard flag) is set."""
    _check_hard_disabled()
    return _hard_disabled


def is_master_enabled() -> bool:
    """Whether the crash site participates in the run path at all."""
    _check_hard_disabled()
    return _master_enabled


def set_master_enabled(enabled: bool) -> bool:
    """Soft enable/disable (hot path). Returns the resulting state.

    Refuses to re-enable a hard-disabled process - that needs a restart
    without the hard flag.
    """
    global _master_enabled
    _check_hard_disabled()
    if _hard_disabled:
        return False
    with _lock:
        _master_enabled = bool(enabled)
    return _master_enabled
