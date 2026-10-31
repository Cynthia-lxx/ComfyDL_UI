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

from comfy.profiling import switch_registry as _switch_registry

#: Registry id of the crash site master switch.
SWITCH_ID = "comfydl.crashsite"


def _hard_disable_probe() -> bool:
    """``--cdl-crashsite-disable`` OR profiling's hard flag (see docstring)."""
    try:
        from comfy import cli_args

        if bool(getattr(cli_args.args, "cdl_crashsite_disable", False)):
            return True
    except Exception:  # pragma: no cover - cli_args always exists in-process
        pass
    # Historical continuity: --cdl-profiling-disable hard-disables the
    # whole analysis stack AND the crash site's capture with it (server.py
    # M1 behaviour - see server.py's provider registration block).
    try:
        from comfy.profiling.proflog import is_hard_disabled as _prof_off

        return bool(_prof_off())
    except Exception:  # noqa: BLE001 - profiling may be absent
        return False


_switch_registry.register_switch(
    SWITCH_ID,
    default=True,
    hot_toggle=True,
    self_meter=True,
    hard_disable_check=_hard_disable_probe,
)


def is_hard_disabled() -> bool:
    """True when ``--cdl-crashsite-disable`` (or profiling's hard flag) is set."""
    return _switch_registry.is_hard_disabled(SWITCH_ID)


def is_master_enabled() -> bool:
    """Whether the crash site participates in the run path at all."""
    return _switch_registry.is_enabled(SWITCH_ID)


def set_master_enabled(enabled: bool) -> bool:
    """Soft enable/disable (hot path). Returns the resulting state.

    Refuses to re-enable a hard-disabled process - that needs a restart
    without the hard flag.
    """
    return _switch_registry.set_enabled(SWITCH_ID, enabled)
