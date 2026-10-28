"""Training checkpoint helper: mid-node continuation after an interrupt.

git-fire M3 (2026-10-13).  When a long training node is interrupted, the
host's lazy crash-site capture cannot help - the training node's output was
never produced (the loop died mid-flight), so nothing was captured and a
resume re-executes the node from scratch.  Instead, the NODE saves its own
complete training state at the moment the interrupt exception tears through
its stack frame, and continues from there when the same prompt runs again.

Identity & freshness (the two-key scheme)
-----------------------------------------
* ``run_key`` - a sha256 over the run's identity (widget values, input
  tensor byte digests, node id) locates the checkpoint file.  Same inputs
  and same node = same file.
* ``prompt_id`` - the checkpoint records the prompt that was interrupted.
  Crash site's resume RE-QUEUES THE SAME prompt_id, so a resumed run finds
  its own checkpoint and continues from the breakpoint step; a fresh Run
  carries a new prompt_id, the stored checkpoint is recognized as stale and
  discarded - the user never accidentally continues old progress.

The current prompt_id/node_id come from
``comfy_execution.utils.get_executing_context``: the host wraps every node
FUNCTION call in that context, and it is None in bare execution (unit tests,
headless scripts) - in that case checkpointing is entirely disabled (graceful
degradation; plain-node tests keep passing unchanged).

Storage: ``<user>/comfydl/checkpoints/<run_key>.pt`` (torch.save payload
dict), written atomically (tmp + os.replace).  Every save also prunes orphan
checkpoints older than ``PRUNE_DAYS`` - interrupted runs that never get
resumed must not accumulate forever.

Scope guard: only interrupts are checkpointed, never arbitrary node errors -
a deterministic error would just replay on resume, so continuing from it is
pointless.  Hard process death is out of scope (same trade as the crash
site's lazy capture).

Master switch (2026-10-28): the whole helper consults
``comfy.crashsite_switch.is_master_enabled()``.  When the crash site is
disabled every entry point short-circuits *before* doing any work - most
importantly ``run_key`` skips the tensor SHA256 digests (a CPU copy +
hash of the full training tensors, done once per training execution), and
``save`` / ``load`` / ``discard`` become no-ops.  The trainer nodes need no
changes: they hold an empty run key and every helper answers "nothing to do".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PRUNE_DAYS = 7


def _master_on() -> bool:
    """The crash site master switch (cheap, no torch import)."""
    try:
        from comfy.crashsite_switch import is_master_enabled

        return is_master_enabled()
    except Exception:  # noqa: BLE001 - switch must never break training
        return True


def checkpoints_root() -> Path:
    """``<user>/comfydl/checkpoints/`` - same user-dir convention as snapshots."""
    from folder_paths import get_user_directory

    root = Path(get_user_directory()) / "comfydl" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def current_prompt_id() -> Optional[str]:
    """The executing prompt's id, or None outside a real host execution."""
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        return ctx.prompt_id if ctx is not None else None
    except Exception:  # noqa: BLE001 - no host, no checkpointing
        return None


def current_node_id() -> str:
    """The executing node's id, or "" outside a real host execution."""
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        return str(ctx.node_id) if ctx is not None else ""
    except Exception:  # noqa: BLE001
        return ""


def _digest(value: Any) -> Any:
    """JSON-able canonical form; tensors become (shape, dtype, sha256)."""
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return {"__tensor__": [list(t.shape), str(t.dtype),
                               hashlib.sha256(t.numpy().tobytes())
                               .hexdigest()]}
    if isinstance(value, dict):
        return {str(k): _digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_digest(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)[:200]


def run_key(parts: Any) -> str:
    """Stable sha256 identity for one training run (see module docstring).

    Returns ``""`` when the crash site is master-disabled - callers keep an
    empty key and every other helper short-circuits on it.
    """
    if not _master_on():
        return ""
    canonical = json.dumps(_digest(parts), sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _path(run_key: str) -> Path:
    safe = "".join(c for c in run_key if c.isalnum() or c in "-._")
    return checkpoints_root() / f"{safe}.pt"


def _prune() -> None:
    """Best-effort cleanup of checkpoints nobody came back for."""
    try:
        deadline = time.time() - PRUNE_DAYS * 86400
        for f in checkpoints_root().glob("*.pt"):
            try:
                if f.stat().st_mtime < deadline:
                    f.unlink()
            except OSError:
                continue
    except Exception:  # noqa: BLE001 - hygiene must never break training
        pass


def save(run_key: str, payload: Dict[str, Any]) -> bool:
    """Atomically store one checkpoint (interrupt path).

    Returns False (and does nothing) when there is no host context - bare
    execution cannot be resumed, so there is nothing to save for.  Also False
    when the crash site is master-disabled (nothing is ever resumed then).
    """
    if not _master_on():
        return False
    prompt_id = current_prompt_id()
    if prompt_id is None:
        return False
    import torch

    root = checkpoints_root()
    target = _path(run_key)
    tmp = target.with_suffix(".pt.tmp")
    torch.save({"run_key": run_key, "prompt_id": prompt_id,
                "saved_at": int(time.time()), "payload": payload}, tmp)
    os.replace(tmp, target)
    _prune()
    return True


def load(run_key: str) -> Optional[Dict[str, Any]]:
    """Load the checkpoint for ``run_key`` if it belongs to THIS prompt.

    A stored checkpoint whose ``prompt_id`` differs from the executing one
    is stale (a fresh Run, not a resume): it is deleted and None returned,
    so the caller trains from scratch.  None also when the crash site is
    master-disabled - no resume, no lookup, no disk touch.
    """
    if not _master_on():
        return None
    prompt_id = current_prompt_id()
    if prompt_id is None:
        return None
    path = _path(run_key)
    if not path.is_file():
        return None
    try:
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or data.get("run_key") != run_key:
            raise ValueError("foreign checkpoint file")
        if data.get("prompt_id") != prompt_id:
            path.unlink(missing_ok=True)
            return None
        payload = data.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - a broken file must not kill training
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def discard(run_key: str) -> None:
    """Remove the checkpoint after a run finished (including early stop).

    No-op when the crash site is master-disabled (nothing was ever written).
    """
    if not _master_on() or not run_key:
        return
    try:
        _path(run_key).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def describe(payload: Dict[str, Any]) -> str:
    """One-line human summary of a checkpoint (logging/report use)."""
    return f"step {payload.get('step')}"


def prune_older_than(days: int) -> List[str]:
    """Explicit cleanup utility (tests / manual hygiene); returns removed names."""
    removed: List[str] = []
    deadline = time.time() - days * 86400
    for f in checkpoints_root().glob("*.pt"):
        try:
            if f.stat().st_mtime < deadline:
                f.unlink()
                removed.append(f.name)
        except OSError:
            continue
    return removed
