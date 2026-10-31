"""Auto-queue primitive (P3 Craft 4, M-C): workflow-level batch repetition.

The experiment bench covers hyper-parameter sweeps *inside* one node
execution (``CdlGridSearchTrain``).  But some scans are workflow-level: a
generation pipeline whose seed/prompt changes per run, or a training graph
whose results must land in the normal output pipeline each round.  For
those, a node needs to ask the *server* to re-queue the very same prompt
when it finishes - the ComfyLab "QUEUE_STATUS self-loop" idea (survey D5),
rebuilt without monkeypatching:

* **Node side** (:class:`CdlBatchLoop`, ``comfy_extras/nodes_autoqueue.py``):
  on every execution it registers one iteration against a ``chain_id`` -
  ``register_iteration`` hands back the 1-based index, whether it is the
  final round, and a readable status.  While rounds remain, the node calls
  :func:`note_requeue`, recording the intent for the *current* prompt.
* **Worker side** (``main.py:prompt_worker``): after a successful
  ``execute()``, the worker calls :func:`take_requeue`; a recorded intent
  means the original queue item is put back - same prompt, same
  extra_data - and the graph runs again with the next iteration index.
  A failed or interrupted execution is never re-queued (an abort must
  actually abort).

Semantics and limits (documented on the node too):

* **Counters are in-memory** (server lifetime) and keyed by ``chain_id``
  plus a signature of the loop configuration - editing ``iterations`` or
  the chain id transparently starts a new chain at 1.  No persistence: a
  restart clears the bookkeeping, which is the safest possible recovery.
* **One intent at a time**: the pending slot is a single value guarded by
  a lock - the worker thread is the only consumer, so there is nothing to
  fan out.  ``take_requeue`` is idempotent by construction (consume ==
  clear).
* **No QUEUE_STATUS custom type**: the ComfyLab output-type protocol is
  not needed - the node exposes ``status``/``index``/``is_final`` as
  ordinary outputs, and ``is_final`` feeds a ``CdlOutputGate`` (an
  ExecutionBlocker wrapper) for "save only the last round" graphs.

Layering: stdlib only - importable from anywhere, including the worker
thread before ComfyUI's heavy modules finish loading.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, Optional

_lock = threading.RLock()

#: chain_id -> (signature, completed_count).  A signature change (the user
#: edited the loop config) transparently restarts the chain at 0.
_counters: Dict[str, tuple] = {}

#: The requeue intent for the prompt currently executing (or None).  One
#: worker thread, one slot - a lock keeps it honest anyway.
_pending_requeue: Optional[str] = None


def _signature(chain_id: str, iterations: int) -> str:
    """Stable short digest of the loop config (config change = new chain)."""
    payload = repr((str(chain_id), int(iterations)))
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()


def register_iteration(chain_id: str, iterations: int) -> Dict[str, Any]:
    """Book one execution of the chain; returns the iteration verdict.

    Out: ``index`` (0-based round), ``iteration`` (1-based), ``iterations``,
         ``is_final`` (this round completes the chain) and ``status``
         (``"3/10"`` style, readable straight into a UI).
    """
    iterations = max(1, int(iterations))
    chain_id = str(chain_id)
    sig = _signature(chain_id, iterations)
    with _lock:
        prev_sig, count = _counters.get(chain_id, (sig, 0))
        if prev_sig != sig:
            count = 0  # config changed: the chain restarts
        index = count
        _counters[chain_id] = (sig, count + 1)
    return {
        "index": index,
        "iteration": index + 1,
        "iterations": iterations,
        "is_final": index + 1 >= iterations,
        "status": f"{index + 1}/{iterations}",
    }


def reset(chain_id: str) -> None:
    """Forget a chain's counter (the ``CdlBatchLoop`` node's restart story)."""
    with _lock:
        _counters.pop(str(chain_id), None)


def note_requeue(chain_id: str) -> None:
    """Record the requeue intent for the prompt currently executing.

    Called by the node *inside* the worker's execute() - the worker picks it
    up after the run succeeds.  One slot: a later call simply overwrites.
    """
    global _pending_requeue
    with _lock:
        _pending_requeue = str(chain_id)


def take_requeue() -> Optional[str]:
    """Consume the pending requeue intent (worker-side, once per prompt)."""
    global _pending_requeue
    with _lock:
        pending = _pending_requeue
        _pending_requeue = None
        return pending


def describe() -> Dict[str, Any]:
    """Live bookkeeping snapshot (debugging + future panel integration)."""
    with _lock:
        return {
            "chains": {
                chain: {"count": count, "signature": sig}
                for chain, (sig, count) in _counters.items()
            },
            "pending_requeue": _pending_requeue,
        }
