"""Pin the live progress reporting of the long-running nodes.

Usage:
    penv\\Scripts\\python.exe cdl_smoke_tests\\test_progress_reporting.py

Why this script exists
----------------------
ComfyUI renders a per-node progress bar when a node drives
``comfy.utils.ProgressBar``: the global hook (installed by ``main.py``) resolves
the *currently executing node* from ``comfy_execution.utils.get_executing_context``
and additionally performs the interrupt check, so a reported loop is both visible
in the UI and cancellable.  This script pins that contract for the loops that can
run for a long time:

* ``Training Loop``            - one update per optimizer step;
* ``Language Model: Train``    - one update per optimizer step;
* ``Language Model: Generate`` - one update per generated token, driven through
  the ``progress`` callback of ``comfy.lm_protocol.generate_tokens``;
* ``Sliding Window``           - one update per sample cut from the stream.

The node checks swap ``comfy.utils.ProgressBar`` for a recorder and assert the
advertised total and update count; a final check installs a real hook to prove
the values actually reach the callback the UI talks to.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))


# ---------------------------------------------------------------------------
# Registry bootstrap: the very same node set the dehydrated host loads.
import asyncio  # noqa: E402

import nodes  # noqa: E402

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))

import torch  # noqa: E402

import comfy.utils  # noqa: E402
from comfy import lm_protocol as mp  # noqa: E402
from comfy_extras import nodes_lm, nodes_nlp, nodes_training  # noqa: E402


# ---------------------------------------------------------------------------
# A recorder standing in for comfy.utils.ProgressBar.


class Recorder:
    """Drop-in replacement capturing how a node drives its progress bar."""

    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id
        self.updates = 0
        self.value = 0

    def update(self, value):
        self.updates += 1
        self.value += value

    def update_absolute(self, value, total=None, preview=None):
        self.updates += 1
        self.value = value
        if total is not None:
            self.total = total


_MADE: list[Recorder] = []
_REAL_PROGRESS_BAR = comfy.utils.ProgressBar


def _install_recorder() -> None:
    _MADE.clear()

    def factory(total, node_id=None):
        bar = Recorder(total, node_id)
        _MADE.append(bar)
        return bar

    comfy.utils.ProgressBar = factory


def _coarse_pair(samples: int = 6, window: int = 4, vocab: int = 16):
    """A valid ``(samples, window)`` context batch plus its next-token targets."""
    generator = torch.Generator().manual_seed(11)
    x = torch.randint(0, vocab, (samples, window), generator=generator).long()
    y = torch.randint(0, vocab, (samples,), generator=generator).long()
    return x, y


LM_SPEC = (mp.EmbeddingSpec(16, 8, True), mp.TransformerBlockSpec(8, 4, 16))


def _build_model():
    return mp.build_model(LM_SPEC, seed=0)


# ---------------------------------------------------------------------------
# T1: Training Loop reports one step at a time.

_install_recorder()
try:
    x, y = _coarse_pair(8, 3)
    targets = torch.randn(8, 1, generator=torch.Generator().manual_seed(3))
    nodes_training.TrainingLoop.execute(
        x=x,
        y=targets,
        optimizer=None,
        params=None,
        hidden="8",
        activation="relu",
        loss="mse",
        steps=25,
        batch_size=0,
        seed=0,
    )
    check(
        "T1 TrainingLoop creates exactly one progress bar",
        len(_MADE) == 1,
        f"got {len(_MADE)}",
    )
    bar = _MADE[0] if _MADE else None
    check(
        "T1 TrainingLoop advertises one unit per optimizer step",
        bar is not None and bar.total == 25,
        f"total={getattr(bar, 'total', None)} (expect 25)",
    )
    check(
        "T1 TrainingLoop updates once per step",
        bar is not None and bar.updates == 25,
        f"updates={getattr(bar, 'updates', None)} (expect 25)",
    )

    # ---- T2: Language Model Train -----------------------------------------
    _install_recorder()
    model = _build_model()
    lm_x, lm_y = _coarse_pair()
    nodes_lm.LanguageModelTrain.execute(
        model=model,
        x=lm_x,
        y=lm_y,
        optimizer=None,
        steps=17,
        batch_size=0,
        seed=0,
    )
    bar = _MADE[0] if _MADE else None
    check(
        "T2 LanguageModelTrain drives one bar of exactly `steps` units",
        bar is not None and bar.total == 17 and bar.updates == 17,
        f"total={getattr(bar, 'total', None)}, updates={getattr(bar, 'updates', None)} (expect 17/17)",
    )

    # ---- T3: Language Model Generate --------------------------------------
    _install_recorder()
    model = _build_model()
    nodes_lm.LanguageModelGenerate.execute(
        model=model,
        vocab=None,
        prefix="the ",
        prefix_ids=torch.tensor([1, 2], dtype=torch.long),
        num_tokens=9,
        temperature=1.0,
        seed=0,
    )
    bar = _MADE[0] if _MADE else None
    check(
        "T3 LanguageModelGenerate drives one bar of exactly `num_tokens` units",
        bar is not None and bar.total == 9 and bar.updates == 9,
        f"total={getattr(bar, 'total', None)}, updates={getattr(bar, 'updates', None)} (expect 9/9)",
    )

    # ---- T4: Sliding Window -----------------------------------------------
    _install_recorder()
    stream = torch.randint(0, 16, (30,), generator=torch.Generator().manual_seed(5)).long()
    window = 4
    nodes_nlp.TextSlidingWindow.execute(ids=stream, window=window)
    bar = _MADE[0] if _MADE else None
    expected = 30 - window
    check(
        "T4 SlidingWindow advertises one unit per sample (stream - window)",
        bar is not None and bar.total == expected,
        f"total={getattr(bar, 'total', None)} (expect {expected})",
    )
    check(
        "T4 SlidingWindow updates once per sample",
        bar is not None and bar.updates == expected,
        f"updates={getattr(bar, 'updates', None)} (expect {expected})",
    )
finally:
    comfy.utils.ProgressBar = _REAL_PROGRESS_BAR


# ---------------------------------------------------------------------------
# T5: the protocol-level callback itself (no node involved).

_steps: list[tuple[int, int]] = []
mp.generate_tokens(
    _build_model(),
    [1, 2, 3],
    num_tokens=5,
    progress=lambda done, total: _steps.append((done, total)),
)
check(
    "T5 generate_tokens reports every generated token in order",
    _steps == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)],
    str(_steps),
)
check(
    "T5 generate_tokens works without a callback",
    mp.generate_tokens(_build_model(), [1], num_tokens=2) is not None,
)

# ---------------------------------------------------------------------------
# T6: a real hook sees the values (this is what main.py installs).

_seen: list[tuple[float, float, object]] = []


def _hook(value, total, preview_image, prompt_id=None, node_id=None):
    _seen.append((value, total, node_id))


comfy.utils.set_progress_bar_global_hook(_hook)
try:
    x, y = _coarse_pair(8, 3)
    targets = torch.randn(8, 1, generator=torch.Generator().manual_seed(3))
    nodes_training.TrainingLoop.execute(
        x=x,
        y=targets,
        optimizer=None,
        params=None,
        hidden="8",
        activation="relu",
        loss="mse",
        steps=12,
        batch_size=0,
        seed=0,
    )
finally:
    comfy.utils.set_progress_bar_global_hook(None)

check("T6 the global hook is called at least once", len(_seen) >= 1, f"{len(_seen)} call(s)")
check(
    "T6 the hook receives the final value == total (12)",
    bool(_seen) and _seen[-1][0] == 12 and _seen[-1][1] == 12,
    str(_seen[-1] if _seen else None),
)

# ---------------------------------------------------------------------------
# Report

print()
print("=== progress reporting ===")
_failed = 0
for _name, _ok, _detail in _RESULTS:
    if _ok:
        print(f"  PASS  {_name}")
    else:
        _failed += 1
        print(f"  FAIL  {_name}  {_detail}")
print(f"=== progress result: {len(_RESULTS) - _failed} PASS, {_failed} FAIL "
      f"(of {len(_RESULTS)}) ===")

raise SystemExit(1 if _failed else 0)
