"""Experiment bench nodes (P3 Craft 2, M-C; ADR-D10).

``CdlGridSearchTrain`` is the production facade of the experiment bench: a
JSON ``param_grid`` drives a cartesian sweep of training hyper-parameters,
every combination runs through the *same* verified machinery as
``TrainingLoop`` (``comfy.training_core.TrainingSession`` - fresh model per
combination, the same early stopping, the same best-snapshot rollback, the
same crash-site checkpointing), and each combination lands as one run in the
runs ledger (``comfy.runs_ledger``, three-layer ADR-D10 schema).

Two deliberate departures from the EternalKernel reference
(``GridSearchTraining``, survey D2):

* **direction is explicit** - the reference hardcoded ``<`` (minimize) and
  even compared against a minimum that included the current run itself
  (``pytorch_nodes.py:1266-1272``, a comparison that can never fire).  Here
  ``direction`` is a ``min``/``max`` combo and the winner is picked once,
  after all combinations finished, from the final (early-stop-truncated)
  loss.
* **the winner is re-verified on the ledger** - ``mark_best`` flips the
  ``best`` flag on the winning run's detail + index row after the scan, so
  the comparison view (Craft 3) can highlight it without recomputing.

Grid keys (whitelist): ``hidden`` (layer widths string), ``activation``,
``loss``, ``steps``, ``batch_size``, ``lr`` (folds into the ``OPTIMIZER``
config via ``dataclasses.replace`` - the config is frozen by design).
Everything else on the node are the *base* settings a grid value overrides,
so sweeping one knob never means retyping the rest.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from typing_extensions import override

import torch

import comfy.utils
from comfy import runs_ledger as rl
from comfy import training_core as tc
from comfy import training_protocol as tp
from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Training"

#: Widget keys a param_grid entry may override (everything else rejected
#: loudly - a typo'd key silently training the same thing 8 times is the
#: worst failure mode a scanner can have).
GRID_KEYS: tuple[str, ...] = (
    "hidden", "activation", "loss", "steps", "batch_size", "lr",
)

#: Directly runnable default: two widths x two learning rates, small enough
#: that a first run finishes in seconds (open-box rule).
DEFAULT_PARAM_GRID = '{"hidden": ["8", "16,8"], "lr": [0.01, 0.003]}'

#: Base settings every combination starts from (grid values override these).
_BASE_KEYS = ("hidden", "activation", "loss", "steps", "batch_size")


def expand_param_grid(param_grid: str) -> list[dict]:
    """Expand the JSON ``param_grid`` into the list of combination dicts.

    Cartesian product over ``{key: [values]}``, insertion order preserved so
    a scan is reproducible run-for-run.  Raises ``ValueError`` with a readable
    message on invalid JSON, unknown keys, non-list values or an empty grid.
    """
    try:
        grid = json.loads(param_grid)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"param_grid is not valid JSON: {exc} (expected e.g. "
            f'{DEFAULT_PARAM_GRID})'
        ) from exc
    if not isinstance(grid, dict) or not grid:
        raise ValueError(
            "param_grid must be a non-empty JSON object mapping a key to a "
            f"list of values; allowed keys: {', '.join(GRID_KEYS)}.")
    for key, values in grid.items():
        if key not in GRID_KEYS:
            raise ValueError(
                f"param_grid key '{key}' is not sweepable; allowed keys: "
                f"{', '.join(GRID_KEYS)}.")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"param_grid['{key}'] must be a non-empty list of values.")
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _effective_base_settings(
    *, hidden: str, activation: str, loss: str, steps: int, batch_size: int,
    combo: dict,
) -> dict:
    """Base widget settings overridden by one grid combination."""
    effective = {
        "hidden": hidden,
        "activation": activation,
        "loss": loss,
        "steps": int(steps),
        "batch_size": int(batch_size),
    }
    for key in _BASE_KEYS:
        if key in combo:
            effective[key] = combo[key] if key in ("hidden", "activation", "loss") \
                else int(combo[key])
    return effective


class CdlGridSearchTrain(io.ComfyNode):
    """Sweep training hyper-parameters over a JSON grid, ledger every run.

    What: the experiment bench.  Expands ``param_grid`` into a cartesian set
          of combinations; for each one builds a *fresh* network (seed =
          ``seed + combo_index`` so runs differ only where the grid says)
          and trains it with exactly the ``Training Loop`` machinery
          (early stopping, best-snapshot rollback, per-combination crash
          checkpoint).  Every combination is recorded as one run in the
          runs ledger (``user/comfydl/runs/``; the ``comfydl.runs`` switch
          gates the recording).
    In:   x / y (TENSOR) - same contract as ``Training Loop`` (last
          dimension = features; ``cross_entropy`` takes class indices or
          one-hot).
          optimizer (OPTIMIZER) / scheduler (SCHEDULER, optional) /
          params (PARAMS, optional warm start) - same slots as
          ``Training Loop``; a grid ``lr`` value overrides the optimizer's
          learning rate per combination.
          Base widgets (a grid value overrides each): ``hidden`` widths,
          ``activation``, ``loss``, ``steps``, ``batch_size`` - identical
          semantics to ``Training Loop``.
          param_grid (STRING, JSON) - e.g.
          ``{"hidden": ["8", "16,8"], "lr": [0.01, 0.003]}``; allowed keys:
          hidden / activation / loss / steps / batch_size / lr.
          direction (COMBO) - ``min`` (default) or ``max``: which final
          loss is "better" (the EternalKernel fix).
          early_stop_patience / early_stop_min_delta - per combination.
          experiment_name (STRING) - groups the scan's runs in the ledger.
    Out:  params (PARAMS) - the *winning* combination's trained parameters.
          best_params (STRING) - JSON of the winning combination.
          loss_history (TENSOR) - the winner's (truncated) loss curve.
          summary (STRING) - the scan table with ``*`` on the winner.
    Caveats: the scan runs inside one node execution - interrupting stops
          the whole scan (the interrupted combination leaves a crash-site
          checkpoint keyed by its combo).  A combination that raises
          aborts the scan (same data, same code path: if one combination
          can fail, all can).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlGridSearchTrain",
            display_name="Grid Search Train",
            category=CATEGORY,
            description=(
                "Sweeps training hyper-parameters over a JSON param_grid, "
                "trains a fresh model per combination with the Training Loop "
                "machinery, records every run to the ledger and returns the "
                "winner."),
            search_aliases=[
                "grid search", "hyperparameter", "sweep", "experiment",
                "param grid", "scan", "train", "autotune",
            ],
            inputs=[
                io.Tensor.Input(
                    "x",
                    tooltip="Inputs, shape (N, in_features); the last dimension is the feature dimension.",
                ),
                io.Tensor.Input(
                    "y",
                    tooltip="Targets: (N, out_features) for mse/l1, or class indices / one-hot for cross_entropy.",
                ),
                io.Optimizer.Input(
                    "optimizer",
                    tooltip="Optimizer settings, linked from an Optimizer node; a grid 'lr' value overrides its learning rate.",
                ),
                io.Params.Input(
                    "params",
                    optional=True,
                    tooltip="Optional warm start applied to every combination's fresh model.",
                ),
                io.Scheduler.Input(
                    "scheduler",
                    optional=True,
                    tooltip="Optional LR scheduler settings, shared by every combination.",
                ),
                io.String.Input(
                    "hidden",
                    default=tc.DEFAULT_HIDDEN,
                    placeholder="e.g. 8 or 16,8 (base value; grid 'hidden' overrides)",
                    tooltip="Base hidden widths; a grid 'hidden' value overrides this per combination.",
                ),
                io.Combo.Input(
                    "activation",
                    options=list(tp.ACTIVATION_OPTIONS),
                    default="relu",
                    tooltip="Base activation; a grid 'activation' value overrides this.",
                ),
                io.Combo.Input(
                    "loss",
                    options=list(tc.LOSS_OPTIONS),
                    default="mse",
                    tooltip="Base loss; a grid 'loss' value overrides this.",
                ),
                io.Int.Input(
                    "steps",
                    default=30,
                    min=1,
                    max=100000,
                    step=1,
                    tooltip="Base optimizer steps per combination (keep scans small; a grid 'steps' value overrides this).",
                ),
                io.Int.Input(
                    "batch_size",
                    default=0,
                    min=0,
                    max=65536,
                    step=1,
                    tooltip="Base samples per step; 0 uses the whole dataset (a grid 'batch_size' value overrides this).",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    tooltip="Base seed; combination i trains with seed + i, so a scan is reproducible.",
                ),
                io.Int.Input(
                    "early_stop_patience",
                    default=0,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip="Early stopping per combination; 0 disables it.",
                ),
                io.Float.Input(
                    "early_stop_min_delta",
                    default=1e-4,
                    min=0.0,
                    max=1.0,
                    step=1e-4,
                    tooltip="Smallest improvement of the smoothed loss that resets the patience counter.",
                ),
                io.String.Input(
                    "param_grid",
                    default=DEFAULT_PARAM_GRID,
                    multiline=True,
                    placeholder='{"hidden": ["8", "16,8"], "lr": [0.01, 0.003]}',
                    tooltip="JSON grid {key: [values]}; keys: hidden / activation / loss / steps / batch_size / lr.",
                ),
                io.Combo.Input(
                    "direction",
                    options=["min", "max"],
                    default="min",
                    tooltip="Which final loss counts as better: min (default) or max.",
                ),
                io.String.Input(
                    "experiment_name",
                    default="gridsearch",
                    tooltip="Name grouping this scan's runs in the ledger and comparison view.",
                ),
            ],
            outputs=[
                io.Params.Output(display_name="params"),
                io.String.Output(display_name="best_params"),
                io.Tensor.Output(display_name="loss_history"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer,
        params=None,
        scheduler=None,
        hidden: str = tc.DEFAULT_HIDDEN,
        activation: str = "relu",
        loss: str = "mse",
        steps: int = 30,
        batch_size: int = 0,
        seed: int = 0,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 1e-4,
        param_grid: str = DEFAULT_PARAM_GRID,
        direction: str = "min",
        experiment_name: str = "gridsearch",
    ) -> io.NodeOutput:
        combos = expand_param_grid(param_grid)
        optimizer, scheduler = tc.resolve_optimizer_scheduler(optimizer, scheduler)

        from comfy.train_checkpoint import (
            current_node_id as ckpt_node_id, run_key as ckpt_run_key)

        experiment = str(experiment_name).strip() or "gridsearch"
        rows: list[dict] = []          # in-memory scan table for the summary
        best_score: float | None = None
        best_index: int | None = None
        best_result: tc.SessionResult | None = None
        best_combo: dict = {}
        sign = -1.0 if direction == "max" else 1.0

        scan_pbar = comfy.utils.ProgressBar(len(combos))
        print(
            f"[Network & Layers] Grid Search: {len(combos)} combination(s), "
            f"direction={direction}, seed base {seed}"
        )

        for index, combo in enumerate(combos):
            effective = _effective_base_settings(
                hidden=hidden, activation=activation, loss=loss,
                steps=steps, batch_size=batch_size, combo=combo)
            combo_optimizer = optimizer
            if "lr" in combo:
                combo_optimizer = dataclasses.replace(
                    optimizer, lr=float(combo["lr"]))
            combo_seed = int(seed) + index
            iterations = max(1, effective["steps"])

            with rl.start_run(
                experiment=experiment,
                params=combo,
                direction=direction,
                metric_name="loss",
                combo_index=index,
                seed=combo_seed,
            ) as run:
                pbar = comfy.utils.ProgressBar(iterations)

                def _progress(step, value, _run=run, _pbar=pbar):
                    _run.append_history(step, value)
                    _pbar.update(1)

                def _ckpt_key(data, _index=index, _combo=combo,
                              _opt=combo_optimizer, _seed=combo_seed,
                              _eff=effective):
                    # Per-combination key: two combinations must never share
                    # a crash checkpoint.
                    return ckpt_run_key({
                        "node": "CdlGridSearchTrain",
                        "node_id": ckpt_node_id(),
                        "combo": _combo,
                        "combo_index": _index,
                        "seed": _seed,
                        "widgets": [
                            _eff["hidden"], _eff["activation"], _eff["loss"],
                            _eff["steps"], _eff["batch_size"],
                            early_stop_patience, early_stop_min_delta,
                        ],
                        "optimizer": _opt,
                        "scheduler": scheduler,
                        "x": data.inputs, "y": data.targets,
                    })

                result = tc.TrainingSession(
                    x=x, y=y, loss=effective["loss"],
                    hidden=effective["hidden"],
                    activation=effective["activation"],
                    optimizer=combo_optimizer, scheduler=scheduler,
                    params=params, steps=effective["steps"],
                    batch_size=effective["batch_size"], seed=combo_seed,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    ckpt_key_fn=_ckpt_key, progress=_progress,
                    label="Grid Search",
                ).run()

            score = float(result.history[-1])
            rows.append({
                "run_id": run.run_id if run.run_id else f"combo-{index}",
                "experiment": experiment,
                "direction": direction,
                "metric_name": "loss",
                "status": "completed",
                "combo_index": index,
                "metric_final": score,
                "params_json": json.dumps(combo, sort_keys=True),
            })
            if best_score is None or sign * score < sign * best_score:
                best_score = score
                best_index = index
                best_result = result
                best_combo = combo
            scan_pbar.update(1)

        if best_index is not None:
            rl.mark_best(rows[best_index]["run_id"])
            rows[best_index]["best"] = 1
        summary = rl.render_experiment(rows)
        print(
            f"[Network & Layers] Grid Search: winner #{best_index} "
            f"{json.dumps(best_combo, sort_keys=True)} "
            f"loss {best_score:.6g} ({direction})"
        )
        return io.NodeOutput(
            tp.module_parameters(best_result.model),
            json.dumps(best_combo, sort_keys=True),
            best_result.history,
            summary,
        )


class ExperimentExtension(ComfyExtension):
    """Registers the experiment-bench node."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlGridSearchTrain]


async def comfy_entrypoint() -> ExperimentExtension:
    return ExperimentExtension()
