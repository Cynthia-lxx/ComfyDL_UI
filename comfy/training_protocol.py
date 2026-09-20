"""Trainable-parameter protocol layer for the dehydrated ComfyUI build.

ComfyUI executes a prompt inside ``torch.inference_mode()`` (``execution.py``),
so no autograd graph can survive a node boundary: a tensor produced by one node
can never be differentiated by another. Everything that needs gradients -
forward, backward and ``optimizer.step()`` - therefore has to happen *inside a
single node*, which is exactly what upstream's ``TrainLoraNode`` does.

This module holds the pieces of that closure which are not node specific:

* :class:`OptimizerConfig` - the hyper-parameters an optimizer node publishes and
  a trainer consumes, plus :func:`build_optimizer` turning them into a real
  ``torch.optim.Optimizer``;
* :class:`MLP` - the tiny fully connected network a trainer trains, with the
  documented parameter naming convention (``layer0.weight``, ``layer0.bias``,
  ...) that makes the trained parameters addressable, saveable and reusable by
  the ordinary ``Basic`` layer nodes;
* parameter bookkeeping - reading a ``PARAMS`` payload, matching it against a
  module (``missing`` / ``skipped`` reporting instead of a silent drop) and
  flattening it back into saveable tensors;
* the text codec that encodes a ``PARAMS`` payload as a ``safetensors`` blob in
  base64, so a trained parameter set can be carried by a *widget* and therefore
  survive inside a saved workflow.

Only ``torch`` and ``safetensors`` are imported here: the module must stay
importable in the dehydrated build, which has neither ``comfy.ldm`` nor
``comfy.lora``. File system work (output folder, file names) deliberately lives
in the nodes, the same way ``nodes_model_merging`` keeps ``_save_bucket``.

Design decisions
----------------
* Deterministic by default. :func:`seeded_rng` seeds the process RNG, builds the
  module and restores the previous state (CPU *and* CUDA), so a training node
  reproduces the same result for the same ``seed`` without stealing the random
  stream of the other nodes in the graph.
* A ``PARAMS`` payload is ``{name: nn.Parameter}``. Anything else is coerced -
  a floating tensor is wrapped into a ``nn.Parameter``, an integer tensor or a
  non tensor entry is dropped *with a warning*, because silently keeping it
  would make ``requires_grad`` fail much later, far away from the cause.
* Warm starting never fails silently. :func:`load_into_module` only copies the
  entries whose shape matches the destination module and reports every missing
  and every skipped key back to the caller, which prints them.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import logging
from typing import Mapping, Sequence

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

#: Prefix of a submodule of :class:`MLP`; ``layer0`` is the first dense layer.
LAYER_PREFIX = "layer"

#: Suffixes of the two parameters of a dense layer, matching ``nn.Linear``.
WEIGHT_SUFFIX = "weight"
BIAS_SUFFIX = "bias"

#: Header of the widget-carried text form of a parameter set. Carries a version
#: so a future format change can be detected instead of misparsed.
PARAMS_TEXT_PREFIX = "CDLPARAMS1:"

#: Activation applied *between* the dense layers of an :class:`MLP`.
ACTIVATION_OPTIONS: tuple[str, ...] = ("relu", "gelu", "tanh", "sigmoid", "none")

#: Optimizers the ``Optimizer`` node can publish, in dropdown order.
OPTIMIZER_OPTIONS: tuple[str, ...] = ("AdamW", "Adam", "SGD", "RMSprop")

#: LR schedulers the ``LR Scheduler`` node can publish, in dropdown order.
SCHEDULER_OPTIONS: tuple[str, ...] = (
    "CosineAnnealingLR",
    "StepLR",
    "ExponentialLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
)

#: Window of the moving average the early-stopping monitor watches.  A raw
#: per-step loss is noisy (mini-batches, dropout), so patience is counted on the
#: smoothed curve; the window is deliberately small so a teaching run of a few
#: hundred steps still sees the trend.
EARLY_STOP_SMOOTHING = 8

#: Default ``t_max`` sentinel of :class:`SchedulerConfig`: ``0`` means "take the
#: number of steps from the trainer that consumes the config".
FOLLOW_TRAINER_STEPS = 0

#: Learning rate of a freshly dropped optimizer node. AdamW's own default is
#: 1e-3, but the tiny teaching networks of this toolchain converge better one
#: order of magnitude higher within the trainer's default step count.
DEFAULT_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    """Hyper-parameters of one ``torch.optim`` optimizer, as a graph value.

    What: the payload of the ``OPTIMIZER`` slot. It is a *configuration*, not a
          live ``torch.optim.Optimizer``: the trainer builds the real optimizer
          from it inside its own call, so nothing device bound and no optimizer
          state is cached by ComfyUI. Optimizer state (Adam's moment estimates)
          is consequently per call and cannot be carried across two prompts -
          ComfyUI gives a node no place to keep Python state between two runs.
    In:   ``name`` selects the optimizer; the remaining fields are read as
          follows. ``Adam`` / ``AdamW``: ``lr``, ``beta1``, ``beta2``, ``eps``,
          ``weight_decay``, ``amsgrad``. ``SGD``: ``lr``, ``momentum``,
          ``weight_decay``. ``RMSprop``: ``lr``, ``beta2`` (as ``alpha``),
          ``eps``, ``momentum``, ``weight_decay``.
    Out: a frozen dataclass; :func:`build_optimizer` turns it into a real
         optimizer, :meth:`describe` into a log line.
    """

    name: str = "AdamW"
    lr: float = DEFAULT_LR
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    amsgrad: bool = False
    grad_clip_norm: float = 0.0
    grad_clip_value: float = 0.0

    def describe(self) -> str:
        """One-line ``repr``-like summary used by the trainer's log line."""
        if self.name == "SGD":
            summary = (
                f"SGD(lr={self.lr:g}, momentum={self.momentum:g}, "
                f"weight_decay={self.weight_decay:g})"
            )
        elif self.name == "RMSprop":
            summary = (
                f"RMSprop(lr={self.lr:g}, alpha={self.beta2:g}, eps={self.eps:g}, "
                f"momentum={self.momentum:g}, weight_decay={self.weight_decay:g})"
            )
        else:
            summary = (
                f"{self.name}(lr={self.lr:g}, betas=({self.beta1:g}, {self.beta2:g}), "
                f"eps={self.eps:g}, weight_decay={self.weight_decay:g}, "
                f"amsgrad={self.amsgrad})"
            )
        if self.grad_clip_norm > 0.0:
            summary += f", clip norm {self.grad_clip_norm:g}"
        if self.grad_clip_value > 0.0:
            summary += f", clip value {self.grad_clip_value:g}"
        return summary


def _finite(value: float, fallback: float, label: str) -> float:
    """Return ``float(value)`` when it is finite, else ``fallback`` with a warning."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        LOGGER.warning(
            "training_protocol: %s=%r is not a number; using %g.", label, value, fallback
        )
        return float(fallback)
    if number != number or number in (float("inf"), float("-inf")):
        LOGGER.warning(
            "training_protocol: %s=%r is not finite; using %g.", label, value, fallback
        )
        return float(fallback)
    return number


def optimizer_config(
    name: str = "AdamW",
    lr: float = DEFAULT_LR,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    amsgrad: bool = False,
    grad_clip_norm: float = 0.0,
    grad_clip_value: float = 0.0,
) -> OptimizerConfig:
    """Normalise widget values into a valid :class:`OptimizerConfig`.

    What: the safe constructor behind the ``Optimizer`` node. Widgets are already
          range limited by the frontend, but a workflow file, an API call or a
          hand-edited ``.json`` can carry anything, and an out-of-range value
          would make ``torch.optim`` raise deep inside the trainer. Every value
          is therefore checked here, reported and clamped once, at the edge.
    In:   the raw widget values; see :class:`OptimizerConfig` for their meaning.
    Out: a frozen config. ``name`` falls back to ``AdamW``, ``lr`` to
         :data:`DEFAULT_LR`, ``eps`` to ``1e-8``, ``weight_decay`` to ``0.0``
         when negative, and ``momentum`` / ``beta1`` / ``beta2`` are clamped
         into ``[0, 1)`` - a beta of exactly 1 makes Adam's update infinite.
    """
    text = str(name or "").strip()
    if text not in OPTIMIZER_OPTIONS:
        LOGGER.warning(
            "training_protocol: optimizer %r is not one of %s; using 'AdamW'.",
            name, ", ".join(OPTIMIZER_OPTIONS),
        )
        text = "AdamW"

    learning_rate = _finite(lr, DEFAULT_LR, "lr")
    if learning_rate < 0.0:
        LOGGER.warning("training_protocol: lr=%r is negative; using %g.", lr, DEFAULT_LR)
        learning_rate = DEFAULT_LR

    first_beta = min(max(_finite(beta1, 0.9, "beta1"), 0.0), 0.999)
    second_beta = min(max(_finite(beta2, 0.999, "beta2"), 0.0), 0.9999)
    momentum_value = min(max(_finite(momentum, 0.9, "momentum"), 0.0), 0.999)
    epsilon = _finite(eps, 1e-8, "eps")
    if epsilon <= 0.0:
        LOGGER.warning("training_protocol: eps=%r is not positive; using 1e-08.", eps)
        epsilon = 1e-8
    decay = _finite(weight_decay, 0.0, "weight_decay")
    if decay < 0.0:
        LOGGER.warning(
            "training_protocol: weight_decay=%r is negative; using 0.0.", weight_decay
        )
        decay = 0.0
    # Clipping is off at 0; a negative value is a typo and is treated as off.
    clip_norm = _finite(grad_clip_norm, 0.0, "grad_clip_norm")
    if clip_norm < 0.0:
        LOGGER.warning(
            "training_protocol: grad_clip_norm=%r is negative; clipping disabled.",
            grad_clip_norm,
        )
        clip_norm = 0.0
    clip_value = _finite(grad_clip_value, 0.0, "grad_clip_value")
    if clip_value < 0.0:
        LOGGER.warning(
            "training_protocol: grad_clip_value=%r is negative; clipping disabled.",
            grad_clip_value,
        )
        clip_value = 0.0

    return OptimizerConfig(
        name=text,
        lr=learning_rate,
        momentum=momentum_value,
        beta1=first_beta,
        beta2=second_beta,
        eps=epsilon,
        weight_decay=decay,
        amsgrad=bool(amsgrad),
        grad_clip_norm=clip_norm,
        grad_clip_value=clip_value,
    )


def build_optimizer(
    config: OptimizerConfig, parameters
) -> torch.optim.Optimizer:
    """Create the real ``torch.optim`` optimizer described by ``config``.

    What: the only place the four supported optimizers are constructed, so the
          trainer never branches on the optimizer name itself.
    In:   config - a normalised :class:`OptimizerConfig`.
          parameters - an iterable of ``nn.Parameter``.
    Out: a fresh optimizer with empty state. ``RMSprop`` reads ``beta2`` as its
         ``alpha`` (the smoothing constant): both are "the decay of the running
         average", so one widget covers both optimizers.
    """
    trainable = [p for p in parameters if isinstance(p, nn.Parameter)]
    if not trainable:
        raise ValueError(
            "training_protocol.build_optimizer: no nn.Parameter to optimize."
        )
    if config.name == "SGD":
        return torch.optim.SGD(
            trainable,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "RMSprop":
        return torch.optim.RMSprop(
            trainable,
            lr=config.lr,
            alpha=config.beta2,
            eps=config.eps,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "Adam":
        return torch.optim.Adam(
            trainable,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    return torch.optim.AdamW(
        trainable,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )


# --------------------------------------------------------------------------- #
# LR schedulers, gradient clipping, early stopping
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    """Hyper-parameters of one ``torch.optim.lr_scheduler``, as a graph value.

    What: the payload of the ``SCHEDULER`` slot. Like
          :class:`OptimizerConfig` it is a *configuration*, not a live scheduler:
          the trainer builds the real scheduler from it inside its own call
          (see :func:`build_scheduler`), so nothing device bound is cached and
          no scheduler state leaks between prompts.
    In:   ``name`` selects the scheduler; the remaining fields are read as
          follows. ``StepLR``: ``step_size``, ``gamma``.
          ``ExponentialLR``: ``gamma``. ``CosineAnnealingLR``: ``t_max``
          (``0`` = the consuming trainer's step count), ``eta_min``.
          ``OneCycleLR``: ``max_lr`` is taken from the wired optimizer's
          ``lr``, ``total_steps`` from the trainer's step count; ``pct_start``
          is the warm-up fraction. ``ReduceLROnPlateau``: ``patience``,
          ``factor``, ``eta_min`` (as ``min_lr``).
    Out: a frozen dataclass; :func:`build_scheduler` turns it into a real
         scheduler, :meth:`describe` into a log line.
    """

    name: str = "CosineAnnealingLR"
    step_size: int = 30
    gamma: float = 0.1
    t_max: int = FOLLOW_TRAINER_STEPS
    eta_min: float = 0.0
    pct_start: float = 0.3
    patience: int = 10
    factor: float = 0.1

    def describe(self) -> str:
        """One-line ``repr``-like summary used by the trainer's log line."""
        if self.name == "StepLR":
            return f"StepLR(step_size={self.step_size}, gamma={self.gamma:g})"
        if self.name == "ExponentialLR":
            return f"ExponentialLR(gamma={self.gamma:g})"
        if self.name == "OneCycleLR":
            return (
                f"OneCycleLR(max_lr=optimizer.lr, pct_start={self.pct_start:g})"
            )
        if self.name == "ReduceLROnPlateau":
            return (
                f"ReduceLROnPlateau(patience={self.patience}, "
                f"factor={self.factor:g}, min_lr={self.eta_min:g})"
            )
        if self.t_max > FOLLOW_TRAINER_STEPS:
            return (
                f"CosineAnnealingLR(T_max={self.t_max}, eta_min={self.eta_min:g})"
            )
        return f"CosineAnnealingLR(T_max=trainer steps, eta_min={self.eta_min:g})"


def scheduler_config(
    name: str = "CosineAnnealingLR",
    step_size: int = 30,
    gamma: float = 0.1,
    t_max: int = FOLLOW_TRAINER_STEPS,
    eta_min: float = 0.0,
    pct_start: float = 0.3,
    patience: int = 10,
    factor: float = 0.1,
) -> SchedulerConfig:
    """Normalise widget values into a valid :class:`SchedulerConfig`.

    What: the safe constructor behind the ``LR Scheduler`` node. Widget values
          are already range limited by the frontend, but a workflow file or an
          API call can carry anything; every value is checked, reported and
          clamped here once, at the edge, so ``torch.optim.lr_scheduler``
          never raises deep inside a trainer.
    In:   the raw widget values; see :class:`SchedulerConfig` for their meaning.
    Out: a frozen config. ``name`` falls back to ``CosineAnnealingLR``;
         ``step_size`` / ``t_max`` / ``patience`` are clamped to ``>= 1``
         (``t_max`` keeps ``0`` as the "follow the trainer" sentinel),
         ``gamma`` / ``factor`` to ``(0, 1)``, ``pct_start`` to ``(0, 1]`` and
         ``eta_min`` to ``>= 0``.
    """
    text = str(name or "").strip()
    if text not in SCHEDULER_OPTIONS:
        LOGGER.warning(
            "training_protocol: scheduler %r is not one of %s; using 'CosineAnnealingLR'.",
            name, ", ".join(SCHEDULER_OPTIONS),
        )
        text = "CosineAnnealingLR"

    def _at_least(value: int, floor: int, label: str, allow_zero: bool) -> int:
        number = int(value)
        if allow_zero and number == 0:
            return 0
        if number < floor:
            LOGGER.warning(
                "training_protocol: %s=%r is below %d; using %d.", label, value, floor, floor
            )
            return floor
        return number

    size = _at_least(step_size, 1, "step_size", allow_zero=False)
    horizon = _at_least(t_max, 1, "t_max", allow_zero=True)
    plateau_patience = _at_least(patience, 1, "patience", allow_zero=False)

    decay = _finite(gamma, 0.1, "gamma")
    if not 0.0 < decay < 1.0:
        LOGGER.warning("training_protocol: gamma=%r is outside (0, 1); using 0.1.", gamma)
        decay = 0.1
    reduction = _finite(factor, 0.1, "factor")
    if not 0.0 < reduction < 1.0:
        LOGGER.warning("training_protocol: factor=%r is outside (0, 1); using 0.1.", factor)
        reduction = 0.1
    warmup = _finite(pct_start, 0.3, "pct_start")
    if not 0.0 < warmup <= 1.0:
        LOGGER.warning("training_protocol: pct_start=%r is outside (0, 1]; using 0.3.", pct_start)
        warmup = 0.3
    floor_lr = _finite(eta_min, 0.0, "eta_min")
    if floor_lr < 0.0:
        LOGGER.warning("training_protocol: eta_min=%r is negative; using 0.0.", eta_min)
        floor_lr = 0.0

    return SchedulerConfig(
        name=text,
        step_size=size,
        gamma=decay,
        t_max=horizon,
        eta_min=floor_lr,
        pct_start=warmup,
        patience=plateau_patience,
        factor=reduction,
    )


def build_scheduler(
    config: SchedulerConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> tuple[object, bool]:
    """Create the real ``torch.optim.lr_scheduler`` described by ``config``.

    What: the only place the five supported schedulers are constructed, so a
          trainer never branches on the scheduler name itself.
    In:   config - a normalised :class:`SchedulerConfig`.
          optimizer - the freshly built optimizer whose param groups carry the
          base learning rate (``OneCycleLR`` reads it as ``max_lr``).
          total_steps - the trainer's step count, used as ``T_max`` when
          ``t_max`` is the "follow the trainer" sentinel ``0`` and as
          ``total_steps`` of ``OneCycleLR``.
    Out: ``(scheduler, needs_metric)``. ``needs_metric`` is ``True`` only for
         ``ReduceLROnPlateau``, whose ``step`` takes the monitored value; the
         caller then steps the scheduler with the smoothed training loss.
         Every other scheduler is stepped without arguments, once per iteration
         after ``optimizer.step()``.
    """
    steps = max(1, int(total_steps))
    base_lr = optimizer.param_groups[0]["lr"]
    if config.name == "StepLR":
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, config.step_size), gamma=config.gamma
            ),
            False,
        )
    if config.name == "ExponentialLR":
        return (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma),
            False,
        )
    if config.name == "OneCycleLR":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=base_lr,
                total_steps=steps,
                pct_start=config.pct_start,
            ),
            False,
        )
    if config.name == "ReduceLROnPlateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.factor,
                patience=config.patience,
                min_lr=config.eta_min,
            ),
            True,
        )
    horizon = config.t_max if config.t_max > FOLLOW_TRAINER_STEPS else steps
    return (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=config.eta_min
        ),
        False,
    )


def clip_gradients(
    config: OptimizerConfig, parameters
) -> None:
    """Apply the gradient clipping a trainer's optimizer config asks for.

    What: the single place ``torch.nn.utils.clip_grad_norm_`` /
          ``clip_grad_value_`` are called, so the two widgets of the
          ``Optimizer`` node cannot drift into different behaviours per trainer.
    In:   config - the trainer's :class:`OptimizerConfig`; ``grad_clip_norm``
          and / or ``grad_clip_value`` are ``0`` (the default) meaning "off".
          parameters - the module's parameters, as passed to the optimizer.
    Out: nothing; the gradients of ``parameters`` are clipped in place. When
         both widgets are set, the norm clip runs first and the value clip
         second - a deterministic, documented combination rather than a silent
         either / or.
    """
    trainable = [p for p in parameters if isinstance(p, nn.Parameter) and p.grad is not None]
    if config.grad_clip_norm > 0.0 and trainable:
        nn.utils.clip_grad_norm_(trainable, config.grad_clip_norm)
    if config.grad_clip_value > 0.0 and trainable:
        nn.utils.clip_grad_value_(trainable, config.grad_clip_value)


def snapshot_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clone of every parameter of ``module`` (an early-stopping save point).

    In:  module - usually the trainer's working copy.
    Out: ``{name: tensor}``; every tensor is a plain clone, so later training
         steps can never reach into the snapshot. ``restore_state`` accepts it.
    """
    return {
        name: value.detach().clone()
        for name, value in module.named_parameters()
    }


def restore_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Copy a :func:`snapshot_state` payload back into ``module`` (a rollback).

    In:  module - the module whose parameters are overwritten.
         state - a payload of :func:`snapshot_state`; entries whose shape no
         longer matches are skipped, which cannot happen for a snapshot of the
         same module but keeps a hand-edited payload from raising mid-roll-back.
    """
    with torch.no_grad():
        for name, value in module.named_parameters():
            saved = state.get(name)
            if saved is not None and tuple(saved.shape) == tuple(value.shape):
                value.copy_(saved.to(device=value.device, dtype=value.dtype))


class EarlyStopTracker:
    """Patience-based early stopping on the moving average of the step loss.

    What: the monitor half of early stopping. A raw per-step loss is noisy, so
          the monitored value is the moving average of the last
          :data:`EARLY_STOP_SMOOTHING` step losses; a value only counts as an
          improvement when it beats the best one by more than ``min_delta``.
          The tracker owns no tensors - the caller snapshots the parameters on
          :meth:`improved` and rolls back on :meth:`update` returning ``True``
          - so the decision is a pure function of the loss sequence and a run
          is exactly reproducible for a given seed.
    In:   patience - how many *steps without improvement* are tolerated before
          stopping; ``0`` disables the monitor entirely.
          min_delta - the smallest improvement that resets the patience.
    Out: call :meth:`update` once per step with the raw step loss; it returns
         ``True`` when training should stop (patience exhausted), and
         :meth:`improved` reports whether the just-updated step set a new best
         (the caller snapshots at that point).
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = max(0, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.best: float | None = None
        self.best_step: int = -1
        self._window: list[float] = []
        self._since_best = 0

    @property
    def enabled(self) -> bool:
        """True when the monitor can ever fire (``patience > 0``)."""
        return self.patience > 0

    def update(self, step: int, value: float) -> bool:
        """Feed one step loss; return ``True`` when training should stop.

        In:  step - the 0-based step index (used for the best-step report).
             value - the raw loss of that step.
        Out: ``False`` while patience is not exhausted or the monitor is off;
             ``True`` once ``patience`` steps passed without an improvement of
             more than ``min_delta``.  The first step always counts as the
             initial best so a flat loss eventually stops the run.
        """
        if not self.enabled:
            return False
        self._window.append(float(value))
        if len(self._window) > EARLY_STOP_SMOOTHING:
            self._window.pop(0)
        monitored = sum(self._window) / len(self._window)
        if self.best is None or monitored < self.best - self.min_delta:
            self.best = monitored
            self.best_step = step
            self._since_best = 0
        else:
            self._since_best += 1
        return self._since_best > self.patience

    @property
    def improved(self) -> bool:
        """True when the most recent :meth:`update` set a new best value."""
        return self.enabled and self._since_best == 0

    def describe_stop(self) -> str:
        """One-line report for the stop case: best step, best value, patience."""
        return (
            f"no improvement for {self._since_best} step(s) "
            f"(patience {self.patience}, min_delta {self.min_delta:g}); "
            f"best monitored loss {self.best:.6g} at step {self.best_step}"
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def seeded_rng(seed: int):
    """Run a block with a seeded torch RNG, then restore the previous one.

    What: makes an initialisation reproducible *without* disturbing anybody else.
          ``torch.manual_seed`` reseeds the CPU generator and every CUDA
          generator, and ``nn.Linear`` draws its weights from the CPU generator
          of the module's device, so both have to be saved and put back - the
          other nodes of the graph rely on the process RNG staying untouched
          (``RegularizationDropout`` makes exactly the same promise).
    In:   seed - the seed to apply for the duration of the block.
    Out: a context manager; the previous CPU and CUDA RNG states are restored on
         the way out, also when the block raises.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class MLP(nn.Module):
    """Fully connected network whose layers are named ``layer0``, ``layer1``, ...

    What: the model a ``Training Loop`` node trains. It is deliberately built
          from plain ``nn.Linear`` layers with a chosen activation *between*
          them and a linear output layer, because that is the shape of every
          teaching model this toolchain targets (linear regression, softmax
          regression, an MLP classifier). The submodules are registered through
          ``setattr`` instead of an ``nn.Sequential`` / ``nn.ModuleList`` so the
          parameter names are ``layer0.weight`` / ``layer0.bias`` rather than
          ``0.weight``: a name that both survives a ``safetensors`` round trip
          and can be typed into the ``Parameters to Tensor`` node to feed the
          ordinary ``Basic`` layer nodes.
    In:   sizes - the layer widths, ``(in_features, hidden..., out_features)``;
          at least two entries. ``activation`` - one of
          :data:`ACTIVATION_OPTIONS`; ``"none"`` makes the network linear.
    Out: an ``nn.Module`` of float32 parameters whose ``named_parameters()`` are
         exactly ``layer{i}.weight`` / ``layer{i}.bias`` in layer order, and
         whose ``forward`` maps ``(N, sizes[0])`` to ``(N, sizes[-1])``.
    """

    def __init__(self, sizes: Sequence[int], activation: str = "relu") -> None:
        super().__init__()
        widths = tuple(int(width) for width in sizes)
        if len(widths) < 2:
            raise ValueError(
                f"training_protocol.MLP: need at least two layer widths, got {widths!r}."
            )
        if any(width <= 0 for width in widths):
            raise ValueError(
                f"training_protocol.MLP: every layer width must be positive, got {widths!r}."
            )
        self.widths = widths
        self.activation = str(activation or "none").strip().lower()
        if self.activation not in ACTIVATION_OPTIONS:
            LOGGER.warning(
                "training_protocol: activation %r is not one of %s; using 'relu'.",
                activation, ", ".join(ACTIVATION_OPTIONS),
            )
            self.activation = "relu"
        for index, (in_features, out_features) in enumerate(zip(widths, widths[1:])):
            setattr(self, f"{LAYER_PREFIX}{index}", nn.Linear(in_features, out_features))

    @property
    def depth(self) -> int:
        """Number of dense layers (``len(widths) - 1``)."""
        return len(self.widths) - 1

    def layer(self, index: int) -> nn.Linear:
        """The dense layer ``layer{index}``."""
        return getattr(self, f"{LAYER_PREFIX}{index}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MLP(widths={self.widths}, activation={self.activation!r})"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply every layer, activating between them; the output layer is linear."""
        result = tensor
        for index in range(self.depth):
            result = self.layer(index)(result)
            if index < self.depth - 1 and self.activation != "none":
                result = apply_activation(self.activation, result)
        return result


def apply_activation(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply one of :data:`ACTIVATION_OPTIONS` to ``tensor``.

    In:  name - the activation's name (case insensitive).
         tensor - any floating point tensor.
    Out: the activated tensor; an unknown name falls back to ``relu`` (with a
         warning), and ``"none"`` returns the tensor unchanged.
    """
    text = str(name or "none").strip().lower()
    if text == "relu":
        return F.relu(tensor)
    if text == "gelu":
        return F.gelu(tensor)
    if text == "tanh":
        return torch.tanh(tensor)
    if text == "sigmoid":
        return torch.sigmoid(tensor)
    if text == "none":
        return tensor
    LOGGER.warning(
        "training_protocol: activation %r is unknown; using 'relu'.", name
    )
    return F.relu(tensor)


def build_mlp(
    sizes: Sequence[int],
    activation: str = "relu",
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MLP:
    """Build an :class:`MLP` with reproducible weights on the requested device.

    In:  sizes - see :class:`MLP`.
         activation - see :class:`MLP`.
         seed - when given, the initialisation is drawn under
         :func:`seeded_rng`, so the same seed always produces the same weights
         and the process RNG is left as it was.
         device / dtype - where the parameters should live; ``None`` keeps the
         default (CPU / float32).
    Out: the module, already moved to ``device`` / ``dtype``.
    """
    if seed is None:
        module = MLP(sizes, activation)
    else:
        with seeded_rng(seed):
            module = MLP(sizes, activation)
    if device is None:
        device = torch.device("cpu")
    return module.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# Parameter payloads
# --------------------------------------------------------------------------- #


def parameter_names(params: Mapping[str, object], limit: int = 8) -> str:
    """Render a payload's keys for an error message.

    In:  params - a ``{name: tensor}`` payload (``PARAMS``).
         limit - how many names to spell out before switching to a ``+N more``
         suffix; a long checkpoint must not produce an unreadable message.
    Out: ``"'layer0.weight', 'layer0.bias', ... (+3 more)"``, or ``"<none>"``
         for an empty payload.
    """
    names = sorted(str(name) for name in params)
    if not names:
        return "<none>"
    head = ", ".join(repr(name) for name in names[: max(1, int(limit))])
    if len(names) <= limit:
        return head
    return f"{head}, ... (+{len(names) - limit} more)"


def as_parameter_dict(params) -> dict[str, nn.Parameter]:
    """Coerce a ``PARAMS`` payload into ``{name: nn.Parameter}``.

    What: the single entry point every node uses to read a ``PARAMS`` value, so
          the coercion - and its warnings - cannot drift between nodes. Slots
          are not type checked by the engine, and a workflow can be edited by
          hand, so a payload may hold a plain tensor (wrapped into a parameter)
          or something that can never be trainable (dropped, with a warning).
    In:   params - a mapping of ``str -> nn.Parameter | torch.Tensor``.
    Out: a new ``dict`` of ``nn.Parameter``; entries that are not floating point
         tensors are dropped and reported, never silently kept.
    """
    if params is None:
        raise ValueError(
            "training_protocol.as_parameter_dict: the PARAMS payload is None; "
            "connect a 'Learnable Parameters' node to that slot."
        )
    if not isinstance(params, Mapping):
        raise TypeError(
            "training_protocol.as_parameter_dict: expected a {name: parameter} "
            f"mapping, got {type(params).__name__}."
        )
    result: dict[str, nn.Parameter] = {}
    dropped: list[str] = []
    for key, value in params.items():
        name = str(key)
        if isinstance(value, nn.Parameter) and value.is_floating_point():
            result[name] = value
        elif isinstance(value, torch.Tensor) and value.is_floating_point():
            result[name] = nn.Parameter(value.detach().clone())
        else:
            dropped.append(name)
    if dropped:
        LOGGER.warning(
            "training_protocol: dropped %d non float parameter(s) (%s); only "
            "floating point tensors can be learnable.",
            len(dropped), parameter_names({name: None for name in dropped}),
        )
    return result


def module_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    """All parameters of ``module``, keyed ``layer{i}.weight`` / ``layer{i}.bias``.

    In:  module - usually an :class:`MLP`.
    Out: an ordered ``dict`` in registration order, which is the payload a
         trainer publishes as its ``PARAMS`` output.
    """
    return {name: value for name, value in module.named_parameters()}


def load_into_module(
    module: nn.Module, params: Mapping[str, torch.Tensor]
) -> tuple[list[str], list[str], list[str]]:
    """Copy a parameter payload into ``module``, reporting every non match.

    What: the warm start of a trainer. ``nn.Module.load_state_dict`` alone is not
          enough here: a shape mismatch raises even with ``strict=False``, and an
          unexpected key is only reported through an exception the node would
          have to catch. This pre-filters the payload, so a payload coming from a
          differently shaped checkpoint still loads everything it *can* and the
          caller can print what was ignored.
    In:   module - the destination module; its parameters are the reference.
          params - the payload to apply; values are cast to the destination
          dtype / device, and non tensor entries are ignored.
    Out: ``(loaded, missing, skipped)`` - names that were copied, names the
         module has but the payload does not, and human readable reasons for the
         entries that were refused (unknown name or mismatching shape).
    """
    destination = dict(module.named_parameters())
    incoming: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in params.items():
        name = str(key)
        target = destination.get(name)
        if target is None:
            skipped.append(f"{name} (no such parameter)")
            continue
        if not isinstance(value, torch.Tensor):
            skipped.append(f"{name} ({type(value).__name__} is not a tensor)")
            continue
        if tuple(value.shape) != tuple(target.shape):
            skipped.append(f"{name} ({tuple(value.shape)} != {tuple(target.shape)})")
            continue
        incoming[name] = value.detach().to(device=target.device, dtype=target.dtype)
    missing = sorted(name for name in destination if name not in incoming)
    if incoming:
        module.load_state_dict(incoming, strict=False)
    return sorted(incoming), missing, skipped


def parameters_to_tensors(params: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Flatten a payload into plain, saveable CPU tensors.

    What: both ``safetensors`` and ``base64`` need contiguous CPU tensors, and a
          payload that has just been trained holds live parameters with a
          ``grad`` and possibly a device. This detaches, moves and makes them
          contiguous in one place.
    In:   params - a ``PARAMS`` payload.
    Out: a new ``dict`` of detached CPU tensors; entries that are not tensors are
         dropped (payloads that reach this point have already been normalised by
         :func:`as_parameter_dict`).
    """
    tensors: dict[str, torch.Tensor] = {}
    for key, value in params.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().to(device="cpu")
        tensors[str(key)] = tensor if tensor.is_contiguous() else tensor.contiguous()
    return tensors


def parameter_count(params: Mapping[str, object]) -> int:
    """Total number of scalar values in a payload (used for log lines)."""
    total = 0
    for value in params.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel())
    return total


# --------------------------------------------------------------------------- #
# Widget-carried text form
# --------------------------------------------------------------------------- #


def encode_parameters(params: Mapping[str, object]) -> str:
    """Encode a payload as ``CDLPARAMS1:<base64 safetensors>``.

    What: the widget-carried half of the persistence story. A widget is what
          survives in a saved workflow, so a trained parameter set can be handed
          to another run as *text* - no files, no shared folder, and the values
          travel with the workflow itself. ``safetensors`` is used rather than
          ``pickle`` so the text can only ever decode into plain tensors.
    In:   params - a ``PARAMS`` payload.
    Out: an ASCII string starting with :data:`PARAMS_TEXT_PREFIX`, safe to paste
         into a string widget.
    """
    tensors = parameters_to_tensors(params)
    if not tensors:
        raise ValueError(
            "training_protocol.encode_parameters: nothing to encode; the payload "
            "holds no tensor."
        )
    blob = safetensors.torch.save(tensors)
    return PARAMS_TEXT_PREFIX + base64.b64encode(blob).decode("ascii")


def decode_parameters(text: str) -> dict[str, torch.Tensor]:
    """Decode the text form produced by :func:`encode_parameters`.

    What: the reading half of the codec. Trailing/leading whitespace and line
          breaks are tolerated (a widget may wrap long text), and the version
          header is optional so a user who copied only the base64 body still
          gets their parameters back.
    In:   text - the widget value.
    Out: a ``{name: torch.Tensor}`` payload (plain tensors; callers wrap them
         into ``nn.Parameter`` through :func:`as_parameter_dict`).
    Raises:
        ValueError: with an actionable message when the text is empty, is not
            valid base64, is not a ``safetensors`` blob, or carries the header
            of an unknown version.
    """
    raw = "".join(str(text or "").split())
    if not raw:
        raise ValueError(
            "the text is empty; paste the 'text' output of a 'Parameters to Text' "
            "node into this widget."
        )
    if PARAMS_TEXT_PREFIX in raw:
        head, _, body = raw.rpartition(PARAMS_TEXT_PREFIX)
        if head:
            raise ValueError(
                "the text contains more than one header; paste exactly one "
                f"{PARAMS_TEXT_PREFIX!r} blob."
            )
        raw = body
    elif raw.startswith("CDLPARAMS"):
        raise ValueError(
            "the text starts with an unknown parameter header; only "
            f"{PARAMS_TEXT_PREFIX!r} is supported."
        )
    if not raw:
        raise ValueError(
            "the text carries the header but no payload; re-copy the whole 'text' "
            "output of the 'Parameters to Text' node."
        )
    padding = "=" * (-len(raw) % 4)
    try:
        blob = base64.b64decode(raw + padding, validate=True)
    except Exception as error:  # noqa: BLE001 - reported as an actionable error
        raise ValueError(
            f"the text is not valid base64 ({error}); paste the 'text' output of a "
            "'Parameters to Text' node unchanged."
        ) from error
    try:
        tensors = safetensors.torch.load(blob)
    except Exception as error:  # noqa: BLE001 - reported as an actionable error
        raise ValueError(
            f"the decoded text is not a safetensors payload ({error}); paste the "
            "'text' output of a 'Parameters to Text' node unchanged."
        ) from error
    if not tensors:
        raise ValueError("the decoded payload holds no tensor.")
    return {str(key): value for key, value in tensors.items()}


__all__ = [
    "ACTIVATION_OPTIONS",
    "BIAS_SUFFIX",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "EARLY_STOP_SMOOTHING",
    "EarlyStopTracker",
    "FOLLOW_TRAINER_STEPS",
    "LAYER_PREFIX",
    "MLP",
    "OPTIMIZER_OPTIONS",
    "OptimizerConfig",
    "PARAMS_TEXT_PREFIX",
    "SCHEDULER_OPTIONS",
    "SchedulerConfig",
    "WEIGHT_SUFFIX",
    "apply_activation",
    "as_parameter_dict",
    "build_mlp",
    "build_optimizer",
    "build_scheduler",
    "clip_gradients",
    "decode_parameters",
    "encode_parameters",
    "load_into_module",
    "module_parameters",
    "optimizer_config",
    "parameter_count",
    "parameter_names",
    "parameters_to_tensors",
    "restore_state",
    "scheduler_config",
    "seeded_rng",
    "snapshot_state",
]
