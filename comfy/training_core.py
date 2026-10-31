"""Training session core (P3 Craft 2, M-C): the shared training machinery.

``TrainingLoop`` (``comfy_extras/nodes_training.py``) has the project's most
battle-tested training loop: batch shuffling under an explicit CPU generator,
the early-stop tracker with best-snapshot rollback, crash-site checkpointing
around ``InterruptProcessingException``, and the ``inference_mode(False)``
escape that makes gradients possible inside ComfyUI at all.  The experiment
bench (``CdlGridSearchTrain``) needs *exactly* that machinery per grid
combination - a fresh model, the same early stop, the same rollback - so the
loop moved here and both nodes delegate to it.

What moved vs what stayed:

* Moved (verbatim where at all possible): the ``x``/``y`` preparation helpers
  (``_TrainingData`` & friends), the loss-name handling, the early-stop
  progress loop, the checkpoint save/restore payload, and the warm-start
  summary.  ``nodes_training`` re-imports the helper names, so every other
  node in that file keeps working unchanged.
* Stayed node-side: widget warnings (step-count threshold), the live
  loss-curve previewer + ``ProgressBar`` (they bind to the *executing node*
  through the progress callback), and the final human-readable log line -
  the two nodes word them differently on purpose.

Extensibility contract: ``TrainingSession`` is configuration-in,
result-out.  Node-specific behaviour rides two hooks:

* ``progress(step, loss_value)`` - called once per optimizer step, at the
  exact point the original loop updated its progress bar (after the early
  stopper had its say, so an early-stopped final step reports no frame -
  preserved quirk).
* ``ckpt_key_fn(data)`` - receives the prepared :class:`_TrainingData` and
  returns a crash-site checkpoint key (or ``None`` to disable
  checkpointing).  ``CdlGridSearchTrain`` folds the combo identity into the
  key so two combinations can never collide on one checkpoint file.

Layering: torch + ``comfy.training_protocol`` + ``comfy.train_checkpoint`` -
the same imports the training nodes already use, nothing app-side.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from comfy import training_metrics as tm
from comfy import training_protocol as tp

#: Loss functions the trainer can minimize (shared with the Loss / Evaluate
#: nodes; see ``comfy.training_metrics``). ``cross_entropy`` and ``perplexity``
#: expect class targets (indices or one-hot), the others regression targets.
LOSS_OPTIONS: tuple[str, ...] = tm.LOSS_OPTIONS

#: Default of the ``hidden`` widget, and the value a typo falls back to.
DEFAULT_HIDDEN = "8"

#: Above this many iterations the trainer warns: the loop is Python, and a user
#: asking for a million steps has almost certainly typed one zero too many.
STEPS_WARN_THRESHOLD = 20000

#: Longest run of names printed in one log line before it is cut short.
NAME_PREVIEW = 8


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _parse_sizes(text: str) -> tuple[int, ...] | None:
    """Parse ``"8"`` or ``"16,8"`` into a tuple of positive ints.

    Accepts comma, semicolon, whitespace and full-width separators so that a
    value typed on a Chinese IME (``"16，8"``) still works.

    Args:
        text: The raw widget value.

    Returns:
        The parsed widths with every non-positive entry dropped, or ``None``
        when the string is empty or holds a non-integer entry; callers fall back
        to their documented default so that a typo cannot break a workflow.
    """
    if text is None:
        return None
    normalized = str(text).replace("，", ",").replace("；", ",").replace(";", ",")
    parts = [part for part in re.split(r"[,\s]+", normalized) if part]
    if not parts:
        return None
    sizes: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return None
        if value > 0:
            sizes.append(value)
    return tuple(sizes)


def _loss_mode(value: str) -> str:
    """Return a valid loss name, falling back to ``mse`` with a warning."""
    text = "" if value is None else str(value).strip().lower()
    if text in LOSS_OPTIONS:
        return text
    _warn(f"loss '{value}' is unknown; using 'mse' (one of {', '.join(LOSS_OPTIONS)}).")
    return "mse"


def _hidden_sizes(value: str) -> tuple[int, ...]:
    """Read the ``hidden`` widget into the widths of the hidden layers.

    Args:
        value: The raw widget value; empty means "no hidden layer", i.e. a single
            dense layer from the features straight to the outputs.

    Returns:
        The hidden widths, without the input / output layer. Text that cannot be
        read falls back to the default ``8``, and non-positive entries are
        dropped, both with a warning, so a typo cannot break a workflow.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    sizes = _parse_sizes(text)
    if sizes is None:
        _warn(
            f"hidden '{value}' could not be read as layer widths; using "
            f"'{DEFAULT_HIDDEN}' instead."
        )
        sizes = _parse_sizes(DEFAULT_HIDDEN) or (8,)
    return tuple(sizes)


@dataclasses.dataclass(frozen=True)
class _TrainingData:
    """The flattened ``(x, y)`` pair a trainer consumes.

    What: turns the two tensors that arrive on the ``x`` / ``y`` slots into the
          ``(samples, features)`` / ``(samples, targets)`` form the loop needs,
          and rejects the combinations that cannot be trained at all. The rules
          are the ones the layer nodes already use - the *last* dimension is the
          feature dimension - so a trainer can sit behind exactly the same nodes
          as a ``Linear`` layer.
    In:   the raw ``x`` / ``y`` tensors plus the selected loss name.
    Out: ``inputs`` (float32, ``(N, features)``), ``targets`` (float32
         ``(N, outputs)`` for a regression loss, int64 ``(N,)`` class indices for
         ``cross_entropy``), the feature / output counts, the sample count and
         the loss name.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    features: int
    outputs: int
    samples: int
    loss: str

    @property
    def classification(self) -> bool:
        """True when the targets are class indices rather than values."""
        return self.loss == "cross_entropy"

    @classmethod
    def from_inputs(cls, x: torch.Tensor, y: torch.Tensor, loss: str) -> "_TrainingData":
        """Flatten and validate ``x`` / ``y``; see the class docstring.

        Raises:
            TypeError: when a slot did not receive a tensor.
            ValueError: when the pair cannot be trained - empty tensors, a
                different number of samples in ``x`` and ``y``, or class targets
                that are negative or not integral.
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "[Network & Layers] Training Loop needs a tensor on both the 'x' "
                f"and the 'y' slot; got {type(x).__name__} and {type(y).__name__}."
            )
        mode = _loss_mode(loss)
        inputs = _flatten_features(x)
        samples, features = (int(inputs.shape[0]), int(inputs.shape[1]))
        if samples == 0 or features == 0:
            raise ValueError(
                f"[Network & Layers] Training Loop got an empty input of shape "
                f"{tuple(x.shape)}; 'x' must hold at least one sample and one feature."
            )
        if mode == "cross_entropy":
            targets, outputs = _class_targets(y, samples)
        else:
            targets = _flatten_features(y).to(dtype=torch.float32)
            if int(targets.shape[0]) != samples:
                raise ValueError(
                    f"[Network & Layers] Training Loop needs as many targets as "
                    f"inputs, but 'x' holds {samples} sample(s) and 'y' holds "
                    f"{int(targets.shape[0])}."
                )
            outputs = int(targets.shape[1])
        inputs = inputs.to(dtype=torch.float32)
        return cls(
            inputs=inputs,
            targets=targets.to(device=inputs.device),
            features=features,
            outputs=outputs,
            samples=samples,
            loss=mode,
        )


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten every dimension but the last one: ``(..., features)`` -> ``(N, features)``.

    A 1-D tensor is read as a single feature column, matching the way
    ``torch.nn.Linear`` treats a bare vector.
    """
    if tensor.dim() == 0:
        raise ValueError(
            "[Network & Layers] Training Loop got a scalar; a sample needs at least "
            "one feature dimension."
        )
    if tensor.dim() == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, tensor.shape[-1])


def _class_targets(y: torch.Tensor, samples: int) -> tuple[torch.Tensor, int]:
    """Read ``y`` as classification targets and return ``(indices, classes)``.

    Accepts the two spellings PyTorch's ``cross_entropy`` understands: class
    indices, or a 2-D one-hot / probability tensor whose last dimension is the
    class count. A scalar target is rejected - it cannot be matched with ``x``.
    """
    if y.dim() >= 2 and y.shape[-1] > 1 and y.is_floating_point():
        probabilities = y.reshape(-1, y.shape[-1])
        if int(probabilities.shape[0]) != samples:
            raise ValueError(
                f"[Network & Layers] Training Loop got {samples} input sample(s) but "
                f"{int(probabilities.shape[0])} one-hot target(s)."
            )
        return probabilities.argmax(dim=-1).to(dtype=torch.long), int(probabilities.shape[1])
    indices = y.reshape(-1)
    if int(indices.shape[0]) != samples:
        raise ValueError(
            f"[Network & Layers] Training Loop got {samples} input sample(s) but "
            f"{int(indices.shape[0])} class target(s)."
        )
    if indices.is_floating_point():
        rounded = indices.round()
        if not bool(torch.equal(indices.detach().cpu(), rounded.detach().cpu())):
            raise ValueError(
                "[Network & Layers] with loss='cross_entropy' the targets on 'y' must "
                "be class indices (integers), or a 2-D one-hot / probability tensor; "
                "float targets are only accepted when they are integral."
            )
        indices = rounded
    indices = indices.to(dtype=torch.long)
    smallest = int(indices.min())
    if smallest < 0:
        raise ValueError(
            f"[Network & Layers] with loss='cross_entropy' class indices must be >= 0, "
            f"but 'y' holds {smallest}."
        )
    return indices, int(indices.max()) + 1


def _loss_value(mode: str, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply the selected loss function to a batch.

    Delegates to :func:`comfy.training_metrics.compute_loss`, the single
    implementation shared with the ``Loss`` and ``Evaluate`` nodes, so the
    trainer's numbers can never drift from the standalone node's.
    """
    return tm.compute_loss(mode, prediction, targets)


def _summarize_warm_start(
    loaded: list[str], missing: list[str], skipped: list[str]
) -> str:
    """One-line report of a warm start, so nothing is dropped silently."""
    parts = [f"warm start: {len(loaded)} parameter(s) loaded"]
    if missing:
        parts.append(
            f"left at their fresh values: "
            f"{tp.parameter_names({name: None for name in missing}, NAME_PREVIEW)}"
        )
    if skipped:
        shown = "; ".join(skipped[:NAME_PREVIEW])
        if len(skipped) > NAME_PREVIEW:
            shown += f"; ... (+{len(skipped) - NAME_PREVIEW} more)"
        parts.append(f"ignored: {shown}")
    return " | ".join(parts)


def resolve_optimizer_scheduler(optimizer, scheduler) -> tuple[Any, Any]:
    """Apply the documented slot-fallbacks for the ``OPTIMIZER``/``SCHEDULER``
    slots (shared by every training-family node, same messages as ever)."""
    if not isinstance(optimizer, tp.OptimizerConfig):
        _warn(
            "the 'optimizer' slot did not receive an Optimizer node; using the "
            "default AdamW settings."
        )
        optimizer = tp.optimizer_config()
    if scheduler is not None and not isinstance(scheduler, tp.SchedulerConfig):
        _warn(
            "the 'scheduler' slot did not receive an LR Scheduler node; "
            "keeping the learning rate constant."
        )
        scheduler = None
    return optimizer, scheduler


# --------------------------------------------------------------------------- #
# The session

@dataclasses.dataclass(frozen=True)
class SessionResult:
    """Everything a finished training session produced."""

    #: the trained (possibly early-stop-rolled-back) module
    model: nn.Module
    #: 1-D detached loss history (truncated at the best step on early stop)
    history: torch.Tensor
    #: ``model(x)`` of the trained network, detached
    predictions: torch.Tensor
    #: step index the early stopper fired on (``None`` = ran to the end)
    stopped_at: Optional[int]
    #: warm-start style notes for the caller's log line
    notes: list[str]
    #: effective step count (``max(1, steps)``)
    iterations: int
    #: effective per-step batch (0 = whole dataset)
    per_step: int
    #: layer widths of the trained MLP (for the caller's log line)
    widths: tuple[int, ...]
    #: the prepared ``(x, y)`` pair the session trained on
    data: "_TrainingData"


class TrainingSession:
    """One training run over ``(x, y)`` - the moved ``TrainingLoop`` core.

    Configure in the constructor, :meth:`run` once.  See the module docstring
    for the ``progress`` / ``ckpt_key_fn`` hooks and what stayed node-side.
    """

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        loss: str = "mse",
        hidden: str = DEFAULT_HIDDEN,
        activation: str = "relu",
        optimizer: Any,
        scheduler: Any = None,
        params: Any = None,
        steps: int = 200,
        batch_size: int = 0,
        seed: int = 0,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 1e-4,
        ckpt_key_fn: Optional[Callable[[_TrainingData], Any]] = None,
        progress: Optional[Callable[[int, float], None]] = None,
        label: str = "Training Loop",
    ) -> None:
        self.x = x
        self.y = y
        self.loss_name = loss
        self.hidden = hidden
        self.activation = activation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.steps = steps
        self.batch_size = batch_size
        self.seed = seed
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.ckpt_key_fn = ckpt_key_fn
        self.progress = progress
        self.label = label
        # Prepared up front so callers can build a checkpoint key from the
        # exact tensors the loop will train on (same payload as ever).
        self.data = _TrainingData.from_inputs(self.x, self.y, self.loss_name)
        self.widths: tuple[int, ...] = (
            (self.data.features,) + _hidden_sizes(self.hidden) + (self.data.outputs,)
        )

    # ------------------------------------------------------------------ #

    def run(self) -> SessionResult:
        """Execute the training loop; see :class:`TrainingSession`."""
        data = self.data
        iterations = max(1, int(self.steps))
        per_step = int(self.batch_size)
        if per_step <= 0 or per_step >= data.samples:
            if per_step > data.samples:
                _warn(
                    f"batch_size {per_step} is larger than the {data.samples} "
                    "sample(s); using the whole dataset every step."
                )
            per_step = 0

        device = data.inputs.device
        shuffler = torch.Generator(device="cpu")
        shuffler.manual_seed(int(self.seed))
        notes: list[str] = []
        history: list[torch.Tensor] = []

        # Everything below runs outside ComfyUI's inference_mode: an inference
        # tensor cannot take part in a backward pass, and the parameters built
        # here have to stay ordinary (grad carrying) tensors.
        with torch.inference_mode(False):
            # Host inference mode (the V1 trainers' documented discipline,
            # see comfydl/nodes/linreg_train.py): the whole prompt evaluates
            # under torch.inference_mode(), so x/y arrived as *inference
            # tensors* - they can be neither updated in place nor saved for
            # backward.  Cloning them INSIDE this normal-mode scope converts
            # them to ordinary tensors and the backward pass below is legal.
            if data.inputs.is_inference() or data.targets.is_inference():
                data = dataclasses.replace(
                    data,
                    inputs=torch.as_tensor(
                        data.inputs, dtype=torch.float32).clone(),
                    targets=torch.as_tensor(data.targets).clone(),
                )
            device = data.inputs.device
            model = tp.build_mlp(self.widths, self.activation, seed=self.seed, device=device)
            if self.params is not None:
                loaded, missing, skipped = tp.load_into_module(
                    model, tp.as_parameter_dict(self.params)
                )
                notes.append(_summarize_warm_start(loaded, missing, skipped))
            trainer = tp.build_optimizer(self.optimizer, model.parameters())
            # Optional LR schedule, built against the real step count so a
            # follow-the-trainer horizon (t_max 0 / OneCycleLR) is exact.
            lr_schedule = None
            schedule_needs_metric = False
            if self.scheduler is not None:
                lr_schedule, schedule_needs_metric = tp.build_scheduler(
                    self.scheduler, trainer, iterations
                )
            # Optional early stopping on the smoothed loss; the tracker owns no
            # tensors, the snapshot / rollback below do.
            stopper = tp.EarlyStopTracker(self.early_stop_patience, self.early_stop_min_delta)
            best_state: dict[str, torch.Tensor] | None = None
            best_length = 0
            stopped_at: int | None = None

            # Breakpoint continuation (crash site M3): a resumed run (same
            # prompt_id) restores the interrupted training state; a fresh Run
            # has a new prompt_id, the stale checkpoint is discarded.
            from comfy.model_management import InterruptProcessingException
            from comfy.train_checkpoint import (
                discard as ckpt_discard,
                load as ckpt_load,
                save as ckpt_save,
            )

            ckpt_run = self.ckpt_key_fn(data) if self.ckpt_key_fn is not None else None
            start_step = 0
            if ckpt_run is not None:
                saved = ckpt_load(ckpt_run)
                if saved is not None:
                    try:
                        model.load_state_dict(saved["model"])
                        trainer.load_state_dict(saved["optimizer"])
                        if lr_schedule is not None and saved.get("scheduler"):
                            lr_schedule.load_state_dict(saved["scheduler"])
                        history = list(saved["history"])
                        stopper.best = saved["stopper_best"]
                        stopper.best_step = int(saved["stopper_best_step"])
                        stopper._window = list(saved["stopper_window"])
                        stopper._since_best = int(saved["stopper_since_best"])
                        best_state = saved["best_state"]
                        best_length = int(saved["best_length"])
                        stopped_at = saved["stopped_at"]
                        shuffler.set_state(saved["shuffler"])
                        torch.set_rng_state(saved["torch_rng"])
                        start_step = int(saved["step"]) + 1
                    except Exception:  # corrupt payload: train fresh
                        start_step = 0
            # The interrupt exception is raised from inside the progress
            # update (the host's hook) - caught so the exact breakpoint state
            # is saved before it propagates (crash site M3).
            step = start_step - 1
            try:
                for step in range(start_step, iterations):
                    if per_step:
                        order = torch.randperm(data.samples, generator=shuffler)[:per_step].to(device)
                        inputs = data.inputs.index_select(0, order)
                        targets = data.targets.index_select(0, order)
                    else:
                        inputs, targets = data.inputs, data.targets
                    step_loss = _loss_value(data.loss, model(inputs), targets)
                    trainer.zero_grad(set_to_none=True)
                    step_loss.backward()
                    if self.optimizer.grad_clip_norm > 0.0 or self.optimizer.grad_clip_value > 0.0:
                        tp.clip_gradients(self.optimizer, model.parameters())
                    trainer.step()
                    if lr_schedule is not None:
                        if schedule_needs_metric:
                            lr_schedule.step(float(step_loss.detach()))
                        else:
                            lr_schedule.step()
                    # Detached scalars, stacked into one tensor at the end: the loss
                    # history must not keep the graph (or the dataset) alive.
                    history.append(step_loss.detach().reshape(()))
                    if stopper.update(step, float(step_loss.detach())):
                        stopped_at = step
                        break
                    if stopper.improved:
                        best_state = tp.snapshot_state(model)
                        best_length = len(history)
                    if self.progress is not None:
                        self.progress(step, float(step_loss.detach()))
            except InterruptProcessingException:
                if ckpt_run is not None:
                    ckpt_save(ckpt_run, {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": trainer.state_dict(),
                        "scheduler": (lr_schedule.state_dict()
                                      if lr_schedule is not None else None),
                        "history": history,
                        "stopper_best": stopper.best,
                        "stopper_best_step": stopper.best_step,
                        "stopper_window": list(stopper._window),
                        "stopper_since_best": stopper._since_best,
                        "best_state": best_state,
                        "best_length": best_length,
                        "stopped_at": stopped_at,
                        "shuffler": shuffler.get_state(),
                        "torch_rng": torch.get_rng_state(),
                    })
                raise
            if stopped_at is not None and best_state is not None:
                # Roll back to the best point and cut the history there, so the
                # outputs are what "the best model of this run" means.
                tp.restore_state(model, best_state)
                del history[best_length:]
                _warn(
                    f"{self.label}: early stop at step {stopped_at + 1} - "
                    f"{stopper.describe_stop()}"
                )
            trainer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.grad = None
            # The run finished (naturally or via early stop) - no continuation
            # is wanted anymore, so the breakpoint file goes away (M3).
            if ckpt_run is not None:
                ckpt_discard(ckpt_run)
            curve = torch.stack(history)
            with torch.no_grad():
                predictions = model(data.inputs).detach()

        return SessionResult(
            model=model,
            history=curve,
            predictions=predictions,
            stopped_at=stopped_at,
            notes=notes,
            iterations=iterations,
            per_step=per_step,
            widths=self.widths,
            data=data,
        )
