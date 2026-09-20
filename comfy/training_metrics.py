"""Shared loss and metric functions for the training nodes.

This module is the *math layer* of the training loop completion: every loss
and metric formula lives here exactly once and is reused by the ``Loss`` and
``Metrics`` nodes, the ``Evaluate`` node and both trainer nodes, so a formula
can never drift between the places it is used.

Like :mod:`comfy.training_protocol`, this module depends only on ``torch`` and
is safe to import inside the dehydrated build.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

__author__ = "ComfyDL_UI contributors"

LOGGER = logging.getLogger("comfydl.training_metrics")

#: Loss functions of the ``Loss`` widget, in dropdown order.
LOSS_OPTIONS: tuple[str, ...] = (
    "mse",
    "l1",
    "smooth_l1",
    "cross_entropy",
    "bce_with_logits",
    "kl_div",
)

#: Metrics of the ``Metrics`` widget, in dropdown order.
METRIC_OPTIONS: tuple[str, ...] = (
    "mae",
    "rmse",
    "accuracy",
    "top_3",
    "top_5",
    "perplexity",
)

#: ``"auto"`` choice of the ``Evaluate`` node: pick the loss / metric from the
#: dtypes and shapes of ``prediction`` and ``target``.
AUTO = "auto"

#: Loss choices of the ``Evaluate`` node = ``auto`` + the explicit list.
AUTO_LOSS_OPTIONS: tuple[str, ...] = (AUTO,) + LOSS_OPTIONS

#: Metric choices of the ``Evaluate`` node = ``auto`` + the explicit list.
AUTO_METRIC_OPTIONS: tuple[str, ...] = (AUTO,) + METRIC_OPTIONS

#: Sentinel for "no metric wanted" in the ``Evaluate`` node.
NO_METRIC = "none"

#: Metric choices of the ``Evaluate`` node including "none" (skip the metric).
EVALUATE_METRIC_OPTIONS: tuple[str, ...] = (AUTO,) + METRIC_OPTIONS + (NO_METRIC,)


def _class_axes(prediction: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Flatten ``prediction`` to ``(N, C)`` and return the class count.

    In:  prediction - logits whose *last* axis is the classes: ``(N, C)`` for a
         plain classifier head, ``(N, L, C)`` e.g. for token logits.
    Out: ``(flat, classes)``. Raises ``ValueError`` with a readable message
         when the tensor has no class axis (dim < 2).
    """
    if prediction.dim() < 2:
        raise ValueError(
            "classification losses need a class axis: expected shape (N, C, ...) "
            f"but got {tuple(prediction.shape)}"
        )
    classes = int(prediction.shape[-1])
    return prediction.reshape(-1, classes), classes


def _as_class_targets(
    target: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
    """Turn ``target`` into flat class indices aligned with a flattened logit
    matrix.

    In:  target - either integer/1-D class indices shaped ``(N,)`` /
         ``(N, d...)`` or a float tensor of the same shape as ``prediction``
         holding probabilities / one-hot rows (the argmax is taken).
         prediction - the logits, used for shape checks only.
    Out: a 1-D ``long`` tensor of length ``N`` matching the flattened logit
         matrix. Raises ``ValueError`` when the shapes cannot be aligned.
    """
    flat, classes = _class_axes(prediction)
    rows = flat.shape[0]
    if target.shape == prediction.shape:
        if not target.is_floating_point():
            raise ValueError(
                "class targets shaped like the prediction must be floats "
                f"(one-hot / probabilities); got dtype {target.dtype}"
            )
        return target.reshape(-1, classes).argmax(dim=1)
    if target.numel() == rows:
        return target.reshape(-1).long()
    raise ValueError(
        "cannot align target with prediction for a classification loss: "
        f"prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
    )


def _same_shape(prediction: torch.Tensor, target: torch.Tensor, label: str) -> None:
    """Raise a readable error unless prediction and target agree in shape."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"{label} needs prediction and target of the same shape; got "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )


def compute_loss(name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute one loss between ``prediction`` and ``target``.

    What: the single implementation of the six loss functions of the training
          nodes, mirroring the ``torch.nn.functional`` reference implementations.
    In:   name - one of :data:`LOSS_OPTIONS`.
          prediction - model output; logits for the classification losses,
          raw values for the regression losses.
          target - ground truth; float tensors of the prediction's shape for
          ``mse`` / ``l1`` / ``smooth_l1`` / ``bce_with_logits``, class indices
          or one-hot / probability rows for ``cross_entropy``, probabilities of
          the prediction's shape for ``kl_div``.
    Out: a scalar ``torch.Tensor`` (mean reduction), carrying gradients w.r.t.
         ``prediction`` when the inputs do.
    """
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("loss inputs must be torch tensors")
    text = str(name or "").strip()
    if text not in LOSS_OPTIONS:
        raise ValueError(f"unknown loss {name!r}; expected one of {LOSS_OPTIONS}")
    if text == "cross_entropy":
        flat = _class_axes(prediction)[0]
        rows = _as_class_targets(target, prediction)
        return F.cross_entropy(flat, rows)
    if text == "bce_with_logits":
        _same_shape(prediction, target, "bce_with_logits")
        return F.binary_cross_entropy_with_logits(prediction, target.float())
    if text == "kl_div":
        _same_shape(prediction, target, "kl_div")
        if target.dim() < 2:
            raise ValueError("kl_div needs at least a (N, C) shape")
        log_probs = F.log_softmax(prediction.float(), dim=-1)
        probs = F.softmax(target.float(), dim=-1)
        return F.kl_div(log_probs, probs, reduction="batchmean")
    if text == "mse":
        _same_shape(prediction, target, "mse")
        return F.mse_loss(prediction.float(), target.float())
    if text == "l1":
        _same_shape(prediction, target, "l1")
        return F.l1_loss(prediction.float(), target.float())
    # smooth_l1
    _same_shape(prediction, target, "smooth_l1")
    return F.smooth_l1_loss(prediction.float(), target.float())


def compute_metric(name: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute one metric between ``prediction`` and ``target``.

    What: the single implementation of the six metrics of the training nodes.
    In:   name - one of :data:`METRIC_OPTIONS`.
          prediction - model output; logits for the classification metrics,
          raw values for the regression metrics.
          target - float tensors of the prediction's shape for ``mae`` /
          ``rmse``; class indices or one-hot / probability rows for
          ``accuracy`` / ``top_3`` / ``top_5`` / ``perplexity``.
    Out: a scalar ``torch.Tensor`` with no gradient, detached from the graph.
    """
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("metric inputs must be torch tensors")
    text = str(name or "").strip()
    if text not in METRIC_OPTIONS:
        raise ValueError(f"unknown metric {name!r}; expected one of {METRIC_OPTIONS}")
    if text in ("mae", "rmse"):
        _same_shape(prediction, target, text)
        diff = (prediction.float() - target.float()).abs()
        if text == "mae":
            return diff.mean().detach()
        return diff.pow(2).mean().sqrt().detach()
    if text == "perplexity":
        flat = _class_axes(prediction)[0]
        rows = _as_class_targets(target, prediction)
        return F.cross_entropy(flat, rows).exp().detach()
    # accuracy / top_k
    flat, classes = _class_axes(prediction)
    rows = _as_class_targets(target, prediction)
    if text == "accuracy":
        k = 1
    elif text == "top_3":
        k = 3
    else:
        k = 5
    k = min(k, classes)
    top = flat.topk(k, dim=1).indices
    correct = (top == rows.reshape(-1, 1)).any(dim=1)
    return correct.float().mean().detach()


def resolve_loss(name: str, prediction: torch.Tensor, target: torch.Tensor) -> str:
    """Resolve the ``auto`` loss choice into a concrete loss name.

    In:  name - ``"auto"`` or a name from :data:`LOSS_OPTIONS`.
         prediction / target - the pair the heuristic inspects.
    Out: ``cross_entropy`` when the target is a class-index tensor (integer
         dtype, or a smaller tensor aligned with the flattened class axis)
         or a one-hot / probability tensor of the prediction's shape that does
         *not* look like plain regression data; ``mse`` otherwise. A concrete
         name is validated and returned unchanged.
    """
    text = str(name or "").strip()
    if text != AUTO:
        if text not in LOSS_OPTIONS:
            raise ValueError(f"unknown loss {name!r}; expected 'auto' or one of {LOSS_OPTIONS}")
        return text
    if _looks_like_classification(prediction, target):
        return "cross_entropy"
    return "mse"


def resolve_metric(name: str, prediction: torch.Tensor, target: torch.Tensor) -> str | None:
    """Resolve the ``auto`` metric choice into a concrete metric name.

    In:  name - ``"auto"``, :data:`NO_METRIC` or a name from
         :data:`METRIC_OPTIONS`.
         prediction / target - the pair the heuristic inspects.
    Out: ``None`` for ``none`` (caller skips the metric);
         ``accuracy`` when the pair looks like classification data
         (see :func:`resolve_loss`); ``mae`` otherwise. A concrete name is
         validated and returned unchanged.
    """
    text = str(name or "").strip()
    if text == NO_METRIC:
        return None
    if text == AUTO:
        return "accuracy" if _looks_like_classification(prediction, target) else "mae"
    if text not in METRIC_OPTIONS:
        raise ValueError(
            f"unknown metric {name!r}; expected 'auto', 'none' or one of {METRIC_OPTIONS}"
        )
    return text


def _looks_like_classification(prediction: torch.Tensor, target: torch.Tensor) -> bool:
    """Heuristic behind ``auto``: is this pair classification data?

    Integer targets that are not shaped like the prediction are class indices;
    float targets of the prediction's shape that hold only 0/1 values across
    the class axis are one-hot rows. Everything else is treated as regression.
    """
    if not target.is_floating_point():
        return True
    if target.shape == prediction.shape and prediction.dim() >= 2:
        classes = int(prediction.shape[1])
        rows = target.reshape(-1, classes)
        if rows.min() >= 0 and rows.max() <= 1 and (rows.sum(dim=1) - 1.0).abs().max() < 1e-3:
            return True
    return False


__all__ = [
    "AUTO",
    "AUTO_LOSS_OPTIONS",
    "AUTO_METRIC_OPTIONS",
    "EVALUATE_METRIC_OPTIONS",
    "LOSS_OPTIONS",
    "METRIC_OPTIONS",
    "NO_METRIC",
    "compute_loss",
    "compute_metric",
    "resolve_loss",
    "resolve_metric",
]
