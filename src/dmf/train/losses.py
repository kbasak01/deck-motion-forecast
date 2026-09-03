"""Loss functions for point, quantile, and Gaussian heads.

All losses operate on **normalised, de-meaned** tensors, not corpus units. Metrics are the
other way round -- always computed in corpus units, so that an RMSE can be compared
against an operational threshold in degrees or metres.

:func:`resolve_loss` maps a head kind to the objective it is fitted with, so the training
loop's call site stays one line and no model can be trained on an objective that does not
match its output shape. Per ``docs/protocol.md`` P5-D4 each model is stopped on **its own**
training objective, which means ``best_val_loss`` is not comparable across heads -- a
pinball loss and an MSE are different quantities and their ratio means nothing. Everything
else about the budget stays identical across models.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from dmf.models.heads import HeadKind

__all__ = [
    "LossFn",
    "gaussian_nll_loss",
    "mae_loss",
    "mse_loss",
    "pinball_loss",
    "resolve_loss",
]

#: A training objective: ``(pred, target) -> scalar``. ``pred`` carries whatever shape the
#: model's head emits -- ``(B, H, C)``, ``(B, H, C, Q)`` or ``(B, H, C, 2)`` -- and
#: ``target`` is always ``(B, H, C)``, dimensionless.
LossFn = Callable[[Tensor, Tensor], Tensor]


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean squared error.

    Args:
        pred: Point forecasts, shape ``(B, H, C)``, dimensionless.
        target: Targets, shape ``(B, H, C)``, dimensionless.

    Returns:
        Scalar loss, dimensionless.

    Raises:
        ValueError: If ``pred`` and ``target`` differ in shape.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape"
        )
    return torch.mean(torch.square(pred - target))


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute mean absolute error.

    Args:
        pred: Point forecasts, shape ``(B, H, C)``, dimensionless.
        target: Targets, shape ``(B, H, C)``, dimensionless.

    Returns:
        Scalar loss, dimensionless.

    Raises:
        ValueError: If ``pred`` and ``target`` differ in shape.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape"
        )
    return torch.mean(torch.abs(pred - target))


def pinball_loss(pred: Tensor, target: Tensor, quantiles: tuple[float, ...]) -> Tensor:
    """Compute the pinball (quantile) loss over a quantile fan.

    For quantile level ``q``, error ``e = target - pred_q``, the per-element loss is
    ``max(q*e, (q-1)*e)``, averaged over batch, horizon, channel and quantile.

    Args:
        pred: Quantile forecasts, shape ``(B, H, C, Q)``, dimensionless.
        target: Targets, shape ``(B, H, C)``, dimensionless. Broadcast across ``Q``.
        quantiles: Quantile levels, each in (0, 1), ascending, length ``Q``.

    Returns:
        Scalar loss, dimensionless.

    Raises:
        ValueError: If ``len(quantiles)`` does not match the ``Q`` axis of ``pred``, or if
            the leading shapes of ``pred`` and ``target`` disagree.
    """
    if pred.ndim != 4:
        raise ValueError(f"pred must have shape (B, H, C, Q), got {tuple(pred.shape)}")
    if tuple(pred.shape[:3]) != tuple(target.shape):
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} disagree on (B, H, C)"
        )
    if len(quantiles) != int(pred.shape[-1]):
        raise ValueError(
            f"{len(quantiles)} quantile levels against a fan of width {int(pred.shape[-1])}"
        )
    levels = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    error = target.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(levels * error, (levels - 1.0) * error))


def gaussian_nll_loss(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Compute the Gaussian negative log-likelihood.

    ``0.5 * (log_var + (target - mean)^2 / exp(log_var))``, averaged over all axes and
    dropping the constant term. The dropped constant is ``0.5 * log(2*pi)`` per element, so
    this agrees with ``torch.nn.functional.gaussian_nll_loss(..., full=False)``, which drops
    the same constant, up to that function's variance clamp at ``eps``.

    Args:
        mean: Predicted means, shape ``(B, H, C)``, dimensionless.
        log_var: Predicted log-variances, shape ``(B, H, C)``, dimensionless. Clamped to
            ``[LOG_VAR_MIN, LOG_VAR_MAX]`` by the model's head
            (:func:`dmf.models.heads.clamp_log_variance`), not here: the bound belongs to
            the parameterisation, and a loss that silently repaired an unbounded head would
            hide the divergence rather than prevent it.
        target: Targets, shape ``(B, H, C)``, dimensionless.

    Returns:
        Scalar loss, dimensionless.

    Raises:
        ValueError: If the three tensors differ in shape.
    """
    if mean.shape != log_var.shape or mean.shape != target.shape:
        raise ValueError(
            f"mean {tuple(mean.shape)}, log_var {tuple(log_var.shape)} and target "
            f"{tuple(target.shape)} must have the same shape"
        )
    return torch.mean(0.5 * (log_var + torch.square(target - mean) * torch.exp(-log_var)))


def resolve_loss(head: HeadKind, quantiles: tuple[float, ...] = ()) -> LossFn:
    """Return the training objective for a head kind.

    One call site in :func:`dmf.train.experiment._fit_one` resolves this and threads it
    through :func:`dmf.train.loop.fit`, so the loop never branches on a head and a new head
    kind arrives with its objective attached.

    Args:
        head: The model's head kind.
        quantiles: Quantile levels, required for the ``quantile`` head and rejected for the
            others.

    Returns:
        A closure ``(pred, target) -> scalar``. ``pred`` is the model's raw output:
        ``(B, H, C)`` scored by MSE for ``point``, ``(B, H, C, Q)`` scored by pinball loss
        for ``quantile``, and ``(B, H, C, 2)`` split into ``pred[..., 0]`` (mean) and
        ``pred[..., 1]`` (log-variance) and scored by Gaussian NLL for ``gaussian``.

    Raises:
        ValueError: If ``head`` is unknown, or if ``quantiles`` does not match it -- which
            is what stops a quantile model being trained on MSE against a fan it cannot
            broadcast to.
    """
    if head == "point":
        if quantiles:
            raise ValueError(f"a point head carries no quantile levels, got {list(quantiles)}")
        return mse_loss
    if head == "quantile":
        if len(quantiles) < 2:
            raise ValueError(f"a quantile head needs at least 2 levels, got {list(quantiles)}")
        levels = tuple(quantiles)

        def _pinball(pred: Tensor, target: Tensor) -> Tensor:
            """Pinball loss at the configured fan."""
            return pinball_loss(pred, target, levels)

        return _pinball
    if head == "gaussian":
        if quantiles:
            raise ValueError(f"a gaussian head carries no quantile levels, got {list(quantiles)}")

        def _nll(pred: Tensor, target: Tensor) -> Tensor:
            """Gaussian NLL over a ``(B, H, C, 2)`` mean/log-variance output."""
            if pred.ndim != 4 or int(pred.shape[-1]) != 2:
                raise ValueError(f"a gaussian head must emit (B, H, C, 2), got {tuple(pred.shape)}")
            return gaussian_nll_loss(pred[..., 0], pred[..., 1], target)

        return _nll
    raise ValueError(f"unknown head kind {head!r}; expected point, quantile or gaussian")
