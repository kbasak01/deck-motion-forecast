"""Loss functions for point, quantile, and Gaussian heads.

All losses operate on **normalised, de-meaned** tensors, not corpus units. Metrics are the
other way round -- always computed in corpus units, so that an RMSE can be compared
against an operational threshold in degrees or metres.
"""

import torch
from torch import Tensor

__all__ = ["gaussian_nll_loss", "mae_loss", "mse_loss", "pinball_loss"]


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
    raise NotImplementedError


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
    raise NotImplementedError


def gaussian_nll_loss(mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
    """Compute the Gaussian negative log-likelihood.

    ``0.5 * (log_var + (target - mean)^2 / exp(log_var))``, averaged over all axes and
    dropping the constant term.

    Args:
        mean: Predicted means, shape ``(B, H, C)``, dimensionless.
        log_var: Predicted log-variances, shape ``(B, H, C)``, dimensionless.
        target: Targets, shape ``(B, H, C)``, dimensionless.

    Returns:
        Scalar loss, dimensionless.

    Raises:
        ValueError: If the three tensors differ in shape.
    """
    raise NotImplementedError
