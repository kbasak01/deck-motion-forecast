"""Accuracy metrics per (model, regime, DOF, horizon).

All inputs are in **corpus units**: degrees for roll and pitch, metres for heave. Skill
score is defined against persistence::

    skill = 1 - MSE_model / MSE_persistence

so that 0 means "no better than repeating the last sample" and 1 means perfect. Negative
values are possible, are meaningful, and are reported rather than clipped.
"""

import pandas as pd

from dmf.typedefs import FloatArray

__all__ = ["mae", "per_dof_horizon_metrics", "rmse", "skill_score"]


def rmse(
    pred: FloatArray, target: FloatArray, axis: int | tuple[int, ...] | None = None
) -> FloatArray:
    """Compute root mean squared error.

    Args:
        pred: Forecasts, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        axis: Axes to reduce over. None reduces over all axes. Pass ``0`` to retain the
            per-(horizon, channel) structure the results tables need.

    Returns:
        RMSE, in the same units as the inputs (degrees for angles, metres for heave).

    Raises:
        ValueError: If ``pred`` and ``target`` differ in shape.
    """
    raise NotImplementedError


def mae(
    pred: FloatArray, target: FloatArray, axis: int | tuple[int, ...] | None = None
) -> FloatArray:
    """Compute mean absolute error.

    Args:
        pred: Forecasts, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        axis: Axes to reduce over. None reduces over all axes.

    Returns:
        MAE, in the same units as the inputs.

    Raises:
        ValueError: If ``pred`` and ``target`` differ in shape.
    """
    raise NotImplementedError


def skill_score(mse_model: FloatArray, mse_persistence: FloatArray) -> FloatArray:
    """Compute skill score against the persistence baseline.

    ``1 - mse_model/mse_persistence``.

    Args:
        mse_model: Model mean squared error, in squared corpus units.
        mse_persistence: Persistence mean squared error over the **same windows**, in
            squared corpus units. Computing the baseline over a different window set is
            the usual way a skill score ends up flattering the model.

    Returns:
        Skill score, dimensionless. Zero means parity with persistence; negative means
        worse than persistence, which is reported as found rather than clipped to zero.

    Raises:
        ValueError: If the shapes differ, or if any element of ``mse_persistence`` is
            zero.
    """
    raise NotImplementedError


def per_dof_horizon_metrics(
    pred: FloatArray,
    target: FloatArray,
    persistence_pred: FloatArray,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
) -> pd.DataFrame:
    """Build the core results table for one (model, regime) pair.

    Args:
        pred: Model forecasts, shape ``(N, H_max, C)``, in corpus units.
        target: Targets, shape ``(N, H_max, C)``, in corpus units.
        persistence_pred: Persistence forecasts over the same windows, shape
            ``(N, H_max, C)``, in corpus units.
        dof_names: Target channel names, length ``C``.
        horizons: Horizons to report, samples, each at most ``H_max``.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds alongside
            samples.

    Returns:
        One row per (DOF, horizon), with columns ``dof``, ``horizon_samples``,
        ``horizon_s``, ``rmse``, ``mae``, ``rmse_persistence``, ``skill``. Units are
        degrees for roll and pitch, metres for heave.

    Raises:
        ValueError: If the three arrays differ in shape, or if a requested horizon exceeds
            ``H_max``.
    """
    raise NotImplementedError
