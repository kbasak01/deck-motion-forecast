"""Accuracy metrics per (model, regime, DOF, horizon).

All inputs are in **corpus units**: degrees for roll and pitch, metres for heave. Skill
score is defined against persistence::

    skill = 1 - MSE_model / MSE_persistence

so that 0 means "no better than repeating the last sample" and 1 means perfect. Negative
values are possible, are meaningful, and are reported rather than clipped.

**Horizon convention, stated as a definition rather than left to be inferred.** "Horizon
``h`` samples" is the error at lead time *exactly* ``h``, never the mean over lead times
1..h. Index ``h`` therefore reads element ``h - 1`` of a zero-based ``(H, ...)`` array.
This is the convention already fixed by
:attr:`dmf.eval.controls.PipelineSanityResult.rmse_pipeline` and by the P2-D9 table in
``docs/protocol.md``; ``tests/test_metrics.py`` pins it with
``test_horizon_metric_is_per_step_not_cumulative``.

**Scale.** A production test partition is ~442 000 windows, so a ``(N, H, C)`` float64
array is ~0.5 GB per model per regime. The table is therefore built from *sums* --
:func:`metrics_table_from_sums` -- which is what :mod:`dmf.eval.runner` streams.
:func:`per_dof_horizon_metrics` keeps the array-shaped signature for small cases and tests
and reduces to the same code path.
"""

import numpy as np
import pandas as pd

from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "mae",
    "metrics_table_from_sums",
    "per_dof_horizon_metrics",
    "rmse",
    "skill_score",
]

#: Column order of the core per-(DOF, horizon) results table. Fixed here so that the
#: runner, the report writer and the tests cannot drift apart.
METRIC_COLUMNS: tuple[str, ...] = (
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_windows",
    "rmse",
    "mae",
    "rmse_persistence",
    "skill",
)


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
    if pred.shape != target.shape:
        raise ValueError(f"pred has shape {pred.shape} but target has shape {target.shape}")
    if pred.size == 0:
        raise ValueError("cannot compute RMSE over an empty array")
    squared_error = np.square(np.asarray(pred, dtype=np.float64) - np.asarray(target, np.float64))
    return np.sqrt(squared_error.mean(axis=axis))


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
    if pred.shape != target.shape:
        raise ValueError(f"pred has shape {pred.shape} but target has shape {target.shape}")
    if pred.size == 0:
        raise ValueError("cannot compute MAE over an empty array")
    absolute_error = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, np.float64))
    return np.asarray(absolute_error.mean(axis=axis), dtype=np.float64)


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
    model = np.asarray(mse_model, dtype=np.float64)
    reference = np.asarray(mse_persistence, dtype=np.float64)
    if model.shape != reference.shape:
        raise ValueError(
            f"mse_model has shape {model.shape} but mse_persistence has shape {reference.shape}; "
            f"the skill denominator must be measured over the same windows as the numerator"
        )
    if np.any(reference == 0.0):
        raise ValueError(
            "mse_persistence contains a zero: the skill score is undefined where persistence "
            "is exact. This happens on a constant channel, which is a corpus bug, not a "
            "perfect forecast."
        )
    return np.asarray(1.0 - model / reference, dtype=np.float64)


def _horizon_index(horizons: tuple[int, ...], max_horizon: int) -> IntArray:
    """Validate the requested horizons and return their zero-based indices.

    Args:
        horizons: Horizons to report, samples. Each must be in ``[1, max_horizon]``.
        max_horizon: Number of forecast steps available, ``H``.

    Returns:
        Zero-based indices, shape ``(len(horizons),)``, i.e. ``h - 1`` per the per-step
        horizon convention documented in the module docstring.

    Raises:
        ValueError: If ``horizons`` is empty or any horizon is out of range.
    """
    if not horizons:
        raise ValueError("horizons must be non-empty")
    bad = [h for h in horizons if h < 1 or h > max_horizon]
    if bad:
        raise ValueError(
            f"horizons {bad} are outside [1, {max_horizon}]; 'horizon h' means the error at "
            f"lead time exactly h samples, so h indexes element h-1 of the horizon axis"
        )
    return np.asarray([h - 1 for h in horizons], dtype=np.int64)


def metrics_table_from_sums(
    sse: FloatArray,
    sae: FloatArray,
    sse_persistence: FloatArray,
    n: int,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
) -> pd.DataFrame:
    """Build the core results table from streamed error sums.

    The production form of :func:`per_dof_horizon_metrics`. Nothing here is ever
    ``(N, H, C)``-shaped, so a 442 000-window test partition costs 1200 float64 values per
    model instead of half a gigabyte.

    ``n`` cancels out of the skill score (``1 - sse_model/sse_persistence``), which is why
    the reference model's own skill is *bitwise* zero rather than zero to rounding.

    Args:
        sse: Model summed squared error, shape ``(H, C)``, squared corpus units.
        sae: Model summed absolute error, shape ``(H, C)``, corpus units.
        sse_persistence: Persistence summed squared error over the **same windows**, shape
            ``(H, C)``, squared corpus units.
        n: Number of windows the sums were accumulated over. Note that consecutive windows
            overlap at ``stride < lookback``, so ``n`` is a window count and **not** an
            independent-sample count; see :func:`dmf.eval.runner.evaluate_models`.
        dof_names: Target channel names, length ``C``.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds alongside
            samples.

    Returns:
        One row per (DOF, horizon), columns :data:`METRIC_COLUMNS`. Units are degrees for
        roll and pitch and metres for heave; ``skill`` is dimensionless.

    Raises:
        ValueError: If the three arrays differ in shape, if ``dof_names`` does not match
            the channel axis, if ``n`` is not positive, if ``fs_hz`` is not positive, or if
            a requested horizon is outside ``[1, H]``.
    """
    model_sse = np.asarray(sse, dtype=np.float64)
    model_sae = np.asarray(sae, dtype=np.float64)
    reference_sse = np.asarray(sse_persistence, dtype=np.float64)
    if model_sse.ndim != 2:
        raise ValueError(f"sse must have shape (H, C), got {model_sse.shape}")
    if not (model_sse.shape == model_sae.shape == reference_sse.shape):
        raise ValueError(
            f"sse {model_sse.shape}, sae {model_sae.shape} and sse_persistence "
            f"{reference_sse.shape} must agree"
        )
    if model_sse.shape[1] != len(dof_names):
        raise ValueError(
            f"sums cover {model_sse.shape[1]} channels but {len(dof_names)} DOF names were "
            f"given: {list(dof_names)}"
        )
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    index = _horizon_index(horizons, model_sse.shape[0])
    # Validated and computed once, over the reported cells only: a zero persistence SSE at
    # some horizon nobody asked for must not fail a table that never reports it.
    skill_grid = skill_score(model_sse[index, :], reference_sse[index, :])

    rows: list[dict[str, object]] = []
    for channel, name in enumerate(dof_names):
        for position, (horizon, step) in enumerate(zip(horizons, index.tolist(), strict=True)):
            cell_sse = float(model_sse[step, channel])
            cell_reference = float(reference_sse[step, channel])
            rows.append(
                {
                    "dof": name,
                    "horizon_samples": int(horizon),
                    "horizon_s": float(horizon) / fs_hz,
                    "n_windows": int(n),
                    "rmse": float(np.sqrt(cell_sse / n)),
                    "mae": float(model_sae[step, channel] / n),
                    "rmse_persistence": float(np.sqrt(cell_reference / n)),
                    "skill": float(skill_grid[position, channel]),
                }
            )
    return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))


def per_dof_horizon_metrics(
    pred: FloatArray,
    target: FloatArray,
    persistence_pred: FloatArray,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
) -> pd.DataFrame:
    """Build the core results table for one (model, regime) pair.

    A thin wrapper: it reduces its arrays to the summed squared and absolute errors and
    delegates to :func:`metrics_table_from_sums`. Materialising ``(N, H, C)`` arrays is
    only viable at test-fixture scale -- on the production corpus each of the three inputs
    is ~0.5 GB -- so production code accumulates the sums instead
    (:func:`dmf.eval.runner.evaluate_models`) and calls the delegate directly.

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
    forecast = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    reference = np.asarray(persistence_pred, dtype=np.float64)
    if forecast.ndim != 3:
        raise ValueError(f"pred must have shape (N, H, C), got {forecast.shape}")
    if not (forecast.shape == truth.shape == reference.shape):
        raise ValueError(
            f"pred {forecast.shape}, target {truth.shape} and persistence_pred "
            f"{reference.shape} must agree; a skill denominator measured over a different "
            f"window set is the usual way a skill score ends up flattering the model"
        )
    if forecast.shape[0] == 0:
        raise ValueError("cannot compute metrics over an empty window set")
    error = forecast - truth
    reference_error = reference - truth
    return metrics_table_from_sums(
        sse=np.square(error).sum(axis=0),
        sae=np.abs(error).sum(axis=0),
        sse_persistence=np.square(reference_error).sum(axis=0),
        n=int(forecast.shape[0]),
        dof_names=dof_names,
        horizons=horizons,
        fs_hz=fs_hz,
    )
