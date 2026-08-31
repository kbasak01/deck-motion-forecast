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

**Skill is not comparable across horizons on this signal, so every table also carries a
normalised RMSE.** Persistence error tracks the target's autocorrelation, so the skill
denominator oscillates with the signal's own period rather than growing with lead time: on
``id``/test the persistence RMSE for roll *falls* from 7.088 deg at 50 samples to 3.794 deg
at 100, because 10 s is close to one roll period (``docs/protocol.md`` P3-D5/P3-D6, pinned
in ``tests/test_models.py::PERSISTENCE_RMSE_ID_TEST``). A skill-vs-horizon curve therefore
shows dips that belong to the reference and not to the model. :func:`nrmse` divides instead
by :func:`signal_std`, the standard deviation of the target over the partition being scored,
which is a property of the signal at that lead time alone: 1.0 means "no better than the
partition mean", lower is better, and it is monotone in difficulty in the way skill is not.

**That is not the "normalised space" this subpackage forbids.** :mod:`dmf.eval` scores in
corpus units and never in units of the *training* normalisation scale. ``signal_std`` is
measured on the held-out targets themselves, in corpus units, and enters only as the
denominator of a dimensionless ratio; no metric here is computed on normalised windows.

**Scale.** A production test partition holds hundreds of thousands of windows, so a
``(N, H, C)`` float64 array runs to several gigabytes per model per regime. The table is
therefore built from *sums* -- :func:`metrics_table_from_sums` -- which is what
:mod:`dmf.eval.runner` streams. The window count is deliberately not quoted here: it is a
function of the corpus and the horizon list and it has already moved once (P3-D6).
:func:`per_dof_horizon_metrics` keeps the array-shaped signature for small cases and tests
and reduces to the same code path.
"""

import numpy as np
import pandas as pd

from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "mae",
    "metrics_table_from_sums",
    "nrmse",
    "per_dof_horizon_metrics",
    "rmse",
    "signal_std",
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
    "signal_std",
    "nrmse",
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


def signal_std(sy: FloatArray, syy: FloatArray, n: int | IntArray | FloatArray) -> FloatArray:
    """Compute the population standard deviation of the target signal.

    ``sqrt(syy/n - (sy/n)**2)`` -- the spread of the target at lead time exactly ``h``, per
    channel, over the whole partition being scored. It is a property of the *targets*, so it
    is identical for every model scored over the same windows and is accumulated once per
    batch rather than once per model (:func:`dmf.eval.runner.evaluate_models`).

    Args:
        sy: Summed target, shape ``(H, C)`` or any broadcastable shape, in **corpus units**
            (degrees for roll and pitch, metres for heave, deg/s and m/s for the rates).
        syy: Summed squared target over the same windows, same shape, in squared corpus
            units.
        n: Number of windows the sums were accumulated over; a scalar, or an array
            broadcastable against ``sy`` when the rows differ in window count. Note that
            windows overlap, so this is a window count and not an independent-sample count;
            the value is the spread of the window population, which is what the reported
            RMSE is averaged over.

    Returns:
        Standard deviation, shape of ``sy``, in **corpus units** -- the same units as
        :func:`rmse`, which is what makes their ratio dimensionless.

    Raises:
        ValueError: If ``sy`` and ``syy`` differ in shape, if ``n`` is not positive, or if
            any variance is zero or non-finite. Refused rather than returned as ``inf`` or
            ``nan``, following :func:`skill_score`: a constant channel is a corpus bug, not
            a perfectly easy forecast.
    """
    sums = np.asarray(sy, dtype=np.float64)
    squares = np.asarray(syy, dtype=np.float64)
    if sums.shape != squares.shape:
        raise ValueError(f"sy has shape {sums.shape} but syy has shape {squares.shape}")
    count = np.asarray(n, dtype=np.float64)
    if np.any(count < 1.0):
        raise ValueError(f"n must be positive, got {n!r}")
    mean = sums / count
    variance = squares / count - mean * mean
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError(
            "the target signal has zero or non-finite variance at some reported cell, so a "
            "normalised RMSE is undefined there. This happens on a constant channel, which "
            "is a corpus bug, not a perfectly easy forecast."
        )
    return np.asarray(np.sqrt(variance), dtype=np.float64)


def nrmse(rmse_value: FloatArray, std_value: FloatArray) -> FloatArray:
    """Normalise an RMSE by the standard deviation of the signal it forecasts.

    ``rmse / signal_std``. 1.0 means "no better than predicting the partition mean of that
    channel at that lead time"; lower is better; there is no upper bound. Reported alongside
    skill because skill's denominator is persistence, whose error tracks the autocorrelation
    and is *not* monotone in lead time on this signal -- see the module docstring.

    Args:
        rmse_value: Root mean squared error, in corpus units, over the same windows as
            ``std_value``.
        std_value: Target standard deviation from :func:`signal_std`, in the same corpus
            units and over the same windows.

    Returns:
        Normalised RMSE, **dimensionless**, shape of the broadcast inputs.

    Raises:
        ValueError: If the shapes do not agree, or if any standard deviation is zero or
            non-finite -- the same refusal :func:`signal_std` makes, repeated here because
            this function can be called with a denominator computed elsewhere.
    """
    numerator = np.asarray(rmse_value, dtype=np.float64)
    denominator = np.asarray(std_value, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError(
            f"rmse_value has shape {numerator.shape} but std_value has shape "
            f"{denominator.shape}; a normalised RMSE must divide by the spread of the same "
            f"targets the error was measured against"
        )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError(
            "signal_std contains a zero or non-finite value: nrmse is undefined where the "
            "target does not vary. This happens on a constant channel, which is a corpus "
            "bug, not a perfectly easy forecast."
        )
    return np.asarray(numerator / denominator, dtype=np.float64)


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
    sy: FloatArray,
    syy: FloatArray,
    n: int,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
) -> pd.DataFrame:
    """Build the core results table from streamed error sums.

    The production form of :func:`per_dof_horizon_metrics`. Nothing here is ever
    ``(N, H, C)``-shaped, so a production test partition costs ``H * C`` float64 values per
    model rather than one entry per window, independently of how many windows it holds.

    ``n`` cancels out of the skill score (``1 - sse_model/sse_persistence``), which is why
    the reference model's own skill is *bitwise* zero rather than zero to rounding.

    Args:
        sse: Model summed squared error, shape ``(H, C)``, squared corpus units.
        sae: Model summed absolute error, shape ``(H, C)``, corpus units.
        sse_persistence: Persistence summed squared error over the **same windows**, shape
            ``(H, C)``, squared corpus units.
        sy: Summed target over the same windows, shape ``(H, C)``, corpus units. Model
            independent, which is why :func:`dmf.eval.runner.evaluate_models` accumulates it
            once per batch and shares it between every model's accumulator.
        syy: Summed squared target over the same windows, shape ``(H, C)``, squared corpus
            units.
        n: Number of windows the sums were accumulated over. Note that consecutive windows
            overlap at ``stride < lookback``, so ``n`` is a window count and **not** an
            independent-sample count; see :func:`dmf.eval.runner.evaluate_models`.
        dof_names: Target channel names, length ``C``.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds alongside
            samples.

    Returns:
        One row per (DOF, horizon), columns :data:`METRIC_COLUMNS`. Units are degrees for
        roll and pitch and metres for heave; ``signal_std`` is in those same corpus units
        and ``skill`` and ``nrmse`` are dimensionless.

    Raises:
        ValueError: If the five sum arrays differ in shape, if ``dof_names`` does not match
            the channel axis, if ``n`` is not positive, if ``fs_hz`` is not positive, if a
            requested horizon is outside ``[1, H]``, or if the target has zero variance at a
            reported cell.
    """
    model_sse = np.asarray(sse, dtype=np.float64)
    model_sae = np.asarray(sae, dtype=np.float64)
    reference_sse = np.asarray(sse_persistence, dtype=np.float64)
    target_sy = np.asarray(sy, dtype=np.float64)
    target_syy = np.asarray(syy, dtype=np.float64)
    if model_sse.ndim != 2:
        raise ValueError(f"sse must have shape (H, C), got {model_sse.shape}")
    if not (
        model_sse.shape
        == model_sae.shape
        == reference_sse.shape
        == target_sy.shape
        == target_syy.shape
    ):
        raise ValueError(
            f"sse {model_sse.shape}, sae {model_sae.shape}, sse_persistence "
            f"{reference_sse.shape}, sy {target_sy.shape} and syy {target_syy.shape} must "
            f"agree"
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
    # Same argument for the normalised RMSE: a constant channel at an unreported horizon
    # must not fail a table that never quotes it. Both grids are (n_horizons, C).
    std_grid = signal_std(target_sy[index, :], target_syy[index, :], n)
    rmse_grid = np.sqrt(model_sse[index, :] / n)
    nrmse_grid = nrmse(rmse_grid, std_grid)

    rows: list[dict[str, object]] = []
    for channel, name in enumerate(dof_names):
        for position, (horizon, step) in enumerate(zip(horizons, index.tolist(), strict=True)):
            cell_reference = float(reference_sse[step, channel])
            rows.append(
                {
                    "dof": name,
                    "horizon_samples": int(horizon),
                    "horizon_s": float(horizon) / fs_hz,
                    "n_windows": int(n),
                    "rmse": float(rmse_grid[position, channel]),
                    "mae": float(model_sae[step, channel] / n),
                    "rmse_persistence": float(np.sqrt(cell_reference / n)),
                    "skill": float(skill_grid[position, channel]),
                    "signal_std": float(std_grid[position, channel]),
                    "nrmse": float(nrmse_grid[position, channel]),
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
        One row per (DOF, horizon), columns :data:`METRIC_COLUMNS`. Units are degrees for
        roll and pitch and metres for heave; ``skill`` and ``nrmse`` are dimensionless.

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
        sy=truth.sum(axis=0),
        syy=np.square(truth).sum(axis=0),
        n=int(forecast.shape[0]),
        dof_names=dof_names,
        horizons=horizons,
        fs_hz=fs_hz,
    )
