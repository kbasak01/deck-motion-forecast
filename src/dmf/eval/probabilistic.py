"""Probabilistic scoring: pinball, CRPS, PICP, interval width, Winkler.

All inputs are in **corpus units** -- degrees for roll and pitch, metres for heave, deg/s
and m/s for the rates -- so that an interval width can be read against a landing limit
directly. Nothing here is ever computed in the units of the training normalisation scale
(:mod:`dmf.eval`).

Gate 5 requires PICP@90 within [0.85, 0.95] on the ``id`` regime. The degradation under
``unseen_seastate`` is expected, is **reported rather than fixed**, and is one of the more
interesting findings available here: it is the same coverage-under-domain-shift story that
conformal prediction runs into, measured on a concrete operational task.

**Coverage is never reported without sharpness.** ``picp`` and ``mean_interval_width`` are
adjacent columns of :data:`PROBABILISTIC_METRIC_COLUMNS` and there is no convenience in this
module that returns a coverage table without a width beside it. Coverage alone is trivially
achievable by widening an interval until it is useless, and ``docs/protocol.md`` P5-D6
predicts before the sweep that ``unseen_heading`` pitch will do exactly that: PICP near 1.0
on a channel clamped ~26 dB down, with a width far wider than the signal.

**CRPS here is an approximation and is labelled as one everywhere it appears.**
:func:`crps_from_quantiles` integrates the pinball loss over the ``Q`` levels that were
actually predicted -- nine of them in ``dmf.models.heads.QUANTILE_FAN_9`` -- so it is a
finite-fan quadrature of the CRPS integral, not the exact CRPS of the predictive
distribution. The results tables say "crps (approx, Q levels)" rather than "CRPS", and the
``n_quantiles`` column of the table carries ``Q`` so that the quadrature the number came
from is visible in the row.

**Horizon convention, identical to :mod:`dmf.eval.metrics`.** "Horizon ``h`` samples" is the
score at lead time *exactly* ``h``, never the mean over lead times 1..h; index ``h`` reads
element ``h - 1`` of a zero-based ``(H, ...)`` axis. The horizon validator is imported from
:mod:`dmf.eval.metrics` rather than re-implemented, so the two tables cannot drift.

**Scale, and why there are two forms of every metric.** A production test partition holds
hundreds of thousands of windows, so materialising a ``(N, H, C, Q)`` float64 quantile fan
over one is terabytes. The exact count is a property of the geometry and the split and moves
whenever either does, so it is deliberately not quoted here. The scalar functions
(:func:`pinball`, :func:`crps_from_quantiles`, :func:`picp`, :func:`mean_interval_width`,
:func:`winkler_score`) are the **definitions and the test oracle**, viable at fixture scale
only. Production streams: a runner accumulates the
per-element terms (:func:`pinball_terms`, :func:`crps_terms`, :func:`coverage_terms`,
:func:`width_terms`, :func:`winkler_terms`, :func:`crossing_terms`) into per-realization sums
of shape ``(n_keys, H, C)`` and hands them to :func:`probabilistic_table_from_sums`. Keeping
the realization axis is what lets the existing realization-level bootstrap
(:func:`dmf.eval.runner.bootstrap_skill_ci`) apply unchanged; every scalar function is
literally ``float(terms.mean())`` of the matching term function, so the two paths are the
same computation by construction and not by inspection.
"""

import numpy as np
import pandas as pd

from dmf.eval.metrics import _horizon_index
from dmf.typedefs import FloatArray

__all__ = [
    "PROBABILISTIC_METRIC_COLUMNS",
    "coverage_terms",
    "crossing_terms",
    "crps_from_quantiles",
    "crps_terms",
    "mean_interval_width",
    "picp",
    "pinball",
    "pinball_terms",
    "probabilistic_table_from_sums",
    "width_terms",
    "winkler_score",
    "winkler_terms",
]

#: Column order of the per-(DOF, horizon) probabilistic results table. Fixed here so that
#: the runner, the report writer and the tests cannot drift apart, mirroring
#: :data:`dmf.eval.metrics.METRIC_COLUMNS`. ``picp`` is immediately followed by
#: ``mean_interval_width``: coverage and sharpness are reported together, always.
#: ``crps`` is the finite-fan approximation described in the module docstring, which is why
#: ``n_quantiles`` travels with it.
PROBABILISTIC_METRIC_COLUMNS: tuple[str, ...] = (
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_windows",
    "n_quantiles",
    "alpha",
    "picp",
    "mean_interval_width",
    "winkler",
    "crps",
    "pinball",
    "crossing_rate",
)


def _validated_levels(quantiles: tuple[float, ...], n_levels: int) -> FloatArray:
    """Validate quantile levels against the fan they label.

    Args:
        quantiles: Quantile levels, each in (0, 1), strictly ascending, length ``Q``.
        n_levels: Length of the ``Q`` axis of the fan being scored.

    Returns:
        The levels as a float64 array of shape ``(Q,)``, dimensionless.

    Raises:
        ValueError: If the length does not match the ``Q`` axis, if the tuple is empty, if
            any level is outside (0, 1), or if the levels are not strictly ascending. The
            last two are refused rather than sorted here: a fan whose levels are out of
            order is mislabelled, and silently reordering the labels would attach the wrong
            level to every column.
    """
    if not quantiles:
        raise ValueError("quantiles must be non-empty")
    if len(quantiles) != n_levels:
        raise ValueError(
            f"quantiles has length {len(quantiles)} but the fan has {n_levels} levels on "
            f"its last axis"
        )
    levels = np.asarray(quantiles, dtype=np.float64)
    if np.any(levels <= 0.0) or np.any(levels >= 1.0):
        raise ValueError(f"quantile levels must lie in (0, 1), got {list(quantiles)}")
    if np.any(np.diff(levels) <= 0.0):
        raise ValueError(f"quantile levels must be strictly ascending, got {list(quantiles)}")
    return levels


def _validated_fan(pred_quantiles: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate a quantile fan against its targets.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.

    Returns:
        Tuple ``(fan, truth)`` as float64 arrays, in corpus units.

    Raises:
        ValueError: If ``pred_quantiles`` is not four-dimensional, if its leading three axes
            do not match ``target``, or if either is empty.
    """
    fan = np.asarray(pred_quantiles, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if fan.ndim != 4:
        raise ValueError(f"pred_quantiles must have shape (N, H, C, Q), got {fan.shape}")
    if truth.shape != fan.shape[:3]:
        raise ValueError(
            f"pred_quantiles {fan.shape} and target {truth.shape} disagree: the leading axes "
            f"of the fan must be the (N, H, C) of the targets it is scored against"
        )
    if fan.size == 0:
        raise ValueError("cannot score an empty window set")
    return fan, truth


def _validated_interval(
    lower: FloatArray, upper: FloatArray, target: FloatArray | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate an interval, and optionally the targets it is scored against.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, same shape, in corpus units.
        target: Targets, same shape, in corpus units, or None when only the interval itself
            is being reduced.

    Returns:
        Tuple ``(lower, upper, target)`` as float64 arrays; ``target`` is an empty array
        when none was supplied.

    Raises:
        ValueError: If the arrays differ in shape or are empty.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"lower has shape {lo.shape} but upper has shape {hi.shape}")
    if lo.size == 0:
        raise ValueError("cannot score an empty window set")
    truth = np.zeros((0,), dtype=np.float64) if target is None else np.asarray(target, np.float64)
    if target is not None and truth.shape != lo.shape:
        raise ValueError(f"target has shape {truth.shape} but the interval has shape {lo.shape}")
    return lo, hi, truth


def _reject_crossed(lower: FloatArray, upper: FloatArray) -> None:
    """Refuse an interval whose lower bound exceeds its upper bound.

    Args:
        lower: Lower interval bound, corpus units.
        upper: Upper interval bound, same shape, corpus units.

    Raises:
        ValueError: If any element of ``lower`` exceeds its ``upper``, which means
            ``dmf.models.heads.sort_quantiles`` was not applied to the fan the endpoints
            were read from. Coverage of a crossed interval is not a coverage.
    """
    crossed = lower > upper
    if bool(np.any(crossed)):
        worst = float(np.max(lower - upper))
        raise ValueError(
            f"{int(np.count_nonzero(crossed))} of {lower.size} interval endpoints are "
            f"crossed (worst lower - upper = {worst:.6g} corpus units); "
            f"dmf.models.heads.sort_quantiles was not applied, and the coverage of a "
            f"crossed interval is not a coverage"
        )


def pinball_terms(
    pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]
) -> FloatArray:
    """Compute the per-element, per-level pinball loss.

    The streaming form of :func:`pinball`: a runner sums these over the windows of each
    realization to a ``(n_keys, H, C, Q)`` accumulator rather than materialising the fan.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), strictly ascending, length ``Q``.

    Returns:
        Pinball loss per (window, horizon, channel, level), shape ``(N, H, C, Q)``, in
        corpus units. Non-negative.

    Raises:
        ValueError: If ``len(quantiles)`` does not match the ``Q`` axis, if the leading
            shapes disagree, if a level is outside (0, 1), or if the levels are not
            strictly ascending.
    """
    fan, truth = _validated_fan(pred_quantiles, target)
    levels = _validated_levels(quantiles, fan.shape[-1])
    error = truth[..., np.newaxis] - fan
    return np.asarray(np.maximum(levels * error, (levels - 1.0) * error), dtype=np.float64)


def crps_terms(
    pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]
) -> FloatArray:
    """Compute the per-element approximate CRPS of a quantile fan.

    ``2 * mean_q pinball_q``: the rectangle-rule quadrature of ``CRPS = 2 * int_0^1 PL_q
    dq``. With the nine equispaced levels of ``dmf.models.heads.QUANTILE_FAN_9`` (0.05 to
    0.95 in steps of 0.1125) the level bins tile (0, 1) almost exactly, but the tails beyond
    0.05 and 0.95 are still only represented by their nearest level, so this is an
    **approximation** whose bias grows with the weight the predictive distribution puts in
    its tails. Reported as "crps (approx, Q levels)", never as exact CRPS.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), strictly ascending, length ``Q``.

    Returns:
        Approximate CRPS per (window, horizon, channel), shape ``(N, H, C)``, in corpus
        units. Non-negative.

    Raises:
        ValueError: If the shapes are inconsistent or the levels are invalid.
    """
    return np.asarray(2.0 * pinball_terms(pred_quantiles, target, quantiles).mean(axis=-1))


def coverage_terms(lower: FloatArray, upper: FloatArray, target: FloatArray) -> FloatArray:
    """Compute the per-element coverage indicator of a prediction interval.

    The streaming form of :func:`picp`: a runner sums these over the windows of each
    realization into an ``n_covered`` accumulator of shape ``(n_keys, H, C)``.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, same shape, in corpus units.
        target: Targets, same shape, in corpus units.

    Returns:
        1.0 where ``lower <= target <= upper`` and 0.0 elsewhere, shape ``(N, H, C)``,
        dimensionless. The interval is **closed**: a target exactly on an endpoint counts as
        covered.

    Raises:
        ValueError: If the three arrays differ in shape, or if any ``lower`` exceeds its
            ``upper`` -- which indicates that ``dmf.models.heads.sort_quantiles`` was not
            applied.
    """
    lo, hi, truth = _validated_interval(lower, upper, target)
    _reject_crossed(lo, hi)
    return np.asarray((truth >= lo) & (truth <= hi), dtype=np.float64)


def width_terms(lower: FloatArray, upper: FloatArray) -> FloatArray:
    """Compute the per-element width of a prediction interval.

    The streaming form of :func:`mean_interval_width`.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, same shape, in corpus units.

    Returns:
        ``upper - lower`` per element, shape ``(N, H, C)``, in corpus units (degrees for
        angles, metres for heave).

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    lo, hi, _ = _validated_interval(lower, upper)
    return np.asarray(hi - lo, dtype=np.float64)


def winkler_terms(
    lower: FloatArray, upper: FloatArray, target: FloatArray, alpha: float = 0.1
) -> FloatArray:
    """Compute the per-element Winkler interval score.

    ``(u - l)`` when the target is covered, plus ``(2/alpha) * distance`` outside the
    interval otherwise. The streaming form of :func:`winkler_score`.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, same shape, in corpus units.
        target: Targets, same shape, in corpus units.
        alpha: Nominal miscoverage rate, so ``alpha = 0.1`` scores a 90 percent interval.

    Returns:
        Winkler score per (window, horizon, channel), shape ``(N, H, C)``, in corpus units.
        Lower is better.

    Raises:
        ValueError: If the shapes differ or ``alpha`` is not in (0, 1).
    """
    lo, hi, truth = _validated_interval(lower, upper, target)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    penalty = (2.0 / alpha) * (np.maximum(lo - truth, 0.0) + np.maximum(truth - hi, 0.0))
    return np.asarray((hi - lo) + penalty, dtype=np.float64)


def crossing_terms(pred_quantiles: FloatArray) -> FloatArray:
    """Flag fan elements that were non-monotone **before** post-hoc sorting.

    A diagnostic, not a metric: it says whether ``dmf.models.heads.sort_quantiles`` was
    doing anything. The accumulator built from it is a count, and the table reports the
    fraction of ``(window, horizon, channel)`` elements that had at least one adjacent
    inversion. A crossing rate of zero means the model's raw fan was already monotone and
    the sort was a no-op; a large rate means the reported intervals depend on the sort, and
    every downstream number should be read with that in mind.

    Args:
        pred_quantiles: **Raw, unsorted** quantile forecasts, shape ``(N, H, C, Q)``, in
            corpus units, with levels ascending along ``Q``. Passing an already-sorted fan
            gives a rate of exactly zero and measures nothing.

    Returns:
        1.0 where any adjacent pair along ``Q`` is strictly decreasing, 0.0 elsewhere,
        shape ``(N, H, C)``, dimensionless. Equal adjacent values are not an inversion.

    Raises:
        ValueError: If ``pred_quantiles`` is not four-dimensional or is empty.
    """
    fan = np.asarray(pred_quantiles, dtype=np.float64)
    if fan.ndim != 4:
        raise ValueError(f"pred_quantiles must have shape (N, H, C, Q), got {fan.shape}")
    if fan.size == 0:
        raise ValueError("cannot score an empty window set")
    inverted = fan[..., 1:] < fan[..., :-1]
    return np.asarray(np.any(inverted, axis=-1), dtype=np.float64)


def pinball(pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]) -> float:
    """Compute mean pinball loss over a quantile fan.

    The mean is taken over windows, horizons, channels **and levels**, so at a single median
    level it is exactly half the MAE.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units,
            sorted ascending along ``Q``.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), ascending, length ``Q``.

    Returns:
        Mean pinball loss, in corpus units.

    Raises:
        ValueError: If ``len(quantiles)`` does not match the ``Q`` axis, the leading
            shapes disagree, or a level is outside (0, 1) or not strictly ascending.
    """
    return float(pinball_terms(pred_quantiles, target, quantiles).mean())


def crps_from_quantiles(
    pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]
) -> float:
    """Approximate the continuous ranked probability score from a quantile fan.

    Computed as the quantile-weighted integral of the pinball loss, which converges to
    CRPS as the fan is refined. With ``Q = 9`` levels this is an approximation, and the
    results tables label it as such rather than reporting it as exact CRPS. See
    :func:`crps_terms` for the quadrature and its bias.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units,
            sorted ascending along ``Q``.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), ascending, length ``Q``.

    Returns:
        Approximate CRPS, in corpus units.

    Raises:
        ValueError: If the shapes are inconsistent or the levels are invalid.
    """
    return float(crps_terms(pred_quantiles, target, quantiles).mean())


def picp(lower: FloatArray, upper: FloatArray, target: FloatArray) -> float:
    """Compute prediction interval coverage probability.

    Never reported alone: :func:`mean_interval_width` is its companion, and
    :data:`PROBABILISTIC_METRIC_COLUMNS` places the two side by side.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.

    Returns:
        Fraction of targets falling within ``[lower, upper]``, dimensionless in [0, 1].
        For a nominal 90 percent interval, Gate 5 requires this in [0.85, 0.95] on the
        ``id`` regime.

    Raises:
        ValueError: If the three arrays differ in shape, or if any ``lower`` exceeds its
            ``upper`` -- which indicates that :func:`dmf.models.heads.sort_quantiles` was
            not applied.
    """
    return float(coverage_terms(lower, upper, target).mean())


def mean_interval_width(lower: FloatArray, upper: FloatArray) -> float:
    """Compute the mean width of the prediction interval.

    Reported alongside PICP, never alone. Coverage is trivially achievable by widening the
    interval until it is useless; the pair is what says whether the intervals are
    informative.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.

    Returns:
        Mean interval width, in corpus units (degrees for angles, metres for heave).

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    return float(width_terms(lower, upper).mean())


def winkler_score(
    lower: FloatArray, upper: FloatArray, target: FloatArray, alpha: float = 0.1
) -> float:
    """Compute the mean Winkler interval score.

    Penalises interval width and adds a miss penalty proportional to how far outside the
    interval the target fell, so it scores coverage and sharpness in a single number
    rather than leaving the trade-off to the reader.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        alpha: Nominal miscoverage rate, so ``alpha = 0.1`` scores a 90 percent interval.

    Returns:
        Mean Winkler score, in corpus units. Lower is better.

    Raises:
        ValueError: If the shapes differ or ``alpha`` is not in (0, 1).
    """
    return float(winkler_terms(lower, upper, target, alpha).mean())


def _reduced_over_keys(
    name: str, values: FloatArray, reference_shape: tuple[int, ...]
) -> FloatArray:
    """Sum a per-realization accumulator over its realization axis.

    Args:
        name: Argument name, for the error message.
        values: Accumulator, shape ``(n_keys, H, C)``.
        reference_shape: The ``(n_keys, H, C)`` every accumulator in the call must share.

    Returns:
        The accumulator summed over realizations, shape ``(H, C)``.

    Raises:
        ValueError: If ``values`` does not have ``reference_shape``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != reference_shape:
        raise ValueError(
            f"{name} has shape {array.shape} but the accumulators must all be "
            f"{reference_shape} = (n_keys, H, C); a probabilistic score summed over a "
            f"different window set is not the score of these windows"
        )
    return np.asarray(array.sum(axis=0), dtype=np.float64)


def _reject_negative(name: str, values: FloatArray, note: str) -> None:
    """Refuse a reported quantity that cannot be negative by construction.

    Args:
        name: Column name, for the error message.
        values: Reported grid, shape ``(n_horizons, C)``.
        note: What a negative value means, appended to the message.

    Raises:
        ValueError: If any element is negative or non-finite.
    """
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} is negative or non-finite at a reported cell: {note}")


def probabilistic_table_from_sums(
    n_covered: FloatArray,
    width_sum: FloatArray,
    winkler_sum: FloatArray,
    crps_sum: FloatArray,
    pinball_sum: FloatArray,
    crossing_count: FloatArray,
    n: int,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Build the probabilistic results table from streamed per-realization sums.

    The production form of the scalar scores in this module, mirroring
    :func:`dmf.eval.metrics.metrics_table_from_sums`. Nothing here is ever ``(N, H, C, Q)``,
    so a test partition costs ``n_keys * H * C * (5 + Q)`` float64 values per (model, head)
    rather than one entry per window.

    The realization axis is kept rather than pre-summed so that the caller can hand the same
    arrays to the realization-level bootstrap (:func:`dmf.eval.runner.bootstrap_skill_ci`);
    it is reduced here exactly the way ``metrics_table_from_sums`` reduces ``sse``/``sae``,
    by summing over realizations and dividing by the window count ``n``.

    Args:
        n_covered: Summed coverage indicator per realization, shape ``(n_keys, H, C)``,
            dimensionless counts from :func:`coverage_terms`.
        width_sum: Summed interval width per realization, same shape, in corpus units
            (:func:`width_terms`).
        winkler_sum: Summed Winkler score per realization, same shape, in corpus units
            (:func:`winkler_terms`), accumulated at the **same** ``alpha`` that is passed
            here -- the argument labels the column, it does not rescale the sum.
        crps_sum: Summed approximate CRPS per realization, same shape, in corpus units
            (:func:`crps_terms`). It is whatever the caller accumulated: the finite-fan
            quadrature for a quantile head, or a closed form for a Gaussian head. For a
            quantile fan it equals ``2 * mean_q`` of ``pinball_sum`` by construction.
        pinball_sum: Summed pinball loss per realization **and level**, shape
            ``(n_keys, H, C, Q)``, in corpus units (:func:`pinball_terms`). Kept per level so
            that the per-level breakdown is available without a second pass; the reported
            ``pinball`` column averages over levels.
        crossing_count: Number of windows whose **raw** fan had at least one adjacent
            inversion, shape ``(n_keys, H, C)``, dimensionless counts from
            :func:`crossing_terms`.
        n: Total windows the sums were accumulated over, summed across realizations. As in
            :mod:`dmf.eval.metrics`, consecutive windows overlap, so this is a window count
            and **not** an independent-sample count.
        dof_names: Target channel names, length ``C``.
        horizons: Horizons to report, samples, each in ``[1, H]``. "Horizon h" is the score
            at lead time exactly ``h``, never a cumulative average over 1..h.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds alongside
            samples.
        alpha: Nominal miscoverage rate of the interval the coverage and Winkler sums were
            accumulated for. 0.1 is PICP@90, the level Gate 5 is read at.

    Returns:
        One row per (DOF, horizon), channel-major and horizon-minor as in
        :func:`dmf.eval.metrics.metrics_table_from_sums`, columns
        :data:`PROBABILISTIC_METRIC_COLUMNS`. ``mean_interval_width``, ``winkler``, ``crps``
        and ``pinball`` are in corpus units (degrees for roll and pitch, metres for heave);
        ``picp``, ``crossing_rate`` and ``alpha`` are dimensionless. ``crps`` is the
        finite-fan approximation of the module docstring, not exact CRPS.

    Raises:
        ValueError: If the five ``(n_keys, H, C)`` accumulators differ in shape, if
            ``pinball_sum`` does not extend that shape with a level axis, if ``dof_names``
            does not match the channel axis, if ``n`` is not positive, if ``fs_hz`` is not
            positive, if ``alpha`` is not in (0, 1), if a requested horizon is outside
            ``[1, H]``, if a coverage or crossing fraction falls outside [0, 1], or if a
            reported width, Winkler, CRPS or pinball value is negative -- a negative mean
            width means the interval endpoints were crossed, i.e.
            ``dmf.models.heads.sort_quantiles`` was not applied.
    """
    covered = np.asarray(n_covered, dtype=np.float64)
    if covered.ndim != 3:
        raise ValueError(f"n_covered must have shape (n_keys, H, C), got {covered.shape}")
    shape = (covered.shape[0], covered.shape[1], covered.shape[2])
    fan_sum = np.asarray(pinball_sum, dtype=np.float64)
    if fan_sum.ndim != 4 or fan_sum.shape[:3] != shape:
        raise ValueError(
            f"pinball_sum must have shape (n_keys, H, C, Q) extending {shape}, got {fan_sum.shape}"
        )
    if covered.shape[2] != len(dof_names):
        raise ValueError(
            f"sums cover {covered.shape[2]} channels but {len(dof_names)} DOF names were "
            f"given: {list(dof_names)}"
        )
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    total_covered = covered.sum(axis=0)
    total_width = _reduced_over_keys("width_sum", width_sum, shape)
    total_winkler = _reduced_over_keys("winkler_sum", winkler_sum, shape)
    total_crps = _reduced_over_keys("crps_sum", crps_sum, shape)
    total_crossing = _reduced_over_keys("crossing_count", crossing_count, shape)
    n_levels = int(fan_sum.shape[3])
    total_pinball = fan_sum.sum(axis=0)

    index = _horizon_index(horizons, shape[1])
    # Validated over the reported cells only, following `metrics_table_from_sums`: a bad
    # accumulator at a horizon nobody asked for must not fail a table that never quotes it.
    picp_grid = total_covered[index, :] / n
    width_grid = total_width[index, :] / n
    winkler_grid = total_winkler[index, :] / n
    crps_grid = total_crps[index, :] / n
    pinball_grid = total_pinball[index, :, :].sum(axis=-1) / (n * n_levels)
    crossing_grid = total_crossing[index, :] / n
    if not np.all(np.isfinite(picp_grid)) or np.any(picp_grid < 0.0) or np.any(picp_grid > 1.0):
        raise ValueError(
            f"picp falls outside [0, 1] at a reported cell (min {np.nanmin(picp_grid):.6g}, "
            f"max {np.nanmax(picp_grid):.6g}); n_covered counts windows and cannot exceed "
            f"n = {n}, so the accumulator and the window count came from different passes"
        )
    if (
        not np.all(np.isfinite(crossing_grid))
        or np.any(crossing_grid < 0.0)
        or np.any(crossing_grid > 1.0)
    ):
        raise ValueError(
            f"crossing_rate falls outside [0, 1] at a reported cell (min "
            f"{np.nanmin(crossing_grid):.6g}, max {np.nanmax(crossing_grid):.6g}); it counts "
            f"windows with at least one inversion and cannot exceed n = {n}"
        )
    _reject_negative(
        "mean_interval_width",
        width_grid,
        "the interval endpoints are crossed on average, which means "
        "dmf.models.heads.sort_quantiles was not applied to the fan they were read from",
    )
    _reject_negative("winkler", winkler_grid, "the Winkler score is a width plus a penalty")
    _reject_negative("crps", crps_grid, "the pinball loss it integrates is non-negative")
    _reject_negative("pinball", pinball_grid, "the pinball loss is non-negative")

    rows: list[dict[str, object]] = []
    for channel, name in enumerate(dof_names):
        for position, horizon in enumerate(horizons):
            rows.append(
                {
                    "dof": name,
                    "horizon_samples": int(horizon),
                    "horizon_s": float(horizon) / fs_hz,
                    "n_windows": int(n),
                    "n_quantiles": n_levels,
                    "alpha": float(alpha),
                    "picp": float(picp_grid[position, channel]),
                    "mean_interval_width": float(width_grid[position, channel]),
                    "winkler": float(winkler_grid[position, channel]),
                    "crps": float(crps_grid[position, channel]),
                    "pinball": float(pinball_grid[position, channel]),
                    "crossing_rate": float(crossing_grid[position, channel]),
                }
            )
    return pd.DataFrame(rows, columns=list(PROBABILISTIC_METRIC_COLUMNS))
