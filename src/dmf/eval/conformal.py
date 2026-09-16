"""Split-conformal calibration: the conformity score, the order statistic, and the pass.

The mathematics and the streaming calibration pass, with no I/O policy and no decision about
where anything is written -- that belongs to :mod:`dmf.eval.conformal_runner`, the same way
:mod:`dmf.eval.probabilistic` holds the metric definitions and :mod:`dmf.eval.prob_runner`
holds the pass that accumulates them.

**The score.** Width-normalised CQR (Romano, Patterson & Candes 2019 for the construction;
Sesia & Candes 2020 for the ratio form). Per ``(horizon, target channel)``, with ``m`` the
point forecast and ``(lo, hi)`` the ``alpha/2`` and ``1 - alpha/2`` members:

    ``s_i = max( (m_i - y_i) / (m_i - lo_i),  (y_i - m_i) / (hi_i - m_i) )``

and ``gamma`` is the ``ceil((n+1)(1-alpha))``-th order statistic of the ascending scores. A
target is covered by the ``gamma``-scaled interval exactly when ``s_i <= gamma``, so this is
split conformal with score ``s``, and the marginal guarantee is

    ``1 - alpha <= P(Y in C(X)) <= 1 - alpha + 1/(n+1)``

under exchangeability of the calibration and test scores.

**The order statistic is the guarantee, and interpolation is not.**
:func:`dmf.train.closed_form.fit_residual_interval` uses ``np.quantile``'s default linear
interpolation and records that the rule "is not load-bearing" at 25 000 windows. That is true
for a floor, whose claim is "here is a reference width", and **false** here, whose claim is a
coverage probability: the finite-sample correction *is* the ``k``-th order statistic and
nothing else. :func:`order_statistic_index` is therefore the only place the level is turned
into an index, and it is tested directly rather than through a calibrated model.

**Exchangeability holds at the realization level, not the window level, and this module says
so.** Consecutive windows are 0.5 s apart on a signal whose roll period is ~12 s
(``dmf.train.closed_form`` records the same fact for the residual floor), so 25 000 windows
are far fewer than 25 000 independent samples and the ``1/(n+1)`` term advertises a precision
the dependence structure does not support. Two responses, and this module implements both:
the window-level ``gamma`` is what the scored rows use, because it is the definition; and a
**realization-level** sensitivity is computed beside it, certifying 90 percent of
realizations' *median* windows rather than 90 percent of windows. That is deliberately a
different and weaker target -- see :func:`_realization_level_gamma` -- and the gap between the
two says how much of the scale is set by the within-realization tail rather than by variation
across records. Measured on the committed run at ``id``/``dlinear_quantile`` the medians are
**0.971** against **0.635**, and across all four regimes ``gamma`` runs 0.77-0.82 against a
realization-level 0.43-0.45 -- roughly half. The gap is large and the calibration table carries
both columns so a reader can see it rather than take this sentence for it. The binding
uncertainty on any reported coverage remains the realization bootstrap interval
:mod:`dmf.eval.prob_runner` already computes.

**What the calibration split contributes, and what it must not.** Only the conformity scores.
Normalisation statistics stay train-fitted and are passed in
(``CLAUDE.md`` non-negotiable 3); the partition is checked structurally rather than
documented, exactly as :func:`dmf.train.closed_form.fit_residual_interval` checks it, because
a calibration fitted on train would be in-sample and one fitted on test is the leak the split
policy exists to prevent.

Units: scores and ``gamma`` are dimensionless. Calibration runs entirely in the model's
normalised space, so ``gamma`` is invariant to the affine map into corpus units.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import normalize_target
from dmf.models.base import BaseForecaster
from dmf.models.conformal import ConformalInterval
from dmf.models.heads import PredictiveDistribution
from dmf.typedefs import FloatArray, IntArray

__all__ = [
    "CONFORMAL_MAX_WINDOWS",
    "ConformalFit",
    "calibrate_models",
    "conformity_scores",
    "order_statistic_index",
]

#: Calibration windows drawn, at a fixed stride over the whole partition. Deliberately the
#: same constant and the same rationale as
#: :data:`dmf.train.closed_form.RESIDUAL_QUANTILE_MAX_WINDOWS`, so that the project's two
#: held-out calibrations draw the same windows: consecutive windows are 0.5 s apart on a ~12 s
#: roll period, so the effective sample size is a small fraction of the nominal one either
#: way, and 25 000 windows still puts ~1250 scores in the 10 percent tail of every
#: ``(horizon, channel)`` cell.
CONFORMAL_MAX_WINDOWS: int = 25_000

#: Smallest calibration half-width, in normalised units, that a conformity score may be
#: divided by. The score is a ratio to the base head's own half-width; where that half-width
#: is ~0 the ratio is unbounded and one degenerate window would set ``gamma`` for the whole
#: cell. Cells hitting this floor are **counted and reported**, never silently dropped.
MIN_HALF_WIDTH: float = 1e-9


@dataclass(frozen=True)
class ConformalFit:
    """One fitted calibration for one model at one regime and seed.

    Attributes:
        gamma: Conformal scale, ``(H, C_out)``, dimensionless, from the window-level scores.
        gamma_realization: The same quantity from one score per calibration realization (its
            median window score), ``(H, C_out)``. A sensitivity on the effective sample
            size, never the scored value.
        n_windows_total: Windows in the calibration partition before sub-sampling.
        n_windows_used: Windows actually scored, ``n``.
        n_realizations: Calibration realizations contributing at least one scored window.
        window_stride: Sub-sampling stride over the global window index.
        order_index: The ``k`` of the ``k``-th order statistic, ``ceil((n+1)(1-alpha))``.
        alpha: Nominal miscoverage the scale was fitted at.
        n_degenerate_cells: ``(horizon, channel)`` cells whose base half-width hit
            :data:`MIN_HALF_WIDTH` on at least one window.
        fitted_on: ``"<regime>/<partition>"`` of the calibration data.
        norm_stats_fitted_on: Provenance label of the normalisation statistics used.
        calibration_time_s: Wall clock of the calibration pass, seconds.
    """

    gamma: FloatArray
    gamma_realization: FloatArray
    n_windows_total: int
    n_windows_used: int
    n_realizations: int
    window_stride: int
    order_index: int
    alpha: float
    n_degenerate_cells: int
    fitted_on: str
    norm_stats_fitted_on: str
    calibration_time_s: float


def order_statistic_index(n: int, alpha: float) -> int:
    """Return ``k`` for the ``k``-th ascending order statistic of the conformity scores.

    ``k = ceil((n + 1) * (1 - alpha))``. This is the whole finite-sample correction, and it
    is an index into sorted scores -- **not** an interpolated quantile. At ``n = 25_000`` and
    ``alpha = 0.1`` it is 22 501, i.e. the 0.90004 empirical quantile.

    Args:
        n: Calibration scores available, ``n >= 1``.
        alpha: Nominal miscoverage, in ``(0, 1)``.

    Returns:
        The 1-based index ``k``, in ``[1, n]``.

    Raises:
        ValueError: If ``n < 1``, if ``alpha`` is outside ``(0, 1)``, or if ``k > n``. The
            last case means the calibration set is too small to certify the requested level
            at all -- the conformal interval would be unbounded -- and it is refused rather
            than silently clipped to the maximum score. At ``alpha = 0.1`` it needs ``n < 9``.
    """
    if n < 1:
        raise ValueError(f"need at least one calibration score, got n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        raise ValueError(
            f"a {1 - alpha:.3g} conformal level needs ceil((n+1)(1-alpha)) = {k} <= n, but "
            f"n = {n}. With this few calibration scores the conformal interval is unbounded; "
            f"clipping to the largest observed score would report a finite interval with no "
            f"guarantee behind it."
        )
    return k


def conformity_scores(
    distribution: PredictiveDistribution,
    target: Tensor,
    *,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Score a batch against its own predictive interval, width-normalised.

    The score is the number of the base head's own interval half-widths by which the target
    misses the point forecast, signed so that a target inside the interval scores ``<= 1``:

        ``s = max( (m - y) / (m - lo),  (y - m) / (hi - m) )``

    so scaling the interval by ``gamma`` covers exactly the targets scoring ``<= gamma``.

    Args:
        distribution: The base head's predictive distribution for the batch, dimensionless.
        target: Targets, ``(B, H, C_out)``, in the same normalised space.
        alpha: Nominal miscoverage; the interval read is the central ``1 - alpha``.

    Returns:
        ``(scores, degenerate)``. ``scores`` is ``(B, H, C_out)``, dimensionless and
        non-negative-by-construction only where the interval is non-degenerate; ``degenerate``
        is a boolean mask of the same shape marking elements whose half-width fell to
        :data:`MIN_HALF_WIDTH`.

    Raises:
        ValueError: If ``target``'s shape does not match the distribution's.
    """
    lower, upper = distribution.interval(alpha)
    median = distribution.point()
    if tuple(target.shape) != tuple(median.shape):
        raise ValueError(f"target has shape {tuple(target.shape)}, expected {tuple(median.shape)}")
    lo_half = (median - lower).clamp_min(MIN_HALF_WIDTH)
    hi_half = (upper - median).clamp_min(MIN_HALF_WIDTH)
    degenerate = ((median - lower) < MIN_HALF_WIDTH) | ((upper - median) < MIN_HALF_WIDTH)
    scores = torch.maximum((median - target) / lo_half, (target - median) / hi_half)
    return scores, degenerate


def calibrate_models(
    models: Mapping[str, BaseForecaster],
    calibration: DeckMotionDataset,
    *,
    alpha: float = 0.1,
    max_windows: int = CONFORMAL_MAX_WINDOWS,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> dict[str, tuple[ConformalInterval, ConformalFit]]:
    """Calibrate every model in one streaming pass over the calibration split.

    One pass, all models: they are scored on identical windows, which is what makes the
    calibrated rows comparable to each other for the same reason
    :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` scores them together.

    Args:
        models: Fitted interval heads, keyed by results-table label. Not modified.
        calibration: The **validation** partition, built with the training split's
            normalisation statistics.
        alpha: Nominal miscoverage. ``0.1`` is the 90 percent interval Gate 5 reads.
        max_windows: Cap on scored windows, applied as a fixed stride over the global window
            index.
        batch_size: Windows per batch. Speed and memory only.
        num_workers: DataLoader worker processes.
        device: Torch device the models run on. Scores are reduced on the CPU in float64.

    Returns:
        ``{label: (calibrated wrapper, fit report)}``. The wrappers are fresh
        :class:`dmf.models.conformal.ConformalInterval` objects holding the passed-in models;
        the inputs are untouched.

    Raises:
        ValueError: If ``calibration`` is not the validation partition, if its normalisation
            statistics were not fitted on a training split, if ``models`` is empty, or if any
            model carries a point head.
    """
    if not models:
        raise ValueError("no models to calibrate")
    if calibration.partition != "val":
        raise ValueError(
            f"conformal scores must be computed on the validation partition, got "
            f"{calibration.regime}/{calibration.partition}. Train scores are the base head's "
            f"own fitting residuals and understate its error by the amount it overfits, so "
            f"the calibrated interval would be too tight in the direction that flatters "
            f"coverage; test scores are the leak itself (CLAUDE.md non-negotiable 2)."
        )
    stats = calibration.norm_stats.subset(calibration.target_columns)
    if not stats.fitted_on.endswith("/train"):
        raise ValueError(
            f"calibration must use training-split normalisation statistics, got "
            f"fitted_on={stats.fitted_on!r} (CLAUDE.md non-negotiable 3)"
        )
    for label, model in models.items():
        if model.head_kind == "point":
            raise ValueError(f"{label} carries a point head and has no interval to calibrate")

    started = time.perf_counter()
    n_total = len(calibration)
    stride = max(1, -(-n_total // max_windows))
    loader = make_dataloader(
        calibration, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=0
    )
    wrapped = {label: ConformalInterval(model).to(device).eval() for label, model in models.items()}

    chunks: dict[str, list[FloatArray]] = {label: [] for label in models}
    degenerate: dict[str, int] = dict.fromkeys(models, 0)
    realization_of: list[IntArray] = []
    position = 0
    per_realization = calibration.windows_per_realization
    with torch.no_grad():
        for x, y, window_mean in loader:
            n = int(x.shape[0])
            # The loader is unshuffled, so `position + i` is the global window index; this
            # offset makes the selection an exact every-`stride`-th window over the whole
            # partition rather than a per-batch approximation of one. Lifted verbatim from
            # dmf.train.closed_form.fit_residual_interval so the two passes agree.
            offset = (-position) % stride
            global_index = np.arange(position, position + n)
            position += n
            if offset >= n:
                continue
            x_sub = x[offset::stride].to(device)
            y_sub = normalize_target(y[offset::stride], stats, window_mean[offset::stride])
            y_sub = y_sub.to(device)
            realization_of.append(global_index[offset::stride] // max(1, per_realization))
            for label, model in wrapped.items():
                raw = model.base(x_sub)
                dist = PredictiveDistribution(raw, model.head_kind, model.base.quantile_levels)
                scores, degen = conformity_scores(dist, y_sub, alpha=alpha)
                chunks[label].append(scores.double().cpu().numpy())
                degenerate[label] += int(degen.any(dim=0).sum().item())

    realizations = np.concatenate(realization_of, axis=0)
    n_used = int(realizations.size)
    order_index = order_statistic_index(n_used, alpha)

    out: dict[str, tuple[ConformalInterval, ConformalFit]] = {}
    for label, model in wrapped.items():
        stacked: FloatArray = np.concatenate(chunks[label], axis=0)
        # np.partition, not np.sort: the k-th order statistic is all that is wanted and the
        # arrays are (25 000, H, C).
        gamma: FloatArray = np.partition(stacked, order_index - 1, axis=0)[order_index - 1]
        gamma_real = _realization_level_gamma(stacked, realizations, alpha)
        model.set_gamma(gamma)
        out[label] = (
            model,
            ConformalFit(
                gamma=gamma,
                gamma_realization=gamma_real,
                n_windows_total=n_total,
                n_windows_used=n_used,
                n_realizations=int(np.unique(realizations).size),
                window_stride=stride,
                order_index=order_index,
                alpha=alpha,
                n_degenerate_cells=degenerate[label],
                fitted_on=f"{calibration.regime}/{calibration.partition}",
                norm_stats_fitted_on=stats.fitted_on,
                calibration_time_s=time.perf_counter() - started,
            ),
        )
    return out


def _realization_level_gamma(
    scores: FloatArray, realizations: IntArray, alpha: float
) -> FloatArray:
    """Return a realization-level sensitivity on ``gamma``.

    One score per calibration realization -- the median of its windows' scores -- then the
    same order statistic across realizations.

    **This is a different target, not the same target at a coarser granularity, and the
    difference is the point.** The window-level ``gamma`` certifies 90 percent of *windows*;
    this one certifies 90 percent of *realizations' median windows*, which is a weaker and
    strictly different statement. It is computed because windows inside a realization overlap
    by 345 of 350 samples, so the window-level ``n`` is not an independent sample count and
    the ``1/(n+1)`` term attached to it is optimistic. The gap between the two therefore
    measures **how much of the window-level scale is set by the within-realization tail rather
    than by variation across realizations** -- a large gap means the calibration is dominated
    by a few hard windows inside otherwise ordinary records.

    It is never the scored value, and no coverage claim in this project rests on it.

    Args:
        scores: Window scores, ``(N, H, C)``, dimensionless.
        realizations: Realization index per window, ``(N,)``.
        alpha: Nominal miscoverage.

    Returns:
        ``(H, C)`` scale factors, dimensionless.
    """
    unique = np.unique(realizations)
    medians = np.stack([np.median(scores[realizations == r], axis=0) for r in unique], axis=0)
    k = order_statistic_index(int(unique.size), alpha)
    picked: FloatArray = np.partition(medians, k - 1, axis=0)[k - 1]
    return picked
