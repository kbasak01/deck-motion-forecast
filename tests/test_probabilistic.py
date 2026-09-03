"""Probabilistic scoring definitions -- Phase 5, and the Gate 5 read-out path.

Every test here is a *definitional anchor*: a point where the metric's value is known
independently of the implementation, so that a rewrite that changes the number fails rather
than silently redefining what the results tables mean.

Covered here:

- **Coverage.** PICP of an oracle Gaussian interval recovers its nominal level to within
  binomial sampling error; a crossed interval raises rather than reporting a coverage,
  because ``lower > upper`` means ``dmf.models.heads.sort_quantiles`` was not applied;
  widening an interval never lowers PICP, which is the monotonicity that makes coverage
  trivially gameable and is why ``docs/protocol.md`` P5-D2 forbids reporting it without a
  width beside it.
- **Pinball.** At a single median level it is *exactly* half the MAE -- bitwise, not
  ``approx`` -- which pins both the loss definition and the fact that the mean runs over the
  level axis as well as over windows, horizons and channels.
- **CRPS.** The approximation is ``2 * mean_q pinball_q``, so a degenerate fan (every level
  predicting the same value) scores exactly the MAE. That anchor is what fixes the
  quadrature: a trapezoidal rule over the nine levels would score 0.9 * MAE there, and the
  column would then be a different quantity from the one the plan asks for. It remains an
  approximation of CRPS from ``Q`` levels and the table labels it as such.
- **Winkler.** Equals the mean interval width when every target is covered, and its miss
  penalty grows linearly in the miss distance with the ``2/alpha`` factor pinned at three
  values of ``alpha`` rather than only at the 0.1 the gate is read at.
- **The streaming path.** ``probabilistic_table_from_sums`` matches the array path
  elementwise on every column, mirroring
  ``tests/test_metrics.py::test_metrics_table_from_sums_matches_the_array_path``. The array
  form is the oracle only: a production ``(N, H, C, Q)`` fan is ~2.8 TB, which is why the
  runner streams per-realization sums.
- **The horizon convention**, mirroring
  ``tests/test_metrics.py::test_horizon_metric_is_per_step_not_cumulative``: "horizon h" is
  the score at lead time exactly ``h``, never the mean over 1..h.
- **The degenerate best case.** A perfectly sharp, perfectly correct forecast scores zero
  width, PICP 1.0, and zero pinball and CRPS -- and a Winkler score of zero, since a
  zero-width interval that always covers has no penalty to pay.
- **Crossing rate**, the diagnostic that says whether post-hoc sorting was doing anything:
  counted on the **raw** fan, so a fan that was sorted before it was measured reports zero
  and measures nothing.

The ``IntervalPredictor`` seam of P5-D1 is exercised in
``tests/test_models.py::test_the_conformal_seam_moves_coverage_without_importing_a_model``,
which is where it belongs: the property it proves is that a calibrating wrapper needs no
model, and asserting that is a statement about the model layer's boundaries.

Sections S1-S7 below deliberately do not touch the model layer, and restate the nine
quantile levels as a local literal so a metric definition never depends on a model
definition. S8 does import it, because a runner test that did not would be testing a mock;
``test_the_scoring_fan_equals_the_model_layers_definition`` is the assertion that keeps the
restated literal and ``dmf.models.heads.QUANTILE_FAN_9`` from drifting apart.

Units: degrees for roll and pitch, metres for heave, samples for horizons, dimensionless for
quantile levels, coverages and rates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm
from torch import Tensor

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import build_split
from dmf.data.windows import window_spec_from_config
from dmf.eval.metrics import mae
from dmf.eval.prob_runner import SCORING_LEVELS, evaluate_probabilistic_models
from dmf.eval.probabilistic import (
    PROBABILISTIC_METRIC_COLUMNS,
    coverage_terms,
    crossing_terms,
    crps_from_quantiles,
    crps_terms,
    mean_interval_width,
    picp,
    pinball,
    pinball_terms,
    probabilistic_table_from_sums,
    width_terms,
    winkler_score,
    winkler_terms,
)
from dmf.eval.runner import evaluate_models
from dmf.models.base import BaseForecaster
from dmf.models.heads import QUANTILE_FAN_9, point_view
from dmf.models.persistence import Persistence
from dmf.typedefs import FloatArray

#: Stored sampling rate, hertz. Mirrors ``tests/conftest.py``.
FS_HZ = 10.0

#: The nine quantile levels of ``docs/protocol.md`` P5-D1: 0.05 to 0.95 in steps of 0.1125,
#: with the median exactly at 0.5 and the outermost pair exactly the 90 percent interval that
#: PICP@90 scores. Spelled out here rather than imported from ``dmf.models.heads`` so that
#: this module -- which the eval side owns -- does not depend on the model layer to state
#: what a quantile level is.
FAN_9: tuple[float, ...] = (0.05, 0.1625, 0.275, 0.3875, 0.5, 0.6125, 0.725, 0.8375, 0.95)

#: Target channel names used throughout, in corpus order.
DOFS: tuple[str, ...] = ("roll", "pitch", "heave")


def _gaussian_fan(centre: FloatArray, sigma: FloatArray, levels: tuple[float, ...]) -> FloatArray:
    """Build an exact Gaussian quantile fan.

    Args:
        centre: Predictive mean, shape ``(N, H, C)``, corpus units.
        sigma: Predictive standard deviation, same shape, corpus units, positive.
        levels: Quantile levels, ascending, length ``Q``.

    Returns:
        Fan of shape ``(N, H, C, Q)``, corpus units, monotone along ``Q`` by construction.
    """
    z = np.asarray(norm.ppf(levels), dtype=np.float64)
    return np.asarray(centre[..., None] + sigma[..., None] * z, dtype=np.float64)


# ---------------------------------------------------------------------------
# Coverage: PICP, interval width, and the crossing refusal
# ---------------------------------------------------------------------------


def test_picp_of_an_oracle_gaussian_interval_recovers_its_nominal_level(
    rng: np.random.Generator,
) -> None:
    """An interval that is right by construction must measure as right.

    The targets are standard normal in degrees and the interval is the exact central 90
    percent one, so PICP has expectation 0.90 and binomial standard error
    ``sqrt(0.9*0.1/N)``. Tolerance is four standard errors: tight enough to catch an
    off-by-one in the level indexing, loose enough not to be a flaky test.
    """
    n_samples = 40_000
    target = rng.normal(size=(n_samples, 1, 1))
    half_width = float(norm.ppf(0.95))
    lower = np.full_like(target, -half_width)
    upper = np.full_like(target, half_width)
    standard_error = np.sqrt(0.9 * 0.1 / n_samples)
    assert picp(lower, upper, target) == pytest.approx(0.90, abs=4.0 * standard_error)
    # Never quoted alone: the width is what says the coverage was not bought by widening.
    assert mean_interval_width(lower, upper) == pytest.approx(2.0 * half_width)


def test_picp_raises_on_a_crossed_interval() -> None:
    """``lower > upper`` means the fan was never sorted; that is not a coverage."""
    lower = np.asarray([[[0.0]], [[2.0]]])
    upper = np.asarray([[[1.0]], [[1.0]]])
    target = np.zeros((2, 1, 1))
    with pytest.raises(ValueError, match="sort_quantiles"):
        picp(lower, upper, target)


def test_widening_an_interval_never_lowers_picp(rng: np.random.Generator) -> None:
    """Coverage is monotone in width, which is exactly why it is never reported alone."""
    target = rng.normal(size=(2_000, 3, 2))
    centre = target + rng.normal(scale=0.5, size=target.shape)
    coverages: list[float] = []
    widths: list[float] = []
    for half_width in (0.1, 0.5, 1.0, 2.0, 8.0):
        lower = centre - half_width
        upper = centre + half_width
        coverages.append(picp(lower, upper, target))
        widths.append(mean_interval_width(lower, upper))
    assert coverages == sorted(coverages)
    assert widths == sorted(widths)
    assert coverages[-1] == 1.0


def test_picp_counts_a_target_exactly_on_an_endpoint_as_covered() -> None:
    """The interval is closed. Stated as a test because the convention is arbitrary."""
    lower = np.zeros((2, 1, 1))
    upper = np.ones((2, 1, 1))
    on_endpoints = np.asarray([[[0.0]], [[1.0]]])
    assert picp(lower, upper, on_endpoints) == 1.0


def test_picp_raises_on_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        picp(np.zeros((4, 2, 1)), np.ones((4, 2, 1)), np.zeros((4, 3, 1)))


def test_mean_interval_width_raises_on_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        mean_interval_width(np.zeros((4, 2, 1)), np.ones((4, 3, 1)))


# ---------------------------------------------------------------------------
# Pinball loss and the CRPS approximation
# ---------------------------------------------------------------------------


def test_pinball_at_the_median_is_exactly_half_the_mae(rng: np.random.Generator) -> None:
    """Not ``approx``: 0.5 * |e| is exact in binary floating point.

    This pins two things at once -- the loss definition, and that the mean runs over the
    level axis as well as the window, horizon and channel axes. A pinball that summed over
    levels instead of averaging would pass at ``Q = 1`` and be nine times too large on the
    real fan, so :func:`test_pinball_scales_with_neither_q_nor_the_window_count` pins the
    other half.
    """
    pred = rng.normal(size=(32, 4, 2))
    target = rng.normal(size=(32, 4, 2))
    assert pinball(pred[..., None], target, (0.5,)) == 0.5 * float(mae(pred, target))


def test_pinball_is_asymmetric_about_its_level() -> None:
    """A worked example, computed by hand.

    At level 0.9 an under-prediction (target above the forecast) costs ``0.9 * 2 = 1.8``
    and an over-prediction of the same size costs ``0.1 * 2 = 0.2``.
    """
    forecast = np.zeros((2, 1, 1, 1))
    target = np.asarray([[[2.0]], [[-2.0]]])
    terms = pinball_terms(forecast, target, (0.9,))
    assert terms.ravel().tolist() == pytest.approx([1.8, 0.2])
    assert pinball(forecast, target, (0.9,)) == pytest.approx(1.0)


def test_pinball_scales_with_neither_q_nor_the_window_count(rng: np.random.Generator) -> None:
    """Duplicating the levels or the windows leaves a mean loss unchanged."""
    target = rng.normal(size=(16, 2, 1))
    fan = _gaussian_fan(target + 0.3, np.full(target.shape, 0.8), FAN_9)
    once = pinball(fan, target, FAN_9)
    twice = pinball(np.concatenate([fan, fan]), np.concatenate([target, target]), FAN_9)
    assert twice == pytest.approx(once)


def test_crps_of_a_degenerate_fan_equals_the_mae(rng: np.random.Generator) -> None:
    """Every level predicting the same value is a point forecast, and CRPS is then MAE.

    This is the anchor that fixes the quadrature to ``2 * mean_q pinball_q``. It holds
    because :data:`FAN_9` is symmetric about 0.5, so the mean level is exactly 0.5.
    """
    point = rng.normal(size=(64, 3, 2))
    target = rng.normal(size=(64, 3, 2))
    degenerate = np.repeat(point[..., None], len(FAN_9), axis=-1)
    assert crps_from_quantiles(degenerate, target, FAN_9) == pytest.approx(
        float(mae(point, target))
    )


def test_crps_is_twice_the_mean_pinball(rng: np.random.Generator) -> None:
    """The definitional relation, checked on a non-degenerate fan."""
    target = rng.normal(size=(64, 3, 2))
    fan = _gaussian_fan(
        target + rng.normal(scale=0.4, size=target.shape), np.full(target.shape, 1.1), FAN_9
    )
    assert crps_from_quantiles(fan, target, FAN_9) == pytest.approx(
        2.0 * pinball(fan, target, FAN_9)
    )


def test_a_sharper_correct_fan_scores_a_lower_crps(rng: np.random.Generator) -> None:
    """CRPS rewards sharpness once the fan is centred on the truth."""
    target = rng.normal(size=(2_000, 2, 1))
    scores = [
        crps_from_quantiles(_gaussian_fan(target, np.full(target.shape, s), FAN_9), target, FAN_9)
        for s in (0.1, 0.5, 2.0)
    ]
    assert scores == sorted(scores)


def test_pinball_rejects_a_level_count_that_does_not_match_the_fan() -> None:
    with pytest.raises(ValueError, match="length 2 but the fan has 9 levels"):
        pinball(np.zeros((4, 2, 1, 9)), np.zeros((4, 2, 1)), (0.1, 0.9))


def test_pinball_rejects_levels_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
        pinball(np.zeros((4, 2, 1, 2)), np.zeros((4, 2, 1)), (0.0, 0.9))


def test_pinball_rejects_non_ascending_levels() -> None:
    """Refused rather than sorted: reordering the labels mislabels every column."""
    with pytest.raises(ValueError, match="strictly ascending"):
        pinball(np.zeros((4, 2, 1, 3)), np.zeros((4, 2, 1)), (0.1, 0.9, 0.5))


def test_pinball_rejects_a_target_that_is_not_the_fan_it_is_scored_against() -> None:
    with pytest.raises(ValueError, match="leading axes"):
        pinball(np.zeros((4, 2, 1, 9)), np.zeros((3, 2, 1)), FAN_9)


# ---------------------------------------------------------------------------
# Winkler interval score
# ---------------------------------------------------------------------------


def test_winkler_equals_the_mean_width_when_every_target_is_covered(
    rng: np.random.Generator,
) -> None:
    target = rng.normal(size=(128, 3, 2))
    lower = target - 1.0
    upper = target + 2.0
    assert winkler_score(lower, upper, target, alpha=0.1) == pytest.approx(3.0)
    assert winkler_score(lower, upper, target, alpha=0.1) == pytest.approx(
        mean_interval_width(lower, upper)
    )


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("miss", [0.0, 0.25, 1.0, 4.0])
def test_winkler_penalty_is_linear_in_the_miss_distance(alpha: float, miss: float) -> None:
    """The penalty is ``(2/alpha) * distance`` outside the interval, either side.

    ``alpha`` is varied because the gate reads a 90 percent interval and a factor that
    happened to be hard-coded at 20 would pass a test written only at ``alpha = 0.1``.
    """
    lower = np.zeros((1, 1, 1))
    upper = np.ones((1, 1, 1))
    expected = 1.0 + (2.0 / alpha) * miss
    assert winkler_score(lower, upper, np.full((1, 1, 1), 1.0 + miss), alpha) == pytest.approx(
        expected
    )
    assert winkler_score(lower, upper, np.full((1, 1, 1), -miss), alpha) == pytest.approx(expected)


def test_winkler_penalty_slope_is_exactly_two_over_alpha() -> None:
    """The factor itself, read off as a slope rather than assumed from one point."""
    alpha = 0.1
    lower = np.zeros((1, 1, 1))
    upper = np.ones((1, 1, 1))
    at_one = winkler_score(lower, upper, np.full((1, 1, 1), 2.0), alpha)
    at_three = winkler_score(lower, upper, np.full((1, 1, 1), 4.0), alpha)
    assert (at_three - at_one) / 2.0 == pytest.approx(2.0 / alpha)


def test_winkler_rejects_an_alpha_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        winkler_score(np.zeros((2, 1, 1)), np.ones((2, 1, 1)), np.zeros((2, 1, 1)), alpha=0.0)


# ---------------------------------------------------------------------------
# Crossing rate: the diagnostic for post-hoc sorting
# ---------------------------------------------------------------------------


def test_crossing_rate_flags_only_elements_with_an_inversion() -> None:
    raw = np.asarray(
        [
            [[[0.0, 1.0, 2.0]]],  # monotone
            [[[0.0, 2.0, 1.0]]],  # one inversion
            [[[1.0, 1.0, 1.0]]],  # ties are not inversions
        ]
    )
    assert crossing_terms(raw).ravel().tolist() == [0.0, 1.0, 0.0]


def test_a_sorted_fan_has_a_zero_crossing_rate(rng: np.random.Generator) -> None:
    """The diagnostic must be taken on the raw fan or it measures nothing."""
    raw = rng.normal(size=(64, 2, 3, len(FAN_9)))
    assert float(crossing_terms(raw).mean()) > 0.0
    assert float(crossing_terms(np.sort(raw, axis=-1)).mean()) == 0.0


# ---------------------------------------------------------------------------
# The streaming table
# ---------------------------------------------------------------------------


def _per_key_sums(
    raw_fan: FloatArray, sorted_fan: FloatArray, target: FloatArray, n_keys: int, alpha: float
) -> dict[str, FloatArray]:
    """Accumulate the per-realization sums the way a runner would.

    Args:
        raw_fan: Unsorted fan, shape ``(n_keys * per_key, H, C, Q)``, corpus units.
        sorted_fan: The same fan sorted ascending along ``Q``, same shape.
        target: Targets, shape ``(n_keys * per_key, H, C)``, corpus units.
        n_keys: Number of realizations the leading axis groups into, contiguously.
        alpha: Nominal miscoverage rate of the interval.

    Returns:
        The six accumulators, each ``(n_keys, H, C)`` except ``pinball_sum``, which is
        ``(n_keys, H, C, Q)``.
    """
    lower = sorted_fan[..., 0]
    upper = sorted_fan[..., -1]

    def by_key(terms: FloatArray) -> FloatArray:
        return np.asarray(
            terms.reshape((n_keys, -1, *terms.shape[1:])).sum(axis=1), dtype=np.float64
        )

    return {
        "n_covered": by_key(coverage_terms(lower, upper, target)),
        "width_sum": by_key(width_terms(lower, upper)),
        "winkler_sum": by_key(winkler_terms(lower, upper, target, alpha)),
        "crps_sum": by_key(crps_terms(sorted_fan, target, FAN_9)),
        "pinball_sum": by_key(pinball_terms(sorted_fan, target, FAN_9)),
        "crossing_count": by_key(crossing_terms(raw_fan)),
    }


def test_probabilistic_table_from_sums_matches_the_array_path(rng: np.random.Generator) -> None:
    """The streaming path and the array path must be the same computation.

    Mirrors ``tests/test_metrics.py::test_metrics_table_from_sums_matches_the_array_path``.
    The array path is only ever an oracle: a production fan is ``434 304 x 150 x 6 x 9``
    float64.
    """
    n_keys, per_key, h_max, alpha = 5, 8, 6, 0.1
    n_windows = n_keys * per_key
    shape = (n_windows, h_max, len(DOFS))
    target = rng.normal(size=shape)
    centre = target + rng.normal(scale=0.4, size=shape)
    sigma = rng.uniform(0.4, 1.6, size=shape)
    raw_fan = _gaussian_fan(centre, sigma, FAN_9) + rng.normal(scale=0.2, size=(*shape, len(FAN_9)))
    sorted_fan = np.sort(raw_fan, axis=-1)
    horizons = (1, 3, 6)

    table = probabilistic_table_from_sums(
        **_per_key_sums(raw_fan, sorted_fan, target, n_keys, alpha),
        n=n_windows,
        dof_names=DOFS,
        horizons=horizons,
        fs_hz=FS_HZ,
        alpha=alpha,
    )
    assert list(table.columns) == list(PROBABILISTIC_METRIC_COLUMNS)
    assert len(table) == len(DOFS) * len(horizons)

    for channel, dof in enumerate(DOFS):
        for horizon in horizons:
            cell = table[(table["dof"] == dof) & (table["horizon_samples"] == horizon)]
            assert len(cell) == 1
            row = cell.iloc[0]
            step = slice(horizon - 1, horizon)
            chan = slice(channel, channel + 1)
            fan_cell = sorted_fan[:, step, chan, :]
            raw_cell = raw_fan[:, step, chan, :]
            target_cell = target[:, step, chan]
            lower_cell = fan_cell[..., 0]
            upper_cell = fan_cell[..., -1]
            assert row["horizon_s"] == pytest.approx(horizon / FS_HZ)
            assert row["n_windows"] == n_windows
            assert row["n_quantiles"] == len(FAN_9)
            assert row["alpha"] == pytest.approx(alpha)
            assert row["picp"] == pytest.approx(picp(lower_cell, upper_cell, target_cell))
            assert row["mean_interval_width"] == pytest.approx(
                mean_interval_width(lower_cell, upper_cell)
            )
            assert row["winkler"] == pytest.approx(
                winkler_score(lower_cell, upper_cell, target_cell, alpha)
            )
            assert row["crps"] == pytest.approx(crps_from_quantiles(fan_cell, target_cell, FAN_9))
            assert row["pinball"] == pytest.approx(pinball(fan_cell, target_cell, FAN_9))
            assert row["crossing_rate"] == pytest.approx(float(crossing_terms(raw_cell).mean()))


def test_probabilistic_table_reports_coverage_next_to_sharpness() -> None:
    """P5-D2: coverage is never reported without a width beside it."""
    columns = list(PROBABILISTIC_METRIC_COLUMNS)
    assert columns.index("mean_interval_width") == columns.index("picp") + 1


def test_probabilistic_horizon_metric_is_per_step_not_cumulative() -> None:
    """Horizon h is the score at lead time exactly h, never the mean over 1..h.

    Mirrors ``tests/test_metrics.py::test_horizon_metric_is_per_step_not_cumulative``. The
    per-window width grows with lead time, so a cumulative reading at h=3 would report the
    mean of {1, 2, 3} = 2.0 where the per-step value is exactly 3.
    """
    n_keys, per_key = 2, 4
    n_windows = n_keys * per_key
    per_step = np.asarray([1.0, 2.0, 3.0])[None, :, None]
    width_sum = np.broadcast_to(per_step, (n_keys, 3, 1)) * per_key
    table = probabilistic_table_from_sums(
        n_covered=np.full((n_keys, 3, 1), float(per_key)),
        width_sum=width_sum.copy(),
        winkler_sum=width_sum.copy(),
        crps_sum=width_sum.copy(),
        pinball_sum=width_sum.copy()[..., None],
        crossing_count=np.zeros((n_keys, 3, 1)),
        n=n_windows,
        dof_names=("roll",),
        horizons=(1, 2, 3),
        fs_hz=FS_HZ,
    )
    assert table["mean_interval_width"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert table["winkler"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert table["crps"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert table["pinball"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    cumulative = [1.0, 1.5, 2.0]
    assert table["mean_interval_width"].tolist() != pytest.approx(cumulative)
    assert table["horizon_s"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def _unit_sums(
    n_keys: int = 2, h_max: int = 4, n_channels: int = 1, n_levels: int = 3
) -> dict[str, FloatArray | int]:
    """Return a well-formed set of accumulators for one window per realization.

    Returns:
        Keyword arguments for :func:`probabilistic_table_from_sums`, describing a covered,
        unit-width, monotone forecast.
    """
    shape = (n_keys, h_max, n_channels)
    return {
        "n_covered": np.ones(shape),
        "width_sum": np.ones(shape),
        "winkler_sum": np.ones(shape),
        "crps_sum": np.full(shape, 0.5),
        "pinball_sum": np.full((*shape, n_levels), 0.25),
        "crossing_count": np.zeros(shape),
        "n": n_keys,
    }


def test_probabilistic_table_rejects_a_horizon_beyond_the_forecast() -> None:
    with pytest.raises(ValueError, match=r"outside \[1, 4\]"):
        probabilistic_table_from_sums(
            **_unit_sums(), dof_names=("roll",), horizons=(5,), fs_hz=FS_HZ
        )


def test_probabilistic_table_rejects_an_accumulator_of_the_wrong_shape() -> None:
    sums = _unit_sums()
    sums["width_sum"] = np.ones((2, 4, 2))
    with pytest.raises(ValueError, match="width_sum has shape"):
        probabilistic_table_from_sums(**sums, dof_names=("roll",), horizons=(1,), fs_hz=FS_HZ)


def test_probabilistic_table_rejects_a_dof_name_list_that_does_not_match_the_sums() -> None:
    with pytest.raises(ValueError, match="1 channels but 3 DOF names"):
        probabilistic_table_from_sums(**_unit_sums(), dof_names=DOFS, horizons=(1,), fs_hz=FS_HZ)


def test_probabilistic_table_refuses_a_negative_mean_width() -> None:
    """The streamed path's only chance to catch an unsorted fan.

    :func:`picp` refuses crossed endpoints elementwise, but a runner that accumulated sums
    from a crossed interval never calls it. A negative mean width is what survives to the
    table, and it is refused there rather than published.
    """
    sums = _unit_sums()
    sums["width_sum"] = np.full((2, 4, 1), -0.5)
    with pytest.raises(ValueError, match="sort_quantiles"):
        probabilistic_table_from_sums(**sums, dof_names=("roll",), horizons=(1,), fs_hz=FS_HZ)


def test_probabilistic_table_refuses_more_covered_windows_than_windows() -> None:
    """A coverage above 1 means the counts and the window total came from different passes."""
    sums = _unit_sums()
    sums["n_covered"] = np.full((2, 4, 1), 3.0)
    with pytest.raises(ValueError, match=r"picp falls outside \[0, 1\]"):
        probabilistic_table_from_sums(**sums, dof_names=("roll",), horizons=(1,), fs_hz=FS_HZ)


def test_probabilistic_table_rejects_an_alpha_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        probabilistic_table_from_sums(
            **_unit_sums(), dof_names=("roll",), horizons=(1,), fs_hz=FS_HZ, alpha=1.0
        )


def test_probabilistic_table_pinball_column_averages_over_levels() -> None:
    """The level axis is averaged, not summed: three levels at 0.25 each report 0.25."""
    table = probabilistic_table_from_sums(
        **_unit_sums(), dof_names=("roll",), horizons=(1, 2), fs_hz=FS_HZ
    )
    assert table["pinball"].tolist() == pytest.approx([0.25, 0.25])
    assert table["n_quantiles"].tolist() == [3, 3]


# ---------------------------------------------------------------------------
# The degenerate best case
# ---------------------------------------------------------------------------


def test_a_perfectly_sharp_and_correct_forecast_scores_zero(rng: np.random.Generator) -> None:
    """Zero width, PICP 1.0, zero pinball, zero CRPS, zero Winkler.

    The joint anchor: every one of these can be made to look good on its own, and only the
    degenerate case pins all five at once.
    """
    target = rng.normal(size=(64, 3, 2))
    fan = np.repeat(target[..., None], len(FAN_9), axis=-1)
    lower = fan[..., 0]
    upper = fan[..., -1]
    assert picp(lower, upper, target) == 1.0
    assert mean_interval_width(lower, upper) == 0.0
    assert pinball(fan, target, FAN_9) == 0.0
    assert crps_from_quantiles(fan, target, FAN_9) == 0.0
    assert winkler_score(lower, upper, target, alpha=0.1) == 0.0
    assert float(crossing_terms(fan).mean()) == 0.0


def test_the_streamed_table_of_a_perfect_forecast_is_zero(rng: np.random.Generator) -> None:
    """The same anchor through the production path."""
    n_keys, per_key = 3, 5
    target = rng.normal(size=(n_keys * per_key, 4, len(DOFS)))
    fan = np.repeat(target[..., None], len(FAN_9), axis=-1)
    table = probabilistic_table_from_sums(
        **_per_key_sums(fan, fan, target, n_keys, 0.1),
        n=n_keys * per_key,
        dof_names=DOFS,
        horizons=(1, 4),
        fs_hz=FS_HZ,
    )
    assert table["picp"].tolist() == [1.0] * len(table)
    for column in ("mean_interval_width", "winkler", "crps", "pinball", "crossing_rate"):
        assert table[column].tolist() == [0.0] * len(table)


def test_the_table_is_a_dataframe_with_one_row_per_cell() -> None:
    table = probabilistic_table_from_sums(
        **_unit_sums(n_channels=3), dof_names=DOFS, horizons=(1, 2, 3), fs_hz=FS_HZ
    )
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 9
    assert table["dof"].tolist() == ["roll"] * 3 + ["pitch"] * 3 + ["heave"] * 3


# --------------------------------------------------------------------------------------
# S8: the streaming runner -- units, key mapping, sorting, and the two passes agreeing.
#
# The scalar metrics above are the definitions. These tests are about the plumbing between
# a model's raw head output and a published column: the normalised-to-corpus-units round
# trip, the window-to-realization mapping, that crossing is counted before sorting, and
# that the distributional pass and the point pass describe the same windows.
# --------------------------------------------------------------------------------------


class _ConstantGaussian(BaseForecaster):
    """Emit a fixed normalised mean and log-variance for every window.

    Deliberately not a trained model: an oracle whose interval is known in closed form is
    the only way to check the unit round trip, because the answer can be written down
    before the code runs.
    """

    SUPPORTED_HEADS = ("point", "gaussian")

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        *,
        mean: Tensor,
        log_var: float,
    ) -> None:
        super().__init__(
            lookback, max_horizon, n_input_channels, n_target_channels, head="gaussian"
        )
        self.register_buffer("mean", mean)
        self.log_var = log_var

    def forward(self, x: Tensor) -> Tensor:
        batch = int(x.shape[0])
        mean = self.mean.expand(batch, self.max_horizon, self.n_target_channels)
        log_var = torch.full_like(mean, self.log_var)
        return torch.stack((mean, log_var), dim=-1)


class _CrossedFan(BaseForecaster):
    """Emit a strictly *descending* quantile fan, so every element crosses."""

    SUPPORTED_HEADS = ("point", "quantile")

    def __init__(
        self, lookback: int, max_horizon: int, n_input_channels: int, n_target_channels: int
    ) -> None:
        super().__init__(
            lookback,
            max_horizon,
            n_input_channels,
            n_target_channels,
            n_quantiles=len(FAN_9),
            head="quantile",
        )

    def forward(self, x: Tensor) -> Tensor:
        batch = int(x.shape[0])
        descending = torch.linspace(1.0, -1.0, len(FAN_9), dtype=torch.float32)
        return descending.expand(batch, self.max_horizon, self.n_target_channels, len(FAN_9))


def _prob_dataset(corpus_root: Path, manifest: pd.DataFrame, cfg: DataConfig) -> DeckMotionDataset:
    """Build the ``id`` test partition, carrying train statistics as the guards require."""
    split = build_split(manifest, "id")
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(corpus_root, split, "train", cfg, spec)
    return DeckMotionDataset(corpus_root, split, "test", cfg, spec, stats=train.norm_stats)


def test_the_scoring_fan_equals_the_model_layers_definition() -> None:
    """The nine levels are defined once, in ``dmf.models.heads``.

    This module spells them out as a literal so the metric tests do not depend on the model
    layer. That duplication is only safe while the two agree, so this is the assertion that
    makes it safe -- without it, changing ``QUANTILE_FAN_9`` would silently leave every
    anchor above testing a fan the project no longer scores.
    """
    assert FAN_9 == QUANTILE_FAN_9
    assert SCORING_LEVELS == QUANTILE_FAN_9


def test_the_runner_refuses_a_point_model(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    n_in = len(dataset.input_columns)
    n_out = len(dataset.target_columns)
    point = Persistence(spec.lookback, spec.max_horizon, n_in, n_out)
    with pytest.raises(ValueError, match="head_kind='point'"):
        evaluate_probabilistic_models(
            {"persistence": point},
            dataset,
            horizons=small_data_cfg.horizons,
            fs_hz=small_data_cfg.fs_hz,
        )


def test_the_runner_refuses_realization_keys_the_point_pass_did_not_score(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """A row's point and distributional columns must describe the same windows."""
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    model = _ConstantGaussian(
        spec.lookback,
        spec.max_horizon,
        len(dataset.input_columns),
        n_out,
        mean=torch.zeros(1, spec.max_horizon, n_out),
        log_var=0.0,
    )
    with pytest.raises(ValueError, match="do not match the point pass"):
        evaluate_probabilistic_models(
            {"g": model},
            dataset,
            horizons=small_data_cfg.horizons,
            fs_hz=small_data_cfg.fs_hz,
            expected_keys=list(dataset.realization_keys)[:-1],
        )


def test_an_oracle_mean_covers_every_window_at_exactly_the_closed_form_width(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The normalised-to-corpus-units round trip, checked against an answer written by hand.

    A Gaussian head whose mean is the *true* target makes every error zero, so coverage is
    1.0 for any positive sigma and the only thing left to check is the width -- which is
    ``2 * z(0.95) * sigma * scale[c]`` in corpus units, per channel. Getting the scale wrong,
    or applying the window mean to the spread as well as to the location, changes that number
    and nothing else in the table would show it. Units are degrees for roll and pitch, metres
    for heave, deg/s and m/s for the rates; sigma is dimensionless.
    """
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)
    log_var = -2.0
    sigma = float(np.exp(0.5 * log_var))

    # An oracle mean in normalised space: the exact normalised target of every window.
    truths = torch.stack([dataset[i][1] for i in range(len(dataset))])
    means = torch.stack([dataset[i][2] for i in range(len(dataset))])
    scale = torch.as_tensor(
        dataset.norm_stats.subset(tuple(dataset.target_columns)).scale, dtype=torch.float32
    )
    normalised = (truths - means) / scale

    # One batch covering the whole partition, so the stored answer lines up with loader
    # order without an index-tracking model.
    class _Answer(BaseForecaster):
        SUPPORTED_HEADS = ("point", "gaussian")

        def __init__(self) -> None:
            super().__init__(
                spec.lookback,
                spec.max_horizon,
                len(dataset.input_columns),
                n_out,
                head="gaussian",
            )
            self.register_buffer("answer", normalised)

        def forward(self, inner: Tensor) -> Tensor:
            del inner
            log_variance = torch.full_like(self.answer, log_var)
            return torch.stack((self.answer, log_variance), dim=-1)

    table, _ = evaluate_probabilistic_models(
        {"oracle": _Answer()},
        dataset,
        horizons=small_data_cfg.horizons,
        fs_hz=small_data_cfg.fs_hz,
        batch_size=len(dataset),
    )
    assert table["picp"].tolist() == [1.0] * len(table)
    z = float(norm.ppf(0.95))
    for _, row in table.iterrows():
        channel = list(dataset.target_columns).index(row["dof"])
        expected = 2.0 * z * sigma * float(scale[channel])
        # rel=1e-6, not exact equality, and the tolerance is stated rather than tuned: the
        # dataset stores its de-meaned, scaled window as float32, so the recovered width
        # carries a relative rounding error of order 2**-24 ~ 6e-8. That is the same floor
        # `persistence_pipeline_sanity` runs at (P2-D9, measured 5.006e-08 over 441 984
        # windows), and anything larger here would be a real unit error rather than storage
        # precision.
        assert row["mean_interval_width"] == pytest.approx(expected, rel=1e-6)


def test_a_descending_fan_is_counted_as_crossed_and_then_sorted(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Crossing is measured on the raw fan; the reported interval is the sorted one.

    A strictly descending fan crosses on every element, so ``crossing_rate`` is exactly 1.0.
    The interval must still be well formed -- a positive width, and no raise from the
    table's ``lower > upper`` refusal -- because
    :class:`dmf.models.heads.PredictiveDistribution` sorts on construction. If crossing were
    measured after the sort it would read 0.0 and this test would fail, which is the point.
    """
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    model = _CrossedFan(
        spec.lookback, spec.max_horizon, len(dataset.input_columns), len(dataset.target_columns)
    )
    table, _ = evaluate_probabilistic_models(
        {"crossed": model},
        dataset,
        horizons=small_data_cfg.horizons,
        fs_hz=small_data_cfg.fs_hz,
    )
    assert table["crossing_rate"].tolist() == [1.0] * len(table)
    assert (table["mean_interval_width"] > 0.0).all()


def test_the_picp_interval_brackets_its_point_estimate_and_is_reproducible_from_its_seed(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    n_out = len(dataset.target_columns)

    def build() -> _ConstantGaussian:
        return _ConstantGaussian(
            spec.lookback,
            spec.max_horizon,
            len(dataset.input_columns),
            n_out,
            mean=torch.zeros(1, spec.max_horizon, n_out),
            log_var=0.0,
        )

    kwargs = {
        "horizons": small_data_cfg.horizons,
        "fs_hz": small_data_cfg.fs_hz,
        "n_boot": 64,
        "bootstrap_seed": 7,
    }
    first, _ = evaluate_probabilistic_models({"g": build()}, dataset, **kwargs)
    second, _ = evaluate_probabilistic_models({"g": build()}, dataset, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert (first["picp_ci_lo"] <= first["picp"]).all()
    assert (first["picp"] <= first["picp_ci_hi"]).all()


def test_the_distributional_and_point_passes_score_the_same_windows(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """``point_view`` is what makes a probabilistic row carry RMSE and skill (P5-D5).

    The two passes build their own loaders, so this asserts the property the ``expected_keys``
    guard exists to protect: identical realization keys and identical window counts. If they
    ever diverge, a row's ``picp`` and its ``rmse`` would describe different data.
    """
    dataset = _prob_dataset(small_corpus, small_manifest, small_data_cfg)
    spec = dataset.window_spec
    n_in = len(dataset.input_columns)
    n_out = len(dataset.target_columns)
    model = _ConstantGaussian(
        spec.lookback,
        spec.max_horizon,
        n_in,
        n_out,
        mean=torch.zeros(1, spec.max_horizon, n_out),
        log_var=0.0,
    )
    point_table, _ = evaluate_models(
        {
            "persistence": Persistence(spec.lookback, spec.max_horizon, n_in, n_out),
            "g": point_view(model),
        },
        dataset,
        persistence_key="persistence",
        horizons=small_data_cfg.horizons,
        fs_hz=small_data_cfg.fs_hz,
        n_boot=16,
    )
    prob_table, _ = evaluate_probabilistic_models(
        {"g": model},
        dataset,
        horizons=small_data_cfg.horizons,
        fs_hz=small_data_cfg.fs_hz,
        n_boot=16,
        expected_keys=list(dataset.realization_keys),
    )
    point_g = point_table[point_table["model"] == "g"].reset_index(drop=True)
    assert prob_table["n_windows"].tolist() == point_g["n_windows"].tolist()
    assert prob_table["dof"].tolist() == point_g["dof"].tolist()
    assert prob_table["horizon_samples"].tolist() == point_g["horizon_samples"].tolist()
