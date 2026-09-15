"""Split-conformal calibration: the guarantee, the guards, and the invariants it must keep.

The coverage claim is tested on synthetic exchangeable data rather than on the corpus, so
that it is a test of the *construction* and not of the corpus's own regularity: if the
finite-sample correction or the score is wrong, these fail without a GPU or a checkpoint.

The guards are tested against the real split machinery, because what they protect is a
property of the split policy -- a calibration fitted on train is in-sample and one fitted on
test is the leak -- and a mock split would not carry that property to be protected.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.eval.conformal import (
    calibrate_models,
    conformity_scores,
    order_statistic_index,
)
from dmf.models.conformal import ConformalInterval
from dmf.models.dlinear import DLinear
from dmf.models.heads import QUANTILE_FAN_9, IntervalPredictor, PredictiveDistribution

ALPHA = 0.1

#: The fixture corpus's validation partition holds six windows, which cannot certify a 90
#: percent level at all -- ``ceil((n+1)(1-alpha)) = 7 > 6`` -- and
#: :func:`dmf.eval.conformal.order_statistic_index` refuses it rather than clipping. That
#: refusal is itself tested above; the two end-to-end tests below therefore ask for a level
#: six windows *can* carry, because what they check is the plumbing and the provenance, not
#: the coverage number. 0.55 rather than 0.5 because its tails are 0.275 and 0.725, which are
#: members of QUANTILE_FAN_9 exactly -- ``quantiles_at`` never interpolates, so an alpha whose
#: tails are not fan members is not readable at all.
FIXTURE_ALPHA = 0.55


def _head(
    head: str,
    *,
    max_horizon: int = 4,
    n_target: int = 2,
    n_input: int = 3,
    lookback: int = 16,
) -> DLinear:
    """Build an untrained interval head at a small geometry.

    Args:
        head: ``"quantile"`` or ``"gaussian"``.
        max_horizon: Forecast length.
        n_target: Target channels.
        n_input: Input channels; must be at least ``n_target`` (P2-D4).
        lookback: Input window length.

    Returns:
        The model, in eval mode.
    """
    model = DLinear(
        lookback=lookback,
        max_horizon=max_horizon,
        n_input_channels=n_input,
        n_target_channels=n_target,
        n_quantiles=len(QUANTILE_FAN_9) if head == "quantile" else 0,
        head=head,
        kernel_size=5,
    )
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# The finite-sample correction
# --------------------------------------------------------------------------------------


def test_the_correction_is_an_order_statistic_and_not_an_interpolated_quantile() -> None:
    """``k = ceil((n+1)(1-alpha))``, indexing sorted scores rather than interpolating.

    The distinction is the whole guarantee: ``dmf.train.closed_form`` may use ``np.quantile``'s
    linear interpolation for the residual *floor*, whose claim is a reference width, but a
    conformal claim is a coverage probability and the order statistic is what delivers it.
    """
    rng = np.random.default_rng(0)
    scores = np.sort(rng.normal(size=19))
    k = order_statistic_index(19, ALPHA)
    assert k == math.ceil(20 * 0.9) == 18
    assert scores[k - 1] == pytest.approx(scores[17])
    # The interpolated quantile is a different number, and that difference is the point.
    assert scores[k - 1] != pytest.approx(float(np.quantile(scores, 0.9)))


def test_a_calibration_set_too_small_to_certify_the_level_is_refused() -> None:
    """``k > n`` means an unbounded interval, which is refused rather than clipped.

    Clipping to the largest observed score would report a finite interval with no guarantee
    behind it -- the failure mode that makes an under-powered calibration look successful.
    """
    with pytest.raises(ValueError, match="unbounded"):
        order_statistic_index(8, ALPHA)
    # One more sample and the level is certifiable.
    assert order_statistic_index(9, ALPHA) == 9


@pytest.mark.parametrize("bad", [0, -1])
def test_the_correction_refuses_a_nonpositive_sample_count(bad: int) -> None:
    """A calibration with no scores is a programming error, not a wide interval."""
    with pytest.raises(ValueError, match="at least one"):
        order_statistic_index(bad, ALPHA)


# --------------------------------------------------------------------------------------
# The coverage guarantee, on exchangeable synthetic data
# --------------------------------------------------------------------------------------


def test_calibration_restores_nominal_coverage_on_exchangeable_data() -> None:
    """A deliberately over-confident fan is corrected to ~90 percent coverage.

    The stub's interval is three times too narrow, so it starts far below nominal. Calibration
    and evaluation draw from one exchangeable pool, which is the condition the guarantee needs
    and the condition that holds in the ``id`` regime and nowhere else.
    """
    rng = np.random.default_rng(7)
    n_cal, n_eval = 4000, 4000
    truth = rng.normal(size=n_cal + n_eval)
    # A fan centred on the truth's mean with a far-too-narrow spread.
    sigma_true, sigma_claimed = 1.0, 1.0 / 3.0
    z = np.asarray([-1.6449, 0.0, 1.6449])
    fan = np.zeros((n_cal + n_eval, 3)) + z * sigma_claimed

    scores = np.max(
        np.stack(
            [
                (fan[:n_cal, 1] - truth[:n_cal]) / (fan[:n_cal, 1] - fan[:n_cal, 0]),
                (truth[:n_cal] - fan[:n_cal, 1]) / (fan[:n_cal, 2] - fan[:n_cal, 1]),
            ]
        ),
        axis=0,
    )
    k = order_statistic_index(n_cal, ALPHA)
    gamma = float(np.partition(scores, k - 1)[k - 1])

    lo = fan[n_cal:, 1] + gamma * (fan[n_cal:, 0] - fan[n_cal:, 1])
    hi = fan[n_cal:, 1] + gamma * (fan[n_cal:, 2] - fan[n_cal:, 1])
    before = float(np.mean((truth[n_cal:] >= fan[n_cal:, 0]) & (truth[n_cal:] <= fan[n_cal:, 2])))
    after = float(np.mean((truth[n_cal:] >= lo) & (truth[n_cal:] <= hi)))

    assert before < 0.5, "the stub must start under-covering for this to be a test"
    assert 0.88 <= after <= 0.92
    # The scale recovered the true-to-claimed ratio, which is what a scale error looks like.
    assert gamma == pytest.approx(sigma_true / sigma_claimed, rel=0.05)


def test_calibration_tightens_an_over_covering_interval() -> None:
    """``gamma < 1`` is a correct outcome, not a suspicious one.

    This project's committed finding is that the deep heads **over-cover** -- PICP up to 0.984
    -- so a calibration that never narrowed anything would be failing to do its job on most of
    the table.
    """
    rng = np.random.default_rng(11)
    n = 8000
    truth = rng.normal(size=n)
    z = np.asarray([-1.6449, 0.0, 1.6449])
    fan = np.zeros((n, 3)) + z * 3.0  # three times too wide
    half = n // 2
    scores = np.max(
        np.stack(
            [
                (fan[:half, 1] - truth[:half]) / (fan[:half, 1] - fan[:half, 0]),
                (truth[:half] - fan[:half, 1]) / (fan[:half, 2] - fan[:half, 1]),
            ]
        ),
        axis=0,
    )
    k = order_statistic_index(half, ALPHA)
    gamma = float(np.partition(scores, k - 1)[k - 1])
    assert gamma < 1.0, "an over-wide interval must be narrowed"
    lo = fan[half:, 1] + gamma * (fan[half:, 0] - fan[half:, 1])
    hi = fan[half:, 1] + gamma * (fan[half:, 2] - fan[half:, 1])
    assert 0.88 <= float(np.mean((truth[half:] >= lo) & (truth[half:] <= hi))) <= 0.92


# --------------------------------------------------------------------------------------
# The wrapper's structural invariants
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("head", ["quantile", "gaussian"])
def test_the_point_forecast_is_bitwise_unchanged(head: str) -> None:
    """Calibration moves the interval and nothing else.

    This is the strongest available form of the additivity Gate 5 requires (P5-D2): if the
    point forecast cannot move, then no RMSE, MAE, skill, nrmse or phase-lag number in this
    project can move either, and the uncalibrated reading survives the arm intact. Asserted
    with ``torch.equal`` rather than a tolerance, because "close" would allow a drift.
    """
    base = _head(head)
    wrapper = ConformalInterval(base)
    wrapper.set_gamma(np.full((base.max_horizon, base.n_target_channels), 1.7))
    wrapper.eval()
    x = torch.randn(5, base.lookback, base.n_input_channels)
    with torch.no_grad():
        before = PredictiveDistribution(base(x), head, base.quantile_levels).point()
        after = wrapper.predict(x).point()
    assert torch.equal(before, after)


@pytest.mark.parametrize("head", ["quantile", "gaussian"])
def test_the_calibrated_width_is_exactly_gamma_times_the_base_width(head: str) -> None:
    """``width_ratio_calibrated / width_ratio_uncalibrated == gamma``, per cell.

    The property that makes the scale factor readable as "this head's fan is 1.34x too narrow
    here", and the reason the construction is multiplicative rather than additive: an additive
    offset would compress the conditional sharpness the floor contrast exists to measure.
    """
    base = _head(head)
    wrapper = ConformalInterval(base)
    rng = np.random.default_rng(3)
    gamma = rng.uniform(0.4, 2.5, size=(base.max_horizon, base.n_target_channels))
    wrapper.set_gamma(gamma)
    wrapper.eval()
    x = torch.randn(6, base.lookback, base.n_input_channels)
    with torch.no_grad():
        lo_b, hi_b = PredictiveDistribution(base(x), head, base.quantile_levels).interval(ALPHA)
        lo_w, hi_w = wrapper.predict(x).interval(ALPHA)
    ratio = ((hi_w - lo_w) / (hi_b - lo_b)).mean(dim=0).numpy()
    assert np.allclose(ratio, gamma, atol=1e-5)


def test_the_calibrated_fan_never_crosses_before_any_post_hoc_sort() -> None:
    """``sort_quantiles`` is a no-op on this model's raw output.

    A positive scale applied to a sorted fan cannot invert it, so crossing is structurally
    impossible here -- the same property P5-D16 records for the residual-interval floor, and
    the reason :func:`dmf.eval.probabilistic._reject_crossed` can never fire on these rows.
    """
    base = _head("quantile")
    wrapper = ConformalInterval(base)
    rng = np.random.default_rng(5)
    wrapper.set_gamma(rng.uniform(0.1, 5.0, size=(base.max_horizon, base.n_target_channels)))
    wrapper.eval()
    x = torch.randn(9, base.lookback, base.n_input_channels)
    with torch.no_grad():
        raw = wrapper(x)
    assert bool((raw.diff(dim=-1) >= -1e-6).all())


def test_an_uncalibrated_wrapper_raises_rather_than_passing_the_base_through() -> None:
    """``gamma = 1`` is a well-formed passthrough, which is exactly why it must not be scored.

    An uncalibrated instance that silently emitted the base's own fan would appear in the
    results table as a head that needed no calibration -- a result, rather than the bug it is.
    """
    wrapper = ConformalInterval(_head("quantile"))
    with pytest.raises(RuntimeError, match="no conformal scale"):
        wrapper(torch.randn(2, 16, 3))


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (0.0, "strictly positive"),
        (-1.0, "strictly positive"),
        (float("nan"), "non-finite"),
        (1e6, "beyond"),
    ],
)
def test_set_gamma_refuses_a_scale_that_would_not_produce_an_interval(
    bad: float, match: str
) -> None:
    """Non-positive, non-finite and absurd scales are refused on installation.

    A zero scale collapses the fan onto the point forecast, which scores as perfect sharpness
    with no coverage; a negative one reverses it. Both are well-formed tensors and neither is
    an interval, so the check belongs on the setter rather than downstream.
    """
    base = _head("quantile")
    wrapper = ConformalInterval(base)
    gamma = np.ones((base.max_horizon, base.n_target_channels))
    gamma[0, 0] = bad
    with pytest.raises(ValueError, match=match):
        wrapper.set_gamma(gamma)


def test_set_gamma_refuses_the_wrong_shape() -> None:
    """One scale per ``(horizon, channel)``; anything else is a transposed axis."""
    base = _head("quantile")
    wrapper = ConformalInterval(base)
    with pytest.raises(ValueError, match="expected"):
        wrapper.set_gamma(np.ones((base.n_target_channels, base.max_horizon)))


def test_the_wrapper_satisfies_the_interval_predictor_seam() -> None:
    """The seam declared in ``heads.py`` gets its first implementer that is not a test double.

    ``tests/test_models.py::test_the_conformal_seam_moves_coverage_without_importing_a_model``
    proves the protocol composes; this asserts the shipped class actually satisfies it, so the
    docstring claim in ``heads.py`` is about the code rather than about an intention.
    """
    base = _head("quantile")
    wrapper = ConformalInterval(base)
    wrapper.set_gamma(np.ones((base.max_horizon, base.n_target_channels)))
    assert isinstance(wrapper, IntervalPredictor)
    assert isinstance(wrapper.predict(torch.randn(2, 16, 3)), PredictiveDistribution)


def test_the_wrapper_imports_no_architecture() -> None:
    """The seam holds a model; it does not know what kind.

    ``heads.py`` states that a wrapper adjusts intervals "without importing or subclassing any
    model". That is a property of the shipped module, so it is checked against the module's
    own source rather than trusted.
    """
    source = Path("src/dmf/models/conformal.py").read_text()
    for architecture in ("dlinear", "lstm", "tcn", "transformer"):
        assert f"import {architecture}" not in source
        assert f"from dmf.models.{architecture}" not in source


def test_a_point_head_has_no_interval_to_calibrate() -> None:
    """Refused at construction, because the failure is in the caller's model selection."""
    point = DLinear(
        lookback=16,
        max_horizon=4,
        n_input_channels=3,
        n_target_channels=2,
        head="point",
        kernel_size=5,
    )
    with pytest.raises(ValueError, match="point forecaster carries no predictive distribution"):
        ConformalInterval(point)


def test_the_parameter_count_includes_the_calibration() -> None:
    """Counted in the direction that does not flatter the arm.

    The calibration is ``H * C`` numbers and adding them to the count is the honest
    accounting: ``EmpiricalResidualInterval`` makes the same choice for the same reason.
    """
    base = _head("quantile")
    wrapper = ConformalInterval(base)
    assert wrapper.n_fitted_parameters == base.n_fitted_parameters + (
        base.max_horizon * base.n_target_channels
    )


# --------------------------------------------------------------------------------------
# The conformity score
# --------------------------------------------------------------------------------------


def test_a_target_is_covered_exactly_when_its_score_is_below_the_scale() -> None:
    """``s_i <= gamma`` iff the ``gamma``-scaled interval covers ``y_i``.

    This equivalence is what makes the order statistic a coverage guarantee rather than a
    heuristic, so it is asserted directly rather than inferred from an empirical rate.
    """
    torch.manual_seed(0)
    base = _head("quantile", max_horizon=3, n_target=2)
    x = torch.randn(64, base.lookback, base.n_input_channels)
    with torch.no_grad():
        dist = PredictiveDistribution(base(x), "quantile", base.quantile_levels)
    target = torch.randn(64, 3, 2) * 2.0
    scores, degenerate = conformity_scores(dist, target, alpha=ALPHA)
    assert not bool(degenerate.any())

    # Strictly between two adjacent scores, so no element sits exactly on the boundary. At an
    # exact tie the equivalence still holds in exact arithmetic but the reconstructed endpoint
    # can land one ULP the wrong side of the target, which would make this a test of floating
    # point rather than of the score.
    ordered = torch.sort(scores.flatten()).values
    midpoint = len(ordered) // 2
    gamma = float((ordered[midpoint] + ordered[midpoint + 1]) / 2.0)
    assert float(ordered[midpoint]) < gamma < float(ordered[midpoint + 1])
    median = dist.point()
    lower, upper = dist.interval(ALPHA)
    lo = median + gamma * (lower - median)
    hi = median + gamma * (upper - median)
    covered = (target >= lo) & (target <= hi)
    assert torch.equal(covered, scores <= gamma)


def test_conformity_scores_refuse_a_mismatched_target() -> None:
    """A shape mismatch is a wiring error and would otherwise broadcast silently."""
    base = _head("quantile", max_horizon=3, n_target=2)
    with torch.no_grad():
        dist = PredictiveDistribution(base(torch.randn(4, 16, 3)), "quantile", base.quantile_levels)
    with pytest.raises(ValueError, match="expected"):
        conformity_scores(dist, torch.randn(4, 3, 3), alpha=ALPHA)


# --------------------------------------------------------------------------------------
# The split guards
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("partition", ["train", "test"])
def test_calibration_refuses_any_partition_but_validation(
    small_corpus: Path, small_data_cfg: DataConfig, partition: str
) -> None:
    """Train scores are in-sample and test scores are the leak.

    Checked structurally, in the same place and for the same reason
    :func:`dmf.train.closed_form.fit_residual_interval` checks it, rather than left to the
    caller to pass the right dataset.
    """
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(load_manifest(small_corpus), "id")
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    wrong = (
        train
        if partition == "train"
        else DeckMotionDataset(
            small_corpus, split, "test", small_data_cfg, spec, stats=train.norm_stats
        )
    )
    model = _head(
        "quantile",
        max_horizon=spec.max_horizon,
        n_target=len(train.target_columns),
        n_input=len(train.input_columns),
        lookback=spec.lookback,
    )
    with pytest.raises(ValueError, match="validation partition"):
        calibrate_models({"m": model}, wrong, alpha=ALPHA)


def test_calibration_records_the_partition_and_statistics_it_used(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """The fit report carries its own provenance, so Gate 10 can check it from the artifact.

    A calibration that merely *was* correct, without a row saying so, is the kind of claim
    this project has had to withdraw before: the artifact has to be able to testify.
    """
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(load_manifest(small_corpus), "id")
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    val = DeckMotionDataset(
        small_corpus, split, "val", small_data_cfg, spec, stats=train.norm_stats
    )
    model = _head(
        "quantile",
        max_horizon=spec.max_horizon,
        n_target=len(train.target_columns),
        n_input=len(train.input_columns),
        lookback=spec.lookback,
    )
    fits = calibrate_models({"m": model}, val, alpha=FIXTURE_ALPHA)
    _, fit = fits["m"]
    assert fit.fitted_on == "id/val"
    assert fit.norm_stats_fitted_on.endswith("/train")
    assert fit.n_windows_used > 0
    assert fit.order_index == order_statistic_index(fit.n_windows_used, FIXTURE_ALPHA)
    assert fit.gamma.shape == (spec.max_horizon, len(train.target_columns))


def test_calibration_wraps_without_mutating_the_base(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """The passed-in models are untouched, so the uncalibrated rows stay scoreable.

    Both arms are scored in the same session, so a calibration that mutated its base in place
    would silently make the uncalibrated table a second copy of the calibrated one.
    """
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(load_manifest(small_corpus), "id")
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    val = DeckMotionDataset(
        small_corpus, split, "val", small_data_cfg, spec, stats=train.norm_stats
    )
    model = _head(
        "quantile",
        max_horizon=spec.max_horizon,
        n_target=len(train.target_columns),
        n_input=len(train.input_columns),
        lookback=spec.lookback,
    )
    x = torch.randn(4, spec.lookback, len(train.input_columns))
    with torch.no_grad():
        before = model(x).clone()
    wrapper, _ = calibrate_models({"m": model}, val, alpha=FIXTURE_ALPHA)["m"]
    with torch.no_grad():
        after = model(x)
    assert torch.equal(before, after)
    assert wrapper.base is model
