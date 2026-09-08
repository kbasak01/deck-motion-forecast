"""Phase-lag diagnostics -- the §6.1 metric that had never been computed.

What is pinned here:

1. **Lag recovery.** A known lag injected into a synthetic narrowband signal is recovered
   to within one sample by the raw argmax, and to much better than one sample by the
   parabolic interpolation.
2. **The interpolation earns its place.** On a deliberately sub-sample lag, the
   interpolated estimate is strictly closer to truth than the raw argmax it refines. If it
   were not, it would be decoration, and ``phase_lag_raw_s`` ships beside ``phase_lag_s``
   precisely so this is checkable on real data too.
3. **The sign convention.** A forecast that is *delayed* relative to the truth returns a
   **positive** lag. Getting this backwards would invert the interpretation of every
   phase-lag row in the results table, and no unit check would catch it.
4. **Amplitude invariance.** Scaling or offsetting either series does not move the peak,
   which is what the normalisation in the estimator is for.
5. **Peak timing.** ``peak_timing_error`` returns positive errors for a late forecast, and
   the half-period tolerance keeps a forecast peak from being matched to a peak in a
   different cycle.

Units: seconds for lags and timing errors, hertz for sampling rates, samples for indices.
"""

import numpy as np
import pytest

from dmf.eval.phase import cross_correlation_lag, peak_timing_error
from dmf.typedefs import FloatArray

#: Sampling rate of every fixture here, hertz. Matches the corpus.
FS_HZ = 10.0

#: Record length, samples. 200 s at 10 Hz -- long enough that the biased correlation's
#: taper is negligible over the lags under test.
N_SAMPLES = 2000

#: Dominant period of the synthetic signal, seconds. Close to the corpus roll period.
PERIOD_S = 12.0


def _narrowband(shift_s: float = 0.0) -> FloatArray:
    """Build a narrowband two-tone signal, optionally shifted forward in time.

    A positive ``shift_s`` produces a series that *lags* the unshifted one: the value at
    time ``t`` is the unshifted value at ``t - shift_s``.

    Args:
        shift_s: Delay applied to the series, **seconds**. May be fractional, which is what
            makes the sub-sample interpolation testable.

    Returns:
        The series, shape ``(N_SAMPLES,)``, dimensionless.
    """
    t = np.arange(N_SAMPLES, dtype=np.float64) / FS_HZ - shift_s
    return np.sin(2.0 * np.pi * t / PERIOD_S) + 0.35 * np.sin(
        2.0 * np.pi * t / (0.6 * PERIOD_S) + 0.7
    )


def test_a_forecast_identical_to_the_truth_has_zero_lag() -> None:
    target = _narrowband()
    assert cross_correlation_lag(target, target, FS_HZ) == pytest.approx(0.0, abs=1e-9)
    assert cross_correlation_lag(target, target, FS_HZ, interpolate=False) == 0.0


@pytest.mark.parametrize("lag_s", [0.1, 0.5, 1.0, 2.0])
def test_a_delayed_forecast_returns_a_positive_lag(lag_s: float) -> None:
    """Sign convention: positive means the forecast is late. This is the load-bearing one."""
    target = _narrowband()
    late = _narrowband(shift_s=lag_s)
    measured = cross_correlation_lag(late, target, FS_HZ)
    assert measured > 0.0
    assert measured == pytest.approx(lag_s, abs=1.0 / FS_HZ)


@pytest.mark.parametrize("lag_s", [-0.1, -0.7, -1.5])
def test_an_early_forecast_returns_a_negative_lag(lag_s: float) -> None:
    target = _narrowband()
    early = _narrowband(shift_s=lag_s)
    measured = cross_correlation_lag(early, target, FS_HZ)
    assert measured < 0.0
    assert measured == pytest.approx(lag_s, abs=1.0 / FS_HZ)


@pytest.mark.parametrize("lag_samples", [1, 3, 7, 15, -4, -11])
def test_the_raw_argmax_recovers_an_integer_sample_lag_exactly(lag_samples: int) -> None:
    target = _narrowband()
    shifted = _narrowband(shift_s=lag_samples / FS_HZ)
    raw = cross_correlation_lag(shifted, target, FS_HZ, interpolate=False)
    assert raw == pytest.approx(lag_samples / FS_HZ, abs=1e-12)


@pytest.mark.parametrize("lag_s", [0.34, 0.67, 1.23, -0.42])
def test_the_raw_argmax_recovers_a_lag_to_within_one_sample(lag_s: float) -> None:
    target = _narrowband()
    shifted = _narrowband(shift_s=lag_s)
    raw = cross_correlation_lag(shifted, target, FS_HZ, interpolate=False)
    assert abs(raw - lag_s) <= 1.0 / FS_HZ


@pytest.mark.parametrize("lag_s", [0.34, 0.67, 1.23, -0.42, 0.05])
def test_parabolic_interpolation_beats_the_raw_argmax_on_a_sub_sample_lag(lag_s: float) -> None:
    """The interpolation must earn the extra column it costs in ``metrics_full.csv``."""
    target = _narrowband()
    shifted = _narrowband(shift_s=lag_s)
    raw = cross_correlation_lag(shifted, target, FS_HZ, interpolate=False)
    fine = cross_correlation_lag(shifted, target, FS_HZ, interpolate=True)
    assert abs(fine - lag_s) < abs(raw - lag_s)
    assert abs(fine - lag_s) < 0.2 / FS_HZ


def test_the_interpolated_lag_stays_within_half_a_sample_of_its_own_argmax() -> None:
    """The auditability property: the refinement can never wander to another peak."""
    target = _narrowband()
    for lag_s in (0.0, 0.13, 0.48, -0.77, 1.61):
        shifted = _narrowband(shift_s=lag_s)
        raw = cross_correlation_lag(shifted, target, FS_HZ, interpolate=False)
        fine = cross_correlation_lag(shifted, target, FS_HZ, interpolate=True)
        assert abs(fine - raw) <= 0.5 / FS_HZ + 1e-12


def test_amplitude_and_offset_do_not_move_the_peak() -> None:
    """What the de-meaning and unit-variance scaling in the estimator are for."""
    target = _narrowband()
    shifted = _narrowband(shift_s=0.6)
    reference = cross_correlation_lag(shifted, target, FS_HZ)
    assert cross_correlation_lag(11.0 * shifted, target, FS_HZ) == pytest.approx(reference)
    assert cross_correlation_lag(shifted + 40.0, target, FS_HZ) == pytest.approx(reference)
    assert cross_correlation_lag(shifted, 0.02 * target - 3.0, FS_HZ) == pytest.approx(reference)


def test_a_constant_or_empty_series_returns_nan_not_zero() -> None:
    """A degenerate channel has no identifiable lag and must not read as perfect timing."""
    target = _narrowband()
    assert np.isnan(cross_correlation_lag(np.zeros(N_SAMPLES), target, FS_HZ))
    assert np.isnan(cross_correlation_lag(target, np.full(N_SAMPLES, 7.0), FS_HZ))
    assert np.isnan(cross_correlation_lag(np.zeros(0), np.zeros(0), FS_HZ))


def test_cross_correlation_lag_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        cross_correlation_lag(np.zeros(10), np.zeros(11), FS_HZ)
    with pytest.raises(ValueError, match="fs_hz must be positive"):
        cross_correlation_lag(np.zeros(10), np.zeros(10), 0.0)


def test_a_persistence_style_forecast_shows_the_lag_it_actually_has() -> None:
    """A forecast that repeats the last observed sample lags the truth by its own horizon.

    This is the pathology the module exists to expose, and the one an RMSE column hides: a
    3 s persistence forecast of a 12 s oscillation is a 3 s-late copy of the truth.
    """
    horizon_s = 3.0
    target = _narrowband()
    shift = int(round(horizon_s * FS_HZ))
    persistence = np.concatenate((np.full(shift, target[0]), target[:-shift]))
    assert cross_correlation_lag(persistence, target, FS_HZ) == pytest.approx(
        horizon_s, abs=1.0 / FS_HZ
    )


# ---------------------------------------------------------------------------
# peak timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag_samples", [1, 4, -3])
def test_peak_timing_error_is_the_injected_shift_and_signed_late_positive(
    lag_samples: int,
) -> None:
    target = _narrowband()
    shifted = _narrowband(shift_s=lag_samples / FS_HZ)
    errors = peak_timing_error(shifted, target, FS_HZ)
    assert errors.size > 5
    assert np.allclose(errors, lag_samples / FS_HZ)
    assert (errors > 0).all() if lag_samples > 0 else (errors < 0).all()


def test_a_perfect_forecast_has_zero_peak_timing_error() -> None:
    target = _narrowband()
    errors = peak_timing_error(target, target, FS_HZ)
    assert errors.size > 5
    assert np.allclose(errors, 0.0)


def test_a_forecast_peak_beyond_the_half_period_is_not_matched() -> None:
    """The tolerance is what keeps a peak from being credited to the wrong cycle."""
    target = _narrowband()
    displaced = _narrowband(shift_s=0.7 * PERIOD_S)
    errors = peak_timing_error(displaced, target, FS_HZ)
    assert np.all(np.abs(errors) <= 0.5 * PERIOD_S + 1.0 / FS_HZ)


def test_peak_timing_error_is_empty_when_the_truth_has_too_few_peaks() -> None:
    ramp = np.linspace(0.0, 1.0, 50)
    assert peak_timing_error(ramp, ramp, FS_HZ).size == 0
    assert peak_timing_error(np.zeros(3), np.zeros(3), FS_HZ).size == 0


def test_peak_timing_error_matches_one_to_one() -> None:
    """Two forecast peaks straddling one true peak cannot both be credited to it."""
    target = np.zeros(60, dtype=np.float64)
    target[20] = 1.0
    target[40] = 1.0
    pred = np.zeros(60, dtype=np.float64)
    pred[19] = 1.0
    pred[21] = 1.0
    pred[41] = 1.0
    errors = peak_timing_error(pred, target, FS_HZ)
    assert errors.size == 2
    assert sorted(np.round(errors, 6).tolist()) == [-0.1, 0.1]


def test_peak_timing_error_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        peak_timing_error(np.zeros(10), np.zeros(11), FS_HZ)
