"""Phase-lag diagnostics.

RMSE hides a specific and important failure: a forecast that reproduces the waveform
correctly but late. On a narrowband oscillation, a lagged forecast can score a respectable
RMSE while being useless for timing a touchdown, because the controller needs to know
*when* the deck is quiet, not merely what its RMS is. Cross-correlation lag exposes that
directly.

The estimator is fixed here and pre-registered as ``docs/protocol.md`` P6-D3:

1. **Normalised.** Both series are de-meaned and divided by their own standard deviation
   before correlating. Scaling a series cannot move an argmax, but a *mean* offset can: at
   ``mode="full"`` the shorter overlap at large ``|lag|`` is zero-padded, so a non-zero mean
   contributes a lag-dependent term. De-meaning removes it, and the unit-variance division
   makes the peak value a correlation coefficient that is readable on its own.
2. **Biased (tapered) rather than overlap-normalised.** The sum at each lag is divided by
   the full sample count ``n``, not by the number of overlapping samples, so correlation
   falls off as ``|lag|`` grows. That taper is wanted: deck motion is narrowband and its
   autocorrelation is quasi-periodic, so a lag of ``d`` and a lag of ``d + T_roll`` are
   nearly equally correlated, and the taper breaks that tie in favour of the replica
   nearest zero. **Consequence, stated rather than hidden: the returned lag is identified
   only modulo the dominant period.** On this corpus that period is ~12 s and the forecast
   lags of interest are ~0.1-2 s, so the ambiguity is not operative -- but it is real, and a
   lag reported near +-6 s should be read as unidentified, not as a measurement.
3. **Parabolic interpolation of the correlation peak**, for sub-sample resolution, with the
   raw argmax available from the same function via ``interpolate=False`` so the
   interpolation is auditable rather than trusted. ``results/e04/metrics_full.csv`` carries
   both as ``phase_lag_s`` and ``phase_lag_raw_s``.

Sign convention, in both functions: **positive means the forecast is late.**
"""

import numpy as np

from dmf.eval.quiescence import match_onsets
from dmf.typedefs import FloatArray

__all__ = ["cross_correlation_lag", "dominant_period", "peak_timing_error"]


def _zscore(x: FloatArray) -> FloatArray | None:
    """De-mean a series and divide by its standard deviation.

    Args:
        x: Series, shape ``(n_samples,)``, in corpus units.

    Returns:
        The standardised series, dimensionless, or None if the series is empty or has
        zero variance, in which case no lag is identifiable.
    """
    if x.size == 0:
        return None
    centred = x - x.mean()
    std = float(np.sqrt(np.mean(np.square(centred))))
    if not std > 0.0:
        return None
    return centred / std


def cross_correlation_lag(
    pred: FloatArray, target: FloatArray, fs_hz: float, *, interpolate: bool = True
) -> float:
    """Find the lag at which forecast and truth are maximally correlated.

    Args:
        pred: Forecast series, shape ``(n_samples,)``, in corpus units.
        target: True series, shape ``(n_samples,)``, in corpus units.
        fs_hz: Sampling rate, hertz.
        interpolate: If True (the default), refine the integer-sample argmax by fitting a
            parabola through the correlation at the peak and its two neighbours, giving
            sub-sample resolution. If False, return the raw argmax lag, which is always an
            integer multiple of ``1 / fs_hz``. Both are reported in the results table --
            an interpolated value that is not bracketed by its own raw argmax is a bug, and
            it is only checkable because both ship.

    Returns:
        Lag at peak cross-correlation, **seconds**. Positive means the forecast lags the
        truth -- the pathology described in the module docstring. A well-behaved forecast
        returns a lag near zero. NaN if either series is empty or constant, since a lag is
        then not identifiable; NaN is returned rather than 0.0 so that a degenerate channel
        cannot be read as perfect timing.

    Raises:
        ValueError: If the two series differ in length or ``fs_hz`` is not positive.
    """
    a = np.asarray(pred, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"pred and target must have the same length, got {a.size} and {b.size}")
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    a_z = _zscore(a)
    b_z = _zscore(b)
    if a_z is None or b_z is None:
        return float("nan")

    n = a_z.size
    corr = np.correlate(a_z, b_z, mode="full") / float(n)
    lags = np.arange(-(n - 1), n, dtype=np.float64)
    peak = int(np.argmax(corr))
    raw_lag = float(lags[peak])
    if not interpolate:
        return raw_lag / fs_hz

    if 0 < peak < corr.size - 1:
        left = float(corr[peak - 1])
        centre = float(corr[peak])
        right = float(corr[peak + 1])
        denominator = left - 2.0 * centre + right
        if denominator < 0.0:
            offset = 0.5 * (left - right) / denominator
            raw_lag += float(np.clip(offset, -0.5, 0.5))
    return raw_lag / fs_hz


def _local_maxima(x: FloatArray) -> FloatArray:
    """Return the indices of the strict interior local maxima of a series.

    Args:
        x: Series, shape ``(n_samples,)``.

    Returns:
        Integer indices ``i`` with ``x[i - 1] < x[i] > x[i + 1]``, ascending, as float64
        so the caller can mix them with sub-sample quantities without a second cast.
    """
    if x.size < 3:
        return np.zeros(0, dtype=np.float64)
    interior = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) + 1
    return interior.astype(np.float64)


def dominant_period(x: FloatArray, fs_hz: float) -> float:
    """Estimate the dominant period of a quasi-periodic series from its peak spacing.

    Written down once and used twice, because two quantities depend on it and they must
    not drift apart: :func:`peak_timing_error`'s matching tolerance is half of it, and
    :mod:`dmf.eval.phase_runner` uses it to decide whether a reported cross-correlation lag
    is *identified* at all -- the module docstring's point 2 is that the lag is identified
    only modulo this quantity.

    The estimator is the median spacing between successive strict interior local maxima,
    which is robust to the occasional double peak a narrowband record shows near a beat
    node in a way that a mean spacing is not.

    Args:
        x: Series, shape ``(n_samples,)``, in corpus units.
        fs_hz: Sampling rate, hertz.

    Returns:
        The dominant period, **seconds**, or NaN when the series has fewer than two peaks
        and no period is estimable.

    Raises:
        ValueError: If ``fs_hz`` is not positive.
    """
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    peaks = _local_maxima(np.asarray(x, dtype=np.float64).reshape(-1))
    if peaks.size < 2:
        return float("nan")
    return float(np.median(np.diff(peaks))) / fs_hz


def peak_timing_error(pred: FloatArray, target: FloatArray, fs_hz: float) -> FloatArray:
    """Measure the timing error of each forecast peak against the nearest true peak.

    Peaks are strict interior local maxima. The matching tolerance is **half the dominant
    period of the true series**, estimated as half the median spacing between successive
    true peaks: a forecast peak further away than that is closer to a different cycle and
    is not the same peak. Matching is the same one-to-one greedy-by-proximity rule
    :func:`dmf.eval.quiescence.match_onsets` applies to window onsets, reused rather than
    reimplemented so that "one predicted event cannot claim two true ones" holds identically
    in both metrics.

    Args:
        pred: Forecast series, shape ``(n_samples,)``, in corpus units.
        target: True series, shape ``(n_samples,)``, in corpus units.
        fs_hz: Sampling rate, hertz.

    Returns:
        Signed timing error per matched peak, **seconds**, shape ``(n_matched_peaks,)``,
        ordered by true-peak time ascending. Positive means the forecast peak arrives late.
        Empty if the true series has fewer than two peaks, since the half-period tolerance
        is then not estimable.

    Raises:
        ValueError: If the two series differ in length.
    """
    a = np.asarray(pred, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"pred and target must have the same length, got {a.size} and {b.size}")
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    pred_peaks = _local_maxima(a).astype(np.int64)
    true_peaks = _local_maxima(b).astype(np.int64)
    if true_peaks.size < 2 or pred_peaks.size == 0:
        return np.zeros(0, dtype=np.float64)

    period_s = dominant_period(b, fs_hz)
    matched_pred, matched_true, _ = match_onsets(
        pred_peaks, true_peaks, fs_hz, tolerance_s=0.5 * period_s
    )
    if matched_pred.size == 0:
        return np.zeros(0, dtype=np.float64)
    return (pred_peaks[matched_pred] - true_peaks[matched_true]).astype(np.float64) / fs_hz
