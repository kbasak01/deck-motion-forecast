"""Phase-lag diagnostics.

RMSE hides a specific and important failure: a forecast that reproduces the waveform
correctly but late. On a narrowband oscillation, a lagged forecast can score a respectable
RMSE while being useless for timing a touchdown, because the controller needs to know
*when* the deck is quiet, not merely what its RMS is. Cross-correlation lag exposes that
directly.
"""

from dmf.typedefs import FloatArray

__all__ = ["cross_correlation_lag", "peak_timing_error"]


def cross_correlation_lag(pred: FloatArray, target: FloatArray, fs_hz: float) -> float:
    """Find the lag at which forecast and truth are maximally correlated.

    Args:
        pred: Forecast series, shape ``(n_samples,)``, in corpus units.
        target: True series, shape ``(n_samples,)``, in corpus units.
        fs_hz: Sampling rate, hertz.

    Returns:
        Lag at peak cross-correlation, **seconds**. Positive means the forecast lags the
        truth -- the pathology described in the module docstring. A well-behaved forecast
        returns a lag near zero.

    Raises:
        ValueError: If the two series differ in length or ``fs_hz`` is not positive.
    """
    raise NotImplementedError


def peak_timing_error(pred: FloatArray, target: FloatArray, fs_hz: float) -> FloatArray:
    """Measure the timing error of each forecast peak against the nearest true peak.

    Args:
        pred: Forecast series, shape ``(n_samples,)``, in corpus units.
        target: True series, shape ``(n_samples,)``, in corpus units.
        fs_hz: Sampling rate, hertz.

    Returns:
        Signed timing error per matched peak, **seconds**, shape ``(n_matched_peaks,)``.
        Positive means the forecast peak arrives late.

    Raises:
        ValueError: If the two series differ in length.
    """
    raise NotImplementedError
