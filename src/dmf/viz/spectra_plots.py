"""Wave-spectrum and physics-validation figures.

These figures back ``results/physics_validation.md``, the Gate 1 artifact.
"""

from matplotlib.figure import Figure

from dmf.typedefs import FloatArray

__all__ = ["plot_autocorrelation", "plot_response_spectra", "plot_spectrum_overlay"]


def plot_spectrum_overlay(
    w_rad_s: FloatArray,
    s_analytic: FloatArray,
    w_welch_rad_s: FloatArray,
    s_welch: FloatArray,
    hs_m: float,
    tp_s: float,
) -> Figure:
    """Overlay the analytic JONSWAP spectrum with the Welch PSD of a synthesised record.

    Gate 1 requires agreement within 15 percent across [0.4, 1.5] rad/s.

    Args:
        w_rad_s: Frequencies of the analytic spectrum, radians per second.
        s_analytic: Analytic spectral density, m^2 s/rad.
        w_welch_rad_s: Frequencies of the Welch estimate, radians per second.
        s_welch: Welch spectral density estimate, m^2 s/rad.
        hs_m: Requested significant wave height, metres, for the title.
        tp_s: Requested peak period, seconds, for the title.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_autocorrelation(
    lag_s: FloatArray,
    acf: FloatArray,
    synthesis_period_s: float,
) -> Figure:
    """Plot the autocorrelation of an elevation record, marking the synthesis period.

    This is the anti-periodicity figure. A spike at ``2*pi/dw`` means the frequency grid
    was not jittered, the record repeats, and the forecasting task has been reduced to
    memorising a loop. The marker is drawn whether or not a spike is present, so that its
    absence is visible evidence rather than an unstated assumption.

    Args:
        lag_s: Lag axis, seconds.
        acf: Autocorrelation at each lag, dimensionless.
        synthesis_period_s: The period ``2*pi/dw`` at which a uniform grid would repeat,
            seconds.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_response_spectra(
    w_e_rad_s: FloatArray,
    spectra: dict[str, FloatArray],
    wn_rad_s: dict[str, float],
) -> Figure:
    """Plot per-DOF response spectra with their natural frequencies marked.

    Args:
        w_e_rad_s: Encounter frequency axis, radians per second.
        spectra: Response spectral density per DOF name. Units are deg^2 s/rad for roll
            and pitch, m^2 s/rad for heave.
        wn_rad_s: Undamped natural angular frequency per DOF name, radians per second.
            Marked as vertical lines: Gate 1 requires the beam-seas roll spectrum to peak
            within 5 percent of ``wn_roll``.

    Returns:
        The figure.

    Raises:
        ValueError: If ``spectra`` and ``wn_rad_s`` have different key sets.
    """
    raise NotImplementedError
