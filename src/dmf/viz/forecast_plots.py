"""Forecast-versus-truth figures."""

from matplotlib.figure import Figure

from dmf.typedefs import FloatArray

__all__ = ["plot_forecast_vs_truth", "plot_interval_fan", "plot_lead_time_histogram"]


def plot_forecast_vs_truth(
    t_s: FloatArray,
    truth: FloatArray,
    forecast: FloatArray,
    persistence: FloatArray,
    dof_name: str,
    units: str,
    horizon_s: float,
) -> Figure:
    """Plot a forecast against truth and the persistence baseline.

    Persistence is drawn on the same axes as a matter of policy: a forecast curve shown
    alone looks impressive on a narrowband signal regardless of whether it beat repeating
    the last sample.

    Args:
        t_s: Time axis, seconds.
        truth: True trajectory, in corpus units.
        forecast: Model forecast over the same span, in corpus units.
        persistence: Persistence forecast over the same span, in corpus units.
        dof_name: Channel name for the axis label, e.g. ``"roll"``.
        units: Unit label for the y-axis, ``"deg"`` or ``"m"``.
        horizon_s: Forecast horizon, seconds, for the title.

    Returns:
        The figure.

    Raises:
        ValueError: If the series differ in length.
    """
    raise NotImplementedError


def plot_interval_fan(
    t_s: FloatArray,
    truth: FloatArray,
    median: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    dof_name: str,
    units: str,
    coverage: float,
) -> Figure:
    """Plot a predictive median with its interval band against truth.

    Args:
        t_s: Time axis, seconds.
        truth: True trajectory, in corpus units.
        median: Predictive median, in corpus units.
        lower: Lower interval bound, in corpus units.
        upper: Upper interval bound, in corpus units.
        dof_name: Channel name, e.g. ``"roll"``.
        units: Unit label, ``"deg"`` or ``"m"``.
        coverage: Nominal coverage of the band, dimensionless in (0, 1), e.g. 0.9. The
            **measured** PICP is annotated alongside it, since a band labelled only with
            its nominal coverage says nothing about whether it is calibrated.

    Returns:
        The figure.

    Raises:
        ValueError: If the series differ in length.
    """
    raise NotImplementedError


def plot_lead_time_histogram(
    lead_times_s: FloatArray,
    threshold_name: str,
    base_rate_value: float,
) -> Figure:
    """Plot the distribution of quiescent-window lead times.

    Args:
        lead_times_s: Lead time per matched window, seconds.
        threshold_name: Threshold set label, ``"permissive"`` or ``"strict"``.
        base_rate_value: Fraction of time the deck is genuinely quiescent, dimensionless.
            Annotated on the figure, because a lead-time distribution is not interpretable
            without knowing how often the event occurs at all.

    Returns:
        The figure.
    """
    raise NotImplementedError
