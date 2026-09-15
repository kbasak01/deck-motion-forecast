"""Forecast-versus-truth figures.

Every figure here draws a reference the reader would otherwise have to imagine. A forecast
curve on a narrowband signal looks impressive whether or not it beat repeating the last
sample, so :func:`plot_forecast_vs_truth` draws persistence on the same axes; an interval
labelled only with its nominal coverage says nothing about calibration, so
:func:`plot_interval_fan` annotates the coverage actually measured on the span it draws;
and a lead-time distribution is uninterpretable without knowing how often the event occurs,
so :func:`plot_lead_time_histogram` annotates the base rate (CLAUDE.md, "base rates in
quiescence detection").

Units are carried in the signature rather than inferred: angles are **degrees** and heave is
**metres**, matching the corpus, and the caller passes the label. Time axes are **seconds**.

No analysis happens here. Coverage annotated by :func:`plot_interval_fan` is the empirical
hit rate of the three series it was handed, which is a property of the drawn span and is
not the PICP of a results table -- those are computed in :mod:`dmf.eval` and committed to a
CSV. Simulated results only.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dmf.typedefs import FloatArray

__all__ = [
    "IntervalPanel",
    "plot_forecast_vs_truth",
    "plot_interval_fan",
    "plot_interval_fan_grid",
    "plot_lead_time_histogram",
]

#: Colour for the truth trace. Black, and drawn last, so it is never hidden by a forecast.
TRUTH_COLOUR: str = "0.1"

#: Colour for the persistence reference. Grey and dashed: it is a baseline, not a result.
PERSISTENCE_COLOUR: str = "0.55"


def _require_same_length(**series: FloatArray) -> int:
    """Check that every named series is one-dimensional and of one common length.

    Args:
        series: Named one-dimensional arrays.

    Returns:
        The common length.

    Raises:
        ValueError: If any series is not one-dimensional, or if two differ in length.
    """
    lengths = {}
    for name, values in series.items():
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
        lengths[name] = int(array.size)
    if len(set(lengths.values())) > 1:
        raise ValueError(f"series differ in length: {lengths}")
    return next(iter(lengths.values()))


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
    _require_same_length(t_s=t_s, truth=truth, forecast=forecast, persistence=persistence)
    figure, axes = plt.subplots(figsize=(9.0, 3.6))
    axes.plot(
        t_s,
        persistence,
        color=PERSISTENCE_COLOUR,
        linewidth=1.0,
        linestyle="--",
        label="persistence",
    )
    axes.plot(t_s, forecast, color="tab:blue", linewidth=1.4, label="forecast")
    axes.plot(t_s, truth, color=TRUTH_COLOUR, linewidth=1.2, label="truth")
    axes.set_xlabel("time (s)")
    axes.set_ylabel(f"{dof_name} ({units})")
    axes.set_title(f"{dof_name} forecast at {horizon_s:g} s lead -- simulated")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


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
    _require_same_length(t_s=t_s, truth=truth, median=median, lower=lower, upper=upper)
    figure, axes = plt.subplots(figsize=(9.0, 3.6))
    measured = _draw_interval_fan(
        axes, t_s, truth, median, lower, upper, None, coverage, legend=True
    )
    axes.set_xlabel("time (s)")
    axes.set_ylabel(f"{dof_name} ({units})")
    axes.set_title(
        f"{dof_name}: nominal {coverage:.0%} band, measured {measured:.0%} on this span"
        " -- simulated"
    )
    figure.tight_layout()
    return figure


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
    values = np.asarray(lead_times_s, dtype=np.float64).reshape(-1)
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    if values.size:
        # Bin count from the sample size, not a constant. A matched-onset count here is in
        # the low hundreds, and a fixed 40 bins over 146 matches renders sampling noise as
        # if it were structure in the lead-time distribution.
        bins = int(min(40, max(8, round(np.sqrt(values.size)))))
        axes.hist(values, bins=bins, color="tab:blue", alpha=0.75, edgecolor="white", linewidth=0.4)
        median = float(np.median(values))
        axes.axvline(
            median,
            color=TRUTH_COLOUR,
            linewidth=1.2,
            linestyle="--",
            label=f"median {median:.2f} s",
        )
        axes.legend(loc="upper right", fontsize=8)
    axes.set_xlabel("lead time (s) -- positive means flagged before the window opened")
    axes.set_ylabel("matched windows")
    axes.set_title(f"Quiescent-window lead time, {threshold_name} thresholds -- simulated")
    # The base rate goes on the figure, not in the caption. An F1 or a lead-time
    # distribution quoted without it is not interpretable, and captions get separated from
    # figures (CLAUDE.md, "base rates in quiescence detection").
    axes.annotate(
        f"base rate {base_rate_value:.4f}  (n = {values.size})",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7"},
    )
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def _draw_interval_fan(
    axes: Axes,
    t_s: FloatArray,
    truth: FloatArray,
    median: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    persistence: FloatArray | None,
    coverage: float,
    *,
    legend: bool,
) -> float:
    """Draw one interval fan onto existing axes and return its measured coverage.

    Shared by :func:`plot_interval_fan` and :func:`plot_interval_fan_grid` so that a
    single-panel figure and a panel of a grid cannot drift apart in what they draw.

    Args:
        axes: Axes to draw on.
        t_s: Time axis, seconds.
        truth: True trajectory, in corpus units.
        median: Predictive median, in corpus units.
        lower: Lower interval bound, in corpus units.
        upper: Upper interval bound, in corpus units.
        persistence: Persistence forecast over the same span, in corpus units, or None.
        coverage: Nominal coverage, dimensionless in (0, 1).
        legend: Whether to draw a legend on these axes.

    Returns:
        The fraction of ``truth`` inside ``[lower, upper]``, dimensionless. This is the
        hit rate of the drawn span, **not** the PICP of a results table, which is a mean
        over a whole test partition.
    """
    time = np.asarray(t_s, dtype=np.float64)
    truth_a = np.asarray(truth, dtype=np.float64)
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    measured = float(np.mean((truth_a >= lower_a) & (truth_a <= upper_a)))

    axes.fill_between(
        time, lower_a, upper_a, color="tab:blue", alpha=0.22, label=f"{coverage:.0%} interval"
    )
    if persistence is not None:
        # Non-negotiable 4 in visual form. Without it a narrowband forecast looks good
        # whether or not it beat repeating the last sample.
        axes.plot(
            time,
            np.asarray(persistence, dtype=np.float64),
            color=PERSISTENCE_COLOUR,
            linewidth=1.0,
            linestyle="--",
            label="persistence",
        )
    axes.plot(
        time,
        np.asarray(median, dtype=np.float64),
        color="tab:blue",
        linewidth=1.4,
        label="predictive median",
    )
    axes.plot(time, truth_a, color=TRUTH_COLOUR, linewidth=1.2, label="truth")
    # Points the band misses are marked. A band that misses in a burst and one that misses
    # uniformly can share a coverage number and mean very different things operationally.
    missed = (truth_a < lower_a) | (truth_a > upper_a)
    if bool(missed.any()):
        axes.plot(
            time[missed],
            truth_a[missed],
            linestyle="none",
            marker="o",
            markersize=3.0,
            color="tab:red",
            label="outside the band",
        )
    axes.grid(True, alpha=0.3)
    if legend:
        axes.legend(loc="upper right", fontsize=8, ncol=2)
    return measured


@dataclass(frozen=True)
class IntervalPanel:
    """One panel of a multi-panel interval figure.

    Attributes:
        title: Panel heading, naming the model and the regime it was trained on. Both
            belong in the title: two panels of the same cell differing only in training
            regime is the comparison the headline figure exists to make.
        t_s: Time axis, seconds.
        truth: True trajectory, in corpus units.
        median: Predictive median, in corpus units.
        lower: Lower interval bound, in corpus units.
        upper: Upper interval bound, in corpus units.
        persistence: Persistence forecast over the same span, in corpus units.
    """

    title: str
    t_s: FloatArray
    truth: FloatArray
    median: FloatArray
    lower: FloatArray
    upper: FloatArray
    persistence: FloatArray


def plot_interval_fan_grid(
    panels: Sequence[IntervalPanel],
    dof_name: str,
    units: str,
    coverage: float,
    horizon_s: float,
    cell_label: str,
) -> Figure:
    """Plot several interval fans of the same cell as stacked panels.

    Each panel is annotated with the coverage **measured on the span drawn**, which is what
    makes the figure a calibration statement rather than a picture of a band. Panels share
    an x axis and a y scale so that widths are comparable by eye: an interval that looks
    tighter because its panel is zoomed differently is the failure this guards against.

    Args:
        panels: Panels to draw, top to bottom. Must be non-empty.
        dof_name: Channel name, e.g. ``"roll"``.
        units: Unit label, ``"deg"`` or ``"m"``.
        coverage: Nominal coverage of every band, dimensionless in (0, 1).
        horizon_s: Forecast lead time, seconds, for the figure title.
        cell_label: The grid cell being drawn, e.g. ``"SS5, beam seas, 12 kn"``.

    Returns:
        The figure.

    Raises:
        ValueError: If ``panels`` is empty, or if any panel's series differ in length.
    """
    if not panels:
        raise ValueError("panels must be non-empty")
    figure, axes_list = plt.subplots(
        len(panels), 1, figsize=(9.5, 3.0 * len(panels) + 0.8), sharex=True, sharey=True
    )
    axes_seq = np.atleast_1d(axes_list).tolist()
    for index, (panel, axes) in enumerate(zip(panels, axes_seq, strict=True)):
        _require_same_length(
            t_s=panel.t_s,
            truth=panel.truth,
            median=panel.median,
            lower=panel.lower,
            upper=panel.upper,
            persistence=panel.persistence,
        )
        measured = _draw_interval_fan(
            axes,
            panel.t_s,
            panel.truth,
            panel.median,
            panel.lower,
            panel.upper,
            panel.persistence,
            coverage,
            legend=index == 0,
        )
        axes.set_title(
            f"{panel.title} -- nominal {coverage:.0%}, measured {measured:.0%} on this span",
            fontsize=10,
        )
        axes.set_ylabel(f"{dof_name} ({units})")
    axes_seq[-1].set_xlabel("time (s)")
    figure.suptitle(
        f"{dof_name} forecast at {horizon_s:g} s lead -- {cell_label} -- SIMULATED, "
        "no real deck data",
        fontsize=11,
    )
    figure.tight_layout()
    return figure
