"""Plotting.

Figures read from ``results/`` and write to ``results/``. No plotting function defines
analysis logic -- if a number appears in a figure, it was computed in :mod:`dmf.eval` and
written to a CSV first, so that every figure is traceable to a committed table.
"""

from dmf.viz.forecast_plots import (
    IntervalPanel,
    plot_forecast_vs_truth,
    plot_interval_fan,
    plot_interval_fan_grid,
    plot_lead_time_histogram,
)

__all__ = [
    "IntervalPanel",
    "plot_forecast_vs_truth",
    "plot_interval_fan",
    "plot_interval_fan_grid",
    "plot_lead_time_histogram",
]
