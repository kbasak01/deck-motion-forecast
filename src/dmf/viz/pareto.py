"""Accuracy-versus-latency Pareto figures."""

import pandas as pd
from matplotlib.figure import Figure

__all__ = ["plot_latency_accuracy_pareto"]


def plot_latency_accuracy_pareto(
    df: pd.DataFrame,
    skill_col: str = "skill_mean",
    latency_col: str = "p50_ms",
) -> Figure:
    """Plot skill score against latency, marking the Pareto frontier.

    Args:
        df: One row per (model, execution provider, batch size), with the skill and
            latency columns named below plus a ``label`` column.
        skill_col: Column holding skill score vs persistence, dimensionless. Higher is
            better.
        latency_col: Column holding latency, **milliseconds**. Lower is better.

    Returns:
        The figure.

    Raises:
        KeyError: If a named column is absent from ``df``.
    """
    raise NotImplementedError
