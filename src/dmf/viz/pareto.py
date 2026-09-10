"""Accuracy-versus-latency Pareto figures."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

__all__ = ["PARETO_REQUIRED_COLUMNS", "pareto_frontier", "plot_latency_accuracy_pareto"]

#: Columns every Pareto frame must carry beyond the two named at the call site. ``label``
#: is required rather than optional because an unannotated Pareto plot is unreadable: the
#: entire point is which *named* configuration is on the frontier.
PARETO_REQUIRED_COLUMNS: tuple[str, ...] = ("label",)


def pareto_frontier(df: pd.DataFrame, skill_col: str, latency_col: str) -> pd.Series:
    """Mark the rows that no other row beats on both axes.

    A row is dominated when some other row is at least as accurate **and** at least as fast,
    and strictly better on one of the two. Ties are both kept, so two identical
    configurations do not eliminate each other.

    Args:
        df: One row per configuration.
        skill_col: Column holding skill score vs persistence, dimensionless. Higher is
            better.
        latency_col: Column holding latency, milliseconds. Lower is better.

    Returns:
        Boolean series, index-aligned to ``df``, True where the row is non-dominated.
    """
    skill = df[skill_col].to_numpy()
    latency = df[latency_col].to_numpy()
    on_frontier = []
    for value, cost in zip(skill, latency, strict=True):
        better_or_equal = (skill >= value) & (latency <= cost)
        strictly_better = better_or_equal & ((skill > value) | (latency < cost))
        on_frontier.append(not bool(strictly_better.any()))
    return pd.Series(on_frontier, index=df.index, name="on_frontier")


def plot_latency_accuracy_pareto(
    df: pd.DataFrame,
    skill_col: str = "skill_mean",
    latency_col: str = "p50_ms",
    group_col: str = "backend",
) -> Figure:
    """Plot skill score against latency, marking the Pareto frontier.

    Latency is on a log axis: the configurations span roughly two orders of magnitude, and
    on a linear axis every CPU row collapses onto the origin, which is exactly the region
    the deployment argument turns on.

    **Only the frontier is annotated.** The first version of this figure labelled all 28
    points and produced an unreadable pile: the accuracy axis takes one value per model, so
    every backend of a model lands on the same horizontal line. Configurations are
    distinguished by colour from ``group_col`` with a legend, the model naming each row is
    written once at the left margin, and the text labels are spent on the points a reader
    is actually choosing between.

    Args:
        df: One row per (model, execution provider, batch size), with the skill and
            latency columns named below plus a ``label`` column.
        skill_col: Column holding skill score vs persistence, dimensionless. Higher is
            better.
        latency_col: Column holding latency, **milliseconds**. Lower is better.
        group_col: Column to colour by, if present. Absent, every point is drawn in one
            colour; it is not required, so a caller with a bare (label, skill, latency)
            frame still gets a figure.

    Returns:
        The figure.

    Raises:
        KeyError: If a named column is absent from ``df``.
    """
    missing = [
        column
        for column in (skill_col, latency_col, *PARETO_REQUIRED_COLUMNS)
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"missing columns {missing}; frame carries {list(df.columns)}")

    frontier = pareto_frontier(df, skill_col, latency_col)
    figure, axes = plt.subplots(figsize=(9.0, 5.5))
    groups = (
        sorted(set(df[group_col].astype(str))) if group_col in df.columns else ["configuration"]
    )
    colours = plt.get_cmap("tab10")
    for index, group in enumerate(groups):
        mask = (
            df[group_col].astype(str) == group
            if group_col in df.columns
            else pd.Series(True, index=df.index)
        )
        axes.scatter(
            df.loc[mask & ~frontier, latency_col],
            df.loc[mask & ~frontier, skill_col],
            marker="o",
            facecolors="none",
            edgecolors=colours(index),
            label=group,
        )
        axes.scatter(
            df.loc[mask & frontier, latency_col],
            df.loc[mask & frontier, skill_col],
            marker="o",
            s=70,
            color=colours(index),
        )
    # Headroom for the frontier labels, which alternate above and below their points: the
    # lowest one is drawn downward and without this lands on top of the x tick labels. Set
    # before the twin axis below copies these limits, or the two axes disagree.
    axes.margins(y=0.16)
    ordered = df.loc[frontier].sort_values(latency_col)
    axes.plot(ordered[latency_col], ordered[skill_col], color="0.3", linewidth=1.0, zorder=0)
    # Frontier labels alternate above and below the line. Adjacent frontier points can sit
    # a few hundredths of a millisecond apart on a log axis -- `tcn` and `tcn_quantile` land
    # at 0.501 and 0.514 ms -- and a constant offset overprints one label on the other.
    for position, (_, row) in enumerate(ordered.iterrows()):
        above = position % 2 == 1
        axes.annotate(
            str(row["label"]),
            (row[latency_col], row[skill_col]),
            textcoords="offset points",
            xytext=(9, 7 if above else -14),
            fontsize=8,
            fontweight="bold",
            va="bottom" if above else "top",
        )
    if "model" in df.columns:
        # Model names go on a right-hand axis rather than inside the plot. Every backend of
        # a model shares one skill value, so each model is a horizontal row of points and a
        # tick names it exactly once -- an in-plot annotation collided with both the data
        # and the frontier labels, which is what made the first version unreadable.
        rows_by_skill = df.groupby("model")[skill_col].first().sort_values()
        names = axes.twinx()
        names.set_ylim(axes.get_ylim())
        names.set_yticks(list(rows_by_skill.to_numpy()))
        names.set_yticklabels([str(name) for name in rows_by_skill.index], fontsize=8, color="0.35")
        names.tick_params(axis="y", length=0)
        for spine in names.spines.values():
            spine.set_visible(False)
    axes.set_xscale("log")
    axes.set_xlabel("latency p50 (ms, lower is better)")
    axes.set_ylabel("skill score vs persistence (higher is better)")
    axes.set_title("Accuracy versus latency -- simulated corpus, measured on one machine")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="lower right", fontsize=8, title="backend")
    figure.tight_layout()
    return figure
