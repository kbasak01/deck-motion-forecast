"""The accuracy-versus-latency Pareto figure -- Gate 7's second artifact.

The figure itself cannot be asserted, but the one piece of *logic* behind it can: which
rows are on the frontier. A wrong frontier is the failure mode that matters here, because
it is invisible -- the plot still looks like a Pareto plot, and the reader takes the marked
points to be the ones worth deploying.

Everything else in `dmf.viz.pareto` is drawing, and drawing is checked by looking at it.
"""

import matplotlib
import pandas as pd
import pytest

from dmf.viz.pareto import pareto_frontier, plot_latency_accuracy_pareto

matplotlib.use("Agg")

#: Four configurations with a known answer. `slow-and-bad` is dominated by every other row;
#: `fast-and-good` dominates it on both axes. The middle two trade off against each other
#: and are both on the frontier, which is the case a naive "best skill" or "lowest latency"
#: rule gets wrong.
FRAME = pd.DataFrame(
    {
        "label": ["fast-and-good", "fast-and-bad", "slow-and-good", "slow-and-bad"],
        "p50_ms": [0.5, 0.4, 3.0, 4.0],
        "skill_mean": [0.80, 0.60, 0.90, 0.55],
    }
)


def test_the_frontier_is_the_set_of_rows_nothing_beats_on_both_axes() -> None:
    """Dominated rows are excluded; rows that trade accuracy against latency are kept."""
    frontier = pareto_frontier(FRAME, "skill_mean", "p50_ms")
    on = set(FRAME.loc[frontier, "label"])
    assert on == {"fast-and-bad", "fast-and-good", "slow-and-good"}
    # `slow-and-bad` is slower AND less accurate than `fast-and-good`, so it cannot be on it.
    assert "slow-and-bad" not in on


def test_identical_configurations_do_not_eliminate_each_other() -> None:
    """A tie on both axes leaves both rows on the frontier, not neither.

    Strict domination is the criterion. With a non-strict one, two rows measuring the same
    configuration twice -- which is exactly what the required re-run produces -- would
    remove each other and the frontier would come back empty.
    """
    tied = pd.DataFrame(
        {"label": ["a", "b"], "p50_ms": [1.0, 1.0], "skill_mean": [0.7, 0.7]},
    )
    assert pareto_frontier(tied, "skill_mean", "p50_ms").all()


def test_a_missing_column_raises_rather_than_plotting_something_else() -> None:
    """The skill, latency and label columns are all required by name."""
    with pytest.raises(KeyError, match="skill_mean"):
        plot_latency_accuracy_pareto(FRAME.drop(columns=["skill_mean"]))
    with pytest.raises(KeyError, match="label"):
        plot_latency_accuracy_pareto(FRAME.drop(columns=["label"]))


def test_the_figure_carries_one_point_per_configuration_on_a_log_latency_axis() -> None:
    """Every row is drawn exactly once, split across the dominated and frontier series."""
    figure = plot_latency_accuracy_pareto(FRAME)
    axes = figure.axes[0]
    assert axes.get_xscale() == "log"
    drawn = sum(collection.get_offsets().shape[0] for collection in axes.collections)
    assert drawn == len(FRAME)


def test_only_the_frontier_is_annotated() -> None:
    """Labels are spent on the points a reader is choosing between, not on all of them.

    The first version of the Phase 7 figure annotated all 28 configurations and produced an
    unreadable pile: the accuracy axis takes one value per model, so every backend of a
    model lands on the same horizontal line. This asserts the fix rather than leaving it to
    whoever next opens the PNG.
    """
    figure = plot_latency_accuracy_pareto(FRAME)
    annotated = {text.get_text() for text in figure.axes[0].texts}
    frontier = set(FRAME.loc[pareto_frontier(FRAME, "skill_mean", "p50_ms"), "label"])
    assert frontier <= annotated
    assert "slow-and-bad" not in annotated


def test_configurations_are_coloured_by_backend_when_the_column_is_there() -> None:
    """One series per backend, so a CPU point and a CUDA point are never the same colour.

    `torch-eager` is measured on both devices in the Phase 7 sweep, and the caller folds the
    device into this column for exactly that reason.
    """
    grouped = FRAME.assign(backend=["ort-cpu", "ort-cpu", "ort-trt", "ort-trt"])
    axes = plot_latency_accuracy_pareto(grouped).axes[0]
    legend = axes.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"ort-cpu", "ort-trt"}
