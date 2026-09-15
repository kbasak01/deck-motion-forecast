"""The forecast figures -- Phase 9's two artifacts.

Drawing is checked by looking at it. What is asserted here is the small amount of *logic*
these functions carry, all of which is invisible in the rendered image and wrong-looking in
none of it:

* the length guard, because three series of different lengths silently produce a figure
  where the forecast is plotted against the wrong time axis;
* the measured-coverage annotation, because a band labelled with its nominal coverage alone
  is exactly the claim this project refuses to make (P5-D15);
* the base-rate annotation, because a lead-time distribution without it is not
  interpretable (CLAUDE.md, "base rates in quiescence detection").
"""

import matplotlib
import numpy as np
import pytest

from dmf.viz.forecast_plots import (
    IntervalPanel,
    plot_forecast_vs_truth,
    plot_interval_fan,
    plot_interval_fan_grid,
    plot_lead_time_histogram,
)

matplotlib.use("Agg")

T = np.linspace(0.0, 10.0, 101)
TRUTH = np.sin(T)


def _texts(figure: matplotlib.figure.Figure) -> str:
    """Return every string drawn on the figure: titles, labels, legend and annotations."""
    axes = figure.axes[0]
    parts = [axes.get_title(), axes.get_xlabel(), axes.get_ylabel()]
    parts += [text.get_text() for text in axes.texts]
    legend = axes.get_legend()
    if legend is not None:
        parts += [text.get_text() for text in legend.get_texts()]
    return " | ".join(parts)


def test_forecast_vs_truth_draws_persistence_beside_the_forecast() -> None:
    """Non-negotiable 4 in visual form: the baseline is on the axes, not in the caption."""
    figure = plot_forecast_vs_truth(T, TRUTH, TRUTH * 0.9, TRUTH * 0.5, "roll", "deg", 3.0)
    labels = {line.get_label() for line in figure.axes[0].get_lines()}
    assert "persistence" in labels
    assert "truth" in labels
    assert "forecast" in labels


def test_forecast_vs_truth_refuses_mismatched_series() -> None:
    """A short baseline would be drawn against the wrong times rather than raising."""
    with pytest.raises(ValueError, match="differ in length"):
        plot_forecast_vs_truth(T, TRUTH, TRUTH, TRUTH[:-1], "roll", "deg", 3.0)


def test_interval_fan_annotates_the_coverage_it_measured_not_only_the_nominal() -> None:
    """A band covering half its targets must say so, next to the 90 % it claims."""
    # Half the samples sit outside a band of half-width 0.5 centred on zero.
    truth = np.where(np.arange(T.size) % 2 == 0, 0.0, 5.0)
    median = np.zeros_like(T)
    figure = plot_interval_fan(T, truth, median, median - 0.5, median + 0.5, "roll", "deg", 0.9)
    drawn = _texts(figure)
    assert "90%" in drawn, drawn
    assert "50%" in drawn, drawn


def test_interval_fan_reports_full_coverage_when_the_band_contains_everything() -> None:
    """The measured figure is a hit rate, so a band that never misses reads 100 %."""
    median = np.zeros_like(T)
    figure = plot_interval_fan(T, TRUTH, median, median - 10.0, median + 10.0, "roll", "deg", 0.9)
    assert "100%" in _texts(figure)


def test_interval_fan_refuses_mismatched_series() -> None:
    """Bounds of a different length would silently shorten the band."""
    median = np.zeros_like(T)
    with pytest.raises(ValueError, match="differ in length"):
        plot_interval_fan(T, TRUTH, median, median[:-1], median, "roll", "deg", 0.9)


def test_lead_time_histogram_carries_its_base_rate() -> None:
    """An F1 or a lead-time distribution without its base rate is not a result."""
    figure = plot_lead_time_histogram(np.array([0.5, 1.0, 1.5, 2.0]), "strict", 0.0245)
    drawn = _texts(figure)
    assert "0.0245" in drawn, drawn
    assert "strict" in drawn, drawn


def test_lead_time_histogram_survives_an_empty_cell() -> None:
    """A cell with no scorable onset must render as an empty figure, not raise.

    `not scorable` is a real outcome in this project (P6-D19), and the figure that
    accompanies it still has to carry the base rate that explains why.
    """
    figure = plot_lead_time_histogram(np.array([]), "permissive", 1.0)
    assert "1.0000" in _texts(figure)


def _panel(title: str, lower: float, upper: float) -> IntervalPanel:
    """Build a panel whose band is a constant offset from a zero median."""
    median = np.zeros_like(T)
    return IntervalPanel(
        title=title,
        t_s=T,
        truth=TRUTH,
        median=median,
        lower=median + lower,
        upper=median + upper,
        persistence=TRUTH * 0.5,
    )


def test_the_grid_annotates_each_panel_with_its_own_measured_coverage() -> None:
    """Two models of the same cell must not share one coverage number.

    The headline figure exists to put a well-calibrated band beside a miscalibrated one,
    which only works if each panel carries the coverage it actually achieved.
    """
    figure = plot_interval_fan_grid(
        [_panel("wide", -10.0, 10.0), _panel("narrow", -0.01, 0.01)],
        "roll",
        "deg",
        0.9,
        3.0,
        "SS5, beam seas",
    )
    titles = [axes.get_title() for axes in figure.axes]
    assert any("wide" in t and "100%" in t for t in titles), titles
    assert any("narrow" in t and "0%" in t for t in titles), titles


def test_the_grid_shares_a_y_scale_so_widths_are_comparable_by_eye() -> None:
    """A panel zoomed to its own band would make a wide interval look tight."""
    figure = plot_interval_fan_grid(
        [_panel("wide", -10.0, 10.0), _panel("narrow", -0.01, 0.01)],
        "roll",
        "deg",
        0.9,
        3.0,
        "SS5, beam seas",
    )
    limits = {axes.get_ylim() for axes in figure.axes}
    assert len(limits) == 1, limits


def test_the_grid_draws_persistence_in_every_panel() -> None:
    """Non-negotiable 4 applies per panel, not once per figure."""
    figure = plot_interval_fan_grid(
        [_panel("a", -1.0, 1.0), _panel("b", -1.0, 1.0)], "roll", "deg", 0.9, 3.0, "SS5"
    )
    for axes in figure.axes:
        assert "persistence" in {line.get_label() for line in axes.get_lines()}


def test_the_grid_refuses_an_empty_panel_list() -> None:
    """An empty figure would render as a blank PNG rather than an error."""
    with pytest.raises(ValueError, match="non-empty"):
        plot_interval_fan_grid([], "roll", "deg", 0.9, 3.0, "SS5")


def test_the_grid_refuses_a_panel_whose_series_disagree() -> None:
    """The length guard must apply inside the grid, not only on the single-panel path."""
    median = np.zeros_like(T)
    bad = IntervalPanel("bad", T, TRUTH, median, median[:-1], median, TRUTH)
    with pytest.raises(ValueError, match="differ in length"):
        plot_interval_fan_grid([bad], "roll", "deg", 0.9, 3.0, "SS5")
