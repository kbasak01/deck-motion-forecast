"""The committed forecast trace -- the artifact the headline figure is a function of.

The extraction pass needs the corpus and the committed checkpoints and is exercised by
running it; what is asserted here is the round trip, because that is what makes
`make figures` reproducible without either. A trace that silently loses a field, or
reorders its panels, would still render a plausible-looking figure.
"""

from pathlib import Path

import numpy as np
import pytest

from dmf.viz.traces import (
    HEADLINE_ALPHA,
    HEADLINE_CELL,
    HEADLINE_DOF,
    HEADLINE_MODELS,
    ForecastTrace,
    load_traces,
    save_traces,
)

FIELDS = ("t_s", "truth", "median", "lower", "upper", "persistence")


def _trace(model: str, offset: float) -> ForecastTrace:
    """Build a trace whose every field is distinguishable from every other."""
    t = np.linspace(0.0, 5.0, 11)
    return ForecastTrace(
        model=model,
        regime="id",
        seed=0,
        cell="SS5|90.0|12.0|frigate",
        dof="roll",
        horizon_s=3.0,
        t_s=t,
        truth=t + offset,
        median=t + offset + 1.0,
        lower=t + offset - 1.0,
        upper=t + offset + 3.0,
        persistence=t + offset + 7.0,
    )


def test_a_saved_trace_round_trips_field_for_field(tmp_path: Path) -> None:
    """Every field returns with the value it was written with, on every panel."""
    written = [_trace("dlinear_quantile", 0.0), _trace("tcn_quantile", 100.0)]
    path = tmp_path / "trace.npz"
    save_traces(path, written)
    read = load_traces(path)

    assert [t.model for t in read] == [t.model for t in written]
    for before, after in zip(written, read, strict=True):
        assert after.regime == before.regime
        assert after.seed == before.seed
        # The cell, channel and lead time are what the figure's caption is built from, so
        # they have to survive the round trip or the caption stops describing the data.
        assert after.cell == before.cell
        assert after.dof == before.dof
        assert after.horizon_s == before.horizon_s
        for field in FIELDS:
            np.testing.assert_allclose(getattr(after, field), getattr(before, field))


def test_panel_order_is_preserved(tmp_path: Path) -> None:
    """Panels are drawn in the order they were written, so the order must survive.

    The two panels differ only by their label and their values; swapping them would put
    each model's name over the other model's band.
    """
    path = tmp_path / "trace.npz"
    save_traces(path, [_trace("second", 100.0), _trace("first", 0.0)])
    read = load_traces(path)
    assert [t.model for t in read] == ["second", "first"]
    assert read[0].truth[0] == pytest.approx(100.0)


def test_saving_nothing_is_refused(tmp_path: Path) -> None:
    """An empty trace file would render as a blank figure rather than an error."""
    with pytest.raises(ValueError, match="non-empty"):
        save_traces(tmp_path / "trace.npz", [])


def test_the_headline_constants_are_the_cell_the_plan_specifies() -> None:
    """Plan item 2 fixes the channel, the sea state, the heading and the nominal coverage.

    Pinned because the figure is the first thing a reader sees and a silent change of cell
    would be invisible in the rendered image -- the caption is generated from these.
    """
    assert HEADLINE_DOF == "roll"
    assert HEADLINE_CELL[0] == "SS5"
    assert HEADLINE_CELL[1] == 90.0, "beam seas"
    assert pytest.approx(0.1) == HEADLINE_ALPHA, "a 90 % band, as Gate 5 is read on"
    # Two models, so the figure cannot show only the flattering one.
    assert len(HEADLINE_MODELS) == 2


def test_the_trace_carries_the_cell_its_caption_claims() -> None:
    """The committed trace must testify to its own caption.

    The headline figure's title names a sea state, a heading, a speed and a lead time. If
    those live only in module constants, a trace re-extracted at a different cell renders
    under a caption that is silently wrong -- and nothing in the artifact contradicts it.
    """
    from pathlib import Path

    path = Path("results/headline_trace.npz")
    if not path.is_file():
        pytest.skip("committed trace absent; `make figures-extract` writes it")
    traces = load_traces(path)
    assert traces, "the committed trace holds no panels"
    assert {t.cell for t in traces} == {"SS5|90.0|12.0|frigate"}
    assert {t.dof for t in traces} == {"roll"}
    assert {t.horizon_s for t in traces} == {3.0}
