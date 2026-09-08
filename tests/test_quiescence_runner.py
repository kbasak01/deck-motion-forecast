"""The quiescent-window detector's geometry, tested against hand-built trajectories.

:mod:`dmf.eval.quiescence` is unit-tested in ``tests/test_quiescence.py``; this module
tests the thing that stands between those definitions and a results row -- *where the
detector stands in time*. Every case here is built on a fabricated 100-sample corpus whose
quiescent runs were placed by hand, so the expected onset, the expected earliest flag and
the expected lead time are arithmetic that can be done on paper and are written into the
assertions as literals rather than recomputed by a second copy of the implementation.

What is pinned, and why each case exists (``docs/protocol.md`` P6-D2, P6-D5, P6-D7):

1. **The fast run finder matches the reference detector.** ``sustained_runs`` is a batched
   rewrite of the run-length filter inside
   :func:`dmf.eval.quiescence.detect_quiescent_mask`; it exists only for speed, and its
   docstring claims row-by-row agreement with the definition. That claim is checked on
   random masks rather than trusted.
2. **The earliest-flag dedupe.** Three decision times see one absolute onset; the table
   must report **one** predicted onset, flagged at the earliest of the three, because that
   earliest flag is what the lead time measures.
3. **Lead time = true onset - earliest flag**, in seconds, asserted as a literal.
4. **The +-0.5 s matching tolerance is inclusive**: a predicted onset exactly 5 samples
   from the truth matches at 10 Hz, and 6 samples does not.
5. **False alarms per minute** is unmatched predicted onsets over the *evaluated* duration,
   which starts at the first decision time and not at sample 0.
6. **The not-scorable cell (P6-D7).** A cell whose truth never leaves the limits has zero
   interior onsets. It must ship ``scorable = False`` with NaN precision/recall/F1 -- never
   0.0, which reads as a model failure, and never 1.0, which reads as a success -- while
   still carrying the base rate that says why.
7. **The interval rule is two-sided (P6-D5).** ``max(|q05|, |q95|) <= limit``, not the
   plan's one-sided 0.05 quantile. The fixture is built so the two readings disagree: its
   ``q05`` is far below ``-limit`` while satisfying ``q05 <= limit``, so a one-sided
   implementation would flag a landing window that the deck may be 5 degrees outside.
8. **The degenerate detector and the constant forecasters**, which are what make every
   other F1 in the table readable.

Units follow the package convention: degrees for roll and pitch, metres per second for
heave rate, seconds for durations and lead times, hertz for sampling rates, samples for
indices.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import NormStats, build_norm_stats
from dmf.data.splits import RealizationKey, Split
from dmf.data.windows import window_spec_from_config
from dmf.eval.quiescence import (
    PERMISSIVE,
    QuiescenceThresholds,
    detect_quiescent_mask,
    precision_recall_f1,
    sustain_samples,
)
from dmf.eval.quiescence_runner import (
    ALWAYS_QUIESCENT,
    CELL_LEVEL,
    QUIESCENCE_COLUMNS,
    RATE_MATCHED,
    SEA_STATE_LEVEL,
    bootstrap_f1_ci,
    decision_channel_index,
    evaluate_quiescence,
    rate_matched_onsets,
    sustained_runs,
)
from dmf.sim.generate import MANIFEST_NAME, RealizationSpec, realization_path
from dmf.typedefs import BoolArray, FloatArray

#: Sampling rate of every fixture here, hertz. Matches the corpus.
FS_HZ = 10.0

#: Record length of a fabricated realization, samples. Ten seconds at 10 Hz.
N_SAMPLES = 100

#: Window geometry of the fabricated corpus: 1 s lookback, 2 s horizon, 0.5 s stride.
#: Deliberately the production *shape* at a tenth of the production size, so the arithmetic
#: below can be done by hand: window starts 0, 5, .. 70; decision times 9, 14, .. 79; the
#: forecast at decision time ``t`` covers absolute samples ``t+1 .. t+20``.
LOOKBACK = 10
HORIZONS: tuple[int, ...] = (5, 10, 20)
MAX_HORIZON = 20
STRIDE = 5
N_WINDOWS = 15
FIRST_DECISION = LOOKBACK - 1
LAST_COVERED = 70 + LOOKBACK + MAX_HORIZON

#: The six motion channels the fabricated corpus stores, in canonical order.
CHANNELS: tuple[str, ...] = (
    "roll",
    "pitch",
    "heave",
    "roll_rate",
    "pitch_rate",
    "heave_rate",
)

#: Probe limits: 1 deg, 1 deg, 1 m/s, sustained 0.5 s = 5 samples. Round numbers, so an
#: in-limit sample (0.0) and an out-of-limit one (10.0) are nowhere near the comparison
#: boundary -- that boundary is covered in ``tests/test_quiescence.py``.
PROBE = QuiescenceThresholds(1.0, 1.0, 1.0, 0.5, "probe")

#: The hand-placed quiescent run of the ``SS5`` realization: samples 40..59 inclusive.
ONSET_SAMPLE = 40
RUN_STOP = 60

#: Value written where the deck is inside the probe limits, and where it is outside.
INSIDE = 0.0
OUTSIDE = 10.0

#: The realization keys of the fabricated corpus.
KEY_ONSET: RealizationKey = ("SS5", 45.0, 0.0, "frigate", 0)
KEY_QUIET: RealizationKey = ("SS3", 45.0, 0.0, "frigate", 0)

#: A second ``SS5`` **cell** -- same sea state, different heading -- and one that has nothing
#: to detect. It exists so that the cell rows and the sea-state roll-up are different rows:
#: with one cell per sea state the two levels coincide and the distinction P6-D19 forced
#: could not be tested at all.
KEY_ONSET_BEAM: RealizationKey = ("SS5", 90.0, 0.0, "frigate", 0)


def _trajectory(quiescent: BoolArray) -> FloatArray:
    """Build a six-channel record that is inside the probe limits exactly where told.

    Args:
        quiescent: Per-sample flag, shape ``(N_SAMPLES,)``.

    Returns:
        Array of shape ``(N_SAMPLES, 6)`` in corpus units, columns :data:`CHANNELS`.
    """
    inside = np.where(quiescent, INSIDE, OUTSIDE).astype(np.float64)
    block = np.zeros((quiescent.size, len(CHANNELS)), dtype=np.float64)
    for name in ("roll", "pitch", "heave_rate"):
        block[:, CHANNELS.index(name)] = inside
    return block


def _write_realization(root: Path, key: RealizationKey, block: FloatArray) -> None:
    """Write one fabricated realization to the corpus layout.

    Args:
        root: Corpus root.
        key: ``(ss, heading_deg, speed_kn, vessel, seed)``.
        block: Record of shape ``(n_samples, 6)``, columns :data:`CHANNELS`.
    """
    ss, heading, speed, vessel, seed = key
    path = root / realization_path(
        RealizationSpec(seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({name: block[:, i] for i, name in enumerate(CHANNELS)})
    frame.to_parquet(path)


@pytest.fixture(scope="module")
def fabricated_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the two hand-built realizations this module reasons about.

    ``SS5`` carries exactly one interior quiescent run, samples 40..59, so it has one
    scorable onset. ``SS3`` is inside the limits for its whole record, so its only run
    begins at sample 0 and ends at the record end: it has **zero** scorable onsets and is
    the P6-D7 not-scorable cell in miniature.

    Returns:
        The corpus root.
    """
    root = tmp_path_factory.mktemp("fabricated_corpus") / "corpus"
    pattern = np.zeros(N_SAMPLES, dtype=bool)
    pattern[ONSET_SAMPLE:RUN_STOP] = True
    _write_realization(root, KEY_ONSET, _trajectory(pattern))
    _write_realization(root, KEY_QUIET, _trajectory(np.ones(N_SAMPLES, dtype=bool)))
    # The beam cell of SS5 is quiet throughout, so it has no scorable onset while the
    # 45 deg cell of the same sea state has one. That is the P6-D19 shape in miniature:
    # scorability is a property of the cell, and a sea-state row averages the two.
    _write_realization(root, KEY_ONSET_BEAM, _trajectory(np.ones(N_SAMPLES, dtype=bool)))
    # A manifest is written so the root is a well-formed corpus even though nothing in
    # this module reads it: the dataset reads the manifest only under sea-state
    # conditioning, and a half-built corpus root is the kind of fixture that misleads the
    # next reader.
    pd.DataFrame(
        [
            {"ss": key[0], "heading": key[1], "speed": key[2], "vessel": key[3], "seed": key[4]}
            for key in (KEY_ONSET, KEY_QUIET, KEY_ONSET_BEAM)
        ]
    ).to_parquet(root / MANIFEST_NAME)
    return root


def _data_cfg(**overrides: object) -> DataConfig:
    """Return the fabricated corpus's task definition.

    Args:
        **overrides: Fields to replace, e.g. ``target_dofs``.

    Returns:
        The configuration.
    """
    cfg = DataConfig(
        fs_hz=FS_HZ,
        lookback=LOOKBACK,
        horizons=HORIZONS,
        target_dofs=CHANNELS,
        input_channels=CHANNELS,
        stride=STRIDE,
        observation_mode="ideal",
        revin=False,
        condition_on_sea_state=False,
    )
    return replace(cfg, **overrides)  # type: ignore[arg-type]


def _unit_stats(channels: tuple[str, ...]) -> NormStats:
    """Return unit-scale training statistics over ``channels``.

    A scale of exactly 1.0 makes ``invert_norm`` an addition of the window mean and
    nothing else, so a prescribed forecast in corpus units survives the round trip
    bit-exactly and the assertions below are about the detector rather than about float
    error.

    Args:
        channels: Motion channel names, in dataset order.

    Returns:
        Statistics labelled ``"id/train"``.
    """
    return build_norm_stats(
        scale=np.ones(len(channels), dtype=np.float64),
        channels=channels,
        fitted_on="id/train",
        n_realizations=1,
    )


def _dataset(
    root: Path, keys: tuple[RealizationKey, ...], cfg: DataConfig | None = None
) -> DeckMotionDataset:
    """Build a test partition over the fabricated corpus.

    Args:
        root: Corpus root.
        keys: Realizations to include, in any order (the dataset sorts them).
        cfg: Task definition; :func:`_data_cfg` by default.

    Returns:
        The dataset, scaled by unit statistics labelled ``"id/train"``.
    """
    config = cfg if cfg is not None else _data_cfg()
    split = Split(
        regime="id",
        train_keys=frozenset(keys),
        val_keys=frozenset(),
        test_keys=frozenset(keys),
    )
    columns = tuple(config.input_channels)
    return DeckMotionDataset(
        root, split, "test", config, window_spec_from_config(config), stats=_unit_stats(columns)
    )


class _Prescribed(nn.Module):
    """A forecaster handed its whole answer, window by window, in dataset order.

    The point of the fixture is that no model is under test here. What is under test is
    what the runner does with a forecast whose onsets are known exactly, so the forecast is
    written down rather than produced.
    """

    def __init__(self, answer: Tensor) -> None:
        """Store the normalised answer.

        Args:
            answer: Model-space output for every window, ``(N, H, C)`` or ``(N, H, C, K)``.
        """
        super().__init__()
        self.register_buffer("answer", answer)
        self._cursor = 0

    def forward(self, x: Tensor) -> Tensor:
        """Return the next ``len(x)`` rows of the stored answer.

        Args:
            x: Input window batch, used only for its length.

        Returns:
            The answer rows for this batch.
        """
        answer: Tensor = self.answer
        batch = int(x.shape[0])
        start = self._cursor
        self._cursor = (start + batch) % int(answer.shape[0])
        return answer[start : start + batch]


def _to_model_space(dataset: DeckMotionDataset, values: FloatArray) -> Tensor:
    """Convert a prescribed corpus-unit forecast into the model's normalised space.

    Args:
        dataset: The partition the forecast is for.
        values: Corpus-unit forecast, ``(N, H, C)`` or ``(N, H, C, K)``.

    Returns:
        The same forecast in model space, so that the runner's ``invert_norm`` returns
        ``values`` unchanged.
    """
    mean = torch.stack([dataset[i][2] for i in range(len(dataset))]).double()
    scale = torch.as_tensor(
        dataset.norm_stats.subset(dataset.target_columns).scale, dtype=torch.float64
    )
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 4:
        return ((tensor - mean[..., None]) / scale[:, None]).float()
    return ((tensor - mean) / scale).float()


def _prescribe(dataset: DeckMotionDataset, values: FloatArray) -> _Prescribed:
    """Build a point forecaster that emits ``values`` in corpus units.

    Args:
        dataset: The partition the forecast is for.
        values: Corpus-unit forecast, shape ``(N, H, C)``.

    Returns:
        The forecaster.
    """
    return _Prescribed(_to_model_space(dataset, values))


def _constant_forecast(dataset: DeckMotionDataset, value: float) -> FloatArray:
    """Return a corpus-unit forecast array filled with one value.

    Args:
        dataset: The partition, for its shape.
        value: The value, corpus units.

    Returns:
        Array of shape ``(N, H, C)``.
    """
    shape = (len(dataset), dataset.window_spec.max_horizon, len(dataset.target_columns))
    return np.full(shape, value, dtype=np.float64)


def _oracle_forecast(dataset: DeckMotionDataset) -> FloatArray:
    """Return the true future of every window, in corpus units and dataset order.

    Args:
        dataset: The partition.

    Returns:
        Array of shape ``(N, H, C)``.
    """
    return np.stack([dataset[i][1].numpy().astype(np.float64) for i in range(len(dataset))])


def _decision_times(dataset: DeckMotionDataset) -> list[int]:
    """Return the absolute decision time of every window, in dataset order.

    Args:
        dataset: The partition.

    Returns:
        One decision time per window, ``start + lookback - 1``.
    """
    lookback = dataset.window_spec.lookback
    return [dataset.describe_window(i)[1] + lookback - 1 for i in range(len(dataset))]


def _row(table: pd.DataFrame, **where: object) -> pd.Series:
    """Return the single row matching ``where``.

    Defaults to the **cell** level. Since P6-D19 the summary carries two rows per reported
    quantity -- one per ``(sea state, heading, speed)`` grid cell and one sea-state roll-up
    over them -- so a selector that names only the sea state is ambiguous, and these
    fixtures put one cell in each sea state, which is the level their assertions are about.
    A test that wants the roll-up passes ``group_level=SEA_STATE_LEVEL`` explicitly.

    Args:
        table: A quiescence summary table.
        **where: Column/value pairs identifying one row.

    Returns:
        That row.
    """
    where.setdefault("group_level", CELL_LEVEL)
    mask = np.ones(len(table), dtype=bool)
    for column, value in where.items():
        mask &= (table[column] == value).to_numpy()
    selected = table.loc[mask]
    assert len(selected) == 1, f"expected exactly one row for {where}, got {len(selected)}"
    return selected.iloc[0]


def _score(
    dataset: DeckMotionDataset,
    models: dict[str, nn.Module],
    *,
    include_always_quiescent: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detector over one partition with the probe thresholds.

    Args:
        dataset: The partition.
        models: Detectors to score.
        include_always_quiescent: Whether to add the degenerate detector.

    Returns:
        Tuple ``(summary, lead_times)``.
    """
    return evaluate_quiescence(
        models,
        dataset,
        fs_hz=FS_HZ,
        threshold_sets=(PROBE,),
        include_always_quiescent=include_always_quiescent,
        batch_size=4096,
    )


# ---------------------------------------------------------------------------
# The batched run finder against the reference definition
# ---------------------------------------------------------------------------


def test_sustained_runs_matches_the_reference_detector_row_by_row(
    rng: np.random.Generator,
) -> None:
    flags = rng.random((32, 40)) < 0.55
    min_run = 5
    rows, starts, stops = sustained_runs(flags, min_run)
    for row in range(flags.shape[0]):
        # detect_quiescent_mask applies the same run-length filter to a single row, via a
        # different code path: thresholds on three series rather than a precomputed mask.
        series = np.where(flags[row], INSIDE, OUTSIDE)
        expected = detect_quiescent_mask(
            series, series, series, replace(PROBE, sustain_s=min_run / FS_HZ), FS_HZ
        )
        got = np.zeros(flags.shape[1], dtype=bool)
        for start, stop in zip(starts[rows == row], stops[rows == row], strict=True):
            got[start:stop] = True
        assert np.array_equal(got, expected)


def test_sustained_runs_returns_runs_in_row_then_start_order(rng: np.random.Generator) -> None:
    flags = rng.random((8, 30)) < 0.6
    rows, starts, _ = sustained_runs(flags, 3)
    order = np.lexsort((starts, rows))
    assert np.array_equal(order, np.arange(rows.size))


def test_sustained_runs_rejects_a_non_positive_minimum() -> None:
    with pytest.raises(ValueError, match="min_run must be positive"):
        sustained_runs(np.ones((2, 4), dtype=bool), 0)


def test_sustained_runs_rejects_a_one_dimensional_mask() -> None:
    with pytest.raises(ValueError, match="flags must have shape"):
        sustained_runs(np.ones(4, dtype=bool), 1)


# ---------------------------------------------------------------------------
# Geometry: dedupe, matching, lead time, false alarms
# ---------------------------------------------------------------------------


def test_the_hand_built_geometry_is_what_the_runner_sees(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    assert len(dataset) == N_WINDOWS
    assert _decision_times(dataset) == list(range(FIRST_DECISION, 80, STRIDE))
    assert sustain_samples(PROBE.sustain_s, FS_HZ) == 5


def test_only_the_earliest_decision_time_that_predicts_an_onset_is_kept(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    forecast = _oracle_forecast(dataset)
    # Hand arithmetic. The run occupies absolute samples 40..59. A decision time t sees it
    # at forecast index 39 - t, which must be > 0 (index 0 means "already quiescent"), and
    # the part of the run inside the 20-sample horizon must be at least 5 samples long:
    #   39 - t > 0            -> t <= 38
    #   min(60, t + 21) - 40 >= 5 -> t >= 24
    # The decision times are 9, 14, ... 79, so exactly {24, 29, 34} witness this onset.
    witnesses = [
        t
        for t in _decision_times(dataset)
        if ONSET_SAMPLE - t - 1 > 0 and min(RUN_STOP, t + 1 + MAX_HORIZON) - ONSET_SAMPLE >= 5
    ]
    assert witnesses == [24, 29, 34]

    summary, leads = _score(dataset, {"oracle": _prescribe(dataset, forecast)})
    row = _row(summary, model="oracle", ss="SS5")
    assert row["n_true_onsets"] == 1
    assert row["n_pred_onsets"] == 1, "three decision times witnessed one onset, not three"
    assert row["n_matched"] == 1
    assert leads["flag_sample"].tolist() == [min(witnesses)]
    assert leads["true_onset_sample"].tolist() == [ONSET_SAMPLE]


def test_lead_time_is_the_true_onset_minus_the_earliest_flag(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, leads = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    # (40 - 24) / 10 Hz = 1.6 s, and every lead quantile of a one-match cell is that value.
    assert leads["lead_s"].tolist() == pytest.approx([1.6])
    row = _row(summary, model="oracle", ss="SS5")
    assert row["lead_p10"] == pytest.approx(1.6)
    assert row["lead_p50"] == pytest.approx(1.6)
    assert row["lead_p90"] == pytest.approx(1.6)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert row["false_alarms_per_min"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("predicted_onset", "matches"),
    [(ONSET_SAMPLE + 5, True), (ONSET_SAMPLE + 6, False)],
)
def test_the_matching_tolerance_is_half_a_second_inclusive(
    fabricated_corpus: Path, predicted_onset: int, matches: bool
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    forecast = _constant_forecast(dataset, OUTSIDE)
    # One decision time predicts a 10-sample quiescent run beginning at `predicted_onset`.
    flag = 34
    index = _decision_times(dataset).index(flag)
    offset = predicted_onset - flag - 1
    forecast[index, offset : offset + 10, :] = INSIDE
    summary, leads = _score(dataset, {"probe": _prescribe(dataset, forecast)})
    row = _row(summary, model="probe", ss="SS5")

    assert row["n_pred_onsets"] == 1
    assert abs(predicted_onset - ONSET_SAMPLE) / FS_HZ == pytest.approx(0.5 if matches else 0.6), (
        "the fixture must straddle the tolerance, not sit safely on one side of it"
    )
    assert bool(row["n_matched"] == 1) is matches
    if matches:
        # Lead time is measured to the flag, not to the predicted onset: (40 - 34)/10.
        assert leads["lead_s"].tolist() == pytest.approx([0.6])
    else:
        assert leads.empty


def test_false_alarms_per_minute_uses_the_evaluated_duration(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    forecast = _constant_forecast(dataset, OUTSIDE)
    # A run predicted at absolute 50..59 by the decision time at 44: 1.0 s from the true
    # onset at 40, i.e. outside the 0.5 s tolerance, so it is a false alarm and the true
    # onset is missed.
    index = _decision_times(dataset).index(44)
    forecast[index, 5:15, :] = INSIDE
    summary, leads = _score(dataset, {"probe": _prescribe(dataset, forecast)})
    row = _row(summary, model="probe", ss="SS5")

    assert row["n_pred_onsets"] == 1
    assert row["n_matched"] == 0
    assert leads.empty
    # The evaluated span runs from the first decision time (9) to the record end: 90
    # samples, 9.0 s. Not 10.0 s: nothing before the first decision time was ever scorable.
    assert row["duration_s"] == pytest.approx(9.0)
    assert row["false_alarms_per_min"] == pytest.approx(60.0 / 9.0)
    assert row["precision"] == pytest.approx(0.0)
    assert row["recall"] == pytest.approx(0.0)
    assert row["f1"] == pytest.approx(0.0)


def test_a_constant_forecast_can_never_predict_an_onset(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    models = {
        "always_outside": _prescribe(dataset, _constant_forecast(dataset, OUTSIDE)),
        "always_inside": _prescribe(dataset, _constant_forecast(dataset, INSIDE)),
    }
    summary, _ = _score(dataset, models)
    # The structural consequence the module docstring states: persistence and window_mean
    # hold one value across the horizon, so their trajectories are either in-limit from
    # index 0 ("already quiescent", no onset) or out of limits throughout. F1 = 0 there is
    # a statement about them as detectors, not about their RMSE, and n_pred_onsets says so.
    for name in models:
        row = _row(summary, model=name, ss="SS5")
        assert row["n_pred_onsets"] == 0
        assert row["f1"] == pytest.approx(0.0)
        assert row["scorable"]


def test_the_base_rate_is_measured_over_the_evaluated_span(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, _ = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    row = _row(summary, model="oracle", ss="SS5")
    # 20 quiescent samples out of the 90 that follow the first decision time.
    assert row["base_rate"] == pytest.approx(20.0 / 90.0)
    assert row["n_realizations"] == 1


# ---------------------------------------------------------------------------
# P6-D7: a cell with no scorable onset is not scored
# ---------------------------------------------------------------------------


def test_a_cell_with_no_true_onsets_is_not_scorable_and_renders_no_f1(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_QUIET,))
    summary, _ = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    row = _row(summary, model="oracle", ss="SS3")

    assert row["n_true_onsets"] == 0
    assert row["scorable"] is np.False_ or row["scorable"] is False
    for column in ("precision", "recall", "f1"):
        assert math.isnan(float(row[column])), (
            f"{column} must be NaN in a cell with nothing to detect: 0.0 reads as a model "
            f"failure and 1.0 reads as a success, and neither is true here"
        )
    # The base rate is still filled in, because it is the column that says *why*.
    assert row["base_rate"] == pytest.approx(1.0)
    # The one raw onset is the record starting inside a quiescent window, and it is
    # excluded rather than scored -- the exclusion is reported, not silent.
    assert row["n_excluded_true"] == 1


def test_the_not_scorable_row_is_not_pooled_with_a_scorable_one(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET, KEY_QUIET))
    summary, _ = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    assert sorted(summary["ss"].unique()) == ["SS3", "SS5"]
    quiet = _row(summary, model="oracle", ss="SS3")
    onset = _row(summary, model="oracle", ss="SS5")
    assert not bool(quiet["scorable"])
    assert bool(onset["scorable"])
    assert onset["f1"] == pytest.approx(1.0)
    # P6-D7 item 3: grouping, not a column. A pooled row would have averaged a perfect F1
    # with an undefined one over a cell whose base rate is 1.0.
    assert quiet["base_rate"] == pytest.approx(1.0)
    assert onset["base_rate"] == pytest.approx(20.0 / 90.0)


# ---------------------------------------------------------------------------
# P6-D5: the interval rule is two-sided
# ---------------------------------------------------------------------------


class _PrescribedQuantile(_Prescribed):
    """A quantile-head forecaster handed its whole fan."""

    head_kind = "quantile"
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    n_output_params = 3


def _quantile_model(dataset: DeckMotionDataset, fan: FloatArray) -> _PrescribedQuantile:
    """Build a quantile forecaster emitting ``fan`` in corpus units.

    Args:
        dataset: The partition the forecast is for.
        fan: Corpus-unit fan, shape ``(N, H, C, 3)`` at levels 0.05, 0.5, 0.95.

    Returns:
        The forecaster.
    """
    return _PrescribedQuantile(_to_model_space(dataset, fan))


def test_the_interval_rule_is_two_sided_not_the_one_sided_lower_quantile(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    median = _oracle_forecast(dataset)
    fan = np.stack([median - 5.0, median, median + 0.05], axis=-1)
    # The fixture is built so the two readings disagree. The plan's one-sided test -- "the
    # 0.05 quantile of each channel" below the limit -- passes here: -5.0 <= 1.0. The
    # two-sided test P6-D5 records refuses it, because max(|q05|, |q95|) = 5.0 > 1.0 and a
    # deck that may be 5 degrees over is not a landing window.
    quiet = np.abs(median) <= PROBE.roll_deg
    assert (fan[..., 0][quiet] <= PROBE.roll_deg).all()
    assert (np.abs(fan[..., 0][quiet]) > PROBE.roll_deg).all()

    summary, _ = _score(dataset, {"fan": _quantile_model(dataset, fan)})
    point = _row(summary, model="fan", rule="point", ss="SS5")
    interval = _row(summary, model="fan", rule="interval", ss="SS5")

    assert point["n_pred_onsets"] == 1, "the median forecast is the oracle and finds the onset"
    assert interval["n_pred_onsets"] == 0, (
        "max(|q05|, |q95|) exceeds the limit everywhere, so the interval rule finds no "
        "landing window; a one-sided |q05| <= limit reading would have found one"
    )
    assert interval["scorable"]
    assert interval["f1"] == pytest.approx(0.0)


def test_the_interval_rule_flags_when_both_bounds_are_inside_the_limits(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    median = _oracle_forecast(dataset)
    # A tight fan around the oracle: both bounds sit inside the limits wherever the median
    # does, so the interval rule reproduces the point rule exactly.
    fan = np.stack([median - 0.05, median, median + 0.05], axis=-1)
    summary, leads = _score(dataset, {"fan": _quantile_model(dataset, fan)})
    point = _row(summary, model="fan", rule="point", ss="SS5")
    interval = _row(summary, model="fan", rule="interval", ss="SS5")
    assert interval["n_pred_onsets"] == point["n_pred_onsets"] == 1
    assert interval["f1"] == pytest.approx(1.0)
    assert sorted(leads["rule"].unique()) == ["interval", "point"]


def test_a_point_model_is_scored_on_the_point_rule_only(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, _ = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    assert summary.loc[summary["model"] == "oracle", "rule"].unique().tolist() == ["point"]


# ---------------------------------------------------------------------------
# The degenerate detector, and the table's shape
# ---------------------------------------------------------------------------


def test_the_always_quiescent_detector_has_perfect_recall_and_poor_precision(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, _ = _score(
        dataset,
        {"oracle": _prescribe(dataset, _oracle_forecast(dataset))},
        include_always_quiescent=True,
    )
    row = _row(summary, model=ALWAYS_QUIESCENT, ss="SS5")
    # It flags an onset at every decision time, so it cannot miss one; precision is what
    # punishes it, which is the P6-D2 correction to the docstring's original claim.
    assert row["recall"] == pytest.approx(1.0)
    assert row["n_pred_onsets"] == N_WINDOWS
    assert row["precision"] == pytest.approx(1.0 / N_WINDOWS)
    assert row["false_alarms_per_min"] == pytest.approx(60.0 * (N_WINDOWS - 1) / 9.0)
    assert row["f1"] < _row(summary, model="oracle", ss="SS5")["f1"]


def test_the_summary_carries_every_declared_column_and_no_other(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET, KEY_QUIET))
    summary, leads = _score(
        dataset,
        {"oracle": _prescribe(dataset, _oracle_forecast(dataset))},
        include_always_quiescent=True,
    )
    assert tuple(summary.columns) == QUIESCENCE_COLUMNS
    # Every F1 sits in a row that also carries its base rate and its onset counts, which is
    # what makes it interpretable at all (CLAUDE.md Known traps).
    for column in ("base_rate", "n_true_onsets", "n_pred_onsets"):
        assert column in summary.columns
    assert summary["observation_mode"].unique().tolist() == ["ideal"]
    assert summary["regime"].unique().tolist() == ["id"]
    assert not leads.empty


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_decision_channel_index_refuses_a_dataset_without_heave_rate() -> None:
    with pytest.raises(ValueError, match="heave_rate"):
        decision_channel_index(("roll", "pitch", "heave"))


def test_decision_channel_index_resolves_the_imu_spelling() -> None:
    columns = ("roll_imu", "pitch_imu", "heave_imu", "heave_rate_imu")
    assert decision_channel_index(columns) == (0, 1, 3)


def test_evaluate_quiescence_refuses_an_attitude_only_arm(fabricated_corpus: Path) -> None:
    cfg = _data_cfg(target_dofs=("roll", "pitch", "heave"))
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,), cfg)
    with pytest.raises(ValueError, match="not forecast targets"):
        _score(dataset, {"probe": _prescribe(dataset, _constant_forecast(dataset, OUTSIDE))})


def test_evaluate_quiescence_refuses_an_empty_model_set(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    with pytest.raises(ValueError, match="models is empty"):
        _score(dataset, {})


def test_evaluate_quiescence_refuses_a_non_positive_sampling_rate(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    with pytest.raises(ValueError, match="fs_hz must be positive"):
        evaluate_quiescence(
            {"probe": _prescribe(dataset, _constant_forecast(dataset, OUTSIDE))},
            dataset,
            fs_hz=0.0,
            threshold_sets=(PROBE,),
        )


def test_evaluate_quiescence_refuses_a_model_of_the_wrong_output_shape(
    fabricated_corpus: Path,
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    truncated = _Prescribed(
        _to_model_space(dataset, _oracle_forecast(dataset))[:, :, :3].contiguous()
    )
    with pytest.raises(ValueError, match="returned shape"):
        _score(dataset, {"short": truncated})


# ---------------------------------------------------------------------------
# Cross-check against an independent implementation of the P6-D2 geometry
# ---------------------------------------------------------------------------


def _reference_onsets(
    dataset: DeckMotionDataset, forecast: FloatArray, thresholds: QuiescenceThresholds
) -> list[tuple[int, int]]:
    """Recompute the predicted onsets by the slow, obvious route.

    Loops one trajectory at a time through :func:`dmf.eval.quiescence.detect_quiescent_mask`
    -- the *definition* -- rather than through the batched run finder the runner uses, so
    an agreement between the two is evidence about the fast path.

    Args:
        dataset: The single-realization partition.
        forecast: Corpus-unit forecast, shape ``(N, H, C)``.
        thresholds: The limit set.

    Returns:
        Every ``(absolute onset sample, decision time)`` pair the forecast contains, before
        the earliest-flag dedupe, so a test can see how much the dedupe removed.
    """
    channels = decision_channel_index(dataset.target_columns)
    pairs: list[tuple[int, int]] = []
    for index, decision in enumerate(_decision_times(dataset)):
        trajectory = forecast[index]
        mask = detect_quiescent_mask(
            trajectory[:, channels[0]],
            trajectory[:, channels[1]],
            trajectory[:, channels[2]],
            thresholds,
            FS_HZ,
        )
        for offset in np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1]))).tolist():
            if offset == 0:
                continue
            absolute = decision + 1 + offset
            if FIRST_DECISION < absolute < LAST_COVERED:
                pairs.append((absolute, decision))
    return pairs


def test_the_runner_agrees_with_a_slow_reference_implementation(
    fabricated_corpus: Path, rng: np.random.Generator
) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    # A forecast with many short and long in-limit stretches, so the sustain filter, the
    # index-0 exclusion and the dedupe all fire repeatedly rather than once.
    pattern = rng.random((len(dataset), MAX_HORIZON)) < 0.8
    forecast = np.broadcast_to(
        np.where(pattern[:, :, None], INSIDE, OUTSIDE), (*pattern.shape, len(CHANNELS))
    ).astype(np.float64)
    pairs = _reference_onsets(dataset, forecast, PROBE)
    earliest: dict[int, int] = {}
    for absolute, decision in pairs:
        earliest.setdefault(absolute, decision)
    assert len(pairs) > len(earliest) > 5, (
        "the fixture must exercise the dedupe -- several decision times predicting one "
        "absolute onset -- and not only the happy path"
    )

    summary, leads = _score(dataset, {"probe": _prescribe(dataset, forecast)})
    row = _row(summary, model="probe", ss="SS5")
    assert row["n_pred_onsets"] == len(earliest)
    for _, lead in leads.iterrows():
        onset = int(lead["true_onset_sample"])
        flag = int(lead["flag_sample"])
        assert flag in earliest.values(), (
            f"the runner flagged at {flag}, which the reference never does"
        )
        assert lead["lead_s"] == pytest.approx((onset - flag) / FS_HZ)


def test_permissive_thresholds_are_the_shipped_defaults(fabricated_corpus: Path) -> None:
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, _ = evaluate_quiescence(
        {"probe": _prescribe(dataset, _constant_forecast(dataset, INSIDE))},
        dataset,
        fs_hz=FS_HZ,
        include_always_quiescent=False,
    )
    # The default threshold_sets argument is (PERMISSIVE, STRICT): both ship, always.
    assert sorted(summary["threshold_set"].unique()) == ["permissive", "strict"]
    assert PERMISSIVE.name == "permissive"


# ---------------------------------------------------------------------------
# The reporting unit: cells, and the roll-up that is marked as one
# ---------------------------------------------------------------------------


def test_the_cells_of_one_sea_state_are_reported_separately_from_their_roll_up(
    fabricated_corpus: Path,
) -> None:
    """P6-D19: scorability is a property of the cell, so the cell is the reporting unit.

    ``SS5`` here holds two cells: 45 deg has one scorable onset, 90 deg has none. The
    sea-state row averages a scorable cell with an unscorable one, which is the same pooling
    P6-D7 item 3 forbids one level up -- so it is emitted **beside** the cells and marked
    ``group_level = "sea_state"``, never instead of them.
    """
    dataset = _dataset(fabricated_corpus, (KEY_ONSET, KEY_ONSET_BEAM))
    summary, _ = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    scoped = summary.loc[(summary["model"] == "oracle") & (summary["ss"] == "SS5")]
    cells = scoped.loc[scoped["group_level"] == CELL_LEVEL]
    rollup = scoped.loc[scoped["group_level"] == SEA_STATE_LEVEL]

    assert sorted(cells["heading_deg"]) == [45.0, 90.0]
    assert len(rollup) == 1
    # A roll-up is told apart by its column, not by a NaN heading -- but the coordinates it
    # does not have are absent rather than invented.
    assert bool(rollup["heading_deg"].isna().all())
    assert bool(rollup["speed_kn"].isna().all())

    quiet = cells.loc[cells["heading_deg"] == 90.0].iloc[0]
    active = cells.loc[cells["heading_deg"] == 45.0].iloc[0]
    assert not bool(quiet["scorable"])
    assert np.isnan(float(quiet["f1"]))
    assert np.isnan(float(quiet["f1_ci_lo"]))
    assert bool(active["scorable"])
    # The roll-up's counts are exactly the cells' counts summed; it is a re-grouping of the
    # same matches and not a second scoring pass.
    for column in ("n_true_onsets", "n_pred_onsets", "n_matched", "n_realizations"):
        assert int(rollup.iloc[0][column]) == int(cells[column].sum())
    # And it is scorable even though one of its cells is not, which is exactly why the two
    # levels may not be conflated: the roll-up hides the cell with nothing to detect.
    assert bool(rollup.iloc[0]["scorable"])


def test_the_lead_times_are_emitted_once_from_the_cell_pass(fabricated_corpus: Path) -> None:
    """The roll-up re-groups matches; it does not produce new ones."""
    dataset = _dataset(fabricated_corpus, (KEY_ONSET, KEY_ONSET_BEAM))
    summary, leads = _score(dataset, {"oracle": _prescribe(dataset, _oracle_forecast(dataset))})
    matched = summary.loc[
        (summary["model"] == "oracle") & (summary["group_level"] == CELL_LEVEL), "n_matched"
    ].sum()
    assert len(leads) == int(matched)
    assert "group_level" not in leads.columns


# ---------------------------------------------------------------------------
# The F1 interval
# ---------------------------------------------------------------------------


def test_the_f1_bootstrap_reproduces_the_point_estimate_under_uniform_weights() -> None:
    """The identity the fast path rests on, asserted against the reference implementation.

    ``F1 = 2 * sum(matched) / (sum(predicted) + sum(true))`` is
    :func:`dmf.eval.quiescence.precision_recall_f1`'s ``2PR/(P+R)``. If that were only
    nearly true the interval would be centred on a different quantity from the F1 printed
    beside it.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        n_true = rng.integers(0, 8, size=6)
        n_pred = rng.integers(0, 8, size=6)
        n_matched = np.minimum(n_true, n_pred)
        _, _, reference = precision_recall_f1(
            int(n_matched.sum()), int(n_pred.sum()), int(n_true.sum())
        )
        total = int(n_pred.sum()) + int(n_true.sum())
        fast = 2.0 * int(n_matched.sum()) / total if total else 0.0
        assert fast == pytest.approx(reference)


def test_the_f1_interval_is_deterministic_and_brackets_a_constant_cell() -> None:
    counts = np.array([3, 3, 3, 3, 3, 3], dtype=np.int64)
    lo, hi = bootstrap_f1_ci(counts, counts, counts)
    # Every realization is identical, so every resample is identical: the interval is a
    # point, and a point is the honest answer rather than a suspiciously narrow one.
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
    again = bootstrap_f1_ci(counts, counts, counts)
    assert (lo, hi) == again


def test_a_sparser_cell_gets_a_wider_f1_interval() -> None:
    """A sparse cell gets a wider interval, which is the whole point of printing one.

    P6-D19: SS6/`strict` holds ~2.6 scorable onsets per realization and is where the F1 is
    least determined. The interval is what says so, and it is reported rather than
    suppressed.
    """
    rng = np.random.default_rng(7)
    dense_true = rng.integers(30, 40, size=40).astype(np.int64)
    dense_matched = (dense_true * 0.5).astype(np.int64)
    sparse_true = rng.integers(0, 4, size=40).astype(np.int64)
    sparse_matched = (sparse_true * 0.5).astype(np.int64)
    dense_lo, dense_hi = bootstrap_f1_ci(dense_matched, dense_true, dense_true)
    sparse_lo, sparse_hi = bootstrap_f1_ci(sparse_matched, sparse_true, sparse_true)
    assert (sparse_hi - sparse_lo) > (dense_hi - dense_lo)


def test_an_unscorable_cell_has_no_f1_interval_rather_than_a_zero_one() -> None:
    zeros = np.zeros(5, dtype=np.int64)
    lo, hi = bootstrap_f1_ci(zeros, np.full(5, 3, dtype=np.int64), zeros)
    assert np.isnan(lo) and np.isnan(hi)


def test_the_f1_interval_refuses_arguments_that_do_not_describe_one_partition() -> None:
    counts = np.ones(4, dtype=np.int64)
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_f1_ci(counts, counts, np.ones(3, dtype=np.int64))
    with pytest.raises(ValueError, match="n_boot must be positive"):
        bootstrap_f1_ci(counts, counts, counts, n_boot=0)
    with pytest.raises(ValueError, match=r"ci_level must be in \(0, 1\)"):
        bootstrap_f1_ci(counts, counts, counts, ci_level=1.0)


# ---------------------------------------------------------------------------
# The chance-level detector
# ---------------------------------------------------------------------------


def test_rate_matched_onsets_are_uniform_deterministic_and_inside_the_span() -> None:
    onsets, flags = rate_matched_onsets(4, 9, 89)
    assert np.array_equal(onsets, rate_matched_onsets(4, 9, 89)[0]), "no RNG anywhere"
    assert bool((onsets > 9).all()) and bool((onsets < 89).all())
    assert onsets.size == 4
    # Uniform: the gaps are equal to within one sample of the floor rounding.
    gaps = np.diff(onsets)
    assert int(gaps.max() - gaps.min()) <= 1
    # Each flag is the latest decision moment strictly before its onset: the shortest lead
    # consistent with predicting it, so the chance detector cannot win the lead-time column.
    assert np.array_equal(flags, onsets - 1)
    assert rate_matched_onsets(0, 9, 89)[0].size == 0
    # A span too small for the requested count yields nothing rather than duplicates.
    assert rate_matched_onsets(50, 9, 20)[0].size == 0
    with pytest.raises(ValueError, match="non-negative"):
        rate_matched_onsets(-1, 9, 89)


def test_the_rate_matched_detector_emits_the_true_onset_count_and_scores_at_chance(
    fabricated_corpus: Path,
) -> None:
    """``always_quiescent`` is not the chance level, and this is the detector that is.

    ``always_quiescent`` flags at every decision time, so its precision is
    ``n_true / n_decisions`` **by construction** -- it is punished for over-flagging and its
    low F1 is arithmetic about the decision grid. ``rate_matched`` emits exactly as many
    onsets as the truth holds, so that explanation is removed and what is left is that it
    does not know *when*.
    """
    dataset = _dataset(fabricated_corpus, (KEY_ONSET,))
    summary, _ = _score(
        dataset,
        {"oracle": _prescribe(dataset, _oracle_forecast(dataset))},
        include_always_quiescent=True,
    )
    chance = _row(summary, model=RATE_MATCHED, ss="SS5")
    degenerate = _row(summary, model=ALWAYS_QUIESCENT, ss="SS5")
    oracle = _row(summary, model="oracle", ss="SS5")

    assert int(chance["n_pred_onsets"]) == int(chance["n_true_onsets"])
    assert int(degenerate["n_pred_onsets"]) == N_WINDOWS
    # Rate-matched, so precision and recall are the same number; the degenerate detector
    # buys recall 1.0 with a precision fixed by the grid.
    assert chance["precision"] == pytest.approx(chance["recall"])
    assert degenerate["precision"] == pytest.approx(int(degenerate["n_true_onsets"]) / N_WINDOWS)
    # And a detector that actually knows when beats both.
    assert oracle["f1"] > chance["f1"]
    assert oracle["f1"] > degenerate["f1"]
    # The chance detector's false-alarm rate is its onset rate, not the decision rate: that
    # is the whole difference between the two references.
    assert chance["false_alarms_per_min"] < degenerate["false_alarms_per_min"]
