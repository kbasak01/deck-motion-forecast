"""The matched-origin scoring path: the dataset subset, the driver, and the trap.

What is pinned here, in the order the failure would happen:

1. **Matching on the window START compares forecasts of different absolute times.** The
   error P6-D4 item 1 was written to prevent is asserted *as an error*, on real windows
   pulled out of two datasets, not merely described in a docstring. Two arms sharing a start
   index forecast samples 300 apart at the production geometry; two arms sharing an origin
   forecast the same samples, and the second is what
   :class:`dmf.data.dataset.DeckMotionDataset`'s ``origins`` argument expresses.
2. **The subset changes the window population and nothing else.** Same normalisation
   statistics, same tensors for the windows it keeps, same realization-major layout, and
   ``windows_per_realization`` still uniform -- which every consumer of that property
   assumes.
3. **A matched table proves it is matched from its own rows.** The provenance
   :func:`dmf.eval.ablations.matched_origin_provenance` writes is re-derived by
   :func:`dmf.eval.ablations.unmatched_origin_reason`, and an arm's own committed table --
   the thing P6-D15 finding 1 found on disk -- is refused even when it is sitting in the
   matched-origin directory.
4. **The driver re-scores and does not retrain.** It restricts the test partition only, and
   refuses to emit the quiescence or phase-lag passes beside a matched accuracy row.

The production arithmetic (1151 / 1131 / 1091 per realization, origins 399..5849) is checked
directly against the table in P6-D4, because that entry is what the assembler's refusal and
this module's acceptance are both written from.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmf.config import DataConfig, load_data
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import build_split
from dmf.data.windows import WindowSpec, window_origins, window_spec_from_config
from dmf.eval.ablations import (
    ARMS,
    MATCHED_ORIGIN_COLUMNS,
    matched_origin_provenance,
    matched_origins,
    matched_window_counts,
    origin_window_indices,
    start_matching_compares_different_times,
    unmatched_origin_reason,
)
from dmf.eval.assemble import (
    ARM_ROWS_FILE,
    MATCHED_ORIGIN_ARMS,
    MATCHED_ORIGIN_SUBDIR,
    read_arm_frames,
)
from dmf.eval.matched import (
    corpus_n_samples,
    lookback_arm_specs,
    matched_lookback_plan,
    score_matched_lookback,
)

#: The three production lookbacks, samples.
PRODUCTION_LOOKBACKS: tuple[int, ...] = (100, 200, 400)

#: The production horizon set and stride, from ``configs/data/default.yaml``.
PRODUCTION_HORIZONS: tuple[int, ...] = (10, 20, 30, 50, 100, 150)
PRODUCTION_STRIDE: int = 5

#: Production realization length, samples.
PRODUCTION_SAMPLES: int = 6000


def _production_specs() -> list[WindowSpec]:
    """Return the three lookback arms' production geometries."""
    return [
        WindowSpec(lookback=lookback, horizons=PRODUCTION_HORIZONS, stride=PRODUCTION_STRIDE)
        for lookback in PRODUCTION_LOOKBACKS
    ]


# ---------------------------------------------------------------------------
# 1. The trap
# ---------------------------------------------------------------------------


def test_start_matching_compares_different_times_at_the_production_geometry() -> None:
    """P6-D4 item 1's counter-example, at the two lookbacks furthest apart."""
    short, _, long = _production_specs()
    misaligned, offset = start_matching_compares_different_times(short, long)
    assert misaligned
    assert offset == -300
    # 300 samples is 30 s at 10 Hz -- twice the longest horizon the project reports, so a
    # start-matched row would not merely be noisy, it would describe a different event.
    assert abs(offset) / 10.0 == 30.0


def test_matching_on_the_start_index_forecasts_different_samples(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """The wrong rule, asserted wrong on real windows rather than on arithmetic alone.

    Two datasets differing only in lookback are built over one realization set. At the same
    *window index* -- i.e. the same start sample -- the targets they hold are different
    absolute samples, and the difference is exactly the lookback difference. At the same
    *origin* they are the same samples, to the bit.
    """
    short_cfg = replace(small_data_cfg, lookback=25)
    long_cfg = replace(small_data_cfg, lookback=50)
    short_spec = window_spec_from_config(short_cfg)
    long_spec = window_spec_from_config(long_cfg)
    split = build_split(pd.read_parquet(small_corpus / "manifest.parquet"), "id")

    short_stats = DeckMotionDataset(small_corpus, split, "train", short_cfg, short_spec).norm_stats
    long_stats = DeckMotionDataset(small_corpus, split, "train", long_cfg, long_spec).norm_stats
    short = DeckMotionDataset(small_corpus, split, "test", short_cfg, short_spec, stats=short_stats)
    long = DeckMotionDataset(small_corpus, split, "test", long_cfg, long_spec, stats=long_stats)

    # Same window index, i.e. same start sample: the two forecast different absolute times.
    _, short_start = short.describe_window(3)
    _, long_start = long.describe_window(3)
    assert short_start == long_start
    _, short_y, _ = short[3]
    _, long_y, _ = long[3]
    assert not np.allclose(short_y.numpy(), long_y.numpy())

    # Same origin: the two forecast the same absolute times, bitwise.
    origins = matched_origins([short_spec, long_spec], short.n_samples)
    short_index = origin_window_indices(short_spec, short.n_samples, origins)
    long_index = origin_window_indices(long_spec, long.n_samples, origins)
    for position in (0, len(origins) // 2, len(origins) - 1):
        _, y_short, _ = short[int(short_index[position])]
        _, y_long, _ = long[int(long_index[position])]
        assert np.array_equal(y_short.numpy(), y_long.numpy())


# ---------------------------------------------------------------------------
# 2. The dataset subset
# ---------------------------------------------------------------------------


def test_the_origin_subset_keeps_exactly_the_requested_windows(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(pd.read_parquet(small_corpus / "manifest.parquet"), "id")
    stats = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec).norm_stats
    full = DeckMotionDataset(small_corpus, split, "test", small_data_cfg, spec, stats=stats)
    keep = window_origins(full.n_samples, spec)[2:5]
    subset = DeckMotionDataset(
        small_corpus, split, "test", small_data_cfg, spec, stats=full.norm_stats, origins=keep
    )

    assert subset.is_origin_subset and not full.is_origin_subset
    assert subset.windows_per_realization == len(keep)
    assert len(subset) == len(subset.realization_keys) * len(keep)
    assert np.array_equal(subset.window_origins, keep)
    assert np.array_equal(subset.window_starts, keep - spec.lookback + 1)
    # Realization-major, as `dmf.eval.runner._key_index` requires.
    for index in range(len(subset)):
        key, start = subset.describe_window(index)
        assert key == subset.realization_keys[index // len(keep)]
        assert start == int(keep[index % len(keep)]) - spec.lookback + 1


def test_the_subset_returns_the_same_tensors_the_full_dataset_does(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(pd.read_parquet(small_corpus / "manifest.parquet"), "id")
    stats = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec).norm_stats
    full = DeckMotionDataset(small_corpus, split, "test", small_data_cfg, spec, stats=stats)
    origins = window_origins(full.n_samples, spec)
    keep = origins[1:4]
    subset = DeckMotionDataset(
        small_corpus, split, "test", small_data_cfg, spec, stats=full.norm_stats, origins=keep
    )
    index = origin_window_indices(spec, full.n_samples, keep)
    for realization in range(len(subset.realization_keys)):
        for position, source in enumerate(index):
            got = subset[realization * len(keep) + position]
            want = full[realization * full.windows_per_realization + int(source)]
            for left, right in zip(got, want, strict=True):
                assert np.array_equal(left.numpy(), right.numpy())


def test_the_subset_does_not_move_the_normalisation_statistics(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """Non-negotiable 3 is untouched by the matching: the scale is still the train split's.

    Fitted on the training partition, subset or not, and over the whole series either way --
    so a matched re-scoring is read against the same denominator the unmatched run used.
    """
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(pd.read_parquet(small_corpus / "manifest.parquet"), "id")
    full = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    keep = window_origins(full.n_samples, spec)[:3]
    subset = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec, origins=keep)
    assert subset.norm_stats.fitted_on == full.norm_stats.fitted_on == "id/train"
    assert np.array_equal(subset.norm_stats.scale, full.norm_stats.scale)


@pytest.mark.parametrize(
    ("origins", "message"),
    [
        ([], "empty"),
        ([[10, 20]], "one-dimensional"),
        ([70, 45], "ascending"),
        ([45, 45], "ascending"),
        ([46], "not scored"),
        ([10**6], "not scored"),
    ],
)
def test_an_origin_set_this_geometry_does_not_score_is_refused(
    small_corpus: Path, small_data_cfg: DataConfig, origins: list[object], message: str
) -> None:
    spec = window_spec_from_config(small_data_cfg)
    split = build_split(pd.read_parquet(small_corpus / "manifest.parquet"), "id")
    stats = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec).norm_stats
    with pytest.raises(ValueError, match=message):
        DeckMotionDataset(
            small_corpus,
            split,
            "test",
            small_data_cfg,
            spec,
            stats=stats,
            origins=np.asarray(origins, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# 3. The provenance a matched table must carry
# ---------------------------------------------------------------------------


def test_the_production_window_counts_reproduce_p6_d4() -> None:
    """The table in P6-D4 item 1, recomputed rather than transcribed."""
    specs = _production_specs()
    counts = matched_window_counts(specs, PRODUCTION_SAMPLES)
    assert counts == {100: (1151, 1091), 200: (1131, 1091), 400: (1091, 1091)}
    origins = matched_origins(specs, PRODUCTION_SAMPLES)
    assert (int(origins[0]), int(origins[-1]), origins.size) == (399, 5849, 1091)
    # Nested, so the intersection IS the L=400 set.
    assert np.array_equal(origins, window_origins(PRODUCTION_SAMPLES, specs[-1]))


def test_the_provenance_round_trips_through_its_own_verifier() -> None:
    specs = _production_specs()
    for spec, arm in zip(specs, ("lookback_10s", "lookback_20s", "lookback_40s"), strict=True):
        record = matched_origin_provenance(spec, specs, PRODUCTION_SAMPLES)
        assert set(record) == set(MATCHED_ORIGIN_COLUMNS)
        assert unmatched_origin_reason(record, arm) == ""
        with_counts = {**record, "n_windows": 1091 * 384, "n_realizations": 384}
        assert unmatched_origin_reason(with_counts, arm) == ""


def test_a_table_without_the_provenance_columns_is_not_a_matched_table() -> None:
    """An arm's own committed rows: the exact thing P6-D15 finding 1 found on disk."""
    reason = unmatched_origin_reason({"model": "persistence", "skill": 0.0}, "lookback_10s")
    assert "matched_n_samples" in reason
    assert "different absolute times" in reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("matched_n_origins", 1151),
        ("matched_origin_first", 99),
        ("matched_origin_last", 5849 - 5),
        ("matched_windows_before", 1091),
        ("matched_n_samples", 3000),
    ],
)
def test_a_provenance_that_does_not_reproduce_is_refused(field: str, value: int) -> None:
    record = dict(matched_origin_provenance(_production_specs()[0], _production_specs(), 6000))
    record[field] = value
    assert unmatched_origin_reason(record, "lookback_10s") != ""


def test_a_row_whose_window_count_contradicts_its_origin_count_is_refused() -> None:
    record = dict(matched_origin_provenance(_production_specs()[0], _production_specs(), 6000))
    # 1151 per realization: the arm's own full window set, wearing matched provenance.
    record["n_realizations"] = 384
    record["n_windows"] = 1151 * 384
    reason = unmatched_origin_reason(record, "lookback_10s")
    assert "per realization" in reason


def test_the_arm_registry_and_the_configs_agree_on_every_lookback() -> None:
    """The registry's ``lookback`` is what the ablation table is labelled from."""
    arms = lookback_arm_specs()
    assert set(arms) == set(MATCHED_ORIGIN_ARMS)
    for arm, (spec, configs) in arms.items():
        assert spec.lookback == ARMS[arm].lookback
        assert configs, f"arm {arm} has no config"


# ---------------------------------------------------------------------------
# 4. The driver
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matched_run(
    small_corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[object, Path]:
    """Re-score the two closed-form lookback arms on the small corpus.

    ``lookback_20s`` is left out: its config is ``e02_deep``, whose SGD rows need committed
    checkpoints, and a missing checkpoint is a hard error rather than a silent refit. The two
    arms scored here carry closed-form and trivial models only.
    """
    out = tmp_path_factory.mktemp("matched") / MATCHED_ORIGIN_SUBDIR
    artifacts = score_matched_lookback(
        small_corpus,
        out_root=out,
        arms=("lookback_10s", "lookback_40s"),
        regimes=("unseen_heading",),
        batch_size=512,
    )
    return artifacts, out


def test_the_driver_writes_one_matched_table_per_arm(matched_run: tuple[object, Path]) -> None:
    artifacts, out = matched_run
    assert set(artifacts.paths) == {"lookback_10s", "lookback_40s"}  # type: ignore[attr-defined]
    for arm, path in artifacts.paths.items():  # type: ignore[attr-defined]
        assert path == out / arm / ARM_ROWS_FILE
        frame = pd.read_csv(path)
        assert not frame.empty
        for column in MATCHED_ORIGIN_COLUMNS:
            assert frame[column].nunique() == 1, column


def test_every_arm_is_scored_on_the_same_origins(matched_run: tuple[object, Path]) -> None:
    artifacts, _ = matched_run
    frames = {arm: pd.read_csv(path) for arm, path in artifacts.paths.items()}  # type: ignore[attr-defined]
    summaries = {
        arm: tuple(
            frame[name].iloc[0]
            for name in ("matched_n_origins", "matched_origin_first", "matched_origin_last")
        )
        for arm, frame in frames.items()
    }
    assert len(set(summaries.values())) == 1
    # And the scored window count follows the origin count, per realization.
    for frame in frames.values():
        per_realization = frame["n_windows"] / frame["n_realizations"]
        assert (per_realization == frame["matched_n_origins"]).all()


def test_the_matched_tables_are_accepted_by_the_assembler(
    matched_run: tuple[object, Path], tmp_path: Path
) -> None:
    artifacts, out = matched_run
    root = tmp_path / "results" / "e04"
    (root / MATCHED_ORIGIN_SUBDIR).mkdir(parents=True)
    for arm, path in artifacts.paths.items():  # type: ignore[attr-defined]
        destination = root / MATCHED_ORIGIN_SUBDIR / arm
        destination.mkdir()
        destination.joinpath(ARM_ROWS_FILE).write_bytes(path.read_bytes())
        frames, reason = read_arm_frames(tmp_path / "results", arm)
        assert reason == "", reason
        assert len(frames) == 1


def test_the_assembler_still_refuses_an_arms_own_table_in_the_matched_directory(
    matched_run: tuple[object, Path], tmp_path: Path
) -> None:
    """The refusal P6-D15 finding 1 rests on must survive the arrival of a producer.

    A committed ``baselines_by_seed.csv`` moved into the matched directory carries no
    provenance columns and must be refused there exactly as it is refused anywhere else --
    the directory is not evidence.
    """
    artifacts, _ = matched_run
    frame = pd.read_csv(artifacts.paths["lookback_10s"])  # type: ignore[attr-defined]
    raw = frame.drop(columns=list(MATCHED_ORIGIN_COLUMNS))
    root = tmp_path / "results" / "e04" / MATCHED_ORIGIN_SUBDIR / "lookback_10s"
    root.mkdir(parents=True)
    raw.to_csv(root / ARM_ROWS_FILE, index=False)
    frames, reason = read_arm_frames(tmp_path / "results", "lookback_10s")
    assert frames == ()
    assert "not a matched table" in reason


def test_a_matched_table_whose_provenance_does_not_reproduce_is_refused(
    matched_run: tuple[object, Path], tmp_path: Path
) -> None:
    artifacts, _ = matched_run
    frame = pd.read_csv(artifacts.paths["lookback_40s"])  # type: ignore[attr-defined]
    # Internally consistent -- the declared origin count and the scored window count still
    # agree -- so only re-deriving the intersection from n_samples, stride, max horizon and
    # the arm lookbacks can catch it. That is the check this test is about.
    frame["matched_n_origins"] = frame["matched_n_origins"] + 40
    frame["n_windows"] = frame["matched_n_origins"] * frame["n_realizations"]
    root = tmp_path / "results" / "e04" / MATCHED_ORIGIN_SUBDIR / "lookback_40s"
    root.mkdir(parents=True)
    frame.to_csv(root / ARM_ROWS_FILE, index=False)
    _, reason = read_arm_frames(tmp_path / "results", "lookback_40s")
    assert "does not reproduce" in reason


def test_a_row_scored_on_a_different_window_set_is_refused_even_if_others_are_not(
    matched_run: tuple[object, Path], tmp_path: Path
) -> None:
    """The check is row-wise, because ``n_windows`` is a per-regime count.

    Four regimes hold different numbers of realizations, so a check that looked only at the
    first row -- or only at files with one distinct value -- would cover one regime or none.
    """
    artifacts, _ = matched_run
    frame = pd.read_csv(artifacts.paths["lookback_10s"])  # type: ignore[attr-defined]
    frame.loc[frame.index[-1], "n_windows"] = int(frame["n_windows"].iloc[-1]) + 5
    root = tmp_path / "results" / "e04" / MATCHED_ORIGIN_SUBDIR / "lookback_10s"
    root.mkdir(parents=True)
    frame.to_csv(root / ARM_ROWS_FILE, index=False)
    frames, reason = read_arm_frames(tmp_path / "results", "lookback_10s")
    assert frames == ()
    assert "1 row(s)" in reason


def test_the_matched_rows_carry_the_columns_a_contrast_needs(
    matched_run: tuple[object, Path],
) -> None:
    artifacts, _ = matched_run
    frame = pd.read_csv(artifacts.paths["lookback_10s"])  # type: ignore[attr-defined]
    for column in ("model", "regime", "dof", "horizon_samples", "horizon_s", "seed", "n_params"):
        assert column in frame.columns
    for column in ("skill", "nrmse"):
        assert column in frame.columns
    # Persistence scores exactly zero against itself on the matched windows too: the
    # denominator was measured over the same restricted set as the numerator.
    own = frame.loc[frame["model"] == "persistence", "skill"].to_numpy()
    assert np.all(own == 0.0)
    # No quiescence or phase-lag column: neither pass is defined on a thinned origin set.
    assert "phase_lag_s" not in frame.columns


def test_the_driver_refuses_to_pair_a_quiescence_or_phase_pass_with_a_matched_set(
    small_corpus: Path, tmp_path: Path
) -> None:
    from dmf.config import load_experiment
    from dmf.eval.scoring import score_regime

    cfg = load_experiment(Path("configs/experiment/e04d_lookback_10s_ood.yaml"))
    origins, _, _ = matched_lookback_plan(small_corpus)
    with pytest.raises(ValueError, match="matched origin set"):
        score_regime(
            cfg,
            small_corpus,
            "unseen_heading",
            with_quiescence=True,
            with_phase_lag=False,
            test_origins=origins,
        )


def test_the_plan_is_taken_over_every_arm_even_when_one_is_scored(small_corpus: Path) -> None:
    """Restricting ``arms`` must not widen the origin set the scored arm is matched on."""
    origins, n_samples, plan = matched_lookback_plan(small_corpus)
    assert n_samples == corpus_n_samples(small_corpus)
    assert set(plan) == set(MATCHED_ORIGIN_ARMS)
    binding = max(spec.lookback for spec, _ in plan.values())
    spec = next(spec for spec, _ in plan.values() if spec.lookback == binding)
    assert np.array_equal(origins, window_origins(n_samples, spec))


def test_the_small_corpus_geometry_matches_the_production_data_configs() -> None:
    """The arms differ in lookback and in nothing else, which is what makes it an ablation."""
    configs = {
        arm: load_data(Path("configs/data") / f"{ARMS[arm].data_config}.yaml")
        for arm in MATCHED_ORIGIN_ARMS
    }
    varying = {
        name
        for name in vars(next(iter(configs.values())))
        if len({getattr(cfg, name) for cfg in configs.values()}) > 1
    }
    assert varying == {"lookback"}
