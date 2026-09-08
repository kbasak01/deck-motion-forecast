"""The score-from-committed-artifacts driver.

What is pinned here:

1. **The five passes describe the same windows.** One partition is built per (experiment,
   regime) and handed to all of them, so a row's ``rmse``, its ``phase_lag_s`` and its
   ``picp`` cannot come from different window populations. The test checks the tables agree
   on the model set, the regime and the DOFs, and that the persistence reference scores
   exactly zero skill against itself.
2. **The provenance guard fires before anything is built** (P3-D11). A test partition
   scaled by another regime's training statistics is refused, and the refusal is the
   *existing* :func:`dmf.eval.runner._check_norm_provenance`, not a copy.
3. **A pass that cannot run is recorded, not skipped silently.** The attitude-only arm does
   not forecast ``heave_rate``, so it cannot supply the quiescence detector's decision
   variable; the driver must carry the reason in ``skipped`` rather than emitting an empty
   quiescence table that reads as "no landing windows were found".
4. **Nothing is written outside the results directory the caller named**, and no empty CSV
   is written at all.

The experiment used here holds only closed-form and parameter-free models, so the driver's
``load_or_fit`` path runs without a checkpoint tree; the SGD branch of ``load_or_fit`` is
covered in ``tests/test_models.py``.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmf.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import build_split
from dmf.data.windows import window_spec_from_config
from dmf.eval.quiescence_runner import SYNTHETIC_DETECTORS
from dmf.eval.scoring import SCORING_ARTIFACTS, score_experiment, score_regime

#: Horizons of the small-corpus geometry, samples.
HORIZONS: tuple[int, ...] = (5, 10, 20)

#: Realizations the phase pass sub-samples in these tests. Two, for speed; the sub-sampling
#: rule itself is tested in ``tests/test_phase_runner.py``.
N_PHASE = 2


def _train_cfg() -> TrainConfig:
    """Return a training block the closed-form rows never read."""
    return TrainConfig(
        epochs=1,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        warmup_frac=0.0,
        grad_clip=1.0,
        patience=1,
        amp_dtype="off",
        num_workers=0,
    )


def _experiment(cfg: DataConfig, regimes: tuple[str, ...] = ("id",)) -> ExperimentConfig:
    """Return an experiment of closed-form and parameter-free models only."""
    return ExperimentConfig(
        name="e04_test",
        data=cfg,
        models=(
            ModelConfig(
                name="persistence", head="point", quantiles=(), params={}, label="persistence"
            ),
            ModelConfig(
                name="window_mean", head="point", quantiles=(), params={}, label="window_mean"
            ),
            ModelConfig(
                name="dlinear_ols",
                head="point",
                quantiles=(),
                params={"kernel_size": 5, "ridge": 1e-6},
                label="dlinear_ols",
            ),
        ),
        train=_train_cfg(),
        seeds=(0, 1, 2),
        regimes=regimes,
    )


@pytest.fixture(scope="module")
def scored(
    small_corpus: Path, small_manifest: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> tuple[object, Path]:
    """Score the fixture experiment once, into a temporary results directory."""
    from dmf.config import load_data

    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(base, lookback=50, horizons=HORIZONS, stride=25)
    out = tmp_path_factory.mktemp("scoring_results")
    artifacts = score_experiment(
        _experiment(cfg),
        small_corpus,
        results_dir=out,
        n_phase_realizations=N_PHASE,
        batch_size=2048,
    )
    return artifacts, out


def test_every_declared_table_is_written_and_none_is_empty(scored: tuple[object, Path]) -> None:
    artifacts, out = scored
    written = artifacts.paths  # type: ignore[attr-defined]
    # No probabilistic model in this experiment, so that table is absent rather than empty:
    # an empty CSV in results/ cannot be told from a table with nothing to report.
    assert set(written) == set(SCORING_ARTIFACTS) - {"probabilistic"}
    for name, path in written.items():
        assert path == out / SCORING_ARTIFACTS[name]
        assert path.is_file()
        assert len(pd.read_csv(path)) > 0
    assert not (out / SCORING_ARTIFACTS["probabilistic"]).exists()


def test_the_accuracy_table_carries_its_persistence_reference_at_exactly_zero_skill(
    scored: tuple[object, Path],
) -> None:
    artifacts, _ = scored
    metrics = artifacts.metrics_full  # type: ignore[attr-defined]
    own = metrics.loc[metrics["model"] == "persistence", "skill"].to_numpy()
    assert own.size > 0
    assert np.all(own == 0.0), "skill is 1 - sse/sse_persistence from the same accumulator"
    assert set(metrics["model"]) == {"persistence", "window_mean", "dlinear_ols"}
    assert set(metrics["regime"]) == {"id"}
    assert set(metrics["horizon_samples"]) == set(HORIZONS)


def test_every_metric_row_carries_its_phase_lag_and_its_run_metadata(
    scored: tuple[object, Path],
) -> None:
    artifacts, _ = scored
    metrics = artifacts.metrics_full  # type: ignore[attr-defined]
    for column in ("rmse", "skill", "nrmse", "signal_std", "skill_ci_lo", "skill_ci_hi"):
        assert column in metrics.columns
    # P6-D3: both readings ship, and the identifiability flag ships with them, so a lag
    # near half a period is readable as unidentified rather than as a measurement.
    for column in ("phase_lag_s", "phase_lag_raw_s", "dominant_period_s", "phase_lag_identified"):
        assert column in metrics.columns
        assert metrics[column].notna().all()
    # Parameter counts and fit times appear in every model-comparison table.
    for column in ("n_params", "fit_time_s", "deterministic", "seed"):
        assert column in metrics.columns


def test_the_per_cell_table_marginalises_back_to_the_pooled_one(
    scored: tuple[object, Path],
) -> None:
    artifacts, _ = scored
    metrics = artifacts.metrics_full  # type: ignore[attr-defined]
    cells = artifacts.metrics_by_cell  # type: ignore[attr-defined]
    assert set(cells["model"]) == set(metrics["model"])
    for column in ("vessel", "ss", "heading_deg", "speed_kn"):
        assert column in cells.columns
    pooled = (
        cells.groupby(["model", "dof", "horizon_samples"], as_index=False)[["sse", "n_windows"]]
        .sum()
        .assign(rmse=lambda f: np.sqrt(f["sse"] / f["n_windows"]))
    )
    merged = metrics.merge(pooled, on=["model", "dof", "horizon_samples"], suffixes=("", "_cell"))
    assert len(merged) == len(metrics)
    assert merged["rmse"].to_numpy() == pytest.approx(merged["rmse_cell"].to_numpy(), rel=1e-9)


def test_the_quiescence_table_is_grouped_by_sea_state_with_its_base_rate(
    scored: tuple[object, Path],
) -> None:
    artifacts, _ = scored
    quiescence = artifacts.quiescence  # type: ignore[attr-defined]
    assert not quiescence.empty
    assert {"ss", "base_rate", "n_true_onsets", "scorable", "threshold_set", "rule"} <= set(
        quiescence.columns
    )
    # Every F1 sits beside its base rate, and no row pools sea states (P6-D7 item 3).
    scorable = quiescence.loc[quiescence["scorable"].astype(bool)]
    assert scorable["base_rate"].notna().all()
    assert quiescence["ss"].nunique() >= 1
    # Both synthetic detectors ride along, labelled as detectors rather than as runs.
    synthetic = quiescence["model"].isin(SYNTHETIC_DETECTORS)
    assert set(SYNTHETIC_DETECTORS) <= set(quiescence["model"])
    assert (quiescence.loc[synthetic, "train_seed"] == -1).all()
    assert set(quiescence.loc[~synthetic, "train_seed"]) == {0}


def test_the_attitude_only_arm_records_why_it_has_no_quiescence_table(
    small_corpus: Path, small_manifest: pd.DataFrame, tmp_path: Path
) -> None:
    from dmf.config import load_data

    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(
        base,
        lookback=50,
        horizons=HORIZONS,
        stride=25,
        target_dofs=("roll", "pitch", "heave"),
        input_channels=("roll", "pitch", "heave"),
    )
    artifacts = score_experiment(
        _experiment(cfg),
        small_corpus,
        results_dir=tmp_path,
        n_phase_realizations=N_PHASE,
        with_phase_lag=False,
    )
    assert artifacts.quiescence.empty
    reason = artifacts.skipped["id/quiescence"]
    # The reason names the missing channel, so a reader of `skipped` learns why rather than
    # that something went wrong. An empty quiescence table with no reason would read as
    # "the detector found nothing", which is a claim about the models.
    assert "heave_rate" in reason
    assert not (tmp_path / SCORING_ARTIFACTS["quiescence"]).exists()
    assert (tmp_path / SCORING_ARTIFACTS["metrics_full"]).is_file()


def test_scoring_refuses_a_partition_scaled_by_another_regimes_statistics(
    small_corpus: Path, small_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dmf.config import load_data

    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(base, lookback=50, horizons=HORIZONS, stride=25)
    spec = window_spec_from_config(cfg)
    foreign = DeckMotionDataset(
        small_corpus, build_split(small_manifest, "unseen_vessel"), "train", cfg, spec
    )
    foreign_stats = foreign.norm_stats
    original = DeckMotionDataset.norm_stats

    # The dataset cannot know which regime the caller meant, so the guard is downstream:
    # here the `id` partitions are handed `unseen_vessel/train` statistics, which is the
    # exact failure P3-D11 exists for -- id/train spans realizations another regime holds
    # out, and a cross-regime scale silently moves every RMSE and every skill denominator.
    monkeypatch.setattr(
        DeckMotionDataset,
        "norm_stats",
        property(lambda _self: foreign_stats),
        raising=True,
    )
    try:
        with pytest.raises(ValueError, match="normalisation statistics"):
            score_regime(_experiment(cfg), small_corpus, "id", n_phase_realizations=N_PHASE)
    finally:
        monkeypatch.setattr(DeckMotionDataset, "norm_stats", original, raising=True)


def test_scoring_refuses_an_experiment_without_its_persistence_reference(
    small_corpus: Path, tmp_path: Path
) -> None:
    from dmf.config import load_data

    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(base, lookback=50, horizons=HORIZONS, stride=25)
    experiment = _experiment(cfg)
    kept = tuple(m for m in experiment.models if m.label != "persistence")
    without = replace(experiment, models=kept)
    with pytest.raises(ValueError, match="every skill score needs its persistence"):
        score_experiment(without, small_corpus, results_dir=tmp_path)


def test_scoring_refuses_an_unknown_regime(small_corpus: Path, tmp_path: Path) -> None:
    from dmf.config import load_data

    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(base, lookback=50, horizons=HORIZONS, stride=25)
    with pytest.raises(ValueError, match="unknown regime"):
        score_experiment(_experiment(cfg), small_corpus, results_dir=tmp_path, regimes=("id_ish",))
