"""Metric definitions and evaluation integrity -- Gate 6, and the Phase 3 eval path.

Covered here:

- Skill score reproduced by hand on one worked example, equal to 0 **bitwise** when the
  model's predictions equal the persistence baseline's, and undefined (raising) when the
  denominator is zero.
- The horizon convention: "horizon h" is the error at lead time exactly ``h`` samples,
  never the mean over lead times 1..h. An off-by-one or a cumulative reading here would
  make every reported number a different quantity from the P2-D9 persistence table.
- ``per_dof_horizon_metrics`` and ``metrics_table_from_sums`` are the same computation, so
  the streaming path used in production is checked against the array path used in tests.
- One evaluation pass scores every model on identical windows, and the skill denominator of
  every model is the reference model's accumulator over those same windows.
- Errors are attributed to the right realization: the accumulator's key mapping is checked
  against :meth:`dmf.data.dataset.DeckMotionDataset.describe_window` rather than assumed.
- Per-cell sums marginalise back to the pooled numbers exactly, so the cell breakdown and
  the headline table cannot disagree.
- The bootstrap resamples **realizations**, not windows, and brackets the point estimate.
- ``aggregate_over_seeds`` refuses a group with fewer than three seeds; ``aggregate_results``
  routes deterministic rows around it with ``n_seeds = 1`` and a **NaN** std.

Controls, from the validation protocol:

- **Shuffle-label control.** The protocol's stated null ("skill collapses to ~0") is wrong
  for this task: a model fitted to time-shuffled targets degenerates to the conditional
  mean, which inverts to the window mean, and the window mean *beats* persistence at long
  horizons on a narrowband signal. ``test_the_window_mean_beats_persistence_at_long_horizon``
  measures that on the simulated corpus; the control therefore tests against the window
  mean and is exercised in both directions here.
- **Untrained-model control.** A randomly initialised model must score worse than
  persistence.
- **Pipeline sanity.** Owned by ``tests/test_windows.py``; this module only checks that
  supplying a model to the control leaves the Phase 2 default path byte-identical.

Units: degrees for roll and pitch, metres for heave, samples for horizons.
"""

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from dmf.config import DataConfig, ObservationMode, load_data
from dmf.data.dataset import DeckMotionDataset, resolve_columns
from dmf.data.normalize import is_train_partition
from dmf.data.splits import Regime, build_split
from dmf.data.windows import window_spec_from_config
from dmf.eval.controls import persistence_pipeline_sanity, shuffle_control, untrained_control
from dmf.eval.metrics import (
    mae,
    metrics_table_from_sums,
    per_dof_horizon_metrics,
    rmse,
    skill_score,
)
from dmf.eval.report import (
    BASELINES_COLUMNS,
    aggregate_over_seeds,
    aggregate_results,
    build_baselines_markdown,
    build_baselines_table,
    to_markdown,
    write_table,
)
from dmf.eval.runner import (
    bootstrap_skill_ci,
    evaluate_models,
    marginalize_cells,
    per_cell_metrics,
)

#: Horizons reported by the small-corpus fixture geometry, samples.
SMALL_REPORTED_HORIZONS: tuple[int, ...] = (5, 10, 20)

#: The Gate 3 horizon, samples: "0.8 skill at 3 s on roll" (``docs/IMPLEMENTATION_PLAN.md``)
#: at the corpus 10 Hz. This is a property of the gate, not of the data config, so it is
#: written here -- but ``test_the_gate_cell_exists_in_the_production_geometry`` checks it
#: against the config's ``fs_hz`` and ``horizons`` rather than assuming they agree.
GATE_HORIZON_SAMPLES = 30

#: Sampling rate of the corpus, hertz.
FS_HZ = 10.0


def _production_data_cfg() -> DataConfig:
    """Load the task definition the Gate 3 report is rendered from.

    Returns:
        ``configs/data/default.yaml`` as a :class:`dmf.config.DataConfig`. Loaded, never
        mirrored: ``target_dofs`` and ``horizons`` have been revised once already, and the
        test files that hand-copied them broke.
    """
    return load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")


# ---------------------------------------------------------------------------
# Test doubles. Real models live in src/dmf/models/ and are read-only to this
# module's owner; these stand in for them so that the evaluation path is tested
# for what it does with a forecast, not for how the forecast was produced.
# ---------------------------------------------------------------------------


class _Persistence(nn.Module):
    """The inline persistence expression the Gate 2 control validated."""

    def __init__(self, max_horizon: int, n_targets: int) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.n_targets = n_targets

    def forward(self, x: Tensor) -> Tensor:
        return x[:, -1:, : self.n_targets].expand(-1, self.max_horizon, -1)


class _WindowMean(nn.Module):
    """Forecast zero in dimensionless space, i.e. the window mean in corpus units."""

    def __init__(self, max_horizon: int, n_targets: int) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.n_targets = n_targets

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros(
            (x.shape[0], self.max_horizon, self.n_targets), dtype=x.dtype, device=x.device
        )


class _Scaled(nn.Module):
    """Persistence scaled by a constant, a deliberately worse or better forecaster."""

    def __init__(self, max_horizon: int, n_targets: int, factor: float) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.n_targets = n_targets
        self.factor = factor

    def forward(self, x: Tensor) -> Tensor:
        base = x[:, -1:, : self.n_targets].expand(-1, self.max_horizon, -1)
        return base * self.factor


class _Oracle(nn.Module):
    """A model that cheats: it is handed the answer at construction time."""

    def __init__(self, max_horizon: int, n_targets: int, answer: Tensor) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.n_targets = n_targets
        self.register_buffer("answer", answer)

    def forward(self, x: Tensor) -> Tensor:
        del x
        return self.answer


def _dataset(
    corpus_root: Path, manifest: pd.DataFrame, cfg: DataConfig, regime: Regime, partition: str
) -> DeckMotionDataset:
    """Build one partition's dataset, propagating train statistics where required."""
    split = build_split(manifest, regime)
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(corpus_root, split, "train", cfg, spec)
    if partition == "train":
        return train
    return DeckMotionDataset(corpus_root, split, partition, cfg, spec, stats=train.norm_stats)


def _oracle_answer(dataset: DeckMotionDataset) -> Tensor:
    """Return the exact normalised targets of every window, in loader order.

    Used to build a model that cheats. A negative control must be shown to fire, and the
    only way to be certain it fires is to hand a model the answer.
    """
    truths = torch.stack([dataset[i][1] for i in range(len(dataset))])
    means = torch.stack([dataset[i][2] for i in range(len(dataset))])
    scale = torch.as_tensor(
        dataset.norm_stats.subset(dataset.target_columns).scale, dtype=torch.float32
    )
    return (truths - means) / scale


def _baseline_models(dataset: DeckMotionDataset) -> dict[str, nn.Module]:
    """Return the persistence and window-mean doubles sized for a dataset."""
    horizon = dataset.window_spec.max_horizon
    targets = len(dataset.target_columns)
    return {
        "persistence": _Persistence(horizon, targets),
        "window_mean": _WindowMean(horizon, targets),
    }


# ---------------------------------------------------------------------------
# Skill score
# ---------------------------------------------------------------------------


def test_skill_score_matches_a_hand_computed_example() -> None:
    got = skill_score(np.asarray([1.0, 4.0]), np.asarray([4.0, 2.0]))
    assert got == pytest.approx([0.75, -1.0])


def test_skill_score_is_exactly_zero_against_itself() -> None:
    """Not ``approx``: the reference model's own row must be bitwise 0.0."""
    reference = np.asarray([3.7, 0.001, 1e9])
    assert np.all(skill_score(reference, reference) == 0.0)


def test_skill_score_is_negative_and_unclipped_when_worse_than_persistence() -> None:
    got = skill_score(np.asarray([100.0]), np.asarray([1.0]))
    assert got[0] == pytest.approx(-99.0)


def test_skill_score_raises_on_a_zero_denominator() -> None:
    with pytest.raises(ValueError, match="contains a zero"):
        skill_score(np.asarray([1.0, 1.0]), np.asarray([1.0, 0.0]))


def test_skill_score_raises_on_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same windows"):
        skill_score(np.zeros((2, 3)), np.ones((2, 4)))


# ---------------------------------------------------------------------------
# RMSE, MAE, and the horizon convention
# ---------------------------------------------------------------------------


def test_rmse_and_mae_on_a_worked_example() -> None:
    pred = np.zeros((2, 1, 1))
    target = np.asarray([[[3.0]], [[4.0]]])
    assert float(rmse(pred, target)) == pytest.approx(np.sqrt(12.5))
    assert float(mae(pred, target)) == pytest.approx(3.5)


def test_mae_raises_on_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        mae(np.zeros((2, 3, 1)), np.zeros((2, 4, 1)))


def test_horizon_metric_is_per_step_not_cumulative() -> None:
    """Horizon h is the error at lead time exactly h, never the mean over 1..h.

    The array below has a step error that grows with lead time. If a horizon metric were
    cumulative, the reported value at h=3 would be the RMS of {1, 2, 3} = 2.16; per-step it
    is exactly 3.
    """
    per_step = np.asarray([1.0, 2.0, 3.0])[:, None]
    pred = np.zeros((1, 3, 1))
    target = per_step[None, :, :]
    table = per_dof_horizon_metrics(
        pred=pred,
        target=target,
        persistence_pred=np.zeros((1, 3, 1)),
        dof_names=("roll",),
        horizons=(1, 2, 3),
        fs_hz=FS_HZ,
    )
    assert table["rmse"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    cumulative = [np.sqrt(np.mean(per_step[: h + 1] ** 2)) for h in range(3)]
    assert table["rmse"].tolist() != pytest.approx(cumulative)


def test_horizon_seconds_come_from_the_sampling_rate() -> None:
    table = metrics_table_from_sums(
        sse=np.full((50, 1), 2.0),
        sae=np.full((50, 1), 1.0),
        sse_persistence=np.full((50, 1), 4.0),
        n=2,
        dof_names=("roll",),
        horizons=(10, 30, 50),
        fs_hz=FS_HZ,
    )
    assert table["horizon_s"].tolist() == pytest.approx([1.0, 3.0, 5.0])
    assert table["skill"].tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_metrics_table_from_sums_matches_the_array_path(rng: np.random.Generator) -> None:
    """F3: the streaming path and the array path must be the same computation."""
    pred = rng.normal(size=(64, 8, 3))
    target = rng.normal(size=(64, 8, 3))
    reference = rng.normal(size=(64, 8, 3))
    array_path = per_dof_horizon_metrics(
        pred, target, reference, ("roll", "pitch", "heave"), (1, 4, 8), FS_HZ
    )
    sums_path = metrics_table_from_sums(
        sse=np.square(pred - target).sum(axis=0),
        sae=np.abs(pred - target).sum(axis=0),
        sse_persistence=np.square(reference - target).sum(axis=0),
        n=64,
        dof_names=("roll", "pitch", "heave"),
        horizons=(1, 4, 8),
        fs_hz=FS_HZ,
    )
    pd.testing.assert_frame_equal(array_path, sums_path)


def test_metrics_table_rejects_a_horizon_beyond_the_forecast() -> None:
    with pytest.raises(ValueError, match=r"outside \[1, 8\]"):
        metrics_table_from_sums(
            np.ones((8, 1)), np.ones((8, 1)), np.ones((8, 1)), 1, ("roll",), (9,), FS_HZ
        )


def test_per_dof_horizon_metrics_rejects_a_different_persistence_window_set() -> None:
    with pytest.raises(ValueError, match="flattering"):
        per_dof_horizon_metrics(
            pred=np.zeros((4, 2, 1)),
            target=np.zeros((4, 2, 1)),
            persistence_pred=np.zeros((3, 2, 1)),
            dof_names=("roll",),
            horizons=(1,),
            fs_hz=FS_HZ,
        )


# ---------------------------------------------------------------------------
# The single-pass runner
# ---------------------------------------------------------------------------


def test_evaluate_models_scores_the_reference_as_exactly_zero_skill(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    models = _baseline_models(dataset)
    table, accumulators = evaluate_models(
        models,
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=64,
    )
    own = table.loc[table["model"] == "persistence", "skill"].to_numpy()
    assert np.all(own == 0.0)
    assert set(accumulators) == set(models)
    assert accumulators["persistence"].n_windows == len(dataset)


def test_every_model_is_scored_over_identical_windows(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The point of the single pass: one denominator, structurally, for every model."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    models = _baseline_models(dataset)
    models["scaled"] = _Scaled(dataset.window_spec.max_horizon, len(dataset.target_columns), 0.5)
    table, _ = evaluate_models(
        models,
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=64,
    )
    per_model = table.groupby("model")["rmse_persistence"].nunique()
    assert (per_model == len(SMALL_REPORTED_HORIZONS) * len(dataset.target_columns)).all()
    pivot = table.pivot_table(
        index=["dof", "horizon_samples"], columns="model", values="rmse_persistence"
    )
    for name in models:
        assert np.allclose(pivot[name].to_numpy(), pivot["persistence"].to_numpy())
    assert table["n_windows"].nunique() == 1


def test_runner_rmse_matches_the_pipeline_sanity_control(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Two independent accumulation paths over the same windows must agree.

    Not bitwise: the control inverts the normalisation in float32 and widens afterwards,
    while the runner widens first and inverts in float64. The gap is the float32 storage
    floor of order ``2**-24``, the same quantity P2-D9 measured as ``max_rel_diff =
    5.006e-08``, and the runner is the more accurate of the two. Anything larger than that
    floor would mean the two paths are computing different quantities.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    table, _ = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    control = persistence_pipeline_sanity(dataset, small_corpus)
    rows = table[table["model"] == "persistence"]
    for _, row in rows.iterrows():
        channel = list(dataset.target_columns).index(str(row["dof"]))
        step = int(row["horizon_samples"]) - 1
        assert float(row["rmse"]) == pytest.approx(
            float(control.rmse_pipeline[step, channel]), rel=1e-6
        )


def test_evaluate_models_requires_the_persistence_denominator(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    with pytest.raises(ValueError, match="non-negotiable 4"):
        evaluate_models(
            _baseline_models(dataset),
            dataset,
            persistence_key="not_a_model",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=256,
            num_workers=0,
        )


def _score(dataset: DeckMotionDataset) -> pd.DataFrame:
    """Run the scoring pass on ``dataset`` with the standard baseline model set.

    Args:
        dataset: The partition to score.

    Returns:
        The runner's metric table.
    """
    table, _ = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    return table


def _relabelled_id_test(
    corpus_root: Path, manifest: pd.DataFrame, cfg: DataConfig, fitted_on: str
) -> DeckMotionDataset:
    """Build an ``id/test`` dataset whose statistics carry an arbitrary provenance label.

    The relabelling uses ``dataclasses.replace`` on the frozen
    :class:`dmf.data.normalize.NormStats` rather than a pipeline, because there is no
    pipeline that produces these labels: ``dmf.data.normalize.build_norm_stats`` refuses to
    construct statistics that are not train-fitted. Everything else here is the public
    path, and the fact that the dataset constructor accepts the result without complaint is
    the point -- it is why the check has to live in the scoring pass.

    Args:
        corpus_root: Fixture corpus root.
        manifest: Fixture corpus metadata.
        cfg: Channel and window settings.
        fitted_on: Provenance label to stamp onto the statistics.

    Returns:
        The ``id`` regime's test partition, scaled by relabelled ``id`` training statistics.
    """
    split = build_split(manifest, "id")
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(corpus_root, split, "train", cfg, spec)
    relabelled = dataclasses.replace(train.norm_stats, fitted_on=fitted_on)
    return DeckMotionDataset(corpus_root, split, "test", cfg, spec, stats=relabelled)


def test_evaluate_models_accepts_statistics_from_its_own_training_split(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The legitimate path must stay open: this is how every caller builds a test set.

    ``dmf.train.experiment.run_experiment`` and every other caller pass
    ``train.norm_stats`` from the same regime's split, which is exactly what
    :func:`_dataset` does.
    """
    for regime in ("id", "unseen_seastate", "unseen_heading", "unseen_vessel"):
        dataset = _dataset(small_corpus, small_manifest, small_data_cfg, regime, "test")
        assert dataset.norm_stats.fitted_on == f"{regime}/train"
        assert not _score(dataset).empty


def test_evaluate_models_refuses_statistics_fitted_on_another_regime(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Cross-regime statistics: a real dataset, not a mutated label.

    ``unseen_vessel/test`` is built here carrying genuine ``id/train`` statistics. The
    dataset constructor accepts this -- it demands *some* train statistics for a held-out
    partition and cannot know which regime the caller meant -- so before this guard the
    pass would have scored and reported numbers. Note the sharper instance of the same
    hole, which needs no second corpus to be dangerous: ``id``'s training pool spans SS6
    and the beam heading, so ``id/train`` statistics on ``unseen_seastate/test`` or
    ``unseen_heading/test`` are fitted over the very variance those regimes hold out.
    """
    spec = window_spec_from_config(small_data_cfg)
    id_train = DeckMotionDataset(
        small_corpus, build_split(small_manifest, "id"), "train", small_data_cfg, spec
    )
    foreign = DeckMotionDataset(
        small_corpus,
        build_split(small_manifest, "unseen_vessel"),
        "test",
        small_data_cfg,
        spec,
        stats=id_train.norm_stats,
    )
    assert foreign.regime == "unseen_vessel"
    assert foreign.norm_stats.fitted_on == "id/train"
    with pytest.raises(ValueError, match="non-negotiable 3"):
        _score(foreign)


def test_evaluate_models_refuses_statistics_not_fitted_on_a_training_partition(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The label is replaced after construction, because there is no honest way to get it.

    ``dmf.data.normalize.build_norm_stats`` refuses to construct statistics whose
    ``fitted_on`` is not a training partition, so no real pipeline can produce this object
    -- ``dataclasses.replace`` on the frozen dataclass is the only route, the same
    smuggling technique ``tests/test_splits.py`` uses on ``apply_norm``. The guard exists
    for the paths that bypass the constructor: a checkpoint rebuilt field by field, or code
    that relabels statistics to make them fit.
    """
    dataset = _relabelled_id_test(small_corpus, small_manifest, small_data_cfg, "id/test")
    with pytest.raises(ValueError, match="not a training partition"):
        _score(dataset)


def test_evaluate_models_refuses_a_provenance_label_with_no_regime(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """A bare ``"train"`` label passes ``is_train_partition`` but records no regime.

    Nothing in ``src/`` produces one -- ``DeckMotionDataset`` writes ``f"{regime}/train"``
    -- but ``fitted_on`` is a free-form string on the public ``NormStats`` constructor, so
    the label is again replaced after construction to reach this branch. It fails closed:
    a guard that a relabelling can switch off is not a guard.
    """
    assert is_train_partition("train")
    dataset = _relabelled_id_test(small_corpus, small_manifest, small_data_cfg, "train")
    with pytest.raises(ValueError, match="does not have the '<regime>/train' shape"):
        _score(dataset)


def test_accumulator_attributes_error_to_the_right_realization(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The per-key mapping is checked against the dataset's own provenance, not assumed.

    If ``key_index = idx // windows_per_realization`` were wrong, every per-cell number and
    every bootstrap interval would be attributed to the wrong realization while the pooled
    totals stayed correct -- a failure that is invisible in the headline table.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    per_realization = dataset.windows_per_realization
    for index in (0, 1, per_realization - 1, per_realization, len(dataset) - 1):
        key, _ = dataset.describe_window(index)
        assert dataset.realization_keys[index // per_realization] == key
    _, accumulators = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=64,
        num_workers=0,
        n_boot=8,
    )
    counts = accumulators["persistence"].n_per_key
    assert torch.equal(counts, torch.full_like(counts, per_realization))


def test_per_cell_metrics_marginalise_back_to_the_pooled_table(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The cell breakdown and the headline table are the same sums, regrouped."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    table, accumulators = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    cells = per_cell_metrics(
        accumulators,
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
    )
    pooled = marginalize_cells(cells, ())
    merged = table.merge(pooled, on=["model", "dof", "horizon_samples"], suffixes=("", "_cell"))
    assert len(merged) == len(table)
    assert np.allclose(merged["rmse"], merged["rmse_cell"], rtol=1e-12)
    assert np.allclose(merged["skill"], merged["skill_cell"], rtol=1e-12, atol=1e-15)
    assert merged["n_windows"].equals(merged["n_windows_cell"])


def test_per_cell_metrics_split_the_corpus_grid(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    _, accumulators = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    cells = per_cell_metrics(
        accumulators,
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        by=("heading_deg",),
    )
    assert set(cells["heading_deg"].unique()) == {135.0, 90.0, 45.0}
    assert int(cells["n_windows"].sum()) == len(dataset) * len(accumulators) * len(
        SMALL_REPORTED_HORIZONS
    ) * len(dataset.target_columns)


def test_per_cell_metrics_rejects_an_unknown_axis(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    _, accumulators = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    with pytest.raises(ValueError, match="unknown cell axes"):
        per_cell_metrics(
            accumulators,
            dataset,
            persistence_key="persistence",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            by=("seed",),
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_resamples_realizations_and_brackets_the_point_estimate(
    rng: np.random.Generator,
) -> None:
    keys, horizon, channels = 40, 4, 2
    sse_model = torch.from_numpy(np.abs(rng.normal(1.0, 0.2, (keys, horizon, channels))))
    sse_reference = torch.from_numpy(np.abs(rng.normal(2.0, 0.2, (keys, horizon, channels))))
    lo, hi = bootstrap_skill_ci(sse_model, sse_reference, horizons=(1, horizon), n_boot=500, seed=7)
    point = 1.0 - sse_model.sum(0).numpy() / sse_reference.sum(0).numpy()
    for index, step in enumerate((0, horizon - 1)):
        assert np.all(lo[index] <= point[step] + 1e-12)
        assert np.all(hi[index] >= point[step] - 1e-12)
    assert np.all(hi > lo)


def test_bootstrap_is_reproducible_from_its_seed(rng: np.random.Generator) -> None:
    sse_model = torch.from_numpy(np.abs(rng.normal(1.0, 0.2, (12, 3, 1))))
    sse_reference = torch.from_numpy(np.abs(rng.normal(2.0, 0.2, (12, 3, 1))))
    first = bootstrap_skill_ci(sse_model, sse_reference, horizons=(3,), n_boot=200, seed=3)
    second = bootstrap_skill_ci(sse_model, sse_reference, horizons=(3,), n_boot=200, seed=3)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_bootstrap_rejects_a_mismatched_reference() -> None:
    with pytest.raises(ValueError, match="same realizations"):
        bootstrap_skill_ci(torch.ones((4, 2, 1)), torch.ones((3, 2, 1)), horizons=(1,))


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_window_mean_skill_rises_with_horizon(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The direction of the effect behind the corrected shuffle-control null.

    On a narrowband, mean-reverting signal, persistence error saturates near
    ``sqrt(2) * RMS`` while the window-mean forecast saturates near ``1.0 * RMS``, so the
    window mean's skill *rises* with horizon and eventually turns positive. The small-corpus
    fixture only reaches a 2 s horizon against a 12 s roll period, so the crossing itself is
    asserted at the production geometry in
    ``test_real_corpus_window_mean_beats_persistence_at_five_seconds``; here only the
    monotone direction is available, and it is the part that does not depend on geometry.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    table, _ = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
        num_workers=0,
        n_boot=8,
    )
    rows = table[table["model"] == "window_mean"]
    for dof in dataset.target_columns:
        skills = rows[rows["dof"] == dof].sort_values("horizon_samples")["skill"].to_numpy()
        assert np.all(np.diff(skills) > 0.0), (dof, skills)


@pytest.mark.slow
def test_real_corpus_window_mean_beats_persistence_at_five_seconds(
    real_corpus: Path, real_manifest: pd.DataFrame
) -> None:
    """The measurement that makes the protocol's stated shuffle-control null wrong.

    The protocol says shuffle-control skill "must collapse to approximately zero". A model
    fitted to time-shuffled targets degenerates to the conditional mean, which
    :func:`dmf.data.normalize.invert_norm` maps back to the window mean -- and at the
    production 5 s horizon the window mean carries clearly **positive** skill against
    persistence. Testing that model against "skill ~ 0" would report leakage on a clean
    pipeline, which is why :func:`dmf.eval.controls.shuffle_control` tests against the
    window mean instead.
    """
    from dmf.data.splits import Split

    cfg = _production_data_cfg()
    split = build_split(real_manifest, "id")
    subset = Split(
        regime=split.regime,
        train_keys=frozenset(sorted(split.train_keys)[:8]),
        val_keys=split.val_keys,
        test_keys=frozenset(sorted(split.test_keys)[:8]),
    )
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(real_corpus, subset, "train", cfg, spec)
    dataset = DeckMotionDataset(real_corpus, subset, "test", cfg, spec, stats=train.norm_stats)
    table, _ = evaluate_models(
        _baseline_models(dataset),
        dataset,
        persistence_key="persistence",
        horizons=cfg.horizons,
        fs_hz=cfg.fs_hz,
        batch_size=2048,
        num_workers=0,
        n_boot=8,
    )
    # Derived from fs_hz rather than written as 50 and 10: the horizon list has been
    # revised once already, and the claim is about 5 s and 1 s, not about two integers.
    five_s, one_s = int(round(5.0 * cfg.fs_hz)), int(round(1.0 * cfg.fs_hz))
    assert {five_s, one_s} <= set(cfg.horizons)
    at_five = table[(table["model"] == "window_mean") & (table["horizon_samples"] == five_s)]
    assert float(at_five["skill"].min()) > 0.0, at_five
    at_one = table[(table["model"] == "window_mean") & (table["horizon_samples"] == one_s)]
    assert float(at_one["skill"].max()) < 0.0, at_one


def test_shuffle_control_passes_when_the_subject_is_the_window_mean(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    result = shuffle_control(
        dataset,
        shuffled_model=_WindowMean(horizon, targets),
        window_mean_model=_WindowMean(horizon, targets),
        persistence_model=_Persistence(horizon, targets),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert result.worst_excess == pytest.approx(0.0, abs=1e-12)
    assert set(result.rows["control"]) == {"shuffle"}


def test_shuffle_control_catches_a_subject_that_still_knows_the_future(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The converse: a control that cannot fail is not a control."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    with pytest.raises(AssertionError, match="shuffle control failed"):
        shuffle_control(
            dataset,
            shuffled_model=_Oracle(horizon, targets, _oracle_answer(dataset)),
            window_mean_model=_WindowMean(horizon, targets),
            persistence_model=_Persistence(horizon, targets),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=len(dataset),
        )


def test_untrained_control_catches_a_model_that_beats_persistence(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The converse: a control that cannot fail is not a control."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    with pytest.raises(AssertionError, match="untrained control failed"):
        untrained_control(
            dataset,
            untrained_model=_Oracle(horizon, targets, _oracle_answer(dataset)),
            persistence_model=_Persistence(horizon, targets),
            regime="id",
            horizons=SMALL_REPORTED_HORIZONS,
            fs_hz=FS_HZ,
            batch_size=len(dataset),
        )


def test_untrained_control_passes_a_model_worse_than_persistence(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    result = untrained_control(
        dataset,
        untrained_model=_Scaled(horizon, targets, -3.0),
        persistence_model=_Persistence(horizon, targets),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert result.worst_excess < 0.0
    assert bool(result.rows["passed"].all())


def test_untrained_control_accepts_a_window_mean_null(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The protocol's literal criterion is not reachable at 5 s; the null is configurable.

    A small-weight random projection saturates near the window-mean error level, and on the
    real corpus the window mean beats persistence beyond ~2 s. The measured numbers are in
    the :func:`dmf.eval.controls.untrained_control` docstring. Here only the plumbing is
    checked: with a null supplied, the comparison is against the null and the null's own row
    appears in the control's table.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    result = untrained_control(
        dataset,
        untrained_model=_Scaled(horizon, targets, -3.0),
        persistence_model=_Persistence(horizon, targets),
        null_model=_WindowMean(horizon, targets),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        batch_size=256,
    )
    assert result.passed
    assert set(result.rows["null_model"]) == {"null"}
    assert set(result.table["model"]) == {"persistence", "untrained", "null"}
    assert not np.allclose(result.rows["skill_null"].to_numpy(), 0.0)


def test_control_excess_is_a_variance_ratio_not_a_skill_difference(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The control statistic must mean the same thing in every cell.

    ``excess`` is ``1 - MSE_subject/MSE_null``. A plain ``skill_subject - skill_null`` is
    not scale-free: where both models are far worse than persistence, a 0.5% error
    reduction shows up as a skill difference of 0.02 and trips a tolerance that means
    something entirely different at a cell where skill is near 1. Measured on the real
    corpus, an AR(20) fitted to shuffled targets sits at -3.4591 skill against a
    window-mean null at -3.4827: a difference of +0.0235, but an error reduction of 0.53%.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    horizon, targets = dataset.window_spec.max_horizon, len(dataset.target_columns)
    result = untrained_control(
        dataset,
        untrained_model=_Scaled(horizon, targets, 0.5),
        persistence_model=_Persistence(horizon, targets),
        null_model=_WindowMean(horizon, targets),
        regime="id",
        horizons=SMALL_REPORTED_HORIZONS,
        fs_hz=FS_HZ,
        tol=1.0,
        batch_size=256,
    )
    subject = result.rows["skill_subject"].to_numpy()
    null = result.rows["skill_null"].to_numpy()
    assert np.allclose(result.rows["excess"].to_numpy(), 1.0 - (1.0 - subject) / (1.0 - null))
    assert not np.allclose(result.rows["excess"].to_numpy(), subject - null)


def test_supplying_a_model_leaves_the_default_pipeline_sanity_path_unchanged(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """F8: the Phase 2 default must stay byte-identical, and the model path must agree.

    The model handed in here is the inline expression itself, which is what
    :class:`dmf.models.persistence.Persistence` is documented to compute. The bitwise check
    against the real class lives in ``tests/test_models.py``.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    default = persistence_pipeline_sanity(dataset, small_corpus)
    with_model = persistence_pipeline_sanity(
        dataset,
        small_corpus,
        model=_Persistence(dataset.window_spec.max_horizon, len(dataset.target_columns)),
    )
    assert default.model_matches_inline is None
    assert with_model.model_matches_inline is True
    assert np.array_equal(default.rmse_pipeline, with_model.rmse_pipeline)
    assert default.max_rel_diff == with_model.max_rel_diff
    assert default.n_windows == with_model.n_windows


def test_pipeline_sanity_rejects_a_model_that_is_not_the_inline_expression(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    with pytest.raises(AssertionError, match="not bitwise equal"):
        persistence_pipeline_sanity(
            dataset,
            small_corpus,
            model=_Scaled(dataset.window_spec.max_horizon, len(dataset.target_columns), 1.0000001),
        )


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------


def _seeded_frame(n_seeds: int, deterministic: bool = False) -> pd.DataFrame:
    """Build a minimal per-run frame for the aggregation tests."""
    return pd.DataFrame(
        {
            "model": ["m"] * n_seeds,
            "regime": ["id"] * n_seeds,
            "dof": ["roll"] * n_seeds,
            "horizon_samples": [30] * n_seeds,
            "horizon_s": [3.0] * n_seeds,
            "seed": list(range(n_seeds)),
            "deterministic": [deterministic] * n_seeds,
            "n_windows": [100] * n_seeds,
            "rmse": [1.0 + 0.1 * i for i in range(n_seeds)],
            "mae": [0.5] * n_seeds,
            "skill": [0.8] * n_seeds,
            "rmse_persistence": [2.0] * n_seeds,
            "skill_ci_lo": [0.7] * n_seeds,
            "skill_ci_hi": [0.9] * n_seeds,
            "n_params": [0] * n_seeds,
            "fit_time_s": [1.0] * n_seeds,
        }
    )


def test_aggregate_over_seeds_refuses_fewer_than_three_seeds() -> None:
    with pytest.raises(ValueError, match="non-negotiable 5"):
        aggregate_over_seeds(_seeded_frame(2), ("model", "regime", "dof", "horizon_samples"))


def test_aggregate_over_seeds_reports_mean_and_std() -> None:
    out = aggregate_over_seeds(_seeded_frame(3), ("model", "regime", "dof", "horizon_samples"))
    assert len(out) == 1
    assert float(out["rmse_mean"].iloc[0]) == pytest.approx(1.1)
    assert float(out["rmse_std"].iloc[0]) == pytest.approx(0.1)
    assert int(out["n_seeds"].iloc[0]) == 3


def test_aggregate_results_passes_deterministic_rows_through_with_a_nan_std() -> None:
    """NaN, not 0.0: with one observation the sample std is undefined."""
    out = aggregate_results(
        _seeded_frame(1, deterministic=True),
        ("model", "regime", "dof", "horizon_samples"),
        metric_cols=("rmse", "mae", "skill"),
    )
    assert len(out) == 1
    assert int(out["n_seeds"].iloc[0]) == 1
    assert np.isnan(float(out["rmse_std"].iloc[0]))
    assert float(out["rmse_std"].iloc[0]) != 0.0
    assert float(out["rmse_mean"].iloc[0]) == pytest.approx(1.0)


def test_aggregate_results_still_refuses_a_thin_stochastic_group() -> None:
    frame = pd.concat(
        [_seeded_frame(1, deterministic=True), _seeded_frame(2).assign(model="sgd")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="non-negotiable 5"):
        aggregate_results(
            frame,
            ("model", "regime", "dof", "horizon_samples"),
            metric_cols=("rmse", "mae", "skill"),
        )


def test_aggregate_results_requires_the_deterministic_flag() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        aggregate_results(
            _seeded_frame(3).drop(columns=["deterministic"]),
            ("model", "regime", "dof", "horizon_samples"),
            metric_cols=("rmse",),
        )


def test_aggregate_results_refuses_a_deterministic_model_that_moved() -> None:
    frame = _seeded_frame(2, deterministic=True)
    frame.loc[1, "skill"] = 0.5
    with pytest.raises(ValueError, match="more than one distinct"):
        aggregate_results(
            frame,
            ("model", "regime", "dof", "horizon_samples"),
            metric_cols=("skill",),
        )


def test_build_baselines_table_emits_the_gate_schema() -> None:
    frame = pd.concat(
        [
            _seeded_frame(1, deterministic=True),
            _seeded_frame(3).assign(model="dlinear"),
        ],
        ignore_index=True,
    )
    table = build_baselines_table(frame)
    assert list(table.columns) == list(BASELINES_COLUMNS)
    assert len(table) == 2
    deterministic_row = table[table["model"] == "m"].iloc[0]
    assert bool(deterministic_row["deterministic"])
    assert int(deterministic_row["n_seeds"]) == 1
    assert np.isnan(float(deterministic_row["skill_std"]))
    stochastic_row = table[table["model"] == "dlinear"].iloc[0]
    assert int(stochastic_row["n_seeds"]) == 3
    assert float(stochastic_row["rmse_std"]) == pytest.approx(0.1)


def test_build_baselines_table_rejects_a_moving_persistence_denominator() -> None:
    frame = _seeded_frame(3)
    frame.loc[2, "rmse_persistence"] = 2.5
    with pytest.raises(ValueError, match="varies between the runs"):
        build_baselines_table(frame)


def _gate_table(
    mode: ObservationMode,
    *,
    regime: str = "id",
    drop_dofs: tuple[str, ...] = (),
    drop_horizons: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Build an aggregated table over the **production** DOF x horizon grid.

    The grid is loaded from ``configs/data/default.yaml`` rather than mirrored here. A
    hand-copied geometry in a test file is exactly what silently drifts when the task
    definition is revised, so the only thing this function knows is how to turn whatever
    the config says into rows.

    Args:
        mode: Observation mode whose spellings the ``dof`` column carries. ``ideal`` gives
            ``roll``; ``imu`` gives ``roll_imu``, via the same
            :func:`dmf.data.dataset.resolve_columns` the dataset uses.
        regime: Regime label written into every row.
        drop_dofs: Logical DOF names to omit, for the genuinely-missing-cell tests.
        drop_horizons: Horizons to omit, samples, likewise.

    Returns:
        A table with the exact :data:`BASELINES_COLUMNS` schema.
    """
    cfg = _production_data_cfg()
    dofs = tuple(d for d in cfg.target_dofs if d not in drop_dofs)
    horizons = tuple(h for h in cfg.horizons if h not in drop_horizons)
    rows = [
        _seeded_frame(1, deterministic=True).assign(
            model="persistence",
            regime=regime,
            dof=column,
            horizon_samples=horizon,
            horizon_s=horizon / cfg.fs_hz,
        )
        for column, horizon in itertools.product(resolve_columns(dofs, mode), horizons)
    ]
    return build_baselines_table(pd.concat(rows, ignore_index=True))


def _gate_section(rendered: str) -> list[str]:
    """Return the rendered rows of the "The gate cell" table only.

    Args:
        rendered: The whole ``baselines.md`` document.

    Returns:
        The Markdown table lines of that one section. The full per-DOF table later in the
        document does carry a ``dof`` column, so it must be excluded from any comparison
        across observation modes.
    """
    lines = rendered.splitlines()
    start = lines.index("## The gate cell") + 1
    rest = lines[start:]
    stop = next((i for i, line in enumerate(rest) if line.startswith("##")), len(rest))
    return [line for line in rest[:stop] if line.startswith("|")]


def test_the_gate_cell_exists_in_the_production_geometry() -> None:
    """Guards the premise of every test below: 3 s on roll is a cell the config reports."""
    cfg = _production_data_cfg()
    assert "roll" in cfg.target_dofs
    assert GATE_HORIZON_SAMPLES in cfg.horizons
    assert GATE_HORIZON_SAMPLES / cfg.fs_hz == pytest.approx(3.0)


@pytest.mark.parametrize("mode", ["ideal", "imu"])
def test_build_baselines_markdown_finds_the_gate_cell_in_either_spelling(
    mode: ObservationMode,
) -> None:
    """`observation_mode: imu` renames roll to roll_imu; the gate is the same cell.

    Matching on a hardcoded ``"roll"`` used to abort an otherwise complete ``imu`` run at
    its last step, after the CSVs had already been written correctly.
    """
    expected = resolve_columns(("roll",), mode)[0]
    rendered = build_baselines_markdown(_gate_table(mode))
    assert f"Gate cell: **{expected} at 3 s" in rendered
    assert "## The gate cell" in rendered
    # The document names the spelling that was actually scored, not the requested one.
    assert f"Gate cell: **{expected} at 3 s" in build_baselines_markdown(
        _gate_table(mode), gate_dof="roll"
    )


def test_build_baselines_markdown_renders_the_same_gate_row_either_way() -> None:
    """Resolving the spelling must not change which numbers are reported.

    The gate table carries no ``dof`` column -- the DOF is named in the heading text -- so
    the rendered rows must be byte-identical between the two observation modes.
    """
    ideal = _gate_section(build_baselines_markdown(_gate_table("ideal")))
    imu = _gate_section(build_baselines_markdown(_gate_table("imu")))
    assert ideal == imu
    assert len(ideal) > 2


@pytest.mark.parametrize("mode", ["ideal", "imu"])
def test_build_baselines_markdown_still_raises_when_the_gate_cell_is_missing(
    mode: ObservationMode,
) -> None:
    """The failure mode removed is "correct run, wrong spelling", not "cell missing".

    Neither spelling of roll present, the gate horizon not reported, or the gate regime
    not scored: each is a real gap in the run and each must still be fatal.
    """
    with pytest.raises(ValueError, match="absent from the table"):
        build_baselines_markdown(_gate_table(mode, drop_dofs=("roll",)))
    with pytest.raises(ValueError, match="absent from the table"):
        build_baselines_markdown(_gate_table(mode, drop_horizons=(GATE_HORIZON_SAMPLES,)))
    with pytest.raises(ValueError, match="absent from the table"):
        build_baselines_markdown(_gate_table(mode, regime="unseen_vessel"))


def test_build_baselines_markdown_does_not_silently_fall_back_to_another_dof() -> None:
    """Roll absent must raise, not quietly report a neighbouring DOF under its heading.

    ``roll_rate_imu`` is the dangerous near-miss: a prefix or substring rule would take it
    for roll. Resolution goes through the explicit ideal/imu pairing instead, so it does
    not.
    """
    table = _gate_table("imu", drop_dofs=("roll",))
    assert {"pitch_imu", "roll_rate_imu"} <= set(table["dof"])
    with pytest.raises(ValueError, match="absent from the table"):
        build_baselines_markdown(table)


@pytest.mark.parametrize("mode", ["ideal", "imu"])
def test_build_baselines_markdown_resolves_every_production_dof(mode: ObservationMode) -> None:
    """Six DOFs, six horizons: the resolver must address any of them, not only roll."""
    cfg = _production_data_cfg()
    table = _gate_table(mode)
    for dof in cfg.target_dofs:
        expected = resolve_columns((dof,), mode)[0]
        rendered = build_baselines_markdown(table, gate_dof=dof)
        assert f"Gate cell: **{expected} at 3 s" in rendered
        # The imu spelling is equally acceptable as the request.
        assert rendered == build_baselines_markdown(table, gate_dof=expected)


def test_build_baselines_markdown_rejects_a_dof_that_is_not_a_corpus_channel() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        build_baselines_markdown(_gate_table("ideal"), gate_dof="yaw")


def test_to_markdown_renders_nan_as_not_measured() -> None:
    rendered = to_markdown(pd.DataFrame({"skill_std": [np.nan], "ok": [True]}))
    assert "n/a" in rendered
    assert "yes" in rendered
    assert rendered.splitlines()[1] == "|---|---|"


def test_write_table_refuses_an_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty table"):
        write_table(pd.DataFrame({"a": []}), tmp_path / "empty.csv")


def test_write_table_creates_its_parent_directory(tmp_path: Path) -> None:
    path = write_table(pd.DataFrame({"a": [1]}), tmp_path / "nested" / "t.csv")
    assert path.exists()
    assert pd.read_csv(path)["a"].tolist() == [1]
