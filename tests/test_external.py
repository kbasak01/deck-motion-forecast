"""Smoke and integrity tests for :mod:`dmf.eval.external`.

These check the properties that are silent in the output if they are wrong -- the
normalisation provenance guard, the persistence denominator being recomputed on the
external record, and the reference model's bitwise-zero self-skill -- plus that the module
runs end to end on a synthetic record of corpus shape. The full MSS evaluation is not run
here.

Units in the fixtures below follow the corpus convention: degrees for angles, degrees per
second for angular rates, metres for heave, metres per second for heave rate.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset, resolve_columns
from dmf.data.normalize import NormStats, build_norm_stats
from dmf.data.splits import RealizationKey, Split, build_split, load_manifest
from dmf.data.windows import WindowSpec, window_spec_from_config
from dmf.eval.external import (
    EXTERNAL_COLUMNS,
    EXTERNAL_QUIESCENCE_COLUMNS,
    assert_corpus_units,
    corpus_train_norm_stats,
    evaluate_quiescence_trajectories,
    evaluate_trajectories,
)
from dmf.eval.quiescence import PERMISSIVE, STRICT, QuiescenceThresholds
from dmf.eval.quiescence_runner import ALWAYS_QUIESCENT, CELL_LEVEL, evaluate_quiescence
from dmf.eval.runner import evaluate_models
from dmf.models.base import BaseForecaster
from dmf.models.persistence import Persistence
from dmf.sim.generate import RealizationSpec, realization_path

CHANNELS: tuple[str, ...] = ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate")
TARGETS: tuple[str, ...] = CHANNELS
HORIZONS: tuple[int, ...] = (5, 10)
FS_HZ: float = 10.0


def _spec() -> WindowSpec:
    """Return a small window geometry for the synthetic record.

    Returns:
        Lookback 40 samples (4 s at 10 Hz), horizons 5 and 10 samples, stride 20.
    """
    return WindowSpec(lookback=40, horizons=HORIZONS, stride=20)


def _frame(seed: int, n_samples: int = 6000, label: str | None = None) -> pd.DataFrame:
    """Build a synthetic narrowband record in corpus units.

    Args:
        seed: RNG seed for the phase offsets.
        n_samples: Record length, samples, at ``FS_HZ``.
        label: Optional record label stored in ``frame.attrs["record"]``.

    Returns:
        A frame of ``n_samples`` rows and the six corpus motion columns, angles in degrees,
        rates in degrees per second, heave in metres, heave rate in metres per second.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / FS_HZ
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(CHANNELS):
        omega = 2.0 * np.pi * (0.08 + 0.02 * i)
        amplitude = 4.0 - 0.4 * i
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        columns[name] = amplitude * np.sin(omega * t + phase) + 0.05 * rng.standard_normal(
            n_samples
        )
    frame = pd.DataFrame(columns)
    if label is not None:
        frame.attrs["record"] = label
    return frame


def _stats(fitted_on: str = "id/train") -> NormStats:
    """Build normalisation statistics with a chosen provenance label.

    Args:
        fitted_on: Provenance label to record.

    Returns:
        Unit-scale statistics over :data:`CHANNELS`, in corpus units.

    Raises:
        ValueError: If ``fitted_on`` is not a training partition, from
            :func:`dmf.data.normalize.build_norm_stats`.
    """
    return build_norm_stats(
        scale=np.full(len(CHANNELS), 2.0, dtype=np.float64),
        channels=CHANNELS,
        fitted_on=fitted_on,
        n_realizations=7,
    )


class _Zero(BaseForecaster):
    """A model that forecasts the window mean, i.e. zero in normalised space."""

    FIT_KIND = "none"

    def _predict(self, x: Tensor) -> Tensor:
        """Return zeros of the output shape.

        Args:
            x: Input windows, ``(B, L, C_in)``, dimensionless.

        Returns:
            Zeros, ``(B, H, C_out)``, dimensionless.
        """
        return torch.zeros(
            (x.shape[0], self.max_horizon, self.n_target_channels), dtype=x.dtype, device=x.device
        )


def _models(spec: WindowSpec) -> dict[str, BaseForecaster]:
    """Build the reference baseline and one trivial competitor.

    Args:
        spec: Window geometry.

    Returns:
        Mapping of results-table label to model.
    """
    shape = (spec.lookback, spec.max_horizon, len(CHANNELS), len(TARGETS))
    return {"persistence": Persistence(*shape), "window_mean": _Zero(*shape)}


def test_evaluate_trajectories_smoke() -> None:
    """The module runs end to end on a 6000 x 6 synthetic record and emits the columns."""
    spec = _spec()
    frames = [_frame(0, label="synthetic_a"), _frame(1, label="synthetic_b")]
    table = evaluate_trajectories(
        _models(spec),
        frames,
        stats=_stats(),
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        horizons=HORIZONS,
        fs_hz=FS_HZ,
    )
    assert list(table.columns) == list(EXTERNAL_COLUMNS)
    assert len(table) == 2 * 2 * len(TARGETS) * len(HORIZONS)
    assert set(table["record"]) == {"synthetic_a", "synthetic_b"}
    assert (table["n_records"] == 2).all()
    assert np.isfinite(table[["rmse", "mae", "rmse_persistence", "skill"]].to_numpy()).all()
    assert (table["rmse"] > 0.0).all()


def test_reference_model_self_skill_is_bitwise_zero() -> None:
    """Persistence scored against itself is exactly 0.0, not merely near it."""
    spec = _spec()
    table = evaluate_trajectories(
        _models(spec),
        [_frame(2)],
        stats=_stats(),
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        horizons=HORIZONS,
        fs_hz=FS_HZ,
    )
    own = table.loc[table["model"] == "persistence", "skill"].to_numpy()
    assert own.size > 0
    assert np.all(own == 0.0)


def test_persistence_denominator_is_the_external_record() -> None:
    """``rmse_persistence`` equals persistence computed directly on the raw record.

    The pipeline-sanity control, applied to the external path: if the denominator were
    carried over from the corpus, or measured over a different window set, this would not
    hold and no skill score in the table would be a skill score.
    """
    spec = _spec()
    frame = _frame(3)
    table = evaluate_trajectories(
        _models(spec),
        [frame],
        stats=_stats(),
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        horizons=HORIZONS,
        fs_hz=FS_HZ,
    )
    series = frame[list(CHANNELS)].to_numpy(dtype=np.float32).astype(np.float64)
    starts = np.arange(0, series.shape[0] - spec.total_length + 1, spec.stride)
    for horizon in HORIZONS:
        last = series[starts + spec.lookback - 1, :]
        future = series[starts + spec.lookback + horizon - 1, :]
        direct = np.sqrt(np.mean(np.square(last - future), axis=0))
        rows = table[(table["model"] == "persistence") & (table["horizon_samples"] == horizon)]
        reported = rows.set_index("dof").loc[list(TARGETS), "rmse_persistence"].to_numpy()
        np.testing.assert_allclose(reported, direct, rtol=1e-6, atol=1e-6)


def test_non_train_stats_are_refused() -> None:
    """Statistics not fitted on a training split are refused loudly (non-negotiable 3)."""
    spec = _spec()
    leaked = NormStats(
        channels=CHANNELS,
        scale=np.full(len(CHANNELS), 2.0, dtype=np.float64),
        fitted_on="external/mss",
        n_realizations=1,
    )
    with pytest.raises(ValueError, match="CORPUS TRAINING SPLIT"):
        evaluate_trajectories(
            _models(spec),
            [_frame(4)],
            stats=leaked,
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=TARGETS,
            horizons=HORIZONS,
            fs_hz=FS_HZ,
        )


def test_missing_persistence_is_refused() -> None:
    """A model set without the reference baseline is refused (non-negotiable 4)."""
    spec = _spec()
    models = _models(spec)
    del models["persistence"]
    with pytest.raises(ValueError, match="persistence_key"):
        evaluate_trajectories(
            models,
            [_frame(5)],
            stats=_stats(),
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=TARGETS,
            horizons=HORIZONS,
            fs_hz=FS_HZ,
        )


def test_targets_must_be_a_prefix_of_inputs() -> None:
    """A non-prefix target ordering is refused, because persistence slices the prefix."""
    spec = _spec()
    with pytest.raises(ValueError, match="prefix"):
        evaluate_trajectories(
            _models(spec),
            [_frame(6)],
            stats=_stats(),
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=("pitch", "roll"),
            horizons=HORIZONS,
            fs_hz=FS_HZ,
        )


def test_radian_record_is_refused_at_the_boundary() -> None:
    """A record still in radians is caught by the unit tripwire (Phase 8 delta 2)."""
    frame = _frame(7)
    radians = frame.copy()
    for name in CHANNELS:
        radians[name] = np.radians(frame[name].to_numpy())
    with pytest.raises(ValueError, match="DEGREES"):
        assert_corpus_units([radians], CHANNELS)
    assert_corpus_units([frame], CHANNELS)


def test_quantile_output_is_reduced_to_the_median() -> None:
    """A rank-4 output is scored through its 0.5 quantile, as the corpus path does."""
    spec = _spec()

    class _Fan(BaseForecaster):
        """A quantile-shaped model whose median is exactly the window mean."""

        FIT_KIND = "none"

        def forward(self, x: Tensor) -> Tensor:
            """Return a fan centred on zero.

            Args:
                x: Input windows, ``(B, L, C_in)``, dimensionless.

            Returns:
                A ``(B, H, C_out, 9)`` fan, dimensionless, whose 0.5 level is zero.
            """
            offsets = torch.linspace(-1.0, 1.0, 9, dtype=x.dtype, device=x.device)
            base = torch.zeros(
                (x.shape[0], self.max_horizon, self.n_target_channels, 9),
                dtype=x.dtype,
                device=x.device,
            )
            return base + offsets

    models = _models(spec)
    models["fan"] = _Fan(spec.lookback, spec.max_horizon, len(CHANNELS), len(TARGETS))
    table = evaluate_trajectories(
        models,
        [_frame(8)],
        stats=_stats(),
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        horizons=HORIZONS,
        fs_hz=FS_HZ,
    )
    fan = table[table["model"] == "fan"].set_index(["dof", "horizon_samples"])["rmse"]
    mean = table[table["model"] == "window_mean"].set_index(["dof", "horizon_samples"])["rmse"]
    np.testing.assert_allclose(fan.to_numpy(), mean.loc[fan.index].to_numpy(), rtol=1e-6)


def test_corpus_train_norm_stats_carries_train_provenance(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """The recovered statistics are labelled ``<regime>/train`` and cover the inputs."""
    cfg = replace(small_data_cfg)
    spec = window_spec_from_config(cfg)
    stats = corpus_train_norm_stats(small_corpus, "unseen_vessel", cfg, spec)
    assert stats.fitted_on == "unseen_vessel/train"
    assert stats.channels == tuple(cfg.input_channels)
    assert np.all(stats.scale > 0.0)


def test_corpus_train_norm_stats_rejects_unknown_regime(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """An unknown regime name is refused rather than silently defaulting."""
    with pytest.raises(ValueError, match="unknown regime"):
        corpus_train_norm_stats(
            small_corpus, "mss", small_data_cfg, window_spec_from_config(small_data_cfg)
        )


def test_external_path_matches_the_corpus_runner_on_a_corpus_realization(
    small_corpus: Path, small_data_cfg: DataConfig
) -> None:
    """Scoring one corpus realization through both paths gives the same numbers.

    The pipeline-sanity control for this module. :mod:`dmf.eval.external` re-implements
    nothing, but it does re-assemble the windowing, the de-meaning, the scaling and the
    inverse transform outside :class:`dmf.data.dataset.DeckMotionDataset`, and an
    off-by-one in the window index or a de-meaning applied over the wrong axis would show
    up as a slightly different RMSE rather than as an error. Feeding one realization's
    Parquet frame to the external path and the same realization's partition to
    :func:`dmf.eval.runner.evaluate_models` is what makes the two comparable rather than
    merely similar.
    """
    cfg = small_data_cfg
    spec = window_spec_from_config(cfg)
    split = build_split(load_manifest(small_corpus), "id")
    train = DeckMotionDataset(small_corpus, split, "train", cfg, spec)
    key = sorted(split.test_keys)[0]
    one = replace(split, test_keys=frozenset({key}))
    test = DeckMotionDataset(small_corpus, one, "test", cfg, spec, stats=train.norm_stats)

    channels = resolve_columns(cfg.input_channels, cfg.observation_mode)
    targets = resolve_columns(cfg.target_dofs, cfg.observation_mode)
    shape = (spec.lookback, spec.max_horizon, len(channels), len(targets))
    models: dict[str, BaseForecaster] = {
        "persistence": Persistence(*shape),
        "window_mean": _Zero(*shape),
    }

    corpus_table, _ = evaluate_models(
        models,
        test,
        persistence_key="persistence",
        horizons=cfg.horizons,
        fs_hz=cfg.fs_hz,
        batch_size=512,
        num_workers=0,
        n_boot=2,
    )
    ss, heading, speed, vessel, seed = key
    path = small_corpus / realization_path(
        RealizationSpec(seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel)
    )
    frame = pd.read_parquet(path, columns=list(channels))
    external_table = evaluate_trajectories(
        models,
        [frame],
        stats=train.norm_stats,
        spec=spec,
        input_channels=channels,
        target_dofs=targets,
        horizons=cfg.horizons,
        fs_hz=cfg.fs_hz,
    )

    index = ["model", "dof", "horizon_samples"]
    columns = ["n_windows", "rmse", "mae", "rmse_persistence", "skill", "signal_std", "nrmse"]
    left = corpus_table.set_index(index).sort_index()[columns]
    right = external_table.set_index(index).sort_index()[columns]
    assert list(left.index) == list(right.index)
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-10, atol=1e-12)


# --- The quiescence path (Phase 8 carry-forward delta 7). ------------------------------
#
# The one property that matters here is that an MSS quiescence number is the *same
# quantity* as a committed corpus quiescence number. That is not establishable by reading
# the code -- `dmf.eval.external` re-assembles the window loop outside
# `DeckMotionDataset`, and an off-by-one in the decision time or a truth mask taken over
# the wrong span would move every F1 while leaving the table looking entirely normal. So
# the same fabricated realization is scored through `evaluate_quiescence` and through
# `evaluate_quiescence_trajectories` and the two tables are compared row by row.

#: Probe limits for the fabricated realization: 1 deg, 1 deg, 1 m/s, sustained 0.5 s. The
#: shape of the shipped sets at round numbers, so nothing here sits near a comparison
#: boundary -- that boundary is covered in ``tests/test_quiescence.py``.
PROBE: QuiescenceThresholds = QuiescenceThresholds(1.0, 1.0, 1.0, 0.5, "probe")

#: Geometry of the fabricated realization: 100 samples, 1 s lookback, 2 s horizon, 0.5 s
#: stride. The production *shape* at a tenth of the size.
Q_LOOKBACK: int = 10
Q_HORIZONS: tuple[int, ...] = (5, 10, 20)
Q_STRIDE: int = 5
Q_SAMPLES: int = 100

#: The hand-placed quiescent run, samples 40..59 inclusive: one interior onset.
Q_ONSET: int = 40
Q_RUN_STOP: int = 60


class _Ramp(BaseForecaster):
    """A deterministic forecaster whose horizon sweeps in and out of the probe limits.

    Not a model under test. The quiescence metric is driven by *onsets*, and the trivial
    baselines cannot produce one at all -- a constant forecast is either in-limit from index
    0 or out of limits throughout -- so a parity test built only on them would compare two
    empty prediction sides. This one is a fixed function of its input, so the corpus path
    and the external path see the identical forecast for the identical window.
    """

    FIT_KIND = "none"

    def _predict(self, x: Tensor) -> Tensor:
        """Return the last observed sample plus a fixed ramp across the horizon.

        Args:
            x: Input windows, ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, ``(B, H, C_out)``, dimensionless.
        """
        ramp = torch.linspace(-3.0, 3.0, self.max_horizon, dtype=x.dtype, device=x.device)
        last = x[:, -1, : self.n_target_channels]
        return last.unsqueeze(1) + ramp.view(1, self.max_horizon, 1)


def _quiescent_block(quiescent: np.ndarray) -> np.ndarray:
    """Build a six-channel record inside the probe limits exactly where told.

    Args:
        quiescent: Per-sample flag, shape ``(n_samples,)``.

    Returns:
        Array ``(n_samples, 6)`` in corpus units -- degrees, degrees per second, metres,
        metres per second -- columns :data:`CHANNELS`. 0.0 where quiescent, 10.0 elsewhere,
        so no sample sits near a threshold.
    """
    inside = np.where(quiescent, 0.0, 10.0).astype(np.float64)
    block = np.zeros((quiescent.size, len(CHANNELS)), dtype=np.float64)
    for name in ("roll", "pitch", "heave_rate"):
        block[:, CHANNELS.index(name)] = inside
    return block


@pytest.fixture
def quiescence_fixture(tmp_path: Path) -> tuple[Path, DataConfig, RealizationKey]:
    """Write a one-realization corpus with exactly one interior quiescent run.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Tuple ``(corpus_root, data_config, realization_key)``.
    """
    key: RealizationKey = ("SS5", 45.0, 0.0, "frigate", 0)
    root = tmp_path / "corpus"
    pattern = np.zeros(Q_SAMPLES, dtype=bool)
    pattern[Q_ONSET:Q_RUN_STOP] = True
    ss, heading, speed, vessel, seed = key
    path = root / realization_path(
        RealizationSpec(seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    block = _quiescent_block(pattern)
    pd.DataFrame({name: block[:, i] for i, name in enumerate(CHANNELS)}).to_parquet(path)
    cfg = DataConfig(
        fs_hz=FS_HZ,
        lookback=Q_LOOKBACK,
        horizons=Q_HORIZONS,
        target_dofs=CHANNELS,
        input_channels=CHANNELS,
        stride=Q_STRIDE,
        observation_mode="ideal",
        revin=False,
        condition_on_sea_state=False,
    )
    return root, cfg, key


def _quiescence_models(spec: WindowSpec) -> dict[str, BaseForecaster]:
    """Build the detector set both paths are scored on.

    Args:
        spec: Window geometry.

    Returns:
        The two constant baselines -- which predict no onset by construction, and are here
        because dropping them would hide that -- and one forecaster that does.
    """
    shape = (spec.lookback, spec.max_horizon, len(CHANNELS), len(TARGETS))
    return {
        "persistence": Persistence(*shape),
        "window_mean": _Zero(*shape),
        "ramp": _Ramp(*shape),
    }


def _unit_train_stats() -> NormStats:
    """Return unit-scale training statistics over :data:`CHANNELS`.

    A scale of exactly 1.0 makes ``invert_norm`` an addition of the window mean and nothing
    else, so the prescribed forecast survives the round trip bit-exactly and any difference
    between the two paths is the detector rather than float error.

    Returns:
        Statistics labelled ``"id/train"``.
    """
    return build_norm_stats(
        scale=np.ones(len(CHANNELS), dtype=np.float64),
        channels=CHANNELS,
        fitted_on="id/train",
        n_realizations=1,
    )


def _both_quiescence_paths(
    root: Path, cfg: DataConfig, key: RealizationKey
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one realization through the corpus runner and through the external path.

    Args:
        root: Corpus root holding the single realization.
        cfg: Task definition.
        key: The realization key.

    Returns:
        Tuple ``(corpus_cell_rows, external_rows)``.
    """
    spec = window_spec_from_config(cfg)
    stats = _unit_train_stats()
    split = Split(
        regime="id",
        train_keys=frozenset({key}),
        val_keys=frozenset(),
        test_keys=frozenset({key}),
    )
    dataset = DeckMotionDataset(root, split, "test", cfg, spec, stats=stats)
    corpus, _ = evaluate_quiescence(
        _quiescence_models(spec),
        dataset,
        fs_hz=cfg.fs_hz,
        threshold_sets=(PROBE,),
    )
    ss, heading, speed, vessel, seed = key
    frame = pd.read_parquet(
        root
        / realization_path(
            RealizationSpec(
                seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel
            )
        )
    )
    frame.attrs["record"] = "fabricated"
    external = evaluate_quiescence_trajectories(
        _quiescence_models(spec),
        [frame],
        stats=stats,
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        fs_hz=cfg.fs_hz,
        threshold_sets=(PROBE,),
    )
    return corpus[corpus["group_level"] == CELL_LEVEL], external


def test_external_quiescence_matches_the_corpus_runner_row_by_row(
    quiescence_fixture: tuple[Path, DataConfig, RealizationKey],
) -> None:
    """The two paths agree on every reported quantity of every detector.

    The pipeline-sanity control for delta 7. It is the whole basis on which an MSS
    quiescence row may be set beside a committed corpus one: the decision times, the truth
    mask, the exclusion rule, the matching and the lead times are the same code operating
    on the same samples, and this asserts that the re-assembly around them did not move
    any of it.
    """
    corpus, external = _both_quiescence_paths(*quiescence_fixture)
    renamed = external.rename(
        columns={
            "n_onsets_true": "n_true_onsets",
            "n_onsets_pred": "n_pred_onsets",
            "median_lead_time_s": "lead_p50",
        }
    )
    shared = [
        "scorable",
        "base_rate",
        "n_true_onsets",
        "n_pred_onsets",
        "n_matched",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_min",
        "lead_p10",
        "lead_p50",
        "lead_p90",
        "duration_s",
        "n_excluded_true",
        "n_excluded_pred",
    ]
    index = ["model", "threshold_set", "rule"]
    left = corpus.set_index(index).sort_index()[shared]
    right = renamed.set_index(index).sort_index()[shared]
    assert list(left.index) == list(right.index)
    assert left["scorable"].tolist() == right["scorable"].tolist()
    numeric = [c for c in shared if c != "scorable"]
    np.testing.assert_allclose(
        left[numeric].to_numpy(dtype=float), right[numeric].to_numpy(dtype=float), rtol=0.0
    )


def test_the_fabricated_onset_is_actually_detected_by_something(
    quiescence_fixture: tuple[Path, DataConfig, RealizationKey],
) -> None:
    """The parity test is not comparing two empty prediction sides.

    Both constant baselines predict zero onsets by construction, so a parity assertion that
    passed on them alone would prove nothing. This pins that the fixture's one interior
    onset is present on the truth side and that at least one detector flags onsets against
    it.
    """
    _, external = _both_quiescence_paths(*quiescence_fixture)
    assert (external["n_onsets_true"] == 1).all()
    constant = external[external["model"].isin(["persistence", "window_mean"])]
    assert (constant["n_onsets_pred"] == 0).all()
    ramp = external[external["model"] == "ramp"]
    assert int(ramp["n_onsets_pred"].iloc[0]) > 0


def test_every_quiescence_row_carries_its_base_rate(
    quiescence_fixture: tuple[Path, DataConfig, RealizationKey],
) -> None:
    """The base rate sits beside the F1 on every row, and is a real measurement.

    CLAUDE.md §Known traps, and the explicit requirement of Phase 8 delta 7. An F1 shipped
    without its base rate is not interpretable, so the column is asserted present and
    finite on every row rather than assumed to be there.
    """
    _, external = _both_quiescence_paths(*quiescence_fixture)
    assert "base_rate" in external.columns
    rates = external["base_rate"].to_numpy(dtype=float)
    assert np.all(np.isfinite(rates))
    assert np.all((rates >= 0.0) & (rates <= 1.0))
    # The fabricated run is samples 40..59, and the evaluated span is the 90 samples
    # strictly after the first decision time (sample 9) up to the last covered sample.
    np.testing.assert_allclose(rates, 20.0 / 90.0)


def test_the_always_yes_detector_scores_far_worse_on_onsets_than_on_samples(
    quiescence_fixture: tuple[Path, DataConfig, RealizationKey],
) -> None:
    """The P6-D2 contrast, carried onto the external path and measured rather than quoted.

    An always-quiescent detector's per-sample F1 is ``2 * base_rate / (1 + base_rate)`` --
    it knows nothing and is rewarded entirely by the base rate. Scored on *onsets* at the
    0.5 s tolerance it collapses, because it flags at every decision time. That gap is what
    makes the onset formulation credible, and it is what a reader of the MSS table needs in
    front of them, so ``always_yes_sample_f1`` ships on every row beside the onset F1.
    """
    _, external = _both_quiescence_paths(*quiescence_fixture)
    row = external[external["model"] == ALWAYS_QUIESCENT].iloc[0]
    rate = float(row["base_rate"])
    np.testing.assert_allclose(
        float(row["always_yes_sample_f1"]), 2.0 * rate / (1.0 + rate), rtol=1e-12
    )
    assert float(row["recall"]) == 1.0
    assert float(row["f1"]) < float(row["always_yes_sample_f1"])


def test_quiescence_records_are_not_pooled(
    quiescence_fixture: tuple[Path, DataConfig, RealizationKey],
) -> None:
    """Two records produce two rows per detector, so a spread over seeds is recoverable.

    Non-negotiable 5. Pooling the records inside the function would leave the caller with a
    single number and no way to recover the spread over the >= 3 MSS seeds of a cell.
    """
    root, cfg, key = quiescence_fixture
    spec = window_spec_from_config(cfg)
    ss, heading, speed, vessel, seed = key
    frame = pd.read_parquet(
        root
        / realization_path(
            RealizationSpec(
                seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel
            )
        )
    )
    first, second = frame.copy(), frame.copy()
    first.attrs["record"] = "a"
    second.attrs["record"] = "b"
    table = evaluate_quiescence_trajectories(
        _quiescence_models(spec),
        [first, second],
        stats=_unit_train_stats(),
        spec=spec,
        input_channels=CHANNELS,
        target_dofs=TARGETS,
        fs_hz=cfg.fs_hz,
        threshold_sets=(PERMISSIVE, STRICT),
    )
    assert sorted(table["record"].unique()) == ["a", "b"]
    assert sorted(table["threshold_set"].unique()) == ["permissive", "strict"]
    assert not table.duplicated(["model", "record", "threshold_set", "rule"]).any()
    assert list(table.columns) == list(EXTERNAL_QUIESCENCE_COLUMNS)


def test_quiescence_refuses_a_radian_record() -> None:
    """A record still in radians is refused before a single threshold is applied.

    The metric this matters most for. Skill is a ratio and a uniform factor of 57.3 cancels
    exactly out of it; the landing limits are absolute, so the same record would produce a
    plausible-looking quiescence table in which the deck is inside 3 degrees essentially
    always.
    """
    spec = _spec()
    frame = _frame(3)
    frame[list(CHANNELS)] = np.radians(frame[list(CHANNELS)].to_numpy())
    with pytest.raises(ValueError, match="radian range"):
        evaluate_quiescence_trajectories(
            _models(spec),
            [frame],
            stats=_stats(),
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=TARGETS,
            fs_hz=FS_HZ,
        )


def test_quiescence_refuses_an_arm_without_heave_rate() -> None:
    """An arm that does not forecast all three decision channels is refused, not scored.

    Two limits out of three is a different metric wearing the same column names.
    """
    spec = _spec()
    targets = ("roll", "pitch", "heave")
    with pytest.raises(ValueError, match="not forecast targets"):
        evaluate_quiescence_trajectories(
            {"persistence": Persistence(spec.lookback, spec.max_horizon, len(CHANNELS), 3)},
            [_frame(4)],
            stats=_stats(),
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=targets,
            fs_hz=FS_HZ,
        )


def test_quiescence_refuses_non_train_statistics() -> None:
    """The provenance guard covers the quiescence path too, not only the skill path."""
    spec = _spec()
    leaked = NormStats(
        channels=CHANNELS,
        scale=np.full(len(CHANNELS), 2.0, dtype=np.float64),
        fitted_on="external/mss",
        n_realizations=1,
    )
    with pytest.raises(ValueError, match="CORPUS TRAINING SPLIT"):
        evaluate_quiescence_trajectories(
            _models(spec),
            [_frame(5)],
            stats=leaked,
            spec=spec,
            input_channels=CHANNELS,
            target_dofs=TARGETS,
            fs_hz=FS_HZ,
        )
