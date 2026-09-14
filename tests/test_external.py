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
from dmf.data.splits import build_split, load_manifest
from dmf.data.windows import WindowSpec, window_spec_from_config
from dmf.eval.external import (
    EXTERNAL_COLUMNS,
    assert_corpus_units,
    corpus_train_norm_stats,
    evaluate_trajectories,
)
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
