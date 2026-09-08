"""The phase-lag pass: what it is measured on, and what it refuses to measure.

``tests/test_phase.py`` pins the estimator on synthetic sinusoids. This module pins the
*pass*: the stride-1 requirement, the RNG-free realization sub-sample, the identifiability
flag, and the join onto the accuracy table.

The load-bearing case is **persistence**, whose answer is known in advance without any
appeal to the implementation: persistence holds the last observed sample across the whole
horizon, so its lead-``h`` forecast series is the truth series delayed by exactly ``h``
samples, and the measured lag must come out at exactly ``h / fs_hz`` seconds, positive
(late). A pass that reported anything else -- a sign flip, an off-by-one in the origin
arithmetic, a lag divided by the stride -- would be caught by that one number.

Units: seconds for lags and periods, samples for horizons and indices, hertz for rates.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import Split, build_split
from dmf.data.windows import window_spec_from_config
from dmf.eval.phase import cross_correlation_lag
from dmf.eval.phase_runner import (
    IDENTIFIABILITY_FRACTION,
    PHASE_COLUMNS,
    PHASE_JOIN_KEYS,
    build_phase_dataset,
    evaluate_phase_lag,
    join_phase_lag,
    select_phase_realizations,
)

#: Sampling rate of the fixture corpus, hertz.
FS_HZ = 10.0

#: Horizons reported by the small-corpus geometry, samples.
HORIZONS: tuple[int, ...] = (5, 10, 20)

#: Realizations the phase pass runs on in this module. Two, not the production 32: the
#: quantity under test is the geometry, and a stride-1 pass over 32 realizations of the
#: fixture corpus would be a minute of nothing new.
N_REALIZATIONS = 2


class _Persistence(nn.Module):
    """Hold the last observed sample across the horizon."""

    def __init__(self, max_horizon: int, n_targets: int) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.n_targets = n_targets

    def forward(self, x: Tensor) -> Tensor:
        return x[:, -1:, : self.n_targets].expand(-1, self.max_horizon, -1)


class _Answered(nn.Module):
    """A forecaster handed its whole answer, in dataset order."""

    def __init__(self, answer: Tensor) -> None:
        super().__init__()
        self.register_buffer("answer", answer)
        self._cursor = 0

    def forward(self, x: Tensor) -> Tensor:
        answer: Tensor = self.answer
        start = self._cursor
        self._cursor = (start + int(x.shape[0])) % int(answer.shape[0])
        return answer[start : start + int(x.shape[0])]


def _phase_dataset(
    corpus: Path, manifest: pd.DataFrame, cfg: DataConfig, n: int = N_REALIZATIONS
) -> tuple[DeckMotionDataset, Split]:
    """Build the stride-1 sub-sampled ``id`` test partition and its split."""
    split = build_split(manifest, "id")
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(corpus, split, "train", cfg, spec)
    return build_phase_dataset(corpus, split, cfg, train.norm_stats, n_realizations=n), split


def _shifted_answer(dataset: DeckMotionDataset, shift: int) -> _Answered:
    """Return a forecaster whose lead-h series is the truth delayed by ``shift`` origins.

    Built by handing window ``i`` the true future of window ``i - shift`` within the same
    realization, so the injected lag is exactly ``shift / fs_hz`` seconds by construction
    and owes nothing to any model.
    """
    per_realization = dataset.windows_per_realization
    truth = torch.stack([dataset[i][1] for i in range(len(dataset))]).double()
    mean = torch.stack([dataset[i][2] for i in range(len(dataset))]).double()
    scale = torch.as_tensor(
        dataset.norm_stats.subset(dataset.target_columns).scale, dtype=torch.float64
    )
    source = np.arange(len(dataset))
    within = source % per_realization
    source = source - np.minimum(within, shift)
    return _Answered(((truth[source] - mean) / scale).float())


def _reference_lags(dataset: DeckMotionDataset, horizon: int) -> np.ndarray:
    """Recompute persistence's lag per (realization, DOF) by the slow, direct route.

    Rebuilds the two series the estimator is supposed to correlate straight out of the
    dataset -- the last observed sample of each window, and the true value ``horizon``
    samples past that window's origin -- so the runner's origin arithmetic, batching and
    inverse normalisation are checked against a construction that uses none of them.
    """
    per_realization = dataset.windows_per_realization
    n_targets = len(dataset.target_columns)
    scale = dataset.norm_stats.subset(dataset.target_columns).scale
    lags = np.zeros((len(dataset.realization_keys), n_targets), dtype=np.float64)
    for realization in range(len(dataset.realization_keys)):
        base = realization * per_realization
        pred = np.zeros((per_realization, n_targets), dtype=np.float64)
        true = np.zeros((per_realization, n_targets), dtype=np.float64)
        for offset in range(per_realization):
            x, y, mean = dataset[base + offset]
            pred[offset] = x[-1, :n_targets].numpy() * scale + mean.numpy()[0]
            true[offset] = y[horizon - 1].numpy()
        for channel in range(n_targets):
            lags[realization, channel] = cross_correlation_lag(
                pred[:, channel], true[:, channel], FS_HZ, interpolate=False
            )
    return lags


def test_persistence_lags_the_truth_and_the_pass_agrees_with_a_direct_computation(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset, _ = _phase_dataset(small_corpus, small_manifest, small_data_cfg)
    models = {
        "persistence": _Persistence(dataset.window_spec.max_horizon, len(dataset.target_columns))
    }
    table = evaluate_phase_lag(models, dataset, horizons=HORIZONS, fs_hz=FS_HZ)

    assert tuple(table.columns) == PHASE_COLUMNS
    assert len(table) == len(dataset.target_columns) * len(HORIZONS)
    for horizon in HORIZONS:
        rows = table.loc[table["horizon_samples"] == horizon].sort_values("dof")
        expected = _reference_lags(dataset, horizon).mean(axis=0)
        order = np.argsort(np.asarray(dataset.target_columns))
        assert rows["phase_lag_raw_s"].to_numpy() == pytest.approx(expected[order])
        lag = rows["phase_lag_raw_s"].to_numpy()
        # Persistence's lead-h forecast IS the truth h samples ago, so the lag is h/fs --
        # shrunk toward zero by the deliberate taper of P6-D3, which divides by the full
        # sample count rather than by the overlap and therefore pulls a broad correlation
        # peak inward by slope/(2 * curvature). That is ~1-3 samples on this fixture's
        # 531-window series and an order of magnitude less on the production 5651-window
        # one. Asserted as a bound and a direction: the estimator may under-report the
        # delay it was given, and must never over-report it.
        assert (lag > 0.0).all(), "persistence is late, and a negative lag is a sign error"
        assert (lag <= horizon / FS_HZ + 1e-12).all()
    by_horizon = table.groupby("horizon_samples")["phase_lag_raw_s"].mean()
    assert list(by_horizon.index) == list(HORIZONS)
    assert (by_horizon.diff().dropna() > 0).all(), "a longer lead must not be less late"
    assert (table["n_realizations_phase"] == N_REALIZATIONS).all()


def test_the_interpolated_lag_stays_within_half_a_sample_of_its_own_argmax(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset, _ = _phase_dataset(small_corpus, small_manifest, small_data_cfg)
    models = {
        "persistence": _Persistence(dataset.window_spec.max_horizon, len(dataset.target_columns))
    }
    table = evaluate_phase_lag(models, dataset, horizons=HORIZONS, fs_hz=FS_HZ)
    # Both readings ship precisely so this is checkable (P6-D3). The parabolic refinement is
    # clipped to +-0.5 samples, so the two can never differ by more than that; a table where
    # they did would mean the interpolation had wandered to a neighbouring peak.
    delta = (table["phase_lag_s"] - table["phase_lag_raw_s"]).abs()
    assert (delta <= 0.5 / FS_HZ + 1e-12).all()


def test_a_forecast_delayed_by_half_a_period_is_reported_as_unidentified(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset, _ = _phase_dataset(small_corpus, small_manifest, small_data_cfg)
    table = evaluate_phase_lag(
        {"persistence": _Persistence(dataset.window_spec.max_horizon, len(dataset.target_columns))},
        dataset,
        horizons=HORIZONS,
        fs_hz=FS_HZ,
    )
    period = float(table["dominant_period_s"].median())
    shift = int(round(0.6 * period * FS_HZ))
    delayed = evaluate_phase_lag(
        {"delayed": _shifted_answer(dataset, shift)}, dataset, horizons=HORIZONS, fs_hz=FS_HZ
    )
    # The injected delay is beyond a quarter of the dominant period, which is where the
    # +-T replicas of the correlation peak stop being distinguishable. Every such row must
    # ship phase_lag_identified = False: it is the estimator declining to say which cycle it
    # is looking at, and P6-D3 requires that be visible rather than printed as a number.
    unidentified = ~delayed["phase_lag_identified"].to_numpy(dtype=bool)
    assert unidentified.any()
    for _, row in delayed.iterrows():
        expected = abs(row["phase_lag_s"]) <= IDENTIFIABILITY_FRACTION * row["dominant_period_s"]
        assert bool(row["phase_lag_identified"]) is bool(expected)
    # And the honest reading of the short-horizon reference rows is the opposite one.
    assert table.loc[table["horizon_samples"] == HORIZONS[0], "phase_lag_identified"].all()


def test_the_phase_pass_refuses_a_strided_partition(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    split = build_split(small_manifest, "id")
    spec = window_spec_from_config(small_data_cfg)
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    strided = DeckMotionDataset(
        small_corpus, split, "test", small_data_cfg, spec, stats=train.norm_stats
    )
    assert strided.window_spec.stride > 1
    with pytest.raises(ValueError, match="stride-1 partition"):
        evaluate_phase_lag(
            {"persistence": _Persistence(spec.max_horizon, len(strided.target_columns))},
            strided,
            horizons=HORIZONS,
            fs_hz=FS_HZ,
        )


def test_the_phase_pass_refuses_statistics_from_another_regime(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    cfg = replace(small_data_cfg, stride=1)
    id_split = build_split(small_manifest, "id")
    other = build_split(small_manifest, "unseen_vessel")
    spec = window_spec_from_config(cfg)
    id_train = DeckMotionDataset(small_corpus, id_split, "train", cfg, spec)
    # The P3-D11 guard, on this pass too: it reads held-out data and is subject to the same
    # rule as the accuracy pass, which is why it calls the same checker rather than a copy.
    dataset = build_phase_dataset(small_corpus, other, cfg, id_train.norm_stats, n_realizations=1)
    with pytest.raises(ValueError, match="normalisation statistics"):
        evaluate_phase_lag(
            {"persistence": _Persistence(spec.max_horizon, len(dataset.target_columns))},
            dataset,
            horizons=HORIZONS,
            fs_hz=FS_HZ,
        )


# ---------------------------------------------------------------------------
# The realization sub-sample
# ---------------------------------------------------------------------------


def test_the_sub_sample_spans_the_grid_instead_of_taking_the_first_keys(
    small_manifest: pd.DataFrame,
) -> None:
    keys = sorted(build_split(small_manifest, "id").test_keys)
    chosen = select_phase_realizations(keys, 4)
    assert len(chosen) == 4
    assert set(chosen) <= set(keys)
    # The point of the even spacing: the head of the sorted list is one sea state at one
    # heading, so "the first 32" would measure a lag for one cell and report it for the
    # regime. This asserts the selection is wider than that head.
    head = keys[: len(chosen)]
    assert len({key[0] for key in chosen} | {key[1] for key in chosen}) > len(
        {key[0] for key in head} | {key[1] for key in head}
    )


def test_the_sub_sample_carries_no_rng(small_manifest: pd.DataFrame) -> None:
    keys = sorted(build_split(small_manifest, "id").test_keys)
    assert select_phase_realizations(keys, 5) == select_phase_realizations(list(reversed(keys)), 5)


def test_the_sub_sample_returns_everything_when_the_partition_is_small() -> None:
    keys = [("SS5", 45.0, 0.0, "frigate", seed) for seed in range(3)]
    assert select_phase_realizations(keys, 32) == tuple(sorted(keys))


def test_the_sub_sample_refuses_an_empty_partition() -> None:
    with pytest.raises(ValueError, match="keys is empty"):
        select_phase_realizations([])


def test_build_phase_dataset_moves_no_realization_across_the_split(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset, split = _phase_dataset(small_corpus, small_manifest, small_data_cfg)
    assert set(dataset.realization_keys) <= set(split.test_keys)
    assert not set(dataset.realization_keys) & set(split.train_keys)
    assert dataset.window_spec.stride == 1
    assert dataset.window_spec.lookback == small_data_cfg.lookback


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def _phase_frame(models: tuple[str, ...]) -> pd.DataFrame:
    """Return a minimal phase table for the join tests."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "regime": "id",
                "dof": "roll",
                "horizon_samples": horizon,
                "horizon_s": horizon / FS_HZ,
                "phase_lag_s": 0.1,
                "phase_lag_raw_s": 0.1,
                "phase_lag_std_s": 0.0,
                "dominant_period_s": 12.0,
                "phase_lag_identified": True,
                "n_realizations_phase": 32,
            }
            for model in models
            for horizon in HORIZONS
        ],
        columns=list(PHASE_COLUMNS),
    )


def _metrics_frame(models: tuple[str, ...]) -> pd.DataFrame:
    """Return a minimal accuracy table for the join tests."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "regime": "id",
                "dof": "roll",
                "horizon_samples": horizon,
                "rmse": 1.0,
                "skill": 0.5,
            }
            for model in models
            for horizon in HORIZONS
        ]
    )


def test_the_join_attaches_both_readings_and_preserves_the_metric_rows() -> None:
    metrics = _metrics_frame(("persistence", "tcn"))
    joined = join_phase_lag(metrics, _phase_frame(("persistence", "tcn")))
    assert len(joined) == len(metrics)
    assert list(joined["model"]) == list(metrics["model"])
    for column in ("phase_lag_s", "phase_lag_raw_s", "phase_lag_identified"):
        assert column in joined.columns
        assert joined[column].notna().all()
    # The join key is the one the metrics table already has; nothing is duplicated by it.
    assert not joined.duplicated(subset=list(PHASE_JOIN_KEYS)).any()


def test_the_join_refuses_to_leave_a_phase_column_blank() -> None:
    metrics = _metrics_frame(("persistence", "tcn"))
    with pytest.raises(ValueError, match="no phase-lag row"):
        join_phase_lag(metrics, _phase_frame(("persistence",)))


def test_the_join_refuses_a_duplicated_phase_key() -> None:
    phase = _phase_frame(("persistence",))
    doubled = pd.concat([phase, phase], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        join_phase_lag(_metrics_frame(("persistence",)), doubled)


def test_the_join_names_a_missing_key_column() -> None:
    with pytest.raises(ValueError, match="missing the join columns"):
        join_phase_lag(
            _metrics_frame(("persistence",)).drop(columns=["regime"]),
            _phase_frame(("persistence",)),
        )
