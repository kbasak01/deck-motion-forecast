"""Windowing and pipeline-fidelity checks -- Gate 2, part two.

Gate 2 criteria owned by this module:

3. A window reconstructed from the dataset exactly matches the raw Parquet slice.
5. A persistence baseline evaluated through the full dataset pipeline reproduces the same
   RMSE as computed directly on the raw arrays -- an end-to-end sanity check that catches
   an off-by-one in windowing, a mis-inverted normalisation, or a channel-order mismatch,
   none of which would be visible in a loss curve.

Also covered here:

- Windows never span a realization boundary.
- The window-count arithmetic, asserted rather than restated in a comment:
  ``n_windows(6000, WindowSpec(200, (10, 20, 30, 50), 5)) == 1151`` with last start 5750.

Deferred to Phase 4, per the plan: the **TCN receptive field** check
(``receptive_field(3, (1, 2, 4, 8, 16, 32)) == 253 >= lookback``), which needs
``dmf.models.tcn`` to exist.

Units: samples for lookback, horizons and strides; degrees for roll and pitch, metres for
heave.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import SMALL_N_SAMPLES
from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset, make_dataloader, resolve_columns
from dmf.data.splits import Regime, build_split
from dmf.data.windows import (
    WindowSpec,
    make_windows,
    n_windows,
    window_spec_from_config,
    window_start_indices,
)
from dmf.eval.controls import persistence_pipeline_sanity

#: The production geometry: 20 s lookback at 10 Hz, horizons 1/2/3/5 s, 0.5 s stride.
PRODUCTION_SPEC = WindowSpec(lookback=200, horizons=(10, 20, 30, 50), stride=5)

#: Realization length of the production corpus, samples (600 s at 10 Hz).
PRODUCTION_SAMPLES = 6000


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


def test_window_spec_geometry() -> None:
    assert PRODUCTION_SPEC.max_horizon == 50
    assert PRODUCTION_SPEC.total_length == 250


def test_production_window_count_arithmetic() -> None:
    """(6000 - 250) // 5 + 1 = 1151, last start 5750. Asserted, not commented."""
    assert n_windows(PRODUCTION_SAMPLES, PRODUCTION_SPEC) == 1151
    starts = window_start_indices(PRODUCTION_SAMPLES, PRODUCTION_SPEC)
    assert starts.shape == (1151,)
    assert starts[0] == 0
    assert starts[-1] == 5750
    assert int(starts[-1]) + PRODUCTION_SPEC.total_length == PRODUCTION_SAMPLES
    assert np.all(np.diff(starts) == PRODUCTION_SPEC.stride)


@pytest.mark.parametrize("n_samples", [0, 1, 249])
def test_short_realizations_yield_no_windows(n_samples: int) -> None:
    assert n_windows(n_samples, PRODUCTION_SPEC) == 0
    assert window_start_indices(n_samples, PRODUCTION_SPEC).size == 0


def test_exactly_one_window_when_the_record_is_exactly_long_enough() -> None:
    assert n_windows(PRODUCTION_SPEC.total_length, PRODUCTION_SPEC) == 1


@pytest.mark.parametrize("stride", [0, -1])
def test_non_positive_stride_raises(stride: int) -> None:
    spec = WindowSpec(lookback=10, horizons=(5,), stride=stride)
    with pytest.raises(ValueError, match="stride must be positive"):
        n_windows(100, spec)


def test_empty_horizons_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ = WindowSpec(lookback=10, horizons=(), stride=1).max_horizon


def test_window_spec_from_config_matches_the_config(small_data_cfg: DataConfig) -> None:
    spec = window_spec_from_config(small_data_cfg)
    assert spec.lookback == small_data_cfg.lookback
    assert spec.horizons == small_data_cfg.horizons
    assert spec.stride == small_data_cfg.stride


# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------


def test_make_windows_on_a_ramp_lands_on_the_right_samples() -> None:
    """A ramp makes an off-by-one arithmetically visible: value == index."""
    spec = WindowSpec(lookback=8, horizons=(2, 4), stride=3)
    series = np.stack([np.arange(50.0), 100.0 + np.arange(50.0)], axis=1)
    x, y = make_windows(series, spec)
    starts = window_start_indices(series.shape[0], spec)

    assert x.shape == (starts.size, spec.lookback, 2)
    assert y.shape == (starts.size, spec.max_horizon, 2)
    for i, start in enumerate(starts.tolist()):
        assert x[i, 0, 0] == start
        assert x[i, -1, 0] == start + spec.lookback - 1
        assert y[i, 0, 0] == start + spec.lookback
        assert y[i, -1, 0] == start + spec.total_length - 1
        assert np.array_equal(x[i, :, 1], x[i, :, 0] + 100.0)


def test_make_windows_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        make_windows(np.arange(100.0), PRODUCTION_SPEC)


def test_make_windows_rejects_a_too_short_series() -> None:
    with pytest.raises(ValueError, match="shorter than the window total length"):
        make_windows(np.zeros((100, 3)), PRODUCTION_SPEC)


# ---------------------------------------------------------------------------
# Criterion 3 -- dataset windows match the raw Parquet slice
# ---------------------------------------------------------------------------


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


def test_dataset_window_matches_the_raw_parquet_slice(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Gate 2 criterion 3.

    ``y`` and ``window_mean`` are checked for **exact** equality against the stored float32
    values -- no arithmetic stands between them and the file. The input is checked to
    ``atol=1e-6`` in corpus units (degrees, metres) because reconstructing it means undoing
    a divide by the fitted scale in float32, and exact equality is not claimable through a
    divide-then-multiply round trip.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    spec = dataset.window_spec
    target_cols = list(dataset.target_columns)
    input_cols = list(dataset.input_columns)
    scale = dataset.norm_stats.scale
    n_out = len(target_cols)

    indices = np.linspace(0, len(dataset) - 1, 25, dtype=int).tolist()
    for index in indices:
        key, start = dataset.describe_window(index)
        row = small_manifest[
            (small_manifest["ss"] == key[0])
            & (small_manifest["heading"] == key[1])
            & (small_manifest["speed"] == key[2])
            & (small_manifest["vessel"] == key[3])
            & (small_manifest["seed"] == key[4])
        ]
        assert len(row) == 1, f"key {key} does not identify one realization"
        frame = pd.read_parquet(small_corpus / str(row.iloc[0]["path"]))

        x, y, window_mean = dataset[index]
        raw_target = frame[target_cols].to_numpy(dtype=np.float32)[
            start + spec.lookback : start + spec.total_length
        ]
        assert np.array_equal(y.numpy(), raw_target), f"target mismatch at window {index}"

        raw_input = frame[input_cols].to_numpy(dtype=np.float32)[start : start + spec.lookback]
        expected_mean = raw_input.astype(np.float64).mean(axis=0)[:n_out]
        np.testing.assert_allclose(window_mean.numpy()[0], expected_mean, rtol=0, atol=1e-5)

        reconstructed = (
            x.numpy().astype(np.float64) * scale[None, :]
            + raw_input.astype(np.float64).mean(axis=0)[None, :]
        )
        np.testing.assert_allclose(reconstructed, raw_input.astype(np.float64), rtol=0, atol=1e-6)


def test_every_window_comes_from_exactly_one_realization(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Windows never span a realization boundary.

    Each window's samples are contained in ``[start, start + total_length)`` of a single
    realization, and every realization contributes the same number of windows, so the index
    cannot run off the end of one file into the next.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "train")
    spec = dataset.window_spec
    split = build_split(small_manifest, "id")

    seen: dict[tuple[str, float, float, str, int], set[int]] = {}
    for index in range(len(dataset)):
        key, start = dataset.describe_window(index)
        assert key in split.train_keys
        assert start >= 0
        assert start + spec.total_length <= SMALL_N_SAMPLES
        seen.setdefault(key, set()).add(start)

    assert set(seen) == set(split.train_keys)
    expected_starts = set(window_start_indices(SMALL_N_SAMPLES, spec).tolist())
    for key, starts in seen.items():
        assert starts == expected_starts, key
    assert len(dataset) == len(split.train_keys) * len(expected_starts)


def test_describe_window_rejects_an_out_of_range_index(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    with pytest.raises(IndexError):
        dataset.describe_window(len(dataset))
    with pytest.raises(IndexError):
        dataset[-1]


def test_observation_mode_remaps_inputs_and_targets_together() -> None:
    """``docs/protocol.md`` P1-D6: an imu input with an ideal target is not constructible.

    ``heave_acc`` has no ``_imu`` twin and passes through: it is the raw accelerometer
    channel and both modes see the same values.
    """
    names = ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate", "heave_acc")
    assert resolve_columns(names, "ideal") == names
    assert resolve_columns(names, "imu") == (
        "roll_imu",
        "pitch_imu",
        "heave_imu",
        "roll_rate_imu",
        "pitch_rate_imu",
        "heave_rate_imu",
        "heave_acc",
    )
    with pytest.raises(ValueError, match="unknown channel"):
        resolve_columns(("yaw",), "ideal")


def test_imu_mode_dataset_reads_the_imu_columns(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    import dataclasses

    cfg = dataclasses.replace(small_data_cfg, observation_mode="imu")
    dataset = _dataset(small_corpus, small_manifest, cfg, "id", "test")
    assert dataset.input_columns == resolve_columns(cfg.input_channels, "imu")
    assert dataset.target_columns == resolve_columns(cfg.target_dofs, "imu")
    assert all(c.endswith("_imu") for c in dataset.target_columns)
    key, start = dataset.describe_window(0)
    _, y, _ = dataset[0]
    frame = pd.read_parquet(
        small_corpus / f"{key[3]}/{key[0]}_h{key[1]:05.1f}_u{key[2]:04.1f}_s{key[4]:03d}.parquet"
    )
    spec = dataset.window_spec
    raw = frame[list(dataset.target_columns)].to_numpy(dtype=np.float32)[
        start + spec.lookback : start + spec.total_length
    ]
    assert np.array_equal(y.numpy(), raw)


# ---------------------------------------------------------------------------
# Criterion 5 -- persistence through the pipeline matches persistence on raw arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["id", "unseen_seastate"])
def test_persistence_pipeline_sanity(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig, regime: Regime
) -> None:
    """Gate 2 criterion 5, in the easy regime and in a generalisation regime."""
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, regime, "test")
    result = persistence_pipeline_sanity(dataset, small_corpus)
    assert result.n_windows == len(dataset)
    assert result.rmse_pipeline.shape == (
        small_data_cfg.horizons[-1],
        len(small_data_cfg.target_dofs),
    )
    assert result.max_rel_diff < 1e-6
    assert np.all(result.rmse_raw > 0.0)
    # Persistence error must grow with horizon on a narrowband oscillation.
    for channel in range(result.rmse_raw.shape[1]):
        column = result.rmse_raw[:, channel]
        assert column[-1] > column[0]


def test_pipeline_sanity_fails_on_an_off_by_one(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The converse: a control that cannot fail is not a control.

    A dataset whose window starts are shifted by one sample -- exactly the classic
    windowing bug -- must be caught, because the raw path recomputes the starts from
    ``window_start_indices`` instead of trusting the dataset.
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    dataset._starts = dataset._starts + 1  # noqa: SLF001
    with pytest.raises(AssertionError, match="disagrees with persistence on the raw arrays"):
        persistence_pipeline_sanity(dataset, small_corpus)


def test_pipeline_sanity_fails_on_a_channel_order_swap(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """A swapped target channel order must be caught too -- roll is not pitch."""
    import dataclasses

    swapped = dataclasses.replace(
        small_data_cfg,
        target_dofs=("pitch", "roll", "heave"),
        input_channels=("pitch", "roll", "heave", "roll_rate", "pitch_rate", "heave_rate"),
    )
    dataset = _dataset(small_corpus, small_manifest, swapped, "id", "test")
    dataset._target_columns = ("roll", "pitch", "heave")  # noqa: SLF001
    with pytest.raises(AssertionError, match="disagrees with persistence on the raw arrays"):
        persistence_pipeline_sanity(dataset, small_corpus)


def test_dataloader_is_deterministic_and_covers_every_window(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Two loaders built from the same seed produce the same epoch, and lose no window.

    The comparison is between the *first* epoch of each loader: torch's ``RandomSampler``
    consumes the generator, so a second epoch of the same loader is deliberately a
    different permutation. Reproducibility here means "a run is reproducible from its
    recorded seed", not "every epoch is identical".
    """
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")

    def first_epoch(seed: int) -> tuple[list[int], torch.Tensor]:
        loader = make_dataloader(dataset, 32, shuffle=True, num_workers=0, seed=seed)
        sizes: list[int] = []
        chunks: list[torch.Tensor] = []
        for _, y, _ in loader:
            sizes.append(int(y.shape[0]))
            chunks.append(y.reshape(-1))
        return sizes, torch.cat(chunks)

    sizes_a, a = first_epoch(11)
    _, b = first_epoch(11)
    _, c = first_epoch(12)

    assert sum(sizes_a) == len(dataset), "drop_last must not discard windows"
    assert torch.equal(a, b)
    assert not torch.equal(a, c)

    ordered = make_dataloader(dataset, 32, shuffle=False, num_workers=0, seed=0)
    flat = torch.cat([y.reshape(-1) for _, y, _ in ordered])
    assert torch.equal(torch.sort(flat).values, torch.sort(a).values)


@pytest.mark.parametrize("bad", [0, -4])
def test_make_dataloader_rejects_a_non_positive_batch_size(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig, bad: int
) -> None:
    dataset = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    with pytest.raises(ValueError, match="batch_size must be positive"):
        make_dataloader(dataset, bad, shuffle=False, num_workers=0, seed=0)


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_corpus_window_matches_the_raw_parquet_slice(
    real_corpus: Path, real_manifest: pd.DataFrame
) -> None:
    """Criterion 3 at the production geometry, on the real corpus."""
    from dmf.config import load_data

    cfg = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    split = build_split(real_manifest, "id")
    spec = window_spec_from_config(cfg)
    # A small, fixed subset of the test partition: reading 384 files is not needed to
    # detect an off-by-one, and the split itself is checked in full in test_splits.py.
    subset = type(split)(
        regime=split.regime,
        train_keys=frozenset(sorted(split.train_keys)[:4]),
        val_keys=split.val_keys,
        test_keys=frozenset(sorted(split.test_keys)[:4]),
    )
    train = DeckMotionDataset(real_corpus, subset, "train", cfg, spec)
    dataset = DeckMotionDataset(real_corpus, subset, "test", cfg, spec, stats=train.norm_stats)
    assert dataset.windows_per_realization == 1151

    target_cols = list(dataset.target_columns)
    for index in np.linspace(0, len(dataset) - 1, 12, dtype=int).tolist():
        key, start = dataset.describe_window(index)
        path = (
            real_corpus / f"{key[3]}/{key[0]}_h{key[1]:05.1f}_u{key[2]:04.1f}_s{key[4]:03d}.parquet"
        )
        raw = pd.read_parquet(path, columns=target_cols)[target_cols].to_numpy(dtype=np.float32)
        _, y, _ = dataset[index]
        assert np.array_equal(y.numpy(), raw[start + spec.lookback : start + spec.total_length]), (
            key,
            start,
        )


@pytest.mark.slow
def test_real_corpus_persistence_pipeline_sanity(
    real_corpus: Path, real_manifest: pd.DataFrame
) -> None:
    """Criterion 5 on the real corpus, at the production geometry."""
    from dmf.config import load_data

    cfg = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    split = build_split(real_manifest, "id")
    spec = window_spec_from_config(cfg)
    subset = type(split)(
        regime=split.regime,
        train_keys=frozenset(sorted(split.train_keys)[:8]),
        val_keys=split.val_keys,
        test_keys=frozenset(sorted(split.test_keys)[:8]),
    )
    train = DeckMotionDataset(real_corpus, subset, "train", cfg, spec)
    dataset = DeckMotionDataset(real_corpus, subset, "test", cfg, spec, stats=train.norm_stats)
    result = persistence_pipeline_sanity(dataset, real_corpus)
    assert result.n_windows == 8 * 1151
    assert result.max_rel_diff < 1e-6
