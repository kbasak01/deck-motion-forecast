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
  That spec is declared **locally**, in :data:`ARITHMETIC_SPEC`, and is deliberately not the
  shipped task: the identity under test is ``(n - total) // stride + 1``, which must keep
  being checked on a geometry whose answer is known by hand no matter how the task
  definition moves. Everything that means *the task as currently configured* instead reads
  ``configs/data/default.yaml`` through :data:`PRODUCTION_SPEC`. Mirroring the task geometry
  into module constants is what let the Gate 3 revision of ``horizons`` and ``target_dofs``
  break this file silently (``docs/protocol.md`` P3).

- The **TCN receptive field**, which the plan lists here and which was deferred while
  ``dmf.models.tcn`` was a stub. ``receptive_field(3, (1, 2, 4, 8, 16, 32)) == 253`` is
  hand-checkable arithmetic (``1 + 2*2*63``) and is asserted as a literal; the covering
  inequality ``253 >= lookback`` is asserted against :data:`PRODUCTION_SPEC` rather than
  against 200, so a task revision moves it instead of silently passing. The converse is
  asserted too -- a truncated dilation list must both fall short *and* make ``TCN(...)``
  raise -- because a one-sided assertion passes against a constructor that never checks.

- The **sea-state conditioning** channel layout (Phase 6.3 ablation). The one-hot
  indicator must be appended **after every motion channel**, never interleaved, because
  ``dmf.models.persistence.Persistence`` forecasts by slicing the first ``C_out`` input
  channels (P2-D4). The tests assert the layout, that ``target_dofs`` is still exactly that
  prefix, that the indicator is left un-normalised, that its width and column order are the
  same in every partition of every regime, and that Gate 2 criterion 5 still holds with it
  switched on -- which is the check that the appended columns did not disturb the
  persistence denominator. Two further guards live here because they decide how the
  ablation can be read at all: ``revin`` and ``condition_on_sea_state`` together are refused
  at config load (P6-D8: RevIN zeroes a time-constant indicator silently), and the
  ``dlinear``/``dlinear_ols`` blindness to the indicator is asserted as a **measured**
  bitwise identity rather than read off a source line, so the arm's zero contrast can never
  later be misread as evidence that sea state does not matter.

Units: samples for lookback, horizons and strides; degrees for roll and pitch, metres for
heave.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import CORPUS_CONFIG_PATH, DATA_CONFIG_PATH, SMALL_N_SAMPLES
from dmf.config import DataConfig, load_data, load_model, load_sim
from dmf.data.dataset import (
    SEA_STATE_CHANNEL_PREFIX,
    DeckMotionDataset,
    make_dataloader,
    resolve_columns,
    sea_state_channels,
)
from dmf.data.splits import Regime, build_split
from dmf.data.windows import (
    WindowSpec,
    make_windows,
    n_windows,
    window_spec_from_config,
    window_start_indices,
)
from dmf.eval.controls import persistence_pipeline_sanity
from dmf.models.dlinear import DLinear
from dmf.models.dlinear_ols import DLinearOLS
from dmf.models.tcn import TCN, receptive_field

#: The current task definition, read from ``configs/data/default.yaml`` rather than restated
#: here. A test that means "the shipped task" must load the shipped task.
PRODUCTION_CFG = load_data(DATA_CONFIG_PATH)

#: The production window geometry, derived from :data:`PRODUCTION_CFG`.
PRODUCTION_SPEC = window_spec_from_config(PRODUCTION_CFG)

#: Channel counts of the current task, derived, never restated.
N_IN = len(PRODUCTION_CFG.input_channels)
N_OUT = len(PRODUCTION_CFG.target_dofs)

#: Where the shipped model configs live, for the receptive-field check below.
MODEL_CONFIG_ROOT = DATA_CONFIG_PATH.parents[1] / "model"

#: Realization length of the production corpus, samples, derived from
#: ``configs/sim/corpus.yaml`` the same way the generator derives it (``duration_s * fs_hz``;
#: the spin-up is discarded before writing). 600 s at 10 Hz = 6000 rows, which the manifest's
#: ``n_rows`` column confirms.
CORPUS_CFG = load_sim(CORPUS_CONFIG_PATH)
PRODUCTION_SAMPLES = int(round(CORPUS_CFG.duration_s * CORPUS_CFG.fs_hz))

#: A fixed geometry with a hand-checkable window count, independent of the task definition.
#: ``(6000 - 250) // 5 + 1 = 1151``, last start 5750. This was the production geometry before
#: the Gate 3 revision; it is kept as a pinned arithmetic case, not as a mirror of the task.
ARITHMETIC_SPEC = WindowSpec(lookback=200, horizons=(10, 20, 30, 50), stride=5)

#: Record length used with :data:`ARITHMETIC_SPEC`, samples.
ARITHMETIC_SAMPLES = 6000


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


def test_production_window_spec_tracks_the_task_config() -> None:
    """The shipped geometry is whatever ``configs/data/default.yaml`` says it is.

    Asserted as a derivation, not as literals: ``total_length`` is the quantity the corpus
    must be long enough to supply, and it must follow the config's horizons wherever they
    move.
    """
    assert PRODUCTION_SPEC.lookback == PRODUCTION_CFG.lookback
    assert PRODUCTION_SPEC.horizons == tuple(PRODUCTION_CFG.horizons)
    assert PRODUCTION_SPEC.stride == PRODUCTION_CFG.stride
    assert PRODUCTION_SPEC.max_horizon == max(PRODUCTION_CFG.horizons)
    assert PRODUCTION_SPEC.total_length == PRODUCTION_CFG.lookback + max(PRODUCTION_CFG.horizons)
    assert PRODUCTION_SPEC.total_length < PRODUCTION_SAMPLES


def test_window_count_arithmetic_on_a_fixed_geometry() -> None:
    """(6000 - 250) // 5 + 1 = 1151, last start 5750. Asserted, not commented.

    Deliberately pinned to :data:`ARITHMETIC_SPEC` rather than to the shipped task: the
    invariant is the arithmetic, and a hand-computable case keeps testing it across task
    revisions.
    """
    assert n_windows(ARITHMETIC_SAMPLES, ARITHMETIC_SPEC) == 1151
    starts = window_start_indices(ARITHMETIC_SAMPLES, ARITHMETIC_SPEC)
    assert starts.shape == (1151,)
    assert starts[0] == 0
    assert starts[-1] == 5750
    assert int(starts[-1]) + ARITHMETIC_SPEC.total_length == ARITHMETIC_SAMPLES
    assert np.all(np.diff(starts) == ARITHMETIC_SPEC.stride)


def test_production_window_count_is_maximal() -> None:
    """The shipped geometry's own count, tied to the arithmetic rather than to a literal.

    At the Gate 3 geometry this is 1131 windows per 6000-sample realization (down from 1151,
    because ``max_horizon`` moved from 50 to 150 samples); the assertion is written so that
    it states the relationship instead of the number.
    """
    count = n_windows(PRODUCTION_SAMPLES, PRODUCTION_SPEC)
    starts = window_start_indices(PRODUCTION_SAMPLES, PRODUCTION_SPEC)
    assert starts.shape == (count,)
    assert int(starts[-1]) + PRODUCTION_SPEC.total_length <= PRODUCTION_SAMPLES
    assert int(starts[-1]) + PRODUCTION_SPEC.stride + PRODUCTION_SPEC.total_length > (
        PRODUCTION_SAMPLES
    )


@pytest.mark.parametrize("n_samples", [0, 1, 249])
def test_short_realizations_yield_no_windows(n_samples: int) -> None:
    """One sample short of ``ARITHMETIC_SPEC.total_length`` (250) must yield nothing."""
    assert n_windows(n_samples, ARITHMETIC_SPEC) == 0
    assert window_start_indices(n_samples, ARITHMETIC_SPEC).size == 0


def test_exactly_one_window_when_the_record_is_exactly_long_enough() -> None:
    assert n_windows(ARITHMETIC_SPEC.total_length, ARITHMETIC_SPEC) == 1
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
# TCN receptive field vs lookback
# ---------------------------------------------------------------------------

#: The shipped TCN stack, read from ``configs/model/tcn.yaml`` so that the covering check
#: below tests the configuration that actually trains rather than a copy of it.
TCN_CFG = load_model(MODEL_CONFIG_ROOT / "tcn.yaml")


def test_tcn_receptive_field_arithmetic_is_the_hand_computed_value() -> None:
    """1 + 2*(3-1)*(1+2+4+8+16+32) = 1 + 4*63 = 253. Asserted, not commented."""
    dilations = (1, 2, 4, 8, 16, 32)
    assert sum(dilations) == 63
    assert receptive_field(3, dilations) == 253
    assert receptive_field(3, list(dilations)) == 253, "YAML delivers a list, not a tuple"


def test_tcn_receptive_field_covers_the_production_lookback() -> None:
    """The covering inequality, against the shipped config and the shipped task geometry.

    Both sides are read rather than restated: the dilations come from
    ``configs/model/tcn.yaml`` and the lookback from ``configs/data/default.yaml``. A
    receptive field shorter than the lookback means the stack never sees the oldest part of
    its own window, and that shows up nowhere in the loss curve.
    """
    rf = receptive_field(int(TCN_CFG.params["kernel_size"]), list(TCN_CFG.params["dilations"]))
    assert rf >= PRODUCTION_SPEC.lookback
    model = TCN(
        PRODUCTION_SPEC.lookback, PRODUCTION_SPEC.max_horizon, N_IN, N_OUT, **TCN_CFG.params
    )
    assert model.receptive_field == rf


def test_a_truncated_dilation_stack_falls_short_and_is_refused() -> None:
    """The converse. Without it, a constructor that never checks passes the test above."""
    truncated = [1, 2, 4, 8]
    assert receptive_field(3, truncated) == 1 + 2 * 2 * 15
    assert receptive_field(3, truncated) < PRODUCTION_SPEC.lookback
    with pytest.raises(ValueError, match="receptive field"):
        TCN(
            PRODUCTION_SPEC.lookback,
            PRODUCTION_SPEC.max_horizon,
            N_IN,
            N_OUT,
            dilations=truncated,
        )


@pytest.mark.parametrize(
    ("kernel_size", "dilations"),
    [(1, (1, 2)), (0, (1, 2)), (3, (1, 0, 4)), (3, (1, -2))],
)
def test_receptive_field_rejects_a_degenerate_stack(
    kernel_size: int, dilations: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        receptive_field(kernel_size, dilations)


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
    cfg = PRODUCTION_CFG
    split = build_split(real_manifest, "id")
    spec = PRODUCTION_SPEC
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
    # Tied to the arithmetic, not to a literal: 1131 at the Gate 3 geometry, 1151 before it.
    assert dataset.windows_per_realization == n_windows(PRODUCTION_SAMPLES, PRODUCTION_SPEC)

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
    cfg = PRODUCTION_CFG
    split = build_split(real_manifest, "id")
    spec = PRODUCTION_SPEC
    subset = type(split)(
        regime=split.regime,
        train_keys=frozenset(sorted(split.train_keys)[:8]),
        val_keys=split.val_keys,
        test_keys=frozenset(sorted(split.test_keys)[:8]),
    )
    train = DeckMotionDataset(real_corpus, subset, "train", cfg, spec)
    dataset = DeckMotionDataset(real_corpus, subset, "test", cfg, spec, stats=train.norm_stats)
    result = persistence_pipeline_sanity(dataset, real_corpus)
    assert result.n_windows == 8 * n_windows(PRODUCTION_SAMPLES, PRODUCTION_SPEC)
    assert result.max_rel_diff < 1e-6


# ---------------------------------------------------------------------------
# Sea-state conditioning -- the Phase 6.3 ablation's channel layout
# ---------------------------------------------------------------------------


def _conditioned(cfg: DataConfig) -> DataConfig:
    """Return ``cfg`` with the one-hot sea-state indicator switched on."""
    return dataclasses.replace(cfg, condition_on_sea_state=True)


def test_the_shipped_data_configs_leave_sea_state_conditioning_off() -> None:
    """The ablation arm turns it on; the reference arms must not have it on by accident."""
    assert PRODUCTION_CFG.condition_on_sea_state is False
    imu_cfg = load_data(DATA_CONFIG_PATH.parent / "imu.yaml")
    assert imu_cfg.condition_on_sea_state is False


def test_load_data_requires_the_conditioning_key(tmp_path: Path) -> None:
    """Validated like every other key: absent is an error, not a silent default."""
    text = DATA_CONFIG_PATH.read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in text.splitlines() if not line.startswith("condition_on_sea_state")
    )
    path = tmp_path / "no_key.yaml"
    path.write_text(stripped, encoding="utf-8")
    with pytest.raises(ValueError, match="condition_on_sea_state"):
        load_data(path)


def test_sea_state_onehot_is_appended_after_every_motion_channel(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The layout the P2-D4 guard depends on, asserted on the tensor and not on the config.

    The indicator width is *derived* from the corpus manifest, so the small corpus's two
    sea states give two columns where the production corpus gives four. Asserting the
    derivation rather than the literal 4 is what keeps this test honest if the corpus grid
    ever changes.
    """
    cfg = _conditioned(small_data_cfg)
    dataset = _dataset(small_corpus, small_manifest, cfg, "id", "test")
    n_motion = len(cfg.input_channels)

    expected_states = tuple(sorted({str(v) for v in small_manifest["ss"].tolist()}))
    assert dataset.sea_states == expected_states
    assert dataset.sea_state_columns == sea_state_channels(expected_states)
    assert all(c.startswith(SEA_STATE_CHANNEL_PREFIX) for c in dataset.sea_state_columns)

    assert dataset.motion_columns == resolve_columns(cfg.input_channels, cfg.observation_mode)
    assert dataset.input_columns == dataset.motion_columns + dataset.sea_state_columns
    assert len(dataset.input_columns) == n_motion + len(expected_states)

    x, _, _ = dataset[0]
    assert x.shape == (cfg.lookback, n_motion + len(expected_states))


def test_target_dofs_remain_the_first_c_out_input_channels(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """P2-D4 survives conditioning: persistence still slices the right channels."""
    cfg = _conditioned(small_data_cfg)
    dataset = _dataset(small_corpus, small_manifest, cfg, "id", "test")
    n_out = len(dataset.target_columns)
    assert dataset.input_columns[:n_out] == dataset.target_columns
    assert not any(c.startswith(SEA_STATE_CHANNEL_PREFIX) for c in dataset.input_columns[:n_out])


def test_conditioning_leaves_the_motion_channels_bit_identical(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Appending a column must not perturb the columns already there.

    Same normalisation statistics, same de-meaning, same targets -- the only difference in
    the tensor is the block on the right.
    """
    plain = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    conditioned = _dataset(small_corpus, small_manifest, _conditioned(small_data_cfg), "id", "test")
    assert len(plain) == len(conditioned)
    assert plain.norm_stats.channels == conditioned.norm_stats.channels
    assert np.array_equal(plain.norm_stats.scale, conditioned.norm_stats.scale)

    n_motion = len(plain.input_columns)
    for index in np.linspace(0, len(plain) - 1, 12, dtype=int).tolist():
        x_plain, y_plain, mean_plain = plain[index]
        x_cond, y_cond, mean_cond = conditioned[index]
        assert torch.equal(x_plain, x_cond[:, :n_motion])
        assert torch.equal(y_plain, y_cond)
        assert torch.equal(mean_plain, mean_cond)


def test_the_indicator_is_neither_demeaned_nor_scaled(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Exactly one 1.0 and the rest 0.0, on every row of every window.

    Per-window de-meaning would zero the block outright -- the indicator is constant within
    a window -- so "not normalised" is a correctness requirement here, not a preference.
    ``NormStats`` therefore covers the motion channels only.
    """
    cfg = _conditioned(small_data_cfg)
    dataset = _dataset(small_corpus, small_manifest, cfg, "id", "test")
    n_motion = len(dataset.motion_columns)
    assert dataset.norm_stats.channels == dataset.motion_columns
    assert len(dataset.norm_stats.scale) == n_motion

    for index in np.linspace(0, len(dataset) - 1, 20, dtype=int).tolist():
        key, _ = dataset.describe_window(index)
        x, _, _ = dataset[index]
        block = x[:, n_motion:].numpy()
        assert set(np.unique(block).tolist()) <= {0.0, 1.0}
        assert np.all(block.sum(axis=1) == 1.0)
        assert np.array_equal(block[0], block[-1])
        hot = dataset.sea_states[int(np.argmax(block[0]))]
        assert hot == key[0]


@pytest.mark.parametrize("regime", ["id", "unseen_seastate", "unseen_heading", "unseen_vessel"])
def test_the_indicator_vocabulary_is_identical_across_partitions_and_regimes(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig, regime: Regime
) -> None:
    """Derived from the corpus manifest, not from the partition.

    Taking the vocabulary from a partition's own keys would give ``unseen_seastate`` a
    narrower training indicator than its test indicator, and the model would be handed a
    different ``C_in`` at test time. The held-out sea state's column is present in training
    and simply never hot -- extrapolation to an unseen category, which is the ablation's
    stated upper-bound caveat, not leakage.
    """
    cfg = _conditioned(small_data_cfg)
    split = build_split(small_manifest, regime)
    spec = window_spec_from_config(cfg)
    train = DeckMotionDataset(small_corpus, split, "train", cfg, spec)
    expected = tuple(sorted({str(v) for v in small_manifest["ss"].tolist()}))

    for partition in ("train", "val", "test"):
        dataset = (
            train
            if partition == "train"
            else DeckMotionDataset(
                small_corpus, split, partition, cfg, spec, stats=train.norm_stats
            )
        )
        assert dataset.sea_states == expected
        assert dataset.input_columns == train.input_columns

    if regime == "unseen_seastate":
        held_out = expected.index("SS6")
        assert not any(key[0] == "SS6" for key in train.realization_keys)
        train_block = np.stack(
            [train[i][0][0, len(train.motion_columns) :].numpy() for i in (0, len(train) // 2)]
        )
        assert train_block[:, held_out].sum() == 0.0


def test_persistence_pipeline_sanity_holds_with_sea_state_conditioning(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Gate 2 criterion 5 on the conditioned data config.

    The control's forecast is ``x[:, -1:, :C_out]``. That expression is only persistence if
    the first ``C_out`` input channels are the targets, so this test is the end-to-end
    statement that the appended indicator did not move them -- and it re-reads the Parquet
    independently, so an appended column that had somehow shifted the motion block would
    show up as a disagreement rather than as a slightly different number.
    """
    dataset = _dataset(small_corpus, small_manifest, _conditioned(small_data_cfg), "id", "test")
    plain = _dataset(small_corpus, small_manifest, small_data_cfg, "id", "test")
    result = persistence_pipeline_sanity(dataset, small_corpus)
    assert result.max_rel_diff < 1e-6
    assert result.n_windows == len(dataset)
    reference = persistence_pipeline_sanity(plain, small_corpus)
    assert np.allclose(result.rmse_raw, reference.rmse_raw, rtol=0.0, atol=0.0)
    assert np.allclose(result.rmse_pipeline, reference.rmse_pipeline, rtol=0.0, atol=0.0)


def test_revin_and_sea_state_conditioning_are_refused_together(tmp_path: Path) -> None:
    """P6-D8 defect 3, refused at config load rather than left as a comment.

    RevIN normalises the whole ``(B, L, C_in)`` window inside the model's forward pass. The
    indicator is constant along the time axis, so its per-window standard deviation is 0 and
    ``(x - mean) / (0 + eps)`` is exactly zero -- the feature is destroyed and nothing
    raises. The arm would then report "sea-state conditioning has no effect" while never
    having shown the model the sea state.
    """
    text = DATA_CONFIG_PATH.read_text(encoding="utf-8")
    both = text.replace("revin: false", "revin: true").replace(
        "condition_on_sea_state: false", "condition_on_sea_state: true"
    )
    path = tmp_path / "revin_and_ss.yaml"
    path.write_text(both, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot both be true"):
        load_data(path)

    # And not only through YAML: `dataclasses.replace` is how every ablation driver and
    # every test builds a variant config, so the dataclass itself must refuse it too.
    with pytest.raises(ValueError, match="cannot both be true"):
        dataclasses.replace(PRODUCTION_CFG, revin=True, condition_on_sea_state=True)

    # Either flag alone stays legal -- they are separate ablation arms.
    assert dataclasses.replace(PRODUCTION_CFG, revin=True).revin is True
    assert _conditioned(PRODUCTION_CFG).condition_on_sea_state is True


def _one_hot_pair(n_motion: int, n_states: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two windows identical in every motion channel, differing only in the one-hot.

    Args:
        n_motion: Number of motion channels.
        n_states: Number of one-hot sea-state columns.
        seed: Torch seed for the shared motion block.

    Returns:
        Tuple ``(first_state_hot, last_state_hot)``, each of shape ``(2, L, C_in)``.
    """
    torch.manual_seed(seed)
    motion = torch.randn(2, PRODUCTION_SPEC.lookback, n_motion)

    def _with(hot: int) -> torch.Tensor:
        block = torch.zeros(2, PRODUCTION_SPEC.lookback, n_states)
        block[:, :, hot] = 1.0
        return torch.cat((motion, block), dim=2)

    return _with(0), _with(n_states - 1)


def test_dlinear_is_bitwise_blind_to_the_sea_state_indicator() -> None:
    """P6-D8 defect 1, measured rather than inferred: the DLinear family cannot see it.

    ``dmf.models.dlinear.DLinear.forward`` slices ``x[:, :, :C_out]``, so on this task
    (``C_out == 6`` motion channels) every appended indicator column is discarded before the
    decomposition. ``DLinearOLS`` inherits that forward pass unchanged.

    The consequence for Stage 5, and the reason this test is permanent: on the
    sea-state-conditioning arm the DLinear rows will show a paired contrast of **exactly
    zero** against the reference arm. That is a property of the architecture, not a finding
    about sea state, and without this assertion nothing in the repository distinguishes the
    two readings.
    """
    n_motion, n_states = 6, 4
    first, last = _one_hot_pair(n_motion, n_states)
    kwargs = {
        "lookback": PRODUCTION_SPEC.lookback,
        "max_horizon": PRODUCTION_SPEC.max_horizon,
        "n_input_channels": n_motion + n_states,
        "n_target_channels": n_motion,
        "kernel_size": 25,
    }
    for model in (DLinear(**kwargs), DLinearOLS(**kwargs)):
        model.eval()
        with torch.no_grad():
            out_first = model.forward(first)
            out_last = model.forward(last)
        assert out_first.shape == (2, PRODUCTION_SPEC.max_horizon, n_motion)
        assert torch.equal(out_first, out_last)

        # The converse, so this is a statement about the indicator and not about a model
        # that ignores its input: perturbing a motion channel must move the output.
        perturbed = first.clone()
        perturbed[:, :, n_motion - 1] += 1.0
        with torch.no_grad():
            assert not torch.equal(model.forward(perturbed), out_first)


def test_the_tcn_does_see_the_sea_state_indicator() -> None:
    """The other half of the ablation's vehicle question.

    ``tcn`` convolves over all ``C_in`` channels, so it is the only vehicle named in the
    Stage 5 arm table that can actually use the indicator. Asserted so that "the TCN row is
    the SS-conditioning row" is a checked fact rather than an assumption.
    """
    n_motion, n_states = 6, 4
    first, last = _one_hot_pair(n_motion, n_states, seed=1)
    model = TCN(
        lookback=PRODUCTION_SPEC.lookback,
        max_horizon=PRODUCTION_SPEC.max_horizon,
        n_input_channels=n_motion + n_states,
        n_target_channels=n_motion,
    )
    model.eval()
    with torch.no_grad():
        assert not torch.equal(model.forward(first), model.forward(last))
