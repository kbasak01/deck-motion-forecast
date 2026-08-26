"""Split integrity and leakage guards -- Gate 2, part one.

Gate 2 criteria owned by this module:

1. Zero seed overlap between any train and test split, asserted programmatically for all
   four regimes (``id``, ``unseen_seastate``, ``unseen_heading``, ``unseen_vessel``).
2. No time index appears in both a train window and a test window for the same
   realization.
4. Normalisation statistics are computed on train only; a fixture that constructs a
   test-only dataset without training statistics must raise rather than silently refit.

Also covered here:

- The validation partition is drawn from training seeds, never from test seeds.
- ``build_split`` raises rather than returning an empty test set when a regime's condition
  selects nothing (for example ``unseen_vessel`` on a single-vessel corpus).
- The converse of criterion 2: a deliberately corrupted split, with one realization forced
  into both train and test, must make ``assert_no_shared_time_index`` raise. An assertion
  that can never fail is not a test.

Units follow the package convention: headings in degrees, speeds in knots, window lengths
and strides in samples.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import (
    SMALL_N_REALIZATIONS,
    SMALL_N_SAMPLES,
)
from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import apply_norm, fit_norm_stats
from dmf.data.splits import (
    HELDOUT_HEADING_DEG,
    HELDOUT_SEA_STATE,
    HELDOUT_VESSEL,
    PRIMARY_VESSEL,
    REGIMES,
    TEST_SEED_FRAC,
    Regime,
    Split,
    build_split,
    realization_key,
)
from dmf.data.windows import window_spec_from_config

#: Realization counts on the real 2304-row corpus, from the Phase 2 regime table in
#: ``docs/protocol.md``. Asserted by the slow tests below so the table cannot rot.
REAL_COUNTS: dict[str, tuple[int, int, int]] = {
    "id": (1296, 240, 384),
    "unseen_seastate": (1224, 216, 480),
    "unseen_heading": (1224, 216, 480),
    "unseen_vessel": (1632, 288, 384),
}

#: Held-out axis value per regime, as ``(realization-key index, value)``.
HELDOUT_AXIS: dict[str, tuple[int, object]] = {
    "unseen_seastate": (0, HELDOUT_SEA_STATE),
    "unseen_heading": (1, HELDOUT_HEADING_DEG),
    "unseen_vessel": (3, HELDOUT_VESSEL),
}


# ---------------------------------------------------------------------------
# Criterion 1 -- seed disjointness, all four regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIMES)
def test_split_partitions_are_seed_disjoint(small_manifest: pd.DataFrame, regime: Regime) -> None:
    from dmf.data.splits import assert_seed_disjoint

    split = build_split(small_manifest, regime)
    assert_seed_disjoint(split)
    assert not split.train_keys & split.test_keys
    assert not split.val_keys & split.test_keys
    assert not split.train_keys & split.val_keys


@pytest.mark.parametrize("regime", REGIMES)
def test_split_partitions_the_manifest_without_loss_or_duplication(
    small_manifest: pd.DataFrame, regime: Regime
) -> None:
    """Every realization the regime uses appears exactly once across the three partitions.

    Guards the failure mode where a filter silently drops rows: a split that loses half the
    corpus is still perfectly disjoint.
    """
    split = build_split(small_manifest, regime)
    total = len(split.train_keys) + len(split.val_keys) + len(split.test_keys)
    union = split.train_keys | split.val_keys | split.test_keys
    assert len(union) == total, "a realization appears in more than one partition"

    all_keys = {
        realization_key(*row)
        for row in zip(
            small_manifest["ss"].tolist(),
            small_manifest["heading"].tolist(),
            small_manifest["speed"].tolist(),
            small_manifest["vessel"].tolist(),
            small_manifest["seed"].tolist(),
            strict=True,
        )
    }
    assert union <= all_keys
    frigate = {k for k in all_keys if k[3] == PRIMARY_VESSEL}
    if regime == "unseen_vessel":
        assert union == all_keys
    else:
        assert union == frigate, "regimes other than unseen_vessel are confined to the frigate"


@pytest.mark.parametrize("regime", REGIMES)
def test_heldout_axis_never_appears_in_train_or_val(
    small_manifest: pd.DataFrame, regime: Regime
) -> None:
    """Disjoint keys are not enough: the held-out axis value must be absent from training.

    A split can be perfectly key-disjoint and still train on SS6.
    """
    split = build_split(small_manifest, regime)
    if regime == "id":
        test_ordinals = {k[4] for k in split.test_keys}
        train_ordinals = {k[4] for k in split.train_keys} | {k[4] for k in split.val_keys}
        assert not test_ordinals & train_ordinals
        assert min(test_ordinals) > max(train_ordinals), "the id cut is the top of the range"
        return
    index, value = HELDOUT_AXIS[regime]
    assert all(k[index] != value for k in split.train_keys)
    assert all(k[index] != value for k in split.val_keys)
    assert all(k[index] == value for k in split.test_keys)


def test_no_regime_ever_trains_on_the_heldout_vessel(small_manifest: pd.DataFrame) -> None:
    """``s175`` exists to be the ``unseen_vessel`` test set and is never trained on."""
    for regime in REGIMES:
        split = build_split(small_manifest, regime)
        assert all(k[3] == PRIMARY_VESSEL for k in split.train_keys), regime
        assert all(k[3] == PRIMARY_VESSEL for k in split.val_keys), regime


@pytest.mark.parametrize("regime", REGIMES)
def test_validation_seed_ordinals_are_disjoint_from_train_and_test(
    small_manifest: pd.DataFrame, regime: Regime
) -> None:
    """Validation is carved by seed ordinal, so the bare ordinals are disjoint too."""
    split = build_split(small_manifest, regime)
    train_ordinals = {k[4] for k in split.train_keys}
    val_ordinals = {k[4] for k in split.val_keys}
    assert not train_ordinals & val_ordinals
    assert min(val_ordinals) > max(train_ordinals), "validation is the top of the dev range"
    if regime == "id":
        assert not val_ordinals & {k[4] for k in split.test_keys}


def test_id_seed_cut_is_the_top_fraction_of_ordinals(small_manifest: pd.DataFrame) -> None:
    """The ``id`` cut is ``TEST_SEED_FRAC`` of the ordinals present, not a hard-coded 32.

    On the 40-seed corpus the two coincide (seeds 32-39); on this 8-seed fixture the cut is
    seeds 6-7. Hard-coding 32 would make every fixture-scale split return an empty test set.
    """
    split = build_split(small_manifest, "id")
    ordinals = sorted({int(s) for s in small_manifest["seed"].tolist()})
    n_test = math.ceil(TEST_SEED_FRAC * len(ordinals))
    assert sorted({k[4] for k in split.test_keys}) == ordinals[-n_test:]


# ---------------------------------------------------------------------------
# Criterion 2 -- no shared time index, and its converse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIMES)
def test_no_time_index_is_shared_between_partitions(
    small_manifest: pd.DataFrame, small_data_cfg: DataConfig, regime: Regime
) -> None:
    from dmf.data.splits import assert_no_shared_time_index

    spec = window_spec_from_config(small_data_cfg)
    split = build_split(small_manifest, regime)
    assert_no_shared_time_index(
        split, SMALL_N_SAMPLES, spec.lookback, spec.max_horizon, spec.stride
    )


def test_a_corrupted_split_makes_the_time_index_check_raise(
    small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The converse of criterion 2. Without this, the check above is a no-op.

    One test realization is forced into the training set, exactly as a within-realization
    split would do, and the guard must fire.
    """
    from dmf.data.splits import assert_no_shared_time_index, assert_seed_disjoint

    spec = window_spec_from_config(small_data_cfg)
    clean = build_split(small_manifest, "id")
    leaked = next(iter(sorted(clean.test_keys)))
    corrupted = Split(
        regime=clean.regime,
        train_keys=clean.train_keys | {leaked},
        val_keys=clean.val_keys,
        test_keys=clean.test_keys,
    )
    with pytest.raises(AssertionError, match="guard band"):
        assert_no_shared_time_index(
            corrupted, SMALL_N_SAMPLES, spec.lookback, spec.max_horizon, spec.stride
        )
    with pytest.raises(AssertionError, match="appear in both train and test"):
        assert_seed_disjoint(corrupted)


def test_seed_disjointness_catches_a_heldout_axis_leak(small_manifest: pd.DataFrame) -> None:
    """Key-disjoint but SS6 in training: the second property of ``assert_seed_disjoint``."""
    from dmf.data.splits import assert_seed_disjoint

    split = build_split(small_manifest, "unseen_seastate")
    # A key that is genuinely absent from the test set (different seed cell) but carries
    # the held-out sea state, so key-level disjointness alone would pass it.
    ss6_train = ("SS6", HELDOUT_HEADING_DEG, 12.0, PRIMARY_VESSEL, -1)
    corrupted = Split(
        regime=split.regime,
        train_keys=split.train_keys | {ss6_train},
        val_keys=split.val_keys,
        test_keys=split.test_keys,
    )
    assert not corrupted.train_keys & corrupted.test_keys
    with pytest.raises(AssertionError, match="held-out axis value"):
        assert_seed_disjoint(corrupted)


# ---------------------------------------------------------------------------
# build_split contract
# ---------------------------------------------------------------------------


def test_build_split_rejects_an_unknown_regime(small_manifest: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        build_split(small_manifest, "unseen_everything")  # type: ignore[arg-type]


@pytest.mark.parametrize("column", ["ss", "heading", "speed", "vessel", "seed"])
def test_build_split_rejects_a_manifest_missing_a_column(
    small_manifest: pd.DataFrame, column: str
) -> None:
    with pytest.raises(ValueError, match="missing required column"):
        build_split(small_manifest.drop(columns=[column]), "id")


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5])
def test_build_split_rejects_an_out_of_range_val_frac(
    small_manifest: pd.DataFrame, val_frac: float
) -> None:
    with pytest.raises(ValueError, match=r"val_frac must lie in \(0, 1\)"):
        build_split(small_manifest, "id", val_frac=val_frac)


def test_unseen_vessel_on_a_single_vessel_corpus_raises(small_manifest: pd.DataFrame) -> None:
    """An empty test set must be an error, never a silently returned empty frozenset."""
    frigate_only = small_manifest[small_manifest["vessel"] == PRIMARY_VESSEL]
    with pytest.raises(ValueError, match="selected no test realizations"):
        build_split(frigate_only, "unseen_vessel")


def test_unseen_seastate_on_a_single_sea_state_corpus_raises(
    small_manifest: pd.DataFrame,
) -> None:
    ss5_only = small_manifest[small_manifest["ss"] != HELDOUT_SEA_STATE]
    with pytest.raises(ValueError, match="selected no test realizations"):
        build_split(ss5_only, "unseen_seastate")


@pytest.mark.parametrize("regime", REGIMES)
def test_build_split_is_deterministic(small_manifest: pd.DataFrame, regime: Regime) -> None:
    """No RNG anywhere in split construction, so no seed needs recording to reproduce one."""
    first = build_split(small_manifest, regime)
    second = build_split(small_manifest.sample(frac=1.0, random_state=0), regime)
    assert first == second


def test_realization_key_is_dtype_canonical(
    small_corpus: Path, small_manifest: pd.DataFrame
) -> None:
    """Keys built from the manifest (float64/int64) equal keys built from a file (float32/int32).

    Without one canonical cast site these would never compare equal and every set
    intersection in this module would be silently empty -- which would make the leakage
    guards pass unconditionally.
    """
    row = small_manifest.iloc[0]
    from_manifest = realization_key(
        row["ss"], row["heading"], row["speed"], row["vessel"], row["seed"]
    )
    frame = pd.read_parquet(small_corpus / str(row["path"]))
    first = frame.iloc[0]
    assert frame["heading"].dtype == np.float32
    assert frame["seed"].dtype == np.int32
    from_file = realization_key(
        first["ss"], first["heading"], first["speed"], first["vessel"], first["seed"]
    )
    assert from_manifest == from_file


# ---------------------------------------------------------------------------
# Criterion 4 -- normalisation statistics come from train only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["id/val", "id/test", "unseen_seastate/test", "corpus", ""])
def test_fit_norm_stats_refuses_a_non_training_partition(label: str) -> None:
    series = np.linspace(0.0, 1.0, 300).reshape(100, 3)
    with pytest.raises(ValueError, match="training partition"):
        fit_norm_stats(series, ("roll", "pitch", "heave"), label, n_realizations=1)


def test_apply_norm_refuses_statistics_not_fitted_on_train() -> None:
    """The provenance check is re-applied at use, not only at fit.

    A ``NormStats`` smuggled in from a checkpoint must be refused too, which is why
    ``fitted_on`` is a field rather than a local variable.
    """
    import dataclasses

    import torch

    series = np.linspace(0.0, 1.0, 300).reshape(100, 3)
    good = fit_norm_stats(series, ("roll", "pitch", "heave"), "id/train", n_realizations=1)
    smuggled = dataclasses.replace(good, fitted_on="id/test")
    x = torch.zeros(2, 4, 3)
    apply_norm(x, good)
    with pytest.raises(ValueError, match="fitted on"):
        apply_norm(x, smuggled)


def test_fit_norm_stats_rejects_a_zero_variance_channel() -> None:
    series = np.zeros((100, 2))
    series[:, 0] = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="zero variance"):
        fit_norm_stats(series, ("roll", "pitch"), "id/train", n_realizations=1)


def test_test_partition_without_train_stats_raises(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Gate 2 criterion 4: refuse rather than silently refit on the held-out partition."""
    split = build_split(small_manifest, "id")
    spec = window_spec_from_config(small_data_cfg)
    for partition in ("val", "test"):
        with pytest.raises(ValueError, match="requires normalisation statistics"):
            DeckMotionDataset(small_corpus, split, partition, small_data_cfg, spec, stats=None)


def test_unknown_partition_raises(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    split = build_split(small_manifest, "id")
    spec = window_spec_from_config(small_data_cfg)
    with pytest.raises(ValueError, match="unknown partition"):
        DeckMotionDataset(small_corpus, split, "holdout", small_data_cfg, spec, stats=None)


def test_train_partition_fits_and_labels_its_own_statistics(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    split = build_split(small_manifest, "id")
    spec = window_spec_from_config(small_data_cfg)
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    stats = train.norm_stats
    assert stats.fitted_on == "id/train"
    assert stats.n_realizations == len(split.train_keys)
    assert stats.channels == small_data_cfg.input_channels
    assert np.all(stats.scale > 0.0)
    # The val set must accept the train set's statistics unchanged.
    val = DeckMotionDataset(small_corpus, split, "val", small_data_cfg, spec, stats=stats)
    assert val.norm_stats is stats


def test_train_only_statistics_differ_measurably_from_whole_corpus_statistics(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """Positive control for criterion 4.

    A guard against a leak is only worth having if the leak would have moved a number.
    ``unseen_seastate`` is the sharp case: fitting the scale over the whole corpus pulls in
    SS6 variance, which is exactly what the regime holds out.
    """
    split = build_split(small_manifest, "unseen_seastate")
    spec = window_spec_from_config(small_data_cfg)
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    test = DeckMotionDataset(
        small_corpus, split, "test", small_data_cfg, spec, stats=train.norm_stats
    )

    train_series = train.training_series
    leaky_series = np.concatenate([train_series, test.training_series], axis=0)
    leaky = fit_norm_stats(
        leaky_series,
        small_data_cfg.input_channels,
        "id/train",  # the label a leak would wear
        n_realizations=train.norm_stats.n_realizations + len(split.test_keys),
    )
    ratio = leaky.scale / train.norm_stats.scale
    assert np.max(np.abs(ratio - 1.0)) > 0.10, (
        f"whole-corpus statistics differ from train-only statistics by only "
        f"{np.max(np.abs(ratio - 1.0)):.4f}; the positive control is not discriminating"
    )


# ---------------------------------------------------------------------------
# Real corpus
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("regime", REGIMES)
def test_real_corpus_regime_counts_and_disjointness(
    real_manifest: pd.DataFrame, regime: Regime
) -> None:
    """The Phase 2 regime table, asserted against the 2304-realization corpus."""
    from dmf.data.splits import assert_no_shared_time_index, assert_seed_disjoint

    split = build_split(real_manifest, regime)
    assert_seed_disjoint(split)
    assert_no_shared_time_index(split, 6000, 200, 50, 5)
    counts = (len(split.train_keys), len(split.val_keys), len(split.test_keys))
    assert counts == REAL_COUNTS[regime]


@pytest.mark.slow
def test_real_corpus_id_cut_is_seeds_32_to_39(real_manifest: pd.DataFrame) -> None:
    """On the production corpus the fractional cut reproduces the plan's literal seeds."""
    split = build_split(real_manifest, "id")
    assert sorted({k[4] for k in split.test_keys}) == list(range(32, 40))
    assert sorted({k[4] for k in split.val_keys}) == list(range(27, 32))
    assert sorted({k[4] for k in split.train_keys}) == list(range(27))


@pytest.mark.slow
def test_real_corpus_manifest_is_the_documented_size(real_manifest: pd.DataFrame) -> None:
    assert len(real_manifest) == 2304
    assert (real_manifest["vessel"] == PRIMARY_VESSEL).sum() == 1920
    assert (real_manifest["vessel"] == HELDOUT_VESSEL).sum() == 384


def test_small_corpus_is_the_expected_size(small_manifest: pd.DataFrame) -> None:
    assert len(small_manifest) == SMALL_N_REALIZATIONS
    assert set(small_manifest["ss"]) == {"SS5", HELDOUT_SEA_STATE}
    assert HELDOUT_HEADING_DEG in set(small_manifest["heading"])
    assert set(small_manifest["vessel"]) == {PRIMARY_VESSEL, HELDOUT_VESSEL}


def test_norm_stats_subset_selects_the_right_scales_in_the_right_order() -> None:
    """``subset`` is the only thing standing between a target and the wrong scale.

    It must reorder, not merely slice, and it must carry provenance through: a subset that
    lost ``fitted_on`` would slip past ``apply_norm``'s guard.
    """
    series = np.stack(
        [np.linspace(-1.0, 1.0, 200), np.linspace(-4.0, 4.0, 200), np.linspace(-9.0, 9.0, 200)],
        axis=1,
    )
    stats = fit_norm_stats(series, ("roll", "pitch", "heave"), "id/train", n_realizations=3)
    reordered = stats.subset(("heave", "roll"))
    assert reordered.channels == ("heave", "roll")
    np.testing.assert_allclose(reordered.scale, [stats.scale[2], stats.scale[0]])
    assert reordered.fitted_on == "id/train"
    assert reordered.n_realizations == 3
    with pytest.raises(ValueError, match="not in"):
        stats.subset(("yaw",))


def test_dataset_normalisation_agrees_with_demean_window_and_apply_norm(
    small_corpus: Path, small_manifest: pd.DataFrame, small_data_cfg: DataConfig
) -> None:
    """The dataset normalises inline in numpy; these functions do it in torch.

    Two implementations of one transform is how a forward and an inverse drift apart, so
    the agreement is asserted rather than assumed. Tolerance is the float32 storage floor
    of the dataset's returned tensor.
    """
    import torch

    from dmf.data.normalize import apply_norm, demean_window
    from dmf.data.windows import window_spec_from_config

    split = build_split(small_manifest, "id")
    spec = window_spec_from_config(small_data_cfg)
    train = DeckMotionDataset(small_corpus, split, "train", small_data_cfg, spec)
    stats = train.norm_stats

    input_cols = list(train.input_columns)
    for index in np.linspace(0, len(train) - 1, 10, dtype=int).tolist():
        key, start = train.describe_window(index)
        row = small_manifest[
            (small_manifest["ss"] == key[0])
            & (small_manifest["heading"] == key[1])
            & (small_manifest["speed"] == key[2])
            & (small_manifest["vessel"] == key[3])
            & (small_manifest["seed"] == key[4])
        ]
        frame = pd.read_parquet(small_corpus / str(row.iloc[0]["path"]), columns=input_cols)
        raw = frame[input_cols].to_numpy(dtype=np.float32)[start : start + spec.lookback]
        reference, _ = demean_window(torch.from_numpy(raw.copy())[None, ...].double())
        reference = apply_norm(reference, stats)
        x, _, _ = train[index]
        np.testing.assert_allclose(
            x.numpy().astype(np.float64), reference[0].numpy(), rtol=1e-6, atol=1e-6
        )
