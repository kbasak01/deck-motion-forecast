"""Realization-level train/test splitting -- the leakage guard.

Splits are **by realization seed**, never by time window. Two windows drawn from the same
realization share the wave field that generated them; if one lands in train and the other
in test, every metric is inflated and the inflation is invisible in the loss curve. This
is the classic failure mode of time-series portfolio projects, and it is the reason this
module exists as a separate, separately tested unit rather than three lines inside a
dataset constructor.

Four evaluation regimes are built here:

===================  ==========================  ==============  ==============================
Regime               Train                       Test            Question answered
===================  ==========================  ==============  ==============================
``id``               seeds 0-31 of every cell    seeds 32-39     In-distribution accuracy
``unseen_seastate``  SS3, SS4, SS5               SS6             Extrapolation to rougher seas
``unseen_heading``   180, 135, 45 deg            90 deg (beam)   Worst-case roll condition
``unseen_vessel``    frigate                     S175            Transfer across hull dynamics
===================  ==========================  ==============  ==============================

The seed cut of the ``id`` regime is expressed as the fraction :data:`TEST_SEED_FRAC` of
the seed ordinals present, not as the literal ordinal 32: on the 40-seed frigate corpus the
two coincide, but a hard 32 would make every split built from a small test fixture return an
empty test set. See ``docs/protocol.md`` P2-D2.

There is no RNG in this module. A split is a pure function of the manifest, the regime and
``val_frac``, so it is reproducible without recording a seed.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from dmf.data.windows import WindowSpec, window_start_indices
from dmf.sim.generate import MANIFEST_NAME

__all__ = [
    "HELDOUT_HEADING_DEG",
    "HELDOUT_SEA_STATE",
    "HELDOUT_VESSEL",
    "PRIMARY_VESSEL",
    "REGIMES",
    "RealizationKey",
    "Regime",
    "Split",
    "TEST_SEED_FRAC",
    "assert_no_shared_time_index",
    "assert_seed_disjoint",
    "build_split",
    "load_manifest",
    "realization_key",
]

#: The four evaluation regimes. ``id`` measures in-distribution accuracy; the other three
#: measure a specific, named kind of generalisation.
Regime = Literal["id", "unseen_seastate", "unseen_heading", "unseen_vessel"]

#: Tuple form of :data:`Regime`, for iteration in evaluation drivers.
REGIMES: tuple[Regime, ...] = ("id", "unseen_seastate", "unseen_heading", "unseen_vessel")

#: One realization, identified by its grid coordinates, ordered
#: ``(sea_state, heading_deg, speed_kn, vessel, seed_ordinal)``. Built only by
#: :func:`realization_key`, which is what makes keys from the manifest (float64/int64) and
#: keys from a per-realization Parquet file (float32/int32) compare equal.
type RealizationKey = tuple[str, float, float, str, int]

#: The hull every regime except ``unseen_vessel`` is confined to. ``docs/corpus_card.md``.
PRIMARY_VESSEL: str = "frigate"

#: The hull held out by ``unseen_vessel``. It is never trained on, in any regime.
HELDOUT_VESSEL: str = "s175"

#: The sea state held out by ``unseen_seastate``: the roughest in the corpus.
HELDOUT_SEA_STATE: str = "SS6"

#: The heading held out by ``unseen_heading``: beam seas, where roll is largest and pitch
#: is smallest (``docs/protocol.md`` P1-D2).
HELDOUT_HEADING_DEG: float = 90.0

#: Fraction of the distinct seed ordinals that ``id`` holds out for test, taken from the
#: top of the ordinal range. On the 40-seed frigate corpus this is exactly seeds 32-39,
#: the cut named in ``docs/IMPLEMENTATION_PLAN.md`` 2.2.
TEST_SEED_FRAC: float = 0.2

#: Manifest columns a split is built from.
REQUIRED_COLUMNS: tuple[str, ...] = ("ss", "heading", "speed", "vessel", "seed")


def realization_key(ss: Any, heading: Any, speed: Any, vessel: Any, seed: Any) -> RealizationKey:
    """Build the canonical key for one realization.

    Every realization key in the codebase is constructed here. ``manifest.parquet`` stores
    ``heading`` and ``speed`` as float64 and ``seed`` as int64, while a per-realization
    Parquet file stores float32 and int32; keys built from the two sources would not compare
    equal without this single cast site.

    Args:
        ss: Sea state label, e.g. ``"SS5"``.
        heading: Encounter angle, degrees.
        speed: Forward speed, knots.
        vessel: Vessel config stem, e.g. ``"frigate"``.
        seed: Seed ordinal within the grid cell.

    Returns:
        The key, ``(ss, heading_deg, speed_kn, vessel, seed_ordinal)``.
    """
    return (str(ss), float(heading), float(speed), str(vessel), int(seed))


@dataclass(frozen=True)
class Split:
    """A concrete train/validation/test partition of the corpus at realization level.

    Membership is expressed as sets of realization keys rather than as row masks, so that
    disjointness is checkable directly and cheaply, without materialising windows.

    Attributes:
        regime: Which of the four regimes produced this split.
        train_keys: Realization keys assigned to training.
        val_keys: Realization keys assigned to validation. Drawn from the *training*
            seeds, so that the test set is touched exactly once, at the end.
        test_keys: Realization keys assigned to testing.
    """

    regime: Regime
    train_keys: frozenset[tuple[str, float, float, str, int]]
    val_keys: frozenset[tuple[str, float, float, str, int]]
    test_keys: frozenset[tuple[str, float, float, str, int]]


def load_manifest(corpus_root: Path) -> pd.DataFrame:
    """Read the corpus manifest.

    Args:
        corpus_root: Dataset root written by :func:`dmf.sim.generate.generate_corpus`.

    Returns:
        One row per realization, with at least the columns in :data:`REQUIRED_COLUMNS`
        plus ``n_rows`` and ``path``. ``heading`` is degrees, ``speed`` knots.

    Raises:
        FileNotFoundError: If no manifest exists under ``corpus_root``.
    """
    path = corpus_root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"no corpus manifest at {path}")
    return pd.read_parquet(path)


def _keys(frame: pd.DataFrame) -> frozenset[RealizationKey]:
    """Convert manifest rows into canonical realization keys.

    Args:
        frame: Manifest rows carrying :data:`REQUIRED_COLUMNS`.

    Returns:
        The set of keys for those rows.
    """
    return frozenset(
        realization_key(*row)
        for row in zip(
            frame["ss"].tolist(),
            frame["heading"].tolist(),
            frame["speed"].tolist(),
            frame["vessel"].tolist(),
            frame["seed"].tolist(),
            strict=True,
        )
    )


def _top_ordinals(ordinals: frozenset[int], frac: float) -> frozenset[int]:
    """Take the highest ``ceil(frac * n)`` distinct seed ordinals.

    Args:
        ordinals: Distinct seed ordinals available.
        frac: Fraction to hold out, in (0, 1).

    Returns:
        The held-out ordinals. At least one whenever ``ordinals`` is non-empty, so a
        fixture-sized corpus still produces a non-empty partition.
    """
    n_hold = math.ceil(frac * len(ordinals))
    return frozenset(sorted(ordinals)[len(ordinals) - n_hold :]) if n_hold else frozenset()


def build_split(meta: pd.DataFrame, regime: Regime, val_frac: float = 0.15) -> Split:
    """Build the seed-disjoint split for one evaluation regime.

    The test set is selected by the regime's held-out axis value (or, for ``id``, by the
    top :data:`TEST_SEED_FRAC` of seed ordinals). The development pool is the complement,
    confined to :data:`PRIMARY_VESSEL`, and validation takes the top
    ``ceil(val_frac * n_distinct_seed_ordinals)`` ordinals of that pool. Carving validation
    by seed ordinal rather than by a random draw makes the ordinal sets themselves globally
    disjoint and removes all randomness from split construction
    (``docs/protocol.md`` P2-D3).

    Args:
        meta: One row per realization, with columns ``ss``, ``heading`` (degrees),
            ``speed`` (knots), ``vessel``, ``seed``. This is the corpus metadata table,
            not the sample-level corpus.
        regime: Which regime to build.
        val_frac: Fraction of *training* realizations held out for validation, in (0, 1).
            Held out at realization granularity like everything else.

    Returns:
        The populated split.

    Raises:
        ValueError: If ``regime`` is unrecognised, if ``meta`` lacks a required column, or
            if the regime's test condition selects no realizations (for example, asking
            for ``unseen_vessel`` on a corpus containing only one vessel).
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}, expected one of {list(REGIMES)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"meta is missing required column(s) {missing}")
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must lie in (0, 1), got {val_frac}")

    primary = meta[meta["vessel"] == PRIMARY_VESSEL]
    if regime == "id":
        ordinals = frozenset(int(s) for s in primary["seed"].tolist())
        test_ordinals = _top_ordinals(ordinals, TEST_SEED_FRAC)
        is_test = primary["seed"].isin(sorted(test_ordinals))
        test_rows = primary[is_test]
        dev_rows = primary[~is_test]
    elif regime == "unseen_seastate":
        is_test = primary["ss"] == HELDOUT_SEA_STATE
        test_rows = primary[is_test]
        dev_rows = primary[~is_test]
    elif regime == "unseen_heading":
        is_test = primary["heading"].astype(float) == HELDOUT_HEADING_DEG
        test_rows = primary[is_test]
        dev_rows = primary[~is_test]
    else:  # unseen_vessel
        test_rows = meta[meta["vessel"] == HELDOUT_VESSEL]
        dev_rows = primary

    test_keys = _keys(test_rows)
    if not test_keys:
        raise ValueError(
            f"regime {regime!r} selected no test realizations from a corpus of "
            f"{len(meta)} rows: the held-out axis value is absent"
        )

    dev_ordinals = frozenset(int(s) for s in dev_rows["seed"].tolist())
    val_ordinals = _top_ordinals(dev_ordinals, val_frac)
    is_val = dev_rows["seed"].isin(sorted(val_ordinals))
    val_keys = _keys(dev_rows[is_val])
    train_keys = _keys(dev_rows[~is_val])

    for name, keys in (("train", train_keys), ("val", val_keys), ("test", test_keys)):
        if not keys:
            raise ValueError(
                f"regime {regime!r} produced an empty {name} partition "
                f"(val_frac={val_frac}, {len(meta)} realizations in meta)"
            )
    return Split(regime=regime, train_keys=train_keys, val_keys=val_keys, test_keys=test_keys)


def _heldout_axis(regime: Regime) -> tuple[int, object] | None:
    """Return the key element and value that a regime holds out, if any.

    Args:
        regime: The regime.

    Returns:
        A ``(key_index, value)`` pair into :data:`RealizationKey`, or None for ``id``,
        whose held-out axis is the seed ordinal and is checked separately.
    """
    if regime == "unseen_seastate":
        return (0, HELDOUT_SEA_STATE)
    if regime == "unseen_heading":
        return (1, HELDOUT_HEADING_DEG)
    if regime == "unseen_vessel":
        return (3, HELDOUT_VESSEL)
    return None


def assert_seed_disjoint(split: Split) -> None:
    """Assert that the three partitions of a split share no realization.

    Checks three properties, not one:

    1. the three key sets are pairwise disjoint;
    2. the regime's held-out axis value never appears in train or val -- a split can be
       key-disjoint and still train on SS6;
    3. for ``id``, whose held-out axis *is* the seed ordinal, the bare ordinal sets are
       pairwise disjoint too, and for every regime the validation ordinals are disjoint
       from the training ordinals.

    Args:
        split: The split to check.

    Raises:
        AssertionError: If any realization key appears in more than one partition. This is
            deliberately an assertion rather than a returned bool: there is no sensible way
            to continue from a leaking split, and a caller must not be able to ignore it.
    """
    parts: dict[str, frozenset[RealizationKey]] = {
        "train": split.train_keys,
        "val": split.val_keys,
        "test": split.test_keys,
    }
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = parts[left] & parts[right]
            assert not shared, (
                f"regime {split.regime!r}: {len(shared)} realization(s) appear in both "
                f"{left} and {right}, e.g. {sorted(shared)[:3]}"
            )

    axis = _heldout_axis(split.regime)
    if axis is not None:
        index, value = axis
        for name in ("train", "val"):
            offenders = {k for k in parts[name] if k[index] == value}
            assert not offenders, (
                f"regime {split.regime!r}: held-out axis value {value!r} appears in "
                f"{len(offenders)} {name} realization(s), e.g. {sorted(offenders)[:3]}"
            )

    ordinals = {name: {k[4] for k in keys} for name, keys in parts.items()}
    assert not (ordinals["train"] & ordinals["val"]), (
        f"regime {split.regime!r}: seed ordinals {sorted(ordinals['train'] & ordinals['val'])} "
        f"appear in both train and val"
    )
    if split.regime == "id":
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared_ord = ordinals[left] & ordinals[right]
                assert not shared_ord, (
                    f"regime 'id' splits on the seed ordinal, but ordinals "
                    f"{sorted(shared_ord)} appear in both {left} and {right}"
                )


def assert_no_shared_time_index(
    split: Split,
    n_samples_per_realization: int,
    lookback: int,
    max_horizon: int,
    stride: int,
) -> None:
    """Assert that no sample index is reachable from both a train and a test window.

    Vacuously true under realization-level splitting, which is exactly why it is worth
    asserting: it is the check that would catch a future refactor quietly introducing
    within-realization splitting. If such splitting is ever deliberately added, a guard
    band of ``lookback + max_horizon`` samples must separate the segments, and this
    function is where that requirement is enforced.

    Args:
        split: The split to check.
        n_samples_per_realization: Record length, samples.
        lookback: Input window length, samples.
        max_horizon: Longest forecast horizon, samples.
        stride: Window stride, samples.

    Raises:
        AssertionError: If any realization key appears in more than one partition, or if a
            future within-realization split violates the guard band.
    """
    spec = WindowSpec(lookback=lookback, horizons=(max_horizon,), stride=stride)
    starts = window_start_indices(n_samples_per_realization, spec)
    if starts.size == 0:
        covered: tuple[int, int] | None = None
    else:
        covered = (int(starts[0]), int(starts[-1]) + spec.total_length - 1)
    guard = lookback + max_horizon

    parts: dict[str, frozenset[RealizationKey]] = {
        "train": split.train_keys,
        "val": split.val_keys,
        "test": split.test_keys,
    }
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for key in sorted(parts[left] & parts[right]):
                assert covered is not None, (
                    f"realization {key} is in both {left} and {right}; it yields no windows "
                    f"at this geometry, but sharing a realization across partitions is "
                    f"forbidden regardless"
                )
                # Both partitions take *every* window of a shared realization, so the two
                # covered spans are identical and their separation is negative.
                separation = covered[0] - covered[1] - 1
                assert separation >= guard, (
                    f"realization {key} appears in both {left} and {right}: sample spans "
                    f"{covered} and {covered} are separated by {separation} samples, "
                    f"below the required guard band of {guard} "
                    f"(lookback {lookback} + max horizon {max_horizon})"
                )
