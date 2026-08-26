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
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = [
    "REGIMES",
    "Regime",
    "Split",
    "assert_no_shared_time_index",
    "assert_seed_disjoint",
    "build_split",
]

#: The four evaluation regimes. ``id`` measures in-distribution accuracy; the other three
#: measure a specific, named kind of generalisation.
Regime = Literal["id", "unseen_seastate", "unseen_heading", "unseen_vessel"]

#: Tuple form of :data:`Regime`, for iteration in evaluation drivers.
REGIMES: tuple[Regime, ...] = ("id", "unseen_seastate", "unseen_heading", "unseen_vessel")


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


def build_split(meta: pd.DataFrame, regime: Regime, val_frac: float = 0.15) -> Split:
    """Build the seed-disjoint split for one evaluation regime.

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
    raise NotImplementedError


def assert_seed_disjoint(split: Split) -> None:
    """Assert that the three partitions of a split share no realization.

    Args:
        split: The split to check.

    Raises:
        AssertionError: If any realization key appears in more than one partition. This is
            deliberately an assertion rather than a returned bool: there is no sensible way
            to continue from a leaking split, and a caller must not be able to ignore it.
    """
    raise NotImplementedError


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
    raise NotImplementedError
