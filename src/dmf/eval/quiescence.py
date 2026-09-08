"""Quiescent-window detection -- the operational metric.

This is the question a landing controller actually asks: can the model predict, 2-3 s
ahead, a window in which ``|roll|``, ``|pitch|`` and ``|heave rate|`` all stay below
landing limits for a sustained interval? It is what converts "I trained a TCN" into "I
built the module that decides when to commit to touchdown", and it is largely absent from
a deck-motion-forecasting literature dominated by RMSE tables.

Ground truth is obtained by applying the thresholds to the **true** future trajectory;
the prediction is obtained by applying the same thresholds to the **forecast**. Onsets are
matched with a tolerance, since being half a second early on an onset is not the same kind
of error as missing it.

**Always report the base rate.** A model can post a high F1 purely by exploiting a high
base rate of quiescent windows -- at SS3 the deck is quiet most of the time, and a detector
that says "yes" constantly will look excellent. The base rate is what makes the F1
interpretable, so :func:`base_rate` output sits next to every F1 in the results tables.
``tests/test_quiescence.py`` asserts that point numerically rather than restating it. On a
**synthetic** fixture at base rate 0.94 -- a square wave of 12 three-second excursions, not
a corpus cell, and the distinction matters because P6-D19 measured the corpus SS4/permissive
base rate as 0.938/0.895/0.806 by speed rather than a single number -- a detector that always
says "quiescent" scores a **per-sample F1 of 0.969** while knowing nothing at all. Scored on
*onsets* with a 0.5 s tolerance -- which is what the results table reports -- the same
detector still gets **recall 1.0**, so a recall column read on its own is fully gamed, while
precision falls to 11/1200 and F1 to 0.018. The onset formulation is therefore not
base-rate-exploitable in the naive way; recall alone, and any per-sample reading, still is.
That distinction is measured in the test, not assumed here.

**Those onset figures are a property of the fixture's arithmetic, not a chance level.** The
always-yes detector's precision is ``n_true / n_decisions`` by construction, so its F1 moves
with the cell's onset count and says nothing about how hard the cell is. The reference that
does is :data:`dmf.eval.quiescence_runner.RATE_MATCHED`, which emits the true *number* of
onsets at uniformly spaced times: it cannot be dismissed for over-flagging, so its F1 is the
chance of getting the *timing* right.

Two conventions are fixed here and not re-litigated downstream (``docs/protocol.md``
P6-D2):

1. **A run of ``k`` consecutive in-limit samples lasts ``k / fs_hz`` seconds.** This is the
   dwell-counting convention the rest of the project already uses for record length
   (``duration_s = n_samples / fs_hz`` in :class:`dmf.config.SimConfig`), so a 2.0 s
   sustain at 10 Hz is satisfied by exactly 20 samples and failed by 19. The alternative
   endpoint-to-endpoint convention (``(k - 1) / fs_hz``) would require 21 and would make
   the same threshold mean two different things in two places.
2. **Matching is one-to-one, greedy by absolute proximity.** A single predicted onset
   sitting between two true onsets is credited to exactly one of them, so a detector cannot
   inflate recall by emitting one flag near a cluster of true onsets.
3. **A run that begins before the first decision time, or that extends past the end of the
   record, is excluded from both the truth side and the prediction side**, and the excluded
   count is reported (:func:`scorable_onsets`). Its onset is not an event the detector could
   have called: on the truth side it began where nothing was observed, and on the prediction
   side it is right-censored. This is what turns the one raw onset per SS3/permissive
   realization -- the record merely starting inside a quiescent window -- into zero.

**Two things the caller must carry that these functions deliberately do not (P6-D7).**

*Scorability.* :func:`precision_recall_f1` returns 0.0 on a zero denominator, which is the
right answer for a pure function and the wrong thing to print. Measured on the corpus,
SS3/permissive has **zero** scorable onsets in every realization sampled: the deck is inside
permissive limits for the whole 600 s record. That cell must be rendered ``not scorable,
base_rate = 1.000``, never F1 = 0 (which reads as model failure) and never F1 = 1 (which
reads as success). Every results row therefore carries ``n_true`` beside its F1, and a row
with ``n_true == 0`` is a row with no measurement in it.

*Grouping.* F1 is **never pooled across sea states**. At ``permissive`` the base rate is
~1.00 at SS3 and ~0.94 at SS4 with almost no onsets, so a pooled F1 measures corpus
composition rather than a model. The reporting unit is
``(model, regime, threshold_set, rule, sea state)``, and the base rate sits in the same row.
"""

import math
from dataclasses import dataclass

import numpy as np

from dmf.typedefs import BoolArray, FloatArray, IntArray

__all__ = [
    "PERMISSIVE",
    "STRICT",
    "QuiescenceThresholds",
    "base_rate",
    "detect_quiescent_mask",
    "false_alarms_per_minute",
    "lead_times",
    "match_onsets",
    "precision_recall_f1",
    "scorable_onsets",
    "sustain_samples",
    "window_onsets",
]

#: Floating-point slack used when converting a duration in seconds to a sample count and
#: when comparing a timing difference against a tolerance. ``2.0 * 10.0`` is not exactly
#: ``20.0`` for every representable pair, and a run of exactly ``sustain_s`` must pass.
_EPS: float = 1e-9


@dataclass(frozen=True)
class QuiescenceThresholds:
    """Operational limits defining a landable window.

    Attributes:
        roll_deg: Maximum absolute roll, **degrees**.
        pitch_deg: Maximum absolute pitch, **degrees**.
        heave_rate_mps: Maximum absolute heave rate, **metres per second**.
        sustain_s: Minimum duration for which all three limits must hold simultaneously,
            **seconds**.
        name: Threshold-set label, used as a results column value.
    """

    roll_deg: float
    pitch_deg: float
    heave_rate_mps: float
    sustain_s: float
    name: str


#: Permissive landing limits: 3.0 deg roll, 2.0 deg pitch, 0.8 m/s heave rate, sustained
#: for at least 2.0 s.
PERMISSIVE: QuiescenceThresholds = QuiescenceThresholds(3.0, 2.0, 0.8, 2.0, "permissive")

#: Strict landing limits: 1.5 deg roll, 1.0 deg pitch, 0.4 m/s heave rate, sustained for
#: at least 3.0 s.
STRICT: QuiescenceThresholds = QuiescenceThresholds(1.5, 1.0, 0.4, 3.0, "strict")


def sustain_samples(sustain_s: float, fs_hz: float) -> int:
    """Convert a sustain requirement in seconds to a minimum run length in samples.

    The rounding rule is stated once, here, so that the sustain threshold means the same
    thing in the detector and in any test that constructs a run by hand: a run of ``k``
    consecutive samples lasts ``k / fs_hz`` seconds, therefore the minimum run length is
    ``ceil(sustain_s * fs_hz)``, and a run of *exactly* ``sustain_s`` passes. The
    :data:`_EPS` slack absorbs the case where ``sustain_s * fs_hz`` lands a fraction of an
    ulp above an integer.

    Args:
        sustain_s: Required dwell time, **seconds**.
        fs_hz: Sampling rate, hertz.

    Returns:
        Minimum number of consecutive in-limit samples, at least 1.
    """
    return max(1, int(math.ceil(sustain_s * fs_hz - _EPS)))


def _run_bounds(mask: BoolArray) -> tuple[IntArray, IntArray]:
    """Return the half-open bounds of every True run in a boolean mask.

    Args:
        mask: Boolean mask, shape ``(n_samples,)``.

    Returns:
        Tuple ``(starts, stops)`` of int64 arrays, shape ``(n_runs,)`` each, such that
        ``mask[starts[i]:stops[i]]`` is all True and is bounded by False or by the array
        ends.
    """
    flags = np.asarray(mask, dtype=bool).astype(np.int8)
    padded = np.concatenate(([np.int8(0)], flags, [np.int8(0)]))
    edges = np.diff(padded.astype(np.int16))
    starts = np.flatnonzero(edges == 1).astype(np.int64)
    stops = np.flatnonzero(edges == -1).astype(np.int64)
    return starts, stops


def detect_quiescent_mask(
    roll_deg: FloatArray,
    pitch_deg: FloatArray,
    heave_rate_mps: FloatArray,
    thresholds: QuiescenceThresholds,
    fs_hz: float,
) -> BoolArray:
    """Mark the samples that lie inside a sustained quiescent window.

    A sample is marked True only if it belongs to a run of consecutive samples, all of
    which satisfy all three limits, whose duration is at least ``thresholds.sustain_s``.
    The sustain requirement is what distinguishes a landable window from an instantaneous
    threshold crossing.

    Duration is counted as ``k / fs_hz`` for a run of ``k`` samples
    (:func:`sustain_samples`), so a run of exactly ``sustain_s`` passes and one sample
    shorter fails. A NaN sample fails every comparison and therefore breaks the run it sits
    in, which is the conservative reading: an unobserved deck is not a landable deck.

    Args:
        roll_deg: Roll series, **degrees**, shape ``(n_samples,)``.
        pitch_deg: Pitch series, **degrees**, shape ``(n_samples,)``.
        heave_rate_mps: Heave rate series, **metres per second**, shape ``(n_samples,)``.
        thresholds: The limit set to apply.
        fs_hz: Sampling rate, hertz, used to convert ``sustain_s`` to samples.

    Returns:
        Boolean mask, shape ``(n_samples,)``, True inside sustained quiescent windows.

    Raises:
        ValueError: If the three series differ in length.
    """
    roll = np.asarray(roll_deg, dtype=np.float64)
    pitch = np.asarray(pitch_deg, dtype=np.float64)
    heave_rate = np.asarray(heave_rate_mps, dtype=np.float64)
    lengths = {roll.shape, pitch.shape, heave_rate.shape}
    if len(lengths) != 1:
        raise ValueError(
            f"roll, pitch and heave_rate must have the same shape, got "
            f"{roll.shape}, {pitch.shape}, {heave_rate.shape}"
        )

    with np.errstate(invalid="ignore"):
        instantaneous = (
            (np.abs(roll) <= thresholds.roll_deg)
            & (np.abs(pitch) <= thresholds.pitch_deg)
            & (np.abs(heave_rate) <= thresholds.heave_rate_mps)
        )

    minimum = sustain_samples(thresholds.sustain_s, fs_hz)
    sustained: BoolArray = np.zeros(instantaneous.shape, dtype=np.bool_)
    starts, stops = _run_bounds(instantaneous)
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        if stop - start >= minimum:
            sustained[start:stop] = True
    return sustained


def window_onsets(mask: BoolArray) -> IntArray:
    """Find the first sample index of each quiescent window.

    Args:
        mask: Quiescence mask from :func:`detect_quiescent_mask`, shape ``(n_samples,)``.

    Returns:
        Onset sample indices, shape ``(n_windows,)``, ascending.
    """
    starts, _ = _run_bounds(mask)
    return starts


def scorable_onsets(mask: BoolArray, first_decision_idx: int = 0) -> tuple[IntArray, int]:
    """Return the onsets a detector could actually have been scored on, and the rest.

    Two kinds of run carry an onset that is not an event: one that is already under way at
    the first decision time -- its true onset lies where nothing was observed, so the record
    merely starts inside a quiescent window -- and one still under way at the end of the
    record, which is right-censored. Both are excluded, from the truth side and from the
    prediction side alike, and the excluded count is returned so that the exclusion appears
    in the results table instead of silently shrinking a denominator (P6-D7).

    This is not a cosmetic filter. Measured on the corpus, SS3/permissive yields exactly one
    raw onset per 600 s realization and **zero** after this rule, because the deck never
    leaves permissive limits; scoring the raw onset would manufacture a measurement out of
    the record boundary.

    Args:
        mask: Quiescence mask from :func:`detect_quiescent_mask`, shape ``(n_samples,)``.
        first_decision_idx: Index of the first sample at which the detector was able to
            emit a decision, samples. Runs starting at or before it are excluded, since a
            run starting exactly there may have begun earlier.

    Returns:
        Tuple ``(onsets, n_excluded)``: ascending onset sample indices of the runs that
        survive, and the number of runs removed by either rule.

    Raises:
        ValueError: If ``first_decision_idx`` is negative.
    """
    if first_decision_idx < 0:
        raise ValueError(f"first_decision_idx must be non-negative, got {first_decision_idx}")
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    starts, stops = _run_bounds(flags)
    keep = (starts > first_decision_idx) & (stops < flags.size)
    return starts[keep], int(np.count_nonzero(~keep))


def match_onsets(
    predicted_onsets: IntArray,
    true_onsets: IntArray,
    fs_hz: float,
    tolerance_s: float = 0.5,
) -> tuple[IntArray, IntArray, IntArray]:
    """Match predicted window onsets to true onsets within a timing tolerance.

    Matching is one-to-one and greedy by proximity, so one predicted onset cannot claim
    credit for two true onsets.

    Candidate pairs within ``tolerance_s`` are visited in order of increasing absolute
    timing difference, ties broken by true index then predicted index, and a pair is
    accepted only if neither of its members is already spoken for. That ordering is what
    makes the result independent of the input order, which a nearest-neighbour sweep over
    the predictions would not be.

    Args:
        predicted_onsets: Onset sample indices from the forecast.
        true_onsets: Onset sample indices from the true trajectory.
        fs_hz: Sampling rate, hertz.
        tolerance_s: Maximum absolute timing difference for a match, **seconds**.

    Returns:
        Tuple ``(matched_pred_idx, matched_true_idx, unmatched_true_idx)``, each an array
        of indices into the corresponding input array. The two matched arrays are aligned
        pair-wise and are ordered by ``matched_true_idx`` ascending;
        ``unmatched_true_idx`` is ascending.

    Raises:
        ValueError: If ``tolerance_s`` is negative.
    """
    if tolerance_s < 0.0:
        raise ValueError(f"tolerance_s must be non-negative, got {tolerance_s}")
    predicted = np.asarray(predicted_onsets, dtype=np.int64).reshape(-1)
    true = np.asarray(true_onsets, dtype=np.int64).reshape(-1)

    if predicted.size == 0 or true.size == 0:
        empty: IntArray = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.arange(true.size, dtype=np.int64)

    delta_s = np.abs(predicted[:, None] - true[None, :]).astype(np.float64) / fs_hz
    within = np.flatnonzero((delta_s <= tolerance_s + _EPS).reshape(-1))
    pred_idx, true_idx = np.unravel_index(within, delta_s.shape)
    order = np.lexsort((pred_idx, true_idx, delta_s.reshape(-1)[within]))

    pred_taken = np.zeros(predicted.size, dtype=bool)
    true_taken = np.zeros(true.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    for position in order.tolist():
        p = int(pred_idx[position])
        t = int(true_idx[position])
        if pred_taken[p] or true_taken[t]:
            continue
        pred_taken[p] = True
        true_taken[t] = True
        pairs.append((t, p))

    pairs.sort()
    matched_true = np.asarray([t for t, _ in pairs], dtype=np.int64)
    matched_pred = np.asarray([p for _, p in pairs], dtype=np.int64)
    unmatched_true = np.flatnonzero(~true_taken).astype(np.int64)
    return matched_pred, matched_true, unmatched_true


def precision_recall_f1(
    n_matched: int, n_predicted: int, n_true: int
) -> tuple[float, float, float]:
    """Compute precision, recall and F1 on window onsets.

    Args:
        n_matched: Predicted onsets matched to a true onset.
        n_predicted: Total predicted onsets.
        n_true: Total true onsets.

    A zero denominator returns 0.0, which is correct for a pure function and misleading in
    a table: ``n_true == 0`` means the cell had nothing to detect, not that the detector
    failed. The caller keeps ``n_true`` in the row and renders such a cell as *not scorable*
    (P6-D7); see the module docstring.

    Returns:
        Tuple ``(precision, recall, f1)``, each dimensionless in [0, 1]. Any quantity with
        a zero denominator is returned as 0.0.

    Raises:
        ValueError: If any count is negative, or ``n_matched`` exceeds ``n_predicted`` or
            ``n_true``.
    """
    for name, value in (
        ("n_matched", n_matched),
        ("n_predicted", n_predicted),
        ("n_true", n_true),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if n_matched > n_predicted:
        raise ValueError(
            f"n_matched ({n_matched}) exceeds n_predicted ({n_predicted}); matching is "
            f"one-to-one, so a match consumes exactly one predicted onset"
        )
    if n_matched > n_true:
        raise ValueError(
            f"n_matched ({n_matched}) exceeds n_true ({n_true}); matching is one-to-one, "
            f"so a match consumes exactly one true onset"
        )
    precision = n_matched / n_predicted if n_predicted > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return float(precision), float(recall), float(f1)


def lead_times(
    predicted_flag_idx: IntArray,
    true_onsets: IntArray,
    fs_hz: float,
) -> FloatArray:
    """Measure how far ahead of each true onset the model first flagged the window.

    Lead time, not accuracy, is what determines whether a forecast is actionable: a
    correct call issued 0.2 s before touchdown gives the controller no time to commit.

    Args:
        predicted_flag_idx: Sample index at which the model first flagged each matched
            window.
        true_onsets: Sample index of the corresponding true onset.
        fs_hz: Sampling rate, hertz.

    Returns:
        Lead time per matched window, **seconds**, shape ``(n_matched,)``. Positive means
        the model flagged the window before it began.

    Raises:
        ValueError: If the two index arrays differ in length.
    """
    flags = np.asarray(predicted_flag_idx, dtype=np.int64).reshape(-1)
    onsets = np.asarray(true_onsets, dtype=np.int64).reshape(-1)
    if flags.shape != onsets.shape:
        raise ValueError(
            f"predicted_flag_idx and true_onsets must have the same length, got "
            f"{flags.size} and {onsets.size}"
        )
    return (onsets - flags).astype(np.float64) / fs_hz


def false_alarms_per_minute(n_false_positives: int, duration_s: float) -> float:
    """Compute the false-alarm rate.

    This is the number a controls engineer asks for first, because it sets how often the
    autonomy will commit to a touchdown that the deck will not support.

    Args:
        n_false_positives: Predicted onsets with no matching true onset.
        duration_s: Total evaluated record duration, **seconds**.

    Returns:
        False alarms per minute, dimensionless per minute.

    Raises:
        ValueError: If ``duration_s`` is not strictly positive.
    """
    if not duration_s > 0.0:
        raise ValueError(f"duration_s must be strictly positive, got {duration_s}")
    return 60.0 * float(n_false_positives) / float(duration_s)


def base_rate(mask: BoolArray) -> float:
    """Compute the fraction of time the deck is genuinely quiescent.

    Reported next to every F1. Without it, an F1 of 0.9 is uninterpretable: it may
    represent real skill, or it may represent a sea state in which the deck is quiet 85
    percent of the time and a constant "yes" scores nearly as well.

    Args:
        mask: True-trajectory quiescence mask, shape ``(n_samples,)``.

    Returns:
        Fraction of samples inside a sustained quiescent window, dimensionless in [0, 1].
        An empty mask returns 0.0 rather than NaN, so that a regime with no evaluated
        samples does not poison a mean over regimes.
    """
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if flags.size == 0:
        return 0.0
    return float(np.count_nonzero(flags)) / float(flags.size)
