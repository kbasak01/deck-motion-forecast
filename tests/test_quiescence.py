"""Quiescent-window detection -- the operational metric's own unit tests.

What is pinned here, and why each case exists:

1. **The sustain boundary.** A run of exactly ``sustain_s`` passes and one sample shorter
   fails. The conversion from seconds to samples is the whole content of the sustain rule,
   and an off-by-one there silently changes what "landable" means.
2. **Greedy one-to-one matching.** One predicted onset sitting between two true onsets is
   credited to exactly one of them. Without this, a detector inflates recall by emitting a
   single flag near a cluster.
3. **Base-rate exploitation, as a number.** An always-yes detector is scored against a
   corpus-realistic base rate (P6-D7: SS4/permissive is ~0.94) and its F1 is asserted, so
   the module docstring's interpretability warning is demonstrated rather than restated.
   Both readings are pinned: per-sample F1 0.969 (the warning, exactly as stated) and
   onset F1 0.018 with recall still a perfect 1.0 (the warning holds for recall, and the
   onset formulation resists it -- which is a property of the metric, not an assumption).
4. **The zero-scorable-onset cell.** SS3/permissive has no interior onsets at all
   (P6-D7), and the test asserts that the zero-denominator F1 is distinguishable from a
   real zero by the ``n_true`` the caller carries.
5. **Per-sea-state grouping.** F1 pooled across sea states differs from the per-sea-state
   F1s and is dominated by the quiet cells, which is why the reporting unit includes the
   sea state.

Units follow the package convention: degrees for roll and pitch, metres per second for
heave rate, seconds for durations, hertz for sampling rates, samples for indices.
"""

import numpy as np
import pytest

from dmf.eval.quiescence import (
    PERMISSIVE,
    STRICT,
    QuiescenceThresholds,
    base_rate,
    detect_quiescent_mask,
    false_alarms_per_minute,
    lead_times,
    match_onsets,
    precision_recall_f1,
    scorable_onsets,
    sustain_samples,
    window_onsets,
)
from dmf.typedefs import BoolArray, FloatArray, IntArray

#: Sampling rate of every fixture here, hertz. Matches the corpus.
FS_HZ = 10.0

#: Threshold set with a whole-second sustain, used where the arithmetic must be obvious.
ONE_SECOND = QuiescenceThresholds(3.0, 2.0, 0.8, 1.0, "one_second")


def _series(pattern: BoolArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build three motion series that are inside limits exactly where ``pattern`` is True.

    Values are chosen well inside or well outside the permissive limits so the test does
    not depend on comparison-boundary behaviour, which is covered separately.
    """
    inside = np.where(pattern, 0.5, 10.0)
    return inside.astype(np.float64), (0.5 * inside).astype(np.float64), (0.05 * inside)


# ---------------------------------------------------------------------------
# thresholds and the sustain conversion
# ---------------------------------------------------------------------------


def test_shipped_threshold_sets_match_the_plan() -> None:
    assert (PERMISSIVE.roll_deg, PERMISSIVE.pitch_deg, PERMISSIVE.heave_rate_mps) == (3.0, 2.0, 0.8)
    assert PERMISSIVE.sustain_s == 2.0
    assert (STRICT.roll_deg, STRICT.pitch_deg, STRICT.heave_rate_mps) == (1.5, 1.0, 0.4)
    assert STRICT.sustain_s == 3.0


@pytest.mark.parametrize(
    ("sustain_s", "expected"),
    [(2.0, 20), (3.0, 30), (0.05, 1), (0.11, 2), (0.0, 1)],
)
def test_sustain_conversion_rounds_up_and_never_below_one(sustain_s: float, expected: int) -> None:
    assert sustain_samples(sustain_s, FS_HZ) == expected


def test_a_run_of_exactly_the_sustain_duration_passes() -> None:
    pattern = np.zeros(60, dtype=np.bool_)
    pattern[10:30] = True  # 20 samples = 2.0 s at 10 Hz, exactly PERMISSIVE.sustain_s
    mask = detect_quiescent_mask(*_series(pattern), PERMISSIVE, FS_HZ)
    assert mask.sum() == 20
    assert window_onsets(mask).tolist() == [10]


def test_a_run_one_sample_short_of_the_sustain_duration_fails() -> None:
    pattern = np.zeros(60, dtype=np.bool_)
    pattern[10:29] = True  # 19 samples = 1.9 s
    mask = detect_quiescent_mask(*_series(pattern), PERMISSIVE, FS_HZ)
    assert not mask.any()
    assert window_onsets(mask).size == 0


def test_only_the_runs_that_reach_the_sustain_survive() -> None:
    pattern = np.zeros(80, dtype=np.bool_)
    pattern[5:14] = True  # 9 samples, fails a 1.0 s sustain
    pattern[20:35] = True  # 15 samples, passes
    pattern[60:70] = True  # 10 samples, exactly 1.0 s, passes
    mask = detect_quiescent_mask(*_series(pattern), ONE_SECOND, FS_HZ)
    assert window_onsets(mask).tolist() == [20, 60]
    assert mask.sum() == 25


def test_all_three_limits_must_hold_simultaneously() -> None:
    n = 40
    roll = np.full(n, 0.5)
    pitch = np.full(n, 0.5)
    heave_rate = np.full(n, 0.1)
    pitch[15] = 5.0  # breaches pitch alone, and must break the run
    mask = detect_quiescent_mask(roll, pitch, heave_rate, ONE_SECOND, FS_HZ)
    assert window_onsets(mask).tolist() == [0, 16]
    assert not mask[15]


def test_the_limit_is_inclusive_at_the_boundary() -> None:
    n = 30
    at_limit = detect_quiescent_mask(
        np.full(n, PERMISSIVE.roll_deg),
        np.full(n, PERMISSIVE.pitch_deg),
        np.full(n, PERMISSIVE.heave_rate_mps),
        PERMISSIVE,
        FS_HZ,
    )
    assert at_limit.all()
    just_over = detect_quiescent_mask(
        np.full(n, PERMISSIVE.roll_deg + 1e-6),
        np.full(n, PERMISSIVE.pitch_deg),
        np.full(n, PERMISSIVE.heave_rate_mps),
        PERMISSIVE,
        FS_HZ,
    )
    assert not just_over.any()


def test_a_nan_sample_breaks_the_run_it_sits_in() -> None:
    n = 60
    roll = np.full(n, 0.5)
    roll[30] = np.nan
    mask = detect_quiescent_mask(roll, np.full(n, 0.5), np.full(n, 0.1), ONE_SECOND, FS_HZ)
    assert not mask[30]
    assert window_onsets(mask).tolist() == [0, 31]


def test_detect_rejects_series_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        detect_quiescent_mask(np.zeros(10), np.zeros(11), np.zeros(10), PERMISSIVE, FS_HZ)


# ---------------------------------------------------------------------------
# onsets and the exclusion rule
# ---------------------------------------------------------------------------


def test_window_onsets_are_the_first_sample_of_each_run() -> None:
    mask = np.zeros(30, dtype=np.bool_)
    mask[2:6] = True
    mask[10:11] = True
    mask[25:] = True
    assert window_onsets(mask).tolist() == [2, 10, 25]


def test_window_onsets_of_an_empty_or_all_true_mask() -> None:
    assert window_onsets(np.zeros(10, dtype=np.bool_)).size == 0
    assert window_onsets(np.ones(10, dtype=np.bool_)).tolist() == [0]


def test_scorable_onsets_excludes_the_boundary_runs_and_counts_them() -> None:
    mask = np.zeros(40, dtype=np.bool_)
    mask[0:5] = True  # already under way at the first decision time
    mask[10:20] = True  # interior, scorable
    mask[35:] = True  # right-censored by the record end
    onsets, excluded = scorable_onsets(mask)
    assert onsets.tolist() == [10]
    assert excluded == 2


def test_ss3_permissive_geometry_yields_zero_scorable_onsets() -> None:
    """P6-D7: the deck never leaves permissive limits, so the only onset is the record."""
    mask = detect_quiescent_mask(*_series(np.ones(600, dtype=np.bool_)), PERMISSIVE, FS_HZ)
    assert base_rate(mask) == 1.0
    assert window_onsets(mask).tolist() == [0]
    onsets, excluded = scorable_onsets(mask)
    assert onsets.size == 0
    assert excluded == 1


def test_scorable_onsets_respects_a_later_first_decision_time() -> None:
    mask = np.zeros(60, dtype=np.bool_)
    mask[8:20] = True
    mask[30:40] = True
    onsets, excluded = scorable_onsets(mask, first_decision_idx=8)
    assert onsets.tolist() == [30]
    assert excluded == 1


def test_scorable_onsets_rejects_a_negative_decision_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        scorable_onsets(np.zeros(5, dtype=np.bool_), first_decision_idx=-1)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def _idx(*values: int) -> IntArray:
    """Build an int64 index array."""
    return np.asarray(values, dtype=np.int64)


def test_one_predicted_onset_between_two_true_onsets_is_credited_once() -> None:
    """The greedy one-to-one rule: a single flag cannot satisfy two true onsets."""
    true = _idx(100, 108)
    predicted = _idx(104)  # 0.4 s from each, both inside a 0.5 s tolerance
    matched_pred, matched_true, unmatched_true = match_onsets(predicted, true, FS_HZ, 0.5)
    assert matched_pred.size == 1
    assert matched_true.size == 1
    assert unmatched_true.size == 1
    assert set(matched_true.tolist()) | set(unmatched_true.tolist()) == {0, 1}
    precision, recall, f1 = precision_recall_f1(int(matched_pred.size), 1, 2)
    assert precision == 1.0
    assert recall == 0.5
    assert f1 == pytest.approx(2.0 / 3.0)


def test_matching_prefers_the_nearer_true_onset() -> None:
    true = _idx(100, 107)
    predicted = _idx(102)
    _, matched_true, unmatched_true = match_onsets(predicted, true, FS_HZ, 0.5)
    assert matched_true.tolist() == [0]
    assert unmatched_true.tolist() == [1]


def test_matching_is_globally_greedy_not_a_left_to_right_sweep() -> None:
    """Predicted 0 is within tolerance of true 0, but true 0's nearer claimant is pred 1."""
    true = _idx(100, 140)
    predicted = _idx(103, 100)
    matched_pred, matched_true, unmatched_true = match_onsets(predicted, true, FS_HZ, 0.5)
    assert matched_true.tolist() == [0]
    assert matched_pred.tolist() == [1]
    assert unmatched_true.tolist() == [1]


def test_matching_is_independent_of_input_order() -> None:
    true = _idx(50, 90, 130)
    forward = match_onsets(_idx(52, 88, 133), true, FS_HZ, 0.5)
    reversed_ = match_onsets(_idx(133, 88, 52), true, FS_HZ, 0.5)
    assert forward[1].tolist() == reversed_[1].tolist() == [0, 1, 2]
    assert sorted(forward[0].tolist()) == [0, 1, 2]
    assert sorted(reversed_[0].tolist()) == [0, 1, 2]


def test_a_prediction_outside_tolerance_is_not_matched() -> None:
    matched_pred, _, unmatched_true = match_onsets(_idx(106), _idx(100), FS_HZ, 0.5)
    assert matched_pred.size == 0
    assert unmatched_true.tolist() == [0]


def test_a_prediction_exactly_at_the_tolerance_is_matched() -> None:
    matched_pred, _, unmatched_true = match_onsets(_idx(105), _idx(100), FS_HZ, 0.5)
    assert matched_pred.tolist() == [0]
    assert unmatched_true.size == 0


def test_matching_with_an_empty_side_returns_every_true_onset_unmatched() -> None:
    matched_pred, matched_true, unmatched_true = match_onsets(
        np.zeros(0, dtype=np.int64), _idx(10, 20), FS_HZ, 0.5
    )
    assert matched_pred.size == matched_true.size == 0
    assert unmatched_true.tolist() == [0, 1]


def test_match_onsets_rejects_a_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        match_onsets(_idx(1), _idx(1), FS_HZ, -0.1)


# ---------------------------------------------------------------------------
# precision / recall / F1
# ---------------------------------------------------------------------------


def test_precision_recall_f1_on_a_worked_example() -> None:
    precision, recall, f1 = precision_recall_f1(6, 10, 8)
    assert precision == pytest.approx(0.6)
    assert recall == pytest.approx(0.75)
    assert f1 == pytest.approx(2 * 0.6 * 0.75 / 1.35)


@pytest.mark.parametrize(
    ("matched", "predicted", "true"),
    [(0, 0, 5), (0, 5, 0), (0, 0, 0)],
)
def test_a_zero_denominator_returns_zero_not_nan(matched: int, predicted: int, true: int) -> None:
    values = precision_recall_f1(matched, predicted, true)
    assert values == (0.0, 0.0, 0.0)
    assert not any(np.isnan(v) for v in values)


def test_the_not_scorable_cell_is_distinguishable_only_by_its_counts() -> None:
    """P6-D7: an F1 of 0.0 from an empty cell must not be read as a model failure.

    The function cannot tell the two apart, and is not asked to. The counts the caller
    carries can, which is why every results row ships ``n_true`` beside its F1.
    """
    empty_cell = precision_recall_f1(0, 0, 0)
    total_miss = precision_recall_f1(0, 0, 12)
    assert empty_cell == total_miss == (0.0, 0.0, 0.0)
    n_true_empty, n_true_miss = 0, 12
    assert n_true_empty == 0 and n_true_miss > 0


@pytest.mark.parametrize(
    ("matched", "predicted", "true", "message"),
    [
        (-1, 3, 3, "non-negative"),
        (1, -3, 3, "non-negative"),
        (1, 3, -3, "non-negative"),
        (4, 3, 5, "exceeds n_predicted"),
        (4, 5, 3, "exceeds n_true"),
    ],
)
def test_precision_recall_f1_rejects_impossible_counts(
    matched: int, predicted: int, true: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        precision_recall_f1(matched, predicted, true)


# ---------------------------------------------------------------------------
# lead times and false alarms
# ---------------------------------------------------------------------------


def test_lead_time_is_positive_when_the_flag_precedes_the_onset() -> None:
    leads = lead_times(_idx(90, 200), _idx(100, 205), FS_HZ)
    assert leads.tolist() == [1.0, 0.5]


def test_lead_time_is_negative_when_the_flag_is_late() -> None:
    assert lead_times(_idx(105), _idx(100), FS_HZ).tolist() == [-0.5]


def test_lead_times_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        lead_times(_idx(1, 2), _idx(1), FS_HZ)


def test_false_alarms_per_minute_on_a_ten_minute_record() -> None:
    assert false_alarms_per_minute(30, 600.0) == pytest.approx(3.0)
    assert false_alarms_per_minute(0, 600.0) == 0.0


@pytest.mark.parametrize("duration_s", [0.0, -1.0])
def test_false_alarms_rejects_a_non_positive_duration(duration_s: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        false_alarms_per_minute(1, duration_s)


# ---------------------------------------------------------------------------
# base rate -- the interpretability point, asserted
# ---------------------------------------------------------------------------


def test_base_rate_is_the_fraction_of_sustained_samples() -> None:
    mask = np.zeros(100, dtype=np.bool_)
    mask[10:40] = True
    assert base_rate(mask) == pytest.approx(0.30)
    assert base_rate(np.ones(50, dtype=np.bool_)) == 1.0
    assert base_rate(np.zeros(50, dtype=np.bool_)) == 0.0
    assert base_rate(np.zeros(0, dtype=np.bool_)) == 0.0


def test_an_always_yes_detector_scores_high_f1_against_a_high_base_rate() -> None:
    """The module docstring's interpretability warning, as numbers rather than as prose.

    A corpus-realistic SS4/permissive cell (P6-D7: base rate ~0.94): the deck is quiescent
    almost all the time, broken by short excursions. The detector under test says
    "quiescent" always and has learned nothing.

    Two readings, both asserted, because they disagree and the disagreement is the point:

    * **Per-sample**, which is the quantity the base rate is a rate *of*: precision is the
      base rate itself, recall is 1.0 by construction, and F1 is **0.969**. That is the
      warning, demonstrated. An F1 of 0.969 here carries no information whatsoever, and the
      only column that says so is the 0.94 sitting beside it.
    * **On onsets with a 0.5 s tolerance**, which is what §6.2 actually reports: recall is
      still a perfect 1.0 -- so a table showing recall without precision is fully gamed --
      but precision collapses to 0.009 and F1 to 0.018, because a window *onset* is a rare
      event even when the deck is quiet 94 percent of the time.

    The onset formulation is therefore **not** base-rate-exploitable in the naive way, and
    that is a property of the metric worth stating rather than assuming; what remains
    exploitable is recall alone, and the per-sample reading.
    """
    n = 6000
    pattern = np.ones(n, dtype=np.bool_)
    for start in range(300, n, 500):  # 12 excursions of 3.0 s each
        pattern[start : start + 30] = False
    mask = detect_quiescent_mask(*_series(pattern), PERMISSIVE, FS_HZ)
    rate = base_rate(mask)
    assert rate == pytest.approx(0.94, abs=1e-12)

    # Per-sample reading: TP = every quiescent sample, predicted positives = every sample.
    n_quiescent = int(mask.sum())
    precision, recall, f1 = precision_recall_f1(n_quiescent, n, n_quiescent)
    assert precision == pytest.approx(rate)
    assert recall == 1.0
    assert f1 == pytest.approx(0.969072, abs=5e-6)
    assert f1 > 0.95

    # Onset reading, at the production 0.5 s decision stride. The first run is already
    # under way at the first decision time and the last is right-censored, so 11 of the 12
    # excursion-terminated runs carry a scorable onset (the exclusion rule, P6-D7).
    true_onsets, excluded = scorable_onsets(mask)
    assert true_onsets.size == 11
    assert excluded == 2

    predicted = np.arange(0, n, 5, dtype=np.int64)
    matched_pred, _, unmatched_true = match_onsets(predicted, true_onsets, FS_HZ, 0.5)
    onset_p, onset_r, onset_f1 = precision_recall_f1(
        int(matched_pred.size), int(predicted.size), int(true_onsets.size)
    )
    assert unmatched_true.size == 0
    assert onset_r == 1.0
    assert onset_p == pytest.approx(11.0 / 1200.0, abs=1e-9)
    assert onset_f1 == pytest.approx(0.018166, abs=5e-6)
    assert onset_f1 < 0.05 < f1

    # And the false-alarm rate, which is the column a controls engineer reads first:
    # 1189 unmatched flags over a 600 s record.
    assert false_alarms_per_minute(
        int(predicted.size - matched_pred.size), n / FS_HZ
    ) == pytest.approx(118.9)


def test_pooling_across_sea_states_reports_corpus_composition_not_skill() -> None:
    """P6-D7 consequence 3: F1 is never pooled across sea states.

    Two cells, one quiet (few onsets, all recovered) and one rough (many onsets, half
    recovered). The pooled F1 sits well above the rough cell's, and the rough cell is the
    operationally interesting one -- so the pooled row measures which cells the corpus
    happens to contain.
    """
    quiet = {"n_true": 40, "n_pred": 40, "n_matched": 40}
    rough = {"n_true": 120, "n_pred": 100, "n_matched": 50}

    _, _, f1_quiet = precision_recall_f1(quiet["n_matched"], quiet["n_pred"], quiet["n_true"])
    _, _, f1_rough = precision_recall_f1(rough["n_matched"], rough["n_pred"], rough["n_true"])
    _, _, f1_pooled = precision_recall_f1(
        quiet["n_matched"] + rough["n_matched"],
        quiet["n_pred"] + rough["n_pred"],
        quiet["n_true"] + rough["n_true"],
    )

    assert f1_quiet == 1.0
    assert f1_rough == pytest.approx(0.454545, abs=5e-6)
    assert f1_pooled == pytest.approx(0.6, abs=5e-6)
    assert f1_pooled - f1_rough > 0.14


def test_a_perfect_detector_scores_one_and_a_silent_one_scores_zero() -> None:
    mask = np.zeros(400, dtype=np.bool_)
    mask[50:100] = True
    mask[200:260] = True
    onsets, _ = scorable_onsets(mask)
    assert onsets.tolist() == [50, 200]

    matched, _, unmatched = match_onsets(onsets, onsets, FS_HZ, 0.5)
    assert precision_recall_f1(int(matched.size), int(onsets.size), int(onsets.size)) == (
        1.0,
        1.0,
        1.0,
    )
    assert unmatched.size == 0
    assert precision_recall_f1(0, 0, int(onsets.size)) == (0.0, 0.0, 0.0)
