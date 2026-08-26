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
"""

from dataclasses import dataclass

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
    "window_onsets",
]


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
    raise NotImplementedError


def window_onsets(mask: BoolArray) -> IntArray:
    """Find the first sample index of each quiescent window.

    Args:
        mask: Quiescence mask from :func:`detect_quiescent_mask`, shape ``(n_samples,)``.

    Returns:
        Onset sample indices, shape ``(n_windows,)``, ascending.
    """
    raise NotImplementedError


def match_onsets(
    predicted_onsets: IntArray,
    true_onsets: IntArray,
    fs_hz: float,
    tolerance_s: float = 0.5,
) -> tuple[IntArray, IntArray, IntArray]:
    """Match predicted window onsets to true onsets within a timing tolerance.

    Matching is one-to-one and greedy by proximity, so one predicted onset cannot claim
    credit for two true onsets.

    Args:
        predicted_onsets: Onset sample indices from the forecast.
        true_onsets: Onset sample indices from the true trajectory.
        fs_hz: Sampling rate, hertz.
        tolerance_s: Maximum absolute timing difference for a match, **seconds**.

    Returns:
        Tuple ``(matched_pred_idx, matched_true_idx, unmatched_true_idx)``, each an array
        of indices into the corresponding input array.

    Raises:
        ValueError: If ``tolerance_s`` is negative.
    """
    raise NotImplementedError


def precision_recall_f1(
    n_matched: int, n_predicted: int, n_true: int
) -> tuple[float, float, float]:
    """Compute precision, recall and F1 on window onsets.

    Args:
        n_matched: Predicted onsets matched to a true onset.
        n_predicted: Total predicted onsets.
        n_true: Total true onsets.

    Returns:
        Tuple ``(precision, recall, f1)``, each dimensionless in [0, 1]. Any quantity with
        a zero denominator is returned as 0.0.

    Raises:
        ValueError: If any count is negative, or ``n_matched`` exceeds ``n_predicted`` or
            ``n_true``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def base_rate(mask: BoolArray) -> float:
    """Compute the fraction of time the deck is genuinely quiescent.

    Reported next to every F1. Without it, an F1 of 0.9 is uninterpretable: it may
    represent real skill, or it may represent a sea state in which the deck is quiet 85
    percent of the time and a constant "yes" scores nearly as well.

    Args:
        mask: True-trajectory quiescence mask, shape ``(n_samples,)``.

    Returns:
        Fraction of samples inside a sustained quiescent window, dimensionless in [0, 1].
    """
    raise NotImplementedError
