"""Driving the quiescent-window detector over a test partition.

:mod:`dmf.eval.quiescence` defines *what* a quiescent window is. This module defines
*where the detector stands in time*, which §6.2 of the plan does not fix and which changes
every number. The geometry is pre-registered as ``docs/protocol.md`` P6-D2 and implemented
here literally:

1. **Truth once per realization**, from the full true trajectory read back out of the
   corpus -- ``roll``, ``pitch``, ``heave_rate``, the three channels P3-D4 made forecast
   targets precisely so this metric could supply its own decision variable.
2. **A running detector at the window stride.** At decision time ``t`` (the last observed
   sample of a window, ``t = start + L - 1``) the model emits a forecast over the absolute
   samples ``t+1 .. t+H``, i.e. ``t+0.1 .. t+15.0 s`` at the production geometry, and the
   thresholds are applied to that trajectory.
3. **A predicted onset is an absolute time.** Many decision times predict the same onset;
   the **earliest** is kept, because that earliest flag is what lead time measures.
4. **Matching on absolute time at 0.5 s**, one-to-one and greedy by proximity.
5. **False alarms per minute** is unmatched predicted onsets over the evaluated duration.
6. **The reporting unit is
   ``(model, regime, threshold_set, rule, ss, heading_deg, speed_kn)``**, with the sea-state
   roll-up emitted beside it and marked ``group_level = "sea_state"``. F1 is never pooled
   across sea states (P6-D7 item 3) and no longer pooled across a sea state's cells either:
   P6-D19 measured that scorability is a property of the **cell**, because forward speed
   shifts the encounter frequency (P1-D1) and moves the response out of the band this hull
   sits inside permissive limits in. At SS3 / 45 deg / permissive the base rate is 1.000 at
   0 kn and 6 kn with no scorable onset at all, and 0.800 at 12 kn with 53 of them.
7. **Not-scorable cells are marked, not scored.** Ten of the 48 ``permissive`` cells and
   three of the 48 ``strict`` cells hold no interior onset, every one of them at SS3
   (P6-D19, ``results/e04/quiescence_base_rate.csv``); such a row ships ``scorable = False``
   with ``precision``/``recall``/``f1`` and its interval as NaN, never as 0.0 (which reads
   as model failure) and never as 1.0 (which reads as success). ``base_rate`` is still
   filled in, because it is the column that says *why* the cell is not scorable.
8. **Every F1 carries a bootstrap interval over realizations** (``f1_ci_lo``/``f1_ci_hi``,
   :func:`bootstrap_f1_ci`). Aggregating over *training* seeds gives a deterministic
   detector ``n_seeds = 1`` and ``f1_std = NaN``, which is no uncertainty at all on the
   metric this phase turns on; and P4-D13 records that seed spread is the wrong denominator
   even where it exists. The sparse cells are where this matters: SS6 / ``strict`` holds
   about 2.6 scorable onsets per realization, and the interval there is wide and is printed
   rather than suppressed.
8. **In the ``imu`` arm both truth and forecast come from ``imu`` channels** (P1-D6,
   enforced upstream by :func:`dmf.data.dataset.resolve_columns`), so those F1s are not
   comparable to ``ideal`` ones. The ``observation_mode`` column says which.

**Two rules, both reported.** ``point`` thresholds the point forecast. ``interval`` asks
"will the deck *stay* inside limits with 95 % confidence", and per P6-D5 the test is
**two-sided**: the limits are symmetric (``|roll| <= 3.0 deg``), so the conservative
statistic is ``max(|q05|, |q95|) <= limit`` and not the plan's one-sided 0.05 quantile.
That statistic is thresholded by the same comparison the point rule uses, rather than
through a second code path that could drift from the first.

**Where the exclusion rule is applied, and where it deliberately is not.** P6-D2 excludes a
run that begins before the first decision time or extends past the record end, from both
sides. On the truth side both halves apply directly (:func:`quiescence.scorable_onsets`).
On the prediction side the first half applies **per trajectory** -- a run starting at
forecast index 0 means the deck was already inside limits at the decision boundary, so that
trajectory witnessed no onset -- and the second half applies only **at the record level, in
absolute time**, so that both sides are bounded by the same two absolute samples. The
right-censoring half is *not* applied per trajectory, and that is a decision rather than an
omission: dropping runs that reach the end of the 15 s forecast would discard exactly the
long predicted quiet windows this metric exists to find, and would bias every lead time
downward, because the earliest decision time to predict a long window is precisely the one
whose forecast horizon it overruns.

**What the per-trajectory rule costs, quantified rather than waved at.** It cannot express
"the deck becomes quiescent at ``t+0.1``": a forecast that is in-limit from index 0 is read
as "already quiescent", not as an onset. At ``stride = 5`` exactly one decision time in five
sees a given absolute onset at index 0, and the four earlier ones see it at index 6, 11, 16,
21 -- so the onset is still detected, by an *earlier* decision time, with a *longer* lead.
The only detections lost are those in which the single decision time that would have flagged
the onset is the one immediately before it, i.e. detections with essentially zero lead time,
which are operationally worthless by the definition this metric exists to enforce.

**A structural consequence worth stating before anyone reads a zero as a failure.** A
*constant* forecast can never predict an onset. ``persistence`` holds the last observed
sample across the whole horizon and ``window_mean`` holds the window mean, so each of their
trajectories is either in-limit from index 0 or out of limits throughout; neither ever
contains an interior run. Both therefore report ``n_pred_onsets = 0`` and F1 = 0 on this
metric, in every cell, by construction. That is a true statement about them as landing-window
detectors -- they cannot express a transition -- and it is not a statement about their RMSE.
Read the ``n_pred_onsets`` column beside the F1.

**The trivial detectors are in the table, not in a footnote.** ``persistence`` and
``window_mean`` are passed in by the caller like any other model. Two more are generated
here, at the onset level rather than as forecasts, and they bound the metric from opposite
sides:

- ``always_quiescent`` flags an onset at **every** decision time. Its recall is 1.0 and its
  precision is ``n_true / n_decisions`` -- both by construction, neither a measurement. It
  is the over-flagging bound and **it is not the chance level**, which is how it has been
  read.
- ``rate_matched`` emits exactly as many onsets as the truth holds, at uniformly spaced,
  RNG-free times (:func:`rate_matched_onsets`). It is handed the rate and nothing else, so
  it cannot be dismissed for flagging too often; what it does not know is *when*. Its
  expected precision is the chance of a uniform flag landing inside the 0.5 s tolerance of a
  true onset, and a detector that does not beat it has learned nothing about timing.

Neither is a forecaster -- an onset-level detector is a thing no
``forward(x) -> (B, H, C)`` can express -- so neither belongs in ``src/dmf/models/``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from dmf.config import ObservationMode
from dmf.data.dataset import DeckMotionDataset, make_dataloader, resolve_columns
from dmf.data.normalize import invert_norm
from dmf.data.splits import RealizationKey
from dmf.eval.quiescence import (
    PERMISSIVE,
    STRICT,
    QuiescenceThresholds,
    false_alarms_per_minute,
    lead_times,
    match_onsets,
    precision_recall_f1,
    scorable_onsets,
    sustain_samples,
)

# Private on purpose, and imported on purpose: reusing the one draw function is what makes
# every detector in a cell resampled by the identical weights, and reusing the one
# percentile helper means "central 95%" has a single definition in this package.
from dmf.eval.runner import _bootstrap_counts, _percentile_interval
from dmf.models.base import ForecastModel
from dmf.models.heads import HeadKind, PredictiveDistribution, n_output_params
from dmf.sim.generate import RealizationSpec, realization_path
from dmf.typedefs import BoolArray, FloatArray, IntArray

__all__ = [
    "ALWAYS_QUIESCENT",
    "CELL_LEVEL",
    "DECISION_CHANNELS",
    "GROUP_LEVELS",
    "INTERVAL_LEVELS",
    "LEAD_TIME_COLUMNS",
    "QUIESCENCE_BOOTSTRAP_SEED",
    "QUIESCENCE_CI_LEVEL",
    "QUIESCENCE_COLUMNS",
    "QUIESCENCE_N_BOOT",
    "RATE_MATCHED",
    "RULES",
    "SEA_STATE_LEVEL",
    "SYNTHETIC_DETECTORS",
    "bootstrap_f1_ci",
    "decision_channel_index",
    "evaluate_quiescence",
    "rate_matched_onsets",
    "sustained_runs",
]

#: Label of the degenerate detector that flags a quiescent onset at every decision time.
ALWAYS_QUIESCENT: str = "always_quiescent"

#: Label of the chance-level detector that emits the **true number** of onsets at uniformly
#: spaced times.
#:
#: :data:`ALWAYS_QUIESCENT` is not the chance level and was being read as one. It flags at
#: every decision time, so its precision is ``n_true / n_decisions`` **by construction** and
#: its recall is 1.0 by construction; its low F1 is arithmetic about the decision grid, not
#: evidence that the metric resists gaming. The reference a reviewer actually wants is a
#: detector matched on *rate*: it emits exactly as many onsets as the truth holds, so it is
#: not punished for over-flagging, and it places them without looking at the deck. Its
#: expected precision is the chance of a uniformly placed flag landing inside the matching
#: tolerance of a true onset, which is what "F1 = 0.06 means nothing here" is measured
#: against. Both detectors ship, because they bound the metric from two different sides.
RATE_MATCHED: str = "rate_matched"

#: The detectors this module generates rather than runs a forward pass for. Neither is a
#: forecaster and neither belongs in ``src/dmf/models/``: they are defined at the onset
#: level, which is a thing no ``forward(x) -> (B, H, C)`` can express.
SYNTHETIC_DETECTORS: tuple[str, str] = (ALWAYS_QUIESCENT, RATE_MATCHED)

#: Value of ``group_level`` on a row whose unit is one corpus grid cell.
CELL_LEVEL: str = "cell"

#: Value of ``group_level`` on a row that rolls several cells of one sea state together.
#:
#: **Marked by a column, never by convention.** P6-D19 measured that scorability is a
#: property of ``(sea state, heading, speed)`` and not of the sea state: at SS3 / 45 deg /
#: permissive the deck never leaves limits at 0 kn or 6 kn and does at 12 kn, because
#: forward speed shifts the encounter frequency (P1-D1). A sea-state row therefore averages
#: scorable and unscorable cells together, which is the same pooling P6-D7 item 3 forbids
#: one level up. It is still reported -- it is the level the plan's §6.2 is written at --
#: but a reader must be able to tell the two apart from the row itself.
SEA_STATE_LEVEL: str = "sea_state"

#: The two reporting levels, coarsest last.
GROUP_LEVELS: tuple[str, str] = (CELL_LEVEL, SEA_STATE_LEVEL)

#: Bootstrap resamples behind every ``f1_ci_lo``/``f1_ci_hi``. The value
#: :func:`dmf.eval.runner.bootstrap_skill_ci` defaults to, restated so the two cannot drift.
QUIESCENCE_N_BOOT: int = 1000

#: Central confidence level of every quiescence interval.
QUIESCENCE_CI_LEVEL: float = 0.95

#: Seed of the resampling generator, so the interval is a function of the scored counts
#: alone.
QUIESCENCE_BOOTSTRAP_SEED: int = 0

#: The three decision channels, as **logical** names. Resolved through the dataset's
#: observation mode, so the ``imu`` arm thresholds ``roll_imu``/``pitch_imu``/
#: ``heave_rate_imu`` and never mixes an ``imu`` forecast with an ``ideal`` truth (P1-D6).
DECISION_CHANNELS: tuple[str, str, str] = ("roll", "pitch", "heave_rate")

#: The two-sided interval levels of the ``interval`` rule (P6-D5). Both are read off the
#: project's nine-level fan exactly; neither is interpolated.
INTERVAL_LEVELS: tuple[float, float] = (0.05, 0.95)

#: The two decision rules, both reported for any model that can carry both.
RULES: tuple[str, ...] = ("point", "interval")

#: Columns of ``results/e04/quiescence.csv``, in table order. Every F1 is adjacent to its
#: base rate, its onset counts and its bootstrap interval, which together are what make it
#: interpretable at all.
QUIESCENCE_COLUMNS: tuple[str, ...] = (
    "model",
    "regime",
    "observation_mode",
    "threshold_set",
    "rule",
    "group_level",
    "ss",
    # NaN on a sea-state roll-up row, which is why `group_level` is the column that decides
    # how a row is read and these two are only the cell's coordinates.
    "heading_deg",
    "speed_kn",
    "scorable",
    "base_rate",
    "n_true_onsets",
    "n_pred_onsets",
    "n_matched",
    "precision",
    "recall",
    "f1",
    # Over **realizations**, not over training seeds: a deterministic detector has one seed
    # and would otherwise ship `n_seeds = 1, f1_std = NaN`, i.e. no uncertainty at all on
    # the metric the Phase 6 carry-forward calls the only one left that discriminates.
    "f1_ci_lo",
    "f1_ci_hi",
    "n_boot",
    "ci_level",
    "bootstrap_seed",
    "false_alarms_per_min",
    "lead_p10",
    "lead_p50",
    "lead_p90",
    "n_realizations",
    "duration_s",
    "n_excluded_true",
    "n_excluded_pred",
)

#: Columns of ``results/e04/quiescence_lead_times.csv`` -- the raw per-match lead times, so
#: the distribution is available and not only its three quantiles. Emitted once, from the
#: **cell** pass: a row is one matched onset of one realization, and the sea-state roll-up
#: re-groups exactly those matches rather than producing new ones.
LEAD_TIME_COLUMNS: tuple[str, ...] = (
    "model",
    "regime",
    "threshold_set",
    "rule",
    "ss",
    "heading_deg",
    "speed_kn",
    "vessel",
    "seed",
    "true_onset_sample",
    "flag_sample",
    "lead_s",
)


def _observation_mode(columns: Sequence[str]) -> ObservationMode:
    """Infer the observation mode a set of resolved target columns came from.

    Args:
        columns: Corpus column names after observation-mode resolution.

    Returns:
        ``"imu"`` if any column carries the ``_imu`` suffix, else ``"ideal"``. The two are
        never mixed within one dataset (P2-D8), so one column decides.
    """
    return "imu" if any(c.endswith("_imu") for c in columns) else "ideal"


def _head_of(model: ForecastModel) -> tuple[HeadKind, tuple[float, ...], int]:
    """Read a model's head metadata without narrowing the accepted type.

    The parameter type is :class:`dmf.models.base.ForecastModel`, a Protocol that promises
    only ``forward(x) -> (B, H, C_out)``. Every model in this project is also a
    :class:`dmf.models.base.BaseForecaster` and carries ``head_kind``, but requiring that
    would stop this runner from scoring a plain-Protocol detector, which is exactly the
    flexibility the signature exists for. A model that declares no head is therefore taken
    at its Protocol word: a point model, scored on the ``point`` rule only.

    Args:
        model: The model.

    Returns:
        Tuple ``(head_kind, quantile_levels, n_output_params)``.
    """
    head: HeadKind = getattr(model, "head_kind", "point")
    levels: tuple[float, ...] = tuple(getattr(model, "quantile_levels", ()))
    width = int(getattr(model, "n_output_params", n_output_params(head, levels)))
    return head, levels, width


def decision_channel_index(target_columns: Sequence[str]) -> tuple[int, int, int]:
    """Locate roll, pitch and heave rate among a dataset's target channels.

    The detector needs all three, and it needs them as *forecast* channels rather than as
    inputs, since the prediction side thresholds the model's own output. An arm whose
    ``target_dofs`` omits ``heave_rate`` -- ``configs/data/attitude_only.yaml`` is exactly
    that arm -- cannot be scored on this metric, and is refused here rather than scored on
    two limits out of three.

    Args:
        target_columns: The dataset's target channel names, after observation-mode
            resolution, e.g. ``("roll", "pitch", "heave", ...)`` or their ``*_imu`` twins.

    Returns:
        Tuple of column indices for roll, pitch and heave rate, in that order.

    Raises:
        ValueError: If any of the three is absent, naming the missing ones.
    """
    columns = list(target_columns)
    wanted = resolve_columns(DECISION_CHANNELS, _observation_mode(columns))
    missing = [name for name in wanted if name not in columns]
    if missing:
        raise ValueError(
            f"the quiescence detector thresholds |roll|, |pitch| and |heave_rate|, but "
            f"{missing} are not forecast targets of this dataset (targets are {columns}). "
            f"An arm that does not forecast all three cannot supply its own decision "
            f"variable and is not scorable on this metric."
        )
    first, second, third = (columns.index(name) for name in wanted)
    return first, second, third


def sustained_runs(flags: BoolArray, min_run: int) -> tuple[IntArray, IntArray, IntArray]:
    """Find every run of at least ``min_run`` consecutive True samples, row-wise.

    The batched twin of the run-length filter inside
    :func:`dmf.eval.quiescence.detect_quiescent_mask`. It exists because the detector runs
    at every window start of every test realization -- hundreds of thousands of forecast
    trajectories per model per threshold set -- and a Python loop over the reference
    implementation is the whole cost of this metric. ``tests/test_quiescence_runner.py``
    asserts row-by-row agreement with the reference on random masks, so the fast path is
    checked against the definition rather than trusted.

    Args:
        flags: Instantaneous in-limit mask, shape ``(n_rows, n_steps)``.
        min_run: Minimum run length, samples, at least 1.

    Returns:
        Tuple ``(row, start, stop)`` of int64 arrays over the surviving runs, where
        ``flags[row[i], start[i]:stop[i]]`` is all True and ``stop[i] - start[i] >=
        min_run``. Ordered by row, then by start.

    Raises:
        ValueError: If ``flags`` is not two-dimensional or ``min_run`` is not positive.
    """
    if flags.ndim != 2:
        raise ValueError(f"flags must have shape (n_rows, n_steps), got {flags.shape}")
    if min_run < 1:
        raise ValueError(f"min_run must be positive, got {min_run}")
    n_rows, n_steps = flags.shape
    padded = np.zeros((n_rows, n_steps + 2), dtype=np.int8)
    padded[:, 1:-1] = np.asarray(flags, dtype=bool)
    edges = np.diff(padded.astype(np.int16), axis=1)
    start_row, start_col = np.nonzero(edges == 1)
    _, stop_col = np.nonzero(edges == -1)
    # np.nonzero scans row-major, so the i-th start and the i-th stop bound the same run.
    keep = (stop_col - start_col) >= min_run
    return (
        start_row[keep].astype(np.int64),
        start_col[keep].astype(np.int64),
        stop_col[keep].astype(np.int64),
    )


@dataclass
class _DetectorState:
    """Accumulated predicted onsets for one (model, rule, threshold set).

    Attributes:
        earliest_flag: Per realization, a mapping from an absolute predicted onset sample to
            the **earliest** decision-time sample that predicted it.
        n_excluded: Per realization, predicted onsets dropped by the record-level bound.
    """

    earliest_flag: list[dict[int, int]]
    n_excluded: list[int]


@dataclass(frozen=True)
class _TruthState:
    """The truth side of one threshold set, per realization.

    Attributes:
        onsets: Scorable true onsets, absolute samples.
        n_excluded: Runs dropped by the exclusion rule.
        n_quiescent: Samples inside a sustained run within the evaluated span.
        n_evaluated: Samples in the evaluated span.
    """

    onsets: list[IntArray]
    n_excluded: list[int]
    n_quiescent: list[int]
    n_evaluated: list[int]


@dataclass(frozen=True)
class _Geometry:
    """Absolute-time bounds shared by the truth and prediction sides.

    Attributes:
        first_decision: Absolute sample of the first decision time, ``starts[0] + L - 1``.
        last_covered: One past the last absolute sample any forecast reaches.
        horizon: Forecast length, samples.
        decision_times: Absolute decision time of each window start within a realization,
            shape ``(windows_per_realization,)``.
    """

    first_decision: int
    last_covered: int
    horizon: int
    decision_times: IntArray = field(repr=False)


def _geometry(dataset: DeckMotionDataset) -> _Geometry:
    """Derive the shared absolute-time bounds from the dataset's window geometry.

    Args:
        dataset: The partition being scored.

    Returns:
        The bounds.

    Raises:
        RuntimeError: If the window starts are not strictly ascending, which is what makes
            "keep the earliest decision time" a single ``setdefault``.
    """
    spec = dataset.window_spec
    starts = np.asarray(
        [dataset.describe_window(i)[1] for i in range(dataset.windows_per_realization)],
        dtype=np.int64,
    )
    if starts.size > 1 and not np.all(np.diff(starts) > 0):
        raise RuntimeError(
            "window starts are not strictly ascending; 'keep the earliest decision time "
            "that predicts an onset' relies on visiting decision times in order"
        )
    return _Geometry(
        first_decision=int(starts[0] + spec.lookback - 1),
        last_covered=int(starts[-1] + spec.total_length),
        horizon=spec.max_horizon,
        decision_times=starts + spec.lookback - 1,
    )


def _true_trajectories(dataset: DeckMotionDataset) -> FloatArray:
    """Read the three decision channels of every realization from the corpus.

    Read back from Parquet rather than reassembled from windows: the truth mask is a
    property of the realization, not of the windowing, and P6-D2 computes it once per
    realization over the whole record. Reading independently also means an error in the
    window index cannot silently move the truth side along with it.

    Args:
        dataset: The partition.

    Returns:
        Array of shape ``(n_realizations, n_samples, 3)``, corpus units, columns ordered
        roll, pitch, heave rate, in the dataset's realization order.
    """
    columns = list(resolve_columns(DECISION_CHANNELS, _observation_mode(dataset.target_columns)))
    blocks: list[FloatArray] = []
    for key in dataset.realization_keys:
        ss, heading, speed, vessel, seed = key
        path: Path = dataset.corpus_root / realization_path(
            RealizationSpec(
                seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel
            )
        )
        frame = pd.read_parquet(path, columns=columns)
        blocks.append(frame[columns].to_numpy(dtype=np.float64))
    return np.stack(blocks, axis=0)


def _truth_state(
    trajectories: FloatArray,
    thresholds: QuiescenceThresholds,
    fs_hz: float,
    geometry: _Geometry,
) -> _TruthState:
    """Build the truth side once per realization from the full true trajectory.

    Args:
        trajectories: True decision channels, shape ``(n_realizations, n_samples, 3)``.
        thresholds: The limit set.
        fs_hz: Sampling rate, hertz.
        geometry: Shared absolute-time bounds.

    Returns:
        The truth side.
    """
    lower, upper = geometry.first_decision, geometry.last_covered
    minimum = sustain_samples(thresholds.sustain_s, fs_hz)
    limits = np.asarray(
        [thresholds.roll_deg, thresholds.pitch_deg, thresholds.heave_rate_mps],
        dtype=np.float64,
    )
    onsets: list[IntArray] = []
    excluded: list[int] = []
    quiescent: list[int] = []
    evaluated: list[int] = []
    for series in trajectories:
        with np.errstate(invalid="ignore"):
            instantaneous = np.all(np.abs(series[:upper, :]) <= limits[None, :], axis=1)
        mask = np.zeros(instantaneous.shape, dtype=np.bool_)
        rows, starts, stops = sustained_runs(instantaneous[None, :], minimum)
        del rows
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            mask[start:stop] = True
        found, dropped = scorable_onsets(mask, first_decision_idx=lower)
        onsets.append(found)
        excluded.append(dropped)
        # The base rate describes the span the F1 is about, not the whole record.
        quiescent.append(int(np.count_nonzero(mask[lower + 1 :])))
        evaluated.append(int(mask.size - lower - 1))
    return _TruthState(onsets, excluded, quiescent, evaluated)


def _record_onsets(
    state: _DetectorState,
    flags: BoolArray,
    decision_times: IntArray,
    realization_index: IntArray,
    min_run: int,
    geometry: _Geometry,
) -> None:
    """Fold one batch of thresholded forecast trajectories into the predicted onsets.

    See the module docstring for why the right-censoring half of the exclusion rule is
    applied at the record level and not per trajectory.

    Args:
        state: The accumulator to update, in place.
        flags: Instantaneous in-limit mask per trajectory, shape ``(B, H)``.
        decision_times: Absolute decision time of each row, shape ``(B,)``.
        realization_index: Realization index of each row, shape ``(B,)``.
        min_run: Sustain requirement, samples.
        geometry: Shared absolute-time bounds.
    """
    rows, starts, _ = sustained_runs(flags, min_run)
    # A run starting at forecast index 0 means the deck was already inside limits at the
    # decision boundary: this trajectory witnessed no onset.
    interior = starts > 0
    rows, starts = rows[interior], starts[interior]
    if rows.size == 0:
        return
    flagged_at = decision_times[rows]
    absolute = flagged_at + 1 + starts
    realizations = realization_index[rows]
    in_span = (absolute > geometry.first_decision) & (absolute < geometry.last_covered)
    for realization in np.unique(realizations[~in_span]).tolist():
        state.n_excluded[int(realization)] += int(
            np.count_nonzero(realizations[~in_span] == realization)
        )
    for realization, onset, flag in zip(
        realizations[in_span].tolist(),
        absolute[in_span].tolist(),
        flagged_at[in_span].tolist(),
        strict=True,
    ):
        # Windows are visited in ascending start order, so the first decision time to
        # predict an onset is the earliest one; setdefault is what "keep the earliest" is.
        state.earliest_flag[realization].setdefault(onset, flag)


def _fold(
    state: _DetectorState,
    values: FloatArray,
    channels: tuple[int, int, int],
    thresholds: QuiescenceThresholds,
    decision_times: IntArray,
    realization_index: IntArray,
    fs_hz: float,
    geometry: _Geometry,
) -> None:
    """Threshold one batch of forecast trajectories and record their onsets.

    Args:
        state: The accumulator for this (model, rule, threshold set).
        values: The decision statistic per trajectory, shape ``(B, H, C_out)``, corpus
            units. For the point rule this is the forecast; for the interval rule it is the
            two-sided conservative magnitude ``max(|q05|, |q95|)`` (P6-D5). Both are
            compared as ``|value| <= limit``, which is the same test for a non-negative
            statistic -- one thresholding path, two rules.
        channels: Column indices of roll, pitch and heave rate.
        thresholds: The limit set.
        decision_times: Absolute decision time per row.
        realization_index: Realization index per row.
        fs_hz: Sampling rate, hertz.
        geometry: Shared absolute-time bounds.
    """
    roll_c, pitch_c, rate_c = channels
    flags = (
        (np.abs(values[:, :, roll_c]) <= thresholds.roll_deg)
        & (np.abs(values[:, :, pitch_c]) <= thresholds.pitch_deg)
        & (np.abs(values[:, :, rate_c]) <= thresholds.heave_rate_mps)
    )
    _record_onsets(
        state,
        flags,
        decision_times,
        realization_index,
        sustain_samples(thresholds.sustain_s, fs_hz),
        geometry,
    )


def _fold_always_quiescent(state: _DetectorState, geometry: _Geometry, n_keys: int) -> None:
    """Record the onsets of the degenerate detector that says "yes" at every decision time.

    Defined at the onset level rather than as a forecast trajectory, because the two are
    different detectors and the exploitative one is the interesting reference. A model whose
    *trajectory* is quiescent from index 0 predicts no onset at all -- correctly, since the
    onset lies in its past -- and would score F1 = 0, which says nothing about base-rate
    exploitation.

    **What this detector's F1 is and is not.** Its recall is 1.0 by construction and its
    precision is ``n_true / n_decisions`` by construction, so its F1 is a statement about the
    decision grid and the cell's onset count, not about the metric's resistance to gaming.
    P6-D2 quotes recall 1.0, precision 0.00917 and F1 0.0182 for it; those three numbers come
    from a synthetic square-wave fixture in ``tests/test_quiescence.py`` (11 scorable onsets
    against 1200 decision times), not from any corpus cell, and the corpus figure differs by
    cell because ``n_true`` does. :data:`RATE_MATCHED` is the reference that is not fixed by
    construction in this way.

    Args:
        state: The accumulator to fill.
        geometry: Shared absolute-time bounds.
        n_keys: Realization count.
    """
    onsets = geometry.decision_times + 1
    in_span = (onsets > geometry.first_decision) & (onsets < geometry.last_covered)
    kept = onsets[in_span]
    flags = geometry.decision_times[in_span]
    dropped = int(np.count_nonzero(~in_span))
    for realization in range(n_keys):
        state.n_excluded[realization] += dropped
        mapping = state.earliest_flag[realization]
        for onset, flag in zip(kept.tolist(), flags.tolist(), strict=True):
            mapping.setdefault(onset, flag)


def bootstrap_f1_ci(
    n_matched: IntArray,
    n_predicted: IntArray,
    n_true: IntArray,
    *,
    n_boot: int = QUIESCENCE_N_BOOT,
    ci_level: float = QUIESCENCE_CI_LEVEL,
    seed: int = QUIESCENCE_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap an F1 confidence interval by resampling whole **realizations**.

    The realization is the unit that was independently simulated, and it is the only unit
    here that could be resampled honestly: onsets within one realization are not independent
    draws (a rough record is rough throughout), and training seeds are not the sampling
    axis at all -- P4-D13, and the reason ``f1_std`` over seeds is NaN for every
    deterministic detector in this table.

    Implemented with multinomial counts rather than index draws, on the identity

    ``F1 = 2 * sum(matched) / (sum(predicted) + sum(true))``

    which is exactly :func:`dmf.eval.quiescence.precision_recall_f1`'s ``2PR/(P+R)``
    wherever that has a non-zero denominator, and 0.0 where it does not. So a resample is a
    count-weighted ratio and 1000 of them are one small matrix product. The counts come from
    :func:`dmf.eval.runner._bootstrap_counts`, which is a pure function of
    ``(n_units, n_boot, seed)`` -- so every detector scored over the same cell is resampled
    by the identical draw, and two rows of one cell are comparable to each other as well as
    each to zero.

    Args:
        n_matched: Matched onsets per realization.
        n_predicted: Predicted onsets per realization.
        n_true: Scorable true onsets per realization.
        n_boot: Bootstrap resamples.
        ci_level: Central confidence level.
        seed: Seed of the resampling generator.

    Returns:
        Tuple ``(lo, hi)``. Both NaN when the cell holds no realization or no true onset at
        all: an unscorable cell has no F1, so it has no interval either, and 0.0 would read
        as a measured zero.

    Raises:
        ValueError: If the three arrays disagree in length, if ``n_boot`` is not positive,
            or if ``ci_level`` is outside ``(0, 1)``.
    """
    if not (n_matched.shape == n_predicted.shape == n_true.shape):
        raise ValueError(
            f"the per-realization counts must have equal length, got {n_matched.shape}, "
            f"{n_predicted.shape} and {n_true.shape}"
        )
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    n_units = int(n_matched.size)
    if n_units == 0 or int(n_true.sum()) == 0:
        return float("nan"), float("nan")
    counts = _bootstrap_counts(n_units, n_boot, seed)
    numerator = 2.0 * (counts @ n_matched.astype(np.float64))
    denominator = (counts @ n_predicted.astype(np.float64)) + (counts @ n_true.astype(np.float64))
    draws = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    lo, hi = _percentile_interval(draws.reshape(n_boot, 1), ci_level=ci_level, shape=(1, 1))
    return float(lo.reshape(-1)[0]), float(hi.reshape(-1)[0])


@dataclass(frozen=True)
class _GroupCounts:
    """The per-realization counts one reported row is formed from.

    Attributes:
        n_true: Scorable true onsets per realization.
        n_pred: Predicted onsets per realization.
        n_matched: Matched onsets per realization.
        n_excluded_true: Truth-side runs dropped by the exclusion rule, per realization.
        n_excluded_pred: Prediction-side onsets dropped likewise.
        quiescent: Samples inside a sustained true run, per realization.
        evaluated: Samples in the evaluated span, per realization.
        leads: Lead times of every match in the group, seconds.
    """

    n_true: IntArray
    n_pred: IntArray
    n_matched: IntArray
    n_excluded_true: IntArray
    n_excluded_pred: IntArray
    quiescent: IntArray
    evaluated: IntArray
    leads: FloatArray


def _row(
    *,
    model: str,
    regime: str,
    observation_mode: str,
    thresholds: QuiescenceThresholds,
    rule: str,
    group_level: str,
    sea_state: str,
    heading_deg: float,
    speed_kn: float,
    counts: _GroupCounts,
    fs_hz: float,
    n_boot: int,
    ci_level: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Form one reported row from a group's per-realization counts.

    Args:
        model: Model label.
        regime: Evaluation regime.
        observation_mode: ``"ideal"`` or ``"imu"``.
        thresholds: The limit set.
        rule: ``"point"`` or ``"interval"``.
        group_level: :data:`CELL_LEVEL` or :data:`SEA_STATE_LEVEL`.
        sea_state: Sea state label.
        heading_deg: Encounter angle, degrees, or NaN on a roll-up.
        speed_kn: Forward speed, knots, or NaN on a roll-up.
        counts: The group's per-realization counts.
        fs_hz: Sampling rate, hertz.
        n_boot: Bootstrap resamples.
        ci_level: Central confidence level.
        bootstrap_seed: Seed of the resampling generator.

    Returns:
        One record on :data:`QUIESCENCE_COLUMNS`.
    """
    n_true = int(counts.n_true.sum())
    n_pred = int(counts.n_pred.sum())
    n_matched = int(counts.n_matched.sum())
    evaluated = int(counts.evaluated.sum())
    duration_s = evaluated / fs_hz
    scorable = n_true > 0
    precision, recall, f1 = precision_recall_f1(n_matched, n_pred, n_true)
    lo, hi = bootstrap_f1_ci(
        counts.n_matched,
        counts.n_pred,
        counts.n_true,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return {
        "model": model,
        "regime": regime,
        "observation_mode": observation_mode,
        "threshold_set": thresholds.name,
        "rule": rule,
        "group_level": group_level,
        "ss": sea_state,
        "heading_deg": heading_deg,
        "speed_kn": speed_kn,
        "scorable": scorable,
        # By definition base_rate(mask) over the pooled evaluated span; formed from the
        # counts rather than by materialising a multi-million-sample mask.
        "base_rate": (int(counts.quiescent.sum()) / evaluated) if evaluated else 0.0,
        "n_true_onsets": n_true,
        "n_pred_onsets": n_pred,
        "n_matched": n_matched,
        "precision": precision if scorable else float("nan"),
        "recall": recall if scorable else float("nan"),
        "f1": f1 if scorable else float("nan"),
        "f1_ci_lo": lo if scorable else float("nan"),
        "f1_ci_hi": hi if scorable else float("nan"),
        "n_boot": n_boot,
        "ci_level": ci_level,
        "bootstrap_seed": bootstrap_seed,
        "false_alarms_per_min": false_alarms_per_minute(n_pred - n_matched, duration_s),
        "lead_p10": _quantile(counts.leads, 0.10),
        "lead_p50": _quantile(counts.leads, 0.50),
        "lead_p90": _quantile(counts.leads, 0.90),
        "n_realizations": int(counts.n_true.size),
        "duration_s": duration_s,
        "n_excluded_true": int(counts.n_excluded_true.sum()),
        "n_excluded_pred": int(counts.n_excluded_pred.sum()),
    }


def rate_matched_onsets(n_onsets: int, first: int, last: int) -> tuple[IntArray, IntArray]:
    """Place ``n_onsets`` onsets uniformly across a span, with no RNG at all.

    The chance-level detector's geometry. It is handed the true onset **count** and nothing
    else -- not the times, not the deck -- so it is rate-matched and blind. That the count is
    an oracle quantity is the point: it removes the "it flags too often" explanation that
    makes :data:`ALWAYS_QUIESCENT`'s low F1 uninformative, and leaves only "it flags in the
    wrong places". A detector that cannot beat this one has learned nothing about *when*.

    Deterministic rather than random, for the reason P6-D3 gives for its own grid choice: a
    reported reference must not move between runs. The onsets sit at the midpoints of
    ``n_onsets`` equal sub-intervals of the open span, which is the placement whose expected
    match rate is the uniform one and which cannot cluster.

    Each onset's flag is the **latest** decision time strictly before it, i.e. the shortest
    lead consistent with predicting it. Not the earliest: choosing the earliest covering
    decision time would hand the chance detector the longest lead time in the table, and the
    lead-time distribution is a column this reference is supposed to make readable rather
    than win.

    Args:
        n_onsets: How many onsets to emit, non-negative.
        first: Absolute sample of the first decision time. Onsets are placed strictly after
            it, matching the exclusion rule the truth side applies.
        last: One past the last absolute sample any forecast reaches. Onsets are placed
            strictly before it.

    Returns:
        Tuple ``(onsets, flags)``, absolute samples, ascending and strictly inside
        ``(first, last)``. Both empty when ``n_onsets`` is zero or the span holds fewer
        distinct samples than requested -- an unscorable cell gets no chance detector, which
        is the same answer the truth side gives it.

    Raises:
        ValueError: If ``n_onsets`` is negative.
    """
    if n_onsets < 0:
        raise ValueError(f"n_onsets must be non-negative, got {n_onsets}")
    empty = np.zeros(0, dtype=np.int64)
    span = last - first - 1
    if n_onsets == 0 or span < n_onsets:
        return empty, empty
    offsets = (np.arange(n_onsets, dtype=np.float64) + 0.5) * (span / n_onsets)
    onsets = np.unique((first + 1 + np.floor(offsets)).astype(np.int64))
    onsets = onsets[(onsets > first) & (onsets < last)]
    return onsets, (onsets - 1).astype(np.int64)


def _fold_rate_matched(
    state: _DetectorState, geometry: _Geometry, truth: _TruthState, n_keys: int
) -> None:
    """Record the chance-level detector's onsets, one realization at a time.

    Per realization and not once for the partition, because the rate it is matched to is a
    per-realization quantity: SS6 holds ~2.6 scorable onsets per record at ``strict`` and
    SS5 holds ~9 (P6-D19), and a single corpus-wide rate would flatter the sparse cells and
    punish the dense ones.

    Args:
        state: The accumulator to fill.
        geometry: Shared absolute-time bounds.
        truth: The truth side of this threshold set, for the onset counts.
        n_keys: Realization count.
    """
    for realization in range(n_keys):
        onsets, flags = rate_matched_onsets(
            int(truth.onsets[realization].size), geometry.first_decision, geometry.last_covered
        )
        mapping = state.earliest_flag[realization]
        for onset, flag in zip(onsets.tolist(), flags.tolist(), strict=True):
            mapping.setdefault(onset, flag)


def _summarise(
    *,
    model: str,
    regime: str,
    observation_mode: str,
    thresholds: QuiescenceThresholds,
    rule: str,
    keys: Sequence[RealizationKey],
    truth: _TruthState,
    state: _DetectorState,
    fs_hz: float,
    tolerance_s: float,
    n_boot: int = QUIESCENCE_N_BOOT,
    ci_level: float = QUIESCENCE_CI_LEVEL,
    bootstrap_seed: int = QUIESCENCE_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Match onsets per realization and group the result by grid cell, then by sea state.

    Grouping, not merely a column: P6-D7 item 3 for the sea state, P6-D19 for the cell. A
    corpus-pooled F1 is dominated by the cells where the deck is quiet essentially all the
    time, and a *sea-state*-pooled F1 pools scorable cells with unscorable ones -- at SS3 /
    permissive, 10 of 12 cells hold no scorable onset at all and the other two hold 11.12
    and 0.25 per realization, because forward speed moves the response out of the band this
    hull sits inside permissive limits in (P1-D1). Both levels are emitted, and
    ``group_level`` says which a row is.

    Args:
        model: Model label.
        regime: Evaluation regime.
        observation_mode: ``"ideal"`` or ``"imu"``; carried because ``imu`` F1s are not
            comparable to ``ideal`` ones (P1-D6).
        thresholds: The limit set.
        rule: ``"point"`` or ``"interval"``.
        keys: Realization keys, in the dataset's index order.
        truth: The truth side.
        state: The accumulated predictions.
        fs_hz: Sampling rate, hertz.
        tolerance_s: Onset matching tolerance, seconds.
        n_boot: Bootstrap resamples behind the F1 interval.
        ci_level: Central confidence level.
        bootstrap_seed: Seed of the resampling generator.

    Returns:
        Tuple ``(summary_rows, lead_rows)``. The lead rows come from the cell pass only; the
        sea-state rows re-group the same matches rather than producing new ones.
    """
    leads: list[dict[str, object]] = []
    per_realization: dict[tuple[str, float, float], list[int]] = {}
    matched_counts = np.zeros(len(keys), dtype=np.int64)
    predicted_counts = np.zeros(len(keys), dtype=np.int64)
    true_counts = np.zeros(len(keys), dtype=np.int64)
    lead_by_index: list[FloatArray] = []
    for index, key in enumerate(keys):
        per_realization.setdefault((str(key[0]), float(key[1]), float(key[2])), []).append(index)
        true_onsets = truth.onsets[index]
        predicted_map = state.earliest_flag[index]
        predicted = np.asarray(sorted(predicted_map), dtype=np.int64)
        matched_pred, matched_true, _ = match_onsets(predicted, true_onsets, fs_hz, tolerance_s)
        true_counts[index] = int(true_onsets.size)
        predicted_counts[index] = int(predicted.size)
        matched_counts[index] = int(matched_pred.size)
        if matched_pred.size == 0:
            lead_by_index.append(np.zeros(0, dtype=np.float64))
            continue
        flag_samples = np.asarray(
            [predicted_map[int(predicted[i])] for i in matched_pred.tolist()], dtype=np.int64
        )
        matched_onsets = true_onsets[matched_true]
        values = lead_times(flag_samples, matched_onsets, fs_hz)
        lead_by_index.append(values)
        for onset, flag, lead in zip(
            matched_onsets.tolist(), flag_samples.tolist(), values.tolist(), strict=True
        ):
            leads.append(
                {
                    "model": model,
                    "regime": regime,
                    "threshold_set": thresholds.name,
                    "rule": rule,
                    "ss": str(key[0]),
                    "heading_deg": float(key[1]),
                    "speed_kn": float(key[2]),
                    "vessel": str(key[3]),
                    "seed": int(key[4]),
                    "true_onset_sample": int(onset),
                    "flag_sample": int(flag),
                    "lead_s": float(lead),
                }
            )

    def _counts(members: Sequence[int]) -> _GroupCounts:
        picked = np.asarray(members, dtype=np.int64)
        gathered = [lead_by_index[index] for index in members]
        return _GroupCounts(
            n_true=true_counts[picked],
            n_pred=predicted_counts[picked],
            n_matched=matched_counts[picked],
            n_excluded_true=np.asarray(truth.n_excluded, dtype=np.int64)[picked],
            n_excluded_pred=np.asarray(state.n_excluded, dtype=np.int64)[picked],
            quiescent=np.asarray(truth.n_quiescent, dtype=np.int64)[picked],
            evaluated=np.asarray(truth.n_evaluated, dtype=np.int64)[picked],
            leads=(np.concatenate(gathered) if gathered else np.zeros(0, dtype=np.float64)),
        )

    common = {
        "model": model,
        "regime": regime,
        "observation_mode": observation_mode,
        "thresholds": thresholds,
        "rule": rule,
        "fs_hz": fs_hz,
        "n_boot": n_boot,
        "ci_level": ci_level,
        "bootstrap_seed": bootstrap_seed,
    }
    summary: list[dict[str, object]] = [
        _row(
            group_level=CELL_LEVEL,
            sea_state=cell[0],
            heading_deg=cell[1],
            speed_kn=cell[2],
            counts=_counts(per_realization[cell]),
            **common,  # type: ignore[arg-type]
        )
        for cell in sorted(per_realization)
    ]
    by_sea_state: dict[str, list[int]] = {}
    for cell, members in per_realization.items():
        by_sea_state.setdefault(cell[0], []).extend(members)
    summary.extend(
        _row(
            group_level=SEA_STATE_LEVEL,
            sea_state=sea_state,
            heading_deg=float("nan"),
            speed_kn=float("nan"),
            counts=_counts(sorted(by_sea_state[sea_state])),
            **common,  # type: ignore[arg-type]
        )
        for sea_state in sorted(by_sea_state)
    )
    return summary, leads


def _quantile(values: FloatArray, level: float) -> float:
    """Return a quantile of the lead-time sample, or NaN when there is none.

    Args:
        values: Lead times, seconds.
        level: Quantile level in [0, 1].

    Returns:
        The quantile, seconds, or NaN if ``values`` is empty. NaN rather than 0.0: a cell
        with no matched onset has no lead-time distribution, and 0.0 would read as "the
        model flags exactly at the onset".
    """
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, level))


def _interval_bounds(raw: Tensor, head: HeadKind, levels: tuple[float, ...]) -> Tensor:
    """Return a model's two-sided 90 % interval bounds, in the model's own space.

    Args:
        raw: Model output, shape ``(B, H, C, K)``.
        head: The head kind that produced ``raw``.
        levels: The quantile fan levels, empty for a Gaussian head.

    Returns:
        Bounds, shape ``(B, H, C, 2)``, carrying ``(q05, q95)``.
    """
    return PredictiveDistribution(raw, head, levels).quantiles_at(INTERVAL_LEVELS)


def evaluate_quiescence(
    models: Mapping[str, ForecastModel],
    dataset: DeckMotionDataset,
    *,
    fs_hz: float,
    threshold_sets: Sequence[QuiescenceThresholds] = (PERMISSIVE, STRICT),
    tolerance_s: float = 0.5,
    include_always_quiescent: bool = True,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the quiescent-window detector for every model on identical windows.

    Mirrors :func:`dmf.eval.runner.evaluate_models`: it takes pre-built models and does not
    care how they were built, so it composes with a checkpoint loader, a closed-form fit, or
    a synthetic model in a test.

    Args:
        models: Detectors to score, keyed by results-table label. Point models are scored on
            the ``point`` rule only; a model whose ``head_kind`` is not ``"point"`` is
            additionally scored on the two-sided ``interval`` rule (P6-D5).
        dataset: The test partition. Its ``target_dofs`` must include roll, pitch and heave
            rate, since the prediction side thresholds the model's own output.
        fs_hz: Sampling rate, hertz. Passed rather than inferred, exactly as
            :func:`dmf.eval.runner.evaluate_models` takes it.
        threshold_sets: Limit sets to report, each producing its own rows.
        tolerance_s: Onset matching tolerance, seconds.
        include_always_quiescent: Whether to add the two synthetic references,
            :data:`SYNTHETIC_DETECTORS`. Neither costs a forward pass, and between them they
            are what makes every other F1 in the table readable: ``always_quiescent`` bounds
            the metric from the over-flagging side and ``rate_matched`` from the
            right-rate/wrong-time side.
        batch_size: Windows per batch. Affects speed and memory only.
        num_workers: DataLoader worker processes.
        device: Torch device the models run on.

    Returns:
        Tuple ``(summary, lead_times)`` with columns :data:`QUIESCENCE_COLUMNS` and
        :data:`LEAD_TIME_COLUMNS`. ``summary`` holds **two** rows per reported quantity:
        one per ``(sea state, heading, speed)`` grid cell and one per sea state rolling
        those cells up, told apart by ``group_level`` (P6-D19). It is sorted by
        ``(model, threshold_set, rule, group_level, ss, heading_deg, speed_kn)``.

    Raises:
        ValueError: If ``models`` is empty, if ``fs_hz`` is not positive, if the dataset
            does not forecast all three decision channels, or if a model returns an
            unexpected output shape.
        RuntimeError: If the window index does not partition by realization, or if the
            loader does not yield every window.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    spec = dataset.window_spec
    dof_names = tuple(dataset.target_columns)
    channels = decision_channel_index(dof_names)
    n_targets = len(dof_names)
    keys = list(dataset.realization_keys)
    n_keys = len(keys)
    per_realization = dataset.windows_per_realization
    if n_keys * per_realization != len(dataset):
        raise RuntimeError(
            f"dataset reports {len(dataset)} windows but {n_keys} realizations x "
            f"{per_realization} windows each is {n_keys * per_realization}"
        )

    geometry = _geometry(dataset)
    trajectories = _true_trajectories(dataset)
    truth = {
        thresholds.name: _truth_state(trajectories, thresholds, fs_hz, geometry)
        for thresholds in threshold_sets
    }

    def _new_state() -> _DetectorState:
        return _DetectorState([{} for _ in range(n_keys)], [0] * n_keys)

    heads = {name: _head_of(model) for name, model in models.items()}
    states: dict[tuple[str, str, str], _DetectorState] = {}
    for name in models:
        rules = ("point",) if heads[name][0] == "point" else RULES
        for rule in rules:
            for thresholds in threshold_sets:
                states[(name, rule, thresholds.name)] = _new_state()

    stats = dataset.norm_stats.subset(dof_names)
    torch_device = torch.device(device)
    previous_modes: dict[str, bool] = {}
    for name, model in models.items():
        if isinstance(model, nn.Module):
            previous_modes[name] = model.training
            model.to(torch_device)
            model.eval()

    loader = make_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=0
    )
    offset = 0
    try:
        with torch.no_grad():
            for x, _, window_mean in loader:
                batch = int(x.shape[0])
                indices = np.arange(offset, offset + batch, dtype=np.int64)
                realization_index = indices // per_realization
                decision_times = geometry.decision_times[indices % per_realization]
                mean = window_mean.double()
                inputs = x.to(torch_device)
                for name, model in models.items():
                    head, levels, width = heads[name]
                    raw = model.forward(inputs).detach().to("cpu", torch.float64)
                    expected = (batch, spec.max_horizon, n_targets, width)
                    if head == "point":
                        if tuple(raw.shape) != expected[:3]:
                            raise ValueError(
                                f"model {name!r} returned shape {tuple(raw.shape)}, "
                                f"expected {expected[:3]}"
                            )
                        point_raw = raw
                    else:
                        if tuple(raw.shape) != expected:
                            raise ValueError(
                                f"model {name!r} returned shape {tuple(raw.shape)}, "
                                f"expected {expected}"
                            )
                        point_raw = PredictiveDistribution(raw, head, levels).point()
                    point = invert_norm(point_raw, stats, mean).numpy()
                    for thresholds in threshold_sets:
                        _fold(
                            states[(name, "point", thresholds.name)],
                            point,
                            channels,
                            thresholds,
                            decision_times,
                            realization_index,
                            fs_hz,
                            geometry,
                        )
                    if head == "point":
                        continue
                    bounds = invert_norm(_interval_bounds(raw, head, levels), stats, mean).numpy()
                    conservative = np.maximum(np.abs(bounds[..., 0]), np.abs(bounds[..., 1]))
                    for thresholds in threshold_sets:
                        _fold(
                            states[(name, "interval", thresholds.name)],
                            conservative,
                            channels,
                            thresholds,
                            decision_times,
                            realization_index,
                            fs_hz,
                            geometry,
                        )
                offset += batch
    finally:
        for name, was_training in previous_modes.items():
            module = models[name]
            if isinstance(module, nn.Module) and was_training:
                module.train()

    if offset != len(dataset):
        raise RuntimeError(
            f"the loader yielded {offset} windows but the dataset holds {len(dataset)}; "
            f"a detector scored over a changed window population is not the detector"
        )

    if include_always_quiescent:
        for thresholds in threshold_sets:
            state = _new_state()
            states[(ALWAYS_QUIESCENT, "point", thresholds.name)] = state
            _fold_always_quiescent(state, geometry, n_keys)
            chance = _new_state()
            states[(RATE_MATCHED, "point", thresholds.name)] = chance
            _fold_rate_matched(chance, geometry, truth[thresholds.name], n_keys)

    observation_mode = _observation_mode(dof_names)
    summary_rows: list[dict[str, object]] = []
    lead_rows: list[dict[str, object]] = []
    for (name, rule, threshold_name), state in states.items():
        thresholds = next(t for t in threshold_sets if t.name == threshold_name)
        rows, leads = _summarise(
            model=name,
            regime=dataset.regime,
            observation_mode=observation_mode,
            thresholds=thresholds,
            rule=rule,
            keys=keys,
            truth=truth[threshold_name],
            state=state,
            fs_hz=fs_hz,
            tolerance_s=tolerance_s,
        )
        summary_rows.extend(rows)
        lead_rows.extend(leads)

    summary = pd.DataFrame(summary_rows, columns=list(QUIESCENCE_COLUMNS)).sort_values(
        ["model", "threshold_set", "rule", "group_level", "ss", "heading_deg", "speed_kn"],
        ignore_index=True,
    )
    lead_frame = pd.DataFrame(lead_rows, columns=list(LEAD_TIME_COLUMNS))
    return summary, lead_frame
