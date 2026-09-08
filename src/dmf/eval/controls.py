"""Integrity controls -- the checks run before believing any result.

Five controls guard this project -- three on the point path and, since Phase 6, two on the
interval path -- and none of them is a unit test of a function; each is a whole-pipeline
experiment whose expected outcome is known in advance:

1. **Pipeline sanity** (Phase 2, closed in Phase 3). Persistence evaluated through the full
   dataset pipeline must reproduce persistence computed directly on the raw Parquet arrays.
   An off-by-one in the windowing, a mis-inverted normalisation, or a channel-order
   mismatch each break this, and none of the three is visible in a loss curve. P2-D9 left
   an obligation: the control computed its forecast inline because
   :class:`dmf.models.persistence.Persistence` did not exist yet.
   :func:`persistence_pipeline_sanity` now takes an optional ``model``; when supplied it runs
   the real model *and* asserts the
   model's output is **bitwise** equal to the inline expression, which is the tightest
   available closure of that obligation.
2. **Shuffle control** (:func:`shuffle_control`). Retrain the best model on time-shuffled
   targets and check that its skill collapses.
3. **Untrained control** (:func:`untrained_control`). A randomly initialised model must
   score worse than persistence.
4. **Interval shuffle control** (:func:`interval_shuffle_control`). The first shuffle
   control in this project to run on a *head*: an interval refitted on time-shuffled
   targets must not price its own uncertainty better than an unconditional one.
5. **Interval untrained control** (:func:`interval_untrained_control`). The first untrained
   control to run on an *interval*. Reported rather than enforced, for a reason derived in
   its docstring rather than inherited.

Controls 4 and 5 close Phase 6 carry-forward item 6 (``docs/IMPLEMENTATION_PLAN.md`` Phase 6,
and ``docs/protocol.md`` P5-D10): before them, every coverage number this project published
rested on a path no negative control had ever touched. Both score against the **unconditional
residual interval** of P6-D6 -- the baseline a head must beat and the null a control scores
against are the same question asked twice. Their rows go to ``interval_controls.csv``, never
into ``baselines_controls.csv``: a Winkler ratio and a squared-error ratio are not the same
statistic and do not belong in one table.

**The stated null for the shuffle control is wrong for this task, and using it would fire a
false alarm.** The protocol says shuffle-control skill "must collapse to approximately
zero". It will not. Trained on time-shuffled targets, a least-squares model degenerates to
predicting the conditional mean of the target, which in the dimensionless space the models
work in is approximately 0, and which :func:`dmf.data.normalize.invert_norm` maps back to
**the window mean**. The window-mean forecast *beats* persistence at long horizons: for a
narrowband zero-mean signal, persistence error saturates at ``sqrt(2) * RMS`` while the
mean forecast saturates at ``1.0 * RMS``, i.e. roughly ``+0.5`` skill at 5 s. Testing
against zero would report leakage where there is none.

The correct null is therefore the **window-mean forecast**, which is exactly
``DampedPersistence`` in the ``tau -> 0+`` limit, so no new model file is needed. The
control asserts ``skill(shuffled) <= skill(window_mean) + tol``. The same argument applies
to the untrained control at long horizons -- a random-init network with a small-weight
initialisation also sits between the window mean and noise -- so
:func:`untrained_control` reports every (DOF, horizon) cell rather than one summary
number.

**Reported and enforced are different commitments, and the row says which.** Every control
row carries ``enforced``: the ``strict`` argument it was called with, i.e. whether a failure
would have stopped the run. The shuffle and interval-shuffle controls ship ``enforced=True``;
the two untrained controls ship ``enforced=False`` (P3-D9, and :func:`interval_untrained_control`
for the interval half), because their literal criterion is known to be wrong for this task and
tuning it until it passed would be tuning a control to its subject. ``passed`` is still
computed and written for them. A control that is reported but not enforced must be
distinguishable from one that gates the run **by column and not by paragraph**, because only
the column survives into a filtered table.

**The shuffle control asserts only where the target carries signal** (``docs/protocol.md``
P6-D11, which corrects P3-D18). On a channel sitting on its P1-D2 residual floor the
statistic measures a train/test amplitude mismatch of either sign rather than information
flow; :func:`floored_dofs` derives that set from the floor's definition and the partition's
own headings. Excluded cells are computed, written and reported exactly like the rest --
the ``asserted`` column, not their absence, is what marks them. The 2% tolerance is
unchanged.
"""

import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dmf.data.channels import logical_channel
from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import invert_norm
from dmf.data.splits import RealizationKey
from dmf.data.windows import window_origins, window_start_indices
from dmf.eval.gate import GATE5_NOMINAL, GATE5_PICP_BAND
from dmf.eval.prob_runner import evaluate_probabilistic_models
from dmf.eval.runner import evaluate_models
from dmf.models.base import BaseForecaster, ForecastModel
from dmf.models.residual_interval import EmpiricalResidualInterval
from dmf.sim.generate import VESSEL_CONFIG_DIR, RealizationSpec, realization_path
from dmf.sim.response import heading_factor
from dmf.sim.vessel import load_vessel
from dmf.typedefs import FloatArray

__all__ = [
    "CONTROL_COLUMNS",
    "INTERVAL_CONTROL_COLUMNS",
    "INTERVAL_CONTROL_METRICS",
    "INTERVAL_NULL_PICP_BAND",
    "INTERVAL_SHUFFLE_TOL",
    "INTERVAL_UNTRAINED_TOL",
    "RESIDUAL_FLOOR_MARGIN",
    "ControlResult",
    "IntervalControlResult",
    "PIPELINE_SANITY_COLUMNS",
    "PipelineSanityResult",
    "controls_table",
    "dataset_floored_dofs",
    "floored_dofs",
    "interval_controls_table",
    "interval_shuffle_control",
    "interval_untrained_control",
    "persistence_pipeline_sanity",
    "pipeline_sanity_table",
    "shuffle_control",
    "untrained_control",
]

#: Column order of ``results/baselines_controls.csv``. Controls are written to their own
#: file, never merged into the headline model table: a negative control that appears beside
#: the models can be mistaken for one, and CLAUDE.md non-negotiable 6 forbids a results
#: table whose rows do not all mean the same thing.
#: ``excess`` is ``1 - MSE_subject/MSE_null``, i.e. the subject's skill measured against
#: the control's null rather than against persistence; ``skill_subject`` and ``skill_null``
#: are both against persistence, so the row carries all three. See :func:`_control_rows` for
#: why a plain difference of skill scores is the wrong statistic here.
#: ``asserted`` and ``enforced`` answer **different questions** and are never collapsed:
#:
#: - ``asserted`` -- did *this row* take part in the control's pass/fail decision? False
#:   rows are computed, written and reported exactly like the others; only the assertion
#:   skips them. See :func:`floored_dofs` for the one reason a shuffle-control row is ever
#:   False (``docs/protocol.md`` P6-D11, P6-D12).
#: - ``enforced`` -- would a failure of *this control* have stopped the run? It is the
#:   ``strict`` argument the control was called with, recorded on every row rather than
#:   left to the caller's memory. The untrained control runs with ``strict=False`` (P3-D9)
#:   because its literal criterion is known-wrong for this task, so its rows ship
#:   ``enforced=False`` while the shuffle control's ship True.
#:
#: A row can be asserted and unenforced (every untrained row: it counted toward the
#: control's verdict, and the verdict raised nothing), or unasserted and enforced (a floored
#: shuffle cell: the control would have stopped the run, but not because of this row).
#: Collapsing them would make "reported, not enforced" a paragraph again -- and a paragraph
#: is what this column exists to replace, because a reader filtering ``passed == False``
#: cannot see a paragraph.
CONTROL_COLUMNS: tuple[str, ...] = (
    "control",
    "regime",
    "subject_model",
    "null_model",
    "dof",
    "horizon_samples",
    "horizon_s",
    "skill_subject",
    "skill_null",
    "excess",
    "tol",
    "asserted",
    "enforced",
    "passed",
)

#: How far above its residual floor a DOF's directional excitation must sit before the
#: shuffle control asserts on it. A factor of 2 is 6 dB above the floor, i.e. still ~20 dB
#: below the on-axis maximum. The corpus heading grid makes the choice non-critical: at the
#: nearest non-floored heading (45 deg from the DOF's null) the factor is ``0.707``, i.e.
#: **14x** its ``0.05`` floor, so any margin in ``(1, 14)`` selects the same cells.
RESIDUAL_FLOOR_MARGIN: float = 2.0

#: Target channel -> the DOF whose heading factor sets its amplitude. Rate and acceleration
#: channels inherit their position channel's floor: ``pitch_rate`` is the time derivative of
#: a signal the floor has already clamped, so it is clamped by the same factor.
BASE_DOF_BY_CHANNEL: Mapping[str, str] = {
    "roll": "roll",
    "roll_rate": "roll",
    "pitch": "pitch",
    "pitch_rate": "pitch",
    "heave": "heave",
    "heave_rate": "heave",
    "heave_acc": "heave",
}


@cache
def _residuals(vessel: str) -> tuple[float, float]:
    """Return one hull's ``(roll, pitch)`` heading-factor residual floors.

    Read from the same YAML the simulator read, so a corpus regenerated with a different
    floor moves this derivation with it rather than leaving a stale constant in the
    evaluator.

    Args:
        vessel: Vessel config stem, e.g. ``"frigate"``.

    Returns:
        ``(roll_residual, pitch_residual)``, both dimensionless.

    Raises:
        ValueError: If no config exists for ``vessel``.
    """
    path = VESSEL_CONFIG_DIR / f"{vessel}.yaml"
    if not path.exists():
        raise ValueError(f"unknown vessel {vessel!r}: no config at {path}")
    loaded = load_vessel(path)
    return float(loaded.roll_residual), float(loaded.pitch_residual)


def _on_residual_floor(channel: str, heading_deg: float, vessel: str, margin: float) -> bool:
    """Say whether one target channel sits on its P1-D2 residual floor at one heading.

    Args:
        channel: Target channel name, either spelling (``"pitch"`` or ``"pitch_imu"``).
        heading_deg: Encounter angle, degrees. 180 head, 90 beam.
        vessel: Vessel config stem, whose YAML supplies the floors.
        margin: Multiple of the floor the directional factor may reach and still count as
            floored.

    Returns:
        True if the DOF's directional excitation at this heading is within ``margin`` of
        its floor.

    Raises:
        ValueError: If ``channel`` is not a corpus motion channel.
    """
    logical = logical_channel(channel)
    base = BASE_DOF_BY_CHANNEL.get(logical)
    if base is None:  # pragma: no cover -- resolve_columns admits no other channel
        raise ValueError(f"channel {channel!r} maps to no DOF; known: {list(BASE_DOF_BY_CHANNEL)}")
    roll_residual, pitch_residual = _residuals(vessel)
    residual = roll_residual if base == "roll" else pitch_residual
    if base == "heave":
        return False  # heading_factor is identically 1.0 for heave: no floor to sit on
    return heading_factor(base, heading_deg, residual) <= margin * residual


def floored_dofs(
    dof_names: Iterable[str],
    test_keys: Iterable[RealizationKey],
    *,
    margin: float = RESIDUAL_FLOOR_MARGIN,
) -> frozenset[str]:
    """Name the target channels whose test-set signal is nothing but their P1-D2 floor.

    ``docs/protocol.md`` P1-D2 gives roll the heading factor ``sqrt(sin(beta)**2 + eps**2)``
    and pitch ``sqrt(cos(beta)**2 + eps**2)``, with ``eps = 0.05``. Each is clamped at one
    heading -- roll at 180 deg, pitch at 90 deg -- where the channel is a 26 dB-suppressed
    engineering stand-in rather than the physics the DOF is named after. On such a channel
    the shuffle control's statistic is dominated by a train/test amplitude mismatch: the
    shuffled model carries a fitted amplitude from headings where the DOF is real and
    imposes it on a signal that has none. P3-D18 measured that in the safe direction
    (-0.0266 at L=200) and P6-D11 measured the *same mechanism in the failing direction*
    (+0.0552 at L=100), established over six measurements that it is not leakage, and
    narrowed the control's assertion to exclude it.

    **This is derived from the floor, not from the list of cells that failed.** A channel is
    floored for a partition when it is floored at *every* realization in it: one unfloored
    heading in the test set puts real signal on the channel and the control asserts as
    normal.

    Args:
        dof_names: Target channel names, in either observation-mode spelling.
        test_keys: Realization keys of the partition being scored, i.e.
            ``(ss, heading_deg, speed_kn, vessel, seed)``. The headings and hulls are read
            from these, so the answer follows the corpus rather than a hard-coded pair.
        margin: Multiple of the floor the directional factor may reach and still count as
            floored; see :data:`RESIDUAL_FLOOR_MARGIN`.

    Returns:
        The subset of ``dof_names`` that is on its floor across the whole partition. Empty
        for every regime whose test set contains a heading where the DOF is excited; on the
        production corpus this is ``{pitch, pitch_rate}`` for ``unseen_heading`` and empty
        for ``id``, ``unseen_seastate`` and ``unseen_vessel``.

    Raises:
        ValueError: If ``test_keys`` is empty -- an empty partition would floor every
            channel vacuously, which would silently disarm the control.
    """
    keys = list(test_keys)
    if not keys:
        raise ValueError("cannot derive the residual-floor set from an empty partition")
    cells = {(float(key[1]), str(key[3])) for key in keys}
    return frozenset(
        name
        for name in dof_names
        if all(_on_residual_floor(name, heading, vessel, margin) for heading, vessel in cells)
    )


def dataset_floored_dofs(
    dataset: DeckMotionDataset, *, margin: float = RESIDUAL_FLOOR_MARGIN
) -> frozenset[str]:
    """Apply :func:`floored_dofs` to the partition a control is about to score.

    Taken from the dataset rather than from a :class:`dmf.data.splits.Split` so that the
    headings are those of the windows actually scored: a control run on a subset of a
    regime gets the subset's answer, not the regime's.

    Args:
        dataset: The partition being scored.
        margin: See :data:`RESIDUAL_FLOOR_MARGIN`.

    Returns:
        The floored subset of ``dataset.target_columns``.
    """
    return floored_dofs(dataset.target_columns, dataset.realization_keys, margin=margin)


@dataclass(frozen=True)
class PipelineSanityResult:
    """Outcome of the persistence pipeline-sanity control.

    Attributes:
        rmse_pipeline: Per-(horizon, target channel) persistence RMSE measured through the
            dataset, shape ``(H, C_out)``, in corpus units (degrees for roll and pitch,
            metres for heave). The horizon axis is per-step: row ``h - 1`` is the error at
            lead time exactly ``h`` samples, never a cumulative average.
        rmse_raw: The same quantity computed directly from the Parquet files, shape
            ``(H, C_out)``, same units.
        max_rel_diff: Largest relative disagreement between the two, dimensionless.
        n_windows: Number of windows scored. Identical on both paths, or the control
            raises.
        dof_names: Target channel names, length ``C_out``.
        model_matches_inline: True if a real forecaster was supplied and its output was
            bitwise equal to the inline persistence expression on every batch; None if no
            model was supplied (the Phase 2 default). It is never False: a mismatch raises.
    """

    rmse_pipeline: FloatArray
    rmse_raw: FloatArray
    max_rel_diff: float
    n_windows: int
    dof_names: tuple[str, ...]
    model_matches_inline: bool | None = None


@dataclass(frozen=True)
class ControlResult:
    """Outcome of a negative control run through the full evaluation path.

    Attributes:
        control: Control name, e.g. ``"shuffle"`` or ``"untrained"``.
        rows: One row per (DOF, horizon), columns :data:`CONTROL_COLUMNS`. This is the
            frame written to ``results/baselines_controls.csv``. It holds **every** cell
            scored, including the ones the assertion skipped; ``asserted`` says which.
        table: The raw output of :func:`dmf.eval.runner.evaluate_models` for every model the
            control scored, kept so the control's own numbers are auditable rather than
            summarised away.
        worst_excess: Largest ``excess`` over the **asserted** cells, dimensionless.
            Negative means the control behaved as predicted. Reported-only cells are in
            :attr:`worst_excess_reported_only`, never folded into this one.
        tol: Tolerance the excess was compared against.
        passed: Whether the control passed, over the asserted cells only.
        enforced: Whether a failure raised, i.e. the ``strict`` argument this control was
            called with. False means the control is **reported rather than enforced**;
            :attr:`passed` still says what it found, and the ``enforced`` column of
            :attr:`rows` carries the same answer into the CSV so a reader who never sees
            this object cannot mistake a reported control for a gating one.
        excluded_dofs: Target channels reported but not asserted on, sorted. Empty for a
            control that asserted everywhere.
        exclusion_reason: Why ``excluded_dofs`` were excluded, for the message a strict
            failure prints and for the report renderer. Empty when nothing was excluded.
    """

    control: str
    rows: pd.DataFrame
    table: pd.DataFrame
    worst_excess: float
    tol: float
    passed: bool
    enforced: bool
    excluded_dofs: tuple[str, ...] = ()
    exclusion_reason: str = ""

    @property
    def asserted_rows(self) -> pd.DataFrame:
        """Return the rows the pass/fail decision was taken over.

        Returns:
            The subset of :attr:`rows` with ``asserted`` True.
        """
        return self.rows[self.rows["asserted"]]

    @property
    def reported_only_rows(self) -> pd.DataFrame:
        """Return the rows that were computed and reported but not asserted on.

        Kept reachable so the renderer can show them next to the asserted ones: a cell the
        control declines to assert on is a finding, not a gap (CLAUDE.md non-negotiable 6).

        Returns:
            The subset of :attr:`rows` with ``asserted`` False.
        """
        return self.rows[~self.rows["asserted"]]

    @property
    def worst_excess_reported_only(self) -> float | None:
        """Return the worst excess among the reported-only cells.

        Returns:
            The largest ``excess`` over cells the assertion skipped, or None if there were
            none. Never compared against ``tol`` by this class; it exists to be read.
        """
        rows = self.reported_only_rows
        return None if rows.empty else float(rows["excess"].max())


def _raw_persistence_sse(dataset: DeckMotionDataset, corpus_root: Path) -> tuple[FloatArray, int]:
    """Accumulate persistence squared error directly from the Parquet files.

    Deliberately shares nothing with :class:`DeckMotionDataset` except the realization key
    list and the window geometry: it re-reads the files with pandas, recomputes the start
    indices from :func:`dmf.data.windows.window_start_indices`, and uses no normalisation
    and no torch. If the dataset's internal index arithmetic were off by one, this path
    would not follow it.

    On an origin-subset dataset (``dataset.is_origin_subset``) the same start indices are
    recomputed here and then **filtered by origin membership**, with the origins likewise
    recomputed from the geometry rather than read off the dataset. What is taken from the
    dataset is the requested origin *set* -- an input to both paths, not a derived index --
    so the off-by-one check the control exists for still runs on the subset.

    Args:
        dataset: The partition to score.
        corpus_root: Dataset root.

    Returns:
        Tuple ``(sse, n_windows)`` where ``sse`` has shape ``(H, C_out)`` in squared corpus
        units.
    """
    spec = dataset.window_spec
    columns = list(dataset.target_columns)
    horizon_offsets = spec.lookback + np.arange(spec.max_horizon, dtype=np.int64)
    sse = np.zeros((spec.max_horizon, len(columns)), dtype=np.float64)
    count = 0
    kept = dataset.window_origins if dataset.is_origin_subset else None
    for key in dataset.realization_keys:
        frame = pd.read_parquet(corpus_root / _path_for(key), columns=columns)
        series = frame[columns].to_numpy(dtype=np.float64)
        starts = window_start_indices(series.shape[0], spec)
        if kept is not None:
            starts = starts[np.isin(window_origins(series.shape[0], spec), kept)]
        last = series[starts + spec.lookback - 1, :]
        future = series[starts[:, None] + horizon_offsets[None, :], :]
        sse += np.square(future - last[:, None, :]).sum(axis=0)
        count += int(starts.size)
    return sse, count


def _path_for(key: RealizationKey) -> Path:
    """Return the corpus-relative Parquet path for a realization key.

    Args:
        key: ``(ss, heading_deg, speed_kn, vessel, seed)``.

    Returns:
        The relative path.
    """
    ss, heading, speed, vessel, seed = key
    return realization_path(
        RealizationSpec(seed=seed, sea_state=ss, heading_deg=heading, speed_kn=speed, vessel=vessel)
    )


def _pipeline_persistence_sse(
    dataset: DeckMotionDataset,
    batch_size: int,
    model: ForecastModel | None = None,
) -> tuple[FloatArray, int, bool | None]:
    """Accumulate persistence squared error through the dataset pipeline.

    The forecast is formed in *normalised* space, exactly where a model's would be
    (``x[:, -1:, :C_out]`` broadcast over the horizon, which is what
    :class:`dmf.models.persistence.Persistence` is documented to compute), and is then
    mapped back to corpus units through :func:`dmf.data.normalize.invert_norm`.

    With ``model`` None -- the Phase 2 default -- the forecast is that inline expression and
    the control tests the *pipeline*, not the model. With a model supplied, the scored
    forecast is ``model.forward(x)`` and the inline expression becomes the reference it is
    checked against, bitwise. That closes the P2-D9 obligation: the two cannot silently
    diverge, and if they ever do, every skill denominator in the project stops being the
    quantity this control validated.

    Args:
        dataset: The partition to score.
        batch_size: Windows per batch. Affects speed only.
        model: Persistence forecaster to score instead of the inline expression. Must
            produce the inline expression bitwise; anything else raises. This is the
            *persistence* control, so a non-persistence model is a caller error, not a
            finding.

    Returns:
        Tuple ``(sse, n_windows, model_matches_inline)`` where ``sse`` has shape
        ``(H, C_out)`` in squared corpus units and the last element is None when no model
        was supplied.

    Raises:
        AssertionError: If ``model`` is supplied and its output differs from the inline
            expression on any element of any batch.
    """
    stats = dataset.norm_stats.subset(dataset.target_columns)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, seed=0)
    sse = torch.zeros(
        (dataset.window_spec.max_horizon, len(dataset.target_columns)), dtype=torch.float64
    )
    count = 0
    matches: bool | None = None if model is None else True
    with torch.no_grad():
        for x, y, window_mean in loader:
            n_out = y.shape[2]
            inline = x[:, -1:, :n_out].expand(-1, y.shape[1], -1)
            if model is None:
                pred_norm = inline
            else:
                pred_norm = model.forward(x)
                if pred_norm.shape != inline.shape:
                    raise AssertionError(
                        f"the supplied model returned shape {tuple(pred_norm.shape)} but the "
                        f"inline persistence expression has shape {tuple(inline.shape)}"
                    )
                if not torch.equal(pred_norm, inline):
                    worst = (pred_norm - inline).abs().max().item()
                    raise AssertionError(
                        f"the supplied model is not bitwise equal to the inline persistence "
                        f"expression x[:, -1:, :C_out].expand(-1, H, -1) that Gate 2 "
                        f"criterion 5 validated (max |diff| = {worst:.3e}). Every skill "
                        f"denominator in this project is that expression; if the model "
                        f"differs, the denominators are no longer the quantity the control "
                        f"checked (docs/protocol.md P2-D9)."
                    )
            pred = invert_norm(pred_norm, stats, window_mean)
            sse += torch.square(pred.double() - y.double()).sum(dim=0)
            count += int(y.shape[0])
    return sse.numpy(), count, matches


def persistence_pipeline_sanity(
    dataset: DeckMotionDataset,
    corpus_root: Path,
    *,
    rtol: float = 1e-6,
    batch_size: int = 512,
    model: ForecastModel | None = None,
) -> PipelineSanityResult:
    """Check that persistence through the pipeline matches persistence on raw arrays.

    Gate 2 criterion 5. Two paths that share no code beyond the realization key list and the
    window geometry must agree to ``rtol`` on the per-(horizon, DOF) RMSE.

    The two paths are not bitwise identical by construction: the pipeline stores its
    de-meaned, scaled input as float32, so the forecast it recovers carries a relative
    rounding error of order ``2**-24``. ``rtol`` is stated rather than assumed for that
    reason. The measured value on the full ``id/test`` partition is 5.006e-08, i.e. the
    float32 storage floor and nothing else.

    Args:
        dataset: The partition to score. Any partition may be used; the ``test`` partition
            of a regime is the interesting one.
        corpus_root: Dataset root, re-read independently by the raw path.
        rtol: Largest tolerated relative disagreement, dimensionless.
        batch_size: Windows per batch on the pipeline path. Affects speed only.
        model: Optional real persistence forecaster. Supplying it makes the pipeline path
            run the model and additionally asserts the model is bitwise equal to the inline
            expression, which is what P2-D9 requires of Phase 3. Leaving it None reproduces
            the Phase 2 behaviour exactly.

    Returns:
        The measured RMSE on both paths and their largest relative disagreement.

    Raises:
        AssertionError: If the two paths disagree beyond ``rtol``, naming the worst
            (horizon, DOF) cell; if they score different numbers of windows; or if a
            supplied ``model`` is not bitwise equal to the inline expression.
    """
    if rtol <= 0.0:
        raise ValueError(f"rtol must be positive, got {rtol}")
    sse_pipeline, n_pipeline, matches = _pipeline_persistence_sse(dataset, batch_size, model)
    sse_raw, n_raw = _raw_persistence_sse(dataset, corpus_root)
    assert n_pipeline == n_raw, (
        f"pipeline scored {n_pipeline} windows but the raw path scored {n_raw}: the "
        f"dataset's window count disagrees with window_start_indices"
    )
    rmse_pipeline = np.sqrt(sse_pipeline / n_pipeline)
    rmse_raw = np.sqrt(sse_raw / n_raw)
    denom = np.maximum(np.abs(rmse_raw), np.finfo(np.float64).tiny)
    rel = np.abs(rmse_pipeline - rmse_raw) / denom
    flat = int(np.argmax(rel))
    worst_h, worst_c = np.unravel_index(flat, rel.shape)
    max_rel_diff = float(rel[worst_h, worst_c])
    dof_names = tuple(dataset.target_columns)
    assert max_rel_diff <= rtol, (
        f"persistence through the pipeline disagrees with persistence on the raw arrays by "
        f"{max_rel_diff:.3e} (rtol={rtol:.1e}) at horizon {int(worst_h) + 1} samples, DOF "
        f"{dof_names[int(worst_c)]!r}: pipeline {rmse_pipeline[worst_h, worst_c]:.9g} vs raw "
        f"{rmse_raw[worst_h, worst_c]:.9g} over {n_raw} windows"
    )
    return PipelineSanityResult(
        rmse_pipeline=rmse_pipeline,
        rmse_raw=rmse_raw,
        max_rel_diff=max_rel_diff,
        n_windows=n_pipeline,
        dof_names=dof_names,
        model_matches_inline=matches,
    )


#: Column order of ``pipeline_sanity.csv``. **Its own file**, for the same reason the
#: interval controls have theirs: the statistic here is a *relative difference between two
#: RMSEs computed by two independent paths*, not a fraction of a null's error removed, so it
#: shares neither ``excess`` nor ``tol``'s meaning with :data:`CONTROL_COLUMNS` and a reader
#: filtering one file on ``excess`` must not silently pick up rows where that column would
#: mean something else.
#:
#: **How to read a table in which every row passes.** This control *raises* -- it is the one
#: control the training and scoring drivers enforce unconditionally -- so a failing row
#: cannot reach a CSV: the run that would have written it stopped instead. The value of the
#: table is therefore the measured ``rel_diff``, which should sit at the float32 storage
#: floor (~5e-8 on the production ``id/test`` partition, P2-D9) and not at the tolerance;
#: a run whose numbers had drifted to just inside ``rtol`` would be visible here and
#: invisible in a pass/fail column. ``passed`` is written anyway, so the file has the same
#: shape as the other control files and can be read by the same eye.
PIPELINE_SANITY_COLUMNS: tuple[str, ...] = (
    "control",
    "experiment",
    "arm",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "rmse_pipeline",
    "rmse_raw",
    "rel_diff",
    "max_rel_diff",
    "rtol",
    "n_windows",
    "model_matches_inline",
    "asserted",
    "enforced",
    "passed",
)


def pipeline_sanity_table(
    results: Sequence[tuple[PipelineSanityResult, str, str, str]],
    *,
    horizons: tuple[int, ...],
    fs_hz: float,
    rtol: float,
) -> pd.DataFrame:
    """Tabulate pipeline-sanity outcomes for ``pipeline_sanity.csv``.

    Closes the reporting half of the control P6-D15 records as *enforced and unreported* --
    the mirror image of the untrained control, which is reported and unenforced. Nothing
    about the check changes: :func:`persistence_pipeline_sanity` still raises, and these rows
    exist so that the number it measured is in a file rather than only in an exception that
    never fired.

    Rows are emitted at the **reported** horizons, not at all ``max_horizon`` lead times, so
    that this table is read at the same lead times as every other table in the document.
    ``max_rel_diff`` is the maximum over *every* cell the control actually compared, reported
    on each row, so the projection to six horizons cannot hide a disagreement at a lead time
    the report does not print.

    Args:
        results: ``(result, experiment, arm, regime)`` per control run, in report order.
        horizons: Horizons to emit, samples, each in ``[1, H]``.
        fs_hz: Sampling rate, hertz, for the ``horizon_s`` column.
        rtol: The tolerance the control was run at, recorded rather than assumed.

    Returns:
        Frame with columns :data:`PIPELINE_SANITY_COLUMNS`, one row per
        (experiment, arm, regime, DOF, horizon).

    Raises:
        ValueError: If ``results`` is empty, or if a horizon is out of range for a result.
    """
    if not results:
        raise ValueError(
            "no pipeline-sanity results to tabulate; a run whose skill denominators were "
            "never checked against the raw arrays is not audited (Gate 2 criterion 5)"
        )
    rows: list[dict[str, object]] = []
    for result, experiment, arm, regime in results:
        max_horizon = result.rmse_pipeline.shape[0]
        bad = [h for h in horizons if not 1 <= h <= max_horizon]
        if bad:
            raise ValueError(
                f"horizons {bad} are out of range for a control that scored {max_horizon} "
                f"lead times on {regime!r}"
            )
        denom = np.maximum(np.abs(result.rmse_raw), np.finfo(np.float64).tiny)
        rel = np.abs(result.rmse_pipeline - result.rmse_raw) / denom
        for horizon in horizons:
            for channel, name in enumerate(result.dof_names):
                rows.append(
                    {
                        "control": "pipeline_sanity",
                        "experiment": experiment,
                        "arm": arm,
                        "regime": regime,
                        "dof": name,
                        "horizon_samples": horizon,
                        "horizon_s": horizon / fs_hz,
                        "rmse_pipeline": float(result.rmse_pipeline[horizon - 1, channel]),
                        "rmse_raw": float(result.rmse_raw[horizon - 1, channel]),
                        "rel_diff": float(rel[horizon - 1, channel]),
                        "max_rel_diff": result.max_rel_diff,
                        "rtol": rtol,
                        "n_windows": result.n_windows,
                        "model_matches_inline": result.model_matches_inline,
                        # True on both counts, and both are facts about the control rather
                        # than about the row: every cell took part in the comparison, and
                        # the comparison raises. A False here could only be written by a
                        # caller that caught the AssertionError, which nothing does.
                        "asserted": True,
                        "enforced": True,
                        "passed": result.max_rel_diff <= rtol,
                    }
                )
    return pd.DataFrame(rows, columns=list(PIPELINE_SANITY_COLUMNS))


def _skill_by_cell(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Extract one model's per-(DOF, horizon) skill from a runner table.

    Args:
        table: Output of :func:`dmf.eval.runner.evaluate_models`.
        model: Model key to extract.

    Returns:
        Frame with ``dof``, ``horizon_samples``, ``horizon_s`` and ``skill``.

    Raises:
        ValueError: If ``model`` has no rows in ``table``.
    """
    rows = table.loc[table["model"] == model, ["dof", "horizon_samples", "horizon_s", "skill"]]
    if rows.empty:
        raise ValueError(f"model {model!r} has no rows in the evaluation table")
    return rows.reset_index(drop=True)


def _control_rows(
    control: str,
    regime: str,
    subject: str,
    null: str,
    subject_skill: pd.DataFrame,
    null_skill: pd.DataFrame | None,
    tol: float,
    *,
    allow_equality: bool,
    enforced: bool,
    not_asserted: Collection[str] = (),
) -> pd.DataFrame:
    """Assemble the CSV rows for one control, including its pass statistic.

    ``excess`` is **not** a difference of skill scores. It is the subject's skill measured
    against the *null* instead of against persistence::

        excess = 1 - MSE_subject / MSE_null = 1 - (1 - skill_subject)/(1 - skill_null)

    A plain difference ``skill_subject - skill_null`` is not scale-free and fires spuriously
    wherever both models are far worse than persistence. Measured example: on ``id/test`` at
    a 1 s horizon on roll, an AR(20) fitted to time-shuffled targets scores -3.4591 and the
    window mean -3.4827, a difference of +0.0235 -- which would trip a 0.02 tolerance. The
    same pair as a ratio is ``MSE_subject/MSE_null = 0.9948``, i.e. the shuffled model
    removes **0.53%** of the null's error, which is the honest statement of how little it
    learned. One tolerance in this ratio means the same thing in every cell.

    When ``null`` is persistence itself the null skill is 0 and ``excess`` reduces to the
    subject's plain skill score, so the protocol's literal criterion is the special case.

    Args:
        control: Control name.
        regime: Regime the control was run on.
        subject: Key of the model under test.
        null: Key of the model (or constant) it is compared against.
        subject_skill: Per-cell skill of the subject, against persistence.
        null_skill: Per-cell skill of the null, against persistence, or None for the
            constant-zero null (i.e. persistence itself).
        tol: Largest tolerated fraction of the null's MSE the subject may remove.
        allow_equality: If True the control passes at ``excess == tol`` exactly, which is
            what an identical subject and null require.
        enforced: Whether a failure of this control raises, i.e. the ``strict`` argument it
            was called with. Recorded on every row: a control that is reported rather than
            enforced must be distinguishable from one that gates the run by a column, not by
            a paragraph beside the table.
        not_asserted: Target channels whose rows are reported but excluded from the
            control's pass/fail decision. Their ``passed`` value is still computed and
            written, so an excluded cell over tolerance is visible in the CSV rather than
            absent from it; only ``asserted`` marks it as not counted.

    Returns:
        Frame with columns :data:`CONTROL_COLUMNS`.

    Raises:
        ValueError: If the null is a perfect forecaster, which makes the ratio undefined.
    """
    null_values = (
        np.zeros(len(subject_skill)) if null_skill is None else null_skill["skill"].to_numpy()
    )
    subject_values = subject_skill["skill"].to_numpy()
    if np.any(null_values == 1.0):
        raise ValueError(
            "the null model is a perfect forecaster in at least one cell, so the control "
            "statistic 1 - MSE_subject/MSE_null is undefined"
        )
    excess = 1.0 - (1.0 - subject_values) / (1.0 - null_values)
    passed = excess <= tol if allow_equality else excess < tol
    dofs = subject_skill["dof"].to_numpy()
    excluded = set(not_asserted)
    asserted = np.array([str(dof) not in excluded for dof in dofs], dtype=bool)
    frame = pd.DataFrame(
        {
            "control": control,
            "regime": regime,
            "subject_model": subject,
            "null_model": null,
            "dof": dofs,
            "horizon_samples": subject_skill["horizon_samples"].to_numpy(),
            "horizon_s": subject_skill["horizon_s"].to_numpy(),
            "skill_subject": subject_values,
            "skill_null": null_values,
            "excess": excess,
            "tol": tol,
            "asserted": asserted,
            "enforced": enforced,
            "passed": passed,
        }
    )
    return frame[list(CONTROL_COLUMNS)]


def _evaluate_control(
    models: Mapping[str, ForecastModel],
    dataset: DeckMotionDataset,
    *,
    persistence_key: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    batch_size: int,
    num_workers: int,
    device: str,
) -> pd.DataFrame:
    """Score a control's models through the same runner the headline table uses.

    Args:
        models: Models to score.
        dataset: The partition to score.
        persistence_key: Key naming the reference baseline.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The runner's table.
    """
    table, _ = evaluate_models(
        models,
        dataset,
        persistence_key=persistence_key,
        horizons=horizons,
        fs_hz=fs_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        n_boot=1,
    )
    return table


def shuffle_control(
    dataset: DeckMotionDataset,
    *,
    shuffled_model: ForecastModel,
    window_mean_model: ForecastModel,
    persistence_model: ForecastModel,
    regime: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    tol: float = 0.02,
    strict: bool = True,
    floor_margin: float = RESIDUAL_FLOOR_MARGIN,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> ControlResult:
    """Check that a model trained on time-shuffled targets learned nothing.

    The subject is a model refitted on targets whose time ordering was destroyed, so every
    genuine input-to-future relationship is gone. What remains is whatever the model can
    still get from the marginal distribution of the target -- which for a least-squares fit
    is the conditional mean, i.e. the window mean once
    :func:`dmf.data.normalize.invert_norm` has added the offset back.

    The null is therefore the **window-mean forecast**, not zero skill. See the module
    docstring: the window mean beats persistence by roughly ``+0.5`` skill at 5 s on a
    narrowband signal, so a "skill must collapse to ~0" test would report leakage on a clean
    pipeline. Passing means the shuffled model does no better than the window mean it
    degenerated to; failing means information survived the shuffle, which can only be
    leakage.

    **The assertion is taken over the channels carrying signal, not over all of them**
    (``docs/protocol.md`` P6-D11). :func:`dataset_floored_dofs` names the target channels
    whose test-set targets are nothing but their P1-D2 residual floor -- on the production
    corpus, ``pitch`` and ``pitch_rate`` under ``unseen_heading`` and nothing else. There
    the statistic is dominated by a train/test amplitude mismatch that was measured in both
    directions (-0.0266 at L=200, +0.0552 at L=100) and shown over six measurements not to
    be leakage: it reverses sign with lookback, which leaked information cannot do; it grows
    with batch size, which imperfect shuffling cannot do; and it survives re-fitting at four
    shuffle seeds, which coefficient noise cannot do. Those cells are **still scored, still
    written to the CSV with their own ``passed`` value, and reachable from
    :attr:`ControlResult.reported_only_rows`** -- only the assertion skips them, and the
    strict message says how many were skipped and why. The tolerance is unchanged.

    Args:
        dataset: Test partition to score, normally ``id/test``.
        shuffled_model: The model refitted on time-shuffled training targets.
        window_mean_model: The window-mean forecaster, i.e. ``DampedPersistence`` at
            ``tau -> 0+``. Scored in the same pass so the comparison uses identical windows.
        persistence_model: The skill denominator, scored in the same pass for the same
            reason.
        regime: Regime label recorded on the output rows.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        tol: Largest fraction of the window-mean null's MSE the shuffled model may remove
            before the control fails, dimensionless. Measured on ``id/test`` with an AR(20)
            refitted on shuffled targets: **0.53%** at worst, against this 2% default.
        strict: If True, raise when the control fails on an asserted cell. A failed
            integrity control invalidates the headline table, so stopping loudly is the
            default.
        floor_margin: Multiple of the P1-D2 residual floor a DOF's directional excitation
            may reach and still count as floored; see :data:`RESIDUAL_FLOOR_MARGIN`.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The control's outcome, including per-cell rows for
        ``results/baselines_controls.csv``. ``worst_excess`` and ``passed`` cover the
        asserted cells; the rest are in :attr:`ControlResult.reported_only_rows`.

    Raises:
        AssertionError: If ``strict`` and the shuffled model beats the window-mean null by
            more than ``tol`` on a cell that is not on its residual floor.
        ValueError: If every scored channel is on its residual floor, which would leave the
            control asserting on nothing and passing vacuously.
    """
    table = _evaluate_control(
        {
            "persistence": persistence_model,
            "window_mean": window_mean_model,
            "shuffled": shuffled_model,
        },
        dataset,
        persistence_key="persistence",
        horizons=horizons,
        fs_hz=fs_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    subject = _skill_by_cell(table, "shuffled")
    null = _skill_by_cell(table, "window_mean")
    floored = dataset_floored_dofs(dataset, margin=floor_margin)
    rows = _control_rows(
        "shuffle",
        regime,
        "shuffled",
        "window_mean",
        subject,
        null,
        tol,
        allow_equality=True,
        enforced=strict,
        not_asserted=floored,
    )
    asserted = rows[rows["asserted"]]
    if asserted.empty:
        raise ValueError(
            f"every scored channel {sorted(floored)} is on its P1-D2 residual floor in this "
            f"partition, so the shuffle control would assert on nothing. A control with no "
            f"asserted cell passes vacuously; score a partition that contains a heading "
            f"where at least one target DOF is excited."
        )
    worst = float(asserted["excess"].max())
    passed = bool(asserted["passed"].all())
    excluded = tuple(sorted(set(rows.loc[~rows["asserted"], "dof"].tolist())))
    reason = (
        ""
        if not excluded
        else (
            f"{len(excluded)} channel(s) {list(excluded)} are on their P1-D2 residual floor "
            f"across this partition's headings, so their targets are a 26 dB-suppressed "
            f"stand-in rather than physics; the control's statistic there measures a "
            f"train/test amplitude mismatch, not leakage (docs/protocol.md P6-D11). Their "
            f"rows are in the table with asserted=False."
        )
    )
    if excluded:
        # Emitted on a PASS as well as a fail. The exclusion is a narrowing of an integrity
        # control, and a narrowing nobody sees is indistinguishable from a control that was
        # never run: the CSV column and the report section are for the reader afterwards,
        # this is for the person watching the sweep.
        warnings.warn(
            f"shuffle control on regime {regime!r}: {reason}", RuntimeWarning, stacklevel=2
        )
    if strict and not passed:
        worst_row = asserted.loc[asserted["excess"].idxmax()]
        raise AssertionError(
            f"shuffle control failed: a model trained on time-shuffled targets removes "
            f"{worst:.2%} of the window-mean null's error (tol={tol:.2%}) at DOF "
            f"{worst_row['dof']!r}, horizon {worst_row['horizon_samples']} samples "
            f"(shuffled skill {worst_row['skill_subject']:.4f} vs window mean "
            f"{worst_row['skill_null']:.4f}). Destroying the time ordering of the targets "
            f"must destroy the signal; if it does not, information is reaching the model "
            f"by some path other than the inputs."
            + (f" Excluded from this assertion: {reason}" if excluded else "")
        )
    return ControlResult(
        control="shuffle",
        rows=rows,
        table=table,
        worst_excess=worst,
        tol=tol,
        passed=passed,
        enforced=strict,
        excluded_dofs=excluded,
        exclusion_reason=reason,
    )


def untrained_control(
    dataset: DeckMotionDataset,
    *,
    untrained_model: ForecastModel,
    persistence_model: ForecastModel,
    regime: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    null_model: ForecastModel | None = None,
    tol: float = 0.0,
    strict: bool = True,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> ControlResult:
    """Check that a randomly initialised model does not beat a trivial forecast.

    Free: the untrained model is scored as one more entry in an evaluation pass that is
    happening anyway.

    **Measured, and it contradicts the protocol's stated criterion.** The protocol says a
    random-init model "must score worse than persistence", i.e. ``skill < 0``. On the real
    corpus it does not, and not because anything leaked: a freshly initialised
    ``DLinear(L=200, H=50)`` on ``id/test`` scores

    ===============  ======  ======  ======  ======
    DOF              1 s     2 s     3 s     5 s
    ===============  ======  ======  ======  ======
    roll             -3.429  -0.043  -0.875  +0.661
    pitch            -0.787  +0.317  +0.369  +0.696
    heave            -2.125  +0.056  -0.116  +0.680
    ===============  ======  ======  ======  ======

    The reason is the same one that breaks the shuffle control's stated null. A
    small-weight random projection of a de-meaned window produces an output of order the
    signal RMS but uncorrelated with the future, whose error saturates near the same level
    as the window-mean forecast -- and the window mean *beats* persistence beyond about 2 s
    on this narrowband signal (measured: +0.71 roll, +0.69 pitch, +0.70 heave at 5 s). So
    ``skill < 0`` at 5 s asks the untrained model to be worse than a baseline that is itself
    worse than doing nothing.

    **Substituting the window-mean null does not rescue it either**, and that is the deeper
    finding: against a window-mean null the same untrained ``DLinear`` removes **17.5%** of
    the null's error at a 2 s horizon on roll (skill -0.0432 against a null of -0.2639). A
    randomly initialised *linear map of the lookback* is not a null model. It is a bad
    filter of genuine past data, and a bad filter of the recent past of a narrowband signal
    carries real skill. The only initialisation for which this control is guaranteed to pass
    is one whose output is identically the window mean -- which passes by construction and
    therefore tests nothing.

    Every cell of this control is asserted on: the residual-floor narrowing that
    :func:`shuffle_control` applies is specific to the shuffle statistic, whose failing
    direction is what P6-D11 examined, and is not extended here on the strength of an
    argument made about a different control. Its ``asserted`` column is therefore True
    throughout.

    **``asserted`` and ``enforced`` are not the same column and this control is why.** Every
    row here is asserted on -- it counts toward the control's verdict -- and the sweep driver
    runs it with ``strict=False``, so every row also carries ``enforced=False``: the verdict
    is computed, written and read, and it stops nothing. Before that column existed, the only
    thing distinguishing this control from one that gates the run was the paragraph below,
    and a reader filtering the CSV for ``passed == False`` never sees a paragraph.

    So this control is reported rather than relied on for a linear architecture on this
    task. Run it with ``strict=False``, write the per-cell rows to
    ``results/baselines_controls.csv``, and read them next to the trained model's rows;
    the informative comparison is untrained-versus-trained, which the headline table already
    carries. Do not raise the tolerance until it passes -- that would be tuning a control to
    its subject. Leaving ``null_model`` None keeps the protocol's literal criterion, which
    holds at 1 s on every DOF and fails at 5 s on all three.

    Args:
        dataset: Test partition to score.
        untrained_model: A freshly constructed, unfitted model.
        persistence_model: The skill denominator, scored in the same pass over identical
            windows.
        regime: Regime label recorded on the output rows.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        null_model: Optional forecaster the untrained model must not beat. None means the
            null is persistence itself, i.e. the assertion is ``skill < tol``.
        tol: Largest fraction of the null's MSE the untrained model may remove,
            dimensionless. With ``null_model=None`` the null is persistence and this is a
            plain skill threshold.
        strict: If True, raise when the control fails.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The control's outcome, including per-cell rows for
        ``results/baselines_controls.csv``.

    Raises:
        AssertionError: If ``strict`` and the untrained model exceeds the null by ``tol`` or
            more anywhere.
    """
    models: dict[str, ForecastModel] = {
        "persistence": persistence_model,
        "untrained": untrained_model,
    }
    if null_model is not None:
        models["null"] = null_model
    table = _evaluate_control(
        models,
        dataset,
        persistence_key="persistence",
        horizons=horizons,
        fs_hz=fs_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    subject = _skill_by_cell(table, "untrained")
    null = None if null_model is None else _skill_by_cell(table, "null")
    null_name = "persistence" if null is None else "null"
    rows = _control_rows(
        "untrained",
        regime,
        "untrained",
        null_name,
        subject,
        null,
        tol,
        allow_equality=False,
        enforced=strict,
    )
    worst = float(rows["excess"].max())
    passed = bool(rows["passed"].all())
    if strict and not passed:
        worst_row = rows.loc[rows["excess"].idxmax()]
        raise AssertionError(
            f"untrained control failed: a randomly initialised model removes {worst:.2%} of "
            f"the null's error (tol={tol:.2%}) at DOF {worst_row['dof']!r}, horizon "
            f"{worst_row['horizon_samples']} samples -- skill {worst_row['skill_subject']:.4f} "
            f"against a null of {worst_row['skill_null']:.4f}. A model that has seen no data "
            f"must not beat a trivial forecast."
        )
    return ControlResult(
        control="untrained",
        rows=rows,
        table=table,
        worst_excess=worst,
        tol=tol,
        passed=passed,
        enforced=strict,
    )


def controls_table(results: Sequence[ControlResult]) -> pd.DataFrame:
    """Concatenate control outcomes into the frame written to ``baselines_controls.csv``.

    Args:
        results: Control outcomes, in the order they should appear.

    Returns:
        Frame with columns :data:`CONTROL_COLUMNS`.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("no control results to tabulate; a run with no controls is not audited")
    return pd.concat([result.rows for result in results], ignore_index=True)[list(CONTROL_COLUMNS)]


# =======================================================================================
# Interval controls -- the two the project has never run (Phase 6 carry-forward item 6).
#
# `shuffle_control` and `untrained_control` above are both **point** controls: their
# subject emits (B, H, C_out) and their statistic is a ratio of mean squared errors. No
# shuffle control has ever run on a *head*, and no untrained control on an *interval*, so
# every Phase 5 coverage claim rests on an unaudited path. The two functions below close
# that, and they share one null.
#
# **The null is the unconditional residual interval** (`docs/protocol.md` P6-D6):
# `EmpiricalResidualInterval`, a closed-form point forecast plus per-(horizon, channel)
# empirical residual quantiles taken from the validation split. Its width does not depend
# on the input at all, so a head beats it only by being **conditional** -- narrow where the
# deck is predictable, wide where it is not. A subject that cannot beat it has learned
# nothing conditional about its own uncertainty, whatever its PICP looks like in isolation.
# The null is typed as that class rather than as a general forecaster so that a caller
# cannot quietly substitute a kinder one, the same reason `shuffle_control` names its
# window-mean null in the signature.
#
# **The statistic is the same one the point controls use, on a proper scoring rule instead
# of on MSE**: `excess = 1 - loss_subject / loss_null`, i.e. the fraction of the null's loss
# the subject removes. One tolerance then means the same thing in every cell and on every
# metric, which a difference of Winkler scores -- an amount in degrees or metres -- would
# not.
#
# **The residual-floor narrowing of P6-D11 is NOT applied here, and that is a decision, not
# an omission.** See `interval_shuffle_control` for the derivation. The floored set is still
# computed and written, in the `on_residual_floor` column, so the question can be re-asked
# from the artifact rather than from a rerun.
# =======================================================================================

#: Scoring rules the interval controls are read on, in report order. All three are losses:
#: lower is better, so ``1 - subject/null`` is "the fraction of the null's loss the subject
#: removed" for each, exactly as for MSE in :func:`_control_rows`.
#:
#: ``pinball`` is reported and asserted on, and it is **not independent evidence**:
#: :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` accumulates CRPS as the
#: finite-fan quadrature of the pinball loss at the same levels for *every* head kind, so
#: ``crps = 2 * mean_q(pinball)`` identically and the two ``excess`` values are equal to
#: floating-point. It is here because the protocol requires pinball, CRPS, PICP and width to
#: be reported together, and ``tests/test_controls_interval.py`` pins the identity so that a
#: reader cannot mistake two columns for two measurements.
INTERVAL_CONTROL_METRICS: tuple[str, ...] = ("winkler", "crps", "pinball")

#: Column order of ``interval_controls.csv``. **A separate file from
#: ``baselines_controls.csv``**, not extra columns on it: an interval row's statistic is a
#: ratio of Winkler scores and a point row's is a ratio of squared errors, and CLAUDE.md
#: non-negotiable 6 forbids a table whose rows do not all mean the same thing.
#:
#: The frame is long on ``metric``: one row per (DOF, horizon, metric). ``picp_*`` and
#: ``width_*`` are repeated on every metric row of a cell so that **no row states a coverage
#: without the width beside it** -- coverage without sharpness is meaningless, and a
#: maximally wide interval has perfect coverage.
#:
#: ``asserted`` is True on every :func:`interval_untrained_control` row and on every
#: :func:`interval_shuffle_control` row whose **null** is calibrated. It is False exactly
#: where the shuffle control's null is itself outside :data:`INTERVAL_NULL_PICP_BAND`, and
#: the number that decision was taken on is ``picp_null``, on the same row -- so a reader
#: can recompute the narrowing from the CSV instead of trusting this paragraph. The
#: P6-D11/P6-D12 *floored-cell* narrowing is still not applied to this statistic;
#: ``on_residual_floor`` records that derivation anyway, so the question stays answerable
#: from the artifact.
#:
#: ``enforced`` is the same column :data:`CONTROL_COLUMNS` carries and answers the same
#: separate question: whether a failure raised. It is True for
#: :func:`interval_shuffle_control` and False for :func:`interval_untrained_control`, whose
#: ``strict`` defaults to False for a reason derived in its docstring. Before this column
#: existed that distinction lived only in prose, and :class:`IntervalControlResult` carried
#: it on the object while the CSV -- the artifact anybody downstream actually reads -- did
#: not.
INTERVAL_CONTROL_COLUMNS: tuple[str, ...] = (
    "control",
    "regime",
    "subject_model",
    "null_model",
    "dof",
    "horizon_samples",
    "horizon_s",
    "alpha",
    "metric",
    "loss_subject",
    "loss_null",
    "excess",
    "tol",
    "asserted",
    "enforced",
    "on_residual_floor",
    "passed",
    "picp_subject",
    "picp_null",
    "width_subject",
    "width_null",
    "crossing_rate_subject",
    "crossing_rate_null",
)

#: Largest fraction of the null's loss a shuffled interval may remove. **Calibrated from
#: a measurement of this statistic**, no longer inherited by analogy from the point control
#: (``docs/protocol.md`` P6-D21). The first production run scored 432 shuffle cells across
#: all four regimes -- ``results/e04/interval_shuffle_calibration.csv`` -- and its worst
#: **in-distribution** excess was **2.46 percent**, at ``pitch_rate`` / 150 samples on
#: ``id``. This 0.10 is a **4.1x margin** on that worst case, which mirrors P3-D8's
#: convention exactly: that entry measured 0.53 percent on the point statistic and chose
#: 2 percent, a 3.8x margin.
#:
#: **The value is post-hoc, and its legitimacy rests on P6-D16 assumption 1**, which
#: registered the first production run of this control as a calibration -- and recorded that
#: the inherited 0.02 had never been measured on a Winkler ratio -- *before* the control had
#: ever produced a number. Without that pre-registration this would be a tolerance widened
#: to make a failure go away, and it is recorded here as post-hoc rather than presented as
#: if it had always been the design.
#:
#: **Almost no power is given up.** Honest excesses run -60.7 to -373.2 at 1 s and are still
#: within [-1.06, +0.26] at 15 s, so a leaking subject would have to improve on the null by
#: two orders of magnitude relative to where an honest one sits before 0.10 rather than 0.02
#: changes a verdict. What *is* given up is stated rather than buried: at 10-15 s on the
#: least predictable channels neither subject nor null carries much information in its point
#: centre, so the statistic cannot separate "learned nothing" from "learned a little" there.
#: That is a property of the corpus at those leads (P3-D4), and no tolerance recovers it.
INTERVAL_SHUFFLE_TOL: float = 0.10

#: Band the **null's own** PICP@90 must fall inside before :func:`interval_shuffle_control`
#: asserts on a cell. **Not a new threshold**: it is :data:`dmf.eval.gate.GATE5_PICP_BAND`
#: itself, i.e. the ``[0.85, 0.95]`` band ``docs/IMPLEMENTATION_PLAN.md`` Phase 5 states and
#: P5-D2 pre-registered, bound to a second name only because it is applied here to the
#: *reference* rather than to a published head. Identity is asserted in
#: ``tests/test_controls_interval.py`` so the two cannot drift apart.
#:
#: **Why the null's calibration decides this and the subject's excess does not.** The
#: control's statistic is ``1 - loss_subject/loss_null`` on a Winkler or CRPS score. Both
#: objects carry a fan; where the null's fan is badly mis-sized for the partition it meets,
#: a *wider* subject covers better and scores better without knowing anything -- the
#: comparison then measures the reference rather than the subject. On the production corpus
#: this excludes every ``unseen_seastate`` cell, whose null covers 0.22 to 0.80 against a
#: nominal 0.90, and it excludes them **on that property alone**: the predicate reads
#: ``picp_null`` and never the regime name, the channel, or the subject's own excess.
#: See :func:`interval_shuffle_control` for what the narrowing costs.
INTERVAL_NULL_PICP_BAND: tuple[float, float] = GATE5_PICP_BAND

#: The untrained interval control's criterion, verbatim from the protocol's untrained
#: control: the subject must not beat the null at all. Kept at zero rather than widened,
#: for the P3-D9 reason -- a tolerance raised until a control passes is a control tuned to
#: its subject. The control is reported rather than enforced instead; see
#: :func:`interval_untrained_control`.
INTERVAL_UNTRAINED_TOL: float = 0.0

#: Metric columns of the probabilistic table, keyed by the name used in ``metric``.
_INTERVAL_METRIC_COLUMNS: Mapping[str, str] = {
    "winkler": "winkler",
    "crps": "crps",
    "pinball": "pinball",
}


@dataclass(frozen=True)
class IntervalControlResult:
    """Outcome of an interval control scored against the unconditional residual interval.

    The interval twin of :class:`ControlResult`, kept as its own type because its rows
    carry a different statistic on a different schema; :func:`interval_controls_table`
    concatenates these and :func:`controls_table` concatenates the point ones, and the two
    are never merged.

    Attributes:
        control: Control name, ``"interval_shuffle"`` or ``"interval_untrained"``.
        rows: One row per (DOF, horizon, metric), columns
            :data:`INTERVAL_CONTROL_COLUMNS`. Every cell scored is present, including any
            the assertion skipped.
        table: The raw :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` output
            for the subject and the null, kept so the control's own numbers are auditable
            rather than summarised away.
        worst_excess: Largest ``excess`` over the **asserted** cells, dimensionless.
            Negative means the control behaved as predicted -- the subject did not beat the
            null. NaN when no cell was asserted on, which is not a pass and must not read as
            one; :attr:`worst_excess_reported_only` carries the rest.
        tol: Tolerance the excess was compared against.
        passed: Whether every asserted cell is within tolerance. True with zero asserted
            cells means the control had nothing to judge -- read it with
            :attr:`exclusion_reason`, which is non-empty exactly then.
        enforced: Whether a failure raised. False means the control is **reported rather
            than enforced**; :attr:`passed` still says what it found.
        floored_dofs: Target channels the P1-D2 derivation marks as sitting on their
            residual floor in this partition. **Reported, not excluded** -- they are
            asserted on like every other channel.
        excluded_cells: ``(dof, horizon_samples)`` pairs reported but not asserted on,
            sorted. Empty for a control that asserted everywhere, and always empty for
            :func:`interval_untrained_control`.
        exclusion_reason: Why :attr:`excluded_cells` were excluded, for the message a strict
            failure prints and for anyone reading the object. Empty when nothing was
            excluded.
    """

    control: str
    rows: pd.DataFrame
    table: pd.DataFrame
    worst_excess: float
    tol: float
    passed: bool
    enforced: bool
    floored_dofs: tuple[str, ...] = ()
    excluded_cells: tuple[tuple[str, int], ...] = ()
    exclusion_reason: str = ""

    @property
    def asserted_rows(self) -> pd.DataFrame:
        """Return the rows the pass/fail decision was taken over.

        Returns:
            The subset of :attr:`rows` with ``asserted`` True.
        """
        return self.rows[self.rows["asserted"]]

    @property
    def reported_only_rows(self) -> pd.DataFrame:
        """Return the rows computed and reported but not asserted on.

        Kept reachable for the same reason :attr:`ControlResult.reported_only_rows` is: a
        cell the control declines to assert on is a finding, not a gap, and it ships with
        its real numbers (CLAUDE.md non-negotiable 6).

        Returns:
            The subset of :attr:`rows` with ``asserted`` False.
        """
        return self.rows[~self.rows["asserted"]]

    @property
    def worst_excess_reported_only(self) -> float | None:
        """Return the worst excess among the reported-only cells.

        Returns:
            The largest ``excess`` over cells the assertion skipped, or None if there were
            none. Never compared against ``tol`` by this class; it exists to be read.
        """
        rows = self.reported_only_rows
        return None if rows.empty else float(rows["excess"].max())

    @property
    def worst_row(self) -> "pd.Series[Any]":
        """Return the asserted row with the largest ``excess``.

        Returns:
            The single worst cell, for a failure message or a report line.

        Raises:
            ValueError: If no row was asserted on, which :func:`_interval_control_rows`
                does not produce.
        """
        asserted = self.rows[self.rows["asserted"]]
        if asserted.empty:
            raise ValueError(f"{self.control} asserted on no row")
        return asserted.loc[[asserted["excess"].idxmax()]].iloc[0]

    def excess_by_metric(self) -> dict[str, float]:
        """Return the worst asserted ``excess`` for each scoring rule.

        Reported per metric because Winkler and CRPS answer different questions -- Winkler
        prices an interval, CRPS prices the whole fan -- and a subject that beats the null
        on one and not the other is a finding rather than a summary statistic.

        Returns:
            Metric name -> largest ``excess`` over the asserted rows carrying it.
        """
        asserted = self.rows[self.rows["asserted"]]
        return {
            str(metric): float(group["excess"].max())
            for metric, group in asserted.groupby("metric", sort=False)
        }


def _interval_metric_frame(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Extract one model's per-(DOF, horizon) probabilistic scores from a runner table.

    Args:
        table: Output of :func:`dmf.eval.prob_runner.evaluate_probabilistic_models`.
        model: Model key to extract.

    Returns:
        Frame indexed by ``(dof, horizon_samples)`` carrying the scoring rules, the
        coverage, the width and the crossing rate.

    Raises:
        ValueError: If ``model`` has no rows in ``table``.
    """
    wanted = [
        "dof",
        "horizon_samples",
        "horizon_s",
        "alpha",
        "picp",
        "mean_interval_width",
        "crossing_rate",
        *_INTERVAL_METRIC_COLUMNS.values(),
    ]
    rows = table.loc[table["model"] == model, wanted]
    if rows.empty:
        raise ValueError(f"model {model!r} has no rows in the probabilistic evaluation table")
    return rows.set_index(["dof", "horizon_samples"], drop=False)


def _interval_control_rows(
    control: str,
    regime: str,
    subject: str,
    null: str,
    subject_scores: pd.DataFrame,
    null_scores: pd.DataFrame,
    tol: float,
    floored: Collection[str],
    *,
    allow_equality: bool,
    enforced: bool,
    null_picp_band: tuple[float, float] | None,
) -> pd.DataFrame:
    """Assemble the CSV rows for one interval control, including its pass statistic.

    ``excess = 1 - loss_subject / loss_null`` per scoring rule, the same ratio
    :func:`_control_rows` forms from mean squared errors and for the same reason: a
    difference of Winkler scores is an amount in degrees or metres and would need a
    different tolerance in every cell.

    Args:
        control: Control name.
        regime: Regime the control was run on.
        subject: Label of the model under test.
        null: Label of the unconditional interval it is compared against.
        subject_scores: Subject's scores, from :func:`_interval_metric_frame`.
        null_scores: Null's scores, same index.
        tol: Largest tolerated fraction of the null's loss the subject may remove.
        floored: Target channels the P1-D2 derivation marks as sitting on their residual
            floor. **Recorded, not excluded**: these rows are asserted on like every other.
        allow_equality: If True the control passes at ``excess == tol`` exactly, which an
            identical subject and null require.
        enforced: Whether a failure of this control raises, i.e. its ``strict`` argument.
        null_picp_band: Inclusive band the **null's** measured PICP must fall inside for a
            cell to be asserted on, or None to assert on every cell. The predicate reads
            ``null_scores["picp"]`` and nothing else -- not the regime, not the channel, and
            not the subject's own excess -- so it cannot be reached by naming a cell that
            failed. See :data:`INTERVAL_NULL_PICP_BAND`.

    Returns:
        Frame with columns :data:`INTERVAL_CONTROL_COLUMNS`, long on ``metric``. Rows the
        band excludes are present, carry their real ``excess`` and their own ``passed``,
        and differ only in ``asserted``.

    Raises:
        ValueError: If the two models were not scored on the same cells, if the null's
            loss is not positive somewhere -- a zero-loss null makes the ratio undefined,
            and for a Winkler score it would mean a zero-width interval that never missed --
            or if a band was given and the null's PICP is not finite somewhere, which would
            silently drop the cell out of the assertion.
    """
    if list(subject_scores.index) != list(null_scores.index):
        raise ValueError(
            "the subject and the null were scored on different (DOF, horizon) cells; they "
            "must be scored in one pass over identical windows"
        )
    picp_null = null_scores["picp"].to_numpy(dtype=np.float64)
    if null_picp_band is None:
        asserted = np.ones(len(picp_null), dtype=bool)
    else:
        if not np.all(np.isfinite(picp_null)):
            worst = int(np.argmax(~np.isfinite(picp_null)))
            raise ValueError(
                f"the null's PICP is {picp_null[worst]!r} at DOF "
                f"{null_scores['dof'].to_numpy()[worst]!r}, horizon "
                f"{null_scores['horizon_samples'].to_numpy()[worst]}, so whether its "
                f"calibration is inside {null_picp_band} cannot be decided. A non-finite "
                f"coverage would silently make the cell unasserted, i.e. narrow an integrity "
                f"control by accident"
            )
        low, high = null_picp_band
        asserted = (picp_null >= low) & (picp_null <= high)
    frames: list[pd.DataFrame] = []
    for metric, column in _INTERVAL_METRIC_COLUMNS.items():
        loss_subject = subject_scores[column].to_numpy(dtype=np.float64)
        loss_null = null_scores[column].to_numpy(dtype=np.float64)
        if not np.all(loss_null > 0.0):
            worst = int(np.argmin(loss_null))
            raise ValueError(
                f"the null's {metric} is {loss_null[worst]:.6g} at DOF "
                f"{subject_scores['dof'].to_numpy()[worst]!r}, so the control statistic "
                f"1 - loss_subject/loss_null is undefined. A non-positive proper score "
                f"means the null emitted a zero-width interval it never missed, which is "
                f"an unfitted EmpiricalResidualInterval rather than a baseline"
            )
        excess = 1.0 - loss_subject / loss_null
        dofs = subject_scores["dof"].to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "control": control,
                    "regime": regime,
                    "subject_model": subject,
                    "null_model": null,
                    "dof": dofs,
                    "horizon_samples": subject_scores["horizon_samples"].to_numpy(),
                    "horizon_s": subject_scores["horizon_s"].to_numpy(),
                    "alpha": subject_scores["alpha"].to_numpy(),
                    "metric": metric,
                    "loss_subject": loss_subject,
                    "loss_null": loss_null,
                    "excess": excess,
                    "tol": tol,
                    # False only where `null_picp_band` says the NULL is miscalibrated
                    # (P6-D21). Per cell, from the null's own coverage, which is on the row
                    # as `picp_null`. The P6-D11 floored-cell narrowing is still not applied
                    # to this statistic; see `interval_shuffle_control`.
                    "asserted": asserted,
                    # Constant per control, carried per row anyway: the CSV is filtered and
                    # grouped downstream, and a fact that survives only in the object is a
                    # fact the reader of the artifact does not have.
                    "enforced": enforced,
                    "on_residual_floor": np.array(
                        [str(dof) in set(floored) for dof in dofs], dtype=bool
                    ),
                    "passed": excess <= tol if allow_equality else excess < tol,
                    "picp_subject": subject_scores["picp"].to_numpy(),
                    "picp_null": null_scores["picp"].to_numpy(),
                    "width_subject": subject_scores["mean_interval_width"].to_numpy(),
                    "width_null": null_scores["mean_interval_width"].to_numpy(),
                    "crossing_rate_subject": subject_scores["crossing_rate"].to_numpy(),
                    "crossing_rate_null": null_scores["crossing_rate"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[list(INTERVAL_CONTROL_COLUMNS)]


def _evaluate_interval_control(
    subject_model: BaseForecaster,
    null_model: EmpiricalResidualInterval,
    dataset: DeckMotionDataset,
    *,
    subject_label: str,
    null_label: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    alpha: float,
    batch_size: int,
    num_workers: int,
    device: str,
) -> pd.DataFrame:
    """Score a subject and the unconditional interval through the Phase 5 scorer.

    One pass over identical windows, through
    :func:`dmf.eval.prob_runner.evaluate_probabilistic_models` -- the same function the
    published ``probabilistic.csv`` rows are produced by, so the control's numbers and the
    headline table's are the same quantity.

    Args:
        subject_model: The model under test. Must carry a non-point head.
        null_model: The fitted unconditional residual interval.
        dataset: The partition to score.
        subject_label: Results label for the subject.
        null_label: Results label for the null.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        alpha: Nominal miscoverage; 0.1 is the PICP@90 the project reads.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The runner's table, both models.

    Raises:
        ValueError: If the null has not been fitted -- an unfitted
            :class:`dmf.models.residual_interval.EmpiricalResidualInterval` emits a
            zero-width fan, which scores as perfect sharpness and would make every subject
            look like a leak.
    """
    if not bool(null_model.is_fitted):
        raise ValueError(
            "the null EmpiricalResidualInterval has no residual quantiles. Its fan would be "
            "zero-width, scoring as perfect sharpness, and every subject would appear to "
            "remove 100% of its loss. Fit it with dmf.train.closed_form.fit_residual_interval "
            "on the VALIDATION split first (docs/protocol.md P6-D6)"
        )
    table, _ = evaluate_probabilistic_models(
        {subject_label: subject_model, null_label: null_model},
        dataset,
        horizons=horizons,
        fs_hz=fs_hz,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        n_boot=1,
    )
    return table


def interval_shuffle_control(
    dataset: DeckMotionDataset,
    *,
    shuffled_model: BaseForecaster,
    null_model: EmpiricalResidualInterval,
    regime: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    alpha: float = 0.1,
    tol: float = INTERVAL_SHUFFLE_TOL,
    strict: bool = True,
    subject_label: str = "shuffled_interval",
    null_label: str = "residual_interval",
    floor_margin: float = RESIDUAL_FLOOR_MARGIN,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> IntervalControlResult:
    """Check that an interval refitted on time-shuffled targets learned no uncertainty.

    The first shuffle control in this project to run on a **head** rather than on a point
    model (Phase 6 carry-forward item 6). The subject is normally an
    :class:`dmf.models.residual_interval.EmpiricalResidualInterval` refitted end to end on
    time-shuffled targets -- shuffled moments for the point half, and residual quantiles
    taken against the same shuffled targets -- so both halves of the object have had every
    genuine input-to-future relationship destroyed. Any quantile head refitted that way may
    be passed instead; the statistic does not care which.

    The null is the honestly fitted unconditional interval (P6-D6). Passing means the
    shuffled interval prices its own uncertainty no better than an interval that knows only
    the marginal residual distribution. Failing means information survived the shuffle,
    which on the interval path can only be leakage.

    **Why the null is not "zero skill", and why it is not persistence.** The same argument
    the module docstring makes for the point control: a least-squares fit on shuffled
    targets degenerates to the conditional mean, so the shuffled subject's *point* half is
    the window-mean forecast and its residual fan is the fan of window-mean residuals --
    wider than the honest one, and **calibrated to its own larger errors**. That last clause
    is what makes the comparison well posed. A Winkler score is not monotone in width: at
    ``alpha = 0.1`` a fan narrower than its residuals pays ``2/alpha`` times every miss and a
    wider one pays its own width, so there is an optimum, and "the shuffled interval is
    wider, therefore worse" is false in general. Both objects here are calibrated to their
    own residuals on the same validation split, and among calibrated intervals the score is
    monotone in the residual scale -- which is the quantity the shuffle destroyed. The
    unconditional interval is therefore the reference that makes the comparison mean
    something, because it is the same object with the shuffle removed.

    **The control does not assert where the NULL is itself miscalibrated**
    (``docs/protocol.md`` P6-D21). A cell is asserted on only if the null's own measured
    PICP@90 falls inside :data:`INTERVAL_NULL_PICP_BAND`, which *is* Gate 5's registered
    ``[0.85, 0.95]`` band (P5-D2) applied to the reference instead of to a head.

    - **Why.** The statistic is a ratio of two fans' scores. Where the null's fan is badly
      mis-sized for the partition it meets, a *wider* subject covers better and scores
      better without knowing anything: on the production corpus the shuffled subject exceeds
      the old tolerance on 12 ``unseen_seastate`` cells whose null covers 0.22 to 0.80
      against a nominal 0.90, and it wins there **by being wider, not by knowing something**.
      A control whose reference is miscalibrated measures the reference.
    - **What it keys off.** ``picp_null``, per cell, and nothing else -- not the regime, not
      the channel, not the subject's excess -- so no failing cell can be excluded by being
      named. The excluded rows are written with their real ``excess`` and their own
      ``passed``; only ``asserted`` is False, exactly as P6-D12 does for the point control's
      floored cells.
    - **What it costs, stated rather than buried.** On the production corpus this leaves
      ``id`` fully asserted (108/108 cells) but drops **every** ``unseen_seastate`` cell and
      most ``unseen_heading`` ones, so on those regimes the control is reported and cannot
      fail. That is a genuine loss of coverage for the guard, not a fix, and it is the same
      finding the coverage tables report: the unconditional interval does not transfer to an
      unseen sea state. When a regime ends up with no asserted cell at all the control emits
      a :class:`RuntimeWarning` saying so and :attr:`IntervalControlResult.worst_excess` is
      NaN, because "passed with nothing to judge" must not read like a pass.

    **The P6-D11 residual-floor narrowing is still deliberately NOT applied.** P6-D12 recorded
    that extending that argument to a different statistic on inference alone is exactly what
    was refused for :func:`untrained_control`, so the question is answered from the
    statistic rather than by analogy:

    - P6-D11's mechanism is an interaction inside a **mean squared error**. On a floored
      channel the shuffled point model carries an amplitude fitted where the DOF is excited
      and imposes it on a test target that has none; the resulting MSE ratio moves with that
      mismatch and reverses sign with lookback.
    - The statistic here is a ratio of **Winkler or CRPS scores**, and on exactly the cells
      the floor names it degenerates to a ratio of two constants. A floored test channel is
      ~26 dB down, so both fans -- each fitted on the *validation* split, where the channel
      is excited -- are far wider than the residuals they meet, coverage goes to 1, the
      Winkler penalty term vanishes and the score reduces to the interval width, which is a
      fixed number per (DOF, horizon) that no property of the test set can move. Two
      constants fitted on the same unfloored split cannot produce a sign flip, so the
      mechanism has no route into this ratio.
    - **That is a derivation, not a measurement.** It has not been measured at production
      scale, which is the second reason not to narrow: leaving a control at full scope is
      never a hole in the guard, and if a floored cell does fire, the right response is the
      six-measurement investigation P6-D11 ran, not a pre-emptive exclusion. The floored set
      is still derived and written to the ``on_residual_floor`` column, and the failure
      message names it, so that investigation starts from the artifact.

    Args:
        dataset: Test partition to score.
        shuffled_model: The head refitted on time-shuffled training targets.
        null_model: The fitted unconditional residual interval, scored in the same pass so
            the comparison uses identical windows.
        regime: Regime label recorded on the output rows.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        alpha: Nominal miscoverage of the interval the coverage and Winkler columns
            describe. 0.1 is PICP@90.
        tol: Largest fraction of the null's loss the shuffled interval may remove before
            the control fails, dimensionless. See :data:`INTERVAL_SHUFFLE_TOL` -- 0.10,
            calibrated post-hoc from a measured 2.46 percent in-distribution worst case at a
            4.1x margin, under a pre-registration (P6-D16 assumption 1) that this control's
            first production run was a calibration.
        strict: If True, raise when the control fails. A failed integrity control
            invalidates every interval claim in the run, so stopping loudly is the default.
        subject_label: Results label for the subject.
        null_label: Results label for the null.
        floor_margin: Multiple of the P1-D2 residual floor a DOF's excitation may reach and
            still be recorded as floored. Reporting only; see above.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The control's outcome, including per-cell rows for ``interval_controls.csv``.

    Raises:
        AssertionError: If ``strict`` and the shuffled interval beats the unconditional one
            by more than ``tol`` on a cell whose null is calibrated.
        ValueError: If the null is unfitted, if the subject carries a point head, if the
            null's coverage is not finite somewhere, or if ``alpha`` is not the nominal
            level :data:`INTERVAL_NULL_PICP_BAND` was registered at.
    """
    nominal = 1.0 - alpha
    if abs(nominal - GATE5_NOMINAL) > 1e-9:
        # The band is Gate 5's, and Gate 5's is a statement about PICP@90 specifically.
        # Reading a nominal-80 percent interval against [0.85, 0.95] would mark a perfectly
        # calibrated null as miscalibrated and silently narrow the control to nothing, which
        # is the failure mode this whole entry exists to avoid. Refuse instead.
        raise ValueError(
            f"alpha={alpha} makes this a nominal {nominal:.0%} interval, but the "
            f"assertion band {INTERVAL_NULL_PICP_BAND} is Gate 5's PICP@"
            f"{GATE5_NOMINAL:.0%} band (docs/protocol.md P5-D2, P6-D21). Judging a "
            f"{nominal:.0%} interval against it would call a calibrated null miscalibrated "
            f"and quietly leave the control asserting on nothing"
        )
    table = _evaluate_interval_control(
        shuffled_model,
        null_model,
        dataset,
        subject_label=subject_label,
        null_label=null_label,
        horizons=horizons,
        fs_hz=fs_hz,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    floored = dataset_floored_dofs(dataset, margin=floor_margin)
    rows = _interval_control_rows(
        "interval_shuffle",
        regime,
        subject_label,
        null_label,
        _interval_metric_frame(table, subject_label),
        _interval_metric_frame(table, null_label),
        tol,
        floored,
        allow_equality=True,
        enforced=strict,
        # The one narrowing this statistic takes, and it is a property of the NULL: see the
        # docstring above and docs/protocol.md P6-D21.
        null_picp_band=INTERVAL_NULL_PICP_BAND,
    )
    asserted = rows[rows["asserted"]]
    excluded_cells = tuple(
        sorted(
            {
                (str(dof), int(horizon))
                for dof, horizon in zip(
                    rows.loc[~rows["asserted"], "dof"],
                    rows.loc[~rows["asserted"], "horizon_samples"],
                    strict=True,
                )
            }
        )
    )
    low, high = INTERVAL_NULL_PICP_BAND
    reason = (
        ""
        if not excluded_cells
        else (
            f"{len(excluded_cells)} of {len(rows) // len(_INTERVAL_METRIC_COLUMNS)} "
            f"(DOF, horizon) cell(s) have a null whose own PICP@90 is outside Gate 5's "
            f"registered band [{low}, {high}] -- measured range "
            f"[{rows.loc[~rows['asserted'], 'picp_null'].min():.4f}, "
            f"{rows.loc[~rows['asserted'], 'picp_null'].max():.4f}] -- so the statistic there "
            f"prices the reference rather than the subject: a wider subject covers a "
            f"mis-sized null's misses without knowing anything (docs/protocol.md P6-D21). "
            f"Their rows are in the table with asserted=False and their real excess."
        )
    )
    if excluded_cells:
        # Emitted on a PASS as well as a fail, for P6-D12's reason: a narrowing of an
        # integrity control that nobody sees is indistinguishable from a control that was
        # never run, and the CSV column is for the reader afterwards rather than for the
        # person watching the sweep.
        warnings.warn(
            f"interval shuffle control on regime {regime!r}: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
    if asserted.empty:
        # NOT an error, and deliberately not a silent pass either. On `unseen_seastate` the
        # unconditional null is miscalibrated in every cell, which is itself a published
        # finding; raising here would abort the evaluation over a property of the reference,
        # and returning a quiet `passed=True` would let a vacuous control read like a
        # verdict. So: warn, and hand back NaN as the worst excess.
        warnings.warn(
            f"interval shuffle control on regime {regime!r} asserted on NO cell: every "
            f"cell's null is outside [{low}, {high}]. The control ran and its rows are "
            f"reported, but it could not judge this regime, and `passed` here means "
            f"'nothing was judged' rather than 'the subject learned nothing'.",
            RuntimeWarning,
            stacklevel=2,
        )
    worst = float("nan") if asserted.empty else float(asserted["excess"].max())
    passed = bool(asserted["passed"].all())
    if strict and not passed:
        worst_row = asserted.loc[asserted["excess"].idxmax()]
        raise AssertionError(
            f"interval shuffle control failed: an interval refitted on time-shuffled "
            f"targets removes {worst:.2%} of the unconditional interval's "
            f"{worst_row['metric']} (tol={tol:.2%}) at DOF {worst_row['dof']!r}, horizon "
            f"{worst_row['horizon_samples']} samples ({worst_row['loss_subject']:.6g} "
            f"against {worst_row['loss_null']:.6g}). Destroying the time ordering of the "
            f"targets must destroy the conditional structure of the residuals too; if it "
            f"does not, information is reaching the head by some path other than the "
            f"inputs."
            + (
                " This cell is on its P1-D2 residual floor (on_residual_floor=True). The "
                "floored-cell narrowing of docs/protocol.md P6-D11 was NOT extended to "
                "this statistic -- see interval_shuffle_control's docstring for why -- so "
                "the first thing to check is whether that derivation is wrong, and the "
                "second is whether the failure also appears on an unfloored channel."
                if bool(worst_row["on_residual_floor"])
                else ""
            )
            + (f" Reported but not asserted on: {reason}" if excluded_cells else "")
        )
    return IntervalControlResult(
        control="interval_shuffle",
        rows=rows,
        table=table,
        worst_excess=worst,
        tol=tol,
        passed=passed,
        enforced=strict,
        floored_dofs=tuple(sorted(floored)),
        excluded_cells=excluded_cells,
        exclusion_reason=reason,
    )


def interval_untrained_control(
    dataset: DeckMotionDataset,
    *,
    untrained_model: BaseForecaster,
    null_model: EmpiricalResidualInterval,
    regime: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    alpha: float = 0.1,
    tol: float = INTERVAL_UNTRAINED_TOL,
    strict: bool = False,
    subject_label: str = "untrained_head",
    null_label: str = "residual_interval",
    floor_margin: float = RESIDUAL_FLOOR_MARGIN,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> IntervalControlResult:
    """Check that a randomly initialised head does not price uncertainty better than nothing.

    The first untrained control in this project to run on an **interval** rather than on a
    point forecast (Phase 6 carry-forward item 6). The subject is a freshly constructed,
    unfitted quantile or Gaussian head; the null is the fitted unconditional residual
    interval (P6-D6).

    **This control is reported, not enforced, and the default ``strict`` is False.** That
    follows the P3-D9 precedent for :func:`untrained_control`, and the reason it is applied
    here at the *definition* site rather than left to the caller is that the failing
    direction is analysed here rather than discovered in a sweep:

    - A random-init head emits a near-degenerate fan whose width is an artifact of the
      initialisation scale and carries no relation to the error it will make. On a channel
      with real amplitude that is catastrophic -- the Winkler penalty is ``2/alpha`` times
      the miss distance, so a narrow fan that misses is priced at 20x its error at
      ``alpha = 0.1`` -- and the control passes easily.
    - **On a channel sitting on its P1-D2 residual floor it inverts.** There the test target
      is ~26 dB down and close to the window mean, so a near-zero-width fan centred near
      zero is almost never wrong by much, while the null's fan -- fitted on the validation
      split, where the channel is excited -- is wide and pays its full width on every
      window. The untrained head then removes a large fraction of the null's loss without
      having seen a single datum. That is the amplitude-mismatch mechanism of P6-D11
      arriving in the untrained control, which is precisely the risk P6-D12 recorded when it
      declined to narrow :func:`untrained_control`: *"a future caller passing ``strict=True``
      would hit it in a control whose failing direction nobody has analysed"*. It is
      analysed now, and the answer is that the criterion is wrong on those cells rather than
      that those cells should be hidden.
    - So the treatment is P3-D9's: keep the literal criterion, keep the tolerance at zero,
      report every cell, and do not raise. **Do not raise the tolerance until it passes** --
      that would be tuning a control to its subject -- and do not narrow the scope either,
      which would be tuning it to a channel. A failure with ``strict=False`` still emits a
      :class:`RuntimeWarning`, because a control whose result nobody sees is a control that
      was not run.

    The informative reading is untrained-versus-fitted on the same cell, which
    ``interval_controls.csv`` and ``probabilistic.csv`` together already carry.

    Args:
        dataset: Test partition to score.
        untrained_model: A freshly constructed, unfitted head. Must carry a non-point head.
        null_model: The fitted unconditional residual interval, scored in the same pass.
        regime: Regime label recorded on the output rows.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        alpha: Nominal miscoverage. 0.1 is PICP@90.
        tol: Largest fraction of the null's loss the untrained head may remove. Zero is the
            protocol's literal criterion; see :data:`INTERVAL_UNTRAINED_TOL`.
        strict: If True, raise on failure. **Default False**, and the default is the
            decision -- see above.
        subject_label: Results label for the subject.
        null_label: Results label for the null.
        floor_margin: Multiple of the P1-D2 residual floor a DOF's excitation may reach and
            still be recorded as floored. Reporting only: this control asserts on every
            cell, which is what P6-D12 requires of an untrained control.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The control's outcome, including per-cell rows for ``interval_controls.csv``.

    Raises:
        AssertionError: If ``strict`` and the untrained head removes ``tol`` or more of the
            null's loss anywhere.
        ValueError: If the null is unfitted, or if the subject carries a point head.
    """
    table = _evaluate_interval_control(
        untrained_model,
        null_model,
        dataset,
        subject_label=subject_label,
        null_label=null_label,
        horizons=horizons,
        fs_hz=fs_hz,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    floored = dataset_floored_dofs(dataset, margin=floor_margin)
    rows = _interval_control_rows(
        "interval_untrained",
        regime,
        subject_label,
        null_label,
        _interval_metric_frame(table, subject_label),
        _interval_metric_frame(table, null_label),
        tol,
        floored,
        allow_equality=False,
        enforced=strict,
        # NOT narrowed, on either criterion. P6-D12 declined to extend the point control's
        # floored-cell narrowing to an untrained statistic on inference alone, and the same
        # refusal applies to P6-D21's null-calibration narrowing: this control is reported
        # rather than enforced (P3-D9), so a miscalibrated null makes a row harder to read,
        # not a run harder to stop. Every cell is asserted on and every cell ships.
        null_picp_band=None,
    )
    worst = float(rows["excess"].max())
    passed = bool(rows["passed"].all())
    if not passed:
        worst_row = rows.loc[rows["excess"].idxmax()]
        message = (
            f"untrained interval control on regime {regime!r}: a randomly initialised head "
            f"removes {worst:.2%} of the unconditional interval's {worst_row['metric']} "
            f"(tol={tol:.2%}) at DOF {worst_row['dof']!r}, horizon "
            f"{worst_row['horizon_samples']} samples "
            f"(on_residual_floor={bool(worst_row['on_residual_floor'])}). A model that has "
            f"seen no data must not price uncertainty better than the marginal residual "
            f"distribution does."
        )
        if strict:
            raise AssertionError(message)
        # Warned even though the control is not enforced: P3-D9 makes this outcome a
        # reported finding, and a finding nobody sees during a 46 h sweep is indistinguishable
        # from a control that was never run.
        warnings.warn(f"{message} Reported, not enforced (P3-D9).", RuntimeWarning, stacklevel=2)
    return IntervalControlResult(
        control="interval_untrained",
        rows=rows,
        table=table,
        worst_excess=worst,
        tol=tol,
        passed=passed,
        enforced=strict,
        floored_dofs=tuple(sorted(floored)),
    )


def interval_controls_table(results: Sequence[IntervalControlResult]) -> pd.DataFrame:
    """Concatenate interval control outcomes into ``interval_controls.csv``.

    Args:
        results: Control outcomes, in the order they should appear.

    Returns:
        Frame with columns :data:`INTERVAL_CONTROL_COLUMNS`.

    Raises:
        ValueError: If ``results`` is empty -- a run with no controls is not audited.
    """
    if not results:
        raise ValueError(
            "no interval control results to tabulate; a run that makes an interval claim "
            "without an interval control is not audited (Phase 6 carry-forward item 6)"
        )
    return pd.concat([result.rows for result in results], ignore_index=True)[
        list(INTERVAL_CONTROL_COLUMNS)
    ]
