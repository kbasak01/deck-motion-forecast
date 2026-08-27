"""Integrity controls -- the checks run before believing any result.

Three controls guard this project, and none of them is a unit test of a function; each is a
whole-pipeline experiment whose expected outcome is known in advance:

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
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import invert_norm
from dmf.data.splits import RealizationKey
from dmf.data.windows import window_start_indices
from dmf.eval.runner import evaluate_models
from dmf.models.base import ForecastModel
from dmf.sim.generate import RealizationSpec, realization_path
from dmf.typedefs import FloatArray

__all__ = [
    "CONTROL_COLUMNS",
    "ControlResult",
    "PipelineSanityResult",
    "controls_table",
    "persistence_pipeline_sanity",
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
    "passed",
)


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
            frame written to ``results/baselines_controls.csv``.
        table: The raw output of :func:`dmf.eval.runner.evaluate_models` for every model the
            control scored, kept so the control's own numbers are auditable rather than
            summarised away.
        worst_excess: Largest value of ``skill_subject - skill_null`` over the reported
            cells, dimensionless. Negative means the control behaved as predicted.
        tol: Tolerance the excess was compared against.
        passed: Whether the control passed.
    """

    control: str
    rows: pd.DataFrame
    table: pd.DataFrame
    worst_excess: float
    tol: float
    passed: bool


def _raw_persistence_sse(dataset: DeckMotionDataset, corpus_root: Path) -> tuple[FloatArray, int]:
    """Accumulate persistence squared error directly from the Parquet files.

    Deliberately shares nothing with :class:`DeckMotionDataset` except the realization key
    list and the window geometry: it re-reads the files with pandas, recomputes the start
    indices from :func:`dmf.data.windows.window_start_indices`, and uses no normalisation
    and no torch. If the dataset's internal index arithmetic were off by one, this path
    would not follow it.

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
    for key in dataset.realization_keys:
        frame = pd.read_parquet(corpus_root / _path_for(key), columns=columns)
        series = frame[columns].to_numpy(dtype=np.float64)
        starts = window_start_indices(series.shape[0], spec)
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
    frame = pd.DataFrame(
        {
            "control": control,
            "regime": regime,
            "subject_model": subject,
            "null_model": null,
            "dof": subject_skill["dof"].to_numpy(),
            "horizon_samples": subject_skill["horizon_samples"].to_numpy(),
            "horizon_s": subject_skill["horizon_s"].to_numpy(),
            "skill_subject": subject_values,
            "skill_null": null_values,
            "excess": excess,
            "tol": tol,
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
        strict: If True, raise when the control fails. A failed integrity control
            invalidates the headline table, so stopping loudly is the default.
        batch_size: Windows per batch.
        num_workers: DataLoader worker processes.
        device: Torch device.

    Returns:
        The control's outcome, including per-cell rows for
        ``results/baselines_controls.csv``.

    Raises:
        AssertionError: If ``strict`` and the shuffled model beats the window-mean null by
            more than ``tol`` anywhere.
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
    rows = _control_rows(
        "shuffle", regime, "shuffled", "window_mean", subject, null, tol, allow_equality=True
    )
    worst = float(rows["excess"].max())
    passed = bool(rows["passed"].all())
    if strict and not passed:
        worst_row = rows.loc[rows["excess"].idxmax()]
        raise AssertionError(
            f"shuffle control failed: a model trained on time-shuffled targets removes "
            f"{worst:.2%} of the window-mean null's error (tol={tol:.2%}) at DOF "
            f"{worst_row['dof']!r}, horizon {worst_row['horizon_samples']} samples "
            f"(shuffled skill {worst_row['skill_subject']:.4f} vs window mean "
            f"{worst_row['skill_null']:.4f}). Destroying the time ordering of the targets "
            f"must destroy the signal; if it does not, information is reaching the model "
            f"by some path other than the inputs."
        )
    return ControlResult(
        control="shuffle", rows=rows, table=table, worst_excess=worst, tol=tol, passed=passed
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
        "untrained", regime, "untrained", null_name, subject, null, tol, allow_equality=False
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
        control="untrained", rows=rows, table=table, worst_excess=worst, tol=tol, passed=passed
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
