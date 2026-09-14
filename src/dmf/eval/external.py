"""Scoring trained checkpoints on an externally supplied trajectory.

This is the seam Phase 8 needs and the corpus scoring path cannot provide. An MSS record is
a CSV of time series, not a :class:`dmf.data.splits.Split` of Parquet realizations, so
:func:`dmf.eval.runner.evaluate_models` -- which takes a
:class:`dmf.data.dataset.DeckMotionDataset` and derives its geometry, its statistics and
its realization keys from it -- cannot be pointed at one. Everything *downstream* of the
dataset is reused unchanged: the windowing, the normalisation, the inverse transform and
the metrics table are the same functions the committed corpus tables came out of, so an
external row and a corpus row are the same quantity computed the same way.

Four properties are enforced here rather than documented, because each of them is silent in
the output if it is got wrong:

**The scale is the corpus training split's, never the external record's.** Non-negotiable 3
(and Phase 8 carry-forward delta 3). ``stats`` must carry a training-partition provenance
label and this module refuses anything else. Re-deriving a scale from the external record
would make every skill score incomparable to the committed tables while looking entirely
normal -- the errors would still be in degrees and metres, just under a different scaling
than the model was fitted under. :func:`corpus_train_norm_stats` exists so that the correct
statistics are *reachable*: they are recomputed by
:meth:`dmf.data.dataset.DeckMotionDataset._fit_stats` rather than serialised beside a
checkpoint, so without it the only way to obtain them is to rebuild the training dataset by
hand at each call site, which is exactly how the wrong ones get used.

**Persistence is recomputed on the external trajectories.** Non-negotiable 4 (delta 4). The
reference model is scored inside the same window loop as every other model, on the same
windows, so the denominator of every skill score is measured on the external record itself.
A denominator carried over from the corpus would not be a skill score at all. As in
:func:`dmf.eval.runner.evaluate_models`, the reference model's own skill must come out
**bitwise** 0.0 and this module raises if it does not.

**Errors are taken in corpus units.** Predictions are mapped back through
:func:`dmf.data.normalize.invert_norm` before any error is formed, because raw RMSE in
normalised space is not comparable across channels and cannot be checked against a landing
threshold. This matters more here than on the corpus: MSS emits angles in **radians**,
this project stores **degrees**, and a uniform factor-of-57 error cancels exactly out of
the skill score (delta 2). The unit conversion is the caller's job -- ``frames`` are
required to arrive already in corpus units -- but :func:`assert_corpus_units` is offered so
that the assertion happens at the boundary rather than nowhere.

**The window population is the external record's.** One row per (model, record, DOF,
horizon): records are kept separate rather than pooled, so that a mean +- std over >= 3
records or >= 3 seeds is available downstream (non-negotiable 5) instead of a single pooled
number whose spread nobody can recover.

**Two tables, one window population.** :func:`evaluate_trajectories` produces the accuracy
table and :func:`evaluate_quiescence_trajectories` the operational one -- the quiescent-
window detector of :mod:`dmf.eval.quiescence`, which Phase 8 carry-forward delta 7 requires
to be run on the external records with the base rate beside every F1. Both cut their windows
through :func:`_prepare_windows`, so the two tables describe the same windows of the same
record; and the quiescence path imports the corpus decision geometry from
:mod:`dmf.eval.quiescence_runner` rather than restating it, because a quiescence number
computed under a different geometry is not comparable to the committed corpus one and
nothing in the output would say so. The units point above is sharper for that metric than
for skill: the landing limits are **absolute**, so a radians-for-degrees error does not
cancel there the way it cancels out of a ratio.

Units throughout: ``frames`` and the returned ``rmse``/``mae``/``rmse_persistence``/
``signal_std`` columns are in corpus units -- degrees for angles, degrees per second for
angular rates, metres for heave, metres per second for heave rate. ``skill`` and ``nrmse``
are dimensionless. Horizons are samples; ``horizon_s`` is seconds.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import NormStats, apply_norm, demean_window, invert_norm, is_train_partition
from dmf.data.splits import REGIMES, Regime, Split, build_split, load_manifest
from dmf.data.windows import WindowSpec, make_windows, window_start_indices
from dmf.eval.metrics import METRIC_COLUMNS, per_dof_horizon_metrics
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
    window_onsets,
)

# The underscored names are private on purpose, and imported on purpose -- the same move
# :mod:`dmf.eval.quiescence_runner` makes on :func:`dmf.eval.runner._bootstrap_counts`, for
# the same reason. Phase 8 delta 7 requires the MSS quiescence numbers to be computed the
# way the committed corpus numbers were; re-deriving the decision geometry here is exactly
# how they would stop being comparable while still looking like a quiescence table. So the
# geometry container, the accumulator, the thresholding fold and the two synthetic
# detectors are the objects the corpus path uses, not copies of them. Every one of them is
# pure-array and dataset-free, which is what makes the reuse possible at all.
from dmf.eval.quiescence_runner import (
    ALWAYS_QUIESCENT,
    RATE_MATCHED,
    RULES,
    _DetectorState,
    _fold,
    _fold_always_quiescent,
    _fold_rate_matched,
    _Geometry,
    _head_of,
    _interval_bounds,
    _TruthState,
    decision_channel_index,
)
from dmf.models.base import BaseForecaster, ForecastModel
from dmf.models.heads import PredictiveDistribution, point_view, quantile_fan
from dmf.typedefs import BoolArray, FloatArray, IntArray

__all__ = [
    "EXTERNAL_COLUMNS",
    "EXTERNAL_QUIESCENCE_COLUMNS",
    "MAX_PLAUSIBLE_ANGLE_DEG",
    "assert_corpus_units",
    "corpus_train_norm_stats",
    "evaluate_quiescence_trajectories",
    "evaluate_trajectories",
    "record_labels",
]

#: Column order of the table :func:`evaluate_trajectories` returns: the corpus metric
#: columns, keyed by the model and the record they were measured on. Stated here for the
#: same reason :data:`dmf.eval.metrics.METRIC_COLUMNS` is -- so the writer and the reader
#: of the CSV cannot drift apart.
EXTERNAL_COLUMNS: tuple[str, ...] = ("model", "record", *METRIC_COLUMNS, "n_records")

#: Largest plausible magnitude for an attitude channel in **degrees**, used by
#: :func:`assert_corpus_units` as a radians-versus-degrees tripwire. A ship that rolls past
#: 90 deg has capsized, so a record whose angles never leave +-1.6 is almost certainly still
#: in radians. Deliberately loose: the check exists to catch a factor of 57, not to police
#: the physics, which ``src/dmf/sim`` already does.
MAX_PLAUSIBLE_ANGLE_DEG: float = 90.0

#: Channels whose corpus unit is degrees or degrees per second, and which are therefore
#: subject to the radians tripwire. Heave is metres and carries no such ambiguity.
_ANGULAR_CHANNELS: frozenset[str] = frozenset(
    {"roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate"}
)


def _as_regime(name: str) -> Regime:
    """Narrow a regime name to the :data:`dmf.data.splits.Regime` literal.

    Args:
        name: Regime name, e.g. ``"unseen_vessel"``.

    Returns:
        The same name, typed.

    Raises:
        ValueError: If it is not one of the four regimes.
    """
    if name not in REGIMES:
        raise ValueError(f"unknown regime {name!r}, expected one of {list(REGIMES)}")
    return name


def record_labels(frames: Sequence[pd.DataFrame]) -> tuple[str, ...]:
    """Derive a stable label for each external record.

    A label is taken from ``frame.attrs["record"]`` when the loader set one -- which is how
    an MSS CSV's filename survives into the results table -- and is otherwise positional.
    Labels must be unique: two records sharing a label would silently average into one row
    downstream, which is the same failure as pooling records, only harder to see.

    Args:
        frames: External records, corpus schema.

    Returns:
        One label per frame, in order.

    Raises:
        ValueError: If ``frames`` is empty or two records carry the same label.
    """
    if not frames:
        raise ValueError("frames is empty; there is no trajectory to evaluate")
    labels = tuple(
        str(frame.attrs.get("record", f"record_{i:03d}")) for i, frame in enumerate(frames)
    )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"record labels {duplicates} are not unique; each external record must be "
            f"identifiable in the results table so that a spread over records is recoverable"
        )
    return labels


def assert_corpus_units(frames: Sequence[pd.DataFrame], channels: Sequence[str]) -> None:
    """Refuse external records whose angles look like radians.

    Phase 8 carry-forward delta 2: MSS emits **radians**, this corpus stores **degrees**,
    and skill is a ratio of two quantities on the same data, so a uniform factor of 57.3
    cancels exactly and leaves ``skill`` unchanged. Only ``rmse`` and every absolute
    threshold -- the quiescence bands -- move. The headline metric therefore cannot catch
    this, which is why the check sits at the boundary where the data arrives.

    The test is a heuristic and is one-sided on purpose: a record is refused only if
    **every** angular channel stays inside +-``MAX_PLAUSIBLE_ANGLE_DEG / 57.3``, which a
    degree-valued ship record of any realistic sea state does not.

    Args:
        frames: External records, expected to be in corpus units.
        channels: Channel names to inspect; only the angular ones are checked.

    Raises:
        ValueError: If a record's angular channels are all small enough to be radians, or
            if a channel is missing or non-finite.
    """
    angular = [c for c in channels if c in _ANGULAR_CHANNELS]
    radians_bound = MAX_PLAUSIBLE_ANGLE_DEG / 57.29577951308232
    for label, frame in zip(record_labels(frames), frames, strict=True):
        missing = [c for c in channels if c not in frame.columns]
        if missing:
            raise ValueError(f"record {label!r} is missing channel(s) {missing}")
        values = frame[list(channels)].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"record {label!r} carries non-finite samples; an RMSE over NaN is NaN, "
                f"and a skill score formed from two NaNs is not a missing result but a "
                f"plausible-looking one"
            )
        if not angular:
            continue
        peak = float(np.abs(frame[angular].to_numpy(dtype=np.float64)).max())
        if peak <= radians_bound:
            raise ValueError(
                f"record {label!r} has a peak angular magnitude of {peak:.4f} over "
                f"{angular}, which is within the radian range: corpus units are DEGREES "
                f"(CLAUDE.md Style). A factor-of-57 error cancels out of the skill score "
                f"and is visible only in rmse and in the quiescence thresholds. Convert "
                f"with numpy.degrees before calling, or pass frames that are already in "
                f"corpus units"
            )


def corpus_train_norm_stats(
    corpus_root: Path, regime: str, cfg: DataConfig, spec: WindowSpec
) -> NormStats:
    """Recover the training-split normalisation statistics for one regime.

    :class:`dmf.data.normalize.NormStats` are not serialised beside a checkpoint: they are
    recomputed by :meth:`dmf.data.dataset.DeckMotionDataset._fit_stats` each time the
    training partition is built. Evaluating a checkpoint off-corpus therefore needs the
    training dataset reconstructed, and this is the one function that does it -- so that no
    external-evaluation call site is tempted to reach for the statistics it has to hand,
    which are the external record's (non-negotiable 3).

    The training partition is loaded eagerly by the dataset, so this costs the read of one
    regime's training realizations. It is the same construction
    :func:`dmf.eval.scoring.score_regime` performs, and it yields the same object.

    Args:
        corpus_root: Path to the Parquet corpus.
        regime: One of :data:`dmf.data.splits.REGIMES`. Must be the regime the checkpoint
            under evaluation was trained on: statistics from a different regime's training
            split are a different scale, and the resulting errors are not comparable to the
            committed tables for that checkpoint.
        cfg: The task configuration the checkpoint was trained under -- channel selection
            and observation mode in particular, since an ``imu`` model scaled by ``ideal``
            statistics is a silent mismatch.
        spec: Window geometry. Does not affect the fitted scale (the statistics are fitted
            over the whole training series, not over its windows) but is required to build
            the dataset.

    Returns:
        Statistics over the motion channels, in corpus units, labelled
        ``"<regime>/train"``.

    Raises:
        ValueError: If ``regime`` is unknown, if the corpus or its manifest is absent, or
            if the recovered statistics do not carry a training-partition label.
    """
    split: Split = build_split(load_manifest(corpus_root), _as_regime(regime))
    train = DeckMotionDataset(corpus_root, split, "train", cfg, spec)
    stats = train.norm_stats
    if not is_train_partition(stats.fitted_on):
        raise ValueError(
            f"statistics recovered from the {regime!r} training partition are labelled "
            f"{stats.fitted_on!r}, which is not a training partition"
        )
    return stats


def _check_stats_provenance(stats: NormStats) -> None:
    """Refuse normalisation statistics that were not fitted on a training split.

    The loud half of non-negotiable 3 / Phase 8 delta 3. :func:`dmf.data.normalize.apply_norm`
    carries the same guard, but it fires deep inside the window loop after the caller has
    already paid for the pass; this fires before the first window is cut, and it names the
    specific way the external path goes wrong -- statistics derived from the external record
    itself.

    Args:
        stats: The statistics the caller supplied.

    Raises:
        ValueError: If ``stats.fitted_on`` does not denote a training partition.
    """
    if not is_train_partition(stats.fitted_on):
        raise ValueError(
            f"refusing to score an external trajectory with statistics labelled "
            f"{stats.fitted_on!r}: the scale must come from the CORPUS TRAINING SPLIT the "
            f"checkpoint was fitted under (CLAUDE.md non-negotiable 3, Phase 8 delta 3). "
            f"Re-deriving a scale from the external record makes every skill score "
            f"incomparable to the committed tables and is invisible in the output. Use "
            f"dmf.eval.external.corpus_train_norm_stats to obtain the right ones"
        )


def _check_channels(
    stats: NormStats, input_channels: Sequence[str], target_dofs: Sequence[str]
) -> tuple[NormStats, NormStats, list[int]]:
    """Validate the channel selection and slice the statistics to it.

    Args:
        stats: Statistics over (at least) the input channels, in corpus units.
        input_channels: Corpus column names the model consumes, in model input order.
        target_dofs: Corpus column names the model forecasts, in model output order.

    Returns:
        Tuple ``(input_stats, target_stats, target_index)`` where ``target_index`` gives the
        position of each target within ``input_channels``.

    Raises:
        ValueError: If either list is empty or has duplicates, if ``target_dofs`` is not a
            prefix of ``input_channels``, or if a channel is absent from ``stats``.
    """
    inputs = tuple(input_channels)
    targets = tuple(target_dofs)
    if not inputs or not targets:
        raise ValueError("input_channels and target_dofs must both be non-empty")
    for name, names in (("input_channels", inputs), ("target_dofs", targets)):
        if len(set(names)) != len(names):
            raise ValueError(f"{name} has duplicate entries: {list(names)}")
    if inputs[: len(targets)] != targets:
        # The same constraint dmf.config.load_data enforces on DataConfig, restated here
        # because this path does not go through a DataConfig: P2-D4 has
        # dmf.models.persistence.Persistence forecast by slicing the FIRST C_out input
        # channels, so any other ordering makes the reference baseline -- the denominator
        # of every skill score below -- forecast the wrong channel, silently.
        raise ValueError(
            f"target_dofs {list(targets)} must be a prefix of input_channels "
            f"{list(inputs)}: dmf.models.persistence.Persistence forecasts by slicing the "
            f"first {len(targets)} input channels (P2-D4), so any other ordering would "
            f"make the skill denominator a forecast of the wrong channel"
        )
    return stats.subset(inputs), stats.subset(targets), [inputs.index(c) for c in targets]


def _as_point_model(name: str, model: ForecastModel) -> ForecastModel:
    """View a model as a rank-3 point forecaster, as the corpus runner does.

    Probabilistic models are reduced to their own point forecast -- the 0.5 quantile, or the
    Gaussian mean -- through :func:`dmf.models.heads.point_view`, the *same* wrapper
    :func:`dmf.eval.scoring.score_regime` applies before
    :func:`dmf.eval.runner.evaluate_models`. Matching it rather than re-deriving a median
    here is the point: an external row and a corpus row must be the same projection of the
    same model, or the two tables are not comparable.

    A model that declares no head kind but nonetheless emits a rank-4 tensor is handled at
    the output instead, in :func:`_reduce_to_point`.

    Args:
        name: Model key, for error messages.
        model: The model.

    Returns:
        A forecaster whose ``forward`` returns ``(B, H, C_out)`` where the head kind is
        known; ``model`` itself otherwise.

    Raises:
        ValueError: If a model declares a non-point head but is not a
            :class:`dmf.models.base.BaseForecaster`, so that
            :func:`dmf.models.heads.point_view` cannot wrap it.
    """
    head_kind = getattr(model, "head_kind", None)
    if head_kind is None or head_kind == "point":
        return model
    if not isinstance(model, BaseForecaster):
        raise ValueError(
            f"model {name!r} declares head_kind {head_kind!r} but is not a BaseForecaster, "
            f"so its point projection cannot be taken the way the corpus tables take it"
        )
    return cast(ForecastModel, point_view(model))


def _reduce_to_point(name: str, raw: Tensor) -> Tensor:
    """Reduce a model output to a point forecast, tolerating an undeclared quantile fan.

    Args:
        name: Model key, for error messages.
        raw: Model output, ``(B, H, C)`` or ``(B, H, C, Q)``, dimensionless.

    Returns:
        Point forecasts, ``(B, H, C)``, dimensionless. For a rank-4 output the **median**
        quantile is taken, via :class:`dmf.models.heads.PredictiveDistribution` so that the
        fan is sorted first and the 0.5 level is read off rather than interpolated -- the
        same selection :meth:`dmf.models.heads.PredictiveDistribution.point` makes on the
        corpus path (P5-D5).

    Raises:
        ValueError: If ``raw`` is neither rank 3 nor rank 4, or if its trailing axis is not
            a fan width :func:`dmf.models.heads.quantile_fan` defines.
    """
    if raw.ndim == 3:
        return raw
    if raw.ndim != 4:
        raise ValueError(
            f"model {name!r} returned rank {raw.ndim} output {tuple(raw.shape)}; expected "
            f"(B, H, C_out) or (B, H, C_out, Q)"
        )
    levels = quantile_fan(int(raw.shape[-1]))
    return PredictiveDistribution(raw, "quantile", levels).point()


def _record_series(
    frame: pd.DataFrame, label: str, channels: Sequence[str], spec: WindowSpec
) -> FloatArray:
    """Project one external record onto the model's input channels.

    Args:
        frame: The record, corpus schema and corpus units.
        label: Record label, for error messages.
        channels: Corpus column names, in model input order.
        spec: Window geometry, checked against the record length.

    Returns:
        Array of shape ``(n_samples, len(channels))``, float64, corpus units.

    Raises:
        ValueError: If a column is missing, if a sample is non-finite, or if the record is
            shorter than one window.
    """
    missing = [c for c in channels if c not in frame.columns]
    if missing:
        raise ValueError(
            f"record {label!r} is missing channel(s) {missing}; its columns are "
            f"{list(frame.columns)}"
        )
    series = frame[list(channels)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(series)):
        raise ValueError(f"record {label!r} carries non-finite samples in {list(channels)}")
    if series.shape[0] < spec.total_length:
        raise ValueError(
            f"record {label!r} has {series.shape[0]} samples, fewer than the "
            f"{spec.total_length} one window needs (lookback {spec.lookback} + max horizon "
            f"{spec.max_horizon})"
        )
    return series


@contextmanager
def _eval_mode(models: Mapping[str, ForecastModel], device: str) -> Iterator[None]:
    """Put every ``nn.Module`` in ``models`` on ``device`` in eval mode, then restore it.

    Scoring must not depend on whether the caller happened to hand over a model in training
    mode: dropout and batch-norm running statistics would make the table a function of that.
    The prior mode is restored on exit, including on an exception, because a scoring call
    that silently leaves a caller's model in eval mode is a defect in the *caller's* next
    training loop rather than in this one.

    Args:
        models: Models to prepare. Non-module entries are left alone.
        device: Torch device the models run on.

    Yields:
        Nothing; the context is the prepared state.
    """
    previous: dict[str, bool] = {}
    torch_device = torch.device(device)
    for name, model in models.items():
        if isinstance(model, nn.Module):
            previous[name] = model.training
            model.to(torch_device)
            model.eval()
    try:
        yield
    finally:
        for name, was_training in previous.items():
            module = models[name]
            if isinstance(module, nn.Module) and was_training:
                module.train()


def _prepare_windows(
    series: FloatArray, spec: WindowSpec, input_stats: NormStats, target_index: Sequence[int]
) -> tuple[Tensor, Tensor, FloatArray, IntArray]:
    """Cut and normalise one record's windows, exactly as the dataset does.

    The single windowing path of this module: :func:`evaluate_trajectories` and
    :func:`evaluate_quiescence_trajectories` both go through it, so the skill table and the
    quiescence table are measured on bit-for-bit the same windows of the same record. It
    mirrors :meth:`dmf.data.dataset.DeckMotionDataset.__getitem__`: ``x`` is de-meaned and
    scaled, ``y`` and the window mean stay in corpus units, and the mean is carried to the
    inverse transform rather than re-derived. float64 through the transform and float32 at
    the boundary, as the dataset stores and the runner widens -- the cast is where the two
    pipelines could differ in the last bit, and every committed corpus table was produced
    from float32 windows.

    Args:
        series: The record's channels, ``(n_samples, C_in)``, float64, corpus units.
        spec: Window geometry, samples.
        input_stats: Statistics over the input channels, from the corpus training split.
        target_index: Position of each target channel within the input channels.

    Returns:
        Tuple ``(x_norm, window_mean, target, starts)``: model inputs ``(N, L, C_in)``
        float32 and dimensionless; per-window target means ``(N, 1, C_out)`` float32 in
        corpus units; targets ``(N, H, C_out)`` float64 in corpus units; and the absolute
        start sample of each window, ``(N,)``.
    """
    index = list(target_index)
    x_raw, y_raw = make_windows(series, spec)
    x_centred, mean = demean_window(torch.from_numpy(np.ascontiguousarray(x_raw)))
    x_norm = apply_norm(x_centred, input_stats).to(torch.float32)
    window_mean = mean[:, :, index].to(torch.float32)
    target = np.ascontiguousarray(y_raw[:, :, index], dtype=np.float32).astype(np.float64)
    starts = window_start_indices(series.shape[0], spec)
    return x_norm, window_mean, target, starts


def _forecast_record(
    models: Mapping[str, ForecastModel],
    x_norm: Tensor,
    window_mean: Tensor,
    target_stats: NormStats,
    *,
    device: str,
    batch_size: int,
    n_targets: int,
    max_horizon: int,
) -> dict[str, FloatArray]:
    """Run every model over one record's windows and map the output to corpus units.

    Args:
        models: Models to score, keyed by results-table label.
        x_norm: Model inputs, ``(N, L, C_in)``, float32, de-meaned and scaled
            (dimensionless).
        window_mean: Per-window target means, ``(N, 1, C_out)``, float32, corpus units.
        target_stats: Statistics over the target channels, from the corpus training split.
        device: Torch device the models run on. Predictions are always returned on the CPU.
        batch_size: Windows per forward pass. Affects speed and memory only.
        n_targets: Expected ``C_out``.
        max_horizon: Expected ``H``.

    Returns:
        One ``(N, H, C_out)`` float64 array per model, in corpus units.

    Raises:
        ValueError: If a model returns the wrong shape.
    """
    n_windows = int(x_norm.shape[0])
    torch_device = torch.device(device)
    point_models = {name: _as_point_model(name, model) for name, model in models.items()}
    out: dict[str, FloatArray] = {
        name: np.empty((n_windows, max_horizon, n_targets), dtype=np.float64)
        for name in point_models
    }
    with _eval_mode(point_models, device), torch.no_grad():
        for start in range(0, n_windows, batch_size):
            stop = min(start + batch_size, n_windows)
            inputs = x_norm[start:stop].to(torch_device)
            mean = window_mean[start:stop].double()
            for name, model in point_models.items():
                raw = _reduce_to_point(name, model.forward(inputs))
                if tuple(raw.shape) != (stop - start, max_horizon, n_targets):
                    raise ValueError(
                        f"model {name!r} returned shape {tuple(raw.shape)}, expected "
                        f"({stop - start}, {max_horizon}, {n_targets})"
                    )
                pred = invert_norm(raw.detach().to("cpu", torch.float64), target_stats, mean)
                out[name][start:stop] = pred.numpy()
    return out


def evaluate_trajectories(
    models: Mapping[str, ForecastModel],
    frames: Sequence[pd.DataFrame],
    *,
    stats: NormStats,
    spec: WindowSpec,
    input_channels: Sequence[str],
    target_dofs: Sequence[str],
    horizons: tuple[int, ...],
    fs_hz: float,
    persistence_key: str = "persistence",
    device: str = "cpu",
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Score trained models on externally supplied trajectories.

    The external counterpart of :func:`dmf.eval.runner.evaluate_models`, composed from the
    same parts: :func:`dmf.data.windows.make_windows` cuts the windows,
    :func:`dmf.data.normalize.demean_window` and :func:`dmf.data.normalize.apply_norm`
    prepare the inputs, :func:`dmf.data.normalize.invert_norm` returns the output to corpus
    units, and :func:`dmf.eval.metrics.per_dof_horizon_metrics` builds the table. The
    transform mirrors :meth:`dmf.data.dataset.DeckMotionDataset.__getitem__` exactly: ``x``
    is de-meaned and scaled, ``y`` and the window mean stay in corpus units, and the mean is
    carried to the inverse rather than re-derived.

    Every model is scored on the *same* windows of the *same* record inside one loop, and
    the skill denominator of every row is ``models[persistence_key]``'s error on those same
    windows -- recomputed on the external trajectories, never carried over from the corpus
    (non-negotiable 4, Phase 8 delta 4). The reference model's own skill must be bitwise
    0.0 and this function raises if it is not.

    Records are **not pooled**: one row per (model, record, DOF, horizon), so a mean +- std
    over records or seeds is recoverable downstream (non-negotiable 5). Pooling first and
    reporting a single number would discard exactly the spread that makes an out-of-
    distribution result interpretable.

    What this function does not do: it does not check the sea state, the hull or the
    spectrum the external record came from. Which distributional shift is being measured --
    a vessel shift under a matched spectrum, or a sea-state shift under an unmatched one --
    is the caller's to pre-register (Phase 8 delta 1), and the two predict opposite model
    rankings.

    Args:
        models: Models to score, keyed by the label used in the results table. Every model
            must accept ``(B, L, C_in)`` and return ``(B, H, C_out)`` or, for a
            probabilistic head, ``(B, H, C_out, Q)``, from which the median quantile is
            taken. ``nn.Module`` entries are moved to ``device`` and switched to eval mode
            for the duration of the call, with their prior training mode restored on exit.
            Must contain ``persistence_key``.
        frames: External records, one per trajectory, in the corpus schema and **corpus
            units**: degrees for angles, degrees per second for angular rates, metres for
            heave, metres per second for heave rate. MSS records are in radians and must be
            converted before they reach this function; :func:`assert_corpus_units` is the
            boundary check for that. Each must be sampled at ``fs_hz`` and be at least
            ``spec.total_length`` samples long. A label may be supplied per record as
            ``frame.attrs["record"]``.
        stats: Normalisation statistics from the **corpus training split**, in corpus
            units, covering at least ``input_channels``. Must carry a training-partition
            provenance label; anything else is refused (non-negotiable 3, Phase 8 delta 3).
            Obtain them with :func:`corpus_train_norm_stats`.
        spec: Window geometry, samples. Use the geometry the checkpoints were trained
            under; Phase 8 delta 5 keeps this at the default 200-sample lookback.
        input_channels: Corpus column names the model consumes, in model input order.
        target_dofs: Corpus column names the model forecasts, in model output order. Must
            be a prefix of ``input_channels`` (P2-D4).
        horizons: Horizons to report, samples, each in ``[1, spec.max_horizon]``. "Horizon
            ``h``" is the error at lead time exactly ``h`` samples.
        fs_hz: Sampling rate of the external records, hertz, used to report each horizon in
            seconds. A record sampled at a different rate than the corpus would make every
            horizon a different lead time in seconds, so this is the caller's assertion to
            make about the record, not something this function can infer.
        persistence_key: Key into ``models`` naming the reference baseline.
        device: Torch device the models run on. Errors are always accumulated on the CPU in
            float64.
        batch_size: Windows per forward pass. Affects speed and memory only.

    Returns:
        One row per (model, record, DOF, horizon), columns :data:`EXTERNAL_COLUMNS`.
        ``rmse``, ``mae``, ``rmse_persistence`` and ``signal_std`` are in corpus units;
        ``skill`` and ``nrmse`` are dimensionless; ``n_windows`` counts the windows of that
        one record, and ``n_records`` the records in the call.

    Raises:
        ValueError: If ``models`` or ``frames`` is empty, if ``persistence_key`` is absent
            from ``models``, if ``stats`` was not fitted on a training partition, if
            ``target_dofs`` is not a prefix of ``input_channels``, if a record is missing a
            channel, carries non-finite samples, or is shorter than one window, or if a
            model returns the wrong output shape.
        RuntimeError: If the reference model's own skill is not bitwise 0.0.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    if persistence_key not in models:
        raise ValueError(
            f"persistence_key {persistence_key!r} is not among the models {sorted(models)}; "
            f"the skill denominator must be measured on these same external windows, and a "
            f"denominator carried over from the corpus is not a skill score (CLAUDE.md "
            f"non-negotiable 4, Phase 8 delta 4)"
        )
    # Checked before a single window is cut: a provenance failure invalidates every number
    # this function would produce.
    _check_stats_provenance(stats)
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    input_stats, target_stats, target_index = _check_channels(stats, input_channels, target_dofs)
    labels = record_labels(frames)
    dof_names = tuple(target_dofs)
    max_horizon = spec.max_horizon

    tables: list[pd.DataFrame] = []
    for label, frame in zip(labels, frames, strict=True):
        series = _record_series(frame, label, input_channels, spec)
        x_norm, window_mean, target, _ = _prepare_windows(series, spec, input_stats, target_index)
        predictions = _forecast_record(
            models,
            x_norm,
            window_mean,
            target_stats,
            device=device,
            batch_size=batch_size,
            n_targets=len(dof_names),
            max_horizon=max_horizon,
        )
        reference = predictions[persistence_key]
        for name, pred in predictions.items():
            table = per_dof_horizon_metrics(
                pred=pred,
                target=target,
                persistence_pred=reference,
                dof_names=dof_names,
                horizons=horizons,
                fs_hz=fs_hz,
            )
            table.insert(0, "record", label)
            table.insert(0, "model", name)
            tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    combined["n_records"] = len(labels)
    own_skill = combined.loc[combined["model"] == persistence_key, "skill"].to_numpy()
    if not np.all(own_skill == 0.0):
        # The same check dmf.eval.runner.evaluate_models makes, for the same reason: skill
        # is 1 - sse_model/sse_persistence formed from the reference model's own errors, so
        # a non-zero self-skill can only mean the reference row and the skill denominator
        # came from different forecasts -- here, that the persistence denominator is not the
        # one measured on this record.
        raise RuntimeError(
            f"the reference model {persistence_key!r} does not score exactly 0.0 against "
            f"itself on the external records (worst |skill| = {np.abs(own_skill).max():.3e}). "
            f"The denominator and the reference row must be the same forecast on the same "
            f"windows; they are not."
        )
    return combined[list(EXTERNAL_COLUMNS)]


#: Column order of the table :func:`evaluate_quiescence_trajectories` returns.
#:
#: ``base_rate`` sits immediately beside ``f1`` and is not optional (CLAUDE.md
#: §Known traps, Phase 8 carry-forward delta 7): an F1 read without it is a statement
#: about how often the deck happened to be quiet. ``always_yes_sample_f1`` sits beside
#: both, so the per-sample/onset contrast that makes the onset formulation credible is
#: readable from the row rather than from a test fixture. The three ``peak_*`` columns and
#: the four threshold columns are in the table for one reason: the thresholds are
#: **absolute**, so a units error moves the F1 without moving anything else, and a reader
#: must be able to check the record's magnitudes against the limits that were applied to it
#: without re-opening the record.
EXTERNAL_QUIESCENCE_COLUMNS: tuple[str, ...] = (
    "model",
    "record",
    "threshold_set",
    "rule",
    "scorable",
    "base_rate",
    "always_yes_sample_f1",
    "n_onsets_true",
    "n_onsets_pred",
    "n_matched",
    "precision",
    "recall",
    "f1",
    "false_alarms_per_min",
    "lead_p10",
    "median_lead_time_s",
    "lead_p90",
    "n_onsets_true_raw",
    "n_excluded_true",
    "n_excluded_pred",
    "n_windows",
    "duration_s",
    "peak_roll_deg",
    "peak_pitch_deg",
    "peak_heave_rate_mps",
    "roll_limit_deg",
    "pitch_limit_deg",
    "heave_rate_limit_mps",
    "sustain_s",
    "n_records",
)


@dataclass(frozen=True)
class _ExternalTruth:
    """The truth side of one threshold set on one external record.

    Attributes:
        state: The one-realization :class:`dmf.eval.quiescence_runner._TruthState` the
            corpus path's folds consume, so the synthetic detectors can be driven by the
            same code.
        base_rate: Fraction of the evaluated span inside a sustained quiescent window.
        always_yes_sample_f1: Per-sample F1 of a detector that says "quiescent" at every
            sample of the evaluated span.
        n_raw_onsets: Onsets before the exclusion rule, so that ``kept + excluded == raw``
            is checkable from the table.
        peaks: Peak absolute value of roll (deg), pitch (deg) and heave rate (m/s) over the
            evaluated record, for the units audit.
    """

    state: _TruthState
    base_rate: float
    always_yes_sample_f1: float
    n_raw_onsets: int
    peaks: tuple[float, float, float]


def _external_geometry(starts: IntArray, spec: WindowSpec) -> _Geometry:
    """Build the corpus decision geometry for one external record.

    Deliberately the same arithmetic as :func:`dmf.eval.quiescence_runner._geometry`, on the
    same :class:`dmf.eval.quiescence_runner._Geometry`, rather than a second definition of
    where the detector stands in time. At the production geometry -- lookback 200, stride 5,
    max horizon 150, 6000-sample record -- it yields first decision 199, last covered 6000
    and 1131 decision times, which is exactly the span
    ``results/e04/quiescence.csv`` was scored over (580.0 s per realization).

    Args:
        starts: Window start samples, ascending.
        spec: Window geometry, samples.

    Returns:
        The bounds.

    Raises:
        RuntimeError: If the starts are not strictly ascending, which is what makes "keep
            the earliest decision time" a single ``setdefault``.
    """
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


def _external_truth(
    series: FloatArray,
    channels: tuple[int, int, int],
    thresholds: QuiescenceThresholds,
    fs_hz: float,
    geometry: _Geometry,
) -> _ExternalTruth:
    """Build the truth side from the full TRUE trajectory of one external record.

    P6-D2 item 1: the truth mask is a property of the record, not of the windowing, so it is
    computed once over the whole trajectory rather than reassembled from window targets.
    :func:`dmf.eval.quiescence.detect_quiescent_mask` is the definition of a quiescent
    window and is called here unmodified; the corpus runner's ``_truth_state`` applies the
    identical limits through its batched run-finder, which
    ``tests/test_quiescence_runner.py`` pins to this reference row by row.

    Args:
        series: The record's channels, ``(n_samples, C_in)``, corpus units.
        channels: Column indices of roll, pitch and heave rate.
        thresholds: The limit set, in **absolute corpus units** -- degrees and metres per
            second.
        fs_hz: Sampling rate, hertz.
        geometry: Shared absolute-time bounds.

    Returns:
        The truth side.
    """
    roll_c, pitch_c, rate_c = channels
    lower, upper = geometry.first_decision, geometry.last_covered
    decision = series[:upper, :]
    mask: BoolArray = detect_quiescent_mask(
        decision[:, roll_c], decision[:, pitch_c], decision[:, rate_c], thresholds, fs_hz
    )
    onsets, n_excluded = scorable_onsets(mask, first_decision_idx=lower)
    # The base rate describes the span the F1 is about, not the whole record -- the same
    # slice `_truth_state` takes, so the two tables' base rates are the same quantity.
    span = mask[lower + 1 :]
    n_quiescent = int(np.count_nonzero(span))
    n_evaluated = int(span.size)
    # Exactly the always-yes detector scored per sample: it calls every sample quiescent, so
    # it matches every quiescent sample (recall 1.0) and predicts `n_evaluated` of them
    # (precision = base rate). Formed through the same `precision_recall_f1` the onset rows
    # use rather than from the closed form, so there is one definition of F1 in this table.
    _, _, sample_f1 = precision_recall_f1(n_quiescent, n_evaluated, n_quiescent)
    peaks = tuple(float(np.abs(decision[:, c]).max()) for c in (roll_c, pitch_c, rate_c))
    return _ExternalTruth(
        state=_TruthState([onsets], [n_excluded], [n_quiescent], [n_evaluated]),
        base_rate=base_rate(span),
        always_yes_sample_f1=sample_f1,
        n_raw_onsets=int(window_onsets(mask).size),
        peaks=cast(tuple[float, float, float], peaks),
    )


def _quiescence_row(
    *,
    model: str,
    record: str,
    thresholds: QuiescenceThresholds,
    rule: str,
    truth: _ExternalTruth,
    state: _DetectorState,
    fs_hz: float,
    tolerance_s: float,
    n_windows: int,
) -> dict[str, object]:
    """Match one detector's onsets against one record's truth and form the reported row.

    Args:
        model: Model label.
        record: Record label.
        thresholds: The limit set.
        rule: ``"point"`` or ``"interval"``.
        truth: The truth side of this record and threshold set.
        state: The accumulated predictions for this (model, rule, threshold set).
        fs_hz: Sampling rate, hertz.
        tolerance_s: Onset matching tolerance, seconds.
        n_windows: Decision times on this record.

    Returns:
        One record on :data:`EXTERNAL_QUIESCENCE_COLUMNS`.
    """
    true_onsets = truth.state.onsets[0]
    predicted_map = state.earliest_flag[0]
    predicted = np.asarray(sorted(predicted_map), dtype=np.int64)
    matched_pred, matched_true, _ = match_onsets(predicted, true_onsets, fs_hz, tolerance_s)
    n_true = int(true_onsets.size)
    n_pred = int(predicted.size)
    n_matched = int(matched_pred.size)
    if n_matched:
        flags = np.asarray(
            [predicted_map[int(predicted[i])] for i in matched_pred.tolist()], dtype=np.int64
        )
        leads = lead_times(flags, true_onsets[matched_true], fs_hz)
    else:
        leads = np.zeros(0, dtype=np.float64)
    evaluated = truth.state.n_evaluated[0]
    duration_s = evaluated / fs_hz
    scorable = n_true > 0
    precision, recall, f1 = precision_recall_f1(n_matched, n_pred, n_true)
    return {
        "model": model,
        "record": record,
        "threshold_set": thresholds.name,
        "rule": rule,
        # A record with no scorable onset has no measurement in it. NaN, never 0.0 (which
        # reads as model failure) and never 1.0 (which reads as success) -- P6-D7, and the
        # rule the committed corpus table follows. `base_rate` is still filled in, because
        # it is the column that says *why* the record is not scorable.
        "scorable": scorable,
        "base_rate": truth.base_rate,
        "always_yes_sample_f1": truth.always_yes_sample_f1,
        "n_onsets_true": n_true,
        "n_onsets_pred": n_pred,
        "n_matched": n_matched,
        "precision": precision if scorable else float("nan"),
        "recall": recall if scorable else float("nan"),
        "f1": f1 if scorable else float("nan"),
        "false_alarms_per_min": false_alarms_per_minute(n_pred - n_matched, duration_s),
        "lead_p10": _lead_quantile(leads, 0.10),
        "median_lead_time_s": _lead_quantile(leads, 0.50),
        "lead_p90": _lead_quantile(leads, 0.90),
        "n_onsets_true_raw": truth.n_raw_onsets,
        "n_excluded_true": int(truth.state.n_excluded[0]),
        "n_excluded_pred": int(state.n_excluded[0]),
        "n_windows": n_windows,
        "duration_s": duration_s,
        "peak_roll_deg": truth.peaks[0],
        "peak_pitch_deg": truth.peaks[1],
        "peak_heave_rate_mps": truth.peaks[2],
        "roll_limit_deg": thresholds.roll_deg,
        "pitch_limit_deg": thresholds.pitch_deg,
        "heave_rate_limit_mps": thresholds.heave_rate_mps,
        "sustain_s": thresholds.sustain_s,
    }


def _lead_quantile(values: FloatArray, level: float) -> float:
    """Return a quantile of the lead-time sample, or NaN when there is none.

    Args:
        values: Lead times, seconds.
        level: Quantile level in [0, 1].

    Returns:
        The quantile, seconds, or NaN if ``values`` is empty. NaN rather than 0.0: a record
        with no matched onset has no lead-time distribution, and 0.0 would read as "the
        model flagged exactly at the onset".
    """
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, level))


def evaluate_quiescence_trajectories(
    models: Mapping[str, ForecastModel],
    frames: Sequence[pd.DataFrame],
    *,
    stats: NormStats,
    spec: WindowSpec,
    input_channels: Sequence[str],
    target_dofs: Sequence[str],
    fs_hz: float,
    threshold_sets: Sequence[QuiescenceThresholds] = (PERMISSIVE, STRICT),
    tolerance_s: float = 0.5,
    include_always_quiescent: bool = True,
    device: str = "cpu",
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Score the quiescent-window detector on externally supplied trajectories.

    The external counterpart of :func:`dmf.eval.quiescence_runner.evaluate_quiescence`, and
    the discharge of Phase 8 carry-forward delta 7: the operational metric run on the MSS
    strip-theory records rather than only on this project's own corpus.

    **The geometry is the corpus geometry, because it is the corpus geometry's own code.**
    A quiescence number computed under a different decision geometry is not comparable to
    the committed corpus numbers, and the difference does not show up anywhere in the
    output -- which is the whole reason for running the metric on MSS at all. So this
    function does not re-derive P6-D2; it imports it.
    :class:`dmf.eval.quiescence_runner._Geometry` fixes the decision times at the window
    stride (0.5 s at the production arm), :func:`dmf.eval.quiescence_runner._fold` does the
    thresholding and keeps the **earliest** decision time that predicts each absolute onset,
    :func:`dmf.eval.quiescence.scorable_onsets` applies the exclusion rule to the truth side
    and :func:`dmf.eval.quiescence.match_onsets` matches at +-``tolerance_s`` one-to-one.
    The truth mask comes from the full **true** trajectory on roll, pitch and heave rate via
    :func:`dmf.eval.quiescence.detect_quiescent_mask`, with the sustain requirement
    ``ceil(sustain_s * fs_hz)`` samples.

    **The base rate is reported beside every F1**, in the same row, and is never optional
    (CLAUDE.md §Known traps). ``always_yes_sample_f1`` is in the row too: it is the F1 a
    detector that says "quiescent" at every sample scores against that record's own truth
    mask, and the gap between it and the ``always_quiescent`` row's onset F1 is what makes
    the onset formulation credible. On the P6-D2 fixture those two numbers are 0.9691 and
    0.0182; here they are measured on each MSS record instead of quoted.

    **Records are not pooled.** One row per (model, record, threshold set, rule), so that a
    mean +- std over the >= 3 MSS seeds of a cell is recoverable downstream
    (non-negotiable 5). No bootstrap interval is emitted: the corpus table's interval
    resamples whole realizations, and a single record is a single unit, so an interval
    computed here would be a resample of one thing. Aggregate the records, then quote the
    spread.

    **Units are asserted, not assumed.** The thresholds are absolute -- 3.0 deg / 2.0 deg /
    0.8 m/s permissive, 1.5 / 1.0 / 0.4 strict -- so unlike the skill score, which is a
    ratio and in which a factor of 57.3 cancels exactly, this metric moves under a
    radians-for-degrees error and moves plausibly. :func:`assert_corpus_units` is therefore
    run over every record before a single window is cut, and each row carries the record's
    peak roll, pitch and heave rate beside the limits that were applied to it.

    Args:
        models: Detectors to score, keyed by results-table label. Point models are scored on
            the ``point`` rule only; a model whose ``head_kind`` is not ``"point"`` is
            additionally scored on the two-sided ``interval`` rule (P6-D5). Note that a
            *constant* forecast -- ``persistence``, ``window_mean`` -- cannot express a
            transition and therefore predicts no onset at all, by construction rather than
            by failure; read ``n_onsets_pred`` beside its F1.
        frames: External records, one per trajectory, in the corpus schema and **corpus
            units**. Each must be sampled at ``fs_hz`` and be at least ``spec.total_length``
            samples long. A label may be supplied per record as ``frame.attrs["record"]``.
        stats: Normalisation statistics from the **corpus training split**, in corpus units.
            Must carry a training-partition provenance label (non-negotiable 3).
        spec: Window geometry, samples. Use the geometry the checkpoints were trained
            under; the decision stride is ``spec.stride`` and the forecast the detector
            thresholds runs to ``spec.max_horizon``.
        input_channels: Corpus column names the model consumes, in model input order.
        target_dofs: Corpus column names the model forecasts, in model output order. Must
            be a prefix of ``input_channels`` (P2-D4) and must include roll, pitch and heave
            rate, since the prediction side thresholds the model's own output.
        fs_hz: Sampling rate of the external records, hertz. Used for the sustain
            requirement, the matching tolerance, the lead times and the false-alarm rate, so
            a record sampled at another rate would change every one of them.
        threshold_sets: Limit sets to report, each producing its own rows. Both
            :data:`dmf.eval.quiescence.PERMISSIVE` and :data:`dmf.eval.quiescence.STRICT`
            by default, as the committed corpus table carries both.
        tolerance_s: Onset matching tolerance, seconds.
        include_always_quiescent: Whether to add the two synthetic references,
            :data:`dmf.eval.quiescence_runner.ALWAYS_QUIESCENT` and
            :data:`dmf.eval.quiescence_runner.RATE_MATCHED`. Neither costs a forward pass
            and between them they bound the metric from the over-flagging side and the
            right-rate/wrong-time side.
        device: Torch device the models run on.
        batch_size: Windows per forward pass. Affects speed and memory only.

    Returns:
        One row per (model, record, threshold set, rule), columns
        :data:`EXTERNAL_QUIESCENCE_COLUMNS`, sorted by
        ``(model, record, threshold_set, rule)``. ``base_rate``, ``precision``, ``recall``,
        ``f1`` and ``always_yes_sample_f1`` are dimensionless; lead times are seconds;
        ``false_alarms_per_min`` is per minute; ``peak_*`` and the limit columns are in
        corpus units. ``precision``/``recall``/``f1`` are NaN on a record with no scorable
        onset.

    Raises:
        ValueError: If ``models`` or ``frames`` is empty, if ``fs_hz`` is not positive, if
            ``stats`` was not fitted on a training partition, if ``target_dofs`` is not a
            prefix of ``input_channels`` or does not forecast all three decision channels,
            if a record is missing a channel, carries non-finite samples, looks like it is
            still in radians, or is shorter than one window, or if a model returns an
            unexpected output shape.
        RuntimeError: If a record's window starts are not strictly ascending.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    # Before a single window is cut, for the same reason the provenance guard is: both
    # invalidate every number this function would otherwise produce.
    _check_stats_provenance(stats)
    input_stats, target_stats, target_index = _check_channels(stats, input_channels, target_dofs)
    labels = record_labels(frames)
    # Refuses a record whose angles are still in radians. The skill table can survive that
    # error; a table of absolute thresholds cannot.
    assert_corpus_units(frames, input_channels)
    channels = decision_channel_index(target_dofs)
    n_targets = len(tuple(target_dofs))
    heads = {name: _head_of(model) for name, model in models.items()}
    torch_device = torch.device(device)

    rows: list[dict[str, object]] = []
    for label, frame in zip(labels, frames, strict=True):
        series = _record_series(frame, label, input_channels, spec)
        x_norm, window_mean, _target, starts = _prepare_windows(
            series, spec, input_stats, target_index
        )
        geometry = _external_geometry(starts, spec)
        n_windows = int(x_norm.shape[0])
        truth = {
            thresholds.name: _external_truth(series, channels, thresholds, fs_hz, geometry)
            for thresholds in threshold_sets
        }

        def _new_state() -> _DetectorState:
            """Return an accumulator for one (model, rule, threshold set) on one record."""
            return _DetectorState([{}], [0])

        states: dict[tuple[str, str, str], _DetectorState] = {}
        for name in models:
            rules = ("point",) if heads[name][0] == "point" else RULES
            for rule in rules:
                for thresholds in threshold_sets:
                    states[(name, rule, thresholds.name)] = _new_state()

        with _eval_mode(models, device), torch.no_grad():
            for start in range(0, n_windows, batch_size):
                stop = min(start + batch_size, n_windows)
                inputs = x_norm[start:stop].to(torch_device)
                mean = window_mean[start:stop].double()
                decision_times = geometry.decision_times[start:stop]
                # One record is one realization here, so every window folds into slot 0.
                realization_index = np.zeros(stop - start, dtype=np.int64)
                for name, model in models.items():
                    head, levels, width = heads[name]
                    raw = model.forward(inputs).detach().to("cpu", torch.float64)
                    expected = (stop - start, spec.max_horizon, n_targets, width)
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
                    point = invert_norm(point_raw, target_stats, mean).numpy()
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
                    bounds = invert_norm(
                        _interval_bounds(raw, head, levels), target_stats, mean
                    ).numpy()
                    # Two-sided, per P6-D5: the limits are symmetric, so the conservative
                    # statistic is max(|q05|, |q95|) and not the one-sided 0.05 quantile.
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

        if include_always_quiescent:
            for thresholds in threshold_sets:
                degenerate = _new_state()
                states[(ALWAYS_QUIESCENT, "point", thresholds.name)] = degenerate
                _fold_always_quiescent(degenerate, geometry, 1)
                chance = _new_state()
                states[(RATE_MATCHED, "point", thresholds.name)] = chance
                _fold_rate_matched(chance, geometry, truth[thresholds.name].state, 1)

        for (name, rule, threshold_name), state in states.items():
            rows.append(
                _quiescence_row(
                    model=name,
                    record=label,
                    thresholds=next(t for t in threshold_sets if t.name == threshold_name),
                    rule=rule,
                    truth=truth[threshold_name],
                    state=state,
                    fs_hz=fs_hz,
                    tolerance_s=tolerance_s,
                    n_windows=n_windows,
                )
            )

    table = pd.DataFrame(rows)
    table["n_records"] = len(labels)
    return table[list(EXTERNAL_QUIESCENCE_COLUMNS)].sort_values(
        ["model", "record", "threshold_set", "rule"], ignore_index=True
    )
