"""Driving the phase-lag estimator over a test partition, and joining it to the metrics.

:mod:`dmf.eval.phase` defines *what* a phase lag is. This module defines what it is
measured **on**, which §6.1 of the plan does not fix and which changes the number. The
geometry is pre-registered as ``docs/protocol.md`` P6-D3 and implemented here literally.

**The series being correlated.** Forecasts are direct multi-horizon, so there is no single
"forecast time series": at lead ``h`` the series is the sequence of lead-``h`` predictions
across consecutive window origins, and the truth it is compared against is the true signal
sampled at those same origins plus ``h``. At the production ``stride: 5`` those origins are
0.5 s apart, which is a fifth of the resolution needed to resolve a lag on a 12 s roll
period, so this pass runs on a **stride-1** dataset and refuses anything else.

**Which realizations.** Thirty-two test realizations per regime, chosen without any RNG.
P6-D3 says "selected by sorted realization key"; the selection implemented here is
*evenly spaced over* the sorted key list rather than its first 32 entries, and the
difference matters: the key sorts on ``(ss, heading, speed, vessel, seed)``, so the first
32 keys of any regime are all one sea state and one heading, and a phase lag measured there
would be a statement about SS3 in bow seas presented as a statement about the regime. Even
spacing is equally free of RNG and spans the grid. Recorded as a refinement of P6-D3.

**What the number can and cannot say.** The lag is identified only modulo the dominant
period of the signal (see :mod:`dmf.eval.phase`), so every row carries
``dominant_period_s`` and a ``phase_lag_identified`` flag, False when ``|lag|`` exceeds a
quarter of that period -- a quarter, because the ambiguity between the zero replica and the
``+-T`` replicas ties at ``+-T/2``, and half way to a tie is as far as a reading should be
trusted. **A row with ``phase_lag_identified == False`` is not a measurement of lateness**;
it is the estimator saying it cannot tell which cycle it is looking at, and the renderer is
expected to flag it rather than print it like any other number.

Both readings ship, per P6-D3: ``phase_lag_s`` (parabolic peak interpolation) and
``phase_lag_raw_s`` (the integer-sample argmax). An interpolated value that is not within
half a sample of its own raw argmax is a bug, and it is only checkable because both are in
the table.

Aggregation is over realizations, never over concatenated records: two realizations joined
end to end have a discontinuity at the seam that is nobody's phase lag. Each realization
gives one lag per (model, DOF, horizon), and the table reports their mean, their standard
deviation, and the count that entered -- a large ``phase_lag_std_s`` beside a small mean is
itself the finding that the lag is not stable across the grid.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmf.config import DataConfig
from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import NormStats, invert_norm
from dmf.data.splits import RealizationKey, Split
from dmf.data.windows import window_spec_from_config
from dmf.eval.phase import cross_correlation_lag, dominant_period
from dmf.eval.runner import _check_norm_provenance
from dmf.models.base import ForecastModel
from dmf.typedefs import FloatArray

__all__ = [
    "DEFAULT_PHASE_REALIZATIONS",
    "IDENTIFIABILITY_FRACTION",
    "PHASE_COLUMNS",
    "PHASE_JOIN_KEYS",
    "build_phase_dataset",
    "evaluate_phase_lag",
    "join_phase_lag",
    "select_phase_realizations",
]

#: Test realizations per regime the phase pass runs on (P6-D3). Thirty-two stride-1
#: realizations is ~180k windows per model per regime; the whole test partition at stride 1
#: would be ~2.7 M, which buys precision the estimator does not need.
DEFAULT_PHASE_REALIZATIONS: int = 32

#: Fraction of the dominant period beyond which a reported lag is declared unidentified.
#: The zero replica and the ``+-T`` replicas tie at ``T/2``; a quarter period is half way to
#: that tie, which is as far as a reading should be trusted (see the module docstring).
IDENTIFIABILITY_FRACTION: float = 0.25

#: Columns the phase table is joined onto :data:`dmf.eval.metrics.METRIC_COLUMNS` by.
PHASE_JOIN_KEYS: tuple[str, ...] = ("model", "regime", "dof", "horizon_samples")

#: Column order of the phase table.
PHASE_COLUMNS: tuple[str, ...] = (
    *PHASE_JOIN_KEYS,
    "horizon_s",
    "phase_lag_s",
    "phase_lag_raw_s",
    "phase_lag_std_s",
    "dominant_period_s",
    "phase_lag_identified",
    "n_realizations_phase",
)


def select_phase_realizations(
    keys: Sequence[RealizationKey], n: int = DEFAULT_PHASE_REALIZATIONS
) -> tuple[RealizationKey, ...]:
    """Choose the sub-sample of test realizations the phase pass runs on.

    Deterministic and RNG-free, and **evenly spaced over the sorted key list** rather than
    taken from its head: the sort is by ``(ss, heading, speed, vessel, seed)``, so the head
    of the list is one sea state at one heading and a lag measured there is not a lag for
    the regime. See the module docstring.

    Args:
        keys: The partition's realization keys, in any order.
        n: Number of realizations to keep, at least 1.

    Returns:
        The chosen keys, ascending. Fewer than ``n`` only when the partition holds fewer.

    Raises:
        ValueError: If ``keys`` is empty or ``n`` is not positive.
    """
    if not keys:
        raise ValueError("keys is empty; there is no partition to sub-sample")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    ordered = sorted(keys)
    if len(ordered) <= n:
        return tuple(ordered)
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(np.int64))
    return tuple(ordered[int(position)] for position in positions)


def build_phase_dataset(
    corpus_root: Path,
    split: Split,
    cfg: DataConfig,
    stats: NormStats,
    *,
    n_realizations: int = DEFAULT_PHASE_REALIZATIONS,
) -> DeckMotionDataset:
    """Build the stride-1 sub-sampled test partition the phase pass reads.

    The window geometry is the arm's own in every respect except the stride, so the models
    see exactly the inputs they were trained for; only the *sampling of origins* is
    refined, which is what the estimator needs and the only thing it needs.

    Args:
        corpus_root: Path to the Parquet corpus.
        split: The regime's split. Its test keys are sub-sampled; its train and validation
            keys are left alone, so nothing here can move a realization across the split.
        cfg: The arm's task definition. Only ``stride`` is overridden.
        stats: Normalisation statistics from this regime's training split. Passed in rather
            than refitted, for the reason :class:`dmf.data.dataset.DeckMotionDataset`
            refuses to fit them on a held-out partition at all.
        n_realizations: Sub-sample size, per :func:`select_phase_realizations`.

    Returns:
        The dataset, at stride 1, over the chosen test realizations only.
    """
    chosen = select_phase_realizations(sorted(split.test_keys), n_realizations)
    narrowed = Split(
        regime=split.regime,
        train_keys=split.train_keys,
        val_keys=split.val_keys,
        test_keys=frozenset(chosen),
    )
    stride_one = DataConfig(
        fs_hz=cfg.fs_hz,
        lookback=cfg.lookback,
        horizons=cfg.horizons,
        target_dofs=cfg.target_dofs,
        input_channels=cfg.input_channels,
        stride=1,
        observation_mode=cfg.observation_mode,
        revin=cfg.revin,
        condition_on_sea_state=cfg.condition_on_sea_state,
    )
    return DeckMotionDataset(
        corpus_root, narrowed, "test", stride_one, window_spec_from_config(stride_one), stats=stats
    )


def _lag_rows(
    *,
    model: str,
    regime: str,
    predictions: FloatArray,
    targets: FloatArray,
    dof_names: tuple[str, ...],
    horizons: tuple[int, ...],
    fs_hz: float,
) -> list[dict[str, object]]:
    """Reduce one model's per-realization series to its phase-table rows.

    Args:
        model: Model label.
        regime: Evaluation regime.
        predictions: Lead-``h`` forecasts, shape ``(R, W, len(horizons), C)``, corpus units.
        targets: The matching truth, same shape, corpus units.
        dof_names: Target channel names, length ``C``.
        horizons: Reported horizons, samples.
        fs_hz: Sampling rate, hertz.

    Returns:
        One row per (DOF, horizon).
    """
    rows: list[dict[str, object]] = []
    for channel, dof in enumerate(dof_names):
        for index, horizon in enumerate(horizons):
            interpolated: list[float] = []
            raw: list[float] = []
            periods: list[float] = []
            for realization in range(predictions.shape[0]):
                pred = predictions[realization, :, index, channel]
                true = targets[realization, :, index, channel]
                interpolated.append(cross_correlation_lag(pred, true, fs_hz, interpolate=True))
                raw.append(cross_correlation_lag(pred, true, fs_hz, interpolate=False))
                periods.append(dominant_period(true, fs_hz))
            lags = np.asarray(interpolated, dtype=np.float64)
            finite = lags[np.isfinite(lags)]
            mean_lag = float(finite.mean()) if finite.size else float("nan")
            # ddof=1: the spread reported is over the realizations that entered, and a
            # single-realization cell has no spread rather than a spread of zero.
            std_lag = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            period = float(np.nanmedian(np.asarray(periods, dtype=np.float64)))
            identified = bool(np.isfinite(mean_lag) and np.isfinite(period)) and (
                abs(mean_lag) <= IDENTIFIABILITY_FRACTION * period
            )
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "dof": dof,
                    "horizon_samples": int(horizon),
                    "horizon_s": horizon / fs_hz,
                    "phase_lag_s": mean_lag,
                    "phase_lag_raw_s": float(np.nanmean(np.asarray(raw, dtype=np.float64))),
                    "phase_lag_std_s": std_lag,
                    "dominant_period_s": period,
                    "phase_lag_identified": identified,
                    "n_realizations_phase": int(finite.size),
                }
            )
    return rows


def evaluate_phase_lag(
    models: Mapping[str, ForecastModel],
    dataset: DeckMotionDataset,
    *,
    horizons: tuple[int, ...],
    fs_hz: float,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Measure every model's phase lag on identical stride-1 windows.

    Mirrors :func:`dmf.eval.runner.evaluate_models`: pre-built models in, one pass over the
    partition, one table out, and the same normalisation-provenance guard at the head of it
    (P3-D11) -- this pass reads held-out data and is subject to exactly the same rule.

    Args:
        models: Point forecasters keyed by results-table label. A probabilistic model is
            passed through :func:`dmf.models.heads.point_view` by the caller, as it is for
            the accuracy pass, so both tables describe the same projection of it.
        dataset: The stride-1 test partition from :func:`build_phase_dataset`. A coarser
            stride is refused: at ``stride = 5`` the estimator's resolution is 0.5 s, which
            cannot resolve the sub-second lags this metric exists to expose.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        fs_hz: Sampling rate, hertz.
        batch_size: Windows per batch. Speed and memory only.
        num_workers: DataLoader worker processes.
        device: Torch device the models run on.

    Returns:
        A table with columns :data:`PHASE_COLUMNS`, one row per (model, DOF, horizon).

    Raises:
        ValueError: If ``models`` is empty, if ``fs_hz`` is not positive, if the dataset is
            not at stride 1, if a horizon is out of range, if the normalisation provenance
            is wrong, or if a model returns the wrong output shape.
        RuntimeError: If the window index does not partition by realization, or the loader
            does not yield every window.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    if not fs_hz > 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    spec = dataset.window_spec
    if spec.stride != 1:
        raise ValueError(
            f"the phase pass needs a stride-1 partition and this one is at stride "
            f"{spec.stride}: the lead-h forecast series is then sampled every "
            f"{spec.stride / fs_hz:.2f} s, which cannot resolve a lag on a ~12 s roll "
            f"period (docs/protocol.md P6-D3). Build it with build_phase_dataset."
        )
    bad = [h for h in horizons if not 1 <= h <= spec.max_horizon]
    if bad:
        raise ValueError(f"horizons {bad} are outside [1, {spec.max_horizon}]")
    _check_norm_provenance(dataset)

    dof_names = tuple(dataset.target_columns)
    n_targets = len(dof_names)
    n_keys = len(dataset.realization_keys)
    per_realization = dataset.windows_per_realization
    if n_keys * per_realization != len(dataset):
        raise RuntimeError(
            f"dataset reports {len(dataset)} windows but {n_keys} realizations x "
            f"{per_realization} windows each is {n_keys * per_realization}"
        )

    lead_index = torch.as_tensor([h - 1 for h in horizons], dtype=torch.int64)
    shape = (n_keys, per_realization, len(horizons), n_targets)
    truth = np.zeros(shape, dtype=np.float32)
    gathered = {name: np.zeros(shape, dtype=np.float32) for name in models}

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
            for x, y, window_mean in loader:
                batch = int(y.shape[0])
                indices = np.arange(offset, offset + batch)
                rows = indices // per_realization
                columns = indices % per_realization
                truth[rows, columns] = y.index_select(1, lead_index).numpy()
                mean = window_mean.double()
                inputs = x.to(torch_device)
                for name, model in models.items():
                    raw = model.forward(inputs)
                    if tuple(raw.shape) != (batch, spec.max_horizon, n_targets):
                        raise ValueError(
                            f"model {name!r} returned shape {tuple(raw.shape)}, expected "
                            f"({batch}, {spec.max_horizon}, {n_targets})"
                        )
                    pred = invert_norm(raw.detach().to("cpu", torch.float64), stats, mean)
                    gathered[name][rows, columns] = pred.index_select(1, lead_index).float().numpy()
                offset += batch
    finally:
        for name, was_training in previous_modes.items():
            module = models[name]
            if isinstance(module, nn.Module) and was_training:
                module.train()

    if offset != len(dataset):
        raise RuntimeError(
            f"the loader yielded {offset} windows but the dataset holds {len(dataset)}; a "
            f"lag measured over a changed window population is not the lag"
        )

    rows_out: list[dict[str, object]] = []
    for name in models:
        rows_out.extend(
            _lag_rows(
                model=name,
                regime=dataset.regime,
                predictions=gathered[name].astype(np.float64),
                targets=truth.astype(np.float64),
                dof_names=dof_names,
                horizons=horizons,
                fs_hz=fs_hz,
            )
        )
    return pd.DataFrame(rows_out, columns=list(PHASE_COLUMNS)).sort_values(
        list(PHASE_JOIN_KEYS), ignore_index=True
    )


def join_phase_lag(metrics: pd.DataFrame, phase: pd.DataFrame) -> pd.DataFrame:
    """Attach the phase columns to an accuracy table, refusing a partial join.

    A left join that silently drops to NaN is the failure mode this function exists to
    prevent: ``phase_lag_s`` is a column §6.1 requires, and a table in which it is blank for
    half the rows reads as "the models have no lag" rather than as "the join keys did not
    match". Every accuracy row must find exactly one phase row, or this raises and names
    the rows that did not.

    Args:
        metrics: Accuracy table carrying :data:`PHASE_JOIN_KEYS` and the
            :data:`dmf.eval.metrics.METRIC_COLUMNS`.
        phase: Output of :func:`evaluate_phase_lag`, or several of them concatenated.

    Returns:
        ``metrics`` with the phase columns appended, in its own row order.

    Raises:
        ValueError: If either frame is missing a join key, if ``phase`` carries duplicate
            keys, or if any accuracy row has no phase row.
    """
    for name, frame in (("metrics", metrics), ("phase", phase)):
        missing = [key for key in PHASE_JOIN_KEYS if key not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing the join columns {missing}")
    duplicated = phase.duplicated(subset=list(PHASE_JOIN_KEYS)).sum()
    if duplicated:
        raise ValueError(
            f"the phase table has {duplicated} duplicate {list(PHASE_JOIN_KEYS)} rows; one "
            f"lag per (model, regime, dof, horizon) is what the join assumes"
        )
    extra = [c for c in PHASE_COLUMNS if c not in PHASE_JOIN_KEYS and c != "horizon_s"]
    joined = metrics.merge(
        phase[[*PHASE_JOIN_KEYS, *extra]],
        on=list(PHASE_JOIN_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined.loc[joined["_merge"] != "both", list(PHASE_JOIN_KEYS)]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} accuracy rows have no phase-lag row, e.g. "
            f"{unmatched.head(3).to_dict('records')}. A blank phase_lag_s column reads as "
            f"'no lag', not as 'not measured', so a partial join is refused."
        )
    return joined.drop(columns=["_merge"])
