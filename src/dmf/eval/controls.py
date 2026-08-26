"""Integrity controls -- the checks run before believing any result.

Three controls guard this project, and none of them is a unit test of a function; each is a
whole-pipeline experiment whose expected outcome is known in advance:

1. **Pipeline sanity** (here, Phase 2). Persistence evaluated through the full dataset
   pipeline must reproduce persistence computed directly on the raw Parquet arrays. An
   off-by-one in the windowing, a mis-inverted normalisation, or a channel-order mismatch
   each break this, and none of the three is visible in a loss curve.
2. **Shuffle control** (Phase 4). Retrain the best model on time-shuffled targets; the skill
   score must collapse to approximately zero. If it does not, there is leakage.
3. **Untrained control** (Phase 4). A randomly initialised model must score worse than
   persistence.

Controls 2 and 3 arrive with the training loop and get the same home.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import invert_norm
from dmf.data.splits import RealizationKey
from dmf.data.windows import window_start_indices
from dmf.sim.generate import RealizationSpec, realization_path
from dmf.typedefs import FloatArray

__all__ = ["PipelineSanityResult", "persistence_pipeline_sanity"]


@dataclass(frozen=True)
class PipelineSanityResult:
    """Outcome of the persistence pipeline-sanity control.

    Attributes:
        rmse_pipeline: Per-(horizon, target channel) persistence RMSE measured through the
            dataset, shape ``(H, C_out)``, in corpus units (degrees for roll and pitch,
            metres for heave).
        rmse_raw: The same quantity computed directly from the Parquet files, shape
            ``(H, C_out)``, same units.
        max_rel_diff: Largest relative disagreement between the two, dimensionless.
        n_windows: Number of windows scored. Identical on both paths, or the control
            raises.
        dof_names: Target channel names, length ``C_out``.
    """

    rmse_pipeline: FloatArray
    rmse_raw: FloatArray
    max_rel_diff: float
    n_windows: int
    dof_names: tuple[str, ...]


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
    dataset: DeckMotionDataset, batch_size: int
) -> tuple[FloatArray, int]:
    """Accumulate persistence squared error through the dataset pipeline.

    The forecast is formed in *normalised* space, exactly where a model's would be
    (``x[:, -1:, :C_out]`` broadcast over the horizon, which is what
    :class:`dmf.models.persistence.Persistence` is documented to compute), and is then
    mapped back to corpus units through :func:`dmf.data.normalize.invert_norm`. It is
    computed inline rather than by importing the model class because
    ``Persistence.forward`` is Phase 3 work; Phase 3 re-runs this control against the real
    model.

    Args:
        dataset: The partition to score.
        batch_size: Windows per batch. Affects speed only.

    Returns:
        Tuple ``(sse, n_windows)`` where ``sse`` has shape ``(H, C_out)`` in squared corpus
        units.
    """
    stats = dataset.norm_stats.subset(dataset.target_columns)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, seed=0)
    sse = torch.zeros(
        (dataset.window_spec.max_horizon, len(dataset.target_columns)), dtype=torch.float64
    )
    count = 0
    for x, y, window_mean in loader:
        n_out = y.shape[2]
        pred_norm = x[:, -1:, :n_out].expand(-1, y.shape[1], -1)
        pred = invert_norm(pred_norm, stats, window_mean)
        sse += torch.square(pred.double() - y.double()).sum(dim=0)
        count += int(y.shape[0])
    return sse.numpy(), count


def persistence_pipeline_sanity(
    dataset: DeckMotionDataset,
    corpus_root: Path,
    *,
    rtol: float = 1e-6,
    batch_size: int = 512,
) -> PipelineSanityResult:
    """Check that persistence through the pipeline matches persistence on raw arrays.

    Gate 2 criterion 5. Two paths that share no code beyond the realization key list and the
    window geometry must agree to ``rtol`` on the per-(horizon, DOF) RMSE.

    The two paths are not bitwise identical by construction: the pipeline stores its
    de-meaned, scaled input as float32, so the forecast it recovers carries a relative
    rounding error of order ``2**-24``. ``rtol`` is stated rather than assumed for that
    reason.

    Args:
        dataset: The partition to score. Any partition may be used; the ``test`` partition
            of a regime is the interesting one.
        corpus_root: Dataset root, re-read independently by the raw path.
        rtol: Largest tolerated relative disagreement, dimensionless.
        batch_size: Windows per batch on the pipeline path. Affects speed only.

    Returns:
        The measured RMSE on both paths and their largest relative disagreement.

    Raises:
        AssertionError: If the two paths disagree beyond ``rtol``, naming the worst
            (horizon, DOF) cell, or if they score different numbers of windows.
    """
    if rtol <= 0.0:
        raise ValueError(f"rtol must be positive, got {rtol}")
    sse_pipeline, n_pipeline = _pipeline_persistence_sse(dataset, batch_size)
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
    )
