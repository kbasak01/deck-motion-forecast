"""Extract the forecast traces the Phase 9 headline figure is drawn from.

Separated from :mod:`dmf.viz.forecast_plots` on purpose. Drawing needs nothing but arrays;
getting those arrays needs the corpus, the committed checkpoints and a GPU-sized forward
pass. Keeping the two apart is what lets ``make figures`` re-render from a small committed
``.npz`` without any of that, the same property ``make report`` gives the Pareto figure --
and it makes "is the figure a function of the committed trace?" answerable in a second.

**No metric is computed here.** The coverage annotated on the figure is the hit rate of the
drawn span, computed at draw time by :mod:`dmf.viz.forecast_plots`; it is not a PICP, which
is a mean over a whole test partition and lives in ``results/e03/probabilistic.csv``.

Units: traces are returned in **corpus units** -- degrees for roll and pitch, degrees per
second for their rates, metres for heave, metres per second for heave rate -- and time axes
are **seconds** from the start of the realization. Simulated results only.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from dmf.config import ExperimentConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import RealizationKey, Split, build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.models.heads import PredictiveDistribution
from dmf.train.closed_form import TrainingMoments
from dmf.train.experiment import load_or_fit
from dmf.typedefs import FloatArray

__all__ = [
    "HEADLINE_ALPHA",
    "HEADLINE_CELL",
    "HEADLINE_DOF",
    "HEADLINE_HORIZON_S",
    "HEADLINE_MODELS",
    "HEADLINE_SPAN_S",
    "ForecastTrace",
    "extract_trace",
    "load_traces",
    "save_traces",
]

#: The grid cell the headline figure is drawn at, as ``(sea state, heading, speed, vessel)``.
#: ``docs/IMPLEMENTATION_PLAN.md`` Phase 9 item 2 fixes sea state and heading; the speed is
#: the fastest in the grid, where the encounter frequency is furthest from the zero-speed
#: case. 90 deg is beam seas, which is where roll is **largest** -- the roll heading factor
#: is ``hypot(sin(90 deg), 0.05) = 1.001``, so roll here is a real signal and not on the
#: P1-D2 residual floor. Pitch at this heading *is* on that floor, which is why the figure
#: draws roll and why a pitch panel here would be meaningless.
HEADLINE_CELL: tuple[str, float, float, str] = ("SS5", 90.0, 12.0, "frigate")

#: Channel drawn. Plan item 2.
HEADLINE_DOF: str = "roll"

#: Lead time drawn, seconds. Plan item 2.
HEADLINE_HORIZON_S: float = 3.0

#: Nominal interval miscoverage. 1 - alpha = 90 %, matching Gate 5 and every PICP column.
HEADLINE_ALPHA: float = 0.1

#: Seconds of forecast drawn. Long enough to show several roll periods (the frigate's is
#: about 9 s), short enough that individual misses are visible rather than a smear.
HEADLINE_SPAN_S: float = 60.0

#: Models drawn, and why these two. ``dlinear_quantile`` is the best-calibrated family on
#: three regimes of four and the only family that survives the Phase 8 cross-generator test;
#: ``tcn_quantile`` is an order of magnitude sharper and over-covers in exactly this 1-5 s
#: band (P5-D15). Showing one without the other would be choosing the flattering half.
HEADLINE_MODELS: tuple[str, ...] = ("dlinear_quantile", "tcn_quantile")


@dataclass(frozen=True)
class ForecastTrace:
    """One model's forecast of one channel at one lead time, over a contiguous span.

    Attributes:
        model: Model label, e.g. ``"dlinear_quantile"``.
        regime: Evaluation regime the checkpoint was trained under, e.g. ``"id"``.
        seed: Training seed of the checkpoint.
        cell: The grid cell drawn, as ``"<sea state>|<heading_deg>|<speed_kn>|<vessel>"``.
            Carried on the artifact so the figure's caption is checkable from the committed
            trace rather than only from the constants in this module.
        dof: Channel drawn, e.g. ``"roll"``.
        horizon_s: Lead time drawn, seconds.
        t_s: Valid time of each forecast, seconds from the start of the realization.
        truth: The realized motion at those times, corpus units.
        median: Predictive median, corpus units.
        lower: Lower bound of the ``1 - alpha`` interval, corpus units.
        upper: Upper bound of the same interval, corpus units.
        persistence: The persistence forecast -- the value at the forecast origin --
            for the same times, corpus units.
    """

    model: str
    regime: str
    seed: int
    cell: str
    dof: str
    horizon_s: float
    t_s: FloatArray
    truth: FloatArray
    median: FloatArray
    lower: FloatArray
    upper: FloatArray
    persistence: FloatArray


def _cell_realization(dataset: DeckMotionDataset, cell: tuple[str, float, float, str]) -> int:
    """Return the index of the first test realization in the requested grid cell.

    Args:
        dataset: Test partition to search.
        cell: ``(sea state, heading_deg, speed_kn, vessel)``.

    Returns:
        Index into ``dataset.realization_keys``.

    Raises:
        ValueError: If the partition holds no realization in that cell. This is the guard
            that matters: ``unseen_heading`` holds 90 deg out of training, so asking for a
            beam-seas cell from the wrong partition must fail loudly rather than silently
            drawing a different heading.
    """
    keys: Sequence[RealizationKey] = dataset.realization_keys
    for index, key in enumerate(keys):
        if (key[0], float(key[1]), float(key[2]), key[3]) == (cell[0], cell[1], cell[2], cell[3]):
            return index
    raise ValueError(
        f"no realization in cell {cell} in the {dataset.partition!r} partition of regime "
        f"{dataset.regime!r}; it holds {len(keys)} realizations"
    )


def extract_trace(
    cfg: ExperimentConfig,
    *,
    corpus_root: Path,
    checkpoint_root: Path,
    regime: str,
    model_labels: Sequence[str],
    cell: tuple[str, float, float, str] = HEADLINE_CELL,
    dof: str = HEADLINE_DOF,
    horizon_s: float = HEADLINE_HORIZON_S,
    span_s: float = HEADLINE_SPAN_S,
    alpha: float = HEADLINE_ALPHA,
    seed: int = 0,
    device: str = "cpu",
) -> list[ForecastTrace]:
    """Forecast one cell with committed checkpoints and return the drawable traces.

    Nothing is trained. ``load_or_fit`` resolves each SGD model's committed ``state_dict``
    and raises rather than falling back to a random initialisation, so a missing checkpoint
    is a hard error here exactly as it is in ``make eval``.

    Args:
        cfg: The experiment whose models and data settings to use.
        corpus_root: Root of the generated corpus.
        checkpoint_root: Root of the committed checkpoints.
        regime: Evaluation regime, e.g. ``"id"``.
        model_labels: Labels of the models to trace, as named in the experiment config.
        cell: ``(sea state, heading_deg, speed_kn, vessel)`` to draw.
        dof: Channel name to draw.
        horizon_s: Lead time to draw, seconds.
        span_s: Length of the drawn span, seconds.
        alpha: Interval miscoverage, so the band is ``1 - alpha``.
        seed: Training seed of the checkpoint to load.
        device: Torch device for the forward pass.

    Returns:
        One trace per requested model, in the order requested.

    Raises:
        ValueError: If a label is not in the config, if ``dof`` is not a target channel, if
            the requested lead time is not a configured horizon, or if the cell is absent.
    """
    spec = window_spec_from_config(cfg.data)
    fs_hz = cfg.data.fs_hz
    horizon_samples = int(round(horizon_s * fs_hz))
    if horizon_samples not in spec.horizons:
        raise ValueError(
            f"{horizon_s} s is {horizon_samples} samples at {fs_hz} Hz, which is not a "
            f"configured horizon {spec.horizons}"
        )

    by_label = {model.label: model for model in cfg.models}
    missing = [label for label in model_labels if label not in by_label]
    if missing:
        raise ValueError(f"models {missing} are not in experiment {cfg.name!r}")

    split: Split = build_split(load_manifest(corpus_root), regime)  # type: ignore[arg-type]
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats)
    test = DeckMotionDataset(corpus_root, split, "test", cfg.data, spec, stats=train.norm_stats)

    targets = test.target_columns
    if dof not in targets:
        raise ValueError(f"{dof!r} is not a target channel; targets are {targets}")
    channel = targets.index(dof)

    realization = _cell_realization(test, cell)
    per_realization = len(test) // len(test.realization_keys)
    n_windows = min(int(round(span_s * fs_hz / spec.stride)), per_realization)
    indices = [realization * per_realization + offset for offset in range(n_windows)]

    x = torch.stack([test[i][0] for i in indices])
    y = torch.stack([test[i][1] for i in indices]).double()
    window_mean = torch.stack([test[i][2] for i in indices]).double()

    scale_all = test.norm_stats.subset(targets).scale
    scale = torch.as_tensor(scale_all, dtype=torch.float64).reshape(1, 1, len(targets))

    # The forecast origin is the last lookback sample; persistence is its value, held. x is
    # de-meaned and scaled, so it is mapped back the same way the distribution is.
    origin_value = (
        x[:, -1, channel].double() * float(scale_all[channel]) + window_mean[:, 0, channel]
    )
    start_samples = np.array([test.describe_window(i)[1] for i in indices], dtype=np.float64)
    t_s = (start_samples + spec.lookback - 1 + horizon_samples) / fs_hz

    holder: dict[str, TrainingMoments] = {}
    traces: list[ForecastTrace] = []
    for label in model_labels:
        records = load_or_fit(
            by_label[label],
            spec=spec,
            train=train,
            val=val,
            experiment=cfg,
            moments_holder=holder,
            device=device,
            checkpoint_dir=checkpoint_root / cfg.name / regime,
        )
        record = next((r for r in records if r.seed == seed), None)
        if record is None:
            raise ValueError(f"no seed {seed} run for {label!r}; got {[r.seed for r in records]}")
        model = record.model
        with torch.no_grad():
            raw = model.forward(x.to(device)).detach().to("cpu", torch.float64)
        distribution = PredictiveDistribution(raw, model.head_kind, model.quantile_levels).affine(
            scale, window_mean
        )
        lower_t, upper_t = distribution.interval(alpha)
        median_t = distribution.quantiles_at((0.5,))[..., 0]
        step = horizon_samples - 1
        traces.append(
            ForecastTrace(
                model=label,
                regime=regime,
                seed=seed,
                cell="|".join(str(part) for part in cell),
                dof=dof,
                horizon_s=horizon_s,
                t_s=t_s,
                truth=np.asarray(y[:, step, channel].numpy(), dtype=np.float64),
                median=np.asarray(median_t[:, step, channel].numpy(), dtype=np.float64),
                lower=np.asarray(lower_t[:, step, channel].numpy(), dtype=np.float64),
                upper=np.asarray(upper_t[:, step, channel].numpy(), dtype=np.float64),
                persistence=np.asarray(origin_value.numpy(), dtype=np.float64),
            )
        )
    return traces


def save_traces(path: Path, traces: Sequence[ForecastTrace]) -> None:
    """Write traces to a compressed ``.npz`` so the figure can be re-rendered without a GPU.

    Args:
        path: Destination file.
        traces: Traces to write.

    Raises:
        ValueError: If ``traces`` is empty.
    """
    if not traces:
        raise ValueError("traces must be non-empty")
    payload: dict[str, npt.NDArray[Any]] = {
        "labels": np.array([t.model for t in traces]),
        "regimes": np.array([t.regime for t in traces]),
        "seeds": np.array([t.seed for t in traces], dtype=np.int64),
        "cells": np.array([t.cell for t in traces]),
        "dofs": np.array([t.dof for t in traces]),
        "horizons_s": np.array([t.horizon_s for t in traces], dtype=np.float64),
    }
    for index, trace in enumerate(traces):
        for field in ("t_s", "truth", "median", "lower", "upper", "persistence"):
            payload[f"{field}_{index}"] = np.asarray(getattr(trace, field))
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy types the second positional parameter as `allow_pickle`, so a **kwargs of
    # array names does not type-check against the stub. The call is the documented one.
    cast(Any, np.savez_compressed)(path, **payload)


def load_traces(path: Path) -> list[ForecastTrace]:
    """Read traces written by :func:`save_traces`.

    Args:
        path: File to read.

    Returns:
        The traces, in the order they were written.
    """
    with np.load(path, allow_pickle=False) as data:
        labels = [str(v) for v in data["labels"]]
        regimes = [str(v) for v in data["regimes"]]
        seeds = [int(v) for v in data["seeds"]]
        cells = [str(v) for v in data["cells"]]
        dofs = [str(v) for v in data["dofs"]]
        horizons = [float(v) for v in data["horizons_s"]]
        return [
            ForecastTrace(
                model=labels[index],
                regime=regimes[index],
                seed=seeds[index],
                cell=cells[index],
                dof=dofs[index],
                horizon_s=horizons[index],
                t_s=data[f"t_s_{index}"],
                truth=data[f"truth_{index}"],
                median=data[f"median_{index}"],
                lower=data[f"lower_{index}"],
                upper=data[f"upper_{index}"],
                persistence=data[f"persistence_{index}"],
            )
            for index in range(len(labels))
        ]
