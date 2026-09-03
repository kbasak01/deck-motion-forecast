"""Score probabilistic forecasts on identical windows, streaming per-realization sums.

This is the distributional sibling of :func:`dmf.eval.runner.evaluate_models`, and it is a
separate function rather than a widening of that one on purpose. ``evaluate_models``
hard-rejects any output that is not ``(B, H, C_out)``, shares one ``sy``/``syy`` pair across
every accumulator by object identity, and guarantees the reference model's own skill is
bitwise 0.0. Those are load-bearing properties with tests behind them, and a rank-4 branch
threaded through the same loop would put every one of them one refactor away from being
silently untrue.

**The point forecast is not scored here.** Every probabilistic model also ships RMSE, MAE,
skill and ``nrmse`` (P5-D5), and those come from ``evaluate_models`` scoring
:func:`dmf.models.heads.point_view` of the same model -- the *same* code path, the same
persistence denominator and the same realization bootstrap as every other row in the
project. Duplicating the ``sse``/``sae`` accumulation here would produce a second set of
point numbers that could disagree with the first in the last bits, which is exactly the
failure the shared-target-sums check in ``evaluate_models`` exists to prevent.

**Both heads are scored on the same nine levels.** A quantile head reads them off its own
fan; a Gaussian head evaluates ``mean + z(q) * sigma`` at the same levels. The pinball and
CRPS columns are therefore the same estimator applied to two predictive distributions, and
comparing them across heads is comparing like with like. A closed-form Gaussian CRPS would
be a *different* estimator -- exact rather than a nine-level rectangle rule -- and putting
it in the same column as the quantile row's approximation would make the Gaussian head look
better by the quadrature bias alone (P5-D8 records that bias as low).

**Crossing is measured before sorting, in normalised space.** Sorting is applied by
:class:`dmf.models.heads.PredictiveDistribution` on construction, so the raw fan is read for
``crossing_count`` before the distribution is built. Measuring it after would report exactly
zero and measure nothing. Normalised space is equivalent here because
:meth:`dmf.models.heads.PredictiveDistribution.affine` scales by a strictly positive
per-channel constant, which cannot reorder a fan.

Units: predictions are produced in normalised, de-meaned space and mapped back to corpus
units -- degrees, metres, deg/s, m/s -- before any score is taken, so a mean interval width
can be read against a landing limit directly. Coverage and the crossing rate are
dimensionless.

Simulated results only.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.eval.probabilistic import (
    PROBABILISTIC_METRIC_COLUMNS,
    coverage_terms,
    crossing_terms,
    crps_terms,
    pinball_terms,
    probabilistic_table_from_sums,
    width_terms,
    winkler_terms,
)
from dmf.eval.runner import (
    _bootstrap_counts,
    _check_bootstrap_args,
    _check_norm_provenance,
    _in_row_order,
    _key_index,
    _percentile_interval,
)
from dmf.models.base import BaseForecaster
from dmf.models.heads import QUANTILE_FAN_9, PredictiveDistribution
from dmf.typedefs import FloatArray

__all__ = [
    "PROBABILISTIC_TABLE_COLUMNS",
    "ProbAccumulator",
    "bootstrap_picp_ci",
    "evaluate_probabilistic_models",
]

#: Columns the scored table carries, beyond the per-cell metrics themselves.
PROBABILISTIC_TABLE_COLUMNS: tuple[str, ...] = (
    ("model", "head", "n_realizations")
    + PROBABILISTIC_METRIC_COLUMNS
    + ("picp_ci_lo", "picp_ci_hi")
)

#: The levels every head is scored at, whatever it was trained on. See the module docstring.
SCORING_LEVELS: tuple[float, ...] = QUANTILE_FAN_9

#: Nominal miscoverage. ``0.1`` scores the 90 percent interval Gate 5 is read on (P5-D2).
DEFAULT_ALPHA: float = 0.1


@dataclass(frozen=True)
class ProbAccumulator:
    """Per-realization sums of the probabilistic scoring terms.

    Every field is summed over the windows of one realization, so the realization bootstrap
    in :func:`bootstrap_picp_ci` applies without a second pass, and a per-cell breakdown is
    a groupby rather than a re-score. Sums, never means: a mean over realizations is not the
    mean over windows unless every realization contributes equally, and nothing here
    guarantees that for a future non-uniform partition.

    Attributes:
        n_covered: Covered-target counts, ``(n_keys, H, C_out)``, dimensionless.
        width_sum: Summed interval width, ``(n_keys, H, C_out)``, corpus units.
        winkler_sum: Summed Winkler score at ``alpha``, ``(n_keys, H, C_out)``, corpus units.
        crps_sum: Summed approximate CRPS, ``(n_keys, H, C_out)``, corpus units.
        pinball_sum: Summed pinball loss per level, ``(n_keys, H, C_out, Q)``, corpus units.
        crossing_count: Counts of elements whose **raw** fan had at least one adjacent
            inversion, ``(n_keys, H, C_out)``, dimensionless.
        n_per_key: Windows contributing to each realization, ``(n_keys,)``, int64.
        quantiles: The levels scored at, length ``Q``.
        alpha: Nominal miscoverage the interval columns were accumulated at.
    """

    n_covered: Tensor
    width_sum: Tensor
    winkler_sum: Tensor
    crps_sum: Tensor
    pinball_sum: Tensor
    crossing_count: Tensor
    n_per_key: Tensor
    quantiles: tuple[float, ...]
    alpha: float

    @property
    def n_windows(self) -> int:
        """Total windows scored, summed over realizations."""
        return int(self.n_per_key.sum())

    @property
    def n_keys(self) -> int:
        """Number of realizations."""
        return int(self.n_per_key.shape[0])


def _empty_prob_accumulator(
    n_keys: int, max_horizon: int, n_targets: int, n_levels: int, *, alpha: float
) -> ProbAccumulator:
    """Allocate zeroed float64 sums for one model."""
    shape = (n_keys, max_horizon, n_targets)
    return ProbAccumulator(
        n_covered=torch.zeros(shape, dtype=torch.float64),
        width_sum=torch.zeros(shape, dtype=torch.float64),
        winkler_sum=torch.zeros(shape, dtype=torch.float64),
        crps_sum=torch.zeros(shape, dtype=torch.float64),
        pinball_sum=torch.zeros((*shape, n_levels), dtype=torch.float64),
        crossing_count=torch.zeros(shape, dtype=torch.float64),
        n_per_key=torch.zeros((n_keys,), dtype=torch.int64),
        quantiles=SCORING_LEVELS,
        alpha=alpha,
    )


def bootstrap_picp_ci(
    n_covered: Tensor,
    n_per_key: Tensor,
    *,
    horizons: tuple[int, ...],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Bootstrap a confidence interval for PICP by resampling whole realizations.

    PICP is a ratio of sums over realizations -- covered windows over scored windows -- so a
    resample is a count-weighted ratio and ``n_boot`` resamples are one matrix product,
    exactly as :func:`dmf.eval.runner.bootstrap_skill_ci` exploits for skill. Realizations
    are resampled, **never windows**: windows cut from one realization overlap and are not
    independent, and resampling them would give an interval several times too narrow.

    The counts come from :func:`dmf.eval.runner._bootstrap_counts` with the same
    ``(n_keys, n_boot, seed)`` contract, so a PICP interval and a skill interval computed at
    the same seed over the same partition are resampled identically.

    Args:
        n_covered: Covered-target counts per realization, ``(n_keys, H, C_out)``.
        n_per_key: Windows per realization, ``(n_keys,)``.
        horizons: Horizons to report, samples. "Horizon ``h``" reads element ``h - 1``.
        n_boot: Number of resamples.
        ci_level: Central confidence level.
        seed: Seed, so the interval is reproducible.

    Returns:
        Tuple ``(lo, hi)``, each ``(len(horizons), C_out)``, dimensionless.

    Raises:
        ValueError: If a horizon is out of range or the bootstrap arguments are invalid.
    """
    _check_bootstrap_args(
        [tuple(int(d) for d in n_covered.shape)],
        horizons=horizons,
        n_boot=n_boot,
        ci_level=ci_level,
    )
    index = [h - 1 for h in horizons]
    n_keys = int(n_covered.shape[0])
    n_targets = int(n_covered.shape[2])
    covered = np.asarray(n_covered[:, index, :].reshape(n_keys, -1).numpy(), dtype=np.float64)
    per_key = np.asarray(n_per_key.numpy(), dtype=np.float64)
    counts = _bootstrap_counts(n_keys, n_boot, seed)
    numerator = counts @ covered
    denominator = (counts @ per_key)[:, None]
    draws = np.asarray(numerator / denominator, dtype=np.float64)
    return _percentile_interval(draws, ci_level=ci_level, shape=(len(horizons), n_targets))


def evaluate_probabilistic_models(
    models: Mapping[str, BaseForecaster],
    dataset: DeckMotionDataset,
    *,
    horizons: tuple[int, ...],
    fs_hz: float,
    alpha: float = DEFAULT_ALPHA,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str = "cpu",
    n_boot: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 0,
    expected_keys: Sequence[object] | None = None,
) -> tuple[pd.DataFrame, dict[str, ProbAccumulator]]:
    """Score every probabilistic model on identical windows in one pass.

    Args:
        models: Models keyed by results-table label. Each must carry a non-point
            ``head_kind`` and return that head's raw parameters, ``(B, H, C_out, K)``.
        dataset: The partition to score. Its normalisation statistics must have been fitted
            on the training split of this same regime; :func:`_check_norm_provenance`
            asserts both halves before any window is read.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        fs_hz: Sampling rate, hertz, to report each horizon in seconds.
        alpha: Nominal miscoverage; ``0.1`` is the 90 percent interval Gate 5 reads.
        batch_size: Windows per batch. Speed and memory only.
        num_workers: DataLoader worker processes.
        device: Torch device the models run on. Scores are accumulated on the CPU in float64.
        n_boot: Bootstrap resamples for the PICP interval. Whole realizations, never windows.
        ci_level: Central confidence level for ``picp_ci_lo``/``picp_ci_hi``.
        bootstrap_seed: Seed, so the interval is reproducible.
        expected_keys: Realization keys the point pass scored, if it has already run. When
            given, this function raises unless they match its own, in order. The point and
            distributional columns of one row must describe the same windows, and two
            loaders built at different times over a mutated corpus would not.

    Returns:
        Tuple ``(table, accumulators)``. ``table`` carries
        :data:`PROBABILISTIC_TABLE_COLUMNS`, one row per (model, DOF, horizon).

    Raises:
        ValueError: If ``models`` is empty, if any model is a point model, if the
            normalisation provenance is wrong, if a model returns the wrong output shape, or
            if ``expected_keys`` does not match the dataset's.
        RuntimeError: If the window-to-realization mapping does not partition the dataset
            evenly, or if a model's CRPS and pinball sums are inconsistent.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    point_models = sorted(n for n, m in models.items() if m.head_kind == "point")
    if point_models:
        raise ValueError(
            f"models {point_models} carry head_kind='point'; a point model has no "
            f"predictive distribution to score. Score it through "
            f"dmf.eval.runner.evaluate_models instead"
        )
    _check_norm_provenance(dataset)

    spec = dataset.window_spec
    dof_names = tuple(dataset.target_columns)
    n_targets = len(dof_names)
    keys = list(dataset.realization_keys)
    n_keys = len(keys)
    per_realization = dataset.windows_per_realization
    if n_keys * per_realization != len(dataset):
        raise RuntimeError(
            f"dataset reports {len(dataset)} windows but {n_keys} realizations x "
            f"{per_realization} windows each is {n_keys * per_realization}; the window "
            f"index does not partition by realization"
        )
    if expected_keys is not None and list(expected_keys) != keys:
        raise ValueError(
            "the realization keys of the probabilistic pass do not match the point pass's; "
            "the point and distributional columns of a row would describe different windows"
        )

    n_levels = len(SCORING_LEVELS)
    stats = dataset.norm_stats.subset(dof_names)
    scale = torch.as_tensor(stats.scale, dtype=torch.float64).reshape(1, 1, n_targets)
    accumulators = {
        name: _empty_prob_accumulator(n_keys, spec.max_horizon, n_targets, n_levels, alpha=alpha)
        for name in models
    }

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
                key_index = _key_index(offset, batch, per_realization)
                target = y.double()
                mean = window_mean.double()
                inputs = x.to(torch_device)
                target_np = np.asarray(target.numpy(), dtype=np.float64)
                for name, model in models.items():
                    accumulator = accumulators[name]
                    raw = model.forward(inputs).detach().to("cpu", torch.float64)
                    expected = (batch, spec.max_horizon, n_targets, model.n_output_params)
                    if tuple(raw.shape) != expected:
                        raise ValueError(
                            f"model {name!r} returned shape {tuple(raw.shape)}, expected {expected}"
                        )
                    # Read the crossing rate off the RAW fan, before the distribution
                    # sorts it. A positive per-channel scale cannot reorder a fan, so
                    # normalised space gives the same answer as corpus units here.
                    if model.head_kind == "quantile":
                        crossing = crossing_terms(np.asarray(raw.numpy(), dtype=np.float64))
                        accumulator.crossing_count.index_add_(
                            0, key_index, torch.as_tensor(crossing, dtype=torch.float64)
                        )
                    distribution = PredictiveDistribution(
                        raw, model.head_kind, model.quantile_levels
                    ).affine(scale, mean)
                    fan = np.asarray(
                        distribution.quantiles_at(SCORING_LEVELS).numpy(), dtype=np.float64
                    )
                    lower_t, upper_t = distribution.interval(alpha)
                    lower = np.asarray(lower_t.numpy(), dtype=np.float64)
                    upper = np.asarray(upper_t.numpy(), dtype=np.float64)

                    accumulator.n_covered.index_add_(
                        0,
                        key_index,
                        torch.as_tensor(
                            coverage_terms(lower, upper, target_np), dtype=torch.float64
                        ),
                    )
                    accumulator.width_sum.index_add_(
                        0,
                        key_index,
                        torch.as_tensor(width_terms(lower, upper), dtype=torch.float64),
                    )
                    accumulator.winkler_sum.index_add_(
                        0,
                        key_index,
                        torch.as_tensor(
                            winkler_terms(lower, upper, target_np, alpha),
                            dtype=torch.float64,
                        ),
                    )
                    accumulator.crps_sum.index_add_(
                        0,
                        key_index,
                        torch.as_tensor(
                            crps_terms(fan, target_np, SCORING_LEVELS), dtype=torch.float64
                        ),
                    )
                    accumulator.pinball_sum.index_add_(
                        0,
                        key_index,
                        torch.as_tensor(
                            pinball_terms(fan, target_np, SCORING_LEVELS),
                            dtype=torch.float64,
                        ),
                    )
                    accumulator.n_per_key.index_add_(
                        0, key_index, torch.ones((batch,), dtype=torch.int64)
                    )
                offset += batch
    finally:
        for name, was_training in previous_modes.items():
            module = models[name]
            if isinstance(module, nn.Module):
                module.train(was_training)

    frames: list[pd.DataFrame] = []
    for name, accumulator in accumulators.items():
        _check_crps_pinball_consistency(name, accumulator)
        # The per-realization axis is passed through, not pre-summed: the table builder
        # reduces it itself, and that axis is what the realization bootstrap resamples.
        table = probabilistic_table_from_sums(
            n_covered=np.asarray(accumulator.n_covered.numpy(), dtype=np.float64),
            width_sum=np.asarray(accumulator.width_sum.numpy(), dtype=np.float64),
            winkler_sum=np.asarray(accumulator.winkler_sum.numpy(), dtype=np.float64),
            crps_sum=np.asarray(accumulator.crps_sum.numpy(), dtype=np.float64),
            pinball_sum=np.asarray(accumulator.pinball_sum.numpy(), dtype=np.float64),
            crossing_count=np.asarray(accumulator.crossing_count.numpy(), dtype=np.float64),
            n=accumulator.n_windows,
            dof_names=dof_names,
            horizons=horizons,
            fs_hz=fs_hz,
            alpha=alpha,
        )
        lo, hi = bootstrap_picp_ci(
            accumulator.n_covered,
            accumulator.n_per_key,
            horizons=horizons,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=bootstrap_seed,
        )
        table.insert(0, "model", name)
        table.insert(1, "head", models[name].head_kind)
        table.insert(2, "n_realizations", n_keys)
        table["picp_ci_lo"] = _in_row_order(lo, n_targets, len(horizons))
        table["picp_ci_hi"] = _in_row_order(hi, n_targets, len(horizons))
        frames.append(table)

    return pd.concat(frames, ignore_index=True)[list(PROBABILISTIC_TABLE_COLUMNS)], accumulators


def _check_crps_pinball_consistency(name: str, accumulator: ProbAccumulator) -> None:
    """Assert the CRPS and pinball sums describe the same fan.

    P5-D8 records that ``probabilistic_table_from_sums`` cannot make this check -- it takes
    ``crps_sum`` as a free input so a closed-form Gaussian CRPS could be streamed through
    the same table -- and that the guard belongs where the head kind is known. This is that
    place. Both columns here are computed from one fan by the rectangle rule, so
    ``crps == 2 * mean_q(pinball)`` holds by construction and a violation means the two were
    accumulated from different predictions.

    Args:
        name: Model label, for the error message.
        accumulator: The sums to check.

    Raises:
        RuntimeError: If the two columns are inconsistent beyond float64 accumulation error.
    """
    derived = 2.0 * accumulator.pinball_sum.mean(dim=-1)
    if not torch.allclose(derived, accumulator.crps_sum, rtol=1e-9, atol=1e-9):
        worst = float((derived - accumulator.crps_sum).abs().max())
        raise RuntimeError(
            f"model {name!r}: crps_sum and pinball_sum disagree by up to {worst:.3e}; they "
            f"are computed from one fan by one quadrature and must agree by construction, "
            f"so they were accumulated from different predictions"
        )
