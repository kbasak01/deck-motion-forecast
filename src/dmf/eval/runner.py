"""One evaluation pass that scores every model on identical windows.

Why this module exists at all, rather than a loop that calls a metric function per model:

**The skill denominator must be structurally correct, not correct by inspection.** Skill is
``1 - MSE_model/MSE_persistence``, and the single most common way a skill score ends up
flattering a model is for the denominator to be measured over a different window set --
a different partition, a different batch order, a different ``drop_last``. Here every model
is scored inside the same batch loop against the same ``y``, and ``rmse_persistence`` for
every row is read out of ``accumulators[persistence_key]``. There is no code path in which
the two could be different window sets. It also collapses ``n_models x n_regimes``
evaluation passes to ``n_regimes``, which is the cheap part of the argument.

**Errors accumulate per realization key, not globally.** ``EvalAccumulator.sse`` is
``(n_keys, H, C_out)`` float64 -- a few megabytes per accumulator at any corpus geometry
this project will use, so per-key accumulation is free. It buys three things at zero runtime
cost: the per-cell breakdown (:func:`per_cell_metrics`), the realization-level bootstrap
confidence interval (:func:`bootstrap_skill_ci`), and the realization-level checks Phase 6
needs -- none of which then requires re-evaluating anything.

**``n_windows`` is a window count, not an independent-sample count.** Windows are cut at a
stride far shorter than the lookback, so consecutive windows share almost all of their input
samples and their forecast targets overlap; a test partition holds two to three orders of
magnitude more windows than it holds independently simulated realizations. Every uncertainty
statement in this module therefore resamples whole realizations. Treating ``n_windows`` as a
sample size would understate every interval by more than an order of magnitude. (No literal
count appears here on purpose: the one that used to sit in this paragraph was superseded by
the P3-D4 horizon change and went stale in place. Counts are measured, never narrated.)

**Horizon convention.** "Horizon ``h``" is the error at lead time exactly ``h`` samples, so
it reads element ``h - 1`` of the horizon axis. See :mod:`dmf.eval.metrics`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.normalize import invert_norm, is_train_partition
from dmf.eval.metrics import METRIC_COLUMNS, metrics_table_from_sums
from dmf.models.base import ForecastModel
from dmf.typedefs import FloatArray

__all__ = [
    "CELL_COLUMNS",
    "EvalAccumulator",
    "bootstrap_skill_ci",
    "evaluate_models",
    "marginalize_cells",
    "paired_skill_difference_ci",
    "per_cell_metrics",
]

#: Grid axes a realization key can be broken out by, in
#: :data:`dmf.data.splits.RealizationKey` order minus the seed. The ``id`` regime pools
#: 4 headings x 4 sea states x 3 speeds, and per ``docs/protocol.md`` P1-D2 roll in head
#: seas sits on the residual floor, so a pooled roll skill of 0.85 could be 0.93 at beam and
#: 0.6 at head, or the reverse. The pooled number alone cannot tell you which, and the
#: follow-on decision depends on it.
CELL_COLUMNS: tuple[str, ...] = ("vessel", "ss", "heading_deg", "speed_kn")

#: Index of each cell axis within a :data:`dmf.data.splits.RealizationKey`.
_KEY_FIELD_INDEX: dict[str, int] = {"ss": 0, "heading_deg": 1, "speed_kn": 2, "vessel": 3}


@dataclass(frozen=True)
class EvalAccumulator:
    """Per-realization error sums for one model over one partition.

    Sums rather than arrays: a production test partition holds hundreds of thousands of
    windows, so storing the predictions would cost gigabytes per model per regime while
    these sums cost megabytes. The exact counts are measured and written to
    ``baselines_by_seed.csv``; they are not narrated here, because the last narrated one
    went stale in place when P3-D4 changed the horizon set.

    Attributes:
        sse: Summed squared error per realization, shape ``(n_keys, H, C_out)``, float64,
            in **squared corpus units** (degrees squared for roll and pitch, metres squared
            for heave).
        sae: Summed absolute error per realization, shape ``(n_keys, H, C_out)``, float64,
            in corpus units.
        n_per_key: Windows scored per realization, shape ``(n_keys,)``, int64. Uniform
            across a partition by construction; :func:`evaluate_models` raises if it is
            not, since a non-uniform count means the window-index-to-key mapping is wrong.
    """

    sse: Tensor
    sae: Tensor
    n_per_key: Tensor

    @property
    def n_windows(self) -> int:
        """Total windows scored, summed over realizations."""
        return int(self.n_per_key.sum().item())

    @property
    def n_keys(self) -> int:
        """Number of realizations scored."""
        return int(self.n_per_key.shape[0])

    def totals(self) -> tuple[FloatArray, FloatArray]:
        """Reduce over realizations.

        Returns:
            Tuple ``(sse, sae)``, each of shape ``(H, C_out)`` in squared corpus units and
            corpus units respectively.
        """
        return (
            self.sse.sum(dim=0).numpy().astype(np.float64),
            self.sae.sum(dim=0).numpy().astype(np.float64),
        )


def _empty_accumulator(n_keys: int, max_horizon: int, n_targets: int) -> EvalAccumulator:
    """Allocate a zeroed accumulator.

    Args:
        n_keys: Number of realizations in the partition.
        max_horizon: Forecast length ``H``, samples.
        n_targets: Target channel count ``C_out``.

    Returns:
        An accumulator of zeros.
    """
    shape = (n_keys, max_horizon, n_targets)
    return EvalAccumulator(
        sse=torch.zeros(shape, dtype=torch.float64),
        sae=torch.zeros(shape, dtype=torch.float64),
        n_per_key=torch.zeros((n_keys,), dtype=torch.int64),
    )


def _key_index(offset: int, batch: int, windows_per_realization: int) -> Tensor:
    """Map a contiguous run of window indices onto realization indices.

    Valid only for a loader built with ``shuffle=False`` and ``drop_last=False``, which is
    why :func:`evaluate_models` builds the loader itself rather than accepting one.
    :class:`dmf.data.dataset.DeckMotionDataset` lays windows out realization-major, so
    ``key_index = index // windows_per_realization``.

    Args:
        offset: Index of the first window in this batch.
        batch: Batch size.
        windows_per_realization: Windows cut from each realization.

    Returns:
        Realization indices, shape ``(batch,)``, int64.
    """
    return torch.arange(offset, offset + batch, dtype=torch.int64) // windows_per_realization


def _check_norm_provenance(dataset: DeckMotionDataset) -> None:
    """Refuse to score a partition whose normalisation scale has the wrong provenance.

    Two distinct failures, neither of which any earlier check can catch:

    1. **Not a training partition.** :func:`dmf.data.normalize.build_norm_stats` already
       refuses to *construct* such statistics, so reaching this branch means the label was
       replaced after construction (``dataclasses.replace``) or the object was rebuilt from
       a checkpoint without going through the constructor. Cheap to re-assert here, and
       this is the last point before the numbers are produced.
    2. **The right partition of the wrong regime.** This one is genuinely new. Nothing
       upstream can catch it: :class:`dmf.data.dataset.DeckMotionDataset` requires *some*
       train statistics for a held-out partition but has no way to know which regime the
       caller meant, and ``id/train`` statistics are perfectly valid statistics. Yet
       ``id``'s training pool spans SS6 and the 90 deg beam heading, which
       ``unseen_seastate`` and ``unseen_heading`` deliberately hold out, so an ``id/train``
       scale applied to those regimes' test partitions is fitted over exactly the variance
       the regime exists to hold out. Where it is not a leak it is still a silent change to
       every RMSE and to every skill denominator in the table.

    The label is parsed defensively rather than by assuming ``split("/")`` yields two
    parts. ``dataset.py`` writes ``f"{regime}/train"``, but ``fitted_on`` is a free-form
    string on the public :class:`dmf.data.normalize.NormStats` constructor and a bare
    ``"train"`` passes :func:`dmf.data.normalize.is_train_partition`. A label that records
    no regime cannot be checked against the dataset's, so it is refused rather than waved
    through -- an integrity guard that a relabelling can switch off is not a guard.

    Args:
        dataset: The partition about to be scored.

    Raises:
        ValueError: If the statistics were not fitted on a training partition, if their
            provenance label does not name a regime, or if it names a different regime
            from the dataset's.
    """
    label = dataset.norm_stats.fitted_on
    if not is_train_partition(label):
        raise ValueError(
            f"the dataset's normalisation statistics record fitted_on={label!r}, which is "
            f"not a training partition; expected {dataset.regime!r}/train. Scaling a "
            f"held-out partition by statistics that saw held-out variance inflates every "
            f"RMSE and every skill score in this table (CLAUDE.md non-negotiable 3)"
        )
    parts = label.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"the dataset's normalisation statistics record fitted_on={label!r}, which "
            f"does not have the '<regime>/train' shape that "
            f"dmf.data.dataset.DeckMotionDataset writes; expected {dataset.regime!r}/"
            f"train. The regime the scale was fitted on cannot be recovered from this "
            f"label, so it cannot be checked against the {dataset.regime!r} partition "
            f"being scored, and an unverifiable provenance is refused rather than assumed "
            f"(CLAUDE.md non-negotiable 3)"
        )
    if parts[0] != dataset.regime:
        raise ValueError(
            f"the dataset holds the {dataset.regime!r} regime but its normalisation "
            f"statistics record fitted_on={label!r}; expected {dataset.regime!r}/train. "
            f"Another regime's training pool can span the realizations this one holds out "
            f"-- id/train covers SS6 and the 90 deg beam heading, which unseen_seastate "
            f"and unseen_heading exist to withhold -- so a cross-regime scale is fitted "
            f"over the very variance the regime is testing generalisation to, and where "
            f"it is not that, it still silently moves every RMSE and every skill "
            f"denominator (CLAUDE.md non-negotiable 3)"
        )


def evaluate_models(
    models: Mapping[str, ForecastModel],
    dataset: DeckMotionDataset,
    *,
    persistence_key: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str = "cpu",
    n_boot: int = 1000,
    ci_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, EvalAccumulator]]:
    """Score every model on identical windows in one pass over the partition.

    Predictions are produced in normalised space and mapped back to corpus units with
    :func:`dmf.data.normalize.invert_norm` before any error is taken, because a normalised
    RMSE is not comparable across channels and cannot be checked against a landing limit.
    Errors are accumulated in float64; the dataset's float32 storage means that conversion
    is exact.

    Skill is formed from sums, not from RMSE: ``1 - sse_model/sse_persistence``, in which
    the window count cancels. The reference model's own skill is therefore **bitwise** 0.0,
    and this function raises if it is not -- a free check that catches any accidental
    divergence between the reference accumulator and the reference model's own row.

    Args:
        models: Models to score, keyed by the label used in the results table. Every model
            must accept ``(B, L, C_in)`` and return ``(B, H, C_out)``. ``nn.Module``
            entries are moved to ``device`` and switched to eval mode for the duration of
            the call, and their prior training mode is restored on the way out.
        dataset: The partition to score, normally the ``test`` partition of a regime. Its
            normalisation statistics must have been fitted on the training split **of this
            same regime**, and :func:`_check_norm_provenance` asserts both halves of that
            before any window is read. Note that nothing checks it at window-construction
            time: :class:`dmf.data.dataset.DeckMotionDataset` scales each window inline
            against ``stats.scale`` and never calls :func:`dmf.data.normalize.apply_norm`,
            so that function's provenance guard does not run on this path. What the dataset
            contributes is structural: its constructor refuses to build a ``val`` or
            ``test`` partition without an explicit ``stats`` argument, and only a ``train``
            partition may fit its own, so a held-out partition can only ever be scaled by
            statistics handed down from *some* training partition. Which one is recorded in
            ``dataset.norm_stats.fitted_on``, and matching that against ``dataset.regime``
            is the part the dataset cannot do for itself.
        persistence_key: Key into ``models`` naming the reference baseline. Its accumulator
            supplies the denominator of every skill score in the returned table.
        horizons: Horizons to report, samples, each in ``[1, H]``. "Horizon ``h``" is the
            error at lead time exactly ``h``.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds.
        batch_size: Windows per batch. Affects speed and memory only.
        num_workers: DataLoader worker processes.
        device: Torch device the models run on. Errors are always accumulated on the CPU in
            float64.
        n_boot: Bootstrap resamples for the skill confidence interval. Whole realizations
            are resampled, never windows.
        ci_level: Central confidence level for ``skill_ci_lo``/``skill_ci_hi``.
        bootstrap_seed: Seed for the bootstrap, so the interval is reproducible.

    Returns:
        Tuple ``(table, accumulators)``. ``table`` has one row per (model, DOF, horizon)
        with columns ``model``, ``n_realizations`` and :data:`dmf.eval.metrics.METRIC_COLUMNS`
        plus ``skill_ci_lo`` and ``skill_ci_hi``. ``accumulators`` holds the per-realization
        error sums for every model, keyed as ``models`` is, so that per-cell breakdowns and
        further resampling need no second pass.

    Raises:
        ValueError: If ``models`` is empty, if ``persistence_key`` is not in ``models``, if
            ``dataset``'s normalisation statistics were not fitted on the training split of
            ``dataset``'s own regime, if a model returns the wrong output shape, or if a
            requested horizon is out of range.
        RuntimeError: If the window-to-realization mapping does not partition the dataset
            evenly, or if the reference model's own skill is not bitwise 0.0.
    """
    if not models:
        raise ValueError("models is empty; there is nothing to evaluate")
    if persistence_key not in models:
        raise ValueError(
            f"persistence_key {persistence_key!r} is not among the models "
            f"{sorted(models)}; every skill score needs its persistence denominator "
            f"measured over these same windows (CLAUDE.md non-negotiable 4)"
        )
    # Checked before any work: this runs at the head of a scoring pass that can take
    # minutes, and a provenance failure invalidates every number the pass would produce.
    _check_norm_provenance(dataset)
    spec = dataset.window_spec
    dof_names = tuple(dataset.target_columns)
    n_targets = len(dof_names)
    n_keys = len(dataset.realization_keys)
    per_realization = dataset.windows_per_realization
    if n_keys * per_realization != len(dataset):
        raise RuntimeError(
            f"dataset reports {len(dataset)} windows but {n_keys} realizations x "
            f"{per_realization} windows each is {n_keys * per_realization}; the window "
            f"index does not partition by realization"
        )

    stats = dataset.norm_stats.subset(dof_names)
    accumulators = {
        name: _empty_accumulator(n_keys, spec.max_horizon, n_targets) for name in models
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
                keys = _key_index(offset, batch, per_realization)
                target = y.double()
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
                    error = pred - target
                    accumulator = accumulators[name]
                    accumulator.sse.index_add_(0, keys, error.square())
                    accumulator.sae.index_add_(0, keys, error.abs())
                    accumulator.n_per_key.index_add_(
                        0, keys, torch.ones((batch,), dtype=torch.int64)
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
            f"a metric averaged over a changed window population is not the metric"
        )
    for name, accumulator in accumulators.items():
        expected = torch.full((n_keys,), per_realization, dtype=torch.int64)
        if not torch.equal(accumulator.n_per_key, expected):
            raise RuntimeError(
                f"model {name!r} scored an uneven number of windows per realization "
                f"(min {int(accumulator.n_per_key.min())}, max "
                f"{int(accumulator.n_per_key.max())}, expected {per_realization}); the "
                f"window-index-to-realization mapping is wrong"
            )

    reference = accumulators[persistence_key]
    reference_sse, _ = reference.totals()
    tables: list[pd.DataFrame] = []
    for name, accumulator in accumulators.items():
        sse, sae = accumulator.totals()
        table = metrics_table_from_sums(
            sse=sse,
            sae=sae,
            sse_persistence=reference_sse,
            n=accumulator.n_windows,
            dof_names=dof_names,
            horizons=horizons,
            fs_hz=fs_hz,
        )
        lo, hi = bootstrap_skill_ci(
            accumulator.sse,
            reference.sse,
            horizons=horizons,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=bootstrap_seed,
        )
        table.insert(0, "model", name)
        table["n_realizations"] = n_keys
        table["skill_ci_lo"] = _in_row_order(lo, len(dof_names), len(horizons))
        table["skill_ci_hi"] = _in_row_order(hi, len(dof_names), len(horizons))
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    own_skill = combined.loc[combined["model"] == persistence_key, "skill"].to_numpy()
    if not np.all(own_skill == 0.0):
        raise RuntimeError(
            f"the reference model {persistence_key!r} does not score exactly 0.0 against "
            f"itself (worst |skill| = {np.abs(own_skill).max():.3e}). Skill is formed as "
            f"1 - sse_model/sse_persistence from the same accumulator, so this can only "
            f"mean the reference row and the skill denominator came from different sums."
        )
    columns = ["model", *METRIC_COLUMNS, "n_realizations", "skill_ci_lo", "skill_ci_hi"]
    return combined[columns], accumulators


def _in_row_order(values: FloatArray, n_dofs: int, n_horizons: int) -> FloatArray:
    """Flatten a ``(n_horizons, n_dofs)`` array into the results table's row order.

    :func:`dmf.eval.metrics.metrics_table_from_sums` emits rows channel-major, horizon-minor.

    Args:
        values: Array of shape ``(n_horizons, n_dofs)``.
        n_dofs: Number of target channels.
        n_horizons: Number of reported horizons.

    Returns:
        Flattened array of length ``n_dofs * n_horizons``, in table row order.

    Raises:
        ValueError: If ``values`` has the wrong shape.
    """
    if values.shape != (n_horizons, n_dofs):
        raise ValueError(f"expected shape ({n_horizons}, {n_dofs}), got {values.shape}")
    return np.asarray(values.T.reshape(-1), dtype=np.float64)


def _check_bootstrap_args(
    shapes: Sequence[tuple[int, ...]], *, horizons: tuple[int, ...], n_boot: int, ci_level: float
) -> None:
    """Validate the arguments every realization bootstrap in this module shares.

    Args:
        shapes: Shapes of the per-realization SSE tensors, which must all be equal and
            three-dimensional ``(n_keys, H, C_out)``.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        n_boot: Number of bootstrap resamples.
        ci_level: Central confidence level.

    Raises:
        ValueError: If the shapes disagree or are not ``(n_keys, H, C_out)``, if ``n_boot``
            is not positive, if ``ci_level`` is outside ``(0, 1)``, or if a horizon is out
            of range.
    """
    first = shapes[0]
    if any(shape != first for shape in shapes):
        raise ValueError(
            f"every SSE tensor must cover the same realizations and windows, got {list(shapes)}"
        )
    if len(first) != 3:
        raise ValueError(f"expected (n_keys, H, C_out), got {first}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    bad = [h for h in horizons if h < 1 or h > first[1]]
    if bad:
        raise ValueError(f"horizons {bad} are outside [1, {first[1]}]")


def _horizon_slice(sse: Tensor, horizons: tuple[int, ...]) -> FloatArray:
    """Select the reported horizons and flatten to ``(n_keys, len(horizons) * C_out)``.

    Args:
        sse: Per-realization summed squared error, ``(n_keys, H, C_out)``, squared corpus
            units.
        horizons: Horizons to report, samples. "Horizon ``h``" reads element ``h - 1``.

    Returns:
        Float64 array, rows realizations, columns horizon-major then channel.
    """
    index = [h - 1 for h in horizons]
    n_keys = int(sse.shape[0])
    return np.asarray(sse[:, index, :].reshape(n_keys, -1).numpy(), dtype=np.float64)


def _bootstrap_counts(n_keys: int, n_boot: int, seed: int) -> FloatArray:
    """Draw multinomial resample counts over realizations.

    ``skill`` is a ratio of sums over realizations, so a resample is exactly a
    count-weighted sum and 1000 resamples are one small matrix product. The counts are a
    pure function of ``(n_keys, n_boot, seed)``, so two models scored over the same
    partition with the same ``seed`` are resampled **identically** -- which is what makes a
    paired difference (:func:`paired_skill_difference_ci`) possible without re-drawing.

    Args:
        n_keys: Number of realizations in the partition.
        n_boot: Number of bootstrap resamples.
        seed: Seed for the resampling generator.

    Returns:
        Counts, shape ``(n_boot, n_keys)``, float64, each row summing to ``n_keys``.
    """
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.multinomial(n_keys, np.full(n_keys, 1.0 / n_keys), size=n_boot), dtype=np.float64
    )


def _percentile_interval(
    draws: FloatArray, *, ci_level: float, shape: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    """Take a central percentile interval over the bootstrap axis.

    Args:
        draws: Bootstrap draws, shape ``(n_boot, len(horizons) * C_out)``.
        ci_level: Central confidence level, e.g. 0.95 for a 2.5/97.5 percentile interval.
        shape: ``(len(horizons), C_out)`` to reshape each bound to.

    Returns:
        Tuple ``(lo, hi)``, each of shape ``shape``, dimensionless.
    """
    tail = (1.0 - ci_level) / 2.0
    lo = np.quantile(draws, tail, axis=0).reshape(shape)
    hi = np.quantile(draws, 1.0 - tail, axis=0).reshape(shape)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def bootstrap_skill_ci(
    sse_model: Tensor,
    sse_persistence: Tensor,
    *,
    horizons: tuple[int, ...],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Bootstrap a skill confidence interval by resampling whole realizations.

    **Realizations, never windows.** At ``stride = 5`` with a 200-sample lookback,
    consecutive windows share 195 of their 200 input samples and overlapping future
    targets, so a window-level bootstrap would treat ~1150 near-duplicate windows per
    realization as independent draws and would report an interval roughly ``sqrt(1151)``
    times too narrow. The realization is the unit that was independently simulated, so it
    is the unit that gets resampled.

    Implemented with multinomial counts rather than index draws -- ``skill`` is a ratio of
    sums, so a resample is exactly a count-weighted sum -- which makes 1000 resamples one
    small matrix product.

    This interval answers "is this model's skill distinguishable from the Gate 3 threshold
    of 0.8?", which a point estimate near the threshold cannot. It is a different quantity
    from the seed-to-seed standard deviation and is reported in different columns.

    Args:
        sse_model: Per-realization summed squared error, shape ``(n_keys, H, C_out)``.
        sse_persistence: The same for the reference model, over the same realizations and
            the same windows.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        n_boot: Number of bootstrap resamples.
        ci_level: Central confidence level, e.g. 0.95 for a 2.5/97.5 percentile interval.
        seed: Seed for the resampling generator.

    Returns:
        Tuple ``(lo, hi)``, each of shape ``(len(horizons), C_out)``, dimensionless.

    Raises:
        ValueError: If the two arrays disagree in shape, if ``n_boot`` is not positive, if
            ``ci_level`` is not in ``(0, 1)``, or if a resampled persistence SSE is zero.
    """
    _check_bootstrap_args(
        [tuple(sse_model.shape), tuple(sse_persistence.shape)],
        horizons=horizons,
        n_boot=n_boot,
        ci_level=ci_level,
    )
    n_keys = int(sse_model.shape[0])
    n_targets = int(sse_model.shape[2])
    model = _horizon_slice(sse_model, horizons)
    reference = _horizon_slice(sse_persistence, horizons)
    counts = _bootstrap_counts(n_keys, n_boot, seed)
    denominator = counts @ reference
    if np.any(denominator == 0.0):
        raise ValueError(
            "a bootstrap resample produced zero persistence SSE; the skill score is undefined there"
        )
    skill = 1.0 - (counts @ model) / denominator
    return _percentile_interval(skill, ci_level=ci_level, shape=(len(horizons), n_targets))


def paired_skill_difference_ci(
    sse_a: Tensor,
    sse_b: Tensor,
    sse_persistence: Tensor,
    *,
    horizons: tuple[int, ...],
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bootstrap a **paired** interval for ``skill(a) - skill(b)``.

    Every model in a pass is scored on the same realizations, so their errors are strongly
    correlated: on a narrowband corpus a rough realization is rough for all of them. Two
    *unpaired* marginal intervals overlapping therefore says nothing about whether the
    difference is distinguishable from zero -- the shared realization-to-realization
    variation, which dominates both marginals, cancels in the difference. Judging a
    model-vs-model effect by eye from two overlapping marginal intervals is the specific
    error this function exists to remove.

    The pairing is exact rather than approximate: the same resample weights are applied to
    both models and to the reference in the same expression, because
    ``skill(a) - skill(b) = (SSE_b - SSE_a) / SSE_persistence`` under one resample. The
    reference cancels out of the numerator entirely, so the difference does not inherit the
    non-monotonic persistence denominator behaviour of P3-D5 in the *numerator* -- it is
    still divided by it, so it is still a skill-scale quantity and still not comparable
    across horizons.

    **Not wired into the sweep.** :func:`evaluate_models` already holds every model's
    per-realization accumulator in one pass, so this costs one extra matrix product per
    pair; it needs a caller to decide which pairs to report and where to write them.

    Args:
        sse_a: Per-realization summed squared error of the first model, shape
            ``(n_keys, H, C_out)``, squared corpus units (degrees squared for attitudes,
            metres squared for heave).
        sse_b: The same for the second model, over the same realizations and windows.
        sse_persistence: The same for the persistence reference, likewise.
        horizons: Horizons to report, samples, each in ``[1, H]``.
        n_boot: Number of bootstrap resamples.
        ci_level: Central confidence level, e.g. 0.95 for a 2.5/97.5 percentile interval.
        seed: Seed for the resampling generator. The same value used by
            :func:`bootstrap_skill_ci` draws the same resamples, so the paired interval and
            the marginals in one table are computed on one common set of resamples.

    Returns:
        Tuple ``(difference, lo, hi)``, each of shape ``(len(horizons), C_out)`` and
        dimensionless. ``difference`` is the point estimate on the full sample, positive
        when model ``a`` has the higher skill; ``lo``/``hi`` are the percentile interval of
        the paired bootstrap. An interval excluding zero is the claim "``a`` beats ``b`` on
        these realizations"; the point estimate is *not* the centre of the interval and is
        reported separately for that reason.

    Raises:
        ValueError: If the three arrays disagree in shape, if ``n_boot`` is not positive, if
            ``ci_level`` is not in ``(0, 1)``, if a horizon is out of range, or if a
            resampled persistence SSE is zero.
    """
    _check_bootstrap_args(
        [tuple(sse_a.shape), tuple(sse_b.shape), tuple(sse_persistence.shape)],
        horizons=horizons,
        n_boot=n_boot,
        ci_level=ci_level,
    )
    n_keys = int(sse_a.shape[0])
    shape = (len(horizons), int(sse_a.shape[2]))
    a = _horizon_slice(sse_a, horizons)
    b = _horizon_slice(sse_b, horizons)
    reference = _horizon_slice(sse_persistence, horizons)
    total = reference.sum(axis=0)
    if np.any(total == 0.0):
        raise ValueError("persistence SSE is zero; the skill difference is undefined there")
    point = np.asarray(((b.sum(axis=0) - a.sum(axis=0)) / total).reshape(shape), dtype=np.float64)
    counts = _bootstrap_counts(n_keys, n_boot, seed)
    denominator = counts @ reference
    if np.any(denominator == 0.0):
        raise ValueError(
            "a bootstrap resample produced zero persistence SSE; the skill score is undefined there"
        )
    draws = (counts @ b - counts @ a) / denominator
    lo, hi = _percentile_interval(draws, ci_level=ci_level, shape=shape)
    return point, lo, hi


def per_cell_metrics(
    accumulators: Mapping[str, EvalAccumulator],
    dataset: DeckMotionDataset,
    *,
    persistence_key: str,
    horizons: tuple[int, ...],
    fs_hz: float,
    by: Sequence[str] = CELL_COLUMNS,
) -> pd.DataFrame:
    """Break the pooled results out by corpus grid cell.

    Free, because :func:`evaluate_models` already accumulated per realization. It is what
    makes the two standing caveats checkable instead of quotable:

    - ``unseen_heading`` pitch sits on the P1-D2 residual floor -- that regime's test set
      *is* beam seas, where pitch is floored ~26 dB down and driven by an engineering
      stand-in for hull asymmetry rather than by physics. Its persistence RMSE is tiny and
      its skill is dominated by that stand-in.
    - ``id`` pools 4 headings x 4 sea states x 3 speeds, and roll in head seas is on the
      same residual floor.

    The raw ``sse``/``sae``/``sse_persistence``/``n_windows`` sums are emitted alongside the
    derived metrics so that any coarser marginal can be recomputed exactly with
    :func:`marginalize_cells`; averaging the per-cell RMSE column instead would be wrong.

    Args:
        accumulators: Per-realization sums from :func:`evaluate_models`.
        dataset: The partition they were accumulated over, for its realization keys.
        persistence_key: Key naming the reference baseline.
        horizons: Horizons to report, samples.
        fs_hz: Sampling rate, hertz.
        by: Grid axes to group by, a subset of :data:`CELL_COLUMNS`.

    Returns:
        One row per (model, cell, DOF, horizon), with the ``by`` columns, the metric
        columns, and the raw sums.

    Raises:
        ValueError: If ``persistence_key`` is absent, if ``by`` names an unknown axis, or
            if an accumulator does not match the dataset's realization count.
    """
    if persistence_key not in accumulators:
        raise ValueError(f"persistence_key {persistence_key!r} is not among {sorted(accumulators)}")
    unknown = [axis for axis in by if axis not in _KEY_FIELD_INDEX]
    if unknown:
        raise ValueError(f"unknown cell axes {unknown}; known axes are {list(CELL_COLUMNS)}")
    keys = dataset.realization_keys
    for name, accumulator in accumulators.items():
        if accumulator.n_keys != len(keys):
            raise ValueError(
                f"accumulator {name!r} covers {accumulator.n_keys} realizations but the "
                f"dataset holds {len(keys)}"
            )
    dof_names = tuple(dataset.target_columns)
    groups: dict[tuple[str | float | int, ...], list[int]] = {}
    for position, key in enumerate(keys):
        cell = tuple(key[_KEY_FIELD_INDEX[axis]] for axis in by)
        groups.setdefault(cell, []).append(position)

    reference = accumulators[persistence_key]
    frames: list[pd.DataFrame] = []
    for cell_values, positions in sorted(groups.items(), key=repr):
        selector = torch.as_tensor(positions, dtype=torch.int64)
        reference_sse = reference.sse[selector].sum(dim=0).numpy().astype(np.float64)
        n = int(reference.n_per_key[selector].sum().item())
        for name, accumulator in accumulators.items():
            sse = accumulator.sse[selector].sum(dim=0).numpy().astype(np.float64)
            sae = accumulator.sae[selector].sum(dim=0).numpy().astype(np.float64)
            table = metrics_table_from_sums(
                sse=sse,
                sae=sae,
                sse_persistence=reference_sse,
                n=n,
                dof_names=dof_names,
                horizons=horizons,
                fs_hz=fs_hz,
            )
            table.insert(0, "model", name)
            for axis, value in zip(by, cell_values, strict=True):
                table[axis] = value
            table["n_realizations"] = len(positions)
            table["sse"] = _cell_sums(sse, dof_names, horizons)
            table["sae"] = _cell_sums(sae, dof_names, horizons)
            table["sse_persistence"] = _cell_sums(reference_sse, dof_names, horizons)
            frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    ordered = [
        "model",
        *by,
        "dof",
        "horizon_samples",
        "horizon_s",
        "n_realizations",
        "n_windows",
        "rmse",
        "mae",
        "rmse_persistence",
        "skill",
        "sse",
        "sae",
        "sse_persistence",
    ]
    return combined[ordered].sort_values(
        [*by, "model", "dof", "horizon_samples"], ignore_index=True
    )


def _cell_sums(
    values: FloatArray, dof_names: tuple[str, ...], horizons: tuple[int, ...]
) -> list[float]:
    """Flatten an ``(H, C)`` sum array into the metrics table's row order.

    Args:
        values: Array of shape ``(H, C)``.
        dof_names: Target channel names, length ``C``.
        horizons: Reported horizons, samples.

    Returns:
        One value per table row, channel-major and horizon-minor.
    """
    return [float(values[h - 1, c]) for c in range(len(dof_names)) for h in horizons]


def marginalize_cells(cells: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Collapse a per-cell table onto a coarser grouping, correctly.

    Sums the squared and absolute errors and re-derives the metrics, rather than averaging
    the RMSE column: ``mean(sqrt(x))`` is not ``sqrt(mean(x))``, and the cells differ in
    window count.

    Args:
        cells: Output of :func:`per_cell_metrics`.
        by: Axes to keep, e.g. ``("heading_deg",)`` for the per-heading breakdown.

    Returns:
        One row per (model, kept axes, DOF, horizon), with the same metric columns.

    Raises:
        ValueError: If a required column is missing from ``cells``.
    """
    required = {
        "model",
        "dof",
        "horizon_samples",
        "horizon_s",
        "n_windows",
        "sse",
        "sae",
        "sse_persistence",
    }
    missing = sorted(required - set(cells.columns))
    if missing:
        raise ValueError(f"cells is missing columns {missing}; pass the per_cell_metrics table")
    group = ["model", *by, "dof", "horizon_samples", "horizon_s"]
    summed = cells.groupby(group, as_index=False, sort=True)[
        ["n_windows", "n_realizations", "sse", "sae", "sse_persistence"]
    ].sum()
    summed["rmse"] = np.sqrt(summed["sse"] / summed["n_windows"])
    summed["mae"] = summed["sae"] / summed["n_windows"]
    summed["rmse_persistence"] = np.sqrt(summed["sse_persistence"] / summed["n_windows"])
    summed["skill"] = 1.0 - summed["sse"] / summed["sse_persistence"]
    return summed
