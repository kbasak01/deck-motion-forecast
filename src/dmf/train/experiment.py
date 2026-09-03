"""The experiment driver: one config in, one results table out.

``scripts/`` holds argparse wrappers only, so the orchestration lives here where it is
importable and testable.

Per regime, exactly once:

1. :func:`dmf.data.splits.build_split` -- realization-level, no RNG.
2. Build the ``train`` dataset, which fits the normalisation statistics; build ``val`` and
   ``test`` **with those statistics**, never their own.
3. One pass of :func:`dmf.train.closed_form.accumulate_training_moments` at ``P_MAX``,
   serving damped persistence, every AR variant -- all three orders, and the attitude-only
   information set, which is a channel-subset slice of the same moments -- and the
   closed-form DLinear, whose decomposed design rides on the same batches.
4. One :func:`dmf.train.loop.fit` per (SGD model, seed).
5. One :func:`dmf.eval.runner.evaluate_models` pass scoring **every** model on identical
   windows, so the skill denominator is structurally the same for all of them.
6. One :func:`dmf.eval.runner.paired_skill_difference_ci` per configured contrast, reusing
   step 5's accumulators and its bootstrap resample weights, into
   ``paired_contrasts.csv``. No second scoring pass and no second bootstrap draw.

Models are dispatched on :attr:`dmf.models.base.BaseForecaster.FIT_KIND`, not on their
class, so a Phase 4 architecture joins the run by declaring ``FIT_KIND = "sgd"`` and adding
a config -- this module does not change.

Units: horizons and lookbacks in samples, ``fs_hz`` in hertz, wall-clock times in seconds,
errors in corpus units (degrees for roll and pitch, metres for heave).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from dmf.config import ExperimentConfig, ModelConfig
from dmf.data.dataset import DeckMotionDataset, make_dataloader
from dmf.data.splits import Regime, build_split, load_manifest
from dmf.data.windows import WindowSpec, window_spec_from_config
from dmf.eval.controls import (
    ControlResult,
    controls_table,
    persistence_pipeline_sanity,
    shuffle_control,
    untrained_control,
)
from dmf.eval.prob_runner import evaluate_probabilistic_models
from dmf.eval.report import (
    build_baselines_markdown,
    build_baselines_table,
    build_probabilistic_table,
    write_table,
)
from dmf.eval.runner import (
    EvalAccumulator,
    evaluate_models,
    marginalize_cells,
    paired_skill_difference_ci,
    per_cell_metrics,
)
from dmf.models.base import BaseForecaster
from dmf.models.dlinear_ols import DLinearOLS
from dmf.models.heads import point_view
from dmf.models.persistence import TAU_WINDOW_MEAN, DampedPersistence
from dmf.train.closed_form import (
    TrainingMoments,
    accumulate_training_moments,
    fit_ar,
    fit_damped_persistence,
    fit_dlinear_ols,
)
from dmf.train.loop import fit, set_seed
from dmf.train.losses import resolve_loss
from dmf.train.registry import MODEL_REGISTRY, build_model

__all__ = [
    "BOOTSTRAP_CI_LEVEL",
    "BOOTSTRAP_N_BOOT",
    "BOOTSTRAP_SEED",
    "CONTRAST_BASELINES",
    "CONTRAST_DEEP_MODELS",
    "PAIRED_CONTRASTS",
    "PAIRED_CONTRAST_COLUMNS",
    "PERSISTENCE_LABEL",
    "RunRecord",
    "VAL_LOSS_NAMES",
    "run_experiment",
]

#: Label of the reference model. Every skill score's denominator comes from this model's
#: accumulator, so the experiment refuses to run without it in the config.
PERSISTENCE_LABEL = "persistence"

#: The (DOF, horizon) cell the Gate 3 threshold is read at, restated by ``docs/protocol.md``
#: P3-D12 (applied 2026-08-27) from roll at 3 s to **pitch at 10 s**. The threshold value
#: itself is unchanged at 0.8; only the cell moved, to the horizon the landing decision is
#: taken at and the DOF that actually binds. Roll at 3 s is a quarter of the 12 s roll
#: period on the most narrowband channel in the corpus, i.e. the easiest cell there is, and
#: it saturates at 0.9987 whatever the model. Named here rather than passed as a literal:
#: ``results/baselines.md`` is the document the gate decision is read from, and the
#: committed artifact declared the superseded cell while the README claimed the gate passed.
GATE_DOF = "pitch"
GATE_HORIZON_SAMPLES = 100

#: Which objective ``best_val_loss`` reports, by head kind. Written to the per-seed table
#: beside the number, because since Phase 5 the early-stopping criterion is head-specific
#: (``docs/protocol.md`` P5-D4) and the column is otherwise three different quantities
#: sharing one name.
VAL_LOSS_NAMES: dict[str, str] = {
    "point": "mse",
    "quantile": "pinball",
    "gaussian": "gaussian_nll",
}

#: Nominal miscoverage the Phase 5 intervals are scored at. ``0.1`` is the 90 percent
#: interval Gate 5 is read on (``docs/protocol.md`` P5-D2), and it is the level the
#: outermost pair of ``dmf.models.heads.QUANTILE_FAN_9`` was chosen to sit exactly on, so no
#: quantile has to be interpolated to form it.
GATE5_ALPHA = 0.1

#: The band Gate 5 requires PICP@90 to fall inside on the ``id`` regime, verbatim from
#: ``docs/IMPLEMENTATION_PLAN.md`` Phase 5. Unchanged in Phase 5; only the cell it is read
#: at was registered (P5-D2), and it is the same cell Gates 3 and 4 are read at.
GATE5_PICP_BAND = (0.85, 0.95)


#: Separator in the per-run key ``"<label>@<seed>"`` that every model of a regime is scored
#: under, so that three seeds of one model stay three entries in the scoring pass rather
#: than one overwriting the others. Labels come from the config and never contain it, so
#: :func:`_split_run_keys` can invert the join for the committed tables.
RUN_KEY_SEP = "@"


#: Bootstrap settings for **both** the marginal skill intervals computed inside
#: :func:`dmf.eval.runner.evaluate_models` and the paired contrasts computed from the
#: accumulators it returns. Passed explicitly to both rather than left to two default
#: arguments that happen to agree: :func:`dmf.eval.runner.paired_skill_difference_ci` is
#: exact only when it draws the *same* multinomial resample weights the marginals drew,
#: and "the same" must be a property of this module, not a coincidence between two
#: signatures. Values are the runner's current defaults, so no committed number moves.
BOOTSTRAP_N_BOOT = 1000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_SEED = 0

#: Baselines that every deep model is contrasted against in ``paired_contrasts.csv``. Each
#: one supports a different claim, and each is the reason that baseline is in the sweep:
#:
#: - ``ar20`` -- the Gate 3 subject and the strongest fitted baseline (``docs/protocol.md``
#:   P3-D12). "This architecture was worth training" is the claim that it beats AR(20) by
#:   more than the paired interval, and nothing weaker.
#: - ``dlinear_ols`` -- the *converged* linear map, at the exact optimum of its own
#:   objective. Contrasting a deep model against the SGD ``dlinear`` row instead would
#:   credit it with an optimisation gap measured at up to +0.0758 skill (P3-D19).
#: - ``damped_persistence`` -- the reference Gate 4 names.
#: - ``window_mean`` -- the zero-parameter trivial forecast, which beats persistence (the
#:   denominator of every skill score here) in 107 of 144 cells (P3-D20). A deep model that
#:   does not clear it has learned nothing that the window mean does not already carry.
CONTRAST_BASELINES: tuple[str, ...] = (
    "ar20",
    "dlinear_ols",
    "damped_persistence",
    "window_mean",
)

#: Phase 4 architectures. Listed here rather than derived from ``FIT_KIND == "sgd"``,
#: which would also sweep in ``dlinear`` and make the optimisation-gap pair below
#: ambiguous.
CONTRAST_DEEP_MODELS: tuple[str, ...] = ("tcn", "transformer", "lstm")

#: Ordered ``(model_a, model_b)`` label pairs written to ``paired_contrasts.csv``. The
#: reported quantity is ``skill(a) - skill(b)``, so **positive means model_a is better**.
#:
#: Every deep model against every entry of :data:`CONTRAST_BASELINES` -- the deep-vs-
#: baseline claims Phase 4 exists to make -- plus ``dlinear`` against ``dlinear_ols``, the
#: DLinear optimisation gap at whatever epoch budget the config carries. That last pair is
#: not a Phase 4 claim: it is the instrument P3-D19 used to show that the gap is not small
#: in skill, and it has to be re-read at the new budget before any deep-vs-``dlinear``
#: number is quoted.
#:
#: Pairs whose two labels are not both in the experiment being run are skipped silently, so
#: this list does not force every config to carry every model; see :func:`_contrast_frame`.
PAIRED_CONTRASTS: tuple[tuple[str, str], ...] = tuple(
    (deep, baseline) for deep in CONTRAST_DEEP_MODELS for baseline in CONTRAST_BASELINES
) + (("dlinear", "dlinear_ols"),)

#: Exact column order of ``results/paired_contrasts.csv``.
#:
#: ``skill_diff`` is ``skill(model_a) - skill(model_b)`` on the full sample: **positive
#: means model_a is the better model**, and it is stated here because a sign convention
#: that has to be inferred is a sign convention that gets misread. ``ci_lo``/``ci_hi`` are
#: the paired bootstrap percentile interval of that same difference over resampled
#: realizations; an interval excluding zero is the claim "``model_a`` beats ``model_b`` on
#: these realizations". ``skill_diff`` is *not* the centre of that interval and is not
#: derived from it. All three are dimensionless. One row per (regime, pair, seed, DOF,
#: horizon); ``seed`` is the training seed of the stochastic side (0 when both sides are
#: deterministic), never an average over seeds.
PAIRED_CONTRAST_COLUMNS: tuple[str, ...] = (
    "regime",
    "model_a",
    "model_b",
    "seed",
    "dof",
    "horizon_samples",
    "horizon_s",
    "skill_diff",
    "ci_lo",
    "ci_hi",
    "n_realizations",
    "n_boot",
    "ci_level",
    "bootstrap_seed",
)


@dataclass(frozen=True)
class RunRecord:
    """One fitted model instance, ready to be scored.

    Attributes:
        label: Results-table label, e.g. ``"ar20"``.
        seed: Training seed, or 0 for a deterministic model (where the seed cannot enter).
        model: The fitted model, in eval mode.
        deterministic: True if the fit is seed-independent, so the row is exempt from the
            three-seed rule and carries ``skill_std = NaN``.
        n_params: Fitted value count, from
            :attr:`dmf.models.base.BaseForecaster.n_fitted_parameters`.
        fit_time_s: Wall-clock seconds to fit. CPU for closed-form models, device
            wall-clock **including data loading** for SGD models -- not comparable between
            the two, and labelled as such in the report.
        best_epoch: Zero-indexed epoch of the lowest validation loss, or None for a model
            with no epochs. Carried into ``baselines_by_seed.csv`` so that "was this model
            still converging when the budget ran out?" is answerable from the committed
            artifact. The `dlinear` under-convergence that motivated ``dlinear_ols`` had to
            be inferred from wall-clock ratios because these three fields were computed by
            :class:`dmf.train.loop.TrainResult` and then discarded here.
        epochs_run: Epochs completed before early stopping or exhaustion, or None. Equal to
            ``cfg.train.epochs`` means the run hit the cap rather than converging.
        best_val_loss: Lowest validation loss reached, dimensionless, or None. Directly
            comparable to a closed-form model's residual only in units, not in partition --
            and, since Phase 5, not comparable across heads either: see ``val_loss_name``.
        val_loss_name: Which objective ``best_val_loss`` is, one of ``mse``, ``pinball``,
            ``gaussian_nll``, or None for a model with no epochs. Each model is stopped on
            its own training objective (``docs/protocol.md`` P5-D4), so a pinball loss and
            an MSE sit in the same column and their ratio means nothing. The column ships
            beside the number so that the comparison cannot be made by accident.
    """

    label: str
    seed: int
    model: BaseForecaster
    deterministic: bool
    n_params: int
    fit_time_s: float
    best_epoch: int | None = None
    epochs_run: int | None = None
    best_val_loss: float | None = None
    val_loss_name: str | None = None


def _instantiate(
    cfg: ModelConfig, spec: WindowSpec, n_in: int, n_out: int, seed: int
) -> BaseForecaster:
    """Build one model, seeding first so that a random initialisation is reproducible.

    Args:
        cfg: Model configuration.
        spec: Window geometry.
        n_in: Input channel count ``C_in``.
        n_out: Target channel count ``C_out``.
        seed: Seed applied before construction, so that weight initialisation is part of
            what the seed controls.

    Returns:
        The constructed model, on CPU.
    """
    set_seed(seed)
    return build_model(cfg, spec, n_in, n_out)


def _fit_one(
    cfg: ModelConfig,
    *,
    spec: WindowSpec,
    train: DeckMotionDataset,
    val: DeckMotionDataset,
    experiment: ExperimentConfig,
    moments_holder: dict[str, TrainingMoments],
    device: str,
    checkpoint_dir: Path,
) -> list[RunRecord]:
    """Fit one configured model, dispatching on its declared fit kind.

    Args:
        cfg: Model configuration.
        spec: Window geometry.
        train: Training partition.
        val: Validation partition, used only for early stopping.
        experiment: The experiment config, for seeds and optimisation settings.
        moments_holder: One-element cache so the moments pass happens at most once per
            regime, and not at all if the regime has no closed-form model.
        device: Torch device for SGD training.
        checkpoint_dir: Where best-epoch weights are written.

    Returns:
        One record per run: a single deterministic record for closed-form and untrained
        models, one per seed for SGD models.

    Raises:
        ValueError: If the model declares an unknown fit kind.
    """
    n_in = len(train.input_columns)
    n_out = len(train.target_columns)
    kind = MODEL_REGISTRY[cfg.name].FIT_KIND

    if kind == "none":
        model = _instantiate(cfg, spec, n_in, n_out, experiment.seeds[0])
        model.eval()
        return [RunRecord(cfg.label, 0, model, True, model.n_fitted_parameters, 0.0)]

    if kind == "closed_form":
        moments = _moments(moments_holder, train, experiment)
        common = {
            "lookback": spec.lookback,
            "n_input_channels": n_in,
            "n_target_channels": n_out,
        }
        if issubclass(MODEL_REGISTRY[cfg.name], DampedPersistence):
            model, decay_report = fit_damped_persistence(
                moments, fs_hz=experiment.data.fs_hz, channels=train.target_columns, **common
            )
            return [
                RunRecord(
                    cfg.label,
                    0,
                    model,
                    True,
                    decay_report.n_fitted_parameters,
                    decay_report.fit_time_s,
                )
            ]
        if issubclass(MODEL_REGISTRY[cfg.name], DLinearOLS):
            ols_model, ols_report = fit_dlinear_ols(
                moments,
                kernel_size=int(cfg.params["kernel_size"]),
                ridge=float(cfg.params.get("ridge", 0.0)),
                **common,
            )
            return [
                RunRecord(
                    cfg.label,
                    0,
                    ols_model,
                    True,
                    ols_report.n_fitted_parameters,
                    ols_report.fit_time_s,
                )
            ]
        # `n_input_used` is read here rather than defaulted inside `fit_ar` because
        # ignoring it would fit the six-channel model and label it `ar_attitude_only`,
        # which is a silently wrong row rather than a loud failure. Slicing the cached
        # moments costs one extra solve and no extra pass over the training split.
        used = cfg.params.get("n_input_used")
        ar_model, ar_report = fit_ar(
            moments,
            order=int(cfg.params["order"]),
            ridge=float(cfg.params.get("ridge", 0.0)),
            n_input_used=None if used is None else int(used),
            **common,
        )
        return [
            RunRecord(
                cfg.label, 0, ar_model, True, ar_report.n_fitted_parameters, ar_report.fit_time_s
            )
        ]

    if kind != "sgd":
        raise ValueError(f"model {cfg.label!r} declares unknown FIT_KIND {kind!r}")

    records: list[RunRecord] = []
    train_loader = make_dataloader(
        train,
        batch_size=experiment.train.batch_size,
        shuffle=True,
        num_workers=experiment.train.num_workers,
        seed=0,
    )
    val_loader = make_dataloader(
        val,
        batch_size=max(experiment.train.batch_size, 1024),
        shuffle=False,
        num_workers=experiment.train.num_workers,
        seed=0,
    )
    # Resolved once per config, from the config rather than from the built model, so that a
    # config asking for a head its model cannot carry fails in `build_model` and never
    # reaches a mismatched objective here.
    loss_fn = resolve_loss(cfg.head, cfg.quantiles)
    for seed in experiment.seeds:
        model = _instantiate(cfg, spec, n_in, n_out, seed).to(device)
        result = fit(
            model,
            train_loader,
            val_loader,
            experiment.train,
            seed,
            checkpoint_dir,
            loss_fn=loss_fn,
            # The label, not the class name: in `e03` the point, quantile and Gaussian
            # variants of one class share a class name and would write one checkpoint file
            # three times, leaving two runs' weights on disk under a third run's name.
            label=cfg.label,
        )
        model.eval()
        records.append(
            RunRecord(
                cfg.label,
                seed,
                model,
                False,
                model.n_fitted_parameters,
                result.wall_time_s,
                best_epoch=result.best_epoch,
                epochs_run=result.epochs_run,
                best_val_loss=result.best_val_loss,
                val_loss_name=VAL_LOSS_NAMES[cfg.head],
            )
        )
    return records


def _moments(
    holder: dict[str, TrainingMoments], train: DeckMotionDataset, experiment: ExperimentConfig
) -> TrainingMoments:
    """Return the regime's training moments, accumulating them at most once.

    Args:
        holder: Per-regime cache.
        train: Training partition.
        experiment: The experiment config, supplying the AR orders and worker count.

    Returns:
        The :class:`dmf.train.closed_form.TrainingMoments` for this regime.
    """
    if "moments" not in holder:
        holder["moments"] = accumulate_training_moments(
            train,
            max_order=_max_order(experiment),
            num_workers=experiment.train.num_workers,
            decompose_kernel=_decompose_kernel(experiment),
        )
    return holder["moments"]


def _max_order(experiment: ExperimentConfig) -> int:
    """Return the largest AR order any configured model asks for, samples.

    Args:
        experiment: The experiment config.

    Returns:
        The maximum ``order`` over the AR configs, or 1 if none are present.
    """
    orders = [int(m.params["order"]) for m in experiment.models if "order" in m.params]
    return max(orders, default=1)


def _decompose_kernel(experiment: ExperimentConfig) -> int | None:
    """Return the trend kernel the closed-form DLinear needs, samples, or None.

    Read from the configs rather than defaulted, and only for models that are actually
    solved in closed form: accumulating the decomposed design at the wrong kernel would
    solve a different model under the configured label, and accumulating it when nothing
    reads it would spend a 400x400 float64 Gram per batch for nothing.

    Args:
        experiment: The experiment config.

    Returns:
        The kernel size, or None if no closed-form DLinear is configured.

    Raises:
        ValueError: If two closed-form DLinear configs ask for different kernels. One pass
            accumulates one design, so the second row would silently be solved from the
            first one's moments.
    """
    kernels = {
        int(m.params["kernel_size"])
        for m in experiment.models
        if issubclass(MODEL_REGISTRY[m.name], DLinearOLS) and "kernel_size" in m.params
    }
    if len(kernels) > 1:
        raise ValueError(
            f"closed-form DLinear configs ask for different kernel sizes {sorted(kernels)}; "
            f"one moments pass can only carry one decomposed design"
        )
    return kernels.pop() if kernels else None


def _split_run_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Split a composite ``"<label>@<seed>"`` model column into ``model`` and ``seed``.

    The scoring pass has to key on the composite run key; the committed tables must not.
    ``baselines.csv`` and ``baselines_by_seed.csv`` carry the bare label in ``model``, so a
    per-cell table still keyed ``"dlinear@1"`` shares a column name with them while using a
    different key space: it cannot be joined on
    ``(model, regime, dof, horizon_samples)``, and ``baselines.md`` renders three seeds of
    one model as three unrelated models. Undoing the join here rather than in
    :mod:`dmf.eval.runner` leaves the runner's single-key contract intact -- the
    accumulators are keyed the same way and the per-cell breakdown needs them to stay
    distinct.

    Args:
        frame: A table whose ``model`` column holds composite run keys. Not mutated.

    Returns:
        A copy in which ``model`` holds the bare label, with an integer ``seed`` column
        inserted immediately after it. Every other column and the row order are unchanged,
        so no metric is recomputed and no number moves.

    Raises:
        ValueError: If any ``model`` value carries no :data:`RUN_KEY_SEP`, which would mean
            the frame did not come from a per-run scoring pass and the split would silently
            invent a seed.
    """
    keys = frame["model"].astype(str)
    composite = keys.str.contains(RUN_KEY_SEP, regex=False)
    if not bool(composite.all()):
        bad = sorted(set(keys[~composite]))
        raise ValueError(
            f"model column holds {bad} without the {RUN_KEY_SEP!r} run-key separator; "
            f"this frame did not come from a per-run scoring pass"
        )
    parts = keys.str.rsplit(RUN_KEY_SEP, n=1, expand=True)
    out = frame.copy()
    out["model"] = parts[0]
    out.insert(list(frame.columns).index("model") + 1, "seed", parts[1].astype(int))
    return out


def _runs_by_label(
    accumulators: Mapping[str, EvalAccumulator],
) -> dict[str, dict[int, EvalAccumulator]]:
    """Group a scoring pass's accumulators by bare label, keyed by training seed.

    Args:
        accumulators: Per-realization error sums keyed ``"<label>@<seed>"``, as
            :func:`dmf.eval.runner.evaluate_models` returns them.

    Returns:
        ``{label: {seed: accumulator}}``. A stochastic model contributes one entry per
        seed; a deterministic one contributes the single entry ``{0: ...}``.

    Raises:
        ValueError: If a key carries no :data:`RUN_KEY_SEP`, which would mean the mapping
            did not come from a per-run scoring pass and the seed would be invented.
    """
    grouped: dict[str, dict[int, EvalAccumulator]] = {}
    for key, accumulator in accumulators.items():
        label, separator, seed = key.rpartition(RUN_KEY_SEP)
        if not separator:
            raise ValueError(
                f"accumulator key {key!r} carries no {RUN_KEY_SEP!r} run-key separator; "
                f"this mapping did not come from a per-run scoring pass"
            )
        grouped.setdefault(label, {})[int(seed)] = accumulator
    return grouped


def _pair_seeds(
    label_a: str, seeds_a: Sequence[int], label_b: str, seeds_b: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Decide which run of ``a`` is contrasted against which run of ``b``.

    A contrast is only paired if the two sides are the same *thing* being resampled, and
    across seeds there are exactly two cases worth supporting:

    - **Both stochastic, same seed set.** Pair seed by seed. Contrasting seed 0 of one
      model against seed 2 of another adds a seed difference to the architecture
      difference the row claims to measure.
    - **One side deterministic.** A closed-form fit has a single accumulator and the seed
      cannot enter it, so every seed of the stochastic side is contrasted against that one
      accumulator. Each such row is still a genuine paired interval: it is the interval for
      *that run* against the deterministic model.

    Intervals are never averaged across seeds here. The mean of three intervals is not an
    interval for anything -- it excludes seed variance and is invariant to seed
    disagreement, which is the defect ``docs/protocol.md`` P3-D22 records and corrects for
    the marginal CIs. The per-seed rows are what get written; any envelope over them is the
    reader's, and must be labelled as an envelope.

    Args:
        label_a: Label of the first model, for the error message.
        seeds_a: Seeds the first model was fitted at.
        label_b: Label of the second model.
        seeds_b: Seeds the second model was fitted at.

    Returns:
        Triples ``(row_seed, seed_a, seed_b)``. ``row_seed`` is the seed written to the
        table: the stochastic side's seed when one side is deterministic, the shared seed
        when both sides carry the same set.

    Raises:
        ValueError: If the two seed sets differ and neither side is a single deterministic
            run, so no pairing is defined and any choice would be arbitrary.
    """
    left = sorted(seeds_a)
    right = sorted(seeds_b)
    if left == right:
        return [(seed, seed, seed) for seed in left]
    if len(right) == 1:
        return [(seed, seed, right[0]) for seed in left]
    if len(left) == 1:
        return [(seed, left[0], seed) for seed in right]
    raise ValueError(
        f"cannot pair {label_a!r} (seeds {left}) with {label_b!r} (seeds {right}): the seed "
        f"sets differ and neither side is a single deterministic run, so seed-by-seed "
        f"pairing is undefined"
    )


def _contrast_frame(
    accumulators: Mapping[str, EvalAccumulator],
    *,
    regime: str,
    dof_names: Sequence[str],
    horizons: tuple[int, ...],
    fs_hz: float,
    persistence_key: str,
    contrasts: Sequence[tuple[str, str]] = PAIRED_CONTRASTS,
    n_boot: int = BOOTSTRAP_N_BOOT,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compute the paired skill-difference intervals for one regime.

    Reuses the accumulators of the regime's single :func:`dmf.eval.runner.evaluate_models`
    pass, so this costs one small matrix product per (pair, seed) and no second pass over
    the partition. ``n_boot`` and ``bootstrap_seed`` are the ones the marginal intervals in
    that same pass used, which is what makes the pairing exact rather than approximate:
    :func:`dmf.eval.runner._bootstrap_counts` is a pure function of
    ``(n_realizations, n_boot, seed)``, so both models and the reference are reweighted by
    the identical multinomial counts and the realization-to-realization variation that
    dominates both marginals cancels in the difference.

    Pairs naming a label the experiment did not run are skipped **silently**:
    :data:`PAIRED_CONTRASTS` is the full Phase 4 list, and requiring every config to carry
    every model would make it impossible to run a baselines-only or single-architecture
    sweep. The one thing that is not silent is a list that matches nothing while deep
    models are present -- see ``Raises``.

    Args:
        accumulators: Per-realization error sums keyed ``"<label>@<seed>"``.
        regime: Regime name, written to every row.
        dof_names: Target channel names in accumulator channel order (roll, pitch and heave
            in degrees and metres, their rates in degrees per second and metres per second).
        horizons: Horizons to report, samples. "Horizon ``h``" is the error at lead time
            exactly ``h`` samples.
        fs_hz: Sampling rate, hertz, used to report each horizon in seconds.
        persistence_key: Key naming the reference accumulator, whose SSE is the skill
            denominator on both sides of every difference.
        contrasts: Ordered ``(model_a, model_b)`` label pairs. Defaults to
            :data:`PAIRED_CONTRASTS`.
        n_boot: Bootstrap resamples; must equal the value the marginal CIs used.
        ci_level: Central confidence level of the paired interval.
        bootstrap_seed: Bootstrap seed; must equal the value the marginal CIs used.

    Returns:
        A frame with columns :data:`PAIRED_CONTRAST_COLUMNS`, one row per (pair, seed, DOF,
        horizon), rows channel-major and horizon-minor like every other table here. Empty
        (with those columns) when no configured pair is present.

    Raises:
        ValueError: If the reference accumulator is missing, if a deep model from
            :data:`CONTRAST_DEEP_MODELS` was scored but not one configured pair could be
            formed -- a mistyped label in :data:`PAIRED_CONTRASTS` would otherwise write a
            silently empty file -- or if a pair's two seed sets cannot be paired.
    """
    if persistence_key not in accumulators:
        raise ValueError(
            f"persistence_key {persistence_key!r} is not among the scored runs "
            f"{sorted(accumulators)}; the skill difference has no denominator"
        )
    reference = accumulators[persistence_key]
    by_label = _runs_by_label(accumulators)
    rows: list[dict[str, object]] = []
    for label_a, label_b in contrasts:
        if label_a not in by_label or label_b not in by_label:
            continue
        runs_a = by_label[label_a]
        runs_b = by_label[label_b]
        pairing = _pair_seeds(label_a, list(runs_a), label_b, list(runs_b))
        for row_seed, seed_a, seed_b in pairing:
            difference, lo, hi = paired_skill_difference_ci(
                runs_a[seed_a].sse,
                runs_b[seed_b].sse,
                reference.sse,
                horizons=horizons,
                n_boot=n_boot,
                ci_level=ci_level,
                seed=bootstrap_seed,
            )
            for channel, dof in enumerate(dof_names):
                for index, horizon in enumerate(horizons):
                    rows.append(
                        {
                            "regime": regime,
                            "model_a": label_a,
                            "model_b": label_b,
                            "seed": row_seed,
                            "dof": dof,
                            "horizon_samples": horizon,
                            "horizon_s": horizon / fs_hz,
                            # skill(a) - skill(b): positive means model_a is better.
                            "skill_diff": float(difference[index, channel]),
                            "ci_lo": float(lo[index, channel]),
                            "ci_hi": float(hi[index, channel]),
                            "n_realizations": runs_a[seed_a].n_keys,
                            "n_boot": n_boot,
                            "ci_level": ci_level,
                            "bootstrap_seed": bootstrap_seed,
                        }
                    )
    scored_deep = sorted(set(by_label) & set(CONTRAST_DEEP_MODELS))
    if scored_deep and not rows:
        raise ValueError(
            f"deep models {scored_deep} were scored but no pair of PAIRED_CONTRASTS "
            f"matched the run labels {sorted(by_label)}; a mistyped label there writes an "
            f"empty contrast file rather than failing"
        )
    return pd.DataFrame(rows, columns=list(PAIRED_CONTRAST_COLUMNS))


def run_experiment(
    cfg: ExperimentConfig,
    corpus_root: Path,
    *,
    results_dir: Path = Path("results"),
    seeds: Sequence[int] | None = None,
    regimes: Sequence[str] | None = None,
    device: str = "cpu",
    run_controls: bool = True,
    checkpoint_root: Path = Path("artifacts/checkpoints"),
) -> pd.DataFrame:
    """Run one experiment end to end and write its result tables.

    Args:
        cfg: The experiment configuration.
        corpus_root: Path to the Parquet corpus.
        results_dir: Directory the CSV and Markdown artifacts are written to. The
            per-seed, per-cell and paired-contrast tables are rewritten after every regime,
            so a failure late in a four-regime run leaves the completed regimes on disk.
        seeds: Override for ``cfg.seeds``. Fewer than three still raises downstream when
            the stochastic rows are aggregated; the override exists for smoke tests, which
            do not aggregate.
        regimes: Override for ``cfg.regimes``.
        device: Torch device for SGD training and inference.
        run_controls: If True, run the shuffle, untrained and pipeline-sanity controls and
            write ``baselines_controls.csv``. Turned off only by fast smoke tests.
        checkpoint_root: Root for per-regime best-epoch checkpoints.

    Returns:
        The per-run table (``baselines_by_seed.csv``), one row per
        (model, regime, DOF, horizon, run). It is the source of truth; the aggregated
        table is derived from it.

        ``paired_contrasts.csv`` is written alongside it whenever at least one pair of
        :data:`PAIRED_CONTRASTS` is present in the config, with columns
        :data:`PAIRED_CONTRAST_COLUMNS` and one row per (regime, pair, seed, DOF, horizon).
        It is not returned: it is a second view of the same accumulators, and the caller
        that wants it should read the file it is committed as. No file is written when no
        configured pair is present, so its presence is never a promise of contrasts that
        were not computed.

    Raises:
        ValueError: If the reference model is absent from the config, or if a requested
            regime is unknown.
    """
    labels = [m.label for m in cfg.models]
    if PERSISTENCE_LABEL not in labels:
        raise ValueError(
            f"experiment {cfg.name!r} has no {PERSISTENCE_LABEL!r} model; every skill "
            f"score needs its persistence denominator measured over the same windows "
            f"(CLAUDE.md non-negotiable 4). Configured models: {labels}"
        )
    experiment = cfg if seeds is None else _with_seeds(cfg, tuple(int(s) for s in seeds))
    chosen = tuple(regimes) if regimes is not None else experiment.regimes
    spec = window_spec_from_config(experiment.data)
    manifest = load_manifest(corpus_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    by_seed: list[pd.DataFrame] = []
    by_seed_prob: list[pd.DataFrame] = []
    by_cell: list[pd.DataFrame] = []
    control_results: list[ControlResult] = []
    heading_marginals: list[pd.DataFrame] = []
    contrasts: list[pd.DataFrame] = []

    for regime in chosen:
        split = build_split(manifest, _as_regime(regime))
        train = DeckMotionDataset(corpus_root, split, "train", experiment.data, spec)
        val = DeckMotionDataset(
            corpus_root, split, "val", experiment.data, spec, stats=train.norm_stats
        )
        test = DeckMotionDataset(
            corpus_root, split, "test", experiment.data, spec, stats=train.norm_stats
        )
        n_in = len(train.input_columns)
        n_out = len(train.target_columns)
        holder: dict[str, TrainingMoments] = {}

        records: list[RunRecord] = []
        for model_cfg in experiment.models:
            records.extend(
                _fit_one(
                    model_cfg,
                    spec=spec,
                    train=train,
                    val=val,
                    experiment=experiment,
                    moments_holder=holder,
                    device=device,
                    checkpoint_dir=checkpoint_root / experiment.name / regime,
                )
            )

        # One pass, every model, identical windows. Runs are keyed "<label>@<seed>" so that
        # three seeds of one model are three entries rather than one silently overwriting
        # the others -- a dict keyed on the label alone would lose two of every three. The
        # key is split back into `model` and `seed` columns before anything is written, so
        # that every committed table shares one key space.
        keyed = {f"{r.label}@{r.seed}": r for r in records}
        # A probabilistic model is scored for point accuracy through its point projection --
        # the 0.5 quantile, or the Gaussian mean (P5-D5). `point_view` exists so that goes
        # through the SAME `evaluate_models` call as every other row: one pass, one window
        # set, one persistence denominator, one realization bootstrap. Widening
        # `evaluate_models` to accept rank-4 output instead would have put its bitwise-zero
        # reference guarantee and its identity-shared target sums one refactor away from
        # being silently untrue.
        table, accumulators = evaluate_models(
            {
                k: (r.model if r.model.head_kind == "point" else point_view(r.model))
                for k, r in keyed.items()
            },
            test,
            persistence_key=f"{PERSISTENCE_LABEL}@0",
            horizons=experiment.data.horizons,
            fs_hz=experiment.data.fs_hz,
            num_workers=experiment.train.num_workers,
            device=device,
            n_boot=BOOTSTRAP_N_BOOT,
            ci_level=BOOTSTRAP_CI_LEVEL,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        meta = pd.DataFrame(
            [
                {
                    "model": key,
                    "label": record.label,
                    "seed": record.seed,
                    "deterministic": record.deterministic,
                    "n_params": record.n_params,
                    "fit_time_s": record.fit_time_s,
                    # NaN, not a sentinel epoch count: a closed-form model has no epochs,
                    # and 0 would read as "stopped immediately" in the committed CSV.
                    "best_epoch": record.best_epoch,
                    "epochs_run": record.epochs_run,
                    "best_val_loss": record.best_val_loss,
                    "val_loss_name": record.val_loss_name,
                }
                for key, record in keyed.items()
            ]
        )
        table = table.merge(meta, on="model", validate="many_to_one")
        table["model"] = table["label"]
        table["regime"] = regime
        scored = table.drop(columns=["label"])
        by_seed.append(scored)

        # The distributional pass, over the same partition and the same realizations. It is
        # a second pass rather than a branch inside `evaluate_models` for the reason given
        # at the `point_view` call above; `expected_keys` is what asserts the two passes
        # scored the same windows in the same order, so a row's `picp` and its `rmse` can
        # never describe different data.
        prob_models = {k: r.model for k, r in keyed.items() if r.model.head_kind != "point"}
        if prob_models:
            prob_table, _ = evaluate_probabilistic_models(
                prob_models,
                test,
                horizons=experiment.data.horizons,
                fs_hz=experiment.data.fs_hz,
                alpha=GATE5_ALPHA,
                num_workers=experiment.train.num_workers,
                device=device,
                n_boot=BOOTSTRAP_N_BOOT,
                ci_level=BOOTSTRAP_CI_LEVEL,
                bootstrap_seed=BOOTSTRAP_SEED,
                expected_keys=list(test.realization_keys),
            )
            prob_table = prob_table.merge(meta, on="model", validate="many_to_one")
            prob_table["model"] = prob_table["label"]
            prob_table["regime"] = regime
            # `signal_std` is carried over from the point pass rather than recomputed: it is
            # a property of the targets alone, `evaluate_models` already accumulates it once
            # per batch and shares it across every model by object identity (P4-D3), and a
            # second float64 reduction of the same values in a different order would differ
            # in the last bits. It is what makes `width_ratio` -- the sharpness reference
            # that turns "2.3 degrees" into a readable number -- available downstream.
            prob_table = prob_table.drop(columns=["label"]).merge(
                scored[["model", "seed", "dof", "horizon_samples", "signal_std"]],
                on=["model", "seed", "dof", "horizon_samples"],
                validate="one_to_one",
            )
            by_seed_prob.append(prob_table)

        cells = per_cell_metrics(
            accumulators,
            test,
            persistence_key=f"{PERSISTENCE_LABEL}@0",
            horizons=experiment.data.horizons,
            fs_hz=experiment.data.fs_hz,
        )
        cells["regime"] = regime
        # Marginalise *before* splitting the run key. `marginalize_cells` groups on
        # ``model``, so collapsing headings after the split would sum the squared errors of
        # three seeds into one row and report a pooled-over-seeds skill that no run ever
        # achieved. Splitting afterwards is a pure relabelling.
        heading = marginalize_cells(cells, ["heading_deg"]) if regime == "id" else None
        by_cell.append(_split_run_keys(cells))
        if heading is not None:
            heading = _split_run_keys(heading)
            heading["regime"] = regime
            heading_marginals.append(heading)

        # Checkpoint the two accumulating tables to disk once per regime, before the
        # controls run. Deliberately redundant with the final writes after the loop: the
        # shuffle control keeps strict=True because a failed integrity control must stop
        # the sweep loudly rather than be quietly recorded, and these writes are what stop
        # that legitimate stop on regime 3 from also destroying the finished work of
        # regimes 1 and 2 in a run that takes hours. Writing before the controls means the
        # current regime's own scored rows survive its own control failure too. Do not
        # "optimise" these away as duplicated work -- recoverability is the point.
        write_table(pd.concat(by_seed, ignore_index=True), results_dir / "baselines_by_seed.csv")
        write_table(pd.concat(by_cell, ignore_index=True), results_dir / "baselines_by_cell.csv")
        if by_seed_prob:
            per_run_prob = pd.concat(by_seed_prob, ignore_index=True)
            write_table(per_run_prob, results_dir / "probabilistic_by_seed.csv")
            # The *aggregated* probabilistic table is checkpointed per regime as well, which
            # `baselines.csv` is not. The reason is specific rather than a general tidiness
            # preference: this sweep is ~2 days, `probabilistic.csv` is the only file
            # `scripts/gate5.py` reads, and the regime list is ordered so that `id` and
            # `unseen_seastate` -- the two Gate 5 actually turns on -- finish first. Writing
            # it here means `make gate5` is answerable after the first regime instead of
            # only after the last, on a run long enough that the difference is a day.
            write_table(build_probabilistic_table(per_run_prob), results_dir / "probabilistic.csv")

        # Model-vs-model, from the accumulators the scoring pass above already returned:
        # same realizations, same resample weights, no second evaluation and no second
        # bootstrap draw. Two *unpaired* marginal intervals overlapping says nothing about a
        # difference between models scored over identical realizations with strongly
        # correlated errors, and reading them as if it did has already produced a wrong
        # published conclusion here (``docs/protocol.md`` P3-D13, P3-D22).
        #
        # After the two writes above for the same reason the controls are: this can raise on
        # a mistyped contrast label, and that legitimate stop must not also destroy the
        # regime's own scored rows.
        contrast = _contrast_frame(
            accumulators,
            regime=regime,
            dof_names=test.target_columns,
            horizons=experiment.data.horizons,
            fs_hz=experiment.data.fs_hz,
            persistence_key=f"{PERSISTENCE_LABEL}@0",
        )
        if not contrast.empty:
            contrasts.append(contrast)
            write_table(
                pd.concat(contrasts, ignore_index=True), results_dir / "paired_contrasts.csv"
            )

        if run_controls:
            control_results.extend(
                _run_controls(
                    regime=regime,
                    train=train,
                    test=test,
                    experiment=experiment,
                    spec=spec,
                    n_in=n_in,
                    n_out=n_out,
                    records=records,
                    corpus_root=corpus_root,
                    device=device,
                )
            )

    per_run = pd.concat(by_seed, ignore_index=True)
    write_table(per_run, results_dir / "baselines_by_seed.csv")
    write_table(pd.concat(by_cell, ignore_index=True), results_dir / "baselines_by_cell.csv")
    if contrasts:
        write_table(pd.concat(contrasts, ignore_index=True), results_dir / "paired_contrasts.csv")
    aggregated = build_baselines_table(per_run)
    write_table(aggregated, results_dir / "baselines.csv")
    if by_seed_prob:
        per_run_prob = pd.concat(by_seed_prob, ignore_index=True)
        write_table(per_run_prob, results_dir / "probabilistic_by_seed.csv")
        write_table(build_probabilistic_table(per_run_prob), results_dir / "probabilistic.csv")
    controls = controls_table(control_results) if control_results else None
    if controls is not None:
        write_table(controls, results_dir / "baselines_controls.csv")
    markdown = build_baselines_markdown(
        aggregated,
        controls=controls,
        by_heading=heading_marginals[0] if heading_marginals else None,
        # Exactly the condition the `paired_contrasts.csv` write above is guarded by, so
        # the provenance line names that file when and only when it exists: a
        # baselines-only run configures no contrast pair and writes none.
        with_contrasts=bool(contrasts),
        # Explicit, so the document cites the run's own artifacts. Defaulting it would
        # render provenance as bare filenames -- vaguer, but never wrong, which is why the
        # `imu` document previously pointing at the `ideal` CSVs was the failure worth
        # avoiding.
        results_dir=results_dir,
        gate_regime=_gate_regime(chosen),
        gate_dof=GATE_DOF,
        gate_horizon_samples=_gate_horizon(experiment.data.horizons),
    )
    (results_dir / "baselines.md").write_text(markdown, encoding="utf-8")
    return per_run


def _gate_horizon(horizons: tuple[int, ...]) -> int:
    """Return the horizon the Gate 3 threshold is read at, samples.

    :data:`GATE_HORIZON_SAMPLES` -- 100 samples, i.e. 10 s at 10 Hz -- is what the
    production configs report. A cut-down geometry that does not contain it falls back to
    its longest horizon so the report still renders; production is unaffected.

    Args:
        horizons: Reported horizons, samples, ascending.

    Returns:
        The gate horizon, samples.
    """
    return GATE_HORIZON_SAMPLES if GATE_HORIZON_SAMPLES in horizons else horizons[-1]


def _gate_regime(regimes: tuple[str, ...]) -> str:
    """Return the regime the Gate 3 threshold is read on.

    Args:
        regimes: The regimes actually scored in this run.

    Returns:
        ``"id"`` when it was scored -- the gate is an in-distribution difficulty check --
        otherwise the first regime present.
    """
    return "id" if "id" in regimes else regimes[0]


def _run_controls(
    *,
    regime: str,
    train: DeckMotionDataset,
    test: DeckMotionDataset,
    experiment: ExperimentConfig,
    spec: WindowSpec,
    n_in: int,
    n_out: int,
    records: Sequence[RunRecord],
    corpus_root: Path,
    device: str,
) -> list[ControlResult]:
    """Run the negative controls for one regime.

    The shuffle control refits AR on time-shuffled targets. Its null is the **window-mean
    forecast**, not zero skill: a least-squares fit on shuffled targets degenerates to the
    conditional mean, which :func:`dmf.data.normalize.invert_norm` turns back into the
    window mean, and the window mean beats persistence at long horizons on a narrowband
    signal. Testing against zero would report leakage on a clean pipeline.

    Args:
        regime: Regime name, for the control rows.
        train: Training partition, refitted on shuffled targets.
        test: Test partition to score on.
        experiment: The experiment config.
        spec: Window geometry.
        n_in: Input channel count.
        n_out: Target channel count.
        records: The regime's fitted models, for the persistence and untrained subjects.
        corpus_root: Corpus root, re-read independently by the pipeline-sanity control.
        device: Torch device.

    Returns:
        The control outcomes.
    """
    by_label = {r.label: r.model for r in records}
    persistence = by_label[PERSISTENCE_LABEL]
    persistence_pipeline_sanity(test, corpus_root, model=persistence)

    window_mean = DampedPersistence(
        spec.lookback,
        spec.max_horizon,
        n_in,
        n_out,
        tau_samples=torch.full((n_out,), TAU_WINDOW_MEAN, dtype=torch.float64).numpy(),
    )
    window_mean.eval()

    results: list[ControlResult] = []
    ar_cfgs = [m for m in experiment.models if "order" in m.params]
    if ar_cfgs:
        # AR(20) on the full input set is the Gate 3 subject, so it is what the shuffle
        # control refits. Explicit rather than a max() over a predicate: that form returns
        # the first maximal element and so happens to be right only while ar20 precedes
        # ar40 in the config list. `n_input_used` is excluded so that the control can never
        # land on an information-set ablation whatever order that ablation is configured
        # at -- `ar_attitude_only` carried order 20 until the capacity match moved it to 40,
        # and a control run on the ablation would test a model that is not the headline one
        # while depending on config order to notice. The fallback keeps the control running
        # on a cut-down config that ships some other order.
        target = next(
            (m for m in ar_cfgs if int(m.params["order"]) == 20 and "n_input_used" not in m.params),
            ar_cfgs[0],
        )
        shuffled_moments = accumulate_training_moments(
            train,
            max_order=int(target.params["order"]),
            num_workers=experiment.train.num_workers,
            shuffle_targets=True,
        )
        shuffled_model, _ = fit_ar(
            shuffled_moments,
            order=int(target.params["order"]),
            ridge=float(target.params.get("ridge", 0.0)),
            lookback=spec.lookback,
            n_input_channels=n_in,
            n_target_channels=n_out,
        )
        results.append(
            shuffle_control(
                test,
                shuffled_model=shuffled_model,
                window_mean_model=window_mean,
                persistence_model=persistence,
                regime=regime,
                horizons=experiment.data.horizons,
                fs_hz=experiment.data.fs_hz,
                num_workers=experiment.train.num_workers,
                device=device,
            )
        )

    sgd_cfgs = [m for m in experiment.models if MODEL_REGISTRY[m.name].FIT_KIND == "sgd"]
    if sgd_cfgs:
        untrained = _instantiate(sgd_cfgs[0], spec, n_in, n_out, experiment.seeds[0])
        untrained.eval()
        # Scored through its point projection when the subject carries a head, because the
        # control's null and its published Phase 3/4 comparators are point forecasts. What
        # this measures for a quantile head is its untrained MEDIAN, which is the quantity
        # comparable to those rows -- it is not a control on the interval, and no control on
        # an untrained interval exists in this sweep. P4-D15 records the more general
        # version of that gap: `sgd_cfgs[0]` is one model, so no control here covers the
        # architectures the gate is actually about.
        untrained_subject = untrained if untrained.head_kind == "point" else point_view(untrained)
        # strict=False: the outcome is recorded in baselines_controls.csv rather than
        # raised. The protocol's null for this control ("a random-init model must score
        # worse than persistence") is wrong on this task in exactly the way the shuffle
        # control's null was, and for the same reason. A small-weight random
        # initialisation emits something close to zero, which after invert_norm is the
        # window-mean forecast -- and on a narrowband signal the window mean beats
        # persistence from about 2 s out (measured on id/test: window-mean skill +0.37 at
        # 3 s, +0.71 at 5 s). Measured untrained DLinear skill on id/test is +0.32 at 2 s
        # and +0.70 at 5 s, i.e. it reproduces the window mean rather than learning
        # anything. Raising here would fail a clean pipeline.
        #
        # null_model=None is a decision, not an omission. `untrained_control` does accept a
        # null_model, and its docstring records what happens when the window mean is
        # substituted: the same untrained DLinear still removes 17.5% of that null's error
        # at a 2 s horizon on roll (skill -0.0432 against a null of -0.2639). A randomly
        # initialised *linear map of the lookback* is not a null model -- it is a bad filter
        # of genuine past data, and a bad filter of the recent past of a narrowband signal
        # carries real skill. The only initialisation this control is guaranteed to pass is
        # one that emits the window mean exactly, which passes by construction and therefore
        # tests nothing. Since no null makes it sound for a linear architecture, we keep the
        # protocol's literal criterion (null = persistence), which holds at 1 s on every DOF
        # and fails at 5 s on all three, and report the failure rather than tune the null or
        # the tolerance until it disappears. The informative comparison is
        # untrained-versus-trained, which the headline table already carries.
        results.append(
            untrained_control(
                test,
                untrained_model=untrained_subject,
                persistence_model=persistence,
                regime=regime,
                horizons=experiment.data.horizons,
                fs_hz=experiment.data.fs_hz,
                num_workers=experiment.train.num_workers,
                device=device,
                strict=False,
            )
        )
    return results


def _with_seeds(cfg: ExperimentConfig, seeds: tuple[int, ...]) -> ExperimentConfig:
    """Return a copy of the config with different seeds.

    Args:
        cfg: The original configuration.
        seeds: Replacement seeds.

    Returns:
        A new configuration; the original is frozen and unchanged.
    """
    return ExperimentConfig(
        name=cfg.name,
        data=cfg.data,
        models=cfg.models,
        train=cfg.train,
        seeds=seeds,
        regimes=cfg.regimes,
    )


def _as_regime(name: str) -> Regime:
    """Narrow a regime name to the ``Regime`` literal type.

    Args:
        name: Regime name from the config.

    Returns:
        The same string, typed.

    Raises:
        ValueError: If the name is not one of the four regimes.
    """
    if name not in ("id", "unseen_seastate", "unseen_heading", "unseen_vessel"):
        raise ValueError(f"unknown regime {name!r}")
    regime: Regime = name  # type: ignore[assignment]
    return regime
