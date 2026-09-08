"""Scoring an experiment from its committed artifacts, without retraining anything.

``make eval`` must be cheap enough to run on every change, and it must produce numbers that
are the *same* numbers the training sweep produced. Those two requirements are in tension
only if scoring re-fits: :func:`dmf.train.experiment.load_or_fit` resolves each SGD model's
committed ``state_dict`` and refuses to fall back to a random initialisation, and the
closed-form rows are re-solved because a solve is reproducible in a way a 60-epoch run is
not. This module is the driver that puts that together with the four scoring passes.

**One pass per (experiment, regime), over one partition, for every model.** The accuracy
pass (:func:`dmf.eval.runner.evaluate_models`), the per-cell breakdown
(:func:`dmf.eval.runner.per_cell_metrics`), the distributional pass
(:func:`dmf.eval.prob_runner.evaluate_probabilistic_models`), the operational pass
(:func:`dmf.eval.quiescence_runner.evaluate_quiescence`) and the phase-lag pass
(:func:`dmf.eval.phase_runner.evaluate_phase_lag`) all read the *same* test partition built
once here, so no two columns of one row can describe different windows.

**Normalisation provenance is checked before any model is built, not after.** P3-D11: the
statistics must be labelled ``"<regime>/train"`` for *this* regime, and
:func:`dmf.eval.runner._check_norm_provenance` is called directly here rather than being
left to fire inside the accuracy pass. Two reasons, and the second is the operative one:
building the models for a regime costs minutes, and the quiescence pass has no accuracy
pass in front of it when a caller asks for it alone. It is the same function the accuracy
pass calls -- imported, not reimplemented -- so the two cannot drift apart.

**What this module does not do.** It does not aggregate over seeds, it does not render
anything, it does not decide any gate, and it writes no cross-arm table:
:mod:`dmf.eval.assemble` joins the committed per-arm tables into ``ablations.csv``,
``controls.csv``, ``probabilistic_baseline.csv`` and ``reference_reproducibility.csv``
without a corpus, a checkpoint or a GPU, and the two writers share no file name.
``metrics_full.csv`` is written **per run**, one row per
``(model, seed, regime, dof, horizon)``, because that is the source of truth
the aggregate is derived from (the ``baselines_by_seed.csv`` pattern of Phase 3-5).
:func:`dmf.eval.report.aggregate_results` applies the >= 3 seed rule and the P3-D10
deterministic exemption at rendering time.

**Not every arm can be scored on every metric, and that is recorded rather than hidden.**
``configs/data/attitude_only.yaml`` does not forecast ``heave_rate``, so it cannot supply
the quiescence detector's own decision variable (P6-D4 item 3). Rather than crash or
silently emit an empty table, the driver records the reason in
:attr:`ScoringArtifacts.skipped` and carries on.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dmf.config import ExperimentConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import REGIMES, Regime, Split, build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.eval.phase_runner import (
    DEFAULT_PHASE_REALIZATIONS,
    build_phase_dataset,
    evaluate_phase_lag,
    join_phase_lag,
)
from dmf.eval.prob_runner import evaluate_probabilistic_models
from dmf.eval.quiescence_runner import decision_channel_index, evaluate_quiescence
from dmf.eval.report import write_table
from dmf.eval.runner import _check_norm_provenance, evaluate_models, per_cell_metrics
from dmf.models.base import BaseForecaster
from dmf.models.heads import point_view
from dmf.train.closed_form import TrainingMoments
from dmf.train.experiment import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_BOOT,
    BOOTSTRAP_SEED,
    GATE5_ALPHA,
    PERSISTENCE_LABEL,
    RUN_KEY_SEP,
    RunRecord,
    _split_run_keys,
    load_or_fit,
)
from dmf.typedefs import IntArray

__all__ = [
    "SCORING_ARTIFACTS",
    "ScoringArtifacts",
    "score_experiment",
    "score_regime",
]

#: Table name -> file name, for everything this module writes. Named here so a caller can
#: check for the files without hard-coding their spelling, and so the renderer and the Gate
#: 6 predicates read the same mapping the writer used.
SCORING_ARTIFACTS: dict[str, str] = {
    "metrics_full": "metrics_full.csv",
    "metrics_by_cell": "metrics_by_cell.csv",
    "quiescence": "quiescence.csv",
    # Gzipped, alone among the scoring artifacts. The raw per-match dump is ~181 MB of
    # 2M rows and compresses 11x; GitHub hard-rejects a blob over 100 MB, so an
    # uncompressed file could not be committed, and a cited source that is absent from a
    # fresh clone breaks Gate 6 predicate 3 (docs/protocol.md P6-D22). pandas infers the
    # codec from the extension on both read and write, so nothing else changes.
    "quiescence_lead_times": "quiescence_lead_times.csv.gz",
    "probabilistic": "probabilistic_by_seed.csv",
}


@dataclass
class ScoringArtifacts:
    """Everything one scoring run produced.

    Attributes:
        metrics_full: One row per (model, seed, regime, DOF, horizon), with the
            :data:`dmf.eval.metrics.METRIC_COLUMNS`, the skill bootstrap interval, the
            phase-lag columns when the phase pass ran, and the run metadata (parameter
            count, fit time, best epoch) every model-comparison table is required to carry.
        metrics_by_cell: The same, broken out by ``(vessel, ss, heading_deg, speed_kn)``.
        quiescence: The operational table, one row per
            (model, regime, threshold set, rule, sea state).
        quiescence_lead_times: Raw per-match lead times, so the distribution is available
            and not only its quantiles.
        probabilistic: The distributional table, empty when the experiment has no
            probabilistic model.
        skipped: Reasons any pass was not run, keyed ``"<regime>/<pass>"``. A pass that
            could not run is recorded rather than being absent from the output.
        paths: Written files, keyed as :data:`SCORING_ARTIFACTS`.
    """

    metrics_full: pd.DataFrame
    metrics_by_cell: pd.DataFrame
    quiescence: pd.DataFrame
    quiescence_lead_times: pd.DataFrame
    probabilistic: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def _as_regime(name: str) -> Regime:
    """Narrow a regime name to the :data:`dmf.data.splits.Regime` literal.

    Args:
        name: Regime name from a config.

    Returns:
        The same name, typed.

    Raises:
        ValueError: If it is not one of the four regimes.
    """
    if name not in REGIMES:
        raise ValueError(f"unknown regime {name!r}, expected one of {list(REGIMES)}")
    return name


def _run_metadata(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Build the per-run metadata frame joined onto every metric row.

    Parameter counts and wall-clock fit times appear in every model comparison table
    (``CLAUDE.md`` §Style, the plan's reporting rules), and ``deterministic`` is what lets
    :func:`dmf.eval.report.aggregate_results` apply the P3-D10 exemption instead of
    refusing a closed-form model's single row.

    Args:
        records: The run records for one regime.

    Returns:
        One row per run, keyed by the composite run key.
    """
    return pd.DataFrame(
        [
            {
                "model": f"{record.label}{RUN_KEY_SEP}{record.seed}",
                "label": record.label,
                "seed": record.seed,
                "deterministic": record.deterministic,
                "n_params": record.n_params,
                "fit_time_s": record.fit_time_s,
                "best_epoch": record.best_epoch,
                "epochs_run": record.epochs_run,
                "best_val_loss": record.best_val_loss,
                "val_loss_name": record.val_loss_name,
            }
            for record in records
        ]
    )


def _quiescence_block(dataset: DeckMotionDataset) -> str | None:
    """Return why the quiescence pass cannot run on this arm, or None if it can.

    Args:
        dataset: The test partition.

    Returns:
        The reason, verbatim from :func:`dmf.eval.quiescence_runner.decision_channel_index`,
        or None.
    """
    try:
        decision_channel_index(dataset.target_columns)
    except ValueError as error:
        return str(error)
    return None


def score_regime(
    cfg: ExperimentConfig,
    corpus_root: Path,
    regime: str,
    *,
    device: str = "cpu",
    checkpoint_root: Path = Path("artifacts/checkpoints"),
    with_quiescence: bool = True,
    with_phase_lag: bool = True,
    n_phase_realizations: int = DEFAULT_PHASE_REALIZATIONS,
    batch_size: int = 4096,
    test_origins: IntArray | None = None,
) -> ScoringArtifacts:
    """Score one experiment on one regime, from committed artifacts.

    Args:
        cfg: The experiment configuration. Its ``data`` block defines the arm, and the
            reference model :data:`dmf.train.experiment.PERSISTENCE_LABEL` must be among
            its models: every skill score needs its persistence denominator measured over
            these same windows.
        corpus_root: Path to the Parquet corpus.
        regime: One of the four evaluation regimes.
        device: Torch device the models run on.
        checkpoint_root: Root of the committed checkpoints; the directory read is
            ``checkpoint_root / cfg.name / regime``.
        with_quiescence: Whether to run the operational pass.
        with_phase_lag: Whether to run the stride-1 phase pass and join its columns.
        n_phase_realizations: Realizations the phase pass sub-samples (P6-D3).
        batch_size: Windows per batch. Speed and memory only.
        test_origins: Forecast origins the **test** partition is restricted to, or None for
            every window the geometry admits. This is the matched-origin path of P6-D4
            item 1: the lookback arms are only comparable on the intersection of their
            origin sets, and that intersection has to be imposed at scoring time because
            each arm's own run scores its own full set.

            Only the test partition is restricted. Fitting is untouched -- the closed-form
            rows are re-solved from the same full training split the sweep used and the SGD
            rows load the same checkpoints -- so a matched table and the arm's committed
            table differ in their scored window set and in nothing else. Passing it
            together with ``with_quiescence`` or ``with_phase_lag`` is refused: the phase
            pass builds its own stride-1 dataset and the quiescence pass scores sustained
            intervals over consecutive windows, so neither is defined on a thinned
            origin set, and silently reporting them beside a matched accuracy row would
            put two window populations in one table.

    Returns:
        The artifacts for this regime, unwritten.

    Raises:
        ValueError: If the reference model is absent, if the regime is unknown, if the
            normalisation provenance is wrong, or if ``test_origins`` is combined with the
            quiescence or phase-lag pass.
        FileNotFoundError: If a checkpoint an SGD row needs is absent.
    """
    labels = [model.label for model in cfg.models]
    if PERSISTENCE_LABEL not in labels:
        raise ValueError(
            f"experiment {cfg.name!r} has no {PERSISTENCE_LABEL!r} model; every skill "
            f"score needs its persistence denominator measured over the same windows "
            f"(CLAUDE.md non-negotiable 4). Configured models: {labels}"
        )
    if test_origins is not None and (with_quiescence or with_phase_lag):
        raise ValueError(
            "test_origins restricts the test partition to a matched origin set, on which "
            "neither the quiescence pass (it scores sustained intervals over consecutive "
            "windows) nor the phase pass (it builds its own stride-1 dataset) is defined. "
            "Pass with_quiescence=False and with_phase_lag=False, so that a matched table "
            "carries only the columns that were matched"
        )
    spec = window_spec_from_config(cfg.data)
    split: Split = build_split(load_manifest(corpus_root), _as_regime(regime))
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats)
    test = DeckMotionDataset(
        corpus_root, split, "test", cfg.data, spec, stats=train.norm_stats, origins=test_origins
    )
    # P3-D11, before any model is built: a provenance failure invalidates every number this
    # function would produce, and building the models costs minutes.
    _check_norm_provenance(test)

    holder: dict[str, TrainingMoments] = {}
    records: list[RunRecord] = []
    for model_cfg in cfg.models:
        records.extend(
            load_or_fit(
                model_cfg,
                spec=spec,
                train=train,
                val=val,
                experiment=cfg,
                moments_holder=holder,
                device=device,
                checkpoint_dir=checkpoint_root / cfg.name / regime,
            )
        )
    keyed: dict[str, RunRecord] = {f"{r.label}{RUN_KEY_SEP}{r.seed}": r for r in records}
    reference = f"{PERSISTENCE_LABEL}{RUN_KEY_SEP}0"

    # A probabilistic model is scored for point accuracy through its own point projection
    # (P5-D5), so it goes through the SAME accuracy pass as every other row: one window
    # set, one persistence denominator, one realization bootstrap.
    point_models = {
        key: (record.model if record.model.head_kind == "point" else point_view(record.model))
        for key, record in keyed.items()
    }
    table, accumulators = evaluate_models(
        point_models,
        test,
        persistence_key=reference,
        horizons=cfg.data.horizons,
        fs_hz=cfg.data.fs_hz,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        device=device,
        n_boot=BOOTSTRAP_N_BOOT,
        ci_level=BOOTSTRAP_CI_LEVEL,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    table["regime"] = regime

    skipped: dict[str, str] = {}
    if with_phase_lag:
        phase_dataset = build_phase_dataset(
            corpus_root,
            split,
            cfg.data,
            train.norm_stats,
            n_realizations=n_phase_realizations,
        )
        phase = evaluate_phase_lag(
            point_models,
            phase_dataset,
            horizons=cfg.data.horizons,
            fs_hz=cfg.data.fs_hz,
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            device=device,
        )
        table = join_phase_lag(table, phase)
    else:
        skipped[f"{regime}/phase_lag"] = "not requested"

    cells = per_cell_metrics(
        accumulators,
        test,
        persistence_key=reference,
        horizons=cfg.data.horizons,
        fs_hz=cfg.data.fs_hz,
    )
    cells["regime"] = regime

    prob_models: dict[str, BaseForecaster] = {
        key: record.model for key, record in keyed.items() if record.model.head_kind != "point"
    }
    probabilistic = pd.DataFrame()
    if prob_models:
        probabilistic, _ = evaluate_probabilistic_models(
            prob_models,
            test,
            horizons=cfg.data.horizons,
            fs_hz=cfg.data.fs_hz,
            alpha=GATE5_ALPHA,
            batch_size=batch_size,
            num_workers=cfg.train.num_workers,
            device=device,
            n_boot=BOOTSTRAP_N_BOOT,
            ci_level=BOOTSTRAP_CI_LEVEL,
            bootstrap_seed=BOOTSTRAP_SEED,
            # Asserts the two passes scored the same realizations in the same order, so a
            # row's picp and its rmse can never describe different data.
            expected_keys=list(test.realization_keys),
        )
        probabilistic["regime"] = regime

    quiescence = pd.DataFrame()
    lead_times = pd.DataFrame()
    if with_quiescence:
        blocked = _quiescence_block(test)
        if blocked is None:
            quiescence, lead_times = evaluate_quiescence(
                {key: record.model for key, record in keyed.items()},
                test,
                fs_hz=cfg.data.fs_hz,
                batch_size=batch_size,
                num_workers=cfg.train.num_workers,
                device=device,
            )
        else:
            skipped[f"{regime}/quiescence"] = blocked
    else:
        skipped[f"{regime}/quiescence"] = "not requested"

    meta = _run_metadata(records)
    scored = table.merge(meta, on="model", validate="many_to_one")
    scored["model"] = scored["label"]
    scored = scored.drop(columns=["label"])
    if not probabilistic.empty:
        probabilistic = probabilistic.merge(meta, on="model", validate="many_to_one")
        probabilistic["model"] = probabilistic["label"]
        probabilistic = probabilistic.drop(columns=["label"])
    for frame in (quiescence, lead_times):
        if not frame.empty:
            frame["model"] = frame["model"].astype(str)

    return ScoringArtifacts(
        metrics_full=scored,
        metrics_by_cell=_split_run_keys(cells),
        quiescence=_split_quiescence_keys(quiescence),
        quiescence_lead_times=_split_quiescence_keys(lead_times),
        probabilistic=probabilistic,
        skipped=skipped,
    )


def _split_quiescence_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Split composite run keys in a quiescence table, under a non-colliding column name.

    Two things make this different from :func:`dmf.train.experiment._split_run_keys`, and
    both are properties of the quiescence tables rather than preferences.

    **The training seed is called ``train_seed`` here.**
    :data:`dmf.eval.quiescence_runner.LEAD_TIME_COLUMNS` already has a ``seed`` column and
    it means the **realization** seed -- the corpus coordinate that sits beside ``vessel``,
    ``ss``, ``heading_deg`` and ``speed_kn`` -- whereas ``seed`` in every accuracy table
    means the *training* seed. One column name meaning two things in one results directory
    is how a table gets joined wrongly and read confidently, so the training seed takes a
    distinct name in these two tables and only in these two.

    **The synthetic detectors have no run key.**
    :func:`dmf.eval.quiescence_runner.evaluate_quiescence` adds its own rows for
    :data:`dmf.eval.quiescence_runner.SYNTHETIC_DETECTORS` -- ``always_quiescent`` and
    ``rate_matched`` -- which are detectors rather than runs of any model. They are split
    out, labelled ``train_seed = -1``, and put back, rather than being made to carry a seed
    they do not have. The split is on the presence of the run-key separator, so a third
    synthetic detector needs no change here.

    Args:
        frame: A quiescence summary or lead-time table. Not mutated.

    Returns:
        The frame with ``model`` holding the bare label and a ``train_seed`` column beside
        it. ``train_seed = -1`` marks a row that is not a run of any model.
    """
    if frame.empty:
        return frame
    keys = frame["model"].astype(str)
    composite = keys.str.contains(RUN_KEY_SEP, regex=False)
    position = list(frame.columns).index("model") + 1
    runs = frame.loc[composite].copy()
    if not runs.empty:
        parts = runs["model"].astype(str).str.rsplit(RUN_KEY_SEP, n=1, expand=True)
        runs["model"] = parts[0]
        runs.insert(position, "train_seed", parts[1].astype(int))
    synthetic = frame.loc[~composite].copy()
    if not synthetic.empty:
        synthetic.insert(position, "train_seed", -1)
    return pd.concat([runs, synthetic], ignore_index=True)


def score_experiment(
    cfg: ExperimentConfig,
    corpus_root: Path,
    *,
    results_dir: Path,
    regimes: Sequence[str] | None = None,
    device: str = "cpu",
    checkpoint_root: Path = Path("artifacts/checkpoints"),
    with_quiescence: bool = True,
    with_phase_lag: bool = True,
    n_phase_realizations: int = DEFAULT_PHASE_REALIZATIONS,
    batch_size: int = 4096,
    write: bool = True,
) -> ScoringArtifacts:
    """Score an experiment on every requested regime and write its tables.

    Args:
        cfg: The experiment configuration.
        corpus_root: Path to the Parquet corpus.
        results_dir: Directory the CSVs are written to. Named by the caller, never derived
            from ``cfg.name`` here: ``results/``, ``results/imu/``, ``results/e02/`` and
            ``results/e03/`` are the Gate 3-5 audit trails and are read, never regenerated.
        regimes: Override for ``cfg.regimes``.
        device: Torch device the models run on.
        checkpoint_root: Root of the committed checkpoints.
        with_quiescence: Whether to run the operational pass.
        with_phase_lag: Whether to run the phase pass and join its columns.
        n_phase_realizations: Realizations the phase pass sub-samples (P6-D3).
        batch_size: Windows per batch.
        write: Whether to write the CSVs. False is for tests and for callers that want the
            frames only.

    Returns:
        The concatenated artifacts over the scored regimes.

    Raises:
        ValueError: If no regime was requested, or as :func:`score_regime` raises.
    """
    chosen = tuple(regimes) if regimes is not None else tuple(cfg.regimes)
    if not chosen:
        raise ValueError(f"experiment {cfg.name!r} names no regime to score")

    parts: list[ScoringArtifacts] = []
    for regime in chosen:
        parts.append(
            score_regime(
                cfg,
                corpus_root,
                regime,
                device=device,
                checkpoint_root=checkpoint_root,
                with_quiescence=with_quiescence,
                with_phase_lag=with_phase_lag,
                n_phase_realizations=n_phase_realizations,
                batch_size=batch_size,
            )
        )

    def _stack(name: str) -> pd.DataFrame:
        frames = [getattr(part, name) for part in parts if not getattr(part, name).empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    combined = ScoringArtifacts(
        metrics_full=_stack("metrics_full"),
        metrics_by_cell=_stack("metrics_by_cell"),
        quiescence=_stack("quiescence"),
        quiescence_lead_times=_stack("quiescence_lead_times"),
        probabilistic=_stack("probabilistic"),
        skipped={key: value for part in parts for key, value in part.skipped.items()},
    )
    if write:
        for name, filename in SCORING_ARTIFACTS.items():
            frame = getattr(combined, name)
            # Deliberately no empty-table placeholder: `write_table` refuses an empty frame
            # because an empty CSV in results/ cannot be told from a table with nothing to
            # report, and `skipped` is where the absence is explained.
            if not frame.empty:
                combined.paths[name] = write_table(frame, results_dir / filename)
    return combined
