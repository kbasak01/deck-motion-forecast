"""Phase 10 driver: calibrate the committed interval heads and re-score them.

**It re-scores; it does not retrain.** Every head this module calibrates is loaded from the
checkpoint the Phase 5 sweep committed, through
:func:`dmf.train.experiment.load_or_fit` with ``strict=True``, which raises rather than
falling back to training or to a random initialisation. Nothing under
``artifacts/checkpoints/`` is written. The structural sibling is
:mod:`dmf.eval.matched`, which re-scores the same checkpoints on a different *window
population*; this one re-scores them under a different *post-processing*.

**It writes into no committed directory.** Everything lands under ``results/e05/``.
``results/e03/probabilistic.csv`` and ``results/e03/gate5.csv`` are read and never
regenerated -- there is no code path in this module that can write them. That is how Gate 5's
"report the out-of-distribution degradation, do not fix it" (P5-D2) is kept: the uncalibrated
rows remain exactly as committed and the calibrated arm is *additional*, so both readings are
available to a reader and neither replaces the other.

This is also why the arm is **not** added to ``configs/experiment/e03_probabilistic.yaml``.
Gate 5's Reading A enumerates no models -- it is every row of the ``id`` regime at the gate
cell -- so a new label in that config would silently turn the committed "6 of 6" into
"12 of 12" and rewrite a gate artifact as a side effect of adding a model.

**What it deliberately does not do.** No quiescence pass and no phase-lag pass: the interval
decision rule reads the same fan positions, so "does calibration improve the *decision*" is a
real and better question, but it is a second scoring pass over consecutive windows and P10-D1
scopes it out of the first run rather than doing it badly inside it. No point pass either --
the wrapper's point forecast is bitwise identical to its base's (proved in
``tests/test_conformal.py``), so every point metric is already published in
``results/e03/baselines_by_seed.csv`` and re-deriving it would invite two copies that could
drift.

``signal_std`` is joined from the committed ``results/e03/probabilistic_by_seed.csv`` rather
than recomputed. It is a property of the targets alone -- verified invariant across all 144
``(regime, dof, horizon)`` cells of that file -- and :func:`dmf.eval.report.width_ratio` needs
it. Recomputing it in a different reduction order would differ in the last bits and make the
calibrated rows' ``width_ratio`` incomparable to the uncalibrated ones for no reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from dmf.config import ExperimentConfig, ModelConfig
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import Split, assert_seed_disjoint, build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.eval.conformal import CONFORMAL_MAX_WINDOWS, ConformalFit, calibrate_models
from dmf.eval.gate import coverage_degradation
from dmf.eval.prob_runner import evaluate_probabilistic_models
from dmf.eval.report import build_probabilistic_table, write_table
from dmf.eval.runner import _check_norm_provenance
from dmf.train.experiment import load_or_fit
from dmf.train.registry import MODEL_REGISTRY

__all__ = [
    "CONFORMAL_ARTIFACTS",
    "CONFORMAL_CALIBRATION_COLUMNS",
    "CONFORMAL_SUFFIX",
    "run_conformal",
    "run_regime_conformal",
]

#: Files this driver writes, relative to its output root. The family sibling of
#: ``dmf.eval.scoring.SCORING_ARTIFACTS`` and ``dmf.eval.assemble.ASSEMBLED_ARTIFACTS``; no
#: two of the three may name the same file, which ``tests/test_assemble.py`` asserts.
CONFORMAL_ARTIFACTS: dict[str, str] = {
    "conformal_by_seed": "conformal_by_seed.csv",
    "conformal": "conformal.csv",
    "calibration": "conformal_calibration.csv",
    "degradation": "conformal_degradation.csv",
}

#: Appended to the base label to key a calibrated row. Disjoint from every committed label by
#: construction, so the calibrated table joins the committed one instead of colliding with it.
CONFORMAL_SUFFIX: str = "_conformal"

#: The calibration state, one row per (regime, model, seed, dof, horizon). Everything a
#: reader needs to check that the calibration was fitted where it claims and on how much data.
CONFORMAL_CALIBRATION_COLUMNS: tuple[str, ...] = (
    "regime",
    "model",
    "seed",
    "dof",
    "horizon_samples",
    "gamma",
    "gamma_realization",
    "alpha",
    "n_windows_total",
    "n_windows_used",
    "n_realizations",
    "window_stride",
    "order_index",
    "n_degenerate_cells",
    "fitted_on",
    "norm_stats_fitted_on",
    "calibration_time_s",
)


def _interval_head_configs(cfg: ExperimentConfig) -> list[ModelConfig]:
    """Return the SGD interval heads of an experiment, and nothing else.

    Filtering here rather than after ``load_or_fit`` saves the training-moments pass the
    closed-form rows would trigger: this arm calibrates learned heads, and persistence,
    ``window_mean``, ``ar20`` and ``dlinear_ols`` have no interval to calibrate.

    Args:
        cfg: The experiment whose models are wanted.

    Returns:
        The model configs whose head is not ``point`` and whose fit kind is ``sgd``.
    """
    out: list[ModelConfig] = []
    for model_cfg in cfg.models:
        if model_cfg.head == "point":
            continue
        if MODEL_REGISTRY[model_cfg.name].FIT_KIND != "sgd":
            continue
        out.append(model_cfg)
    return out


def _calibration_rows(
    regime: str,
    label: str,
    seed: int,
    fit: ConformalFit,
    dofs: Sequence[str],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Flatten one fit report into per-cell calibration rows.

    Args:
        regime: Evaluation regime.
        label: Base model label, without the conformal suffix.
        seed: Training seed of the base model.
        fit: The fit report.
        dofs: Target channel names, in the model's channel order.
        horizons: Horizons reported, samples.

    Returns:
        One row per ``(dof, horizon)`` on :data:`CONFORMAL_CALIBRATION_COLUMNS`.
    """
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for channel, dof in enumerate(dofs):
            rows.append(
                {
                    "regime": regime,
                    "model": label,
                    "seed": seed,
                    "dof": dof,
                    "horizon_samples": horizon,
                    "gamma": float(fit.gamma[horizon - 1, channel]),
                    "gamma_realization": float(fit.gamma_realization[horizon - 1, channel]),
                    "alpha": fit.alpha,
                    "n_windows_total": fit.n_windows_total,
                    "n_windows_used": fit.n_windows_used,
                    "n_realizations": fit.n_realizations,
                    "window_stride": fit.window_stride,
                    "order_index": fit.order_index,
                    "n_degenerate_cells": fit.n_degenerate_cells,
                    "fitted_on": fit.fitted_on,
                    "norm_stats_fitted_on": fit.norm_stats_fitted_on,
                    "calibration_time_s": fit.calibration_time_s,
                }
            )
    return pd.DataFrame(rows, columns=list(CONFORMAL_CALIBRATION_COLUMNS))


def run_regime_conformal(
    cfg: ExperimentConfig,
    corpus_root: Path,
    regime: str,
    *,
    alpha: float = 0.1,
    checkpoint_root: Path = Path("artifacts/checkpoints"),
    device: str = "cpu",
    batch_size: int = 4096,
    num_workers: int = 4,
    max_windows: int = CONFORMAL_MAX_WINDOWS,
    signal_std_source: Path = Path("results/e03/probabilistic_by_seed.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate and re-score one regime.

    Args:
        cfg: The Phase 5 experiment whose heads and geometry are reused unchanged.
        corpus_root: Corpus root.
        regime: Evaluation regime.
        alpha: Nominal miscoverage.
        checkpoint_root: Committed checkpoints. **Never written to.**
        device: Torch device.
        batch_size: Windows per batch.
        num_workers: DataLoader workers.
        max_windows: Calibration window cap.
        signal_std_source: Committed per-seed table the ``signal_std`` column is joined from.

    Returns:
        ``(scored, calibration)``. ``scored`` is on the ``probabilistic_by_seed.csv`` schema
        with calibrated labels; ``calibration`` is on
        :data:`CONFORMAL_CALIBRATION_COLUMNS`.

    Raises:
        ValueError: If the experiment carries no SGD interval head, or if the ``signal_std``
            join is not one-to-one.
    """
    spec = window_spec_from_config(cfg.data)
    split: Split = build_split(load_manifest(corpus_root), regime)  # type: ignore[arg-type]
    assert_seed_disjoint(split)
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats)
    test = DeckMotionDataset(corpus_root, split, "test", cfg.data, spec, stats=train.norm_stats)
    # Before anything is built, as P3-D11 requires: a provenance failure invalidates every
    # number this function would produce.
    _check_norm_provenance(test)

    heads = _interval_head_configs(cfg)
    if not heads:
        raise ValueError(f"{cfg.name} carries no SGD interval head to calibrate")

    records = []
    for model_cfg in heads:
        records.extend(
            load_or_fit(
                model_cfg,
                spec=spec,
                train=train,
                val=val,
                experiment=cfg,
                moments_holder={},
                device=device,
                checkpoint_dir=checkpoint_root / cfg.name / regime,
            )
        )
    base = {f"{r.label}@{r.seed}": r.model for r in records}
    fits = calibrate_models(
        base,
        val,
        alpha=alpha,
        max_windows=max_windows,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    wrapped = {f"{key}{CONFORMAL_SUFFIX}": model for key, (model, _) in fits.items()}

    scored, _ = evaluate_probabilistic_models(
        wrapped,
        test,
        horizons=tuple(cfg.data.horizons),
        fs_hz=cfg.data.fs_hz,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        expected_keys=list(test.realization_keys),
    )
    meta = pd.DataFrame(
        [
            {
                "model": f"{r.label}@{r.seed}{CONFORMAL_SUFFIX}",
                "label": f"{r.label}{CONFORMAL_SUFFIX}",
                "seed": r.seed,
                "deterministic": False,
                "n_params": fits[f"{r.label}@{r.seed}"][0].n_fitted_parameters,
                "fit_time_s": float("nan"),
                "best_epoch": pd.NA,
                "epochs_run": pd.NA,
                "best_val_loss": pd.NA,
                "val_loss_name": "conformal",
            }
            for r in records
        ]
    )
    scored = scored.merge(meta, on="model", validate="many_to_one")
    scored["model"] = scored["label"]
    scored["regime"] = regime
    scored = scored.drop(columns=["label"])

    widths = pd.read_csv(signal_std_source)
    widths = (
        widths[widths["regime"] == regime][["dof", "horizon_samples", "signal_std"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    before = len(scored)
    scored = scored.merge(widths, on=["dof", "horizon_samples"], validate="many_to_one")
    if len(scored) != before:
        raise ValueError(
            f"the signal_std join changed the row count from {before} to {len(scored)} at "
            f"regime {regime}; signal_std must be one value per (dof, horizon) and this "
            f"table would otherwise carry a width_ratio computed against the wrong reference"
        )

    calibration = pd.concat(
        [
            _calibration_rows(
                regime,
                key.split("@")[0],
                int(key.split("@")[1]),
                fit,
                test.target_columns,
                tuple(cfg.data.horizons),
            )
            for key, (_, fit) in fits.items()
        ],
        ignore_index=True,
    )
    return scored, calibration


def run_conformal(
    cfg: ExperimentConfig,
    corpus_root: Path,
    *,
    out_root: Path,
    regimes: Sequence[str] | None = None,
    alpha: float = 0.1,
    checkpoint_root: Path = Path("artifacts/checkpoints"),
    device: str = "cpu",
    batch_size: int = 4096,
    num_workers: int = 4,
    max_windows: int = CONFORMAL_MAX_WINDOWS,
) -> dict[str, Path]:
    """Calibrate and re-score every regime, writing after each one.

    Written per regime rather than once at the end, for the reason
    :func:`dmf.train.experiment.run_experiment` gives: this is a multi-hour pass and a
    failure in the last regime must not destroy the finished work of the first three.

    Args:
        cfg: The Phase 5 experiment.
        corpus_root: Corpus root.
        out_root: Where the four tables are written.
        regimes: Regimes to run; defaults to the experiment's own list.
        alpha: Nominal miscoverage.
        checkpoint_root: Committed checkpoints, never written to.
        device: Torch device.
        batch_size: Windows per batch.
        num_workers: DataLoader workers.
        max_windows: Calibration window cap.

    Returns:
        The written paths, keyed as :data:`CONFORMAL_ARTIFACTS`.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    selected = list(regimes) if regimes is not None else list(cfg.regimes)
    # Stage status file, in the shape `scripts/run_e04.sh` writes and
    # `scripts/collect_runtimes.py` parses, so this arm's wall clock is derivable from a log
    # rather than from file timestamps. P10-D3 records why the first run's figure is prose:
    # it predates this file.
    status = Path("artifacts/logs/e05/status_conformal.txt")
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.open("a") as handle:
        handle.write(f"start {datetime.now().astimezone().isoformat()} n={len(selected)}\n")
    by_seed: list[pd.DataFrame] = []
    calibrations: list[pd.DataFrame] = []
    written: dict[str, Path] = {}
    for regime in selected:
        scored, calibration = run_regime_conformal(
            cfg,
            corpus_root,
            regime,
            alpha=alpha,
            checkpoint_root=checkpoint_root,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_windows=max_windows,
        )
        by_seed.append(scored)
        calibrations.append(calibration)
        per_run = pd.concat(by_seed, ignore_index=True)
        written["conformal_by_seed"] = write_table(
            per_run, out_root / CONFORMAL_ARTIFACTS["conformal_by_seed"]
        )
        aggregated = build_probabilistic_table(per_run)
        written["conformal"] = write_table(aggregated, out_root / CONFORMAL_ARTIFACTS["conformal"])
        written["calibration"] = write_table(
            pd.concat(calibrations, ignore_index=True),
            out_root / CONFORMAL_ARTIFACTS["calibration"],
        )
        degradation = coverage_degradation(aggregated)
        if not degradation.empty:
            written["degradation"] = write_table(
                degradation, out_root / CONFORMAL_ARTIFACTS["degradation"]
            )
        with status.open("a") as handle:
            handle.write(f"DONE {regime} {datetime.now().astimezone().isoformat()}\n")
    with status.open("a") as handle:
        handle.write(f"end {datetime.now().astimezone().isoformat()}\n")
    return written
