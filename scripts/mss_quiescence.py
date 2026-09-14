#!/usr/bin/env python3
"""Run the quiescent-window detector on the MSS S175 trajectories.

Phase 8 carry-forward delta 7 (`docs/IMPLEMENTATION_PLAN.md`): the operational metric is
the part of the MSS test most exposed to a units error, and it had not been run on the MSS
records at all. This closes that gap. Nothing is retrained: the checkpoints are the
`unseen_vessel` ones `scripts/mss_evaluate.py` scores -- the regime that holds the S175 out
of training entirely -- and the normalisation scale is the corpus `unseen_vessel/train`
split's, never re-derived from the MSS record (delta 3).

Three properties this script exists to preserve, each of which is silent in the output if
it is got wrong:

* **The base rate is beside every F1.** `CLAUDE.md` §Known traps. At SS5 head seas an MSS
  record spends most of its time inside permissive limits, so a permissive F1 read on its
  own says more about the sea state than about the model.
* **The geometry is the corpus geometry.** `dmf.eval.external.evaluate_quiescence_
  trajectories` imports P6-D2's decision times, thresholding fold, exclusion rule and
  matching from `dmf.eval.quiescence_runner` rather than restating them, and
  `tests/test_external.py` scores one fabricated realization through both paths and
  compares the two tables row by row. Without that, an MSS quiescence number would not be
  the same quantity as the committed `results/e04/quiescence.csv` one and nothing in the
  CSV would say so.
* **Units are asserted before scoring.** The landing limits are absolute -- 3.0 deg /
  2.0 deg / 0.8 m/s permissive, 1.5 / 1.0 / 0.4 strict. A radians-for-degrees error cancels
  exactly out of a skill score and does not cancel out of these, so every record goes
  through `assert_corpus_units` first and every row carries the record's peak roll, pitch
  and heave rate beside the limits that were applied to it.

Records are **not pooled**: one row per (model, record, threshold set, rule), so the >= 3
MSS seeds of a cell can be aggregated with a spread downstream (non-negotiable 5).

No sign ablation here, unlike `scripts/mss_evaluate.py`. The truth side of this metric
thresholds `|roll|`, `|pitch|` and `|heave_rate|`, so a global sign flip leaves it exactly
unchanged; only the models' response to sign-flipped *inputs* would move, which is a
question about the models rather than about the detector, and it is already asked in the
skill table.

Both sides are simulations. Nothing here is validation against a real ship.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from dmf.config import ExperimentConfig, load_experiment
from dmf.data.dataset import DeckMotionDataset
from dmf.data.normalize import NormStats
from dmf.data.splits import build_split, load_manifest
from dmf.data.windows import WindowSpec, window_spec_from_config
from dmf.eval.external import assert_corpus_units as assert_external_units
from dmf.eval.external import evaluate_quiescence_trajectories
from dmf.models.base import ForecastModel
from dmf.train.closed_form import TrainingMoments
from dmf.train.experiment import RunRecord, load_or_fit

#: Motion channels the unit assertion operates on.
MOTION_CHANNELS: tuple[str, ...] = (
    "roll",
    "pitch",
    "heave",
    "roll_rate",
    "pitch_rate",
    "heave_rate",
)

#: Models carried into the MSS quiescence table.
#:
#: The same set `scripts/mss_evaluate.py` scores, so the two MSS tables describe the same
#: models on the same records. `persistence` and `window_mean` are kept even though a
#: constant forecast cannot express a transition and therefore predicts **no onset at all**
#: by construction: dropping them would hide a structural property of the metric, and
#: `n_onsets_pred` is in the table precisely so a zero can be read for what it is.
MSS_MODEL_LABELS: tuple[str, ...] = (
    "persistence",
    "window_mean",
    "damped_persistence",
    "ar10",
    "ar20",
    "ar40",
    "dlinear_ols",
    "dlinear",
    "tcn",
    "lstm",
    "transformer",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mss/s175_ss5.yaml"))
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiment/e02_deep.yaml"),
        help="Experiment whose checkpoints are evaluated. The default arm; do not point "
        "this at an ablation (delta 5).",
    )
    parser.add_argument(
        "--regime",
        default="unseen_vessel",
        help="Regime whose checkpoints and training-split normalisation statistics are "
        "used. `unseen_vessel` holds the S175 out of training, which is what makes the "
        "MSS rows comparable to the committed corpus ones.",
    )
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--grid-kind", default="mss", choices=("mss", "corpus"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Windows per forward pass. Affects speed and memory only.",
    )
    return parser


def _load_models(
    experiment_path: Path, regime: str, corpus_root: Path, checkpoint_root: Path, device: str
) -> tuple[dict[str, ForecastModel], ExperimentConfig, tuple[WindowSpec, NormStats]]:
    """Load fitted models and the corpus training-split statistics.

    Copied from `scripts/mss_evaluate.py`, which copies `dmf.eval.scoring.score_regime`, so
    that the models scored here are the same objects the committed tables were produced
    from: SGD rows load their checkpoints, closed-form rows are re-solved from one shared
    moments pass. Nothing is trained and nothing under `artifacts/` is written.

    Args:
        experiment_path: Experiment config whose checkpoints are loaded.
        regime: Regime whose checkpoints and training statistics are used.
        corpus_root: Parquet corpus root.
        checkpoint_root: Root of the committed checkpoints.
        device: Torch device the loaded models are placed on.

    Returns:
        Tuple ``(models, experiment_config, (window_spec, norm_stats))``, with models keyed
        ``label|seed``.
    """
    cfg = load_experiment(experiment_path)
    spec = window_spec_from_config(cfg.data)
    split = build_split(load_manifest(corpus_root), regime)  # type: ignore[arg-type]
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats)
    holder: dict[str, TrainingMoments] = {}
    records: list[RunRecord] = []
    for model_cfg in cfg.models:
        if model_cfg.label not in MSS_MODEL_LABELS:
            continue
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
    models: dict[str, ForecastModel] = {f"{r.label}|{r.seed}": r.model for r in records}
    return models, cfg, (spec, train.norm_stats)


def _report(table: pd.DataFrame) -> None:
    """Print the headline F1 and base rate per cell, per threshold set, to stderr.

    **Per cell, never pooled across the grid.** P6-D7 item 3 forbids pooling F1 across sea
    states and P6-D19 extended that to the cell, because scorability is a property of
    ``(sea state, heading, speed)``: on these MSS records the SS5 head-seas cells hold *no*
    scorable permissive onset at all, and a grid-pooled F1 would silently average them with
    the bow-quartering cells that do. A cell with no scorable onset is marked and not
    scored, exactly as the committed corpus table marks its own.

    Counts are pooled over the cell's records before the ratio is taken -- the way the
    corpus table pools its realizations within a cell -- because F1 is a ratio and a mean of
    per-record F1s is not the F1 of the pooled counts. The per-record rows are in the CSV,
    which is where the spread over seeds lives.

    Args:
        table: The emitted quiescence table.
    """
    point = table[table["rule"] == "point"]
    keys = ["threshold_set", "heading_deg", "speed_kn", "model"]
    grouped = point.groupby(keys, as_index=False).agg(
        n_true=("n_onsets_true", "sum"),
        n_pred=("n_onsets_pred", "sum"),
        n_matched=("n_matched", "sum"),
        base_rate=("base_rate", "mean"),
        sample_f1=("always_yes_sample_f1", "mean"),
        false_alarms=("false_alarms_per_min", "mean"),
        lead=("median_lead_time_s", "median"),
    )
    denominator = grouped["n_pred"] + grouped["n_true"]
    grouped["f1"] = 2.0 * grouped["n_matched"] / denominator.where(denominator > 0)
    for cell, block in grouped.groupby(["threshold_set", "heading_deg", "speed_kn"]):
        threshold_set = str(cell[0])
        heading, speed = float(str(cell[1])), float(str(cell[2]))
        head = (
            f"[quiescence] --- {threshold_set} | heading {heading:5.1f} deg |"
            f" {speed:4.1f} kn | base rate {float(block['base_rate'].iloc[0]):.4f}"
            f" | always-yes per-sample F1 {float(block['sample_f1'].iloc[0]):.4f}"
            f" | true onsets {int(block['n_true'].iloc[0])}"
        )
        if int(block["n_true"].iloc[0]) == 0:
            print(f"{head} | NOT SCORABLE ---", file=sys.stderr)
            continue
        print(f"{head} ---", file=sys.stderr)
        for record in block.sort_values("f1", ascending=False).to_dict("records"):
            print(
                f"[quiescence]   {str(record['model']):22s}"
                f" F1 {float(record['f1']):6.4f}"
                f"  n_pred {int(record['n_pred']):6d}"
                f"  FA/min {float(record['false_alarms']):7.3f}"
                f"  median lead {float(record['lead']):6.2f} s",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    """Score the MSS records on the quiescent-window detector and write the table.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    run_cfg = yaml.safe_load(args.config.read_text())
    fs = float(run_cfg["record"]["fs_hz"])

    print(
        f"[quiescence] loading checkpoints: {args.experiment.name} / {args.regime}",
        file=sys.stderr,
    )
    models, cfg, (spec, stats) = _load_models(
        args.experiment, args.regime, args.corpus_root, args.checkpoint_root, args.device
    )
    print(f"[quiescence] {len(models)} fitted model instances", file=sys.stderr)
    print(
        f"[quiescence] norm stats fitted_on={stats.fitted_on!r} (must be a train split)",
        file=sys.stderr,
    )

    tables: list[pd.DataFrame] = []
    for heading in run_cfg["headings_deg"]:
        for speed in run_cfg["speeds_kn"]:
            pattern = f"S175_h{float(heading):05.1f}_u{float(speed):04.1f}_s*.csv"
            paths = sorted((args.mss_dir / args.grid_kind).glob(pattern))
            if not paths:
                print(f"[quiescence] no records for {heading}/{speed}", file=sys.stderr)
                continue
            frames = []
            for path in paths:
                frame = pd.read_csv(path)
                # Name the realization by its file stem so the MSS seed survives into the
                # table: the >= 3-seed rule has to key on the seed, not on position.
                frame.attrs["record"] = path.stem
                frames.append(frame)
            # Delta 2, at the boundary, and it matters more here than in the skill table:
            # a factor of 57.3 cancels exactly out of a ratio and does not cancel out of an
            # absolute threshold.
            assert_external_units(frames, MOTION_CHANNELS)
            table = evaluate_quiescence_trajectories(
                models,
                frames,
                stats=stats,
                spec=spec,
                input_channels=cfg.data.input_channels,
                target_dofs=cfg.data.target_dofs,
                fs_hz=fs,
                device=args.device,
                batch_size=args.batch_size,
            )
            table.insert(0, "heading_deg", float(heading))
            table.insert(1, "speed_kn", float(speed))
            tables.append(table)
            print(
                f"[quiescence] heading {float(heading):5.1f} {float(speed):4.1f} kn: "
                f"{len(frames)} records, {len(table)} rows",
                file=sys.stderr,
            )

    if not tables:
        print("[quiescence] no records matched the configured grid", file=sys.stderr)
        return 1

    out = pd.concat(tables, ignore_index=True)
    out.insert(0, "ss", str(run_cfg["sea_state"]["name"]))
    out.insert(0, "grid_kind", args.grid_kind)
    out.insert(0, "regime", args.regime)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "quiescence_mss.csv"
    out.to_csv(path, index=False)
    print(f"[quiescence] wrote {path} ({len(out)} rows)", file=sys.stderr)
    _report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
