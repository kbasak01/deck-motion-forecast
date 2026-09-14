#!/usr/bin/env python3
"""Evaluate corpus-trained checkpoints on MSS S175 trajectories, without retraining.

Phase 8, and the point of the phase. The models are loaded from
`artifacts/checkpoints/deep/unseen_vessel/` -- the regime that holds the S175 out of training
entirely -- and scored on MSS strip-theory trajectories they have never seen. The comparison
baseline is the committed `unseen_vessel` rows for the same cells in
`results/e04/metrics_by_cell.csv`, so the only thing differing between the two columns is
which generator produced the S175 motion.

Four things this script does because the carry-forward deltas require them:

* normalisation statistics come from the corpus `unseen_vessel/train` split and are never
  re-derived from the MSS record (delta 3);
* `persistence`, `window_mean` and `dlinear_ols` are recomputed on the MSS trajectories
  themselves, so the skill denominator is measured on the same data as the numerator
  (delta 4);
* the default data arm only -- lookback 200, no ablations, no sea-state conditioning
  (delta 5);
* three seeds, so every model-vs-model statement has a spread (delta 6).

It also runs a **sign-flip ablation**. The MSS-to-corpus sign convention is a derivation
(P8-D3), not a measurement, and a sign error does not cancel in the skill score the way a
units error does. Running both conventions shows whether the conclusion depends on it.

Both sides are simulations. Nothing here is validation against a real ship.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dmf.config import load_experiment
from dmf.data.dataset import DeckMotionDataset
from dmf.data.splits import build_split, load_manifest
from dmf.data.windows import window_spec_from_config
from dmf.eval.external import assert_corpus_units as assert_external_units
from dmf.eval.external import evaluate_trajectories
from dmf.mss.convert import MSS_TO_CORPUS_SIGN
from dmf.train.closed_form import TrainingMoments
from dmf.train.experiment import RunRecord, load_or_fit

#: Models carried into the MSS table. The three baselines are mandatory (delta 4); the deep
#: rows are what the phase is testing. AR rows are dropped only to keep the table readable --
#: they are in the committed corpus table if a reader wants them.
#: Run key of the skill denominator. Models are keyed ``label|seed``; persistence is
#: closed-form and therefore deterministic, so it carries seed 0 only.
PERSISTENCE_RUN_KEY: str = "persistence|0"

#: Motion channels the unit assertion and the rescaling control operate on.
MOTION_CHANNELS: tuple[str, ...] = (
    "roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate",
)

MSS_MODEL_LABELS: tuple[str, ...] = (
    "persistence",
    "window_mean",
    "dlinear_ols",
    "dlinear",
    "tcn",
    "lstm",
    "transformer",
)


def build_parser() -> argparse.ArgumentParser:
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
        "MSS record comparable to the committed corpus rows.",
    )
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--grid-kind", default="mss", choices=("mss", "corpus"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rescale-to-corpus",
        action="store_true",
        help="Control for the amplitude gap: rescale each MSS channel so its RMS matches "
        "the corpus mean for that (heading, speed, dof) cell. Our generator runs ~1.6x hot "
        "against strip theory (P8-D7), so MSS records present to the model at roughly half "
        "the amplitude anything in training had. Skill is scale-invariant, but the "
        "*normalised input* is not, so this separates a normalisation-range effect from a "
        "structural one. Shape is untouched; only the per-channel scale changes.",
    )
    parser.add_argument(
        "--sign-ablation",
        action="store_true",
        help="Also score with heave's sign convention inverted, to show whether the "
        "conclusion depends on the P8-D3 derivation.",
    )
    return parser


def _load_models(
    experiment_path: Path, regime: str, corpus_root: Path, checkpoint_root: Path, device: str
) -> tuple[dict[str, object], object, object]:
    """Load fitted models and the corpus training-split statistics.

    Mirrors `dmf.eval.scoring.score_regime` so that the models here are the same objects
    the committed tables were produced from: SGD rows load their checkpoints, closed-form
    rows are re-solved from one shared moments pass.
    """
    cfg = load_experiment(experiment_path)
    spec = window_spec_from_config(cfg.data)
    split = build_split(load_manifest(corpus_root), regime)  # type: ignore[arg-type]
    train = DeckMotionDataset(corpus_root, split, "train", cfg.data, spec)
    val = DeckMotionDataset(
        corpus_root, split, "val", cfg.data, spec, stats=train.norm_stats
    )
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
    models = {f"{r.label}|{r.seed}": r.model for r in records}
    return models, cfg, (spec, train.norm_stats)


def _rescale_to_corpus(
    frames: list[pd.DataFrame], corpus_root: Path, heading: float, speed: float
) -> list[pd.DataFrame]:
    """Scale each MSS channel to the corpus mean RMS for this cell.

    Per-channel multiplication by a constant. It changes the amplitude the model sees after
    normalisation and leaves every phase, period and cross-channel relationship untouched,
    which is exactly the separation the control needs.

    Args:
        frames: MSS records for one cell.
        corpus_root: Parquet corpus root.
        heading: Encounter angle, degrees.
        speed: Forward speed, knots.

    Returns:
        Rescaled copies. Channels the corpus has at zero amplitude are left alone, since a
        scale factor is undefined there (head-seas roll is the P1-D2 floor).
    """
    pattern = f"SS5_h{heading:05.1f}_u{speed:04.1f}_s*.parquet"
    corpus = [pd.read_parquet(p) for p in sorted((corpus_root / "s175").glob(pattern))]
    out: list[pd.DataFrame] = []
    for frame in frames:
        scaled = frame.copy()
        scaled.attrs.update(frame.attrs)
        for channel in MOTION_CHANNELS:
            target = float(np.mean([np.sqrt(np.mean(c[channel].to_numpy() ** 2)) for c in corpus]))
            have = float(np.sqrt(np.mean(frame[channel].to_numpy() ** 2)))
            if have > 1e-12 and target > 1e-12:
                scaled[channel] = frame[channel].to_numpy() * (target / have)
        out.append(scaled)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cfg = yaml.safe_load(args.config.read_text())
    fs = float(run_cfg["record"]["fs_hz"])

    print(f"[eval] loading checkpoints: {args.experiment.name} / {args.regime}", file=sys.stderr)
    models, cfg, (spec, stats) = _load_models(
        args.experiment, args.regime, args.corpus_root, args.checkpoint_root, args.device
    )
    print(f"[eval] {len(models)} fitted model instances", file=sys.stderr)
    print(
        f"[eval] norm stats fitted_on={stats.fitted_on!r} (must be a train split)",
        file=sys.stderr,
    )

    conventions: list[tuple[str, dict[str, float]]] = [("nominal", dict(MSS_TO_CORPUS_SIGN))]
    if args.sign_ablation:
        flipped = {k: -v for k, v in MSS_TO_CORPUS_SIGN.items()}
        conventions.append(("sign_flipped", flipped))

    tables: list[pd.DataFrame] = []
    for conv_name, _signs in conventions:
        for heading in run_cfg["headings_deg"]:
            for speed in run_cfg["speeds_kn"]:
                pattern = f"S175_h{float(heading):05.1f}_u{float(speed):04.1f}_s*.csv"
                paths = sorted((args.mss_dir / args.grid_kind).glob(pattern))
                if not paths:
                    print(f"[eval] no records for {heading}/{speed}", file=sys.stderr)
                    continue
                frames = []
                for path in paths:
                    frame = pd.read_csv(path)
                    # Name the realization by its file stem so the MSS seed survives into
                    # the results table. A positional index would silently renumber if a
                    # glob ever returned a different set, and the three-seed rule has to
                    # key on the seed, not on position.
                    frame.attrs["record"] = path.stem
                    frames.append(frame)
                # Delta 2, at the boundary: a factor of 57.3 cancels out of `skill` and is
                # visible only in `rmse` and the quiescence thresholds, so the headline
                # metric structurally cannot catch it.
                assert_external_units(frames, MOTION_CHANNELS)
                if conv_name == "sign_flipped":
                    frames = [
                        f.assign(
                            **{
                                c: -f[c]
                                for c in ("roll", "pitch", "heave",
                                          "roll_rate", "pitch_rate", "heave_rate")
                            }
                        )
                        for f in frames
                    ]
                if args.rescale_to_corpus:
                    frames = _rescale_to_corpus(
                        frames, args.corpus_root, float(heading), float(speed)
                    )
                table = evaluate_trajectories(
                    models,
                    frames,
                    stats=stats,
                    spec=spec,
                    input_channels=cfg.data.input_channels,
                    target_dofs=cfg.data.target_dofs,
                    horizons=tuple(cfg.data.horizons),
                    fs_hz=fs,
                    device=args.device,
                    persistence_key=PERSISTENCE_RUN_KEY,
                )
                table.insert(0, "sign_convention", conv_name)
                table.insert(1, "heading_deg", float(heading))
                table.insert(2, "speed_kn", float(speed))
                tables.append(table)
                print(
                    f"[eval] {conv_name:12s} heading {float(heading):5.1f} "
                    f"{float(speed):4.1f} kn: {len(frames)} records scored",
                    file=sys.stderr,
                )

    out = pd.concat(tables, ignore_index=True)
    out.insert(0, "grid_kind", args.grid_kind)
    out.insert(0, "regime", args.regime)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_rescaled" if args.rescale_to_corpus else ""
    path = args.out_dir / f"skill_mss_{args.grid_kind}{suffix}.csv"
    out.to_csv(path, index=False)
    print(f"[eval] wrote {path} ({len(out)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
