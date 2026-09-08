#!/usr/bin/env python3
"""CLI wrapper for the matched-origin re-scoring. Logic lives in :mod:`dmf.eval.matched`.

Run this **after** the Phase 6 sweep and **before** ``make eval``:

    python scripts/rescore_matched.py --all-arms

It re-scores the three lookback arms on the forecast origins they share -- the L=400 arm's
1091 origins, 399..5849 -- and writes one table per arm under
``results/e04/matched_origins/<arm>/baselines_by_seed.csv``, which is where
:func:`dmf.eval.assemble.read_arm_frames` reads them from. Until those files exist,
``ablations.csv`` carries no ``lookback`` rows and Gate 6 predicate 5 cannot pass, because
the arms as committed were each scored on their own full window set (1151 / 1131 / 1091 per
realization) and differencing them would compare forecasts of different absolute times
(``docs/protocol.md`` P6-D4 item 1, P6-D15 finding 1).

**Nothing is retrained.** SGD rows load their committed ``state_dict`` -- a missing
checkpoint is a hard error, never a silent refit -- and closed-form rows are re-solved from
the same full training split the sweep used. Only the *test* window population changes.

**What it costs.** One forward pass per (arm, experiment, regime) over a test partition
thinned to about 95 % of its windows, so roughly the cost of the corresponding scoring pass.
The reference arm ``lookback_20s`` is scored from ``artifacts/checkpoints/deep/``, i.e.
``results/e02/``'s checkpoints; its *committed rows* are not reused, because they were scored
on 1131 origins and are not the matched quantity.

Nothing is written outside ``--out-root``. ``results/``, ``results/imu/``, ``results/e02/``,
``results/e03/`` and the per-arm ``results/e04/*`` directories are read and never touched.
"""

import argparse
import sys
from pathlib import Path

import torch

from dmf.eval.ablations import ABLATIONS
from dmf.eval.matched import LOOKBACK_ABLATION, default_out_root, score_matched_lookback


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser. ``--arm`` and ``--all-arms`` are alternative ways of saying which arms to
        score; giving neither is an error rather than a default, because a partial matched
        set is a real choice and should be made explicitly.
    """
    parser = argparse.ArgumentParser(
        description="Re-score the lookback arms on their shared forecast origins."
    )
    parser.add_argument(
        "--all-arms",
        action="store_true",
        help=f"Score every arm of the lookback ablation: {list(ABLATIONS[LOOKBACK_ABLATION])}.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Score only these arms. The origin set is still the intersection over ALL arms, "
            "so a partial run's rows still pair with the rest."
        ),
    )
    parser.add_argument(
        "--regime",
        type=str,
        nargs="+",
        default=None,
        help="Regimes to score; default is each experiment config's own.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/corpus"), help="Corpus dataset root."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results root; the tables go under <root>/e04/matched_origins/<arm>/.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override the output directory entirely. Default: <results-dir>/e04/matched_origins.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Root of the committed checkpoints. Read, never written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device the forward passes run on.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Windows per scoring batch. Speed only."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Re-score the requested lookback arms and write their matched tables.

    Args:
        argv: Command-line arguments, or None to read ``sys.argv``.

    Returns:
        Process exit code.

    Raises:
        ValueError: If neither ``--arm`` nor ``--all-arms`` was given, or if both were.
    """
    args = build_parser().parse_args(argv)
    if args.all_arms and args.arm is not None:
        raise ValueError("pass --all-arms or --arm, not both")
    if not args.all_arms and args.arm is None:
        raise ValueError(
            "pass --all-arms to score every lookback arm, or --arm to name a subset. There is "
            "no default: a matched set missing an arm renders as an ablation with a missing "
            "row, and choosing that should be explicit"
        )
    out_root = args.out_root or default_out_root(args.results_dir)
    artifacts = score_matched_lookback(
        args.corpus,
        out_root=out_root,
        checkpoint_root=args.checkpoints,
        arms=None if args.all_arms else list(args.arm),
        regimes=None if args.regime is None else list(args.regime),
        device=args.device,
        batch_size=args.batch_size,
    )
    counts = ", ".join(
        f"L={lookback}: {before} -> {after}"
        for lookback, (before, after) in sorted(artifacts.window_counts.items())
    )
    print(
        f"matched on {artifacts.origins.size} origins "
        f"({int(artifacts.origins[0])}..{int(artifacts.origins[-1])}) of a "
        f"{artifacts.n_samples}-sample realization; windows per realization {counts}",
        flush=True,
    )
    for arm, path in sorted(artifacts.paths.items()):
        print(f"rescored {arm} -> {path}", flush=True)
    # Printed, not swallowed: an arm that could not be re-scored leaves the lookback ablation
    # incomplete, and the reader of the log should see which and why at the moment it happens
    # rather than infer it from a table that is smaller than expected.
    for key, reason in sorted(artifacts.skipped.items()):
        print(f"not rescored {key}: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
