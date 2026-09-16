#!/usr/bin/env python3
"""CLI wrapper for Phase 10 split-conformal calibration.

Logic lives in :mod:`dmf.eval.conformal_runner`.

Calibrates the committed Phase 5 interval heads on each regime's validation split and
re-scores them on that regime's test split, writing four tables under ``--out-root``. It
retrains nothing and writes into no committed results directory.

Neither ``--regime`` nor ``--all-regimes`` has a default, for the reason
``scripts/rescore_matched.py`` gives: running three regimes when four were meant produces a
table that is silently partial, and the partial table is indistinguishable from a complete
one once it is on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dmf.config import load_experiment
from dmf.eval.conformal import CONFORMAL_MAX_WINDOWS
from dmf.eval.conformal_runner import run_conformal

#: The Phase 5 experiment whose heads, geometry and checkpoints this arm reuses unchanged.
BASE_EXPERIMENT = Path("configs/experiment/e03_probabilistic.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_EXPERIMENT)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Committed checkpoint root. Never written to.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/e05"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--regime", action="append", help="Regime to run; repeatable.")
    group.add_argument(
        "--all-regimes",
        action="store_true",
        help="Run every regime the experiment config lists.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=CONFORMAL_MAX_WINDOWS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the calibration arm.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    cfg = load_experiment(args.config)
    written = run_conformal(
        cfg,
        args.corpus,
        out_root=args.out_root,
        regimes=None if args.all_regimes else args.regime,
        alpha=args.alpha,
        checkpoint_root=args.checkpoints,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_windows=args.max_windows,
    )
    for key, path in sorted(written.items()):
        print(f"[conformal] {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
