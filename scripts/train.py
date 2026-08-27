#!/usr/bin/env python3
"""CLI wrapper for training. Logic lives in :mod:`dmf.train.experiment`."""

import argparse
import sys
from pathlib import Path

import torch

from dmf.config import load_experiment
from dmf.train.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one experiment configuration.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML.")
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/corpus"), help="Corpus dataset root."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override the config's training seeds. At least three are required.",
    )
    parser.add_argument(
        "--regimes", type=str, nargs="+", default=None, help="Override the config's regimes."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the SGD-fitted models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment(
        load_experiment(args.config),
        args.corpus,
        results_dir=args.results_dir,
        seeds=args.seeds,
        regimes=args.regimes,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
