#!/usr/bin/env python3
"""CLI wrapper for training. Logic lives in :mod:`dmf.train.loop`."""

import argparse
import sys
from pathlib import Path


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
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
