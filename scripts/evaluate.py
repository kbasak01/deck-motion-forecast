#!/usr/bin/env python3
"""CLI wrapper for evaluation. Logic lives in :mod:`dmf.eval`."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate every table in results/.")
    parser.add_argument("--all-regimes", action="store_true", help="Score all four regimes.")
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Score a single regime: id, unseen_seastate, unseen_heading, unseen_vessel.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output directory."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
