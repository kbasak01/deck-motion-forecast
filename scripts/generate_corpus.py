#!/usr/bin/env python3
"""CLI wrapper for corpus generation. Logic lives in :mod:`dmf.sim.generate`."""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the simulated deck-motion corpus.")
    parser.add_argument("--config", type=Path, required=True, help="Simulation config YAML.")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/corpus"), help="Output dataset root."
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker processes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
