#!/usr/bin/env python3
"""CLI wrapper for ONNX export, parity, and latency benchmarking.

Logic lives in :mod:`dmf.deploy`. Parity runs before latency; a fast graph that computes
the wrong thing is worthless.
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export, parity-check, and benchmark a model.")
    parser.add_argument("--export", action="store_true", help="Export the checkpoint to ONNX.")
    parser.add_argument("--parity", action="store_true", help="Run the FP32 parity check.")
    parser.add_argument("--latency", action="store_true", help="Run the latency benchmark.")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Trained checkpoint to export."
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
