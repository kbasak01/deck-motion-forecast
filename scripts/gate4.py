#!/usr/bin/env python3
"""CLI wrapper for the Gate 4 read-out. Logic lives in :mod:`dmf.eval.gate`.

Reads the committed ``baselines.csv`` (and ``paired_contrasts.csv`` when it is there),
writes ``gate4.csv`` and ``gate4.md``, and exits non-zero when the reading the gate is
read at does not pass.
"""

import argparse
import sys
from pathlib import Path

from dmf.eval.gate import build_gate4_markdown, reading_passes, write_gate4_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Gate 4 read-out from committed artifacts. Both readings are "
            "computed and reported: A is the original criterion (3 s vs "
            "damped_persistence, docs/IMPLEMENTATION_PLAN.md Phase 4) and B is the "
            "restated one the gate is read at (10 s on pitch vs the stronger of "
            "damped_persistence and window_mean, docs/protocol.md P4-D1)."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory holding baselines.csv and optionally paired_contrasts.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write gate4.csv and gate4.md. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--gate-reading",
        type=str,
        default="B",
        help="Which reading the exit status is taken from. Both are always reported.",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when the gate reading does not pass (still reports it).",
    )
    parser.add_argument(
        "--print",
        dest="print_report",
        action="store_true",
        help="Also print the rendered document to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readout, csv_path, markdown_path = write_gate4_report(args.results_dir, args.out_dir)
    if args.print_report:
        print(build_gate4_markdown(readout, results_dir=args.results_dir))
    print(f"wrote {csv_path} and {markdown_path}", file=sys.stderr)
    passed = reading_passes(readout, args.gate_reading)
    print(
        f"Gate 4, reading {args.gate_reading}: {'PASS' if passed else 'NOT PASSED'}",
        file=sys.stderr,
    )
    return 0 if passed or args.allow_fail else 1


if __name__ == "__main__":
    sys.exit(main())
