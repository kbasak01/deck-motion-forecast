#!/usr/bin/env python3
"""CLI wrapper for the Gate 5 read-out. Logic lives in :mod:`dmf.eval.gate`.

Reads the committed ``probabilistic.csv``, writes ``gate5.csv``,
``gate5_degradation.csv`` and ``gate5.md``, and exits non-zero when the reading the gate is
read at does not pass.

The degradation table is written whenever the sweep covered more than the ``id`` regime, and
it never affects the exit status: ``docs/IMPLEMENTATION_PLAN.md`` Phase 5 and
``docs/protocol.md`` P5-D2 both require the out-of-distribution degradation to be
**reported rather than fixed**. A regime losing coverage is the finding this phase exists to
produce, not a gate failure.
"""

import argparse
import sys
from pathlib import Path

from dmf.eval.gate import (
    build_gate5_markdown,
    coverage_degradation,
    gate5_reading_passes,
    read_gate5_inputs,
    write_gate5_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Gate 5 read-out from committed artifacts. Both readings are "
            "computed and reported: A is the gate itself (PICP@90 within [0.85, 0.95] at "
            "pitch / 10 s on `id`, the cell registered in docs/protocol.md P5-D2 before "
            "the sweep ran) and B is every cell of that regime, so the pass count is "
            "visible beside the gate cell."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/e03"),
        help="Directory holding probabilistic.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write gate5.csv, gate5_degradation.csv and gate5.md. "
        "Defaults to --results-dir.",
    )
    parser.add_argument(
        "--gate-reading",
        type=str,
        default="A",
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
    readout, csv_path, markdown_path = write_gate5_report(args.results_dir, args.out_dir)
    if args.print_report:
        table = read_gate5_inputs(args.results_dir)
        print(
            build_gate5_markdown(
                readout,
                degradation=coverage_degradation(table),
                results_dir=args.results_dir,
            )
        )
    print(f"wrote {csv_path} and {markdown_path}", file=sys.stderr)
    passed = gate5_reading_passes(readout, args.gate_reading)
    print(
        f"Gate 5, reading {args.gate_reading}: {'PASS' if passed else 'NOT PASSED'}",
        file=sys.stderr,
    )
    return 0 if passed or args.allow_fail else 1


if __name__ == "__main__":
    sys.exit(main())
