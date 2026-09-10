#!/usr/bin/env python3
"""CLI wrapper for the Gate 7 read-out. Logic lives in :mod:`dmf.deploy.gate`.

Reads ``results/parity.csv``, ``results/latency.csv``, the Pareto figure and the README's
latency section, writes ``gate7.csv`` and ``gate7.md``, and exits non-zero unless all three
clauses of ``docs/IMPLEMENTATION_PLAN.md`` §Phase 7 pass.

This script does **not** run the benchmark. A gate read-out must not perturb the numbers it
is reading, and re-running the sweep would take hours and change them -- the same reasoning
that keeps ``scripts/gate6.py`` from running ``make eval``.
"""

import argparse
import sys
from pathlib import Path

from dmf.deploy.gate import build_gate7_markdown, gate7_passes, write_gate7_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Gate 7 read-out from committed artifacts: parity passes, "
            "latency.csv and the Pareto figure exist, and the README states measured "
            "numbers with methodology and no generic speedup multipliers."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory holding parity.csv, latency.csv and latency_pareto.png.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write gate7.csv and gate7.md. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--readme", type=Path, default=Path("README.md"), help="README to check clause 3 against."
    )
    parser.add_argument(
        "--print-report", action="store_true", help="Print the rendered read-out to stdout."
    )
    parser.add_argument(
        "--allow-fail", action="store_true", help="Exit 0 even when the gate does not pass."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readout, csv_path, markdown_path = write_gate7_report(
        args.results_dir, args.out_dir, readme=args.readme
    )
    if args.print_report:
        print(build_gate7_markdown(readout))
    print(f"wrote {csv_path} and {markdown_path}", file=sys.stderr)
    for row in readout.itertuples():
        print(f"  criterion {row.criterion}: {row.verdict} -- {row.detail}", file=sys.stderr)
    passed = gate7_passes(readout)
    print(f"Gate 7: {'PASS' if passed else 'NOT PASSED'}", file=sys.stderr)
    return 0 if passed or args.allow_fail else 1


if __name__ == "__main__":
    sys.exit(main())
