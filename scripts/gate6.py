#!/usr/bin/env python3
"""CLI wrapper for the Gate 6 read-out. Logic lives in :mod:`dmf.eval.gate`.

Reads ``results.md`` and the CSVs it cites, writes ``gate6.csv``, ``gate6_tables.csv`` and
``gate6.md``, and exits non-zero unless all seven pre-registered predicates
(``docs/protocol.md`` P6-D1) pass.

**Predicate 1 -- "`make eval` exits 0" -- cannot be established from artifacts.** No file in
``results/`` testifies that a command was run, so this script does not guess: without
``--eval-exit-code`` that predicate is UNVERIFIED, which is not a pass, and the gate does not
pass either. The intended way to satisfy it is ``make gate6-full``, which runs ``make eval``
first; because make stops on a non-zero exit, reaching the gate step at all is the evidence,
and the target passes ``--eval-exit-code 0`` on that basis.

This script does **not** run ``make eval`` itself. That target scores the whole corpus and
would contend with any training sweep on the same machine, and ``fit_time_s`` is a reported
quantity -- a gate read-out must not perturb the numbers it is reading.
"""

import argparse
import sys
from pathlib import Path

from dmf.eval.gate import (
    GATE6_READING,
    build_gate6_markdown,
    gate6_reading_passes,
    gate6_table_audit,
    read_gate6_inputs,
    write_gate6_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Gate 6 read-out from committed artifacts. Gate 6 is a process "
            "criterion rather than a numeric one, registered in docs/protocol.md P6-D1 as "
            "seven artifact predicates before this machinery existed; all seven are "
            "evaluated and reported, and an UNVERIFIED predicate is not a pass."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory holding results.md and the CSVs it cites.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write gate6.csv, gate6_tables.csv and gate6.md. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--eval-exit-code",
        type=int,
        default=None,
        help=(
            "Exit status of `make eval`, if the caller ran it. Omitted leaves predicate 1 "
            "UNVERIFIED, which is not a pass."
        ),
    )
    parser.add_argument(
        "--no-rerender",
        dest="rerender",
        action="store_false",
        help=(
            "Skip re-rendering results.md from its CSVs. Leaves predicate 2 UNVERIFIED: the "
            "regeneration check is what makes 'not hand-edited' a measurement."
        ),
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when the gate does not pass (still reports it).",
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
    readout, csv_path, markdown_path = write_gate6_report(
        args.results_dir,
        args.out_dir,
        eval_exit_code=args.eval_exit_code,
        rerender=args.rerender,
    )
    if args.print_report:
        evidence = read_gate6_inputs(
            args.results_dir, eval_exit_code=args.eval_exit_code, rerender=args.rerender
        )
        print(
            build_gate6_markdown(
                readout, audit=gate6_table_audit(evidence), results_dir=args.results_dir
            )
        )
    print(f"wrote {csv_path} and {markdown_path}", file=sys.stderr)
    passed = gate6_reading_passes(readout, GATE6_READING)
    for row in readout.itertuples():
        print(f"  criterion {row.criterion}: {row.verdict} -- {row.detail}", file=sys.stderr)
    print(f"Gate 6: {'PASS' if passed else 'NOT PASSED'}", file=sys.stderr)
    return 0 if passed or args.allow_fail else 1


if __name__ == "__main__":
    sys.exit(main())
