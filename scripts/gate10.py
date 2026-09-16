#!/usr/bin/env python3
"""CLI wrapper for the Gate 10 read-out. Logic lives in :mod:`dmf.eval.conformal_gate`.

Reads the committed ``results/e05/`` tables, writes ``gate10.csv`` and ``gate10.md``, and
exits non-zero when a predicate fails.

**A large out-of-distribution degradation is not a failure here.** P10-D1 predicts one, and
predicate 7 requires it to be reported while deliberately setting no bar on its size; a gate
that failed on it would reward hiding the finding (``CLAUDE.md`` non-negotiable 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dmf.eval.conformal_gate import PASS, build_gate10_markdown, read_gate10


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/e05"))
    parser.add_argument("--heads", type=Path, default=Path("results/e03/probabilistic.csv"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when a predicate fails. The predicate still records as a failure "
        "in the artifact; this only changes the exit code.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read Gate 10 and write its artifacts.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame = read_gate10(args.results_dir, args.heads)
    except FileNotFoundError as exc:
        print(f"Gate 10: {exc}", file=sys.stderr)
        return 1

    csv_path = out_dir / "gate10.csv"
    md_path = out_dir / "gate10.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(build_gate10_markdown(frame))

    passed = int((frame["verdict"] == PASS).sum())
    for row in frame.itertuples():
        print(f"  criterion {row.criterion}: {row.verdict} -- {row.note}")
    print(f"Gate 10: {'PASS' if passed == len(frame) else 'NOT PASSED'} ({passed}/{len(frame)})")
    print(f"wrote {csv_path} and {md_path}")
    return 0 if passed == len(frame) or args.allow_fail else 1


if __name__ == "__main__":
    sys.exit(main())
