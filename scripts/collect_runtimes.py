#!/usr/bin/env python3
"""Derive `results/runtime_stages.csv` from the stage logs under `artifacts/logs/`.

The README quotes a per-stage wall clock for reproducing this project. Those figures came
from timestamps in `artifacts/logs/`, and `artifacts/` is gitignored -- so from a fresh
clone they were unverifiable assertions. This script turns the part of that table that has
a log behind it into a committed artifact, and the README's table names which rows it
covers and which rows it does not.

**It does not cover every stage.** The Phase 4 and Phase 5 sweeps predate the status-file
convention and have no stage log; their figures come from `docs/protocol.md` prose. The
Phase 7 and Phase 8 runs were never logged either. Those rows are marked as such in the
README rather than given a number this file could be mistaken for supporting.

Two log formats are read: the e04 drivers write ISO-8601 stamps, the Phase 3 sweep driver
writes `date` output. In both, the first and last stamp bracket the stage.

Units: `hours`, wall clock. Simulated results only.
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

#: ISO-8601 stamps, as `date -Is` writes them in the e04 drivers.
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")

#: Bare `date` output, as the Phase 3 sweep driver writes it.
DATE_RE = re.compile(r"[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ \d{2}:\d{2}:\d{2} [A-Z]{3,4} \d{4}")

#: `date`'s default format, for strptime.
DATE_FMT = "%a %b %d %H:%M:%S %Z %Y"

#: Output columns.
COLUMNS = ("stage", "log", "start", "end", "hours")


def _span(stamps: list[str], parse) -> float:
    """Return the hours between the first and last stamp.

    Args:
        stamps: Timestamps in file order, at least two.
        parse: Callable turning one stamp into a datetime.

    Returns:
        Wall clock, hours.
    """
    return (parse(stamps[-1]) - parse(stamps[0])).total_seconds() / 3600.0


def collect(logs_root: Path) -> list[dict[str, object]]:
    """Read every stage log that carries a bracketing pair of timestamps.

    Args:
        logs_root: Directory holding the run logs, normally ``artifacts/logs``.

    Returns:
        One row per stage, sorted by stage name. Empty if no logs are present, which is the
        case on a fresh clone -- the committed CSV is then the only record.
    """
    rows: list[dict[str, object]] = []
    for status in sorted((logs_root / "e04").glob("status_*.txt")):
        stamps = ISO_RE.findall(status.read_text())
        if len(stamps) >= 2:
            rows.append(
                {
                    "stage": status.stem,
                    "log": str(status),
                    "start": stamps[0],
                    "end": stamps[-1],
                    "hours": round(_span(stamps, datetime.fromisoformat), 3),
                }
            )
    sweep = logs_root / "sweep_status.txt"
    if sweep.is_file():
        stamps = DATE_RE.findall(sweep.read_text())
        if len(stamps) >= 2:
            rows.append(
                {
                    "stage": "sweep_e01_ideal_and_imu",
                    "log": str(sweep),
                    "start": stamps[0],
                    "end": stamps[-1],
                    "hours": round(_span(stamps, lambda s: datetime.strptime(s, DATE_FMT)), 3),
                }
            )
    return sorted(rows, key=lambda row: str(row["stage"]))


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, or None for ``sys.argv[1:]``.

    Returns:
        Process exit code. Non-zero if no log was found, so that a silent empty rewrite of
        the committed artifact is not mistaken for a successful run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("artifacts/logs"))
    parser.add_argument("--out", type=Path, default=Path("results/runtime_stages.csv"))
    args = parser.parse_args(argv)

    rows = collect(args.logs_root)
    if not rows:
        print(
            f"no stage logs under {args.logs_root}; leaving {args.out} untouched", file=sys.stderr
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    total = sum(float(row["hours"]) for row in rows)  # type: ignore[arg-type]
    print(f"wrote {args.out} ({len(rows)} stages, {total:.1f} h logged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
