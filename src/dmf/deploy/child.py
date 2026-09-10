"""Child entry point for one benchmark configuration.

Invoked as ``python -m dmf.deploy.child --job JOB.json --out RESULT.json`` by
:func:`dmf.deploy.harness.run_job_subprocess`. It exists as a module rather than as a
string passed to ``python -c`` so that a failure inside a benchmark child produces a
traceback naming a file and a line, and so that it can be run by hand to reproduce one row.

It measures exactly one configuration and writes exactly one result. All of the reasons
for the process boundary are in :mod:`dmf.deploy.harness`.
"""

import argparse
import json
import sys
from pathlib import Path

from dmf.deploy.bench import write_benchmark_json
from dmf.deploy.harness import job_from_payload, run_job


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser. Both paths are required: this module is not a user-facing CLI and has
        no useful defaults.
    """
    parser = argparse.ArgumentParser(description="Measure one benchmark configuration.")
    parser.add_argument("--job", type=Path, required=True, help="Serialised BenchJob.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the result.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Measure one configuration and write its JSON fragment.

    Args:
        argv: Command-line arguments, or None to read ``sys.argv``.

    Returns:
        Process exit status: 0 on success. Any refusal from the harness -- an unrealized
        execution provider, a missing checkpoint -- propagates as an exception and a
        non-zero status, which the parent reports rather than silently omitting the row.
    """
    args = build_parser().parse_args(argv)
    job = job_from_payload(json.loads(args.job.read_text(encoding="utf-8")))
    write_benchmark_json((run_job(job),), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
