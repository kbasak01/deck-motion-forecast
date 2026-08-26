#!/usr/bin/env python3
"""CLI wrapper for corpus generation. Logic lives in :mod:`dmf.sim.generate`."""

import argparse
import sys
import time
from pathlib import Path

from dmf.config import load_sim
from dmf.sim.generate import MANIFEST_NAME, generate_corpus, realization_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the simulated deck-motion corpus.")
    parser.add_argument("--config", type=Path, required=True, help="Simulation config YAML.")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/corpus"), help="Output dataset root."
    )
    parser.add_argument("--workers", type=int, default=8, help="Worker processes.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid size and exit without simulating anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_sim(args.config)
    specs = realization_grid(cfg)
    rows = len(specs) * int(round(cfg.duration_s * cfg.fs_hz))
    print(
        f"[corpus] {len(specs)} realizations x "
        f"{int(round(cfg.duration_s * cfg.fs_hz))} rows = {rows / 1e6:.2f} M rows "
        f"-> {args.out}",
        flush=True,
    )
    if args.dry_run:
        return 0

    started = time.perf_counter()
    root = generate_corpus(cfg, args.out, args.workers)
    elapsed = time.perf_counter() - started
    on_disk = sum(p.stat().st_size for p in root.rglob("*.parquet"))
    print(
        f"[corpus] done in {elapsed:.1f} s "
        f"({on_disk / 1e6:.1f} MB on disk, manifest at {root / MANIFEST_NAME})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
