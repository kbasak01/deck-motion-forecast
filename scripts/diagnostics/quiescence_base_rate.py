#!/usr/bin/env python3
"""Corpus base rate and onset counts for the quiescence metric (docs/protocol.md P6-D7).

These numbers decide the REPORTING DESIGN -- which cells are scorable, and that F1 is
grouped by sea state rather than pooled -- so they must be reproducible from a committed
artifact rather than quoted from a transcript. The Phase 6 audit (finding N1) recorded that
P6-D7's tables had no producer; this is it.

Reads the corpus only. No model, no GPU, no split: the base rate is a property of the true
trajectories, so every realization of the primary vessel is eligible and the train/test
boundary is irrelevant here.

    python scripts/diagnostics/quiescence_base_rate.py --out results/e04/quiescence_base_rate.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dmf.eval.quiescence import PERMISSIVE, STRICT, base_rate, detect_quiescent_mask, scorable_onsets

THRESHOLDS = (PERMISSIVE, STRICT)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--per-cell",
        type=int,
        default=8,
        help="Realizations per (sea state, heading) cell. P6-D7 used 4; more is strictly better "
        "and the cost is one Parquet read each.",
    )
    parser.add_argument("--vessel", type=str, default="frigate")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the base-rate and onset table."""
    args = build_parser().parse_args(argv)
    meta = pd.read_parquet(args.corpus / "manifest.parquet")
    meta = meta[meta["vessel"] == args.vessel]
    fs_hz = float(meta["fs_hz"].iloc[0])

    rows: list[dict[str, object]] = []
    # Grouped by SPEED as well as heading. Grouping only by (ss, heading) hides a real
    # effect: at SS3/45 deg the deck never leaves permissive limits at 0 and 6 kn (base rate
    # 1.000, zero scorable onsets) but does at 12 kn (0.800, 53 onsets), because forward speed
    # shifts the encounter frequency (P1-D1). A cell is scorable or not per (ss, heading,
    # speed), and P6-D7's original table -- measured on a speed-0 subsample -- said otherwise.
    for (ss, heading, speed), group in meta.groupby(["ss", "heading", "speed"], sort=True):
        chosen = group.sort_values("seed").head(args.per_cell)
        for limits in THRESHOLDS:
            rates, raw, scorable, samples = [], 0, 0, 0
            for path in chosen["path"]:
                frame = pd.read_parquet(
                    args.corpus / path, columns=["roll", "pitch", "heave_rate"]
                )
                mask = detect_quiescent_mask(
                    frame["roll"].to_numpy(np.float64),
                    frame["pitch"].to_numpy(np.float64),
                    frame["heave_rate"].to_numpy(np.float64),
                    limits,
                    fs_hz,
                )
                rates.append(base_rate(mask))
                onsets, _ = scorable_onsets(mask, first_decision_idx=0)
                # `scorable_onsets` drops a run already open at the first decision index; the
                # raw count is kept beside it because their difference IS the P6-D7 finding
                # at SS3/permissive (one raw onset, zero scorable).
                raw += int(np.count_nonzero(np.diff(np.concatenate(([0], mask.view(np.int8)))) == 1))
                scorable += int(onsets.size)
                samples += int(mask.size)
            rows.append(
                {
                    "vessel": args.vessel,
                    "ss": ss,
                    "heading_deg": float(heading),
                    "speed_kn": float(speed),
                    "threshold_set": limits.name,
                    "n_realizations": len(chosen),
                    "base_rate": float(np.mean(rates)),
                    "raw_onsets_per_realization": raw / len(chosen),
                    "scorable_onsets_per_realization": scorable / len(chosen),
                    "duration_s": samples / fs_hz / len(chosen),
                    "scorable": scorable > 0,
                }
            )

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(table)} rows)")
    for name in (limits.name for limits in THRESHOLDS):
        sub = table[table["threshold_set"] == name]
        print(f"\n=== {name}: base rate by (ss, speed) ===")
        print(
            sub.pivot_table(index="ss", columns="speed_kn", values="base_rate")
            .round(3)
            .to_string()
        )
        print(f"=== {name}: scorable onsets/realization by (ss, speed) ===")
        print(
            sub.pivot_table(
                index="ss", columns="speed_kn", values="scorable_onsets_per_realization"
            )
            .round(2)
            .to_string()
        )
        unscorable = sub[~sub["scorable"]]
        print(
            f"=== {name}: unscorable cells: {len(unscorable)} of {len(sub)} "
            f"(ss, heading, speed) ==="
        )
        if len(unscorable):
            print(unscorable[["ss", "heading_deg", "speed_kn", "base_rate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
