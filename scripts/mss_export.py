#!/usr/bin/env python3
"""Generate the MSS S175 cross-validation records and export them to CSV.

Phase 8. Reads `configs/mss/s175_ss5.yaml`, synthesises 6-DOF motion from the MSS ShipX
strip-theory motion RAOs, converts to the corpus storage schema (degrees, metres) and
writes one CSV per realization plus a manifest.

The manifest carries the realized Hs and Tz per record, which is the evidence Gate 8
predicate 1 is read from. This script checks the match and reports it, but does not enforce
it -- `scripts/gate8.py` re-derives the predicate independently, following the precedent set
by `scripts/gate7.py`.

These are simulated trajectories from another simulator, not measurements of a real ship.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from dmf.mss.export import generate_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mss/s175_ss5.yaml"),
        help="Run configuration. Changing the sea state here turns the comparison into a "
        "vessel-AND-sea-state shift, which predicts a different model ranking.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/mss"),
        help="Where the committed spectrum-match summary is written. The trajectories "
        "themselves live under artifacts/ and are gitignored, so Gate 8 predicate 1 has to "
        "read its evidence from a derived table in results/ or it is unverifiable from a "
        "clean checkout.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/mss"),
        help="Destination root; one subdirectory per grid convention, plus manifest.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    print(f"[mss] config {args.config}", file=sys.stderr)
    n = len(cfg["headings_deg"]) * len(cfg["speeds_kn"]) * len(cfg["seeds"])
    print(f"[mss] {n} realizations x {len(cfg['grids']) and 2} grid conventions", file=sys.stderr)

    manifest_path = generate_all(cfg, args.out_dir)
    manifest = pd.read_csv(manifest_path)

    tol = cfg["match_tolerance"]
    print(f"[mss] wrote {len(manifest)} records -> {manifest_path}", file=sys.stderr)
    # The predicate is read on the MEAN over realizations, which is the estimator of the
    # target. A max-over-realizations test would be measuring per-record sampling scatter
    # (sd ~3.5% on Hs at 299 components over a 600 s record), not spectral match. The
    # spread is printed beside the mean rather than hidden, per delta 6.
    for label, col, rel in (
        ("Hs", "hs_rel_err", float(tol["hs_rel"])),
        ("Tz", "tz_rel_err", float(tol["tz_rel"])),
    ):
        mean = float(manifest[col].mean())
        sd = float(manifest[col].std())
        ok = abs(mean) <= rel
        print(
            f"[mss] {label} rel err mean {mean:+.4f} +/- {sd:.4f} (per-record sd) "
            f"vs tol {rel} -> {'OK' if ok else 'OUT OF TOLERANCE'}",
            file=sys.stderr,
        )
    print(
        "[mss] encounter-frame Tz (diagnostic, Doppler-shifted by speed): "
        + ", ".join(
            f"{s:.0f}kn {g['tz_encounter_s'].mean():.2f}s" for s, g in manifest.groupby("speed_kn")
        ),
        file=sys.stderr,
    )
    # Committed evidence for Gate 8 predicate 1. artifacts/ is gitignored, so the manifest
    # alone cannot testify to the spectrum match on a fresh clone.
    match = (
        manifest.groupby("grid_kind")[
            ["hs_target_m", "hs_realized_m", "hs_rel_err", "tz_target_s", "tz_realized_s",
             "tz_rel_err"]
        ]
        .agg(["mean", "std", "count"])
    )
    match.columns = ["_".join(c) for c in match.columns]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    match_path = args.results_dir / "spectrum_match.csv"
    match.reset_index().to_csv(match_path, index=False)
    print(f"[mss] wrote {match_path} (committed evidence for Gate 8 predicate 1)", file=sys.stderr)

    for kind, sub in manifest.groupby("grid_kind"):
        print(
            f"[mss] {kind:6s}: RMS roll {sub['rms_roll'].mean():.4f} deg, "
            f"pitch {sub['rms_pitch'].mean():.4f} deg, heave {sub['rms_heave'].mean():.4f} m",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
