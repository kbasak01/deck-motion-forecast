#!/usr/bin/env python3
"""Compare MSS S175 trajectories against the corpus S175 statistically.

Phase 8. Per-DOF RMS, significant single amplitude and zero-crossing period, plus a Welch
response-spectrum overlay, for each (heading, speed) cell at SS5.

Every MSS number is reported against the corpus's own across-seed spread, per carry-forward
delta 6: "the spectra overlay closely" is unfalsifiable unless the reader is told how far
apart two realizations of the corpus generator routinely land. The z column is that
distance, in corpus standard deviations.

Both sides are simulations. This compares a reduced-order generator against a strip-theory
one; neither is a measurement of a real ship.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dmf.mss.compare import (
    COMPARE_DOFS,
    PSD_BAND_RAD_S,
    compare_against_corpus_spread,
    marginalize_over_speed,
    response_psd,
    summarize_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mss/s175_ss5.yaml"))
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mss"))
    parser.add_argument(
        "--grid-kind",
        default="mss",
        choices=("mss", "corpus"),
        help="Which wave-grid convention to compare. 'mss' is the independent path; "
        "'corpus' shares the corpus wave field and isolates the hull response.",
    )
    return parser


def _corpus_frames(root: Path, heading: float, speed: float) -> list[pd.DataFrame]:
    pattern = f"SS5_h{heading:05.1f}_u{speed:04.1f}_s*.parquet"
    return [pd.read_parquet(p) for p in sorted((root / "s175").glob(pattern))]


def _mss_frames(root: Path, grid: str, heading: float, speed: float) -> list[pd.DataFrame]:
    pattern = f"S175_h{heading:05.1f}_u{speed:04.1f}_s*.csv"
    return [pd.read_csv(p) for p in sorted((root / grid).glob(pattern))]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    fs = float(cfg["record"]["fs_hz"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stat_rows: list[pd.DataFrame] = []
    psd_rows: list[dict[str, object]] = []

    for heading in cfg["headings_deg"]:
        for speed in cfg["speeds_kn"]:
            corpus = _corpus_frames(args.corpus_root, float(heading), float(speed))
            mss = _mss_frames(args.mss_dir, args.grid_kind, float(heading), float(speed))
            if not corpus or not mss:
                print(
                    f"[compare] skipping {heading}/{speed}: "
                    f"{len(corpus)} corpus, {len(mss)} mss records",
                    file=sys.stderr,
                )
                continue

            c_sum = summarize_records(corpus, fs, "corpus")
            m_sum = summarize_records(mss, fs, "mss")
            for metric in ("rms", "sig_amp", "tz_s"):
                table = compare_against_corpus_spread(c_sum, m_sum, metric)
                table.insert(0, "speed_kn", float(speed))
                table.insert(0, "heading_deg", float(heading))
                stat_rows.append(table)

            # Band-averaged PSD agreement, averaged over realizations on each side.
            for dof in COMPARE_DOFS:
                w_ref = None
                spectra: dict[str, FloatArrayLike] = {}
                for label, frames in (("corpus", corpus), ("mss", mss)):
                    acc = []
                    for frame in frames:
                        w, p = response_psd(frame[dof].to_numpy(), fs)
                        acc.append(p)
                        w_ref = w
                    spectra[label] = np.mean(np.vstack(acc), axis=0)
                assert w_ref is not None
                band = (w_ref >= PSD_BAND_RAD_S[0]) & (w_ref <= PSD_BAND_RAD_S[1])
                c_band = float(np.trapezoid(spectra["corpus"][band], w_ref[band]))
                m_band = float(np.trapezoid(spectra["mss"][band], w_ref[band]))
                psd_rows.append(
                    {
                        "heading_deg": float(heading),
                        "speed_kn": float(speed),
                        "dof": dof,
                        "band_lo_rad_s": PSD_BAND_RAD_S[0],
                        "band_hi_rad_s": PSD_BAND_RAD_S[1],
                        "corpus_band_power": c_band,
                        "mss_band_power": m_band,
                        "ratio_mss_over_corpus": m_band / c_band if c_band else float("nan"),
                        "corpus_peak_w": float(w_ref[np.argmax(spectra["corpus"])]),
                        "mss_peak_w": float(w_ref[np.argmax(spectra["mss"])]),
                    }
                )

    stats = pd.concat(stat_rows, ignore_index=True)
    psd = pd.DataFrame(psd_rows)
    marginal = marginalize_over_speed(stats)
    marginal_path = args.out_dir / f"summary_marginal_{args.grid_kind}.csv"
    marginal.to_csv(marginal_path, index=False)
    stats_path = args.out_dir / f"summary_stats_{args.grid_kind}.csv"
    psd_path = args.out_dir / f"psd_band_{args.grid_kind}.csv"
    stats.to_csv(stats_path, index=False)
    psd.to_csv(psd_path, index=False)

    print(f"[compare] grid convention: {args.grid_kind}", file=sys.stderr)
    print(f"[compare] wrote {stats_path} ({len(stats)} rows)", file=sys.stderr)
    print(f"[compare] wrote {psd_path} ({len(psd)} rows)", file=sys.stderr)
    print(f"[compare] wrote {marginal_path} ({len(marginal)} rows)", file=sys.stderr)
    big = marginal[(marginal["metric"] == "rms") & (marginal["ratio_spread_over_speed"] > 0.5)]
    for _, row in big.iterrows():
        print(
            f"[compare] LARGE SPEED DEPENDENCE: {row['dof']} at {row['heading_deg']:.0f} deg, "
            f"per-speed ratio {row['ratio_min_over_speed']:.3f}..{row['ratio_max_over_speed']:.3f} "
            f"-- the marginal ratio {row['ratio_mss_over_corpus']:.3f} hides this",
            file=sys.stderr,
        )

    rms = stats[stats["metric"] == "rms"]
    print("\n[compare] RMS, MSS vs corpus, in corpus across-seed sd:", file=sys.stderr)
    for dof in COMPARE_DOFS:
        sub = rms[rms["dof"] == dof]
        print(
            f"  {dof:6s} corpus {sub['corpus_mean'].mean():8.4f} "
            f"+/- {sub['corpus_sd'].mean():6.4f}   "
            f"mss {sub['mss_mean'].mean():8.4f}   "
            f"ratio {sub['ratio_mss_over_corpus'].mean():6.3f}   "
            f"z {sub['z_in_corpus_sd'].abs().mean():8.2f}",
            file=sys.stderr,
        )
    return 0


FloatArrayLike = np.ndarray

if __name__ == "__main__":
    sys.exit(main())
