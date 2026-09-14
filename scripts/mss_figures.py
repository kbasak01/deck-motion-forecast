#!/usr/bin/env python3
"""Render the Phase 8 figures from the committed CSVs.

No analysis happens here. Every number plotted was computed by `dmf.eval` or
`dmf.mss.compare` and written to a CSV first, per the rule in `dmf/viz/__init__.py`, so each
figure is traceable to a committed table.

Both series in every figure are simulations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dmf.mss.compare import COMPARE_DOFS, response_psd
from dmf.viz.mss_plots import plot_response_spectra_overlay, plot_skill_vs_horizon

PLOT_MODELS: tuple[str, ...] = ("dlinear_ols", "tcn", "lstm", "transformer")
MODEL_COLORS: dict[str, str] = {
    "dlinear_ols": "#14213d",
    "tcn": "#c1442a",
    "lstm": "#2a7f62",
    "transformer": "#8a6d3b",
}
UNITS: dict[str, str] = {"roll": "deg", "pitch": "deg", "heave": "m"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/mss"))
    parser.add_argument("--mss-dir", type=Path, default=Path("artifacts/mss"))
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--grid-kind", default="mss")
    parser.add_argument("--heading", type=float, default=135.0,
                        help="Heading for the spectra overlay. 135 deg by default because "
                             "head-seas roll is identically zero on the MSS side.")
    parser.add_argument("--speed", type=float, default=12.0)
    return parser


#: Headings each DOF is summarised over. Head-seas roll is excluded in advance (P8-D1):
#: MSS gives exactly 0.0000 deg there by port/starboard symmetry and the corpus gives the
#: P1-D2 residual floor, so a skill score against that target is undefined and its
#: persistence denominator is ~0. Including it puts a 1e12 artifact on the roll axis.
DOF_HEADINGS: dict[str, tuple[float, ...]] = {
    "roll": (135.0,),
    "pitch": (180.0, 135.0),
    "heave": (180.0, 135.0),
}


def _skill_series(results_dir: Path, grid_kind: str) -> tuple[np.ndarray, dict]:
    mss = pd.read_csv(results_dir / f"skill_mss_{grid_kind}.csv")
    mss = mss[mss["sign_convention"] == "nominal"].copy()
    mss["label"] = mss["model"].astype(str).str.split("|").str[0]
    mss["seed"] = mss["model"].astype(str).str.split("|").str[-1]

    corpus = pd.read_csv("results/e04/metrics_by_cell.csv")
    corpus = corpus[
        (corpus["regime"] == "unseen_vessel")
        & (corpus["vessel"] == "s175")
        & (corpus["ss"] == "SS5")
        & (corpus["heading_deg"].isin([180.0, 135.0]))
    ].drop_duplicates(
        subset=["model", "seed", "heading_deg", "speed_kn", "dof", "horizon_samples"]
    )

    horizons = np.array(sorted(mss["horizon_s"].unique()), dtype=float)
    series: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}
    for dof in COMPARE_DOFS:
        panel: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        headings = DOF_HEADINGS[dof]
        for model in PLOT_MODELS:
            entry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for source, table, label_col in (
                ("mss", mss[mss["label"] == model], "seed"),
                ("corpus", corpus[corpus["model"] == model], "seed"),
            ):
                sub = table[(table["dof"] == dof) & (table["heading_deg"].isin(headings))]
                if sub.empty:
                    continue
                means, sds = [], []
                for h in horizons:
                    cell = sub[np.isclose(sub["horizon_s"], h)]
                    per_seed = cell.groupby(label_col)["skill"].mean()
                    means.append(float(per_seed.mean()) if len(per_seed) else np.nan)
                    sds.append(float(per_seed.std()) if len(per_seed) > 1 else 0.0)
                entry[source] = (np.array(means), np.nan_to_num(np.array(sds)))
            if entry:
                panel[model] = entry
        series[dof] = panel
    return horizons, series


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. The figure the phase exists for.
    horizons, series = _skill_series(out, args.grid_kind)
    fig = plot_skill_vs_horizon(
        horizons,
        series,
        COMPARE_DOFS,
        "Forecast skill transfers to an independent hydrodynamic model only at short lead\n"
        "S175 at SS5 — solid: corpus `unseen_vessel` (committed) · dashed: MSS strip theory\n"
        "roll shown at 135 deg only: head-seas roll is zero by symmetry in MSS "
        "and the P1-D2 floor in the corpus",
        model_colors=MODEL_COLORS,
    )
    p1 = out / "mss_skill_vs_horizon.png"
    fig.savefig(p1)
    print(f"[fig] wrote {p1}", file=sys.stderr)

    # 2. Response spectra overlay.
    cp = sorted((args.corpus_root / "s175").glob(
        f"SS5_h{args.heading:05.1f}_u{args.speed:04.1f}_s*.parquet"))
    mp = sorted((args.mss_dir / args.grid_kind).glob(
        f"S175_h{args.heading:05.1f}_u{args.speed:04.1f}_s*.csv"))
    if cp and mp:
        corpus_frames = [pd.read_parquet(p) for p in cp]
        mss_frames = [pd.read_csv(p) for p in mp]
        c_psd, m_psd, band = {}, {}, {}
        w_ref = None
        for dof in COMPARE_DOFS:
            acc_c = []
            for frame in corpus_frames:
                w, psd = response_psd(frame[dof].to_numpy(), 10.0)
                acc_c.append(psd)
                w_ref = w
            stack = np.vstack(acc_c)
            c_psd[dof] = stack.mean(axis=0)
            band[dof] = (stack.min(axis=0), stack.max(axis=0))
            acc_m = [response_psd(f[dof].to_numpy(), 10.0)[1] for f in mss_frames]
            m_psd[dof] = np.vstack(acc_m).mean(axis=0)
        assert w_ref is not None
        fig2 = plot_response_spectra_overlay(
            w_ref, c_psd, m_psd, COMPARE_DOFS, UNITS,
            f"Response spectra, S175 at SS5, {args.heading:.0f} deg, {args.speed:.0f} kn "
            f"({len(corpus_frames)} corpus vs {len(mss_frames)} MSS realizations)",
            corpus_band=band,
        )
        p2 = out / "mss_response_spectra.png"
        fig2.savefig(p2)
        print(f"[fig] wrote {p2}", file=sys.stderr)
    else:
        print(f"[fig] skipping spectra overlay: {len(cp)} corpus, {len(mp)} mss", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
