#!/usr/bin/env python3
"""CLI wrapper for the Phase 9 figures. Logic lives in :mod:`dmf.viz`.

``make figures`` is this script with no arguments. It produces the two figures
``docs/IMPLEMENTATION_PLAN.md`` Phase 9 asks for and that nothing in the project had a
generator for:

1. **`results/headline_forecast_intervals.png`** -- plan item 2: forecast versus truth with
   90 % intervals, roll at SS5 beam seas, 3 s lead. Two panels, ``dlinear_quantile`` and
   ``tcn_quantile``, because those two are the project's calibration-versus-sharpness
   trade-off and drawing either alone would be choosing the flattering half.
2. **`results/quiescence_lead_time_hist.png`** -- plan item 4's histogram, drawn for one
   named cell with its base rate on the figure. Never pooled across sea states: P6-D19
   retracted a whole table for exactly that.

**Two stages, and the split is the point.** Extraction needs the corpus, the committed
checkpoints and a forward pass; rendering needs a 40 kB ``.npz`` and a committed CSV. With
``--render-only`` the second runs alone, which is what makes "is this figure a function of
committed artifacts?" answerable without a GPU -- the same property ``make report`` gives
the Pareto figure. The trace is written to ``results/`` and committed.

Nothing here computes a metric. The coverage printed on the headline figure is the hit rate
of the drawn span, computed at draw time; the PICP of the test partition is a different
quantity and lives in ``results/e03/probabilistic.csv``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from dmf.config import load_experiment  # noqa: E402
from dmf.viz.forecast_plots import (  # noqa: E402
    IntervalPanel,
    plot_interval_fan_grid,
    plot_lead_time_histogram,
)
from dmf.viz.traces import (  # noqa: E402
    HEADLINE_ALPHA,
    HEADLINE_MODELS,
    extract_trace,
    load_traces,
    save_traces,
)

#: The experiment carrying the interval heads.
CONFIG: Path = Path("configs/experiment/e03_probabilistic.yaml")

#: Committed trace the headline figure is rendered from.
TRACE_NAME: str = "headline_trace.npz"

#: Figure filenames, under the results root.
HEADLINE_NAME: str = "headline_forecast_intervals.png"
LEAD_TIME_NAME: str = "quiescence_lead_time_hist.png"

#: The cell the lead-time histogram is drawn at, and the model drawn there.
#:
#: ``lstm_gaussian`` is the best-scoring model at this cell on the interval rule, and the
#: cell is scorable -- 90 deg / 12 kn at SS5 under `strict` limits, base rate 0.083. A cell
#: is chosen and named rather than pooled: scorability is a property of the cell, not of the
#: sea state (P6-D19), and 10 of 48 `permissive` cells have nothing in them to detect.
LEAD_TIME_SELECTOR: dict[str, object] = {
    "regime": "id",
    "model": "lstm_gaussian",
    "ss": "SS5",
    "heading_deg": 90.0,
    "speed_kn": 12.0,
    "threshold_set": "strict",
    "rule": "interval",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--regime", default="id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render from the committed trace and CSVs; no corpus, no checkpoints, no GPU",
    )
    return parser


def render_headline(results_dir: Path) -> Path:
    """Render the headline forecast-with-intervals figure from the committed trace.

    Args:
        results_dir: Results root holding the trace, and where the figure is written.

    Returns:
        Path written.
    """
    traces = load_traces(results_dir / TRACE_NAME)
    # Caption from the artifact, not from the module constants. A trace re-extracted at a
    # different cell must retitle the figure rather than inherit a caption that is no longer
    # true of the data beside it.
    cells = {trace.cell for trace in traces}
    if len(cells) != 1:
        raise ValueError(f"traces span more than one grid cell: {sorted(cells)}")
    ss, heading_deg, speed_kn, vessel = traces[0].cell.split("|")
    panels = [
        IntervalPanel(
            title=f"{trace.model} (trained on {trace.regime}, seed {trace.seed})",
            t_s=trace.t_s,
            truth=trace.truth,
            median=trace.median,
            lower=trace.lower,
            upper=trace.upper,
            persistence=trace.persistence,
        )
        for trace in traces
    ]
    figure = plot_interval_fan_grid(
        panels,
        traces[0].dof,
        "deg",
        1.0 - HEADLINE_ALPHA,
        traces[0].horizon_s,
        f"{ss}, beam seas ({float(heading_deg):g} deg), {float(speed_kn):g} kn, {vessel}",
    )
    out = results_dir / HEADLINE_NAME
    figure.savefig(out, dpi=140)
    return out


def render_lead_times(results_dir: Path) -> Path:
    """Render the quiescent-window lead-time histogram for one named cell.

    Args:
        results_dir: Results root holding ``e04/``, and where the figure is written.

    Returns:
        Path written.

    Raises:
        ValueError: If the selector matches no row, which would otherwise render an empty
            histogram that looks like a model that never fired.
    """
    leads = pd.read_csv(results_dir / "e04" / "quiescence_lead_times.csv.gz")
    mask = pd.Series(True, index=leads.index)
    for column, value in LEAD_TIME_SELECTOR.items():
        mask &= leads[column] == value
    selected = leads.loc[mask]
    if selected.empty:
        raise ValueError(f"no lead-time rows match {LEAD_TIME_SELECTOR}")

    summary = pd.read_csv(results_dir / "e04" / "quiescence.csv")
    keys = dict(LEAD_TIME_SELECTOR)
    row_mask = pd.Series(True, index=summary.index)
    for column, value in keys.items():
        row_mask &= summary[column] == value
    rows = summary.loc[row_mask & (summary["group_level"] == "cell")]
    if rows.empty:
        raise ValueError(f"no quiescence summary row matches {keys}")
    base_rate = float(rows["base_rate"].iloc[0])

    figure = plot_lead_time_histogram(
        selected["lead_s"].to_numpy(dtype=float),
        str(LEAD_TIME_SELECTOR["threshold_set"]),
        base_rate,
    )
    figure.axes[0].set_title(
        f"{LEAD_TIME_SELECTOR['model']} on {LEAD_TIME_SELECTOR['ss']}, beam seas, "
        f"{LEAD_TIME_SELECTOR['speed_kn']:g} kn -- "
        f"{LEAD_TIME_SELECTOR['threshold_set']} limits, {LEAD_TIME_SELECTOR['rule']} rule "
        "-- simulated",
        fontsize=10,
    )
    figure.tight_layout()
    out = results_dir / LEAD_TIME_NAME
    figure.savefig(out, dpi=140)
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, or None for ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    if not args.render_only:
        cfg = load_experiment(args.config)
        traces = extract_trace(
            cfg,
            corpus_root=args.corpus_root,
            checkpoint_root=args.checkpoint_root,
            regime=args.regime,
            model_labels=HEADLINE_MODELS,
            seed=args.seed,
            device=args.device,
        )
        save_traces(args.results_dir / TRACE_NAME, traces)
        print(f"wrote {args.results_dir / TRACE_NAME}")
    for path in (render_headline(args.results_dir), render_lead_times(args.results_dir)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
