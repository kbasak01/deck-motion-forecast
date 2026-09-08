#!/usr/bin/env python3
"""CLI wrapper for evaluation. Logic lives in :mod:`dmf.eval`.

``make eval`` is this script with ``--all-regimes`` and nothing else, and it does two
things in order:

1a. **Score, without retraining.** :func:`dmf.eval.scoring.score_experiment` builds each
   test partition once and drives all four passes over it through
   :func:`dmf.train.experiment.load_or_fit`, which resolves an SGD model's committed
   ``state_dict`` and refuses to fall back to a random initialisation. Only the closed-form
   rows are re-solved, because a solve is reproducible in a way a 60-epoch run is not. No
   training happens here; a missing checkpoint is a hard error, not a silent refit.
1b. **Run the controls that are not part of a scoring pass.**
   :func:`dmf.eval.control_runner.run_controls` runs the two interval controls and the
   persistence pipeline-sanity control and writes ``interval_controls.csv`` and
   ``pipeline_sanity.csv``. Both were producer-less until now (``docs/protocol.md`` P6-D15):
   the interval controls were implemented, tested and called by nothing, and pipeline sanity
   raised inside the driver while writing no row anywhere. Nothing here is trained -- the
   null and the shuffled subject are closed-form solves and the untrained subject is
   *supposed* to be unfitted -- but it does need the corpus, so ``--no-controls`` skips it
   and the report then names both files as absent rather than pretending they passed.
   The interval controls run only for a config that **publishes an interval**: on
   ``e02_deep``, where every model has a point head, they would re-measure the unconditional
   floor under a second experiment label and produce two rows that look like two
   measurements of one thing. Pipeline sanity runs for every config.
1c. **Assemble the cross-arm tables.** :func:`dmf.eval.assemble.assemble_phase6_tables`
   joins the committed per-arm tables into ``ablations.csv``, ``controls.csv``,
   ``probabilistic_baseline.csv`` and ``reference_reproducibility.csv``. It reads CSVs only:
   no corpus, no checkpoints, no GPU, and it cannot start a fit. An arm it cannot assemble
   is printed with the reason rather than omitted silently.
2. **Render.** :func:`dmf.eval.report.build_results_report` writes ``results/results.md``
   *from* the CSVs step 1 just wrote and the ones already committed beside them, so every
   number in the document traces to a file by construction (docs/protocol.md P6-D1).

**The lookback ablation is not re-scored here.** Its three arms have to be scored on the
origin set they share (P6-D4 item 1), which is a separate forward pass over the corpus per
arm and per regime; ``scripts/rescore_matched.py`` is that pass, and this script only reads
what it wrote. Folding it in would put a multi-arm re-scoring behind a command that has to be
cheap enough to run on every change.

**Where things are written.** ``--results-dir`` names the results *root* (``results/``).
The scored tables go to ``<root>/e04/`` and the document to ``<root>/results.md``, which is
what ``docs/IMPLEMENTATION_PLAN.md`` §Phase 6 says Phase 6 writes. The renderer is given the
root rather than ``<root>/e04``, because every provenance marker names its source relative to
the document and section 6.4 reads the committed Phase 5 heads from ``<root>/e03/``. ``results/``,
``results/imu/``, ``results/e02/`` and ``results/e03/`` are the Gate 3-5 audit trails: this
script reads them and never writes into them.

``--render-only`` skips **all** of step 1 and rebuilds the document from the
committed CSVs alone. It needs no corpus, no checkpoints and no GPU, which is what makes "is
`results.md` a pure function of the CSVs?" a question anyone can answer in a second. Step 1b
is skipped there too even though it is equally cheap: it *writes* four of the CSVs the
document is rendered from, so leaving it in would make that question answer itself.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from dmf.config import ExperimentConfig, load_experiment
from dmf.eval.ablations import arms_for_experiment
from dmf.eval.assemble import NOT_AN_ARM, assemble_phase6_tables
from dmf.eval.control_runner import CONTROL_ARTIFACTS, run_controls
from dmf.eval.report import PHASE6_SUBDIR, build_results_report, write_table
from dmf.eval.scoring import SCORING_ARTIFACTS, ScoringArtifacts, score_experiment

#: Experiment configs scored by default, and why each is there.
#:
#: ``e02_deep`` is the reference arm: it fits every architecture on all four regimes under
#: the train block every Phase 6 arm copies byte-identically, so it is the source of the
#: §6.1 table and of the point rule of §6.2. ``e03_probabilistic`` carries the interval
#: heads, which §6.4 reports and which the §6.2 interval rule reads its quantiles from.
#:
#: The ablation arms are **not** here. Their tables are assembled into
#: ``results/e04/ablations.csv`` by the ablation driver, from the per-arm runs, and this
#: script renders that file rather than re-deriving it; re-scoring six arms on every
#: ``make eval`` would also make the command far too expensive to run on every change.
DEFAULT_CONFIGS: tuple[Path, ...] = (
    Path("configs/experiment/e02_deep.yaml"),
    Path("configs/experiment/e03_probabilistic.yaml"),
)

#: Name of the rendered document, under the results root.
REPORT_NAME: str = "results.md"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser. ``--all-regimes`` and ``--regime`` are alternative ways of saying which
        regimes to score; giving both is an error rather than a precedence rule, because a
        precedence rule is something a caller has to remember.
    """
    parser = argparse.ArgumentParser(description="Regenerate every table in results/.")
    parser.add_argument("--all-regimes", action="store_true", help="Score all four regimes.")
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Score a single regime: id, unseen_seastate, unseen_heading, unseen_vessel.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Experiment configs to score from their committed checkpoints.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("artifacts/corpus"), help="Corpus dataset root."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Root of the committed checkpoints. Never written to.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device the scoring forward passes run on.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Windows per scoring batch. Speed only."
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help=(
            "Skip the interval and pipeline-sanity controls. The report then names both "
            "files as absent, which reads as a gap rather than as a pass."
        ),
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Reuse the committed scoring CSVs and redo only controls, assembly and render. "
        "Step 1a costs hours and its outputs are deterministic given the checkpoints, so a "
        "controls-only change (a tolerance, a narrowing) must not force a re-score. Mutually "
        "exclusive with --render-only, which skips controls as well.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip scoring and rebuild results.md from the committed CSVs alone.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Render an explicitly incomplete report instead of failing when a required CSV "
            "is absent. The document names every section it could not build."
        ),
    )
    return parser


def resolve_regimes(
    cfg: ExperimentConfig, *, all_regimes: bool, regime: str | None
) -> tuple[str, ...]:
    """Decide which regimes of one config to score.

    Args:
        cfg: The experiment config.
        all_regimes: Whether ``--all-regimes`` was passed.
        regime: The single regime named by ``--regime``, if any.

    Returns:
        The regimes to score, in the config's own order.

    Raises:
        ValueError: If both selectors were given, or if the named regime is not one this
            config was fitted on -- scoring it would need checkpoints that do not exist, and
            failing here says so more clearly than a FileNotFoundError three layers down.
    """
    if all_regimes and regime is not None:
        raise ValueError("pass --all-regimes or --regime, not both")
    if regime is None:
        return tuple(cfg.regimes)
    if regime not in cfg.regimes:
        raise ValueError(
            f"experiment {cfg.name!r} was not fitted on regime {regime!r}; it carries "
            f"{list(cfg.regimes)}"
        )
    return (regime,)


def _arm_of(config: Path) -> str:
    """Return the ablation arm a config feeds, or the not-an-arm sentinel.

    Args:
        config: Path to an experiment config.

    Returns:
        The first arm :func:`dmf.eval.ablations.arms_for_experiment` maps this config's stem
        to -- ``reference`` for ``e02_deep`` -- or :data:`dmf.eval.assemble.NOT_AN_ARM` for a
        config that is not an ablation arm at all, which ``e03_probabilistic`` is not. A
        **non-empty** sentinel either way: the renderer groups the control summary on this
        column and pandas drops NaN group keys.
    """
    try:
        arms = arms_for_experiment(config.stem)
    except ValueError:
        # Not an ablation arm at all. `e03_probabilistic` is the Phase 5 head experiment, and
        # its controls belong in the table anyway -- section 6.4 renders its rows, and a
        # published number whose controls are in a file nobody assembles is a number nobody
        # audited.
        return NOT_AN_ARM
    return arms[0].arm


def _tag(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Label a scored frame with the experiment that produced it.

    Two configs are scored into one set of tables, so a row that does not say which
    experiment it came from cannot be told from a duplicate of another. The column is added
    here rather than in :mod:`dmf.eval.scoring`, which scores one experiment at a time and
    has nothing to say about the others.

    Args:
        frame: A scored table.
        experiment: The config's ``name``.

    Returns:
        A copy carrying ``experiment`` as its first column, or the frame unchanged if empty.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    out.insert(0, "experiment", experiment)
    return out


def score_all(
    configs: list[Path],
    *,
    corpus: Path,
    out_dir: Path,
    checkpoints: Path,
    device: str,
    batch_size: int,
    all_regimes: bool,
    regime: str | None,
) -> ScoringArtifacts:
    """Score every requested config and write one set of tables.

    Args:
        configs: Experiment config paths.
        corpus: Corpus dataset root.
        out_dir: Directory the tables are written to.
        checkpoints: Root of the committed checkpoints.
        device: Torch device.
        batch_size: Windows per batch.
        all_regimes: Whether ``--all-regimes`` was passed.
        regime: The single regime named by ``--regime``, if any.

    Returns:
        The combined artifacts, already written.

    Raises:
        ValueError: As :func:`resolve_regimes` raises.
        FileNotFoundError: If a checkpoint an SGD row needs is absent. Scoring never falls
            back to a random initialisation.
    """
    parts: list[ScoringArtifacts] = []
    skipped: dict[str, str] = {}
    for path in configs:
        cfg = load_experiment(path)
        chosen = resolve_regimes(cfg, all_regimes=all_regimes, regime=regime)
        print(f"scoring {cfg.name} on {list(chosen)} from {checkpoints / cfg.name}", flush=True)
        scored = score_experiment(
            cfg,
            corpus,
            results_dir=out_dir,
            regimes=chosen,
            device=device,
            checkpoint_root=checkpoints,
            batch_size=batch_size,
            write=False,
        )
        for key, reason in scored.skipped.items():
            skipped[f"{cfg.name}/{key}"] = reason
        parts.append(
            ScoringArtifacts(
                metrics_full=_tag(scored.metrics_full, cfg.name),
                metrics_by_cell=_tag(scored.metrics_by_cell, cfg.name),
                quiescence=_tag(scored.quiescence, cfg.name),
                quiescence_lead_times=_tag(scored.quiescence_lead_times, cfg.name),
                probabilistic=_tag(scored.probabilistic, cfg.name),
            )
        )

    def _stack(name: str) -> pd.DataFrame:
        frames = [getattr(part, name) for part in parts if not getattr(part, name).empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    combined = ScoringArtifacts(
        metrics_full=_stack("metrics_full"),
        metrics_by_cell=_stack("metrics_by_cell"),
        quiescence=_stack("quiescence"),
        quiescence_lead_times=_stack("quiescence_lead_times"),
        probabilistic=_stack("probabilistic"),
        skipped=skipped,
    )
    for key, filename in SCORING_ARTIFACTS.items():
        frame = getattr(combined, key)
        # No empty-table placeholder: an empty CSV in results/ cannot be told from a table
        # with nothing to report, and `skipped` is where an absence is explained.
        if not frame.empty:
            combined.paths[key] = write_table(frame, out_dir / filename)
    for key, reason in sorted(combined.skipped.items()):
        print(f"skipped {key}: {reason}", flush=True)
    return combined


def main(argv: list[str] | None = None) -> int:
    """Score from committed artifacts, then render ``results.md`` from the CSVs.

    Args:
        argv: Command-line arguments, or None to read ``sys.argv``.

    Returns:
        Process exit code. 0 on success; the exceptions this raises are deliberately not
        caught, because a report that renders after a scoring failure is the failure mode
        Gate 6 exists to prevent.
    """
    args = build_parser().parse_args(argv)
    out_dir = args.results_dir / PHASE6_SUBDIR
    if args.controls_only and args.render_only:
        raise SystemExit("--controls-only and --render-only are mutually exclusive")
    if args.controls_only and args.no_controls:
        raise SystemExit("--controls-only and --no-controls are mutually exclusive")
    if not args.render_only:
        if args.controls_only:
            print("reusing committed scoring artifacts: --controls-only", flush=True)
        else:
            score_all(
                list(args.config),
                corpus=args.corpus,
                out_dir=out_dir,
                checkpoints=args.checkpoints,
                device=args.device,
                batch_size=args.batch_size,
                all_regimes=bool(args.all_regimes),
                regime=args.regime,
            )
        if args.no_controls:
            print("skipped controls: --no-controls", flush=True)
        else:
            controls = run_controls(
                [load_experiment(path) for path in args.config],
                args.corpus,
                results_dir=out_dir,
                # The config STEM, not `cfg.name`: `configs/experiment/e02_deep.yaml` is
                # named `deep` inside, and these rows must key on the same experiment names
                # `controls.csv` does or the two control tables cannot be read side by side.
                experiments=[path.stem for path in args.config],
                arms=[_arm_of(path) for path in args.config],
                device=args.device,
                batch_size=args.batch_size,
                regimes=None if args.regime is None else [args.regime],
            )
            for name, path in sorted(controls.paths.items()):
                print(f"controls {name} -> {path}", flush=True)
            for name in CONTROL_ARTIFACTS:
                if name not in controls.paths:
                    print(f"no {name} table written", flush=True)
            for key, reason in sorted(controls.skipped.items()):
                print(f"control skipped {key}: {reason}", flush=True)
        assembled = assemble_phase6_tables(args.results_dir, out_dir=out_dir)
        for name, path in sorted(assembled.paths.items()):
            print(f"assembled {name} -> {path}", flush=True)
        # Printed, not swallowed: an arm that could not be assembled is an absent
        # measurement, and the reader of a sweep log should see which one and why at the
        # moment it happens rather than infer it later from a table that is smaller than
        # expected.
        for key, reason in sorted(assembled.skipped.items()):
            print(f"not assembled {key}: {reason}", flush=True)
    # The renderer is handed the results **root**, not the directory it just wrote: every
    # provenance marker names its source relative to the document, and section 6.4 reads the
    # committed Phase 5 heads from `results/e03/` beside it.
    report = build_results_report(
        args.results_dir, args.results_dir / REPORT_NAME, strict=not args.allow_missing
    )
    print(f"wrote {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
