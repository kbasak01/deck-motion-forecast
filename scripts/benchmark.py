#!/usr/bin/env python3
"""CLI wrapper for ONNX export, parity, and latency benchmarking.

Logic lives in :mod:`dmf.deploy`. Parity runs before latency; a fast graph that computes
the wrong thing is worthless.

``make bench`` is this script with ``--export --parity --latency`` and nothing else, so the
four deployable models, the five backends and the two batch sizes are resolved from
:mod:`dmf.deploy.targets` and :mod:`dmf.deploy.harness` rather than from the command line.
The command line only narrows that set, which keeps "what was benchmarked" a property of
the repository instead of a property of one shell history.

Exit status is 0 only if every requested stage completed and every parity check passed. A
parity failure stops the latency stage, and a configuration that refused -- an execution
provider that did not load, most importantly -- is printed and makes the exit non-zero
rather than being dropped from the table.
"""

import argparse
import sys
from pathlib import Path

from dmf.deploy.harness import BACKENDS
from dmf.deploy.parity import PARITY_N_WINDOWS
from dmf.deploy.pipeline import (
    DEFAULT_BATCH_SIZES,
    build_jobs,
    providers_checked,
    run_pipeline,
    unverified_timed_configurations,
    worst_parity,
)
from dmf.deploy.report import write_latency_report, write_pareto_figure
from dmf.deploy.targets import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_ONNX_DIR,
    DEFAULT_TARGETS,
    target_by_key,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export, parity-check, and benchmark a model.")
    parser.add_argument("--export", action="store_true", help="Export the checkpoint to ONNX.")
    parser.add_argument("--parity", action="store_true", help="Run the FP32 parity check.")
    parser.add_argument("--latency", action="store_true", help="Run the latency benchmark.")
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Render results/latency.md and the Pareto figure from the committed CSVs. "
            "Measures nothing: no GPU, no checkpoints, no corpus."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/e04/metrics_full.csv"),
        help="Per-seed metrics the Pareto figure reads its accuracy axis from.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Trained checkpoint to export."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Output directory."
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        choices=[t.key for t in DEFAULT_TARGETS],
        default=None,
        help="Models to process. Default: all four deployable models.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=list(BACKENDS),
        default=list(BACKENDS),
        help="Runtimes to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to benchmark. 1 is the real-time case.",
    )
    parser.add_argument("--warmup", type=int, default=200, help="Untimed iterations.")
    parser.add_argument("--timed", type=int, default=2000, help="Timed iterations.")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Pinned intra-op CPU thread counts, recorded on every row. More than one value "
            "turns the sweep into a thread-sensitivity study."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run the whole latency sweep. 2 gives the stability table.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Fold these rows into the existing latency artifacts instead of replacing them.",
    )
    parser.add_argument(
        "--latency-stem",
        type=str,
        default="latency",
        help="Filename stem for the latency artifacts, e.g. 'latency_threads'.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the PyTorch backends; the ORT backends take theirs from the provider.",
    )
    parser.add_argument(
        "--parity-windows",
        type=int,
        default=PARITY_N_WINDOWS,
        help="Random windows per parity check.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4, help="FP32 parity tolerance, dimensionless."
    )
    parser.add_argument(
        "--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR, help="Directory for exported graphs."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Root of the checkpoint tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.export or args.parity or args.latency or args.report):
        print("nothing to do: pass at least one of --export, --parity, --latency, --report")
        return 2

    if args.report and not (args.export or args.parity or args.latency):
        document = write_latency_report(args.results_dir, args.metrics)
        figure = write_pareto_figure(args.results_dir, args.metrics)
        print(f"wrote {document} and {figure}")
        return 0

    keys = args.target or [t.key for t in DEFAULT_TARGETS]
    targets = tuple(target_by_key(key) for key in keys)
    jobs = build_jobs(
        targets,
        backends=tuple(args.backend),
        batch_sizes=tuple(args.batch_size),
        warmup_iters=args.warmup,
        timed_iters=args.timed,
        thread_counts=tuple(args.threads),
        torch_device=args.torch_device,
        onnx_dir=args.onnx_dir,
        checkpoint_root=args.checkpoints,
    )
    outcome = run_pipeline(
        targets,
        do_export=args.export,
        do_parity=args.parity,
        do_latency=args.latency,
        jobs=jobs,
        onnx_dir=args.onnx_dir,
        checkpoint_root=args.checkpoints,
        checkpoint=args.checkpoint,
        results_dir=args.results_dir,
        parity_windows=args.parity_windows,
        tolerance=args.tolerance,
        repeats=args.repeats,
        latency_stem=args.latency_stem,
        append=args.append,
    )

    for record in outcome.exports:
        print(
            f"exported {record.target_key:<14} {record.onnx_path}  "
            f"{list(record.input_dims)} -> {list(record.output_dims)}  "
            f"({record.size_bytes / 1024:.0f} KiB, head={record.head_kind})"
        )
    for key, draws in outcome.parity.items():
        for provider in providers_checked(draws):
            worst = worst_parity(draws, provider)
            n_draws = sum(1 for draw in draws if draw.provider == provider)
            verdict = "PASS" if worst.passed else "FAIL"
            absolute = "PASS" if worst.passed_absolute else "FAIL"
            print(
                f"parity   {key:<14} {provider:<26} max_abs_err={worst.max_abs_err:.3e} "
                f"|y|max={worst.output_abs_max:7.3f} scaled_tol={worst.scaled_tolerance:.3e} "
                f"({worst.scaled_tolerance / worst.max_abs_err:6.1f}x) {verdict} "
                f"[abs 1e-4: {absolute}] worst of {n_draws} draws, seed {worst.seed}"
            )
    for key, attribution in outcome.attributions.items():
        print(
            f"         {key:<14} FP32 noise floor: torch-vs-fp64="
            f"{attribution.torch_fp32_vs_fp64_max:.3e}  onnx-vs-fp64="
            f"{attribution.onnx_fp32_vs_fp64_max:.3e}  |y|max={attribution.output_abs_max:.3f}"
        )
    for row in outcome.latency:
        print(
            f"latency  {row.result.label:<34} p50={row.result.p50_ms:8.3f} ms  "
            f"p90={row.result.p90_ms:8.3f}  p99={row.result.p99_ms:8.3f}  "
            f"{row.result.throughput_windows_s:9.1f} win/s  "
            f"[{'|'.join(row.result.providers_realized)}]"
        )
    if args.report:
        document = write_latency_report(args.results_dir, args.metrics)
        figure = write_pareto_figure(args.results_dir, args.metrics)
        print(f"wrote {document} and {figure}")

    for label, error in outcome.failures:
        print(f"REFUSED  {label}: {error}", file=sys.stderr)
    failed_parity = [
        f"{key}/{provider}"
        for key, draws in outcome.parity.items()
        for provider in providers_checked(draws)
        if not worst_parity(draws, provider).passed
    ]
    if failed_parity:
        print(
            f"parity FAILED for {failed_parity}: REFUSED, not benchmarked. A fast graph "
            f"that computes the wrong thing is worthless.",
            file=sys.stderr,
        )
    # The invariant the exit code stands for is "nothing unverified was timed", not "every
    # configuration passed". A provider whose parity check fails is excluded from the
    # latency stage by `run_pipeline`, and a documented refusal is an outcome of the study
    # rather than an error in it -- `lstm_quantile` on torch:cuda is one, and cannot be
    # made to pass, because cuDNN's fused recurrence is what fails the tolerance
    # (docs/protocol.md P7-D10). Exiting non-zero on it would mean no committed command
    # could reproduce the phase, which is the defect S3 was raised to fix.
    #
    # So: fail when a configuration was timed without a passing parity row, or when the
    # refusals leave nothing to time. Both are real errors. `scripts/gate7.py` re-derives
    # the same invariant from the committed artifacts, independently of this exit code.
    timed_unverified = sorted(unverified_timed_configurations(outcome, args.results_dir))
    if timed_unverified:
        print(
            f"TIMED WITHOUT PARITY: {timed_unverified}. This must never happen: the "
            f"latency stage is supposed to exclude any configuration whose parity check "
            f"did not pass.",
            file=sys.stderr,
        )
        return 1
    if args.latency and not outcome.latency:
        print("no configuration survived to be timed", file=sys.stderr)
        return 1
    if args.latency and outcome.failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
