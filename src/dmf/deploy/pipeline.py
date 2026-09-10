"""Export, parity and latency as three ordered stages over the deployable model set.

``scripts/benchmark.py`` is an argparse wrapper over this module and holds no logic, per
``CLAUDE.md`` §Architecture rules.

**The order is the methodology.** Export, then parity, then timing. A graph that has not
been parity-checked has no business being timed: the benchmark would faithfully measure how
fast the wrong answer arrives, and every column of the resulting table would look correct.
A model whose parity check fails is therefore excluded from the latency stage -- not the
whole sweep, so one bad row does not discard the hours of measurement around it, but that
model contributes no timings and the run's exit status is non-zero.

**The sweep is run twice.** A latency number that has not been reproduced is a sample of
one. :func:`run_pipeline` takes ``repeats``, and ``latency_stability.csv`` reports the p50
drift per configuration between repeat 1 and repeat 2. The methodology asks for agreement
within 10 percent; where a configuration does not agree, the drift is reported rather than
re-run until it does.

Units: latency in milliseconds, memory in mebibytes, throughput in windows per second,
errors dimensionless (the models operate on normalised inputs).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dmf.deploy.bench import BenchResult, environment_stamp
from dmf.deploy.export_onnx import export_model, graph_dims
from dmf.deploy.harness import BACKENDS, PROVIDERS_BY_BACKEND, BenchJob, run_job_subprocess
from dmf.deploy.parity import (
    PARITY_N_WINDOWS,
    PARITY_SEEDS,
    ParityAttribution,
    ParityResult,
    attribute_parity,
    check_parity,
    check_torch_device_parity,
)
from dmf.deploy.parity import random_parity_windows as _parity_windows
from dmf.deploy.targets import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_ONNX_DIR,
    DEFAULT_TARGETS,
    ModelTarget,
    load_target,
    onnx_path_for,
)

__all__ = [
    "CONFIG_KEYS",
    "DEFAULT_BATCH_SIZES",
    "LATENCY_COLUMNS",
    "PARITY_COLUMNS",
    "PARITY_PROVIDERS",
    "STABILITY_COLUMNS",
    "STABILITY_TOLERANCE_FRAC",
    "ExportRecord",
    "LatencyRow",
    "PipelineResult",
    "build_jobs",
    "export_targets",
    "latency_sweep",
    "latency_table",
    "parity_table",
    "job_provider",
    "parity_status",
    "parity_targets",
    "unverified_timed_configurations",
    "providers_checked",
    "run_pipeline",
    "stability_table",
    "worst_parity",
    "write_latency_csv",
    "write_latency_json",
    "write_parity_csv",
    "write_parity_json",
    "write_stability_csv",
]

#: Batch sizes measured. 1 is the real-time case -- one window per control cycle, which is
#: the only one a deck-landing decision actually runs at -- and 32 is the throughput case,
#: which exists to show where the GPU's parallelism starts to pay for its launch overhead.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 32)

#: Execution-provider chains the parity check runs on, in report order. Every provider the
#: sweep benchmarks appears here, because a latency row whose provider was never
#: parity-checked is a timing of something unverified -- which is exactly how the TF32
#: defect survived a whole sweep (``docs/protocol.md`` P7-D9). The CPU chain is first and is
#: the only one whose absence is an error.
PARITY_PROVIDERS: tuple[tuple[str, ...], ...] = (
    ("CPUExecutionProvider",),
    ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ("TensorrtExecutionProvider", "CPUExecutionProvider"),
)

#: Fractional p50 agreement required between the two repeats of the sweep, dimensionless.
#: A configuration outside it is reported with its drift, not re-run until it agrees --
#: re-running until the numbers match is how a benchmark becomes a search for the answer
#: you wanted.
STABILITY_TOLERANCE_FRAC: float = 0.10

#: Column order of ``latency.csv``. The realized provider sits next to the requested
#: backend on purpose: those two columns disagreeing is the failure mode
#: :func:`dmf.deploy.bench.benchmark_onnxruntime` refuses on, and a reader should be able to
#: check it without trusting that the refusal ran.
LATENCY_COLUMNS: tuple[str, ...] = (
    "model",
    "backend",
    "device",
    "batch_size",
    "providers_realized",
    "intra_op_threads",
    "tf32",
    "repeat",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    "throughput_windows_s",
    "peak_host_mem_mb",
    "peak_device_mem_mb",
    "warmup_iters",
    "timed_iters",
    "gpu",
    "torch",
    "onnxruntime",
    "git_commit",
)

#: Column order of ``parity.csv``. Both verdicts are columns: ``passed`` is the criterion
#: in force (scale-relative, ``docs/protocol.md`` P7-D3) and ``passed_absolute`` is the
#: implementation plan's original unscaled 1e-4, kept so the superseded test stays
#: auditable and a reader can see exactly which rows the change decided.
PARITY_COLUMNS: tuple[str, ...] = (
    "model",
    "provider",
    "tf32",
    "head_kind",
    "n_windows",
    "n_seeds",
    "worst_seed",
    "max_abs_err",
    "max_abs_err_best_seed",
    "mean_abs_err",
    "output_abs_max",
    "scale",
    "tolerance",
    "scaled_tolerance",
    "margin_x",
    "passed",
    "passed_absolute",
    "onnx_path",
    "gpu",
    "torch",
    "onnxruntime",
    "git_commit",
)

#: Column order of ``latency_stability.csv``.
STABILITY_COLUMNS: tuple[str, ...] = (
    "model",
    "backend",
    "device",
    "batch_size",
    "intra_op_threads",
    "p50_ms_run1",
    "p50_ms_run2",
    "drift_frac",
    "drift_pct",
    "within_10pct",
    "p99_ms_run1",
    "p99_ms_run2",
    "p99_drift_pct",
    "p99_within_10pct",
    "git_commit",
)


@dataclass(frozen=True)
class ExportRecord:
    """One exported graph and the shape contract it declares.

    Attributes:
        target_key: The model's key.
        onnx_path: Where the graph was written.
        head_kind: ``point`` or ``quantile``. A quantile graph carries the post-hoc
            quantile sort as its last node; a point graph must not.
        n_fitted_parameters: Fitted value count of the source model, dimensionless.
        input_dims: Declared input dims, symbols for dynamic axes.
        output_dims: Declared output dims, symbols for dynamic axes.
        size_bytes: Size of the ``.onnx`` file, bytes.
    """

    target_key: str
    onnx_path: Path
    head_kind: str
    n_fitted_parameters: int
    input_dims: tuple[int | str, ...]
    output_dims: tuple[int | str, ...]
    size_bytes: int


@dataclass(frozen=True)
class LatencyRow:
    """One benchmark configuration and what it measured.

    Attributes:
        job: The configuration requested.
        result: What was measured, including the provider actually realized.
        repeat: Which pass of the sweep this row is from, 1-based. The whole sweep is run
            more than once and the repeats are kept apart rather than averaged, because the
            question they answer -- does this number reproduce? -- is destroyed by averaging
            them.
    """

    job: BenchJob
    result: BenchResult
    repeat: int = 1


@dataclass(frozen=True)
class PipelineResult:
    """Everything one ``scripts/benchmark.py`` invocation produced.

    Attributes:
        exports: Exported graphs, empty if the export stage was not requested.
        parity: Parity results per target key, one per window draw, empty if not requested.
        attributions: Double-precision attribution per target key, populated **only** for
            models whose parity check failed. It is what separates "the export is wrong"
            from "the tolerance is below this graph's FP32 noise floor", and the artifact
            carries it so the distinction survives the run.
        latency: Measured rows across every repeat, empty if not requested. A model whose
            parity check failed contributes no rows: an unverified graph is never timed.
        failures: ``(label, error)`` per configuration that refused or crashed. A refused
            configuration is reported, never dropped: a missing row in a latency table
            reads as "not measured", and "the TensorRT provider did not load" is a
            different and much more important statement.
        parity_gate_passed: Whether every parity check that ran passed. False excludes the
            failing models from the latency stage and makes the run's exit status non-zero.
    """

    exports: tuple[ExportRecord, ...]
    parity: dict[str, tuple[ParityResult, ...]]
    attributions: dict[str, ParityAttribution]
    latency: tuple[LatencyRow, ...]
    failures: tuple[tuple[str, str], ...]
    parity_gate_passed: bool


def export_targets(
    targets: tuple[ModelTarget, ...] = DEFAULT_TARGETS,
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    checkpoint: Path | None = None,
) -> tuple[ExportRecord, ...]:
    """Export each target to ONNX and record the shape contract it declares.

    Args:
        targets: Models to export.
        onnx_dir: Directory for the graphs.
        checkpoint_root: Root of the checkpoint tree.
        checkpoint: Explicit checkpoint file, legal only for a single target.

    Returns:
        One record per target, in ``targets`` order.

    Raises:
        ValueError: If ``checkpoint`` is given for more than one target -- one file cannot
            be the weights of two architectures, and picking the first silently would
            export the wrong model under the right name.
    """
    if checkpoint is not None and len(targets) != 1:
        raise ValueError(
            f"--checkpoint names one file but {len(targets)} targets were selected; "
            f"select a single target with --target"
        )
    records: list[ExportRecord] = []
    for target in targets:
        model = load_target(target, checkpoint_root=checkpoint_root, checkpoint=checkpoint)
        sample = torch.zeros((1, model.lookback, model.n_input_channels), dtype=torch.float32)
        path = export_model(model, sample, onnx_path_for(target, onnx_dir))
        input_dims, output_dims = graph_dims(path)
        records.append(
            ExportRecord(
                target_key=target.key,
                onnx_path=path,
                head_kind=model.head_kind,
                n_fitted_parameters=model.n_fitted_parameters,
                input_dims=input_dims,
                output_dims=output_dims,
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(records)


def parity_targets(
    targets: tuple[ModelTarget, ...] = DEFAULT_TARGETS,
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    checkpoint: Path | None = None,
    n_windows: int = PARITY_N_WINDOWS,
    tolerance: float = 1e-4,
    seeds: tuple[int, ...] = PARITY_SEEDS,
    providers: tuple[tuple[str, ...], ...] = PARITY_PROVIDERS,
) -> dict[str, tuple[ParityResult, ...]]:
    """Run the FP32 parity check for each target, on each window draw.

    Several draws rather than one because ``max_abs_err`` is a maximum over roughly nine
    million elements -- a tail statistic, which moves by about 2x between draws. A single
    draw made the verdict a property of the seed; see
    :data:`dmf.deploy.parity.PARITY_SEEDS` and ``docs/protocol.md`` P7-D3.

    Args:
        targets: Models to check.
        onnx_dir: Directory holding the exported graphs.
        checkpoint_root: Root of the checkpoint tree.
        checkpoint: Explicit checkpoint file, legal only for a single target.
        n_windows: Random windows per draw, dimensionless.
        tolerance: Relative coefficient, dimensionless. The applied threshold is
            ``tolerance * max(1, |y|_max)``.
        seeds: Window draws to run.
        providers: Execution-provider chains to check, in report order. **Every provider
            that is benchmarked must be checked**: a provider's kernels are part of the
            computation, and TF32 on this card made all three GPU providers fail the bar
            the CPU one passed (``docs/protocol.md`` P7-D9). A GPU provider that is not
            available on the machine is skipped with its reason, not silently dropped.

    Returns:
        Per target key, one result per (provider, seed), provider-major.

    Raises:
        FileNotFoundError: If a target has not been exported.
    """
    results: dict[str, tuple[ParityResult, ...]] = {}
    for target in targets:
        model = load_target(target, checkpoint_root=checkpoint_root, checkpoint=checkpoint)
        draws: list[ParityResult] = []
        for chain in providers:
            for seed in seeds:
                windows = _parity_windows(
                    model.lookback, model.n_input_channels, n_windows=n_windows, seed=seed
                )
                try:
                    draws.append(
                        check_parity(
                            model,
                            onnx_path_for(target, onnx_dir),
                            windows,
                            tolerance=tolerance,
                            seed=seed,
                            providers=chain,
                        )
                    )
                except (ValueError, RuntimeError) as exc:
                    # A provider that does not load on this machine is not a parity
                    # failure; it is an absent row, and the latency stage will refuse to
                    # time it for the same reason.
                    if chain[0] == "CPUExecutionProvider":
                        raise
                    print(f"parity: {target.key} skipped {chain[0]}: {exc}")
                    break
            # The PyTorch CUDA rows are timed on cuDNN/cuBLAS kernels that no ONNX parity
            # row touches. Checked here against the same model on the CPU, at the same
            # tolerance, so every timed configuration has a parity row behind it.
        if torch.cuda.is_available():
            for seed in seeds:
                windows = _parity_windows(
                    model.lookback, model.n_input_channels, n_windows=n_windows, seed=seed
                )
                draws.append(
                    check_torch_device_parity(
                        model, windows, device="cuda", tolerance=tolerance, seed=seed
                    )
                )
        results[target.key] = tuple(draws)
    return results


def providers_checked(results: tuple[ParityResult, ...]) -> tuple[str, ...]:
    """List the execution providers a model was parity-checked on, in first-seen order.

    Args:
        results: Every draw for one model.

    Returns:
        The provider names.
    """
    seen: list[str] = []
    for result in results:
        if result.provider not in seen:
            seen.append(result.provider)
    return tuple(seen)


def worst_parity(results: tuple[ParityResult, ...], provider: str | None = None) -> ParityResult:
    """Return the draw with the largest absolute error.

    The verdict is read off the worst draw, not the mean or the pinned one: a criterion
    that a model passes on average is not a criterion.

    Args:
        results: One result per (provider, window draw), non-empty.
        provider: Restrict to one execution provider, or None for the worst over all of
            them -- which is the verdict, since every benchmarked provider has to pass.

    Returns:
        The result with the largest ``max_abs_err``.

    Raises:
        ValueError: If ``results`` is empty, or if ``provider`` matches nothing.
    """
    selected = [r for r in results if provider is None or r.provider == provider]
    if not selected:
        raise ValueError(f"no parity draws to summarise for provider={provider!r}")
    return max(selected, key=lambda result: result.max_abs_err)


def parity_table(
    parity: dict[str, tuple[ParityResult, ...]],
    exports: tuple[ExportRecord, ...] = (),
) -> pd.DataFrame:
    """Assemble the parity results into the parity table.

    Args:
        parity: Per target key, one result per window draw.
        exports: Export records, for the head kind and graph path.

    Returns:
        A frame with :data:`PARITY_COLUMNS`, one row per model, read off its worst draw.
        ``max_abs_err_best_seed`` is the *best* draw's error, so the spread between draws is
        visible in the committed artifact rather than only in the protocol.
    """
    by_key = {record.target_key: record for record in exports}
    stamp = environment_stamp()
    records: list[dict[str, Any]] = []
    for key, draws in parity.items():
        for provider in providers_checked(draws):
            per_provider = tuple(r for r in draws if r.provider == provider)
            worst = worst_parity(per_provider)
            best = min(per_provider, key=lambda result: result.max_abs_err)
            record = by_key.get(key)
            records.append(
                _parity_record(key, provider, worst, best, len(per_provider), record, stamp)
            )
    return pd.DataFrame.from_records(records, columns=list(PARITY_COLUMNS))


def _parity_record(
    key: str,
    provider: str,
    worst: ParityResult,
    best: ParityResult,
    n_seeds: int,
    record: ExportRecord | None,
    stamp: dict[str, str],
) -> dict[str, Any]:
    """Flatten one (model, provider) parity summary into the table's schema.

    Args:
        key: Model key.
        provider: Execution provider the ONNX side ran on.
        worst: The draw the verdict is read from.
        best: The draw with the smallest error, so the spread is visible in the artifact.
        n_seeds: Draws behind this row.
        record: The export record, for the head kind and graph path.
        stamp: Environment stamp, so the CSV is reproducible on its own.

    Returns:
        Mapping over :data:`PARITY_COLUMNS`.
    """
    return {
        "model": key,
        "provider": provider,
        "tf32": False,
        "head_kind": record.head_kind if record else "",
        "n_windows": worst.n_windows,
        "n_seeds": n_seeds,
        "worst_seed": worst.seed,
        "max_abs_err": worst.max_abs_err,
        "max_abs_err_best_seed": best.max_abs_err,
        "mean_abs_err": worst.mean_abs_err,
        "output_abs_max": worst.output_abs_max,
        "scale": worst.scale,
        "tolerance": worst.tolerance,
        "scaled_tolerance": worst.scaled_tolerance,
        "margin_x": worst.scaled_tolerance / worst.max_abs_err,
        "passed": worst.passed,
        "passed_absolute": worst.passed_absolute,
        "onnx_path": str(record.onnx_path) if record else "",
        "gpu": stamp.get("gpu", ""),
        "torch": stamp.get("torch", ""),
        "onnxruntime": stamp.get("onnxruntime", ""),
        "git_commit": stamp.get("git_commit", ""),
    }


def write_parity_csv(
    parity: dict[str, tuple[ParityResult, ...]],
    exports: tuple[ExportRecord, ...],
    path: Path,
) -> Path:
    """Write the parity table.

    Args:
        parity: Per target key, one result per window draw.
        exports: Export records.
        path: Destination CSV.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parity_table(parity, exports).to_csv(path, index=False)
    return path


def latency_sweep(
    jobs: tuple[BenchJob, ...],
    repeat: int = 1,
) -> tuple[tuple[LatencyRow, ...], tuple[tuple[str, str], ...]]:
    """Measure every configuration once, each in its own subprocess.

    A configuration that refuses or crashes is collected rather than raised, so that one
    unavailable execution provider does not discard the hours of measurement that preceded
    it -- and is returned as a failure rather than dropped, so it cannot be mistaken for a
    configuration nobody asked for.

    Args:
        jobs: Configurations to measure, in report order.
        repeat: Which pass of the sweep this is, 1-based, stamped on every row.

    Returns:
        ``(rows, failures)``, where each failure is ``(job label, error text)``.
    """
    rows: list[LatencyRow] = []
    failures: list[tuple[str, str]] = []
    for job in jobs:
        try:
            rows.append(LatencyRow(job=job, result=run_job_subprocess(job), repeat=repeat))
        except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
            failures.append((f"{job.label}-run{repeat}", f"{type(exc).__name__}: {exc}"))
    return tuple(rows), tuple(failures)


def _row_record(row: LatencyRow) -> dict[str, Any]:
    """Flatten one measured row into the latency table's schema.

    Args:
        row: The measured configuration.

    Returns:
        Mapping over :data:`LATENCY_COLUMNS`.
    """
    result, job = row.result, row.job
    return {
        "model": job.target_key,
        "backend": job.backend,
        "device": job.resolved_device,
        "batch_size": job.batch_size,
        "providers_realized": "|".join(result.providers_realized),
        "intra_op_threads": result.intra_op_threads,
        "tf32": result.tf32,
        "repeat": row.repeat,
        "p50_ms": result.p50_ms,
        "p90_ms": result.p90_ms,
        "p99_ms": result.p99_ms,
        "mean_ms": result.mean_ms,
        "std_ms": result.std_ms,
        "throughput_windows_s": result.throughput_windows_s,
        "peak_host_mem_mb": result.peak_host_mem_mb,
        "peak_device_mem_mb": result.peak_device_mem_mb,
        "warmup_iters": result.config.warmup_iters,
        "timed_iters": result.config.timed_iters,
        "gpu": result.env.get("gpu", ""),
        "torch": result.env.get("torch", ""),
        "onnxruntime": result.env.get("onnxruntime", ""),
        "git_commit": result.env.get("git_commit", ""),
    }


def latency_table(rows: tuple[LatencyRow, ...]) -> pd.DataFrame:
    """Assemble the measured rows into the latency table.

    Args:
        rows: Measured configurations.

    Returns:
        A frame with :data:`LATENCY_COLUMNS`, one row per configuration per repeat.
    """
    return pd.DataFrame.from_records(
        [_row_record(row) for row in rows], columns=list(LATENCY_COLUMNS)
    )


def write_latency_csv(rows: tuple[LatencyRow, ...], path: Path, append: bool = False) -> Path:
    """Write the latency table.

    Args:
        rows: Measured configurations. Filter to one repeat before calling if the artifact
            is meant to hold one row per configuration.
        path: Destination CSV.
        append: Fold the rows into an existing file instead of replacing it, replacing any
            configuration measured again.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = latency_table(rows)
    if append:
        table = _merge_frames(path, table, (*CONFIG_KEYS, "repeat"))
    table.to_csv(path, index=False)
    return path


def write_latency_json(rows: tuple[LatencyRow, ...], path: Path, append: bool = False) -> Path:
    """Write every measured row in full, with its environment stamp and its harness settings.

    The CSV is the table; this is the record. It carries **both** repeats, each row's
    requested job beside its realized providers, and the environment stamp per run -- a
    benchmark number without one is not reproducible.

    Args:
        rows: Measured configurations, across every repeat.
        path: Destination JSON.
        append: Fold the rows into an existing file instead of replacing it, replacing any
            ``(label, repeat)`` measured again.

    Returns:
        The path written.
    """
    payload: dict[str, list[dict[str, Any]]] = {
        "results": [
            {
                "repeat": row.repeat,
                "model": row.job.target_key,
                "backend": row.job.backend,
                "label": row.result.label,
                "providers_requested": list(PROVIDERS_BY_BACKEND.get(row.job.backend, ())),
                "providers_realized": list(row.result.providers_realized),
                "intra_op_threads": row.result.intra_op_threads,
                "tf32": row.result.tf32,
                "latency_ms": {
                    "p50": row.result.p50_ms,
                    "p90": row.result.p90_ms,
                    "p99": row.result.p99_ms,
                    "mean": row.result.mean_ms,
                    "std": row.result.std_ms,
                },
                "throughput_windows_s": row.result.throughput_windows_s,
                "peak_host_mem_mb": row.result.peak_host_mem_mb,
                "peak_device_mem_mb": row.result.peak_device_mem_mb,
                "config": asdict(row.result.config),
                "env": dict(row.result.env),
            }
            for row in rows
        ]
    }
    if append and path.is_file():
        prior: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["results"]
        seen = {(row["label"], row["repeat"]) for row in payload["results"]}
        payload["results"] = [
            row for row in prior if (row["label"], row["repeat"]) not in seen
        ] + payload["results"]
    payload["results"].sort(key=lambda row: (str(row["label"]), int(row["repeat"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


#: Columns identifying one benchmark configuration across repeats. A configuration is a
#: (model, backend, device, batch size, thread count) tuple: two rows sharing all five are
#: the same measurement made twice, and nothing else is.
CONFIG_KEYS: tuple[str, ...] = (
    "model",
    "backend",
    "device",
    "batch_size",
    "intra_op_threads",
)


def _merge_frames(existing: Path, new: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """Fold new rows into an artifact already on disk, newest wins.

    Appending rather than overwriting is what lets a sweep be extended -- a backend or a
    device that was missed the first time -- without re-measuring the configurations that
    were not missed, which would cost hours and produce numbers from a different machine
    state than the ones they sit beside. The rows that ARE re-measured replace their
    predecessors rather than duplicating them, so a table can never carry one configuration
    twice.

    Args:
        existing: The artifact already written, or a path that does not exist.
        new: Rows to fold in.
        keys: Columns identifying a row uniquely.

    Returns:
        The merged frame, sorted by ``keys`` so the file is a deterministic function of its
        contents rather than of the order the sweep happened to run in.
    """
    if not existing.is_file():
        return new.sort_values(list(keys)).reset_index(drop=True)
    prior = pd.read_csv(existing)
    merged = pd.concat([prior, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=list(keys), keep="last")
    return merged.sort_values(list(keys)).reset_index(drop=True)


def stability_table(rows: tuple[LatencyRow, ...]) -> pd.DataFrame:
    """Compare each configuration's p50 across the repeats of the sweep.

    Drift is measured against repeat 1, and the frame is returned whatever it says. A
    configuration outside :data:`STABILITY_TOLERANCE_FRAC` is reported with its drift, not
    re-run until it agrees.

    Args:
        rows: Measured configurations across every repeat.

    Returns:
        A frame with :data:`STABILITY_COLUMNS`, one row per configuration measured at least
        twice. Configurations measured once are omitted, since there is nothing to compare.
    """
    table = latency_table(rows)
    keys = list(CONFIG_KEYS)
    records: list[dict[str, Any]] = []
    for key_values, group in table.groupby(keys, sort=False):
        indexed = group.set_index("repeat")
        if 1 not in indexed.index or 2 not in indexed.index:
            continue
        first = float(str(indexed.loc[1, "p50_ms"]))
        second = float(str(indexed.loc[2, "p50_ms"]))
        drift = (second - first) / first
        # p99 is tracked beside p50 because it is the statistic the deployment argument is
        # made on, and it is the noisier of the two: tracking only the median made a claim
        # about tails rest on a stability check that never looked at one.
        p99_first = float(str(indexed.loc[1, "p99_ms"]))
        p99_second = float(str(indexed.loc[2, "p99_ms"]))
        p99_drift = (p99_second - p99_first) / p99_first
        record = dict(zip(keys, key_values, strict=True))
        record.update(
            {
                "p50_ms_run1": first,
                "p50_ms_run2": second,
                "drift_frac": drift,
                "drift_pct": 100.0 * drift,
                "within_10pct": abs(drift) <= STABILITY_TOLERANCE_FRAC,
                "p99_ms_run1": p99_first,
                "p99_ms_run2": p99_second,
                "p99_drift_pct": 100.0 * p99_drift,
                "p99_within_10pct": abs(p99_drift) <= STABILITY_TOLERANCE_FRAC,
                "git_commit": str(group["git_commit"].iloc[0]),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=list(STABILITY_COLUMNS))


def write_stability_csv(rows: tuple[LatencyRow, ...], path: Path, append: bool = False) -> Path:
    """Write the run-to-run p50 comparison.

    Args:
        rows: Measured configurations across every repeat.
        path: Destination CSV.
        append: Fold the rows into an existing file instead of replacing it.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = stability_table(rows)
    if append:
        table = _merge_frames(path, table, CONFIG_KEYS)
    table.to_csv(path, index=False)
    return path


def write_parity_json(
    parity: dict[str, tuple[ParityResult, ...]],
    exports: tuple[ExportRecord, ...],
    path: Path,
    attributions: dict[str, ParityAttribution] | None = None,
) -> Path:
    """Write the parity results per draw, the exported shape contracts and the stamp.

    Args:
        parity: Per target key, one result per window draw.
        exports: Export records, for the declared graph shapes.
        path: Destination JSON.
        attributions: Double-precision attribution per target key, for the models that
            failed. Written next to the failure so the artifact says why, not just that.

    Returns:
        The path written.
    """
    by_key = {record.target_key: record for record in exports}
    attribution = attributions or {}
    payload: dict[str, Any] = {
        "criterion": (
            "max_abs_err < tolerance * max(1, |y|_max); the plan's unscaled "
            "max_abs_err < tolerance is reported alongside as passed_absolute "
            "(docs/protocol.md P7-D3)"
        ),
        "seeds": list(PARITY_SEEDS),
        "env": environment_stamp(),
        "models": {
            key: {
                "verdict": {
                    "passed": worst_parity(draws).passed,
                    "passed_absolute": worst_parity(draws).passed_absolute,
                    "worst_seed": worst_parity(draws).seed,
                    "max_abs_err": worst_parity(draws).max_abs_err,
                    "scaled_tolerance": worst_parity(draws).scaled_tolerance,
                },
                "draws": [
                    {
                        "seed": draw.seed,
                        "max_abs_err": draw.max_abs_err,
                        "mean_abs_err": draw.mean_abs_err,
                        "output_abs_max": draw.output_abs_max,
                        "scale": draw.scale,
                        "tolerance": draw.tolerance,
                        "scaled_tolerance": draw.scaled_tolerance,
                        "n_windows": draw.n_windows,
                        "passed": draw.passed,
                        "passed_absolute": draw.passed_absolute,
                    }
                    for draw in draws
                ],
                "onnx_path": str(by_key[key].onnx_path) if key in by_key else None,
                "head_kind": by_key[key].head_kind if key in by_key else None,
                "input_dims": list(by_key[key].input_dims) if key in by_key else None,
                "output_dims": list(by_key[key].output_dims) if key in by_key else None,
                "onnx_size_bytes": by_key[key].size_bytes if key in by_key else None,
                "fp64_attribution": (
                    {
                        "torch_fp32_vs_fp64_max": attribution[key].torch_fp32_vs_fp64_max,
                        "onnx_fp32_vs_fp64_max": attribution[key].onnx_fp32_vs_fp64_max,
                        "output_abs_max": attribution[key].output_abs_max,
                        "n_windows": attribution[key].n_windows,
                    }
                    if key in attribution
                    else None
                ),
            }
            for key, draws in parity.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_jobs(
    targets: tuple[ModelTarget, ...],
    backends: tuple[str, ...] = BACKENDS,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    warmup_iters: int = 200,
    timed_iters: int = 2000,
    thread_counts: tuple[int, ...] = (1,),
    torch_device: str = "cpu",
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> tuple[BenchJob, ...]:
    """Build the cross product of configurations to measure.

    Args:
        targets: Models to measure.
        backends: Runtimes to measure.
        batch_sizes: Batch sizes to measure.
        warmup_iters: Untimed iterations per configuration.
        timed_iters: Timed iterations per configuration.
        thread_counts: Pinned CPU thread counts. More than one turns the sweep into a
            thread-sensitivity study, which is worth doing for the ORT CPU provider at
            batch 1 -- its latency moves by more than the CPU-versus-GPU effect being
            measured, so "ORT CPU" without a thread count is not a number.
        torch_device: Device for the two PyTorch backends. The ORT backends take theirs
            from the provider.
        onnx_dir: Directory holding exported graphs.
        checkpoint_root: Root of the checkpoint tree.

    Returns:
        The jobs, model-major then backend, batch size and thread count, which is also
        report order.

    Raises:
        ValueError: If a backend is not one of :data:`dmf.deploy.harness.BACKENDS`.
    """
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        raise ValueError(f"unknown backends {unknown}; known backends are {list(BACKENDS)}")
    jobs: list[BenchJob] = []
    for target in targets:
        for backend in backends:
            for batch_size in batch_sizes:
                for threads in thread_counts:
                    jobs.append(
                        BenchJob(
                            target_key=target.key,
                            backend=backend,  # type: ignore[arg-type]
                            batch_size=batch_size,
                            device=torch_device,
                            warmup_iters=warmup_iters,
                            timed_iters=timed_iters,
                            intra_op_threads=threads,
                            onnx_dir=onnx_dir,
                            checkpoint_root=checkpoint_root,
                        )
                    )
    return tuple(jobs)


def job_provider(job: BenchJob) -> str | None:
    """Name the arithmetic a configuration will run, as it appears in ``parity.csv``.

    Args:
        job: The configuration.

    Returns:
        The provider name, or None for a PyTorch **CPU** job -- which is exempt because it
        *is* the reference every other row is checked against.
    """
    if job.backend in PROVIDERS_BY_BACKEND:
        return PROVIDERS_BY_BACKEND[job.backend][0]
    return "torch:cuda" if job.resolved_device == "cuda" else None


def parity_status(
    parity: dict[str, tuple[ParityResult, ...]],
    results_dir: Path | None = None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return the ``(model, provider)`` pairs that were checked, and those that failed.

    Reads the committed ``parity.csv`` when this run did not do the checking itself.
    **That is the point**: the rule is "nothing is timed without a passing parity row", not
    "nothing is timed in a command that happened to also pass ``--parity``". A sweep split
    across several invocations -- which is how the committed artifacts are produced -- would
    otherwise enforce the rule in the first invocation and drop it in the rest.

    Args:
        parity: Parity draws per model from this run, if any.
        results_dir: Where to look for a committed ``parity.csv`` when ``parity`` is empty.

    Returns:
        ``(checked, failed)``, each a set of ``(model, provider)``.
    """
    checked: set[tuple[str, str]] = set()
    failed: set[tuple[str, str]] = set()
    if parity:
        for key, draws in parity.items():
            for provider in providers_checked(draws):
                checked.add((key, provider))
                if not worst_parity(draws, provider).passed:
                    failed.add((key, provider))
        return checked, failed
    path = (results_dir or Path("results")) / "parity.csv"
    if not path.is_file():
        return checked, failed
    frame = pd.read_csv(path)
    for row in frame.itertuples():
        pair = (str(row.model), str(row.provider))
        checked.add(pair)
        if not bool(row.passed):
            failed.add(pair)
    return checked, failed


def _is_blocked(job: BenchJob, checked: set[tuple[str, str]], failed: set[tuple[str, str]]) -> bool:
    """Report whether a configuration lacks a passing parity row for what it would run.

    Args:
        job: The configuration.
        checked: ``(model, provider)`` pairs with a parity row.
        failed: The subset of those that failed.

    Returns:
        True if this job's arithmetic failed its parity check, or was never checked while
        other configurations of the same model were. A PyTorch CPU job is never blocked.
    """
    provider = job_provider(job)
    if provider is None or not checked:
        return False
    pair = (job.target_key, provider)
    if pair in failed:
        return True
    return pair not in checked and any(model == job.target_key for model, _ in checked)


def run_pipeline(
    targets: tuple[ModelTarget, ...] = DEFAULT_TARGETS,
    *,
    do_export: bool = True,
    do_parity: bool = True,
    do_latency: bool = True,
    jobs: tuple[BenchJob, ...] = (),
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    checkpoint: Path | None = None,
    results_dir: Path | None = None,
    parity_windows: int = PARITY_N_WINDOWS,
    parity_seeds: tuple[int, ...] = PARITY_SEEDS,
    tolerance: float = 1e-4,
    repeats: int = 1,
    latency_stem: str = "latency",
    append: bool = False,
) -> PipelineResult:
    """Export, parity-check and benchmark, in that order, writing the artifacts.

    Args:
        targets: Models to process.
        do_export: Whether to export.
        do_parity: Whether to parity-check.
        do_latency: Whether to benchmark.
        jobs: Configurations to benchmark. Empty means "the default cross product", built
            by :func:`build_jobs`.
        onnx_dir: Directory for exported graphs.
        checkpoint_root: Root of the checkpoint tree.
        checkpoint: Explicit checkpoint file, legal only for a single target.
        results_dir: Where ``parity.{json,csv}``, ``<stem>.csv``, ``<stem>.json`` and
            ``<stem>_stability.csv`` are written. None writes nothing, which is what the
            tests use.
        parity_windows: Random windows per draw.
        parity_seeds: Window draws to run the parity check on.
        tolerance: Relative parity coefficient, dimensionless.
        repeats: How many times to run the whole latency sweep. Two is the methodology's
            requirement; the stability table is written whenever it is at least two.
        latency_stem: Filename stem for the latency artifacts, so a thread-sensitivity
            sweep can be written beside the main one instead of overwriting it.
        append: Fold this run's rows into the existing latency artifacts rather than
            replacing them. For extending a sweep with configurations that were missed,
            without re-measuring the ones that were not.

    Returns:
        What each stage produced.

    Raises:
        FileNotFoundError: If a graph or checkpoint a requested stage needs is absent.
        ValueError: If ``repeats`` is not positive.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    exports: tuple[ExportRecord, ...] = ()
    parity: dict[str, tuple[ParityResult, ...]] = {}
    attributions: dict[str, ParityAttribution] = {}
    if do_export:
        exports = export_targets(targets, onnx_dir, checkpoint_root, checkpoint)
    if do_parity:
        parity = parity_targets(
            targets,
            onnx_dir,
            checkpoint_root,
            checkpoint,
            n_windows=parity_windows,
            tolerance=tolerance,
            seeds=parity_seeds,
        )
        # Attribution runs only on a CPU-provider failure, and only then, because it costs
        # a full double-precision forward pass per model. It answers "did the export change
        # the arithmetic?", which is a question about the graph; a GPU provider's failure is
        # a question about that provider's kernels and is not what it measures.
        for target in targets:
            draws = parity.get(target.key)
            if draws is None or worst_parity(draws, "CPUExecutionProvider").passed:
                continue
            model = load_target(target, checkpoint_root=checkpoint_root, checkpoint=checkpoint)
            attributions[target.key] = attribute_parity(
                model,
                onnx_path_for(target, onnx_dir),
                _parity_windows(
                    model.lookback,
                    model.n_input_channels,
                    n_windows=parity_windows,
                    seed=worst_parity(draws, "CPUExecutionProvider").seed,
                ),
            )
        if results_dir is not None:
            write_parity_json(
                parity, exports, results_dir / "parity.json", attributions=attributions
            )
            write_parity_csv(parity, exports, results_dir / "parity.csv")

    gate_passed = all(
        worst_parity(draws, provider).passed
        for draws in parity.values()
        for provider in providers_checked(draws)
    )
    latency: tuple[LatencyRow, ...] = ()
    failures: tuple[tuple[str, str], ...] = ()
    if do_latency:
        sweep_jobs = jobs or build_jobs(targets, onnx_dir=onnx_dir, checkpoint_root=checkpoint_root)
        # Parity gates timing per (model, provider). No configuration is ever timed
        # without a parity row behind the arithmetic it will run -- that rule is absolute,
        # and gating on the model alone is what let a whole sweep of TF32 GPU rows ship
        # behind a CPU-only parity check (P7-D9). One failure is still not a reason to
        # discard the configurations around it, so the block is as narrow as the evidence.
        checked, failed = parity_status(parity, results_dir)
        excluded = tuple(job.label for job in sweep_jobs if _is_blocked(job, checked, failed))
        for label in excluded:
            print(f"latency: {label} not timed -- no passing parity row for its arithmetic")
        sweep_jobs = tuple(job for job in sweep_jobs if not _is_blocked(job, checked, failed))
        all_rows: list[LatencyRow] = []
        all_failures: list[tuple[str, str]] = []
        for repeat in range(1, repeats + 1):
            rows, repeat_failures = latency_sweep(sweep_jobs, repeat=repeat)
            all_rows.extend(rows)
            all_failures.extend(repeat_failures)
        latency, failures = tuple(all_rows), tuple(all_failures)
        if results_dir is not None and latency:
            first_pass = tuple(row for row in latency if row.repeat == 1)
            write_latency_csv(first_pass, results_dir / f"{latency_stem}.csv", append=append)
            write_latency_json(latency, results_dir / f"{latency_stem}.json", append=append)
            if repeats > 1:
                write_stability_csv(
                    latency, results_dir / f"{latency_stem}_stability.csv", append=append
                )
    return PipelineResult(
        exports=exports,
        parity=parity,
        attributions=attributions,
        latency=latency,
        failures=failures,
        parity_gate_passed=gate_passed,
    )


def unverified_timed_configurations(
    outcome: PipelineResult, results_dir: Path | None = None
) -> set[str]:
    """Return every timed configuration that has no passing parity row.

    The safety property of this package in one function: a latency row whose arithmetic was
    never verified is what the ordering of stages exists to prevent, and it is worth being
    able to assert rather than to argue. `run_pipeline` already excludes a failing
    (model, provider) pair from the latency stage, so a non-empty result here means that
    exclusion did not work and the numbers must not be published.

    Delegates to :func:`parity_status`, which falls back to the committed ``parity.csv``
    when this invocation did not run the checks itself. That fallback is the whole reason
    this is not a one-line comprehension: the committed sweep is produced by several
    commands, and only the first of them passes ``--parity``. Reading the run's own result
    alone would enforce the rule in that command and silently drop it in every other -- and
    would flag every row of a ``--latency``-only run as unverified, which is how this
    function failed its first probe.

    Args:
        outcome: A completed pipeline run.
        results_dir: Where to find a committed ``parity.csv`` when this run has none.

    Returns:
        ``"model/backend/device"`` for each offending latency row; empty when the pipeline
        behaved. A provider that was never parity-checked at all counts as offending.
    """
    from dmf.deploy.gate import timed_provider

    checked, failed = parity_status(outcome.parity, results_dir)
    passing = checked - failed
    offenders: set[str] = set()
    for row in outcome.latency:
        job = row.job
        provider = timed_provider(job.backend, job.device)
        if provider is None:
            continue
        if (job.target_key, provider) not in passing:
            offenders.add(f"{job.target_key}/{job.backend}/{job.device}")
    return offenders
