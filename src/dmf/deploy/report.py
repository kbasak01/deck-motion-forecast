"""Rendering ``results/latency.md`` and the accuracy-versus-latency figure.

A pure function of the committed CSVs. Nothing here measures anything: it reads
``parity.csv``, ``latency.csv``, ``latency_threads.csv`` and ``latency_stability.csv``, and
writes a document in which every table carries the Gate 6 provenance contract --

    ``<!-- dmf-table id=<slug> section=<n> source=<path> csv_rows=<int> rows=<int>
    [select="..."] -->``

-- emitted by :func:`dmf.eval.report.dmf_table_marker`, the same function the Phase 6
document uses, so the two cannot drift. **The absence of a ``select`` field is a claim**:
it says the rendered table is the whole CSV, row for row. Where a table is a subset, the
predicate that produced it is written out, and it is a predicate a reader can run.

Keeping the renderer separate from the harness is what makes "is the document a function of
the CSVs?" answerable in one second: ``scripts/benchmark.py --report`` needs no GPU, no
checkpoints and no corpus, and cannot start a measurement.

Units: latency milliseconds, memory mebibytes, throughput windows per second, skill
dimensionless.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from dmf.eval.report import (
    RenderedTableSpec,
    dmf_table_marker,
    table_source_marker,
    to_markdown,
)
from dmf.train.experiment import GATE_DOF, GATE_HORIZON_SAMPLES
from dmf.viz.pareto import plot_latency_accuracy_pareto

__all__ = [
    "LATENCY_REPORT_NAME",
    "PARETO_FIGURE_NAME",
    "REPORT_SECTION",
    "LoadedCsv",
    "build_latency_report",
    "gate_cell_skill",
    "pareto_frame",
    "write_latency_report",
    "write_pareto_figure",
]

#: Document this module writes, under the results root.
LATENCY_REPORT_NAME: str = "latency.md"

#: Figure this module writes, under the results root.
PARETO_FIGURE_NAME: str = "latency_pareto.png"

#: Section of ``docs/IMPLEMENTATION_PLAN.md`` every table here belongs to.
REPORT_SECTION: str = "7"

#: Columns of ``latency.csv`` shown in the per-batch tables, in display order. The full
#: file carries the environment stamp columns too; they are constant down the sweep and are
#: stated once in the document's header instead of repeated on forty rows.
_LATENCY_DISPLAY: tuple[str, ...] = (
    "model",
    "backend",
    "device",
    "providers_realized",
    "intra_op_threads",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "mean_ms",
    "std_ms",
    "throughput_windows_s",
    "peak_host_mem_mb",
    "peak_device_mem_mb",
)

#: Columns of ``parity.csv`` shown. Both verdicts are here on purpose: ``passed`` is the
#: criterion in force and ``passed_absolute`` the implementation plan's original unscaled
#: 1e-4 (``docs/protocol.md`` P7-D3).
_PARITY_DISPLAY: tuple[str, ...] = (
    "model",
    "head_kind",
    "n_seeds",
    "worst_seed",
    "max_abs_err",
    "max_abs_err_best_seed",
    "mean_abs_err",
    "output_abs_max",
    "scaled_tolerance",
    "margin_x",
    "passed",
    "passed_absolute",
)

#: Columns of ``latency_stability.csv`` shown.
_STABILITY_DISPLAY: tuple[str, ...] = (
    "model",
    "backend",
    "device",
    "batch_size",
    "p50_ms_run1",
    "p50_ms_run2",
    "drift_pct",
    "within_10pct",
    "p99_ms_run1",
    "p99_ms_run2",
    "p99_drift_pct",
    "p99_within_10pct",
)


@dataclass(frozen=True)
class LoadedCsv:
    """One source CSV and the row count its provenance marker declares.

    Attributes:
        frame: The parsed table.
        rows: Data rows in the file, header excluded. Declared in the marker so that a
            table rendered from a stale or truncated file is visible without opening it.
        display: Path as the marker names it, relative to the results root.
    """

    frame: pd.DataFrame
    rows: int
    display: str


def _load(results_dir: Path, name: str) -> LoadedCsv:
    """Read one source CSV.

    Args:
        results_dir: The results root.
        name: Filename under it.

    Returns:
        The loaded source.

    Raises:
        FileNotFoundError: If the file is absent. The document is not rendered from a
            partial set: a missing table would read as "not measured".
    """
    path = results_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent; run `scripts/benchmark.py --export --parity --latency` first"
        )
    frame = pd.read_csv(path)
    return LoadedCsv(frame=frame, rows=len(frame), display=name)


def _block(
    table_id: str,
    source: LoadedCsv,
    display: pd.DataFrame,
    select: str = "",
    float_fmt: str = "{:.4f}",
) -> list[str]:
    """Render one table under both provenance markers.

    The machine-readable marker is emitted last, immediately above the table's header row,
    because the Gate read-out associates a table with the last non-blank line before it.

    Args:
        table_id: Stable slug for this table.
        source: The CSV it was read from.
        display: The frame to render.
        select: Predicate describing how the rendered rows were derived. Empty only when
            the table is the whole file, row for row.
        float_fmt: Float formatting.

    Returns:
        The Markdown lines.
    """
    spec = RenderedTableSpec(
        table_id=table_id,
        section=REPORT_SECTION,
        source=source.display,
        csv_rows=source.rows,
        rendered_rows=len(display),
        select=select,
    )
    return [
        table_source_marker(spec.table_id, spec.source, spec.csv_rows, spec.rendered_rows),
        "",
        dmf_table_marker(spec),
        to_markdown(display, float_fmt=float_fmt),
        "",
    ]


def gate_cell_skill(metrics_csv: Path) -> pd.DataFrame:
    """Read each model's skill score at the pre-registered gate cell.

    The cell is ``docs/protocol.md`` P3-D12's: **pitch at 10 s on the ``id`` regime**, which
    is the horizon a landing decision is taken at and the DOF that binds. It is not the cell
    at which the deep models look best -- roll at 3 s saturates near 0.999 for everything --
    and using the pre-registered one rather than the flattering one is the point.

    Args:
        metrics_csv: ``results/e04/metrics_full.csv``, per seed.

    Returns:
        One row per model with ``skill_mean``, ``skill_std`` and ``n_seeds``. The mean is
        over the three training seeds, per ``CLAUDE.md`` non-negotiable 5 -- **which is not
        the same population as the latency rows**, since those are measured on the seed-0
        graph alone. Latency is a property of the architecture and the runtime; accuracy is
        not, and the two columns of the Pareto plot therefore come from different
        populations by design. Stated here because it is invisible in the figure.

    Raises:
        FileNotFoundError: If the metrics file is absent.
        ValueError: If the gate cell is empty, which means the file is not what it claims.
    """
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"{metrics_csv} is absent; run `make eval` first")
    frame = pd.read_csv(metrics_csv)
    cell = frame[
        (frame["regime"] == "id")
        & (frame["dof"] == GATE_DOF)
        & (frame["horizon_samples"] == GATE_HORIZON_SAMPLES)
    ]
    if cell.empty:
        raise ValueError(
            f"{metrics_csv} carries no rows for the gate cell "
            f"(regime=id, dof={GATE_DOF}, horizon_samples={GATE_HORIZON_SAMPLES})"
        )
    grouped = cell.groupby("model")["skill"].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["model", "skill_mean", "skill_std", "n_seeds"]
    return grouped


def pareto_frame(
    results_dir: Path,
    metrics_csv: Path,
    batch_size: int = 1,
) -> pd.DataFrame:
    """Join measured latency to gate-cell accuracy, one row per (model, backend).

    Args:
        results_dir: The results root, holding ``latency.csv``.
        metrics_csv: ``results/e04/metrics_full.csv``.
        batch_size: Which batch size to plot. 1 is the operational case.

    Returns:
        Columns ``label``, ``model``, ``backend``, ``p50_ms``, ``skill_mean``,
        ``skill_std``, ``n_seeds``.

    Raises:
        FileNotFoundError: If a source is absent.
        ValueError: If no latency row survives the batch-size filter, or if a benchmarked
            model has no accuracy row -- a silently dropped join is how a Pareto plot comes
            to show three of four models with nothing saying so.
    """
    latency = _load(results_dir, "latency.csv").frame
    selected = latency[latency["batch_size"] == batch_size]
    if selected.empty:
        raise ValueError(f"latency.csv carries no rows at batch_size={batch_size}")
    skill = gate_cell_skill(metrics_csv)
    joined = selected.merge(skill, on="model", how="left")
    missing = sorted(set(joined.loc[joined["skill_mean"].isna(), "model"]))
    if missing:
        raise ValueError(
            f"no gate-cell accuracy for benchmarked models {missing}; a Pareto plot with a "
            f"dropped join shows fewer models than were measured and says nothing about it"
        )
    # The device is folded into the backend name rather than left as a separate column:
    # `torch-eager` was measured on both, and a legend that coloured the two the same would
    # put a CPU point and a CUDA point in one series -- the same misreading the `device`
    # column exists to prevent in the tables.
    joined["backend"] = joined["backend"] + "/" + joined["device"]
    joined["label"] = joined["model"] + "/" + joined["backend"]
    return joined[
        ["label", "model", "backend", "p50_ms", "skill_mean", "skill_std", "n_seeds"]
    ].reset_index(drop=True)


def write_pareto_figure(
    results_dir: Path,
    metrics_csv: Path,
    path: Path | None = None,
    batch_size: int = 1,
) -> Path:
    """Write the accuracy-versus-latency figure.

    Args:
        results_dir: The results root.
        metrics_csv: ``results/e04/metrics_full.csv``.
        path: Destination PNG. Defaults to ``<results_dir>/latency_pareto.png``.
        batch_size: Which batch size to plot.

    Returns:
        The path written.
    """
    frame = pareto_frame(results_dir, metrics_csv, batch_size=batch_size)
    figure: Figure = plot_latency_accuracy_pareto(frame)
    figure.text(
        0.01,
        0.005,
        (
            f"Simulated corpus. Accuracy: skill vs persistence at the pre-registered gate "
            f"cell ({GATE_DOF} @ {GATE_HORIZON_SAMPLES / 10:.0f} s, id), mean of 3 seeds. "
            f"The 1-5 s operational ranking differs (docs/protocol.md P4-D14). "
            f"Latency: p50 at batch {batch_size}, 1 intra-op thread, seed-0 graph, "
            f"one machine (RTX A4000 / i9-10980XE)."
        ),
        ha="left",
        va="bottom",
        fontsize=6.5,
        wrap=True,
    )
    figure.subplots_adjust(bottom=0.20)
    destination = path or results_dir / PARETO_FIGURE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    return destination


def _fastest(
    frame: pd.DataFrame,
    model: str,
    backend: str,
    column: str = "p50_ms",
    device: str | None = None,
) -> float:
    """Return one cell of the latency table.

    **The selection must be unique.** Since the PyTorch backends are measured on both
    devices, ``(model, backend)`` no longer identifies a row, and a lookup that quietly took
    the first match put a CPU number into a sentence about CUDA in the first draft of this
    document -- the exact misreading the ``device`` column exists to prevent. An ambiguous
    selection therefore raises rather than picking.

    Args:
        frame: A latency frame already filtered to one batch size.
        model: Model key.
        backend: Backend key.
        column: Column to read.
        device: ``"cpu"`` or ``"cuda"``, required wherever the backend runs on both.

    Returns:
        The value, or NaN if that configuration is not in the frame.

    Raises:
        ValueError: If more than one row matches.
    """
    rows = frame[(frame["model"] == model) & (frame["backend"] == backend)]
    if device is not None:
        rows = rows[rows["device"] == device]
    if len(rows) > 1:
        raise ValueError(
            f"({model}, {backend}, device={device}) matches {len(rows)} rows on devices "
            f"{sorted(set(rows['device']))}; name the device"
        )
    return float(rows[column].iloc[0]) if len(rows) else float("nan")


def _gpu_rows_beating_cpu(batch1: pd.DataFrame, parity_frame: pd.DataFrame) -> str:
    """Say which parity-passing GPU configurations beat the ORT CPU provider, if any.

    Counted rather than asserted. The sentence this replaces read "every GPU configuration
    that passes the parity bar is slower than the ORT CPU provider at batch 1 on both
    recurrent models", which was true of the sweep it was written against and false of the
    one it shipped with: `lstm` on eager PyTorch-CUDA passes parity at 1.905x margin and is
    1.25x faster than ORT CPU, in both repeats. The refusal of `lstm_quantile` on the same
    backend (P7-D10) does not generalise to `lstm`, and writing as though it did turned one
    measured exclusion into a claim about every GPU path.

    Args:
        batch1: Latency rows at batch 1.
        parity_frame: The parity table, for each row's verdict.

    Returns:
        One or two sentences naming the configurations that beat ORT CPU, or saying that
        none does.
    """
    passing = {
        (str(row.model), str(row.provider)) for row in parity_frame.itertuples() if bool(row.passed)
    }
    beats: list[str] = []
    for model in sorted(set(batch1["model"])):
        rows = batch1[batch1["model"] == model]
        cpu = rows[rows["backend"] == "ort-cpu"]
        if cpu.empty:
            continue
        cpu_p50 = float(str(cpu["p50_ms"].iloc[0]))
        for row in rows.itertuples():
            if str(row.device) != "cuda":
                continue
            provider = (
                "torch:cuda"
                if str(row.backend).startswith("torch-")
                else {
                    "ort-cuda": "CUDAExecutionProvider",
                    "ort-trt": "TensorrtExecutionProvider",
                }.get(str(row.backend), "")
            )
            if (model, provider) not in passing:
                continue
            if float(str(row.p50_ms)) < cpu_p50:
                beats.append(
                    f"`{model}/{row.backend}` at {float(str(row.p50_ms)):.3f} ms against "
                    f"{cpu_p50:.3f}"
                )
    if not beats:
        return (
            "**Every GPU configuration that passes the parity bar is slower than the ORT CPU "
            "provider at batch 1, on every model measured.**"
        )
    joined = f"{'; '.join(beats[:-1])}; and {beats[-1]}" if len(beats) > 1 else beats[0]
    return (
        f"**It does not, however, leave the CPU ahead of every verified GPU row**: {joined}. "
        "Those pass parity and are not excluded by anything; the withdrawal above is one "
        "configuration of one model, not a statement about the GPU."
    )


def _closing_paragraph(lq_torch_cuda: float | None, lq_cpu: float, survivors: str) -> str:
    """State where accuracy and the CPU argument disagree, if they still do.

    An earlier draft asserted that `lstm_quantile` -- the most accurate model at the gate
    cell -- had its fastest and lowest-tail configuration on the GPU, and framed the CPU
    choice as a trade against it. That configuration was PyTorch eager on CUDA, and P7-D10
    withdrew it: cuDNN's fused recurrence is what made it fast and is also why it misses the
    parity bar by 5.2x, so it is refused rather than timed. The paragraph is written from
    whether that row exists, so it cannot outlive the measurement it described.

    Args:
        lq_torch_cuda: `lstm_quantile` eager-CUDA p50 in ms, or None if it was refused.
        survivors: Sentence from :func:`_gpu_rows_beating_cpu`, naming the parity-passing
            GPU rows that still beat ORT CPU.
        lq_cpu: `lstm_quantile` ORT CPU p50, milliseconds.

    Returns:
        One paragraph.
    """
    if lq_torch_cuda is not None:
        return (
            "**The case is weakest exactly where the accuracy is best.** `lstm_quantile` is "
            "the most accurate model at the gate cell, and its fastest configuration is on "
            f"the GPU: a deployment wanting that model *and* the CPU pays about {lq_cpu:.2f} "
            f"ms p50 instead of {lq_torch_cuda:.2f} ms. Still inside the budget, but a trade "
            "rather than a free choice."
        )
    return (
        "**The place this conclusion used to be weakest has closed, and it is worth being "
        "careful rather than pleased about that.** `lstm_quantile` is the most accurate model "
        "at the gate cell, and an earlier draft of this document reported that its fastest "
        "and lowest-tail configuration was on the GPU -- eager PyTorch, at roughly half the "
        f"ORT CPU provider's {lq_cpu:.2f} ms -- and framed the CPU choice as a trade against "
        "accuracy. Per-provider parity withdrew that row: cuDNN's fused LSTM is what made it "
        "fast and is also why it misses this project's tolerance by 5.2x, and the same model "
        "with cuDNN disabled runs at 32.3 ms, sixteen times the CPU provider "
        "(`docs/protocol.md` P7-D10). That row was removed by a control this phase added "
        "late, and a study that had not added it would have published the opposite -- which "
        "is a reason to hold what remains more carefully, not less. " + survivors
    )


def _cpu_vs_cuda_sentence(models: list[str], ratio: "Callable[[str, str, str], float]") -> str:
    """State how the ORT CPU provider compares to the ORT CUDA provider, from the ratios.

    Written from the numbers because the first version was not. It asserted that the CPU
    provider "beats the CUDA provider at batch 1 on every model measured" and then printed
    `0.88x` for `tcn_quantile` in the same sentence -- a ratio below 1 being, by its own
    definition two clauses earlier, the CPU losing. The claim had been true of the sweep it
    was written against and was not re-derived when TF32 was disabled and the sweep re-run
    (docs/protocol.md P7-D9). Counting the wins is what stops the sentence outliving the
    measurement.

    Args:
        models: Model keys, in report order.
        ratio: ``(model, slower, faster) -> float`` over batch-1 p50.

    Returns:
        One bolded sentence, with every model's ratio and the exceptions named.
    """
    wins = [m for m in models if ratio(m, "ort-cuda", "ort-cpu") > 1.0]
    losses = [m for m in models if m not in wins]
    figures = ", ".join(f"{ratio(m, 'ort-cuda', 'ort-cpu'):.2f}x on `{m}`" for m in models)
    if not losses:
        return (
            "**Within ONNX Runtime, the CPU provider beats the CUDA provider at batch 1 on "
            f"every model measured**: {figures}."
        )
    if not wins:
        return (
            "**Within ONNX Runtime, the CPU provider does not beat the CUDA provider at "
            f"batch 1 on any model**: {figures} (a ratio below 1 is the CPU losing)."
        )
    named = ", ".join(f"`{m}`" for m in losses)
    return (
        "**Within ONNX Runtime, the CPU provider beats the CUDA provider at batch 1 on "
        f"{len(wins)} of the {len(models)} models**, and loses on {named}: {figures} -- a "
        "ratio below 1 is the CPU losing. The recurrent models carry the effect; the "
        "convolutional ones are close enough that the sign is not the interesting part of "
        "the number."
    )


def _frontier_sentence(pareto: pd.DataFrame) -> str:
    """Name the p50 Pareto frontier's members, and say whether it is all-GPU.

    Derived because the hard-coded version outlived two re-runs: it named
    `lstm_quantile/torch-eager/cuda`, a row that per-provider parity later refused
    (P7-D10), and called the frontier "entirely GPU" after an ORT CPU configuration had
    joined it. That is the third comparative sentence in this module found asserting a
    membership it no longer had (P7-D12).

    Args:
        pareto: The plotted frame from :func:`pareto_frame`, carrying `label`, `skill_mean`
            and `p50_ms`. The frontier is recomputed here from the same function the figure
            uses, so the text and the picture cannot disagree.

    Returns:
        One paragraph.
    """
    from dmf.viz.pareto import pareto_frontier

    frontier = pareto_frontier(pareto, "skill_mean", "p50_ms")
    members = pareto[frontier].sort_values("p50_ms")
    labels = [str(row.label) for row in members.itertuples()]
    named = (
        f"{', '.join(f'`{x}`' for x in labels[:-1])} and `{labels[-1]}`"
        if len(labels) > 1
        else f"`{labels[0]}`"
    )
    cpu_members = [x for x in labels if "/cpu" in x]
    if cpu_members:
        character = (
            "it is NOT all-GPU: "
            + ", ".join(f"`{x}`" for x in cpu_members)
            + (" is on it" if len(cpu_members) == 1 else " are on it")
            + ", holding the accurate end"
        )
    else:
        character = "on this corpus it is entirely GPU, and ORT CPU is dominated on median"
    return (
        "**Read the frontier with its axis in mind: it is a p50 frontier, and "
        f"{character}** -- {named}. Median is not the statistic the control loop is "
        "designed against, and the CPU configurations are measured while using none of the "
        "GPU that perception needs; a p99 frontier drawn from the same file ranks them "
        "differently. One figure cannot carry both, so the ranking here is the median one "
        "and the tail argument stays in the text above. **The frontier's composition depends "
        "on a refusal**: `lstm_quantile` on PyTorch-CUDA ran at roughly 1.2 ms with the same "
        "skill and would have dominated the accurate end; it is absent because it fails "
        "parity (`docs/protocol.md` P7-D10), not because it was measured and lost."
    )


def _instability_paragraph(frame: pd.DataFrame, batch1: pd.DataFrame) -> str:
    """Say where the run-to-run instability is, branching on which statistic actually fails.

    The version this replaces was written when p50 carried the failures and was not
    re-derived when the corrected sweep moved them all to p99. With the p50 set empty it
    printed "It does not: 0 of 52 configurations exceeded it on p50" -- contradicting its own
    number -- then "the worst being nothing  at batch 0 (0.0 percent)", the empty-frame
    fallbacks leaking into published prose, then a bolded universal quantifier over the empty
    set. Four false sentences from one stale assumption, which is why this is computed.

    Args:
        frame: The full stability table, both statistics, all batch sizes.
        batch1: The same table restricted to batch 1.

    Returns:
        One paragraph.
    """
    p50_fail = frame[~frame["within_10pct"]]
    p99_fail = frame[~frame["p99_within_10pct"]]
    b1_p99_fail = batch1[~batch1["p99_within_10pct"]]

    def _worst(rows: pd.DataFrame, column: str) -> str:
        if rows.empty:
            return "none"
        index = rows[column].abs().idxmax()
        return (
            f"`{rows.at[index, 'model']}/{rows.at[index, 'backend']}/"
            f"{rows.at[index, 'device']}` at batch "
            f"{int(str(rows.at[index, 'batch_size']))} "
            f"({abs(float(str(rows.at[index, column]))):.1f} percent)"
        )

    if p50_fail.empty:
        head = (
            "**The two statistics disagree about whether this sweep is stable, and the one "
            f"that fails is the one the argument uses.** On p50 every configuration holds: "
            f"{len(frame)} of {len(frame)} agree within 10 percent across the two runs, so "
            "the medians quoted above are reproducible. On **p99** that is not true: "
            f"{len(p99_fail)} of {len(frame)} drift further, {len(b1_p99_fail)} of them at "
            f"batch 1, the worst being {_worst(p99_fail, 'p99_drift_pct')}. "
        )
    else:
        head = (
            f"**Instability, stated as a fact and not as an excuse.** {len(p50_fail)} of "
            f"{len(frame)} configurations drift more than 10 percent on p50, the worst "
            f"{_worst(p50_fail, 'drift_pct')}; on p99, {len(p99_fail)} of {len(frame)} do, "
            f"the worst {_worst(p99_fail, 'p99_drift_pct')}. "
        )
    by_backend = p99_fail.groupby("backend").size().sort_values(ascending=False)
    where = ", ".join(f"{count} `{backend}`" for backend, count in by_backend.items())
    return head + (
        f"The tail failures fall as {where}. That is why every tail claim above is "
        "restricted to what holds in **both** runs rather than read off run 1, and why the "
        "one comparison whose margin is smaller than its comparator's drift is written as a "
        "tie rather than an ordering."
    )


def _threshold_reach(parity_frame: pd.DataFrame) -> str:
    """Say which rows the scale-relative criterion decides, and what turns on them.

    Counted, because the asserted version said "the change decides exactly one row of four"
    and named only `lstm_quantile`. With parity moved to every benchmarked provider (P7-D9)
    the change decides four rows across two models, and one of them -- `lstm` on torch:cuda
    -- goes on to hold the best batch-1 median for its model and a seat on the Pareto
    frontier. A threshold change that selects a frontier member is not a footnote about one
    row.

    Args:
        parity_frame: The parity table, carrying `passed` and `passed_absolute`.

    Returns:
        One or two sentences.
    """
    decided = parity_frame[parity_frame["passed"] & ~parity_frame["passed_absolute"]]
    if decided.empty:
        return (
            "**The change decides nothing here**: every row passes the plan's original "
            "unscaled 1e-4 as well."
        )
    names = sorted({f"`{row.model}/{row.provider}`" for row in decided.itertuples()})
    models = sorted({str(row.model) for row in decided.itertuples()})
    joined = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
    plural = "s" if len(models) > 1 else ""
    return (
        f"**The change decides {len(decided)} of the {len(parity_frame)} (model, provider) "
        f"rows**, across {len(models)} model{plural}: {joined} pass the scale-relative "
        "criterion and fail the unscaled one. That matters beyond a count, because "
        "`lstm/torch:cuda` is among them and it holds the best batch-1 median for its model "
        "and a place on the Pareto frontier below -- under the plan's original criterion it "
        "would have been refused and the frontier would have a different member. PyTorch's "
        "own FP32 output misses the unscaled bar against the same model in FP64, which is "
        "what rules out an export defect and is the measurement the change rests on."
    )


def _torch_cuda_paragraph(batch1: pd.DataFrame, parity_frame: pd.DataFrame) -> str:
    """Describe the PyTorch-on-CUDA rows, including the ones that were refused.

    Written from the tables rather than as prose because the membership of this set is a
    measurement: a model whose PyTorch CUDA path fails its parity check is **not timed**, so
    a sentence naming it would describe a row that is not in the file. Saying which models
    are absent, and why, is the CLAUDE.md non-negotiable-6 half of the same fact.

    Args:
        batch1: Latency rows at batch 1.
        parity_frame: The parity table, for the ``torch:cuda`` verdicts.

    Returns:
        One paragraph.
    """
    torch_cuda = batch1[(batch1["backend"] == "torch-eager") & (batch1["device"] == "cuda")]
    refused = sorted(
        str(row.model)
        for row in parity_frame.itertuples()
        if str(row.provider) == "torch:cuda" and not bool(row.passed)
    )
    parts: list[str] = []
    for model in sorted(set(torch_cuda["model"])):
        rows = torch_cuda[torch_cuda["model"] == model].reset_index(drop=True)
        cpu = batch1[(batch1["model"] == model) & (batch1["backend"] == "ort-cpu")]
        cuda = batch1[(batch1["model"] == model) & (batch1["backend"] == "ort-cuda")]
        if rows.empty or cpu.empty or cuda.empty:
            continue
        parts.append(
            f"`{model}` {float(rows.at[0, 'p50_ms']):.3f} ms, against ORT CUDA's "
            f"{float(cuda['p50_ms'].iloc[0]):.3f} and ORT CPU's "
            f"{float(cpu['p50_ms'].iloc[0]):.3f}"
        )
    text = (
        "Eager PyTorch on CUDA at batch 1: " + "; ".join(parts) + ". Where it is fast on the "
        "recurrent models it is cuDNN's fused LSTM kernel doing it -- one launch for the "
        "whole 200-step recurrence, where ORT's CUDA LSTM on this build issues many -- and "
        "on the convolutional graphs it is the slowest GPU option, because ~150 unfused "
        "kernel launches per forward pass is exactly the cost ORT's graph execution removes."
        if parts
        else "No PyTorch CUDA row survived the parity check, so none is timed."
    )
    if refused:
        text += (
            f" **{', '.join(f'`{m}`' for m in refused)} is absent from every PyTorch CUDA "
            "row above: its GPU forward pass does not reproduce the CPU one within this "
            "project's tolerance, so it was refused rather than timed** (`parity.csv`, "
            "`docs/protocol.md` P7-D10). That is the model whose PyTorch CUDA number "
            "previously qualified the CPU conclusion; the qualification is withdrawn, and "
            "withdrawing it makes the CPU case stronger, which is a reason to be careful "
            "about it rather than pleased with it."
        )
    return text


def _argmin_label(
    stability_b1: pd.DataFrame, model: str, column: str, ort_only: bool = False
) -> str:
    """Name the configuration with the smallest value of one column, for one model.

    Args:
        stability_b1: Stability rows restricted to batch 1, carrying both runs.
        model: Model key.
        column: Column to minimise, e.g. ``p99_ms_run2``.
        ort_only: Restrict to the ONNX Runtime backends.

    Returns:
        ``backend/device value`` for the winning row, or ``"n/a"`` if there is none.
    """
    rows = stability_b1[stability_b1["model"] == model]
    if ort_only:
        rows = rows[rows["backend"].str.startswith("ort")]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return "n/a"
    index = int(rows[column].idxmin())
    value = float(str(rows.at[index, column]))
    return f"{rows.at[index, 'backend']}/{rows.at[index, 'device']} {value:.3f} ms"


def _agree_count(
    stability_b1: pd.DataFrame,
    models: list[str],
    ort_only: bool = False,
    want: str = "ort-cpu",
) -> int:
    """Count the models where one backend has the best p99 in **both** runs.

    A superlative that holds in one run and reverses in the other is not a finding, and the
    only way to know which it is is to check both. This is what restricts the tail claim,
    and it is computed rather than asserted so it cannot go stale against the table beside
    it.

    Args:
        stability_b1: Stability rows restricted to batch 1.
        models: Models to check.
        ort_only: Restrict the field to the ONNX Runtime backends.
        want: The backend the claim is about.

    Returns:
        Number of models where ``want`` wins the tail in both repeats.
    """
    return sum(
        _argmin_label(stability_b1, model, "p99_ms_run1", ort_only).startswith(want)
        and _argmin_label(stability_b1, model, "p99_ms_run2", ort_only).startswith(want)
        for model in models
    )


def build_latency_report(results_dir: Path, metrics_csv: Path | None = None) -> str:
    """Render ``latency.md`` from the committed CSVs.

    Args:
        results_dir: The results root, holding ``parity.csv``, ``latency.csv``,
            ``latency_threads.csv`` and ``latency_stability.csv``.
        metrics_csv: ``results/e04/metrics_full.csv``, for the accuracy column of the
            Pareto section. None omits that paragraph rather than inventing it.

    Returns:
        The document.

    Raises:
        FileNotFoundError: If a source CSV is absent.
    """
    parity = _load(results_dir, "parity.csv")
    latency = _load(results_dir, "latency.csv")
    threads = _load(results_dir, "latency_threads.csv")
    stability = _load(results_dir, "latency_stability.csv")

    env_row = latency.frame.iloc[0]
    batch1 = latency.frame[latency.frame["batch_size"] == 1]
    stab1 = stability.frame[stability.frame["batch_size"] == 1]
    batch32 = latency.frame[latency.frame["batch_size"] == 32]
    drifted = stability.frame[~stability.frame["within_10pct"]]

    lines: list[str] = [
        "# Latency and numerical parity -- Phase 7",
        "",
        "**All results in this project are from simulated vessel motion. No real deck data "
        "is used.** These are latency measurements of exported graphs; they say nothing "
        "about accuracy beyond the skill column of the Pareto figure.",
        "",
        "Generated by `scripts/benchmark.py --report` from the CSVs cited above each table. "
        "Every table is reproducible from its source file; the marker above each one names "
        "that file, its row count, and the predicate selecting the rows shown.",
        "",
        "## Environment",
        "",
        f"- CPU: Intel Core i9-10980XE, 18 cores / 36 threads. "
        f"{env_row['intra_op_threads']} intra-op thread is pinned per configuration -- not "
        "the fastest setting available, and the thread sweep below shows what it costs and "
        "on which model it *saves*.",
        f"- GPU: {env_row['gpu']} (workstation card, **standing in for an embedded target** "
        "-- absolute numbers do not transfer to a flight controller).",
        f"- torch {env_row['torch']}, onnxruntime {env_row['onnxruntime']}, opset 18, "
        f"commit `{env_row['git_commit']}`.",
        f"- {env_row['warmup_iters']} warmup then {env_row['timed_iters']} timed iterations "
        "per configuration, each configuration in its own subprocess, "
        "`torch.cuda.synchronize()` around every timed GPU region.",
        "- One iteration is measured **host to host**: NumPy window in, NumPy forecast out, "
        "including the device transfers on the GPU paths. That is what a flight controller "
        "experiences, and it is the same definition for every backend.",
        "- The full environment stamp, per run, is in `latency.json`.",
        "",
        "## Parity",
        "",
        "Run before any timing, **once per execution provider that is benchmarked** -- "
        "including PyTorch on CUDA, whose cuDNN kernels no ONNX parity row touches. A "
        "provider's kernels are part of the computation, and checking only the CPU provider "
        "is how a whole sweep of TF32 GPU rows shipped without anything raising "
        "(`docs/protocol.md` P7-D9). The criterion is `max_abs_err < 1e-4 * max(1, |y|max)`, "
        "read over the worst of five 1000-window draws; the implementation plan's original "
        "unscaled `max_abs_err < 1e-4` is reported beside it as `passed_absolute`. "
        "`docs/protocol.md` P7-D3 records that threshold change and the measurement behind "
        "it. " + _threshold_reach(parity.frame),
        "",
    ]
    lines += _block(
        "parity",
        parity,
        parity.frame[list(_PARITY_DISPLAY)],
        float_fmt="{:.3e}",
    )
    lines += [
        "## Latency, batch 1 -- the operational case",
        "",
        "One window per control cycle is what a landing decision actually runs at.",
        "",
    ]
    lines += _block("latency_b1", latency, batch1[list(_LATENCY_DISPLAY)], "batch_size == 1")

    def _ratio(model: str, slow: str, fast: str) -> float:
        """Return how many times slower one backend is than another, on one model."""
        return _fastest(batch1, model, slow) / _fastest(batch1, model, fast)

    def _ratio_range(model: str, slow: str, fast: str) -> str:
        """Render a backend ratio as the range it spans across the two repeats.

        A ratio quoted to two decimals from one run implies a precision the re-run does not
        support -- the same four comparisons move by up to a fifth between repeats. The
        range is the honest form and it is computed from `latency_stability.csv`, so it
        cannot drift from the stability table below.
        """
        values = []
        for column in ("p50_ms_run1", "p50_ms_run2"):
            rows = stab1[stab1["model"] == model]
            slow_row = rows[rows["backend"] == slow]
            fast_row = rows[rows["backend"] == fast]
            if len(slow_row) == 1 and len(fast_row) == 1:
                values.append(float(slow_row[column].iloc[0]) / float(fast_row[column].iloc[0]))
        if not values:
            return "n/a"
        low, high = min(values), max(values)
        return f"{low:.1f}x" if abs(high - low) < 0.05 else f"{low:.1f}-{high:.1f}x"

    def _batch_cost(model: str, backend: str) -> str:
        """Render the extra wall time batch 32 costs over batch 1, as a range over repeats.

        Quoted as a range because the two repeats disagree materially -- run 1 gives about a
        percent and run 2 several times that. The launch-bound conclusion survives either;
        the round number does not.
        """
        percents = []
        for column in ("p50_ms_run1", "p50_ms_run2"):
            rows = stab1[(stab1["model"] == model) & (stab1["backend"] == backend)]
            wide = stability.frame[
                (stability.frame["model"] == model)
                & (stability.frame["backend"] == backend)
                & (stability.frame["batch_size"] == 32)
            ]
            if len(rows) == 1 and len(wide) == 1:
                one = float(rows[column].iloc[0])
                many = float(wide[column].iloc[0])
                percents.append(100.0 * (many - one) / one)
        if not percents:
            return "n/a"
        return f"{min(percents):.0f}-{max(percents):.0f} percent"

    def _best(model: str, column: str) -> tuple[str, float]:
        """Return the fastest ``(backend/device, value)`` for one model at batch 1."""
        rows = batch1[batch1["model"] == model].reset_index(drop=True)
        if rows.empty:
            return "n/a", float("nan")
        index = int(rows[column].idxmin())
        return (
            f"{rows.at[index, 'backend']}/{rows.at[index, 'device']}",
            float(str(rows.at[index, column])),
        )

    # Derived from the table, not written down: a hard-coded model list silently drops a
    # model added to `dmf.deploy.targets` and silently keeps one removed from it.
    models = sorted(set(batch1["model"]))

    # `lstm_quantile` on torch:cuda fails its parity check and is therefore never timed
    # (P7-D10), so this lookup must be allowed to come back empty rather than raising. A
    # row that is absent because the configuration was refused is a fact about the study,
    # not a missing value to paper over.
    def _maybe(
        model: str, backend: str, column: str = "p50_ms", device: str = "cuda"
    ) -> float | None:
        """Return one latency cell, or None when the configuration was never timed."""
        rows = batch1[
            (batch1["model"] == model)
            & (batch1["backend"] == backend)
            & (batch1["device"] == device)
        ]
        return None if rows.empty else float(str(rows[column].iloc[0]))

    _lq_torch_cuda = _maybe("lstm_quantile", "torch-eager")

    # Which backend actually holds the best p99 per model, derived rather than asserted.
    # An earlier draft of this section claimed ORT CPU held the best tail of every ORT
    # provider on all four models; TensorRT beats it on `tcn_quantile` by 3.7 percent, and
    # the claim was contradicted two clauses later by the report's own number. Counting the
    # winners from the table is the fix that cannot drift away from the data again.
    _ort_backends = ("ort-cpu", "ort-cuda", "ort-trt")

    def _p99_winner(model: str, pool: tuple[str, ...] | None = None) -> tuple[str, float]:
        """Return the (backend/device, p99_ms) holding the best tail for one model."""
        rows = batch1[batch1["model"] == model]
        if pool is not None:
            rows = rows[rows["backend"].isin(pool)]
        index = rows["p99_ms"].idxmin()
        return (
            f"{rows.at[index, 'backend']}/{rows.at[index, 'device']}",
            float(str(rows.at[index, "p99_ms"])),
        )

    def _names(keys: list[str]) -> str:
        """Render model names as a comma list with a trailing 'and'."""
        marked = [f"`{key}`" for key in keys]
        if len(marked) <= 1:
            return "".join(marked)
        return f"{', '.join(marked[:-1])} and {marked[-1]}"

    def _count(keys: list[str]) -> str:
        """Spell a small count, so the sentence reads as prose."""
        return (
            ("none", "one", "two", "three", "four")[len(keys)] if len(keys) < 5 else str(len(keys))
        )

    _ort_p99_wins = [m for m in models if _p99_winner(m, _ort_backends)[0] == "ort-cpu/cpu"]
    _field_p99_wins = [m for m in models if _p99_winner(m)[0] == "ort-cpu/cpu"]

    def _exception(wins: list[str], pool: tuple[str, ...] | None) -> str:
        """Name every model where ORT CPU does not hold the best tail, with the numbers."""
        losses = [m for m in models if m not in wins]
        if not losses:
            return ""
        parts = []
        for model in losses:
            label, value = _p99_winner(model, pool)
            cpu = _fastest(batch1, model, "ort-cpu", "p99_ms")
            parts.append(f"`{model}`, where {label} is better ({value:.3f} vs {cpu:.3f} ms)")
        joined = f"{'; '.join(parts[:-1])}; and {parts[-1]}" if len(parts) > 1 else parts[0]
        return f"The exception{'s are' if len(parts) > 1 else ' is'} {joined}. "

    _ort_p99_exception = _exception(_ort_p99_wins, _ort_backends)
    _field_p99_exception = _exception(_field_p99_wins, None)
    lines += [
        _cpu_vs_cuda_sentence(models, _ratio) + " At a few "
        "hundred microseconds of compute, launch and transfer overhead dominate and the GPU "
        "never gets to work. The mechanism is visible directly in the table: ORT CUDA on "
        f"`tcn` takes {_fastest(batch1, 'tcn', 'ort-cuda'):.3f} ms at batch 1 and "
        f"{_fastest(batch32, 'tcn', 'ort-cuda'):.3f} ms at batch 32 -- **32 times the work "
        f"for {_batch_cost('tcn', 'ort-cuda')} more wall time across the two repeats**, "
        "which is what launch-bound looks like. It is also positive evidence that the GPU "
        "timings are genuinely synchronised: an un-synchronised timer would not have "
        "produced a batch-32 number that large.",
        "",
        '**But "CPU beats GPU" is too coarse a headline, in two separate ways.**',
        "",
        "*First, the ONNX Runtime GPU result is architecture-dependent.* The TensorRT "
        "provider beats ORT CPU on median for the two convolutional graphs "
        f"({_fastest(batch1, 'tcn', 'ort-trt'):.3f} vs "
        f"{_fastest(batch1, 'tcn', 'ort-cpu'):.3f} ms on `tcn`; "
        f"{_fastest(batch1, 'tcn_quantile', 'ort-trt'):.3f} vs "
        f"{_fastest(batch1, 'tcn_quantile', 'ort-cpu'):.3f} ms on `tcn_quantile`) and loses "
        f"badly on both recurrent ones ({_fastest(batch1, 'lstm', 'ort-trt'):.3f} vs "
        f"{_fastest(batch1, 'lstm', 'ort-cpu'):.3f} ms). A 200-step recurrence is a chain of "
        "small dependent kernels; a dilated convolution stack is not.",
        "",
        "*Second, PyTorch's own CUDA path behaves differently again -- where it could be "
        "verified.* " + _torch_cuda_paragraph(batch1, parity.frame),
        "",
        "**The fastest configuration per model at batch 1, on median and on p99.** Each row "
        "is a minimum over the configurations that were *timed*, and `lstm_quantile` was "
        "chosen from one fewer than the others: its PyTorch-CUDA path is refused, so its "
        "winner is the best of what survived parity rather than the best of the field.",
        "",
        "| model | best p50 | | best p99 | |",
        "|---|---|---|---|---|",
        *[
            f"| `{model}` | {_best(model, 'p50_ms')[0]} | {_best(model, 'p50_ms')[1]:.3f} ms "
            f"| {_best(model, 'p99_ms')[0]} | {_best(model, 'p99_ms')[1]:.3f} ms |"
            for model in models
        ],
        "",
        "**On p99 the answer depends on which run you read, so both are shown.** "
        "`latency_stability.csv` carries each configuration's p99 in both repeats; the "
        "fastest-tail configuration per model, in each:",
        "",
        "| model | best p99, run 1 | best p99, run 2 | best ORT p99, run 1 / run 2 |",
        "|---|---|---|---|",
        *[
            f"| `{model}` | {_argmin_label(stab1, model, 'p99_ms_run1')} "
            f"| {_argmin_label(stab1, model, 'p99_ms_run2')} "
            f"| {_argmin_label(stab1, model, 'p99_ms_run1', ort_only=True)} / "
            f"{_argmin_label(stab1, model, 'p99_ms_run2', ort_only=True)} |"
            for model in models
        ],
        "",
        "**Restricted to what holds in both runs**: ORT CPU has the best tail of any ONNX "
        f"Runtime provider on {_agree_count(stab1, models, ort_only=True)} of "
        f"{len(models)} models, and the best tail of the whole field on "
        f"{_agree_count(stab1, models, ort_only=False)} of {len(models)}. The ORT-internal "
        "statement is the one the deployment argument uses and the one that survives both "
        "runs; the whole-field version does not, because the PyTorch CUDA rows move between "
        "repeats by more than their separation from it. TensorRT buys a better median on "
        "the convolutional graphs and gives it back in the tail.",
        "",
        "## Latency, batch 32 -- the throughput case",
        "",
    ]
    lines += _block("latency_b32", latency, batch32[list(_LATENCY_DISPLAY)], "batch_size == 32")
    lines += [
        "**At batch 32 the GPU wins overwhelmingly**: TensorRT reaches "
        f"{_fastest(batch32, 'tcn', 'ort-trt', 'throughput_windows_s'):.0f} windows/s on "
        f"`tcn` against ORT CPU's "
        f"{_fastest(batch32, 'tcn', 'ort-cpu', 'throughput_windows_s'):.0f}. It does not "
        "change the deployment conclusion, and saying so is not special pleading: **a "
        "landing aircraft forecasts one deck.** Batch 32 is a throughput datapoint, not the "
        "mission, and there is no operational configuration in which 32 independent deck "
        "windows are available to be batched.",
        "",
        "## ORT CPU thread sensitivity, batch 1",
        "",
        "The main sweep pins one intra-op thread, which is the deployment-honest setting for "
        "a processor shared with the rest of an autopilot. It is also not the fastest "
        "setting available, and the difference runs in opposite directions by architecture, "
        "so both are reported.",
        "",
    ]
    lines += _block("latency_threads", threads, threads.frame[list(_LATENCY_DISPLAY)])

    def _thread_row(model: str) -> dict[str, float]:
        """Summarise one model's thread sweep: 1-thread, best, and max-thread p50."""
        rows = threads.frame[threads.frame["model"] == model].sort_values("intra_op_threads")
        indexed = rows.reset_index(drop=True)
        best = int(indexed["p50_ms"].idxmin())
        last = len(indexed) - 1
        return {
            "one": float(str(indexed.at[0, "p50_ms"])),
            "best": float(str(indexed.at[best, "p50_ms"])),
            "best_threads": float(str(indexed.at[best, "intra_op_threads"])),
            "best_p99": float(str(indexed.at[best, "p99_ms"])),
            "most": float(str(indexed.at[last, "p50_ms"])),
            "max_threads": float(str(indexed.at[last, "intra_op_threads"])),
        }

    thread_models = sorted(set(threads.frame["model"]))
    trends = {model: _thread_row(model) for model in thread_models}
    # Which model gains most and which loses most, read off the sweep rather than named in
    # the prose: a hard-coded pair goes stale the first time the model set changes.
    gainer = min(trends, key=lambda m: trends[m]["best"] / trends[m]["one"])
    loser = max(trends, key=lambda m: trends[m]["most"] / trends[m]["one"])
    lines += [
        f"The largest gain is `{gainer}`: p50 {trends[gainer]['one']:.3f} ms at 1 thread "
        f"against {trends[gainer]['best']:.3f} ms at "
        f"{int(trends[gainer]['best_threads'])}, where its **tail** "
        f"({trends[gainer]['best_p99']:.3f} ms p99) also beats the best GPU tail for that "
        f"model in the main sweep -- in **both** repeats, which has to be said because that "
        f"comparator is one of the nine p99 failures and moves 13.5 percent between them "
        f"(0.766 then 0.870 ms). The median half of that comparison is not claimed: the "
        "two differ by a few percent across two measurement sessions whose shared 1-thread "
        "configuration itself differs by a similar amount, so it is inside the noise.",
        "",
        f"The largest loss is `{loser}`, which degrades monotonically -- "
        f"{trends[loser]['one']:.3f} ms at 1 thread to {trends[loser]['most']:.3f} ms at "
        f"{int(trends[loser]['max_threads'])} -- because a 200-step recurrence does not "
        'parallelise and the extra threads only add synchronisation. **An "ORT CPU" '
        "number without its thread count is not a number**, which is why the count is a "
        "column on every row of every table here.",
        "",
        "## Run-to-run stability",
        "",
        "The whole sweep was run twice. The methodology requires p50 to agree within 10 "
        f"percent per configuration. **It does not: {len(drifted)} of "
        f"{len(stability.frame)} configurations exceeded it on p50, and "
        f"{int((~stability.frame['p99_within_10pct']).sum())} of {len(stability.frame)} on "
        "p99** -- the tail is the noisier statistic and is also the one the deployment "
        "argument is made on, so it is tracked here rather than left uncomputed. Reported "
        "rather than re-run until they agreed.",
        "",
    ]
    lines += _block(
        "latency_stability",
        stability,
        stability.frame.reindex(
            stability.frame["drift_pct"].abs().sort_values(ascending=False).index
        )[list(_STABILITY_DISPLAY)],
        float_fmt="{:.3f}",
    )
    lines += [
        _instability_paragraph(stability.frame, stab1),
        "",
        "Section 5.4 of the implementation plan makes sub-10-percent drift a "
        "deployment-validation checkbox. This is recorded as a **partial failure of that "
        "checkbox**, with every failing row named in the table above and in "
        "`docs/protocol.md` P7-D13.",
        "",
    ]
    if metrics_csv is not None and metrics_csv.is_file():
        skill = gate_cell_skill(metrics_csv)
        deployed = skill[skill["model"].isin(sorted(set(latency.frame["model"])))]
        lines += [
            "## Accuracy versus latency",
            "",
            f"`{PARETO_FIGURE_NAME}` plots skill against p50 at batch 1. Accuracy is the "
            f"skill score versus persistence at the pre-registered gate cell ({GATE_DOF} at "
            f"{GATE_HORIZON_SAMPLES / 10:.0f} s, `id` regime, `docs/protocol.md` P3-D12), "
            "**mean over three training seeds**; the 1-5 s operational band ranks the "
            "models differently (P4-D14), so the frontier is not a general statement about "
            "which model is better.",
            "",
            "**Four of the project's eighteen models are here.** The Pareto plot covers "
            "the two deep architectures and their quantile heads, which is what Phase 7 "
            "exports (`dmf.deploy.targets`); it does **not** include the `dlinear` family, "
            "the AR baselines, `persistence` or `window_mean`. That omission runs against "
            "the figure's own argument rather than for it: `dlinear_ols` is a single matrix "
            "multiply, would almost certainly sit at the low-latency end on the CPU, and is "
            "the row most likely to be non-dominated there. It is absent because it was not "
            "exported, not because it was measured and lost.",
            "",
            _frontier_sentence(pareto_frame(results_dir, metrics_csv)),
            "",
            "Note the two axes come from different populations by construction: accuracy is "
            "a three-seed mean, latency is measured on the seed-0 graph alone. Latency is a "
            "property of the architecture and the runtime and does not move with the seed; "
            "accuracy does, which is why it carries three.",
            "",
            "| model | skill @ gate cell (mean of 3 seeds) | std |",
            "|---|---|---|",
            *[
                f"| {row.model} | {row.skill_mean:.4f} | {row.skill_std:.4f} |"
                for row in deployed.itertuples()
            ],
            "",
        ]
    lines += [
        "## What this implies for deployment",
        "",
        "Every configuration measured clears a 10 Hz control cycle by an order of magnitude: "
        f"the slowest batch-1 median in the whole sweep is {batch1['p50_ms'].max():.2f} ms "
        "against a 100 ms budget. **The choice is therefore not about feasibility, it is "
        "about which resource the forecaster spends.**",
        "",
        "On one CPU thread through the ONNX Runtime CPU provider, the convolutional "
        f"forecaster runs at {_fastest(batch1, 'tcn', 'ort-cpu'):.2f} ms p50 and "
        f"{_fastest(batch1, 'tcn', 'ort-cpu', 'p99_ms'):.2f} ms p99 -- not the best tail in "
        "the sweep, which belongs to TensorRT on the quantile graph, but within a few "
        "hundredths of a millisecond of it and reached without a GPU -- while the same graph "
        "on ORT's CUDA provider is "
        f"{_ratio('tcn', 'ort-cuda', 'ort-cpu'):.1f}x slower at batch 1. That is the case "
        "for putting a deck-motion predictor on the flight controller's CPU rather than "
        "contending for the GPU that perception is using, and it rests on the **shape** of "
        "the result -- launch overhead dominating a few hundred microseconds of compute -- "
        "not on these absolute numbers, which were measured on a workstation CPU and an RTX "
        "A4000 and will not transfer to an embedded target.",
        "",
        _closing_paragraph(
            _lq_torch_cuda,
            _fastest(batch1, "lstm_quantile", "ort-cpu"),
            _gpu_rows_beating_cpu(batch1, parity.frame),
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_latency_report(
    results_dir: Path,
    metrics_csv: Path | None = None,
    path: Path | None = None,
) -> Path:
    """Render and write ``latency.md``.

    Args:
        results_dir: The results root.
        metrics_csv: ``results/e04/metrics_full.csv``, or None to omit the accuracy section.
        path: Destination. Defaults to ``<results_dir>/latency.md``.

    Returns:
        The path written.
    """
    destination = path or results_dir / LATENCY_REPORT_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_latency_report(results_dir, metrics_csv), encoding="utf-8")
    return destination
