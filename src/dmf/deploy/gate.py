"""The Gate 7 read-out.

``docs/IMPLEMENTATION_PLAN.md`` §Phase 7 states the gate as three clauses: **parity passes**;
``results/latency.csv`` **and a Pareto plot exist**; the README **states measured numbers
with methodology, and cites no generic speedup multipliers**.

The first two are file predicates. The third is the interesting one, because "cites no
generic speedup multipliers" is the kind of criterion that is normally checked by a human
reading charitably. It is checked here instead, narrowly enough to mean something: **every
``Nx`` claim in the README's Phase 7 section must be reproducible as a ratio of two p50
values for the same model at the same batch size**, and every ``… ms`` figure must appear
in one of the committed latency tables (``latency.csv`` and the thread sweep beside it).

**The first version of this check was vacuous and is worth recording as such.** It formed
ratios over *all* p50 values in the file -- 5 701 of them from 76 rows -- and accepted
anything within 2 percent of any one. Measured acceptance was 100 percent of values in
[0.5, 50], so a README claiming "NVIDIA reports up to 12x speedup with TensorRT" passed.
Restricting the pool to same-model, same-batch pairs -- the only ratios this project's own
sentences ever make -- cuts it to roughly half that range.

**Roughly half is not a strong check, and the read-out says so rather than implying
otherwise.** :func:`multiplier_acceptance` measures what fraction of plausible multipliers
the current pool would accept and prints it in the gate detail, so the clause's strength is
a reported number instead of an assumption. With seven backends measured per model there
are simply many real ratios, and that is a limit of the method, not something to hide. The
consequence for the write-up is that a multiplier should never be the load-bearing form of
a claim: state both endpoints in milliseconds -- which the stricter half of this clause
checks against the committed tables -- and let the ratio be the summary.

Nothing here measures anything: the read-out is computed from committed artifacts, so it
cannot perturb the numbers it is reading (the same reason ``scripts/gate6.py`` refuses to
run ``make eval`` itself).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dmf.eval.gate import VERDICT_FAIL, VERDICT_PASS, VERDICT_UNVERIFIED

__all__ = [
    "GATE7_CRITERIA",
    "GATE7_METHODOLOGY_TOKENS",
    "README_SECTION_HEADING",
    "Gate7Evidence",
    "build_gate7_markdown",
    "gate7_passes",
    "gate7_readout",
    "timed_provider",
    "multiplier_acceptance",
    "read_gate7_inputs",
    "write_gate7_report",
]

#: The three clauses, in plan order. Ids are strings so the frame's ``criterion`` column
#: matches the Gate 6 convention.
GATE7_CRITERIA: tuple[tuple[str, str], ...] = (
    ("1", "every benchmarked configuration has a passing FP32 parity row"),
    ("2", "results/latency.csv and the Pareto figure exist and are non-empty"),
    ("3", "README states measured numbers with methodology and no generic multipliers"),
)

#: Heading that marks the README's Phase 7 section. Matched case-insensitively on a line
#: beginning with ``#``.
README_SECTION_HEADING: str = "inference latency"

#: Methodology the README section must state, as ``(token, why it matters)``. Each is
#: matched case-insensitively as a substring, so "2000 timed iterations" satisfies
#: ``timed``. They are the facts without which a latency number cannot be interpreted:
#: how long the harness warmed up, how many iterations it timed, that the tail is reported
#: and not only the mean, that GPU regions were synchronised, how many CPU threads were
#: pinned, and that the corpus is simulated.
GATE7_METHODOLOGY_TOKENS: tuple[tuple[str, str], ...] = (
    ("warmup", "an unwarmed measurement times allocation and clock ramp"),
    ("timed", "the iteration count is the sample size"),
    ("p99", "a mean without a tail hides the iteration that misses the deadline"),
    ("synchron", "un-synchronised CUDA timing is the field's most common error"),
    ("thread", "ORT CPU latency swings with thread count, so a row without one is not a number"),
    ("simulated", "CLAUDE.md non-negotiable 1"),
)

#: Tolerance on reproducing a quoted ``Nx`` multiplier from two measured p50 values,
#: dimensionless. Wide enough to absorb the rounding of a figure printed to one or two
#: decimals, narrow enough that a number from somewhere else will not fit.
_MULTIPLIER_TOL: float = 0.02

#: Columns a pair of p50 values must agree on before their ratio counts as a claim this
#: project can make. Comparing two backends is a statement about backends; comparing two
#: models, or two batch sizes, is a statement about something else, and pooling all three
#: made the check accept essentially any number (see the module docstring).
_RATIO_KEYS: tuple[str, ...] = ("model", "batch_size")

#: Tolerance on reproducing a quoted millisecond figure from the latency table,
#: dimensionless (relative). A figure quoted to two decimals is within half a unit of the
#: last place, i.e. well inside this.
_MS_TOL: float = 0.01

#: Columns holding a measured latency, across every committed table. The ``_run1``/``_run2``
#: forms are the stability table's, which is where the second repeat lives.
_LATENCY_COLUMN_RE = re.compile(r"(p50|p90|p99|mean)_ms(_run\d+)?")

_MS_RE = re.compile(r"(\d+\.\d+)\s*ms\b")
_MULTIPLIER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\b", re.IGNORECASE)


@dataclass(frozen=True)
class Gate7Evidence:
    """Everything the read-out is computed from.

    Attributes:
        parity: ``results/parity.csv``, or None if absent.
        latency: ``results/latency.csv``, or None if absent.
        latency_extra: Every other committed latency table -- at present the thread
            sensitivity sweep. Clause 3 pools it with ``latency``: the criterion is that a
            quoted number is *measured and committed*, and a figure from
            ``latency_threads.csv`` is both. Excluding it would have failed the gate for
            quoting the thread sweep, which is the opposite of what the clause is for.
        pareto_exists: Whether the Pareto figure is on disk.
        readme_section: The README's Phase 7 section, or ``""`` if it has none.
        results_dir: Where the artifacts were read from.
    """

    parity: pd.DataFrame | None
    latency: pd.DataFrame | None
    latency_extra: tuple[pd.DataFrame, ...]
    pareto_exists: bool
    readme_section: str
    results_dir: Path


def _readme_section(readme: Path) -> str:
    """Extract the README's Phase 7 section.

    Args:
        readme: Path to ``README.md``.

    Returns:
        The section text from its heading to the next heading of the same or higher level,
        or ``""`` if the README has no such heading.
    """
    if not readme.is_file():
        return ""
    lines = readme.read_text(encoding="utf-8").splitlines()
    start = None
    level = 0
    for index, line in enumerate(lines):
        if line.startswith("#") and README_SECTION_HEADING in line.lower():
            start = index
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        return ""
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("#") and (len(line) - len(line.lstrip("#"))) <= level:
            return "\n".join(lines[start:index])
    return "\n".join(lines[start:])


def read_gate7_inputs(results_dir: Path, readme: Path = Path("README.md")) -> Gate7Evidence:
    """Read the committed artifacts the gate is judged from.

    Args:
        results_dir: Directory holding ``parity.csv``, ``latency.csv`` and the figure.
        readme: Path to the README.

    Returns:
        The evidence. A missing artifact is None or False rather than an exception: an
        absent file is a gate failure to be reported, not a crash.
    """
    parity_path = results_dir / "parity.csv"
    latency_path = results_dir / "latency.csv"
    # Every committed latency table, the stability one included: its p50/p99 columns are
    # the SECOND run's measurements and nothing else records them, so a document quoting a
    # run-to-run range would otherwise be checked against half its own evidence.
    extra = tuple(pd.read_csv(path) for path in sorted(results_dir.glob("latency_*.csv")))
    return Gate7Evidence(
        parity=pd.read_csv(parity_path) if parity_path.is_file() else None,
        latency=pd.read_csv(latency_path) if latency_path.is_file() else None,
        latency_extra=extra,
        pareto_exists=(results_dir / "latency_pareto.png").is_file(),
        readme_section=_readme_section(readme),
        results_dir=results_dir,
    )


def _parity_verdict(evidence: Gate7Evidence) -> tuple[str, str]:
    """Judge clause 1.

    Args:
        evidence: The artifacts.

    Returns:
        ``(verdict, detail)``.
    """
    if evidence.parity is None or evidence.parity.empty:
        return VERDICT_UNVERIFIED, f"{evidence.results_dir / 'parity.csv'} is absent"
    frame = evidence.parity
    passing = {
        (str(row.model), str(row.provider)) for row in frame.itertuples() if bool(row.passed)
    }
    failed_rows = sorted(
        f"{row.model}/{row.provider}" for row in frame.itertuples() if not bool(row.passed)
    )
    unscaled = sorted(
        f"{row.model}/{row.provider}" for row in frame.itertuples() if not bool(row.passed_absolute)
    )
    n_seeds = int(frame["n_seeds"].max())
    detail = (
        f"{len(frame)} (model, provider) pairs checked over {n_seeds} window draws each; "
        f"worst margin {frame['margin_x'].min():.1f}x. Rows that fail the criterion and are "
        f"consequently NOT benchmarked: {failed_rows or 'none'}. Unscaled 1e-4 additionally "
        f"fails for {unscaled or 'nothing'} (P7-D3 records the criterion change)."
    )

    # The clause is about what was *timed*: a failing parity row whose configuration was
    # excluded from the sweep is the gate working, not the gate failing. What must not
    # exist is a latency row whose arithmetic has no passing parity row -- which is exactly
    # the state the TF32 defect shipped in (P7-D9).
    if evidence.latency is None or evidence.latency.empty:
        return VERDICT_UNVERIFIED, detail + " No latency table to cross-check against."
    unverified = sorted(
        {
            f"{row.model}/{row.backend}/{row.device}"
            for row in evidence.latency.itertuples()
            if timed_provider(str(row.backend), str(row.device)) is not None
            and (str(row.model), str(timed_provider(str(row.backend), str(row.device))))
            not in passing
        }
    )
    cpu_only = {model for model, provider in passing if provider == "CPUExecutionProvider"}
    missing_cpu = sorted(set(frame["model"]) - cpu_only)
    detail += f" Benchmarked configurations without a passing parity row: {unverified or 'none'}."
    if unverified or missing_cpu:
        return VERDICT_FAIL, detail + f" Models with no passing CPU-EP row: {missing_cpu}."
    return VERDICT_PASS, detail


def timed_provider(backend: str, device: str) -> str | None:
    """Name the arithmetic one latency row ran, as ``parity.csv`` spells it.

    Args:
        backend: The ``backend`` column.
        device: The ``device`` column.

    Returns:
        The provider name, or None for a PyTorch CPU row, which is the reference every
        other row is checked against and has nothing to be checked against itself.
    """
    if backend.startswith("torch-"):
        return "torch:cuda" if device == "cuda" else None
    return {
        "ort-cpu": "CPUExecutionProvider",
        "ort-cuda": "CUDAExecutionProvider",
        "ort-trt": "TensorrtExecutionProvider",
    }.get(backend)


def _artifact_verdict(evidence: Gate7Evidence) -> tuple[str, str]:
    """Judge clause 2.

    Args:
        evidence: The artifacts.

    Returns:
        ``(verdict, detail)``.
    """
    missing: list[str] = []
    if evidence.latency is None or evidence.latency.empty:
        missing.append("latency.csv")
    if not evidence.pareto_exists:
        missing.append("latency_pareto.png")
    if missing:
        return VERDICT_FAIL, f"absent or empty: {missing}"
    assert evidence.latency is not None
    return (
        VERDICT_PASS,
        f"latency.csv carries {len(evidence.latency)} configurations over "
        f"{evidence.latency['backend'].nunique()} backends and "
        f"{evidence.latency['batch_size'].nunique()} batch sizes; the Pareto figure exists.",
    )


def _readme_verdict(evidence: Gate7Evidence) -> tuple[str, str]:
    """Judge clause 3.

    Args:
        evidence: The artifacts.

    Returns:
        ``(verdict, detail)``. Three sub-checks, all required: the section exists and states
        the methodology tokens; every millisecond figure it quotes is in ``latency.csv``;
        every ``Nx`` multiplier it quotes is a ratio of two measured p50 values.
    """
    section = evidence.readme_section
    if not section:
        return VERDICT_FAIL, f"README has no '#... {README_SECTION_HEADING}' section"
    if evidence.latency is None or evidence.latency.empty:
        return VERDICT_UNVERIFIED, "latency.csv absent, so quoted numbers cannot be checked"

    lowered = section.lower()
    missing_tokens = [token for token, _ in GATE7_METHODOLOGY_TOKENS if token not in lowered]

    frames = [evidence.latency, *evidence.latency_extra]
    measured: set[float] = set()
    for frame in frames:
        for column in frame.columns:
            if _LATENCY_COLUMN_RE.fullmatch(str(column)):
                measured.update(float(value) for value in frame[column])
    quoted_ms = [float(match) for match in _MS_RE.findall(section)]
    unmatched_ms = [
        value
        for value in quoted_ms
        if not any(abs(value - m) <= max(_MS_TOL * m, 0.005) for m in measured)
    ]

    ratios = _same_cell_ratios(frames)
    quoted_x = [float(match) for match in _MULTIPLIER_RE.findall(section)]
    unmatched_x = [
        value
        for value in quoted_x
        if not any(abs(value - r) <= max(_MULTIPLIER_TOL * value, 0.005) for r in ratios)
    ]

    detail = (
        f"{len(quoted_ms)} millisecond figures quoted, {len(unmatched_ms)} not in "
        f"the committed latency tables; {len(quoted_x)} multipliers quoted, "
        f"{len(unmatched_x)} not reproducible as a same-model same-batch p50 ratio "
        f"(pool of {len(ratios)} ratios, which would accept "
        f"{100 * multiplier_acceptance(ratios):.0f} percent of plausible multipliers -- "
        f"the ms check is the stricter half); methodology tokens missing: "
        f"{missing_tokens or 'none'}."
    )
    if missing_tokens or unmatched_ms or unmatched_x:
        return VERDICT_FAIL, detail + f" unmatched ms={unmatched_ms}, unmatched x={unmatched_x}"
    return VERDICT_PASS, detail


def multiplier_acceptance(
    ratios: set[float], low: float = 0.5, high: float = 50.0, n: int = 4000
) -> float:
    """Measure how permissive the multiplier check currently is.

    A check whose acceptance region is most of the plausible range is not evidence, and the
    only way to know which it is is to measure it. Reported in the gate detail so the
    clause's strength ships beside its verdict.

    Args:
        ratios: The accepted pool.
        low: Lower end of the plausible multiplier range, dimensionless.
        high: Upper end.
        n: Grid points.

    Returns:
        Fraction of a uniform grid over ``[low, high]`` that the pool would accept,
        dimensionless.
    """
    if not ratios:
        return 0.0
    step = (high - low) / max(n - 1, 1)
    grid = [low + step * index for index in range(n)]
    accepted = sum(
        any(abs(value - r) <= max(_MULTIPLIER_TOL * value, 0.005) for r in ratios) for value in grid
    )
    return accepted / len(grid)


def _same_cell_ratios(frames: list[pd.DataFrame]) -> set[float]:
    """Build the pool of p50 ratios a claim may be reproduced from.

    Args:
        frames: The committed latency tables.

    Returns:
        Every ratio of two p50 values sharing a model and a batch size -- i.e. every
        backend-versus-backend comparison this project can make. Ratios across models or
        across batch sizes are excluded: they are statements about something other than a
        backend, and pooling them is what made the check vacuous.
    """
    ratios: set[float] = set()
    for frame in frames:
        columns = [c for c in frame.columns if str(c).startswith("p50_ms")]
        if not columns:
            continue
        keys = [key for key in _RATIO_KEYS if key in frame.columns]
        for _, group in frame.groupby(keys) if keys else [((), frame)]:
            for column in columns:
                values = [float(value) for value in group[column]]
                ratios.update(a / b for a in values for b in values if b > 0)
    return ratios


def gate7_readout(evidence: Gate7Evidence) -> pd.DataFrame:
    """Compute the three-clause read-out.

    Args:
        evidence: The committed artifacts.

    Returns:
        One row per clause, with ``criterion``, ``description``, ``verdict`` and ``detail``.
    """
    judges = (_parity_verdict, _artifact_verdict, _readme_verdict)
    rows = []
    for (criterion, description), judge in zip(GATE7_CRITERIA, judges, strict=True):
        verdict, detail = judge(evidence)
        rows.append(
            {
                "criterion": criterion,
                "description": description,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return pd.DataFrame.from_records(rows)


def gate7_passes(readout: pd.DataFrame) -> bool:
    """Report whether every clause passes.

    Args:
        readout: The frame :func:`gate7_readout` returned.

    Returns:
        True only if every row is :data:`dmf.eval.gate.VERDICT_PASS`. An ``UNVERIFIED`` row
        is not a pass.
    """
    return bool(len(readout) == len(GATE7_CRITERIA) and (readout["verdict"] == VERDICT_PASS).all())


def build_gate7_markdown(readout: pd.DataFrame) -> str:
    """Render the read-out.

    Args:
        readout: The frame :func:`gate7_readout` returned.

    Returns:
        The document.
    """
    verdict = "PASS" if gate7_passes(readout) else "NOT PASSED"
    lines = [
        "# Gate 7 read-out",
        "",
        "`docs/IMPLEMENTATION_PLAN.md` §Phase 7: parity passes; `results/latency.csv` and a "
        "Pareto plot exist; the README states measured numbers with methodology and cites no "
        "generic speedup multipliers. Computed from committed artifacts by "
        "`scripts/gate7.py`; nothing here measures anything.",
        "",
        f"**Gate 7: {verdict}**",
        "",
        "| criterion | description | verdict | detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {row.criterion} | {row.description} | {row.verdict} | {row.detail} |"
        for row in readout.itertuples()
    ]
    lines += [
        "",
        "Clause 3 is checked syntactically rather than read charitably: every `Nx` claim in "
        "the README's latency section must be reproducible as a ratio of two p50 values "
        "**for the same model at the same batch size**, and every millisecond figure must "
        "appear in a committed latency table. It cannot tell a speedup from any other "
        "ratio, and a fabricated multiple that coincided with a real one would pass; what "
        "it catches is a multiplier that is not a ratio of two numbers this machine "
        "produced. Its first version pooled every p50 pair in the file and accepted "
        "essentially any value -- see `dmf.deploy.gate`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_gate7_report(
    results_dir: Path,
    out_dir: Path | None = None,
    readme: Path = Path("README.md"),
) -> tuple[pd.DataFrame, Path, Path]:
    """Compute the read-out and write ``gate7.csv`` and ``gate7.md``.

    Args:
        results_dir: Directory holding the artifacts.
        out_dir: Where to write. Defaults to ``results_dir``.
        readme: Path to the README.

    Returns:
        ``(readout, csv_path, markdown_path)``.
    """
    destination = out_dir or results_dir
    destination.mkdir(parents=True, exist_ok=True)
    readout = gate7_readout(read_gate7_inputs(results_dir, readme))
    csv_path = destination / "gate7.csv"
    markdown_path = destination / "gate7.md"
    readout.to_csv(csv_path, index=False)
    markdown_path.write_text(build_gate7_markdown(readout), encoding="utf-8")
    return readout, csv_path, markdown_path
