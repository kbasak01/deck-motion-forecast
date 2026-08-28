"""Results table assembly and rendering.

Every number that appears in ``results/results.md`` or the README traces back to a CSV
written by this module. ``make eval`` regenerates all of it end to end; nothing in
``results/`` is hand-edited.

Model comparisons are reported as **mean +/- std over at least three seeds**, and no model
is dropped for underperforming. If DLinear beats the transformer, both rows appear.

**Deterministic models are exempt from the three-seed rule, and the exemption is visible in
the table.** Persistence, damped persistence and AR are closed-form: the seed cannot enter
their fit, so three identical rows would measure nothing. They emit one row with
``deterministic=True``, ``n_seeds=1`` and ``<metric>_std = NaN``. **NaN, not 0.0**: with one
observation the sample standard deviation is undefined, and ``0.0`` would claim a
measurement that was never made. :func:`aggregate_over_seeds` stays strict and still refuses
any group with fewer than three seeds; :func:`aggregate_results` is the router that sends
stochastic rows through it and passes deterministic rows around it.

``skill_std`` and ``skill_ci_lo``/``skill_ci_hi`` are different quantities and live in
different columns. The first is seed-to-seed spread of the fitting procedure; the second is
a bootstrap over held-out **realizations** (:func:`dmf.eval.runner.bootstrap_skill_ci`) and
is the one that says whether a skill score near a gate threshold is distinguishable from it.

**That second pair is a confidence interval only on a single-seed row.** The bootstrap runs
per run, so aggregating over seeds has to combine finished intervals, and no arithmetic on
finished intervals recovers a calibrated interval for the seed mean -- averaging them, which
is what this module used to do, produces something narrower than any input interval's own
coverage justifies and hides seed disagreement entirely. Multi-seed rows therefore carry the
**envelope** (lowest low, highest high): conservative, contains every seed's interval, and it
widens rather than hides when the seeds disagree. :func:`baselines_caveats` says so next to
every rendered table. A calibrated seed-mean interval would have to pool the bootstrap
resamples across seeds inside :mod:`dmf.eval.runner`, which is a re-run, not an aggregation.
"""

import contextlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from dmf.data.channels import channel_aliases

__all__ = [
    "BASELINES_ARTIFACTS",
    "BASELINES_COLUMNS",
    "BASELINES_GROUP_COLS",
    "BASELINES_METRIC_COLS",
    "aggregate_over_seeds",
    "aggregate_results",
    "baselines_caveats",
    "build_baselines_markdown",
    "build_baselines_table",
    "build_results_report",
    "to_markdown",
    "write_table",
]

#: Minimum seeds for a stochastic model-vs-model comparison (CLAUDE.md non-negotiable 5).
MIN_SEEDS = 3

#: Grouping that identifies one comparison cell in ``results/baselines.csv``.
BASELINES_GROUP_COLS: tuple[str, ...] = ("model", "regime", "dof", "horizon_samples")

#: Metrics aggregated to mean and std over seeds.
BASELINES_METRIC_COLS: tuple[str, ...] = ("rmse", "mae", "skill")

#: Exact column order of ``results/baselines.csv``, the Gate 3 artifact.
BASELINES_COLUMNS: tuple[str, ...] = (
    "model",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_seeds",
    "deterministic",
    "n_windows",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "rmse_persistence",
    "skill_mean",
    "skill_std",
    "skill_ci_lo",
    "skill_ci_hi",
    "n_params",
    "fit_time_s_mean",
)

#: The artifacts a baselines run writes, in the order the provenance line names them, each
#: paired with what it holds. Single source of truth: the ``baselines.md`` provenance line is
#: generated from this rather than written out in prose, so a renamed or added artifact
#: cannot leave the document pointing at a file that no longer exists. The *directory* is
#: supplied by the caller and is deliberately not baked in here -- a run writing to
#: ``results/imu/`` used to emit a provenance line naming ``results/baselines*.csv``, i.e.
#: the other observation mode's files, which the last caveat below explicitly warns against
#: mixing with these numbers.
BASELINES_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("baselines.csv", "aggregated"),
    ("baselines_by_seed.csv", "one row per run, the source of truth"),
    ("baselines_by_cell.csv", "per grid cell"),
    ("baselines_controls.csv", "negative controls"),
)

#: Artifacts written only when the negative controls are run, so the provenance line must
#: not name them otherwise.
_CONTROL_ARTIFACTS: frozenset[str] = frozenset({"baselines_controls.csv"})


def _grouped(count: int) -> str:
    """Render an integer with space-separated thousands, e.g. ``1 234 567``.

    Args:
        count: A non-negative count.

    Returns:
        The digits grouped in threes by spaces, matching the style of the prose these
        numbers are embedded in.
    """
    return f"{count:,}".replace(",", " ")


def _display_dir(results_dir: Path) -> str:
    """Render a results directory as a path prefix for the provenance line.

    Args:
        results_dir: Directory the artifacts were written to.

    Returns:
        A POSIX prefix ending in ``/`` (e.g. ``"results/imu/"``), relative to the current
        working directory when it is below it so that a committed document names a
        repo-relative path, and empty for the current directory itself.
    """
    path = Path(results_dir)
    # Not below the CWD (absolute, or a sibling): name it as given rather than guess.
    with contextlib.suppress(ValueError):
        path = path.relative_to(Path.cwd())
    text = path.as_posix().rstrip("/")
    return "" if text in ("", ".") else f"{text}/"


def _provenance_line(results_dir: Path | None, *, with_controls: bool) -> str:
    """Render the sentence naming the files a rendered document was built from.

    Args:
        results_dir: Directory the artifacts were written to. ``None`` means the caller did
            not say, in which case the files are named by bare filename and located
            relative to the document -- true in any directory, where a guessed
            ``results/`` prefix would be false in all but one.
        with_controls: Whether the controls table was written. Naming a file that was not
            written is the same defect as naming the wrong one.

    Returns:
        One Markdown sentence.
    """
    named = [
        (name, what)
        for name, what in BASELINES_ARTIFACTS
        if with_controls or name not in _CONTROL_ARTIFACTS
    ]
    if results_dir is None:
        where = "the CSVs beside this document"
        prefix = ""
    else:
        where = "the CSVs it was written from"
        prefix = _display_dir(results_dir)
    rendered = [f"`{prefix}{name}` ({what})" for name, what in named]
    joined = (
        rendered[0] if len(rendered) == 1 else ", ".join(rendered[:-1]) + " and " + rendered[-1]
    )
    return f"Every number here traces to {where}: {joined}."


def baselines_caveats(
    table: pd.DataFrame | None = None, *, gate_regime: str = "id"
) -> tuple[str, ...]:
    """Caveats that must travel with the baseline tables.

    Each one is a case where a number is real but means something other than what it looks
    like. Two of them are **measured from** ``table`` rather than written as literals. That
    is not fastidiousness: a window count pinned in this prose survived the P3-D4 task
    revision that changed it (441 984 -> 434 304, P3-D6) and was rendered into every
    committed ``baselines.md``. The table-free wording therefore carries no geometry number
    at all, and the wording used when a table is supplied reads the number off the table it
    is printed beside, so the two cannot disagree.

    Args:
        table: The aggregated table from :func:`build_baselines_table`. When given, the
            window count and the interval caveat are derived from it.
        gate_regime: Regime whose test-partition window count is quoted, matching the
            document's gate cell. A count for another regime would be true but irrelevant.

    Returns:
        Markdown sentences, one caveat each, in rendering order.
    """
    scored = table if table is not None and not table.empty else None
    counts: list[int] = []
    if scored is not None and {"regime", "n_windows"} <= set(scored.columns):
        counts = sorted({int(v) for v in scored.loc[scored["regime"] == gate_regime, "n_windows"]})
    # One distinct count means every row of that regime was scored over one window set, which
    # is what the single-pass runner guarantees. Anything else and quoting a single number
    # would be a claim the table does not support, so the sentence drops the number instead.
    scale = (
        f"The {_grouped(counts[0])} windows of `{gate_regime}/test` reported here are"
        if len(counts) == 1
        else "Test windows are"
    )
    # Unknown means "assume it applies": a caveat printed unnecessarily costs a line, an
    # omitted one costs a misread interval.
    multi_seed = (
        scored is None or "n_seeds" not in scored.columns or bool((scored["n_seeds"] > 1).any())
    )
    caveats = [
        "**These are simulated results.** No real deck data is used anywhere in this project.",
        "**`unseen_heading` pitch sits on the P1-D2 residual floor.** That regime's test set "
        "*is* beam seas, where the pitch heading factor is floored at `eps = 0.05` -- about "
        "26 dB below its maximum -- and is therefore driven by an engineering stand-in for hull "
        "asymmetry rather than by the pitch physics. Its persistence RMSE is correspondingly "
        "tiny and its skill is dominated by that stand-in. Never quote that cell without this "
        "sentence.",
        "**The `id` regime pools 4 headings x 4 sea states x 3 speeds**, and per P1-D2 roll in "
        "head seas is on the same residual floor. A pooled roll skill of 0.85 could be 0.93 at "
        "beam and 0.6 at head, or the reverse; `baselines_by_cell.csv` is what distinguishes "
        "them, and the follow-on decision depends on which it is.",
        "**`n_windows` is a window count, not an independent-sample count.** " + scale + " cut "
        "at a stride far shorter than the lookback, so consecutive windows share almost all of "
        "their input samples and their forecast targets overlap. They come from a much smaller "
        "number of independently simulated realizations -- `baselines_by_seed.csv` carries that "
        "count in `n_realizations` -- which is why `skill_ci_lo`/`skill_ci_hi` bootstrap whole "
        "realizations and never windows.",
    ]
    if multi_seed:
        caveats.append(
            "**`skill_ci_lo`/`skill_ci_hi` is a confidence interval only on a single-seed "
            "row.** The bootstrap over held-out realizations runs per *run*, so a row with "
            "`n_seeds = 1` carries that run's interval unchanged, but a row with "
            "`n_seeds > 1` can only combine finished intervals. It carries their **envelope** "
            "-- lowest low, highest high -- which is conservative, contains every seed's own "
            "interval, and widens when the seeds disagree instead of hiding it. It is *not* a "
            "calibrated interval for the seed-mean skill: that would have to pool the "
            "bootstrap resamples across seeds at scoring time. The per-run intervals are in "
            "`baselines_by_seed.csv`; do not read a multi-seed row's interval as a 95% "
            "statement."
        )
    caveats += [
        "**`fit_time_s_mean` is not apples-to-apples.** It is CPU wall-clock for the "
        "closed-form models and GPU wall-clock *including data loading* for the SGD-fitted "
        "ones. No throughput claim is made from it.",
        "**`imu` observation mode is not a drop-in comparison.** Per P1-D6/P2-D8, `imu` inputs "
        "must be scored against `imu` targets, which changes the persistence denominator; skill "
        "scores across observation modes are therefore not comparable.",
    ]
    return tuple(caveats)


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a results table to CSV.

    Args:
        df: The table to write.
        path: Destination under ``results/``. Parent directories are created if absent.

    Returns:
        The path written.

    Raises:
        ValueError: If ``df`` is empty. An empty CSV in ``results/`` is worse than a missing
            one, because a downstream reader cannot tell it from a table with nothing to
            report.
    """
    if df.empty:
        raise ValueError(f"refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _metric_columns(df: pd.DataFrame, group_cols: Sequence[str]) -> tuple[str, ...]:
    """Infer the numeric columns to aggregate.

    Args:
        df: Per-seed results.
        group_cols: Columns identifying a comparison cell.

    Returns:
        Numeric, non-boolean columns that are neither ``seed`` nor a grouping column.
    """
    excluded = {*group_cols, "seed"}
    return tuple(
        name
        for name in df.columns
        if name not in excluded
        and pd.api.types.is_numeric_dtype(df[name])
        and not pd.api.types.is_bool_dtype(df[name])
    )


def _carry_constant_columns(
    df: pd.DataFrame, group_cols: Sequence[str], handled: Sequence[str]
) -> pd.DataFrame:
    """Carry through columns that are constant within every group.

    Args:
        df: Per-seed results.
        group_cols: Columns identifying a comparison cell.
        handled: Columns already accounted for and not to be carried.

    Returns:
        One row per group holding the constant columns, or an empty frame of group keys if
        there are none. Columns that vary within a group are dropped rather than
        silently reduced.
    """
    keys = list(group_cols)
    skip = {*keys, *handled, "seed"}
    candidates = [name for name in df.columns if name not in skip]
    grouped = df.groupby(keys, as_index=False, sort=False)
    carried = grouped[keys].first()
    for name in candidates:
        counts = df.groupby(keys, sort=False)[name].nunique(dropna=False)
        if bool((counts <= 1).all()):
            carried = carried.merge(grouped[[*keys, name]].first(), on=keys, validate="one_to_one")
    return carried


def aggregate_over_seeds(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    *,
    metric_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Collapse per-seed rows into mean and standard deviation.

    Args:
        df: Per-seed results, with a ``seed`` column and one or more metric columns.
        group_cols: Columns identifying a comparison cell, e.g.
            ``("model", "regime", "dof", "horizon_samples")``.
        metric_cols: Columns to aggregate. If None, every numeric non-boolean column that
            is neither ``seed`` nor a grouping column is aggregated.

    Returns:
        One row per group, with ``<metric>_mean``, ``<metric>_std`` and ``n_seeds``
        columns. Units are unchanged from the input. Non-metric columns that are constant
        within every group are carried through unchanged; columns that vary are dropped.

    Raises:
        ValueError: If ``df`` has no ``seed`` column, or if any group contains fewer than
            three seeds -- a comparison over fewer seeds than that measures initialisation
            noise, so it is refused rather than reported with a caveat.
    """
    if "seed" not in df.columns:
        raise ValueError("df has no 'seed' column; per-seed aggregation needs one")
    missing = [name for name in group_cols if name not in df.columns]
    if missing:
        raise ValueError(f"group columns {missing} are not in the frame: {list(df.columns)}")
    if df.empty:
        raise ValueError("cannot aggregate an empty frame")
    keys = list(group_cols)
    seeds = df.groupby(keys, sort=False)["seed"].nunique()
    thin = seeds[seeds < MIN_SEEDS]
    if not thin.empty:
        raise ValueError(
            f"{len(thin)} group(s) carry fewer than {MIN_SEEDS} seeds, e.g. "
            f"{thin.index[0]} with {int(thin.iloc[0])}. A model-vs-model comparison over "
            f"fewer seeds than that measures initialisation noise (CLAUDE.md "
            f"non-negotiable 5). Deterministic models go through aggregate_results, which "
            f"routes them around this check and records n_seeds=1 with a NaN std."
        )
    metrics = _metric_columns(df, keys) if metric_cols is None else metric_cols
    absent = [name for name in metrics if name not in df.columns]
    if absent:
        raise ValueError(f"metric columns {absent} are not in the frame")
    grouped = df.groupby(keys, as_index=False, sort=False)
    out = grouped[keys].first()
    for name in metrics:
        stats = df.groupby(keys, sort=False)[name].agg(["mean", "std"]).reset_index()
        stats = stats.rename(columns={"mean": f"{name}_mean", "std": f"{name}_std"})
        out = out.merge(stats, on=keys, validate="one_to_one")
    out = out.merge(seeds.rename("n_seeds").reset_index(), on=keys, validate="one_to_one")
    carried = _carry_constant_columns(df, keys, metrics)
    if len(carried.columns) > len(keys):
        out = out.merge(carried, on=keys, validate="one_to_one")
    return out


def aggregate_results(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
    *,
    metric_cols: tuple[str, ...],
    deterministic_col: str = "deterministic",
) -> pd.DataFrame:
    """Aggregate a mixed table of deterministic and stochastic runs.

    Stochastic rows go through :func:`aggregate_over_seeds` unchanged, three-seed minimum
    and all. Deterministic rows -- closed-form models, whose fit the seed cannot enter --
    pass through with ``n_seeds = 1`` and ``<metric>_std = NaN``.

    ``NaN`` rather than ``0.0`` is deliberate: with a single observation the sample standard
    deviation is undefined, and writing ``0.0`` would assert a measured spread of zero. The
    claim that these models *are* deterministic is carried by a bitwise-equality test on the
    fitted coefficients, which is strictly stronger than three identical rows because it
    shows the seed cannot enter rather than observing that it did not.

    Args:
        df: Per-run results, one row per (cell, run), with a ``seed`` column and a boolean
            ``deterministic_col``.
        group_cols: Columns identifying a comparison cell.
        metric_cols: Metric columns to aggregate.
        deterministic_col: Boolean column marking rows exempt from the seed rule.

    Returns:
        One row per group, with ``<metric>_mean``, ``<metric>_std``, ``n_seeds`` and the
        deterministic flag, plus any non-metric column constant within its group.

    Raises:
        ValueError: If ``deterministic_col`` is absent, if a deterministic group holds more
            than one distinct set of metric values, or if a stochastic group has fewer than
            three seeds.
    """
    if deterministic_col not in df.columns:
        raise ValueError(
            f"df has no {deterministic_col!r} column; without it there is no way to tell a "
            f"closed-form model's single row from a stochastic model missing two seeds"
        )
    flag = df[deterministic_col].astype(bool)
    parts: list[pd.DataFrame] = []
    if bool((~flag).any()):
        parts.append(aggregate_over_seeds(df.loc[~flag], group_cols, metric_cols=metric_cols))
    if bool(flag.any()):
        parts.append(_passthrough_deterministic(df.loc[flag], group_cols, metric_cols))
    return pd.concat(parts, ignore_index=True)


def _passthrough_deterministic(
    df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]
) -> pd.DataFrame:
    """Pass deterministic rows through with ``n_seeds = 1`` and a NaN std.

    Args:
        df: The deterministic subset.
        group_cols: Columns identifying a comparison cell.
        metric_cols: Metric columns.

    Returns:
        One row per group.

    Raises:
        ValueError: If a group holds more than one distinct value of any metric -- a model
            declared deterministic that produced two different numbers is a contradiction,
            not something to average.
    """
    keys = list(group_cols)
    for name in metric_cols:
        spread = df.groupby(keys, sort=False)[name].nunique(dropna=False)
        inconsistent = spread[spread > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"{len(inconsistent)} deterministic group(s) hold more than one distinct "
                f"{name!r}, e.g. {inconsistent.index[0]}. A model marked deterministic must "
                f"produce the same number every time; averaging the difference away would "
                f"hide the contradiction."
            )
    grouped = df.groupby(keys, as_index=False, sort=False)
    out = grouped[keys].first()
    for name in metric_cols:
        first = grouped[[*keys, name]].first().rename(columns={name: f"{name}_mean"})
        out = out.merge(first, on=keys, validate="one_to_one")
        out[f"{name}_std"] = np.nan
    out["n_seeds"] = 1
    carried = _carry_constant_columns(df, keys, metric_cols)
    if len(carried.columns) > len(keys):
        out = out.merge(carried, on=keys, validate="one_to_one")
    return out


def build_baselines_table(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Build ``results/baselines.csv`` from the per-run table.

    ``baselines_by_seed.csv`` is the source of truth -- one row per (model, regime, DOF,
    horizon, run). This function is the only place the aggregated Gate 3 artifact is built,
    so its schema is defined once.

    Args:
        by_seed: Per-run results. Must carry ``model``, ``regime``, ``dof``,
            ``horizon_samples``, ``horizon_s``, ``seed``, ``deterministic``, ``n_windows``,
            ``rmse``, ``mae``, ``rmse_persistence``, ``skill``, ``skill_ci_lo``,
            ``skill_ci_hi``, ``n_params`` and ``fit_time_s``.

    Returns:
        One row per (model, regime, DOF, horizon), columns :data:`BASELINES_COLUMNS`,
        sorted by ``regime, model, dof, horizon_samples``. ``skill_ci_lo``/``skill_ci_hi``
        are the per-run realization bootstrap interval for a single-seed row and the
        **envelope** of the per-run intervals for a multi-seed one -- see the module
        docstring and :func:`baselines_caveats`; they are not a calibrated interval for the
        seed mean, and the per-run intervals stay available in ``baselines_by_seed.csv``.

    Raises:
        ValueError: If a required column is missing, or if ``rmse_persistence`` or
            ``n_windows`` varies between the runs of one cell -- that would mean the skill
            denominator was measured over different window sets for different seeds, which
            is exactly the failure the single-pass runner exists to make impossible.
    """
    required = {
        *BASELINES_GROUP_COLS,
        "horizon_s",
        "seed",
        "deterministic",
        "n_windows",
        "rmse_persistence",
        "skill_ci_lo",
        "skill_ci_hi",
        "n_params",
        "fit_time_s",
        *BASELINES_METRIC_COLS,
    }
    missing = sorted(required - set(by_seed.columns))
    if missing:
        raise ValueError(f"by_seed is missing columns {missing}")
    keys = list(BASELINES_GROUP_COLS)
    for name in ("rmse_persistence", "n_windows"):
        spread = by_seed.groupby(keys, sort=False)[name].nunique(dropna=False)
        varying = spread[spread > 1]
        if not varying.empty:
            raise ValueError(
                f"{name!r} varies between the runs of {len(varying)} cell(s), e.g. "
                f"{varying.index[0]}. Every model in a regime is scored in one pass over "
                f"one window set, so this can only mean the runs were not comparable."
            )
    aggregated = aggregate_results(by_seed, BASELINES_GROUP_COLS, metric_cols=BASELINES_METRIC_COLS)
    carried = by_seed.groupby(keys, as_index=False, sort=False).agg(
        horizon_s=("horizon_s", "first"),
        deterministic=("deterministic", "first"),
        n_windows=("n_windows", "first"),
        rmse_persistence=("rmse_persistence", "first"),
        # Envelope, not mean. The bootstrap runs per run, so a multi-seed cell arrives here
        # as several finished intervals and no arithmetic on finished intervals yields a
        # calibrated interval for the seed mean. The mean of them, which this used to take,
        # is narrower than any of its inputs justifies and is invariant to seed disagreement
        # -- precisely the thing a reader would use the column to detect. min/max is
        # conservative, contains every seed's interval, and widens when the seeds disagree.
        # For a deterministic (n_seeds = 1) row it is that run's interval exactly, unchanged.
        # `baselines_caveats` states this next to every rendered table.
        skill_ci_lo=("skill_ci_lo", "min"),
        skill_ci_hi=("skill_ci_hi", "max"),
        n_params=("n_params", "first"),
        fit_time_s_mean=("fit_time_s", "mean"),
    )
    aggregated = aggregated.drop(
        columns=[c for c in carried.columns if c not in keys and c in aggregated.columns]
    )
    out = aggregated.merge(carried, on=keys, validate="one_to_one")
    return out[list(BASELINES_COLUMNS)].sort_values(
        ["regime", "model", "dof", "horizon_samples"], ignore_index=True
    )


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a results table as a GitHub-flavoured Markdown table.

    Args:
        df: The table to render.
        float_fmt: Format string applied to float columns. ``NaN`` renders as ``n/a``
            rather than as a number, because a NaN standard deviation means "not measured",
            not "zero".

    Returns:
        The rendered table, ready to paste into ``results/results.md``.

    Raises:
        ValueError: If ``df`` has no columns.
    """
    if len(df.columns) == 0:
        raise ValueError("cannot render a table with no columns")

    def cell(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float | np.floating):
            return "n/a" if np.isnan(float(value)) else float_fmt.format(float(value))
        return str(value)

    header = [str(name) for name in df.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _gate_dof_candidates(gate_dof: str) -> tuple[str, ...]:
    """Return the spellings of ``gate_dof`` to look for, caller's own spelling first.

    Args:
        gate_dof: Either spelling of a corpus motion channel.

    Returns:
        ``("roll", "roll_imu")`` for ``"roll"``, ``("roll_imu", "roll")`` for ``"roll_imu"``.

    Raises:
        ValueError: If ``gate_dof`` is not a corpus motion channel under either spelling.
    """
    aliases = channel_aliases(gate_dof)
    return (gate_dof, *(name for name in aliases if name != gate_dof))


def _resolve_gate_dof(scoped: pd.DataFrame, gate_dof: str) -> str | None:
    """Find which spelling of the gate DOF the scored table actually uses.

    The corpus carries every motion channel under two names -- the clean one (``roll``,
    degrees) and the noisy twin the ``imu`` observation mode substitutes for it
    (``roll_imu``, degrees) -- and :func:`dmf.data.dataset.resolve_columns` decides between
    them at window-construction time. The Gate 3 cell is defined by *intent* (roll at 3 s
    in ``id``, per ``docs/IMPLEMENTATION_PLAN.md``), and that intent survives both
    observation modes, so the row is located by logical channel rather than by string
    equality on one spelling. Matching on a hardcoded ``"roll"`` used to abort an otherwise
    finished ``imu`` run at its very last step.

    The caller's exact spelling is tried first, so passing ``gate_dof="roll_imu"``
    explicitly is honoured even in the (currently impossible) case of a table holding both.

    Args:
        scoped: Rows of the table already restricted to the gate regime and horizon.
        gate_dof: Either spelling of the gate channel, e.g. ``"roll"`` or ``"roll_imu"``.

    Returns:
        The spelling present in ``scoped``, or None if no spelling of the channel is
        there. A genuinely missing gate cell is a real failure and stays fatal, but the
        caller raises it, because only the caller knows the regime and horizon that
        complete the cell's address.
    """
    candidates = _gate_dof_candidates(gate_dof)
    present = set(scoped["dof"].unique())
    return next((name for name in candidates if name in present), None)


def build_baselines_markdown(
    table: pd.DataFrame,
    *,
    controls: pd.DataFrame | None = None,
    by_heading: pd.DataFrame | None = None,
    results_dir: Path | None = None,
    gate_regime: str = "id",
    gate_dof: str = "roll",
    gate_horizon_samples: int = 30,
    gate_threshold: float = 0.8,
) -> str:
    """Render ``results/baselines.md``, the document the Gate 3 decision is read from.

    Gate 3 says: "read it. If AR(p) already achieves 0.8 skill at 3 s on roll, your task is
    too easy." That is a decision, so the document leads with the cell the decision turns
    on, its bootstrap interval, and the per-heading breakdown that says whether a pooled
    number is hiding two different regimes -- not with a wall of rows.

    Args:
        table: The aggregated table from :func:`build_baselines_table`.
        controls: Optional control rows from :func:`dmf.eval.controls.controls_table`.
        by_heading: Optional per-heading marginal of the gate regime, from
            :func:`dmf.eval.runner.marginalize_cells`.
        results_dir: Directory the CSVs this document is built from were written to, used
            for the provenance line. **Pass the same directory the document is written
            to.** Left ``None`` the files are named by bare filename and located "beside
            this document", which is true wherever the run wrote them; a hardcoded
            ``results/`` prefix was not, and an ``imu`` run rendered into
            ``results/imu/baselines.md`` used to cite the ``ideal`` run's four CSVs by name
            -- the files its own last caveat says must not be mixed with these numbers.
        gate_regime: Regime the gate threshold applies to.
        gate_dof: DOF the gate threshold applies to, as a **logical** channel name. Under
            ``observation_mode: imu`` the table holds the noisy twin (``roll_imu``)
            instead, and the row is matched by logical channel either way; the rendered
            document names the spelling that was actually scored.
        gate_horizon_samples: Horizon the gate threshold applies to, samples.
        gate_threshold: The Gate 3 skill threshold.

    Returns:
        The rendered Markdown document.

    Raises:
        ValueError: If ``gate_dof`` is not a corpus motion channel, or if the gate cell is
            absent from ``table`` under either of its spellings.
    """
    scoped = table[
        (table["regime"] == gate_regime) & (table["horizon_samples"] == gate_horizon_samples)
    ]
    resolved_dof = _resolve_gate_dof(scoped, gate_dof) if not scoped.empty else None
    if resolved_dof is None:
        raise ValueError(
            f"the gate cell (regime={gate_regime!r}, dof={gate_dof!r} or its imu twin, "
            f"horizon_samples={gate_horizon_samples}) is absent from the table; the "
            f"spellings looked for were {list(_gate_dof_candidates(gate_dof))} and the "
            f"DOFs present at that regime and horizon are {sorted(scoped['dof'].unique())}"
        )
    gate = scoped[scoped["dof"] == resolved_dof]
    horizon_s = float(gate["horizon_s"].iloc[0])
    parts: list[str] = [
        "# Phase 3 baselines (Gate 3)",
        "",
        f"Simulated results only. Gate cell: **{resolved_dof} at {horizon_s:g} s "
        f"({gate_horizon_samples} samples), `{gate_regime}` regime**, threshold "
        f"{gate_threshold:g} skill vs persistence.",
        "",
        _provenance_line(results_dir, with_controls=controls is not None and not controls.empty),
        "",
        "## The gate cell",
        "",
        to_markdown(
            gate[
                [
                    "model",
                    "n_seeds",
                    "deterministic",
                    "rmse_mean",
                    "rmse_persistence",
                    "skill_mean",
                    "skill_std",
                    "skill_ci_lo",
                    "skill_ci_hi",
                    "n_params",
                ]
            ].sort_values("model", ignore_index=True)
        ),
        "",
        "`skill_std` is seed-to-seed spread; `skill_ci_lo`/`skill_ci_hi` are a bootstrap "
        "over held-out **realizations** and are what decide whether a value near the "
        "threshold is distinguishable from it -- on a single-seed row. On a multi-seed row "
        "they are the envelope of the per-run intervals, not a calibrated interval for the "
        "seed mean; see the caveats.",
        "",
    ]
    if by_heading is not None and not by_heading.empty:
        parts += [
            "## The gate cell, broken out by heading",
            "",
            "Pooled numbers on `id` average over 4 headings, and per P1-D2 roll in head "
            "seas sits on the residual floor. The breakdown, not the pooled value, is what "
            "the follow-on decision depends on.",
            "",
            to_markdown(by_heading),
            "",
        ]
    if controls is not None and not controls.empty:
        summary = controls.groupby(["control", "subject_model", "null_model"], as_index=False).agg(
            worst_excess=("excess", "max"), tol=("tol", "first"), passed=("passed", "all")
        )
        parts += [
            "## Negative controls",
            "",
            "Kept out of the headline table so they cannot be mistaken for models.",
            "",
            to_markdown(summary),
            "",
            "The shuffle control's null is the **window-mean forecast**, not zero skill: a "
            "model fitted to time-shuffled targets degenerates to the conditional mean, "
            "which inverts to the window mean, and the window mean beats persistence at "
            "long horizons on a narrowband signal. Testing against zero would report "
            "leakage on a clean pipeline.",
            "",
        ]
    parts += ["## Full table", ""]
    for regime in sorted(table["regime"].unique()):
        subset = table[table["regime"] == regime]
        parts += [
            f"### `{regime}`",
            "",
            to_markdown(
                subset[
                    [
                        "model",
                        "dof",
                        "horizon_s",
                        "n_seeds",
                        "rmse_mean",
                        "rmse_std",
                        "mae_mean",
                        "rmse_persistence",
                        "skill_mean",
                        "skill_std",
                        "skill_ci_lo",
                        "skill_ci_hi",
                        "n_params",
                        "fit_time_s_mean",
                    ]
                ].reset_index(drop=True)
            ),
            "",
        ]
    parts += ["## Caveats", ""]
    parts += [f"- {caveat}" for caveat in baselines_caveats(table, gate_regime=gate_regime)]
    parts.append("")
    return "\n".join(parts)


def build_results_report(results_dir: Path, out_path: Path) -> Path:
    """Assemble every committed CSV in ``results/`` into the Markdown report.

    Args:
        results_dir: Directory of committed result CSVs.
        out_path: Destination Markdown file, normally ``results/results.md``.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If an expected CSV is missing, so that a partially regenerated
            report cannot be mistaken for a complete one.
        NotImplementedError: Always, for now. This assembles *every* table in the project
            and is Phase 6 work; Phase 3's gate artifact is built by
            :func:`build_baselines_markdown`.
    """
    raise NotImplementedError
