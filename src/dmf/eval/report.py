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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from dmf.data.channels import channel_aliases

# Imported rather than restated: the two level labels are written by the runner and read
# here, and two spellings of "sea_state" in two modules is how a roll-up silently stops
# being distinguishable from a cell row.
from dmf.eval.quiescence_runner import CELL_LEVEL, SEA_STATE_LEVEL

__all__ = [
    "ABLATION_CI_ENVELOPES",
    "ABLATION_FLAG_COLUMNS",
    "ABLATION_TABLE_IDS",
    "BASELINES_ARTIFACTS",
    "BASELINES_COLUMNS",
    "BASELINES_GROUP_COLS",
    "BASELINES_METRIC_COLS",
    "LONG_BAND",
    "NOT_SCORABLE",
    "PROBABILISTIC_COLUMNS",
    "PHASE6_SUBDIR",
    "REPORT_SOURCES",
    "REVIN_VEHICLES",
    "SHORT_BAND",
    "TABLE_SOURCE_PATTERN",
    "TABLE_SOURCE_PREFIX",
    "LoadedSource",
    "ReportSource",
    "RenderedTableSpec",
    "TableSource",
    "ablation_table_id",
    "aggregate_over_seeds",
    "aggregate_results",
    "baselines_caveats",
    "build_ablation_table",
    "build_baselines_markdown",
    "build_baselines_table",
    "build_point_table",
    "build_probabilistic_table",
    "build_probabilistic_view",
    "build_quiescence_table",
    "build_results_report",
    "dmf_table_marker",
    "horizon_band",
    "load_report_sources",
    "parse_table_sources",
    "results_report_caveats",
    "split_ablation_rows",
    "split_controls",
    "table_source_marker",
    "to_markdown",
    "write_table",
]

#: Minimum seeds for a stochastic model-vs-model comparison (CLAUDE.md non-negotiable 5).
MIN_SEEDS = 3

#: Grouping that identifies one comparison cell in ``results/baselines.csv``.
BASELINES_GROUP_COLS: tuple[str, ...] = ("model", "regime", "dof", "horizon_samples")

#: Metrics aggregated to mean and std over seeds. ``nrmse`` rides with them because it is a
#: per-run model quantity like the other three; its denominator ``signal_std`` is *not* here,
#: because that is a property of the targets and is constant across the runs of a cell --
#: :func:`build_baselines_table` asserts that and carries it through unaggregated.
BASELINES_METRIC_COLS: tuple[str, ...] = ("rmse", "mae", "skill", "nrmse")

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
    "signal_std",
    "nrmse_mean",
    "nrmse_std",
    "n_params",
    "fit_time_s_mean",
)

#: The artifacts a baselines run writes, in the order the provenance line names them, each
#: paired with what it holds. An entry whose contents are not rendered into the document
#: says so: the sentence these are embedded in is "every number here traces to", so a file
#: no number here is read from must not be left implying otherwise.
#:
#: Single source of truth: the ``baselines.md`` provenance line is generated from this
#: rather than written out in prose, so a renamed or added artifact cannot leave the
#: document pointing at a file that no longer exists. The *directory* is
#: supplied by the caller and is deliberately not baked in here -- a run writing to
#: ``results/imu/`` used to emit a provenance line naming ``results/baselines*.csv``, i.e.
#: the other observation mode's files, which the last caveat below explicitly warns against
#: mixing with these numbers.
BASELINES_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("baselines.csv", "aggregated"),
    ("baselines_by_seed.csv", "one row per run, the source of truth"),
    ("baselines_by_cell.csv", "per grid cell"),
    ("paired_contrasts.csv", "paired model-vs-model skill differences, not rendered here"),
    ("baselines_controls.csv", "negative controls"),
)

#: Artifacts written only when the negative controls are run, so the provenance line must
#: not name them otherwise.
_CONTROL_ARTIFACTS: frozenset[str] = frozenset({"baselines_controls.csv"})

#: Artifacts written only when at least one configured model-vs-model contrast pair was
#: present in the run, so the provenance line must not name them otherwise. Same rule as
#: :data:`_CONTROL_ARTIFACTS`, for the same reason: a baselines-only run writes no
#: ``paired_contrasts.csv``, and a document asserting a file that is not beside it is the
#: P3-D21 defect in a second instance.
_CONTRAST_ARTIFACTS: frozenset[str] = frozenset({"paired_contrasts.csv"})


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


def _provenance_line(results_dir: Path | None, *, with_controls: bool, with_contrasts: bool) -> str:
    """Render the sentence naming the files a rendered document was built from.

    Args:
        results_dir: Directory the artifacts were written to. ``None`` means the caller did
            not say, in which case the files are named by bare filename and located
            relative to the document -- true in any directory, where a guessed
            ``results/`` prefix would be false in all but one.
        with_controls: Whether the controls table was written. Naming a file that was not
            written is the same defect as naming the wrong one.
        with_contrasts: Whether ``paired_contrasts.csv`` was written, i.e. whether the run
            held at least one configured contrast pair. Same rule as ``with_controls``: a
            baselines-only run writes no such file and this document must not name it.

    Returns:
        One Markdown sentence.
    """
    omitted: set[str] = set()
    if not with_controls:
        omitted |= _CONTROL_ARTIFACTS
    if not with_contrasts:
        omitted |= _CONTRAST_ARTIFACTS
    named = [(name, what) for name, what in BASELINES_ARTIFACTS if name not in omitted]
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
        "**`nrmse_mean` is there because skill is not comparable across horizons on this "
        "signal.** Persistence error tracks the target's autocorrelation, so the skill "
        "denominator oscillates with the signal's own period instead of growing with lead "
        "time: on `id`/test in the `ideal` mode, persistence RMSE for roll *falls* from "
        "7.09 deg at 50 samples (5 s) to 3.79 deg at 100 (10 s), because 10 s is about one "
        "roll period (P3-D5/P3-D6). A skill-vs-horizon curve therefore has dips that belong "
        "to the reference and not to the model. `nrmse_mean` is `rmse_mean / signal_std`, "
        "where `signal_std` is the standard deviation of the held-out target itself at that "
        "lead time in corpus units: 1.0 means no better than predicting the partition mean, "
        "lower is better, and it is the column to read when the horizon or the vessel "
        "varies.",
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
            ``rmse``, ``mae``, ``rmse_persistence``, ``skill``, ``nrmse``, ``signal_std``,
            ``skill_ci_lo``, ``skill_ci_hi``, ``n_params`` and ``fit_time_s``.

    Returns:
        One row per (model, regime, DOF, horizon), columns :data:`BASELINES_COLUMNS`,
        sorted by ``regime, model, dof, horizon_samples``. ``skill_ci_lo``/``skill_ci_hi``
        are the per-run realization bootstrap interval for a single-seed row and the
        **envelope** of the per-run intervals for a multi-seed one -- see the module
        docstring and :func:`baselines_caveats`; they are not a calibrated interval for the
        seed mean, and the per-run intervals stay available in ``baselines_by_seed.csv``.

    Raises:
        ValueError: If a required column is missing, or if ``rmse_persistence``,
            ``signal_std`` or ``n_windows`` varies between the runs of one cell -- that
            would mean the skill or nrmse denominator was measured over different window
            sets for different seeds, which is exactly the failure the single-pass runner
            exists to make impossible.
    """
    required = {
        *BASELINES_GROUP_COLS,
        "horizon_s",
        "seed",
        "deterministic",
        "n_windows",
        "rmse_persistence",
        "signal_std",
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
    # `signal_std` is here rather than assumed constant: it is a property of the targets
    # alone, so it cannot legitimately differ between the runs of one cell, and
    # `_carry_constant_columns` would silently *drop* it if it did -- leaving the schema
    # selection below to fail with a missing-column error that says nothing about the cause.
    for name in ("rmse_persistence", "signal_std", "n_windows"):
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
    with_contrasts: bool = False,
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
        with_contrasts: Whether the run wrote ``paired_contrasts.csv`` beside this
            document, which it does only when at least one configured model-vs-model
            contrast pair was present -- a baselines-only run writes none. The provenance
            line names the file only when this is ``True``; defaulting to ``False`` keeps
            a caller that does not say from asserting a file that is not there, which is
            the P3-D21 defect. Not otherwise rendered: the contrasts are read from the CSV,
            not from this document.
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
        _provenance_line(
            results_dir,
            with_controls=controls is not None and not controls.empty,
            with_contrasts=with_contrasts,
        ),
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
        # Grouped over the asserted rows, because those are the ones the pass/fail decision
        # was taken over; the reported-only rows follow in their own table rather than being
        # averaged into this one or dropped. A control's verdict and the cells it declined
        # to assert on are two different statements and are rendered as two.
        asserted = controls[controls["asserted"]] if "asserted" in controls else controls
        summary = asserted.groupby(["control", "subject_model", "null_model"], as_index=False).agg(
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
        reported_only = (
            controls[~controls["asserted"]] if "asserted" in controls else controls.iloc[:0]
        )
        if not reported_only.empty:
            worst = reported_only.groupby(["control", "regime", "dof"], as_index=False).agg(
                worst_excess=("excess", "max"), tol=("tol", "first"), passed=("passed", "all")
            )
            parts += [
                "### Cells reported but not asserted on",
                "",
                "These channels sit on their P1-D2 residual floor across the whole test "
                "partition, so their targets are a 26 dB-suppressed engineering stand-in "
                "rather than the physics the DOF is named after. The shuffle statistic "
                "there is dominated by a train/test amplitude mismatch, measured in both "
                "directions and shown not to be leakage (`docs/protocol.md` P6-D11). They "
                "are scored and shown; they do not decide the control.",
                "",
                to_markdown(worst),
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
                        "nrmse_mean",
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


# --------------------------------------------------------------------------------------
# Phase 5: the probabilistic table.
#
# Structured exactly like the baselines table above -- one row per (model, head, regime,
# DOF, horizon), aggregated over seeds by the same rules, with the same >= 3 seed floor and
# the same envelope treatment of bootstrap intervals -- so that a reader who has learned to
# read one can read the other. What differs is what the columns mean, and that is the point
# of the caveats below rather than of a different schema.
# --------------------------------------------------------------------------------------

#: Grouping keys. ``head`` joins the baselines keys because one backbone ships two heads,
#: and a table keyed on ``model`` alone would silently average a quantile row into a
#: Gaussian one.
PROBABILISTIC_GROUP_COLS: tuple[str, ...] = ("model", "head", "regime", "dof", "horizon_samples")

#: Metrics aggregated mean +/- std over seeds.
PROBABILISTIC_METRIC_COLS: tuple[str, ...] = (
    "picp",
    "mean_interval_width",
    "width_ratio",
    "winkler",
    "crps",
    "pinball",
    "crossing_rate",
)

#: Schema of ``probabilistic.csv``.
PROBABILISTIC_COLUMNS: tuple[str, ...] = (
    "model",
    "head",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_seeds",
    "deterministic",
    "n_windows",
    "n_quantiles",
    "alpha",
    "picp_mean",
    "picp_std",
    "picp_ci_lo",
    "picp_ci_hi",
    "mean_interval_width_mean",
    "mean_interval_width_std",
    "width_ratio_mean",
    "width_ratio_std",
    "signal_std",
    "winkler_mean",
    "winkler_std",
    "crps_mean",
    "crps_std",
    "pinball_mean",
    "pinball_std",
    "crossing_rate_mean",
    "crossing_rate_std",
    "n_params",
    "val_loss_name",
    "fit_time_s_mean",
)

#: Artifacts a Phase 5 run writes, and what each is for. Rendered into every document's
#: provenance line, as :data:`BASELINES_ARTIFACTS` is.
PROBABILISTIC_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("probabilistic.csv", "aggregated"),
    ("probabilistic_by_seed.csv", "one row per run, the source of truth"),
    ("baselines.csv", "the point accuracy of the same runs, scored against persistence"),
)


def width_ratio(mean_width: pd.Series, signal_std: pd.Series, alpha: pd.Series) -> pd.Series:
    """Express interval width as a fraction of an unconditional interval's width.

    A mean interval width in degrees is not interpretable on its own: 2 deg is sharp for
    roll at 15 s and uselessly wide for pitch at 1 s. The reference is the width an
    unconditional Gaussian interval would need to reach the same nominal level using only
    the scored partition's own spread -- ``2 * z(1 - alpha/2) * signal_std`` -- so

        ``width_ratio = mean_interval_width / (2 * z(1 - alpha/2) * signal_std)``

    and **1.0 means the interval is no sharper than knowing nothing but the partition's
    variance**. This is deliberately the same device ``nrmse`` uses for RMSE (P4-D3), where
    1.0 means "no better than predicting the partition mean", and it exists for the same
    reason: on this corpus the absolute magnitudes flatter every model, so a dimensionless
    reference is what makes a column readable across DOFs and horizons.

    It is a **sharpness** measure and says nothing about calibration on its own. A ratio
    below 1.0 with PICP at nominal is a genuinely informative interval; a ratio below 1.0
    with PICP well under nominal is just an interval that is too narrow.

    Args:
        mean_width: Mean interval width, corpus units.
        signal_std: Standard deviation of the target at that lead time and channel over the
            scored partition, corpus units. Comes from the point pass, which already
            computes it for ``nrmse``.
        alpha: Nominal miscoverage the width was measured at.

    Returns:
        Dimensionless ratio, one value per input row.

    Raises:
        ValueError: If any ``signal_std`` is not finite and strictly positive.
    """
    std = np.asarray(signal_std, dtype=np.float64)
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError(
            "signal_std must be finite and strictly positive to normalise an interval "
            "width; a zero-variance channel has no unconditional interval to compare to"
        )
    z = norm.ppf(1.0 - np.asarray(alpha, dtype=np.float64) / 2.0)
    return pd.Series(
        np.asarray(mean_width, dtype=np.float64) / (2.0 * z * std),
        index=mean_width.index,
        dtype=float,
    )


def build_probabilistic_table(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Build ``results/<dir>/probabilistic.csv`` from the per-run probabilistic table.

    ``probabilistic_by_seed.csv`` is the source of truth; this is the only place the
    aggregated artifact is built, so its schema is defined once.

    ``picp_ci_lo``/``picp_ci_hi`` are the **envelope** of the per-run realization bootstrap
    intervals, min of the lows and max of the highs, for exactly the reason P3-D22 records
    for ``skill_ci``: the mean of several finished intervals is not an interval for
    anything, and it is invariant to seed disagreement, which is the one thing a reader
    would consult it to detect.

    Args:
        by_seed: Per-run rows carrying :data:`PROBABILISTIC_GROUP_COLS`, ``horizon_s``,
            ``seed``, ``deterministic``, ``n_windows``, ``n_quantiles``, ``alpha``,
            ``signal_std``, ``picp_ci_lo``, ``picp_ci_hi``, ``n_params``, ``val_loss_name``,
            ``fit_time_s`` and every column in :data:`PROBABILISTIC_METRIC_COLS` except
            ``width_ratio``, which is derived here.

    Returns:
        One row per (model, head, regime, DOF, horizon), columns
        :data:`PROBABILISTIC_COLUMNS`, sorted by ``regime, model, head, dof,
        horizon_samples``.

    Raises:
        ValueError: If a required column is missing, or if ``signal_std``, ``n_windows``,
            ``n_quantiles`` or ``alpha`` varies between the runs of one cell -- which would
            mean the runs of a single cell were not scored over one window set at one
            nominal level, and no aggregate of them would mean anything.
    """
    derived = {"width_ratio"}
    required = {
        *PROBABILISTIC_GROUP_COLS,
        "horizon_s",
        "seed",
        "deterministic",
        "n_windows",
        "n_quantiles",
        "alpha",
        "signal_std",
        "picp_ci_lo",
        "picp_ci_hi",
        "n_params",
        "val_loss_name",
        "fit_time_s",
        *(set(PROBABILISTIC_METRIC_COLS) - derived),
    }
    missing = sorted(required - set(by_seed.columns))
    if missing:
        raise ValueError(f"by_seed is missing columns {missing}")

    frame = by_seed.copy()
    frame["width_ratio"] = width_ratio(
        frame["mean_interval_width"], frame["signal_std"], frame["alpha"]
    )
    keys = list(PROBABILISTIC_GROUP_COLS)
    for name in ("signal_std", "n_windows", "n_quantiles", "alpha"):
        spread = frame.groupby(keys, sort=False)[name].nunique(dropna=False)
        varying = spread[spread > 1]
        if not varying.empty:
            raise ValueError(
                f"{name!r} varies between the runs of {len(varying)} cell(s), e.g. "
                f"{varying.index[0]}. Every model in a regime is scored in one pass over "
                f"one window set at one nominal level, so this can only mean the runs were "
                f"not comparable."
            )

    aggregated = aggregate_results(
        frame, PROBABILISTIC_GROUP_COLS, metric_cols=PROBABILISTIC_METRIC_COLS
    )
    carried = frame.groupby(keys, as_index=False, sort=False).agg(
        horizon_s=("horizon_s", "first"),
        deterministic=("deterministic", "first"),
        n_windows=("n_windows", "first"),
        n_quantiles=("n_quantiles", "first"),
        alpha=("alpha", "first"),
        signal_std=("signal_std", "first"),
        # Envelope, never the mean -- see the docstring and P3-D22.
        picp_ci_lo=("picp_ci_lo", "min"),
        picp_ci_hi=("picp_ci_hi", "max"),
        n_params=("n_params", "first"),
        val_loss_name=("val_loss_name", "first"),
        fit_time_s_mean=("fit_time_s", "mean"),
    )
    aggregated = aggregated.drop(
        columns=[c for c in carried.columns if c not in keys and c in aggregated.columns]
    )
    merged = aggregated.merge(carried, on=keys, validate="one_to_one")
    ordered = merged.sort_values(
        ["regime", "model", "head", "dof", "horizon_samples"], kind="stable"
    ).reset_index(drop=True)
    return ordered[list(PROBABILISTIC_COLUMNS)]


# --------------------------------------------------------------------------------------
# Phase 6: results.md, rendered *from* the CSVs.
#
# Gate 6 is a traceability criterion, not a numeric one (docs/protocol.md P6-D1): every
# table in this document must name the CSV it was read from and state that file's row
# count. That is made a structural property of the renderer rather than a claim about it --
# a number that is not in a CSV cannot be rendered, because every table below is a
# projection of a frame this module read off disk.
#
# **Each table carries its provenance twice, from one record, and that is deliberate.**
# :func:`table_source_marker` renders a visible line a human reads and greps for; a
# provenance statement nobody can see is one nobody notices has gone stale.
# :func:`dmf_table_marker` renders the machine contract :data:`dmf.eval.gate.
# GATE6_MARKER_SPEC` fixes, as the last non-blank line before the table, which is what the
# Gate 6 read-out parses. Both come from the same :class:`RenderedTableSpec`, so the two
# cannot disagree, and ``tests/test_report.py`` asserts they never do.
#
# Three counts, three different things:
#   ``csv_rows`` -- data rows of the named CSV, header excluded. Predicate 4 checks this.
#   ``rows``     -- rows of the Markdown table below the marker.
#   ``select``   -- how the second was derived from the first. Nearly every table here is a
#                   projection: a per-run table aggregated over seeds (P6-D10), or one
#                   ablation's rows out of five. A table declaring no ``select`` is asserted
#                   to be the whole file, so the filter is declared rather than assumed.
# --------------------------------------------------------------------------------------

#: Prefix every human-readable provenance line starts with. Greppable on its own.
TABLE_SOURCE_PREFIX: str = "Source: "

#: The exact grammar of the human-readable provenance line. Public so that a reader of the
#: document and a checker of it parse the same thing.
TABLE_SOURCE_PATTERN: str = (
    r"^Source: `(?P<path>[^`]+)` \| csv_rows: (?P<csv_rows>\d+) \| "
    r"rendered_rows: (?P<rendered_rows>\d+) \| table_id: (?P<table_id>[A-Za-z0-9_.\-]+)$"
)

_TABLE_SOURCE_RE = re.compile(TABLE_SOURCE_PATTERN)

#: Subdirectory of the results root holding the Phase 6 CSVs. The document is written to the
#: root beside ``results/e02/`` and ``results/e03/``, and every marker names its source
#: **relative to that root**, so ``results_dir / source`` resolves for any caller and the
#: rendered text does not depend on the current working directory. A document whose content
#: varied with the directory it was rendered from could not be byte-compared against a
#: re-render, which is how Gate 6 predicate 2 checks it was not hand-edited.
PHASE6_SUBDIR: str = "e04"


@dataclass(frozen=True)
class TableSource:
    """One rendered table's provenance, parsed back out of the document.

    Attributes:
        table_id: Stable identifier of the rendered table, unique within the document.
        path: The CSV the table was read from, relative to the results root.
        csv_rows: Data rows of that CSV, header excluded.
        rendered_rows: Rows of the Markdown table under the marker, header and rule
            excluded.
    """

    table_id: str
    path: str
    csv_rows: int
    rendered_rows: int


@dataclass(frozen=True)
class RenderedTableSpec:
    """Everything both provenance markers of one table are rendered from.

    One record, two renderings: a reader's line and a parser's comment. They cannot drift
    because neither is written by hand.

    Attributes:
        table_id: Stable slug. The Gate 6 read-out matches its required-table list against
            these, so they are a vocabulary rather than free text.
        section: Which section of ``docs/IMPLEMENTATION_PLAN.md`` the table belongs to.
        source: The CSV, relative to the results root.
        csv_rows: Data rows of that CSV.
        rendered_rows: Rows of the rendered table.
        select: How the rendered rows were derived from the file's. Empty means the table
            *is* the file, row for row, which is a strictly stronger claim.
    """

    table_id: str
    section: str
    source: str
    csv_rows: int
    rendered_rows: int
    select: str = ""


def table_source_marker(table_id: str, path: str, csv_rows: int, rendered_rows: int) -> str:
    """Render the human-readable provenance line.

    Args:
        table_id: Stable identifier, matching ``[A-Za-z0-9_.-]+``.
        path: The CSV the table was read from, relative to the results root.
        csv_rows: Data rows of that CSV, header excluded.
        rendered_rows: Rows of the table rendered under this marker.

    Returns:
        One line matching :data:`TABLE_SOURCE_PATTERN`.

    Raises:
        ValueError: If ``table_id`` does not match the grammar, if ``path`` contains a
            backtick, or if either count is negative -- any of which would emit a line that
            cannot be parsed, which is indistinguishable from a table with no provenance.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", table_id):
        raise ValueError(f"table_id {table_id!r} must match [A-Za-z0-9_.-]+")
    if "`" in path:
        raise ValueError(f"path {path!r} contains a backtick and would break the marker")
    if csv_rows < 0 or rendered_rows < 0:
        raise ValueError(f"row counts must be non-negative, got {csv_rows} and {rendered_rows}")
    return (
        f"{TABLE_SOURCE_PREFIX}`{path}` | csv_rows: {csv_rows} | "
        f"rendered_rows: {rendered_rows} | table_id: {table_id}"
    )


def dmf_table_marker(spec: RenderedTableSpec) -> str:
    """Render the machine-readable marker the Gate 6 read-out parses.

    The contract is :data:`dmf.eval.gate.GATE6_MARKER_SPEC`::

        <!-- dmf-table id=<slug> section=<6.1|6.2|6.3> source=<path/to.csv>
             csv_rows=<int> rows=<int> [select="<filter>"] -->

    and it must be the **last non-blank line before the table**, which is what
    :func:`_table_block` guarantees.

    Args:
        spec: The table's provenance record.

    Returns:
        One HTML comment line.

    Raises:
        ValueError: If a field would break the grammar -- an unparseable marker is worth
            nothing, so it fails where it is written rather than where it is read.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", spec.table_id):
        raise ValueError(f"table_id {spec.table_id!r} must match [A-Za-z0-9_.-]+")
    if any(char.isspace() for char in spec.source) or "-->" in spec.source:
        raise ValueError(f"source {spec.source!r} cannot be written as a bare marker value")
    if '"' in spec.select or "-->" in spec.select:
        raise ValueError(f"select {spec.select!r} would break the marker's quoting")
    parts = [
        "<!-- dmf-table",
        f"id={spec.table_id}",
        f"section={spec.section}",
        f"source={spec.source}",
        f"csv_rows={spec.csv_rows}",
        f"rows={spec.rendered_rows}",
    ]
    if spec.select:
        parts.append(f'select="{spec.select}"')
    return " ".join([*parts, "-->"])


def parse_table_sources(markdown: str) -> tuple[TableSource, ...]:
    """Parse every human-readable provenance line out of a rendered document.

    Args:
        markdown: The rendered document.

    Returns:
        One :class:`TableSource` per line, in document order.
    """
    found: list[TableSource] = []
    for line in markdown.splitlines():
        match = _TABLE_SOURCE_RE.match(line.strip())
        if match is not None:
            found.append(
                TableSource(
                    table_id=match["table_id"],
                    path=match["path"],
                    csv_rows=int(match["csv_rows"]),
                    rendered_rows=int(match["rendered_rows"]),
                )
            )
    return tuple(found)


def _relative_to(path: Path, root: Path) -> str:
    """Render a path relative to the results root.

    Args:
        path: The file, possibly containing ``..``.
        root: The results root every marker names its source relative to.

    Returns:
        A POSIX path relative to ``root``, or an absolute one if the file is not below it.
        Never relative to the current working directory: the document must be a
        deterministic function of its inputs, and a cwd-relative path is not.
    """
    resolved = Path(path).resolve()
    with contextlib.suppress(ValueError):
        return resolved.relative_to(Path(root).resolve()).as_posix()
    return resolved.as_posix()


@dataclass(frozen=True)
class ReportSource:
    """One CSV the report reads, and what happens when it is not there.

    Attributes:
        key: Name the renderer refers to the frame by.
        candidates: Filenames tried in order, relative first to ``<root>/e04`` and then to
            the root itself. More than one where a table has two legitimate producers --
            ``probabilistic_baseline.csv`` is the joined artifact the Phase 6 plan names,
            and ``probabilistic_by_seed.csv`` is what :mod:`dmf.eval.scoring` writes when
            the residual-interval floor is scored alongside the heads in one pass. The
            marker names whichever was actually read.
        what: One line describing the contents, used in the inputs list.
        required: Whether a section of the report cannot be rendered without it. A missing
            required source is fatal under ``strict``: a partially regenerated report that
            silently omits section 6.3 is worse than no report, because it looks complete.
        rendered: Whether any table is rendered from it. ``False`` entries are named in the
            inputs list *without* a provenance marker, so that the marker set stays in
            one-to-one correspondence with the rendered tables.
    """

    key: str
    candidates: tuple[str, ...]
    what: str
    required: bool
    rendered: bool = True


#: Every CSV ``results.md`` reads, in the order the inputs list names them.
REPORT_SOURCES: tuple[ReportSource, ...] = (
    ReportSource(
        key="metrics_full",
        candidates=("metrics_full.csv",),
        what="per-run point accuracy: rmse, mae, skill, nrmse, phase lag (section 6.1)",
        required=True,
    ),
    ReportSource(
        key="quiescence",
        candidates=("quiescence.csv",),
        what="quiescent-window detection by (model, regime, threshold set, rule, sea state)",
        required=True,
    ),
    ReportSource(
        key="ablations",
        candidates=("ablations.csv",),
        what="the five ablations, as paired contrasts against their reference arm",
        required=True,
    ),
    ReportSource(
        key="controls",
        candidates=("controls.csv",),
        what=(
            "the point integrity controls of every arm -- shuffle and untrained -- with "
            "`asserted` and `enforced`. The pipeline-sanity control raises in the training "
            "driver and writes no row, so it is not in this file"
        ),
        required=True,
    ),
    ReportSource(
        key="interval_controls",
        candidates=("interval_controls.csv",),
        what=(
            "the interval controls, on their own schema: a ratio of Winkler scores is not a "
            "ratio of squared errors and the two never share a table"
        ),
        required=False,
    ),
    ReportSource(
        key="pipeline_sanity",
        candidates=("pipeline_sanity.csv",),
        what=(
            "the persistence pipeline-sanity control: persistence through the dataset "
            "against persistence recomputed on the raw Parquet, per (DOF, horizon). Its "
            "statistic is a relative RMSE difference, not a fraction of a null's error "
            "removed, so it is neither in `controls.csv` nor in `interval_controls.csv`"
        ),
        required=False,
    ),
    ReportSource(
        key="probabilistic",
        candidates=("probabilistic_baseline.csv", "probabilistic_by_seed.csv"),
        what=(
            "the residual-interval floor **only** -- one `residual_interval` row per "
            "(regime, DOF, horizon), and no learned head. The heads are in "
            "`e03/probabilistic.csv` and are differenced against this file in section 6.4"
        ),
        required=True,
    ),
    ReportSource(
        key="e03_heads",
        candidates=("../e03/probabilistic.csv", "e03/probabilistic.csv"),
        what="the Phase 5 heads as committed, read for the section 6.4 comparison",
        required=False,
    ),
    ReportSource(
        key="reproducibility",
        candidates=("reference_reproducibility.csv",),
        what="the RevIN arm's closed-form rows against results/e02/ (P6-D13)",
        required=False,
    ),
    ReportSource(
        key="metrics_by_cell",
        candidates=("metrics_by_cell.csv",),
        what="section 6.1 broken out by (vessel, sea state, heading, speed)",
        required=False,
        rendered=False,
    ),
    ReportSource(
        key="quiescence_lead_times",
        candidates=("quiescence_lead_times.csv",),
        what="raw per-match lead times, so the distribution is not only its quantiles",
        required=False,
        rendered=False,
    ),
)


@dataclass(frozen=True)
class LoadedSource:
    """A source CSV that was found, with the row count its markers must state.

    Attributes:
        spec: The declaration it was loaded for.
        path: The file actually read.
        display: The file as the markers name it: relative to the results root.
        rows: Data rows, header excluded.
        frame: The parsed table.
    """

    spec: ReportSource
    path: Path
    display: str
    rows: int
    frame: pd.DataFrame


def load_report_sources(
    results_dir: Path, *, strict: bool = True
) -> tuple[dict[str, LoadedSource], tuple[ReportSource, ...]]:
    """Read every declared source that is present under the results root.

    Args:
        results_dir: The results **root** (``results/``), which holds ``results.md``, the
            Phase 6 CSVs under ``e04/`` and the frozen Gate 3-5 directories beside it. A
            caller that passes ``results/e04`` directly is also honoured, because each
            candidate is looked for under ``<root>/e04`` and then under ``<root>``.
        strict: Whether a missing **required** source is fatal.

    Returns:
        A mapping of source key to the loaded frame, and the specs that were not found.

    Raises:
        FileNotFoundError: Under ``strict``, if any required source is absent. Every
            missing file is named at once rather than one per run, because a caller fixing
            them wants the list.
    """
    loaded: dict[str, LoadedSource] = {}
    missing: list[ReportSource] = []
    for spec in REPORT_SOURCES:
        path = next(
            (
                candidate
                for name in spec.candidates
                for candidate in (results_dir / PHASE6_SUBDIR / name, results_dir / name)
                if candidate.is_file()
            ),
            None,
        )
        if path is None:
            missing.append(spec)
            continue
        frame = pd.read_csv(path)
        loaded[spec.key] = LoadedSource(
            spec=spec,
            path=path,
            display=_relative_to(path, results_dir),
            rows=len(frame),
            frame=frame,
        )
    absent_required = [spec for spec in missing if spec.required]
    if strict and absent_required:
        named = ", ".join(
            f"{spec.key} (looked for {' or '.join(spec.candidates)})" for spec in absent_required
        )
        raise FileNotFoundError(
            f"{len(absent_required)} required result table(s) are missing under "
            f"{results_dir} and {results_dir / PHASE6_SUBDIR}: {named}. A report rendered "
            f"without them would omit a section of docs/IMPLEMENTATION_PLAN.md 6 while "
            f"looking complete, which is what Gate 6 predicate 5 exists to prevent. Pass "
            f"strict=False to render an explicitly incomplete document that names what it "
            f"is missing."
        )
    return loaded, tuple(missing)


# --------------------------------------------------------------------------------------
# Aggregation: the >= 3 seed rule applied to every Phase 6 table, once.
# --------------------------------------------------------------------------------------


def _present(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return the requested columns that exist, in the requested order.

    Args:
        frame: The table.
        columns: Column names in display order.

    Returns:
        The subset that is present. Used for display selection only -- every column a
        table's *meaning* depends on is checked explicitly by its renderer, so that an
        absent base rate or an absent interval width fails rather than quietly disappearing
        from the rendered columns.
    """
    return [name for name in columns if name in frame.columns]


def _group_cols(frame: pd.DataFrame, base: tuple[str, ...]) -> tuple[str, ...]:
    """Prepend ``experiment`` to a grouping key when the table carries one.

    ``scripts/evaluate.py`` scores more than one experiment config into one set of tables,
    and the same label appears in several of them -- ``persistence`` is in every config, by
    construction, because it is the skill denominator. Grouping on the label alone would
    average two *different runs* into one cell: for a closed-form row that is caught by
    :func:`_passthrough_deterministic` refusing two distinct values, but for a stochastic
    row it would silently average six runs and report ``n_seeds = 3``.

    Args:
        frame: The table.
        base: The grouping key the metric is defined on.

    Returns:
        ``base``, with ``"experiment"`` in front when that column exists.
    """
    return ("experiment", *base) if "experiment" in frame.columns else base


def _attach_deterministic(
    frame: pd.DataFrame, lookup: Mapping[str, bool], *, missing_default: bool = True
) -> pd.DataFrame:
    """Give a table the ``deterministic`` flag :func:`aggregate_results` routes on.

    Only the accuracy tables carry the flag natively. The quiescence and ablation tables are
    keyed on the same model labels, so the flag is *read* from the accuracy table rather
    than guessed -- it is a property of the fitting procedure, not of the metric.

    A label absent from the lookup (on this corpus, exactly
    :data:`dmf.eval.quiescence_runner.SYNTHETIC_DETECTORS`) defaults to ``missing_default``.
    Defaulting to True is safe rather than optimistic: :func:`_passthrough_deterministic`
    raises if a group so marked holds more than one distinct value of any metric, so a label
    wrongly defaulted here fails loudly instead of being averaged.

    Args:
        frame: A table with a ``model`` column and no ``deterministic`` column.
        lookup: Model label -> whether its fit carries no RNG.
        missing_default: Flag for labels the lookup does not know.

    Returns:
        A copy carrying ``deterministic``. Returned unchanged if it already had one.
    """
    if "deterministic" in frame.columns:
        return frame
    out = frame.copy()
    out["deterministic"] = [bool(lookup.get(str(label), missing_default)) for label in out["model"]]
    return out


def _deterministic_lookup(metrics_full: pd.DataFrame) -> dict[str, bool]:
    """Build the model -> ``deterministic`` map from the accuracy table.

    Args:
        metrics_full: The per-run accuracy table.

    Returns:
        One entry per model label. A label is deterministic only if *every* one of its rows
        says so; a table disagreeing with itself resolves to False, which routes the label
        through the strict three-seed path rather than the exemption.
    """
    if "deterministic" not in metrics_full.columns:
        return {}
    grouped = metrics_full.groupby("model")["deterministic"].all()
    return {str(label): bool(flag) for label, flag in grouped.items()}


def _aggregated_view(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...],
    metric_cols: tuple[str, ...],
    *,
    source: str,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Return a seed-aggregated view of a per-run table, or pass an aggregate through.

    Every Phase 6 metric table is written **per run** (P6-D10), so the >= 3 seed rule is
    applied here, at render time, by the one implementation that knows the deterministic
    exemption. A table that has already been through that path (``n_seeds`` present, no seed
    column) is passed through so the two producers cannot double-aggregate.

    Args:
        frame: Per-run rows, or an already aggregated table.
        group_cols: Columns identifying a comparison cell.
        metric_cols: Metrics to aggregate. Only those present are used, so a table lacking
            an optional metric renders without it rather than failing.
        source: File name, for the error messages.
        seed_col: Column holding the *training* seed. ``train_seed`` in the quiescence
            tables, where ``seed`` already means the realization seed.

    Returns:
        One row per group with ``<metric>_mean``, ``<metric>_std`` and ``n_seeds``.

    Raises:
        ValueError: If the table is neither per-run nor aggregated -- i.e. holds several
            rows per cell with no seed column, in which case there is no way to tell three
            seeds from three duplicates and no aggregate of it would mean anything.
    """
    metrics = tuple(name for name in metric_cols if name in frame.columns)
    if seed_col in frame.columns:
        working = frame.rename(columns={seed_col: "seed"}) if seed_col != "seed" else frame
        return aggregate_results(working, group_cols, metric_cols=metrics)
    if "n_seeds" in frame.columns:
        absent = [f"{name}_mean" for name in metrics if f"{name}_mean" not in frame.columns]
        if absent:
            raise ValueError(
                f"{source} looks aggregated (it has n_seeds) but is missing {absent}; an "
                f"aggregate written by this module names its metrics '<metric>_mean'"
            )
        return frame
    raise ValueError(
        f"{source} has neither a {seed_col!r} column nor 'n_seeds', so it can be neither "
        f"aggregated nor recognised as an aggregate. Rendering its rows as the table would "
        f"present per-seed rows as a model comparison (CLAUDE.md non-negotiable 5)."
    )


def _envelope_and_carried(
    frame: pd.DataFrame,
    group_cols: tuple[str, ...],
    *,
    first: Sequence[str] = (),
    envelopes: Sequence[tuple[str, str]] = (),
    all_flags: Sequence[str] = (),
) -> pd.DataFrame:
    """Collapse the non-metric columns of a per-run table onto its groups.

    Args:
        frame: Per-run rows.
        group_cols: Columns identifying a cell.
        first: Columns constant within a cell, taken from its first row.
        envelopes: ``(column, "min"|"max")`` pairs. Bootstrap interval bounds are combined
            as an **envelope**, never a mean: no arithmetic on finished intervals recovers a
            calibrated interval for the seed mean, and the mean of them is invariant to the
            seed disagreement a reader consults it to detect (P3-D22).
        all_flags: Boolean columns reduced with ``all`` -- a flag that holds for the cell
            only if it held for every run of it.

    Returns:
        One row per group holding the requested columns.
    """
    keys = list(group_cols)
    grouped = frame.groupby(keys, as_index=False, sort=False)
    out = grouped[keys].first()
    for name in first:
        if name in frame.columns:
            out = out.merge(grouped[[*keys, name]].first(), on=keys, validate="one_to_one")
    for name, how in envelopes:
        if name in frame.columns:
            agg = frame.groupby(keys, sort=False)[name].agg(how).reset_index()
            out = out.merge(agg, on=keys, validate="one_to_one")
    for name in all_flags:
        if name in frame.columns:
            agg = frame.groupby(keys, sort=False)[name].all().reset_index()
            out = out.merge(agg, on=keys, validate="one_to_one")
    return out


# --------------------------------------------------------------------------------------
# Section 6.1 -- the full metric table.
# --------------------------------------------------------------------------------------

#: Cell of the point table.
POINT_GROUP_COLS: tuple[str, ...] = ("model", "regime", "dof", "horizon_samples")

#: Metrics aggregated mean +/- std over seeds. ``fit_time_s`` rides along because a model
#: comparison table carries parameter counts and wall-clock train time (the plan's reporting
#: rules); the phase-lag columns ride along because P6-D3 ships both readings, the
#: interpolated one and the raw argmax, so the interpolation is auditable.
POINT_METRIC_COLS: tuple[str, ...] = (
    "rmse",
    "mae",
    "skill",
    "nrmse",
    "fit_time_s",
    "phase_lag_s",
    "phase_lag_raw_s",
    "dominant_period_s",
)

#: Columns the section cannot be rendered without. ``nrmse`` is here because skill alone is
#: not comparable across horizons on this signal (P3-D5/P4-D3): the persistence denominator
#: oscillates with the roll period, so a skill column read across horizons without its
#: nrmse beside it says something about the signal rather than about the model.
POINT_REQUIRED: tuple[str, ...] = (
    "model",
    "regime",
    "dof",
    "horizon_samples",
    "rmse",
    "skill",
    "nrmse",
    "rmse_persistence",
)


def _phase_lag_display(frame: pd.DataFrame) -> pd.Series:
    """Render the phase-lag column, refusing to print an unidentified lag as a number.

    The estimator is identified only modulo the dominant period (P6-D3), and
    ``phase_lag_identified`` is ``|lag| <= 0.25 * dominant_period_s`` (P6-D10). A 10 s or
    15 s lead on a ~12 s roll period legitimately falls outside that, and printing the
    argmax there would present a property of the signal's own periodicity as a measurement
    of the model's timing.

    Args:
        frame: The aggregated point table.

    Returns:
        A string column: the lag in seconds where it is identified, ``unidentified``
        where it is not, and ``not measured`` where the phase pass did not run.
    """
    if "phase_lag_s_mean" not in frame.columns:
        return pd.Series(["not measured"] * len(frame), index=frame.index, dtype=object)
    identified = (
        frame["phase_lag_identified"].astype(bool)
        if "phase_lag_identified" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    values = frame["phase_lag_s_mean"].astype(float)
    return pd.Series(
        [
            "not measured"
            if not np.isfinite(value)
            else (f"{value:+.3f}" if flag else "unidentified")
            for value, flag in zip(values, identified, strict=True)
        ],
        index=frame.index,
        dtype=object,
    )


def build_point_table(metrics_full: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``metrics_full.csv`` over seeds into the section 6.1 table.

    Args:
        metrics_full: The per-run accuracy table :mod:`dmf.eval.scoring` writes, one row per
            (model, seed, regime, DOF, horizon).

    Returns:
        One row per (model, regime, DOF, horizon), with ``<metric>_mean``/``<metric>_std``,
        ``n_seeds``, the bootstrap skill envelope and the run metadata.

    Raises:
        ValueError: If a column the section's meaning depends on is missing.
    """
    absent = [name for name in POINT_REQUIRED if name not in metrics_full.columns]
    if absent:
        raise ValueError(
            f"metrics_full is missing {absent}; section 6.1 reports rmse, mae, skill, nrmse "
            f"and phase lag, and every skill score is reported beside the persistence RMSE "
            f"it is measured against (CLAUDE.md non-negotiable 4)"
        )
    group = _group_cols(metrics_full, POINT_GROUP_COLS)
    aggregated = _aggregated_view(metrics_full, group, POINT_METRIC_COLS, source="metrics_full.csv")
    carried = _envelope_and_carried(
        metrics_full,
        group,
        first=(
            "horizon_s",
            "n_windows",
            "rmse_persistence",
            "signal_std",
            "n_params",
            "n_realizations_phase",
        ),
        envelopes=(("skill_ci_lo", "min"), ("skill_ci_hi", "max")),
        all_flags=("phase_lag_identified",),
    )
    keys = list(group)
    overlap = [c for c in carried.columns if c not in keys and c in aggregated.columns]
    return aggregated.drop(columns=overlap).merge(carried, on=keys, validate="one_to_one")


def _point_display(table: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns section 6.1 renders.

    Args:
        table: The output of :func:`build_point_table`, all regimes.

    Returns:
        A display frame. ``nrmse_mean`` sits beside ``skill_mean`` and
        ``rmse_persistence`` beside both, so neither can be read without its reference.
    """
    display = table.copy()
    display["phase_lag_s"] = _phase_lag_display(display)
    columns = _present(
        display,
        (
            "experiment",
            "model",
            "regime",
            "dof",
            "horizon_s",
            "n_seeds",
            "deterministic",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "rmse_persistence",
            "skill_mean",
            "skill_std",
            "skill_ci_lo",
            "skill_ci_hi",
            "nrmse_mean",
            "phase_lag_s",
            "n_params",
            "fit_time_s_mean",
        ),
    )
    return display[columns].sort_values(
        _present(display, ("regime", "model", "dof", "horizon_s")),
        ignore_index=True,
        kind="stable",
    )


# --------------------------------------------------------------------------------------
# Section 6.2 -- quiescent-window detection.
# --------------------------------------------------------------------------------------

#: Cell of the quiescence table at the finest reported level. The sea state, the heading and
#: the speed are all **grouping keys, not columns to average over** (P6-D7 item 3, P6-D19):
#: at ``permissive`` the deck is inside limits for essentially the whole record at SS3 and
#: SS4, so a corpus-pooled F1 measures the corpus composition -- and within one sea state,
#: forward speed decides whether the cell is scorable at all, so a sea-state-pooled F1 pools
#: scorable cells with unscorable ones.
QUIESCENCE_GROUP_COLS: tuple[str, ...] = (
    "model",
    "regime",
    "observation_mode",
    "threshold_set",
    "rule",
    "ss",
    "heading_deg",
    "speed_kn",
)

#: Cell of the sea-state roll-up. The heading and the speed are **dropped rather than
#: NaN-grouped**: a NaN group key is silently discarded by ``groupby``, which is how
#: :data:`dmf.eval.assemble.NOT_AN_ARM` came to exist, so the two levels are aggregated
#: separately and concatenated rather than aggregated together.
QUIESCENCE_ROLLUP_GROUP_COLS: tuple[str, ...] = (
    "model",
    "regime",
    "observation_mode",
    "threshold_set",
    "rule",
    "ss",
)

#: Column saying which of the two levels a row is, and the two values it takes. Never
#: inferred from a NaN heading: a roll-up is marked, not recognised.
QUIESCENCE_LEVEL_COL: str = "group_level"

#: Metrics aggregated mean +/- std over training seeds.
QUIESCENCE_METRIC_COLS: tuple[str, ...] = (
    "base_rate",
    "precision",
    "recall",
    "f1",
    "false_alarms_per_min",
    "lead_p10",
    "lead_p50",
    "lead_p90",
    "n_true_onsets",
    "n_pred_onsets",
    "n_matched",
    "duration_s",
    "n_excluded_true",
    "n_excluded_pred",
)

#: Columns the section cannot be rendered without. ``base_rate`` is here because an F1
#: without it is not interpretable (``CLAUDE.md`` §Known traps, plan §6.2) and ``scorable``
#: because a cell with no true onsets must render as neither a success nor a failure.
QUIESCENCE_REQUIRED: tuple[str, ...] = (
    "model",
    "regime",
    "threshold_set",
    "rule",
    "ss",
    "base_rate",
    "f1",
    "precision",
    "recall",
    "scorable",
    "n_true_onsets",
)

#: The F1 interval, combined across training seeds as an envelope and never as a mean, for
#: the reason :func:`_envelope_and_carried` records. It is a bootstrap over **realizations**
#: and is the only uncertainty a deterministic detector's F1 has: aggregating over training
#: seeds gives it ``n_seeds = 1`` and ``f1_std = NaN``.
QUIESCENCE_CI_ENVELOPES: tuple[tuple[str, str], ...] = (
    ("f1_ci_lo", "min"),
    ("f1_ci_hi", "max"),
)

#: What a cell with no scorable onset renders as. Never F1 = 0, which reads as a model
#: failure, and never F1 = 1, which reads as a success: SS3 at ``permissive`` has the deck
#: inside limits for the entire record, so there is nothing there to detect (P6-D7 item 1).
NOT_SCORABLE: str = "not scorable"

#: Columns blanked to :data:`NOT_SCORABLE` on an unscorable row.
#:
#: Three groups of columns are deliberately **not** among them. ``base_rate`` and
#: ``n_true_onsets`` are what say *why* the cell is unscorable. ``n_pred_onsets`` and
#: ``false_alarms_per_min`` stay because they are defined without any true onset at all: a
#: detector that flags an onset where there is none to flag is exactly what a false alarm
#: is, and blanking that number would hide the one thing still measurable in the cell.
#: Precision, recall, F1 and the lead-time quantiles all need a true onset to exist.
_UNSCORABLE_COLUMNS: tuple[str, ...] = (
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "f1_ci_lo",
    "f1_ci_hi",
    "lead_p10_mean",
    "lead_p50_mean",
    "lead_p90_mean",
)


def build_quiescence_table(
    quiescence: pd.DataFrame, deterministic: Mapping[str, bool]
) -> pd.DataFrame:
    """Aggregate ``quiescence.csv`` over training seeds into the section 6.2 table.

    Args:
        quiescence: The per-run operational table, keyed on ``train_seed`` rather than
            ``seed`` because ``seed`` already means the realization seed in the lead-time
            table beside it.
        deterministic: Model label -> whether its fit carries no RNG, read from the accuracy
            table so that the exemption is the same one section 6.1 applied.

    Returns:
        One row per (model, regime, observation mode, threshold set, rule, sea state).

    Raises:
        ValueError: If a column the section's meaning depends on is missing.
    """
    absent = [name for name in QUIESCENCE_REQUIRED if name not in quiescence.columns]
    if absent:
        raise ValueError(
            f"quiescence.csv is missing {absent}; every F1 is reported beside its base rate "
            f"and grouped by sea state, and a cell with no scorable onset is reported as "
            f"'{NOT_SCORABLE}' rather than as an F1 (docs/protocol.md P6-D7)"
        )
    frame = _attach_deterministic(quiescence, deterministic)
    if QUIESCENCE_LEVEL_COL not in frame.columns:
        # A table written before P6-D19 has one level and does not say so. Treated as the
        # cell level only if it carries the cell axes; otherwise it *is* the roll-up, and
        # labelling it as a cell would claim a resolution it never had.
        level = CELL_LEVEL if {"heading_deg", "speed_kn"} <= set(frame.columns) else SEA_STATE_LEVEL
        frame = frame.assign(**{QUIESCENCE_LEVEL_COL: level})
    parts: list[pd.DataFrame] = []
    for level, columns in (
        (CELL_LEVEL, QUIESCENCE_GROUP_COLS),
        (SEA_STATE_LEVEL, QUIESCENCE_ROLLUP_GROUP_COLS),
    ):
        scoped = frame.loc[frame[QUIESCENCE_LEVEL_COL].astype(str) == level]
        if scoped.empty:
            continue
        group = _group_cols(
            scoped,
            (QUIESCENCE_LEVEL_COL, *(name for name in columns if name in scoped.columns)),
        )
        aggregated = _aggregated_view(
            scoped,
            group,
            QUIESCENCE_METRIC_COLS,
            source="quiescence.csv",
            seed_col="train_seed",
        )
        carried = _envelope_and_carried(
            scoped,
            group,
            first=("n_realizations", "n_boot", "ci_level", "bootstrap_seed"),
            envelopes=QUIESCENCE_CI_ENVELOPES,
            all_flags=("scorable",),
        )
        keys = list(group)
        overlap = [c for c in carried.columns if c not in keys and c in aggregated.columns]
        parts.append(
            aggregated.drop(columns=overlap).merge(carried, on=keys, validate="one_to_one")
        )
    if not parts:
        return frame.iloc[:0]
    return pd.concat(parts, ignore_index=True)


def _blank_unscorable(table: pd.DataFrame) -> pd.DataFrame:
    """Replace every score on an unscorable cell with :data:`NOT_SCORABLE`.

    Args:
        table: An aggregated quiescence table.

    Returns:
        A copy. Rows whose ``scorable`` is False carry :data:`NOT_SCORABLE` in the columns
        that need a true onset to mean anything, and their real numbers everywhere else.
    """
    display = table.copy()
    scorable = display["scorable"].astype(bool) if "scorable" in display else None
    if scorable is not None and not bool(scorable.all()):
        for name in _UNSCORABLE_COLUMNS:
            if name in display.columns:
                display[name] = display[name].astype(object)
                display.loc[~scorable, name] = NOT_SCORABLE
    return display


def _quiescence_display(table: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns section 6.2 renders, blanking unscorable cells.

    Args:
        table: The output of :func:`build_quiescence_table`, all regimes and both
            threshold sets. The grouping keys stay on the row rather than in a heading, so
            a reader cannot lose track of which cell a number belongs to and a checker can
            read the grouping off the header.

    Returns:
        A display frame with ``base_rate_mean`` and ``n_true_onsets_mean`` immediately
        before precision, recall and F1, and :data:`NOT_SCORABLE` in place of every score
        on a cell with no scorable onset.
    """
    display = _blank_unscorable(table)
    columns = _present(
        display,
        (
            "experiment",
            "model",
            "regime",
            "threshold_set",
            "rule",
            "group_level",
            "ss",
            "heading_deg",
            "speed_kn",
            "n_seeds",
            "scorable",
            "base_rate_mean",
            "n_true_onsets_mean",
            "n_pred_onsets_mean",
            "n_matched_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "f1_std",
            # Beside the F1 and not in a table of its own: `f1_std` is the spread over
            # training seeds and is NaN for every deterministic detector here, so this
            # bootstrap over realizations is the only uncertainty most of these rows carry.
            "f1_ci_lo",
            "f1_ci_hi",
            "false_alarms_per_min_mean",
            "lead_p10_mean",
            "lead_p50_mean",
            "lead_p90_mean",
            "duration_s_mean",
            "n_excluded_true_mean",
            "n_excluded_pred_mean",
        ),
    )
    return display[columns].sort_values(
        _present(
            display,
            ("regime", "threshold_set", "ss", "heading_deg", "speed_kn", "rule", "model"),
        ),
        ignore_index=True,
        kind="stable",
    )


def _quiescence_base_rate_display(table: pd.DataFrame) -> pd.DataFrame:
    """Render the base rate on its own, as a property of the truth.

    The base rate is printed beside every F1 as well; this table exists because it is not a
    model quantity at all -- it is what decides whether a cell is scorable, and P6-D7 shows
    it decides that independently of any model. ``n_true_onsets`` is the quantity scoring
    actually turns on: a cell can sit at base rate 1.000 and hold nothing to detect.

    Args:
        table: The output of :func:`build_quiescence_table`.

    Returns:
        One row per (regime, threshold set, sea state), with ``truth_consistent`` recording
        whether every model in the cell saw the same truth. A False there would mean two
        models were scored against different ground truth, which is a defect and not a
        result, so it is a column rather than an assumption.
    """
    keys = _present(
        table, ("group_level", "regime", "threshold_set", "ss", "heading_deg", "speed_kn")
    )
    columns = _present(
        table,
        (
            "scorable",
            "base_rate_mean",
            "n_true_onsets_mean",
            "n_realizations",
            "duration_s_mean",
            "n_excluded_true_mean",
        ),
    )
    grouped = table.groupby(keys, as_index=False, sort=True)
    out = grouped[keys].first()
    for name in columns:
        out = out.merge(grouped[[*keys, name]].first(), on=keys, validate="one_to_one")
    consistent = pd.Series(True, index=out.index)
    for name in ("base_rate_mean", "n_true_onsets_mean"):
        if name in table.columns:
            spread = table.groupby(keys, sort=True)[name].agg(lambda s: float(s.max() - s.min()))
            consistent &= np.isclose(spread.to_numpy(), 0.0, atol=1e-9)
    out["truth_consistent"] = consistent.to_numpy()
    return out


def _quiescence_lead_time_display(table: pd.DataFrame) -> pd.DataFrame:
    """Render the lead-time distribution and the false-alarm rate.

    Args:
        table: The output of :func:`build_quiescence_table`.

    Returns:
        A display frame. The base rate rides along because it is what makes the false-alarm
        rate readable: a detector on a deck that is quiet 95% of the time has far more
        opportunity to be right by accident.
    """
    display = _blank_unscorable(table)
    columns = _present(
        display,
        (
            "experiment",
            "model",
            "regime",
            "threshold_set",
            "rule",
            "group_level",
            "ss",
            "heading_deg",
            "speed_kn",
            "n_seeds",
            "scorable",
            "base_rate_mean",
            "n_matched_mean",
            "lead_p10_mean",
            "lead_p50_mean",
            "lead_p90_mean",
            "false_alarms_per_min_mean",
            "duration_s_mean",
        ),
    )
    return display[columns].sort_values(
        _present(
            display,
            ("regime", "threshold_set", "ss", "heading_deg", "speed_kn", "rule", "model"),
        ),
        ignore_index=True,
        kind="stable",
    )


# --------------------------------------------------------------------------------------
# Section 6.3 -- the five ablations.
# --------------------------------------------------------------------------------------

#: Cell of the ablation table. ``logical_dof`` rather than ``dof``: the ``imu`` arm's rows
#: are keyed on the resolved corpus columns (``roll_imu``), and a contrast joined on ``dof``
#: would silently match nothing.
ABLATION_GROUP_COLS: tuple[str, ...] = (
    "ablation",
    "arm",
    "reference_arm",
    "model",
    "regime",
    "logical_dof",
    "horizon_samples",
)

#: Metrics aggregated over seeds. Only ``skill`` and ``nrmse`` cross an observation-mode
#: boundary (P6-D4 item 4); ``dmf.eval.ablations.assert_columns_comparable`` enforces that
#: upstream, and this module renders whichever columns the contrast table carries.
ABLATION_METRIC_COLS: tuple[str, ...] = (
    "skill",
    "skill_reference",
    "skill_diff",
    "nrmse",
    "nrmse_reference",
    "nrmse_diff",
)

#: The paired interval on ``skill_diff``, combined across seeds as an **envelope** and never
#: as a mean, for the reason :func:`_envelope_and_carried` records: no arithmetic on finished
#: intervals recovers a calibrated interval for the seed mean.
#:
#: These columns exist because ``skill_diff_std`` is not an answer. It is the spread over
#: *training* seeds, which is NaN for every deterministic vehicle -- and a deterministic
#: vehicle is the only vehicle four of the five arms have on the out-of-distribution regimes
#: -- and where it is defined it measures initialisation and data order rather than the
#: realization sampling that decides whether a skill difference is real (P4-D13).
ABLATION_CI_ENVELOPES: tuple[tuple[str, str], ...] = (
    ("skill_diff_ci_lo", "min"),
    ("skill_diff_ci_hi", "max"),
)

#: Flags that must travel with an ablation row, because a zero contrast that is an
#: architectural artefact must not read as evidence of no effect (P6-D8).
#:
#: - ``vehicle_blind_to_arm`` -- DLinear slices ``x[:, :, :C_out]``, so it is bitwise blind
#:   to the sea-state indicator and its contrast is exactly zero by construction.
#: - ``capacity_confounded`` -- AR(20) gains 66% more parameters from the one-hot's four
#:   extra channels, so a win there is not attributable to sea-state information.
#: - ``not_fittable`` -- AR(20) on ``ss_conditioned``/``unseen_seastate`` has a singular
#:   design because the SS6 indicator is constant zero in training.
#: - ``privileged_information`` -- the sea state is estimated online at deployment, not
#:   known, so no row of that arm is deployable. It was called ``upper_bound`` until P6-D18
#:   measured that the arm is **not** an upper bound: conditioning costs -0.0445 mean skill
#:   on ``unseen_seastate``, so the "bound" lies below the baseline it was to bound.
#: - ``parameter_matched`` -- measured per row from ``param_delta_frac`` and therefore the
#:   exact negation of ``capacity_confounded``. ``arm_parameter_matched`` beside it is the
#:   arm's design intent, which is a different and coarser claim: the channels arm is built
#:   to hold parameters fixed and does so for its closed-form vehicle only, so its ``tcn``
#:   rows are arm-matched and row-unmatched at -15.25%.
#: - ``architecture_differs`` -- the contrast pairs two networks rather than one network
#:   under two conditions. True only on ``lookback_40s``, whose vehicle ``tcn_l400`` carries
#:   the extra dilation stage a 40 s receptive field needs (221 636 parameters against
#:   ``tcn``'s 196 804). The join that renders the row is on the logical vehicle
#:   (:func:`dmf.eval.ablations.with_logical_model`); this flag and ``model_reference`` are
#:   what stop it reading as one network at two input lengths.
#: - ``on_residual_floor`` -- the P1-D2 floor, and the last artefact of the same kind. The
#:   others stop an artefactual *zero* reading as "no effect"; this one stops an artefactual
#:   *large number* reading as an effect. Five arms score
#:   ``unseen_heading``/``ar20``/``pitch_rate``/10 s and their ``skill_diff`` runs
#:   -0.000003 (``revin``), +0.000585 (``ss_conditioned``), +3.29 (``lookback_40s``),
#:   +30.30 (``lookback_10s``) and +41.67 (``imu``) against ``signal_std = 0.0658 deg/s``,
#:   while the unfloored rows of the same file have median ``|skill_diff| = 0.0002``. On a
#:   26 dB suppressed channel the skill denominator is floor too, so the ratio is arithmetic
#:   about the floor. (P6-D20 retracted a ``+30.27`` quoted here from the audit rather than
#:   derived; the whole set is printed instead of one member of it, per P6-D17.) Derived by
#:   :func:`dmf.eval.ablations.cell_floored_dofs` from
#:   :func:`dmf.eval.controls.floored_dofs` -- the same derivation the shuffle-control
#:   narrowing uses (P6-D12) -- so the two cannot disagree.
ABLATION_FLAG_COLUMNS: tuple[str, ...] = (
    "vehicle_blind_to_arm",
    "capacity_confounded",
    "not_fittable",
    "privileged_information",
    "parameter_matched",
    "arm_parameter_matched",
    "architecture_differs",
    "on_residual_floor",
)

#: The arm whose closed-form rows do not belong in its own ablation table, and the vehicles
#: that do. ``dmf.models.base.revin_applies`` is True only for ``FIT_KIND == "sgd"``, so
#: every closed-form row of the RevIN arm must equal the reference arm's by construction.
#: Listing five "no effect" rows in a table about RevIN's effect would read as a result;
#: they are a **reproducibility control** on reading ``results/e02/`` as the reference arm,
#: and they are rendered under that heading instead (P6-D13).
REVIN_ARM: str = "revin"

#: The vehicles RevIN applies to, and therefore the only rows the RevIN ablation table lists.
REVIN_VEHICLES: frozenset[str] = frozenset({"tcn", "tcn_l400"})


def split_ablation_rows(ablations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the contrast table into the rows each part of section 6.3 renders.

    Args:
        ablations: The per-run contrast table.

    Returns:
        ``(contrasts, unfittable, revin_closed_form)``. The second holds rows that have no
        fit at all and carry the reason; the third holds the RevIN arm's closed-form rows,
        which are a reproducibility control rather than an ablation result (P6-D13). Both
        are **reported**, never dropped -- CLAUDE.md non-negotiable 6 forbids dropping a
        row, and both of these would otherwise be read as an effect of zero.
    """
    frame = ablations
    if "not_fittable" in frame.columns:
        reason = frame["not_fittable"].fillna("").astype(str).str.strip()
        unfittable = frame.loc[reason != ""]
        frame = frame.loc[reason == ""]
    else:
        unfittable = frame.iloc[:0]
    if "arm" in frame.columns and "model" in frame.columns:
        is_revin_closed_form = (frame["arm"] == REVIN_ARM) & ~frame["model"].astype(str).isin(
            REVIN_VEHICLES
        )
        revin_closed_form = frame.loc[is_revin_closed_form]
        frame = frame.loc[~is_revin_closed_form]
    else:
        revin_closed_form = frame.iloc[:0]
    return frame, unfittable, revin_closed_form


def build_ablation_table(
    ablations: pd.DataFrame, deterministic: Mapping[str, bool]
) -> pd.DataFrame:
    """Aggregate the contrast table over seeds into the section 6.3 table.

    Args:
        ablations: The contrast rows, already split by :func:`split_ablation_rows`.
        deterministic: Model label -> whether its fit carries no RNG.

    Returns:
        One row per (ablation, arm, model, regime, logical DOF, horizon).

    Raises:
        ValueError: If a column the section's meaning depends on is missing.
    """
    required = ("ablation", "arm", "model", "regime", "logical_dof", "horizon_samples")
    absent = [name for name in required if name not in ablations.columns]
    if absent:
        raise ValueError(f"ablations.csv is missing {absent}")
    if not any(name in ablations.columns for name in ABLATION_METRIC_COLS):
        raise ValueError(
            f"ablations.csv carries none of {list(ABLATION_METRIC_COLS)}; an ablation table "
            f"with no metric is not a table"
        )
    frame = _attach_deterministic(ablations, deterministic, missing_default=False)
    group = _group_cols(frame, tuple(name for name in ABLATION_GROUP_COLS if name in frame.columns))
    aggregated = _aggregated_view(frame, group, ABLATION_METRIC_COLS, source="ablations.csv")
    carried = _envelope_and_carried(
        frame,
        group,
        first=(
            "horizon_s",
            "dof",
            "observation_mode",
            "lookback",
            "reference_lookback",
            "n_params",
            "n_params_reference",
            # Carried beside the flag, not only used to compute it: `capacity_confounded`
            # is a threshold on this number, and a flag whose input is not printed cannot
            # be checked. NaN means the flag was declared rather than measured, which on
            # this corpus happens only on the rows that have no fit.
            "param_delta_frac",
            "note",
            # Carried, because an interval that is absent must say why on the row that
            # lacks it. `ci_resample_unit` rides along for the same reason: a reader must
            # not have to know from elsewhere that one draw here is a grid cell rather than
            # a realization (dmf.eval.ablations.CI_RESAMPLE_UNIT).
            "skill_diff_ci_reason",
            "ci_resample_unit",
            "ci_n_units",
            "ci_n_realizations",
            "n_boot",
            "ci_level",
            "bootstrap_seed",
            *ABLATION_FLAG_COLUMNS,
        ),
        envelopes=ABLATION_CI_ENVELOPES,
    )
    keys = list(group)
    overlap = [c for c in carried.columns if c not in keys and c in aggregated.columns]
    return aggregated.drop(columns=overlap).merge(carried, on=keys, validate="one_to_one")


def _ablation_display(table: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns section 6.3 renders.

    Args:
        table: The output of :func:`build_ablation_table`, scoped to one ablation.

    Returns:
        A display frame carrying every flag column the source table had, and ``nrmse``
        beside ``skill`` wherever both were contrasted.
    """
    table = table.copy()
    if "not_fittable" in table.columns:
        # Read back from CSV an empty reason is NaN, which `to_markdown` renders "n/a" --
        # "not measured" where the truth is "this row fits fine". Blank it explicitly.
        table["not_fittable"] = table["not_fittable"].fillna("").astype(str)
    columns = _present(
        table,
        (
            "arm",
            "reference_arm",
            "model",
            "regime",
            "logical_dof",
            "horizon_s",
            "n_seeds",
            "lookback",
            "n_params",
            "param_delta_frac",
            "skill_mean",
            "skill_std",
            "skill_reference_mean",
            "skill_diff_mean",
            "skill_diff_std",
            # Beside the difference, never in a separate table: `skill_diff_std` is the
            # spread over training seeds and is NaN on every deterministic vehicle, so the
            # paired interval is the only uncertainty most of these rows have.
            "skill_diff_ci_lo",
            "skill_diff_ci_hi",
            "ci_resample_unit",
            "ci_n_units",
            "skill_diff_ci_reason",
            "nrmse_mean",
            "nrmse_reference_mean",
            "nrmse_diff_mean",
            *ABLATION_FLAG_COLUMNS,
        ),
    )
    order = _present(table, ("arm", "regime", "model", "logical_dof", "horizon_s"))
    return table[columns].sort_values(order, ignore_index=True, kind="stable")


# --------------------------------------------------------------------------------------
# Section 6.4 -- the probabilistic floor and the heads scored beside it.
# --------------------------------------------------------------------------------------

#: Where the 1-5 s band ends, seconds. The corpus horizons are 1, 2, 3, 5, 10 and 15 s.
BAND_SPLIT_S: float = 5.0

#: Labels of the two bands P5-D15 forbids pooling.
SHORT_BAND: str = "1-5 s"
LONG_BAND: str = "10-15 s"


def horizon_band(horizon_s: float) -> str:
    """Label a horizon with the band its coverage may be read in.

    P5-D15: the deep heads are calibrated 12 of 12 at 10-15 s on ``id`` and over-cover in
    18-23 of 24 cells at 1-5 s, so a coverage figure averaged over both bands averages two
    opposite behaviours and describes neither. Every coverage row this module renders
    carries this label, and no rendered row spans two of them.

    Args:
        horizon_s: Lead time, seconds.

    Returns:
        :data:`SHORT_BAND`, :data:`LONG_BAND`, or the horizon's own label if the corpus grows
        a lead time outside both -- an unknown horizon gets a band of its own rather than
        being folded into whichever is nearer.
    """
    if horizon_s <= BAND_SPLIT_S:
        return SHORT_BAND
    if horizon_s <= 15.0:
        return LONG_BAND
    return f"{horizon_s:g} s"


def build_probabilistic_view(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Return the section 6.4 table, aggregating a per-run frame if that is what it is.

    Args:
        frame: Either the per-run probabilistic table or an already aggregated one.
        source: File name, for the error messages.

    Returns:
        The aggregated table with a ``band`` column added.

    Raises:
        ValueError: If the frame is neither, or if it carries a coverage without a width.
    """
    if "experiment" in frame.columns and int(frame["experiment"].nunique()) > 1:
        # The Phase 5 schema is keyed on (model, head, regime, dof, horizon) and has no
        # room for the experiment that produced the row, so two configs scoring the same
        # label would be averaged into one cell. Refused rather than merged: the fix is to
        # score them into separate files, not to average across them here.
        raise ValueError(
            f"{source} holds rows from {sorted(frame['experiment'].unique())}; the "
            f"probabilistic schema is not keyed on the experiment, so aggregating it here "
            f"would average two different runs of the same label into one row"
        )
    if "n_seeds" in frame.columns and "seed" not in frame.columns:
        table = frame.copy()
    else:
        table = build_probabilistic_table(frame)
    for pair in (("picp_mean", "mean_interval_width_mean"),):
        coverage, width = pair
        if coverage in table.columns and width not in table.columns:
            raise ValueError(
                f"{source} states a coverage ({coverage}) with no interval width beside it. "
                f"Coverage without sharpness is meaningless -- a maximally wide interval has "
                f"perfect coverage -- so the two are reported together or not at all."
            )
    if "horizon_s" not in table.columns:
        raise ValueError(
            f"{source} has no horizon_s column, so its coverage rows cannot be labelled with "
            f"the band they belong to and could not be shown not to pool them (P5-D15)"
        )
    table["band"] = [horizon_band(float(value)) for value in table["horizon_s"]]
    return table


def _probabilistic_display(table: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns section 6.4 renders.

    Args:
        table: The output of :func:`build_probabilistic_view`, all regimes.

    Returns:
        A display frame in which ``picp_mean`` is followed immediately by the two sharpness
        columns and preceded by the band, so no coverage number can be read on its own.
    """
    columns = _present(
        table,
        (
            "model",
            "head",
            "regime",
            "dof",
            "band",
            "horizon_s",
            "n_seeds",
            "picp_mean",
            "picp_std",
            "picp_ci_lo",
            "picp_ci_hi",
            "mean_interval_width_mean",
            "width_ratio_mean",
            "signal_std",
            "winkler_mean",
            "crps_mean",
            "pinball_mean",
            "crossing_rate_mean",
            "n_params",
        ),
    )
    order = _present(table, ("regime", "model", "head", "dof", "horizon_s"))
    return table[columns].sort_values(order, ignore_index=True, kind="stable")


#: What a head is contrasted against, and the keys the floor is unique on. The floor is
#: unconditional, so it has exactly one row per ``(regime, dof, horizon)`` -- no model and no
#: head enter it, which is precisely the property that makes it a floor (P6-D6).
FLOOR_JOIN_KEYS: tuple[str, ...] = ("regime", "dof", "horizon_samples")

#: The floor model label, as ``probabilistic_baseline.csv`` writes it.
FLOOR_MODEL: str = "residual_interval"


def build_floor_contrast(heads: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """Difference every learned head against the unconditional floor, cell by cell.

    P6-D6 defines the floor and states the argument this table is the rendering of: the
    floor is unconditional, so on ``id`` it is near-perfectly calibrated **by construction**
    and a learned head beats it only by making its interval *conditional* -- narrower where
    the deck is predictable, wider where it is not. A head that matches the floor on coverage
    while matching it on width has learned nothing about its own uncertainty, however good
    its PICP looks in isolation. Until this table existed that argument was rendered as prose
    beside two tables that were never differenced, so a reader had to do the join by eye
    across 864 and 144 rows.

    ``width_ratio`` is the sharpness axis and it is the one to read: both sides are
    normalised by the same ``signal_std``, so ``width_ratio_diff`` is dimensionless and
    negative means the head is **sharper** than the floor. ``picp_diff`` is the coverage
    axis, and it is meaningless on its own -- a head that is wider everywhere will cover
    better everywhere.

    Args:
        heads: The aggregated learned-head table (``results/e03/probabilistic.csv``).
        floor: The aggregated floor table (``results/e04/probabilistic_baseline.csv``).

    Returns:
        One row per head row, carrying the floor's coverage and width beside the head's own
        and the two differences. Head rows with no floor cell are **kept**, with NaN
        differences: dropping them would silently narrow the comparison to the cells the
        floor happens to cover.

    Raises:
        ValueError: If either side is missing a column the contrast is defined by, or if the
            floor is not unique on :data:`FLOOR_JOIN_KEYS` -- a floor with two rows per cell
            is not the unconditional object P6-D6 defines, and joining it would duplicate
            every head row silently.
    """
    needed = (*FLOOR_JOIN_KEYS, "picp_mean", "width_ratio_mean")
    for name, frame in (("the heads table", heads), ("the floor table", floor)):
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing {missing}, so no floor contrast is defined")
    keys = list(FLOOR_JOIN_KEYS)
    # Narrowed to the floor's own rows first. The file is named for the floor and on the
    # production tree holds nothing else, but a caller handing in a wider table would
    # otherwise fan every head row out against whatever else was in it -- and the failure
    # would look like a duplicated-row bug rather than like the wrong comparator.
    if "model" in floor.columns:
        floor = floor.loc[floor["model"].astype(str) == FLOOR_MODEL]
        if floor.empty:
            raise ValueError(
                f"the floor table carries no {FLOOR_MODEL!r} rows, so there is no "
                f"unconditional comparator to difference the heads against (P6-D6)"
            )
    if bool(floor.duplicated(subset=keys).any()):
        raise ValueError(
            f"the floor table is not unique on {keys}; the floor is unconditional by "
            f"definition (P6-D6), so more than one row per cell means it is not the object "
            f"this contrast is against"
        )
    right = floor[[*keys, "picp_mean", "width_ratio_mean", "mean_interval_width_mean"]].rename(
        columns={
            "picp_mean": "floor_picp",
            "width_ratio_mean": "floor_width_ratio",
            "mean_interval_width_mean": "floor_mean_interval_width",
        }
    )
    merged = heads.merge(right, on=keys, how="left", validate="many_to_one")
    merged["picp_diff"] = merged["picp_mean"] - merged["floor_picp"]
    merged["width_ratio_diff"] = merged["width_ratio_mean"] - merged["floor_width_ratio"]
    return merged


def _floor_contrast_display(table: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns the head-minus-floor table renders.

    Args:
        table: The output of :func:`build_floor_contrast`.

    Returns:
        A display frame in which each difference sits beside both numbers it was taken
        between, so no difference can be read without its two sides.
    """
    columns = _present(
        table,
        (
            "model",
            "head",
            "regime",
            "dof",
            "band",
            "horizon_s",
            "n_seeds",
            "picp_mean",
            "floor_picp",
            "picp_diff",
            "width_ratio_mean",
            "floor_width_ratio",
            "width_ratio_diff",
            "mean_interval_width_mean",
            "floor_mean_interval_width",
            "signal_std",
        ),
    )
    order = _present(table, ("regime", "model", "head", "dof", "horizon_s"))
    return table[columns].sort_values(order, ignore_index=True, kind="stable")


# --------------------------------------------------------------------------------------
# The integrity controls.
# --------------------------------------------------------------------------------------


def _control_summary(controls: pd.DataFrame, *, by: Sequence[str]) -> pd.DataFrame:
    """Summarise control rows to one row per control.

    Args:
        controls: Control rows.
        by: Grouping columns; those absent from the frame are ignored.

    Returns:
        One row per group with the worst excess, the tolerance, the verdict over the group,
        whether the group is enforced, and the number of cells it covers. ``worst_excess``
        is a maximum because the criterion is one-sided: the subject may be arbitrarily
        worse than the null and only an *improvement* on it is evidence of leakage.

        ``enforced`` is reduced with ``all``, so a group mixing enforced and unenforced rows
        reads False. That is the conservative direction: it under-claims enforcement rather
        than over-claiming it, and a group that mixes the two is a grouping bug worth seeing.
        A table with no ``enforced`` column predates the distinction and gets no such column
        here -- inventing one would assert a commitment the artifact never recorded.
    """
    keys = _present(controls, by)
    if not keys:
        raise ValueError(
            f"none of the grouping columns {list(by)} are in the control table, whose "
            f"columns are {list(controls.columns)}"
        )
    named: dict[str, tuple[str, str]] = {"n_cells": (keys[0], "size")}
    if "excess" in controls.columns:
        named["worst_excess"] = ("excess", "max")
    if "tol" in controls.columns:
        named["tol"] = ("tol", "first")
    if "passed" in controls.columns:
        named["passed"] = ("passed", "all")
    if "enforced" in controls.columns:
        named["enforced"] = ("enforced", "all")
    return controls.groupby(keys, as_index=False, sort=True).agg(**named)


def split_controls(controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split control rows into the asserted ones and the reported-only ones.

    P6-D12: a cell whose test-set signal sits on its P1-D2 residual floor is computed,
    written and shown, but does not take part in the control's pass/fail decision. The two
    are rendered as two tables and never folded into one verdict -- a control's verdict and
    the cells it declined to assert on are different statements.

    Args:
        controls: The control table.

    Returns:
        ``(asserted, reported_only)``. A table with no ``asserted`` column predates the
        narrowing and is treated as fully asserted, which is what it was.
    """
    if "asserted" not in controls.columns:
        return controls, controls.iloc[:0]
    flag = controls["asserted"].astype(bool)
    return controls.loc[flag], controls.loc[~flag]


# --------------------------------------------------------------------------------------
# The document.
# --------------------------------------------------------------------------------------

#: Ablation key -> the table slug the Gate 6 read-out's required-table list uses. Absent
#: keys fall back to ``ablation_<key>``; the map exists because the arm registry calls the
#: sea-state ablation ``sea_state_conditioning`` and the gate calls its table
#: ``ablation_ss_conditioning``, and a slug that differs by a synonym is a table the gate
#: reports as missing while the reader is looking straight at it.
ABLATION_TABLE_IDS: Mapping[str, str] = {
    "observation_mode": "ablation_observation_mode",
    "channels": "ablation_channels",
    "sea_state_conditioning": "ablation_ss_conditioning",
    "lookback": "ablation_lookback",
    "normalization": "ablation_normalization",
}


def ablation_table_id(ablation: str) -> str:
    """Return the table slug for one ablation.

    Args:
        ablation: The ablation key as :mod:`dmf.eval.ablations` spells it.

    Returns:
        The slug, from :data:`ABLATION_TABLE_IDS` or derived.
    """
    return ABLATION_TABLE_IDS.get(ablation, f"ablation_{ablation}")


def results_report_caveats() -> tuple[str, ...]:
    """Return the caveats every reading of this document is subject to.

    Returns:
        One string per caveat. These are properties of the corpus and of the reporting
        rules, not of any particular run, so they are fixed text rather than derived from
        the tables -- a caveat that disappears when a table is empty is a caveat that was
        never enforced.
    """
    return (
        "**Simulated results only.** The corpus is JONSWAP-driven vessel simulation; no "
        "real deck data enters this project at any point, and no number here is evidence "
        "about a real ship.",
        "**The task is easy for a structural reason, and every absolute number here is "
        "flattered by it** (P3-D1). There is no process noise anywhere in the generator and "
        "the vessel RAO is a narrowband filter, so the deck record is a finite sum of "
        "sinusoids -- which satisfies an *exact* linear recursion. Least-squares AR over "
        "200 lags x 6 channels is therefore **identifying** the system rather than "
        "approximating it, which is why the short-lead skill in section 6.1 runs to "
        "0.9976-1.0000 at 1 s on `id` (`ar40` 0.9991-1.0000, `dlinear_ols` 0.9996-1.0000). "
        "The shuffle control settles that this is not leakage: refit on time-shuffled "
        "targets the same estimator scores `excess = 0.001` against a 2% tolerance. Read "
        "**relative** model comparisons and **OOD degradation** as findings; do not read an "
        "absolute skill here as an achievable-accuracy claim.",
        "**Every table is mean +/- std over >= 3 seeds**, except closed-form rows, which "
        "carry `deterministic = yes`, `n_seeds = 1` and a `n/a` std -- with one observation "
        "the sample standard deviation is undefined, and `0.0` would claim a measurement "
        "that was never made (P3-D10). `deterministic = yes` means 'carries no RNG', not "
        "'bit-identical across runs' (P6-D13).",
        "**`skill_std` and `skill_ci_lo`/`skill_ci_hi` are different quantities.** The first "
        "is seed-to-seed spread; the second is a bootstrap over held-out realizations, and "
        "on a multi-seed row it is the **envelope** of the per-run intervals, not a "
        "calibrated interval for the seed mean (P3-D22).",
        "**Skill is not comparable across horizons on this signal** (P3-D5). The persistence "
        "denominator oscillates with the roll period, so `nrmse` is printed beside every "
        "skill score and is the column to read down a horizon column with.",
        "**Phase lag is identified only modulo the dominant period** (P6-D3). Rows where "
        "`|lag| > 0.25 * dominant_period_s` render as `unidentified` rather than as a "
        "number; the tapered estimator also shrinks `|lag|` toward zero, so a lag near 0 is "
        "not by itself evidence of perfect timing (P6-D10).",
        "**Every F1 is printed beside its base rate, and the reporting cell is (sea state, "
        "heading, speed) rather than the sea state** (P6-D7 item 3, corrected by P6-D19). "
        "A cell with no scorable onset renders `not scorable`, because both F1 = 0 and "
        "F1 = 1 would be wrong about a cell holding nothing to detect. **Scorability is a "
        "property of the cell, not of the sea state**: P6-D7's claim that SS3 at "
        "`permissive` has no scorable onset was measured on a speed-0 subsample and is "
        "retracted. Forward speed shifts the encounter frequency, so 10 of SS3's 12 cells "
        "at `permissive` are unscorable and two are not -- SS3/45 deg/12 kn carries 11.12 "
        "scorable onsets per realization, and the base-rate table below counts 410 true "
        "onsets in it on `id`. A sea-state roll-up pools scorable cells with unscorable "
        "ones and is marked `group_level = sea_state` for that reason.",
        "**Coverage and sharpness are printed together and never pooled across horizon "
        "bands** (P5-D11, P5-D15). A maximally wide interval has perfect coverage, and the "
        "deep heads are calibrated at 10-15 s while over-covering at 1-5 s, so one number "
        "across both averages two opposite behaviours.",
        "**Coverage degradation under `unseen_seastate` is reported, not fixed** (P5-D12). "
        "The degradation is the finding.",
        "**The `imu` arm is not raw-RMSE-comparable to `ideal`, but its skill is** "
        "(P6-D4 item 4 as corrected by P6-D23). Both input and target are `imu` on that arm "
        "(P2-D8), so the two modes forecast different targets with different `signal_std` "
        "and no raw-scale column may cross the boundary. The **persistence denominators**, "
        "however, agree to about 1%: 0.4361 against 0.4417 at heave / 1 s on `id`, a ratio "
        "of **0.9872**, and 0.9706-1.0008 with mean **0.9963** across every DOF and horizon. "
        "The published claim that the `imu` denominator is 'roughly halved' (0.218 vs 0.437) "
        "quoted P1-D6's **cross-mode** pairing -- `heave_imu` at t forecasting *true* heave "
        "at t+h -- which P2-D8 makes structurally impossible and this arm does not use; it "
        "was wrong by roughly 40x in the ratio. The correction **strengthens** the arm: with "
        "the denominators agreeing to 1%, the measured skill loss is a genuine observability "
        "cost rather than a scale artefact to be hedged.",
        "**The sea-state-conditioning arm consumes privileged information, and is NOT an "
        "upper bound** (P6-D4 item 5, superseded by P6-D18). At deployment the sea state is "
        "estimated online from the same motion record the forecaster consumes, so no row of "
        "that arm is a deployable result -- that much stands, and is what the "
        "`privileged_information` flag records. What does not stand is the word *bound*: "
        "conditioning buys +0.0014 mean skill on `id` and costs **-0.0445** on "
        "`unseen_seastate`, where 24 of 36 cells exclude zero and all 24 are negative. A "
        "bound that lies below the unconditioned baseline on the regime that matters is not "
        "an upper bound on anything. The `id` gain is small but resolvable, not absent: "
        "P6-D20 retracted the 'not measurable' reading, which had compared a paired "
        "difference against a marginal interval.",
        "**No model is dropped for underperforming** (CLAUDE.md non-negotiable 6). Rows that "
        "cannot exist at all -- a singular design, a vehicle blind to the ablated feature -- "
        "are reported with the reason rather than omitted.",
        "**`fit_time_s_mean` is not apples-to-apples.** It is CPU wall-clock for the "
        "closed-form models and GPU wall-clock including data loading for the SGD-fitted "
        "ones. No throughput claim is made from it.",
    )


def _table_block(
    table_id: str,
    section: str,
    source: LoadedSource,
    display: pd.DataFrame,
    *,
    select: str = "",
) -> list[str]:
    """Render one table with both provenance markers.

    The machine-readable marker is emitted **last**, immediately above the table's header
    row, because :func:`dmf.eval.gate.parse_rendered_tables` associates a table with the
    last non-blank line before it. The human-readable line sits above it.

    Args:
        table_id: Stable identifier of this table within the document.
        section: Section of the plan this table belongs to.
        source: The CSV it was read from.
        display: The frame to render.
        select: How the rendered rows were derived from the file's. Left empty only when
            the table is the whole file row for row.

    Returns:
        The Markdown lines.
    """
    spec = RenderedTableSpec(
        table_id=table_id,
        section=section,
        source=source.display,
        csv_rows=source.rows,
        rendered_rows=len(display),
        select=select,
    )
    return [
        table_source_marker(spec.table_id, spec.source, spec.csv_rows, spec.rendered_rows),
        "",
        dmf_table_marker(spec),
        to_markdown(display),
        "",
    ]


def _missing_block(spec: ReportSource, results_dir: Path) -> list[str]:
    """Render the stand-in for a section whose source CSV is absent.

    Deliberately carries **no** provenance marker and no table: the marker set stays in
    one-to-one correspondence with the rendered tables, so a missing section cannot be
    mistaken by the Gate 6 read-out for a rendered one.

    Args:
        spec: The declaration that could not be satisfied.
        results_dir: Directory that was searched.

    Returns:
        Markdown lines naming what is missing and what it would have held.
    """
    tried = " or ".join(f"`{name}`" for name in spec.candidates)
    return [
        f"**Not rendered.** {tried} is absent under `{results_dir}/{PHASE6_SUBDIR}`, so this "
        f"section has no source. It would hold: {spec.what}. This document is therefore "
        f"incomplete, and it says so here rather than reading as though the section had "
        f"nothing to report.",
        "",
    ]


def _source_spec(key: str) -> ReportSource:
    """Return the declaration for one source key.

    Args:
        key: The key.

    Returns:
        Its :class:`ReportSource`.

    Raises:
        KeyError: If no source is declared under that key.
    """
    for spec in REPORT_SOURCES:
        if spec.key == key:
            return spec
    raise KeyError(f"no report source declared under {key!r}")


#: What ``scripts/evaluate.py`` scores by default, and therefore the whole population of
#: sections 6.1 and 6.2. ``DEFAULT_CONFIGS`` is ``(e02_deep, e03_probabilistic)``: the
#: unablated reference arm and the Phase 5 interval heads. **No Phase 6 ablation arm is in
#: it** -- re-scoring six arms on every ``make eval`` would make the command unrunnable on
#: every change -- so an arm appears in this document only as a paired contrast in section
#: 6.3, and on the operational metric it does not appear at all.
SCORED_CONFIG_SCOPE: str = (
    "**Scope: this table covers the reference arm and the Phase 5 heads only.** "
    "`scripts/evaluate.py` scores `DEFAULT_CONFIGS = (e02_deep, e03_probabilistic)`, which "
    "is the unablated reference arm (P6-D4) and the interval heads of section 6.4. **None "
    "of the six Phase 6 ablation arms is scored here.** They reach this document only as "
    "paired contrasts in section 6.3, against the reference arm's rows and never as "
    "free-standing accuracy numbers, and the `experiment` column below names every run the "
    "table does contain. Re-scoring the arms on every `make eval` would put a multi-hour "
    "corpus pass behind a command that has to be cheap enough to run on every change; the "
    "consequence is a scope limit, and it is stated rather than left to be inferred from a "
    "column."
)


def _experiments_in(source: LoadedSource) -> str:
    """Return a sentence naming the runs a section's source table actually holds.

    Derived from the file rather than declared, so it cannot go stale against a re-scored
    corpus: a scope claim that is written down and not re-derived is exactly the failure
    P6-D19 records.

    Args:
        source: The loaded source CSV.

    Returns:
        A Markdown sentence, or the empty string if the table carries no ``experiment``
        column to derive it from.
    """
    if "experiment" not in source.frame.columns:
        return ""
    names = sorted({str(value) for value in source.frame["experiment"].dropna().unique()})
    listed = ", ".join(f"`{name}`" for name in names)
    return f"Runs present in `{source.display}`: {listed} ({len(names)} of them)."


def _section_point(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render section 6.1.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = [
        "## 6.1 Point accuracy",
        "",
        "One row per (model, regime, DOF, horizon), aggregated over seeds. All four regimes "
        "are in one table with `regime` on the row: a per-regime heading would let a reader "
        "carry a number out of the cell it was measured in.",
        "",
        "`rmse_persistence` is the skill denominator, measured on the same windows in the "
        "same pass; `nrmse` is RMSE over the held-out targets' own standard deviation, and "
        "1.0 there means no better than predicting the partition mean.",
        "",
        SCORED_CONFIG_SCOPE,
        "",
    ]
    source = loaded.get("metrics_full")
    if source is None:
        return parts + _missing_block(_source_spec("metrics_full"), results_dir)
    scope = _experiments_in(source)
    if scope:
        parts += [scope, ""]
    table = build_point_table(source.frame)
    parts += _table_block(
        "core_metrics",
        "6.1",
        source,
        _point_display(table),
        select="one row per (model, regime, dof, horizon), aggregated over seeds",
    )
    cells = loaded.get("metrics_by_cell")
    if cells is not None:
        parts += [
            f"The same metrics broken out by `(vessel, sea state, heading, speed)` are in "
            f"`{cells.display}` ({cells.rows} rows). Not rendered here, and no number above "
            f"is read from it.",
            "",
        ]
    return parts


def _section_quiescence(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render section 6.2.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = [
        "## 6.2 Quiescent-window detection",
        "",
        "The operational metric, at both threshold sets, scored on window **onsets** at a "
        "0.5 s matching tolerance. The table's unit is (model, regime, threshold set, rule, "
        "sea state, heading, speed), and `group_level` says whether a row is one such cell "
        "or the sea-state **roll-up** over its cells. F1 is never pooled across sea states "
        "(P6-D7 item 3) -- a pooled F1 on this corpus measures the corpus composition, "
        "because at `permissive` the deck is inside limits for essentially the whole record "
        "at SS3 and SS4 -- and the cell rows exist because it should not be pooled within "
        "one sea state either: P6-D19 measured scorability as a property of the cell, since "
        "forward speed shifts the encounter frequency and at SS3/45 deg/`permissive` the "
        "deck never leaves limits at 0 or 6 kn and does at 12 kn. A roll-up row averages "
        "scorable and unscorable cells together and is marked as such rather than mixed in.",
        "",
        "Two rules are reported. **point** thresholds the point forecast; **interval** "
        "requires `max(|q05|, |q95|) <= limit` per channel per lead, which is a two-sided "
        "reading of the plan's one-sided text and is therefore stricter -- a recall "
        "difference between the rules is partly definitional (P6-D5).",
        "",
        "`f1_ci_lo`/`f1_ci_hi` are a bootstrap interval over **realizations**, not over "
        "training seeds: every detector in this table that is not an SGD model has one "
        "training seed, so `f1_std` is NaN for it, and P4-D13 records that seed spread is "
        "the wrong denominator even where it exists. The sparse cells are where this "
        "matters, and their intervals are wide and are printed rather than suppressed.",
        "",
        "Two synthetic detectors bound the metric from opposite sides, and neither is a "
        "model. `always_quiescent` flags at every decision time: its recall is 1.0 and its "
        "precision is `n_true / n_decisions`, **both by construction**, so its F1 is a fact "
        "about the decision grid rather than evidence that the metric resists gaming. "
        "`rate_matched` emits exactly as many onsets as the truth holds, at uniformly "
        "spaced RNG-free times -- it is handed the rate and not the timing, so it is the "
        "chance level for *when*, and a detector that does not beat it has learned nothing "
        "about timing.",
        "",
        SCORED_CONFIG_SCOPE,
        "",
        "**No ablation arm is scored on the operational metric at all**, and that is a "
        "stronger limit than section 6.1's. One arm could not be: `attitude_only` does not "
        "forecast `heave_rate`, which the thresholds test, so its driver writes no "
        "`quiescence.csv` and records the reason rather than an empty file (P6-D10). The "
        "other five *could* have been and were not, for the cost reason above -- so the "
        "named exclusion of `attitude_only` should not be read as implying the rest were "
        "included. What follows is the reference arm's detector performance, with the "
        "Phase 5 heads supplying the `interval` rule.",
        "",
    ]
    source = loaded.get("quiescence")
    if source is None:
        return parts + _missing_block(_source_spec("quiescence"), results_dir)
    scope = _experiments_in(source)
    if scope:
        parts += [scope, ""]
    metrics = loaded.get("metrics_full")
    lookup = _deterministic_lookup(metrics.frame) if metrics is not None else {}
    table = build_quiescence_table(source.frame, lookup)
    parts += ["### Detection", ""]
    parts += _table_block(
        "quiescence_detection",
        "6.2",
        source,
        _quiescence_display(table),
        select=(
            "one row per (group level, model, regime, threshold set, rule, sea state, "
            "heading, speed), aggregated over training seeds"
        ),
    )
    parts += [
        "### Base rate",
        "",
        "The base rate is a property of the *truth*, not of any model, so it is also shown "
        "on its own: it is what decides whether a cell is scorable at all, and it is beside "
        "every F1 above as well (P6-D7). `n_true_onsets` is the quantity scoring actually "
        "turns on -- a cell can have a base rate of 1.0 and no onset to detect.",
        "",
    ]
    parts += _table_block(
        "quiescence_base_rate",
        "6.2",
        source,
        _quiescence_base_rate_display(table),
        select=(
            "one row per (group level, regime, threshold set, sea state, heading, speed); "
            "the truth side only"
        ),
    )
    parts += [
        "### Lead time and false alarms",
        "",
        "Lead time is `true onset - earliest flag`: several decision times predict the same "
        "onset and the earliest is kept, because that earliest flag is what an operator "
        "would act on (P6-D2). False alarms per minute is unmatched predicted onsets over "
        "evaluated duration.",
        "",
    ]
    parts += _table_block(
        "quiescence_lead_time",
        "6.2",
        source,
        _quiescence_lead_time_display(table),
        select=(
            "one row per (group level, model, regime, threshold set, rule, sea state, "
            "heading, speed), aggregated over training seeds; the lead-time quantiles and "
            "false-alarm rate"
        ),
    )
    lead = loaded.get("quiescence_lead_times")
    if lead is not None:
        parts += [
            f"The distribution is not only its three quantiles: the raw per-match lead times "
            f"are in `{lead.display}` ({lead.rows} rows), which this document summarises "
            f"rather than reproduces.",
            "",
        ]
    return parts


def _resample_units_note(frame: pd.DataFrame) -> str:
    """Say how many resampling units each regime's paired interval was drawn from.

    Derived from the rows rather than declared. The number decides how much an interval on
    an out-of-distribution row is worth: a bootstrap over 12 clusters can only ever place
    its percentiles at 12 distinct configurations' worth of resolution, and every headline
    this document reads off `unseen_seastate` or `unseen_heading` rests on that.

    Args:
        frame: The contrast rows, carrying ``regime`` and ``ci_n_units``.

    Returns:
        A Markdown sentence, or the empty string if the columns are absent.
    """
    if not {"regime", "ci_n_units"} <= set(frame.columns):
        return ""
    counts = (
        frame.dropna(subset=["ci_n_units"])
        .groupby("regime")["ci_n_units"]
        .apply(lambda values: sorted({int(value) for value in values}))
    )
    if counts.empty:
        return ""
    listed = "; ".join(
        f"`{regime}` {'/'.join(str(value) for value in values)}"
        for regime, values in counts.items()
    )
    return (
        f"**Units behind each paired interval, counted from the rows themselves**: {listed}. "
        f"The two out-of-distribution regimes hold out one level of one factor, so their "
        f"grids are 12 cells against 48 on `id` and `unseen_vessel` -- **every "
        f"out-of-distribution interval in this section is a bootstrap over 12 clusters**, "
        f"including the RevIN result, and a 12-unit percentile interval is a coarse object. "
        f"It is reported at that width rather than narrowed by resampling a finer unit the "
        f"committed tables do not carry (`dmf.eval.ablations.CI_RESAMPLE_UNIT`)."
    )


def _section_ablations(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render section 6.3, including the rows that are not ablation results.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = [
        "## 6.3 Ablations",
        "",
        "Each arm is paired against its reference arm on identical windows, identical "
        "origins and the DOFs both forecast. `skill_diff` is **arm minus reference**, so "
        "positive means the ablated arm scores higher.",
        "",
        "`skill_diff_ci_lo`/`skill_diff_ci_hi` are a **paired** bootstrap interval on that "
        "difference: the arm and its reference are weighted by the same resample draw, so "
        "the realization-to-realization variation that dominates each marginal cancels. "
        "Read the difference against this interval and never against a marginal interval "
        "from section 6.1 -- two overlapping marginals say nothing about whether a paired "
        "difference is distinguishable from zero.",
        "",
        "`ci_resample_unit` says what one draw resamples, and on these rows it is a **corpus "
        "grid cell** rather than a realization, because no committed table carries "
        "per-realization SSE. That makes the interval conservative: measured against the "
        "5616 realization-paired contrasts committed in `results/e02/paired_contrasts.csv`, "
        "the cell-resampled interval is wider by a median factor of 2.8 on `id` and "
        "`unseen_vessel` and 6.3-7.1 on the two out-of-distribution regimes, and it was "
        "narrower in 1 of those 5616. An interval that excludes zero here would also exclude "
        "zero under a realization bootstrap; one that contains zero may be doing so only "
        "because of the coarser unit.",
        "",
        "**Those four numbers -- 2.8, 6.3-7.1 and 1 of 5616 -- are the one substantive "
        "quantity in this document that is not read from a CSV.** They are hard-coded prose "
        "in `dmf.eval.report`, derived once at implementation time; no committed script "
        "emits them and no file beside this one carries them, so Gate 6's traceability "
        "contract (every rendered number traces to a named CSV) does not cover them and "
        "nothing here re-derives them on a re-render. They are stated because the direction "
        "they claim -- conservative -- is what licenses reading a zero-excluding interval as "
        "real, and a reader is entitled to know that this particular licence rests on an "
        "unaudited derivation rather than on an artifact. Two things reduce that exposure "
        "without removing it. The direction is argued **structurally** and independently of "
        "the measurement: a cluster resample of a fixed factorial design charges the "
        "statistic for between-cell heterogeneity that is design rather than sampling, so "
        "the widening is not a contingent fact about this corpus. And the magnitude is "
        "**reconstructible read-only from two committed files** -- resample the grid cells "
        "of `results/e02/baselines_by_cell.csv` under the same `(n_units, n_boot, seed)` "
        "draw the ablation intervals use, and compare each width against the realization "
        "interval on the same row of `results/e02/paired_contrasts.csv`. A reader can "
        "therefore check the claim; what does not exist is a committed script that checks it "
        "on every render.",
        "",
        "`skill_diff_std` is the spread over **training seeds**. It is NaN for every "
        "deterministic vehicle, and where it is defined it measures initialisation and data "
        "order rather than realization sampling, so it is not the denominator a difference "
        "should be judged against (P4-D13).",
        "",
    ]
    source = loaded.get("ablations")
    if source is None:
        return parts + _missing_block(_source_spec("ablations"), results_dir)
    metrics = loaded.get("metrics_full")
    lookup = _deterministic_lookup(metrics.frame) if metrics is not None else {}
    units = _resample_units_note(source.frame)
    if units:
        parts += [units, ""]
    parts += [
        "**The contrast is joined on the logical vehicle, not on the model label** "
        "(`dmf.eval.ablations.with_logical_model`). One arm spells its deep vehicle "
        "differently from its reference: the L=400 arm's is `tcn_l400`, carrying the extra "
        "dilation stage a 40 s receptive field needs, against the reference's `tcn`. Joining "
        "on the label dropped all 216 of those fitted, scored, committed rows from every "
        "table in this section without an empty frame or a `not_fittable` reason to notice, "
        "because the arm's five closed-form vehicles still matched. `model_reference` and "
        "`architecture_differs` are on every row so that the pairing is visible: where "
        "`architecture_differs` is `yes` the contrast compares two networks, and "
        "`param_delta_frac` (+0.1262 there) says by how much they differ in capacity.",
        "",
    ]
    contrasts, unfittable, revin_closed_form = split_ablation_rows(source.frame)
    absent_flags = [name for name in ABLATION_FLAG_COLUMNS if name not in source.frame.columns]
    if absent_flags:
        named = ", ".join(f"`{name}`" for name in absent_flags)
        parts += [
            f"**The source table does not carry {named}.** Those flags are what keep a zero "
            f"contrast that is an architectural artefact from reading as evidence of no "
            f"effect (P6-D8), and they cannot be reconstructed here from the metrics alone; "
            f"this document reports their absence rather than inferring them.",
            "",
        ]
    rendered_ablations: set[str] = set()
    if not contrasts.empty:
        table = build_ablation_table(contrasts, lookup)
        for ablation in sorted(table["ablation"].astype(str).unique()):
            scoped = table[table["ablation"].astype(str) == ablation]
            rendered_ablations.add(ablation)
            parts += [f"### `{ablation}`", ""]
            note = scoped["note"].astype(str).iloc[0] if "note" in scoped.columns else ""
            if note:
                parts += [f"Reading instruction carried from the arm registry: {note}.", ""]
            parts += _table_block(
                ablation_table_id(ablation),
                "6.3",
                source,
                _ablation_display(scoped),
                select=f"the {ablation} rows, aggregated over seeds",
            )
    absent_ablations = sorted(set(ABLATION_TABLE_IDS) - rendered_ablations)
    if absent_ablations:
        named = ", ".join(f"`{name}`" for name in absent_ablations)
        parts += [
            f"**{named} contributed no row to the source table**, so no table is rendered "
            f"for it. That is an absent measurement, not a null result, and it is stated "
            f"here rather than left as a gap in the section.",
            "",
        ]
    if not unfittable.empty:
        parts += [
            "### Rows that have no fit",
            "",
            "Reported rather than omitted. An absent row and a row with nothing in it look "
            "the same to a reader only if the second is never written (P6-D8 defect 2).",
            "",
        ]
        columns = _present(
            unfittable, ("ablation", "arm", "model", "regime", "not_fittable", "note")
        )
        # De-duplicated, not aggregated: these rows carry no metric to average, and the
        # same reason repeated once per seed would read as three findings.
        parts += _table_block(
            "ablation_not_fittable",
            "6.3",
            source,
            unfittable[columns].drop_duplicates().reset_index(drop=True),
            select="the rows with a `not_fittable` reason, de-duplicated over seeds",
        )
    if not revin_closed_form.empty:
        excluded = ", ".join(
            f"`{name}`" for name in sorted(set(revin_closed_form["model"].astype(str)))
        )
        parts += [
            "### The RevIN arm's closed-form rows are a reproducibility control, not an "
            "ablation result",
            "",
            f'`dmf.models.base.revin_applies` is True only for `FIT_KIND == "sgd"`, so RevIN '
            f"cannot apply to {excluded}, and every one of those rows must equal the "
            f"reference arm's by construction. Listing them as RevIN rows would put 'no "
            f"effect' results into a table about RevIN's effect, so the RevIN ablation table "
            f"above lists the `tcn` rows only and these are shown here instead (P6-D13). "
            f"What they *do* test is that reading `results/e02/` as the reference arm "
            f"reproduces: the same corpus, split, solver and scorer, run weeks apart under a "
            f"changed codebase.",
            "",
        ]
        parts += _table_block(
            "ablation_revin_closed_form",
            "6.3",
            source,
            _ablation_display(build_ablation_table(revin_closed_form, lookup)),
            select="the RevIN arm's rows for models RevIN does not apply to",
        )
    reproducibility = loaded.get("reproducibility")
    if reproducibility is not None:
        parts += _table_block(
            "reference_reproducibility",
            "6.3",
            reproducibility,
            reproducibility.frame.reset_index(drop=True),
        )
    return parts


def _section_probabilistic(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render section 6.4.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = [
        "## 6.4 Probabilistic: the residual-interval floor and the heads beside it",
        "",
        "`residual_interval` is a point forecast plus per-(DOF, horizon) empirical residual "
        "quantiles taken from the **validation** split. It is unconditional -- its width at "
        "a given cell is the same for every window -- so on `id` it is near-perfectly "
        "calibrated by construction. That is what makes it discriminating: a learned head "
        "beats it only by making its interval conditional, narrow where the deck is "
        "predictable and wide where it is not. Matching it on coverage *and* on width means "
        "the head has learned nothing about its own uncertainty (P6-D6).",
        "",
        "`band` is on every row and no row spans two bands (P5-D15). `width_ratio` is the "
        "mean width over the width an unconditional Gaussian interval would need at the same "
        "nominal level, so 1.0 means no sharper than knowing only the partition's variance.",
        "",
    ]
    source = loaded.get("probabilistic")
    floor_table: pd.DataFrame | None = None
    if source is None:
        parts += _missing_block(_source_spec("probabilistic"), results_dir)
    else:
        floor_table = build_probabilistic_view(source.frame, source=source.display)
        parts += _table_block(
            "probabilistic_floor",
            "6.4",
            source,
            _probabilistic_display(floor_table),
            select=(
                "one row per (model, head, regime, dof, horizon), aggregated over seeds; "
                "never pooled across horizon bands"
            ),
        )
    heads = loaded.get("e03_heads")
    if heads is not None:
        parts += [
            "### The Phase 5 heads as committed",
            "",
            "Read from the Gate 5 audit trail, which is never regenerated. Shown as its own "
            "table rather than joined into the one above, so each number keeps a single "
            "source file and a single row count.",
            "",
        ]
        head_table = build_probabilistic_view(heads.frame, source=heads.display)
        parts += _table_block(
            "probabilistic_heads",
            "6.4",
            heads,
            _probabilistic_display(head_table),
            select="every row, with the horizon band labelled",
        )
        if source is not None and floor_table is not None:
            parts += [
                "### Head minus floor",
                "",
                "The section's argument, differenced rather than asserted. The floor is "
                "unconditional -- one interval width per (regime, DOF, horizon), the same "
                "for every window -- so on `id` it is near-perfectly calibrated by "
                "construction and **a head beats it only by being conditional**. "
                "`width_ratio_diff` is the axis that answers that: both sides are "
                "normalised by the same `signal_std`, so it is dimensionless, and negative "
                "means the head's interval is **sharper** than the floor's at that cell. "
                "`picp_diff` is beside it and must not be read alone -- a head that is "
                "wider everywhere covers better everywhere, which is the failure mode this "
                "whole section exists to expose (P6-D6).",
                "",
                f"Each head row is joined to its floor cell on "
                f"`{', '.join(FLOOR_JOIN_KEYS)}`; the floor carries no model and no head, "
                f"which is what makes it a floor. A head row with no floor cell is kept "
                f"with NaN differences rather than dropped, so the join cannot narrow the "
                f"comparison silently. Both sides of every difference are printed beside "
                f"it: the head's own numbers come from `{heads.display}` and the floor's "
                f"from `{source.display}`, and neither is re-derived here.",
                "",
            ]
            contrast = build_floor_contrast(head_table, floor_table)
            parts += _table_block(
                "probabilistic_head_minus_floor",
                "6.4",
                heads,
                _floor_contrast_display(contrast),
                select=(
                    f"every row of {heads.display}, joined to the {FLOOR_MODEL} floor of "
                    f"{source.display} on ({', '.join(FLOOR_JOIN_KEYS)}) and differenced"
                ),
            )
    return parts


#: Why a **point** control row is reported but not asserted on: the P6-D11/P6-D12
#: floored-cell narrowing, which is the only reason a point row is ever unasserted.
POINT_REPORTED_ONLY_REASON: str = (
    "These channels sit on their P1-D2 residual floor across the whole test partition, so "
    "their targets are a 26 dB-suppressed stand-in rather than the physics the DOF is named "
    "after. The shuffle statistic there is dominated by a train/test amplitude mismatch, "
    "measured in both directions and shown not to be leakage (P6-D11). They are scored and "
    "shown; they do not decide the control."
)

#: Why an **interval** control row is reported but not asserted on. A different criterion
#: from the point one, and stated separately rather than sharing a paragraph: the floored-cell
#: narrowing is deliberately NOT applied to the interval statistic, so printing the point
#: reason over these rows would state a derivation that was explicitly refused for them.
INTERVAL_REPORTED_ONLY_REASON: str = (
    "These cells have a **null** whose own PICP@90 falls outside Gate 5's registered "
    "`[0.85, 0.95]` band, so the interval shuffle control declines to judge them (P6-D21): "
    "against a mis-sized reference a wider subject scores better by covering the null's "
    "misses, and the statistic then measures the reference rather than the subject. The "
    "decision is taken per cell from `picp_null`, which is on every row, so it can be "
    "recomputed from the file. **This is a narrowing, not a fix** -- a leak confined to "
    "these cells would not be caught, and on the production corpus it removes every "
    "`unseen_seastate` cell, which is the same finding the coverage tables report. The "
    "floored-cell narrowing of P6-D11 is *not* applied here; `on_residual_floor` records "
    "that derivation without acting on it."
)


def _unjudged_groups_note(frame: pd.DataFrame, asserted: pd.DataFrame) -> list[str]:
    """Name every (control, regime) group whose rows were all excluded from the assertion.

    A group with no asserted cell **disappears from the summary table above**, because that
    table is grouped over asserted rows. An absent row reads as "not run" or is not read at
    all, and neither is what happened: the control ran, produced numbers, and could not
    judge them. P6-D21's narrowing makes this reachable in production -- on the corpus the
    interval shuffle control's null is miscalibrated in every ``unseen_seastate`` cell -- so
    the absence is stated rather than left to be noticed.

    Args:
        frame: Every control row of one family.
        asserted: The subset the verdict was taken over.

    Returns:
        Markdown lines, empty when every group had at least one asserted cell.
    """
    keys = [key for key in ("control", "regime") if key in frame.columns]
    if not keys:
        return []
    everything = set(map(tuple, frame[keys].astype(str).to_numpy()))
    judged = set(map(tuple, asserted[keys].astype(str).to_numpy())) if not asserted.empty else set()
    unjudged = sorted(everything - judged)
    if not unjudged:
        return []
    named = ", ".join("/".join(group) for group in unjudged)
    return [
        f"**{len(unjudged)} group(s) have no asserted cell at all and are therefore absent "
        f"from the table above**: {named}. The control ran there and every cell was excluded "
        f"from its assertion, so it could not judge that group. That is not a pass, and the "
        f"rows are in the reported-only table below with their real numbers.",
        "",
    ]


def _section_controls(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render the integrity controls.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = [
        "## Integrity controls",
        "",
        "Kept out of every model table so they cannot be mistaken for models.",
        "",
        "The shuffle control's null is the **window-mean forecast**, not zero skill: a model "
        "fitted to time-shuffled targets degenerates to the conditional mean, which inverts "
        "to the window mean, and the window mean beats persistence at long horizons on a "
        "narrowband signal. Testing against zero would report leakage on a clean pipeline "
        "(P3-D8). `excess` is `1 - MSE_subject/MSE_null`, one-sided, and `worst_excess` is "
        "its maximum over the cells of the group.",
        "",
        "**Asserted and reported-only cells are two tables and never one verdict** (P6-D12). "
        "A cell whose test-set signal sits on its P1-D2 residual floor is computed and shown "
        "but does not decide the control; the narrowing costs detection power on exactly "
        "those cells and that cost is stated rather than absorbed.",
        "",
        "**The two control families narrow on different criteria, and the reason is written "
        "beside each table below.** The point shuffle control stops asserting on channels "
        "sitting on their residual floor (P6-D11/P6-D12). The interval shuffle control stops "
        "asserting where the *null* is itself outside Gate 5's registered PICP band "
        "`[0.85, 0.95]` (P6-D21), because a ratio of interval scores against a mis-sized "
        "reference prices the reference: a wider subject covers a miscalibrated null's misses "
        "without knowing anything. Neither narrowing removes a row from the file, and where a "
        'regime ends up with no asserted interval cell at all its `passed` means "nothing was '
        'judged" rather than "the subject learned nothing".',
        "",
        "**`passed` is not the same commitment for every control, and the `enforced` column "
        "is where that is written down.** `enforced` is the `strict` argument the control was "
        "called with: True means a failure stops the run, False means the verdict is computed, "
        "written and read but raises nothing. The shuffle controls are enforced. The untrained "
        "controls are **reported, not enforced** (P3-D9): their literal criterion -- a "
        "random-initialised model must not beat persistence -- is known to be wrong for this "
        "signal at long horizons, and their failures ship rather than being tuned away. "
        "`enforced` and `asserted` are different questions: `asserted` says whether a *row* "
        "counted toward its control's verdict, `enforced` says whether that verdict stopped "
        "anything. Every untrained row is asserted and unenforced.",
        "",
        "**The pipeline-sanity control is enforced *and* now reported**, in its own table "
        "below. It raises inside the driver -- it is the one control that stops a run "
        "unconditionally -- so a *failing* row cannot appear in that table at all: the run "
        "that would have written it stopped instead. Read the measured `rel_diff` rather "
        "than the `passed` column. It should sit at the float32 storage floor (5.006e-08 on "
        "the production `id/test` partition, P2-D9); a run that had drifted to just inside "
        "the tolerance would look identical in `passed` and obvious in `rel_diff`.",
        "",
    ]
    for key, label, by, reported_only_reason in (
        # `experiment` first, and `_control_summary` drops it where a table does not carry
        # it. `controls.csv` concatenates the controls of every arm that was run, and a
        # summary grouped without it would report one `worst_excess` over seven arms with
        # nothing saying which arm produced it -- the P5-D17 failure mode (a true statement
        # about a subset published as a statement about the whole) with the subset hidden
        # in the grouping rather than in the prose.
        (
            "controls",
            "point",
            ("experiment", "arm", "control", "regime", "subject_model", "null_model"),
            POINT_REPORTED_ONLY_REASON,
        ),
        (
            "interval_controls",
            "interval",
            ("experiment", "arm", "control", "regime", "subject_model", "null_model", "metric"),
            INTERVAL_REPORTED_ONLY_REASON,
        ),
    ):
        source = loaded.get(key)
        if source is None:
            spec = _source_spec(key)
            if spec.required:
                parts += _missing_block(spec, results_dir)
            else:
                named = " or ".join(f"`{name}`" for name in spec.candidates)
                parts += [
                    f"**No {label} controls table.** {named} is absent under "
                    f"`{results_dir}/{PHASE6_SUBDIR}`, so no {label} control is reported "
                    f"here. An interval claim made without an interval control is unaudited, "
                    f"and that is a gap rather than a pass.",
                    "",
                ]
            continue
        asserted, reported_only = split_controls(source.frame)
        parts += [f"### {label.capitalize()} controls: asserted", ""]
        parts += _table_block(
            f"controls_{label}_asserted",
            "controls",
            source,
            _control_summary(asserted, by=by),
            select="the rows the pass/fail decision was taken over, summarised per control",
        )
        parts += _unjudged_groups_note(source.frame, asserted)
        if not reported_only.empty:
            parts += [
                f"### {label.capitalize()} controls: reported, not asserted on",
                "",
                reported_only_reason,
                "",
            ]
            parts += _table_block(
                f"controls_{label}_reported_only",
                "controls",
                source,
                _control_summary(
                    reported_only, by=("experiment", "arm", "control", "regime", "dof")
                ),
                select=(
                    "the cells excluded from the assertion, summarised per "
                    "(experiment, arm, control, regime, dof)"
                ),
            )
    parts += _pipeline_sanity_block(loaded, results_dir)
    return parts


def _pipeline_sanity_block(loaded: Mapping[str, LoadedSource], results_dir: Path) -> list[str]:
    """Render the pipeline-sanity control, or say that it did not run.

    Its own table rather than a third block of the control summary above: ``excess`` there is
    the fraction of a null's error a subject removed, and the statistic here is the relative
    disagreement between two independent computations of the *same* quantity. One column name
    meaning two things in one section is how a table gets read confidently and wrongly.

    Args:
        loaded: The sources that were found.
        results_dir: The results root.

    Returns:
        Markdown lines.
    """
    parts = ["### Pipeline sanity: persistence through the dataset vs. on the raw arrays", ""]
    source = loaded.get("pipeline_sanity")
    if source is None:
        spec = _source_spec("pipeline_sanity")
        named = " or ".join(f"`{name}`" for name in spec.candidates)
        return parts + [
            f"**Not rendered.** {named} is absent under `{results_dir}/{PHASE6_SUBDIR}`. The "
            f"control still runs -- it raises inside the driver, so any run that completed "
            f"passed it -- but no run wrote its measured disagreement to a file, so this "
            f"document cannot state the number. That is the *enforced and unreported* half of "
            f"P6-D15, and its absence is reported here rather than left to be inferred.",
            "",
        ]
    frame = source.frame
    worst = float(frame["rel_diff"].max()) if "rel_diff" in frame.columns else float("nan")
    tolerance = float(frame["rtol"].iloc[0]) if "rtol" in frame.columns else float("nan")
    parts += [
        f"Two paths that share nothing but the realization list and the window geometry: the "
        f"forecast is formed in normalised space and inverted through the dataset on one "
        f"side, and recomputed directly from the Parquet with no torch and no normalisation "
        f"on the other. Worst relative disagreement over every cell rendered here: "
        f"**{worst:.3e}** against a tolerance of {tolerance:.1e}. Gate 2 criterion 5, and the "
        f"reason every skill denominator in this document is the quantity it claims to be.",
        "",
    ]
    columns = _present(
        frame,
        (
            "experiment",
            "arm",
            "regime",
            "dof",
            "horizon_s",
            "rmse_pipeline",
            "rmse_raw",
            "rel_diff",
            "rtol",
            "n_windows",
            "model_matches_inline",
            "enforced",
            "passed",
        ),
    )
    display = frame.sort_values(
        _present(frame, ("experiment", "arm", "regime", "dof", "horizon_samples")),
        kind="stable",
        ignore_index=True,
    )
    parts += _table_block(
        "pipeline_sanity",
        "controls",
        source,
        # No `select`: every row of the file is rendered, so the marker's csv_rows and rows
        # must coincide -- the strongest form of Gate 6 predicate 4 (P6-D14), and it applies
        # here because this table is not a filtered view of anything.
        display[columns],
    )
    return parts


def build_results_report(results_dir: Path, out_path: Path, *, strict: bool = True) -> Path:
    """Assemble every committed Phase 6 CSV into ``results/results.md``.

    The document is rendered **from** the CSVs: every table below is a projection of a frame
    this function read off disk, and each one carries a provenance marker naming that file,
    its row count and the projection applied. "Every number traces to a CSV" is therefore a
    structural property of the renderer rather than a claim about it, which is what Gate 6
    predicates 3 and 4 check (docs/protocol.md P6-D1).

    Per-run tables are aggregated here, at render time, by :func:`aggregate_results`:
    ``metrics_full.csv`` is written one row per (model, seed, regime, DOF, horizon) (P6-D10),
    and rendering those rows as the table would breach CLAUDE.md non-negotiable 5 while
    looking like a complete table.

    **The output is a deterministic function of the inputs.** No timestamp, hostname or
    wall-clock figure appears in it, and every path it names is relative to ``results_dir``,
    so re-rendering the same CSVs reproduces the document byte for byte. That is what makes
    "regenerated by ``make eval``, not hand-edited" checkable.

    Args:
        results_dir: The results **root**, normally ``results/``: it holds ``results.md``,
            the Phase 6 CSVs under ``e04/``, and the frozen Gate 3-5 directories that
            section 6.4 reads the committed heads from. Passing ``results/e04`` directly is
            also honoured.
        out_path: Destination Markdown file, normally ``results/results.md``.
        strict: Whether a missing required CSV is fatal. ``True`` by default: a report that
            silently omits a section of the plan is worse than no report, because it looks
            complete. ``False`` renders an explicitly incomplete document that names every
            section it could not build.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: Under ``strict``, if an expected CSV is missing, so that a
            partially regenerated report cannot be mistaken for a complete one.
        ValueError: If a table is present but lacks a column its meaning depends on -- a
            skill without its persistence denominator, an F1 without its base rate, a
            coverage without its width.
    """
    loaded, missing = load_report_sources(results_dir, strict=strict)
    parts: list[str] = [
        "# Phase 6 results -- short-horizon deck motion forecasting",
        "",
        "**These are simulated results.** The corpus is JONSWAP-driven vessel simulation; no "
        "real deck data enters this project at any point.",
        "",
        "Regenerated end to end by `make eval`. Nothing here is hand-edited: the document is "
        "a deterministic function of the CSVs named below, so re-rendering them reproduces "
        "it byte for byte.",
        "",
        "## Provenance",
        "",
        "Every table below is preceded by two lines carrying the same record -- one for a "
        "reader, one for the Gate 6 read-out:",
        "",
        "```",
        "Source: `<path>` | csv_rows: <n> | rendered_rows: <m> | table_id: <id>",
        "<!-- dmf-table id=<id> section=<6.1|6.2|6.3|6.4|controls> source=<path> "
        'csv_rows=<n> rows=<m> [select="<filter>"] -->',
        "```",
        "",
        "`<path>` is relative to this document's directory. `csv_rows` is the data-row count "
        "of that file, header excluded; `rows` is the row count of the table below the "
        "marker; `select` says how the second was derived from the first, and a table "
        "declaring no `select` is claiming to be the whole file row for row. A section whose "
        "source is absent carries no marker at all, so the marker set and the rendered "
        "tables stay in one-to-one correspondence.",
        "",
        "Read here:",
        "",
    ]
    for spec in REPORT_SOURCES:
        source = loaded.get(spec.key)
        named = " or ".join(f"`{name}`" for name in spec.candidates)
        if source is None:
            state = "**absent**" + (" (required)" if spec.required else " (optional)")
            parts.append(f"- {named} -- {spec.what}; {state}")
        elif spec.rendered:
            parts.append(f"- `{source.display}` ({source.rows} rows) -- {spec.what}")
        else:
            parts.append(
                f"- `{source.display}` ({source.rows} rows) -- {spec.what}; **not rendered "
                f"here**, and no number below is read from it"
            )
    parts.append("")
    absent_required = [spec for spec in missing if spec.required]
    if absent_required:
        names = ", ".join(f"`{spec.candidates[0]}`" for spec in absent_required)
        parts += [
            f"**This document is incomplete.** {names} could not be found, and the sections "
            f"that would have been rendered from them say so where they would have appeared.",
            "",
        ]
    parts += _section_point(loaded, results_dir)
    parts += _section_quiescence(loaded, results_dir)
    parts += _section_ablations(loaded, results_dir)
    parts += _section_probabilistic(loaded, results_dir)
    parts += _section_controls(loaded, results_dir)
    parts += ["## Caveats", ""]
    parts += [f"- {caveat}" for caveat in results_report_caveats()]
    parts.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
