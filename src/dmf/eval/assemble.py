"""Assembling the four cross-arm Phase 6 tables from committed per-arm artifacts.

``results.md`` reads nine CSVs. :mod:`dmf.eval.scoring` writes five of them by scoring one
experiment against the corpus; this module writes the other four by **joining tables that
already exist**. Until it did, ``make eval`` failed on the first missing file -- correctly,
but it meant the pipeline had no producer at all for section 6.3, for the controls section
or for section 6.4's floor.

**Why a sibling module and not more of** :mod:`dmf.eval.scoring`. The two do different work
and need different things to run:

============================  =================================  ============================
                              :mod:`dmf.eval.scoring`            this module
============================  =================================  ============================
reads                         the corpus and the checkpoints     committed CSVs only
needs a GPU                   yes, for the SGD forward passes    no
unit of work                  one (experiment, regime)           one *cross-arm join*
can it refit                  no, and it raises if asked to      it cannot even try
cost                          minutes to hours                   under a second
============================  =================================  ============================

The operative row is the last two. An ablation contrast pairs an arm's committed rows
against another arm's committed rows; nothing about that requires the corpus, and folding it
into the scoring driver would make a table that is a *join* look like a table that is a
*measurement*. It would also put a function that can start a 46 h sweep in the same call
path as one that reads seven CSVs, and this project has already recorded what happens when
an expensive path is reachable from a cheap-looking one. Keeping them apart means
``--render-only`` and this module together answer "is ``results.md`` a pure function of the
committed CSVs?" without touching a GPU.

**What this module does not produce, and who does.** ``interval_controls.csv`` and
``pipeline_sanity.csv`` are the other two files ``results.md`` reads, and they are **not**
assembled here: both require the corpus and the dataset pipeline -- an interval control refits
a closed-form interval and scores it, and the pipeline-sanity control re-reads the Parquet --
which is exactly the contract this module does not have. :mod:`dmf.eval.control_runner` writes
them, and ``tests/test_assemble.py`` asserts that every source the renderer declares is
written by exactly one of the three drivers, so "which module produces this file" is answered
by a set relation rather than by a grep.

**Nothing here is computed a second time.** Every number is copied or differenced from a
committed table; the module owns no metric of its own. What it does own is the refusal to
join two tables that are not comparable -- :func:`dmf.eval.ablations.contrast_table` enforces
the observation-mode rule (P6-D4 item 4) and the logical-channel join, and
:data:`MATCHED_ORIGIN_ARMS` refuses the lookback contrast unless it is reading a table that
proves, from its own rows, that it was scored on the shared origin set (P6-D4 item 1). The
producer of such a table is :func:`dmf.eval.matched.score_matched_lookback`; the refusal
does not consult it, and a table's directory is never taken as evidence of what it is.
An arm that cannot be assembled is recorded in
:attr:`AssembledArtifacts.skipped` with the reason, exactly as
:attr:`dmf.eval.scoring.ScoringArtifacts.skipped` records a pass that could not run: an
absent measurement is not a null result, and the renderer already says so where a table is
missing.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dmf.eval.ablations import (
    ARMS,
    CROSS_MODE_COLUMNS,
    MATCHED_ORIGIN_COLUMNS,
    REFERENCE_ARM,
    arms_for_experiment,
    contrast_table,
    not_fittable_rows,
    unmatched_origin_reason,
)
from dmf.eval.controls import CONTROL_COLUMNS
from dmf.eval.report import PHASE6_SUBDIR, build_probabilistic_table, write_table

__all__ = [
    "ARM_CELLS_FILE",
    "ASSEMBLED_ARTIFACTS",
    "BITWISE_TOL",
    "CONTROL_SOURCE_DIRS",
    "ENFORCED_BY_CONTROL",
    "MATCHED_ORIGIN_ARMS",
    "MATCHED_ORIGIN_SUBDIR",
    "NOT_AN_ARM",
    "REFERENCE_DIR",
    "REPRODUCIBILITY_ARM_DIR",
    "REPRODUCIBILITY_QUANTITIES",
    "RESIDUAL_FLOOR_EXPERIMENT",
    "AssembledArtifacts",
    "assemble_phase6_tables",
    "build_ablations_table",
    "build_controls_table",
    "build_probabilistic_baseline",
    "build_reference_reproducibility",
    "read_arm_cells",
    "read_arm_frames",
    "read_arm_rows",
]

#: Table name -> file name, for everything this module writes. The twin of
#: :data:`dmf.eval.scoring.SCORING_ARTIFACTS`, and between them they cover every source
#: :data:`dmf.eval.report.REPORT_SOURCES` declares required.
ASSEMBLED_ARTIFACTS: dict[str, str] = {
    "ablations": "ablations.csv",
    "controls": "controls.csv",
    "probabilistic_baseline": "probabilistic_baseline.csv",
    "reference_reproducibility": "reference_reproducibility.csv",
}

#: Committed directory the reference arm's rows are read from, relative to the results root.
#: ``results/e02/`` fitted both vehicles under a byte-identical train block; it is the Gate 4
#: audit trail and is read, never regenerated (P6-D4).
REFERENCE_DIR: str = "e02"

#: Per-run accuracy table each arm is contrasted from. **Per seed, not aggregated**: the
#: contrast is paired seed-wise (seed 0 of the arm against seed 0 of the reference) and the
#: ``>= 3`` seed aggregation belongs to the renderer (P6-D10). Reading the aggregated
#: ``baselines.csv`` here would difference two means and lose the pairing.
ARM_ROWS_FILE: str = "baselines_by_seed.csv"

#: Aggregated accuracy table, read only by the reproducibility control, which compares two
#: *published* tables rather than two runs.
ARM_SUMMARY_FILE: str = "baselines.csv"

#: Per-arm control table written by the training driver.
ARM_CONTROLS_FILE: str = "baselines_controls.csv"

#: Per-grid-cell accuracy table, and the **only** committed table with a resampling unit
#: finer than the whole partition. It carries ``sse``, ``sse_persistence`` and
#: ``n_realizations`` per ``(vessel, ss, heading, speed)``, which is what
#: :func:`dmf.eval.ablations.contrast_uncertainty` draws the paired interval from and what
#: :func:`dmf.eval.ablations.cell_floored_dofs` reads the headings and hulls out of. An arm
#: that did not commit one gets NaN bounds and a reason, never a seed-spread stand-in.
ARM_CELLS_FILE: str = "baselines_by_cell.csv"

#: Arms whose rows may **not** be read from any committed per-arm table, because a lookback
#: contrast is only paired on the intersection of the three origin sets (P6-D4 item 1) and
#: every committed run scored its arm's own *full* set. Measured on the committed tables at
#: ``unseen_heading``, persistence, roll, 1 s: 552 480 windows at L=100, 542 880 at L=200 and
#: 523 680 at L=400, i.e. 1151 / 1131 / 1091 per realization against the 1091 the matched
#: comparison requires. Differencing those tables would compare forecasts of different
#: absolute times, which is the failure P6-D4 item 1 exists to prevent and which
#: :func:`dmf.eval.ablations.start_matching_compares_different_times` exists to demonstrate.
#: The rows must come from a re-scoring on
#: :func:`dmf.eval.ablations.matched_origins` --
#: :func:`dmf.eval.matched.score_matched_lookback` is the producer -- and must **prove it
#: from their own columns**: every row carries
#: :data:`dmf.eval.ablations.MATCHED_ORIGIN_COLUMNS`, :func:`_unmatched_reason` re-derives the
#: intersection from them, and a table that does not reproduce is refused exactly as an absent
#: one is. An arm's own committed table carries none of those columns, so pointing this module
#: at the raw arms -- or copying one into the matched directory -- is refused rather than
#: silently joined.
MATCHED_ORIGIN_ARMS: frozenset[str] = frozenset({"lookback_10s", "lookback_20s", "lookback_40s"})

#: Where a matched-origin re-scoring would be written, under ``<root>/e04/``. Keyed by **arm**
#: and not by experiment: ``lookback_20s`` and ``reference`` are two readings of one training
#: run (P6-D10), so an experiment-keyed path could not hold both.
MATCHED_ORIGIN_SUBDIR: str = "matched_origins"

#: Value of ``arm`` for an experiment that is not an ablation arm. A **non-empty** sentinel,
#: and that is not cosmetic: the renderer summarises the controls with
#: ``groupby(["experiment", "arm", ...])``, an empty string round-trips through CSV as NaN,
#: and pandas drops NaN group keys -- so an empty value here deletes those rows from the
#: rendered table while leaving them in the file, which is the worst of both. Measured
#: before it was fixed: all 288 ``e03_probabilistic`` rows vanished from the summary.
NOT_AN_ARM: str = "not_an_arm"

#: Directories whose ``baselines_controls.csv`` goes into ``controls.csv``, as
#: ``(directory relative to the results root, experiment, arm)``. ``e03_probabilistic`` is the
#: Phase 5 head experiment rather than an ablation arm, and its controls belong in this table
#: anyway: section 6.4 renders its rows, and a published number whose controls are in a file
#: nobody assembles is a number nobody audited. The e04 arm directories are discovered from
#: :data:`ARMS` rather than listed, so a new arm needs no edit here.
CONTROL_SOURCE_DIRS: tuple[tuple[str, str, str], ...] = (
    (REFERENCE_DIR, "e02_deep", REFERENCE_ARM),
    ("e03", "e03_probabilistic", NOT_AN_ARM),
)

#: Control name -> whether the training driver runs it with ``strict=True``, i.e. whether a
#: failure stops the sweep. Used **only** to backfill the ``enforced`` column of control
#: tables written before that column existed; every table written after carries its own
#: value, taken from the ``strict`` argument at the call site.
#:
#: This is a claim about two drivers: ``dmf.train.experiment._run_controls`` runs the point
#: controls during a training sweep, and :func:`dmf.eval.control_runner.run_regime_controls`
#: runs the interval controls and the pipeline-sanity control from committed artifacts.
#: ``tests/test_assemble.py`` parses **both** functions' sources, asserts the mapping agrees
#: with the ``strict`` keywords actually passed, and asserts that every key here has a
#: caller -- an entry for a control nothing runs would claim an enforcement nobody made.
ENFORCED_BY_CONTROL: Mapping[str, bool] = {
    "shuffle": True,
    "untrained": False,
    "interval_shuffle": True,
    "interval_untrained": False,
}

#: Experiment config stem of the probabilistic floor (P6-D6). Closed-form and trivial models
#: only, so it costs minutes and carries no seed variance.
RESIDUAL_FLOOR_EXPERIMENT: str = "e04g_residual_floor"

#: The arm the reproducibility control reads, relative to ``<root>/e04/``. Its closed-form
#: rows *must* equal the reference arm's by construction -- ``revin_applies`` is True only
#: for ``FIT_KIND == "sgd"`` -- so comparing them tests the data path, not RevIN (P6-D13).
REPRODUCIBILITY_ARM_DIR: str = "e04f_revin_ood"

#: Quantities the reproducibility control compares, in report order. The first two are
#: properties of the corpus, the split and the window geometry alone and carry no model, so
#: they are the ones a reader should look at first: if they move, the reference reuse is
#: unsound whatever the model columns say.
REPRODUCIBILITY_QUANTITIES: tuple[str, ...] = (
    "rmse_persistence",
    "signal_std",
    "rmse_mean",
    "mae_mean",
    "skill_mean",
    "nrmse_mean",
)

#: Absolute difference below which two published numbers are called identical. 1e-9 is far
#: below the fourth decimal every table renders and four orders of magnitude below the
#: narrowest bootstrap interval on these cells, so a row inside it cannot move a printed
#: digit or a conclusion.
BITWISE_TOL: float = 1e-9

#: Value of ``model``/``quantity`` on a margin row, i.e. one aggregating over that axis.
#: A margin is taken over models only: a maximum over *quantities* would take the largest of
#: an RMSE in degrees, a skill score and a dimensionless nrmse and print it as one number.
MARGIN_LABEL: str = "any"


@dataclass
class AssembledArtifacts:
    """The four cross-arm tables, and the reason for every one that could not be built.

    Attributes:
        ablations: The paired contrasts of section 6.3, one row per
            (experiment, ablation, arm, model, regime, DOF, horizon, seed), plus the
            placeholder rows for fits that do not exist.
        controls: Every arm's point controls in one table, carrying ``asserted`` and
            ``enforced``.
        probabilistic_baseline: The unconditional residual-interval floor on the
            ``probabilistic.csv`` schema.
        reference_reproducibility: The P6-D13 control, as a committed artifact rather than a
            number in a protocol entry.
        skipped: Why a table, or one arm of one table, is absent. Keyed
            ``"<table>/<what>"``. An assembly that could not run is recorded rather than
            being silently smaller.
        sources: Table name -> the files it was read from, relative to the results root, so
            a row's provenance is answerable without re-running anything.
        paths: Written files, keyed as :data:`ASSEMBLED_ARTIFACTS`.
    """

    ablations: pd.DataFrame
    controls: pd.DataFrame
    probabilistic_baseline: pd.DataFrame
    reference_reproducibility: pd.DataFrame
    skipped: dict[str, str] = field(default_factory=dict)
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def _phase6_dir(results_root: Path) -> Path:
    """Return the directory the Phase 6 tables live in.

    Args:
        results_root: The results **root**, normally ``results/``. A caller that passes
            ``results/e04`` directly is honoured, matching
            :func:`dmf.eval.report.load_report_sources`.

    Returns:
        ``<root>/e04`` if it exists, else ``<root>`` itself.
    """
    nested = results_root / PHASE6_SUBDIR
    return nested if nested.is_dir() else results_root


def _root_of(results_root: Path) -> Path:
    """Return the results root given either the root or its Phase 6 subdirectory.

    Args:
        results_root: The results root or ``<root>/e04``.

    Returns:
        The root, so that ``e02/`` and ``e03/`` resolve beside ``e04/``.
    """
    return results_root.parent if results_root.name == PHASE6_SUBDIR else results_root


def read_arm_frames(
    results_root: Path, arm: str
) -> tuple[tuple[tuple[str, pd.DataFrame], ...], str]:
    """Read one arm's per-seed accuracy rows **per experiment**, or say why it cannot.

    Per experiment and not concatenated, because two experiments can score the same label on
    the same regime: ``e04c_ss_conditioned`` covers ``id`` and ``unseen_seastate`` for the
    deep vehicle and ``e04c_ss_conditioned_ar_id`` covers ``id`` for AR(20) -- and both carry
    ``persistence`` on ``id``, since every config must (it is the skill denominator).
    Concatenating first would give two rows for one join key and the contrast's
    ``validate="one_to_one"`` would refuse the whole arm. Contrasting each file separately
    keeps both rows, distinguishable by ``experiment``, which is the rule
    :func:`dmf.eval.report._group_cols` already applies: two different runs of one label are
    two rows, never one average.

    Args:
        results_root: The results root.
        arm: Arm key.

    Returns:
        ``((experiment, frame), ...)`` and a reason. The tuple is empty exactly when the arm
        contributed nothing, and the reason is non-empty whenever something is missing --
        including a *partially* run arm, which returns frames and a reason together.

    Raises:
        KeyError: If ``arm`` is not in :data:`dmf.eval.ablations.ARMS`.
    """
    spec = ARMS[arm]
    root = _root_of(results_root)
    phase6 = _phase6_dir(results_root)
    if arm in MATCHED_ORIGIN_ARMS:
        path = phase6 / MATCHED_ORIGIN_SUBDIR / arm / ARM_ROWS_FILE
        if path.is_file():
            frame = _tagged(path, arm)
            reason = _unmatched_reason(frame, arm, path)
            return ((), reason) if reason else (((arm, frame),), "")
        return (), (
            f"the lookback contrast is paired only on the intersection of the three origin "
            f"sets (docs/protocol.md P6-D4 item 1), and every committed run scored its own "
            f"arm's full set: 1151 windows per realization at L=100, 1131 at L=200 and 1091 "
            f"at L=400. Differencing those tables would compare forecasts of different "
            f"absolute times. This arm therefore needs a re-scoring on "
            f"dmf.eval.ablations.matched_origins written to {path}; nothing produces one "
            f"yet, so the arm is skipped rather than assembled from the unmatched rows"
        )
    if arm == REFERENCE_ARM:
        path = root / REFERENCE_DIR / ARM_ROWS_FILE
        if path.is_file():
            return ((spec.experiments[0], _tagged(path, spec.experiments[0])),), ""
        return (), f"the reference arm is read from {path}, which is absent"
    paths = [phase6 / name / ARM_ROWS_FILE for name in spec.experiments]
    found = [path for path in paths if path.is_file()]
    if not found:
        return (), (
            f"none of this arm's experiments have been run: looked for "
            f"{[str(path) for path in paths]}"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    reason = "" if not missing else f"partial arm: {missing} absent"
    return tuple((path.parent.name, _tagged(path, path.parent.name)) for path in found), reason


def _unmatched_reason(frame: pd.DataFrame, arm: str, path: Path) -> str:
    """Say why a table found in the matched-origin directory is not a matched table.

    **The directory a file sits in is not evidence.** Copying an arm's own
    ``baselines_by_seed.csv`` -- which scored that arm's full window set -- into
    ``<root>/e04/matched_origins/<arm>/`` would otherwise turn the refusal that P6-D15
    finding 1 rests on into a silent join, and the resulting table would look exactly like a
    correct one. So the file is required to carry
    :data:`dmf.eval.ablations.MATCHED_ORIGIN_COLUMNS`, to carry **one** value of each on
    every row, and to state an origin set that
    :func:`dmf.eval.ablations.unmatched_origin_reason` can re-derive from the four facts the
    intersection depends on. A committed per-arm table has none of those columns and is
    refused on the first check.

    Args:
        frame: The table as read.
        arm: The arm it claims to be.
        path: Where it was read from, for the message.

    Returns:
        Empty if the table is a correct matched table for ``arm``; otherwise the reason it
        is not, phrased as the assembler's refusal.
    """
    if frame.empty:
        return f"the matched table at {path} holds no rows"
    present = [name for name in MATCHED_ORIGIN_COLUMNS if name in frame.columns]
    varying = sorted(name for name in present if frame[name].nunique(dropna=False) != 1)
    if varying:
        return (
            f"the matched table at {path} carries more than one value of {varying}, so its "
            f"rows were not all scored on one origin set. dmf.eval.matched writes one "
            f"provenance per arm, on every row"
        )
    record: dict[str, object] = {name: frame[name].iloc[0] for name in present}
    # Row-wise, not once on the first row: `n_windows` is a per-regime count -- four regimes
    # hold different numbers of realizations -- so a single-row check would silently cover
    # one regime and a uniqueness check would cover none.
    # `matched_n_origins` in the guard as well as in the arithmetic: without it the table has
    # no origin count to check against, and that absence is the *other* refusal below, which
    # says something far more useful than an arithmetic mismatch against NaN.
    if {"n_windows", "n_realizations", "matched_n_origins"} <= set(frame.columns):
        expected = frame["n_realizations"] * frame["matched_n_origins"]
        wrong = frame.loc[(frame["n_realizations"] > 0) & (frame["n_windows"] != expected)]
        if not wrong.empty:
            row = wrong.iloc[0]
            return (
                f"{len(wrong)} row(s) of {path} report a window count that is not the "
                f"declared origin count times the realization count: {int(row['n_windows'])} "
                f"over {int(row['n_realizations'])} realizations against "
                f"{int(row['matched_n_origins'])} matched origins. The rows were scored on a "
                f"different window set from the one the table declares"
            )
    reason = unmatched_origin_reason(record, arm)
    if not reason:
        return ""
    return (
        f"the table at {path} is in the matched-origin directory but is not a matched "
        f"table: {reason}. Rows are read from a re-scoring on "
        f"dmf.eval.ablations.matched_origins (dmf.eval.matched.score_matched_lookback), "
        f"never from an arm's own committed table, whatever directory it is found in"
    )


def _matched_pair_reason(
    arm_rows: pd.DataFrame, reference_rows: pd.DataFrame, arm: str, reference: str
) -> str:
    """Say why two matched tables are not matched **to each other**.

    :func:`_unmatched_reason` checks each side against the intersection it re-derives from
    its own row, which is enough only while both sides describe the same corpus geometry.
    Two arms scored on realizations of different length, or at different strides, each
    reproduce their own claim and pair on nothing -- so the defining facts are compared
    across the pair as well.

    Args:
        arm_rows: The arm's matched rows.
        reference_rows: Its reference arm's matched rows.
        arm: Arm key.
        reference: Reference arm key.

    Returns:
        Empty if the two describe one origin set, otherwise the reason they do not.
    """
    shared = (
        "matched_n_samples",
        "matched_stride",
        "matched_max_horizon",
        "matched_lookbacks",
        "matched_n_origins",
        "matched_origin_first",
        "matched_origin_last",
        "matched_origin_stride",
    )
    for name in shared:
        if name not in arm_rows.columns or name not in reference_rows.columns:
            return f"one side carries no {name!r} column"
        left, right = arm_rows[name].iloc[0], reference_rows[name].iloc[0]
        if left != right:
            return (
                f"arm {arm!r} declares {name}={left!r} and its reference {reference!r} "
                f"declares {right!r}. Each table reproduces its own claim, but the two "
                f"describe different origin sets, so their rows are not paired"
            )
    return ""


def _tagged(path: Path, experiment: str) -> pd.DataFrame:
    """Read one accuracy table and label every row with the run that produced it.

    Args:
        path: The CSV.
        experiment: Provenance label.

    Returns:
        The frame with ``experiment`` as its leading column.
    """
    frame = pd.read_csv(path)
    frame.insert(0, "experiment", experiment)
    return frame


def read_arm_rows(results_root: Path, arm: str) -> tuple[pd.DataFrame | None, str]:
    """Read one arm's per-seed accuracy rows as a single frame.

    The concatenated view of :func:`read_arm_frames`, for the reference side of a contrast
    and for callers that only want the rows. The **arm** side goes through
    :func:`read_arm_frames` instead; see its docstring for why concatenating first can make
    an arm unjoinable.

    Args:
        results_root: The results root.
        arm: Arm key.

    Returns:
        ``(frame, reason)``. The frame is None exactly when the arm contributed nothing.

    Raises:
        KeyError: If ``arm`` is not in :data:`dmf.eval.ablations.ARMS`.
    """
    frames, reason = read_arm_frames(results_root, arm)
    if not frames:
        return None, reason
    return pd.concat([frame for _, frame in frames], ignore_index=True), reason


def read_arm_cells(results_root: Path, arm: str) -> tuple[pd.DataFrame | None, str]:
    """Read one arm's per-grid-cell rows, or say why the arm has none.

    The resampling unit of every ablation interval. This is deliberately a *separate* reader
    from :func:`read_arm_frames`: the per-cell table is not needed to state a contrast, only
    to put an interval on it, so an arm that committed no cell table still contributes rows
    -- with the reason its interval is absent on every one of them. Failing the whole arm
    for a missing uncertainty column would delete a measured difference to protest a missing
    error bar.

    The matched-origin arms are the case this returns None for on the production tree:
    :func:`dmf.eval.matched.score_matched_lookback` re-scores them into pooled rows, and
    reading the *unmatched* per-cell table beside them would resample a window population
    the contrast was not taken over.

    Args:
        results_root: The results root.
        arm: Arm key.

    Returns:
        ``(frame, reason)``, concatenated over the arm's experiments. The frame is None
        exactly when no per-cell table exists.

    Raises:
        KeyError: If ``arm`` is not in :data:`dmf.eval.ablations.ARMS`.
    """
    spec = ARMS[arm]
    root = _root_of(results_root)
    phase6 = _phase6_dir(results_root)
    if arm in MATCHED_ORIGIN_ARMS:
        path = phase6 / MATCHED_ORIGIN_SUBDIR / arm / ARM_CELLS_FILE
        if not path.is_file():
            return None, (
                f"arm {arm!r} is scored on the matched origin set, and the re-scoring at "
                f"{path} has written no per-cell table. Its own experiments' committed "
                f"per-cell rows cover 1151 or 1131 origins per realization, not the 1091 "
                f"the contrast was taken over (P6-D4 item 1), so resampling them would "
                f"put an interval on a different window population"
            )
        return _tagged(path, arm), ""
    if arm == REFERENCE_ARM:
        path = root / REFERENCE_DIR / ARM_CELLS_FILE
        if not path.is_file():
            return None, f"the reference arm's per-cell table is read from {path}, which is absent"
        return _tagged(path, spec.experiments[0]), ""
    paths = [phase6 / name / ARM_CELLS_FILE for name in spec.experiments]
    found = [path for path in paths if path.is_file()]
    if not found:
        return None, (
            f"none of arm {arm!r}'s experiments committed a per-cell table: looked for "
            f"{[str(path) for path in paths]}"
        )
    frame = pd.concat([_tagged(path, path.parent.name) for path in found], ignore_index=True)
    missing = [str(path) for path in paths if not path.is_file()]
    return frame, ("" if not missing else f"partial per-cell coverage: {missing} absent")


def build_ablations_table(
    results_root: Path,
    *,
    columns: Sequence[str] = CROSS_MODE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assemble ``ablations.csv`` from the committed per-arm tables.

    Every arm is paired against its own reference arm by
    :func:`dmf.eval.ablations.contrast_table`, which is where the matching rules live: the
    join is on the *logical* channel so an ``imu`` row cannot silently match nothing, the
    observation-mode guard refuses a raw-scale column across the ``ideal``/``imu`` boundary
    (P6-D4 item 4), and the pairing is seed-wise where both sides carry seeds.

    Args:
        results_root: The results root.
        columns: Metric columns to difference. The default is the two that may cross an
            observation-mode boundary, which is also the pair section 6.3 renders; every arm
            must use the same list, because the rows of the four ablations end up in one
            file and a column present for some arms only would read as a missing measurement.

    Returns:
        ``(frame, skipped)``. The frame carries :data:`dmf.eval.ablations.CONTRAST_COLUMNS`,
        an ``experiment`` provenance column, the requested metrics and the rows that have no
        fit at all. ``skipped`` maps ``"<arm>"`` to the reason it contributed nothing.
    """
    skipped: dict[str, str] = {}
    parts: list[pd.DataFrame] = []
    reference_cache: dict[str, pd.DataFrame | None] = {}
    cell_cache: dict[str, pd.DataFrame | None] = {}

    def _reference(arm: str) -> pd.DataFrame | None:
        if arm not in reference_cache:
            frame, reason = read_arm_rows(results_root, arm)
            reference_cache[arm] = frame
            if reason:
                skipped[f"rows/{arm}"] = reason
        return reference_cache[arm]

    def _cells(arm: str) -> pd.DataFrame | None:
        if arm not in cell_cache:
            frame, reason = read_arm_cells(results_root, arm)
            cell_cache[arm] = frame
            if reason:
                # Recorded, not fatal. The arm still contributes its contrast rows; what it
                # loses is the interval on them, and the row itself says so.
                skipped[f"cells/{arm}"] = reason
        return cell_cache[arm]

    for arm, spec in ARMS.items():
        if arm == spec.reference:
            # A reference is not contrasted against itself. `reference` and `lookback_20s`
            # are both references; the second is also an arm of the lookback ablation, and
            # the lookback ablation reads it as one.
            continue
        arm_frames, arm_reason = read_arm_frames(results_root, arm)
        if arm_reason:
            skipped[f"rows/{arm}"] = arm_reason
        reference_rows = _reference(spec.reference)
        if not arm_frames or reference_rows is None:
            missing = "this arm" if not arm_frames else f"its reference {spec.reference!r}"
            detail = arm_reason if not arm_frames else skipped.get(f"rows/{spec.reference}", "")
            skipped[arm] = f"no contrast: {missing} has no committed rows. {detail}".strip()
            continue
        if arm in MATCHED_ORIGIN_ARMS or spec.reference in MATCHED_ORIGIN_ARMS:
            mismatch = _matched_pair_reason(
                pd.concat([frame for _, frame in arm_frames], ignore_index=True),
                reference_rows,
                arm,
                spec.reference,
            )
            if mismatch:
                skipped[arm] = f"no contrast: {mismatch}"
                continue
        for experiment, arm_rows in arm_frames:
            try:
                parts.append(
                    contrast_table(
                        arm_rows,
                        reference_rows,
                        arm,
                        columns=columns,
                        arm_cells=_cells(arm),
                        reference_cells=_cells(spec.reference),
                    )
                )
            except ValueError as error:
                # One arm's unjoinable file must not stop the other six arms' tables from
                # being written -- `make eval` finishing with a recorded gap is strictly
                # more useful than `make eval` failing. The full message is carried, so the
                # gap is diagnosable from the log rather than only visible as a short table.
                skipped[f"{arm}/{experiment}"] = f"no contrast from {experiment}: {error}"
        placeholders = not_fittable_rows(arm)
        if not placeholders.empty:
            placeholders = placeholders.copy()
            # Blank rather than a guess: no run produced these rows, so naming one would be
            # a provenance claim about a fit that does not exist. They are rendered as their
            # own de-duplicated table and never grouped, so nothing joins on this column.
            placeholders["experiment"] = ""
            parts.append(placeholders)

    if not parts:
        return pd.DataFrame(), skipped
    frame = pd.concat(parts, ignore_index=True)
    ordered = [
        "experiment",
        *[name for name in frame.columns if name != "experiment"],
    ]
    return (
        frame[ordered].sort_values(
            ["ablation", "arm", "regime", "model", "logical_dof", "horizon_samples"],
            kind="stable",
            ignore_index=True,
        ),
        skipped,
    )


def _backfill_control_flags(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Give a control table the ``asserted`` and ``enforced`` columns it may predate.

    Args:
        frame: One arm's control rows.
        source: Path, for the error message.

    Returns:
        A copy carrying both columns. ``asserted`` defaults to True, which is what a table
        written before the P6-D12 narrowing *was*: every scored cell decided the verdict.
        ``enforced`` is taken from :data:`ENFORCED_BY_CONTROL`, which records how the
        training driver calls each control.

    Raises:
        ValueError: If the table names a control the enforcement mapping does not know.
            Guessing would put a commitment on a row that nobody made.
    """
    out = frame.copy()
    if "asserted" not in out.columns:
        out["asserted"] = True
    if "enforced" not in out.columns:
        unknown = sorted(set(out["control"].astype(str)) - set(ENFORCED_BY_CONTROL))
        if unknown:
            raise ValueError(
                f"{source} carries control(s) {unknown} with no `enforced` column and no "
                f"entry in ENFORCED_BY_CONTROL, so whether a failure would have stopped the "
                f"run is unknown. Add the control to the mapping (and to the test that "
                f"checks it against dmf.train.experiment._run_controls) rather than "
                f"defaulting it: a wrong value here claims an enforcement nobody made"
            )
        out["enforced"] = [ENFORCED_BY_CONTROL[str(name)] for name in out["control"]]
    return out


def build_controls_table(results_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assemble ``controls.csv`` from every arm's committed control table.

    Args:
        results_root: The results root.

    Returns:
        ``(frame, skipped)``. The frame is ``experiment``, ``arm`` and
        :data:`dmf.eval.controls.CONTROL_COLUMNS`; ``skipped`` records every declared source
        that is absent. Rows are never pooled across arms here: the two provenance columns
        lead the table, and the renderer groups on them.

    Raises:
        ValueError: If a provenance value is empty, which would round-trip through CSV as
            NaN and be dropped by the renderer's ``groupby`` -- present in the file, absent
            from the table.
    """
    root = _root_of(results_root)
    phase6 = _phase6_dir(results_root)
    sources: list[tuple[Path, str, str]] = [
        (root / directory / ARM_CONTROLS_FILE, experiment, arm)
        for directory, experiment, arm in CONTROL_SOURCE_DIRS
    ]
    experiments = sorted({name for spec in ARMS.values() for name in spec.experiments})
    sources += [
        (phase6 / name / ARM_CONTROLS_FILE, name, arms_for_experiment(name)[0].arm)
        for name in experiments
        if (phase6 / name).is_dir()
    ]

    skipped: dict[str, str] = {}
    parts: list[pd.DataFrame] = []
    for path, experiment, arm in sources:
        if not path.is_file():
            skipped[experiment] = f"no control table at {path}"
            continue
        frame = _backfill_control_flags(pd.read_csv(path), str(path))
        frame.insert(0, "arm", arm)
        frame.insert(0, "experiment", experiment)
        parts.append(frame[["experiment", "arm", *CONTROL_COLUMNS]])
    if not parts:
        return pd.DataFrame(), skipped
    frame = pd.concat(parts, ignore_index=True)
    for column in ("experiment", "arm"):
        blank = frame[column].astype(str).str.strip() == ""
        if bool(blank.any()):
            raise ValueError(
                f"{int(blank.sum())} control row(s) carry an empty {column!r}. The renderer "
                f"groups the control summary on it, an empty value reads back from CSV as "
                f"NaN, and pandas drops NaN group keys -- so those rows would be in the "
                f"file and absent from the table. Use NOT_AN_ARM rather than an empty string"
            )
    return frame, skipped


def build_probabilistic_baseline(results_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assemble ``probabilistic_baseline.csv``: the residual-interval floor (P6-D6).

    The floor is a ``dlinear_ols`` point forecast plus per-(DOF, horizon) empirical residual
    quantiles taken from the **validation** split -- never train, which would be in-sample
    and over-tight, and never test, which would be the leak the split policy exists to
    prevent. It is unconditional, so on ``id`` it is near-perfectly calibrated with no
    learning at all, which is exactly what makes it discriminating: a head beats it only by
    making its interval conditional.

    **The floor's rows and the Phase 5 heads' rows stay in two files.**
    :func:`dmf.eval.report.build_probabilistic_view` refuses a probabilistic frame holding
    more than one experiment, because that schema is not keyed on the experiment and
    aggregating it would average two different runs of one label into one row. This table is
    therefore written on the ``probabilistic.csv`` schema so it *joins* the committed
    ``results/e03/probabilistic.csv`` -- same columns, disjoint model labels -- and the
    renderer shows the two side by side rather than merged.

    Args:
        results_root: The results root.

    Returns:
        ``(frame, skipped)``. The frame is aggregated onto
        :data:`dmf.eval.report.PROBABILISTIC_COLUMNS`; ``skipped`` explains an absent table
        or an unchecked join.
    """
    phase6 = _phase6_dir(results_root)
    root = _root_of(results_root)
    path = phase6 / RESIDUAL_FLOOR_EXPERIMENT / "probabilistic_by_seed.csv"
    if not path.is_file():
        return pd.DataFrame(), {
            "probabilistic_baseline": (
                f"no probabilistic floor at {path}. It is produced by "
                f"`scripts/train.py --config configs/experiment/{RESIDUAL_FLOOR_EXPERIMENT}"
                f".yaml --results-dir {phase6 / RESIDUAL_FLOOR_EXPERIMENT}`, which is "
                f"closed-form on all four regimes and costs minutes. Section 6.4 without it "
                f"reports six learned heads and no trivial comparator, which is the Phase 5 "
                f"gap P6-D6 exists to close"
            )
        }
    frame = build_probabilistic_table(pd.read_csv(path))
    skipped: dict[str, str] = {}
    heads = root / "e03" / "probabilistic.csv"
    if heads.is_file():
        _assert_joins_heads(frame, pd.read_csv(heads), str(heads))
    else:
        skipped["probabilistic_baseline/join"] = (
            f"{heads} is absent, so the floor's rows could not be checked for label "
            f"collisions against the committed Phase 5 heads"
        )
    return frame, skipped


def _assert_joins_heads(floor: pd.DataFrame, heads: pd.DataFrame, source: str) -> None:
    """Refuse a floor table that would collide with the committed heads.

    The two tables are read as one section. If a label appeared in both, a reader
    concatenating them would get two rows for one cell and no way to tell which run each came
    from -- the schema carries no experiment.

    Args:
        floor: The assembled floor table.
        heads: The committed Phase 5 table.
        source: Path of the committed table, for the message.

    Raises:
        ValueError: If any (model, head, regime, DOF, horizon) key is in both tables.
    """
    keys = ["model", "head", "regime", "dof", "horizon_samples"]
    if not set(keys) <= set(heads.columns):
        return
    overlap = floor.merge(heads[keys].drop_duplicates(), on=keys, how="inner")
    if not overlap.empty:
        offending = sorted(set(overlap["model"].astype(str)))
        raise ValueError(
            f"the residual-interval floor and {source} share {len(overlap)} cell(s) on "
            f"model(s) {offending}. The probabilistic schema is not keyed on the experiment, "
            f"so two rows for one cell cannot be told apart; score the floor under a label "
            f"of its own rather than reusing a Phase 5 one"
        )


def _difference_rows(joined: pd.DataFrame, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the absolute and relative differences of one quantity.

    Args:
        joined: The paired rows, carrying ``<quantity>`` from the arm and
            ``<quantity>_reference`` from the reference.
        quantity: Column name.

    Returns:
        ``(absolute, relative)``. The relative figure is undefined where the reference is
        exactly zero and is NaN there rather than infinite: a difference of zero against a
        reference of zero is a reproduction, not an infinite error, and every such cell on
        this corpus (``skill`` of persistence, which is 0 by construction) is one.
    """
    arm = joined[quantity].to_numpy(dtype=float)
    reference_values = joined[f"{quantity}_reference"].to_numpy(dtype=float)
    absolute = np.abs(arm - reference_values)
    reference = np.abs(reference_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(reference > 0.0, absolute / reference, np.nan)
    return absolute, relative


def build_reference_reproducibility(results_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Assemble ``reference_reproducibility.csv``: the P6-D13 control, as an artifact.

    Six of the seven ablation arms are contrasted against ``results/e02/`` **read rather than
    re-run**. That reuse is an assumption about the data path, and this control is what turns
    it into a measurement: the RevIN arm's closed-form rows must equal the reference arm's by
    construction (``revin_applies`` is True only for ``FIT_KIND == "sgd"``), and it was fitted
    independently, weeks later, under a changed codebase, on the two regimes ``results/e02/``
    also covers. A deterministic closed-form row that failed to reproduce would invalidate the
    reuse that section 6.3 depends on.

    **The table is a (model, quantity) grid with a margin over models.** P6-D13 reported two
    tables -- one by quantity and one by model -- and the second's "70 of 72 rows differ" is a
    count on ``rmse_mean`` alone, which the entry does not say. Crossing the two axes makes
    that explicit: the same join gives 71 differing rows on ``nrmse_mean`` and 50 on
    ``skill_mean``, all of the same ~5e-7 in ``rmse``, and a reader can see which column each
    figure came from. There is no margin over quantities, because a maximum over an RMSE in
    degrees, a dimensionless nrmse and a skill score is not a number.

    Args:
        results_root: The results root.

    Returns:
        ``(frame, skipped)``. One row per (model, quantity) plus a ``model = "any"`` margin
        per quantity, carrying the row count, how many rows differ by more than
        :data:`BITWISE_TOL`, and the largest absolute and relative differences.
    """
    root = _root_of(results_root)
    phase6 = _phase6_dir(results_root)
    arm_path = phase6 / REPRODUCIBILITY_ARM_DIR / ARM_SUMMARY_FILE
    reference_path = root / REFERENCE_DIR / ARM_SUMMARY_FILE
    missing = [str(path) for path in (arm_path, reference_path) if not path.is_file()]
    if missing:
        return pd.DataFrame(), {
            "reference_reproducibility": (
                f"cannot run: {missing} absent. This control is what makes reading "
                f"results/e02/ as the reference arm a measurement rather than an assumption "
                f"(docs/protocol.md P6-D13)"
            )
        }
    arm = pd.read_csv(arm_path)
    reference = pd.read_csv(reference_path)
    keys = ["model", "regime", "dof", "horizon_samples"]
    quantities = [
        name
        for name in REPRODUCIBILITY_QUANTITIES
        if name in arm.columns and name in reference.columns
    ]
    joined = arm.merge(
        reference[[*keys, *quantities]],
        on=keys,
        how="inner",
        suffixes=("", "_reference"),
        validate="one_to_one",
    )
    if joined.empty:
        return pd.DataFrame(), {
            "reference_reproducibility": (
                f"{arm_path} and {reference_path} share no (model, regime, dof, horizon) "
                f"row, so nothing could be compared"
            )
        }
    regimes = ",".join(sorted(joined["regime"].astype(str).unique()))
    rows: list[dict[str, object]] = []
    for quantity in quantities:
        absolute, relative = _difference_rows(joined, quantity)
        differing = absolute > BITWISE_TOL
        models = joined["model"].astype(str).to_numpy()
        for model in [*sorted(set(models)), MARGIN_LABEL]:
            mask = np.ones(len(joined), dtype=bool) if model == MARGIN_LABEL else models == model
            rows.append(
                {
                    "arm": REPRODUCIBILITY_ARM_DIR,
                    "arm_source": str(arm_path.relative_to(root)),
                    "reference_source": str(reference_path.relative_to(root)),
                    "regimes": regimes,
                    "model": model,
                    "quantity": quantity,
                    "n_rows": int(mask.sum()),
                    "n_differing": int(np.count_nonzero(differing[mask])),
                    "max_abs_diff": float(np.max(absolute[mask])),
                    "max_rel_diff": float(_nanmax_or_zero(relative[mask])),
                    "tol": BITWISE_TOL,
                    "bitwise_identical": bool(not differing[mask].any()),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["quantity", "model"], kind="stable", ignore_index=True)
    return frame, {}


def _nanmax_or_zero(values: np.ndarray) -> float:
    """Return the largest finite value, or 0.0 when every entry is undefined.

    Args:
        values: Relative differences, possibly all NaN.

    Returns:
        The maximum ignoring NaN, or 0.0 if there is nothing to take a maximum of. Zero and
        not NaN: every cell whose relative difference is undefined has an *absolute*
        difference of zero on this corpus, so "no relative disagreement" is the honest
        reading, and the absolute column beside it carries the evidence.
    """
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def assemble_phase6_tables(
    results_root: Path,
    *,
    out_dir: Path | None = None,
    write: bool = True,
) -> AssembledArtifacts:
    """Build the four cross-arm tables and write them beside the scored ones.

    Args:
        results_root: The results **root**, normally ``results/``. ``e02/`` and ``e03/`` are
            read from beside ``e04/`` and are never written to.
        out_dir: Where to write. Defaults to ``<root>/e04``, which is where
            :func:`dmf.eval.report.load_report_sources` looks.
        write: Whether to write the CSVs. False is for tests and for callers that want the
            frames only.

    Returns:
        The artifacts, with a reason recorded for every table that could not be built. An
        empty table is **not** written: an empty CSV in ``results/`` cannot be told from a
        table with nothing to report, and :attr:`AssembledArtifacts.skipped` is where the
        absence is explained.
    """
    destination = out_dir if out_dir is not None else _phase6_dir(results_root)
    ablations, ablation_skips = build_ablations_table(results_root)
    controls, control_skips = build_controls_table(results_root)
    probabilistic, probabilistic_skips = build_probabilistic_baseline(results_root)
    reproducibility, reproducibility_skips = build_reference_reproducibility(results_root)

    skipped: dict[str, str] = {}
    for table, part in (
        ("ablations", ablation_skips),
        ("controls", control_skips),
        ("probabilistic_baseline", probabilistic_skips),
        ("reference_reproducibility", reproducibility_skips),
    ):
        for key, reason in part.items():
            skipped[f"{table}/{key}"] = reason

    artifacts = AssembledArtifacts(
        ablations=ablations,
        controls=controls,
        probabilistic_baseline=probabilistic,
        reference_reproducibility=reproducibility,
        skipped=skipped,
    )
    if write:
        for name, filename in ASSEMBLED_ARTIFACTS.items():
            frame = getattr(artifacts, name)
            if not frame.empty:
                artifacts.paths[name] = write_table(frame, destination / filename)
    return artifacts
