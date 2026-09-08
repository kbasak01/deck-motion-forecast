"""The Gate 4 read-out, computed from the committed artifacts by repository code.

Gate 4 is a decision, and a decision that is taken by an ad-hoc script is a decision that
cannot be re-taken. Everything here reads ``results/<dir>/baselines.csv`` (and, when it is
there, ``results/<dir>/paired_contrasts.csv``) and produces the verdict table -- so the
number a reader sees and the number the gate was read at are the same object.

**Two readings ship, and both are computed.** They differ only in which cell the criterion
is evaluated at; the criterion itself is identical.

- **Reading A** -- the original, verbatim from ``docs/IMPLEMENTATION_PLAN.md`` §Phase 4:
  "all deep models beat damped persistence at 3 s horizon on the ``id`` regime by a margin
  exceeding the seed-to-seed standard deviation". Read at 30 samples, reference
  ``damped_persistence``, over every DOF the table reports at that cell.
- **Reading B** -- the restatement recorded in ``docs/protocol.md`` P4-D1 and the one the
  gate is read at: the decision horizon (100 samples / 10 s) on the binding DOF (pitch),
  against the **stronger of ``damped_persistence`` and ``window_mean``, resolved per cell**.
  ``window_mean`` is a zero-parameter forecast and it beats ``damped_persistence`` in 54 of
  144 cells (P3-D20), so which of the two is the reference is a measured fact about the
  cell, not an assumption -- :data:`GATE4_COLUMNS` carries both candidates' skill in
  ``reference_pool`` so the resolution can be checked rather than trusted.

Reading A is reported whether or not it passes. Per the user instruction recorded in P4-D1,
a deep model failing to beat damped persistence by more than its seed spread *is* the
result, and CLAUDE.md non-negotiable 6 forbids dropping it or relegating it to a footnote:
every row appears in the table body with its verdict.

**The margin rule, and its boundary.** ``margin = skill_mean(model) - skill_mean(reference)``
and the verdict is :data:`VERDICT_PASS` iff ``margin > skill_std(model)``, **strictly**.
``margin == skill_std`` is :data:`VERDICT_FAIL`: the criterion says "exceeding", and a
margin equal to the spread does not exceed it.

**A NaN standard deviation is never a pass.** Deterministic models carry ``skill_std = NaN``
by design (P3-D10) -- with one observation the sample standard deviation is undefined, and
NaN is how that is written. ``margin > NaN`` is ``False`` in IEEE arithmetic, so a naive
comparison would silently *fail* such a row, and any code that "fixed" that by filling the
NaN with 0.0 would make every positive margin pass. Neither is acceptable for a deep model,
which is stochastic and must carry a real spread: a NaN (or non-finite, or fewer than
:data:`dmf.eval.report.MIN_SEEDS` seeds) std on a deep model row is marked
:data:`VERDICT_UNVERIFIED` and named in the rendered document. It is an unmeasured gate, not
a passed one and not a failed one.

**The margin test and the paired bootstrap answer different questions, and both are
reported.** The margin test asks whether the model's advantage is larger than the spread of
its own fitting procedure across seeds. The paired contrast (P4-D4) asks whether the
advantage survives resampling the held-out realizations, with both models scored on the same
resample so the realization-to-realization severity cancels. A margin can exceed the seed std
while the paired interval still spans zero, and the reverse. They are two columns, never one
verdict.

Units: ``skill_mean``, ``skill_std``, ``margin``, ``nrmse_mean``, ``skill_diff`` and the
paired interval bounds are all dimensionless. ``horizon_samples`` is samples at the corpus
rate; ``horizon_s`` is seconds. Nothing here is in corpus units, because nothing here is an
error magnitude -- ``nrmse_mean`` rides along precisely so a cell can be read without the
horizon-comparability trap of P3-D5, where the skill denominator oscillates with the roll
period and a skill-vs-horizon reading attributes a property of persistence to the model.

Simulated results only; no real deck data enters any artifact this module reads.
"""

import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dmf.eval.report import (
    MIN_SEEDS,
    _display_dir,
    _resolve_gate_dof,
    to_markdown,
    write_table,
)

__all__ = [
    "GATE4_COLUMNS",
    "GATE4_DEEP_MODELS",
    "GATE4_READINGS",
    "GATE4_REGIME",
    "PAIRED_ABSENT",
    "PAIRED_BELOW_ZERO",
    "PAIRED_EXCEEDS_ZERO",
    "PAIRED_SPANS_ZERO",
    "READING_A",
    "READING_B",
    "VERDICT_FAIL",
    "VERDICT_PASS",
    "VERDICT_UNVERIFIED",
    "DEGRADATION_COLUMNS",
    "GATE5_COLUMNS",
    "GATE5_DOF",
    "GATE5_HORIZON_SAMPLES",
    "GATE5_PICP_BAND",
    "GATE5_REGIME",
    "GATE6_COLUMNS",
    "GATE6_CRITERIA",
    "GATE6_MARKER_SPEC",
    "GATE6_READING",
    "GATE6_REQUIRED_TABLES",
    "GATE6_RESULTS_MD",
    "GATE6_TABLE_COLUMNS",
    "Gate6Evidence",
    "GateReading",
    "RenderedTable",
    "build_gate4_markdown",
    "build_gate5_markdown",
    "build_gate6_markdown",
    "coverage_degradation",
    "gate4_notes",
    "gate4_readout",
    "gate5_notes",
    "gate5_readout",
    "gate5_reading_passes",
    "gate6_notes",
    "gate6_readout",
    "gate6_reading_passes",
    "gate6_table_audit",
    "parse_rendered_tables",
    "read_gate4_inputs",
    "read_gate5_inputs",
    "read_gate6_inputs",
    "reading_passes",
    "write_gate4_report",
    "write_gate5_report",
    "write_gate6_report",
]

#: Regime both readings are taken on. ``id`` is what Gate 4 names; the out-of-distribution
#: regimes are reported by the baselines tables and are not what this gate turns on.
GATE4_REGIME = "id"

#: The Phase 4 architectures the gate is about. Deliberately **not** imported from
#: :data:`dmf.train.experiment.CONTRAST_DEEP_MODELS`: that module pulls in torch and the
#: whole training stack, and this is a pandas-only reading layer (the same argument P3-D14
#: made for :mod:`dmf.data.channels`). The duplication is pinned by a test that asserts the
#: two tuples are equal, which is the P4-D4 pattern -- a fact stated twice and checked, not
#: stated twice and hoped over.
GATE4_DEEP_MODELS: tuple[str, ...] = ("tcn", "transformer", "lstm")

#: Verdict of the margin criterion: the margin strictly exceeds the seed-to-seed std.
VERDICT_PASS = "PASS"

#: Verdict of the margin criterion: the margin does not exceed the seed-to-seed std. This
#: includes the exact boundary ``margin == skill_std``; "exceeding" is strict.
VERDICT_FAIL = "FAIL"

#: Verdict of the margin criterion when it could not be evaluated: a deep model row whose
#: ``skill_std`` is not a finite number, or which carries fewer than :data:`MIN_SEEDS`
#: seeds. Never a pass -- and never silently a fail either, because the gate was not
#: measured rather than measured and missed.
VERDICT_UNVERIFIED = "UNVERIFIED"

#: Paired-bootstrap outcome: the interval lies entirely above zero, i.e. the deep model
#: beats the reference on resampled realizations, in every seed of the join.
PAIRED_EXCEEDS_ZERO = "excludes 0 (model better)"

#: Paired-bootstrap outcome: the interval lies entirely below zero, i.e. the *reference*
#: beats the deep model.
PAIRED_BELOW_ZERO = "excludes 0 (reference better)"

#: Paired-bootstrap outcome: the interval contains zero. Note that this can coexist with a
#: :data:`VERDICT_PASS` margin, which is why the two are separate columns.
PAIRED_SPANS_ZERO = "spans 0"

#: No ``paired_contrasts.csv`` row matched this (model, reference, regime, DOF, horizon).
#: The read-out degrades to the margin test alone and says so.
PAIRED_ABSENT = "n/a"


@dataclass(frozen=True)
class GateReading:
    """One cell the Gate 4 criterion is evaluated at.

    Attributes:
        key: Short label, ``"A"`` or ``"B"``, carried in the ``reading`` column.
        title: One-line description of what the reading is and where it comes from.
        horizon_samples: Lead time the criterion is read at, in samples at the corpus rate.
        references: Candidate reference models, in tie-break order. More than one means the
            reference is resolved per cell as the ``argmax`` of ``skill_mean`` among them;
            an exact tie resolves to the earlier entry, so the resolution is deterministic.
        dofs: Logical DOF names the reading covers, or ``None`` for "every DOF the table
            reports at this cell". Names are logical (``"pitch"``) and are resolved to the
            spelling the run actually scored (``"pitch_imu"`` under ``observation_mode:
            imu``) by :func:`dmf.eval.report._resolve_gate_dof`, never by a suffix rule --
            ``roll_rate_imu`` is the near-miss a substring rule mis-resolves to roll
            (P3-D14).
    """

    key: str
    title: str
    horizon_samples: int
    references: tuple[str, ...]
    dofs: tuple[str, ...] | None


#: The original criterion, verbatim from ``docs/IMPLEMENTATION_PLAN.md`` §Phase 4: damped
#: persistence, 3 s (30 samples at 10 Hz), ``id``, every DOF.
READING_A = GateReading(
    key="A",
    title="original criterion (docs/IMPLEMENTATION_PLAN.md Phase 4): 3 s vs damped_persistence",
    horizon_samples=30,
    references=("damped_persistence",),
    dofs=None,
)

#: The restated criterion (``docs/protocol.md`` P4-D1) and the one the gate is read at: the
#: decision horizon (10 s / 100 samples) on the binding DOF (pitch), against the stronger of
#: the two trivial baselines in that exact cell.
READING_B = GateReading(
    key="B",
    title=(
        "restated criterion (docs/protocol.md P4-D1): 10 s on pitch vs the stronger of "
        "damped_persistence and window_mean, resolved per cell"
    ),
    horizon_samples=100,
    references=("damped_persistence", "window_mean"),
    dofs=("pitch",),
)

#: Both readings, in report order. Reading A first because it is the one the plan states.
GATE4_READINGS: tuple[GateReading, ...] = (READING_A, READING_B)

#: Exact column order of the emitted verdict table (``gate4.csv``).
#:
#: ``margin = skill_mean - reference_skill_mean``; ``verdict`` is
#: :data:`VERDICT_PASS` iff ``margin > skill_std`` strictly. ``reference_pool`` records every
#: candidate's ``skill_mean`` in the cell, so a reader can check the resolution rather than
#: take it. ``paired_*`` come from ``paired_contrasts.csv`` and are a **different question**
#: from the margin test; ``paired_ci_lo``/``paired_ci_hi`` on a multi-seed join are the
#: envelope (lowest low, highest high) of the per-seed intervals, following the
#: :func:`dmf.eval.report.build_baselines_table` precedent, never their mean.
GATE4_COLUMNS: tuple[str, ...] = (
    "reading",
    "reading_title",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "model",
    "n_seeds",
    "skill_mean",
    "skill_std",
    "nrmse_mean",
    "reference",
    "reference_skill_mean",
    "reference_pool",
    "margin",
    "verdict",
    "verdict_reason",
    "paired_model_b",
    "paired_n_seeds",
    "paired_skill_diff",
    "paired_ci_lo",
    "paired_ci_hi",
    "paired_verdict",
)

#: Columns of ``gate4.csv`` rendered into the per-reading Markdown tables. The full frame is
#: wider than a readable table, but nothing load-bearing is dropped: ``verdict`` is here,
#: and ``reference_pool`` and ``verdict_reason`` get their own sections rather than being
#: cut.
_RENDERED_COLUMNS: tuple[str, ...] = (
    "model",
    "dof",
    "n_seeds",
    "skill_mean",
    "nrmse_mean",
    "skill_std",
    "reference",
    "reference_skill_mean",
    "margin",
    "verdict",
    "paired_skill_diff",
    "paired_ci_lo",
    "paired_ci_hi",
    "paired_verdict",
)

#: Columns :func:`gate4_readout` needs in the aggregated baselines table.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "model",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_seeds",
    "skill_mean",
    "skill_std",
)

#: Columns :func:`gate4_readout` needs in ``paired_contrasts.csv`` before it will join it.
#: A file missing any of them is reported as unusable rather than joined on a guess.
_CONTRAST_COLUMNS: tuple[str, ...] = (
    "regime",
    "model_a",
    "model_b",
    "seed",
    "dof",
    "horizon_samples",
    "skill_diff",
    "ci_lo",
    "ci_hi",
)


def _scope(table: pd.DataFrame, regime: str, reading: GateReading) -> pd.DataFrame:
    """Restrict an aggregated baselines table to one reading's regime and horizon.

    Args:
        table: Aggregated table in the :data:`dmf.eval.report.BASELINES_COLUMNS` schema.
        regime: Regime the gate is read on.
        reading: The reading whose horizon selects the rows.

    Returns:
        The rows at that (regime, horizon).

    Raises:
        ValueError: If a required column is missing, or if the cell is empty. An empty gate
            cell is a real gap in the run -- the wrong regime scored, or the horizon not
            reported -- and returning an empty verdict would present that gap as "nothing
            failed".
    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"the baselines table is missing columns {missing}")
    scoped = table[
        (table["regime"] == regime) & (table["horizon_samples"] == reading.horizon_samples)
    ]
    if scoped.empty:
        raise ValueError(
            f"reading {reading.key}: the gate cell (regime={regime!r}, "
            f"horizon_samples={reading.horizon_samples}) is absent from the table; the "
            f"regimes present are {sorted(table['regime'].unique())} and the horizons "
            f"present are {sorted(table['horizon_samples'].unique())}"
        )
    return scoped


def _reading_dofs(scoped: pd.DataFrame, reading: GateReading, regime: str) -> tuple[str, ...]:
    """Resolve the DOF spellings a reading covers in this table.

    Args:
        scoped: Rows already restricted to the reading's regime and horizon.
        reading: The reading, whose ``dofs`` is either explicit logical names or ``None``
            for every DOF present.
        regime: Regime, for the error message only.

    Returns:
        The corpus spellings actually present, in table order.

    Raises:
        ValueError: If a named DOF is absent under either of its spellings, or is not a
            corpus motion channel at all.
    """
    if reading.dofs is None:
        return tuple(str(name) for name in scoped["dof"].unique())
    resolved: list[str] = []
    for dof in reading.dofs:
        found = _resolve_gate_dof(scoped, dof)
        if found is None:
            raise ValueError(
                f"reading {reading.key}: the gate cell (regime={regime!r}, dof={dof!r} or "
                f"its imu twin, horizon_samples={reading.horizon_samples}) is absent from "
                f"the table; the DOFs present at that regime and horizon are "
                f"{sorted(scoped['dof'].unique())}"
            )
        resolved.append(found)
    return tuple(resolved)


def _one_row(cell: pd.DataFrame, model: str, reading: GateReading, dof: str) -> dict[str, Any]:
    """Return the single aggregated row for one model in one cell.

    Args:
        cell: Rows restricted to the reading's regime, horizon and one DOF.
        model: Model label to look for.
        reading: The reading, for the error message.
        dof: DOF spelling, for the error message.

    Returns:
        That model's row as a mapping from column name to value.

    Raises:
        ValueError: If the model is absent from the cell, or if the cell holds more than one
            row for it -- the table is one row per (model, regime, DOF, horizon), so a
            duplicate means the frame was concatenated from two runs and picking either
            row would silently report one of them.
    """
    rows = cell[cell["model"] == model]
    if rows.empty:
        raise ValueError(
            f"reading {reading.key}: model {model!r} is absent from the table at "
            f"(dof={dof!r}, horizon_samples={reading.horizon_samples}); the models present "
            f"in that cell are {sorted(cell['model'].unique())}"
        )
    if len(rows) > 1:
        raise ValueError(
            f"reading {reading.key}: model {model!r} has {len(rows)} rows at (dof={dof!r}, "
            f"horizon_samples={reading.horizon_samples}); the aggregated table is one row "
            f"per cell, so this frame holds more than one run's table"
        )
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def _resolve_reference(
    cell: pd.DataFrame, reading: GateReading, dof: str
) -> tuple[str, float, str]:
    """Pick the reference the criterion is read against in this cell.

    With one candidate this is that candidate. With several -- Reading B's
    ``damped_persistence`` vs ``window_mean`` -- it is the ``argmax`` of ``skill_mean``
    **in this exact cell**, because which of the two is stronger is a measured property of
    the cell (P3-D20 records both directions occurring across the grid) and assuming either
    one would set the bar wherever the assumption happened to fall. An exact tie resolves to
    the earlier entry of ``reading.references``, so the choice is deterministic.

    Args:
        cell: Rows restricted to the reading's regime, horizon and one DOF.
        reading: The reading, whose ``references`` are the candidates.
        dof: DOF spelling, for the error message.

    Returns:
        ``(name, skill_mean, pool)`` where ``pool`` renders every candidate's ``skill_mean``
        so the resolution is auditable from the row itself.

    Raises:
        ValueError: If any candidate is absent from the cell, or carries a non-finite
            ``skill_mean``. The restated gate is defined as the maximum over both
            candidates; quietly falling back to whichever one is present would report a
            weaker bar under the stronger criterion's name.
    """
    skills: dict[str, float] = {}
    for name in reading.references:
        row = _one_row(cell, name, reading, dof)
        value = float(row["skill_mean"])
        if not np.isfinite(value):
            raise ValueError(
                f"reading {reading.key}: reference {name!r} at (dof={dof!r}, "
                f"horizon_samples={reading.horizon_samples}) has a non-finite skill_mean "
                f"({value}); the margin has no reference"
            )
        skills[name] = value
    order = list(reading.references)
    winner = max(order, key=lambda name: (skills[name], -order.index(name)))
    pool = ", ".join(f"{name}={skills[name]:.4f}" for name in order)
    return winner, skills[winner], pool


def _margin_verdict(margin: float, skill_std: float, n_seeds: int, model: str) -> tuple[str, str]:
    """Apply the margin criterion to one row.

    The rule is ``margin > skill_std``, strictly: the criterion says "exceeding", so the
    exact boundary ``margin == skill_std`` is a :data:`VERDICT_FAIL`.

    A non-finite ``skill_std`` is :data:`VERDICT_UNVERIFIED`, never a pass and never a
    silent fail. NaN is how "not measured" is written for a deterministic model (P3-D10);
    a deep model is stochastic, so a NaN there means the row does not carry the quantity
    the criterion compares against. Fewer than :data:`MIN_SEEDS` seeds is the same
    condition seen from the other side and is treated identically.

    Args:
        margin: ``skill_mean(model) - skill_mean(reference)``, dimensionless.
        skill_std: Seed-to-seed standard deviation of the model's skill, dimensionless.
        n_seeds: Number of seeds behind ``skill_std``.
        model: Model label, for the reason string.

    Returns:
        ``(verdict, reason)``. The reason is always populated, including on a pass, so a
        row can be read without recomputing the comparison.

    Raises:
        ValueError: If ``skill_std`` is negative, which is not a spread.
    """
    if skill_std < 0.0:
        raise ValueError(
            f"{model!r} carries a negative skill_std ({skill_std}); that is not a spread"
        )
    if not np.isfinite(skill_std):
        return (
            VERDICT_UNVERIFIED,
            f"skill_std is not a finite number ({skill_std}) on n_seeds={n_seeds}: the "
            f"seed spread the criterion compares the margin against was never measured, "
            f"so this cell is unverified rather than passed",
        )
    if n_seeds < MIN_SEEDS:
        return (
            VERDICT_UNVERIFIED,
            f"n_seeds={n_seeds} is below the {MIN_SEEDS}-seed minimum (CLAUDE.md "
            f"non-negotiable 5), so skill_std={skill_std:.6f} is not a seed-to-seed spread",
        )
    if margin > skill_std:
        return (
            VERDICT_PASS,
            f"margin {margin:.6f} exceeds skill_std {skill_std:.6f}",
        )
    if margin == skill_std:
        return (
            VERDICT_FAIL,
            f"margin {margin:.6f} equals skill_std {skill_std:.6f} exactly; the criterion "
            f"is 'exceeding', which is strict",
        )
    return (
        VERDICT_FAIL,
        f"margin {margin:.6f} does not exceed skill_std {skill_std:.6f}",
    )


def _paired_columns(
    contrasts: pd.DataFrame | None,
    *,
    regime: str,
    model: str,
    reference: str,
    dof: str,
    horizon_samples: int,
) -> dict[str, object]:
    """Join the matching paired-bootstrap contrast, if there is one.

    ``paired_contrasts.csv`` holds one row per (regime, pair, seed, DOF, horizon) and emits
    no aggregated view on purpose (P4-D4), so this reduces over seeds here: the point
    estimate is the mean of the per-seed ``skill_diff``, and the interval is the
    **envelope** (lowest ``ci_lo``, highest ``ci_hi``) of the per-seed intervals. The
    envelope, not their mean: no arithmetic on finished intervals yields a calibrated
    interval for the seed mean, the mean of them is narrower than any input justifies and
    is invariant to seed disagreement, and the envelope makes "excludes zero" mean "in
    every seed". This is the same rule and the same reasoning as
    :func:`dmf.eval.report.build_baselines_table`.

    Args:
        contrasts: The contrasts frame, or ``None`` when the run wrote none.
        regime: Regime of the gate cell.
        model: Deep model, joined as ``model_a``.
        reference: Resolved reference, joined as ``model_b``.
        dof: DOF spelling of the gate cell.
        horizon_samples: Horizon of the gate cell, samples.

    Returns:
        The five ``paired_*`` columns. When no row matches -- an absent file, or a pair the
        run did not configure -- ``paired_verdict`` is :data:`PAIRED_ABSENT` and the
        numeric columns are NaN, so the read-out degrades to the margin test alone rather
        than failing.
    """
    absent: dict[str, object] = {
        "paired_model_b": reference,
        "paired_n_seeds": float("nan"),
        "paired_skill_diff": float("nan"),
        "paired_ci_lo": float("nan"),
        "paired_ci_hi": float("nan"),
        "paired_verdict": PAIRED_ABSENT,
    }
    if contrasts is None or contrasts.empty:
        return absent
    if any(name not in contrasts.columns for name in _CONTRAST_COLUMNS):
        return absent
    matched = contrasts[
        (contrasts["regime"] == regime)
        & (contrasts["model_a"] == model)
        & (contrasts["model_b"] == reference)
        & (contrasts["dof"] == dof)
        & (contrasts["horizon_samples"] == horizon_samples)
    ]
    if matched.empty:
        return absent
    lo = float(matched["ci_lo"].min())
    hi = float(matched["ci_hi"].max())
    if lo > 0.0:
        verdict = PAIRED_EXCEEDS_ZERO
    elif hi < 0.0:
        verdict = PAIRED_BELOW_ZERO
    else:
        verdict = PAIRED_SPANS_ZERO
    return {
        "paired_model_b": reference,
        "paired_n_seeds": float(matched["seed"].nunique()),
        "paired_skill_diff": float(matched["skill_diff"].mean()),
        "paired_ci_lo": lo,
        "paired_ci_hi": hi,
        "paired_verdict": verdict,
    }


def gate4_readout(
    table: pd.DataFrame,
    *,
    contrasts: pd.DataFrame | None = None,
    readings: Sequence[GateReading] = GATE4_READINGS,
    deep_models: Sequence[str] = GATE4_DEEP_MODELS,
    regime: str = GATE4_REGIME,
) -> pd.DataFrame:
    """Compute the Gate 4 verdict table from an aggregated baselines table.

    One row per (reading, model, DOF), both readings, every deep model, no row dropped for
    failing (CLAUDE.md non-negotiable 6). See the module docstring for the criterion, the
    strict boundary, and why a NaN ``skill_std`` is :data:`VERDICT_UNVERIFIED`.

    Args:
        table: Aggregated table in the :data:`dmf.eval.report.BASELINES_COLUMNS` schema,
            i.e. ``baselines.csv``. ``nrmse_mean`` is used if present and reported as NaN
            if not -- every Phase 3 artifact predates that column (P4-D3).
        contrasts: ``paired_contrasts.csv`` as a frame, or ``None``. Optional: its absence
            leaves the ``paired_*`` columns empty and is noted in the rendered document,
            not an error.
        readings: Cells to read the criterion at. Defaults to both.
        deep_models: Models the gate is about.
        regime: Regime the gate is read on.

    Returns:
        A frame with columns :data:`GATE4_COLUMNS`, ordered by reading, then DOF as the
        table orders them, then ``deep_models`` order.

    Raises:
        ValueError: If a required column is missing; if a gate cell is absent (wrong
            regime, unreported horizon, missing DOF, missing model, or a missing reference
            candidate); if a model appears twice in one cell; or if a ``skill_std`` is
            negative. A missing gate cell raises rather than returning an empty verdict,
            because "no rows" and "nothing failed" render identically.
    """
    rows: list[dict[str, object]] = []
    for reading in readings:
        scoped = _scope(table, regime, reading)
        has_nrmse = "nrmse_mean" in scoped.columns
        for dof in _reading_dofs(scoped, reading, regime):
            cell = scoped[scoped["dof"] == dof]
            reference, reference_skill, pool = _resolve_reference(cell, reading, dof)
            for model in deep_models:
                row = _one_row(cell, model, reading, dof)
                skill = float(row["skill_mean"])
                skill_std = float(row["skill_std"])
                n_seeds = int(row["n_seeds"])
                margin = skill - reference_skill
                verdict, reason = _margin_verdict(margin, skill_std, n_seeds, model)
                rows.append(
                    {
                        "reading": reading.key,
                        "reading_title": reading.title,
                        "regime": regime,
                        "dof": dof,
                        "horizon_samples": int(reading.horizon_samples),
                        "horizon_s": float(row["horizon_s"]),
                        "model": model,
                        "n_seeds": n_seeds,
                        "skill_mean": skill,
                        "skill_std": skill_std,
                        "nrmse_mean": float(row["nrmse_mean"]) if has_nrmse else float("nan"),
                        "reference": reference,
                        "reference_skill_mean": reference_skill,
                        "reference_pool": pool,
                        "margin": margin,
                        "verdict": verdict,
                        "verdict_reason": reason,
                        **_paired_columns(
                            contrasts,
                            regime=regime,
                            model=model,
                            reference=reference,
                            dof=dof,
                            horizon_samples=reading.horizon_samples,
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=list(GATE4_COLUMNS))


def reading_passes(readout: pd.DataFrame, key: str) -> bool:
    """Whether every row of one reading is a :data:`VERDICT_PASS`.

    Args:
        readout: The frame from :func:`gate4_readout`.
        key: Reading key, e.g. ``"B"``.

    Returns:
        ``True`` only if the reading has at least one row and every one of them passed.
        :data:`VERDICT_UNVERIFIED` is not a pass.

    Raises:
        ValueError: If ``key`` names no reading in ``readout`` -- an absent reading would
            otherwise vacuously "pass".
    """
    subset = readout[readout["reading"] == key]
    if subset.empty:
        raise ValueError(
            f"reading {key!r} is absent from the read-out; the readings present are "
            f"{sorted(readout['reading'].unique())}"
        )
    return bool((subset["verdict"] == VERDICT_PASS).all())


def gate4_notes(readout: pd.DataFrame) -> tuple[str, ...]:
    """Caveats that must travel with the verdict table.

    Args:
        readout: The frame from :func:`gate4_readout`, used to decide which conditional
            notes apply.

    Returns:
        Markdown bullet bodies, without the leading ``- ``.
    """
    notes = [
        "**The criterion is strict.** `verdict` is PASS iff `margin > skill_std`; "
        "`margin == skill_std` is FAIL, because the gate says the margin must *exceed* "
        "the seed-to-seed standard deviation.",
        "**`skill_std` covers initialisation and data-order variance (P4-D13).** "
        "`_fit_one` builds the loaders once per model config, but the shuffle generator's "
        "state advances every epoch and is never reset, so each seed trains on a different "
        "batch order and the three-seed spread is the full one the criterion intends. Note "
        "what this costs in exchange: seed k's batch order depends on how many epochs seeds "
        "0..k-1 ran, so the seeds are consecutive segments of one stream rather than "
        "independent draws, and a change to the epoch cap changes the data order of every "
        "seed after the first.",
        "**The spread is not the same quantity in every row (P4-D6).** Every SGD row's "
        "`skill_std` covers initialisation and data-order variance; `lstm` and "
        "`transformer` additionally carry cuDNN/Flash-Attention backward nondeterminism "
        "that `tcn` and `dlinear` do not. Since the criterion compares each model's margin "
        "against its own spread, the noisier model faces the harder bar.",
        "**`nrmse_mean` is beside `skill_mean` because skill is not comparable across "
        "horizons on this signal (P3-D5).** Persistence error tracks the autocorrelation "
        "and is non-monotone in lead time, so a skill-vs-horizon reading shows dips that "
        "belong to the reference. `nrmse = rmse / signal_std` of the scored partition at "
        "that exact lead time; 1.0 is 'no better than the partition mean' and lower is "
        "better.",
        "**Simulated results only.** Every artifact read here comes from the JONSWAP-driven "
        "vessel simulation; no real deck data is involved.",
    ]
    if bool((readout["verdict"] == VERDICT_UNVERIFIED).any()):
        notes.insert(
            0,
            "**UNVERIFIED is not a pass and not a fail.** A deep model is stochastic and "
            "must carry a real seed spread. Where `skill_std` is NaN or the row carries "
            f"fewer than {MIN_SEEDS} seeds, the criterion has no quantity to compare the "
            "margin against, and filling that NaN with 0.0 would make every positive "
            "margin 'exceed' it. Those rows are marked UNVERIFIED and the gate is not read "
            "as passed.",
        )
    if bool((readout["paired_verdict"] != PAIRED_ABSENT).any()):
        notes[3:3] = [
            "**`margin` and `paired_skill_diff` answer different questions and are not "
            "collapsed into one verdict.** The margin test asks whether the advantage "
            "exceeds the spread of the fitting procedure across seeds; the paired "
            "bootstrap asks whether it survives resampling held-out realizations with both "
            "models scored on the same resample. A margin can exceed the seed std while the "
            "paired interval still spans zero, and the reverse.",
            "**`paired_ci_lo`/`paired_ci_hi` are the envelope over seeds, not a mean of "
            "intervals.** The contrasts file emits one interval per seed and no aggregated "
            "view (P4-D4); the envelope contains every seed's interval, so `excludes 0` "
            "here means every seed excluded zero. It is not a calibrated interval for the "
            "seed mean.",
        ]
    if bool((readout["paired_verdict"] == PAIRED_ABSENT).any()):
        notes.append(
            "**Some rows carry no paired contrast.** The contrasts file was absent, "
            "unusable, or held no row for that (model, reference, DOF, horizon). Those "
            "rows are the margin test alone; the corroboration is missing, not negative.",
        )
    if bool(readout["nrmse_mean"].isna().all()):
        notes.append(
            "**`nrmse_mean` is empty for every row.** The table read has no `nrmse_mean` "
            "column -- every artifact written before P4-D3 predates it -- so the "
            "horizon-comparability check P3-D5 mandates cannot be made from this document.",
        )
    return tuple(notes)


def _gate_provenance_line(results_dir: Path | None, *, with_contrasts: bool) -> str:
    """Render the sentence naming the files this document was built from.

    Deliberately not :func:`dmf.eval.report._provenance_line`: that names all five
    baselines artifacts, and this document reads two of them. "Every number here traces
    to" is a claim about the files named, so naming a file no number here comes from is
    the same defect as naming the wrong one (P3-D21).

    Args:
        results_dir: Directory the artifacts were read from, or ``None`` when the caller
            did not say, in which case the files are located "beside this document" --
            true in any directory, where a guessed ``results/`` prefix would be false in
            all but one.
        with_contrasts: Whether ``paired_contrasts.csv`` was actually joined.

    Returns:
        One Markdown sentence.
    """
    names = ["baselines.csv"]
    if with_contrasts:
        names.append("paired_contrasts.csv")
    prefix = "" if results_dir is None else _display_dir(results_dir)
    where = "the CSVs beside this document" if results_dir is None else "the CSVs it was read from"
    rendered = [f"`{prefix}{name}`" for name in names]
    joined = rendered[0] if len(rendered) == 1 else " and ".join(rendered)
    return f"Every number here traces to {where}: {joined}."


def _headline(readout: pd.DataFrame, reading: GateReading) -> str:
    """Render one reading's outcome as a sentence that names its failures.

    Args:
        readout: The frame from :func:`gate4_readout`.
        reading: The reading to summarise.

    Returns:
        A Markdown line. Non-passing rows are named in it, so a reader who reads only the
        headline still sees which model failed -- the table body carries them too.
    """
    subset = readout[readout["reading"] == reading.key]
    counts = subset["verdict"].value_counts()
    n_pass = int(counts.get(VERDICT_PASS, 0))
    total = len(subset)
    outcome = "PASS" if n_pass == total else "NOT PASSED"
    parts = [f"**Reading {reading.key} -- {outcome}**: {n_pass} of {total} rows pass."]
    for verdict in (VERDICT_FAIL, VERDICT_UNVERIFIED):
        named = subset[subset["verdict"] == verdict]
        if not named.empty:
            listed = ", ".join(
                f"`{row.model}`/{row.dof}" for row in named.itertuples(index=False, name="Row")
            )
            parts.append(f"{verdict}: {listed}.")
    return " ".join(parts)


def build_gate4_markdown(
    readout: pd.DataFrame,
    *,
    results_dir: Path | None = None,
    readings: Sequence[GateReading] = GATE4_READINGS,
) -> str:
    """Render the Gate 4 read-out as Markdown.

    The document leads with each reading's outcome and names every non-passing row in that
    first sentence as well as in the table body, so an underperforming model cannot be
    missed by a reader who stops after the headline (CLAUDE.md non-negotiable 6).

    Args:
        readout: The frame from :func:`gate4_readout`.
        results_dir: Directory the CSVs were read from, for the provenance line. **Pass the
            same directory the artifacts came from**; left ``None`` they are located
            "beside this document", which is true wherever they are.
        readings: The readings to render, in order. Readings absent from ``readout`` are
            skipped, so a caller that computed one reading can render one.

    Returns:
        The rendered Markdown document.

    Raises:
        ValueError: If ``readout`` is empty. An empty gate document reads as "nothing
            failed".
    """
    if readout.empty:
        raise ValueError("refusing to render an empty Gate 4 read-out")
    with_contrasts = bool((readout["paired_verdict"] != PAIRED_ABSENT).any())
    parts: list[str] = [
        "# Gate 4 read-out",
        "",
        "Simulated results only. Gate 4: **all deep models beat the reference by a margin "
        "exceeding the seed-to-seed standard deviation**. Two readings of the cell, both "
        "computed here; the gate is read at Reading B (`docs/protocol.md` P4-D1) and "
        "Reading A is reported whether or not it passes.",
        "",
        _gate_provenance_line(results_dir, with_contrasts=with_contrasts),
        "",
    ]
    present = [reading for reading in readings if reading.key in set(readout["reading"])]
    parts += ["## Outcome", ""]
    parts += [f"- {_headline(readout, reading)}" for reading in present]
    parts.append("")
    for reading in present:
        subset = readout[readout["reading"] == reading.key].reset_index(drop=True)
        horizon_s = float(subset["horizon_s"].iloc[0])
        parts += [
            f"## Reading {reading.key} -- {reading.title}",
            "",
            f"Regime `{subset['regime'].iloc[0]}`, horizon {horizon_s:g} s "
            f"({int(subset['horizon_samples'].iloc[0])} samples). "
            f"`margin = skill_mean - reference_skill_mean`; PASS iff `margin > skill_std`.",
            "",
            _headline(readout, reading),
            "",
            to_markdown(subset[list(_RENDERED_COLUMNS)]),
            "",
        ]
        if len(reading.references) > 1:
            resolution = subset[["dof", "reference", "reference_pool"]].drop_duplicates(
                ignore_index=True
            )
            parts += [
                f"Reference resolved per cell as the strongest of "
                f"{', '.join(f'`{name}`' for name in reading.references)}; both candidates' "
                f"skill in the cell is shown so the resolution can be checked, since which "
                f"one wins varies across the grid (P3-D20).",
                "",
                to_markdown(resolution),
                "",
            ]
        flagged = subset[subset["verdict"] != VERDICT_PASS]
        if not flagged.empty:
            parts += ["Why each non-passing row did not pass:", ""]
            parts += [
                f"- `{row.model}` / {row.dof}: **{row.verdict}** -- {row.verdict_reason}"
                for row in flagged.itertuples(index=False, name="Row")
            ]
            parts.append("")
    parts += ["## Notes", ""]
    parts += [f"- {note}" for note in gate4_notes(readout)]
    parts.append("")
    return "\n".join(parts)


def read_gate4_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read the committed artifacts the Gate 4 read-out is computed from.

    Args:
        results_dir: Directory holding ``baselines.csv`` and, optionally,
            ``paired_contrasts.csv``.

    Returns:
        ``(baselines, contrasts)``. ``contrasts`` is ``None`` when the file is absent --
        a baselines-only run writes none, and the read-out degrades to the margin test.

    Raises:
        FileNotFoundError: If ``baselines.csv`` is absent. Without it there is no gate.
    """
    baselines_path = results_dir / "baselines.csv"
    if not baselines_path.exists():
        raise FileNotFoundError(
            f"{baselines_path} does not exist; the Gate 4 read-out is computed from the "
            f"committed baselines table, not from a re-run"
        )
    baselines = pd.read_csv(baselines_path)
    contrasts_path = results_dir / "paired_contrasts.csv"
    contrasts = pd.read_csv(contrasts_path) if contrasts_path.exists() else None
    return baselines, contrasts


def write_gate4_report(
    results_dir: Path, out_dir: Path | None = None
) -> tuple[pd.DataFrame, Path, Path]:
    """Read the artifacts, compute the read-out, and write both the CSV and the document.

    Args:
        results_dir: Directory holding ``baselines.csv`` and optionally
            ``paired_contrasts.csv``.
        out_dir: Where to write ``gate4.csv`` and ``gate4.md``. Defaults to
            ``results_dir``.

    Returns:
        ``(readout, csv_path, markdown_path)``.

    Raises:
        FileNotFoundError: If ``baselines.csv`` is absent.
        ValueError: Anything :func:`gate4_readout` raises -- a missing gate cell is fatal.
    """
    destination = results_dir if out_dir is None else out_dir
    baselines, contrasts = read_gate4_inputs(results_dir)
    readout = gate4_readout(baselines, contrasts=contrasts)
    csv_path = write_table(readout, destination / "gate4.csv")
    markdown_path = destination / "gate4.md"
    markdown_path.write_text(
        build_gate4_markdown(readout, results_dir=results_dir), encoding="utf-8"
    )
    return readout, csv_path, markdown_path


# ======================================================================================
# Gate 5 -- probabilistic calibration.
#
# Same shape as the Gate 4 read-out above and for the same reason: the number a reader sees
# and the number the gate was read at are one object, computed by repository code from the
# committed CSVs rather than by a script that cannot be re-run.
#
# The criterion is `docs/IMPLEMENTATION_PLAN.md` Phase 5 verbatim -- PICP@90 within
# [0.85, 0.95] on `id` -- and the band is unchanged. What `docs/protocol.md` P5-D2 added,
# BEFORE the sweep ran, is the cell it is read at, because the plan does not say and
# coverage varies strongly with lead time. Choosing that afterwards would be cell selection.
# ======================================================================================

#: Regime the gate is read on. The out-of-distribution regimes are reported beside it and
#: are explicitly **not** what it turns on: P5-D2 and the task both require the degradation
#: to be reported rather than fixed, so a regime failing here is a finding, not a gate.
GATE5_REGIME = "id"

#: The cell Reading A is read at -- the same cell Gates 3 and 4 are read at (P3-D12, P4-D1),
#: so all three phases turn on one place: the decision horizon, on the DOF that binds.
GATE5_DOF = "pitch"
GATE5_HORIZON_SAMPLES = 100

#: The band, verbatim from ``docs/IMPLEMENTATION_PLAN.md`` Phase 5. Inclusive at both ends:
#: the plan says "within [0.85, 0.95]", and a coverage landing exactly on 0.95 is within it.
GATE5_PICP_BAND: tuple[float, float] = (0.85, 0.95)

#: Nominal coverage the band applies to. Carried as a named constant so the document can
#: state what "PICP@90" means rather than leaving the reader to infer it from ``alpha``.
GATE5_NOMINAL = 0.90

#: Schema of ``gate5.csv``.
GATE5_COLUMNS: tuple[str, ...] = (
    "reading",
    "model",
    "head",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "n_seeds",
    "picp_mean",
    "picp_std",
    "picp_ci_lo",
    "picp_ci_hi",
    "band_lo",
    "band_hi",
    "verdict",
    "mean_interval_width_mean",
    "width_ratio_mean",
    "crossing_rate_mean",
    "n_params",
)

#: Schema of ``gate5_degradation.csv``.
DEGRADATION_COLUMNS: tuple[str, ...] = (
    "model",
    "head",
    "regime",
    "dof",
    "horizon_samples",
    "horizon_s",
    "picp_id",
    "picp_ood",
    "picp_delta",
    "width_id",
    "width_ood",
    "width_delta",
    "width_ratio_id",
    "width_ratio_ood",
)


def _picp_verdict(row: "pd.Series[Any]") -> str:
    """Classify one coverage row against the band.

    A row that cannot be evaluated is :data:`VERDICT_UNVERIFIED`, never a pass. Following
    the Gate 4 precedent exactly: ``nan >= 0.85`` is ``False`` in IEEE arithmetic, so a
    naive comparison would silently *fail* an unmeasured row, and filling the NaN would
    silently pass it. Both are wrong, and the distinction is the reader's to see.

    Args:
        row: One aggregated coverage row.

    Returns:
        One of :data:`VERDICT_PASS`, :data:`VERDICT_FAIL`, :data:`VERDICT_UNVERIFIED`.
    """
    picp = float(row["picp_mean"])
    seeds = row.get("n_seeds", np.nan)
    if not np.isfinite(picp):
        return VERDICT_UNVERIFIED
    if not np.isfinite(float(seeds)) or int(seeds) < MIN_SEEDS:
        return VERDICT_UNVERIFIED
    low, high = GATE5_PICP_BAND
    return VERDICT_PASS if low <= picp <= high else VERDICT_FAIL


def gate5_readout(
    table: pd.DataFrame,
    *,
    regime: str = GATE5_REGIME,
    dof: str = GATE5_DOF,
    horizon_samples: int = GATE5_HORIZON_SAMPLES,
) -> pd.DataFrame:
    """Compute both Gate 5 readings from the aggregated probabilistic table.

    - **Reading A**, the gate: one row per (model, head) at :data:`GATE5_DOF` /
      :data:`GATE5_HORIZON_SAMPLES`.
    - **Reading B**, the surround: every cell of the regime, so the pass count is visible
      beside the gate cell rather than discovered later. Gates 3 and 4 both needed a "what
      the gate does not say" section written after the fact; here it is part of the read-out.

    Args:
        table: ``probabilistic.csv``, i.e. :data:`dmf.eval.report.PROBABILISTIC_COLUMNS`.
        regime: Regime to read. Defaults to ``id``, which is what the gate names.
        dof: DOF Reading A is read at. Defaults to the registered cell; parameterised only
            so a fixture-scale table can be exercised, never so a caller can choose a
            kinder cell after seeing the numbers.
        horizon_samples: Horizon Reading A is read at, samples. Same caveat as ``dof``.

    Returns:
        One frame with :data:`GATE5_COLUMNS`, Reading A rows first.

    Raises:
        ValueError: If a required column is missing, or if the gate cell is absent -- a
            missing gate cell is fatal, never an empty pass.
    """
    missing = sorted(
        {
            "model",
            "head",
            "regime",
            "dof",
            "horizon_samples",
            "horizon_s",
            "n_seeds",
            "picp_mean",
            "picp_std",
            "picp_ci_lo",
            "picp_ci_hi",
            "mean_interval_width_mean",
            "width_ratio_mean",
            "crossing_rate_mean",
            "n_params",
        }
        - set(table.columns)
    )
    if missing:
        raise ValueError(f"the probabilistic table is missing columns {missing}")

    scoped = table[table["regime"] == regime]
    if scoped.empty:
        raise ValueError(
            f"the probabilistic table carries no rows for regime {regime!r}; Gate 5 is read "
            f"on {GATE5_REGIME!r} and cannot be read from a table that does not contain it"
        )
    dof_column = _resolve_gate_dof(scoped, dof)
    cell = scoped[(scoped["dof"] == dof_column) & (scoped["horizon_samples"] == horizon_samples)]
    if cell.empty:
        raise ValueError(
            f"the gate cell (regime={regime!r}, dof={dof_column!r}, "
            f"horizon_samples={horizon_samples}) is absent from the probabilistic "
            f"table. Gate 5 cannot be read, and an absent cell is not a pass"
        )

    frames = []
    for key, rows in ((READING_A.key, cell), (READING_B.key, scoped)):
        frame = rows.copy()
        frame.insert(0, "reading", key)
        frames.append(frame)
    readout = pd.concat(frames, ignore_index=True)
    readout["band_lo"] = GATE5_PICP_BAND[0]
    readout["band_hi"] = GATE5_PICP_BAND[1]
    readout["verdict"] = readout.apply(_picp_verdict, axis=1)
    ordered = readout.sort_values(
        ["reading", "model", "head", "dof", "horizon_samples"], kind="stable"
    ).reset_index(drop=True)
    return ordered[list(GATE5_COLUMNS)]


def gate5_reading_passes(readout: pd.DataFrame, key: str) -> bool:
    """Report whether every row of one reading passes.

    Args:
        readout: The frame :func:`gate5_readout` returned.
        key: ``"A"`` or ``"B"``.

    Returns:
        ``True`` only if the reading has rows and every one is :data:`VERDICT_PASS`. An
        :data:`VERDICT_UNVERIFIED` row is not a pass.

    Raises:
        ValueError: If ``key`` names no reading in ``readout``.
    """
    rows = readout[readout["reading"] == key]
    if rows.empty:
        raise ValueError(
            f"reading {key!r} is not in the read-out; it carries {sorted(set(readout['reading']))}"
        )
    return bool((rows["verdict"] == VERDICT_PASS).all())


def coverage_degradation(table: pd.DataFrame, *, base_regime: str = GATE5_REGIME) -> pd.DataFrame:
    """Tabulate how coverage and sharpness move from ``id`` to each held-out regime.

    **This is the finding, not a defect to fix.** ``docs/IMPLEMENTATION_PLAN.md`` Phase 5
    and P5-D2 both require the degradation to be reported rather than corrected: it is the
    coverage-under-domain-shift story that motivates split conformal prediction, measured on
    a concrete operational task.

    **The comparison is UNPAIRED, and that has to be stated wherever it is read.** ``id``
    and ``unseen_seastate`` score different realizations -- different seeds in the first
    case, an entirely held-out sea state in the second -- so there is no common resample
    over which a paired interval could be drawn. Each side carries its own realization
    bootstrap and the difference carries none. ``docs/protocol.md`` P3-D13 records this
    project publishing a wrong conclusion twice from exactly this mistake, which is why the
    delta columns ship without an interval rather than with one that would be wrong.

    ``width_delta`` is reported beside ``picp_delta`` and neither is interpretable alone. An
    out-of-distribution interval can hold its coverage purely by getting wider, which is not
    calibration surviving the shift; and P5-D6 predicts, before the sweep, that
    ``unseen_heading`` will show near-perfect coverage at meaningless width on pitch for the
    P1-D2 residual-floor reason.

    Args:
        table: ``probabilistic.csv``.
        base_regime: The reference regime. Defaults to ``id``.

    Returns:
        One row per (model, head, out-of-distribution regime, DOF, horizon), columns
        :data:`DEGRADATION_COLUMNS`. Empty if the table carries only ``base_regime``.

    Raises:
        ValueError: If ``base_regime`` is absent from the table.
    """
    keys = ["model", "head", "dof", "horizon_samples"]
    base = table[table["regime"] == base_regime]
    if base.empty:
        raise ValueError(
            f"the probabilistic table carries no {base_regime!r} rows, so there is nothing "
            f"to measure degradation against"
        )
    other = table[table["regime"] != base_regime]
    if other.empty:
        return pd.DataFrame(columns=list(DEGRADATION_COLUMNS))
    columns = [*keys, "horizon_s", "picp_mean", "mean_interval_width_mean", "width_ratio_mean"]
    merged = other[[*columns, "regime"]].merge(
        base[columns], on=keys, suffixes=("_ood", "_id"), validate="many_to_one"
    )
    merged["picp_id"] = merged["picp_mean_id"]
    merged["picp_ood"] = merged["picp_mean_ood"]
    merged["picp_delta"] = merged["picp_ood"] - merged["picp_id"]
    merged["width_id"] = merged["mean_interval_width_mean_id"]
    merged["width_ood"] = merged["mean_interval_width_mean_ood"]
    merged["width_delta"] = merged["width_ood"] - merged["width_id"]
    merged["width_ratio_id"] = merged["width_ratio_mean_id"]
    merged["width_ratio_ood"] = merged["width_ratio_mean_ood"]
    merged["horizon_s"] = merged["horizon_s_ood"]
    ordered = merged.sort_values(
        ["regime", "model", "head", "dof", "horizon_samples"], kind="stable"
    ).reset_index(drop=True)
    return ordered[list(DEGRADATION_COLUMNS)]


#: Columns rendered into the Gate 5 tables. The CSV keeps everything; the document keeps
#: what a reader making the decision needs, with coverage and width **always adjacent** --
#: PICP alone is trivially gameable by widening, and the protocol forbids reporting it that
#: way (P5-D2, and the `forecast-protocol` skill).
_GATE5_RENDERED: tuple[str, ...] = (
    "model",
    "head",
    "dof",
    "horizon_s",
    "n_seeds",
    "picp_mean",
    "picp_std",
    "picp_ci_lo",
    "picp_ci_hi",
    "mean_interval_width_mean",
    "width_ratio_mean",
    "verdict",
)


def _gate5_headline(readout: pd.DataFrame, key: str) -> str:
    """Summarise one reading in a sentence that names every row that did not pass."""
    rows = readout[readout["reading"] == key]
    total = len(rows)
    passed = int((rows["verdict"] == VERDICT_PASS).sum())
    verdict = "PASS" if passed == total and total else "NOT PASSED"
    sentence = f"**Reading {key} -- {verdict}**: {passed} of {total} rows inside the band."
    failures = rows[rows["verdict"] != VERDICT_PASS]
    if failures.empty:
        return sentence
    named = ", ".join(
        f"`{r.model}`/{r.head} {r.dof}@{r.horizon_s:g}s {r.picp_mean:.3f} ({r.verdict})"
        for r in list(failures.itertuples())[:8]
    )
    more = "" if len(failures) <= 8 else f", and {len(failures) - 8} more"
    return f"{sentence} Outside: {named}{more}."


def gate5_notes(readout: pd.DataFrame, degradation: pd.DataFrame | None = None) -> tuple[str, ...]:
    """Return the caveats that must travel with any Gate 5 number.

    Args:
        readout: The frame from :func:`gate5_readout`.
        degradation: The frame from :func:`coverage_degradation`, if it was computed.

    Returns:
        Caveat lines, rendered as a bullet list by :func:`build_gate5_markdown`.
    """
    notes = [
        "Simulated results only. The generator has no process noise, so the achievable "
        "sharpness is unrealistically high and every interval here is narrower than one "
        "fitted to real deck motion would be (`docs/protocol.md` P4-D16).",
        "**Coverage is never a result on its own.** A wide enough interval covers "
        "everything. `width_ratio` is the reading that makes the width interpretable: it is "
        "the interval width over that of an unconditional interval matched to the scored "
        "partition's own spread, so **1.0 means no sharper than knowing only the variance**, "
        "the same device `nrmse` provides for RMSE (P4-D3).",
        "**The heads are not parameter-matched** to each other or to their Phase 4 point "
        "rows: the final projection widens with the head, so a quantile head carries ~9x the "
        "head parameters of a point one (P5-D5). A head-vs-head difference is not an "
        "architecture result.",
        "**`best_val_loss` is not comparable across heads** -- each model is early-stopped on "
        "its own objective, named in `val_loss_name` (P5-D4).",
    ]
    unverified = readout[readout["verdict"] == VERDICT_UNVERIFIED]
    if not unverified.empty:
        named = ", ".join(
            f"`{r.model}`/{r.head} {r.dof}@{r.horizon_s:g}s"
            for r in list(unverified.itertuples())[:6]
        )
        notes.append(
            f"**{len(unverified)} row(s) are UNVERIFIED, which is not a pass**: {named}. A "
            f"row with fewer than {MIN_SEEDS} seeds or a non-finite coverage is an unmeasured "
            f"gate, not a passed one."
        )
    if degradation is not None and not degradation.empty:
        regimes = ", ".join(f"`{name}`" for name in sorted(set(degradation["regime"])))
        notes.append(
            f"The degradation table covers {regimes} and is **reported, not fixed** -- it is "
            f"the finding this phase exists to produce, and the same coverage-under-shift "
            f"story that motivates conformal prediction. The deltas are **unpaired**: the "
            f"two regimes score different realizations, so no common resample exists and no "
            f"interval on a delta would be honest (P3-D13)."
        )
        notes.append(
            "On `unseen_heading`, coverage on pitch and pitch_rate is **predicted in advance** "
            "to be near 1.0 at meaningless width: that regime's test set is beam seas, where "
            "the pitch heading factor sits on the P1-D2 residual floor ~26 dB down, so a head "
            "fitted where pitch has amplitude emits intervals scaled to a signal the test set "
            "does not contain (P5-D6). Near-perfect coverage there is not calibration."
        )
    return tuple(notes)


def build_gate5_markdown(
    readout: pd.DataFrame,
    *,
    degradation: pd.DataFrame | None = None,
    results_dir: Path | None = None,
) -> str:
    """Render the Gate 5 read-out as Markdown.

    Args:
        readout: The frame from :func:`gate5_readout`.
        degradation: The frame from :func:`coverage_degradation`, rendered as its own
            section when given.
        results_dir: Directory the CSVs were read from, for the provenance line.

    Returns:
        The rendered Markdown document.

    Raises:
        ValueError: If ``readout`` is empty -- an empty gate document reads as "nothing
            failed".
    """
    if readout.empty:
        raise ValueError("refusing to render an empty Gate 5 read-out")
    where = "beside this document" if results_dir is None else f"`{_display_dir(results_dir)}`"
    low, high = GATE5_PICP_BAND
    parts: list[str] = [
        "# Gate 5 read-out",
        "",
        f"Simulated results only. Gate 5: **PICP@{GATE5_NOMINAL:.0%} within "
        f"[{low}, {high}] on the `{GATE5_REGIME}` regime** "
        "(`docs/IMPLEMENTATION_PLAN.md` Phase 5, band unchanged). The cell it is read at "
        "was registered before the sweep ran (`docs/protocol.md` P5-D2) and is the cell "
        "Gates 3 and 4 are read at: the decision horizon, on the DOF that binds.",
        "",
        f"Computed from `probabilistic.csv` in {where}; the point accuracy of the same runs "
        f"is in `baselines.csv` beside it, and the per-run source of truth is "
        f"`probabilistic_by_seed.csv`.",
        "",
        "## Outcome",
        "",
    ]
    present = [key for key in ("A", "B") if key in set(readout["reading"])]
    parts += [f"- {_gate5_headline(readout, key)}" for key in present]
    parts.append("")

    titles = {
        "A": f"the gate -- {GATE5_DOF} at {GATE5_HORIZON_SAMPLES} samples",
        "B": f"the surround -- every cell of `{GATE5_REGIME}`",
    }
    for key in present:
        subset = readout[readout["reading"] == key].reset_index(drop=True)
        parts += [
            f"## Reading {key} -- {titles[key]}",
            "",
            f"Regime `{subset['regime'].iloc[0]}`. PASS iff "
            f"`{low} <= picp_mean <= {high}`, inclusive at both ends.",
            "",
            _gate5_headline(readout, key),
            "",
            to_markdown(subset[list(_GATE5_RENDERED)]),
            "",
        ]

    if degradation is not None and not degradation.empty:
        parts += [
            "## Coverage under distribution shift -- reported, not fixed",
            "",
            "Each row is one cell's move from `id` to a held-out regime. Read `picp_delta` "
            "and `width_delta` **together**: an interval that keeps its coverage by growing "
            "has not kept its calibration.",
            "",
            to_markdown(degradation),
            "",
        ]

    notes = gate5_notes(readout, degradation)
    parts += ["## Notes", "", *[f"- {note}" for note in notes], ""]
    return "\n".join(parts)


def read_gate5_inputs(results_dir: Path) -> pd.DataFrame:
    """Read ``probabilistic.csv`` from a results directory.

    Args:
        results_dir: Directory holding the artifacts.

    Returns:
        The aggregated probabilistic table.

    Raises:
        FileNotFoundError: If ``probabilistic.csv`` is absent.
    """
    path = results_dir / "probabilistic.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; Gate 5 is read from the committed artifacts of a Phase 5 "
            f"sweep, never from a live model"
        )
    return pd.read_csv(path)


def write_gate5_report(
    results_dir: Path,
    out_dir: Path | None = None,
    *,
    dof: str = GATE5_DOF,
    horizon_samples: int = GATE5_HORIZON_SAMPLES,
) -> tuple[pd.DataFrame, Path, Path]:
    """Read the artifacts, compute the read-out and the degradation table, and write both.

    Args:
        results_dir: Directory holding ``probabilistic.csv``.
        out_dir: Where to write ``gate5.csv``, ``gate5_degradation.csv`` and ``gate5.md``.
            Defaults to ``results_dir``.
        dof: DOF Reading A is read at. See :func:`gate5_readout`.
        horizon_samples: Horizon Reading A is read at, samples. See :func:`gate5_readout`.

    Returns:
        ``(readout, csv_path, markdown_path)``.

    Raises:
        FileNotFoundError: If ``probabilistic.csv`` is absent.
        ValueError: Anything :func:`gate5_readout` raises -- a missing gate cell is fatal.
    """
    destination = results_dir if out_dir is None else out_dir
    table = read_gate5_inputs(results_dir)
    readout = gate5_readout(table, dof=dof, horizon_samples=horizon_samples)
    degradation = coverage_degradation(table)
    csv_path = write_table(readout, destination / "gate5.csv")
    if not degradation.empty:
        write_table(degradation, destination / "gate5_degradation.csv")
    markdown_path = destination / "gate5.md"
    markdown_path.write_text(
        build_gate5_markdown(readout, degradation=degradation, results_dir=results_dir),
        encoding="utf-8",
    )
    return readout, csv_path, markdown_path


# ======================================================================================
# Gate 6 -- traceability and reproducibility.
#
# Same shape as the Gate 4 and Gate 5 read-outs above and for the same reason: the verdict
# a reader sees and the verdict the gate was read at are one object, computed by repository
# code rather than by a script that cannot be re-run.
#
# What differs is the kind of criterion. Gates 3-5 are numeric -- a skill margin, a coverage
# band -- and each turns on a cell chosen before the sweep ran. Gate 6 is a **process**
# criterion, so `docs/protocol.md` P6-D1 registered no threshold and no cell, and instead
# wrote it out as seven artifact predicates. This section implements exactly those seven,
# in P6-D1's order and numbering, and adds none: a criterion invented after the artifacts
# exist is the thing pre-registration is for.
#
# **The seven predicates need a machine-readable document, and this is where the contract
# is stated.** Predicates 3 and 4 -- every rendered table names an existing CSV, and the
# row count it states matches that CSV's -- are the operative ones, because they make "every
# number traceable to a CSV" a structural property of the renderer rather than a claim
# about it. A prose sentence naming a file is not checkable; a marker is. `results.md` must
# therefore precede every rendered table with
#
#     <!-- dmf-table id=<slug> section=<6.1|6.2|6.3|...> source=<path/to.csv>
#          csv_rows=<int> rows=<int> [select="<filter>"] -->
#
# as the last non-blank line before the table's header row. See :data:`GATE6_MARKER_SPEC`
# for the field semantics and :func:`parse_rendered_tables` for the parser.
#
# **Predicate 4 is not literally checkable, and that is recorded rather than worked around.**
# Most rendered tables are a filtered view of a larger CSV -- one regime, one horizon band --
# so "the row count stated equals the row count of the named CSV" is false for them by
# construction, and a gate that enforced it literally would fail every honest document. It
# is therefore read as three sub-checks: the stated row count matches the rows actually
# rendered (the document is self-consistent), the stated CSV row count matches the file (the
# provenance is real), and for a table declaring no `select` the two coincide (P6-D1's
# literal reading, where it applies). All three must hold; which tables are filtered is
# reported.
# ======================================================================================

#: The document Gate 6 is read on.
GATE6_RESULTS_MD = "results.md"

#: The one reading. Gates 4 and 5 ship two readings each because their criterion could
#: honestly be read at more than one cell; Gate 6 has no cell, so there is one key and it
#: covers all seven predicates.
GATE6_READING = "gate"

#: Human-readable statement of the marker contract, rendered into ``gate6.md`` so that the
#: document the renderer's author reads and the string the parser accepts are one object.
GATE6_MARKER_SPEC = (
    "<!-- dmf-table id=<slug> section=<6.1|6.2|6.3> source=<path/to.csv> "
    'csv_rows=<int> rows=<int> [select="<filter>"] -->'
)

#: Marker keys that must be present on every table marker.
GATE6_MARKER_REQUIRED: tuple[str, ...] = ("id", "source", "csv_rows", "rows")

#: The tables ``docs/IMPLEMENTATION_PLAN.md`` sections 6.1, 6.2 and 6.3 require, as
#: ``(section, id)``. Predicate 5 is a set containment against this tuple, so this constant
#: **is** the required-table list and a renderer's ``id`` slugs must match it exactly.
#:
#: Taken from the plan and from nothing else. The interval quiescence rule (P6-D5), the
#: interval controls (Phase 6 carry-forward item 6) and the reproducibility control (P6-D13)
#: are all Phase 6 obligations and none of them is in this tuple, because P6-D1 was written
#: before them and adding a predicate to a pre-registered gate after the fact is the failure
#: pre-registration exists to prevent. :func:`gate6_notes` names them instead, so the reader
#: sees what the gate does not cover.
GATE6_REQUIRED_TABLES: tuple[tuple[str, str], ...] = (
    ("6.1", "core_metrics"),
    ("6.2", "quiescence_detection"),
    ("6.2", "quiescence_lead_time"),
    ("6.2", "quiescence_base_rate"),
    ("6.3", "ablation_observation_mode"),
    ("6.3", "ablation_channels"),
    ("6.3", "ablation_ss_conditioning"),
    ("6.3", "ablation_lookback"),
    ("6.3", "ablation_normalization"),
)

#: The seven predicates, verbatim in substance from ``docs/protocol.md`` P6-D1 and in its
#: numbering. Stored as data so that the rendered document, the CSV and the pass rule all
#: read one list; a criterion that appeared in the prose and not in the frame would be a
#: criterion nobody evaluated.
GATE6_CRITERIA: tuple[tuple[str, str], ...] = (
    (
        "1",
        "`make eval` exits 0 on a checkout holding `artifacts/corpus/` and "
        "`artifacts/checkpoints/`.",
    ),
    ("2", "`results/results.md` exists and is regenerated by that command, not hand-edited."),
    (
        "3",
        "Every table rendered in `results.md` names the CSV it was read from, and that file "
        "exists.",
    ),
    (
        "4",
        "The row count `results.md` states for each table equals the row count of the named CSV.",
    ),
    ("5", "Every 6.1, 6.2 and 6.3 table required by the plan is present."),
    ("6", "Every F1 row carries its base rate in the same row."),
    (
        "7",
        "Every coverage row carries an interval width in the same row, and no coverage row "
        "pools the 1-5 s and 10-15 s bands.",
    ),
)

#: Schema of ``gate6.csv``: one row per predicate.
GATE6_COLUMNS: tuple[str, ...] = (
    "reading",
    "criterion",
    "statement",
    "verdict",
    "n_checked",
    "n_failed",
    "detail",
)

#: Schema of ``gate6_tables.csv``: one row per table rendered in ``results.md``. This is the
#: evidence predicates 3, 4, 6 and 7 are computed from, written out so that a failure can be
#: read at the table that caused it rather than at the summary.
GATE6_TABLE_COLUMNS: tuple[str, ...] = (
    "table_id",
    "section",
    "source",
    "source_exists",
    "csv_rows_stated",
    "csv_rows_actual",
    "rows_stated",
    "rows_rendered",
    "filtered",
    "has_f1",
    "has_base_rate",
    "has_picp",
    "has_width",
    "has_horizon",
    "line",
    "problems",
)

#: The marker itself.
_MARKER_RE = re.compile(r"^<!--\s*dmf-table\s+(?P<attrs>.*?)\s*-->\s*$")

#: ``key=value`` pairs inside a marker; values may be bare or double-quoted.
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?:"(?P<quoted>[^"]*)"|(?P<bare>\S+))')

#: Column-name patterns predicates 6 and 7 turn on. Matched against the rendered header
#: cells, not against a CSV, because the predicates are about what a **row of the document**
#: states -- a base rate that exists in a CSV but not beside the F1 a reader is looking at is
#: exactly the trap `CLAUDE.md` names.
_F1_COLUMN_RE = re.compile(r"(?:^|_)f1(?:_|$)", re.IGNORECASE)
_BASE_RATE_COLUMN_RE = re.compile(r"base[_ ]?rate", re.IGNORECASE)
_PICP_COLUMN_RE = re.compile(r"picp|coverage", re.IGNORECASE)
_WIDTH_COLUMN_RE = re.compile(r"width", re.IGNORECASE)
_HORIZON_COLUMN_RE = re.compile(r"^horizon(_s|_samples)?$", re.IGNORECASE)
_POOLED_COLUMN_RE = re.compile(r"pooled|all_horizons|horizon_band", re.IGNORECASE)


@dataclass(frozen=True)
class RenderedTable:
    """One Markdown table found in ``results.md``, with whatever marker preceded it.

    Attributes:
        table_id: The marker's ``id``, or empty for a table with no marker -- an **orphan**,
            which fails predicate 3 by definition: a table nobody can trace is the case the
            predicate exists to catch.
        section: The marker's ``section``, or the section
            :data:`GATE6_REQUIRED_TABLES` assigns to this ``id``, or empty.
        source: The marker's ``source``, a path relative to the results directory.
        csv_rows_stated: Row count the marker claims the source CSV has, or None.
        rows_stated: Row count the marker claims this table renders, or None.
        select: The marker's ``select`` filter description; empty means the table is
            claimed to be the whole CSV, which is the only case P6-D1's predicate 4 can be
            read literally on.
        columns: Header cells of the rendered table, in order.
        rows_rendered: Body rows actually present under the header.
        line: 1-based line number of the table's header row, for a message that points at
            the document.
        marker_error: Why the marker was unusable, empty if it was fine.
    """

    table_id: str
    section: str
    source: str
    csv_rows_stated: int | None
    rows_stated: int | None
    select: str
    columns: tuple[str, ...]
    rows_rendered: int
    line: int
    marker_error: str = ""

    @property
    def filtered(self) -> bool:
        """Whether the marker declares this table a filtered view of its CSV.

        Returns:
            True if a ``select`` was declared.
        """
        return bool(self.select)


@dataclass(frozen=True)
class Gate6Evidence:
    """Everything the seven predicates are evaluated from, gathered by one IO pass.

    Separated from :func:`gate6_readout` so that the read-out is a pure function of
    evidence, exactly as :func:`gate4_readout` is a pure function of ``baselines.csv``. A
    gate that reads the filesystem inside its own verdict logic cannot be exercised on a
    fixture, and one that cannot be exercised on a fixture is one nobody has watched fail.

    Attributes:
        results_dir: The directory ``results.md`` and its CSVs were read from.
        results_md_exists: Whether ``results.md`` is there at all.
        tables: Every Markdown table found in it, in document order.
        csv_row_counts: Row count of each CSV a marker named, keyed by the marker's
            ``source`` string. Absent keys are files that do not exist.
        eval_exit_code: Exit status of ``make eval``, or None if this invocation was not
            told. **None is UNVERIFIED, never a pass**: predicate 1 is about a command
            having been run, and no artifact can testify to that on its own.
        corpus_present: Whether ``artifacts/corpus/`` exists -- predicate 1's precondition.
        checkpoints_present: Whether ``artifacts/checkpoints/`` exists -- likewise.
        rerender_matches: Whether re-running the renderer over the same CSVs reproduces
            ``results.md`` byte for byte, or None if that could not be attempted.
        rerender_error: Why it could not be attempted, or how it differed.
    """

    results_dir: Path
    results_md_exists: bool
    tables: tuple[RenderedTable, ...]
    csv_row_counts: Mapping[str, int]
    eval_exit_code: int | None = None
    corpus_present: bool = False
    checkpoints_present: bool = False
    rerender_matches: bool | None = None
    rerender_error: str = ""


def parse_rendered_tables(text: str) -> tuple[RenderedTable, ...]:
    """Find every Markdown table in a document and the marker that should precede it.

    A table is a run of consecutive lines beginning with ``|`` whose second line is a
    separator row. Its marker is the **last non-blank line before it**, which is a strict
    rule on purpose: a marker several paragraphs up could be read as belonging to either of
    two tables, and predicate 3 is worth nothing if the association is ambiguous.

    Args:
        text: The contents of ``results.md``.

    Returns:
        One :class:`RenderedTable` per table found, in document order. A table with no
        marker is returned with an empty ``table_id`` and a ``marker_error`` saying so,
        rather than skipped -- an untraceable table must appear in the audit, not vanish
        from it.
    """
    lines = text.splitlines()
    sections = {table_id: section for section, table_id in GATE6_REQUIRED_TABLES}
    tables: list[RenderedTable] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = index
        block: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index].strip())
            index += 1
        if len(block) < 2 or set(block[1].replace("|", "").replace(" ", "")) - set("-:") != set():
            continue  # not a table: a stray pipe line, or a separator that is not one
        columns = tuple(cell.strip() for cell in block[0].strip("|").split("|"))
        rows_rendered = len(block) - 2
        marker_line = start - 1
        while marker_line >= 0 and not lines[marker_line].strip():
            marker_line -= 1
        match = _MARKER_RE.match(lines[marker_line].strip()) if marker_line >= 0 else None
        if match is None:
            tables.append(
                RenderedTable(
                    table_id="",
                    section="",
                    source="",
                    csv_rows_stated=None,
                    rows_stated=None,
                    select="",
                    columns=columns,
                    rows_rendered=rows_rendered,
                    line=start + 1,
                    marker_error=(
                        f"no `{GATE6_MARKER_SPEC.split()[1]}` marker on the last non-blank "
                        f"line before this table"
                    ),
                )
            )
            continue
        attrs = {
            found.group("key"): (
                found.group("quoted") if found.group("quoted") is not None else found.group("bare")
            )
            for found in _ATTR_RE.finditer(match.group("attrs"))
        }
        missing = [key for key in GATE6_MARKER_REQUIRED if key not in attrs]
        table_id = attrs.get("id", "")
        tables.append(
            RenderedTable(
                table_id=table_id,
                section=attrs.get("section", sections.get(table_id, "")),
                source=attrs.get("source", ""),
                csv_rows_stated=_as_int(attrs.get("csv_rows")),
                rows_stated=_as_int(attrs.get("rows")),
                select=attrs.get("select", ""),
                columns=columns,
                rows_rendered=rows_rendered,
                line=start + 1,
                marker_error=(
                    "" if not missing else f"marker is missing required key(s) {missing}"
                ),
            )
        )
    return tuple(tables)


def _as_int(value: str | None) -> int | None:
    """Parse a marker value as an integer, returning None rather than raising.

    Args:
        value: The raw marker value, or None if the key was absent.

    Returns:
        The integer, or None if it was absent or unparseable. A malformed count is reported
        by predicate 4 as a failure, which is more useful than a traceback out of the parser.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _table_problems(table: RenderedTable, counts: Mapping[str, int]) -> dict[str, list[str]]:
    """Evaluate one table against the predicates that are about tables.

    Args:
        table: The rendered table.
        counts: Row count of each CSV a marker named, keyed by the ``source`` string.

    Returns:
        Predicate id -> the problems that predicate found with this table. An empty list
        means the table satisfies it; a table is only counted against a predicate it is
        actually subject to (predicate 6 says nothing about a table with no F1 column).
    """
    problems: dict[str, list[str]] = {key: [] for key, _ in GATE6_CRITERIA}
    if table.marker_error:
        problems["3"].append(table.marker_error)
    elif not table.source:
        problems["3"].append("marker declares no `source`")
    elif table.source not in counts:
        problems["3"].append(f"named CSV `{table.source}` does not exist")

    if table.rows_stated is None or table.csv_rows_stated is None:
        problems["4"].append("marker does not state both `rows` and `csv_rows` as integers")
    else:
        if table.rows_stated != table.rows_rendered:
            problems["4"].append(
                f"marker states rows={table.rows_stated} but {table.rows_rendered} body "
                f"row(s) are rendered"
            )
        actual = counts.get(table.source)
        if actual is None:
            problems["4"].append(f"cannot count rows of `{table.source}`")
        elif table.csv_rows_stated != actual:
            problems["4"].append(
                f"marker states csv_rows={table.csv_rows_stated} but `{table.source}` "
                f"holds {actual}"
            )
        elif not table.filtered and table.rows_stated != actual:
            problems["4"].append(
                f"marker declares no `select`, so P6-D1 predicate 4 is read literally here: "
                f"rows={table.rows_stated} must equal the CSV's {actual}"
            )

    columns = table.columns
    has_f1 = any(_F1_COLUMN_RE.search(name) for name in columns)
    has_base_rate = any(_BASE_RATE_COLUMN_RE.search(name) for name in columns)
    if has_f1 and not has_base_rate:
        problems["6"].append(
            "carries an F1 column and no base-rate column; F1 against an unstated base rate "
            "is not interpretable (CLAUDE.md known traps)"
        )
    has_picp = any(_PICP_COLUMN_RE.search(name) for name in columns)
    has_width = any(_WIDTH_COLUMN_RE.search(name) for name in columns)
    has_horizon = any(_HORIZON_COLUMN_RE.match(name) for name in columns)
    if has_picp:
        if not has_width:
            problems["7"].append(
                "carries a coverage column and no width column; a maximally wide interval "
                "has perfect coverage, so coverage alone states nothing"
            )
        if not has_horizon:
            problems["7"].append(
                "carries a coverage column and no per-row `horizon_s`/`horizon_samples` "
                "column, so its rows pool lead times. P5-D15 measured the deep heads "
                "calibrated 12 of 12 at 10-15 s and over-covering in 18-23 of 24 cells at "
                "1-5 s: one pooled number averages two opposite behaviours"
            )
        pooled = [name for name in columns if _POOLED_COLUMN_RE.search(name)]
        if pooled:
            problems["7"].append(f"coverage table carries pooled column(s) {pooled}")
    return problems


def gate6_table_audit(evidence: Gate6Evidence) -> pd.DataFrame:
    """Tabulate every rendered table and what the predicates found wrong with it.

    Written to ``gate6_tables.csv``. This is where a Gate 6 failure is actually read: the
    read-out says which predicate failed and how many tables tripped it, and this frame says
    which ones.

    Args:
        evidence: The gathered evidence.

    Returns:
        One row per table, columns :data:`GATE6_TABLE_COLUMNS`. Empty if the document
        rendered no tables at all -- which the read-out reports as a failure of predicate 3,
        not as an empty pass.
    """
    rows: list[dict[str, object]] = []
    for table in evidence.tables:
        problems = _table_problems(table, evidence.csv_row_counts)
        flat = [f"[{key}] {text}" for key, items in problems.items() for text in items]
        rows.append(
            {
                "table_id": table.table_id,
                "section": table.section,
                "source": table.source,
                "source_exists": table.source in evidence.csv_row_counts,
                "csv_rows_stated": table.csv_rows_stated,
                "csv_rows_actual": evidence.csv_row_counts.get(table.source),
                "rows_stated": table.rows_stated,
                "rows_rendered": table.rows_rendered,
                "filtered": table.filtered,
                "has_f1": any(_F1_COLUMN_RE.search(name) for name in table.columns),
                "has_base_rate": any(_BASE_RATE_COLUMN_RE.search(name) for name in table.columns),
                "has_picp": any(_PICP_COLUMN_RE.search(name) for name in table.columns),
                "has_width": any(_WIDTH_COLUMN_RE.search(name) for name in table.columns),
                "has_horizon": any(_HORIZON_COLUMN_RE.match(name) for name in table.columns),
                "line": table.line,
                "problems": "; ".join(flat),
            }
        )
    return pd.DataFrame(rows, columns=list(GATE6_TABLE_COLUMNS))


def _criterion_1(evidence: Gate6Evidence) -> tuple[str, int, int, str]:
    """Evaluate P6-D1 predicate 1: ``make eval`` exits 0 on a checkout with the artifacts."""
    missing = [
        name
        for name, present in (
            ("artifacts/corpus/", evidence.corpus_present),
            ("artifacts/checkpoints/", evidence.checkpoints_present),
        )
        if not present
    ]
    if missing:
        return (
            VERDICT_UNVERIFIED,
            1,
            0,
            f"the predicate's precondition is not met on this checkout: {missing} absent. "
            f"Not a failure of `make eval` and not a pass",
        )
    if evidence.eval_exit_code is None:
        return (
            VERDICT_UNVERIFIED,
            1,
            0,
            "no `make eval` exit status was supplied. No artifact testifies that a command "
            "was run, so this is an unmeasured predicate, not a passed one; run the gate "
            "after `make eval` in the same make invocation (`make gate6-full`) or pass "
            "--eval-exit-code",
        )
    if evidence.eval_exit_code == 0:
        return VERDICT_PASS, 1, 0, "`make eval` exited 0"
    return VERDICT_FAIL, 1, 1, f"`make eval` exited {evidence.eval_exit_code}"


def _criterion_2(evidence: Gate6Evidence) -> tuple[str, int, int, str]:
    """Evaluate P6-D1 predicate 2: ``results.md`` exists and is generated, not hand-edited."""
    if not evidence.results_md_exists:
        return VERDICT_FAIL, 1, 1, f"{GATE6_RESULTS_MD} is not in {evidence.results_dir}"
    if evidence.rerender_matches is None:
        return (
            VERDICT_UNVERIFIED,
            1,
            0,
            f"the document exists but could not be re-rendered, so "
            f"'not hand-edited' is unmeasured: {evidence.rerender_error}",
        )
    if evidence.rerender_matches:
        return (
            VERDICT_PASS,
            1,
            0,
            "re-running the renderer over the committed CSVs reproduces the document byte "
            "for byte, so every line in it came from a CSV",
        )
    return (
        VERDICT_FAIL,
        1,
        1,
        f"re-rendering does not reproduce the committed document: {evidence.rerender_error}",
    )


def _criterion_5(evidence: Gate6Evidence) -> tuple[str, int, int, str]:
    """Evaluate P6-D1 predicate 5: every table the plan requires is present."""
    present = {table.table_id for table in evidence.tables}
    missing = [
        f"{section} `{table_id}`"
        for section, table_id in GATE6_REQUIRED_TABLES
        if table_id not in present
    ]
    total = len(GATE6_REQUIRED_TABLES)
    if missing:
        return (
            VERDICT_FAIL,
            total,
            len(missing),
            f"{len(missing)} of {total} required table(s) absent: {', '.join(missing)}",
        )
    return VERDICT_PASS, total, 0, f"all {total} required tables are present"


def gate6_readout(evidence: Gate6Evidence) -> pd.DataFrame:
    """Compute the Gate 6 verdict, one row per pre-registered predicate.

    Pure: everything it reads is in ``evidence``, so the whole gate can be exercised on a
    synthetic document. :func:`read_gate6_inputs` is the only part that touches disk.

    Args:
        evidence: The gathered evidence, from :func:`read_gate6_inputs` or built directly.

    Returns:
        Seven rows in P6-D1's order, columns :data:`GATE6_COLUMNS`. A predicate that could
        not be evaluated is :data:`VERDICT_UNVERIFIED`, which is not a pass -- the Gate 4
        and Gate 5 rule, for the same reason: an unmeasured gate is neither passed nor
        failed, and filling the gap either way is a decision the reader should make.
    """
    per_table = [_table_problems(table, evidence.csv_row_counts) for table in evidence.tables]
    n_tables = len(evidence.tables)

    def table_predicate(key: str, subject: str) -> tuple[str, int, int, str]:
        failures = [
            (table, problems[key])
            for table, problems in zip(evidence.tables, per_table, strict=True)
            if problems[key]
        ]
        if key == "3" and n_tables == 0:
            return (
                VERDICT_FAIL,
                0,
                0,
                f"{GATE6_RESULTS_MD} renders no tables at all; an empty document is not a "
                f"traceable one",
            )
        if not failures:
            return VERDICT_PASS, n_tables, 0, f"{n_tables} table(s) checked, {subject}"
        named = "; ".join(
            f"`{table.table_id or '(unmarked)'}` at line {table.line}: {items[0]}"
            for table, items in failures[:6]
        )
        more = "" if len(failures) <= 6 else f", and {len(failures) - 6} more"
        return VERDICT_FAIL, n_tables, len(failures), f"{named}{more}"

    outcomes: dict[str, tuple[str, int, int, str]] = {
        "1": _criterion_1(evidence),
        "2": _criterion_2(evidence),
        "3": table_predicate("3", "each names a CSV that exists"),
        "4": table_predicate("4", "each states a row count matching its source"),
        "5": _criterion_5(evidence),
        "6": table_predicate("6", "no F1 column stands without its base rate"),
        "7": table_predicate("7", "no coverage column stands without a width and a lead time"),
    }
    rows = []
    for key, statement in GATE6_CRITERIA:
        verdict, checked, failed, detail = outcomes[key]
        rows.append(
            {
                "reading": GATE6_READING,
                "criterion": key,
                "statement": statement,
                "verdict": verdict,
                "n_checked": checked,
                "n_failed": failed,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows, columns=list(GATE6_COLUMNS))


def gate6_reading_passes(readout: pd.DataFrame, key: str = GATE6_READING) -> bool:
    """Report whether every predicate of one reading passes.

    Args:
        readout: The frame :func:`gate6_readout` returned.
        key: :data:`GATE6_READING` for the gate as a whole, or a single criterion id
            (``"1"`` .. ``"7"``) to ask about one predicate. Naming one predicate reports
            it; it does not narrow the gate, which :func:`write_gate6_report` and
            ``scripts/gate6.py`` always take over all seven.

    Returns:
        True only if the selected rows exist and every one is :data:`VERDICT_PASS`. An
        :data:`VERDICT_UNVERIFIED` row is not a pass.

    Raises:
        ValueError: If ``key`` names neither the reading nor a criterion in ``readout``.
    """
    rows = readout if key == GATE6_READING else readout[readout["criterion"] == key]
    if rows.empty:
        raise ValueError(
            f"{key!r} names neither the reading {GATE6_READING!r} nor a criterion in the "
            f"read-out, which carries {sorted(set(readout['criterion']))}"
        )
    return bool((rows["verdict"] == VERDICT_PASS).all())


def gate6_notes(readout: pd.DataFrame, audit: pd.DataFrame | None = None) -> tuple[str, ...]:
    """Return the caveats that must travel with any Gate 6 verdict.

    Args:
        readout: The frame from :func:`gate6_readout`.
        audit: The frame from :func:`gate6_table_audit`, if it was computed.

    Returns:
        Caveat lines, rendered as a bullet list by :func:`build_gate6_markdown`.
    """
    notes = [
        "Simulated results only. Nothing this gate checks is evidence about real deck "
        "motion; it is evidence that the numbers in `results.md` came from the CSVs beside "
        "it.",
        "**This is a reproducibility and traceability gate and nothing else** "
        "(`docs/protocol.md` P6-D1). It does not test that any Phase 6 number is correct, "
        "that the quiescence detector is well specified, or that an ablation contrast is "
        "fair. Those are P6-D2 through P6-D6 and the integrity controls, and a PASS here "
        "must not be read as covering them.",
        "**Predicate 4 is read as three sub-checks, not literally.** Most rendered tables "
        "are a filtered view of a larger CSV, so 'the stated row count equals the CSV's' is "
        "false for them by construction. The gate requires instead that the stated row "
        "count match the rows rendered, that the stated CSV row count match the file, and "
        "that the two coincide wherever a table declares no `select`. The `filtered` column "
        "of `gate6_tables.csv` says which tables the literal reading applied to.",
        "**The required-table list is the plan's and nothing more.** The interval quiescence "
        "rule (P6-D5), the two interval controls (Phase 6 carry-forward item 6) and the "
        "`results/e02/` reproducibility control (P6-D13) are all Phase 6 obligations that "
        "this gate does **not** require a table for, because P6-D1 was registered before "
        "they existed and adding predicates to a pre-registered gate after the artifacts "
        "exist defeats the point of registering it. Their absence would be invisible here.",
    ]
    unverified = readout[readout["verdict"] == VERDICT_UNVERIFIED]
    if not unverified.empty:
        named = ", ".join(f"criterion {row.criterion}" for row in unverified.itertuples())
        notes.append(
            f"**{len(unverified)} predicate(s) are UNVERIFIED, which is not a pass**: "
            f"{named}. An unmeasured predicate is an unmeasured gate."
        )
    if audit is not None and not audit.empty:
        orphans = audit[audit["table_id"] == ""]
        if not orphans.empty:
            lines = ", ".join(str(line) for line in orphans["line"].tolist()[:8])
            notes.append(
                f"**{len(orphans)} table(s) carry no provenance marker** (document line(s) "
                f"{lines}). A table nobody can trace to a CSV is exactly what predicate 3 "
                f"exists to catch, and it is counted as a failure rather than skipped."
            )
        filtered = int(audit["filtered"].sum())
        if filtered:
            notes.append(
                f"{filtered} of {len(audit)} rendered table(s) declare a `select` filter, so "
                f"P6-D1's literal predicate 4 does not apply to them and the two weaker "
                f"sub-checks carried the verdict there."
            )
    return tuple(notes)


def build_gate6_markdown(
    readout: pd.DataFrame,
    *,
    audit: pd.DataFrame | None = None,
    results_dir: Path | None = None,
) -> str:
    """Render the Gate 6 read-out as Markdown.

    Args:
        readout: The frame from :func:`gate6_readout`.
        audit: The frame from :func:`gate6_table_audit`, rendered as its own section when
            given.
        results_dir: Directory the document and CSVs were read from, for the provenance
            line.

    Returns:
        The rendered Markdown document.

    Raises:
        ValueError: If ``readout`` is empty -- an empty gate document reads as "nothing
            failed".
    """
    if readout.empty:
        raise ValueError("refusing to render an empty Gate 6 read-out")
    where = "beside this document" if results_dir is None else f"`{_display_dir(results_dir)}`"
    passed = int((readout["verdict"] == VERDICT_PASS).sum())
    total = len(readout)
    headline = "PASS" if passed == total else "NOT PASSED"
    parts: list[str] = [
        "# Gate 6 read-out",
        "",
        "Simulated results only. Gate 6: **`results/results.md` regenerated end-to-end by "
        "`make eval`, containing every table above, every number traceable to a CSV in "
        "`results/`** (`docs/IMPLEMENTATION_PLAN.md` Phase 6). Unlike Gates 3-5 it is a "
        "process criterion with no threshold and no cell, so `docs/protocol.md` P6-D1 "
        "registered it as seven artifact predicates before this machinery existed. Those "
        "seven, in that numbering, are what is evaluated below.",
        "",
        f"Read over {GATE6_RESULTS_MD} and the CSVs in {where}.",
        "",
        "## Outcome",
        "",
        f"**{headline}**: {passed} of {total} pre-registered predicates satisfied.",
        "",
        to_markdown(readout[["criterion", "statement", "verdict", "n_failed", "detail"]]),
        "",
        "## The provenance contract",
        "",
        "Predicates 3 and 4 are checkable only if the document says, per table, where its "
        "numbers came from. The renderer must emit, as the last non-blank line before every "
        "Markdown table:",
        "",
        "```",
        GATE6_MARKER_SPEC,
        "```",
        "",
        "`source` is a path relative to the results directory; `csv_rows` is that file's row "
        "count; `rows` is the number of body rows the table renders; `select` is a "
        "human-readable description of the filter applied, and its **absence is a claim** "
        "that the table is the whole CSV, which is the case P6-D1's predicate 4 is read "
        "literally on.",
        "",
    ]
    if audit is not None and not audit.empty:
        parts += [
            "## Per-table audit",
            "",
            "The evidence predicates 3, 4, 6 and 7 are computed from. A failure is read "
            "here, at the table that caused it, rather than at the summary above.",
            "",
            to_markdown(
                audit[
                    [
                        "table_id",
                        "section",
                        "source",
                        "source_exists",
                        "csv_rows_stated",
                        "csv_rows_actual",
                        "rows_stated",
                        "rows_rendered",
                        "filtered",
                        "problems",
                    ]
                ]
            ),
            "",
        ]
    notes = gate6_notes(readout, audit)
    parts += ["## Notes", "", *[f"- {note}" for note in notes], ""]
    return "\n".join(parts)


def read_gate6_inputs(
    results_dir: Path,
    *,
    eval_exit_code: int | None = None,
    rerender: bool = True,
    repo_root: Path | None = None,
) -> Gate6Evidence:
    """Gather everything the seven predicates need, in one IO pass.

    The only function in this section that touches disk. It reads ``results.md``, parses its
    tables, counts the rows of every CSV a marker names, and -- unless asked not to --
    re-renders the document from those same CSVs into a temporary directory so that
    predicate 2 is a measurement rather than an assumption.

    **Re-rendering is cheap and does not touch the GPU**: it reads committed CSVs and writes
    Markdown. It also never writes into ``results_dir``.

    Args:
        results_dir: Directory holding ``results.md`` and the CSVs it cites.
        eval_exit_code: Exit status of ``make eval``, if the caller ran it. None leaves
            predicate 1 UNVERIFIED, which is not a pass.
        rerender: Whether to attempt the regeneration check for predicate 2.
        repo_root: Where ``artifacts/corpus/`` and ``artifacts/checkpoints/`` are looked
            for. Defaults to the current working directory.

    Returns:
        The evidence, ready for :func:`gate6_readout`.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist. A missing *document* is a
            predicate 2 failure and is reported as one; a missing *directory* is a caller
            error, because it means the gate was pointed somewhere else entirely.
    """
    if not results_dir.exists():
        raise FileNotFoundError(
            f"{results_dir} does not exist; Gate 6 is read over a directory of committed "
            f"artifacts, and pointing it at a missing one would report every predicate as "
            f"failed for the wrong reason"
        )
    document = results_dir / GATE6_RESULTS_MD
    text = document.read_text(encoding="utf-8") if document.exists() else ""
    tables = parse_rendered_tables(text)
    counts: dict[str, int] = {}
    for table in tables:
        if not table.source or table.source in counts:
            continue
        path = results_dir / table.source
        if path.exists():
            counts[table.source] = int(len(pd.read_csv(path)))
    root = Path.cwd() if repo_root is None else repo_root
    matches: bool | None = None
    error = ""
    if not document.exists():
        error = f"{GATE6_RESULTS_MD} is not there to compare against"
    elif not rerender:
        error = "the regeneration check was disabled by the caller (--no-rerender)"
    else:
        matches, error = _rerender_matches(results_dir, text)
    return Gate6Evidence(
        results_dir=results_dir,
        results_md_exists=document.exists(),
        tables=tables,
        csv_row_counts=counts,
        eval_exit_code=eval_exit_code,
        corpus_present=(root / "artifacts" / "corpus").exists(),
        checkpoints_present=(root / "artifacts" / "checkpoints").exists(),
        rerender_matches=matches,
        rerender_error=error,
    )


def _rerender_matches(results_dir: Path, committed: str) -> tuple[bool | None, str]:
    """Re-render ``results.md`` from the committed CSVs and compare it byte for byte.

    This is what makes "not hand-edited" checkable. It also imposes a requirement on the
    renderer that is worth stating: ``results.md`` must be a **deterministic function of the
    CSVs**. A timestamp, a hostname or a wall-clock figure in the document would make this
    check fail on a document nobody had touched.

    Args:
        results_dir: Directory of committed CSVs.
        committed: The committed document's text.

    Returns:
        ``(matches, detail)``. ``matches`` is None when the comparison could not be made at
        all, which the read-out reports as UNVERIFIED rather than as either verdict.
    """
    from dmf.eval.report import build_results_report

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / GATE6_RESULTS_MD
        try:
            build_results_report(results_dir, out)
        except NotImplementedError:
            return None, (
                "dmf.eval.report.build_results_report is not implemented yet, so the "
                "document cannot be regenerated from its CSVs"
            )
        except Exception as exc:  # noqa: BLE001 - a renderer crash is a finding, not a stack trace
            return False, f"the renderer raised {type(exc).__name__}: {exc}"
        rendered = out.read_text(encoding="utf-8")
    if rendered == committed:
        return True, ""
    committed_lines = committed.splitlines()
    rendered_lines = rendered.splitlines()
    for index, (left, right) in enumerate(zip(committed_lines, rendered_lines, strict=False), 1):
        if left != right:
            return False, (
                f"first difference at line {index}: committed {left!r}, regenerated {right!r}"
            )
    return False, (
        f"the documents agree on their first {min(len(committed_lines), len(rendered_lines))} "
        f"line(s) but differ in length: committed {len(committed_lines)}, regenerated "
        f"{len(rendered_lines)}"
    )


def write_gate6_report(
    results_dir: Path,
    out_dir: Path | None = None,
    *,
    eval_exit_code: int | None = None,
    rerender: bool = True,
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Gather the evidence, compute the read-out and the table audit, and write both.

    Args:
        results_dir: Directory holding ``results.md`` and its CSVs.
        out_dir: Where to write ``gate6.csv``, ``gate6_tables.csv`` and ``gate6.md``.
            Defaults to ``results_dir``.
        eval_exit_code: Exit status of ``make eval``, if the caller ran it.
        rerender: Whether to attempt the regeneration check for predicate 2.
        repo_root: Where the ``artifacts/`` preconditions are looked for.

    Returns:
        ``(readout, csv_path, markdown_path)``.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist.
    """
    destination = results_dir if out_dir is None else out_dir
    evidence = read_gate6_inputs(
        results_dir, eval_exit_code=eval_exit_code, rerender=rerender, repo_root=repo_root
    )
    readout = gate6_readout(evidence)
    audit = gate6_table_audit(evidence)
    csv_path = write_table(readout, destination / "gate6.csv")
    if not audit.empty:
        write_table(audit, destination / "gate6_tables.csv")
    markdown_path = destination / "gate6.md"
    markdown_path.write_text(
        build_gate6_markdown(readout, audit=audit, results_dir=results_dir), encoding="utf-8"
    )
    return readout, csv_path, markdown_path
