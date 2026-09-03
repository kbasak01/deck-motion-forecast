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

from collections.abc import Sequence
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
    "GateReading",
    "build_gate4_markdown",
    "gate4_notes",
    "gate4_readout",
    "read_gate4_inputs",
    "reading_passes",
    "write_gate4_report",
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
