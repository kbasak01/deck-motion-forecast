"""Gate 10's predicates, including the one that must never acquire a threshold.

The gate is a *process* gate: Phase 10 predicts that calibration will fail to restore coverage
under sea-state shift, so a bar on the out-of-distribution result would be a bar set by
whoever already knew the answer. These tests protect that property as much as they protect the
arithmetic -- ``test_predicate_7_imposes_no_threshold_on_the_degradation`` reads the shipped
source and fails if a later edit adds one.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from dmf.eval import conformal_gate as gate10

REGIMES = ("id", "unseen_seastate", "unseen_heading", "unseen_vessel")


def _conformal_frame(picp: float = 0.90) -> pd.DataFrame:
    """Build a minimal calibrated table.

    Args:
        picp: Coverage to report at every cell.

    Returns:
        One row per (regime, model), at the Gate 5 cell.
    """
    rows = []
    for regime in REGIMES:
        for model in ("tcn_quantile_conformal", "lstm_gaussian_conformal"):
            rows.append(
                {
                    "model": model,
                    "head": "quantile",
                    "regime": regime,
                    "dof": "pitch",
                    "horizon_samples": 100,
                    "horizon_s": 10.0,
                    "n_seeds": 3,
                    "deterministic": False,
                    "n_windows": 434304,
                    "picp_mean": picp,
                }
            )
    return pd.DataFrame(rows)


def _heads_frame() -> pd.DataFrame:
    """Build a minimal uncalibrated table with the same schema and disjoint labels.

    Returns:
        The frame.
    """
    frame = _conformal_frame(0.86)
    frame["model"] = frame["model"].str.removesuffix("_conformal")
    return frame


def _calibration_frame(fitted_on: str | None = None, stats: str | None = None) -> pd.DataFrame:
    """Build a minimal calibration table.

    Args:
        fitted_on: Override the provenance label, to exercise the guard.
        stats: Override the normalisation provenance label.

    Returns:
        The frame.
    """
    rows = []
    for regime in REGIMES:
        rows.append(
            {
                "regime": regime,
                "model": "tcn_quantile",
                "seed": 0,
                "dof": "pitch",
                "horizon_samples": 100,
                "fitted_on": fitted_on or f"{regime}/val",
                "norm_stats_fitted_on": stats or f"{regime}/train",
                "n_windows_used": 24677,
            }
        )
    return pd.DataFrame(rows)


def test_predicate_2_rejects_a_label_collision_with_the_committed_heads() -> None:
    """A calibrated row that shadows an uncalibrated one destroys the comparison.

    The two tables are read as one section; a shared label would give a reader two rows for
    one cell with no way to tell which run each came from.
    """
    conformal = _conformal_frame()
    conformal["model"] = conformal["model"].str.removesuffix("_conformal")
    ok, note = gate10._predicate_2(conformal, _heads_frame())
    assert not ok
    assert "collision" in note


def test_predicate_2_rejects_a_different_window_population() -> None:
    """Two tables scored on different windows are not comparable, however they are labelled."""
    conformal = _conformal_frame()
    conformal.loc[0, "n_windows"] = 1
    ok, note = gate10._predicate_2(conformal, _heads_frame())
    assert not ok
    assert "window count" in note


def test_predicate_2_accepts_a_disjoint_table_on_the_same_schema() -> None:
    """The intended state: same columns, same windows, disjoint labels."""
    ok, _ = gate10._predicate_2(_conformal_frame(), _heads_frame())
    assert ok


@pytest.mark.parametrize(
    ("fitted_on", "stats", "needle"),
    [
        ("id/train", None, "not fitted on"),
        ("id/test", None, "not fitted on"),
        (None, "id/val", "not train-fitted"),
    ],
)
def test_predicate_3_rejects_calibration_from_the_wrong_partition(
    fitted_on: str | None, stats: str | None, needle: str
) -> None:
    """Calibrating on train is in-sample; on test it is the leak; statistics must be train.

    These are the same three failures ``dmf.eval.conformal.calibrate_models`` refuses at run
    time. The gate re-checks them **from the committed artifact**, because a run that was
    correct and an artifact that can prove it are different claims.
    """
    ok, note = gate10._predicate_3(_calibration_frame(fitted_on, stats))
    assert not ok
    assert needle in note


def test_predicate_3_accepts_validation_fitted_calibration() -> None:
    """The intended state."""
    ok, _ = gate10._predicate_3(_calibration_frame())
    assert ok


def test_predicate_4_fails_when_the_gate_cell_is_absent() -> None:
    """An absent cell is not a pass -- the rule Gate 5 states and this gate inherits."""
    ok, note = gate10._predicate_4(_conformal_frame().iloc[0:0])
    assert not ok
    assert "absent is not a pass" in note


def test_predicate_4_fails_when_in_distribution_coverage_misses_the_band() -> None:
    """``id`` is the one regime with a guarantee, so a miss there is an implementation bug."""
    ok, note = gate10._predicate_4(_conformal_frame(picp=0.70))
    assert not ok
    assert "outside" in note


def test_predicate_4_passes_at_nominal_coverage() -> None:
    """The correctness check the construction must satisfy."""
    ok, note = gate10._predicate_4(_conformal_frame(picp=0.901))
    assert ok
    assert "worst deviation" in note


def test_predicate_5_rejects_a_two_seed_row() -> None:
    """Non-negotiable 5, applied to the calibrated rows like any other model-vs-model claim."""
    frame = _conformal_frame()
    frame.loc[0, "n_seeds"] = 2
    ok, _ = gate10._predicate_5(frame)
    assert not ok


def test_predicate_7_imposes_no_threshold_on_the_degradation() -> None:
    """The non-predicate, protected as a predicate.

    P5-D2 and P10-D1 both require the out-of-distribution behaviour to be reported rather than
    fixed. A gate that failed on a large degradation would reward hiding it, so predicate 7
    checks only that the table **exists and covers every OOD regime**. This test reads the
    shipped source and fails if a comparison against a degradation magnitude ever appears in
    it -- the same style of source-level assertion ``tests/test_assemble.py`` uses to keep
    ``ENFORCED_BY_CONTROL`` honest.
    """
    source = inspect.getsource(gate10._predicate_7)
    tree = ast.parse(ast.unparse(ast.parse(source)))
    compared: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            rendered = ast.unparse(node)
            if "picp_delta" in rendered or "width_delta" in rendered:
                compared.append(rendered)
    assert not compared, (
        f"predicate 7 compares a degradation magnitude: {compared}. Gate 10 must not bar a "
        f"large degradation -- P10-D1 predicts one, and a gate that failed on it would "
        f"reward hiding the finding."
    )


def test_predicate_7_requires_the_degradation_table_to_exist(tmp_path: Path) -> None:
    """Reported rather than omitted: an absent table is the one way this predicate fails."""
    ok, note = gate10._predicate_7(tmp_path, _conformal_frame())
    assert not ok
    assert "absent" in note


def test_predicate_7_passes_with_every_ood_regime_present(tmp_path: Path) -> None:
    """A degradation of any size passes, which is the point."""
    frame = pd.DataFrame(
        [
            {"regime": regime, "picp_delta": -0.4, "width_delta": 1.0}
            for regime in gate10.OOD_REGIMES
        ]
    )
    frame.to_csv(tmp_path / "conformal_degradation.csv", index=False)
    ok, note = gate10._predicate_7(tmp_path, _conformal_frame())
    assert ok
    assert "no threshold applied by design" in note


def test_the_protected_tables_are_the_gate_5_record() -> None:
    """Predicate 1 protects exactly the files that carry the uncalibrated reading.

    If this arm could modify them it could rewrite a gate that was closed on different
    evidence, which is the failure the additive design exists to prevent.
    """
    assert "results/e03/probabilistic.csv" in gate10.PROTECTED
    assert "results/e03/gate5.csv" in gate10.PROTECTED
