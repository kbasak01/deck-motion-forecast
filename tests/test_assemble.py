"""The producer for the four tables ``results.md`` reads and nothing used to write.

``dmf.eval.scoring`` writes five of the nine sources the renderer declares. Until
:mod:`dmf.eval.assemble` existed, ``ablations.csv``, ``controls.csv``,
``probabilistic_baseline.csv`` and ``reference_reproducibility.csv`` had **no producer at
all**, so ``make eval`` failed on the first missing file and section 6.3 could not be
rendered by any route. What is pinned here:

1. **Every source the renderer reads has a producer, optional ones included.** Asserted as a
   set relation between :data:`dmf.eval.report.REPORT_SOURCES` and the three drivers that
   write CSVs -- :data:`dmf.eval.scoring.SCORING_ARTIFACTS`,
   :data:`dmf.eval.control_runner.CONTROL_ARTIFACTS` and
   :data:`dmf.eval.assemble.ASSEMBLED_ARTIFACTS` -- so a source added to the renderer with
   nothing writing it fails here rather than in a 46 h sweep's last step. *Optional* is
   included deliberately: ``interval_controls.csv`` and ``pipeline_sanity.csv`` were declared,
   rendered-if-present and written by nobody, so the document could only ever render their
   "absent" branch (P6-D15), and an assertion restricted to required sources would not have
   caught it.
2. **The lookback contrast is refused, not approximated.** The committed arms scored
   1151/1131/1091 windows per realization -- their own full origin sets -- and differencing
   those compares forecasts of different absolute times (P6-D4 item 1). The assembler skips
   the whole ablation with that reason rather than joining what is on disk.
3. **``enforced`` is backfilled from the driver and never guessed.** A control name the
   enforcement map does not know raises, and the map itself is checked against
   ``dmf.train.experiment._run_controls`` by parsing it.
4. **The capacity flag is measured on the real committed rows**, not asserted from a model
   name, and it fires on more than the one arm P6-D8 anticipated.

Units: windows are counts, parameter fractions are dimensionless.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmf.eval import assemble, control_runner
from dmf.eval.ablations import contrast_table
from dmf.eval.assemble import (
    ASSEMBLED_ARTIFACTS,
    ENFORCED_BY_CONTROL,
    MATCHED_ORIGIN_ARMS,
    REPRODUCIBILITY_QUANTITIES,
    assemble_phase6_tables,
    build_controls_table,
    build_probabilistic_baseline,
    build_reference_reproducibility,
    read_arm_rows,
)
from dmf.eval.control_runner import CONTROL_ARTIFACTS
from dmf.eval.report import (
    PHASE6_SUBDIR,
    PROBABILISTIC_COLUMNS,
    REPORT_SOURCES,
    _control_summary,
    split_controls,
)
from dmf.eval.scoring import SCORING_ARTIFACTS
from dmf.train import experiment

#: Repository root, for the committed-results tests.
REPO = Path(__file__).resolve().parents[1]

#: The committed results root. Read only; these tests never write into it.
RESULTS = REPO / "results"

#: Horizons the synthetic tables carry, samples and seconds.
HORIZONS: tuple[tuple[int, float], ...] = ((10, 1.0), (50, 5.0))


def _by_seed(
    params: dict[str, int],
    *,
    regimes: tuple[str, ...] = ("id",),
    dofs: tuple[str, ...] = ("roll", "pitch"),
    skill: float = 0.60,
) -> pd.DataFrame:
    """Return a frame shaped like a committed ``baselines_by_seed.csv``.

    Args:
        params: Model label -> fitted parameter count.
        regimes: Regimes to emit.
        dofs: Target channels.
        skill: Skill value on every row, so a contrast's difference is exactly computable.

    Returns:
        One row per (model, regime, DOF, horizon), seed 0, deterministic.
    """
    return pd.DataFrame(
        [
            {
                "model": model,
                "regime": regime,
                "dof": dof,
                "horizon_samples": samples,
                "horizon_s": seconds,
                "n_windows": 1000,
                "rmse": 1.0,
                "mae": 0.8,
                "rmse_persistence": 2.0,
                "skill": skill,
                "signal_std": 3.0,
                "nrmse": 0.4,
                "seed": 0,
                "deterministic": True,
                "n_params": count,
                "fit_time_s": 1.0,
                "best_epoch": np.nan,
                "epochs_run": np.nan,
                "best_val_loss": np.nan,
                "val_loss_name": "mse",
            }
            for model, count in params.items()
            for regime in regimes
            for dof in dofs
            for samples, seconds in HORIZONS
        ]
    )


def _controls_frame(*, with_flags: bool) -> pd.DataFrame:
    """Return a frame shaped like a committed ``baselines_controls.csv``.

    Args:
        with_flags: Whether to include the ``asserted`` and ``enforced`` columns. False is
            the shape of every table written before Phase 6, which is what the backfill is
            for.

    Returns:
        Two controls x two cells.
    """
    rows = []
    for control, subject, null in (
        ("shuffle", "shuffled", "window_mean"),
        ("untrained", "untrained", "persistence"),
    ):
        for dof in ("roll", "pitch"):
            row: dict[str, object] = {
                "control": control,
                "regime": "id",
                "subject_model": subject,
                "null_model": null,
                "dof": dof,
                "horizon_samples": 10,
                "horizon_s": 1.0,
                "skill_subject": -3.0,
                "skill_null": -3.0,
                "excess": 0.001,
                "tol": 0.02,
                "passed": True,
            }
            if with_flags:
                row["asserted"] = True
                row["enforced"] = control == "shuffle"
            rows.append(row)
    return pd.DataFrame(rows)


def _tree(root: Path, *, with_lookback: bool = False) -> Path:
    """Write a minimal committed-results tree: a reference arm and one ablation arm.

    Args:
        root: Results root to create.
        with_lookback: Whether to also write a lookback arm's directory, so the refusal can
            be shown to be a rule rather than an absence.

    Returns:
        The root.
    """
    reference = {"persistence": 0, "ar20": 108_900, "dlinear_ols": 60_300}
    (root / "e02").mkdir(parents=True, exist_ok=True)
    _by_seed(reference).to_csv(root / "e02" / "baselines_by_seed.csv", index=False)
    _controls_frame(with_flags=False).to_csv(root / "e02" / "baselines_controls.csv", index=False)

    arm_dir = root / PHASE6_SUBDIR / "e04c_ss_conditioned_ar_id"
    arm_dir.mkdir(parents=True, exist_ok=True)
    conditioned = {"persistence": 0, "ar20": 180_900, "dlinear_ols": 60_300}
    _by_seed(conditioned, skill=0.65).to_csv(arm_dir / "baselines_by_seed.csv", index=False)
    _controls_frame(with_flags=True).to_csv(arm_dir / "baselines_controls.csv", index=False)

    if with_lookback:
        lookback = root / PHASE6_SUBDIR / "e04d_lookback_10s_ood"
        lookback.mkdir(parents=True, exist_ok=True)
        _by_seed({"persistence": 0, "dlinear_ols": 30_300}).to_csv(
            lookback / "baselines_by_seed.csv", index=False
        )
    return root


# ---------------------------------------------------------------------------
# Every required source has a producer
# ---------------------------------------------------------------------------


#: The three drivers that write the CSVs ``results.md`` is rendered from, and the file names
#: each claims. :mod:`dmf.eval.scoring` scores an experiment against the corpus,
#: :mod:`dmf.eval.control_runner` runs the controls that are not part of a scoring pass, and
#: :mod:`dmf.eval.assemble` joins committed tables. Kept as data so the "has a producer"
#: assertions below read one list rather than three greps.
_PRODUCERS: dict[str, dict[str, str]] = {
    "dmf.eval.scoring": SCORING_ARTIFACTS,
    "dmf.eval.control_runner": CONTROL_ARTIFACTS,
    "dmf.eval.assemble": ASSEMBLED_ARTIFACTS,
}

#: Sources that are deliberately not produced by any of them: committed audit trails, read
#: and never regenerated. ``results/e03/probabilistic.csv`` is the Gate 5 record.
_EXTERNAL_SOURCES: frozenset[str] = frozenset({"e03_heads"})


def _produced_files() -> set[str]:
    """Return every file name one of the three drivers writes."""
    return {name for artifacts in _PRODUCERS.values() for name in artifacts.values()}


def test_every_required_report_source_is_written_by_one_of_the_producers() -> None:
    produced = _produced_files()
    unproduced = [
        spec.key for spec in REPORT_SOURCES if spec.required and not set(spec.candidates) & produced
    ]
    # A required source with no producer is exactly the state Phase 6 was in: `make eval`
    # fails on the first missing file, which is correct behaviour and an unusable pipeline.
    assert not unproduced, unproduced


def test_every_rendered_source_has_a_producer_even_when_it_is_optional() -> None:
    """The failure this catches is a renderer expecting a table nothing writes.

    ``interval_controls.csv`` and ``pipeline_sanity.csv`` were exactly that: a section of the
    document, a `ReportSource` entry, tested control functions -- and no caller, so the
    document could only ever render the "absent" branch (P6-D15). Optional is not the same as
    unproduced: optional means *this* run may not have written it, not that no run can.
    """
    produced = _produced_files()
    orphans = [
        spec.key
        for spec in REPORT_SOURCES
        if spec.rendered
        and spec.key not in _EXTERNAL_SOURCES
        and not set(spec.candidates) & produced
    ]
    assert not orphans, orphans


def test_no_two_producers_write_the_same_file() -> None:
    names = [name for artifacts in _PRODUCERS.values() for name in artifacts.values()]
    # Two writers for one path is a race whose loser is silently discarded, and the reader
    # cannot tell which one produced the file it got.
    assert len(names) == len(set(names)), sorted(names)


def test_the_producer_writes_the_four_tables_it_can_build(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    artifacts = assemble_phase6_tables(root)
    assert set(artifacts.paths) == {"ablations", "controls"}
    for name, path in artifacts.paths.items():
        assert path.name == ASSEMBLED_ARTIFACTS[name]
        assert path.is_file()
    # The two it could not build say why, and are not written empty: an empty CSV in
    # results/ cannot be told from a table with nothing to report.
    assert not (root / PHASE6_SUBDIR / "probabilistic_baseline.csv").exists()
    assert any("probabilistic" in key for key in artifacts.skipped)
    assert any("reference_reproducibility" in key for key in artifacts.skipped)


# ---------------------------------------------------------------------------
# The lookback refusal
# ---------------------------------------------------------------------------


def test_the_lookback_contrast_is_refused_until_the_origins_are_matched(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results", with_lookback=True)
    artifacts = assemble_phase6_tables(root, write=False)
    # The arm's directory exists and holds a perfectly well formed table. It is still not
    # read: its rows were scored on its own full origin set, and differencing them against
    # the reference would compare forecasts of different absolute times (P6-D4 item 1).
    assert "lookback" not in set(artifacts.ablations["ablation"].astype(str))
    reasons = [value for key, value in artifacts.skipped.items() if "lookback" in key]
    assert reasons
    assert any("matched_origins" in reason for reason in reasons)
    assert any("1151" in reason and "1091" in reason for reason in reasons)


def test_the_refusal_names_every_lookback_arm_including_the_reference(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results", with_lookback=True)
    for arm in MATCHED_ORIGIN_ARMS:
        frame, reason = read_arm_rows(root, arm)
        # `lookback_20s` is the L=200 arm re-scored on the matched set -- NOT the committed
        # results/e02/ rows, which are the same run scored on its own full set (P6-D10).
        assert frame is None
        assert "origin" in reason


def _matched_tree(root: Path, *, skill_by_arm: dict[str, float] | None = None) -> Path:
    """Write a results tree whose three lookback arms ARE origin-matched.

    The counterpart of ``_tree(with_lookback=True)``: same directory layout, but the rows
    come from :func:`dmf.eval.matched.score_matched_lookback`'s schema -- one provenance
    record per arm, at the production geometry -- so the assembler can re-derive the
    intersection and accept them.

    Args:
        root: Results root to create.
        skill_by_arm: Skill per arm, so a contrast's difference is exactly computable.

    Returns:
        The root.
    """
    from dmf.data.windows import WindowSpec
    from dmf.eval.ablations import ARMS, matched_origin_provenance
    from dmf.eval.assemble import MATCHED_ORIGIN_SUBDIR

    skills = skill_by_arm or {"lookback_10s": 0.50, "lookback_20s": 0.60, "lookback_40s": 0.64}
    horizons = tuple(samples for samples, _ in HORIZONS)
    specs = {
        arm: WindowSpec(lookback=ARMS[arm].lookback, horizons=horizons, stride=5)
        for arm in MATCHED_ORIGIN_ARMS
    }
    for arm, spec in specs.items():
        frame = _by_seed({"persistence": 0, "dlinear_ols": 60_300}, skill=skills[arm])
        provenance = matched_origin_provenance(spec, list(specs.values()), 6000)
        # The window count follows the declared origin count, over 8 realizations, because
        # the assembler cross-checks exactly that -- and the intersection here is the one
        # THIS fixture's horizon set gives, not the production 1091.
        frame["n_windows"] = int(provenance["matched_n_origins"]) * 8
        frame["n_realizations"] = 8
        for column, value in provenance.items():
            frame[column] = value
        directory = root / PHASE6_SUBDIR / MATCHED_ORIGIN_SUBDIR / arm
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "baselines_by_seed.csv", index=False)
    return root


def test_the_lookback_ablation_is_assembled_once_the_arms_are_origin_matched(
    tmp_path: Path,
) -> None:
    """The Gate 6 predicate 5 blocker, closed.

    ``ablation_lookback`` had no producer at any level (P6-D15 finding 1). With matched
    tables on disk the assembler emits it -- and emits it *against the re-scored L=200 arm*,
    not against the committed ``results/e02/`` rows, which were scored on 1131 origins.
    """
    root = _matched_tree(_tree(tmp_path / "results"))
    artifacts = assemble_phase6_tables(root, write=False)
    rows = artifacts.ablations
    lookback = rows[rows["ablation"].astype(str) == "lookback"]
    assert not lookback.empty
    assert set(lookback["arm"]) == {"lookback_10s", "lookback_40s"}
    assert set(lookback["reference_arm"]) == {"lookback_20s"}
    assert set(lookback["lookback"]) == {100, 400}
    assert set(lookback["reference_lookback"]) == {200}
    # arm minus reference, exactly: -0.10 for the short arm, +0.04 for the long one.
    diffs = lookback.groupby("arm")["skill_diff"].first().round(6).to_dict()
    assert diffs == {"lookback_10s": -0.10, "lookback_40s": 0.04}
    # The only lookback entry left in `skipped` is the missing per-cell table, and it costs
    # the arm its interval rather than its rows: an ablation is not deleted to protest a
    # missing error bar, and the row itself says the interval is absent and why.
    gaps = {key: value for key, value in artifacts.skipped.items() if "lookback" in key}
    assert set(gaps) == {
        "ablations/cells/lookback_10s",
        "ablations/cells/lookback_20s",
        "ablations/cells/lookback_40s",
    }
    assert lookback["skill_diff_ci_lo"].isna().all()
    assert lookback["skill_diff_ci_hi"].isna().all()
    assert lookback["skill_diff_ci_reason"].astype(str).str.startswith("no paired ").all()


def test_the_lookback_arms_must_agree_with_each_other_on_the_origin_set(tmp_path: Path) -> None:
    """Each side reproducing its own claim is not enough: they must be the same claim."""
    from dmf.eval.assemble import MATCHED_ORIGIN_SUBDIR

    root = _matched_tree(_tree(tmp_path / "results"))
    path = root / PHASE6_SUBDIR / MATCHED_ORIGIN_SUBDIR / "lookback_20s" / "baselines_by_seed.csv"
    frame = pd.read_csv(path)
    # A reference scored on a different corpus: internally consistent, and matched to nothing.
    from dmf.data.windows import WindowSpec
    from dmf.eval.ablations import ARMS, matched_origin_provenance

    horizons = tuple(samples for samples, _ in HORIZONS)
    specs = [
        WindowSpec(lookback=ARMS[arm].lookback, horizons=horizons, stride=5)
        for arm in ("lookback_10s", "lookback_20s", "lookback_40s")
    ]
    elsewhere = matched_origin_provenance(specs[1], specs, 3000)
    for column, value in elsewhere.items():
        frame[column] = value
    frame["n_windows"] = int(elsewhere["matched_n_origins"]) * 8
    frame.to_csv(path, index=False)
    artifacts = assemble_phase6_tables(root, write=False)
    assert "lookback" not in set(artifacts.ablations["ablation"].astype(str))
    reasons = [value for key, value in artifacts.skipped.items() if "lookback" in key]
    assert any("different origin sets" in reason for reason in reasons)


def test_two_experiments_scoring_one_label_on_one_regime_stay_two_rows(tmp_path: Path) -> None:
    """The collision the running sweep will produce, handled before it lands.

    ``e04c_ss_conditioned`` (deep, ``id`` and ``unseen_seastate``) and
    ``e04c_ss_conditioned_ar_id`` (AR(20), ``id``) are two experiments of one arm, and both
    carry ``persistence`` on ``id`` because every config must -- it is the skill
    denominator. Concatenating them before the contrast gives two rows for one join key and
    ``validate="one_to_one"`` refuses the whole arm, taking the other six arms' tables down
    with it. Contrasted per experiment, both rows survive, distinguishable by
    ``experiment``, which is the rule the renderer already applies: two runs of one label
    are two rows, never one average.
    """
    root = _tree(tmp_path / "results")
    twin = root / PHASE6_SUBDIR / "e04c_ss_conditioned_ood"
    twin.mkdir(parents=True, exist_ok=True)
    source = root / PHASE6_SUBDIR / "e04c_ss_conditioned_ar_id" / "baselines_by_seed.csv"
    pd.read_csv(source).to_csv(twin / "baselines_by_seed.csv", index=False)

    artifacts = assemble_phase6_tables(root, write=False)
    rows = artifacts.ablations
    assert set(rows["experiment"].dropna()) >= {"e04c_ss_conditioned_ar_id", twin.name}
    cell = rows[
        (rows["model"] == "persistence")
        & (rows["regime"] == "id")
        & (rows["logical_dof"] == "roll")
        & (rows["horizon_samples"] == 10)
    ]
    assert len(cell) == 2
    assert set(cell["experiment"]) == {"e04c_ss_conditioned_ar_id", twin.name}


# ---------------------------------------------------------------------------
# The observation-mode boundary survives assembly
# ---------------------------------------------------------------------------


def test_the_assembled_imu_rows_carry_the_guard_that_makes_raw_rmse_refusable() -> None:
    arm_path = RESULTS / PHASE6_SUBDIR / "e04a_obs_mode_ood" / "baselines_by_seed.csv"
    reference_path = RESULTS / "e02" / "baselines_by_seed.csv"
    if not arm_path.is_file():
        pytest.skip("the imu arm has not been scored yet")
    arm = pd.read_csv(arm_path)
    reference = pd.read_csv(reference_path)
    shared = sorted(set(arm["model"]) & set(reference["model"]))
    table = contrast_table(
        arm.loc[arm["model"].isin(shared)],
        reference.loc[
            reference["model"].isin(shared) & reference["regime"].isin(set(arm["regime"]))
        ],
        "imu",
    )
    assert not bool(table["raw_rmse_comparable"].any())
    assert set(table["observation_mode"]) == {"imu"}
    assert set(table["reference_observation_mode"]) == {"ideal"}
    # Only the two dimensionless columns crossed the boundary; no raw-scale column is in
    # the assembled table at all, so there is nothing for a reader to difference by hand.
    assert "rmse" not in table.columns
    with pytest.raises(ValueError, match="must not cross that boundary"):
        contrast_table(arm, reference, "imu", columns=("skill", "rmse"))


# ---------------------------------------------------------------------------
# `enforced`
# ---------------------------------------------------------------------------


def test_enforced_is_backfilled_for_a_table_that_predates_the_column(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    controls, _ = build_controls_table(root)
    assert {"experiment", "arm", "asserted", "enforced"} <= set(controls.columns)
    untrained = controls[controls["control"] == "untrained"]
    shuffle = controls[controls["control"] == "shuffle"]
    assert not untrained.empty
    # The untrained control is reported, not enforced -- on the backfilled reference rows
    # and on the arm rows that carried the column themselves, which is what makes the
    # backfill a reconstruction of the same fact rather than a second convention.
    assert not bool(untrained["enforced"].any())
    assert bool(shuffle["enforced"].all())
    # And it is asserted throughout: the two columns answer different questions.
    assert bool(untrained["asserted"].all())


def test_the_rendered_control_summary_accounts_for_every_row_of_the_source() -> None:
    """No control row may be present in the file and absent from the table.

    The renderer groups the summary on ``(experiment, arm, control, regime, ...)``. An empty
    provenance value round-trips through CSV as NaN and pandas drops NaN group keys, so a
    blank ``arm`` deletes those rows from the rendered table while leaving them in the file.
    Measured when ``e03_probabilistic`` carried an empty arm: all 288 of its rows vanished
    from a summary that still cited a 1152-row source. Asserted as a conservation law --
    the summary's cell counts must sum to the source's row count -- rather than as a check
    on the one column that happened to break it.
    """
    controls = RESULTS / PHASE6_SUBDIR / "controls.csv"
    if not controls.is_file():
        pytest.skip("controls.csv has not been assembled into the committed results")
    frame = pd.read_csv(controls)
    asserted, reported_only = split_controls(frame)
    covered = 0
    for part, by in (
        (asserted, ("experiment", "arm", "control", "regime", "subject_model", "null_model")),
        (reported_only, ("experiment", "arm", "control", "regime", "dof")),
    ):
        if not part.empty:
            covered += int(_control_summary(part, by=by)["n_cells"].sum())
    assert covered == len(frame)


def test_a_provenance_value_that_would_vanish_from_the_summary_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path / "results")
    controls, _ = build_controls_table(root)
    assert set(controls["arm"]) == {"reference", "ss_conditioned"}
    assert not (controls["arm"].astype(str).str.strip() == "").any()
    # Declaring a source with a blank arm -- which is what `e03_probabilistic` had before
    # NOT_AN_ARM existed -- is refused rather than written, because the file would then
    # state one row count in its marker and render a smaller table under it.
    monkeypatch.setattr(assemble, "CONTROL_SOURCE_DIRS", (("e02", "e02_deep", ""),))
    with pytest.raises(ValueError, match="empty"):
        build_controls_table(root)


def test_an_unknown_control_is_refused_rather_than_defaulted(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    frame = _controls_frame(with_flags=False)
    frame["control"] = "some_new_control"
    frame.to_csv(root / "e02" / "baselines_controls.csv", index=False)
    with pytest.raises(ValueError, match="ENFORCED_BY_CONTROL"):
        build_controls_table(root)


#: Control function name -> the control key :data:`ENFORCED_BY_CONTROL` records it under.
#: Both drivers together must cover every key: a control in the map that nothing calls is a
#: claim about a run nobody made.
_CONTROL_KEY_BY_FUNCTION: dict[str, str] = {
    "shuffle_control": "shuffle",
    "untrained_control": "untrained",
    "interval_shuffle_control": "interval_shuffle",
    "interval_untrained_control": "interval_untrained",
}


def _strict_keywords(function: object) -> dict[str, bool]:
    """Return the ``strict`` argument one driver passes to each control it calls.

    Parsed from the driver's own source rather than read from a comment: the backfill claims
    to know how each control is called, and a claim about another module's source is
    checkable against that source.

    Args:
        function: The driver function to parse.

    Returns:
        Control function name -> the ``strict`` value passed, defaulting to the control's own
        default, which is True for the two enforced controls and False for
        ``interval_untrained_control``.
    """
    defaults = {name: name != "interval_untrained_control" for name in _CONTROL_KEY_BY_FUNCTION}
    tree = ast.parse(inspect.getsource(function))  # type: ignore[arg-type]
    found: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in _CONTROL_KEY_BY_FUNCTION:
            continue
        strict = next((kw for kw in node.keywords if kw.arg == "strict"), None)
        found[node.func.id] = (
            defaults[node.func.id] if strict is None else bool(ast.literal_eval(strict.value))
        )
    return found


def test_the_enforcement_map_matches_the_training_driver() -> None:
    passed = _strict_keywords(experiment._run_controls)
    assert passed == {"shuffle_control": True, "untrained_control": False}
    assert ENFORCED_BY_CONTROL["shuffle"] is passed["shuffle_control"]
    assert ENFORCED_BY_CONTROL["untrained"] is passed["untrained_control"]


def test_the_enforcement_map_matches_the_interval_control_driver() -> None:
    """The successor to the test that asserted this gap was open.

    ``interval_shuffle_control`` and ``interval_untrained_control`` used to exist, be tested,
    and be called by nothing -- Phase 6 carry-forward item 6 closed in code and open in
    artifacts (P6-D15). :mod:`dmf.eval.control_runner` is now their caller, so the assertion
    changes from "nobody calls them" to "the map describes how they are called": the shuffle
    control is enforced, the untrained one is reported (P3-D9).
    """
    passed = _strict_keywords(control_runner.run_regime_controls)
    assert passed == {"interval_shuffle_control": True, "interval_untrained_control": False}
    assert ENFORCED_BY_CONTROL["interval_shuffle"] is passed["interval_shuffle_control"]
    assert ENFORCED_BY_CONTROL["interval_untrained"] is passed["interval_untrained_control"]


def test_every_control_in_the_map_has_a_caller() -> None:
    """A mapping entry for a control nothing runs claims an enforcement nobody made."""
    called = {
        _CONTROL_KEY_BY_FUNCTION[name]
        for driver in (experiment._run_controls, control_runner.run_regime_controls)
        for name in _strict_keywords(driver)
    }
    assert called == set(ENFORCED_BY_CONTROL)


# ---------------------------------------------------------------------------
# The probabilistic floor
# ---------------------------------------------------------------------------


def _floor_by_seed(model: str = "residual_interval") -> pd.DataFrame:
    """Return a frame shaped like a committed ``probabilistic_by_seed.csv``."""
    return pd.DataFrame(
        [
            {
                "model": model,
                "head": "quantile",
                "regime": regime,
                "dof": "roll",
                "horizon_samples": samples,
                "horizon_s": seconds,
                "seed": 0,
                "deterministic": True,
                "n_windows": 1000,
                "n_quantiles": 9,
                "alpha": 0.1,
                "signal_std": 3.0,
                "picp": 0.90,
                "mean_interval_width": 1.0,
                "winkler": 1.2,
                "crps": 0.3,
                "pinball": 0.15,
                "crossing_rate": 0.0,
                "picp_ci_lo": 0.88,
                "picp_ci_hi": 0.92,
                "n_params": 60_300,
                "val_loss_name": "pinball",
                "fit_time_s": 1.0,
            }
            for regime in ("id", "unseen_seastate")
            for samples, seconds in HORIZONS
        ]
    )


def test_the_floor_lands_on_the_schema_that_joins_the_committed_heads(tmp_path: Path) -> None:
    root = _tree(tmp_path / "results")
    floor_dir = root / PHASE6_SUBDIR / "e04g_residual_floor"
    floor_dir.mkdir(parents=True, exist_ok=True)
    _floor_by_seed().to_csv(floor_dir / "probabilistic_by_seed.csv", index=False)
    frame, skipped = build_probabilistic_baseline(root)
    # Same columns as results/e03/probabilistic.csv, so the floor and the heads can be read
    # as one section without either being re-derived.
    assert tuple(frame.columns) == PROBABILISTIC_COLUMNS
    committed = RESULTS / "e03" / "probabilistic.csv"
    if committed.is_file():
        assert tuple(pd.read_csv(committed).columns) == tuple(frame.columns)
    assert "probabilistic_baseline/join" in skipped or not skipped
    # Closed-form: one row per cell, `n_seeds = 1`, and a NaN std rather than a 0.0 that
    # would claim a spread nobody measured (P3-D10).
    assert set(frame["n_seeds"]) == {1}
    assert frame["picp_std"].isna().all()


def test_a_floor_that_would_collide_with_a_committed_head_is_refused(tmp_path: Path) -> None:
    committed = RESULTS / "e03" / "probabilistic.csv"
    if not committed.is_file():
        pytest.skip("no committed Phase 5 heads")
    heads = pd.read_csv(committed)
    stolen = str(heads["model"].iloc[0])
    root = _tree(tmp_path / "results")
    (root / "e03").mkdir(parents=True, exist_ok=True)
    heads.to_csv(root / "e03" / "probabilistic.csv", index=False)
    floor_dir = root / PHASE6_SUBDIR / "e04g_residual_floor"
    floor_dir.mkdir(parents=True, exist_ok=True)
    floor = _floor_by_seed(stolen)
    floor["regime"] = str(heads["regime"].iloc[0])
    floor["dof"] = str(heads["dof"].iloc[0])
    floor["head"] = str(heads["head"].iloc[0])
    floor["horizon_samples"] = int(heads["horizon_samples"].iloc[0])
    floor.to_csv(floor_dir / "probabilistic_by_seed.csv", index=False)
    # The probabilistic schema is not keyed on the experiment, so one label in both files
    # gives two rows for one cell with no way to tell them apart.
    with pytest.raises(ValueError, match="share"):
        build_probabilistic_baseline(root)


# ---------------------------------------------------------------------------
# The reproducibility control, against the real committed CSVs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reproducibility() -> pd.DataFrame:
    """Return the reproducibility table built from the committed results, or skip."""
    arm = RESULTS / PHASE6_SUBDIR / "e04f_revin_ood" / "baselines.csv"
    if not arm.is_file() or not (RESULTS / "e02" / "baselines.csv").is_file():
        pytest.skip("the RevIN arm or the reference arm is not committed")
    frame, skipped = build_reference_reproducibility(RESULTS)
    assert not skipped
    return frame


def test_the_corpus_level_quantities_reproduce_exactly(reproducibility: pd.DataFrame) -> None:
    """P6-D13's headline: the two quantities that carry no model are bitwise identical.

    ``rmse_persistence`` and ``signal_std`` are properties of the corpus, the split and the
    window geometry alone. They reproduce to 0.000e+00 across a fit run weeks later under a
    changed codebase, which is what makes reading ``results/e02/`` as the reference arm a
    measured claim rather than an assumption six of the seven arms depend on.
    """
    margin = reproducibility[reproducibility["model"] == "any"].set_index("quantity")
    for quantity in ("rmse_persistence", "signal_std"):
        row = margin.loc[quantity]
        assert float(row["max_abs_diff"]) == 0.0
        assert float(row["max_rel_diff"]) == 0.0
        assert bool(row["bitwise_identical"])
        assert int(row["n_differing"]) == 0
        assert int(row["n_rows"]) == 360


def test_the_published_quantity_figures_are_reproduced(reproducibility: pd.DataFrame) -> None:
    expected = {
        "rmse_mean": (6.674e-07, 3.95e-06),
        "mae_mean": (3.420e-07, 3.22e-06),
        "skill_mean": (4.812e-05, 4.78e-06),
        "nrmse_mean": (1.015e-05, 3.95e-06),
    }
    margin = reproducibility[reproducibility["model"] == "any"].set_index("quantity")
    for quantity, (absolute, relative) in expected.items():
        row = margin.loc[quantity]
        assert float(row["max_abs_diff"]) == pytest.approx(absolute, rel=1e-3)
        assert float(row["max_rel_diff"]) == pytest.approx(relative, rel=1e-2)


def test_only_ar20_moves_and_the_row_count_says_which_column_that_is(
    reproducibility: pd.DataFrame,
) -> None:
    """P6-D13's by-model table, with the column it was computed on made explicit.

    The entry reports "``ar20``: 70 of 72 rows differ, max abs 6.674e-07" without saying
    which quantity that counts. It is ``rmse_mean``; the same join gives 71 differing rows on
    ``nrmse_mean`` and 50 on ``skill_mean``. Crossing model against quantity is what makes
    that answerable from the artifact instead of from a rerun.
    """
    frame = reproducibility[reproducibility["model"] != "any"]
    for model in ("persistence", "window_mean", "damped_persistence", "dlinear_ols"):
        rows = frame[frame["model"] == model]
        assert len(rows) == len(REPRODUCIBILITY_QUANTITIES)
        assert bool(rows["bitwise_identical"].all()), model
        assert int(rows["n_rows"].max()) == 72

    ar20 = frame[frame["model"] == "ar20"].set_index("quantity")
    assert int(ar20.loc["rmse_mean", "n_differing"]) == 70
    assert float(ar20.loc["rmse_mean", "max_abs_diff"]) == pytest.approx(6.673885e-07, rel=1e-4)
    assert int(ar20.loc["nrmse_mean", "n_differing"]) == 71
    assert int(ar20.loc["skill_mean", "n_differing"]) == 50
    # Every difference is orders of magnitude below the fourth decimal every table renders,
    # so no printed digit and no conclusion moves.
    assert float(ar20["max_abs_diff"].max()) < 1e-4


def test_there_is_no_margin_over_quantities(reproducibility: pd.DataFrame) -> None:
    # A maximum over an RMSE in degrees, a dimensionless nrmse and a skill score is not a
    # number, and printing one would invite exactly the misreading P6-D13's by-model table
    # already produced.
    assert "any" not in set(reproducibility["quantity"])
    assert set(reproducibility["quantity"]) == set(REPRODUCIBILITY_QUANTITIES)


# ---------------------------------------------------------------------------
# The capacity flag on the real committed rows
# ---------------------------------------------------------------------------


def _committed_contrast(experiment: str, arm: str) -> pd.DataFrame:
    """Contrast one committed arm directory against the committed reference arm."""
    arm_path = RESULTS / PHASE6_SUBDIR / experiment / "baselines_by_seed.csv"
    reference_path = RESULTS / "e02" / "baselines_by_seed.csv"
    if not arm_path.is_file() or not reference_path.is_file():
        pytest.skip(f"{experiment} has not been scored yet")
    arm_rows = pd.read_csv(arm_path)
    reference = pd.read_csv(reference_path)
    shared = sorted(set(arm_rows["model"]) & set(reference["model"]))
    return contrast_table(
        arm_rows.loc[arm_rows["model"].isin(shared)],
        reference.loc[
            reference["model"].isin(shared) & reference["regime"].isin(set(arm_rows["regime"]))
        ],
        arm,
    )


def test_on_the_sea_state_arm_exactly_ar20_is_capacity_confounded() -> None:
    """P6-D8 defect 4, measured from the committed rows rather than transcribed.

    The one-hot's four extra channels cost AR(20) 66.12% more parameters and cost the
    channel-shared and trivial vehicles nothing. If the conditioned AR row beat the
    unconditioned one, the increase would not be attributable to sea-state information -- so
    the flag has to land on that model and only that model on this arm.
    """
    table = _committed_contrast("e04c_ss_conditioned_ood", "ss_conditioned")
    flagged = table.groupby("model")["capacity_confounded"].all()
    assert set(flagged[flagged].index) == {"ar20"}
    ar20 = table[table["model"] == "ar20"]
    assert float(ar20["param_delta_frac"].max()) == pytest.approx(0.6612, abs=1e-4)
    assert set(ar20["n_params"]) == {180_900}
    assert set(ar20["n_params_reference"]) == {108_900}
    # And the arm's other vehicles are flagged blind instead, which is the other half of why
    # a zero contrast there is not evidence of no effect.
    blind = table.groupby("model")["vehicle_blind_to_arm"].all()
    assert set(blind[blind].index) == {
        "persistence",
        "window_mean",
        "damped_persistence",
        "dlinear_ols",
    }
    for model in blind[blind].index:
        rows = table[table["model"] == model]
        assert float(rows["skill_diff"].abs().max()) == 0.0, model


def test_the_channels_arm_is_capacity_confounded_too_and_the_flag_says_so() -> None:
    """The sea-state arm is not the only capacity-confounded one, and the flag says so.

    The ``attitude_only`` arm forecasts three channels instead of six, so AR(20) drops from
    108 900 parameters to 27 450 and ``DampedPersistence`` from 6 to 3, while the
    channel-shared ``dlinear_ols`` is exactly matched at 60 300 -- which is the cell P6-D4
    item 2 calls clean and the one a name-based flag would have gotten wrong in both
    directions. A flag derived from the counts follows the corpus; one derived from a model
    name would have to be edited for every arm.
    """
    table = _committed_contrast("e04b_channels_ood", "attitude_only")
    flagged = table.groupby("model")["capacity_confounded"].all()
    assert set(flagged[flagged].index) == {"ar20", "damped_persistence"}
    deltas = table.groupby("model")["param_delta_frac"].first()
    assert deltas["ar20"] == pytest.approx(-0.7479, abs=1e-4)
    assert deltas["damped_persistence"] == pytest.approx(-0.5)
    assert deltas["dlinear_ols"] == 0.0
    assert not bool(table.loc[table["model"] == "dlinear_ols", "capacity_confounded"].any())
