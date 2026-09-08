"""``results/results.md``: the renderer, and the properties Gate 6 reads off it.

What is pinned here:

1. **The document is a projection of the CSVs.** Every rendered table carries a provenance
   marker naming an existing file and stating that file's row count, and the marker set is
   in one-to-one correspondence with the rendered tables. That is what makes "every number
   traces to a CSV" structural rather than a claim (``docs/protocol.md`` P6-D1, predicates
   3 and 4).
2. **Per-run rows are aggregated, not rendered.** ``metrics_full.csv`` is written one row
   per (model, seed, regime, DOF, horizon) (P6-D10). Rendering those rows as the table
   would breach CLAUDE.md non-negotiable 5 while looking like a complete table, so the
   tests check the rendered row count is the *aggregated* one and that a two-seed
   stochastic model is refused outright.
3. **The reporting rules that are easy to satisfy in prose and easy to break in code.**
   nrmse beside every skill; base rate beside every F1 and grouped by sea state; interval
   width beside every coverage and never pooled across the two horizon bands; a cell with
   no scorable onset rendered as neither a success nor a failure; a lag that is
   unidentified never printed as a number; the flags that stop a zero contrast reading as
   evidence of no effect.
4. **The schema the renderer expects is the schema the scorer writes.** The last test
   scores the fixture corpus through :mod:`dmf.eval.scoring` and renders that, so a column
   renamed on the producing side fails here rather than in a 46 h sweep's last step.

The synthetic frames below are deliberately small and hand-built: every property under
test is a *structural* one, and a fixture whose numbers came from a model would make a
failure ambiguous between the renderer and the model.
"""

import itertools
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmf.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, load_data
from dmf.eval.assemble import NOT_AN_ARM
from dmf.eval.report import (
    LONG_BAND,
    NOT_SCORABLE,
    PHASE6_SUBDIR,
    SHORT_BAND,
    TABLE_SOURCE_PATTERN,
    RenderedTableSpec,
    _control_summary,
    _resample_units_note,
    build_ablation_table,
    build_floor_contrast,
    build_point_table,
    build_probabilistic_view,
    build_quiescence_table,
    build_results_report,
    dmf_table_marker,
    horizon_band,
    parse_table_sources,
    split_ablation_rows,
    split_controls,
    table_source_marker,
)
from dmf.eval.scoring import SCORING_ARTIFACTS, score_experiment

#: Horizons of the synthetic frames: one in each band P5-D15 forbids pooling.
HORIZONS: tuple[tuple[int, float], ...] = ((10, 1.0), (150, 15.0))

#: (label, deterministic, seeds) for the synthetic model set. Three seeds for the SGD row,
#: one for the closed-form rows, which is the P3-D10 exemption in fixture form.
MODELS: tuple[tuple[str, bool, int], ...] = (
    ("persistence", True, 1),
    ("dlinear_ols", True, 1),
    ("tcn", False, 3),
)


def _metrics_full(models: tuple[tuple[str, bool, int], ...] = MODELS) -> pd.DataFrame:
    """Return a synthetic per-run accuracy table.

    Args:
        models: (label, deterministic, seed count) triples.

    Returns:
        One row per (model, seed, regime, DOF, horizon). The 15 s rows carry
        ``phase_lag_identified = False``, which is the legitimate case on a ~12 s roll
        period (P6-D3).
    """
    rows = []
    for (label, deterministic, n_seeds), regime, dof, (samples, seconds) in itertools.product(
        models, ("id", "unseen_seastate"), ("roll", "pitch"), HORIZONS
    ):
        for seed in range(n_seeds):
            rows.append(
                {
                    "model": label,
                    "seed": seed,
                    "regime": regime,
                    "dof": dof,
                    "horizon_samples": samples,
                    "horizon_s": seconds,
                    "n_windows": 1000,
                    "rmse": 0.5 + 0.01 * seed,
                    "mae": 0.4,
                    "rmse_persistence": 1.0,
                    "skill": 0.75 + 0.001 * seed,
                    "signal_std": 2.0,
                    "nrmse": 0.25,
                    "skill_ci_lo": 0.70 - 0.01 * seed,
                    "skill_ci_hi": 0.80 + 0.01 * seed,
                    "phase_lag_s": 0.2 if samples == 10 else 5.9,
                    "phase_lag_raw_s": 0.2 if samples == 10 else 6.0,
                    "dominant_period_s": 12.0,
                    "phase_lag_identified": samples == 10,
                    "n_realizations_phase": 32,
                    "deterministic": deterministic,
                    "n_params": 1000,
                    "fit_time_s": 1.0 + seed,
                }
            )
    return pd.DataFrame(rows)


def _quiescence() -> pd.DataFrame:
    """Return a synthetic per-run quiescence table.

    Returns:
        One row per (model, train seed, threshold set, rule, sea state). SS3 at
        ``permissive`` is the unscorable cell of P6-D7 item 1: base rate 1.0, no interior
        onset, and one predicted onset so that the false-alarm rate stays a real number
        there.
    """
    rows = []
    for label, _deterministic, n_seeds in (*MODELS, ("always_quiescent", True, 1)):
        for threshold_set, ss in itertools.product(("permissive", "strict"), ("SS3", "SS5")):
            scorable = not (threshold_set == "permissive" and ss == "SS3")
            for rule in ("point", "interval"):
                for seed in range(n_seeds):
                    rows.append(
                        {
                            "model": label,
                            "train_seed": -1 if label == "always_quiescent" else seed,
                            "regime": "id",
                            "observation_mode": "ideal",
                            "threshold_set": threshold_set,
                            "rule": rule,
                            "ss": ss,
                            "scorable": scorable,
                            "base_rate": 0.5 if scorable else 1.0,
                            "n_true_onsets": 30 if scorable else 0,
                            "n_pred_onsets": 28 if scorable else 1,
                            "n_matched": 20 if scorable else 0,
                            "precision": 0.7 if scorable else 0.0,
                            "recall": 0.66 if scorable else 0.0,
                            "f1": 0.68 if scorable else 0.0,
                            "false_alarms_per_min": 0.5,
                            "lead_p10": 1.0,
                            "lead_p50": 2.0,
                            "lead_p90": 3.0,
                            "n_realizations": 16,
                            "duration_s": 600.0,
                            "n_excluded_true": 1,
                            "n_excluded_pred": 0,
                        }
                    )
    return pd.DataFrame(rows)


def _ablations(*, with_capacity_flag: bool = True) -> pd.DataFrame:
    """Return a synthetic per-run contrast table.

    Args:
        with_capacity_flag: Whether to carry the ``capacity_confounded`` column.

    Returns:
        Rows for two ablations, plus the RevIN arm's closed-form row (which belongs under
        the reproducibility control, P6-D13) and an unfittable AR row (P6-D8 defect 2).
    """
    rows = []
    combos = (
        ("observation_mode", "imu", ("dlinear_ols", True, 1), "id"),
        ("normalization", "revin", ("tcn", False, 3), "id"),
        ("normalization", "revin", ("dlinear_ols", True, 1), "unseen_heading"),
        # The lookback arm, which had no producer at all until the matched-origin re-scoring
        # existed (P6-D15 finding 1). Its rows reach the renderer by the same path as every
        # other arm's, so what is tested here is that the section renders them under the
        # slug Gate 6 predicate 5 requires.
        ("lookback", "lookback_40s", ("dlinear_ols", True, 1), "id"),
    )
    for ablation, arm, (label, deterministic, n_seeds), regime in combos:
        for seed in range(n_seeds):
            rows.append(
                {
                    "ablation": ablation,
                    "arm": arm,
                    "reference_arm": "reference",
                    "model": label,
                    "seed": seed,
                    "regime": regime,
                    "observation_mode": "ideal",
                    "logical_dof": "roll",
                    "dof": "roll",
                    "horizon_samples": 10,
                    "horizon_s": 1.0,
                    "lookback": 200,
                    "reference_lookback": 200,
                    "parameter_matched": True,
                    "privileged_information": False,
                    # The production grid: 48 cells on `id` and `unseen_vessel`, 12 on the
                    # two regimes that hold out one level of one factor. Carried in the
                    # fixture because it is what the paired interval was drawn from, and a
                    # document that does not say so lets a 12-cluster interval read like a
                    # 48-cluster one.
                    "ci_resample_unit": "grid_cell",
                    "ci_n_units": 48 if regime in ("id", "unseen_vessel") else 12,
                    "vehicle_blind_to_arm": False,
                    "not_fittable": "",
                    "note": "a reading instruction",
                    "skill": 0.70,
                    "skill_reference": 0.71,
                    "skill_diff": -0.01,
                    "nrmse": 0.30,
                    "nrmse_reference": 0.29,
                    "nrmse_diff": 0.01,
                    "deterministic": deterministic,
                }
            )
    rows.append(
        {
            "ablation": "sea_state_conditioning",
            "arm": "ss_conditioned",
            "reference_arm": "reference",
            "model": "ar20",
            "seed": 0,
            "regime": "unseen_seastate",
            "observation_mode": "ideal",
            "logical_dof": "",
            "dof": "",
            "horizon_samples": -1,
            "horizon_s": -0.1,
            "lookback": 200,
            "reference_lookback": 200,
            "parameter_matched": False,
            "privileged_information": True,
            "ci_resample_unit": "grid_cell",
            "ci_n_units": np.nan,
            "vehicle_blind_to_arm": False,
            "not_fittable": "not fittable: the SS6 indicator is constant zero in training",
            "note": "UPPER BOUND",
            "skill": np.nan,
            "skill_reference": np.nan,
            "skill_diff": np.nan,
            "nrmse": np.nan,
            "nrmse_reference": np.nan,
            "nrmse_diff": np.nan,
            "deterministic": True,
        }
    )
    frame = pd.DataFrame(rows)
    if with_capacity_flag:
        frame["capacity_confounded"] = frame["model"] == "ar20"
    return frame


def _controls() -> pd.DataFrame:
    """Return a synthetic control table with one asserted and one reported-only cell."""
    rows = []
    for control, subject, null in (
        ("shuffle", "shuffled", "window_mean"),
        ("untrained", "untrained", "persistence"),
    ):
        for regime, dof, asserted in (("id", "roll", True), ("unseen_heading", "pitch", False)):
            rows.append(
                {
                    "control": control,
                    "regime": regime,
                    "subject_model": subject,
                    "null_model": null,
                    "dof": dof,
                    "horizon_samples": 10,
                    "horizon_s": 1.0,
                    "skill_subject": -3.0,
                    "skill_null": -3.0,
                    "excess": 0.001 if asserted else 0.055,
                    "tol": 0.02,
                    "asserted": asserted,
                    # The shuffle control stops the run; the untrained control does not
                    # (P3-D9). Two different commitments, one column.
                    "enforced": control == "shuffle",
                    "passed": asserted,
                }
            )
    return pd.DataFrame(rows)


def _probabilistic(*, with_width: bool = True) -> pd.DataFrame:
    """Return a synthetic per-run probabilistic table.

    Args:
        with_width: Whether to carry ``mean_interval_width``. False builds the frame the
            renderer must refuse: a coverage with no sharpness beside it.

    Returns:
        Rows for the unconditional floor and one learned head, at one horizon in each band.
    """
    rows = []
    for label, deterministic, n_seeds in (
        ("residual_interval", True, 1),
        ("tcn_quantile", False, 3),
    ):
        for regime, (samples, seconds) in itertools.product(("id", "unseen_seastate"), HORIZONS):
            for seed in range(n_seeds):
                rows.append(
                    {
                        "model": label,
                        "head": "quantile",
                        "regime": regime,
                        "dof": "roll",
                        "horizon_samples": samples,
                        "horizon_s": seconds,
                        "seed": seed,
                        "deterministic": deterministic,
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
                        "n_params": 100,
                        "val_loss_name": "pinball",
                        "fit_time_s": 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame if with_width else frame.drop(columns=["mean_interval_width"])


def _write_results_dir(root: Path, **overrides: pd.DataFrame | None) -> Path:
    """Write a complete synthetic Phase 6 results **root**.

    The Phase 6 CSVs live under ``<root>/e04`` beside the frozen Gate 3-5 directories, and
    ``results.md`` is written to the root itself, which is the layout
    ``docs/IMPLEMENTATION_PLAN.md`` §Phase 6 fixes and the one the Gate 6 read-out resolves
    marker paths against.

    Args:
        root: The results root to create.
        overrides: ``<stem>=frame`` replaces a default table; ``<stem>=None`` omits it.

    Returns:
        The root written.
    """
    defaults: dict[str, pd.DataFrame | None] = {
        "metrics_full": _metrics_full(),
        "quiescence": _quiescence(),
        "ablations": _ablations(),
        "controls": _controls(),
        "probabilistic_baseline": _probabilistic(),
    }
    defaults.update(overrides)
    phase6 = root / PHASE6_SUBDIR
    phase6.mkdir(parents=True, exist_ok=True)
    for stem, frame in defaults.items():
        if frame is not None:
            frame.to_csv(phase6 / f"{stem}.csv", index=False)
    return root


def _cells(row: str) -> list[str]:
    """Split one Markdown table row into its cell texts.

    Args:
        row: A rendered row, pipe-delimited.

    Returns:
        The trimmed cells, without the leading and trailing empties.
    """
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _column(table: list[str], name: str) -> list[str]:
    """Return one column of a rendered table by header name.

    Reading a rendered table by column rather than by substring matters here: several
    columns of these tables legitimately hold the same number, so a substring assertion
    can pass or fail for a reason that has nothing to do with the column under test.

    Args:
        table: The rendered lines, header first.
        name: Header of the column wanted.

    Returns:
        That column's cells, one per body row.

    Raises:
        AssertionError: If the header is not in the table.
    """
    header = _cells(table[0])
    assert name in header, f"{name!r} is not a column of this table: {header}"
    index = header.index(name)
    return [_cells(row)[index] for row in table[2:]]


def _rendered_tables(text: str) -> dict[str, list[str]]:
    """Split a document into ``table_id -> rendered lines`` for the table under each marker.

    Args:
        text: The rendered document.

    Returns:
        The Markdown rows of each table, header row first.
    """
    tables: dict[str, list[str]] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(TABLE_SOURCE_PATTERN, line.strip())
        if match is None:
            continue
        block: list[str] = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith("|"):
                block.append(candidate)
            elif block:
                break
        tables[match["table_id"]] = block
    return tables


@pytest.fixture
def report(tmp_path: Path) -> tuple[str, Path]:
    """Render the complete synthetic report once."""
    root = _write_results_dir(tmp_path / "results")
    out = build_results_report(root, root / "results.md")
    return out.read_text(encoding="utf-8"), root


# ---------------------------------------------------------------------------
# 1. Provenance: the property Gate 6 predicates 3 and 4 are read from.
# ---------------------------------------------------------------------------


#: The machine-readable marker the Gate 6 read-out parses, as this module's tests read it
#: back. Deliberately a second, independent expression of the grammar: a test that imported
#: the writer's own regex would pass against any format the writer happened to emit.
_DMF_MARKER_RE = re.compile(
    r"^<!-- dmf-table id=(?P<id>\S+) section=(?P<section>\S+) source=(?P<source>\S+) "
    r"csv_rows=(?P<csv_rows>\d+) rows=(?P<rows>\d+)(?: select=\"(?P<select>[^\"]*)\")? -->$"
)


def test_the_marker_round_trips_through_its_own_grammar() -> None:
    line = table_source_marker("s61.id", "results/e04/metrics_full.csv", 40, 12)
    (parsed,) = parse_table_sources(f"prose above\n{line}\nprose below")
    assert parsed.table_id == "s61.id"
    assert parsed.path == "results/e04/metrics_full.csv"
    assert (parsed.csv_rows, parsed.rendered_rows) == (40, 12)


@pytest.mark.parametrize(
    ("table_id", "path"),
    [("has spaces", "results/x.csv"), ("ok", "results/`x`.csv")],
)
def test_a_marker_the_readout_could_not_parse_is_refused(table_id: str, path: str) -> None:
    # A line the Gate 6 read-out cannot parse is indistinguishable from a table with no
    # provenance at all, so it fails at write time rather than at read time.
    with pytest.raises(ValueError):
        table_source_marker(table_id, path, 1, 1)


def test_every_rendered_table_names_an_existing_csv_and_states_its_row_count(
    report: tuple[str, Path],
) -> None:
    text, root = report
    sources = parse_table_sources(text)
    assert sources, "the document rendered no table at all"
    for source in sources:
        # Relative to the document's own directory, never to the working directory: the
        # Gate 6 read-out resolves `results_dir / source`, and a cwd-relative path would
        # also make the document a non-deterministic function of its inputs.
        assert not Path(source.path).is_absolute()
        path = root / source.path
        assert path.is_file(), f"{source.table_id} names {source.path}, which does not exist"
        assert source.csv_rows == len(pd.read_csv(path))


def test_the_marker_set_is_one_to_one_with_the_rendered_tables(
    report: tuple[str, Path],
) -> None:
    text, _ = report
    sources = parse_table_sources(text)
    ids = [source.table_id for source in sources]
    assert len(ids) == len(set(ids)), "two tables share a table_id, so a check on one is ambiguous"
    tables = _rendered_tables(text)
    assert set(tables) == set(ids)
    # Rows of the rendered table, excluding the header and the separator rule.
    for source in sources:
        assert source.rendered_rows == len(tables[source.table_id]) - 2


def _machine_markers(text: str) -> dict[str, dict[str, str]]:
    """Parse every ``<!-- dmf-table ... -->`` marker out of a document.

    Args:
        text: The rendered document.

    Returns:
        ``table_id -> field mapping``, with ``select`` empty when the marker omits it.
    """
    found: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        match = _DMF_MARKER_RE.match(line.strip())
        if match is not None:
            fields = match.groupdict()
            found[fields["id"]] = {k: (v or "") for k, v in fields.items()}
    return found


def test_the_two_markers_of_every_table_carry_the_same_record(
    report: tuple[str, Path],
) -> None:
    text, _ = report
    machine = _machine_markers(text)
    visible = {source.table_id: source for source in parse_table_sources(text)}
    # One record, two renderings. If these could disagree, the document a human reads and
    # the document the Gate 6 read-out checks would be two different claims.
    assert set(machine) == set(visible)
    for table_id, fields in machine.items():
        assert fields["source"] == visible[table_id].path
        assert int(fields["csv_rows"]) == visible[table_id].csv_rows
        assert int(fields["rows"]) == visible[table_id].rendered_rows


def test_the_machine_marker_is_the_last_line_before_its_table(
    report: tuple[str, Path],
) -> None:
    text, _ = report
    lines = text.splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|")
        and index + 1 < len(lines)
        and lines[index + 1].startswith("|")
        and set(lines[index + 1]) <= set("|-: ")
    ]
    assert headers
    for index in headers:
        above = index - 1
        while above >= 0 and not lines[above].strip():
            above -= 1
        # `dmf.eval.gate.parse_rendered_tables` associates a table with the last non-blank
        # line before it. A marker one paragraph further up would belong to either of two
        # tables, and a provenance check whose association is ambiguous is worth nothing.
        assert _DMF_MARKER_RE.match(lines[above].strip()), (
            f"the table at line {index + 1} has no machine marker directly above it"
        )


def test_a_table_that_is_not_its_whole_file_says_how_it_was_filtered(
    report: tuple[str, Path],
) -> None:
    text, _ = report
    for table_id, fields in _machine_markers(text).items():
        if fields["csv_rows"] != fields["rows"]:
            # Declaring no `select` is the strictly stronger claim -- "this table is the
            # file, row for row" -- and the Gate 6 read-out enforces it literally there.
            assert fields["select"], f"{table_id} is a projection and does not say so"


def test_a_machine_marker_that_could_not_be_parsed_is_refused() -> None:
    spec = RenderedTableSpec(
        table_id="core_metrics",
        section="6.1",
        source="e04/metrics_full.csv",
        csv_rows=4,
        rendered_rows=2,
    )
    assert _DMF_MARKER_RE.match(dmf_table_marker(spec))
    for broken in (
        replace(spec, source="e04/metrics full.csv"),
        replace(spec, select='a "quoted" filter'),
        replace(spec, table_id="not a slug"),
    ):
        with pytest.raises(ValueError):
            dmf_table_marker(broken)


def test_the_required_sections_are_all_present(report: tuple[str, Path]) -> None:
    text, _ = report
    ids = {source.table_id for source in parse_table_sources(text)}
    # The slugs are a vocabulary, not free text: the Gate 6 read-out matches its
    # required-table list against exactly these strings, so a synonym is a table the gate
    # reports as missing while the reader is looking straight at it.
    required = {
        "core_metrics",
        "quiescence_detection",
        "quiescence_base_rate",
        "quiescence_lead_time",
        "ablation_observation_mode",
        "ablation_normalization",
        "probabilistic_floor",
        "controls_point_asserted",
    }
    assert required <= ids, f"missing {sorted(required - ids)}"


def test_both_threshold_sets_are_reported(report: tuple[str, Path]) -> None:
    table = _rendered_tables(report[0])["quiescence_detection"]
    assert set(_column(table, "threshold_set")) == {"permissive", "strict"}
    # Both rules too: the interval rule is stricter by definition (P6-D5), so a reader
    # comparing them has to see both.
    assert set(_column(table, "rule")) == {"point", "interval"}


def test_a_missing_required_table_is_fatal_by_default(tmp_path: Path) -> None:
    root = _write_results_dir(tmp_path / "results", ablations=None)
    with pytest.raises(FileNotFoundError, match="ablations.csv"):
        build_results_report(root, root / "results.md")


def test_an_incomplete_report_says_so_and_carries_no_marker_for_what_is_missing(
    tmp_path: Path,
) -> None:
    root = _write_results_dir(tmp_path / "results", ablations=None)
    text = build_results_report(root, root / "results.md", strict=False).read_text()
    assert "This document is incomplete" in text
    assert "Not rendered" in text
    # The absent section contributes no marker, so a read-out counting markers cannot
    # mistake it for a rendered table.
    assert not [s for s in parse_table_sources(text) if s.table_id.startswith("ablation_")]


# ---------------------------------------------------------------------------
# 2. The seed rule.
# ---------------------------------------------------------------------------


def test_the_document_renders_seed_aggregates_and_not_per_run_rows(
    report: tuple[str, Path],
) -> None:
    text, root = report
    per_run = pd.read_csv(root / PHASE6_SUBDIR / "metrics_full.csv")
    cells = per_run.groupby(["model", "regime", "dof", "horizon_samples"]).ngroups
    rendered = sum(
        source.rendered_rows
        for source in parse_table_sources(text)
        if source.table_id == "core_metrics"
    )
    assert rendered == cells < len(per_run)
    table = _rendered_tables(text)["core_metrics"]
    assert "n_seeds" in table[0]
    # The three-seed row says three and the closed-form row says one, with a NaN std
    # rendered as "n/a" rather than as a measured zero.
    seeds = dict(zip(_column(table, "model"), _column(table, "n_seeds"), strict=True))
    assert seeds["tcn"] == "3"
    assert seeds["persistence"] == "1"
    # NaN, not 0.0: with one observation the sample standard deviation is undefined, and
    # 0.0 would claim a measurement that was never made (P3-D10).
    stds = dict(zip(_column(table, "model"), _column(table, "skill_std"), strict=True))
    assert stds["persistence"] == "n/a"
    assert float(stds["tcn"]) > 0.0


def test_two_experiments_scoring_the_same_label_are_not_averaged_together() -> None:
    # `scripts/evaluate.py` scores several configs into one metrics_full.csv, and every
    # config carries `persistence` because it is the skill denominator. Without the
    # experiment in the key, the two three-seed `tcn` runs below would collapse into one
    # row claiming n_seeds = 3 over six runs.
    left = _metrics_full(models=(("tcn", False, 3),)).assign(experiment="e02_deep")
    right = _metrics_full(models=(("tcn", False, 3),)).assign(experiment="e04a_obs_mode")
    table = build_point_table(pd.concat([left, right], ignore_index=True))
    assert set(table["experiment"]) == {"e02_deep", "e04a_obs_mode"}
    assert len(table) == 2 * table.groupby(["model", "regime", "dof", "horizon_samples"]).ngroups
    assert set(table["n_seeds"]) == {3}


def test_a_two_seed_stochastic_model_is_refused_rather_than_averaged() -> None:
    thin = _metrics_full(models=(("tcn", False, 2),))
    with pytest.raises(ValueError, match="fewer than 3 seeds"):
        build_point_table(thin)


def test_no_model_is_dropped_from_the_point_table(report: tuple[str, Path]) -> None:
    text, root = report
    per_run = pd.read_csv(root / PHASE6_SUBDIR / "metrics_full.csv")
    rendered = set(_column(_rendered_tables(text)["core_metrics"], "model"))
    assert rendered == set(per_run["model"])


# ---------------------------------------------------------------------------
# 3. Section 6.1: skill, nrmse and the phase lag that is not a measurement.
# ---------------------------------------------------------------------------


def test_nrmse_and_the_persistence_denominator_sit_beside_every_skill(
    report: tuple[str, Path],
) -> None:
    header = _rendered_tables(report[0])["core_metrics"][0]
    for column in ("skill_mean", "nrmse_mean", "rmse_persistence"):
        assert column in header
    columns = _cells(header)
    # Beside, not merely present: skill read across horizons without nrmse next to it is a
    # statement about the persistence denominator's own periodicity (P3-D5/P4-D3).
    assert abs(columns.index("nrmse_mean") - columns.index("skill_mean")) <= 4


def test_a_skill_column_with_no_persistence_denominator_is_refused() -> None:
    with pytest.raises(ValueError, match="rmse_persistence"):
        build_point_table(_metrics_full().drop(columns=["rmse_persistence"]))


def test_an_unidentified_phase_lag_is_not_rendered_as_a_number(
    report: tuple[str, Path],
) -> None:
    table = _rendered_tables(report[0])["core_metrics"]
    paired = list(zip(_column(table, "horizon_s"), _column(table, "phase_lag_s"), strict=True))
    assert paired
    for horizon, lag in paired:
        if float(horizon) >= 15.0:
            # The 15 s rows exceed a quarter of the 12 s dominant period, so their argmax
            # is ambiguous between the zero and the +-T replicas (P6-D3/P6-D10).
            assert lag == "unidentified"
        else:
            assert lag == "+0.200"


# ---------------------------------------------------------------------------
# 4. Section 6.2: base rates, sea-state grouping, and the unscorable cell.
# ---------------------------------------------------------------------------


def test_every_f1_row_carries_its_base_rate_in_the_same_row(report: tuple[str, Path]) -> None:
    header = _rendered_tables(report[0])["quiescence_detection"][0]
    assert "f1_mean" in header and "base_rate_mean" in header
    columns = _cells(header)
    # Beside, not merely present: the base rate is what makes recall interpretable, so it
    # is read without scanning across ten columns.
    assert columns.index("base_rate_mean") < columns.index("f1_mean")
    # And every table in the document that carries an F1 anywhere carries a base rate,
    # which is the predicate the Gate 6 read-out evaluates column by column.
    for table in _rendered_tables(report[0]).values():
        names = _cells(table[0])
        if any(re.search(r"(?:^|_)f1(?:_|$)", name) for name in names):
            assert any("base_rate" in name for name in names)


def test_f1_is_grouped_by_sea_state_and_never_pooled(report: tuple[str, Path]) -> None:
    header = _rendered_tables(report[0])["quiescence_detection"][0]
    assert "| ss |" in header
    # The grouping keys stay on the row: a table that pooled sea states, threshold sets or
    # regimes into a heading could not be checked for pooling at all (P6-D7 item 3).
    for key in ("regime", "threshold_set", "rule"):
        assert f"| {key} |" in header
    assert set(_column(_rendered_tables(report[0])["quiescence_detection"], "ss")) == {
        "SS3",
        "SS5",
    }


def test_a_cell_with_no_scorable_onset_is_neither_a_success_nor_a_failure(
    report: tuple[str, Path],
) -> None:
    table = _rendered_tables(report[0])["quiescence_detection"]
    columns = {
        name: _column(table, name)
        for name in (
            "ss",
            "threshold_set",
            "f1_mean",
            "precision_mean",
            "recall_mean",
            "base_rate_mean",
            "n_true_onsets_mean",
            "false_alarms_per_min_mean",
            "lead_p50_mean",
        )
    }
    ss3 = [
        index
        for index, (ss, threshold_set) in enumerate(
            zip(columns["ss"], columns["threshold_set"], strict=True)
        )
        if ss == "SS3" and threshold_set == "permissive"
    ]
    assert ss3
    for index in ss3:
        # Never F1 = 0 (which reads as a model failure) and never F1 = 1 (which reads as a
        # success): there is nothing to detect in this cell (P6-D7 item 1).
        for name in ("f1_mean", "precision_mean", "recall_mean", "lead_p50_mean"):
            assert columns[name][index] == NOT_SCORABLE
        # What says why the cell is unscorable stays a number.
        assert columns["base_rate_mean"][index] == "1.0000"
        assert columns["n_true_onsets_mean"][index] == "0.0000"
        # So does the false-alarm rate, which is defined without any true onset at all.
        assert columns["false_alarms_per_min_mean"][index] == "0.5000"


def test_the_degenerate_detector_is_in_the_table(report: tuple[str, Path]) -> None:
    # Base-rate exploitation is quantified rather than warned about (P6-D2).
    assert "always_quiescent" in _column(
        _rendered_tables(report[0])["quiescence_detection"], "model"
    )


def test_a_quiescence_table_without_its_base_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="base_rate"):
        build_quiescence_table(_quiescence().drop(columns=["base_rate"]), {})


def test_the_deterministic_flag_is_read_from_the_accuracy_table_not_guessed() -> None:
    lookup = {"tcn": False, "persistence": True, "dlinear_ols": True}
    table = build_quiescence_table(_quiescence(), lookup)
    seeds = table.set_index(["model", "threshold_set", "rule", "ss"])["n_seeds"]
    assert int(seeds.loc[("tcn", "strict", "point", "SS5")]) == 3
    assert int(seeds.loc[("persistence", "strict", "point", "SS5")]) == 1
    # The synthetic detector is not a run of any model and has no training seed.
    assert int(seeds.loc[("always_quiescent", "strict", "point", "SS5")]) == 1


# ---------------------------------------------------------------------------
# 5. Section 6.3: the flags that stop a zero contrast reading as a finding.
# ---------------------------------------------------------------------------


def test_the_revin_arm_lists_only_the_tcn_rows(report: tuple[str, Path]) -> None:
    tables = _rendered_tables(report[0])
    assert set(_column(tables["ablation_normalization"], "model")) == {"tcn"}
    # Not dropped: the closed-form rows are reported under the reproducibility control,
    # because "no effect" rows in a table about RevIN's effect would read as a result and
    # CLAUDE.md non-negotiable 6 forbids dropping them (P6-D13).
    assert "dlinear_ols" in _column(tables["ablation_revin_closed_form"], "model")
    assert "reproducibility control" in report[0]


def test_the_split_keeps_every_row_somewhere() -> None:
    frame = _ablations()
    contrasts, unfittable, revin = split_ablation_rows(frame)
    assert len(contrasts) + len(unfittable) + len(revin) == len(frame)
    assert set(unfittable["model"]) == {"ar20"}
    assert set(revin["model"]) == {"dlinear_ols"}


def test_a_row_that_cannot_be_fitted_is_reported_with_its_reason(
    report: tuple[str, Path],
) -> None:
    table = _rendered_tables(report[0])["ablation_not_fittable"]
    assert "ar20" in _column(table, "model")
    assert any("constant zero in training" in reason for reason in _column(table, "not_fittable"))


def test_the_flags_travel_with_every_ablation_row(report: tuple[str, Path]) -> None:
    header = _rendered_tables(report[0])["ablation_observation_mode"][0]
    # `privileged_information`, not `upper_bound`: P6-D18 measured that the sea-state arm is
    # not an upper bound, and the column that renders on 792 rows must not carry a retracted
    # claim. What survives is that the information is unavailable at deployment.
    for flag in (
        "vehicle_blind_to_arm",
        "capacity_confounded",
        "privileged_information",
        "parameter_matched",
    ):
        assert flag in header
    assert "upper_bound" not in header


def test_a_flag_the_source_does_not_carry_is_reported_rather_than_inferred(
    tmp_path: Path,
) -> None:
    root = _write_results_dir(tmp_path / "results", ablations=_ablations(with_capacity_flag=False))
    text = build_results_report(root, root / "results.md").read_text()
    assert "does not carry `capacity_confounded`" in text


def test_the_ablation_contrast_is_aggregated_over_seeds() -> None:
    contrasts, _, _ = split_ablation_rows(_ablations())
    table = build_ablation_table(contrasts, {})
    tcn = table.loc[table["model"] == "tcn"]
    assert int(tcn["n_seeds"].iloc[0]) == 3
    assert float(tcn["skill_diff_mean"].iloc[0]) == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# 6. Section 6.4: coverage with sharpness, split by band.
# ---------------------------------------------------------------------------


def test_horizon_bands_are_the_two_p5_d15_names() -> None:
    assert horizon_band(1.0) == horizon_band(5.0) == SHORT_BAND
    assert horizon_band(10.0) == horizon_band(15.0) == LONG_BAND
    # A horizon outside both gets a band of its own rather than being folded into the
    # nearer one, which would pool two behaviours under one label.
    assert horizon_band(30.0) == "30 s"


def test_every_coverage_row_carries_a_width_and_a_band(report: tuple[str, Path]) -> None:
    header = _rendered_tables(report[0])["probabilistic_floor"][0]
    assert "picp_mean" in header
    assert "mean_interval_width_mean" in header and "width_ratio_mean" in header
    assert "| band |" in header
    # A per-row horizon too: without it the row's coverage would be a lead-time average
    # whatever the band column said.
    assert "| horizon_s |" in header


def test_no_rendered_coverage_row_pools_the_two_bands(report: tuple[str, Path]) -> None:
    bands = _column(_rendered_tables(report[0])["probabilistic_floor"], "band")
    assert bands
    for band in bands:
        assert band in (SHORT_BAND, LONG_BAND), f"{band!r} is neither band, so it pools or omits"
    assert set(bands) == {SHORT_BAND, LONG_BAND}


def test_a_coverage_with_no_width_beside_it_is_refused() -> None:
    with pytest.raises(ValueError, match="width"):
        build_probabilistic_view(_probabilistic(with_width=False), source="synthetic")


def test_the_unconditional_floor_is_in_the_table_beside_the_learned_head(
    report: tuple[str, Path],
) -> None:
    models = set(_column(_rendered_tables(report[0])["probabilistic_floor"], "model"))
    assert {"residual_interval", "tcn_quantile"} <= models


def test_coverage_under_unseen_seastate_is_reported_rather_than_omitted(
    report: tuple[str, Path],
) -> None:
    # P5-D12: the degradation is the finding, so the out-of-distribution regime is on the
    # row rather than dropped.
    regimes = set(_column(_rendered_tables(report[0])["probabilistic_floor"], "regime"))
    assert "unseen_seastate" in regimes


def _heads_and_floor(tmp_path: Path) -> tuple[str, Path]:
    """Render a report carrying both the floor and a committed Phase 5 heads file.

    The production layout: ``<root>/e04/probabilistic_baseline.csv`` holds the floor alone
    and ``<root>/e03/probabilistic.csv`` holds the heads, which is exactly the pair section
    6.4 has to difference.

    Args:
        tmp_path: Test temporary directory.

    Returns:
        ``(document text, results root)``.
    """
    frame = _probabilistic()
    floor = frame.loc[frame["model"] == "residual_interval"].copy()
    heads = frame.loc[frame["model"] == "tcn_quantile"].copy()
    # The head is twice as wide as the floor at the same coverage: sharper is negative, so
    # this row must render a POSITIVE width_ratio_diff and a zero picp_diff.
    heads["mean_interval_width"] = 2.0
    root = _write_results_dir(tmp_path / "results", probabilistic_baseline=floor)
    (root / "e03").mkdir(parents=True, exist_ok=True)
    heads.to_csv(root / "e03" / "probabilistic.csv", index=False)
    return build_results_report(root, root / "results.md").read_text(), root


def test_the_heads_are_differenced_against_the_floor_rather_than_argued_about(
    tmp_path: Path,
) -> None:
    """P6-D6's argument, rendered as a contrast instead of as prose beside two tables."""
    text, _ = _heads_and_floor(tmp_path)
    table = _rendered_tables(text)["probabilistic_head_minus_floor"]
    header = table[0]
    for column in ("picp_mean", "floor_picp", "picp_diff"):
        assert column in header
    for column in ("width_ratio_mean", "floor_width_ratio", "width_ratio_diff"):
        assert column in header, "a difference is printed beside both sides of it"
    assert set(_column(table, "model")) == {"tcn_quantile"}
    # Same coverage, twice the width: the head has learned nothing about its own
    # uncertainty, and it is the sharpness column that says so.
    assert {float(value) for value in _column(table, "picp_diff")} == {0.0}
    assert all(float(value) > 0.0 for value in _column(table, "width_ratio_diff"))


def test_the_floor_contrast_keeps_a_head_cell_the_floor_does_not_cover(
    tmp_path: Path,
) -> None:
    """Dropping it would narrow the comparison to the floor's cells, silently."""
    frame = _probabilistic()
    floor = frame.loc[frame["model"] == "residual_interval"].copy()
    heads = frame.loc[frame["model"] == "tcn_quantile"].copy()
    floor_view = build_probabilistic_view(floor, source="floor")
    heads_view = build_probabilistic_view(heads, source="heads")
    narrowed = floor_view.loc[floor_view["regime"] != "unseen_seastate"]
    contrast = build_floor_contrast(heads_view, narrowed)
    assert len(contrast) == len(heads_view)
    missing = contrast.loc[contrast["regime"] == "unseen_seastate"]
    assert missing["picp_diff"].isna().all()
    assert missing["width_ratio_diff"].isna().all()


def test_a_floor_table_that_is_not_a_floor_is_refused(tmp_path: Path) -> None:
    frame = _probabilistic()
    view = build_probabilistic_view(frame, source="both")
    heads = view.loc[view["model"] == "tcn_quantile"]
    with pytest.raises(ValueError, match="unconditional comparator|not unique"):
        build_floor_contrast(heads, heads)


def test_the_provenance_line_does_not_claim_the_floor_file_holds_the_heads(
    report: tuple[str, Path],
) -> None:
    """It held one model and said it held every learned head beside it (audit S8)."""
    text = report[0]
    assert "the residual-interval floor and every learned head scored beside it" not in text
    assert "the residual-interval floor **only**" in text


# ---------------------------------------------------------------------------
# 7. The controls.
# ---------------------------------------------------------------------------


def test_asserted_and_reported_only_cells_are_two_tables_and_not_one_verdict(
    report: tuple[str, Path],
) -> None:
    ids = {source.table_id for source in parse_table_sources(report[0])}
    assert {"controls_point_asserted", "controls_point_reported_only"} <= ids
    asserted = "\n".join(_rendered_tables(report[0])["controls_point_asserted"])
    reported = "\n".join(_rendered_tables(report[0])["controls_point_reported_only"])
    # The failing floored cell is visible, and it is not in the table the verdict is read
    # from (P6-D12).
    assert "0.0550" in reported and "0.0550" not in asserted
    assert "unseen_heading" in reported


def test_a_control_table_predating_the_narrowing_is_treated_as_fully_asserted() -> None:
    legacy = _controls().drop(columns=["asserted"])
    asserted, reported_only = split_controls(legacy)
    assert len(asserted) == len(legacy)
    assert reported_only.empty


def test_the_untrained_control_is_labelled_reported_not_enforced(
    report: tuple[str, Path],
) -> None:
    assert "reported, not enforced" in report[0]


def test_the_enforced_column_is_rendered_beside_the_verdict(report: tuple[str, Path]) -> None:
    """The distinction is a column, not only a paragraph.

    A reader who filters the table on `passed` sees the untrained control's failures without
    reading the prose above it; `enforced` is what tells them, in the same row, that those
    failures stop nothing.
    """
    table = _rendered_tables(report[0])["controls_point_asserted"]
    assert "enforced" in _cells(table[0])
    verdicts = dict(zip(_column(table, "control"), _column(table, "enforced"), strict=True))
    assert verdicts == {"shuffle": "yes", "untrained": "no"}
    # And the two columns are not the same question: the reported-only table holds cells
    # that were NOT asserted on inside a control that IS enforced.
    excluded = _rendered_tables(report[0])["controls_point_reported_only"]
    assert "enforced" in _cells(excluded[0])
    assert "yes" in _column(excluded, "enforced")


def test_a_control_table_that_never_recorded_enforcement_gets_no_such_column() -> None:
    """An absent commitment is not invented at render time.

    ``results/e02/baselines_controls.csv`` predates both flags. Rendering an `enforced`
    column for it would state a commitment nobody recorded, which is worse than not showing
    one -- the whole point of the column is that it carries a fact rather than an inference.
    """
    legacy = _controls().drop(columns=["enforced"])
    summary = _control_summary(legacy, by=("control", "regime"))
    assert "enforced" not in summary.columns
    assert "passed" in summary.columns


def test_the_missing_interval_control_is_called_a_gap(report: tuple[str, Path]) -> None:
    assert "No interval controls table" in report[0]


def _interval_controls() -> pd.DataFrame:
    """Return a synthetic ``interval_controls.csv`` with one narrowed cell.

    The narrowing that produces ``asserted=False`` here is P6-D21's -- the **null's** own
    PICP@90 outside Gate 5's band -- not the floored-cell one, which is deliberately never
    applied to an interval statistic (P6-D12). The two rows differ in exactly that.
    """
    rows = []
    for control, subject, enforced in (
        ("interval_shuffle", "shuffled_interval", True),
        ("interval_untrained", "untrained_head", False),
    ):
        for regime, picp_null, asserted in (("id", 0.902, True), ("unseen_seastate", 0.331, False)):
            rows.append(
                {
                    "control": control,
                    "regime": regime,
                    "subject_model": subject,
                    "null_model": "residual_interval",
                    "dof": "heave",
                    "horizon_samples": 100,
                    "horizon_s": 10.0,
                    "alpha": 0.1,
                    "metric": "winkler",
                    "loss_subject": 1.0,
                    "loss_null": 1.5 if asserted else 1.36,
                    "excess": -0.5 if asserted else 0.2649,
                    "tol": 0.10,
                    # False only where the NULL is miscalibrated, and the number that
                    # decision was taken on is in `picp_null` on the same row.
                    "asserted": asserted or control == "interval_untrained",
                    "enforced": enforced,
                    "on_residual_floor": False,
                    "passed": asserted,
                    "picp_subject": 0.9,
                    "picp_null": picp_null,
                    "width_subject": 2.0,
                    "width_null": 1.5,
                    "crossing_rate_subject": 0.0,
                    "crossing_rate_null": 0.0,
                    "experiment": "e03_probabilistic",
                    "arm": NOT_AN_ARM,
                }
            )
    return pd.DataFrame(rows)


def _section_of(text: str, heading: str) -> str:
    """Return the lines under one Markdown heading, up to the next heading of any depth."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("#")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_the_interval_narrowing_is_reported_with_its_own_reason(tmp_path: Path) -> None:
    """The reported-only prose must state the criterion that actually excluded the rows.

    Both control families render a "reported, not asserted on" table, and until P6-D21 the
    only narrowing in the project was the floored-cell one, so that reason was hard-coded
    for both. Printing it over interval rows would state a derivation that was explicitly
    **refused** for this statistic (P6-D12) instead of the one that was applied.
    """
    root = _write_results_dir(tmp_path / "results", interval_controls=_interval_controls())
    text = build_results_report(root, root / "results.md").read_text(encoding="utf-8")
    ids = {source.table_id for source in parse_table_sources(text)}
    assert {"controls_interval_asserted", "controls_interval_reported_only"} <= ids
    interval = _section_of(text, "### Interval controls: reported, not asserted on")
    assert "PICP@90" in interval and "[0.85, 0.95]" in interval
    assert "P6-D21" in interval
    # The floored-cell reason is the POINT control's, and it is not stated over these rows.
    assert "26 dB-suppressed" not in interval
    point = _section_of(text, "### Point controls: reported, not asserted on")
    assert "26 dB-suppressed" in point
    assert "PICP@90" not in point
    # And the excluded interval cell ships with its real excess rather than being dropped.
    excluded = _rendered_tables(text)["controls_interval_reported_only"]
    assert "0.2649" in "\n".join(excluded)
    assert "unseen_seastate" in "\n".join(excluded)


def test_a_group_with_no_asserted_cell_is_named_rather_than_silently_absent(
    tmp_path: Path,
) -> None:
    """The asserted summary is grouped over asserted rows, so a fully excluded group vanishes.

    On the production corpus P6-D21's narrowing excludes every `unseen_seastate` cell of the
    interval shuffle control, which would leave that regime absent from the verdict table
    with nothing saying why. An absence reads as "not run" or is not read at all; neither is
    what happened.
    """
    root = _write_results_dir(tmp_path / "results", interval_controls=_interval_controls())
    text = build_results_report(root, root / "results.md").read_text(encoding="utf-8")
    interval = _section_of(text, "### Interval controls: asserted")
    assert "no asserted cell at all" in interval
    assert "interval_shuffle/unseen_seastate" in interval
    # The group that WAS judged is not named, and the untrained control is judged everywhere.
    assert "interval_shuffle/id" not in interval
    assert "interval_untrained/unseen_seastate" not in interval
    # The same sentence is emitted for the point family when it applies -- this fixture
    # excludes `unseen_heading` there -- so the note follows the data rather than the label.
    point = _section_of(text, "### Point controls: asserted")
    assert "shuffle/unseen_heading" in point and "untrained/unseen_heading" in point
    assert "shuffle/id" not in point


def _pipeline_sanity() -> pd.DataFrame:
    """Return a frame shaped like a committed ``pipeline_sanity.csv``."""
    return pd.DataFrame(
        [
            {
                "control": "pipeline_sanity",
                "experiment": "e02_deep",
                "arm": "reference",
                "regime": regime,
                "dof": dof,
                "horizon_samples": samples,
                "horizon_s": seconds,
                "rmse_pipeline": 1.234567,
                "rmse_raw": 1.234567,
                "rel_diff": 3.21e-08,
                "max_rel_diff": 3.21e-08,
                "rtol": 1e-6,
                "n_windows": 441_984,
                "model_matches_inline": True,
                "asserted": True,
                "enforced": True,
                "passed": True,
            }
            for regime in ("id", "unseen_seastate")
            for dof in ("roll", "pitch")
            for samples, seconds in HORIZONS
        ]
    )


def test_the_lookback_ablation_renders_under_the_slug_the_gate_requires(
    report: tuple[str, Path],
) -> None:
    """Gate 6 predicate 5 names ``ablation_lookback``; nothing produced it until now."""
    from dmf.eval.gate import GATE6_REQUIRED_TABLES

    assert ("6.3", "ablation_lookback") in GATE6_REQUIRED_TABLES
    tables = _rendered_tables(report[0])
    assert "ablation_lookback" in tables
    assert "ablation_lookback contributed no row" not in report[0]


def test_the_pipeline_sanity_control_is_reported_with_its_measured_disagreement(
    tmp_path: Path,
) -> None:
    """The 'enforced and unreported' half of P6-D15, once it has a producer."""
    root = _write_results_dir(tmp_path / "results", pipeline_sanity=_pipeline_sanity())
    out = build_results_report(root, root / "results.md")
    text = out.read_text(encoding="utf-8")
    assert "enforced *and* now reported" in text
    # The measured number, not only the verdict: a run drifted to just inside the tolerance
    # passes and would be invisible in `passed`. Deliberately not the 5.006e-08 the prose
    # quotes from P2-D9, so that this asserts the rendered value rather than the paragraph.
    assert "3.210e-08" in text
    table = _rendered_tables(text)["pipeline_sanity"]
    assert "rel_diff" in _cells(table[0])
    assert "rmse_pipeline" in _cells(table[0]) and "rmse_raw" in _cells(table[0])
    assert set(_column(table, "enforced")) == {"yes"}


def test_a_pipeline_sanity_table_that_is_absent_is_called_a_gap(report: tuple[str, Path]) -> None:
    """Absent means "nobody wrote the number down", not "the control did not run"."""
    text = report[0]
    assert "pipeline_sanity.csv" in text
    assert "enforced and unreported" in text
    assert "pipeline_sanity" not in _rendered_tables(text)


# ---------------------------------------------------------------------------
# 8. The caveats that are not derived from any table.
# ---------------------------------------------------------------------------


def test_the_simulation_only_caveat_is_in_the_first_paragraph(report: tuple[str, Path]) -> None:
    head = report[0].split("\n\n")[1]
    assert "simulated results" in head.lower()


def test_the_document_states_the_reporting_rules_it_was_built_under(
    report: tuple[str, Path],
) -> None:
    text = report[0]
    for phrase in (
        "P6-D7",
        "P5-D15",
        "P3-D22",
        "no model is dropped",
    ):
        assert phrase.lower() in text.lower()


def test_no_retracted_claim_is_rendered(report: tuple[str, Path]) -> None:
    """Four claims this project retracted in its own protocol, pinned as absent.

    Each was rendered in a shipped document after the entry retracting it was written, which
    is the failure mode the whole caveat block exists against: a correction recorded in
    ``docs/protocol.md`` and not propagated to the artifact a reader actually reads.
    """
    text = report[0]
    retracted = (
        # P6-D23: the 0.218/0.437 pair is P1-D6's cross-mode pairing, which this arm does
        # not use. Measured same-mode, the denominators agree to about 1%.
        "roughly halves the 1 s persistence denominator",
        # P6-D19: scorability is a property of (sea state, heading, speed). SS3 at
        # `permissive` has 11.12 scorable onsets per realization at 45 deg / 12 kn.
        "so there is nothing to detect there",
        # P6-D18: conditioning costs -0.0445 mean skill on unseen_seastate, so the "bound"
        # lies below the baseline it was supposed to bound.
        "The sea-state-conditioning arm is an upper bound",
        # P6-D20: no row of the assembled table carries it.
        "+30.27",
    )
    for claim in retracted:
        assert claim not in text, f"{claim!r} was retracted in docs/protocol.md and is rendered"


def test_the_document_carries_the_corrections_that_replaced_them(
    report: tuple[str, Path],
) -> None:
    text = report[0]
    for phrase in (
        # P6-D23, quoting the measured ratio rather than hedging the arm away.
        "0.9872",
        "0.9963",
        # P3-D1: why the absolute numbers are flattered, which the README carried and this
        # document did not.
        "no process noise",
        "identifying",
        # P6-D19's corrected reading of scorability.
        "property of the cell",
        # P6-D18 / P6-D20.
        "privileged_information",
        "-0.0445",
    ):
        assert phrase in text, f"{phrase!r} is a correction this document must carry"


def test_both_scored_sections_state_that_they_cover_the_reference_arm_only(
    report: tuple[str, Path],
) -> None:
    """`DEFAULT_CONFIGS` is (e02_deep, e03_probabilistic): no ablation arm is scored here.

    The document explained why ``attitude_only`` cannot be scored on quiescence, which reads
    as though the other five were included. They were not, and the reason is cost rather
    than definition -- a different kind of gap, and one a reader cannot infer from a column.
    """
    text = report[0]
    point, rest = text.split("## 6.2", 1)
    quiescence = rest.split("## 6.3", 1)[0]
    for section in (point.split("## 6.1", 1)[1], quiescence):
        assert "DEFAULT_CONFIGS" in section
        assert "None of the six Phase 6 ablation arms is scored here" in section
    assert "No ablation arm is scored on the operational metric at all" in quiescence


def test_the_hard_coded_conservatism_factor_declares_that_it_has_no_producer(
    report: tuple[str, Path],
) -> None:
    """The one substantive number in the document that Gate 6 cannot trace to a CSV."""
    text = report[0]
    assert "2.8" in text and "6.3-7.1" in text
    assert "not read from a CSV" in text
    assert "no committed script" in text


def test_the_paired_intervals_declare_how_many_units_they_were_drawn_from(
    report: tuple[str, Path],
) -> None:
    """S4: every out-of-distribution interval in section 6.3 is a 12-cluster bootstrap.

    The count is derived from the rows rather than declared, so the sentence follows the
    corpus; what is pinned here is that it is rendered at all, and that it names the number
    a reader has to know before reading an out-of-distribution interval as evidence.
    """
    text = report[0]
    assert "Units behind each paired interval" in text
    # Derived from the fixture's own rows: `id` is a 48-cell grid and the regime that holds
    # out one level of one factor is a 12-cell one.
    assert "`id` 48" in text and "`unseen_heading` 12" in text
    assert "12 clusters" in text


def test_the_unit_note_is_derived_from_the_rows_and_not_written_down() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["id", "id", "unseen_seastate", "unseen_heading"],
            "ci_n_units": [48.0, 48.0, 12.0, 12.0],
        }
    )
    note = _resample_units_note(frame)
    assert "`id` 48" in note and "`unseen_seastate` 12" in note and "`unseen_heading` 12" in note
    # A table with no interval at all gets no sentence, rather than a sentence about zero.
    assert _resample_units_note(frame.drop(columns=["ci_n_units"])) == ""
    assert _resample_units_note(frame.assign(ci_n_units=np.nan)) == ""


# ---------------------------------------------------------------------------
# 9. The renderer reads the schema the scorer writes.
# ---------------------------------------------------------------------------


def _fixture_experiment(cfg: DataConfig) -> ExperimentConfig:
    """Return a closed-form-only experiment the fixture corpus can score without training."""
    return ExperimentConfig(
        name="e04_report_test",
        data=cfg,
        models=(
            ModelConfig(
                name="persistence", head="point", quantiles=(), params={}, label="persistence"
            ),
            ModelConfig(
                name="window_mean", head="point", quantiles=(), params={}, label="window_mean"
            ),
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=256,
            lr=1e-3,
            weight_decay=0.0,
            warmup_frac=0.0,
            grad_clip=1.0,
            patience=1,
            amp_dtype="off",
            num_workers=0,
        ),
        seeds=(0, 1, 2),
        regimes=("id",),
    )


def test_the_renderer_reads_what_the_scorer_writes(small_corpus: Path, tmp_path: Path) -> None:
    base = load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    cfg = replace(base, lookback=50, horizons=(5, 10, 20), stride=25)
    root = tmp_path / "results"
    score_experiment(
        _fixture_experiment(cfg),
        small_corpus,
        results_dir=root / PHASE6_SUBDIR,
        n_phase_realizations=2,
        batch_size=2048,
    )
    assert (root / PHASE6_SUBDIR / SCORING_ARTIFACTS["metrics_full"]).is_file()
    # strict=False because this fixture run produces no ablation or control table: the
    # point of the test is the schema agreement, and an incomplete document says so.
    text = build_results_report(root, root / "results.md", strict=False).read_text()
    sources = parse_table_sources(text)
    rendered = {source.table_id for source in sources}
    assert {"core_metrics", "quiescence_detection"} <= rendered
    for source in sources:
        assert source.csv_rows == len(pd.read_csv(root / source.path))
    # Every column sections 6.1 and 6.2 need is one the scorer actually wrote.
    header = _rendered_tables(text)["core_metrics"][0]
    for column in ("skill_mean", "nrmse_mean", "rmse_persistence", "phase_lag_s", "n_params"):
        assert column in header
    assert "base_rate_mean" in _rendered_tables(text)["quiescence_detection"][0]


# ---------------------------------------------------------------------------
# 10. `make eval`: the wrapper that has to exit 0 for Gate 6 to be reachable.
# ---------------------------------------------------------------------------


def _evaluate_module() -> object:
    """Import ``scripts/evaluate.py`` by path.

    ``scripts/`` holds argparse wrappers and is not an importable package, but the wiring
    it does -- which directory is scored, which is rendered, and that rendering does not
    retrain -- is exactly what Gate 6 predicate 1 turns on, so it is tested rather than
    assumed.

    Returns:
        The loaded module.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("dmf_scripts_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wrapper_keeps_the_flags_make_eval_passes() -> None:
    args = _evaluate_module().build_parser().parse_args(["--all-regimes"])  # type: ignore[attr-defined]
    assert args.all_regimes is True
    assert args.regime is None
    assert args.results_dir == Path("results")


def test_naming_a_regime_the_experiment_was_not_fitted_on_fails_early() -> None:
    module = _evaluate_module()
    cfg = _fixture_experiment(
        load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")
    )
    assert module.resolve_regimes(cfg, all_regimes=True, regime=None) == ("id",)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not fitted on regime"):
        module.resolve_regimes(cfg, all_regimes=False, regime="unseen_vessel")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not both"):
        module.resolve_regimes(cfg, all_regimes=True, regime="id")  # type: ignore[attr-defined]


def test_render_only_rebuilds_the_document_from_the_csvs_alone(tmp_path: Path) -> None:
    # No corpus, no checkpoints, no GPU: if the document is a pure function of the CSVs,
    # this succeeds, and if it is not, it cannot.
    _write_results_dir(tmp_path)
    module = _evaluate_module()
    code = module.main(["--render-only", "--results-dir", str(tmp_path)])  # type: ignore[attr-defined]
    assert code == 0
    out = tmp_path / "results.md"
    assert out.is_file()
    sources = parse_table_sources(out.read_text(encoding="utf-8"))
    assert sources
    assert all((tmp_path / source.path).is_file() for source in sources)
