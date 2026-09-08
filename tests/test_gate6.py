"""The Gate 6 read-out: seven pre-registered predicates over a rendered report.

Gates 3-5 turn on a number. Gate 6 turns on a document, so its tests are about parsing and
about what the verdict does when a predicate cannot be evaluated -- which is the case that
matters, because ``UNVERIFIED`` silently becoming ``PASS`` is how a traceability gate stops
meaning anything.

Every document here is synthetic. That is deliberate: ``docs/protocol.md`` P6-D1 registered
the predicates before the renderer existed, and a gate that can only be exercised against
the one real ``results.md`` is a gate nobody has watched fail.

The marker contract under test is :data:`dmf.eval.gate.GATE6_MARKER_SPEC`.
"""

from pathlib import Path

import pandas as pd
import pytest

from dmf.eval.gate import (
    GATE6_COLUMNS,
    GATE6_CRITERIA,
    GATE6_READING,
    GATE6_REQUIRED_TABLES,
    GATE6_RESULTS_MD,
    GATE6_TABLE_COLUMNS,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNVERIFIED,
    Gate6Evidence,
    build_gate6_markdown,
    gate6_notes,
    gate6_reading_passes,
    gate6_readout,
    gate6_table_audit,
    parse_rendered_tables,
    read_gate6_inputs,
    write_gate6_report,
)


def _marker(table_id: str, source: str, csv_rows: int, rows: int, select: str = "") -> str:
    """Render one provenance marker in the contract's exact form."""
    text = f"<!-- dmf-table id={table_id} source={source} csv_rows={csv_rows} rows={rows}"
    if select:
        text += f' select="{select}"'
    return text + " -->"


def _table(columns: list[str], n_rows: int) -> str:
    """Render a Markdown table with ``n_rows`` body rows and the given header."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for index in range(n_rows):
        lines.append("| " + " | ".join(f"v{index}" for _ in columns) + " |")
    return "\n".join(lines)


def _document(blocks: list[str], heading: str = "# Results") -> str:
    """Assemble a synthetic ``results.md``."""
    return "\n\n".join([heading, *blocks]) + "\n"


def _required_blocks(counts: dict[str, int]) -> list[str]:
    """Render one compliant table per table the plan requires."""
    blocks = []
    for _, table_id in GATE6_REQUIRED_TABLES:
        rows = counts.get(table_id, 3)
        blocks.append(
            _marker(table_id, f"{table_id}.csv", rows, rows)
            + "\n\n"
            + _table(["model", "regime", "value"], rows)
        )
    return blocks


def _evidence(
    text: str,
    counts: dict[str, int],
    *,
    tmp_path: Path,
    eval_exit_code: int | None = 0,
    rerender_matches: bool | None = True,
) -> Gate6Evidence:
    """Build an evidence set directly, so the read-out is exercised without touching disk."""
    return Gate6Evidence(
        results_dir=tmp_path,
        results_md_exists=True,
        tables=parse_rendered_tables(text),
        csv_row_counts=counts,
        eval_exit_code=eval_exit_code,
        corpus_present=True,
        checkpoints_present=True,
        rerender_matches=rerender_matches,
        rerender_error="" if rerender_matches else "differs",
    )


def _clean_counts() -> dict[str, int]:
    """Row counts matching the compliant document :func:`_required_blocks` renders."""
    return {f"{table_id}.csv": 3 for _, table_id in GATE6_REQUIRED_TABLES}


# ---------------------------------------------------------------------------------------
# The parser. Predicates 3, 4, 6 and 7 are only as good as the association between a marker
# and the table below it, so that association is pinned first.
# ---------------------------------------------------------------------------------------


def test_a_marker_attaches_to_the_table_it_precedes() -> None:
    text = _document(
        [_marker("core_metrics", "baselines.csv", 144, 12) + "\n\n" + _table(["a"], 12)]
    )
    tables = parse_rendered_tables(text)
    assert len(tables) == 1
    assert tables[0].table_id == "core_metrics"
    assert tables[0].source == "baselines.csv"
    assert tables[0].csv_rows_stated == 144
    assert tables[0].rows_stated == 12
    assert tables[0].rows_rendered == 12
    assert tables[0].columns == ("a",)
    assert not tables[0].filtered


def test_a_select_filter_is_parsed_and_marks_the_table_filtered() -> None:
    text = _document(
        [
            _marker("core_metrics", "baselines.csv", 144, 12, select="regime == id")
            + "\n\n"
            + _table(["a"], 12)
        ]
    )
    table = parse_rendered_tables(text)[0]
    assert table.filtered
    assert table.select == "regime == id"


def test_a_table_with_no_marker_is_reported_not_skipped() -> None:
    """An untraceable table is exactly what predicate 3 exists to catch."""
    text = _document(["Some prose.\n\n" + _table(["a", "b"], 2)])
    tables = parse_rendered_tables(text)
    assert len(tables) == 1
    assert tables[0].table_id == ""
    assert tables[0].marker_error


def test_prose_between_the_marker_and_the_table_breaks_the_association() -> None:
    """The marker must be the last non-blank line before the table, and only that.

    A marker several paragraphs up could be read as belonging to either of two tables, and a
    provenance claim whose subject is ambiguous is not a provenance claim.
    """
    text = _document(
        [
            _marker("core_metrics", "baselines.csv", 4, 4)
            + "\n\nA paragraph of commentary.\n\n"
            + _table(["a"], 4)
        ]
    )
    assert parse_rendered_tables(text)[0].table_id == ""


# ---------------------------------------------------------------------------------------
# The seven predicates.
# ---------------------------------------------------------------------------------------


def test_a_compliant_document_passes_every_predicate(tmp_path: Path) -> None:
    readout = gate6_readout(
        _evidence(_document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path)
    )
    assert tuple(readout.columns) == GATE6_COLUMNS
    assert list(readout["criterion"]) == [key for key, _ in GATE6_CRITERIA]
    assert set(readout["verdict"]) == {VERDICT_PASS}, readout[["criterion", "detail"]]
    assert gate6_reading_passes(readout)


def test_a_missing_eval_exit_code_is_unverified_and_not_a_pass(tmp_path: Path) -> None:
    """No artifact testifies that a command was run, so the gate refuses to assume it."""
    readout = gate6_readout(
        _evidence(
            _document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path, eval_exit_code=None
        )
    )
    assert readout.loc[readout["criterion"] == "1", "verdict"].item() == VERDICT_UNVERIFIED
    assert not gate6_reading_passes(readout)
    assert not gate6_reading_passes(readout, "1")
    assert gate6_reading_passes(readout, "3")


def test_a_nonzero_eval_exit_code_fails_predicate_one(tmp_path: Path) -> None:
    readout = gate6_readout(
        _evidence(
            _document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path, eval_exit_code=2
        )
    )
    assert readout.loc[readout["criterion"] == "1", "verdict"].item() == VERDICT_FAIL


def test_a_missing_precondition_is_unverified_rather_than_failed(tmp_path: Path) -> None:
    """`make eval` cannot be blamed for a checkout that has no corpus to evaluate."""
    evidence = _evidence(_document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path)
    stripped = Gate6Evidence(
        results_dir=evidence.results_dir,
        results_md_exists=True,
        tables=evidence.tables,
        csv_row_counts=evidence.csv_row_counts,
        eval_exit_code=0,
        corpus_present=False,
        checkpoints_present=True,
        rerender_matches=True,
    )
    readout = gate6_readout(stripped)
    assert readout.loc[readout["criterion"] == "1", "verdict"].item() == VERDICT_UNVERIFIED


def test_a_hand_edited_document_fails_predicate_two(tmp_path: Path) -> None:
    readout = gate6_readout(
        _evidence(
            _document(_required_blocks({})),
            _clean_counts(),
            tmp_path=tmp_path,
            rerender_matches=False,
        )
    )
    assert readout.loc[readout["criterion"] == "2", "verdict"].item() == VERDICT_FAIL


def test_an_unavailable_renderer_leaves_predicate_two_unverified(tmp_path: Path) -> None:
    readout = gate6_readout(
        _evidence(
            _document(_required_blocks({})),
            _clean_counts(),
            tmp_path=tmp_path,
            rerender_matches=None,
        )
    )
    assert readout.loc[readout["criterion"] == "2", "verdict"].item() == VERDICT_UNVERIFIED


def test_a_named_csv_that_does_not_exist_fails_predicate_three(tmp_path: Path) -> None:
    counts = _clean_counts()
    del counts["core_metrics.csv"]
    readout = gate6_readout(_evidence(_document(_required_blocks({})), counts, tmp_path=tmp_path))
    row = readout[readout["criterion"] == "3"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "does not exist" in row["detail"]


def test_a_document_with_no_tables_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """An empty document satisfies every table predicate trivially. It must not pass."""
    readout = gate6_readout(_evidence(_document(["Nothing here."]), {}, tmp_path=tmp_path))
    assert readout.loc[readout["criterion"] == "3", "verdict"].item() == VERDICT_FAIL
    assert not gate6_reading_passes(readout)


def test_a_stated_row_count_that_does_not_match_the_rendering_fails_predicate_four(
    tmp_path: Path,
) -> None:
    blocks = _required_blocks({})
    blocks[0] = _marker("core_metrics", "core_metrics.csv", 3, 7) + "\n\n" + _table(["a"], 3)
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    row = readout[readout["criterion"] == "4"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "body row" in row["detail"]


def test_a_stated_csv_row_count_that_does_not_match_the_file_fails_predicate_four(
    tmp_path: Path,
) -> None:
    counts = _clean_counts()
    counts["core_metrics.csv"] = 99
    readout = gate6_readout(_evidence(_document(_required_blocks({})), counts, tmp_path=tmp_path))
    row = readout[readout["criterion"] == "4"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "holds 99" in row["detail"]


def test_an_unfiltered_table_must_render_its_whole_csv(tmp_path: Path) -> None:
    """P6-D1's predicate 4 read literally, on the only tables it can be read literally on."""
    blocks = _required_blocks({})
    blocks[0] = _marker("core_metrics", "core_metrics.csv", 12, 3) + "\n\n" + _table(["a"], 3)
    counts = _clean_counts()
    counts["core_metrics.csv"] = 12
    readout = gate6_readout(_evidence(_document(blocks), counts, tmp_path=tmp_path))
    row = readout[readout["criterion"] == "4"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "literally" in row["detail"]


def test_the_same_table_passes_once_it_declares_its_filter(tmp_path: Path) -> None:
    """The filtered case, which the literal reading would fail on every honest document."""
    blocks = _required_blocks({})
    blocks[0] = (
        _marker("core_metrics", "core_metrics.csv", 12, 3, select="regime == id")
        + "\n\n"
        + _table(["a"], 3)
    )
    counts = _clean_counts()
    counts["core_metrics.csv"] = 12
    readout = gate6_readout(_evidence(_document(blocks), counts, tmp_path=tmp_path))
    assert readout.loc[readout["criterion"] == "4", "verdict"].item() == VERDICT_PASS


def test_a_missing_required_table_fails_predicate_five(tmp_path: Path) -> None:
    blocks = _required_blocks({})[1:]
    counts = _clean_counts()
    readout = gate6_readout(_evidence(_document(blocks), counts, tmp_path=tmp_path))
    row = readout[readout["criterion"] == "5"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "core_metrics" in row["detail"]


def test_an_f1_column_without_a_base_rate_fails_predicate_six(tmp_path: Path) -> None:
    """The named trap in CLAUDE.md: F1 against an unstated base rate says nothing."""
    blocks = _required_blocks({})
    blocks[1] = (
        _marker("quiescence_detection", "quiescence_detection.csv", 3, 3)
        + "\n\n"
        + _table(["model", "precision", "recall", "f1"], 3)
    )
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    row = readout[readout["criterion"] == "6"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "base-rate" in row["detail"]


def test_an_f1_column_with_a_base_rate_passes_predicate_six(tmp_path: Path) -> None:
    blocks = _required_blocks({})
    blocks[1] = (
        _marker("quiescence_detection", "quiescence_detection.csv", 3, 3)
        + "\n\n"
        + _table(["model", "precision", "recall", "f1", "base_rate"], 3)
    )
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    assert readout.loc[readout["criterion"] == "6", "verdict"].item() == VERDICT_PASS


def test_a_coverage_column_without_a_width_fails_predicate_seven(tmp_path: Path) -> None:
    """A maximally wide interval has perfect coverage, so coverage alone states nothing."""
    blocks = _required_blocks({})
    blocks[0] = (
        _marker("core_metrics", "core_metrics.csv", 3, 3)
        + "\n\n"
        + _table(["model", "horizon_s", "picp"], 3)
    )
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    row = readout[readout["criterion"] == "7"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "width" in row["detail"]


def test_a_coverage_table_that_pools_lead_times_fails_predicate_seven(tmp_path: Path) -> None:
    """P5-D15: the deep heads behave oppositely at 1-5 s and 10-15 s, so a pooled row lies."""
    blocks = _required_blocks({})
    blocks[0] = (
        _marker("core_metrics", "core_metrics.csv", 3, 3)
        + "\n\n"
        + _table(["model", "picp", "mean_interval_width"], 3)
    )
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    row = readout[readout["criterion"] == "7"].iloc[0]
    assert row["verdict"] == VERDICT_FAIL
    assert "pool" in row["detail"]


def test_a_per_horizon_coverage_table_with_a_width_passes_predicate_seven(tmp_path: Path) -> None:
    blocks = _required_blocks({})
    blocks[0] = (
        _marker("core_metrics", "core_metrics.csv", 3, 3)
        + "\n\n"
        + _table(["model", "horizon_s", "picp", "mean_interval_width"], 3)
    )
    readout = gate6_readout(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    assert readout.loc[readout["criterion"] == "7", "verdict"].item() == VERDICT_PASS


# ---------------------------------------------------------------------------------------
# The audit frame, the notes and the document.
# ---------------------------------------------------------------------------------------


def test_the_audit_frame_names_the_table_that_failed(tmp_path: Path) -> None:
    blocks = _required_blocks({})
    blocks[0] = _marker("core_metrics", "missing.csv", 3, 3) + "\n\n" + _table(["a"], 3)
    audit = gate6_table_audit(_evidence(_document(blocks), _clean_counts(), tmp_path=tmp_path))
    assert tuple(audit.columns) == GATE6_TABLE_COLUMNS
    failing = audit[audit["table_id"] == "core_metrics"].iloc[0]
    assert not failing["source_exists"]
    assert "[3]" in failing["problems"]


def test_the_notes_say_what_the_gate_does_not_cover(tmp_path: Path) -> None:
    """A PASS on a traceability gate must not read as a PASS on correctness."""
    evidence = _evidence(_document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path)
    notes = gate6_notes(gate6_readout(evidence), gate6_table_audit(evidence))
    joined = " ".join(notes)
    assert "reproducibility and traceability gate" in joined
    assert "interval controls" in joined
    assert "Simulated results only" in joined


def test_the_document_renders_and_names_the_marker_contract(tmp_path: Path) -> None:
    evidence = _evidence(_document(_required_blocks({})), _clean_counts(), tmp_path=tmp_path)
    readout = gate6_readout(evidence)
    text = build_gate6_markdown(readout, audit=gate6_table_audit(evidence), results_dir=tmp_path)
    assert "# Gate 6 read-out" in text
    assert "dmf-table" in text
    assert "PASS" in text


def test_an_empty_readout_is_refused() -> None:
    with pytest.raises(ValueError, match="empty Gate 6 read-out"):
        build_gate6_markdown(pd.DataFrame(columns=list(GATE6_COLUMNS)))


# ---------------------------------------------------------------------------------------
# The IO half.
# ---------------------------------------------------------------------------------------


def test_reading_a_directory_counts_the_rows_of_the_csvs_the_markers_name(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1, 2, 3, 4]}).to_csv(tmp_path / "core_metrics.csv", index=False)
    (tmp_path / GATE6_RESULTS_MD).write_text(
        _document([_marker("core_metrics", "core_metrics.csv", 4, 4) + "\n\n" + _table(["a"], 4)]),
        encoding="utf-8",
    )
    evidence = read_gate6_inputs(tmp_path, rerender=False)
    assert evidence.results_md_exists
    assert evidence.csv_row_counts == {"core_metrics.csv": 4}
    assert evidence.rerender_matches is None


def test_a_missing_document_fails_predicate_two_rather_than_raising(tmp_path: Path) -> None:
    evidence = read_gate6_inputs(tmp_path, eval_exit_code=0, rerender=False)
    readout = gate6_readout(evidence)
    assert readout.loc[readout["criterion"] == "2", "verdict"].item() == VERDICT_FAIL


def test_a_missing_results_directory_is_a_caller_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_gate6_inputs(tmp_path / "nowhere")


def test_write_gate6_report_writes_all_three_artifacts(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "core_metrics.csv", index=False)
    (tmp_path / GATE6_RESULTS_MD).write_text(
        _document([_marker("core_metrics", "core_metrics.csv", 3, 3) + "\n\n" + _table(["a"], 3)]),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    readout, csv_path, markdown_path = write_gate6_report(tmp_path, out, rerender=False)
    assert csv_path.exists()
    assert markdown_path.exists()
    assert (out / "gate6_tables.csv").exists()
    # Predicates 1 and 2 are unmeasured here, so the gate does not pass -- and predicate 5
    # fails, because one table is not the nine the plan requires.
    assert not gate6_reading_passes(readout, GATE6_READING)


def test_the_criteria_and_the_readout_cannot_drift() -> None:
    """Seven predicates in the constant, seven rows in the frame, one list."""
    assert len(GATE6_CRITERIA) == 7
    assert [key for key, _ in GATE6_CRITERIA] == [str(index) for index in range(1, 8)]
