"""The producer for ``interval_controls.csv`` and ``pipeline_sanity.csv``.

Both files are the subject of ``docs/protocol.md`` P6-D15's closing paragraphs: two interval
controls implemented, tested and **never called**, and one pipeline-sanity control enforced
and **never reported**. What is pinned here is that the driver closes both gaps without
changing what either control means.

1. Both tables are written, on their own schemas. An interval row's statistic is a ratio of
   Winkler scores and a point row's is a ratio of squared errors; they never share a file.
2. ``enforced`` is the ``strict`` argument at the call site and nothing else -- True for the
   interval shuffle control and for pipeline sanity, False for the interval untrained
   control (P3-D9) -- and ``tests/test_assemble.py`` checks the same fact against
   :data:`dmf.eval.assemble.ENFORCED_BY_CONTROL` by parsing this driver's source.
3. **No interval row states a coverage without a width beside it.** Coverage without
   sharpness is meaningless: a maximally wide interval has perfect coverage.
4. The provenance columns are never empty. An empty ``arm`` round-trips through CSV as NaN
   and pandas drops NaN group keys, which once deleted 288 rows from a rendered summary
   while leaving them in the file (P6-D15 defect 4).
5. A control that could not run is recorded with its reason, never absent.
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from dmf.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from dmf.eval.assemble import NOT_AN_ARM
from dmf.eval.control_runner import (
    CONTROL_ARTIFACTS,
    PIPELINE_SANITY_RTOL,
    ControlArtifacts,
    run_controls,
)
from dmf.eval.controls import (
    INTERVAL_CONTROL_COLUMNS,
    INTERVAL_NULL_PICP_BAND,
    PIPELINE_SANITY_COLUMNS,
)

#: Horizons of the small-corpus geometry, samples.
HORIZONS: tuple[int, ...] = (5, 10, 20)

#: The nine levels every quantile row in this project is scored at.
QUANTILES: tuple[float, ...] = (
    0.05,
    0.1625,
    0.275,
    0.3875,
    0.5,
    0.6125,
    0.725,
    0.8375,
    0.95,
)


def _train_cfg() -> TrainConfig:
    """Return a training block nothing in this module reads: no row here is trained."""
    return TrainConfig(
        epochs=1,
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        warmup_frac=0.0,
        grad_clip=1.0,
        patience=1,
        amp_dtype="off",
        num_workers=0,
    )


def _data_cfg() -> DataConfig:
    """Return the production channel selection at the small corpus's window geometry."""
    base = load_default()
    return replace(base, lookback=50, horizons=HORIZONS, stride=25)


def load_default() -> DataConfig:
    """Return ``configs/data/default.yaml``."""
    from dmf.config import load_data

    return load_data(Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml")


def _experiment(*, with_head: bool = True, with_point_solver: bool = True) -> ExperimentConfig:
    """Return an experiment of closed-form and untrained models only.

    Args:
        with_head: Whether to configure an interval head, i.e. a subject for the untrained
            interval control.
        with_point_solver: Whether to configure ``dlinear_ols``, which is where the
            unconditional null's kernel and ridge come from.
    """
    models = [
        ModelConfig(name="persistence", head="point", quantiles=(), params={}, label="persistence"),
    ]
    if with_point_solver:
        models.append(
            ModelConfig(
                name="dlinear_ols",
                head="point",
                quantiles=(),
                params={"kernel_size": 5, "ridge": 1e-6},
                label="dlinear_ols",
            )
        )
    if with_head:
        models.append(
            ModelConfig(
                name="dlinear",
                head="quantile",
                quantiles=QUANTILES,
                params={"kernel_size": 5, "individual": False},
                label="dlinear_quantile",
            )
        )
    return ExperimentConfig(
        name="e04_controls_test",
        data=_data_cfg(),
        models=tuple(models),
        train=_train_cfg(),
        seeds=(0, 1, 2),
        regimes=("id",),
    )


@pytest.fixture(scope="module")
def controlled(
    small_corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ControlArtifacts, Path]:
    """Run every control once, into a temporary results directory."""
    out = tmp_path_factory.mktemp("control_results")
    artifacts = run_controls(
        [_experiment()], small_corpus, results_dir=out, arms=("reference",), batch_size=512
    )
    return artifacts, out


def test_both_control_tables_are_written(controlled: tuple[ControlArtifacts, Path]) -> None:
    artifacts, out = controlled
    assert set(artifacts.paths) == set(CONTROL_ARTIFACTS)
    for name, filename in CONTROL_ARTIFACTS.items():
        assert artifacts.paths[name] == out / filename
        assert len(pd.read_csv(out / filename)) > 0


def test_the_interval_table_keeps_its_own_schema(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    artifacts, _ = controlled
    frame = artifacts.interval_controls
    assert list(frame.columns) == ["experiment", "arm", *INTERVAL_CONTROL_COLUMNS]
    assert set(frame["control"]) == {"interval_shuffle", "interval_untrained"}
    # The point controls' statistic is not here and must never be: a ratio of Winkler scores
    # and a ratio of squared errors do not share a column name.
    assert "skill_subject" not in frame.columns


def test_enforcement_is_the_strict_argument_and_differs_by_control(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    artifacts, _ = controlled
    frame = artifacts.interval_controls
    shuffle = frame[frame["control"] == "interval_shuffle"]
    untrained = frame[frame["control"] == "interval_untrained"]
    assert bool(shuffle["enforced"].all())
    assert not bool(untrained["enforced"].any())
    # The untrained control is asserted on every cell: neither the P6-D11 floored-cell
    # narrowing (P6-D12) nor the P6-D21 null-calibration one is extended to it.
    assert bool(untrained["asserted"].all())
    # The shuffle control's `asserted` is exactly the band predicate on the NULL's own
    # coverage (P6-D21) -- recomputed here from the committed column rather than trusted, so
    # a narrowing that started keying off anything else would fail this test. On this fixture
    # the null under-covers on some cells, which is why the two values differ at all.
    low, high = INTERVAL_NULL_PICP_BAND
    in_band = (shuffle["picp_null"] >= low) & (shuffle["picp_null"] <= high)
    assert (shuffle["asserted"] == in_band).all()
    # And the excluded rows are present with their real numbers, not dropped.
    assert shuffle.loc[~shuffle["asserted"], "excess"].notna().all()
    assert "on_residual_floor" in frame.columns


def test_the_shuffle_control_passed_and_would_have_stopped_the_run(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    """It is enforced, so a failure raises -- reaching this point is itself the evidence."""
    artifacts, _ = controlled
    shuffle = artifacts.interval_controls
    shuffle = shuffle[shuffle["control"] == "interval_shuffle"]
    assert bool(shuffle["passed"].all())
    assert (shuffle["excess"] <= shuffle["tol"]).all()


def test_no_interval_row_states_a_coverage_without_a_width(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    artifacts, _ = controlled
    frame = artifacts.interval_controls
    for column in ("picp_subject", "picp_null", "width_subject", "width_null"):
        assert column in frame.columns
        assert frame[column].notna().all()


def test_the_pipeline_sanity_table_reports_what_the_assertion_measured(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    artifacts, _ = controlled
    frame = artifacts.pipeline_sanity
    assert list(frame.columns) == list(PIPELINE_SANITY_COLUMNS)
    assert set(frame["control"]) == {"pipeline_sanity"}
    assert bool(frame["enforced"].all()) and bool(frame["asserted"].all())
    assert bool(frame["passed"].all())
    # The value of the table is the measured disagreement, which should sit at the float32
    # storage floor rather than just inside the tolerance -- 5.006e-08 on the production
    # id/test partition (P2-D9), and ~4e-07 on this fixture, whose windows are far fewer and
    # whose amplitudes are far smaller. A run that had drifted to just inside the tolerance
    # would be visible in this column and invisible in `passed`, which is why the column is
    # written at all.
    assert (frame["rel_diff"] <= frame["rtol"]).all()
    assert float(frame["max_rel_diff"].max()) < PIPELINE_SANITY_RTOL
    # The persistence model was scored, not the inline expression alone, and it matched it
    # bitwise -- which is what makes the control a statement about every skill denominator.
    assert bool(frame["model_matches_inline"].all())
    assert (frame["n_windows"] > 0).all()
    assert set(frame["horizon_samples"]) == set(HORIZONS)


def test_every_row_says_which_experiment_and_arm_produced_it(
    controlled: tuple[ControlArtifacts, Path],
) -> None:
    artifacts, _ = controlled
    for frame in (artifacts.interval_controls, artifacts.pipeline_sanity):
        for column in ("experiment", "arm"):
            values = frame[column].astype(str)
            assert values.notna().all()
            assert (values.str.strip() != "").all()
            assert values.nunique() == 1


def test_a_config_with_no_closed_form_point_row_records_why_it_has_no_interval_control(
    small_corpus: Path, tmp_path: Path
) -> None:
    artifacts = run_controls(
        # A head, so there IS an interval claim to audit, and no closed-form point row, so
        # the unconditional null cannot be solved at the kernel and ridge the run's own point
        # forecast uses.
        [_experiment(with_point_solver=False, with_head=True)],
        small_corpus,
        results_dir=tmp_path,
        batch_size=512,
    )
    assert artifacts.interval_controls.empty
    assert not (tmp_path / CONTROL_ARTIFACTS["interval_controls"]).exists()
    reason = artifacts.skipped["e04_controls_test/id/interval"]
    assert "residual_interval" in reason and "dlinear_ols" in reason
    # The pipeline-sanity control still ran: every run has one.
    assert not artifacts.pipeline_sanity.empty


def test_an_experiment_that_publishes_no_interval_gets_no_interval_control(
    small_corpus: Path, tmp_path: Path
) -> None:
    """An interval control on a run with no interval claim audits nothing.

    ``e02_deep`` is that run: every model in it has a point head. Running the control there
    anyway would re-measure the same unconditional floor under a second experiment label,
    which reads as two measurements of one thing.
    """
    artifacts = run_controls(
        [_experiment(with_head=False)], small_corpus, results_dir=tmp_path, batch_size=512
    )
    assert artifacts.interval_controls.empty
    assert "publishes no interval" in artifacts.skipped["e04_controls_test/id/interval"]
    assert not artifacts.pipeline_sanity.empty


def test_a_closed_form_interval_gets_the_shuffle_control_but_no_untrained_one(
    small_corpus: Path, tmp_path: Path
) -> None:
    """An unfitted closed-form solve is not a randomly initialised model.

    An unfitted ``EmpiricalResidualInterval`` is an empty object rather than an untrained
    model -- it raises instead of emitting the zero-width fan that would score as perfect
    sharpness -- so the untrained control is skipped with that reason while the shuffle
    control, which needs no random initialisation, still runs.
    """
    cfg = _experiment(with_head=False)
    floor = ModelConfig(
        name="residual_interval",
        head="quantile",
        quantiles=QUANTILES,
        params={"kernel_size": 5, "ridge": 1e-6},
        label="residual_interval",
    )
    cfg = replace(cfg, models=(*cfg.models, floor))
    artifacts = run_controls([cfg], small_corpus, results_dir=tmp_path, batch_size=512)
    assert set(artifacts.interval_controls["control"]) == {"interval_shuffle"}
    reason = artifacts.skipped["e04_controls_test/id/interval_untrained"]
    assert "no SGD interval head" in reason


def test_a_regime_the_config_was_not_fitted_on_is_skipped_with_its_reason(
    small_corpus: Path, tmp_path: Path
) -> None:
    artifacts = run_controls(
        [_experiment()],
        small_corpus,
        results_dir=tmp_path,
        regimes=("unseen_vessel",),
        batch_size=512,
    )
    assert artifacts.skipped["e04_controls_test/unseen_vessel"].startswith("experiment")
    assert artifacts.interval_controls.empty


def test_the_default_arm_label_is_a_non_empty_sentinel() -> None:
    assert NOT_AN_ARM and NOT_AN_ARM.strip() == NOT_AN_ARM


def test_a_run_with_no_configs_is_refused(small_corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not audited"):
        run_controls([], small_corpus, results_dir=tmp_path)


def test_the_two_tables_reach_the_rendered_document(
    controlled: tuple[ControlArtifacts, Path], tmp_path: Path
) -> None:
    """The half of the gap that code alone does not close: the files must also render.

    A control with a producer and no place in the document is the same defect as a control
    with no producer, one step further down the pipeline.
    """
    from dmf.eval.report import PHASE6_SUBDIR, build_results_report

    artifacts, _ = controlled
    phase6 = tmp_path / "results" / PHASE6_SUBDIR
    phase6.mkdir(parents=True)
    for name, filename in CONTROL_ARTIFACTS.items():
        getattr(artifacts, name).to_csv(phase6 / filename, index=False)
    # Explicitly incomplete: this fixture has no scored tables, and the point here is only
    # that the two control tables are read, rendered and marked.
    text = build_results_report(
        tmp_path / "results", tmp_path / "results" / "results.md", strict=False
    ).read_text(encoding="utf-8")
    assert "interval_controls.csv" in text and "pipeline_sanity.csv" in text
    for marker in ("controls_interval_asserted", "pipeline_sanity"):
        assert f"id={marker} " in text
    assert "No interval controls table" not in text
