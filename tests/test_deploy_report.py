"""The Phase 7 document renderer and the Gate 7 read-out.

These two modules generate or judge **every** claim in `results/latency.md`, the README's
latency section and `results/gate7.md`, and until this file existed neither had a single
test. That is not a hypothetical gap: the first version of the gate's "no generic speedup
multipliers" check formed ratios over every p50 value in the file and accepted essentially
any number, so a README citing a vendor's multiple would have passed. A test with the
fabricated string in it is what closes that, and
`test_a_fabricated_vendor_multiplier_is_rejected` is that test.

What is asserted here:

- the gate's millisecond check rejects a figure that is in no committed table, and accepts
  one that is;
- the gate's multiplier check rejects a fabricated multiple and accepts a real
  same-model, same-batch backend ratio;
- the check reports how permissive it is, and is not fully permissive;
- the methodology tokens are required, so a section that quotes numbers without saying how
  they were taken cannot pass;
- the renderer's comparative claims -- best-tail-in-both-runs, the fastest configuration
  per model -- are computed from the tables rather than written down, and are asserted on
  synthetic tables whose answer is known by construction;
- every rendered table's provenance marker declares the row count of the file it came from.

The fixtures are synthetic and hermetic: no committed artifact, no GPU, no corpus.
"""

from pathlib import Path

import pandas as pd
import pytest

from dmf.deploy.bench import BenchResult
from dmf.deploy.gate import (
    GATE7_METHODOLOGY_TOKENS,
    Gate7Evidence,
    _readme_verdict,
    gate7_passes,
    gate7_readout,
    multiplier_acceptance,
)
from dmf.deploy.harness import BenchJob
from dmf.deploy.parity import ParityResult
from dmf.deploy.pipeline import LatencyRow, PipelineResult, unverified_timed_configurations
from dmf.deploy.report import _agree_count, _argmin_label, build_latency_report, gate_cell_skill
from dmf.eval.gate import VERDICT_FAIL, VERDICT_PASS

#: A minimal latency table: two models, two backends, one batch size. `fast/ort-cpu` is
#: exactly half `fast/ort-cuda`, so 2.0x is a real same-model ratio and 2.5x is not.
LATENCY = pd.DataFrame(
    {
        "model": ["fast", "fast", "slow", "slow"],
        "backend": ["ort-cpu", "ort-cuda", "ort-cpu", "ort-cuda"],
        "device": ["cpu", "cuda", "cpu", "cuda"],
        "batch_size": [1, 1, 1, 1],
        "providers_realized": [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ],
        "intra_op_threads": [1, 1, 1, 1],
        "tf32": [False, False, False, False],
        "repeat": [1, 1, 1, 1],
        "p50_ms": [1.000, 2.000, 4.000, 8.000],
        "p90_ms": [1.100, 2.200, 4.400, 8.800],
        "p99_ms": [1.200, 2.400, 4.800, 9.600],
        "mean_ms": [1.050, 2.100, 4.200, 8.400],
        "std_ms": [0.010, 0.020, 0.040, 0.080],
        "throughput_windows_s": [1000.0, 500.0, 250.0, 125.0],
        "peak_host_mem_mb": [800.0, 900.0, 800.0, 900.0],
        "peak_device_mem_mb": [0.0, 70.0, 0.0, 70.0],
        "warmup_iters": [200] * 4,
        "timed_iters": [2000] * 4,
        "gpu": ["TEST GPU"] * 4,
        "torch": ["9.9.9"] * 4,
        "onnxruntime": ["9.9.9"] * 4,
        "git_commit": ["deadbee"] * 4,
    }
)

#: A methodology paragraph that satisfies every required token, so the token check is not
#: what any multiplier test is actually measuring.
METHODOLOGY = (
    "## Inference latency\n"
    "200 warmup then 2000 timed iterations, p50/p90/p99 reported, "
    "torch.cuda.synchronize() around every timed GPU region, 1 intra-op thread pinned, "
    "TF32 off, simulated corpus.\n"
)


def _evidence(section_body: str, parity_passed: bool = True) -> Gate7Evidence:
    """Assemble gate evidence around one README section body.

    Args:
        section_body: Text appended to the methodology paragraph.
        parity_passed: Whether the synthetic parity table passes.

    Returns:
        The evidence object.
    """
    # One row per (model, provider) that the latency table benchmarks, which is what
    # clause 1 cross-checks against.
    parity = pd.DataFrame(
        {
            "model": ["fast", "fast", "slow", "slow"],
            "provider": [
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
                "CUDAExecutionProvider",
            ],
            "n_seeds": [5] * 4,
            "margin_x": [10.0] * 4,
            "passed": [parity_passed, True, True, True],
            "passed_absolute": [True] * 4,
        }
    )
    return Gate7Evidence(
        parity=parity,
        latency=LATENCY,
        latency_extra=(),
        pareto_exists=True,
        readme_section=METHODOLOGY + section_body,
        results_dir=Path("results"),
    )


def test_a_fabricated_vendor_multiplier_is_rejected() -> None:
    """The string that passed the first version of this check must now fail it.

    2.5x is not a ratio of any two p50 values sharing a model and a batch size in the table
    above; 2.0x is. The earlier version pooled ratios across models as well and accepted
    both -- and everything else in [0.5, 50].
    """
    verdict, detail = _readme_verdict(
        _evidence("NVIDIA reports up to 2.5x speedup with TensorRT.\n")
    )
    assert verdict == VERDICT_FAIL
    assert "2.5" in detail


def test_a_real_same_model_backend_ratio_is_accepted() -> None:
    """A multiplier this machine actually produced passes."""
    verdict, _ = _readme_verdict(
        _evidence("ORT CUDA is 2.0x slower than ORT CPU on `fast` at batch 1.\n")
    )
    assert verdict == VERDICT_PASS


def test_a_cross_model_ratio_is_not_a_backend_claim_and_is_rejected() -> None:
    """A cross-model ratio is not a statement about a backend, and is rejected.

    `slow/ort-cpu` over `fast/ort-cpu` is exactly 4.0x, and pooling ratios like it with the
    backend comparisons is what made the first version of this check vacuous.
    """
    verdict, _ = _readme_verdict(_evidence("The GPU gives a 4.0x speedup.\n"))
    assert verdict == VERDICT_FAIL


def test_a_millisecond_figure_from_nowhere_is_rejected() -> None:
    """A quoted latency must appear in a committed table, to the printed precision."""
    assert _readme_verdict(_evidence("It runs in 1.000 ms.\n"))[0] == VERDICT_PASS
    assert _readme_verdict(_evidence("It runs in 1.234 ms.\n"))[0] == VERDICT_FAIL


@pytest.mark.parametrize("token", [token for token, _ in GATE7_METHODOLOGY_TOKENS])
def test_every_methodology_token_is_required(token: str) -> None:
    """Dropping any one of the six required statements fails clause 3.

    Numbers without their methodology are decoration: a p50 taken without warmup, without
    synchronisation or at an unstated thread count is a different quantity with the same
    name.
    """
    crippled = METHODOLOGY.replace(token, "xxxx")
    evidence = Gate7Evidence(
        parity=_evidence("").parity,
        latency=LATENCY,
        latency_extra=(),
        pareto_exists=True,
        readme_section=crippled + "It runs in 1.000 ms.\n",
        results_dir=Path("results"),
    )
    verdict, detail = _readme_verdict(evidence)
    assert verdict == VERDICT_FAIL
    assert token in detail


def test_the_multiplier_check_reports_how_permissive_it_is() -> None:
    """The acceptance rate is measured, not assumed, and it is not everything.

    A check that accepts any plausible number is not evidence. Reporting the fraction it
    would accept is what keeps that visible in the read-out instead of resting on the fact
    that a check exists.
    """
    ratios = {2.0, 0.5}
    acceptance = multiplier_acceptance(ratios)
    assert 0.0 < acceptance < 0.05
    assert multiplier_acceptance(set()) == 0.0


def test_a_failing_parity_row_for_a_benchmarked_configuration_fails_the_gate() -> None:
    """Clause 1 is not advisory: a timed configuration whose arithmetic failed sinks it."""
    readout = gate7_readout(_evidence("It runs in 1.000 ms.\n", parity_passed=False))
    assert not gate7_passes(readout)
    assert readout.loc[readout["criterion"] == "1", "verdict"].iloc[0] == VERDICT_FAIL


def test_a_benchmarked_provider_with_no_parity_row_at_all_fails_the_gate() -> None:
    """The exact shape of the TF32 defect: GPU rows timed, only the CPU provider checked.

    A parity table that certifies one provider certifies one provider. Before this check
    existed, a sweep could -- and did -- ship CUDA and TensorRT latency rows behind a
    CPU-only parity check, with every column of the output looking correct.
    """
    evidence = _evidence("It runs in 1.000 ms.\n")
    cpu_only = evidence.parity
    assert cpu_only is not None
    stripped = cpu_only[cpu_only["provider"] == "CPUExecutionProvider"]
    readout = gate7_readout(
        Gate7Evidence(
            parity=stripped,
            latency=evidence.latency,
            latency_extra=(),
            pareto_exists=True,
            readme_section=evidence.readme_section,
            results_dir=evidence.results_dir,
        )
    )
    assert readout.loc[readout["criterion"] == "1", "verdict"].iloc[0] == VERDICT_FAIL
    detail = str(readout.loc[readout["criterion"] == "1", "detail"].iloc[0])
    assert "ort-cuda" in detail


def test_a_parity_failure_on_a_configuration_that_was_not_timed_does_not_fail_the_gate() -> None:
    """Excluding a failing configuration is the gate working, not the gate failing.

    `lstm_quantile` on PyTorch CUDA fails parity on the reference machine and is therefore
    not benchmarked; the clause has to distinguish "we refused to time it" from "we timed
    something we never verified".
    """
    evidence = _evidence("It runs in 1.000 ms.\n")
    parity = evidence.parity
    assert parity is not None
    extra = pd.concat(
        [
            parity,
            pd.DataFrame(
                {
                    "model": ["fast"],
                    "provider": ["torch:cuda"],
                    "n_seeds": [5],
                    "margin_x": [0.2],
                    "passed": [False],
                    "passed_absolute": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    readout = gate7_readout(
        Gate7Evidence(
            parity=extra,
            latency=LATENCY,  # carries no torch rows at all
            latency_extra=(),
            pareto_exists=True,
            readme_section=evidence.readme_section,
            results_dir=evidence.results_dir,
        )
    )
    assert readout.loc[readout["criterion"] == "1", "verdict"].iloc[0] == VERDICT_PASS
    assert "fast/torch:cuda" in str(readout.loc[readout["criterion"] == "1", "detail"].iloc[0])


# ---------------------------------------------------------------------------------------
# The renderer's comparative claims
# ---------------------------------------------------------------------------------------

#: A stability table where the best tail changes between runs for `flip` and does not for
#: `firm`. The renderer must report one of those as a both-runs claim and not the other.
STABILITY = pd.DataFrame(
    {
        "model": ["firm", "firm", "flip", "flip"],
        "backend": ["ort-cpu", "torch-eager", "ort-cpu", "torch-eager"],
        "device": ["cpu", "cuda", "cpu", "cuda"],
        "batch_size": [1, 1, 1, 1],
        "intra_op_threads": [1, 1, 1, 1],
        "p50_ms_run1": [1.0, 2.0, 1.0, 2.0],
        "p50_ms_run2": [1.0, 2.0, 1.0, 2.0],
        "drift_frac": [0.0, 0.0, 0.0, 0.0],
        "drift_pct": [0.0, 0.0, 0.0, 0.0],
        "within_10pct": [True, True, True, True],
        "p99_ms_run1": [1.0, 2.0, 1.0, 2.0],
        "p99_ms_run2": [1.0, 2.0, 3.0, 2.0],
        "p99_drift_pct": [0.0, 0.0, 200.0, 0.0],
        "p99_within_10pct": [True, True, False, True],
        "git_commit": ["deadbee"] * 4,
    }
)


def test_the_best_tail_claim_counts_only_what_holds_in_both_runs() -> None:
    """`firm` keeps its best-tail backend across repeats; `flip` does not.

    This is the guard on the sentence the deployment argument leans on. Asserted on a table
    built so the answer is known: a claim read off run 1 alone would count both.
    """
    assert _agree_count(STABILITY, ["firm", "flip"], want="ort-cpu") == 1
    assert _argmin_label(STABILITY, "firm", "p99_ms_run2").startswith("ort-cpu")
    assert _argmin_label(STABILITY, "flip", "p99_ms_run2").startswith("torch-eager")
    assert _argmin_label(STABILITY, "flip", "p99_ms_run1").startswith("ort-cpu")


def test_argmin_can_be_restricted_to_the_onnx_runtime_backends() -> None:
    """The ORT-internal claim and the whole-field claim are different claims."""
    assert _argmin_label(STABILITY, "flip", "p99_ms_run2", ort_only=True).startswith("ort-cpu")
    assert _argmin_label(STABILITY, "absent", "p99_ms_run2") == "n/a"


def test_gate_cell_skill_reads_the_preregistered_cell_and_refuses_an_empty_one(
    tmp_path: Path,
) -> None:
    """Accuracy comes from pitch at 10 s on `id`, averaged over seeds, or it raises.

    Silently averaging the wrong cell -- or an empty selection producing NaN -- is how a
    Pareto plot comes to show an accuracy axis nobody chose.
    """
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "a", "b"],
            "regime": ["id", "id", "unseen_heading", "id"],
            "dof": ["pitch", "pitch", "pitch", "roll"],
            "horizon_samples": [100, 100, 100, 100],
            "seed": [0, 1, 0, 0],
            "skill": [0.80, 0.90, 0.10, 0.50],
        }
    )
    path = tmp_path / "metrics_full.csv"
    frame.to_csv(path, index=False)
    result = gate_cell_skill(path)
    assert set(result["model"]) == {"a"}
    assert float(result.loc[result["model"] == "a", "skill_mean"].iloc[0]) == pytest.approx(0.85)
    assert int(result.loc[result["model"] == "a", "n_seeds"].iloc[0]) == 2

    empty = tmp_path / "empty.csv"
    frame[frame["regime"] == "nowhere"].to_csv(empty, index=False)
    with pytest.raises(ValueError, match="gate cell"):
        gate_cell_skill(empty)


def test_every_rendered_table_declares_the_row_count_of_its_source(tmp_path: Path) -> None:
    """The provenance marker's `csv_rows` must be the file's, and `rows` the table's.

    A marker that claimed a row count it did not have would make the whole traceability
    contract decorative, and the document is long enough that nobody would notice by eye.
    """
    import re

    (tmp_path / "parity.csv").write_text(
        "model,provider,tf32,head_kind,n_windows,n_seeds,worst_seed,max_abs_err,"
        "max_abs_err_best_seed,mean_abs_err,output_abs_max,scale,tolerance,scaled_tolerance,"
        "margin_x,passed,passed_absolute,onnx_path,gpu,torch,onnxruntime,git_commit\n"
        "fast,CPUExecutionProvider,False,point,1000,5,0,1e-05,1e-05,1e-06,1.0,1.0,0.0001,"
        "0.0001,10.0,True,True,a.onnx,TEST GPU,9.9.9,9.9.9,deadbee\n",
        encoding="utf-8",
    )
    # The renderer needs a batch-32 section too; the values are irrelevant to the markers.
    both_batches = pd.concat([LATENCY, LATENCY.assign(batch_size=32)], ignore_index=True)
    both_batches.to_csv(tmp_path / "latency.csv", index=False)
    LATENCY.assign(intra_op_threads=4).to_csv(tmp_path / "latency_threads.csv", index=False)
    STABILITY.to_csv(tmp_path / "latency_stability.csv", index=False)

    document = build_latency_report(tmp_path)
    markers = re.findall(r"<!-- dmf-table ([^>]*)-->", document)
    assert len(markers) == 5
    fields = [
        dict(part.split("=", 1) for part in marker.split() if "=" in part) for marker in markers
    ]
    by_source = {field["source"]: field for field in fields}
    assert by_source["latency.csv"]["csv_rows"] == str(2 * len(LATENCY))
    assert by_source["latency_stability.csv"]["csv_rows"] == str(len(STABILITY))
    # The batch-1 table is a subset and must say by which predicate; the thread sweep is the
    # whole file and must not carry a `select` at all, which is the stronger claim.
    assert 'select="batch_size == 1"' in document
    assert by_source["latency_threads.csv"].get("select") is None


def _bench_result(job: BenchJob) -> BenchResult:
    """A measured row with placeholder timings: the invariant reads only its identity."""
    return BenchResult(
        label=f"{job.target_key}-{job.backend}-b{job.batch_size}",
        p50_ms=1.0,
        p90_ms=1.0,
        p99_ms=1.0,
        mean_ms=1.0,
        std_ms=0.0,
        throughput_windows_s=1000.0,
        peak_host_mem_mb=0.0,
        peak_device_mem_mb=0.0,
        providers_realized=("CPUExecutionProvider",),
        intra_op_threads=1,
        tf32=False,
        config=job.bench_config(),
        env={},
    )


def _parity(provider: str, *, passed: bool) -> ParityResult:
    """One parity verdict for one provider, with the fields the invariant reads."""
    return ParityResult(
        max_abs_err=1e-5 if passed else 1e-2,
        mean_abs_err=1e-6,
        n_windows=8,
        tolerance=1e-4,
        passed=passed,
        provider=provider,
    )


def _latency_row(target: str, backend: str, device: str) -> LatencyRow:
    """One timed row, carrying only what the invariant reads off it."""
    job = BenchJob(target_key=target, backend=backend, device=device)
    return LatencyRow(job=job, result=_bench_result(job), repeat=1)


def _pipeline(
    parity: dict[str, tuple[ParityResult, ...]], rows: tuple[LatencyRow, ...]
) -> PipelineResult:
    """A pipeline result carrying just a parity table and a set of timed rows."""
    return PipelineResult(
        exports=(),
        parity=parity,
        attributions={},
        latency=rows,
        failures=(),
        parity_gate_passed=all(r.passed for draws in parity.values() for r in draws),
    )


def test_timing_a_configuration_whose_parity_failed_is_reported_as_the_invariant_breaking(
    tmp_path: Path,
) -> None:
    """The one thing the stage ordering exists to prevent must be assertable, not argued.

    `lstm_quantile` on torch:cuda fails parity permanently (P7-D10). If a row for it ever
    reaches the latency table, every downstream number is a timing of something unverified,
    and `scripts/benchmark.py` must exit non-zero rather than publish it.
    """
    outcome = _pipeline(
        {"lstm_quantile": (_parity("torch:cuda", passed=False),)},
        (_latency_row("lstm_quantile", "torch-eager", "cuda"),),
    )
    assert unverified_timed_configurations(outcome, tmp_path) == {"lstm_quantile/torch-eager/cuda"}


def test_a_refused_configuration_that_was_correctly_excluded_breaks_nothing(
    tmp_path: Path,
) -> None:
    """The documented refusal is an outcome, not an error (P7-D11).

    A parity failure whose configuration was kept out of the latency stage is exactly what
    is supposed to happen, and must not fail the run -- otherwise no committed command can
    reproduce the phase, which is the S3 defect.
    """
    outcome = _pipeline(
        {
            "lstm_quantile": (
                _parity("torch:cuda", passed=False),
                _parity("CPUExecutionProvider", passed=True),
            )
        },
        (_latency_row("lstm_quantile", "ort-cpu", "cpu"),),
    )
    assert unverified_timed_configurations(outcome, tmp_path) == set()


def test_a_pytorch_cpu_row_needs_no_parity_row_of_its_own(tmp_path: Path) -> None:
    """PyTorch on the CPU is the reference every other configuration is checked against."""
    outcome = _pipeline({}, (_latency_row("tcn", "torch-eager", "cpu"),))
    assert unverified_timed_configurations(outcome, tmp_path) == set()


def test_a_latency_only_run_reads_the_committed_parity_table(tmp_path: Path) -> None:
    """The committed sweep is several commands and only the first passes `--parity`.

    Reading this run's own parity result alone would enforce the invariant in that first
    command and drop it in every other -- and would flag every row of a `--latency`-only
    invocation as unverified, which is exactly what the first version of this function did
    when it was probed (docs/protocol.md P7-D11).
    """
    pd.DataFrame(
        [
            {"model": "tcn", "provider": "CPUExecutionProvider", "passed": True},
            {"model": "lstm_quantile", "provider": "torch:cuda", "passed": False},
        ]
    ).to_csv(tmp_path / "parity.csv", index=False)

    verified = _pipeline({}, (_latency_row("tcn", "ort-cpu", "cpu"),))
    assert unverified_timed_configurations(verified, tmp_path) == set()

    refused = _pipeline({}, (_latency_row("lstm_quantile", "torch-eager", "cuda"),))
    assert unverified_timed_configurations(refused, tmp_path) == {"lstm_quantile/torch-eager/cuda"}

    unchecked = _pipeline({}, (_latency_row("tcn", "ort-trt", "cuda"),))
    assert unverified_timed_configurations(unchecked, tmp_path) == {"tcn/ort-trt/cuda"}
