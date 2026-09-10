"""ONNX export parity and benchmark-harness integrity -- Gate 7.

Empty in Phase 0. Implemented in Phase 7.

Covered here:

- FP32 parity: ``max_abs_err < 1e-4`` between PyTorch and ONNX Runtime over 1000 random
  windows, for every exported model.
- The exported graph has a dynamic batch axis and a fixed sequence axis.
- Output shape matches ``BaseForecaster.output_shape`` for both point and quantile heads.
- ``benchmark_onnxruntime`` records the provider actually selected, not the one requested,
  so an unavailable CUDA provider cannot silently produce a "CUDA" row holding CPU timings.
- ``benchmark_torch`` raises when asked to time a CUDA run with ``synchronize=False``:
  un-synchronised CUDA timing produces impossibly fast numbers, and the harness must refuse
  to generate them rather than leave it to the reader to notice.

Three things the phase added to that pre-registered list, each because implementing it
turned up something the list did not anticipate:

**The quantile sort is in the graph, and that is asserted.** ``BaseForecaster.forward``
does not sort; the sort lives downstream in ``PredictiveDistribution``. A naively traced
quantile graph therefore ships crossing quantiles -- the trap ``CLAUDE.md`` names -- so
``dmf.deploy.export_onnx.wrap_for_export`` appends it, for a ``quantile`` head only.
``test_gaussian_head_is_never_wrapped_in_the_quantile_sort`` is the other half: sorting a
Gaussian head's ``(mean, log_var)`` axis is silent, shape-preserving and catastrophic
(``docs/protocol.md`` P5-D1).

**The criterion is scale-relative, and the verdict is read over five window draws.** The
plan's ``max_abs_err < 1e-4`` absolute is not something FP32 can deliver for
``lstm_quantile``: on outputs reaching ``|y| = 9.79``, PyTorch's own FP32 forward pass is
``1.06e-4`` away from the same model in FP64, so the absolute test fails against the
reference itself. The criterion in force is ``max_abs_err < 1e-4 * max(1, |y|_max)``
(``docs/protocol.md`` P7-D3, adopted by the user), the unscaled verdict is still computed
and reported as ``passed_absolute``, and both are asserted here. The check runs on the five
draws of ``dmf.deploy.parity.PARITY_SEEDS`` and the verdict is the worst of them, because
``max_abs_err`` is a maximum over a tail and moves about 2x between draws -- measured on
``lstm_quantile`` at 8.5e-05, 9.6e-05, 9.7e-05, 1.8e-04, 2.0e-04, which straddles 1e-4 and
made the *absolute* verdict a property of the seed rather than of the model.

**The declared output dims of a quantile graph had to be repaired.** ``torch.sort`` exports
as ``TopK``, whose ``K`` is a graph input, so ONNX shape inference abandons every trailing
axis and the graph came out declaring ``(batch, TopKoutput_dim_1, TopKoutput_dim_2,
TopKoutput_dim_3)``. Those axes are fixed properties of the trained model, and
``test_exported_graph_declares_a_dynamic_batch_axis_and_fixed_everything_else`` asserts the
repair for the quantile head as well as the point one -- which is the head where it was
needed.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from dmf.deploy.bench import BenchConfig, benchmark_onnxruntime, benchmark_torch
from dmf.deploy.export_onnx import (
    INPUT_NAME,
    export_model,
    graph_dims,
    graph_input_geometry,
    wrap_for_export,
)
from dmf.deploy.harness import BenchJob, run_job_subprocess
from dmf.deploy.parity import (
    PARITY_N_WINDOWS,
    PARITY_SEEDS,
    attribute_parity,
    check_parity,
    random_parity_windows,
)
from dmf.deploy.pipeline import export_targets
from dmf.deploy.targets import DEFAULT_TARGETS, ModelTarget, load_target, onnx_path_for
from dmf.models.dlinear import DLinear
from dmf.models.heads import QUANTILE_FAN_9

#: Root of the committed checkpoint tree. Every test that needs trained weights skips
#: without it, so the suite still runs on a fresh clone that has not trained anything.
CHECKPOINT_ROOT = Path("artifacts/checkpoints")

#: Geometry of the throwaway models the fast tests export, in samples and channels. Small
#: enough that an export plus a session plus a few hundred inferences is under a second,
#: which is what keeps the harness tests runnable on every change.
TINY_LOOKBACK = 32
TINY_HORIZON = 8
TINY_CHANNELS = 3

#: Gate 7's relative coefficient, dimensionless. Restated here so a test reads the number it
#: asserts. The threshold applied is this times ``max(1, |y|_max)``.
PARITY_TOLERANCE = 1e-4

#: The five ``lstm_quantile`` draws the criterion change was decided on, as
#: ``(seed, onnx-vs-torch, torch-vs-fp64, onnx-vs-fp64)``, dimensionless. Two of the five
#: exceed the unscaled 1e-4 and three do not, which is what made a single-draw absolute
#: verdict a coin flip; the worst ratio of ONNX to torch error is 1.98, which is what the
#: factor in `test_no_export_defect_hides_in_the_lstm_quantile_parity_gap` is sized against.
LSTM_QUANTILE_DRAWS: tuple[tuple[int, float, float, float], ...] = (
    (0, 9.727e-05, 7.020e-05, 9.978e-05),
    (1, 9.632e-05, 1.004e-04, 1.283e-04),
    (20260909, 1.809e-04, 1.094e-04, 2.165e-04),
    (42, 8.535e-05, 1.082e-04, 1.453e-04),
    (7, 2.024e-04, 1.340e-04, 2.155e-04),
)

#: Slack allowed between the ONNX graph's distance from FP64 and PyTorch's own, as a factor.
#: The five draws above top out at 1.98, so 2.0 -- the value this test first shipped with --
#: passed by one percent and was a latent flake. 3.0 is above the measured spread and still
#: far below what an actual export defect would produce, which would move the ONNX error by
#: the size of the defect rather than by a rounding factor.
FP64_GAP_FACTOR = 3.0


def _tiny_model(head: str) -> DLinear:
    """Build an untrained DLinear with one of the three heads.

    DLinear because it is the one model in the project that carries all three heads and is
    cheap enough to export inside a fast test.

    Args:
        head: ``"point"``, ``"quantile"`` or ``"gaussian"``.

    Returns:
        The model, in eval mode. Untrained: every property under test here is a property of
        the graph's structure and arithmetic, not of the weights' values.
    """
    n_quantiles = len(QUANTILE_FAN_9) if head == "quantile" else 0
    model = DLinear(
        lookback=TINY_LOOKBACK,
        max_horizon=TINY_HORIZON,
        n_input_channels=TINY_CHANNELS,
        n_target_channels=TINY_CHANNELS,
        kernel_size=5,
        n_quantiles=n_quantiles,
        head=head,  # type: ignore[arg-type]
    )
    model.eval()
    return model


@pytest.fixture
def tiny_point_graph(tmp_path: Path) -> tuple[DLinear, Path]:
    """Export a tiny point model and return it with its graph."""
    model = _tiny_model("point")
    sample = torch.zeros((1, TINY_LOOKBACK, TINY_CHANNELS))
    return model, export_model(model, sample, tmp_path / "point.onnx")


@pytest.fixture
def tiny_quantile_graph(tmp_path: Path) -> tuple[DLinear, Path]:
    """Export a tiny quantile model and return it with its graph."""
    model = _tiny_model("quantile")
    sample = torch.zeros((1, TINY_LOOKBACK, TINY_CHANNELS))
    return model, export_model(model, sample, tmp_path / "quantile.onnx")


@pytest.fixture(scope="session")
def exported_targets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Export all four deployable models once for the whole session.

    Session-scoped because four exports plus four checkpoint loads is the expensive part of
    this file, and every slow test wants the same four graphs.

    Returns:
        Graph path per target key.

    Raises:
        pytest.skip.Exception: If the committed checkpoints are absent.
    """
    if not CHECKPOINT_ROOT.is_dir():
        pytest.skip(f"{CHECKPOINT_ROOT} absent; trained checkpoints are needed to export")
    onnx_dir = tmp_path_factory.mktemp("onnx")
    records = export_targets(DEFAULT_TARGETS, onnx_dir, CHECKPOINT_ROOT)
    return {record.target_key: record.onnx_path for record in records}


# ---------------------------------------------------------------------------------------
# 1. FP32 parity over 1000 random windows, for every exported model.
# ---------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("target", DEFAULT_TARGETS, ids=[t.key for t in DEFAULT_TARGETS])
def test_fp32_parity_holds_over_a_thousand_random_windows(
    target: ModelTarget, exported_targets: dict[str, Path]
) -> None:
    """PyTorch and ONNX Runtime agree over 1000 random windows, on every one of five draws.

    The whole of Gate 7 rests on this: a graph that has not been checked here has no
    business being timed, because the benchmark would faithfully measure how fast the wrong
    answer arrives and every column of the table would look correct.

    Asserted on the WORST of the five draws, not the pinned one. The criterion is
    `max_abs_err < 1e-4 * max(1, |y|max)`; the raw absolute error and its verdict against
    the plan's unscaled 1e-4 are asserted to be present and self-consistent, so the
    superseded criterion stays visible in the artifact rather than being replaced by it.

    The reference is `wrap_for_export(model)`, not `model.forward`, so that the quantile
    sort the graph carries is part of what is compared rather than a difference the
    tolerance absorbs.
    """
    model = load_target(target, checkpoint_root=CHECKPOINT_ROOT)
    draws = [
        check_parity(
            model,
            exported_targets[target.key],
            random_parity_windows(model.lookback, model.n_input_channels, seed=seed),
            tolerance=PARITY_TOLERANCE,
            seed=seed,
        )
        for seed in PARITY_SEEDS
    ]
    worst = max(draws, key=lambda result: result.max_abs_err)
    assert len(draws) == len(PARITY_SEEDS)
    assert worst.n_windows == PARITY_N_WINDOWS
    assert worst.tolerance == PARITY_TOLERANCE
    assert worst.scale == max(1.0, worst.output_abs_max)
    assert worst.scaled_tolerance == PARITY_TOLERANCE * worst.scale
    assert worst.max_abs_err < worst.scaled_tolerance
    assert worst.passed
    # The unscaled verdict is still computed, and it is the stricter of the two.
    assert worst.passed_absolute == (worst.max_abs_err < PARITY_TOLERANCE)
    assert not worst.passed_absolute or worst.passed


@pytest.mark.slow
def test_no_export_defect_hides_in_the_lstm_quantile_parity_gap(
    exported_targets: dict[str, Path],
) -> None:
    """`lstm_quantile` misses the unscaled 1e-4 by rounding, not by computing something else.

    Measured against the same model in double precision, which stands in for the exact
    answer. If the export had changed the computation, the ONNX graph would be further from
    FP64 than PyTorch's own FP32 output is, by roughly the size of the defect. It is not:
    the two are within a rounding factor of each other and BOTH exceed 1e-4, which is what
    identifies the unscaled criterion rather than the export as the thing that does not fit,
    and which is the evidence behind the scale-relative criterion (`docs/protocol.md`
    P7-D3).

    This is the assertion that would catch a real regression in the LSTM export. It matters
    more now than it did before the criterion changed: the scaled threshold gives this model
    a 4.9x margin, so a genuine defect of a few times 1e-4 would pass the parity check and
    fail only here.
    """
    target = next(t for t in DEFAULT_TARGETS if t.key == "lstm_quantile")
    model = load_target(target, checkpoint_root=CHECKPOINT_ROOT)
    windows = random_parity_windows(model.lookback, model.n_input_channels)
    attribution = attribute_parity(model, exported_targets["lstm_quantile"], windows)
    # Within a rounding factor of the model it came from. Sized against the five measured
    # draws in LSTM_QUANTILE_DRAWS, whose worst ratio is 1.98 -- so the 2.0 this shipped
    # with passed by one percent and would have flaked on another draw.
    assert max(onnx / torch_gap for _, _, torch_gap, onnx in LSTM_QUANTILE_DRAWS) < FP64_GAP_FACTOR
    assert attribution.onnx_fp32_vs_fp64_max < FP64_GAP_FACTOR * attribution.torch_fp32_vs_fp64_max
    # And the FP32 floor really is above the tolerance, which is the whole claim.
    assert attribution.torch_fp32_vs_fp64_max > PARITY_TOLERANCE * 0.5
    assert attribution.output_abs_max > 1.0


def test_the_scale_floor_holds_a_small_output_model_to_the_plans_absolute_tolerance(
    tmp_path: Path,
) -> None:
    """A model whose outputs are below unit scale gets exactly the unscaled 1e-4.

    The floor at 1 is what keeps the criterion change from being a general loosening: it can
    only ever relax the test for outputs that are genuinely large, and for everything else
    the applied threshold is the number the implementation plan wrote down. Asserted on a
    zeroed model, so the output is exactly 0 and the floor is exercised rather than
    approached.
    """
    model = _tiny_model("point")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = export_model(
        model, torch.zeros((1, TINY_LOOKBACK, TINY_CHANNELS)), tmp_path / "zeroed.onnx"
    )
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=8)
    result = check_parity(model, path, windows, tolerance=PARITY_TOLERANCE, seed=3)
    assert result.output_abs_max == 0.0
    assert result.scale == 1.0
    assert result.scaled_tolerance == PARITY_TOLERANCE
    assert result.max_abs_err == 0.0
    assert result.passed and result.passed_absolute
    assert result.seed == 3


def test_the_threshold_scales_with_the_reference_outputs_own_magnitude(tmp_path: Path) -> None:
    """A model with order-1000 outputs is held to 1e-4 relative, not 1e-4 absolute.

    The scale is read off the PyTorch reference, never off the graph under test: a graph
    that produced a large number would otherwise widen the tolerance it is being judged by.
    """
    model = _tiny_model("point")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(1000.0)
    path = export_model(
        model, torch.zeros((1, TINY_LOOKBACK, TINY_CHANNELS)), tmp_path / "amplified.onnx"
    )
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=8)
    result = check_parity(model, path, windows, tolerance=PARITY_TOLERANCE)
    assert result.output_abs_max > 1.0
    assert result.scale == result.output_abs_max
    assert result.scaled_tolerance == pytest.approx(PARITY_TOLERANCE * result.output_abs_max)
    assert result.scaled_tolerance > PARITY_TOLERANCE
    assert result.passed


def test_parity_refuses_windows_that_do_not_match_the_declared_input_shape(
    tiny_point_graph: tuple[DLinear, Path],
) -> None:
    """Windows of the wrong geometry raise rather than broadcast into a small error."""
    model, path = tiny_point_graph
    wrong = random_parity_windows(TINY_LOOKBACK + 1, TINY_CHANNELS, n_windows=4)
    with pytest.raises(ValueError, match="declares"):
        check_parity(model, path, wrong)


def test_parity_refuses_a_graph_that_was_never_exported(
    tiny_point_graph: tuple[DLinear, Path], tmp_path: Path
) -> None:
    """A missing graph is a hard error, not an empty result."""
    model, _ = tiny_point_graph
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=4)
    with pytest.raises(FileNotFoundError):
        check_parity(model, tmp_path / "absent.onnx", windows)


# ---------------------------------------------------------------------------------------
# 2. Dynamic batch axis, fixed sequence axis.
# ---------------------------------------------------------------------------------------


def test_exported_graph_declares_a_dynamic_batch_axis_and_fixed_everything_else(
    tiny_point_graph: tuple[DLinear, Path], tiny_quantile_graph: tuple[DLinear, Path]
) -> None:
    """Batch is a symbol; sequence, channel and horizon axes are integers.

    Read off the graph's declared dims rather than from the shape of some run's output,
    which is the only way to tell a dynamic axis from a fixed one without exporting twice.

    The sequence axis is fixed deliberately: a dynamic one blocks kernel selection and buys
    nothing for a model whose lookback is a fixed property of its weights. The quantile
    graph is checked too, because that is the head whose trailing axes ONNX shape inference
    gives up on -- see the module docstring.
    """
    for model, path in (tiny_point_graph, tiny_quantile_graph):
        input_dims, output_dims = graph_dims(path)
        assert input_dims == ("batch", TINY_LOOKBACK, TINY_CHANNELS)
        assert output_dims[0] == "batch"
        assert output_dims[1:] == model.output_shape(1)[1:]
        assert all(isinstance(dim, int) for dim in output_dims[1:])
        assert graph_input_geometry(path) == (TINY_LOOKBACK, TINY_CHANNELS)


def test_export_refuses_a_sample_input_of_the_wrong_geometry() -> None:
    """A trace at the wrong L or C_in would fix the wrong axes into the graph."""
    model = _tiny_model("point")
    with pytest.raises(ValueError, match="declares"):
        export_model(model, torch.zeros((1, TINY_LOOKBACK + 1, TINY_CHANNELS)), Path("unused.onnx"))
    with pytest.raises(ValueError, match="shape"):
        export_model(model, torch.zeros((TINY_LOOKBACK, TINY_CHANNELS)), Path("unused.onnx"))


# ---------------------------------------------------------------------------------------
# 3. Output shape matches `output_shape`, for both heads.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
def test_graph_output_shape_matches_the_models_declared_output_shape(
    batch_size: int,
    tiny_point_graph: tuple[DLinear, Path],
    tiny_quantile_graph: tuple[DLinear, Path],
) -> None:
    """At two batch sizes, and for both heads, ORT returns exactly `output_shape(B)`.

    Two batch sizes rather than one because the point of the dynamic axis is that the graph
    is correct at a batch it was not traced at -- it was traced at 1.
    """
    for model, path in (tiny_point_graph, tiny_quantile_graph):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = np.zeros((batch_size, TINY_LOOKBACK, TINY_CHANNELS), dtype=np.float32)
        (out,) = session.run([], {INPUT_NAME: x})
        assert out.shape == model.output_shape(batch_size)


def test_quantile_graph_output_is_non_decreasing_along_the_quantile_axis(
    tiny_quantile_graph: tuple[DLinear, Path],
) -> None:
    """The sort is IN the graph, so an ONNX consumer cannot receive crossing quantiles.

    Pinball loss does not constrain ordering and `BaseForecaster.forward` does not sort --
    the sort lives in `PredictiveDistribution`, which a deployed ONNX consumer does not
    have. An untrained head is used on purpose: its quantiles cross constantly, so this
    would fail loudly if the sort node were dropped from the export.
    """
    model, path = tiny_quantile_graph
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, TINY_LOOKBACK, TINY_CHANNELS)).astype(np.float32)
    (out,) = session.run([], {INPUT_NAME: x})
    assert np.all(np.diff(out, axis=-1) >= 0.0)
    # The unsorted head really does cross, so the assertion above is not vacuous.
    with torch.no_grad():
        raw = model.forward(torch.from_numpy(x)).numpy()
    assert np.any(np.diff(raw, axis=-1) < 0.0)


@pytest.mark.slow
def test_the_real_quantile_graphs_are_non_decreasing_too(
    exported_targets: dict[str, Path],
) -> None:
    """The same, on the two trained quantile models that would actually be deployed."""
    for key in ("tcn_quantile", "lstm_quantile"):
        session = ort.InferenceSession(
            str(exported_targets[key]), providers=["CPUExecutionProvider"]
        )
        lookback, n_channels = graph_input_geometry(exported_targets[key])
        x = random_parity_windows(lookback, n_channels, n_windows=8).astype(np.float32)
        (out,) = session.run([], {INPUT_NAME: x})
        assert np.all(np.diff(out, axis=-1) >= 0.0), key


def test_gaussian_head_is_never_wrapped_in_the_quantile_sort() -> None:
    """A Gaussian head's `(mean, log_var)` axis is not a fan and must never be sorted.

    Sorting it swaps the two on every element where `log_var < mean`: shape-preserving,
    silent, and it produces intervals that are wrong without being malformed
    (`docs/protocol.md` P5-D1). The export wrapper dispatches on the head KIND, not on the
    output's rank, which is what makes a rank-4 Gaussian output safe here.
    """
    gaussian = _tiny_model("gaussian")
    assert wrap_for_export(gaussian) is gaussian
    point = _tiny_model("point")
    assert wrap_for_export(point) is point
    quantile = _tiny_model("quantile")
    assert wrap_for_export(quantile) is not quantile


# ---------------------------------------------------------------------------------------
# 4. The realized provider is recorded, not the requested one.
# ---------------------------------------------------------------------------------------


def test_benchmark_records_the_provider_the_session_actually_realized(
    tiny_point_graph: tuple[DLinear, Path],
) -> None:
    """A CPU-provider run records `CPUExecutionProvider`, plus the pinned thread count."""
    _, path = tiny_point_graph
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=8)
    cfg = BenchConfig(warmup_iters=2, timed_iters=5, batch_size=1, device="cpu")
    result = benchmark_onnxruntime(path, windows, cfg, "tiny-ort-cpu", ("CPUExecutionProvider",))
    assert result.providers_realized[0] == "CPUExecutionProvider"
    assert result.intra_op_threads == cfg.intra_op_threads
    assert result.p50_ms > 0.0
    assert result.peak_device_mem_mb == 0.0
    assert result.env["git_commit"]


def test_benchmark_refuses_a_session_that_fell_back_to_another_provider(
    tiny_point_graph: tuple[DLinear, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA request realized as CPU raises instead of returning a mislabelled row.

    This is the failure this phase exists to prevent, and it cannot be provoked on a
    machine where CUDA loads correctly -- so the session is stubbed to do exactly what a
    real ORT session does when `libcublas` is not resident: return a working CPU session
    and log an error nobody reads. `ort.get_available_providers()` would still list CUDA
    here, which is precisely why the harness never consults it.
    """
    _, path = tiny_point_graph
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=4)

    class _FellBackToCPU:
        """A session that was asked for CUDA and quietly returned CPU."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_providers(self) -> list[str]:
            """Report the provider that actually loaded."""
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", _FellBackToCPU)
    cfg = BenchConfig(warmup_iters=1, timed_iters=1, batch_size=1, device="cuda")
    with pytest.raises(ValueError, match="realized"):
        benchmark_onnxruntime(
            path, windows, cfg, "tiny-ort-cuda", ("CUDAExecutionProvider", "CPUExecutionProvider")
        )


def test_benchmark_refuses_a_graph_that_does_not_exist(tmp_path: Path) -> None:
    """A missing graph raises rather than producing an empty row."""
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=4)
    with pytest.raises(FileNotFoundError):
        benchmark_onnxruntime(tmp_path / "absent.onnx", windows, BenchConfig(), "missing")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize(
    ("backend", "expected"),
    [("ort-cuda", "CUDAExecutionProvider"), ("ort-trt", "TensorrtExecutionProvider")],
)
def test_gpu_providers_are_genuinely_realized_not_silently_cpu(
    backend: str, expected: str, exported_targets: dict[str, Path]
) -> None:
    """On this machine, the CUDA and TensorRT providers really do load.

    The complement of the stubbed test above: that one asserts the harness refuses a
    fallback, this one asserts there is nothing to refuse here, which is what makes the
    committed GPU rows meaningful. It goes through the subprocess path so that the
    `dmf.deploy.providers` preload is exercised in a fresh interpreter -- the same
    condition `make bench` runs in.
    """
    onnx_dir = next(iter(exported_targets.values())).parent
    job = BenchJob(
        target_key="tcn",
        backend=backend,  # type: ignore[arg-type]
        batch_size=1,
        warmup_iters=2,
        timed_iters=5,
        onnx_dir=onnx_dir,
        checkpoint_root=CHECKPOINT_ROOT,
    )
    result = run_job_subprocess(job)
    assert result.providers_realized[0] == expected
    assert result.p50_ms > 0.0


# ---------------------------------------------------------------------------------------
# 5. Un-synchronised CUDA timing is refused.
# ---------------------------------------------------------------------------------------


def test_benchmark_torch_refuses_unsynchronised_cuda_timing() -> None:
    """`device="cuda"` with `synchronize=False` raises, on any machine.

    Without the synchronise, the timed region ends when the kernels have been *queued*, and
    the harness would report a few microseconds for work that has not happened. That is the
    single most common benchmarking error there is and it produces numbers that look like a
    result, so the harness refuses to generate them rather than leaving a reader to notice
    that 20 microseconds is impossible.

    The refusal is checked before the CUDA-availability probe, so it raises for the right
    reason on a CPU-only machine too -- otherwise this assertion would only be exercised
    where a GPU happens to exist.
    """
    model = _tiny_model("point")
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=4)
    cfg = BenchConfig(warmup_iters=1, timed_iters=1, device="cuda", synchronize=False)
    with pytest.raises(ValueError, match="synchronize=False"):
        benchmark_torch(model, windows, cfg, "tiny-torch-cuda-unsynced")


def test_benchmark_torch_refuses_cuda_when_no_device_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With synchronisation asked for and no GPU present, the refusal names the device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _tiny_model("point")
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=4)
    cfg = BenchConfig(warmup_iters=1, timed_iters=1, device="cuda", synchronize=True)
    with pytest.raises(ValueError, match="no CUDA device"):
        benchmark_torch(model, windows, cfg, "tiny-torch-cuda-absent")


def test_benchmark_torch_measures_and_stamps_a_cpu_run() -> None:
    """The PyTorch CPU path produces ordered percentiles, a thread count and a stamp."""
    model = _tiny_model("point")
    windows = random_parity_windows(TINY_LOOKBACK, TINY_CHANNELS, n_windows=8)
    cfg = BenchConfig(warmup_iters=2, timed_iters=10, batch_size=2, device="cpu")
    result = benchmark_torch(model, windows, cfg, "tiny-torch-eager-cpu")
    assert result.p50_ms <= result.p90_ms <= result.p99_ms
    assert result.mean_ms > 0.0
    assert result.throughput_windows_s == pytest.approx(cfg.batch_size / (result.mean_ms / 1e3))
    assert result.providers_realized == ("torch-eager:cpu",)
    assert result.intra_op_threads == cfg.intra_op_threads
    assert result.peak_device_mem_mb == 0.0
    assert result.peak_host_mem_mb > 0.0
    assert result.config == cfg


def test_the_harness_offers_no_way_to_configure_an_unsynchronised_gpu_timing() -> None:
    """Every job built by the harness synchronises, whatever else it sets.

    The refusal in `benchmark_torch` is the last line of defence; this is the one that means
    it never has to fire. `BenchJob.bench_config` is the only construction site of a
    `BenchConfig` in the benchmark path, and it hard-codes `synchronize=True`.
    """
    job = BenchJob(target_key="tcn", backend="torch-eager", device="cuda")
    assert job.bench_config().synchronize
    assert replace(job, backend="ort-cuda").bench_config().device == "cuda"
    assert replace(job, backend="ort-cpu").bench_config().device == "cpu"


def test_onnx_paths_carry_the_regime_and_seed_they_were_exported_from() -> None:
    """A graph is weights, not just an architecture, so its filename says which weights."""
    target = next(t for t in DEFAULT_TARGETS if t.key == "tcn")
    path = onnx_path_for(target, Path("artifacts/onnx"))
    assert path.name == "tcn_id_seed0.onnx"
    assert onnx_path_for(replace(target, seed=2)).name == "tcn_id_seed2.onnx"
