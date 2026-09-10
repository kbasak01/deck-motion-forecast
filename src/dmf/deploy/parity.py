"""Numerical parity between the PyTorch model and its exported ONNX graph.

Run before any timing measurement: benchmarking a graph that has not been parity-checked
measures how fast the wrong answer arrives.

**The criterion is scale-relative, and that is a threshold change.** The implementation
plan and ``CLAUDE.md`` both state it as ``max_abs_err < 1e-4`` absolute over 1000 random
windows. Measured, one of the four deployable models cannot pass that in FP32 *and neither
can PyTorch* -- ``lstm_quantile``'s own FP32 forward pass differs from the same model in
FP64 by more than 1e-4, on outputs reaching ``|y| = 9.8``, so the absolute form of the test
asks single precision for something it does not have. The criterion applied here is
therefore

    ``max_abs_err < tolerance * max(1, |y|_max)``

with ``tolerance`` unchanged at 1e-4. ``docs/protocol.md`` P7-D3 records the change, the
measurement behind it, and that the user adopted it explicitly. **The raw absolute
``max_abs_err`` and its verdict against the unscaled 1e-4 are reported alongside**, on
:class:`ParityResult` and in every artifact, so the plan's original criterion stays
auditable rather than being quietly superseded.

**One draw is not a measurement.** ``max_abs_err`` is a maximum over ~9 million elements,
i.e. a tail statistic, and it moves by about 2x between window draws: on
``lstm_quantile`` five 1000-window draws gave 8.5e-05, 9.6e-05, 9.7e-05, 1.8e-04 and
2.0e-04, which straddles the unscaled 1e-4 and would have made the absolute verdict a
property of the seed rather than of the model. :data:`PARITY_SEEDS` therefore runs the
check on **five** draws and the reported verdict is the worst of them.

The reference is the **CPU** execution provider. It is the deterministic one, and the
question this module asks is "did the export change the arithmetic?", not "do two
accelerators agree?" -- mixing the two would let a CUDA kernel's fused reduction show up
as an export defect.

The PyTorch side is :func:`dmf.deploy.export_onnx.wrap_for_export`, not ``model.forward``:
for a quantile head the exported graph carries the post-hoc quantile sort, so comparing
against the raw head would charge the sort to the tolerance and, worse, would pass just as
happily against a graph that had omitted it.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dmf.deploy.export_onnx import INPUT_NAME, wrap_for_export
from dmf.deploy.providers import disable_tf32, preload_gpu_libraries, session_providers
from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = [
    "PARITY_CHUNK",
    "PARITY_N_WINDOWS",
    "PARITY_SEED",
    "PARITY_SEEDS",
    "ParityAttribution",
    "ParityResult",
    "attribute_parity",
    "check_parity",
    "check_torch_device_parity",
    "random_parity_windows",
]

#: Window count Gate 7 is read at, dimensionless.
PARITY_N_WINDOWS: int = 1000

#: Seed for the random parity windows. Recorded rather than left to the caller's default so
#: that a parity number in a committed artifact is reproducible exactly; a tolerance passed
#: by luck on one draw and failed on the next would otherwise be indistinguishable from a
#: flaky runtime.
PARITY_SEED: int = 20260909

#: The draws the reported verdict is taken over, worst-case. Five rather than one because
#: ``max_abs_err`` is a maximum over a tail and moves ~2x between draws (see the module
#: docstring); these are the five the criterion change was decided on, kept verbatim so the
#: committed artifact is comparable to the measurement in ``docs/protocol.md`` P7-D3.
PARITY_SEEDS: tuple[int, ...] = (PARITY_SEED, 0, 1, 7, 42)

#: Windows pushed through both runtimes per call, dimensionless. The comparison is chunked
#: because a quantile model's output at 1000 windows is (1000, 150, 6, 9) float32 = 324 MiB
#: per runtime; the error statistics are accumulated across chunks, so the result is
#: identical to a single-shot comparison.
PARITY_CHUNK: int = 50


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a parity check.

    Attributes:
        max_abs_err: Largest absolute elementwise difference between the PyTorch and ONNX
            outputs, dimensionless (the models operate on normalised inputs). Reported
            unscaled, always, whatever the verdict is read against.
        mean_abs_err: Mean absolute elementwise difference, dimensionless.
        n_windows: Number of random windows compared.
        tolerance: The relative coefficient, dimensionless -- 1e-4, unchanged from the
            implementation plan. It is multiplied by :attr:`scale` to give the threshold
            actually applied.
        passed: Whether ``max_abs_err < scaled_tolerance``. **This is the verdict.**
        output_abs_max: Largest absolute value in the reference output, dimensionless. An
            absolute error is not interpretable without it: 1e-4 against an output of order
            1 is 1e-4 relative, and against an output of order 10 it is 1e-5.
        scale: ``max(1, output_abs_max)``. Floored at 1 so that a model whose outputs are
            *smaller* than unit scale is held to the plan's absolute 1e-4 and cannot be
            passed by a lenient scaling -- the change is only ever allowed to loosen the
            test for outputs that are genuinely large.
        scaled_tolerance: ``tolerance * scale``, dimensionless. The threshold applied.
        passed_absolute: Whether ``max_abs_err < tolerance``, i.e. the plan's **original**
            unscaled criterion. Carried so the superseded test stays auditable and a reader
            can see which rows the change actually decided.
        seed: Seed of the window draw this result is from, or -1 when the caller did not
            say. A verdict that moves between draws (module docstring) is not
            interpretable without it.
        provider: Execution provider the ONNX side ran on. Parity is checked **once per
            provider that is benchmarked**, not only on the CPU one: a provider's kernels
            are part of the computation, and TF32 on an Ampere card made every GPU provider
            fail this bar while the CPU provider passed (``docs/protocol.md`` P7-D9). A
            latency row whose provider has no parity row is a timing of something
            unverified.
    """

    max_abs_err: float
    mean_abs_err: float
    n_windows: int
    tolerance: float
    passed: bool
    output_abs_max: float = 0.0
    scale: float = 1.0
    scaled_tolerance: float = 1e-4
    passed_absolute: bool = False
    seed: int = -1
    provider: str = "CPUExecutionProvider"


def random_parity_windows(
    lookback: int,
    n_input_channels: int,
    n_windows: int = PARITY_N_WINDOWS,
    seed: int = PARITY_SEED,
) -> FloatArray:
    """Draw the random windows the parity check runs on.

    Standard normal, because the windows reaching ``forward`` are de-meaned per window and
    divided by the training-split channel scale, so N(0, 1) is the space the model actually
    sees. Random rather than corpus-drawn on purpose: the check should cover the input
    space rather than only the region this corpus happens to occupy, since an export defect
    localised outside the training distribution is still an export defect.

    Args:
        lookback: Input window length ``L``, samples.
        n_input_channels: Input channel count ``C_in``.
        n_windows: Number of windows to draw, dimensionless.
        seed: Seed for the draw, recorded on the benchmark artifact.

    Returns:
        Windows, shape ``(n_windows, L, C_in)``, dimensionless.
    """
    rng = np.random.default_rng(seed)
    drawn: FloatArray = rng.standard_normal((n_windows, lookback, n_input_channels))
    return drawn


def check_parity(
    model: BaseForecaster,
    onnx_path: Path,
    windows: FloatArray,
    tolerance: float = 1e-4,
    seed: int = -1,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> ParityResult:
    """Compare PyTorch and ONNX Runtime outputs over a batch of random windows.

    Args:
        model: The source PyTorch model, in eval mode.
        onnx_path: The exported graph.
        windows: Random input windows, shape ``(n_windows, L, C_in)``, dimensionless.
            Gate 7 uses 1000. Random rather than corpus-drawn, so that the check covers
            the input space rather than only the region the corpus happens to occupy.
        tolerance: Relative coefficient, dimensionless. The threshold applied is
            ``tolerance * max(1, |y|_max)`` -- see the module docstring and
            ``docs/protocol.md`` P7-D3 for why the criterion is scale-relative and why the
            unscaled verdict is reported beside it rather than dropped.
        seed: Seed the windows were drawn at, recorded on the result. The verdict moves
            between draws, so a result without its seed cannot be reproduced.
        providers: Execution providers for the ONNX side, in priority order. The PyTorch
            reference is always CPU FP32; running this against a GPU provider therefore
            asks a strictly larger question than "did the export change the arithmetic?" --
            it asks whether that provider's kernels reproduce it too, which is the question
            TF32 answered badly (P7-D9).

    Returns:
        The parity result. Returned rather than asserted, so that the caller can record
        the measured error in the benchmark artifact even on a pass.

    Raises:
        FileNotFoundError: If ``onnx_path`` does not exist.
        ValueError: If the requested provider is not the one the session realized, if
            ``windows`` does not match the model's declared input shape, or if
            the graph's output shape does not match
            :meth:`dmf.models.base.BaseForecaster.output_shape`. The shape check is here
            and not only in the tests because an elementwise comparison between two arrays
            that broadcast against each other would otherwise report a small error for two
            genuinely different outputs.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"no exported graph at {onnx_path}")
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape (n, L, C_in), got {windows.shape}")
    if windows.shape[1:] != (model.lookback, model.n_input_channels):
        raise ValueError(
            f"windows are (L={windows.shape[1]}, C_in={windows.shape[2]}) but the model "
            f"declares (L={model.lookback}, C_in={model.n_input_channels})"
        )
    if windows.shape[0] < 1:
        raise ValueError("windows must carry at least one window")

    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    session = ort.InferenceSession(str(onnx_path), providers=session_providers(providers))
    realized = tuple(str(name) for name in session.get_providers())
    if providers[0] not in realized:
        raise ValueError(
            f"requested {providers[0]!r} for the parity check but the session realized "
            f"{list(realized)}; a parity row recorded against a provider that did not load "
            f"would certify the wrong runtime"
        )
    reference = wrap_for_export(model)
    reference.eval()

    max_abs_err = 0.0
    abs_err_sum = 0.0
    n_elements = 0
    output_abs_max = 0.0
    with torch.no_grad():
        for start in range(0, windows.shape[0], PARITY_CHUNK):
            chunk = np.ascontiguousarray(windows[start : start + PARITY_CHUNK], dtype=np.float32)
            expected_shape = model.output_shape(chunk.shape[0])
            torch_out = reference(torch.from_numpy(chunk)).numpy()
            (onnx_out,) = session.run([], {INPUT_NAME: chunk})
            for name, array in (("pytorch", torch_out), ("onnx", onnx_out)):
                if tuple(array.shape) != expected_shape:
                    raise ValueError(
                        f"{name} output shape {tuple(array.shape)} does not match the "
                        f"model's declared output_shape {expected_shape}"
                    )
            err = np.abs(torch_out.astype(np.float64) - onnx_out.astype(np.float64))
            max_abs_err = max(max_abs_err, float(err.max()))
            abs_err_sum += float(err.sum())
            n_elements += int(err.size)
            # Scale is read off the PyTorch reference, not off the ONNX output: the
            # reference is the thing being reproduced, so the tolerance must not widen
            # because the graph under test produced a large number.
            output_abs_max = max(output_abs_max, float(np.abs(torch_out).max()))

    scale = max(1.0, output_abs_max)
    return ParityResult(
        max_abs_err=max_abs_err,
        mean_abs_err=abs_err_sum / n_elements,
        n_windows=int(windows.shape[0]),
        tolerance=tolerance,
        passed=max_abs_err < tolerance * scale,
        output_abs_max=output_abs_max,
        scale=scale,
        scaled_tolerance=tolerance * scale,
        passed_absolute=max_abs_err < tolerance,
        seed=seed,
        provider=providers[0],
    )


@dataclass(frozen=True)
class ParityAttribution:
    """Whether a parity gap is an export defect or the FP32 noise floor.

    A parity failure has exactly two causes and they call for opposite responses: the
    export changed the computation (fix the export), or single precision cannot resolve the
    tolerance for this model (the criterion is wrong for this graph, and saying so is a
    decision to record, not a threshold to quietly widen). Distinguishing them by eye from
    a single ``max_abs_err`` is not possible, so this measures the third quantity that
    settles it -- the same model in **double** precision, which stands in for the exact
    answer.

    If ``onnx_fp32_vs_fp64_max`` is no larger than ``torch_fp32_vs_fp64_max``, the exported
    graph is at least as close to the exact result as the PyTorch model it came from, and no
    export defect exists to find: the two FP32 runtimes simply round differently, and their
    difference is bounded by the sum of two errors neither of which is wrong.

    Attributes:
        torch_fp32_vs_fp64_max: Largest absolute difference between the PyTorch model in
            FP32 and the same model in FP64, dimensionless. The FP32 noise floor of the
            reference itself.
        onnx_fp32_vs_fp64_max: Largest absolute difference between the ONNX graph in FP32
            and the PyTorch model in FP64, dimensionless.
        output_abs_max: Largest absolute output value seen, dimensionless. An absolute
            tolerance is only interpretable next to it: 1e-4 on an output of order 1 is
            1e-4 relative, and on an output of order 10 it is 1e-5.
        n_windows: Windows compared, dimensionless.
    """

    torch_fp32_vs_fp64_max: float
    onnx_fp32_vs_fp64_max: float
    output_abs_max: float
    n_windows: int


def attribute_parity(
    model: BaseForecaster,
    onnx_path: Path,
    windows: FloatArray,
) -> ParityAttribution:
    """Measure both runtimes against a double-precision reference.

    Run when :func:`check_parity` fails, to say which of the two failure causes it was. The
    FP64 model is a deep copy: ``.double()`` mutates in place, and casting the caller's
    model would leave every later measurement running in a precision nobody asked for.

    Args:
        model: The source PyTorch model, in eval mode.
        onnx_path: The exported graph.
        windows: The same windows :func:`check_parity` was run on, shape
            ``(n_windows, L, C_in)``, dimensionless.

    Returns:
        The attribution.

    Raises:
        FileNotFoundError: If ``onnx_path`` does not exist.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"no exported graph at {onnx_path}")

    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    fp32 = wrap_for_export(model).eval()
    fp64 = wrap_for_export(copy.deepcopy(model).double()).eval()

    torch_gap = 0.0
    onnx_gap = 0.0
    out_absmax = 0.0
    with torch.no_grad():
        for start in range(0, windows.shape[0], PARITY_CHUNK):
            chunk = np.ascontiguousarray(windows[start : start + PARITY_CHUNK], dtype=np.float32)
            exact = fp64(torch.from_numpy(chunk).double()).numpy()
            torch_out = fp32(torch.from_numpy(chunk)).numpy().astype(np.float64)
            (onnx_out,) = session.run([], {INPUT_NAME: chunk})
            torch_gap = max(torch_gap, float(np.abs(torch_out - exact).max()))
            onnx_gap = max(onnx_gap, float(np.abs(onnx_out.astype(np.float64) - exact).max()))
            out_absmax = max(out_absmax, float(np.abs(exact).max()))

    return ParityAttribution(
        torch_fp32_vs_fp64_max=torch_gap,
        onnx_fp32_vs_fp64_max=onnx_gap,
        output_abs_max=out_absmax,
        n_windows=int(windows.shape[0]),
    )


#: Provider label recorded for a PyTorch-on-CUDA parity row. Not an ONNX Runtime provider
#: name, and deliberately shaped like one so a single ``provider`` column can carry both:
#: the question is the same either way -- does the thing that was *timed* reproduce the
#: reference FP32 computation?
TORCH_CUDA_PROVIDER: str = "torch:cuda"


def check_torch_device_parity(
    model: BaseForecaster,
    windows: FloatArray,
    device: str = "cuda",
    tolerance: float = 1e-4,
    seed: int = -1,
) -> ParityResult:
    """Compare the PyTorch model on a device against the same model on the CPU.

    The ONNX parity check certifies the exported graph under each execution provider. It
    says nothing about the **PyTorch CUDA rows**, which are timed on cuDNN and cuBLAS
    kernels that no other check touches -- and those are exactly the kernels TF32 changed
    (``docs/protocol.md`` P7-D9). Without this, four of the sixteen GPU latency rows would
    be timings of an arithmetic nobody verified.

    Args:
        model: The trained model, in eval mode. Restored to the CPU before returning.
        windows: Random input windows, shape ``(n_windows, L, C_in)``, dimensionless.
        device: Device to check, normally ``"cuda"``.
        tolerance: Relative coefficient, dimensionless; the threshold applied is
            ``tolerance * max(1, |y|_max)`` as everywhere else.
        seed: Seed the windows were drawn at, recorded on the result.

    Returns:
        The parity result, with ``provider`` set to ``torch:<device>``.

    Raises:
        ValueError: If ``device`` is not available.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but no CUDA device is available")
    disable_tf32()
    reference = wrap_for_export(model).eval()

    max_abs_err = 0.0
    abs_err_sum = 0.0
    n_elements = 0
    output_abs_max = 0.0
    with torch.no_grad():
        for start in range(0, windows.shape[0], PARITY_CHUNK):
            chunk = np.ascontiguousarray(windows[start : start + PARITY_CHUNK], dtype=np.float32)
            tensor = torch.from_numpy(chunk)
            cpu_out = reference(tensor).numpy().astype(np.float64)
            reference.to(device)
            device_out = reference(tensor.to(device)).detach().cpu().numpy().astype(np.float64)
            reference.to("cpu")
            err = np.abs(cpu_out - device_out)
            max_abs_err = max(max_abs_err, float(err.max()))
            abs_err_sum += float(err.sum())
            n_elements += int(err.size)
            output_abs_max = max(output_abs_max, float(np.abs(cpu_out).max()))

    scale = max(1.0, output_abs_max)
    return ParityResult(
        max_abs_err=max_abs_err,
        mean_abs_err=abs_err_sum / n_elements,
        n_windows=int(windows.shape[0]),
        tolerance=tolerance,
        passed=max_abs_err < tolerance * scale,
        output_abs_max=output_abs_max,
        scale=scale,
        scaled_tolerance=tolerance * scale,
        passed_absolute=max_abs_err < tolerance,
        seed=seed,
        provider=f"torch:{device}",
    )
