"""Latency and throughput benchmarking.

Methodology, which is the part that actually determines whether the numbers mean anything:

- 200 warmup iterations before any timed iteration, then 2000 timed iterations.
- ``torch.cuda.synchronize()`` around every timed region on CUDA. Un-synchronised CUDA
  timing is the most common benchmarking error there is, and it produces numbers that look
  impossibly fast. A number that looks too good is wrong, not impressive.
- Percentiles reported, not just the mean. A controller cares about p99, because that is
  the iteration that misses the control deadline.
- Every run stamped with hardware, driver, CUDA, torch and ORT versions. A benchmark
  number without an environment stamp is not reproducible, and a reviewer will notice.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = [
    "BenchConfig",
    "BenchResult",
    "benchmark_onnxruntime",
    "benchmark_torch",
    "environment_stamp",
    "write_benchmark_json",
]


@dataclass(frozen=True)
class BenchConfig:
    """Benchmark harness settings.

    Attributes:
        warmup_iters: Untimed iterations run before measurement, to let kernel autotuning,
            memory allocation and clock ramp settle.
        timed_iters: Timed iterations.
        batch_size: Batch size. 1 is the real-time case; 32 is the throughput case.
        device: ``"cpu"`` or ``"cuda"``.
        synchronize: Whether to synchronise the device around each timed region. Must be
            True on CUDA for the numbers to mean anything.
    """

    warmup_iters: int = 200
    timed_iters: int = 2000
    batch_size: int = 1
    device: str = "cpu"
    synchronize: bool = True


@dataclass(frozen=True)
class BenchResult:
    """Measured latency and throughput for one configuration.

    Attributes:
        label: Configuration name, e.g. ``"ort-cpu-b1"``.
        p50_ms: Median per-iteration latency, **milliseconds**.
        p90_ms: 90th-percentile latency, milliseconds.
        p99_ms: 99th-percentile latency, milliseconds.
        mean_ms: Mean latency, milliseconds.
        std_ms: Latency standard deviation, milliseconds.
        throughput_windows_s: Windows processed per second.
        peak_host_mem_mb: Peak host memory, mebibytes.
        peak_device_mem_mb: Peak device memory, mebibytes, or 0.0 on CPU.
        config: The harness settings used.
        env: Environment stamp from :func:`environment_stamp`.
    """

    label: str
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput_windows_s: float
    peak_host_mem_mb: float
    peak_device_mem_mb: float
    config: BenchConfig
    env: dict[str, str]


def environment_stamp() -> dict[str, str]:
    """Collect the environment metadata that makes a benchmark reproducible.

    Returns:
        Mapping with CPU model, GPU name, NVIDIA driver version, CUDA version, and the
        ``torch``, ``onnx`` and ``onnxruntime`` versions. Written into every benchmark
        artifact.
    """
    raise NotImplementedError


def benchmark_torch(
    model: BaseForecaster,
    windows: FloatArray,
    cfg: BenchConfig,
    label: str,
    compile_model: bool = False,
) -> BenchResult:
    """Benchmark the PyTorch model, eager or compiled.

    Args:
        model: The trained model, in eval mode and under ``torch.no_grad``.
        windows: Input windows to cycle through, shape ``(n_windows, L, C_in)``,
            dimensionless.
        cfg: Harness settings.
        label: Configuration name recorded on the result.
        compile_model: Whether to wrap the model in ``torch.compile`` first. The warmup
            iterations must absorb compilation, or the first timed iteration measures the
            compiler.

    Returns:
        The measured result.

    Raises:
        ValueError: If ``cfg.device`` is ``"cuda"`` but no CUDA device is available, or if
            ``cfg.device`` is ``"cuda"`` and ``cfg.synchronize`` is False.
    """
    raise NotImplementedError


def benchmark_onnxruntime(
    onnx_path: Path,
    windows: FloatArray,
    cfg: BenchConfig,
    label: str,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> BenchResult:
    """Benchmark an ONNX graph under a given execution provider.

    Args:
        onnx_path: The exported graph, already parity-checked.
        windows: Input windows to cycle through, shape ``(n_windows, L, C_in)``,
            dimensionless.
        cfg: Harness settings.
        label: Configuration name recorded on the result.
        providers: ONNX Runtime execution providers in priority order, e.g.
            ``("CUDAExecutionProvider", "CPUExecutionProvider")``. The provider actually
            selected is recorded on the result, since ORT silently falls back when a
            requested provider is unavailable -- which would otherwise produce a "CUDA"
            row containing CPU timings.

    Returns:
        The measured result.

    Raises:
        FileNotFoundError: If ``onnx_path`` does not exist.
        ValueError: If none of ``providers`` is available in this ORT build.
    """
    raise NotImplementedError


def write_benchmark_json(results: tuple[BenchResult, ...], path: Path) -> Path:
    """Write benchmark results, with environment stamps, to JSON.

    Args:
        results: The results to write.
        path: Destination file under ``results/``.

    Returns:
        The path written.
    """
    raise NotImplementedError


def _as_serialisable(result: BenchResult) -> dict[str, Any]:
    """Flatten a result into a JSON-serialisable mapping.

    Args:
        result: The result to flatten.

    Returns:
        Mapping with the latency fields in milliseconds, the harness settings, and the
        environment stamp.
    """
    raise NotImplementedError
