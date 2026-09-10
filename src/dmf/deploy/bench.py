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

Three further decisions this module makes, each of which changes the numbers:

**Host to host.** One iteration is measured from "a window is a NumPy array in host memory"
to "the forecast is a NumPy array in host memory", for every backend. That includes the
host-to-device copy and the copy back on the CUDA paths, and it is the quantity a flight
controller experiences. Timing device-resident tensors instead would flatter CUDA against
a CPU provider that has no transfer to hide, and the comparison this phase exists to make
is precisely CPU-versus-CUDA at batch 1.

**Threads are pinned and recorded.** ORT CPU-provider latency moves by more than the
effect being measured when the thread count changes, so both ``intra_op_num_threads`` and
``torch.set_num_threads`` are set from :attr:`BenchConfig.intra_op_threads` and the value
that took effect is read back onto :attr:`BenchResult.intra_op_threads`.

**The realized provider is read back, never assumed.** ``ort.get_available_providers()``
reports what the wheel was built with; a session whose requested provider failed to load
falls back to CPU and runs perfectly well (:mod:`dmf.deploy.providers`). This module reads
``session.get_providers()`` after construction and refuses to return a result whose
requested provider is absent, because the alternative failure mode -- a "TensorRT" row
holding CPU timings -- is invisible in every column of the output table.
"""

import json
import os
import platform
import resource
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dmf.deploy.export_onnx import INPUT_NAME
from dmf.deploy.providers import disable_tf32, preload_gpu_libraries, session_providers
from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = [
    "BYTES_PER_MIB",
    "GPU_PROVIDERS",
    "BenchConfig",
    "BenchResult",
    "benchmark_onnxruntime",
    "benchmark_torch",
    "environment_stamp",
    "write_benchmark_json",
]

#: Bytes per mebibyte. Memory is reported in MiB, not MB.
BYTES_PER_MIB: float = 1024.0 * 1024.0

#: ORT execution providers that place compute on the GPU. Used to decide whether a device
#: memory figure and a ``torch.cuda.synchronize()`` are meaningful for a session.
GPU_PROVIDERS: frozenset[str] = frozenset({"CUDAExecutionProvider", "TensorrtExecutionProvider"})

#: Percentiles reported, in percent. p99 is the one a real-time control loop is designed
#: against; a mean alone hides exactly the tail that misses a deadline.
_PERCENTILES: tuple[float, float, float] = (50.0, 90.0, 99.0)


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
        intra_op_threads: CPU threads each backend may use within one operator. Pinned
            rather than left to the runtime's default, which is core-count dependent and
            therefore not reproducible on another machine. The default of 1 is the
            deployment-honest setting for this study: the target is a flight controller
            sharing a CPU with the rest of the autopilot, and a benchmark that quietly
            takes every core measures a machine nobody is deploying to. Recorded on the
            result, so a table that mixes settings cannot be assembled by accident.
    """

    warmup_iters: int = 200
    timed_iters: int = 2000
    batch_size: int = 1
    device: str = "cpu"
    synchronize: bool = True
    intra_op_threads: int = 1


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
        peak_host_mem_mb: Peak host memory, mebibytes: this process's ``ru_maxrss``, which
            is a per-configuration peak only because each configuration runs in its own
            subprocess (:mod:`dmf.deploy.harness`). **Read it as a floor, not as a model
            footprint.** Every child imports torch -- the ORT children too, because that is
            what preloads the CUDA libraries -- and on the reference machine that import
            alone is about 1.19 GiB, which dominates every row and is common to all of
            them. Differences between rows are informative; the absolute value is mostly
            the interpreter.
        peak_device_mem_mb: Peak device memory, mebibytes, or 0.0 on CPU. For an ORT GPU
            session it is the drop in device free memory across the run
            (``torch.cuda.mem_get_info``), so it includes the provider's CUDA context and,
            on TensorRT, its engine; for a torch run it is
            ``torch.cuda.max_memory_allocated``, which counts tensors only. The two are not
            the same quantity and are not comparable across backends.
        providers_realized: Execution providers the session **actually** reports after
            construction, in ORT's priority order, or the torch backend and device for a
            PyTorch row. The requested provider is in :attr:`config` and this is what was
            obtained; the two differing is the failure this field exists to expose, and
            :func:`benchmark_onnxruntime` refuses rather than reports when they do.
        intra_op_threads: Thread count that took effect, read back from the runtime rather
            than copied from the request.
        tf32: Whether TF32 was permitted on the GPU paths. **False on every row this
            project publishes.** For the PyTorch backends it is read back from
            ``torch.backends.cuda.matmul.allow_tf32``; for the ORT backends it is the literal
            ``False``, because ONNX Runtime exposes no way to read a provider option back off
            a session. That asymmetry is worth knowing given P7-D9 was "TF32 was on and
            nothing raised": for the ORT paths this column records what was *requested*, and
            the control that actually establishes the arithmetic is the per-provider parity
            check, which fails by five to fifty times when TF32 is on. It is a column rather
            than a footnote because it changes what
            is being timed: with TF32 on, the GPU rows compute a 10-bit-mantissa
            approximation that fails this project's parity bar, while the CPU rows compute
            full FP32 (``docs/protocol.md`` P7-D9). It is trivially False on a CPU-only row
            and is still recorded there, so the column never has to be interpreted.
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
    providers_realized: tuple[str, ...]
    intra_op_threads: int
    tf32: bool
    config: BenchConfig
    env: dict[str, str]


def _command_output(command: tuple[str, ...]) -> str:
    """Run a command and return its first line of output, or a marker on failure.

    Args:
        command: Argument vector.

    Returns:
        The first non-empty output line, stripped, or ``"unavailable"`` if the command is
        missing or fails. A stamp field is never allowed to abort a benchmark.
    """
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    for line in completed.stdout.splitlines():
        if line.strip():
            return line.strip()
    return "unavailable"


def _cpu_model() -> str:
    """Return the CPU model name.

    Returns:
        The ``model name`` field of ``/proc/cpuinfo``, or ``platform.processor()`` where
        that file does not exist.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _git_commit() -> str:
    """Return the commit the benchmark was run at.

    Returns:
        The short commit hash, suffixed ``-dirty`` if the working tree has uncommitted
        changes. A benchmark number whose commit is unknown is not reproducible, and one
        taken on a dirty tree is not reproducible either -- so the second case is labelled
        rather than hidden.
    """
    commit = _command_output(("git", "rev-parse", "--short", "HEAD"))
    if commit == "unavailable":
        return commit
    status = _command_output(("git", "status", "--porcelain"))
    return f"{commit}-dirty" if status not in {"unavailable", ""} else commit


def environment_stamp() -> dict[str, str]:
    """Collect the environment metadata that makes a benchmark reproducible.

    Returns:
        Mapping with CPU model, GPU name, NVIDIA driver version, CUDA version, and the
        ``torch``, ``onnx`` and ``onnxruntime`` versions. Written into every benchmark
        artifact. Also carries the TensorRT version, the Python version and platform, the
        default torch thread count, and the git commit -- the commit because a latency
        table whose source revision is unknown cannot be re-measured, and the TensorRT
        version because the EP links a specific ``libnvinfer`` soname
        (:mod:`dmf.deploy.providers`).
    """
    import onnx
    import onnxruntime as ort

    try:
        import tensorrt

        tensorrt_version = str(tensorrt.__version__)
    except ImportError:
        tensorrt_version = "absent"

    gpu_name = "none"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    driver = _command_output(("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"))
    return {
        "cpu": _cpu_model(),
        "cpu_count": str(len(os.sched_getaffinity(0))),
        "gpu": gpu_name,
        "nvidia_driver": driver,
        "cuda": str(torch.version.cuda or "none"),
        # `torch.backends.cudnn.version` carries no annotation; the cuDNN build is worth
        # stamping on a CUDA-EP benchmark, so the call is kept and the gap named.
        "cudnn": str(torch.backends.cudnn.version() or "none"),  # type: ignore[no-untyped-call]
        "torch": str(torch.__version__),
        "onnx": str(onnx.__version__),
        "onnxruntime": str(ort.__version__),
        "onnxruntime_package": "onnxruntime-gpu",
        "tensorrt": tensorrt_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_default_threads": str(torch.get_num_threads()),
        "git_commit": _git_commit(),
    }


def _peak_host_mem_mb() -> float:
    """Return this process's peak resident set size.

    ``ru_maxrss`` is a high-water mark that is never reset, so it is only a *per
    configuration* peak because :mod:`dmf.deploy.harness` runs each configuration in its
    own subprocess. Read in the same process that ran the iterations.

    Returns:
        Peak resident set size, mebibytes. Linux reports ``ru_maxrss`` in kibibytes.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _cuda_free_bytes() -> int:
    """Return free device memory, bytes, or 0 when there is no CUDA device.

    Returns:
        Free bytes on device 0.
    """
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _batches(windows: FloatArray, batch_size: int) -> list[FloatArray]:
    """Slice windows into fixed-size batches, cycling if there are not enough.

    Several distinct batches rather than one reused buffer, so that a runtime cannot cache
    an output for an input it has already seen and so that the input is not guaranteed to
    be in L1 on every iteration.

    Args:
        windows: Input windows, shape ``(n_windows, L, C_in)``, dimensionless.
        batch_size: Windows per batch.

    Returns:
        At least one contiguous float32 batch of shape ``(batch_size, L, C_in)``.

    Raises:
        ValueError: If ``windows`` is not rank 3 or ``batch_size`` is not positive.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape (n, L, C_in), got {windows.shape}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    n = int(windows.shape[0])
    n_batches = max(1, n // batch_size)
    out: list[FloatArray] = []
    for i in range(n_batches):
        idx = [(i * batch_size + j) % n for j in range(batch_size)]
        out.append(np.ascontiguousarray(windows[idx], dtype=np.float32))
    return out


def _stats(samples_s: list[float]) -> tuple[float, float, float, float, float]:
    """Reduce per-iteration wall times to the reported latency statistics.

    Percentiles are taken over the full sample vector rather than from a running estimate,
    which is why every iteration's time is stored.

    Args:
        samples_s: Per-iteration wall time, seconds.

    Returns:
        ``(p50, p90, p99, mean, std)``, milliseconds.
    """
    arr = np.asarray(samples_s, dtype=np.float64) * 1e3
    p50, p90, p99 = (float(v) for v in np.percentile(arr, _PERCENTILES))
    return p50, p90, p99, float(arr.mean()), float(arr.std(ddof=0))


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
        The measured result. ``providers_realized`` holds the torch backend and the device
        actually used, so a PyTorch row and an ORT row carry the same kind of evidence.

    Raises:
        ValueError: If ``cfg.device`` is ``"cuda"`` but no CUDA device is available, or if
            ``cfg.device`` is ``"cuda"`` and ``cfg.synchronize`` is False. The second
            refusal is the point: without a synchronise, the timed region ends when the
            kernels have been *queued*, and the harness would report a few microseconds for
            work that has not happened. Producing that number and leaving a reader to
            notice it is impossible is not an option the harness offers.
    """
    if cfg.device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {cfg.device!r}")
    if cfg.device == "cuda":
        # Checked before the availability probe on purpose: refusing an un-synchronised
        # CUDA timing is a property of the request, not of the hardware, so it has to
        # raise for that reason on a machine with no GPU too -- otherwise the refusal is
        # only testable where it is least likely to be tested.
        if not cfg.synchronize:
            raise ValueError(
                "device='cuda' with synchronize=False would time kernel *launches*, not "
                "kernel completions, and report an impossibly fast result; "
                "un-synchronised CUDA timing is refused"
            )
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but no CUDA device is available")

    disable_tf32()
    torch.set_num_threads(cfg.intra_op_threads)
    device = torch.device(cfg.device)
    model.eval()
    model.to(device)
    runnable: Callable[[Tensor], Tensor] = torch.compile(model) if compile_model else model
    batches = _batches(windows, cfg.batch_size)
    on_cuda = cfg.device == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    def _one(batch: FloatArray) -> None:
        """Run one host-to-host iteration: NumPy in, NumPy out."""
        tensor = torch.from_numpy(batch).to(device)
        out = runnable(tensor)
        # .cpu() on CUDA both materialises the result on the host and blocks; on CPU it is
        # a no-op view. Either way the iteration ends with the forecast in host memory.
        out.detach().cpu().numpy()

    with torch.no_grad():
        for i in range(cfg.warmup_iters):
            _one(batches[i % len(batches)])
        if on_cuda and cfg.synchronize:
            torch.cuda.synchronize(device)

        samples_s: list[float] = []
        for i in range(cfg.timed_iters):
            batch = batches[i % len(batches)]
            start = time.perf_counter()
            _one(batch)
            if on_cuda and cfg.synchronize:
                torch.cuda.synchronize(device)
            samples_s.append(time.perf_counter() - start)

    p50, p90, p99, mean, std = _stats(samples_s)
    peak_device = float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_MIB if on_cuda else 0.0
    backend = "torch-compile" if compile_model else "torch-eager"
    return BenchResult(
        label=label,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        throughput_windows_s=cfg.batch_size / (mean / 1e3),
        peak_host_mem_mb=_peak_host_mem_mb(),
        peak_device_mem_mb=peak_device,
        providers_realized=(f"{backend}:{cfg.device}",),
        intra_op_threads=torch.get_num_threads(),
        tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        config=cfg,
        env=environment_stamp(),
    )


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
        ValueError: If ``providers`` is empty, or if the first-priority provider is absent
            from ``session.get_providers()`` after construction. Checked against the
            realized session and **not** against ``ort.get_available_providers()``, which
            reports build-time capability: the wheel used here advertises all three
            providers on a machine where two of them would fall back to CPU without the
            preload in :mod:`dmf.deploy.providers`.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"no exported graph at {onnx_path}")
    if not providers:
        raise ValueError("providers must name at least one execution provider")

    import onnxruntime as ort

    preload_gpu_libraries()
    disable_tf32()
    wants_gpu = bool(GPU_PROVIDERS & set(providers))
    free_before = _cuda_free_bytes() if wants_gpu else 0

    options = ort.SessionOptions()
    options.intra_op_num_threads = cfg.intra_op_threads
    options.inter_op_num_threads = cfg.intra_op_threads
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=session_providers(providers)
    )
    realized = tuple(str(p) for p in session.get_providers())
    if providers[0] not in realized:
        raise ValueError(
            f"requested {providers[0]!r} but the session realized {list(realized)}; ORT "
            f"falls back silently and still returns a working session, so reporting this "
            f"row would publish {realized[0] if realized else 'no'} timings under the "
            f"{providers[0]} label"
        )

    torch.set_num_threads(cfg.intra_op_threads)
    batches = _batches(windows, cfg.batch_size)
    on_gpu = bool(GPU_PROVIDERS & set(realized))
    sync = on_gpu and cfg.synchronize and torch.cuda.is_available()
    free_min = free_before

    for i in range(cfg.warmup_iters):
        session.run([], {INPUT_NAME: batches[i % len(batches)]})
    if sync:
        torch.cuda.synchronize()
    if wants_gpu:
        free_min = min(free_min, _cuda_free_bytes())

    samples_s = []
    for i in range(cfg.timed_iters):
        batch = batches[i % len(batches)]
        start = time.perf_counter()
        # `run` blocks until the outputs are materialised as host NumPy arrays, so the
        # copy back is itself the synchronisation on a GPU provider. The explicit
        # synchronise below is belt-and-braces and costs nothing at these iteration counts.
        session.run([], {INPUT_NAME: batch})
        if sync:
            torch.cuda.synchronize()
        samples_s.append(time.perf_counter() - start)
    if wants_gpu:
        free_min = min(free_min, _cuda_free_bytes())

    p50, p90, p99, mean, std = _stats(samples_s)
    peak_device = max(0.0, float(free_before - free_min) / BYTES_PER_MIB) if wants_gpu else 0.0
    return BenchResult(
        label=label,
        p50_ms=p50,
        p90_ms=p90,
        p99_ms=p99,
        mean_ms=mean,
        std_ms=std,
        throughput_windows_s=cfg.batch_size / (mean / 1e3),
        peak_host_mem_mb=_peak_host_mem_mb(),
        peak_device_mem_mb=peak_device,
        providers_realized=realized,
        intra_op_threads=int(options.intra_op_num_threads),
        tf32=False,
        config=cfg,
        env=environment_stamp(),
    )


def _as_serialisable(result: BenchResult) -> dict[str, Any]:
    """Flatten a result into a JSON-serialisable mapping.

    Args:
        result: The result to flatten.

    Returns:
        Mapping with the latency fields in milliseconds, the harness settings, and the
        environment stamp.
    """
    payload: dict[str, Any] = asdict(result)
    payload["providers_realized"] = list(result.providers_realized)
    payload["config"] = asdict(result.config)
    payload["env"] = dict(result.env)
    return payload


def write_benchmark_json(results: tuple[BenchResult, ...], path: Path) -> Path:
    """Write benchmark results, with environment stamps, to JSON.

    Args:
        results: The results to write.
        path: Destination file under ``results/``.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": [_as_serialisable(result) for result in results]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
