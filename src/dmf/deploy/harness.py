"""One benchmark configuration, run in its own process.

Every configuration -- (model, backend, device, batch size) -- is measured in a **fresh
subprocess** that writes a one-result JSON fragment the parent merges. Three things make
that not merely tidy:

1. ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a high-water mark that is never reset.
   In a single process it would report the largest footprint any configuration so far had
   reached, so the first row would be honest and every later row would inherit it. Per
   process, it is a per-configuration peak.
2. ``torch.compile`` leaves compiled artifacts and guards in the process, and a CUDA
   context, once created, does not go away. A CPU row measured after a CUDA row in the same
   process is measured against a different allocator state than one measured before it.
3. The TensorRT EP builds an engine on first inference. Building it inside the process that
   also holds a torch CUDA context and an ORT CUDA session is the configuration in which
   provider-loading failures are hardest to attribute.

The child entry point is :mod:`dmf.deploy.child`, an importable module invoked with
``python -m``, not a string passed to ``-c``: a traceback from a benchmark child should
name a file and a line like any other.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from dmf.deploy.bench import BenchConfig, BenchResult, benchmark_onnxruntime, benchmark_torch
from dmf.deploy.export_onnx import graph_input_geometry
from dmf.deploy.parity import random_parity_windows
from dmf.deploy.providers import tf32_environment
from dmf.deploy.targets import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_ONNX_DIR,
    load_target,
    onnx_path_for,
    target_by_key,
)
from dmf.typedefs import FloatArray

__all__ = [
    "BACKENDS",
    "BENCH_WINDOW_SEED",
    "DEVICE_BY_BACKEND",
    "PROVIDERS_BY_BACKEND",
    "Backend",
    "BenchJob",
    "bench_result_from_payload",
    "job_from_payload",
    "job_payload",
    "run_job",
    "run_job_subprocess",
]

#: The five configurations Phase 7 compares.
Backend = Literal["torch-eager", "torch-compile", "ort-cpu", "ort-cuda", "ort-trt"]

#: In report order: the two PyTorch rows a deployment would be replacing, then the three
#: ONNX Runtime execution providers.
BACKENDS: tuple[Backend, ...] = (
    "torch-eager",
    "torch-compile",
    "ort-cpu",
    "ort-cuda",
    "ort-trt",
)

#: Execution providers requested per ORT backend, in ORT's priority order. The CPU provider
#: is listed as the fallback on the GPU rows because ORT requires a complete fallback chain
#: for any subgraph a GPU provider cannot take -- which is exactly why the realized
#: provider has to be read back rather than inferred from this table.
PROVIDERS_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "ort-cpu": ("CPUExecutionProvider",),
    "ort-cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "ort-trt": ("TensorrtExecutionProvider", "CPUExecutionProvider"),
}

#: Device implied by each ORT backend. The PyTorch backends take their device from the job,
#: because "PyTorch eager on the CPU" and "PyTorch eager on the GPU" are both rows this
#: study wants.
DEVICE_BY_BACKEND: dict[str, str] = {
    "ort-cpu": "cpu",
    "ort-cuda": "cuda",
    "ort-trt": "cuda",
}

#: Seed for the benchmark input windows. Different from
#: :data:`dmf.deploy.parity.PARITY_SEED` on purpose: the timing draw is not the draw parity
#: was established on, so a graph cannot be timed on exactly the inputs it was verified
#: against. Latency does not depend on the values, but the independence is free.
BENCH_WINDOW_SEED: int = 20260910

#: Distinct windows held in the cycling buffer, dimensionless. Enough that a batch of 32 is
#: drawn from more than one buffer and no iteration reuses the immediately preceding input,
#: small enough that the buffer stays resident.
BENCH_N_WINDOWS: int = 64

#: Seconds a child configuration may take before the parent gives up. Generous because the
#: TensorRT EP builds its engine inside the first warmup iteration.
CHILD_TIMEOUT_S: float = 3600.0


@dataclass(frozen=True)
class BenchJob:
    """One benchmark configuration.

    Attributes:
        target_key: Key of the :class:`dmf.deploy.targets.ModelTarget` being measured.
        backend: Which runtime to measure.
        batch_size: Windows per inference call. 1 is the real-time case, 32 the throughput
            case.
        device: ``"cpu"`` or ``"cuda"``. Ignored for the ORT backends, whose device is
            fixed by :data:`DEVICE_BY_BACKEND`.
        warmup_iters: Untimed iterations.
        timed_iters: Timed iterations.
        intra_op_threads: Pinned CPU thread count.
        onnx_dir: Directory holding exported graphs.
        checkpoint_root: Root of the checkpoint tree, for the PyTorch backends.
        n_windows: Distinct windows in the cycling input buffer.
        window_seed: Seed for the input draw.
    """

    target_key: str
    backend: Backend
    batch_size: int = 1
    device: str = "cpu"
    warmup_iters: int = 200
    timed_iters: int = 2000
    intra_op_threads: int = 1
    onnx_dir: Path = DEFAULT_ONNX_DIR
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    n_windows: int = BENCH_N_WINDOWS
    window_seed: int = BENCH_WINDOW_SEED

    @property
    def resolved_device(self) -> str:
        """Device this job actually runs on, ``"cpu"`` or ``"cuda"``."""
        return DEVICE_BY_BACKEND.get(self.backend, self.device)

    @property
    def label(self) -> str:
        """Row label, e.g. ``"tcn-ort-cpu-b1-t1"`` or ``"tcn-torch-eager-cuda-b1-t1"``.

        The device is in the label for the PyTorch rows only, where it is a free choice;
        the ORT backend names already carry theirs. The thread count is always in it: two
        rows of a thread-sensitivity sweep differ in nothing else, and a label that
        collapsed them would make the two measurements indistinguishable in
        ``latency.json``.
        """
        parts = [self.target_key, self.backend]
        if self.backend not in DEVICE_BY_BACKEND:
            parts.append(self.resolved_device)
        parts.extend([f"b{self.batch_size}", f"t{self.intra_op_threads}"])
        return "-".join(parts)

    def bench_config(self) -> BenchConfig:
        """Return the :class:`dmf.deploy.bench.BenchConfig` this job implies.

        Returns:
            The harness settings, with ``synchronize`` always True. The harness offers no
            way to switch it off for a CUDA run and this is the only construction site, so
            an un-synchronised GPU timing cannot be produced by configuration.
        """
        return BenchConfig(
            warmup_iters=self.warmup_iters,
            timed_iters=self.timed_iters,
            batch_size=self.batch_size,
            device=self.resolved_device,
            synchronize=True,
            intra_op_threads=self.intra_op_threads,
        )


def run_job(job: BenchJob) -> BenchResult:
    """Run one benchmark configuration in the current process.

    The two backend families are given the geometry of their input buffer from different
    places, deliberately. A PyTorch job reads it off the loaded model; an ONNX Runtime job
    reads it off the graph's own declared axes and **never loads the checkpoint**, which is
    both the deployment situation and what keeps a torch model's weights out of the ORT
    row's ``ru_maxrss``.

    Args:
        job: The configuration to measure.

    Returns:
        The measured result.

    Raises:
        ValueError: If ``job.backend`` is not one of :data:`BACKENDS`, or from the
            backend's own refusals -- an unavailable CUDA device, or a requested execution
            provider the session did not realize.
        FileNotFoundError: If the graph or checkpoint the job needs is absent.
    """
    if job.backend not in BACKENDS:
        raise ValueError(f"unknown backend {job.backend!r}; known backends are {list(BACKENDS)}")
    target = target_by_key(job.target_key)
    cfg = job.bench_config()

    if job.backend in {"torch-eager", "torch-compile"}:
        model = load_target(target, checkpoint_root=job.checkpoint_root)
        windows = _windows(job, model.lookback, model.n_input_channels)
        return benchmark_torch(
            model,
            windows,
            cfg,
            job.label,
            compile_model=job.backend == "torch-compile",
        )

    onnx_path = onnx_path_for(target, job.onnx_dir)
    lookback, n_channels = graph_input_geometry(onnx_path)
    windows = _windows(job, lookback, n_channels)
    return benchmark_onnxruntime(
        onnx_path,
        windows,
        cfg,
        job.label,
        providers=PROVIDERS_BY_BACKEND[job.backend],
    )


def _windows(job: BenchJob, lookback: int, n_input_channels: int) -> FloatArray:
    """Draw the input windows one job cycles through.

    Standard normal, as in the parity check and for the same reason: the windows reaching
    a forecaster are de-meaned and scaled, so that is the space it sees. The values do not
    affect latency for any operator in these graphs -- there is no data-dependent control
    flow -- but drawing them from the right space costs nothing and removes the question.

    Args:
        job: The job, for the buffer size and seed.
        lookback: Input window length ``L``, samples.
        n_input_channels: Input channel count ``C_in``.

    Returns:
        Windows, shape ``(n_windows, L, C_in)``, dimensionless.
    """
    return random_parity_windows(
        lookback,
        n_input_channels,
        n_windows=max(job.n_windows, job.batch_size),
        seed=job.window_seed,
    )


def job_payload(job: BenchJob) -> dict[str, Any]:
    """Serialise a job to a JSON-compatible mapping.

    Args:
        job: The job.

    Returns:
        Mapping with ``Path`` fields rendered as strings.
    """
    payload: dict[str, Any] = {
        "target_key": job.target_key,
        "backend": job.backend,
        "batch_size": job.batch_size,
        "device": job.device,
        "warmup_iters": job.warmup_iters,
        "timed_iters": job.timed_iters,
        "intra_op_threads": job.intra_op_threads,
        "onnx_dir": str(job.onnx_dir),
        "checkpoint_root": str(job.checkpoint_root),
        "n_windows": job.n_windows,
        "window_seed": job.window_seed,
    }
    return payload


def job_from_payload(payload: dict[str, Any]) -> BenchJob:
    """Rebuild a job from :func:`job_payload`'s output.

    Args:
        payload: The serialised job.

    Returns:
        The job.
    """
    backend: Backend = payload["backend"]
    return BenchJob(
        target_key=str(payload["target_key"]),
        backend=backend,
        batch_size=int(payload["batch_size"]),
        device=str(payload["device"]),
        warmup_iters=int(payload["warmup_iters"]),
        timed_iters=int(payload["timed_iters"]),
        intra_op_threads=int(payload["intra_op_threads"]),
        onnx_dir=Path(str(payload["onnx_dir"])),
        checkpoint_root=Path(str(payload["checkpoint_root"])),
        n_windows=int(payload["n_windows"]),
        window_seed=int(payload["window_seed"]),
    )


def bench_result_from_payload(payload: dict[str, Any]) -> BenchResult:
    """Rebuild a :class:`dmf.deploy.bench.BenchResult` from its serialised form.

    Args:
        payload: One element of the ``results`` list written by
            :func:`dmf.deploy.bench.write_benchmark_json`.

    Returns:
        The result.
    """
    config = BenchConfig(**payload["config"])
    fields = {k: v for k, v in payload.items() if k not in {"config", "providers_realized", "env"}}
    return BenchResult(
        providers_realized=tuple(str(p) for p in payload["providers_realized"]),
        config=config,
        env={str(k): str(v) for k, v in payload["env"].items()},
        **fields,
    )


def run_job_subprocess(
    job: BenchJob,
    python: str | None = None,
    timeout_s: float = CHILD_TIMEOUT_S,
) -> BenchResult:
    """Run one benchmark configuration in a fresh child process.

    Args:
        job: The configuration to measure.
        python: Interpreter to spawn. Defaults to the current one, so a venv is inherited
            without being named.
        timeout_s: Seconds before the child is killed.

    Returns:
        The result the child measured and wrote.

    Raises:
        RuntimeError: If the child exits non-zero or writes no fragment. The child's
            stderr is included: a refusal to report a mislabelled provider arrives that
            way, and swallowing it would turn a deliberate refusal into a missing row.
    """
    with tempfile.TemporaryDirectory(prefix="dmf-bench-") as tmp:
        job_path = Path(tmp) / "job.json"
        out_path = Path(tmp) / "result.json"
        job_path.write_text(json.dumps(job_payload(job)), encoding="utf-8")
        completed = subprocess.run(
            [
                python or sys.executable,
                "-m",
                "dmf.deploy.child",
                "--job",
                str(job_path),
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            # NVIDIA_TF32_OVERRIDE is read when the CUDA context is created, so it has to
            # be in the child's environment from the start; setting it inside the child
            # would already be too late if anything had touched the device first.
            env=tf32_environment(),
        )
        if completed.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                f"benchmark child for {job.label} exited {completed.returncode}\n"
                f"{completed.stderr.strip()}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    result = bench_result_from_payload(payload["results"][0])
    return replace(result, label=job.label)
