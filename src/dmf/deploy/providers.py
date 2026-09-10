"""Making ONNX Runtime's GPU execution providers actually loadable.

``onnxruntime.get_available_providers()`` reports what the wheel was **built** with, not
what will load in this process. Both GPU providers need their shared libraries resident
before ``InferenceSession`` is constructed, and when they are not, ORT logs an error,
falls back to the CPU provider and **still returns a working session** -- which is how a
row labelled "CUDA" or "TensorRT" comes to hold CPU timings. Nothing downstream raises,
and the only symptom is a GPU that is suspiciously as fast as the CPU.

Measured on the reference machine (RTX A4000, driver 595.95, onnxruntime-gpu 1.29.0,
torch 2.13.0+cu130, tensorrt-cu13 10.16.1.11), with **no** ``LD_LIBRARY_PATH`` set:

- ``import torch`` pulls in ``libcublas``/``libcublasLt``/``libcudnn`` -> CUDA EP loads;
- ``import tensorrt`` pulls in ``libnvinfer.so.10`` -> TensorRT EP loads;
- with both imported first, session creation realizes ``CUDAExecutionProvider`` and
  ``TensorrtExecutionProvider`` respectively;
- without them it realizes ``CPUExecutionProvider`` for all three requests.

So the fix is an import, not an environment variable. That matters here: ``make bench``
has to work from a clean shell (the same requirement commit 70130a1 fixed for the other
targets), and an ``LD_LIBRARY_PATH`` set in one developer's shell is exactly the kind of
state that makes a benchmark unreproducible.

The complementary half of the guard lives in :func:`dmf.deploy.bench.benchmark_onnxruntime`,
which reads ``session.get_providers()`` back and refuses to report a result whose requested
provider is not in it.
"""

import contextlib

__all__ = [
    "CUDA_PROVIDER_OPTIONS",
    "PRELOADED_LIBRARY_IMPORTS",
    "PROVIDER_OPTIONS",
    "TF32_OVERRIDE_ENV",
    "disable_tf32",
    "preload_gpu_libraries",
    "session_providers",
    "tf32_environment",
]

#: The imports whose side effect is the preload, in the order they are attempted, with the
#: provider each one enables. Named here so the reason for an otherwise unused import is
#: readable at the call site.
PRELOADED_LIBRARY_IMPORTS: tuple[tuple[str, str], ...] = (
    ("torch", "CUDAExecutionProvider"),
    ("tensorrt", "TensorrtExecutionProvider"),
)


def preload_gpu_libraries() -> None:
    """Import the packages whose shared libraries ORT's GPU providers need.

    Call before **every** ``InferenceSession`` construction, including the CPU-provider
    ones: it is cheap after the first call (both imports are cached in ``sys.modules``) and
    a call site that only preloads "when using the GPU" is one edit away from being wrong.

    ``torch`` is a hard dependency and is imported unguarded. ``tensorrt`` is imported
    best-effort: a machine without it must still be able to benchmark the CPU and CUDA
    providers, and the TensorRT row is then absent rather than silently CPU -- the check in
    :func:`dmf.deploy.bench.benchmark_onnxruntime` is what turns the missing library into a
    refusal instead of a mislabelled number.

    Returns:
        None. The effect is entirely in the process's loaded shared libraries.
    """
    import torch  # noqa: F401  (import side effect: libcublas/libcublasLt/libcudnn)

    # Absent TensorRT is a supported configuration; a mislabelled TensorRT row is not, and
    # that is refused downstream by reading the realized provider back off the session.
    with contextlib.suppress(ImportError):
        import tensorrt  # noqa: F401  (import side effect: libnvinfer.so.10)


#: NVIDIA's process-wide TF32 kill switch. Read by cuBLAS, cuDNN **and TensorRT**, and the
#: only lever that reaches all three: ORT 1.29's TensorRT EP exposes no TF32 provider option
#: at all (``trt_tf32_enable`` is rejected as an invalid option, and the shipped
#: ``libonnxruntime_providers_tensorrt.so`` contains no ``tf32`` symbol), while TensorRT's
#: builder enables ``BuilderFlag::kTF32`` by default. It must be set **before** the CUDA
#: context is created, which is why it is an environment variable and not an API call.
TF32_OVERRIDE_ENV: str = "NVIDIA_TF32_OVERRIDE"

#: Provider options passed to the CUDA execution provider. ``use_tf32`` defaults to 1 in
#: ORT, so this is a change, not a restatement of the default.
CUDA_PROVIDER_OPTIONS: dict[str, str] = {"use_tf32": "0"}

#: Options by provider name. The TensorRT EP is absent deliberately: it has no TF32 option
#: to pass, and passing an unknown one makes ORT log an error and **fall back to CPU while
#: still returning a working session** -- the P7-D1 failure mode. TensorRT is held to FP32
#: by :data:`TF32_OVERRIDE_ENV` instead.
PROVIDER_OPTIONS: dict[str, dict[str, str]] = {"CUDAExecutionProvider": CUDA_PROVIDER_OPTIONS}


def tf32_environment() -> dict[str, str]:
    """Return the environment a benchmark child must run under.

    Args:
        None.

    Returns:
        A copy of this process's environment with :data:`TF32_OVERRIDE_ENV` set to ``"0"``.
        Passed to every subprocess by :func:`dmf.deploy.harness.run_job_subprocess`, because
        the variable is read when the CUDA context is created and a child that inherited an
        unset value would build TF32 kernels regardless of anything the parent did.
    """
    import os

    return {**os.environ, TF32_OVERRIDE_ENV: "0"}


def disable_tf32() -> None:
    """Force full FP32 arithmetic on every GPU path in this process.

    **Why this exists.** On an Ampere card TF32 is the default for convolutions and matrix
    multiplies in cuBLAS, cuDNN and TensorRT: a 10-bit mantissa instead of 23. Measured on
    the exported graphs of this project, against the CPU execution provider as reference,
    that costs three orders of magnitude of accuracy -- ``tcn`` 4.9e-02 against a scaled
    tolerance of 9.1e-03, i.e. **the GPU rows were failing this project's own parity bar**
    while the CPU rows ran full FP32. Timing an approximation against an exact computation
    and reporting the ratio as a CPU-versus-GPU result is not a comparison
    (``docs/protocol.md`` P7-D9).

    Three switches, because no one of them covers the field: the environment variable
    reaches TensorRT and every CUDA library, the ORT provider option reaches the CUDA EP
    explicitly (belt and braces, and it is visible in the artifact), and the torch flags
    reach the PyTorch backends.

    Call **before** any CUDA context is created -- before session construction and before
    the first tensor reaches the device.

    Returns:
        None. The effect is in this process's environment and torch's global flags.
    """
    import os

    import torch

    os.environ[TF32_OVERRIDE_ENV] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The string form of the same request, which also covers matmul paths that consult it.
    torch.set_float32_matmul_precision("highest")


def session_providers(
    providers: tuple[str, ...],
) -> list[str | tuple[str, dict[str, str]]]:
    """Attach each provider's options, for ``InferenceSession(providers=...)``.

    One construction site, so a session created anywhere in this package carries the same
    arithmetic. A provider with no entry in :data:`PROVIDER_OPTIONS` is passed as a bare
    name, which is what ORT expects.

    Args:
        providers: Provider names in priority order.

    Returns:
        The list ORT takes, with options attached where there are any.
    """
    return [
        (name, PROVIDER_OPTIONS[name]) if name in PROVIDER_OPTIONS else name for name in providers
    ]
