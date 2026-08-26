"""Numerical parity between the PyTorch model and its exported ONNX graph.

Gate 7 requires ``max_abs_err < 1e-4`` in FP32 over 1000 random windows. Run before any
timing measurement: benchmarking a graph that has not been parity-checked measures how
fast the wrong answer arrives.
"""

from dataclasses import dataclass
from pathlib import Path

from dmf.models.base import BaseForecaster
from dmf.typedefs import FloatArray

__all__ = ["ParityResult", "check_parity"]


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a parity check.

    Attributes:
        max_abs_err: Largest absolute elementwise difference between the PyTorch and ONNX
            outputs, dimensionless (the models operate on normalised inputs).
        mean_abs_err: Mean absolute elementwise difference, dimensionless.
        n_windows: Number of random windows compared.
        tolerance: The threshold applied, dimensionless.
        passed: Whether ``max_abs_err < tolerance``.
    """

    max_abs_err: float
    mean_abs_err: float
    n_windows: int
    tolerance: float
    passed: bool


def check_parity(
    model: BaseForecaster,
    onnx_path: Path,
    windows: FloatArray,
    tolerance: float = 1e-4,
) -> ParityResult:
    """Compare PyTorch and ONNX Runtime outputs over a batch of random windows.

    Args:
        model: The source PyTorch model, in eval mode.
        onnx_path: The exported graph.
        windows: Random input windows, shape ``(n_windows, L, C_in)``, dimensionless.
            Gate 7 uses 1000. Random rather than corpus-drawn, so that the check covers
            the input space rather than only the region the corpus happens to occupy.
        tolerance: Maximum permitted absolute error, dimensionless.

    Returns:
        The parity result. Returned rather than asserted, so that the caller can record
        the measured error in the benchmark artifact even on a pass.

    Raises:
        FileNotFoundError: If ``onnx_path`` does not exist.
        ValueError: If ``windows`` does not match the model's declared input shape.
    """
    raise NotImplementedError
