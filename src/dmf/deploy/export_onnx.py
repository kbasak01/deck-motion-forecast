"""ONNX export.

Opset 18, **dynamic batch axis only**. The sequence length is fixed deliberately: a
dynamic sequence axis prevents the runtime from selecting specialised kernels and buys
nothing here, since the lookback is a fixed property of the trained model.
"""

from pathlib import Path

from torch import Tensor

from dmf.models.base import BaseForecaster

__all__ = ["export_model"]


def export_model(
    model: BaseForecaster,
    sample_input: Tensor,
    path: Path,
    opset: int = 18,
) -> Path:
    """Export a trained model to ONNX.

    Args:
        model: The trained model. Placed in eval mode and moved to CPU before tracing, so
            that the exported graph does not carry device-specific nodes.
        sample_input: A representative input window, shape ``(B, L, C_in)``,
            dimensionless. Its ``L`` and ``C_in`` become fixed in the graph; ``B`` is made
            dynamic.
        path: Destination ``.onnx`` file. Lives under ``artifacts/`` and is gitignored.
        opset: ONNX opset version.

    Returns:
        The path written.

    Raises:
        ValueError: If ``sample_input`` does not match the model's declared ``L`` and
            ``C_in``.
    """
    raise NotImplementedError
