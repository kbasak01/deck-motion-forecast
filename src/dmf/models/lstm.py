"""LSTM forecaster -- recurrent encoder with a direct multi-horizon head.

The recurrence encodes the lookback; the head projects the final hidden state to all ``H``
steps at once. The decoder is deliberately not recurrent: rolling out the horizon would
compound error and would produce a loop in the exported ONNX graph.
"""

from torch import Tensor

from dmf.models.base import BaseForecaster

__all__ = ["LSTMForecaster"]


class LSTMForecaster(BaseForecaster):
    """Two-layer LSTM encoder with a linear direct multi-horizon head."""

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_quantiles: int = 0,
    ) -> None:
        """Configure the model.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            hidden_size: LSTM hidden state width.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between layers, in [0, 1).
            n_quantiles: Quantile count ``Q``, or 0 for a point head.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Encode the lookback and project the final hidden state to the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        raise NotImplementedError
