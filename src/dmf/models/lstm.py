"""LSTM forecaster -- recurrent encoder with a direct multi-horizon head.

The recurrence encodes the lookback; the head projects the final hidden state to all ``H``
steps at once. The decoder is deliberately not recurrent: rolling out the horizon would
compound error and would produce a loop in the exported ONNX graph.
"""

from typing import ClassVar

from torch import Tensor, nn

from dmf.models.base import BaseForecaster
from dmf.models.heads import HeadKind, clamp_log_variance
from dmf.train.registry import register_model

__all__ = ["LSTMForecaster"]


@register_model("lstm")
class LSTMForecaster(BaseForecaster):
    """Two-layer LSTM encoder with a linear direct multi-horizon head."""

    FIT_KIND = "sgd"
    #: All three heads: the projection widens by K and the reshape branch handles each.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("point", "quantile", "gaussian")

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
        head: HeadKind | None = None,
    ) -> None:
        """Configure the model.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            hidden_size: LSTM hidden state width.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between layers, in [0, 1). ``nn.LSTM``
                applies it to the output of every layer but the last, so a single-layer
                stack is built with dropout 0 rather than emitting a torch warning about a
                setting that would have no effect.
            n_quantiles: Quantile count ``Q``, or 0 for a point head.
            head: Output head kind -- ``"point"``, ``"quantile"`` or ``"gaussian"``.
                Defaults to ``"quantile"`` when ``n_quantiles > 0`` and ``"point"``
                otherwise, so a point build is unchanged and a Gaussian head is asked for
                by name (``docs/protocol.md`` P5-D1).

        Raises:
            ValueError: If ``num_layers`` is not positive.
        """
        super().__init__(
            lookback, max_horizon, n_input_channels, n_target_channels, n_quantiles, head
        )
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self._head_width = max_horizon * n_target_channels * self.n_output_params
        self.head = nn.Linear(hidden_size, self._head_width)

    def _predict(self, x: Tensor) -> Tensor:
        """Encode the lookback and project the final hidden state to the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` for a point head, ``(B, H, C_out, Q)`` for
            a quantile head, or ``(B, H, C_out, 2)`` carrying ``(mean, log_var)`` for a
            Gaussian one. Dimensionless.
        """
        _, (h_n, _) = self.lstm(x)
        out: Tensor = self.head(h_n[-1])
        # out: (B, H * C_out * K) -> (B, H, C_out[, K])
        if self.head_kind == "point":
            return out.view(x.shape[0], self.max_horizon, self.n_target_channels)
        fanned = out.view(
            x.shape[0], self.max_horizon, self.n_target_channels, self.n_output_params
        )
        return clamp_log_variance(fanned) if self.head_kind == "gaussian" else fanned
