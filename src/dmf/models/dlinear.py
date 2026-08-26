"""DLinear -- series decomposition followed by one linear layer per component.

Included specifically as a model that might beat the transformer. On long-horizon
time-series forecasting, linear models are repeatedly competitive with attention-based
ones, and reporting that honestly is the point of having it here.
"""

from torch import Tensor

from dmf.models.base import BaseForecaster

__all__ = ["DLinear", "moving_average", "series_decompose"]


def moving_average(x: Tensor, kernel_size: int) -> Tensor:
    """Compute a centred moving average along the time axis.

    Args:
        x: Input windows, shape ``(B, L, C)``, dimensionless.
        kernel_size: Averaging window, samples. Odd values keep the output centred.

    Returns:
        Smoothed series, shape ``(B, L, C)``, dimensionless. Edges are handled by
        replicating the boundary values, so the output length matches the input.

    Raises:
        ValueError: If ``kernel_size`` is not positive or exceeds ``L``.
    """
    raise NotImplementedError


def series_decompose(x: Tensor, kernel_size: int) -> tuple[Tensor, Tensor]:
    """Split a series into a moving-average trend and a remainder.

    Args:
        x: Input windows, shape ``(B, L, C)``, dimensionless.
        kernel_size: Trend-extraction window, samples.

    Returns:
        Tuple ``(trend, remainder)``, each of shape ``(B, L, C)``, dimensionless, summing
        back to ``x``.
    """
    raise NotImplementedError


class DLinear(BaseForecaster):
    """Decomposition linear forecaster.

    Decomposes each channel into trend and remainder, maps each component from ``L`` to
    ``H`` with its own linear layer, and sums the two. Channel-independent by default:
    one shared pair of ``(L -> H)`` maps applied to every channel.
    """

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        kernel_size: int = 25,
        individual: bool = False,
        n_quantiles: int = 0,
    ) -> None:
        """Configure the model.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            kernel_size: Trend-extraction window, samples.
            individual: If True, learn a separate linear map per channel instead of
                sharing one across channels.
            n_quantiles: Quantile count ``Q``, or 0 for a point head.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Forecast by summing the trend and remainder projections.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        raise NotImplementedError
