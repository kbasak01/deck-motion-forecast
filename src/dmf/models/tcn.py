"""Temporal convolutional network -- WaveNet-style dilated causal convolutions.

**Receptive field.** For a stack of residual blocks each containing two causal
convolutions of kernel size ``k`` at dilation ``d``, the receptive field is::

    RF = 1 + 2*(k - 1)*sum(dilations)

With ``k = 3`` and ``dilations = [1, 2, 4, 8, 16, 32]`` (sum 63)::

    RF = 1 + 2*2*63 = 253

which covers the default ``lookback = 200``. This must hold: a receptive field shorter
than the lookback means the model silently cannot see the start of its own input window,
which shows up nowhere in the loss curve. The arithmetic is asserted in
``tests/test_windows.py`` against :func:`receptive_field`, not merely stated here.
"""

from torch import Tensor, nn

from dmf.models.base import BaseForecaster

__all__ = ["TCN", "TemporalBlock", "receptive_field"]


def receptive_field(kernel_size: int, dilations: tuple[int, ...], convs_per_block: int = 2) -> int:
    """Compute the receptive field of a dilated causal convolution stack.

    ``RF = 1 + convs_per_block*(kernel_size - 1)*sum(dilations)``.

    Args:
        kernel_size: Convolution kernel size ``k``, samples.
        dilations: Dilation factor of each residual block, in stack order.
        convs_per_block: Causal convolutions per residual block.

    Returns:
        Receptive field, samples.

    Raises:
        ValueError: If ``kernel_size`` is less than 2, or any dilation is not positive.
    """
    raise NotImplementedError


class TemporalBlock(nn.Module):
    """One residual block: two weight-normalised dilated causal convolutions.

    Causality is enforced by left-padding each convolution by ``(k-1)*d`` samples and
    trimming the same number from the right. Padding symmetrically instead would let each
    output step see future samples -- a leak that is invisible in training and inflates
    every short-horizon metric.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        """Configure the block.

        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            kernel_size: Convolution kernel size, samples.
            dilation: Dilation factor, samples between kernel taps.
            dropout: Dropout probability, in [0, 1).
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Feature map, shape ``(B, C_feat, L)`` -- channels-first, the layout torch
                convolutions expect, transposed once at the stack boundary rather than in
                every block.

        Returns:
            Feature map, shape ``(B, out_channels, L)``.
        """
        raise NotImplementedError


class TCN(BaseForecaster):
    """Dilated causal convolution stack with a direct multi-horizon head."""

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        n_quantiles: int = 0,
    ) -> None:
        """Configure the model and verify its receptive field.

        Args:
            lookback: Input window length ``L``, samples.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            n_filters: Feature channels per residual block.
            kernel_size: Convolution kernel size, samples.
            dilations: Dilation factor per residual block, in stack order.
            dropout: Dropout probability, in [0, 1).
            n_quantiles: Quantile count ``Q``, or 0 for a point head.

        Raises:
            ValueError: If ``receptive_field(kernel_size, dilations) < lookback``. Checked
                at construction so that a misconfigured stack fails immediately rather
                than training to a quietly degraded result.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Encode the lookback with the dilated stack and project to the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        raise NotImplementedError
