"""DLinear -- series decomposition followed by one linear layer per component.

Included specifically as a model that might beat the transformer. On long-horizon
time-series forecasting, linear models are repeatedly competitive with attention-based
ones, and reporting that honestly is the point of having it here.

This file holds the **canonical, channel-independent** DLinear of the original paper: roll
is forecast from roll history alone, pitch from pitch, heave from heave. Under the P3
geometry the target set is all six channels, so the model is handed all of them -- what it
cannot do is let one output channel read another's history, which AR(p) does. A
DLinear-vs-AR gap therefore confounds architecture with per-output information set, and
that confound is *not* closed by a multi-channel DLinear: at the P3 geometry (L=200, H=150,
C_in=C_out=6) such a variant carries 2 161 800 fitted parameters against this one's 60 300,
so it trades an information-set confound for a 36x capacity confound. The information set
is isolated on the AR side instead, where it is free of both --
``configs/model/ar_attitude_only.yaml`` is AR(40) on the three attitude channels, matched
to ``ar20`` in ridge, solver, moments and -- since 40 x 3 = 20 x 6 = 120 features -- in
parameter count too (docs/protocol.md P3).

This model's own optimisation gap is measured rather than assumed: ``dlinear_ols``
(:mod:`dmf.models.dlinear_ols`) solves the identical map in closed form and is reported as
its own row, so the difference between the two is the shortfall of 60 epochs of Adam and
not part of any architecture comparison.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from dmf.models.base import BaseForecaster
from dmf.train.registry import register_model

__all__ = ["DLinear", "moving_average", "series_decompose"]


def moving_average(x: Tensor, kernel_size: int) -> Tensor:
    """Compute a centred moving average along the time axis.

    Edges are **replicated**, not zero-padded. Zero padding would pull the trend toward
    zero at both ends of every window, and since the last lookback sample is exactly the
    quantity persistence forecasts from, a depressed trend there biases the remainder
    channel at the one sample that matters most.

    Args:
        x: Input windows, shape ``(B, L, C)``, dimensionless.
        kernel_size: Averaging window, samples. Odd values keep the output centred.

    Returns:
        Smoothed series, shape ``(B, L, C)``, dimensionless. Edges are handled by
        replicating the boundary values, so the output length matches the input.

    Raises:
        ValueError: If ``kernel_size`` is not positive or exceeds ``L``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, L, C), got {tuple(x.shape)}")
    lookback = int(x.shape[1])
    if not 1 <= kernel_size <= lookback:
        raise ValueError(f"kernel_size must be in [1, L={lookback}], got {kernel_size}")
    if kernel_size == 1:
        return x.clone()
    front = (kernel_size - 1) // 2
    back = kernel_size // 2
    padded = F.pad(x.transpose(1, 2), (front, back), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return smoothed.transpose(1, 2)


def series_decompose(x: Tensor, kernel_size: int) -> tuple[Tensor, Tensor]:
    """Split a series into a moving-average trend and a remainder.

    Args:
        x: Input windows, shape ``(B, L, C)``, dimensionless.
        kernel_size: Trend-extraction window, samples.

    Returns:
        Tuple ``(trend, remainder)``, each of shape ``(B, L, C)``, dimensionless, summing
        back to ``x``.
    """
    trend = moving_average(x, kernel_size)
    return trend, x - trend


@register_model("dlinear")
class DLinear(BaseForecaster):
    """Decomposition linear forecaster.

    Decomposes each channel into trend and remainder, maps each component from ``L`` to
    ``H`` with its own linear layer, and sums the two. Channel-independent by default:
    one shared pair of ``(L -> H)`` maps applied to every channel.

    Only the first ``C_out`` input channels are read, which under the P2-D4 prefix rule are
    exactly the target channels. That is the paper's model and it is the honest one to
    label ``dlinear``.
    """

    FIT_KIND = "sgd"

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

        Raises:
            ValueError: If ``kernel_size`` is not in ``[1, lookback]``.
        """
        super().__init__(lookback, max_horizon, n_input_channels, n_target_channels, n_quantiles)
        if not 1 <= kernel_size <= lookback:
            raise ValueError(f"kernel_size must be in [1, lookback={lookback}], got {kernel_size}")
        self.kernel_size = kernel_size
        self.individual = individual
        self._per_channel_out = max_horizon * max(n_quantiles, 1)
        n_maps = n_target_channels if individual else 1
        self.trend = nn.ModuleList(
            nn.Linear(lookback, self._per_channel_out) for _ in range(n_maps)
        )
        self.remainder = nn.ModuleList(
            nn.Linear(lookback, self._per_channel_out) for _ in range(n_maps)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forecast by summing the trend and remainder projections.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        targets = x[:, :, : self.n_target_channels]
        trend, remainder = series_decompose(targets, self.kernel_size)
        trend = trend.transpose(1, 2)
        remainder = remainder.transpose(1, 2)
        if self.individual:
            parts = [
                self.trend[c](trend[:, c, :]) + self.remainder[c](remainder[:, c, :])
                for c in range(self.n_target_channels)
            ]
            out = torch.stack(parts, dim=1)
        else:
            out = self.trend[0](trend) + self.remainder[0](remainder)
        # out: (B, C_out, H * max(Q, 1)) -> (B, H, C_out[, Q])
        if self.n_quantiles > 0:
            fanned = out.view(
                x.shape[0], self.n_target_channels, self.max_horizon, self.n_quantiles
            )
            return fanned.permute(0, 2, 1, 3)
        return out.transpose(1, 2)
