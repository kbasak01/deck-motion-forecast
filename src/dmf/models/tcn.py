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

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from dmf.models.base import BaseForecaster
from dmf.train.registry import register_model

__all__ = ["TCN", "TemporalBlock", "receptive_field"]


def receptive_field(kernel_size: int, dilations: Sequence[int], convs_per_block: int = 2) -> int:
    """Compute the receptive field of a dilated causal convolution stack.

    ``RF = 1 + convs_per_block*(kernel_size - 1)*sum(dilations)``.

    Args:
        kernel_size: Convolution kernel size ``k``, samples.
        dilations: Dilation factor of each residual block, in stack order. Accepts any
            sequence, because YAML delivers a list where the architecture is written as a
            tuple.
        convs_per_block: Causal convolutions per residual block.

    Returns:
        Receptive field, samples.

    Raises:
        ValueError: If ``kernel_size`` is less than 2, or any dilation is not positive.
    """
    if kernel_size < 2:
        raise ValueError(f"kernel_size must be at least 2, got {kernel_size}")
    dilation_list = [int(d) for d in dilations]
    if any(d < 1 for d in dilation_list):
        raise ValueError(f"dilations must all be positive, got {dilation_list}")
    if convs_per_block < 1:
        raise ValueError(f"convs_per_block must be positive, got {convs_per_block}")
    return 1 + convs_per_block * (kernel_size - 1) * sum(dilation_list)


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
        super().__init__()
        #: Samples of left padding, all of which are trimmed off the right of the output.
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.pad,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=self.pad,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def _causal(self, conv: nn.Module, x: Tensor) -> Tensor:
        """Convolve with symmetric-length output, then drop the samples that see ahead.

        ``nn.Conv1d`` pads both ends, so the last ``self.pad`` output steps are the ones
        whose support extends past the end of the input. Trimming them from the **right**
        is what leaves output step ``t`` a function of inputs ``<= t`` only.

        Args:
            conv: The dilated convolution to apply.
            x: Feature map, shape ``(B, C_in, L)``.

        Returns:
            Feature map, shape ``(B, C_out, L)``.
        """
        out: Tensor = conv(x)
        return out[:, :, : -self.pad] if self.pad > 0 else out

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Feature map, shape ``(B, C_feat, L)`` -- channels-first, the layout torch
                convolutions expect, transposed once at the stack boundary rather than in
                every block.

        Returns:
            Feature map, shape ``(B, out_channels, L)``.
        """
        out = self.drop1(self.relu1(self._causal(self.conv1, x)))
        out = self.drop2(self.relu2(self._causal(self.conv2, out)))
        residual = x if self.downsample is None else self.downsample(x)
        activated: Tensor = self.relu_out(out + residual)
        return activated


@register_model("tcn")
class TCN(BaseForecaster):
    """Dilated causal convolution stack with a direct multi-horizon head."""

    FIT_KIND = "sgd"

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        n_filters: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
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
            dilations: Dilation factor per residual block, in stack order. YAML delivers a
                list; it is coerced to a tuple here so the stored attribute matches the
                documented type.
            dropout: Dropout probability, in [0, 1).
            n_quantiles: Quantile count ``Q``, or 0 for a point head.

        Raises:
            ValueError: If ``receptive_field(kernel_size, dilations) < lookback``. Checked
                at construction so that a misconfigured stack fails immediately rather
                than training to a quietly degraded result.
        """
        super().__init__(lookback, max_horizon, n_input_channels, n_target_channels, n_quantiles)
        self.dilations: tuple[int, ...] = tuple(int(d) for d in dilations)
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.receptive_field = receptive_field(kernel_size, self.dilations)
        if self.receptive_field < lookback:
            raise ValueError(
                f"receptive field {self.receptive_field} < lookback {lookback}: the stack "
                f"cannot see the start of its own window (kernel_size={kernel_size}, "
                f"dilations={list(self.dilations)})"
            )
        blocks: list[nn.Module] = []
        channels = n_input_channels
        for dilation in self.dilations:
            blocks.append(TemporalBlock(channels, n_filters, kernel_size, dilation, dropout))
            channels = n_filters
        #: The dilated stack, exposed as an attribute so that
        #: ``tests/test_models.py`` can assert causality on its ``(B, C, L)`` feature map,
        #: which the horizon head -- reading only the last step -- cannot show.
        self.blocks = nn.Sequential(*blocks)
        self._head_width = max_horizon * n_target_channels * max(n_quantiles, 1)
        self.head = nn.Linear(n_filters, self._head_width)

    def forward(self, x: Tensor) -> Tensor:
        """Encode the lookback with the dilated stack and project to the horizon.

        Only the **last** encoded step is read. That is the principled choice here rather
        than a saving: because the receptive field covers the whole lookback, the final
        step is already a function of every input sample, so flattening all ``L`` steps
        would add capacity without adding information.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        features: Tensor = self.blocks(x.transpose(1, 2))
        out: Tensor = self.head(features[:, :, -1])
        # out: (B, H * C_out * max(Q, 1)) -> (B, H, C_out[, Q])
        if self.n_quantiles > 0:
            return out.view(x.shape[0], self.max_horizon, self.n_target_channels, self.n_quantiles)
        return out.view(x.shape[0], self.max_horizon, self.n_target_channels)
