"""Encoder-only transformer with patch embedding.

The lookback is cut into non-overlapping patches of 10 samples (1 s at ``fs = 10 Hz``),
giving 20 tokens for the default ``lookback = 200``. Patching is not an optimisation
detail: token-per-sample attention over 200 steps is quadratic in a length that carries
very little information per step, and it trains measurably worse on a narrowband signal
than the patched form.
"""

from torch import Tensor, nn

from dmf.models.base import BaseForecaster

__all__ = ["PatchEmbedding", "TransformerForecaster"]


class PatchEmbedding(nn.Module):
    """Cut a lookback window into non-overlapping patches and linearly embed them."""

    def __init__(self, patch_len: int, n_input_channels: int, d_model: int) -> None:
        """Configure the embedding.

        Args:
            patch_len: Samples per patch. At ``fs = 10 Hz``, 10 samples is 1 s.
            n_input_channels: Input channel count ``C_in``. All channels of a patch are
                flattened into one token, so the model attends over time rather than over
                channels.
            d_model: Embedding width.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of windows into a token sequence.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless. ``L`` must be an
                exact multiple of ``patch_len``.

        Returns:
            Token embeddings, shape ``(B, L//patch_len, d_model)``.

        Raises:
            ValueError: If ``L`` is not a multiple of ``patch_len``.
        """
        raise NotImplementedError


class TransformerForecaster(BaseForecaster):
    """Encoder-only transformer with learned positional encoding.

    Flatten-and-project head rather than a decoder: the horizon is emitted in one pass,
    consistent with every other model here.
    """

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        patch_len: int = 10,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_quantiles: int = 0,
    ) -> None:
        """Configure the model.

        Args:
            lookback: Input window length ``L``, samples. Must be a multiple of
                ``patch_len``.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``.
            n_target_channels: Target channel count ``C_out``.
            patch_len: Samples per token.
            d_model: Embedding and attention width.
            n_layers: Encoder layer count.
            n_heads: Attention heads per layer. ``d_model`` must be divisible by this.
            d_ff: Feed-forward hidden width.
            dropout: Dropout probability, in [0, 1).
            n_quantiles: Quantile count ``Q``, or 0 for a point head.

        Raises:
            ValueError: If ``lookback`` is not a multiple of ``patch_len``, or ``d_model``
                is not divisible by ``n_heads``.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Embed, attend over patches, and project to the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` or ``(B, H, C_out, Q)``, dimensionless.
        """
        raise NotImplementedError
