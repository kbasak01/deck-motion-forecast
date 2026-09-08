"""Encoder-only transformer with patch embedding.

The lookback is cut into non-overlapping patches of 10 samples (1 s at ``fs = 10 Hz``),
giving 20 tokens for the default ``lookback = 200``. Patching is not an optimisation
detail: token-per-sample attention over 200 steps is quadratic in a length that carries
very little information per step, and it trains measurably worse on a narrowband signal
than the patched form.

**Where the capacity sits.** The flatten-and-project head maps ``n_tokens * d_model``
(20 x 128 = 2560) to ``H * C_out`` (150 x 6 = 900), which is 2 304 900 of the model's
2 712 708 parameters -- 85% of the model, and 38x the whole of DLinear. That is the head
the implementation plan specifies and it ships as specified; the resulting capacity spread
across the Phase 4 line-up is a caveat to report beside the results table, not something to
engineer away by quietly substituting a pooled or last-token head.
"""

from typing import ClassVar

import torch
from torch import Tensor, nn

from dmf.models.base import BaseForecaster
from dmf.models.heads import HeadKind, clamp_log_variance
from dmf.train.registry import register_model

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
        super().__init__()
        self.patch_len = patch_len
        self.n_input_channels = n_input_channels
        self.d_model = d_model
        self.projection = nn.Linear(patch_len * n_input_channels, d_model)

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
        lookback = int(x.shape[1])
        if lookback % self.patch_len != 0:
            raise ValueError(
                f"lookback ({lookback}) must be a multiple of patch_len ({self.patch_len})"
            )
        n_tokens = lookback // self.patch_len
        patches = x.reshape(x.shape[0], n_tokens, self.patch_len * self.n_input_channels)
        embedded: Tensor = self.projection(patches)
        return embedded


@register_model("transformer")
class TransformerForecaster(BaseForecaster):
    """Encoder-only transformer with learned positional encoding.

    Flatten-and-project head rather than a decoder: the horizon is emitted in one pass,
    consistent with every other model here.

    No causal mask is applied. The encoder reads the lookback only, every sample of which
    is in the past of the forecast origin, so masking would restrict the encoder without
    removing any leak -- unlike the TCN's causal padding, which is load-bearing because its
    convolutions are applied at every step of the window.
    """

    FIT_KIND = "sgd"
    #: All three heads: the projection widens by K and the reshape branch handles each.
    SUPPORTED_HEADS: ClassVar[tuple[HeadKind, ...]] = ("point", "quantile", "gaussian")

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
        head: HeadKind | None = None,
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
            head: Output head kind -- ``"point"``, ``"quantile"`` or ``"gaussian"``.
                Defaults to ``"quantile"`` when ``n_quantiles > 0`` and ``"point"``
                otherwise, so a point build is unchanged and a Gaussian head is asked for
                by name (``docs/protocol.md`` P5-D1).

        Raises:
            ValueError: If ``lookback`` is not a multiple of ``patch_len``, or ``d_model``
                is not divisible by ``n_heads``.
        """
        super().__init__(
            lookback, max_horizon, n_input_channels, n_target_channels, n_quantiles, head
        )
        if patch_len < 1:
            raise ValueError(f"patch_len must be positive, got {patch_len}")
        if lookback % patch_len != 0:
            raise ValueError(
                f"lookback ({lookback}) must be a multiple of patch_len ({patch_len}); "
                f"a ragged final patch would silently drop the oldest samples"
            )
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_tokens = lookback // patch_len
        self.patch_embed = PatchEmbedding(patch_len, n_input_channels, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._head_width = max_horizon * n_target_channels * self.n_output_params
        self.head = nn.Linear(self.n_tokens * d_model, self._head_width)

    def _predict(self, x: Tensor) -> Tensor:
        """Embed, attend over patches, and project to the horizon.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Forecasts, shape ``(B, H, C_out)`` for a point head, ``(B, H, C_out, Q)`` for
            a quantile head, or ``(B, H, C_out, 2)`` carrying ``(mean, log_var)`` for a
            Gaussian one. Dimensionless.
        """
        tokens: Tensor = self.embed_dropout(self.patch_embed(x) + self.pos_embed)
        encoded: Tensor = self.encoder(tokens)
        out: Tensor = self.head(encoded.reshape(x.shape[0], self.n_tokens * self.d_model))
        # out: (B, H * C_out * K) -> (B, H, C_out[, K])
        if self.head_kind == "point":
            return out.view(x.shape[0], self.max_horizon, self.n_target_channels)
        fanned = out.view(
            x.shape[0], self.max_horizon, self.n_target_channels, self.n_output_params
        )
        return clamp_log_variance(fanned) if self.head_kind == "gaussian" else fanned
