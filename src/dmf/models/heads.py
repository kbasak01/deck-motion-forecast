"""Output heads: point, quantile, and Gaussian.

Structured so that a ``ConformalWrapper`` can be added later around
:class:`QuantileHead` without touching any model. Split conformal calibration is the
natural next step for these intervals, and the head boundary is where it attaches.
"""

from torch import Tensor, nn

__all__ = ["GaussianHead", "PointHead", "QuantileHead", "sort_quantiles"]


def sort_quantiles(y: Tensor) -> Tensor:
    """Sort a quantile fan ascending along the quantile axis.

    Pinball loss does not constrain quantile ordering, so a trained quantile head produces
    crossing quantiles on some fraction of inputs -- the 0.6 quantile below the 0.4, say.
    A crossed interval has negative width, which makes PICP and mean interval width
    meaningless. Sorting post-hoc is the standard remedy and is applied unconditionally
    before any probabilistic metric is computed.

    Args:
        y: Quantile forecasts, shape ``(B, H, C, Q)``, dimensionless.

    Returns:
        The same forecasts with the ``Q`` axis sorted ascending, shape ``(B, H, C, Q)``.
    """
    raise NotImplementedError


class PointHead(nn.Module):
    """Project an encoder representation to ``(H, C_out)`` point forecasts."""

    def __init__(self, d_in: int, max_horizon: int, n_target_channels: int) -> None:
        """Configure the head.

        Args:
            d_in: Width of the flattened encoder representation.
            max_horizon: Forecast length ``H``, samples.
            n_target_channels: Target channel count ``C_out``.
        """
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tensor:
        """Project to point forecasts.

        Args:
            z: Encoder representation, shape ``(B, d_in)``.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.
        """
        raise NotImplementedError


class QuantileHead(nn.Module):
    """Project an encoder representation to ``(H, C_out, Q)`` quantile forecasts.

    Trained with pinball loss; see :func:`dmf.train.losses.pinball_loss`.
    """

    def __init__(
        self,
        d_in: int,
        max_horizon: int,
        n_target_channels: int,
        quantiles: tuple[float, ...],
    ) -> None:
        """Configure the head.

        Args:
            d_in: Width of the flattened encoder representation.
            max_horizon: Forecast length ``H``, samples.
            n_target_channels: Target channel count ``C_out``.
            quantiles: Quantile levels, each in (0, 1), strictly ascending. The default
                fan is the nine levels 0.05 to 0.95 in steps of 0.1125, whose outermost
                pair gives the 90 percent interval scored by PICP@90.

        Raises:
            ValueError: If ``quantiles`` is not strictly ascending or contains a level
                outside (0, 1).
        """
        raise NotImplementedError

    def forward(self, z: Tensor) -> Tensor:
        """Project to a quantile fan.

        Args:
            z: Encoder representation, shape ``(B, d_in)``.

        Returns:
            Quantile forecasts, shape ``(B, H, C_out, Q)``, dimensionless. Not sorted;
            callers apply :func:`sort_quantiles` before scoring.
        """
        raise NotImplementedError


class GaussianHead(nn.Module):
    """Project an encoder representation to a per-step mean and log-variance.

    Cheaper than a quantile fan and trained with Gaussian NLL, but it assumes a symmetric
    predictive distribution. Whether that assumption costs anything on deck motion is an
    empirical question the Phase 5 comparison answers.
    """

    def __init__(self, d_in: int, max_horizon: int, n_target_channels: int) -> None:
        """Configure the head.

        Args:
            d_in: Width of the flattened encoder representation.
            max_horizon: Forecast length ``H``, samples.
            n_target_channels: Target channel count ``C_out``.
        """
        raise NotImplementedError

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Project to a mean and log-variance.

        Args:
            z: Encoder representation, shape ``(B, d_in)``.

        Returns:
            Tuple ``(mean, log_var)``, each of shape ``(B, H, C_out)``, dimensionless.
            ``log_var`` is returned rather than a variance so that the parameterisation is
            unconstrained and the NLL stays numerically stable.
        """
        raise NotImplementedError
