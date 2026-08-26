"""Per-window de-meaning and train-split-only channel scaling.

Two stages, deliberately separated:

1. **Per-window de-meaning.** Subtract the lookback-window mean from each window and add it
   back to the prediction. This removes the slowly varying offset that the model has no
   business learning, and is applied identically at train and test time using only
   information inside the window itself.
2. **Global per-channel scale, fitted on the training split only.** Fitting the scale over
   the whole corpus leaks test-set variance into training. It is a small leak and it does
   not produce an obviously wrong number, which is precisely why it needs a structural
   guard rather than care: :class:`NormStats` records the split it was fitted on, and
   :func:`apply_norm` refuses statistics that were not fitted on a training partition.
"""

from dataclasses import dataclass

from torch import Tensor, nn

from dmf.typedefs import FloatArray

__all__ = ["NormStats", "RevIN", "apply_norm", "demean_window", "fit_norm_stats", "invert_norm"]


@dataclass(frozen=True)
class NormStats:
    """Per-channel normalisation statistics and their provenance.

    Attributes:
        channels: Channel names, in the order the statistics are stored.
        scale: Per-channel scale, shape ``(n_channels,)``. Units are per-channel: degrees
            for angles, degrees per second for angular rates, metres for heave.
        fitted_on: Description of the partition the statistics were fitted on, e.g.
            ``"id/train"``. Carried so that the leak in point 2 of the module docstring is
            detectable after the fact, from a saved checkpoint, rather than only at the
            moment of fitting.
        n_realizations: Number of realizations the statistics were fitted over.
    """

    channels: tuple[str, ...]
    scale: FloatArray
    fitted_on: str
    n_realizations: int


def fit_norm_stats(
    train_series: FloatArray,
    channels: tuple[str, ...],
    fitted_on: str,
    n_realizations: int,
) -> NormStats:
    """Fit per-channel scale statistics on training realizations only.

    Args:
        train_series: Concatenated training-split samples, shape
            ``(n_samples, n_channels)``, in corpus units. Must contain **only** training
            realizations; the caller is responsible for having applied the split first.
        channels: Channel names matching the columns of ``train_series``.
        fitted_on: Provenance label recorded on the result, e.g. ``"id/train"``. Must name
            a training partition.
        n_realizations: Number of realizations represented in ``train_series``.

    Returns:
        The fitted statistics.

    Raises:
        ValueError: If ``fitted_on`` does not denote a training partition, if the column
            count does not match ``channels``, or if any channel has zero variance.
    """
    raise NotImplementedError


def demean_window(x: Tensor) -> tuple[Tensor, Tensor]:
    """Subtract the per-window, per-channel mean from a batch of input windows.

    Args:
        x: Input windows, shape ``(B, L, C)``, in corpus units.

    Returns:
        Tuple ``(x_centred, mean)`` where ``x_centred`` has shape ``(B, L, C)`` and
        ``mean`` has shape ``(B, 1, C)``. ``mean`` must be retained and passed to
        :func:`invert_norm`, since the prediction is only meaningful once it is added back.
    """
    raise NotImplementedError


def apply_norm(x: Tensor, stats: NormStats) -> Tensor:
    """Divide each channel by its fitted scale.

    Args:
        x: De-meaned windows, shape ``(B, L, C)``, in corpus units.
        stats: Statistics fitted on the training split.

    Returns:
        Scaled windows, shape ``(B, L, C)``, dimensionless.

    Raises:
        ValueError: If ``stats`` was not fitted on a training partition, or if its channel
            count does not match ``C``.
    """
    raise NotImplementedError


def invert_norm(y: Tensor, stats: NormStats, window_mean: Tensor) -> Tensor:
    """Map a model output back into corpus units.

    Multiplies by the fitted scale and adds back the per-window mean. Metrics are always
    computed in corpus units -- degrees, metres -- never in normalised space, because a
    normalised RMSE is not comparable across channels and cannot be checked against an
    operational threshold.

    Args:
        y: Model output, shape ``(B, H, C)`` for a point head or ``(B, H, C, Q)`` for a
            quantile head, dimensionless.
        stats: The statistics used in the forward direction.
        window_mean: Per-window means from :func:`demean_window`, shape ``(B, 1, C)``.

    Returns:
        Predictions in corpus units, same shape as ``y``.

    Raises:
        ValueError: If the shapes of ``y``, ``stats`` and ``window_mean`` are inconsistent.
    """
    raise NotImplementedError


class RevIN(nn.Module):
    """Reversible instance normalisation, as an alternative to plain de-meaning.

    Offered behind a config flag (``DataConfig.revin``) and ablated in Phase 6 rather than
    adopted by default, since on a de-meaned narrowband signal its benefit is an empirical
    question rather than a foregone conclusion.
    """

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        """Initialise the layer.

        Args:
            n_channels: Number of input channels ``C``.
            eps: Numerical floor added to the standard deviation.
            affine: Whether to learn a per-channel affine transform after normalising.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Normalise a batch of windows, caching the statistics for inversion.

        Args:
            x: Input windows, shape ``(B, L, C)``, in corpus units.

        Returns:
            Normalised windows, shape ``(B, L, C)``, dimensionless.
        """
        raise NotImplementedError

    def inverse(self, y: Tensor) -> Tensor:
        """Undo the most recent :meth:`forward`, restoring corpus units.

        Args:
            y: Model output, shape ``(B, H, C)``, dimensionless.

        Returns:
            Output in corpus units, shape ``(B, H, C)``.

        Raises:
            RuntimeError: If called before :meth:`forward`.
        """
        raise NotImplementedError
