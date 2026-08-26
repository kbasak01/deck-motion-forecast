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

import numpy as np
import torch
from torch import Tensor, nn

from dmf.typedefs import FloatArray

__all__ = [
    "NormStats",
    "RevIN",
    "apply_norm",
    "build_norm_stats",
    "demean_window",
    "fit_norm_stats",
    "invert_norm",
    "is_train_partition",
]


def is_train_partition(fitted_on: str) -> bool:
    """Report whether a provenance label names a training partition.

    Args:
        fitted_on: Provenance label, e.g. ``"id/train"`` or ``"train"``.

    Returns:
        True if the label is exactly ``"train"`` or ends in ``"/train"``. Anything else --
        ``"id/val"``, ``"unseen_seastate/test"``, ``"corpus"`` -- is refused by
        :func:`fit_norm_stats` and :func:`apply_norm`.
    """
    return fitted_on == "train" or fitted_on.endswith("/train")


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

    def subset(self, channels: tuple[str, ...]) -> "NormStats":
        """Restrict the statistics to a subset of channels, preserving provenance.

        The dataset scales its inputs with the full input-channel statistics but inverts
        its targets with the target-channel subset, so :func:`invert_norm` can keep a strict
        shape check instead of silently slicing whatever it is handed.

        Args:
            channels: Channel names to retain, in the order wanted on the result.

        Returns:
            Statistics over ``channels`` only, with ``fitted_on`` and ``n_realizations``
            carried through unchanged.

        Raises:
            ValueError: If any requested channel is absent.
        """
        unknown = [c for c in channels if c not in self.channels]
        if unknown:
            raise ValueError(f"channels {unknown} are not in {list(self.channels)}")
        index = [self.channels.index(c) for c in channels]
        return NormStats(
            channels=tuple(channels),
            scale=self.scale[index],
            fitted_on=self.fitted_on,
            n_realizations=self.n_realizations,
        )


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
    if not is_train_partition(fitted_on):
        raise ValueError(
            f"normalisation statistics must be fitted on a training partition, got "
            f"fitted_on={fitted_on!r}. Fitting on val, test, or the whole corpus leaks "
            f"held-out variance into training (CLAUDE.md non-negotiable 3)."
        )
    if train_series.ndim != 2:
        raise ValueError(f"train_series must be 2-D, got shape {train_series.shape}")
    if train_series.shape[1] != len(channels):
        raise ValueError(
            f"train_series has {train_series.shape[1]} columns but {len(channels)} channel "
            f"names were given: {list(channels)}"
        )
    if train_series.shape[0] == 0:
        raise ValueError("train_series is empty")
    scale = np.asarray(train_series, dtype=np.float64).std(axis=0)
    return build_norm_stats(scale, channels, fitted_on, n_realizations)


def build_norm_stats(
    scale: FloatArray,
    channels: tuple[str, ...],
    fitted_on: str,
    n_realizations: int,
) -> NormStats:
    """Validate and wrap a pre-computed per-channel scale.

    Split out of :func:`fit_norm_stats` so that a caller which accumulates second moments
    streaming over realizations -- as :class:`dmf.data.dataset.DeckMotionDataset` does, to
    avoid materialising a float64 copy of the whole training split -- passes through the
    same provenance and zero-variance checks.

    Args:
        scale: Per-channel scale, shape ``(n_channels,)``, in corpus units.
        channels: Channel names, matching ``scale``.
        fitted_on: Provenance label; must name a training partition.
        n_realizations: Number of realizations represented.

    Returns:
        The validated statistics.

    Raises:
        ValueError: If ``fitted_on`` does not denote a training partition, if the length of
            ``scale`` does not match ``channels``, or if any channel has zero variance.
    """
    if not is_train_partition(fitted_on):
        raise ValueError(
            f"normalisation statistics must be fitted on a training partition, got "
            f"fitted_on={fitted_on!r}"
        )
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (len(channels),):
        raise ValueError(f"scale has shape {scale.shape}, expected ({len(channels)},)")
    dead = [name for name, s in zip(channels, scale.tolist(), strict=True) if s <= 0.0]
    if dead:
        raise ValueError(f"channels {dead} have zero variance on the training split")
    return NormStats(
        channels=tuple(channels),
        scale=scale,
        fitted_on=fitted_on,
        n_realizations=n_realizations,
    )


def demean_window(x: Tensor) -> tuple[Tensor, Tensor]:
    """Subtract the per-window, per-channel mean from a batch of input windows.

    Args:
        x: Input windows, shape ``(B, L, C)``, in corpus units.

    Returns:
        Tuple ``(x_centred, mean)`` where ``x_centred`` has shape ``(B, L, C)`` and
        ``mean`` has shape ``(B, 1, C)``. ``mean`` must be retained and passed to
        :func:`invert_norm`, since the prediction is only meaningful once it is added back.

    Raises:
        ValueError: If ``x`` is not three-dimensional.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, L, C), got {tuple(x.shape)}")
    mean = x.mean(dim=1, keepdim=True)
    return x - mean, mean


def _scale_tensor(stats: NormStats, like: Tensor) -> Tensor:
    """Return the fitted scale as a tensor on the dtype and device of ``like``.

    Args:
        stats: The statistics.
        like: Tensor whose dtype and device the scale should match.

    Returns:
        Scale, shape ``(C,)``.
    """
    return torch.as_tensor(stats.scale, dtype=like.dtype, device=like.device)


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
    if not is_train_partition(stats.fitted_on):
        raise ValueError(
            f"refusing to normalise with statistics fitted on {stats.fitted_on!r}: only "
            f"training-partition statistics may be applied (CLAUDE.md non-negotiable 3)"
        )
    if x.shape[-1] != len(stats.channels):
        raise ValueError(
            f"x has {x.shape[-1]} channels but stats cover {len(stats.channels)}: "
            f"{list(stats.channels)}"
        )
    return x / _scale_tensor(stats, x)


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
    if y.ndim not in (3, 4):
        raise ValueError(f"y must have shape (B, H, C) or (B, H, C, Q), got {tuple(y.shape)}")
    n_channels = len(stats.channels)
    if y.shape[2] != n_channels:
        raise ValueError(
            f"y has {y.shape[2]} channels but stats cover {n_channels}: {list(stats.channels)}"
        )
    if window_mean.ndim != 3 or window_mean.shape[1] != 1:
        raise ValueError(f"window_mean must have shape (B, 1, C), got {tuple(window_mean.shape)}")
    if window_mean.shape[0] != y.shape[0] or window_mean.shape[2] != n_channels:
        raise ValueError(
            f"window_mean {tuple(window_mean.shape)} is inconsistent with y "
            f"{tuple(y.shape)} and {n_channels} channels"
        )
    scale = _scale_tensor(stats, y)
    mean = window_mean.to(dtype=y.dtype)
    if y.ndim == 4:
        return y * scale[:, None] + mean[..., None]
    return y * scale + mean


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

        Raises:
            ValueError: If ``n_channels`` is not positive or ``eps`` is not positive.
        """
        super().__init__()
        if n_channels < 1:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Normalise a batch of windows, caching the statistics for inversion.

        Args:
            x: Input windows, shape ``(B, L, C)``, in corpus units.

        Returns:
            Normalised windows, shape ``(B, L, C)``, dimensionless.

        Raises:
            ValueError: If ``x`` is not ``(B, L, C)`` with ``C == n_channels``.
        """
        if x.ndim != 3 or x.shape[-1] != self.n_channels:
            raise ValueError(f"x must have shape (B, L, {self.n_channels}), got {tuple(x.shape)}")
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False) + self.eps
        self._mean = mean.detach()
        self._std = std.detach()
        out = (x - mean) / std
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def inverse(self, y: Tensor) -> Tensor:
        """Undo the most recent :meth:`forward`, restoring corpus units.

        Args:
            y: Model output, shape ``(B, H, C)``, dimensionless.

        Returns:
            Output in corpus units, shape ``(B, H, C)``.

        Raises:
            RuntimeError: If called before :meth:`forward`.
            ValueError: If ``y`` is inconsistent with the cached statistics.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.inverse called before RevIN.forward")
        if y.ndim != 3 or y.shape[-1] != self.n_channels:
            raise ValueError(f"y must have shape (B, H, {self.n_channels}), got {tuple(y.shape)}")
        if y.shape[0] != self._mean.shape[0]:
            raise ValueError(
                f"y has batch {y.shape[0]} but the cached statistics have {self._mean.shape[0]}"
            )
        out = y
        if self.affine:
            out = (out - self.bias) / self.weight
        return out * self._std + self._mean
