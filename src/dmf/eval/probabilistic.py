"""Probabilistic scoring: pinball, CRPS, PICP, interval width, Winkler.

All inputs are in **corpus units** -- degrees for roll and pitch, metres for heave -- so
that an interval width can be read against a landing limit directly.

Gate 5 requires PICP@90 within [0.85, 0.95] on the ``id`` regime. The degradation under
``unseen_seastate`` is expected, is **reported rather than fixed**, and is one of the more
interesting findings available here: it is the same coverage-under-domain-shift story that
conformal prediction runs into, measured on a concrete operational task.
"""

from dmf.typedefs import FloatArray

__all__ = ["crps_from_quantiles", "mean_interval_width", "picp", "pinball", "winkler_score"]


def pinball(pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]) -> float:
    """Compute mean pinball loss over a quantile fan.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units,
            sorted ascending along ``Q``.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), ascending, length ``Q``.

    Returns:
        Mean pinball loss, in corpus units.

    Raises:
        ValueError: If ``len(quantiles)`` does not match the ``Q`` axis, or the leading
            shapes disagree.
    """
    raise NotImplementedError


def crps_from_quantiles(
    pred_quantiles: FloatArray, target: FloatArray, quantiles: tuple[float, ...]
) -> float:
    """Approximate the continuous ranked probability score from a quantile fan.

    Computed as the quantile-weighted integral of the pinball loss, which converges to
    CRPS as the fan is refined. With ``Q = 9`` levels this is an approximation, and the
    results tables label it as such rather than reporting it as exact CRPS.

    Args:
        pred_quantiles: Quantile forecasts, shape ``(N, H, C, Q)``, in corpus units,
            sorted ascending along ``Q``.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        quantiles: Quantile levels, each in (0, 1), ascending, length ``Q``.

    Returns:
        Approximate CRPS, in corpus units.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    raise NotImplementedError


def picp(lower: FloatArray, upper: FloatArray, target: FloatArray) -> float:
    """Compute prediction interval coverage probability.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.

    Returns:
        Fraction of targets falling within ``[lower, upper]``, dimensionless in [0, 1].
        For a nominal 90 percent interval, Gate 5 requires this in [0.85, 0.95] on the
        ``id`` regime.

    Raises:
        ValueError: If the three arrays differ in shape, or if any ``lower`` exceeds its
            ``upper`` -- which indicates that :func:`dmf.models.heads.sort_quantiles` was
            not applied.
    """
    raise NotImplementedError


def mean_interval_width(lower: FloatArray, upper: FloatArray) -> float:
    """Compute the mean width of the prediction interval.

    Reported alongside PICP, never alone. Coverage is trivially achievable by widening the
    interval until it is useless; the pair is what says whether the intervals are
    informative.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.

    Returns:
        Mean interval width, in corpus units (degrees for angles, metres for heave).

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    raise NotImplementedError


def winkler_score(
    lower: FloatArray, upper: FloatArray, target: FloatArray, alpha: float = 0.1
) -> float:
    """Compute the mean Winkler interval score.

    Penalises interval width and adds a miss penalty proportional to how far outside the
    interval the target fell, so it scores coverage and sharpness in a single number
    rather than leaving the trade-off to the reader.

    Args:
        lower: Lower interval bound, shape ``(N, H, C)``, in corpus units.
        upper: Upper interval bound, shape ``(N, H, C)``, in corpus units.
        target: Targets, shape ``(N, H, C)``, in corpus units.
        alpha: Nominal miscoverage rate, so ``alpha = 0.1`` scores a 90 percent interval.

    Returns:
        Mean Winkler score, in corpus units. Lower is better.

    Raises:
        ValueError: If the shapes differ or ``alpha`` is not in (0, 1).
    """
    raise NotImplementedError
