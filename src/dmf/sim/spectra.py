"""JONSWAP spectrum, spectral moments, and random-phase wave-elevation synthesis.

All frequencies in this module are **radians per second**. Spectral densities are
**m^2 s/rad**, i.e. ``integral S(w) dw`` has units of m^2 and equals the variance of the
sea-surface elevation. Nothing here is expressed in Hz; if a Hz quantity ever appears in a
caller it must be converted with ``w = 2*pi*f`` and ``S(w) = S_f(f)/(2*pi)`` before
comparison.

The spectral form is the Hs-scaled DNV-RP-C205 JONSWAP::

    wp    = 2*pi/Tp
    sigma = 0.07 for w <= wp, else 0.09
    r     = exp(-(w - wp)**2 / (2 * sigma**2 * wp**2))
    S(w)  = (5/16) * Hs**2 * wp**4 * w**-5 * exp(-1.25*(wp/w)**4)
            * (1 - 0.287*log(gamma)) * gamma**r

The ``(1 - 0.287*ln gamma)`` factor normalises the peak enhancement so that
``integral S dw = Hs**2/16`` for every ``gamma``, which is what makes the Gate 1
``Hs = 4*sqrt(m0)`` recovery check meaningful.
"""

from dataclasses import dataclass

import numpy as np

from dmf.typedefs import FloatArray

__all__ = [
    "WaveComponents",
    "hs_from_moments",
    "jonswap",
    "sample_components",
    "spectral_moment",
    "synthesis_period_s",
    "synthesize_elevation",
    "tz_from_moments",
]

#: Time-sample block size for the superposition inner loop. The synthesis is an
#: ``(n_samples, n_components)`` outer product; evaluating it in blocks keeps the temporary
#: in cache instead of allocating a single 86 MB array for an hour-long record.
_TIME_CHUNK = 8192


def jonswap(w: FloatArray, hs_m: float, tp_s: float, gamma: float = 3.3) -> FloatArray:
    """Evaluate the Hs-scaled JONSWAP spectral density.

    Args:
        w: Angular frequencies, radians per second. Strictly positive entries only;
            non-positive entries are returned as zero density rather than as ``inf``,
            because ``w**-5`` diverges at the origin.
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: Peak-enhancement factor, dimensionless. ``1.0`` recovers
            Pierson-Moskowitz; ``3.3`` is the standard North Sea value.

    Returns:
        Spectral density ``S(w)`` in m^2 s/rad, same shape as ``w``.

    Raises:
        ValueError: If ``hs_m <= 0``, ``tp_s <= 0`` or ``gamma < 1``.
    """
    if hs_m <= 0.0:
        raise ValueError(f"hs_m must be positive, got {hs_m}")
    if tp_s <= 0.0:
        raise ValueError(f"tp_s must be positive, got {tp_s}")
    if gamma < 1.0:
        raise ValueError(f"gamma must be >= 1, got {gamma}")

    w_arr = np.asarray(w, dtype=np.float64)
    positive = w_arr > 0.0
    # A dummy positive value keeps ``log`` and ``(wp/w)**4`` finite on the masked-out
    # entries; the density there is overwritten with zero below.
    w_safe = np.where(positive, w_arr, 1.0)

    wp = 2.0 * np.pi / tp_s
    sigma = np.where(w_safe <= wp, 0.07, 0.09)
    r = np.exp(-((w_safe - wp) ** 2) / (2.0 * sigma**2 * wp**2))
    normalisation = 1.0 - 0.287 * np.log(gamma)

    # Evaluated in log space: the ``w**-5`` prefactor overflows and the
    # ``exp(-1.25*(wp/w)**4)`` tail underflows at small ``w``, and their product is a
    # finite zero that a naive ``inf * 0`` would turn into ``nan``.
    log_prefactor = np.log(5.0 / 16.0 * hs_m**2 * wp**4)
    with np.errstate(over="ignore"):
        log_shape = log_prefactor - 5.0 * np.log(w_safe) - 1.25 * (wp / w_safe) ** 4
    density = np.exp(log_shape) * normalisation * gamma**r
    return np.asarray(np.where(positive, density, 0.0), dtype=np.float64)


def spectral_moment(w: FloatArray, s: FloatArray, n: int) -> float:
    """Compute the ``n``-th spectral moment ``m_n = integral(w**n * S(w) dw)``.

    Integration is trapezoidal over the supplied grid, so the grid, not the integrand,
    sets the accuracy. This matters for ``m2``: the integrand ``w**2 * S(w)`` decays only
    as ``w**-3``, so ``Tz = 2*pi*sqrt(m0/m2)`` converges slowly and the upper integration
    limit materially changes the answer. Integrate to ``w_max = 30 rad/s`` to recover the
    published ``Tp/Tz = 1.286`` at ``gamma = 3.3``.

    Args:
        w: Angular frequencies, radians per second, strictly increasing.
        s: Spectral density at ``w``, m^2 s/rad. Same shape as ``w``.
        n: Moment order. ``0`` gives variance (m^2), ``2`` gives m^2/s^2.

    Returns:
        The moment ``m_n``, units m^2 (rad/s)^n.

    Raises:
        ValueError: If ``w`` and ``s`` have different shapes, if ``w`` is not strictly
            increasing, or if ``n`` is negative.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    if w_arr.shape != s_arr.shape:
        raise ValueError(f"w and s must have the same shape, got {w_arr.shape} and {s_arr.shape}")
    if w_arr.ndim != 1 or w_arr.size < 2:
        raise ValueError(f"w must be a 1-D grid with at least two points, got shape {w_arr.shape}")
    if not bool(np.all(np.diff(w_arr) > 0.0)):
        raise ValueError("w must be strictly increasing")
    if n < 0:
        raise ValueError(f"moment order must be non-negative, got {n}")
    return float(np.trapezoid(w_arr**n * s_arr, w_arr))


def hs_from_moments(m0: float) -> float:
    """Recover significant wave height from the zeroth spectral moment.

    Args:
        m0: Zeroth spectral moment, m^2 (equivalently the elevation variance).

    Returns:
        Significant wave height ``4*sqrt(m0)``, metres.

    Raises:
        ValueError: If ``m0 < 0``.
    """
    if m0 < 0.0:
        raise ValueError(f"m0 must be non-negative, got {m0}")
    return 4.0 * float(np.sqrt(m0))


def tz_from_moments(m0: float, m2: float) -> float:
    """Recover the mean zero-upcrossing period from spectral moments.

    Args:
        m0: Zeroth spectral moment, m^2.
        m2: Second spectral moment, m^2/s^2.

    Returns:
        Zero-upcrossing period ``2*pi*sqrt(m0/m2)``, seconds.

    Raises:
        ValueError: If ``m0 < 0`` or ``m2 <= 0``.
    """
    if m0 < 0.0:
        raise ValueError(f"m0 must be non-negative, got {m0}")
    if m2 <= 0.0:
        raise ValueError(f"m2 must be positive, got {m2}")
    return 2.0 * np.pi * float(np.sqrt(m0 / m2))


@dataclass(frozen=True)
class WaveComponents:
    """A finite set of regular wave components approximating a JONSWAP sea.

    The elevation is reconstructed as
    ``eta(t) = sum_i amplitude_m[i] * cos(w_rad_s[i]*t + phase_rad[i])``.

    Attributes:
        w_rad_s: Component angular frequencies, radians per second, shape ``(n,)``.
            Each entry lies strictly inside its own bin, ``w_lo_i <= w_i < w_lo_i + dw_i``.
        dw_rad_s: Bin widths, radians per second, shape ``(n,)``. Equal-width by
            construction; retained per component so that the amplitude scaling
            ``sqrt(2*S*dw)`` is auditable after the fact.
        amplitude_m: Component amplitudes, metres, shape ``(n,)``.
        phase_rad: Component phases, radians in ``[0, 2*pi)``, shape ``(n,)``. The **same**
            phase set drives every DOF in :mod:`dmf.sim.response`, which is what preserves
            the physical roll/pitch/heave phase relationships.
    """

    w_rad_s: FloatArray
    dw_rad_s: FloatArray
    amplitude_m: FloatArray
    phase_rad: FloatArray


def sample_components(
    hs_m: float,
    tp_s: float,
    gamma: float,
    n_components: int,
    w_min_rad_s: float,
    w_max_rad_s: float,
    rng: np.random.Generator,
    jitter: bool = True,
) -> WaveComponents:
    """Draw a random-phase component set from a JONSWAP spectrum.

    The band ``[w_min_rad_s, w_max_rad_s]`` is split into ``n_components`` equal bins of
    width ``dw = (w_max - w_min)/n``. Frequencies are placed at
    ``w_i = w_lo_i + u_i*dw`` with ``u_i ~ U(0, 1)`` when ``jitter`` is True, and at the
    bin **lower edge** (``u_i = 0``) when it is False. Amplitudes are
    ``sqrt(2*S(w_i)*dw)`` metres and phases are drawn ``U(0, 2*pi)``.

    Setting ``jitter=False`` is provided only so that the anti-periodicity test can
    demonstrate the failure it guards against. An un-jittered grid is a uniform comb, so
    every component frequency is an integer multiple of ``dw`` whenever ``w_min/dw`` is an
    integer, and the record then repeats **exactly** with period ``2*pi/dw``. When
    ``w_min/dw`` is not an integer the repeat is a rigid phase rotation of the same
    waveform rather than an identity, which is just as memorisable by a forecaster. Never
    use ``jitter=False`` for a corpus.

    Args:
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: JONSWAP peak-enhancement factor, dimensionless.
        n_components: Number of components. 200-400 is the working range.
        w_min_rad_s: Lower band edge, radians per second.
        w_max_rad_s: Upper band edge, radians per second.
        rng: Explicit generator. All randomness in the simulator flows through this
            argument; there are no module-level ``np.random`` calls.
        jitter: If True, draw each frequency uniformly inside its bin.

    Returns:
        The sampled :class:`WaveComponents`.

    Raises:
        ValueError: If ``n_components < 1``, or if ``w_min_rad_s`` is not strictly
            positive and strictly less than ``w_max_rad_s``.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if w_min_rad_s <= 0.0:
        raise ValueError(f"w_min_rad_s must be strictly positive, got {w_min_rad_s}")
    if w_min_rad_s >= w_max_rad_s:
        raise ValueError(
            f"w_min_rad_s must be strictly less than w_max_rad_s, "
            f"got {w_min_rad_s} and {w_max_rad_s}"
        )

    dw = (w_max_rad_s - w_min_rad_s) / n_components
    # Same arithmetic as ``np.linspace(w_min, w_max, n + 1)[:-1]``, bit for bit, so that a
    # caller reconstructing the bin edges independently agrees exactly. With ``jitter``
    # off this also makes every ``w_i`` an exact integer multiple of ``dw`` whenever
    # ``w_min/dw`` is an integer, which is what the anti-periodicity converse test needs.
    w_lo = w_min_rad_s + np.arange(n_components, dtype=np.float64) * dw
    offset = (
        rng.uniform(0.0, 1.0, n_components) if jitter else np.zeros(n_components, dtype=np.float64)
    )
    w_rad_s = w_lo + offset * dw
    phase_rad = rng.uniform(0.0, 2.0 * np.pi, n_components)
    amplitude_m = np.sqrt(2.0 * jonswap(w_rad_s, hs_m, tp_s, gamma) * dw)
    return WaveComponents(
        w_rad_s=np.asarray(w_rad_s, dtype=np.float64),
        dw_rad_s=np.full(n_components, dw, dtype=np.float64),
        amplitude_m=np.asarray(amplitude_m, dtype=np.float64),
        phase_rad=np.asarray(phase_rad, dtype=np.float64),
    )


def synthesize_elevation(components: WaveComponents, t_s: FloatArray) -> FloatArray:
    """Reconstruct the sea-surface elevation time series by linear superposition.

    Computes ``eta(t) = sum_i a_i * cos(w_i*t + phi_i)``.

    Args:
        components: Component set from :func:`sample_components`.
        t_s: Sample times, seconds. Need not start at zero; the caller is responsible for
            discarding the spin-up transient by slicing the result.

    Returns:
        Elevation, metres, shape ``t_s.shape``. Its variance converges to
        ``sum(amplitude_m**2)/2``, which is the discrete counterpart of ``m0`` over the
        synthesis band.
    """
    t_arr = np.asarray(t_s, dtype=np.float64)
    flat_t = t_arr.reshape(-1)
    out = np.empty(flat_t.size, dtype=np.float64)
    w = components.w_rad_s
    amplitude = components.amplitude_m
    phase = components.phase_rad
    for start in range(0, flat_t.size, _TIME_CHUNK):
        stop = min(start + _TIME_CHUNK, flat_t.size)
        theta = np.outer(flat_t[start:stop], w) + phase
        out[start:stop] = np.cos(theta) @ amplitude
    return np.asarray(out.reshape(t_arr.shape), dtype=np.float64)


def synthesis_period_s(components: WaveComponents) -> float:
    """Return the period at which an un-jittered component set would repeat.

    This is ``2*pi/dw``. It is the quantity the anti-periodicity guard is built around: if
    it is shorter than the record length and the frequencies were not jittered, the record
    is a loop and any forecaster trained on it memorises the loop instead of learning the
    dynamics.

    Args:
        components: Component set. Bins are equal-width by construction; the minimum bin
            width is used so the returned value is the longest period the grid could
            produce.

    Returns:
        Synthesis period, seconds.
    """
    dw = float(np.min(components.dw_rad_s))
    if dw <= 0.0:
        raise ValueError(f"bin widths must be positive, got minimum {dw}")
    return 2.0 * np.pi / dw
