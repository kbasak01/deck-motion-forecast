"""JONSWAP spectrum, spectral moments, and random-phase wave synthesis.

Frequencies are **radians per second** throughout. The spectral density ``S(w)`` has units
of m^2 s/rad, so that ``integral(S(w) dw)`` is in m^2.

The DNV-RP-C205 Hs-scaled JONSWAP form used here is::

    S(w) = (5/16) * Hs^2 * wp^4 * w^-5 * exp(-1.25 * (wp/w)^4)
           * (1 - 0.287*ln(gamma)) * gamma^r
    r     = exp(-(w - wp)^2 / (2 * sigma^2 * wp^2))
    sigma = 0.07 for w <= wp, 0.09 for w > wp
    wp    = 2*pi/Tp
"""

import numpy as np

from dmf.typedefs import FloatArray

__all__ = [
    "component_amplitudes",
    "hs_from_m0",
    "jittered_frequency_grid",
    "jonswap",
    "peak_frequency",
    "spectral_moment",
    "synthesize_elevation",
    "tz_from_moments",
]


def peak_frequency(tp_s: float) -> float:
    """Convert a spectral peak period to a peak angular frequency.

    Args:
        tp_s: Spectral peak period, seconds.

    Returns:
        Peak angular frequency ``wp = 2*pi/tp_s``, radians per second.

    Raises:
        ValueError: If ``tp_s`` is not strictly positive.
    """
    raise NotImplementedError


def jonswap(w_rad_s: FloatArray, hs_m: float, tp_s: float, gamma: float) -> FloatArray:
    """Evaluate the Hs-scaled JONSWAP spectral density.

    Args:
        w_rad_s: Angular frequencies at which to evaluate the spectrum, radians per
            second. Must be strictly positive; ``S`` diverges as ``w -> 0``.
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: Peak-enhancement factor, dimensionless. 1.0 recovers Pierson-Moskowitz.

    Returns:
        Spectral density at each requested frequency, m^2 s/rad, same shape as
        ``w_rad_s``.

    Raises:
        ValueError: If ``hs_m`` or ``tp_s`` is not strictly positive, if ``gamma < 1``,
            or if any element of ``w_rad_s`` is not strictly positive.
    """
    raise NotImplementedError


def spectral_moment(w_rad_s: FloatArray, s_m2_s_rad: FloatArray, n: int) -> float:
    """Compute the n-th spectral moment by trapezoidal integration.

    ``m_n = integral(w^n * S(w) dw)``.

    Args:
        w_rad_s: Angular frequencies, radians per second, strictly ascending.
        s_m2_s_rad: Spectral density at those frequencies, m^2 s/rad.
        n: Moment order. ``n=0`` gives variance (m^2), ``n=2`` gives the second moment
            (m^2/s^2) used for the zero-crossing period.

    Returns:
        The n-th spectral moment, units m^2 (rad/s)^n.

    Raises:
        ValueError: If the two arrays differ in shape or if ``w_rad_s`` is not ascending.
    """
    raise NotImplementedError


def hs_from_m0(m0_m2: float) -> float:
    """Recover significant wave height from the zeroth spectral moment.

    ``Hs = 4*sqrt(m0)``. Gate 1 requires this to reproduce the requested ``Hs`` to within
    2 percent when applied to the truncated, discretised spectrum actually synthesised.

    Args:
        m0_m2: Zeroth spectral moment (surface elevation variance), m^2.

    Returns:
        Significant wave height, metres.

    Raises:
        ValueError: If ``m0_m2`` is negative.
    """
    raise NotImplementedError


def tz_from_moments(m0_m2: float, m2_m2_rad2_s2: float) -> float:
    """Compute the mean zero-crossing period from spectral moments.

    ``Tz = 2*pi*sqrt(m0/m2)``. For ``gamma = 3.3`` the ratio ``Tz/Tp`` is expected to lie
    in roughly 0.71-0.78; Gate 1 asserts this band.

    Args:
        m0_m2: Zeroth spectral moment, m^2.
        m2_m2_rad2_s2: Second spectral moment, m^2 rad^2/s^2.

    Returns:
        Mean zero-crossing period, seconds.

    Raises:
        ValueError: If either moment is not strictly positive.
    """
    raise NotImplementedError


def jittered_frequency_grid(
    w_min_rad_s: float,
    w_max_rad_s: float,
    n_components: int,
    rng: np.random.Generator,
    *,
    jitter: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Build the component frequency grid for random-phase synthesis.

    The band is divided into ``n_components`` equal bins of width ``dw``. With
    ``jitter=True`` each component frequency is drawn uniformly inside its own bin
    (``w_i = w_lo_i + u_i*dw``, ``u_i ~ U(0,1)``); with ``jitter=False`` the bin centre is
    used.

    Jitter is not cosmetic. A uniform grid makes the synthesised record exactly periodic
    with period ``2*pi/dw``. If that period is shorter than the record length, the
    "forecasting" task collapses into memorising a repeat, and every downstream result is
    invalid. ``tests/test_spectra.py`` asserts the absence of an autocorrelation spike at
    that period.

    Args:
        w_min_rad_s: Lower band edge, radians per second, strictly positive.
        w_max_rad_s: Upper band edge, radians per second.
        n_components: Number of wave components. 200-400 is the intended range.
        rng: Seeded generator supplying the within-bin offsets.
        jitter: Whether to jitter within bins. False is provided only so that the
            periodicity failure mode can be demonstrated in a test.

    Returns:
        Tuple ``(w, dw)`` where ``w`` holds the component frequencies in radians per
        second, shape ``(n_components,)``, and ``dw`` holds the per-component bin width in
        radians per second, same shape. ``dw`` is uniform but returned per-component so
        that non-uniform banding can be introduced later without a signature change.

    Raises:
        ValueError: If the band is empty or ``n_components`` is not positive.
    """
    raise NotImplementedError


def component_amplitudes(s_m2_s_rad: FloatArray, dw_rad_s: FloatArray) -> FloatArray:
    """Convert a discretised spectrum into per-component wave amplitudes.

    ``a_i = sqrt(2 * S(w_i) * dw_i)``.

    Args:
        s_m2_s_rad: Spectral density at each component frequency, m^2 s/rad.
        dw_rad_s: Bin width associated with each component, radians per second.

    Returns:
        Component amplitudes, metres, same shape as the inputs.

    Raises:
        ValueError: If the two arrays differ in shape.
    """
    raise NotImplementedError


def synthesize_elevation(
    t_s: FloatArray,
    w_rad_s: FloatArray,
    amplitude_m: FloatArray,
    phase_rad: FloatArray,
) -> FloatArray:
    """Synthesise a wave elevation record by superposition of harmonic components.

    ``eta(t) = sum_i a_i * cos(w_i*t + phi_i)``.

    The same ``phase_rad`` array must be reused when synthesising the DOF responses in
    :mod:`dmf.sim.response`, so that the physical phase relationships between elevation,
    roll, pitch and heave are preserved. Those relationships are precisely what a
    multivariate forecaster is expected to exploit; drawing fresh phases per DOF would
    destroy them.

    Args:
        t_s: Time samples, seconds, shape ``(n_samples,)``.
        w_rad_s: Component frequencies, radians per second, shape ``(n_components,)``.
        amplitude_m: Component amplitudes, metres, shape ``(n_components,)``.
        phase_rad: Component phases, radians, drawn from U(0, 2*pi), shape
            ``(n_components,)``.

    Returns:
        Surface elevation at each time sample, metres, shape ``(n_samples,)``.

    Raises:
        ValueError: If the three component arrays differ in shape.
    """
    raise NotImplementedError
