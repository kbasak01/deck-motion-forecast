"""Encounter-frequency transformation and heading conventions.

Heading convention, used consistently across the package: ``beta_deg`` is the encounter
angle in degrees, where **180 is head seas** (vessel steaming into the waves), 90 is beam
seas, and 0 is following seas.

The encounter frequency for deep water is::

    w_e = w - (w^2 * U / g) * cos(beta)

Note the sign convention this implies: with ``beta = 180`` (head seas) ``cos(beta) = -1``
and ``w_e > w``, which is the physically correct Doppler up-shift.
"""

from dmf.typedefs import BoolArray, FloatArray

__all__ = [
    "GRAVITY_M_S2",
    "deep_water_wavenumber",
    "encounter_frequency",
    "encounter_frequency_is_monotonic",
    "knots_to_mps",
]

#: Standard gravitational acceleration, metres per second squared.
GRAVITY_M_S2: float = 9.80665


def knots_to_mps(u_kn: float) -> float:
    """Convert forward speed from knots to metres per second.

    Args:
        u_kn: Speed, knots.

    Returns:
        Speed, metres per second.
    """
    raise NotImplementedError


def deep_water_wavenumber(w_rad_s: FloatArray) -> FloatArray:
    """Compute the deep-water wavenumber from angular frequency.

    Uses the deep-water dispersion relation ``k = w^2/g``, valid when the water depth
    exceeds roughly half the wavelength. The corpus assumes deep water throughout; no
    finite-depth correction is applied.

    Args:
        w_rad_s: Angular frequencies, radians per second.

    Returns:
        Wavenumbers, radians per metre, same shape as ``w_rad_s``.
    """
    raise NotImplementedError


def encounter_frequency(w_rad_s: FloatArray, speed_mps: float, beta_deg: float) -> FloatArray:
    """Transform absolute wave frequencies into encounter frequencies.

    ``w_e = w - (w^2 * U / g) * cos(beta)``, with ``beta`` measured per the module
    convention (180 = head seas).

    Args:
        w_rad_s: Absolute wave angular frequencies, radians per second.
        speed_mps: Vessel forward speed, metres per second.
        beta_deg: Encounter angle, degrees. Converted to radians internally.

    Returns:
        Encounter angular frequencies, radians per second, same shape as ``w_rad_s``.
        May contain negative values in following seas, which correspond to waves
        overtaking the vessel.

    Raises:
        ValueError: If ``speed_mps`` is negative.
    """
    raise NotImplementedError


def encounter_frequency_is_monotonic(
    w_rad_s: FloatArray, speed_mps: float, beta_deg: float
) -> BoolArray:
    """Flag components for which the encounter-frequency map is locally invertible.

    ``dw_e/dw = 1 - 2*w*U*cos(beta)/g`` changes sign in following and quartering seas
    (``cos(beta) > 0``), so distinct absolute frequencies can map to the same encounter
    frequency. Where that happens, a response computed by naive per-component lookup is
    wrong, and the regime must either be excluded from the corpus or handled explicitly.
    This project excludes it: :func:`dmf.sim.generate.realization_grid` rejects any
    (speed, heading) cell for which this predicate is not True across the whole synthesis
    band, and records the rejection in the corpus card.

    Args:
        w_rad_s: Absolute wave angular frequencies, radians per second.
        speed_mps: Vessel forward speed, metres per second.
        beta_deg: Encounter angle, degrees.

    Returns:
        Boolean mask, same shape as ``w_rad_s``, True where ``dw_e/dw > 0``.
    """
    raise NotImplementedError
