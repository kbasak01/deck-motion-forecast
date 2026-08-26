"""Encounter kinematics: deep-water wave number and Doppler-shifted encounter frequency.

Frequencies are **radians per second**, wave numbers **radians per metre**, speeds
**metres per second** (knots are converted at the config boundary only), and headings
**degrees** at the public boundary with the convention 180 = head seas, 90 = beam seas,
0 = following seas. ``g = 9.80665 m/s^2``.
"""

import numpy as np

from dmf.typedefs import FloatArray

__all__ = [
    "GRAVITY_M_S2",
    "KNOT_M_S",
    "encounter_frequency",
    "is_encounter_monotonic",
    "knots_to_m_s",
    "wave_number",
]

#: Standard gravity, metres per second squared.
GRAVITY_M_S2: float = 9.80665

#: One international knot in metres per second.
KNOT_M_S: float = 0.514444


def knots_to_m_s(kn: float) -> float:
    """Convert a speed from knots to metres per second.

    Args:
        kn: Speed, knots.

    Returns:
        Speed, metres per second.
    """
    return kn * KNOT_M_S


def wave_number(w: FloatArray) -> FloatArray:
    """Deep-water wave number from angular frequency.

    Uses the deep-water dispersion relation ``k = w**2/g``. The corpus band
    ``[0.2, 2.5] rad/s`` corresponds to wavelengths from about 10 m to 1500 m; deep water
    is assumed throughout and no finite-depth correction is applied.

    Args:
        w: Angular frequencies, radians per second.

    Returns:
        Wave numbers, radians per metre, same shape as ``w``.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    return np.asarray(w_arr**2 / GRAVITY_M_S2, dtype=np.float64)


def encounter_frequency(w: FloatArray, speed_m_s: float, heading_deg: float) -> FloatArray:
    """Doppler-shifted frequency at which the vessel meets the waves.

    Computes ``w_e = w - (w**2 * U / g) * cos(beta)`` with ``beta`` in radians internally.
    Head seas (``beta = 180 deg``, ``cos beta = -1``) raise the encounter frequency;
    following seas (``beta = 0``) lower it and can drive ``w_e`` negative, which
    physically means the waves overtake the ship. The **signed** value is returned and
    must be kept in the cosine argument of the response synthesis; only the response
    transfer function is evaluated at ``abs(w_e)``, so that damping stays dissipative.

    Args:
        w: Wave angular frequencies in the earth frame, radians per second.
        speed_m_s: Forward speed, metres per second, non-negative.
        heading_deg: Encounter angle, degrees. 180 head, 90 beam, 0 following.

    Returns:
        Signed encounter frequencies, radians per second, same shape as ``w``.

    Raises:
        ValueError: If ``speed_m_s`` is negative.
    """
    if speed_m_s < 0.0:
        raise ValueError(f"speed_m_s must be non-negative, got {speed_m_s}")
    w_arr = np.asarray(w, dtype=np.float64)
    cos_beta = float(np.cos(np.radians(heading_deg)))
    return np.asarray(w_arr - w_arr**2 * speed_m_s * cos_beta / GRAVITY_M_S2, dtype=np.float64)


def is_encounter_monotonic(w: FloatArray, speed_m_s: float, heading_deg: float) -> bool:
    """Report whether ``w -> w_e`` is one-to-one across the supplied band.

    The derivative is ``dw_e/dw = 1 - 2*w*U*cos(beta)/g``, which is positive for every
    ``w`` when ``cos(beta) <= 0`` (head and bow-quartering seas) and changes sign at
    ``w_crit = g / (2*U*cos(beta))`` when ``cos(beta) > 0`` (following and stern-quartering
    seas). Two distinct wave frequencies then arrive at the same encounter frequency, so an
    encounter-frequency response spectrum is not a simple change of variable of the wave
    spectrum.

    The corpus keeps those cells rather than excluding them. Time-domain superposition
    stays valid because each component is propagated independently with its own signed
    ``w_e``; what breaks is only the frequency-domain change of variable, which this
    project never relies on. This predicate exists so that the non-monotonic cells are
    labelled explicitly instead of passing silently.

    Args:
        w: Wave angular frequencies, radians per second, covering the band of interest.
        speed_m_s: Forward speed, metres per second.
        heading_deg: Encounter angle, degrees.

    Returns:
        True if ``dw_e/dw`` has one sign across the whole of ``w``, False otherwise.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    cos_beta = float(np.cos(np.radians(heading_deg)))
    derivative = 1.0 - 2.0 * w_arr * speed_m_s * cos_beta / GRAVITY_M_S2
    return bool(np.all(derivative > 0.0) or np.all(derivative < 0.0))
