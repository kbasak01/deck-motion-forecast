"""Vessel parameter container and loading.

Natural periods and damping ratios are the parameters that decide how interesting the
forecasting problem is. Roll is lightly damped (``zeta`` 0.05-0.12) and therefore
narrowband, resonant, and predictable over a few periods; heave is heavily damped
(``zeta`` 0.25-0.45) and broadband. Per-DOF skill scores are expected to differ sharply
for this reason, and that difference is a reportable result rather than a bug.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Vessel", "load_vessel", "natural_frequency"]


@dataclass(frozen=True)
class Vessel:
    """Linear seakeeping parameters for one hull.

    Attributes:
        name: Short identifier, e.g. ``"frigate"``. Stored as a Parquet metadata column.
        length_m: Length between perpendiculars, metres.
        beam_m: Moulded beam, metres.
        draft_m: Mean draft, metres. Enters the Smith factor for heave excitation.
        tn_roll_s: Undamped natural roll period, seconds. Typically 10-16.
        zeta_roll: Roll damping ratio, dimensionless. Typically 0.05-0.12.
        tn_pitch_s: Undamped natural pitch period, seconds. Typically 6-9.
        zeta_pitch: Pitch damping ratio, dimensionless. Typically 0.30-0.50.
        tn_heave_s: Undamped natural heave period, seconds. Typically 7-10.
        zeta_heave: Heave damping ratio, dimensionless. Typically 0.25-0.45.
        k_roll: Roll excitation gain, degrees of roll per unit wave slope (rad/rad),
            before heading and attenuation factors.
        k_pitch: Pitch excitation gain, degrees of pitch per unit wave slope (rad/rad),
            before heading and attenuation factors.
        k_heave: Heave excitation gain, metres of heave per metre of wave elevation,
            before the Smith factor and attenuation.
    """

    name: str
    length_m: float
    beam_m: float
    draft_m: float
    tn_roll_s: float
    zeta_roll: float
    tn_pitch_s: float
    zeta_pitch: float
    tn_heave_s: float
    zeta_heave: float
    k_roll: float
    k_pitch: float
    k_heave: float


def natural_frequency(tn_s: float) -> float:
    """Convert an undamped natural period to an undamped natural angular frequency.

    Args:
        tn_s: Undamped natural period, seconds.

    Returns:
        Undamped natural angular frequency ``wn = 2*pi/tn_s``, radians per second.

    Raises:
        ValueError: If ``tn_s`` is not strictly positive.
    """
    raise NotImplementedError


def load_vessel(path: Path) -> Vessel:
    """Load a vessel parameter set from YAML.

    Args:
        path: Path to a file under ``configs/sim/vessels/``.

    Returns:
        The parsed vessel parameters. Periods are seconds, lengths metres, damping ratios
        dimensionless, exactly as documented on :class:`Vessel`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing or a damping ratio is outside (0, 1).
    """
    raise NotImplementedError
