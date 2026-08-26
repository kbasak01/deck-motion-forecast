"""Vessel dynamic parameters and their YAML representation.

A vessel is reduced to three uncoupled second-order modes (roll, pitch, heave) plus the
geometry needed for the wavelength-vs-length rolloff and the Smith pressure-reduction
factor. This is a deliberately linear, uncoupled model: see the limitations note in
:mod:`dmf.sim.response`.

Periods are **seconds**, lengths **metres**, damping ratios and gains dimensionless.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DofParams", "Vessel", "load_vessel"]


@dataclass(frozen=True)
class DofParams:
    """Second-order modal parameters for one degree of freedom.

    Attributes:
        tn_s: Undamped natural period, seconds. The natural frequency used by the physics
            is ``wn = 2*pi/tn_s`` radians per second.
        zeta: Damping ratio, dimensionless. Roll is lightly damped (0.05-0.12), which is
            what makes it narrowband and comparatively predictable; pitch and heave are
            heavily damped (0.25-0.50) and broadband.
        gain: Static response gain, dimensionless. For roll and pitch it converts wave
            slope (radians) to attitude (radians); for heave it converts Smith-corrected
            elevation (metres) to heave (metres). This is the single calibration knob that
            sets the absolute motion amplitudes, and therefore the number Gate 1
            criterion 7 checks.
    """

    tn_s: float
    zeta: float
    gain: float


@dataclass(frozen=True)
class Vessel:
    """A hull's seakeeping parameter set.

    Attributes:
        name: Short identifier used as a Parquet metadata column value, e.g. ``"frigate"``.
        length_m: Length between perpendiculars, metres. Sets the wavelength-vs-length
            rolloff, so short waves do not drive full-amplitude response.
        draft_m: Mean draft, metres. Sets the Smith factor ``exp(-k*draft)`` on heave
            excitation.
        beam_m: Moulded beam, metres. Carried for reporting; the uncoupled modal model does
            not use it.
        roll: Roll modal parameters. Angles in radians internally, degrees when stored.
        pitch: Pitch modal parameters. Angles in radians internally, degrees when stored.
        heave: Heave modal parameters. Metres throughout.
        roll_residual: Dimensionless floor ``eps`` in the roll heading factor
            ``sqrt(sin(beta)**2 + eps**2)``. Stands in for hull asymmetry and
            short-crested residual roll excitation, which a strictly unidirectional
            ``abs(sin(beta))`` model throws away. Without it, roll is identically zero in
            head and following seas -- a quarter of the corpus each -- and every
            skill-vs-persistence score there becomes 0/0. At ``eps = 0.05`` head-seas roll
            sits about 26 dB below beam seas, which keeps Gate 1 criterion 6 a real test.
        pitch_residual: The same floor for pitch, ``sqrt(cos(beta)**2 + eps**2)``,
            dimensionless. Introduced for exactly the reason ``roll_residual`` was: a pure
            ``abs(cos(beta))`` factor makes pitch identically zero at 90 deg, so every
            beam-seas realization would carry a dead pitch channel and its
            skill-vs-persistence score would be 0/0. That is not a corner case here -- the
            ``unseen_heading`` evaluation regime holds out beam seas, so its entire test
            set would be affected. Physically it stands in for the same neglected effects
            as the roll floor: hull asymmetry and short-crested residual excitation.
    """

    name: str
    length_m: float
    draft_m: float
    beam_m: float
    roll: DofParams
    pitch: DofParams
    heave: DofParams
    roll_residual: float
    pitch_residual: float = 0.05


def load_vessel(path: Path) -> Vessel:
    """Load a vessel definition from YAML.

    The YAML keys mirror the dataclass field names exactly, with ``roll``, ``pitch`` and
    ``heave`` as nested mappings of :class:`DofParams` fields. Units are as documented on
    the dataclasses: seconds, metres, dimensionless.

    Args:
        path: Path to a file under ``configs/sim/vessels/``.

    Returns:
        The parsed vessel.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, or if any period, length or damping
            ratio is non-positive.
    """
    if not path.exists():
        raise FileNotFoundError(f"vessel config not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of a vessel config must be a mapping")

    def _dof(key: str) -> DofParams:
        entry = raw.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: missing or malformed '{key}' mapping")
        missing = {"tn_s", "zeta", "gain"} - set(entry)
        if missing:
            raise ValueError(f"{path}: '{key}' is missing {sorted(missing)}")
        tn_s = float(entry["tn_s"])
        zeta = float(entry["zeta"])
        gain = float(entry["gain"])
        if tn_s <= 0.0:
            raise ValueError(f"{path}: {key}.tn_s must be positive, got {tn_s}")
        if zeta <= 0.0:
            raise ValueError(f"{path}: {key}.zeta must be positive, got {zeta}")
        return DofParams(tn_s=tn_s, zeta=zeta, gain=gain)

    missing_top = {"name", "length_m", "draft_m", "beam_m", "roll_residual"} - set(raw)
    if missing_top:
        raise ValueError(f"{path}: missing required keys {sorted(missing_top)}")
    length_m = float(raw["length_m"])
    draft_m = float(raw["draft_m"])
    beam_m = float(raw["beam_m"])
    for label, value in (("length_m", length_m), ("draft_m", draft_m), ("beam_m", beam_m)):
        if value <= 0.0:
            raise ValueError(f"{path}: {label} must be positive, got {value}")
    roll_residual = float(raw["roll_residual"])
    pitch_residual = float(raw.get("pitch_residual", 0.05))
    for label, value in (
        ("roll_residual", roll_residual),
        ("pitch_residual", pitch_residual),
    ):
        if value < 0.0:
            raise ValueError(f"{path}: {label} must be non-negative, got {value}")
    return Vessel(
        name=str(raw["name"]),
        length_m=length_m,
        draft_m=draft_m,
        beam_m=beam_m,
        roll=_dof("roll"),
        pitch=_dof("pitch"),
        heave=_dof("heave"),
        roll_residual=roll_residual,
        pitch_residual=pitch_residual,
    )
