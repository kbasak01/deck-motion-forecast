"""Load the MSS ShipX vessel structure from its MATLAB file.

`HYDRO/vessels_shipx/s175/s175.mat` in the MSS toolbox is a MATLAB v5 MAT-file, which
`scipy.io.loadmat` reads directly. No MATLAB or Octave licence or installation is needed to
obtain the RAO tables themselves; Octave is used elsewhere in Phase 8 only to check our
synthesis against MSS's own `waveMotionRAO.m`.

Units, as stored by MSS (verified against the long-wave limit, see
:func:`assert_long_wave_limit`):

* ``motionRAO.amp[dof]``  -- metres per metre of wave amplitude for surge/sway/heave
  (DOF 1-3), **radians** per metre for roll/pitch/yaw (DOF 4-6).
* ``motionRAO.phase[dof]`` -- radians.
* ``motionRAO.w``          -- wave angular frequency, rad/s.
* ``headings``             -- radians, **0 following seas, pi head seas**, i.e. the same
  convention the corpus uses in degrees. See :mod:`dmf.mss.convert` for how this was
  established; MSS's own source comments disagree with each other on the point.

DOF ordering is the SNAME 6-DOF order: surge, sway, heave, roll, pitch, yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.io import loadmat

from dmf.typedefs import ComplexArray, FloatArray

#: SNAME 6-DOF names, in MSS/ShipX storage order.
DOF_NAMES: Final[tuple[str, ...]] = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: Index into :data:`DOF_NAMES` for the three DOFs the corpus forecasts.
DOF_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(DOF_NAMES)}

#: Which DOFs are angular (RAO in rad/m) rather than translational (RAO in m/m).
ANGULAR_DOFS: Final[frozenset[str]] = frozenset({"roll", "pitch", "yaw"})

GRAVITY_M_S2: Final[float] = 9.80665


@dataclass(frozen=True)
class MSSVessel:
    """One MSS vessel structure, restricted to what motion synthesis needs.

    Attributes:
        name: Hull name, from ``vessel.main.name``.
        w_rad_s: RAO frequency grid, rad/s, shape ``(n_freq,)``, ascending.
        headings_rad: RAO heading grid, radians, shape ``(n_head,)``, ascending over
            ``[0, 2*pi)``. 0 is following seas and pi is head seas.
        amp: Per-DOF RAO amplitude, 6 arrays of shape ``(n_freq, n_head, n_speed)``.
            Units m/m for DOF 1-3 and rad/m for DOF 4-6.
        phase: Per-DOF RAO phase, radians, same shapes as ``amp``.
        length_m: Length between perpendiculars, metres.
        beam_m: Beam, metres.
        draft_m: Draft, metres.
        gm_t_m: Transverse metacentric height, metres.
        k44_m: Roll radius of gyration, metres.
        gravity_m_s2: Gravitational acceleration as MSS stores it, m/s^2. MSS ships
            9.8100004 where this project uses 9.80665. The difference is physically
            negligible but not numerically: it enters the encounter frequency, and with an
            absolute time axis beginning at t = 120 s the accumulated phase drift reaches
            ~1% pointwise. The MSS value is used for MSS records so that the Octave parity
            check compares like with like.
        source: Path the structure was read from.
    """

    name: str
    w_rad_s: FloatArray
    headings_rad: FloatArray
    amp: tuple[FloatArray, ...]
    phase: tuple[FloatArray, ...]
    length_m: float
    beam_m: float
    draft_m: float
    gm_t_m: float
    k44_m: float
    gravity_m_s2: float
    source: Path

    @property
    def roll_period_s(self) -> float:
        """Natural roll period from ``T = 2*pi*k44/sqrt(g*GM_T)``, seconds.

        Reported so the MSS hull's roll mode can be compared against the period our own
        ``configs/sim/vessels/s175.yaml`` derives from an *assumed* GM.
        """
        return float(2.0 * np.pi * self.k44_m / np.sqrt(self.gravity_m_s2 * self.gm_t_m))

    def heading_index(self, heading_deg: float) -> int:
        """Index of the RAO heading grid point nearest ``heading_deg``.

        Args:
            heading_deg: Encounter angle in the corpus convention (180 head, 0 following).

        Returns:
            Index into :attr:`headings_rad`.
        """
        target = np.deg2rad(float(heading_deg) % 360.0)
        return int(np.argmin(np.abs(self.headings_rad - target)))

    def complex_rao(self, dof: str, heading_deg: float, speed_index: int = 0) -> ComplexArray:
        """Complex RAO for one DOF on the native frequency grid.

        Real and imaginary parts are formed *before* any interpolation elsewhere, which is
        how MSS's own ``waveMotionRAO.m`` avoids phase-unwrapping artifacts.

        Args:
            dof: One of :data:`DOF_NAMES`.
            heading_deg: Encounter angle, degrees, corpus convention.
            speed_index: Index into the RAO speed axis. Defaults to 0 (zero speed), which
                is what MSS's own ``waveMotionRAO.m`` uses; forward speed enters through
                the encounter frequency rather than through the RAO table.

        Returns:
            Complex array of shape ``(n_freq,)``, units m/m or rad/m.

        Raises:
            ValueError: If ``dof`` is not a known DOF name.
        """
        if dof not in DOF_INDEX:
            raise ValueError(f"unknown dof {dof!r}, expected one of {DOF_NAMES}")
        i = DOF_INDEX[dof]
        j = self.heading_index(heading_deg)
        a = np.asarray(self.amp[i][:, j, speed_index], dtype=np.float64)
        p = np.asarray(self.phase[i][:, j, speed_index], dtype=np.float64)
        return np.asarray(a * np.exp(1j * p), dtype=np.complex128)


def load_mss_vessel(path: Path) -> MSSVessel:
    """Read an MSS vessel ``.mat`` file.

    Args:
        path: Path to e.g. ``mss/upstream/HYDRO/vessels_shipx/s175/s175.mat``.

    Returns:
        The parsed :class:`MSSVessel`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the file has no ``vessel`` variable or no ``motionRAO`` field.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"MSS vessel file not found: {path}. Clone the toolbox first, e.g. "
            "`git clone https://github.com/cybergalactic/MSS.git mss/upstream`."
        )
    raw = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if "vessel" not in raw:
        raise KeyError(f"{path} has no 'vessel' variable; variables: {sorted(raw)}")
    vessel = raw["vessel"]
    if not hasattr(vessel, "motionRAO"):
        raise KeyError(
            f"{path} has no 'motionRAO' field. Force RAOs alone are not enough to "
            "synthesise motion without integrating the vessel equations."
        )
    rao = vessel.motionRAO
    main = vessel.main
    amp = tuple(np.asarray(a, dtype=np.float64) for a in rao.amp)
    phase = tuple(np.asarray(p, dtype=np.float64) for p in rao.phase)
    return MSSVessel(
        name=str(main.name),
        w_rad_s=np.asarray(rao.w, dtype=np.float64),
        headings_rad=np.asarray(vessel.headings, dtype=np.float64),
        amp=amp,
        phase=phase,
        length_m=float(main.Lpp),
        beam_m=float(main.B),
        draft_m=float(main.T),
        gm_t_m=float(main.GM_T),
        k44_m=float(main.k44),
        gravity_m_s2=float(main.g),
        source=path,
    )


def assert_long_wave_limit(vessel: MSSVessel, *, rtol: float = 0.10) -> None:
    """Check the RAO tables against the analytic long-wave limit.

    As ``w -> 0`` a ship contours the wave: heave amplitude tends to one metre per metre of
    wave amplitude, and pitch amplitude tends to the wave slope ``k = w**2/g`` radians per
    metre. This is the check that pins down the *units* of the angular RAOs (rad/m, not
    deg/m) without trusting a comment, and it is what established that MSS heave is
    positive-down (its phase tends to pi, not 0, while contouring).

    Args:
        vessel: Loaded MSS vessel.
        rtol: Relative tolerance on both limits.

    Raises:
        AssertionError: If either limit is violated.
    """
    head = 180.0
    w = vessel.w_rad_s
    lo = int(np.argmin(np.abs(w - 0.25)))
    heave = vessel.complex_rao("heave", head)
    pitch = vessel.complex_rao("pitch", head)

    heave_amp = float(np.abs(heave[lo]))
    if not np.isclose(heave_amp, 1.0, rtol=rtol):
        raise AssertionError(
            f"heave RAO at w={w[lo]:.4f} rad/s is {heave_amp:.4f} m/m, expected ~1.0 "
            "(a ship contours a long wave). The RAO table is not what we think it is."
        )

    k = float(w[lo] ** 2 / GRAVITY_M_S2)
    pitch_amp = float(np.abs(pitch[lo]))
    if not np.isclose(pitch_amp, k, rtol=2.0 * rtol):
        raise AssertionError(
            f"pitch RAO at w={w[lo]:.4f} rad/s is {pitch_amp:.6f} but the wave slope is "
            f"k={k:.6f}. If the ratio is ~57 the table is in degrees per metre, not "
            "radians per metre."
        )
