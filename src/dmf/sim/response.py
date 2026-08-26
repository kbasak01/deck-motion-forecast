"""Linear vessel response: per-DOF transfer functions and time-domain motion synthesis.

Each degree of freedom is a linear second-order system driven by the wave field and
evaluated at the encounter frequency::

    H_dof(w, w_e) = gain * heading_factor(beta) * F_exc(w) * rolloff(k, L)
                    / (1 - (w_e/wn)**2 + 2j*zeta*(w_e/wn))

with excitations

===== ============================================ ==============================
DOF    ``F_exc(w)`` per unit elevation amplitude     Heading factor
===== ============================================ ==============================
Heave  Smith factor ``exp(-k*draft)``, dimensionless ``1`` (weak dependence)
Pitch  wave slope ``k``, radians per metre           ``sqrt(cos(beta)**2 + eps_p**2)``
Roll   wave slope ``k``, radians per metre           ``sqrt(sin(beta)**2 + eps_r**2)``
===== ============================================ ==============================

Units: ``w``, ``w_e`` and ``wn`` are radians per second; ``k`` is radians per metre;
headings are degrees at the public boundary and radians internally; roll and pitch
transfer magnitudes are radians per metre of wave amplitude, heave is metres per metre.
:class:`MotionRecord` converts angles to degrees at the storage boundary.

Modelling limitations carried forward to the README: linear seakeeping only, with no
nonlinear roll damping, no parametric resonance and no green water; unidirectional
(long-crested) seas with no spreading and no swell/wind-sea bimodality; the three modes are
uncoupled, so there is no roll-yaw or pitch-heave coupling; heading and speed are constant
within a realization.
"""

from dataclasses import dataclass

import numpy as np

from dmf.sim.encounter import encounter_frequency, wave_number
from dmf.sim.spectra import WaveComponents
from dmf.sim.vessel import DofParams, Vessel
from dmf.typedefs import ComplexArray, FloatArray

__all__ = [
    "DOF_NAMES",
    "MotionRecord",
    "dof_transfer",
    "heading_factor",
    "length_rolloff",
    "second_order_transfer",
    "synthesize_motion",
]

#: The three simulated degrees of freedom, in the order used by every public API here.
DOF_NAMES: tuple[str, str, str] = ("roll", "pitch", "heave")

#: Time-sample block size for the superposition inner loop, matching
#: :mod:`dmf.sim.spectra`. Keeps the ``(n_samples, n_components)`` trig temporaries in
#: cache rather than materialising an hour-long record in one allocation.
_TIME_CHUNK = 8192


def second_order_transfer(w_e: FloatArray, wn_rad_s: float, zeta: float) -> ComplexArray:
    """Unit-gain second-order transfer function evaluated at encounter frequency.

    Returns ``1 / (1 - (w_e/wn)**2 + 2j*zeta*(w_e/wn))``. Limits, which
    ``tests/test_response.py`` asserts: magnitude ``1`` as ``w_e -> 0``, ``1/(2*zeta)`` at
    ``w_e = wn``, and ``-> 0`` as ``w_e -> inf``. The DOF gain is applied by
    :func:`dof_transfer`, not here.

    Callers must pass ``abs(w_e)``. In following seas the signed encounter frequency goes
    negative, and a negative argument would flip the sign of the damping term and turn the
    mode into an energy source. The sign belongs in the cosine argument of the time-domain
    synthesis, not in the transfer function.

    Args:
        w_e: Encounter frequencies, radians per second, non-negative.
        wn_rad_s: Undamped natural frequency, radians per second, positive.
        zeta: Damping ratio, dimensionless, positive.

    Returns:
        Complex transfer function, dimensionless, same shape as ``w_e``. The argument is
        the phase lag of the response behind the excitation, in radians.

    Raises:
        ValueError: If ``wn_rad_s <= 0``, ``zeta <= 0``, or any entry of ``w_e`` is
            negative.
    """
    if wn_rad_s <= 0.0:
        raise ValueError(f"wn_rad_s must be positive, got {wn_rad_s}")
    if zeta <= 0.0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    w_arr = np.asarray(w_e, dtype=np.float64)
    if bool(np.any(w_arr < 0.0)):
        raise ValueError(
            "w_e must be non-negative; pass abs(w_e) and carry the sign in the "
            "time-domain cosine argument instead"
        )
    ratio = w_arr / wn_rad_s
    denominator = (1.0 - ratio**2) + 2.0j * zeta * ratio
    return np.asarray(1.0 / denominator, dtype=np.complex128)


def heading_factor(dof: str, heading_deg: float, residual: float) -> float:
    """Directional excitation factor for one DOF at one encounter angle.

    ==== =========================================== ========================================
    DOF   Factor                                       Rationale
    ==== =========================================== ========================================
    roll  ``sqrt(sin(beta)**2 + residual**2)``         beam seas roll the ship
    pitch ``sqrt(cos(beta)**2 + residual**2)``         head and following seas pitch the ship
    heave ``1.0``                                      heave depends only weakly on heading
    ==== =========================================== ========================================

    Both roll and pitch carry a residual floor. Without it the factor is identically zero
    at one of the corpus headings -- roll in head and following seas, pitch in beam seas --
    which gives a dead channel whose skill score against persistence is 0/0. Beam seas are
    the whole test set of the ``unseen_heading`` evaluation regime, so a zero pitch floor
    would not be a corner case there but the entire regime. Physically the floor stands in
    for hull asymmetry and for the short-crested residual excitation that a strictly
    unidirectional model discards; it is an engineering stand-in, not a derived quantity,
    and is flagged as such in the README limitations.

    At ``residual = 0.05`` the off-axis heading sits about 26 dB below the on-axis one, so
    Gate 1 criterion 6 (roll RMS drops by at least 10x from beam to head seas) remains a
    real test.

    Args:
        dof: One of :data:`DOF_NAMES`.
        heading_deg: Encounter angle, degrees. 180 head, 90 beam, 0 following. Converted
            to radians internally.
        residual: The floor for the requested DOF, dimensionless -- that is,
            :attr:`dmf.sim.vessel.Vessel.roll_residual` for roll and
            :attr:`dmf.sim.vessel.Vessel.pitch_residual` for pitch. Ignored for heave.

    Returns:
        Dimensionless factor in ``[0, sqrt(1 + residual**2)]``.

    Raises:
        ValueError: If ``dof`` is not in :data:`DOF_NAMES`, or ``residual`` is negative.
    """
    if dof not in DOF_NAMES:
        raise ValueError(f"unknown dof {dof!r}, expected one of {DOF_NAMES}")
    if residual < 0.0:
        raise ValueError(f"residual must be non-negative, got {residual}")
    if dof == "heave":
        return 1.0
    beta_rad = np.radians(heading_deg)
    on_axis = np.sin(beta_rad) if dof == "roll" else np.cos(beta_rad)
    return float(np.hypot(on_axis, residual))


def length_rolloff(k: FloatArray, length_m: float) -> FloatArray:
    """Wavelength-versus-ship-length attenuation of the wave excitation.

    Uses ``exp(-(k*L/(4*pi))**2)``, i.e. unity for waves much longer than the ship and
    decaying as a Gaussian once the wavelength drops below about ``L``. Physically this
    stands in for the cancellation that occurs when several wave crests act along the hull
    at once; a hull does not respond to ripples. The functional form is a smooth
    engineering approximation, not a strip-theory result, and is flagged as such in the
    README limitations.

    At ``k*L = 4*pi`` (wavelength ``= L/2``) the factor is ``1/e``.

    Args:
        k: Wave numbers, radians per metre, non-negative.
        length_m: Ship length, metres, positive.

    Returns:
        Attenuation factor in ``(0, 1]``, same shape as ``k``, monotonically decreasing
        in ``k``.

    Raises:
        ValueError: If ``length_m <= 0`` or any entry of ``k`` is negative.
    """
    if length_m <= 0.0:
        raise ValueError(f"length_m must be positive, got {length_m}")
    k_arr = np.asarray(k, dtype=np.float64)
    if bool(np.any(k_arr < 0.0)):
        raise ValueError("wave numbers must be non-negative")
    return np.asarray(np.exp(-((k_arr * length_m / (4.0 * np.pi)) ** 2)), dtype=np.float64)


def dof_transfer(
    dof: str,
    w: FloatArray,
    w_e: FloatArray,
    vessel: Vessel,
    heading_deg: float,
) -> ComplexArray:
    """Full complex response amplitude operator for one DOF.

    Combines gain, heading factor, wave excitation, length rolloff and the second-order
    resonance::

        H = gain * heading_factor * F_exc(w) * length_rolloff(k, L)
            * second_order_transfer(abs(w_e), wn, zeta)

    The excitation is evaluated at the **wave** frequency ``w`` (the wave number and Smith
    factor are properties of the wave field, not of the ship's motion through it) while the
    resonance is evaluated at ``abs(w_e)``. The sign of ``w_e`` is not used here; it is
    applied by :func:`synthesize_motion` in the cosine argument.

    Args:
        dof: One of :data:`DOF_NAMES`.
        w: Wave angular frequencies in the earth frame, radians per second.
        w_e: Signed encounter frequencies, radians per second, same shape as ``w``, from
            :func:`dmf.sim.encounter.encounter_frequency`.
        vessel: Hull parameters.
        heading_deg: Encounter angle, degrees.

    Returns:
        Complex RAO, same shape as ``w``. Units are radians of roll or pitch per metre of
        wave amplitude, or metres of heave per metre of wave amplitude.

    Raises:
        ValueError: If ``dof`` is unknown or ``w`` and ``w_e`` have different shapes.
    """
    if dof not in DOF_NAMES:
        raise ValueError(f"unknown dof {dof!r}, expected one of {DOF_NAMES}")
    w_arr = np.asarray(w, dtype=np.float64)
    w_e_arr = np.asarray(w_e, dtype=np.float64)
    if w_arr.shape != w_e_arr.shape:
        raise ValueError(
            f"w and w_e must have the same shape, got {w_arr.shape} and {w_e_arr.shape}"
        )

    params, residual = _dof_params(dof, vessel)
    k = wave_number(w_arr)
    # Heave is driven by the Smith (pressure-reduction) factor: the orbital pressure that
    # drives heave decays as exp(-k*z) with depth, so a deep hull feels short waves only
    # weakly. Roll and pitch are driven by wave slope, k*a radians per metre of amplitude.
    excitation = np.exp(-k * vessel.draft_m) if dof == "heave" else k
    static = (
        params.gain
        * heading_factor(dof, heading_deg, residual)
        * excitation
        * length_rolloff(k, vessel.length_m)
    )
    resonance = second_order_transfer(np.abs(w_e_arr), 2.0 * np.pi / params.tn_s, params.zeta)
    return np.asarray(static * resonance, dtype=np.complex128)


@dataclass(frozen=True)
class MotionRecord:
    """One realization of vessel motion, at the storage unit convention.

    Angles are **degrees** and angular rates **degrees per second** here, because this is
    the storage boundary; the physics that produced them worked in radians throughout.

    Attributes:
        t_s: Sample times, seconds, shape ``(n,)``.
        roll_deg: Roll angle, degrees, shape ``(n,)``. Positive to starboard.
        pitch_deg: Pitch angle, degrees, shape ``(n,)``. Positive bow-up.
        heave_m: Heave displacement, metres, shape ``(n,)``. Positive up.
        roll_rate_dps: Roll rate, degrees per second, shape ``(n,)``.
        pitch_rate_dps: Pitch rate, degrees per second, shape ``(n,)``.
        heave_rate_m_s: Heave velocity, metres per second, shape ``(n,)``.
        heave_acc_m_s2: Heave acceleration, metres per second squared, shape ``(n,)``.
            This is the channel the ``imu`` observation mode double-integrates.
    """

    t_s: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    heave_m: FloatArray
    roll_rate_dps: FloatArray
    pitch_rate_dps: FloatArray
    heave_rate_m_s: FloatArray
    heave_acc_m_s2: FloatArray


def synthesize_motion(
    components: WaveComponents,
    vessel: Vessel,
    heading_deg: float,
    speed_m_s: float,
    t_s: FloatArray,
) -> MotionRecord:
    """Synthesize a motion record by linear superposition over the wave components.

    For each DOF and each component ``i``::

        x(t)  = sum_i a_i * abs(H_i) * cos(w_e_i*t + phi_i + arg(H_i))

    where ``w_e_i`` is the **signed** encounter frequency and ``H_i`` is evaluated at
    ``abs(w_e_i)``. Every DOF uses the same phase set ``phi_i`` as the elevation, which is
    what preserves the physical phase relationships between roll, pitch and heave -- the
    structure a multivariate forecaster is meant to exploit.

    Rates and accelerations are differentiated **analytically in the frequency domain**,
    never by finite differences::

        d/dt  cos(w_e*t + psi) = -w_e * sin(w_e*t + psi)
        d2/dt2 cos(w_e*t + psi) = -w_e**2 * cos(w_e*t + psi)

    Finite differencing would inject a frequency-dependent amplitude and phase error and
    would amplify the highest synthesis frequency the most, exactly where the physics is
    least trustworthy. The signed ``w_e`` matters here: a following-seas component with
    ``w_e < 0`` contributes a rate of the opposite sign, and squaring it for the
    acceleration is what makes the sign convention self-consistent.

    Args:
        components: Wave component set, shared across all DOFs.
        vessel: Hull parameters.
        heading_deg: Encounter angle, degrees. 180 head, 90 beam, 0 following.
        speed_m_s: Forward speed, metres per second.
        t_s: Sample times, seconds. The caller slices off the spin-up transient; this
            function evaluates a steady-state superposition and has no transient of its
            own, so the discard exists only to decorrelate the ``t = 0`` phase alignment
            across realizations.

    Returns:
        The motion record, angles in degrees and rates in degrees per second.

    Raises:
        ValueError: If ``speed_m_s`` is negative.
    """
    if speed_m_s < 0.0:
        raise ValueError(f"speed_m_s must be non-negative, got {speed_m_s}")
    t_arr = np.asarray(t_s, dtype=np.float64)
    w = components.w_rad_s
    w_e = encounter_frequency(w, speed_m_s, heading_deg)

    # One complex phasor per component per DOF: z = a * H * exp(1j*phi). The wave phase
    # ``phi`` is shared across all three DOFs, which is exactly what preserves their
    # physical phase relationships. Writing the superposition as
    # ``x(t) = Re(sum_i z_i * exp(1j*w_e_i*t))`` makes analytic differentiation a
    # multiplication by ``1j*w_e`` and keeps the signed encounter frequency in one place.
    wave_phasor = components.amplitude_m * np.exp(1j * components.phase_rad)
    z_roll = dof_transfer("roll", w, w_e, vessel, heading_deg) * wave_phasor
    z_pitch = dof_transfer("pitch", w, w_e, vessel, heading_deg) * wave_phasor
    z_heave = dof_transfer("heave", w, w_e, vessel, heading_deg) * wave_phasor
    channels = np.stack(
        [
            z_roll,
            z_pitch,
            z_heave,
            1j * w_e * z_roll,
            1j * w_e * z_pitch,
            1j * w_e * z_heave,
            -(w_e**2) * z_heave,
        ],
        axis=1,
    )

    flat_t = t_arr.reshape(-1)
    out = np.empty((flat_t.size, channels.shape[1]), dtype=np.float64)
    real_part = np.ascontiguousarray(channels.real)
    imag_part = np.ascontiguousarray(channels.imag)
    for start in range(0, flat_t.size, _TIME_CHUNK):
        stop = min(start + _TIME_CHUNK, flat_t.size)
        theta = np.outer(flat_t[start:stop], w_e)
        # Re(z * exp(1j*theta)) = Re(z)*cos(theta) - Im(z)*sin(theta).
        out[start:stop] = np.cos(theta) @ real_part - np.sin(theta) @ imag_part

    to_deg = float(np.degrees(1.0))
    return MotionRecord(
        t_s=t_arr,
        roll_deg=out[:, 0] * to_deg,
        pitch_deg=out[:, 1] * to_deg,
        heave_m=out[:, 2],
        roll_rate_dps=out[:, 3] * to_deg,
        pitch_rate_dps=out[:, 4] * to_deg,
        heave_rate_m_s=out[:, 5],
        heave_acc_m_s2=out[:, 6],
    )


def _dof_params(dof: str, vessel: Vessel) -> tuple[DofParams, float]:
    """Return the modal parameters and heading-factor residual for one DOF.

    Args:
        dof: One of :data:`DOF_NAMES`.
        vessel: Hull parameters.

    Returns:
        A ``(params, residual)`` pair. ``residual`` is dimensionless and is zero for
        heave, whose heading factor has no floor because it is already unity everywhere.

    Raises:
        ValueError: If ``dof`` is not in :data:`DOF_NAMES`.
    """
    if dof == "roll":
        return vessel.roll, vessel.roll_residual
    if dof == "pitch":
        return vessel.pitch, vessel.pitch_residual
    if dof == "heave":
        return vessel.heave, 0.0
    raise ValueError(f"unknown dof {dof!r}, expected one of {DOF_NAMES}")
