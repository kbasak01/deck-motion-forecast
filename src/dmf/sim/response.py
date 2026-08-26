"""Second-order DOF response to wave excitation.

Each degree of freedom is modelled as a linear second-order system driven by wave
elevation (heave) or wave slope (roll, pitch)::

    H_dof(w_e) = K_dof * F_exc(w) / (1 - (w_e/wn)^2 + 2j*zeta*(w_e/wn))

The response is synthesised by superposition in the frequency domain **reusing the phase
set drawn for the elevation record**. That is what preserves the physical phase
relationships between roll, pitch and heave, which is exactly the structure a multivariate
forecaster should be able to exploit.

Angle convention: transfer-function gains are returned in the units documented on
:class:`dmf.sim.vessel.Vessel` (degrees per unit wave slope for roll and pitch, metres per
metre for heave), so synthesised roll and pitch series are in **degrees** and heave in
**metres**, matching the Parquet schema.
"""

from dmf.sim.vessel import Vessel
from dmf.typedefs import ComplexArray, FloatArray

__all__ = [
    "differentiate_series",
    "heave_transfer",
    "length_attenuation",
    "pitch_transfer",
    "roll_transfer",
    "second_order_response",
    "smith_factor",
    "synthesize_dof",
]


def second_order_response(w_e_rad_s: FloatArray, wn_rad_s: float, zeta: float) -> ComplexArray:
    """Evaluate the complex second-order response function.

    ``1 / (1 - (w_e/wn)^2 + 2j*zeta*(w_e/wn))``. The magnitude gives the dynamic
    amplification and the argument gives the phase lag between excitation and response;
    both matter, because discarding the phase would decorrelate the DOFs.

    Args:
        w_e_rad_s: Encounter angular frequencies, radians per second. The absolute value
            is used, so negative encounter frequencies from following seas are handled
            without producing a spurious sign flip.
        wn_rad_s: Undamped natural angular frequency of the DOF, radians per second.
        zeta: Damping ratio, dimensionless, in (0, 1).

    Returns:
        Complex response factor at each encounter frequency, dimensionless, same shape as
        ``w_e_rad_s``.

    Raises:
        ValueError: If ``wn_rad_s`` is not strictly positive or ``zeta`` is outside (0, 1).
    """
    raise NotImplementedError


def smith_factor(k_rad_m: FloatArray, draft_m: float) -> FloatArray:
    """Compute the Smith (pressure-attenuation) factor for heave excitation.

    ``exp(-k * draft)``. Accounts for the decay of the wave-induced pressure field with
    depth, so short waves excite the hull less than their surface amplitude suggests.

    Args:
        k_rad_m: Wavenumbers, radians per metre.
        draft_m: Mean draft, metres.

    Returns:
        Attenuation factor in (0, 1], dimensionless, same shape as ``k_rad_m``.

    Raises:
        ValueError: If ``draft_m`` is negative.
    """
    raise NotImplementedError


def length_attenuation(k_rad_m: FloatArray, length_m: float) -> FloatArray:
    """Compute the wavelength-versus-ship-length excitation rolloff.

    Uses ``exp(-(k*L/(4*pi))^2)``, so that waves much shorter than the hull do not drive a
    full-amplitude response. This is a documented modelling choice, not a derivation: the
    exact rolloff form is a stand-in for the strip-theory integration of pressure over a
    hull that a full seakeeping code would perform, and the corpus card records it as a
    limitation.

    Args:
        k_rad_m: Wavenumbers, radians per metre.
        length_m: Length between perpendiculars, metres.

    Returns:
        Attenuation factor in (0, 1], dimensionless, same shape as ``k_rad_m``.

    Raises:
        ValueError: If ``length_m`` is not strictly positive.
    """
    raise NotImplementedError


def heave_transfer(
    w_rad_s: FloatArray,
    w_e_rad_s: FloatArray,
    vessel: Vessel,
) -> ComplexArray:
    """Build the heave transfer function from wave elevation to heave displacement.

    Heave is driven by wave **elevation**, attenuated by the Smith factor and the
    length rolloff, and has only a weak heading dependence, so no heading factor is
    applied here.

    Args:
        w_rad_s: Absolute wave angular frequencies, radians per second. Used for the
            wavenumber-dependent attenuation terms.
        w_e_rad_s: Encounter angular frequencies, radians per second. Used for the
            resonant denominator.
        vessel: Hull parameters supplying ``tn_heave_s``, ``zeta_heave``, ``k_heave``,
            ``draft_m`` and ``length_m``.

    Returns:
        Complex transfer function, metres of heave per metre of wave amplitude, same
        shape as ``w_rad_s``.

    Raises:
        ValueError: If ``w_rad_s`` and ``w_e_rad_s`` differ in shape.
    """
    raise NotImplementedError


def pitch_transfer(
    w_rad_s: FloatArray,
    w_e_rad_s: FloatArray,
    beta_deg: float,
    vessel: Vessel,
) -> ComplexArray:
    """Build the pitch transfer function from wave elevation to pitch angle.

    Pitch is driven by wave **slope** (``k*a``, radians) and scales with ``|cos(beta)|``,
    peaking in head and following seas and vanishing in beam seas.

    Args:
        w_rad_s: Absolute wave angular frequencies, radians per second.
        w_e_rad_s: Encounter angular frequencies, radians per second.
        beta_deg: Encounter angle, degrees, per the :mod:`dmf.sim.encounter` convention
            (180 = head seas).
        vessel: Hull parameters supplying ``tn_pitch_s``, ``zeta_pitch``, ``k_pitch`` and
            ``length_m``.

    Returns:
        Complex transfer function, **degrees** of pitch per metre of wave amplitude, same
        shape as ``w_rad_s``.

    Raises:
        ValueError: If ``w_rad_s`` and ``w_e_rad_s`` differ in shape.
    """
    raise NotImplementedError


def roll_transfer(
    w_rad_s: FloatArray,
    w_e_rad_s: FloatArray,
    beta_deg: float,
    vessel: Vessel,
) -> ComplexArray:
    """Build the roll transfer function from wave elevation to roll angle.

    Roll is driven by wave **slope** (``k*a``, radians) and scales with ``|sin(beta)|``,
    peaking in beam seas and vanishing in head and following seas. Because ``zeta_roll``
    is small, this transfer function is sharply peaked near ``wn_roll``; Gate 1 asserts
    that the beam-seas roll response spectrum peaks within 5 percent of ``wn_roll``, and
    that head-seas roll RMS is at least an order of magnitude smaller.

    Args:
        w_rad_s: Absolute wave angular frequencies, radians per second.
        w_e_rad_s: Encounter angular frequencies, radians per second.
        beta_deg: Encounter angle, degrees, per the :mod:`dmf.sim.encounter` convention.
        vessel: Hull parameters supplying ``tn_roll_s``, ``zeta_roll``, ``k_roll`` and
            ``length_m``.

    Returns:
        Complex transfer function, **degrees** of roll per metre of wave amplitude, same
        shape as ``w_rad_s``.

    Raises:
        ValueError: If ``w_rad_s`` and ``w_e_rad_s`` differ in shape.
    """
    raise NotImplementedError


def synthesize_dof(
    t_s: FloatArray,
    w_e_rad_s: FloatArray,
    amplitude_m: FloatArray,
    phase_rad: FloatArray,
    transfer: ComplexArray,
) -> FloatArray:
    """Synthesise one DOF time series by frequency-domain superposition.

    ``x(t) = sum_i |H_i| * a_i * cos(w_e_i*t + phi_i + arg(H_i))``.

    ``phase_rad`` must be the **same** array used for the elevation record in
    :func:`dmf.sim.spectra.synthesize_elevation`. Drawing fresh phases here would produce
    roll, pitch and heave series that are individually plausible but mutually
    uncorrelated, silently removing the cross-channel structure the forecaster is meant to
    learn.

    Args:
        t_s: Time samples, seconds, shape ``(n_samples,)``.
        w_e_rad_s: Encounter angular frequencies, radians per second, shape
            ``(n_components,)``. Encounter rather than absolute frequency is used here
            because the response is observed in the vessel frame.
        amplitude_m: Component wave amplitudes, metres, shape ``(n_components,)``.
        phase_rad: Component phases, radians, shape ``(n_components,)``.
        transfer: Complex transfer function per component, shape ``(n_components,)``.
            Its units set the output units: degrees for roll and pitch, metres for heave.

    Returns:
        The DOF time series, shape ``(n_samples,)``, in the units carried by ``transfer``.

    Raises:
        ValueError: If the four component arrays differ in shape.
    """
    raise NotImplementedError


def differentiate_series(x: FloatArray, fs_hz: float) -> FloatArray:
    """Differentiate a uniformly sampled series with respect to time.

    Used to produce rate and acceleration channels from displacement channels. A spectral
    or centred-difference scheme is required rather than a forward difference, since a
    forward difference imposes a half-sample phase lag that would show up as a systematic
    error in the phase-lag metric of :mod:`dmf.eval.phase`.

    Args:
        x: Uniformly sampled series, shape ``(n_samples,)``. Units are arbitrary and set
            the output units.
        fs_hz: Sampling rate, hertz.

    Returns:
        Time derivative at each sample, shape ``(n_samples,)``, in input units per second
        (degrees per second for angles, metres per second for heave).

    Raises:
        ValueError: If ``fs_hz`` is not strictly positive.
    """
    raise NotImplementedError
