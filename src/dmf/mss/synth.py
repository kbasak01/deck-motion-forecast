"""Random-phase synthesis of 6-DOF motion from MSS strip-theory RAOs.

This is a NumPy port of MSS's ``LIBRARY/environment/waveMotionRAO.m`` (Fossen 2021,
eqs. 10.83 and 10.105). Two deliberate differences from the m-file, both recorded in
`docs/protocol.md`:

1. **Vectorised over time.** The m-file returns a 6x1 column for one scalar ``t`` and is
   driven from a loop. For a constant heading and speed the complex RAO is constant in
   time, so it is hoisted out and the component sum is evaluated over the whole time axis
   at once. The arithmetic is identical; only the loop order changes.
2. **Phases are an argument, not a hidden global.** ``waveMotionRAO.m`` seeds
   ``rng(12345,"twister")`` into a ``persistent`` variable, so every fresh MATLAB/Octave
   process yields the *same* realisation. Drawing three "seeds" from repeated calls would
   silently produce one record three times. Here the phases are passed in.

Both differences are checked rather than asserted: :mod:`dmf.mss.octave` runs the
unmodified m-file and requires agreement with this module.

Two frequency-grid conventions are offered, and which one is used is a scientific choice,
not a detail:

* :func:`sample_mss_grid` reproduces MSS's own ``waveDirectionalSpectrum.m`` -- bins over
  ``[0, omega_max]`` with a frequency drawn uniformly inside each bin. This is the fully
  independent path: MSS spectrum, MSS RAOs.
* :func:`sample_corpus_grid` reuses :func:`dmf.sim.spectra.sample_components`, so the wave
  field is bit-for-bit the corpus's own. This is the *controlled* path: the only thing that
  differs from a corpus realisation is the transfer function, which isolates the vessel
  response as the cause of any change in forecast skill.

Both grids jitter frequencies within bins, so neither carries the synthesis-periodicity
artifact that `tests/test_spectra.py` guards against; MSS independently arrived at the same
mitigation.

Everything here is simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from dmf.mss.vessel import DOF_INDEX, DOF_NAMES, MSSVessel
from dmf.sim.spectra import jonswap, sample_components
from dmf.typedefs import ComplexArray, FloatArray

GRAVITY_M_S2: Final[float] = 9.80665


@dataclass(frozen=True)
class WaveGrid:
    """A discretised wave field: one amplitude and phase per frequency component.

    Attributes:
        w_rad_s: Component frequencies, rad/s, shape ``(n,)``.
        amplitude_m: Component amplitudes ``sqrt(2*S*dw)``, metres, shape ``(n,)``.
        phase_rad: Component phases, radians, shape ``(n,)``.
        source: ``"mss"`` or ``"corpus"``, recording which convention produced the grid.
    """

    w_rad_s: FloatArray
    amplitude_m: FloatArray
    phase_rad: FloatArray
    source: str


@dataclass(frozen=True)
class MSSMotion:
    """6-DOF wave-frequency motion, in MSS's own units and sign convention.

    No unit or sign conversion has been applied; that happens in :mod:`dmf.mss.convert`.

    Attributes:
        t_s: Sample times, seconds, shape ``(n_t,)``.
        eta: Displacements, shape ``(6, n_t)``. Metres for surge/sway/heave, **radians**
            for roll/pitch/yaw. SNAME axes (x forward, y starboard, z **down**).
        eta_dot: Time derivatives of ``eta``, shape ``(6, n_t)``, m/s and rad/s.
        elevation_m: Wave elevation as seen by the moving ship, metres, shape ``(n_t,)``,
            positive up. Synthesised at the **encounter** frequency, which is what MSS's
            own ``waveMotionRAO.m`` returns. Its zero-crossing period is Doppler-shifted by
            forward speed and is therefore *not* the sea state's Tz.
        elevation_earth_m: Wave elevation in the earth frame, metres, shape ``(n_t,)``,
            positive up. Synthesised at the **wave** frequency, so its Hs and Tz are
            properties of the sea state alone and are speed-independent. This is the series
            the spectrum-match predicate must be read from.
        w_encounter_rad_s: Encounter frequency per component, rad/s, shape ``(n,)``.
    """

    t_s: FloatArray
    eta: FloatArray
    eta_dot: FloatArray
    elevation_m: FloatArray
    elevation_earth_m: FloatArray
    w_encounter_rad_s: FloatArray


def sample_mss_grid(
    hs_m: float,
    tp_s: float,
    gamma: float,
    n_components: int,
    omega_max_rad_s: float,
    rng: np.random.Generator,
) -> WaveGrid:
    """Draw a wave grid using MSS's ``waveDirectionalSpectrum.m`` convention.

    The band ``[0, omega_max]`` is split into ``n_components`` equal bins; each component
    sits at its bin midpoint displaced by ``U(-dw/2, dw/2)``. Amplitudes are
    ``sqrt(2*S*dw)``.

    Args:
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: JONSWAP peak enhancement factor.
        n_components: Number of frequency bins.
        omega_max_rad_s: Upper frequency limit, rad/s.
        rng: Random generator supplying jitter and phases.

    Returns:
        The sampled :class:`WaveGrid`.

    Raises:
        ValueError: If ``n_components`` is not positive or ``omega_max_rad_s`` is not.
    """
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if omega_max_rad_s <= 0.0:
        raise ValueError(f"omega_max_rad_s must be positive, got {omega_max_rad_s}")
    dw = float(omega_max_rad_s) / n_components
    mid = (np.arange(n_components, dtype=np.float64) + 0.5) * dw
    w = mid + (rng.random(n_components) - 0.5) * dw
    s = jonswap(w, hs_m=hs_m, tp_s=tp_s, gamma=gamma)
    amp = np.sqrt(2.0 * s * dw)
    phase = rng.random(n_components) * 2.0 * np.pi
    return WaveGrid(w_rad_s=w, amplitude_m=amp, phase_rad=phase, source="mss")


def sample_corpus_grid(
    hs_m: float,
    tp_s: float,
    gamma: float,
    n_components: int,
    w_min_rad_s: float,
    w_max_rad_s: float,
    rng: np.random.Generator,
) -> WaveGrid:
    """Draw a wave grid identical to the one the corpus generator uses.

    Delegates to :func:`dmf.sim.spectra.sample_components`, so a realisation built on this
    grid differs from a corpus realisation *only* in the transfer function applied to it.

    Args:
        hs_m: Significant wave height, metres.
        tp_s: Spectral peak period, seconds.
        gamma: JONSWAP peak enhancement factor.
        n_components: Number of frequency bins.
        w_min_rad_s: Lower band limit, rad/s.
        w_max_rad_s: Upper band limit, rad/s.
        rng: Random generator supplying jitter and phases.

    Returns:
        The sampled :class:`WaveGrid`.
    """
    comp = sample_components(
        hs_m=hs_m,
        tp_s=tp_s,
        gamma=gamma,
        n_components=n_components,
        w_min_rad_s=w_min_rad_s,
        w_max_rad_s=w_max_rad_s,
        rng=rng,
        jitter=True,
    )
    return WaveGrid(
        w_rad_s=comp.w_rad_s,
        amplitude_m=comp.amplitude_m,
        phase_rad=comp.phase_rad,
        source="corpus",
    )


def mss_encounter_frequency(
    w_rad_s: FloatArray,
    speed_m_s: float,
    heading_deg: float,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> FloatArray:
    """Encounter frequency in MSS's formulation.

    MSS computes ``w_e = |w - (w**2/g)*U*cos(beta)|`` (``waveMotionRAO.m``), taking the
    absolute value rather than carrying the sign. For head seas (``beta = 180 deg``,
    ``cos beta = -1``) this is ``w + w**2*U/g``, monotonically increasing with no sign
    change at any speed in this study, so the abs() is inert here. It is reproduced anyway
    so that the Octave parity check compares like with like, and because the corpus's own
    ``dmf.sim.encounter`` deliberately keeps the sign instead -- a difference that only
    matters for following and stern-quartering seas, neither of which is in this phase.

    Args:
        w_rad_s: Wave frequencies, rad/s.
        speed_m_s: Forward speed, metres per second.
        heading_deg: Encounter angle, degrees, 180 head and 0 following.
        gravity_m_s2: Gravitational acceleration. Pass ``vessel.gravity_m_s2`` for MSS
            records; MSS ships 9.8100004 against this project's 9.80665, and on an absolute
            time axis starting at 120 s that gap alone moves the record by ~1% pointwise.

    Returns:
        Encounter frequencies, rad/s, same shape as ``w_rad_s``.
    """
    w = np.asarray(w_rad_s, dtype=np.float64)
    beta = np.deg2rad(float(heading_deg))
    return np.abs(w - (w**2 / float(gravity_m_s2)) * float(speed_m_s) * np.cos(beta))


def _interpolate_rao(
    vessel: MSSVessel, dof: str, heading_deg: float, w_target: FloatArray, speed_index: int
) -> ComplexArray:
    """Interpolate one DOF's complex RAO in frequency and heading.

    Follows ``waveMotionRAO.m`` exactly, in both respects that matter:

    * **Real and imaginary parts are interpolated, never amplitude and phase.** MSS makes
      this choice explicitly to avoid phase-unwrapping artifacts at RAO nulls.
    * **Frequency first, then heading.** MSS interpolates each heading column onto the
      target frequency grid and only then interpolates across headings.

    Interpolating in heading is not optional. The RAO tables are on a 10 degree grid, so the
    corpus heading 180 deg lands on a node but 135 deg does not. Snapping to the nearest
    node instead reproduces MSS exactly at 180 deg and diverges by 38% at 135 deg -- which
    is where roll is the live channel, since head-seas roll is zero by symmetry. The Octave
    parity check is what caught this.

    Args:
        vessel: Loaded MSS vessel.
        dof: One of :data:`dmf.mss.vessel.DOF_NAMES`.
        heading_deg: Encounter angle, degrees, corpus convention.
        w_target: Frequencies to interpolate onto, rad/s.
        speed_index: RAO speed-axis index.

    Returns:
        Complex RAO at ``w_target``, shape ``(len(w_target),)``.
    """
    i = DOF_INDEX[dof]
    amp = np.asarray(vessel.amp[i][:, :, speed_index], dtype=np.float64)
    phase = np.asarray(vessel.phase[i][:, :, speed_index], dtype=np.float64)
    native_re = amp * np.cos(phase)
    native_im = amp * np.sin(phase)

    w_native = vessel.w_rad_s
    n_head = vessel.headings_rad.size
    re_f = np.empty((w_target.size, n_head), dtype=np.float64)
    im_f = np.empty((w_target.size, n_head), dtype=np.float64)
    for j in range(n_head):
        re_f[:, j] = np.interp(w_target, w_native, native_re[:, j])
        im_f[:, j] = np.interp(w_target, w_native, native_im[:, j])

    beta = float(np.deg2rad(float(heading_deg) % 360.0))
    headings = vessel.headings_rad
    re = _interp_linear_extrap(beta, headings, re_f)
    im = _interp_linear_extrap(beta, headings, im_f)
    return np.asarray(re + 1j * im, dtype=np.complex128)


def _interp_linear_extrap(x: float, xp: FloatArray, fp: FloatArray) -> FloatArray:
    """Linear interpolation in the heading axis with linear extrapolation at the ends.

    Reproduces MATLAB/Octave ``interp1(..., 'linear', 'extrap')``, which ``np.interp`` does
    not: ``np.interp`` clamps outside the range instead of extrapolating. The heading grid
    runs 0 to 350 degrees, so any heading above 350 falls in the extrapolated gap.

    Args:
        x: Query point.
        xp: Sample points, ascending, shape ``(m,)``.
        fp: Values, shape ``(n, m)`` -- interpolation is along the last axis.

    Returns:
        Interpolated values, shape ``(n,)``.
    """
    m = xp.size
    j = int(np.searchsorted(xp, x) - 1)
    j = max(0, min(j, m - 2))
    x0, x1 = float(xp[j]), float(xp[j + 1])
    w = (x - x0) / (x1 - x0)
    return np.asarray(fp[:, j] * (1.0 - w) + fp[:, j + 1] * w, dtype=np.float64)


def synthesize_mss_motion(
    vessel: MSSVessel,
    grid: WaveGrid,
    heading_deg: float,
    speed_m_s: float,
    t_s: FloatArray,
    *,
    speed_index: int = 0,
) -> MSSMotion:
    """Synthesise 6-DOF wave-frequency motion by linear superposition.

    For each DOF, ``eta(t) = sum_k |H_k| * a_k * cos(w_e,k * t + arg(H_k) + phi_k)`` and
    ``eta_dot`` is the analytic derivative (multiply by ``i*w_e``), never a finite
    difference -- matching the corpus generator's convention so that the rate channels are
    comparable.

    Args:
        vessel: Loaded MSS vessel with motion RAOs.
        grid: Wave field to drive the hull with.
        heading_deg: Encounter angle, degrees, 180 head and 0 following.
        speed_m_s: Forward speed, metres per second.
        t_s: Sample times, seconds, shape ``(n_t,)``.
        speed_index: RAO speed-axis index; 0 (zero speed) reproduces MSS's own choice.

    Returns:
        The synthesised :class:`MSSMotion`, in MSS units and signs.

    Raises:
        ValueError: If ``t_s`` is not one-dimensional.
    """
    t = np.asarray(t_s, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t_s must be one-dimensional, got shape {t.shape}")

    w_e = mss_encounter_frequency(
        grid.w_rad_s, speed_m_s, heading_deg, gravity_m_s2=vessel.gravity_m_s2
    )
    amp = grid.amplitude_m
    phi = grid.phase_rad

    # (n_components, n_t). The wave field itself, before any RAO is applied.
    argument = np.outer(w_e, t) + phi[:, None]
    elevation = np.sum(amp[:, None] * np.cos(argument), axis=0)
    # Same components at the wave frequency rather than the encounter frequency. Hs is
    # identical either way (the amplitudes are unchanged), but Tz is not: at 12 kn head
    # seas the encounter-frame record is compressed by roughly a third.
    elevation_earth = np.sum(
        amp[:, None] * np.cos(np.outer(grid.w_rad_s, t) + phi[:, None]), axis=0
    )

    eta = np.zeros((6, t.size), dtype=np.float64)
    eta_dot = np.zeros((6, t.size), dtype=np.float64)
    for i, dof in enumerate(DOF_NAMES):
        h = _interpolate_rao(vessel, dof, heading_deg, grid.w_rad_s, speed_index)
        mag = np.abs(h) * amp
        pha = np.angle(h)
        total = argument + pha[:, None]
        eta[i] = np.sum(mag[:, None] * np.cos(total), axis=0)
        # d/dt cos(w_e t + c) = -w_e sin(w_e t + c)
        eta_dot[i] = -np.sum((mag * w_e)[:, None] * np.sin(total), axis=0)

    return MSSMotion(
        t_s=t,
        eta=eta,
        eta_dot=eta_dot,
        elevation_m=elevation,
        elevation_earth_m=elevation_earth,
        w_encounter_rad_s=w_e,
    )
