"""Physics invariants for vessel response -- Gate 1, part two.

Empty in Phase 0. Implemented in Phase 1.

Gate 1 criteria owned by this module:

6. In beam seas the roll response spectrum peaks within 5 percent of ``wn_roll``; in head
   seas roll RMS drops by at least an order of magnitude.
7. Roll RMS at SS5 beam seas lands in a physically sensible range (single-digit degrees).
   Print the value and check it against published seakeeping figures rather than only
   asserting a wide band.
8. Increasing forward speed in head seas shifts the response spectrum peak to higher
   ``w_e``.

Also covered here:

- Encounter frequency: ``w_e > w`` in head seas, ``w_e < w`` in following seas, and
  :func:`dmf.sim.encounter.is_encounter_monotonic` returns False somewhere in the band for
  the following-seas cases that the corpus grid must handle explicitly.
- Directional monotonicity: changing ``gamma``, ``Hs``, ``Tp``, ``beta`` and ``U`` each
  move the output in the physically correct direction.
- Phase coherence: roll, pitch and heave synthesised from a shared phase set are
  cross-correlated, and are not when the phase set is redrawn per DOF.

Units: frequencies rad/s, headings degrees at the boundary, speeds m/s inside the physics
and knots only where the name says so, roll and pitch degrees on the record, heave metres.
Every test prints what it measured; run with ``-s`` when recording Gate 1.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from scipy import signal

from conftest import FS_HZ, HEADINGS_DEG, SEA_STATES, SPEEDS_KN, W_MAX_RAD_S, W_MIN_RAD_S
from dmf.config import SeaState
from dmf.sim.encounter import (
    GRAVITY_M_S2,
    encounter_frequency,
    is_encounter_monotonic,
    knots_to_m_s,
    wave_number,
)
from dmf.sim.response import (
    MotionRecord,
    dof_transfer,
    heading_factor,
    length_rolloff,
    second_order_transfer,
    synthesize_motion,
)
from dmf.sim.spectra import WaveComponents, jonswap, sample_components
from dmf.sim.vessel import Vessel
from dmf.typedefs import FloatArray

RECORD_S = 3600.0
SHORT_RECORD_S = 1200.0
N_COMPONENTS = 300

HEAD_SEAS_DEG = 180.0
BEAM_SEAS_DEG = 90.0
FOLLOWING_SEAS_DEG = 0.0

#: Non-monotonic ``w -> w_e`` cells of the corpus grid, as (heading_deg, speed_kn).
#:
#: ``dw_e/dw = 1 - 2*w*U*cos(beta)/g`` flips sign at ``w_crit = g/(2*U*cos beta)``, which
#: only exists for ``cos beta > 0``. Over the synthesis band [0.2, 2.5] rad/s that puts
#: ``w_crit`` at 1.124 rad/s for (45 deg, 12 kn) and at 2.247 rad/s for (45 deg, 6 kn) --
#: both inside the band. 135 deg and 180 deg have ``cos beta <= 0`` and are monotonic at
#: every speed; 90 deg has ``cos beta = 0``, so ``w_e = w`` exactly.
NON_MONOTONIC_CELLS: frozenset[tuple[float, float]] = frozenset({(45.0, 6.0), (45.0, 12.0)})


def _time_axis(duration_s: float) -> FloatArray:
    return np.arange(0.0, duration_s, 1.0 / FS_HZ)


def _components(
    sea: SeaState,
    rng: np.random.Generator,
    n_components: int = N_COMPONENTS,
) -> WaveComponents:
    return sample_components(
        hs_m=sea.hs_m,
        tp_s=sea.tp_s,
        gamma=sea.gamma,
        n_components=n_components,
        w_min_rad_s=W_MIN_RAD_S,
        w_max_rad_s=W_MAX_RAD_S,
        rng=rng,
        jitter=True,
    )


def _record(
    sea: SeaState,
    vessel: Vessel,
    rng: np.random.Generator,
    heading_deg: float,
    speed_kn: float = 0.0,
    duration_s: float = SHORT_RECORD_S,
) -> MotionRecord:
    return synthesize_motion(
        _components(sea, rng),
        vessel,
        heading_deg,
        knots_to_m_s(speed_kn),
        _time_axis(duration_s),
    )


def _averaged_psd(signals: list[FloatArray], nperseg: int) -> tuple[FloatArray, FloatArray]:
    """Return (angular frequency rad/s, PSD per rad/s) averaged over independent records.

    ``scipy.signal.welch`` returns a one-sided PSD per Hz; the conversion to per rad/s is
    ``S(w) = S_f(f)/(2*pi)`` with ``w = 2*pi*f``. The peak *location* is unaffected by the
    scale factor, but the analytic cross-check needs the right units.
    """
    total: FloatArray | None = None
    freq_hz: FloatArray = np.zeros(0)
    for x in signals:
        freq_hz, p = signal.welch(
            x,
            fs=FS_HZ,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
            detrend="constant",
        )
        total = p if total is None else total + p
    assert total is not None
    return 2.0 * np.pi * freq_hz, total / (len(signals) * 2.0 * np.pi)


def _interpolated_peak(w: FloatArray, psd: FloatArray, w_lo: float, w_hi: float) -> float:
    """Locate a spectral peak to sub-bin accuracy by parabolic fit on the log-PSD.

    Gate 1 criterion 6 allows 5 percent of ``wn_roll = 0.5236 rad/s``, i.e. 0.026 rad/s.
    Even at nperseg = 4096 the raw bin spacing is 0.0153 rad/s, so bin-centre reporting
    alone would burn half the tolerance on quantisation. A parabola through the three
    log-PSD samples straddling the maximum is exact for a Gaussian peak and good to a few
    percent of a bin for the near-Lorentzian roll resonance.
    """
    band = (w >= w_lo) & (w <= w_hi)
    idx = int(np.flatnonzero(band)[np.argmax(psd[band])])
    y0, y1, y2 = np.log(psd[idx - 1]), np.log(psd[idx]), np.log(psd[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
    dw = float(w[1] - w[0])
    return float(w[idx] + delta * dw)


def _max_abs_xcorr(a: FloatArray, b: FloatArray, max_lag_s: float = 20.0) -> float:
    an = (a - a.mean()) / a.std()
    bn = (b - b.mean()) / b.std()
    n = an.size
    max_lag = int(max_lag_s * FS_HZ)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        c = (
            float(np.mean(an[: n - lag] * bn[lag:]))
            if lag >= 0
            else float(np.mean(an[-lag:] * bn[: n + lag]))
        )
        best = max(best, abs(c))
    return best


# ---------------------------------------------------------------------------
# Encounter kinematics
# ---------------------------------------------------------------------------


def test_encounter_frequency_direction() -> None:
    w = np.linspace(W_MIN_RAD_S, W_MAX_RAD_S, 1001)
    u = knots_to_m_s(12.0)

    w_e_head = encounter_frequency(w, u, HEAD_SEAS_DEG)
    w_e_follow = encounter_frequency(w, u, FOLLOWING_SEAS_DEG)
    w_e_quarter = encounter_frequency(w, u, 45.0)
    w_e_still = encounter_frequency(w, 0.0, HEAD_SEAS_DEG)
    w_e_beam = encounter_frequency(w, u, BEAM_SEAS_DEG)

    print(
        f"[enc] U=12 kn ({u:.3f} m/s): head w_e/w in "
        f"[{(w_e_head / w).min():.3f}, {(w_e_head / w).max():.3f}], following w_e min = "
        f"{w_e_follow.min():+.3f} rad/s (negative means the waves overtake the ship)"
    )
    assert np.all(w_e_head > w)
    assert np.all(w_e_follow < w)
    assert np.all(w_e_quarter < w)
    assert np.allclose(w_e_still, w)
    assert np.allclose(w_e_beam, w)


def test_encounter_monotonicity_over_corpus_grid() -> None:
    w = np.linspace(W_MIN_RAD_S, W_MAX_RAD_S, 4001)
    flagged: list[tuple[float, float, float]] = []
    for heading_deg in HEADINGS_DEG:
        for speed_kn in SPEEDS_KN:
            u = knots_to_m_s(speed_kn)
            monotonic = is_encounter_monotonic(w, u, heading_deg)
            expected = (heading_deg, speed_kn) not in NON_MONOTONIC_CELLS
            if not monotonic:
                cos_beta = float(np.cos(np.radians(heading_deg)))
                flagged.append((heading_deg, speed_kn, GRAVITY_M_S2 / (2.0 * u * cos_beta)))
            assert monotonic is expected, (
                f"heading={heading_deg} deg, speed={speed_kn} kn: "
                f"is_encounter_monotonic returned {monotonic}, expected {expected}"
            )
    print(
        "[enc] non-monotonic cells (heading deg, speed kn, w_crit rad/s): "
        + ", ".join(f"({h:.0f}, {s:.0f}, {c:.3f})" for h, s, c in flagged)
    )
    assert {(h, s) for h, s, _ in flagged} == NON_MONOTONIC_CELLS


def test_wave_number_is_deep_water_dispersion() -> None:
    w = np.linspace(0.2, 2.5, 11)
    k = wave_number(w)
    wavelength_m = 2.0 * np.pi / k
    print(
        f"[enc] wavelength at w=0.2 rad/s: {wavelength_m[0]:.1f} m, "
        f"at w=2.5 rad/s: {wavelength_m[-1]:.1f} m"
    )
    assert np.allclose(k, w**2 / GRAVITY_M_S2)


# ---------------------------------------------------------------------------
# Transfer-function shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zeta", [0.05, 0.08, 0.12, 0.35])
def test_second_order_transfer_limits(zeta: float) -> None:
    wn = 2.0 * np.pi / 12.0
    h = second_order_transfer(np.array([0.0, wn, 1000.0 * wn]), wn, zeta)
    mags = np.abs(h)
    print(
        f"[rao] zeta={zeta:.2f}: |H(0)|={mags[0]:.4f} (expect 1), |H(wn)|={mags[1]:.4f} "
        f"(expect {1.0 / (2.0 * zeta):.4f}), |H(1000 wn)|={mags[2]:.3e} (expect 0)"
    )
    assert mags[0] == pytest.approx(1.0, rel=1e-12)
    assert mags[1] == pytest.approx(1.0 / (2.0 * zeta), rel=1e-12)
    assert mags[2] < 1e-5
    # Response lags excitation: the phase is zero at DC and -pi/2 at resonance.
    assert np.angle(h[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.angle(h[1]) == pytest.approx(-np.pi / 2.0, abs=1e-12)


def test_heading_factors(frigate: Vessel) -> None:
    eps_roll = frigate.roll_residual
    eps_pitch = frigate.pitch_residual
    roll_beam = heading_factor("roll", BEAM_SEAS_DEG, eps_roll)
    roll_head = heading_factor("roll", HEAD_SEAS_DEG, eps_roll)
    roll_follow = heading_factor("roll", FOLLOWING_SEAS_DEG, eps_roll)
    print(
        f"[hdg] roll factor: beam={roll_beam:.4f}, head={roll_head:.4f}, "
        f"following={roll_follow:.4f}  -> head is "
        f"{20.0 * np.log10(roll_head / roll_beam):.1f} dB below beam"
    )
    assert roll_beam == pytest.approx(np.hypot(1.0, eps_roll), rel=1e-12)
    assert roll_head == pytest.approx(eps_roll, abs=1e-12)
    assert roll_follow == pytest.approx(eps_roll, abs=1e-12)

    # Pitch carries the same residual floor as roll, for the same reason: a pure
    # abs(cos(beta)) factor is identically zero at 90 deg, so the pitch channel would be
    # dead in every beam-seas realization and its skill score against persistence would be
    # 0/0 -- and beam seas are the entire test set of the ``unseen_heading`` regime.
    pitch_head = heading_factor("pitch", HEAD_SEAS_DEG, eps_pitch)
    pitch_beam = heading_factor("pitch", BEAM_SEAS_DEG, eps_pitch)
    print(
        f"[hdg] pitch factor: head={pitch_head:.4f}, beam={pitch_beam:.4f}  -> beam is "
        f"{20.0 * np.log10(pitch_beam / pitch_head):.1f} dB below head "
        f"(ratio {pitch_head / pitch_beam:.2f}x)"
    )
    assert pitch_head == pytest.approx(np.hypot(1.0, eps_pitch), rel=1e-12)
    assert pitch_beam == pytest.approx(eps_pitch, abs=1e-12)
    assert pitch_head / pitch_beam == pytest.approx(20.0, rel=0.01)

    assert heading_factor("heave", BEAM_SEAS_DEG, eps_roll) == pytest.approx(1.0, rel=1e-12)


def test_length_rolloff_is_bounded_and_decaying(frigate: Vessel) -> None:
    k = wave_number(np.linspace(0.05, 4.0, 500))
    rolloff = length_rolloff(k, frigate.length_m)
    at_zero = length_rolloff(np.array([0.0]), frigate.length_m)
    half_ship = length_rolloff(np.array([4.0 * np.pi / frigate.length_m]), frigate.length_m)
    print(
        f"[rolloff] L={frigate.length_m:.0f} m: max={rolloff.max():.4f}, "
        f"min={rolloff.min():.3e}, value at k*L=4*pi = {half_ship[0]:.4f} (expect 1/e)"
    )
    assert np.all(rolloff > 0.0)
    assert np.all(rolloff <= 1.0)
    assert np.all(np.diff(rolloff) < 0.0)
    assert at_zero[0] == pytest.approx(1.0, rel=1e-12)
    assert half_ship[0] == pytest.approx(float(np.exp(-1.0)), rel=1e-12)


def test_heave_transfer_tends_to_gain_at_long_waves(frigate: Vessel) -> None:
    w = np.array([1e-4])
    h = dof_transfer("heave", w, w, frigate, BEAM_SEAS_DEG)
    w_high = np.array([5.0])
    h_high = dof_transfer("heave", w_high, w_high, frigate, BEAM_SEAS_DEG)
    print(
        f"[rao] heave |H(w->0)|={np.abs(h)[0]:.6f} (expect gain={frigate.heave.gain}), "
        f"|H(5 rad/s)|={np.abs(h_high)[0]:.3e}"
    )
    assert np.abs(h)[0] == pytest.approx(frigate.heave.gain, rel=1e-3)
    assert np.abs(h_high)[0] < 1e-3 * frigate.heave.gain


# ---------------------------------------------------------------------------
# Invariant 6a -- beam-seas roll spectrum peaks at the roll natural frequency
# ---------------------------------------------------------------------------

ROLL_PSD_NPERSEG = 4096
ROLL_PSD_N_SEEDS = 4


@pytest.mark.slow
@pytest.mark.parametrize("sea", [SEA_STATES[2], SEA_STATES[3]], ids=["SS5", "SS6"])
def test_beam_seas_roll_peaks_at_natural_frequency(
    sea: SeaState, frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    wn = 2.0 * np.pi / frigate.roll.tn_s
    records = [
        _record(sea, frigate, make_rng(200 + s), BEAM_SEAS_DEG, 0.0, RECORD_S).roll_deg
        for s in range(ROLL_PSD_N_SEEDS)
    ]
    w, psd = _averaged_psd(records, ROLL_PSD_NPERSEG)
    peak = _interpolated_peak(w, psd, 0.5 * wn, 2.0 * wn)

    # Analytic cross-check: at U = 0 the encounter frequency equals the wave frequency, so
    # the response spectrum is simply |H(w)|^2 S(w) and its peak can be computed directly.
    w_fine = np.linspace(W_MIN_RAD_S, W_MAX_RAD_S, 200_001)
    h = dof_transfer("roll", w_fine, w_fine, frigate, BEAM_SEAS_DEG)
    analytic = np.abs(h) ** 2 * jonswap(w_fine, sea.hs_m, sea.tp_s, sea.gamma)
    analytic_peak = float(w_fine[int(np.argmax(analytic))])

    err_pct = 100.0 * (peak - wn) / wn
    print(
        f"[inv6a] {sea.name} beam seas: wn_roll={wn:.4f} rad/s, Welch peak={peak:.4f} "
        f"({err_pct:+.2f} %), analytic peak={analytic_peak:.4f} "
        f"({100.0 * (analytic_peak - wn) / wn:+.2f} %)"
    )
    assert abs(err_pct) < 5.0
    assert peak == pytest.approx(analytic_peak, rel=0.03)


# ---------------------------------------------------------------------------
# Invariant 6b -- head-seas roll RMS drops by at least an order of magnitude
# ---------------------------------------------------------------------------


def test_head_seas_roll_rms_collapses(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    sea = SEA_STATES[2]
    beam = _record(sea, frigate, make_rng(300), BEAM_SEAS_DEG).roll_deg
    head = _record(sea, frigate, make_rng(300), HEAD_SEAS_DEG).roll_deg
    ratio = float(np.std(beam) / np.std(head))
    print(
        f"[inv6b] {sea.name}: roll RMS beam={np.std(beam):.3f} deg, "
        f"head={np.std(head):.4f} deg, ratio={ratio:.2f}x "
        f"({20.0 * np.log10(ratio):.1f} dB)"
    )
    assert ratio >= 10.0


# ---------------------------------------------------------------------------
# Invariant 7 -- SS5 beam-seas roll RMS is physically sensible
# ---------------------------------------------------------------------------


def test_ss5_beam_roll_rms_is_single_digit_degrees(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    """SS5 beam-seas roll RMS must land in single-digit degrees.

    The vessel ``gain`` is the calibration anchor for this number: it is the only knob that
    scales absolute motion amplitude, so if this value comes out wrong the fix belongs in
    ``configs/sim/vessels/frigate.yaml``, not in the physics. Published frigate seakeeping
    figures put beam-seas roll at roughly 2-4 degrees RMS in SS5, so a result at the top of
    the asserted band should be treated as suspicious even though it passes.
    """
    sea = SEA_STATES[2]
    rms = [
        float(np.std(_record(sea, frigate, make_rng(400 + s), BEAM_SEAS_DEG).roll_deg))
        for s in range(3)
    ]
    mean_rms = float(np.mean(rms))
    print(
        f"\n[inv7] *** SS5 beam-seas roll RMS = {mean_rms:.3f} deg *** "
        f"(per-seed {', '.join(f'{v:.3f}' for v in rms)}; "
        f"significant single amplitude ~{2.0 * mean_rms:.2f} deg, "
        f"expected published range 2-4 deg RMS)\n"
    )
    assert 1.0 <= mean_rms <= 9.0


# ---------------------------------------------------------------------------
# Invariant 8 -- forward speed shifts the response peak to higher encounter frequency
# ---------------------------------------------------------------------------

PITCH_PSD_NPERSEG = 2048
PITCH_PSD_N_SEEDS = 4


@pytest.mark.slow
def test_forward_speed_shifts_pitch_peak_up(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    """Head-seas response peak moves to higher ``w_e`` as speed rises.

    Pitch, not roll: roll at zeta = 0.08 is so lightly damped that its response spectrum
    stays pinned to ``wn_roll`` whatever the excitation is shifted to, so it cannot detect
    an encounter-frequency shift. Pitch at zeta = 0.40 is broadband enough that its peak
    tracks the excitation.
    """
    sea = SEA_STATES[2]
    peaks: list[float] = []
    for speed_kn in SPEEDS_KN:
        records = [
            _record(sea, frigate, make_rng(500 + s), HEAD_SEAS_DEG, speed_kn, RECORD_S).pitch_deg
            for s in range(PITCH_PSD_N_SEEDS)
        ]
        w, psd = _averaged_psd(records, PITCH_PSD_NPERSEG)
        peaks.append(_interpolated_peak(w, psd, 0.3, 2.0))
    print(
        "[inv8] head-seas pitch response peak vs speed: "
        + ", ".join(f"{s:.0f} kn -> {p:.4f} rad/s" for s, p in zip(SPEEDS_KN, peaks, strict=True))
    )
    assert peaks[0] < peaks[1] < peaks[2]


# ---------------------------------------------------------------------------
# Directional monotonicity
# ---------------------------------------------------------------------------


def test_roll_rms_scales_exactly_linearly_with_hs(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    """A linear system driven by a spectrum whose amplitudes scale as ``Hs`` is exactly linear.

    ``A_i = sqrt(2*S(w_i)*dw)`` and ``S`` is proportional to ``Hs**2``, so with the same
    seed the two records are the same waveform scaled by the Hs ratio. This is asserted as
    an equality, not as a band: any departure means the amplitude scaling is wrong.
    """
    low = SeaState(name="low", hs_m=2.0, tp_s=9.7, gamma=3.3)
    high = SeaState(name="high", hs_m=4.0, tp_s=9.7, gamma=3.3)
    rms_low = float(np.std(_record(low, frigate, make_rng(600), BEAM_SEAS_DEG).roll_deg))
    rms_high = float(np.std(_record(high, frigate, make_rng(600), BEAM_SEAS_DEG).roll_deg))
    ratio = rms_high / rms_low
    print(
        f"[mono] roll RMS {rms_low:.4f} deg at Hs=2 m -> {rms_high:.4f} deg at Hs=4 m; "
        f"ratio={ratio:.6f} (expect exactly 2)"
    )
    assert ratio == pytest.approx(2.0, rel=1e-6)


def test_roll_rms_rises_as_damping_falls(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    sea = SEA_STATES[2]
    rms: list[float] = []
    zetas = [0.05, 0.08, 0.12]
    for zeta in zetas:
        vessel = dataclasses.replace(frigate, roll=dataclasses.replace(frigate.roll, zeta=zeta))
        rms.append(float(np.std(_record(sea, vessel, make_rng(700), BEAM_SEAS_DEG).roll_deg)))
    print(
        "[mono] roll RMS vs damping: "
        + ", ".join(f"zeta={z:.2f} -> {r:.3f} deg" for z, r in zip(zetas, rms, strict=True))
    )
    assert rms[0] > rms[1] > rms[2]


def test_roll_rms_peaks_when_tp_approaches_roll_period(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    tps = [7.5, 9.7, 12.0, 15.0]
    rms = [
        float(
            np.std(
                _record(
                    SeaState(name=f"Tp{tp}", hs_m=3.3, tp_s=tp, gamma=3.3),
                    frigate,
                    make_rng(800),
                    BEAM_SEAS_DEG,
                ).roll_deg
            )
        )
        for tp in tps
    ]
    resonant = int(np.argmax(rms))
    print(
        f"[mono] roll RMS vs Tp (Tn_roll={frigate.roll.tn_s:.1f} s): "
        + ", ".join(f"Tp={tp:.1f} s -> {r:.3f} deg" for tp, r in zip(tps, rms, strict=True))
        + f"  -> peak at Tp={tps[resonant]:.1f} s"
    )
    assert tps[resonant] == pytest.approx(12.0)


def test_roll_rms_is_maximal_in_beam_seas(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    sea = SEA_STATES[2]
    rms = {
        heading: float(np.std(_record(sea, frigate, make_rng(900), heading).roll_deg))
        for heading in HEADINGS_DEG
    }
    print(
        "[mono] roll RMS vs heading: "
        + ", ".join(f"{h:.0f} deg -> {r:.3f}" for h, r in rms.items())
    )
    assert max(rms, key=lambda h: rms[h]) == BEAM_SEAS_DEG


# ---------------------------------------------------------------------------
# Phase coherence across DOFs
# ---------------------------------------------------------------------------


def test_shared_phase_set_couples_the_dofs(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    """Roll, pitch and heave must come from one phase set, and it must be observable.

    A multivariate forecaster's whole advantage over three independent univariate ones is
    the deterministic phase relationship between the DOFs. If a future refactor ever draws
    phases per DOF, the corpus silently loses that structure and every multivariate result
    becomes meaningless -- so the coupling is asserted, and the de-coupled control is
    asserted too.
    """
    sea = SEA_STATES[2]
    heading = 135.0
    components = _components(sea, make_rng(1000))
    t = _time_axis(SHORT_RECORD_S)
    shared = synthesize_motion(components, frigate, heading, 0.0, t)

    redraw = make_rng(1001)
    scrambled = dataclasses.replace(
        components, phase_rad=redraw.uniform(0.0, 2.0 * np.pi, components.phase_rad.size)
    )
    independent = synthesize_motion(scrambled, frigate, heading, 0.0, t)

    coupled = _max_abs_xcorr(shared.roll_deg, shared.pitch_deg)
    decoupled = _max_abs_xcorr(shared.roll_deg, independent.pitch_deg)
    coupled_rh = _max_abs_xcorr(shared.roll_deg, shared.heave_m)
    print(
        f"[phase] max |xcorr| roll-pitch: shared phases={coupled:.3f}, "
        f"independent phases={decoupled:.3f}; roll-heave shared={coupled_rh:.3f}"
    )
    assert coupled > 0.5
    assert coupled_rh > 0.5
    assert decoupled < 0.35
    assert coupled > 2.0 * decoupled


# ---------------------------------------------------------------------------
# Record integrity
# ---------------------------------------------------------------------------


def test_motion_record_shapes_and_rate_consistency(
    frigate: Vessel, make_rng: Callable[[int], np.random.Generator]
) -> None:
    """Analytic rates must agree with a high-order numerical derivative of the record.

    This does not make finite differencing the definition -- the rates are differentiated
    in the frequency domain -- but a central difference on a 10 Hz record whose energy sits
    below 2.5 rad/s is accurate to better than a percent, which is enough to catch a
    missing ``w_e`` factor, a sign error, or degrees/radians confusion in the rate channel.
    """
    sea = SEA_STATES[2]
    rec = _record(sea, frigate, make_rng(1100), BEAM_SEAS_DEG, 0.0, 600.0)
    n = rec.t_s.size
    for name in (
        "roll_deg",
        "pitch_deg",
        "heave_m",
        "roll_rate_dps",
        "pitch_rate_dps",
        "heave_rate_m_s",
        "heave_acc_m_s2",
    ):
        assert getattr(rec, name).shape == (n,)

    numeric_rate = np.gradient(rec.roll_deg, 1.0 / FS_HZ)
    numeric_acc = np.gradient(rec.heave_rate_m_s, 1.0 / FS_HZ)
    interior = slice(10, -10)
    rate_err = float(
        np.std(rec.roll_rate_dps[interior] - numeric_rate[interior])
        / np.std(numeric_rate[interior])
    )
    acc_diff = rec.heave_acc_m_s2[interior] - numeric_acc[interior]
    acc_err = float(np.std(acc_diff) / np.std(numeric_acc[interior]))
    print(
        f"[record] analytic vs central-difference: roll rate rel error={rate_err:.4f}, "
        f"heave acceleration rel error={acc_err:.4f}"
    )
    assert rate_err < 0.02
    assert acc_err < 0.02
