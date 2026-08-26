"""Physics invariants for JONSWAP synthesis -- Gate 1, part one.

Empty in Phase 0. Phase 1 must implement these before writing ``src/dmf/sim/spectra.py``;
this is the phase where an agentic coder is most likely to produce plausible-looking wrong
physics, and the tests are the only defence.

Gate 1 criteria owned by this module:

1. ``4*sqrt(m0)`` recovers the requested ``Hs`` to within 2 percent.
2. Zero-crossing period ``Tz = 2*pi*sqrt(m0/m2)`` falls in ``Tz/Tp`` in 0.71-0.78 for
   ``gamma = 3.3``.
3. Welch PSD of a synthesised 3600 s elevation record matches the analytic ``S(w)`` within
   15 percent across [0.4, 1.5] rad/s.
4. Peak amplitudes of the elevation record follow a Rayleigh distribution
   (KS test, ``p > 0.01``).
5. **Anti-periodicity.** Autocorrelation of a 1 hr record shows no spurious spike at
   ``2*pi/dw``. This is the single most important test in the project: a uniform frequency
   grid makes the record repeat, the forecaster memorises the repeat, and every downstream
   result becomes meaningless without anything looking wrong. The test should also assert
   the converse -- that synthesis with ``jitter=False`` *does* produce the spike -- so that
   it is demonstrably capable of failing.

Every test here prints the quantity it measured. Run the suite with ``-s`` when filling in
the Gate 1 report so the gate is recorded as numbers rather than as green ticks.

Units throughout: frequencies rad/s, spectral density m^2 s/rad, elevation m, time s. The
only Hz quantity is the sampling rate, and it is converted at the ``scipy.signal.welch``
boundary with ``w = 2*pi*f`` and ``S(w) = S_f(f)/(2*pi)``.
"""

import ast
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy import signal, stats

import dmf.sim
from conftest import FS_HZ, SEA_STATES, W_MAX_RAD_S, W_MIN_RAD_S
from dmf.config import SeaState
from dmf.sim.spectra import (
    WaveComponents,
    hs_from_moments,
    jonswap,
    sample_components,
    spectral_moment,
    synthesis_period_s,
    synthesize_elevation,
    tz_from_moments,
)
from dmf.typedefs import FloatArray

#: Integration grid for the analytic moment checks. The upper limit is load-bearing: the
#: ``m2`` integrand ``w**2 * S(w)`` decays only as ``w**-3``, so the integral converges
#: slowly and truncating too early inflates ``Tz``. See ``test_tz_over_tp_in_band``.
ANALYTIC_W_MIN = 0.05
ANALYTIC_W_MAX = 30.0
ANALYTIC_N = 400_000

#: Record length and component count used by the synthesis tests, seconds / dimensionless.
RECORD_S = 3600.0
N_COMPONENTS = 300


def _analytic_w() -> FloatArray:
    return np.linspace(ANALYTIC_W_MIN, ANALYTIC_W_MAX, ANALYTIC_N)


def _time_axis(duration_s: float, fs_hz: float = FS_HZ) -> FloatArray:
    return np.arange(0.0, duration_s, 1.0 / fs_hz)


def _components(
    sea: SeaState,
    rng: np.random.Generator,
    n_components: int = N_COMPONENTS,
    jitter: bool = True,
) -> WaveComponents:
    return sample_components(
        hs_m=sea.hs_m,
        tp_s=sea.tp_s,
        gamma=sea.gamma,
        n_components=n_components,
        w_min_rad_s=W_MIN_RAD_S,
        w_max_rad_s=W_MAX_RAD_S,
        rng=rng,
        jitter=jitter,
    )


def _unbiased_acf(x: FloatArray, max_lag: int) -> FloatArray:
    """Normalised autocorrelation with the ``1/(N - lag)`` (unbiased) denominator.

    The biased ``1/N`` estimator shrinks every lag by ``1 - lag/N``, which at the 817 s
    synthesis period of a 3600 s record is a 23 percent haircut -- enough that a perfectly
    periodic signal would score 0.77 instead of 1.0 and the converse half of the
    anti-periodicity test would lose its teeth.
    """
    y = x - x.mean()
    n = y.size
    spec = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[: max_lag + 1]
    acf = acf / (n - np.arange(max_lag + 1))
    return np.asarray(acf / acf[0], dtype=np.float64)


def _zero_crossing_crests(eta: FloatArray) -> FloatArray:
    """Maximum of ``eta`` between successive upward zero crossings.

    Zero-crossing crests, not all local maxima. For a broadband process the set of all
    local maxima follows a Rice distribution whose narrowbandedness parameter is not 1, so
    testing it against Rayleigh would be wrong by construction and would fail for correct
    physics.
    """
    y = eta - eta.mean()
    up = np.flatnonzero((y[:-1] <= 0.0) & (y[1:] > 0.0))
    return np.array([y[up[i] : up[i + 1]].max() for i in range(up.size - 1)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Invariant 1 -- Hs recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sea", SEA_STATES, ids=[s.name for s in SEA_STATES])
@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.3])
def test_hs_recovered_from_m0(sea: SeaState, gamma: float) -> None:
    w = _analytic_w()
    s = jonswap(w, sea.hs_m, sea.tp_s, gamma)
    m0 = spectral_moment(w, s, 0)
    hs_hat = hs_from_moments(m0)
    err_pct = 100.0 * (hs_hat - sea.hs_m) / sea.hs_m
    print(
        f"[inv1] {sea.name} gamma={gamma:.1f}: Hs_req={sea.hs_m:.3f} m  "
        f"4*sqrt(m0)={hs_hat:.4f} m  error={err_pct:+.3f} %"
    )
    assert abs(err_pct) < 2.0


# ---------------------------------------------------------------------------
# Invariant 2 -- Tz/Tp band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sea", SEA_STATES, ids=[s.name for s in SEA_STATES])
def test_tz_over_tp_in_band(sea: SeaState) -> None:
    # The integration limit, not the threshold, is what decides this test. Measured on the
    # reference implementation: w_max = 2.5 -> Tz/Tp = 0.803 (a false failure), w_max = 6
    # -> 0.782 (still outside), w_max = 30 -> 0.7778, which matches the published
    # Tp/Tz = 1.286 at gamma = 3.3. The margin against the 0.78 ceiling is only ~0.3 %.
    # If this test fails, the fix is the integration band, NOT the threshold.
    w = _analytic_w()
    s = jonswap(w, sea.hs_m, sea.tp_s, 3.3)
    m0 = spectral_moment(w, s, 0)
    m2 = spectral_moment(w, s, 2)
    tz = tz_from_moments(m0, m2)
    ratio = tz / sea.tp_s
    print(
        f"[inv2] {sea.name}: Tp={sea.tp_s:.2f} s  Tz={tz:.3f} s  Tz/Tp={ratio:.4f}  "
        f"Tp/Tz={1.0 / ratio:.4f}"
    )
    assert 0.71 <= ratio <= 0.78


# ---------------------------------------------------------------------------
# Invariant 3 -- Welch PSD versus analytic spectrum
# ---------------------------------------------------------------------------

#: Welch design for invariant 3. nperseg = 2048 at fs = 10 Hz gives a resolution of
#: 0.0307 rad/s and 35 half-overlapped segments per 3600 s record; averaging 8 independent
#: seeds gives ~280 segments, so the per-bin sampling sigma is ~6 %. Band-averaging into
#: 0.1 rad/s bands (3.3 Welch bins each) takes that to ~3 %, making the 15 % tolerance a
#: ~5 sigma test rather than a coin flip.
#:
#: nperseg = 1024 (0.061 rad/s) was measured to be too coarse, not too noisy: the
#: low-frequency flank of JONSWAP falls like exp(-1.25*(wp/w)**4), and smoothing that with
#: a 0.061 rad/s kernel biases the [0.40, 0.50] rad/s band by +22 % at SS5. That is
#: estimator bias, not a physics error, and doubling the segment length removes it.
WELCH_NPERSEG = 2048
WELCH_N_SEEDS = 8
PSD_BAND_RAD_S = (0.4, 1.5)
PSD_BAND_WIDTH_RAD_S = 0.1


@pytest.mark.slow
@pytest.mark.parametrize("sea", [SEA_STATES[2], SEA_STATES[3]], ids=["SS5", "SS6"])
def test_welch_psd_matches_analytic(
    sea: SeaState, make_rng: Callable[[int], np.random.Generator]
) -> None:
    t = _time_axis(RECORD_S)
    psd_f: FloatArray | None = None
    freq_hz: FloatArray = np.zeros(0)
    for seed in range(WELCH_N_SEEDS):
        eta = synthesize_elevation(_components(sea, make_rng(seed)), t)
        freq_hz, p = signal.welch(
            eta,
            fs=FS_HZ,
            nperseg=WELCH_NPERSEG,
            noverlap=WELCH_NPERSEG // 2,
            window="hann",
            detrend="constant",
        )
        psd_f = p if psd_f is None else psd_f + p
    assert psd_f is not None
    psd_f = psd_f / WELCH_N_SEEDS

    # scipy.signal.welch returns a one-sided PSD per Hz; convert to per rad/s.
    w = 2.0 * np.pi * freq_hz
    psd_w = psd_f / (2.0 * np.pi)
    analytic = jonswap(np.maximum(w, 1e-9), sea.hs_m, sea.tp_s, sea.gamma)

    edges = np.arange(PSD_BAND_RAD_S[0], PSD_BAND_RAD_S[1] + 1e-9, PSD_BAND_WIDTH_RAD_S)
    devs: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (w >= lo) & (w < hi)
        if not mask.any():
            continue
        devs.append(100.0 * (psd_w[mask].mean() - analytic[mask].mean()) / analytic[mask].mean())
    worst = max(devs, key=abs)
    print(
        f"[inv3] {sea.name}: {len(devs)} bands over {PSD_BAND_RAD_S} rad/s, "
        f"nperseg={WELCH_NPERSEG}, {WELCH_N_SEEDS} seeds -> max |deviation| = {abs(worst):.2f} % "
        f"(signed {worst:+.2f} %)"
    )
    assert abs(worst) < 15.0


# ---------------------------------------------------------------------------
# Invariant 4 -- Rayleigh crest distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crests_are_rayleigh(seed: int, make_rng: Callable[[int], np.random.Generator]) -> None:
    sea = SEA_STATES[2]
    eta = synthesize_elevation(_components(sea, make_rng(seed)), _time_axis(RECORD_S))
    crests = _zero_crossing_crests(eta)
    # Scale fixed at sqrt(m0) from the record variance, never fitted: fitting the scale to
    # the same sample invalidates the KS p-value (the null distribution is no longer the
    # tabulated one) and turns a real test into a rubber stamp.
    scale = float(np.std(eta - eta.mean()))
    result = stats.kstest(crests, "rayleigh", args=(0.0, scale))
    print(
        f"[inv4] seed={seed}: n_crests={crests.size}  sqrt(m0)={scale:.4f} m  "
        f"KS D={result.statistic:.4f}  p={result.pvalue:.4f}"
    )
    assert crests.size > 300
    assert result.pvalue > 0.01


# ---------------------------------------------------------------------------
# Invariant 5 -- anti-periodicity. The most important test in the project.
# ---------------------------------------------------------------------------

#: 299 components over [0.2, 2.5] rad/s gives dw = 2.3/299 and w_min/dw = 26 exactly, so an
#: un-jittered grid is a comb of exact integer multiples of dw and the un-jittered record is
#: *exactly* periodic with period 2*pi/dw = 816.8 s. That exactness is what lets the
#: converse assertions below be sharp (r = 0.995, repeat to 1e-12) instead of approximate.
#: With a band whose w_min/dw is not an integer the un-jittered repeat is a rigid phase
#: rotation of the same waveform rather than an identity -- equally memorisable, but it
#: would make the converse test's threshold arbitrary.
ANTIPERIOD_N_COMPONENTS = 299
ACF_MAX_LAG_S = 1500.0
ACF_BACKGROUND_LAG_S = (200.0, 1500.0)
ACF_HARMONIC_EXCLUSION_S = 20.0

#: Threshold on |r(T_period)| for a jittered record. The spec's 0.10 is inside the sampling
#: noise of the estimator: at large lag the ACF of a 3600 s JONSWAP record fluctuates with
#: sigma ~ sqrt(tau_correlation/T_record) ~ 0.06, so 0.10 is 1.7 sigma and would flake on
#: roughly one seed in ten. Measured |r(T)| over eight seeds spans 0.004-0.093. 0.20 is
#: ~3.3 sigma, still an order of magnitude below the un-jittered value of 0.995, and the
#: peak-versus-background ratio below is the assertion that actually discriminates.
ACF_JITTERED_MAX = 0.20
ACF_PEAK_TO_BACKGROUND_MAX = 1.5
ACF_UNJITTERED_MIN = 0.95


def _acf_diagnostics(eta: FloatArray, period_s: float) -> tuple[float, float, float]:
    max_lag = int(round(ACF_MAX_LAG_S * FS_HZ))
    acf = _unbiased_acf(eta, max_lag)
    lags_s = np.arange(max_lag + 1) / FS_HZ

    at_period = float(acf[int(round(period_s * FS_HZ))])

    background_mask = (lags_s >= ACF_BACKGROUND_LAG_S[0]) & (lags_s <= ACF_BACKGROUND_LAG_S[1])
    for harmonic in range(1, int(ACF_MAX_LAG_S / period_s) + 2):
        background_mask &= np.abs(lags_s - harmonic * period_s) > ACF_HARMONIC_EXCLUSION_S
    background = float(np.percentile(np.abs(acf[background_mask]), 99.9))

    near_mask = np.abs(lags_s - period_s) <= ACF_HARMONIC_EXCLUSION_S
    peak_near = float(np.abs(acf[near_mask]).max())
    return at_period, peak_near, background


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_autocorrelation_spike_at_synthesis_period(
    seed: int, make_rng: Callable[[int], np.random.Generator]
) -> None:
    sea = SEA_STATES[2]
    components = _components(sea, make_rng(seed), n_components=ANTIPERIOD_N_COMPONENTS)
    period_s = synthesis_period_s(components)

    # Guard first: if the synthesis period fell outside the record the test could not fail
    # even if the synthesis were a perfect loop, and it would pass for the wrong reason.
    assert period_s < RECORD_S, (
        f"synthesis period {period_s:.1f} s is not inside the {RECORD_S:.0f} s record; "
        "this test cannot detect periodicity under these settings"
    )

    eta = synthesize_elevation(components, _time_axis(RECORD_S))
    at_period, peak_near, background = _acf_diagnostics(eta, period_s)
    ratio = peak_near / background
    print(
        f"[inv5] seed={seed}: T_period={period_s:.2f} s ({RECORD_S / period_s:.2f} repeats "
        f"inside the record)  r(T)={at_period:+.4f}  peak_near={peak_near:.4f}  "
        f"background_p99.9={background:.4f}  ratio={ratio:.3f}"
    )
    assert abs(at_period) < ACF_JITTERED_MAX
    assert ratio < ACF_PEAK_TO_BACKGROUND_MAX


def test_unjittered_synthesis_does_show_the_spike(
    make_rng: Callable[[int], np.random.Generator],
) -> None:
    """The converse: proof that the anti-periodicity test above is capable of failing."""
    sea = SEA_STATES[2]
    components = _components(sea, make_rng(0), n_components=ANTIPERIOD_N_COMPONENTS, jitter=False)
    period_s = synthesis_period_s(components)
    assert period_s < RECORD_S

    eta = synthesize_elevation(components, _time_axis(RECORD_S))
    at_period, peak_near, background = _acf_diagnostics(eta, period_s)
    print(
        f"[inv5-converse] UN-JITTERED: T_period={period_s:.2f} s  r(T)={at_period:+.4f}  "
        f"background_p99.9={background:.4f}  ratio={peak_near / background:.2f}"
    )
    assert at_period > ACF_UNJITTERED_MIN
    assert peak_near / background > ACF_PEAK_TO_BACKGROUND_MAX


def test_unjittered_record_repeats_exactly_and_jittered_does_not(
    make_rng: Callable[[int], np.random.Generator],
) -> None:
    sea = SEA_STATES[2]
    t = _time_axis(RECORD_S)

    unjittered = _components(sea, make_rng(0), ANTIPERIOD_N_COMPONENTS, jitter=False)
    period_s = synthesis_period_s(unjittered)
    eta = synthesize_elevation(unjittered, t)
    eta_shift = synthesize_elevation(unjittered, t + period_s)
    rel_unjittered = float(np.abs(eta - eta_shift).max() / np.std(eta))

    jittered = _components(sea, make_rng(0), ANTIPERIOD_N_COMPONENTS, jitter=True)
    eta_j = synthesize_elevation(jittered, t)
    eta_j_shift = synthesize_elevation(jittered, t + synthesis_period_s(jittered))
    rel_jittered = float(np.sqrt(np.mean((eta_j - eta_j_shift) ** 2)) / np.std(eta_j))

    print(
        f"[inv5-shift] eta(t) vs eta(t+T): un-jittered max|diff|/std={rel_unjittered:.3e}, "
        f"jittered rms|diff|/std={rel_jittered:.3f}"
    )
    assert rel_unjittered < 1e-9
    assert rel_jittered > 0.5


def test_periodogram_has_no_harmonic_comb(make_rng: Callable[[int], np.random.Generator]) -> None:
    """Frequency-domain twin of the ACF check: no line spectrum on the ``1/T_period`` comb.

    Checking a single line at ``f = 1/T_period`` is vacuous here -- the lowest synthesis
    frequency is ``26*dw``, so nothing ever sits in that bin. What a uniform grid actually
    produces is a *comb*: every component frequency is an integer multiple of ``dw``, so all
    the energy lands on multiples of ``1/T_period``. The record length is set to an exact
    integer number of synthesis periods so those harmonics fall exactly on FFT bins.
    """
    sea = SEA_STATES[2]
    n_periods = 4
    for jitter in (True, False):
        components = _components(sea, make_rng(0), ANTIPERIOD_N_COMPONENTS, jitter=jitter)
        period_s = synthesis_period_s(components)
        eta = synthesize_elevation(components, _time_axis(n_periods * period_s))
        power = np.abs(np.fft.rfft(eta - eta.mean())) ** 2
        comb_fraction = float(power[::n_periods].sum() / power.sum())
        print(
            f"[inv5-comb] jitter={jitter}: fraction of power on multiples of "
            f"1/T_period = {comb_fraction:.4f}"
        )
        if jitter:
            assert comb_fraction < 0.5
        else:
            assert comb_fraction > 0.99


# ---------------------------------------------------------------------------
# Supporting invariants
# ---------------------------------------------------------------------------


def test_jittered_frequencies_lie_inside_their_bins(rng: np.random.Generator) -> None:
    components = _components(SEA_STATES[2], rng)
    edges = np.linspace(W_MIN_RAD_S, W_MAX_RAD_S, N_COMPONENTS + 1)
    lo = edges[:-1]
    offsets = (components.w_rad_s - lo) / components.dw_rad_s
    print(
        f"[support] bin offsets u_i: min={offsets.min():.4f} max={offsets.max():.4f} "
        f"mean={offsets.mean():.4f} (expect U(0,1))"
    )
    assert np.all(components.w_rad_s >= lo)
    assert np.all(components.w_rad_s < lo + components.dw_rad_s)
    assert np.allclose(components.dw_rad_s, (W_MAX_RAD_S - W_MIN_RAD_S) / N_COMPONENTS)


def test_synthesis_period_is_two_pi_over_dw(rng: np.random.Generator) -> None:
    components = _components(SEA_STATES[2], rng)
    expected = 2.0 * np.pi / ((W_MAX_RAD_S - W_MIN_RAD_S) / N_COMPONENTS)
    got = synthesis_period_s(components)
    print(f"[support] synthesis period: {got:.3f} s (expected {expected:.3f} s)")
    assert got == pytest.approx(expected, rel=1e-12)


def test_same_seed_reproduces_bitwise(make_rng: Callable[[int], np.random.Generator]) -> None:
    a = _components(SEA_STATES[2], make_rng(11))
    b = _components(SEA_STATES[2], make_rng(11))
    c = _components(SEA_STATES[2], make_rng(12))
    t = _time_axis(60.0)
    eta_a = synthesize_elevation(a, t)
    eta_b = synthesize_elevation(b, t)
    eta_c = synthesize_elevation(c, t)
    print(
        f"[support] same-seed max|diff|={np.abs(eta_a - eta_b).max():.3e}, "
        f"different-seed rms|diff|={np.sqrt(np.mean((eta_a - eta_c) ** 2)):.4f} m"
    )
    assert np.array_equal(a.w_rad_s, b.w_rad_s)
    assert np.array_equal(a.phase_rad, b.phase_rad)
    assert np.array_equal(eta_a, eta_b)
    assert not np.allclose(eta_a, eta_c)


@pytest.mark.parametrize("sea", SEA_STATES, ids=[s.name for s in SEA_STATES])
def test_discrete_variance_matches_band_limited_m0(sea: SeaState, rng: np.random.Generator) -> None:
    """``sum(A**2)/2`` must reproduce ``m0`` integrated over the *synthesis band*.

    The comparison is against the band-limited moment, not the full-band one: truncating
    JONSWAP at [0.2, 2.5] rad/s discards 0.1-1.1 percent of the variance depending on sea
    state, and charging the discretisation for that truncation would confuse two different
    approximations.
    """
    components = _components(sea, rng)
    discrete = float(np.sum(components.amplitude_m**2) / 2.0)
    w_band = np.linspace(W_MIN_RAD_S, W_MAX_RAD_S, 200_001)
    m0_band = spectral_moment(w_band, jonswap(w_band, sea.hs_m, sea.tp_s, sea.gamma), 0)
    err_pct = 100.0 * (discrete - m0_band) / m0_band
    print(
        f"[support] {sea.name}: sum(A^2)/2={discrete:.5f} m^2  m0(band)={m0_band:.5f} m^2  "
        f"error={err_pct:+.3f} %"
    )
    assert abs(err_pct) < 2.0


def test_sim_package_imports_no_torch() -> None:
    """Architecture rule from CLAUDE.md: ``src/dmf/sim/`` is pure NumPy.

    Enforced by AST-parsing every module rather than by convention, because a stray
    ``import torch`` inside a helper would otherwise only surface as a mysterious 2 GB
    import cost in the corpus generator.
    """
    sim_dir = Path(dmf.sim.__file__).parent
    modules = sorted(sim_dir.glob("*.py"))
    assert modules, f"no modules found under {sim_dir}"
    offenders: list[str] = []
    for module_path in modules:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "torch" or name.startswith("torch.") for name in names):
                offenders.append(f"{module_path.name}:{node.lineno}")
    print(f"[support] scanned {len(modules)} modules under {sim_dir} for torch imports")
    assert not offenders, f"torch imported under dmf.sim at {offenders}"
