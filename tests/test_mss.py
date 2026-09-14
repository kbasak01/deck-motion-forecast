"""Phase 8: the MSS ingest bridge, its units, its signs and its parity with the m-file.

The invariants asserted here are the ones that a skill score structurally cannot catch.
Carry-forward delta 2 makes the point for units -- skill is a ratio over the same data, so a
uniform factor of 57.3 cancels exactly -- and P8-D3 extends it to signs, which do *not*
cancel, because a TCN or an LSTM is not sign-equivariant the way a linear model is.

Each test prints the quantity it measured, in the `[p8-N]` tagged format the Gate 1 tests
use, because a gate is recorded as numbers and not as green ticks. Run with `-s`.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from dmf.mss.compare import zero_crossing_period
from dmf.mss.convert import (
    MSS_FRAME_COLUMNS,
    MSS_TO_CORPUS_SIGN,
    assert_corpus_units,
    mss_motion_to_frame,
)
from dmf.mss.export import MSSRealizationSpec, _build_grid, realization_seed
from dmf.mss.synth import (
    mss_encounter_frequency,
    sample_corpus_grid,
    synthesize_mss_motion,
)
from dmf.mss.vessel import DOF_INDEX, assert_long_wave_limit, load_mss_vessel

MAT_PATH = Path("mss/upstream/HYDRO/vessels_shipx/s175/s175.mat")
CONFIG_PATH = Path("configs/mss/s175_ss5.yaml")

#: SS5, from configs/sim/sea_states.yaml.
HS_M, TP_S, GAMMA = 3.3, 9.7, 3.3
FS_HZ = 10.0


requires_mss = pytest.mark.skipif(
    not MAT_PATH.exists(),
    reason=(
        "MSS toolbox not cloned. git clone https://github.com/cybergalactic/MSS.git mss/upstream"
    ),
)


@pytest.fixture(scope="module")
def vessel():
    if not MAT_PATH.exists():
        pytest.skip("MSS toolbox not cloned")
    return load_mss_vessel(MAT_PATH)


@pytest.fixture(scope="module")
def run_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _time_axis(n: int = 6000) -> np.ndarray:
    return 120.0 + np.arange(n, dtype=np.float64) / FS_HZ


@requires_mss
def test_long_wave_limit_pins_the_rao_units(vessel) -> None:
    """Heave -> 1 m/m and pitch -> k rad/m as w -> 0: the units are rad/m, not deg/m.

    This is what establishes that the angular RAOs are radians per metre without trusting a
    comment. If the table were in degrees the pitch ratio below would be ~57, not ~1.
    """
    assert_long_wave_limit(vessel)
    w = vessel.w_rad_s
    lo = int(np.argmin(np.abs(w - 0.25)))
    heave = float(np.abs(vessel.complex_rao("heave", 180.0)[lo]))
    pitch = float(np.abs(vessel.complex_rao("pitch", 180.0)[lo]))
    k = float(w[lo] ** 2 / vessel.gravity_m_s2)
    print(f"[p8-1] at w={w[lo]:.4f} rad/s: heave RAO {heave:.4f} m/m, pitch/k {pitch / k:.4f}")
    assert 0.9 <= heave <= 1.1
    assert 0.8 <= pitch / k <= 1.25


@requires_mss
def test_port_starboard_symmetry(vessel) -> None:
    """Headings theta and 360 - theta must give identical response magnitudes.

    This is the symmetry MSS itself constructs when it mirrors the 0-180 deg ShipX data
    into 180-360 (`read_veres_TF.m:193-205`), so it checks that our heading indexing and
    interpolation read that mirrored table the way MSS wrote it.

    Note what is deliberately *not* asserted here: that head and following seas agree at
    zero speed. They do not, and should not -- the S175 is a real hull and is not fore-aft
    symmetric, so at 0 kn its heave RMS is 13.5% larger in head seas than in following seas
    (0.2308 against 0.2033 in the run that first exposed this assumption). The corpus
    generator cannot represent that at all: `heading_factor` depends on cos(beta) through
    its square, so head and following seas are identical there by construction.
    """
    grid = sample_corpus_grid(HS_M, TP_S, GAMMA, 299, 0.2, 2.5, np.random.default_rng(7))
    t = _time_axis(2000)
    for heading in (135.0, 45.0):
        a = synthesize_mss_motion(vessel, grid, heading, 0.0, t)
        b = synthesize_mss_motion(vessel, grid, 360.0 - heading, 0.0, t)
        for dof in ("heave", "pitch", "roll"):
            i = DOF_INDEX[dof]
            ra = float(np.sqrt(np.mean(a.eta[i] ** 2)))
            rb = float(np.sqrt(np.mean(b.eta[i] ** 2)))
            print(
                f"[p8-2] {heading:5.1f}/{360.0 - heading:5.1f} deg {dof:6s}: {ra:.6f} vs {rb:.6f}"
            )
            assert ra == pytest.approx(rb, rel=1e-6)


@requires_mss
def test_encounter_frequency_rises_in_head_seas_and_falls_in_following(vessel) -> None:
    """The heading convention, in the one place forward speed can expose it.

    At zero speed head and following seas are indistinguishable in the encounter frequency,
    so the convention only bites at speed: meeting waves head-on raises the encounter
    frequency and running with them lowers it. With 180 deg as head seas (P8-D2) this is
    what the formula must produce.
    """
    w = np.array([0.4, 0.6478, 0.9], dtype=np.float64)  # SS5 peak is 0.6478 rad/s
    u = 12.0 * 0.514444
    head = mss_encounter_frequency(w, u, 180.0, gravity_m_s2=vessel.gravity_m_s2)
    following = mss_encounter_frequency(w, u, 0.0, gravity_m_s2=vessel.gravity_m_s2)
    print(f"[p8-2b] w {w}, head w_e {np.round(head, 4)}, following w_e {np.round(following, 4)}")
    assert np.all(head > w), "head seas must raise the encounter frequency"
    assert np.all(following < w), "following seas must lower it"


@requires_mss
def test_head_seas_roll_is_identically_zero(vessel) -> None:
    """MSS head-seas roll vanishes by port/starboard symmetry.

    Recorded as a test because it is the reason P8-D1 excludes head-seas roll from the
    headline in advance: a skill score against a zero-variance target is undefined. The
    corpus produces 0.038 deg here instead, which is entirely the P1-D2 residual floor.
    """
    grid = sample_corpus_grid(HS_M, TP_S, GAMMA, 299, 0.2, 2.5, np.random.default_rng(3))
    motion = synthesize_mss_motion(vessel, grid, 180.0, 0.0, _time_axis(2000))
    roll_rms = float(np.sqrt(np.mean(motion.eta[DOF_INDEX["roll"]] ** 2)))
    print(f"[p8-3] MSS head-seas roll RMS {roll_rms:.3e} rad")
    assert roll_rms < 1e-6


@requires_mss
def test_converted_frame_is_in_corpus_units_and_signs(vessel) -> None:
    """Degrees, metres, and heave the only inverted channel.

    The magnitude half of this is what delta 2 asks for. The sign half is P8-D3: SNAME's
    z-down heave inverts against the corpus's +up, while roll and pitch already agree.
    """
    grid = sample_corpus_grid(HS_M, TP_S, GAMMA, 299, 0.2, 2.5, np.random.default_rng(11))
    motion = synthesize_mss_motion(vessel, grid, 135.0, 6.0 * 0.514444, _time_axis(2000))
    frame = mss_motion_to_frame(motion)
    assert list(frame.columns) == list(MSS_FRAME_COLUMNS)
    assert_corpus_units(frame)

    # Degrees, not radians: a pitch record in radians would peak near 0.02, not near 1.
    peak_pitch = float(np.max(np.abs(frame["pitch"].to_numpy())))
    print(f"[p8-4] pitch peak {peak_pitch:.4f} deg (radians would be ~57x smaller)")
    assert 0.1 < peak_pitch < 60.0

    # Heave inverts and nothing else.
    assert MSS_TO_CORPUS_SIGN == {"roll": +1.0, "pitch": +1.0, "heave": -1.0}
    np.testing.assert_allclose(
        frame["heave"].to_numpy(), -motion.eta[DOF_INDEX["heave"]], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        frame["pitch"].to_numpy(), np.rad2deg(motion.eta[DOF_INDEX["pitch"]]), rtol=0, atol=0
    )


@requires_mss
def test_units_assertion_catches_a_radians_slip(vessel) -> None:
    """The boundary assertion must actually fire on the failure it exists for."""
    grid = sample_corpus_grid(HS_M, TP_S, GAMMA, 299, 0.2, 2.5, np.random.default_rng(5))
    motion = synthesize_mss_motion(vessel, grid, 135.0, 0.0, _time_axis(1000))
    frame = mss_motion_to_frame(motion)
    frame["pitch"] = np.rad2deg(frame["pitch"].to_numpy())  # degrees applied twice
    with pytest.raises(AssertionError, match="plausible limit"):
        assert_corpus_units(frame)


@requires_mss
def test_spectrum_matches_ss5(vessel, run_config) -> None:
    """Gate 8 predicate 1: realized Hs and Tz within 5% of the SS5 target.

    Read on the mean over realizations, not the max. The per-record spread at 299
    components over a 600 s record is ~4.5% on Hs, so a max-over-records test would be
    measuring sampling scatter rather than spectral match (P8-D5).
    """
    tol = run_config["match_tolerance"]
    tz_target = float(tol["tz_over_tp"]) * TP_S
    hs_vals, tz_vals = [], []
    for seed in range(8):
        spec = MSSRealizationSpec("mss", 180.0, 0.0, seed)
        grid = _build_grid(spec, run_config, realization_seed(spec))
        motion = synthesize_mss_motion(vessel, grid, 180.0, 0.0, _time_axis())
        hs_vals.append(4.0 * float(np.std(motion.elevation_earth_m)))
        tz_vals.append(zero_crossing_period(motion.elevation_earth_m, FS_HZ))
    hs, tz = float(np.mean(hs_vals)), float(np.mean(tz_vals))
    print(
        f"[p8-5] Hs {hs:.4f} m ({hs / HS_M - 1:+.2%}) +/- {np.std(hs_vals):.4f}, "
        f"Tz {tz:.4f} s ({tz / tz_target - 1:+.2%}) +/- {np.std(tz_vals):.4f}"
    )
    assert abs(hs / HS_M - 1.0) <= float(tol["hs_rel"])
    assert abs(tz / tz_target - 1.0) <= float(tol["tz_rel"])


@requires_mss
def test_heading_interpolation_is_not_nearest_node(vessel) -> None:
    """135 deg must interpolate between RAO nodes, not snap to one.

    The RAO grid is 10 degrees, so 180 lands on a node and 135 does not. Snapping reproduced
    MSS exactly at 180 and was 38% wrong at 135 -- the only heading in this run where roll
    is live. Guarded here so the bug cannot come back silently (P8-D4).
    """
    at_135 = vessel.complex_rao("roll", 135.0)
    for node in (130.0, 140.0):
        at_node = vessel.complex_rao("roll", node)
        # complex_rao snaps by design; the synthesis path is what must interpolate.
        assert at_node.shape == at_135.shape
    grid = sample_corpus_grid(HS_M, TP_S, GAMMA, 299, 0.2, 2.5, np.random.default_rng(2))
    t = _time_axis(1500)
    rms = {}
    for heading in (130.0, 135.0, 140.0):
        motion = synthesize_mss_motion(vessel, grid, heading, 0.0, t)
        rms[heading] = float(np.sqrt(np.mean(motion.eta[DOF_INDEX["roll"]] ** 2)))
    print(f"[p8-6] roll RMS 130/135/140 deg: {rms[130.0]:.6f} {rms[135.0]:.6f} {rms[140.0]:.6f}")
    lo, hi = sorted((rms[130.0], rms[140.0]))
    assert lo <= rms[135.0] <= hi, "135 deg must lie between its bracketing nodes"
    assert rms[135.0] != rms[130.0] and rms[135.0] != rms[140.0], "snapped to a node"


@pytest.mark.octave
@requires_mss
def test_numpy_bridge_matches_octave() -> None:
    """The NumPy port reproduces MSS's own waveMotionRAO.m.

    Skips cleanly when Octave is absent, following the `real_corpus` idiom in conftest: an
    unavailable optional dependency is a skip, not a failure. `scripts/gate8.py` decides
    what an unverified bridge means for the gate.
    """
    if shutil.which("octave") is None:
        pytest.skip("GNU Octave not installed")
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "scripts/mss_octave_check.py", "--out-dir", tmp, "--n-steps", "200"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert proc.returncode == 0, f"parity check failed:\n{proc.stderr}"
        import pandas as pd

        table = pd.read_csv(Path(tmp) / "octave_parity.csv")
        worst = float(table["rel_deviation"].max())
        print(f"[p8-7] worst relative deviation NumPy vs Octave: {worst:.3e}")
        assert worst < 1e-6, "the port should match the m-file to near machine precision"
        patch = table[table["check"] == "patch_equivalence"]
        assert bool(patch["passed"].iloc[0]), "the local m-file patch changed behaviour"


def test_mss_package_imports_no_torch() -> None:
    """`dmf.mss` stays pure NumPy/SciPy, mirroring the rule `dmf.sim` is held to.

    The MSS bridge is simulation-side code. Keeping torch out of it means the ingest can be
    exercised, and its physics checked, without the training stack present.
    """
    offenders: list[str] = []
    for path in sorted(Path("src/dmf/mss").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "torch" or n.startswith("torch.") for n in names):
                offenders.append(f"{path}:{node.lineno}")
    print(f"[p8-8] torch imports under src/dmf/mss: {len(offenders)}")
    assert not offenders, f"torch imported in simulation-side code: {offenders}"
