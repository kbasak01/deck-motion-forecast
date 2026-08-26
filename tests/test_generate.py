"""Corpus generation, seeding, and observation-model invariants.

These tests protect three properties that nothing downstream can recover from if they are
wrong:

1. **Grid completeness.** The corpus is exactly the grid the plan specifies -- 2304
   realizations, 40 seeds per cell for the frigate and 8 for the held-out S175 -- and it
   *includes* the two non-monotonic encounter-frequency cells rather than silently
   dropping them (``docs/protocol.md`` P1-D1).
2. **Seed determinism, independent of worker count.** A realization is a pure function of
   its grid coordinates. If the seed came from a shared counter, a parent generator, or
   task ordering, the corpus would depend on ``n_workers`` and "reproducible" would be a
   claim rather than a fact.
3. **The spin-up is really discarded**, and the ``imu`` observation model corrupts what it
   is supposed to corrupt and nothing else.

Units follow the package convention: seconds, metres, degrees for angles, knots for speed,
hertz only for the sampling rate and filter cutoffs.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from conftest import FS_HZ
from dmf.config import SimConfig, load_sim
from dmf.sim.generate import (
    CORPUS_COLUMNS,
    CORPUS_DTYPES,
    MANIFEST_NAME,
    RealizationSpec,
    generate_corpus,
    realization_grid,
    realization_path,
    realization_seed_sequence,
    simulate_realization,
)
from dmf.sim.imu import (
    HEAVE_HIGHPASS_HZ,
    IDEAL_COLUMNS,
    IMU_COLUMNS,
    apply_observation_model,
    bias_random_walk,
    heave_from_vertical_acc,
    highpass_biquad,
)

CORPUS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sim" / "corpus.yaml"

#: Expected corpus size: 4 sea states x 4 headings x 3 speeds x (40 + 8) seeds.
EXPECTED_TOTAL = 2304
EXPECTED_FRIGATE = 1920
EXPECTED_S175 = 384

#: (heading_deg, speed_kn) cells where ``dw_e/dw`` changes sign inside [0.2, 2.5] rad/s.
NON_MONOTONIC_CELLS: frozenset[tuple[float, float]] = frozenset({(45.0, 6.0), (45.0, 12.0)})


@pytest.fixture(scope="module")
def sim_cfg() -> SimConfig:
    """Return the corpus configuration actually used by ``make data``."""
    return load_sim(CORPUS_CONFIG)


@pytest.fixture(scope="module")
def spec() -> RealizationSpec:
    """Return a representative realization: SS5 beam seas, frigate, at rest."""
    return RealizationSpec(
        seed=7, sea_state="SS5", heading_deg=90.0, speed_kn=0.0, vessel="frigate"
    )


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_has_the_specified_size(sim_cfg: SimConfig) -> None:
    grid = realization_grid(sim_cfg)
    counts = Counter(s.vessel for s in grid)
    assert len(grid) == EXPECTED_TOTAL
    assert counts["frigate"] == EXPECTED_FRIGATE
    assert counts["s175"] == EXPECTED_S175
    assert len({(s.vessel, s.sea_state, s.heading_deg, s.speed_kn, s.seed) for s in grid}) == len(
        grid
    )


def test_grid_seed_ordinals_are_contiguous_per_cell(sim_cfg: SimConfig) -> None:
    """Seed ordinals must form a dense range per cell.

    The splits slice on that ordinal -- the ``id`` regime uses seeds 0-31 for training and
    32-39 for test -- so it is an index, not an arbitrary label.
    """
    per_cell: dict[tuple[str, str, float, float], list[int]] = {}
    for s in realization_grid(sim_cfg):
        per_cell.setdefault((s.vessel, s.sea_state, s.heading_deg, s.speed_kn), []).append(s.seed)
    for (vessel, *_), seeds in per_cell.items():
        expected = sim_cfg.seeds_for(vessel)
        assert sorted(seeds) == list(range(expected))


def test_grid_keeps_the_non_monotonic_encounter_cells(sim_cfg: SimConfig) -> None:
    """P1-D1: the awkward cells are kept and documented, never filtered out.

    A grid that quietly dropped them would leave the heading axis unbalanced across speeds
    and would make the ``unseen_heading`` regime a different experiment than it claims.
    """
    present = {(s.heading_deg, s.speed_kn) for s in realization_grid(sim_cfg)}
    assert present >= NON_MONOTONIC_CELLS
    for cell in NON_MONOTONIC_CELLS:
        for vessel in sim_cfg.vessels:
            n = sum(
                1
                for s in realization_grid(sim_cfg)
                if (s.heading_deg, s.speed_kn) == cell and s.vessel == vessel
            )
            assert n == len(sim_cfg.sea_states) * sim_cfg.seeds_for(vessel)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_seed_sequence_depends_on_every_coordinate() -> None:
    base = RealizationSpec(
        seed=3, sea_state="SS4", heading_deg=135.0, speed_kn=6.0, vessel="frigate"
    )
    variants = [
        RealizationSpec(4, "SS4", 135.0, 6.0, "frigate"),
        RealizationSpec(3, "SS5", 135.0, 6.0, "frigate"),
        RealizationSpec(3, "SS4", 90.0, 6.0, "frigate"),
        RealizationSpec(3, "SS4", 135.0, 12.0, "frigate"),
        RealizationSpec(3, "SS4", 135.0, 6.0, "s175"),
    ]
    entropies = {tuple(realization_seed_sequence(v).entropy) for v in [base, *variants]}
    assert len(entropies) == len(variants) + 1
    assert tuple(realization_seed_sequence(base).entropy) == tuple(
        realization_seed_sequence(RealizationSpec(3, "SS4", 135.0, 6.0, "frigate")).entropy
    )


def test_realization_is_bitwise_reproducible(spec: RealizationSpec, sim_cfg: SimConfig) -> None:
    """The same spec must give the same numbers on a repeat call, in isolation."""
    first = simulate_realization(spec, sim_cfg)
    second = simulate_realization(spec, sim_cfg)
    pd.testing.assert_frame_equal(first, second)
    for column in (*IDEAL_COLUMNS, *IMU_COLUMNS):
        assert np.array_equal(first[column].to_numpy(), second[column].to_numpy())


def test_different_seeds_give_different_records(spec: RealizationSpec, sim_cfg: SimConfig) -> None:
    other = RealizationSpec(
        seed=spec.seed + 1,
        sea_state=spec.sea_state,
        heading_deg=spec.heading_deg,
        speed_kn=spec.speed_kn,
        vessel=spec.vessel,
    )
    a = simulate_realization(spec, sim_cfg)["roll"].to_numpy()
    b = simulate_realization(other, sim_cfg)["roll"].to_numpy()
    assert not np.array_equal(a, b)
    # Independent realizations of the same cell. The bound is loose on purpose: roll is
    # narrowband (zeta = 0.06), so a 600 s record holds only ~50 roll cycles and the
    # sample correlation of two independent records has a standard deviation of order
    # 0.2. Anything at or near 1.0 would mean the two seeds share a wave field; 0.5 is
    # comfortably clear of the sampling noise without being a coin flip.
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.5


# ---------------------------------------------------------------------------
# Record shape and spin-up
# ---------------------------------------------------------------------------


def test_record_shape_dtypes_and_time_axis(spec: RealizationSpec, sim_cfg: SimConfig) -> None:
    df = simulate_realization(spec, sim_cfg)
    assert len(df) == int(round(sim_cfg.duration_s * sim_cfg.fs_hz)) == 6000
    assert tuple(df.columns) == CORPUS_COLUMNS
    assert {name: str(dtype) for name, dtype in df.dtypes.items()} == CORPUS_DTYPES
    assert not bool(df.isna().to_numpy().any())
    t = df["t"].to_numpy()
    assert np.allclose(np.diff(t), 1.0 / sim_cfg.fs_hz)
    assert t[-1] - t[0] == pytest.approx(sim_cfg.duration_s - 1.0 / sim_cfg.fs_hz)
    assert df["seed"].unique().tolist() == [spec.seed]
    assert df["ss"].unique().tolist() == [spec.sea_state]
    assert df["vessel"].unique().tolist() == [spec.vessel]


def test_spinup_is_discarded(spec: RealizationSpec, sim_cfg: SimConfig) -> None:
    """The stored record must start at ``spinup_s``, not at the synthesis origin.

    At ``t = 0`` every wave component is at ``cos(phi_i)``, a phase alignment shared by
    every realization in the corpus. Retaining it would give a forecaster a fixed, learnable
    landmark at the start of every record.
    """
    df = simulate_realization(spec, sim_cfg)
    assert float(df["t"].iloc[0]) == pytest.approx(sim_cfg.spinup_s)

    # The same synthesis with nothing discarded: same seed, same component set, same
    # 7200-sample time axis, so the retained window must be its tail and not its head.
    undiscarded = SimConfig(
        **{
            **sim_cfg.__dict__,
            "duration_s": sim_cfg.duration_s + sim_cfg.spinup_s,
            "spinup_s": 0.0,
        }
    )
    full = simulate_realization(spec, undiscarded)
    n_spin = int(round(sim_cfg.spinup_s * sim_cfg.fs_hz))
    assert float(full["t"].iloc[0]) == 0.0
    for column in ("roll", "heave", "roll_imu", "heave_imu"):
        assert np.array_equal(df[column].to_numpy(), full[column].to_numpy()[n_spin:])
        assert not np.allclose(df[column].to_numpy(), full[column].to_numpy()[:-n_spin])


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


def test_ideal_mode_is_the_identity(rng: np.random.Generator) -> None:
    df = _toy_frame()
    out = apply_observation_model(df, "ideal", FS_HZ, rng)
    pd.testing.assert_frame_equal(out, df)


def test_imu_mode_changes_attitude_and_heave_only(rng: np.random.Generator) -> None:
    df = _toy_frame()
    out = apply_observation_model(df, "imu", FS_HZ, rng)
    assert np.array_equal(out["t"].to_numpy(), df["t"].to_numpy())
    assert np.array_equal(out["heave_acc"].to_numpy(), df["heave_acc"].to_numpy())
    for channel in ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate"):
        assert not np.array_equal(out[channel].to_numpy(), df[channel].to_numpy())
    # Attitude corruption is small: white noise plus a slow bias, not a rescaling.
    for channel in ("roll", "pitch"):
        residual = out[channel].to_numpy() - df[channel].to_numpy()
        assert float(np.std(residual)) < 0.1


def test_imu_heave_error_grows_towards_low_frequency(
    spec: RealizationSpec, sim_cfg: SimConfig
) -> None:
    """The reconstruction error must be a low-frequency phenomenon.

    Measured on a real broadband realization rather than a toy sinusoid, and band by band
    as a *relative* error, because true heave carries almost no energy below 0.06 Hz and an
    absolute error spectrum would therefore hide the distortion entirely.

    The error is not small in the wave band either -- the two causal high-pass stages lead
    true heave by about 48 degrees at 0.1 Hz, which alone accounts for a relative error
    power of ``abs(1 - exp(1j*48 deg))**2 = 0.66``. What the ``imu`` mode claims is that the
    error *grows monotonically towards low frequency*, and that is what is asserted.
    """
    df = simulate_realization(spec, sim_cfg)
    true = df["heave"].to_numpy().astype(np.float64)
    err = df["heave_imu"].to_numpy().astype(np.float64) - true
    f, p_true = signal.welch(true, fs=sim_cfg.fs_hz, nperseg=2048)
    _, p_err = signal.welch(err, fs=sim_cfg.fs_hz, nperseg=2048)

    bands = [(0.03, 0.06), (0.06, 0.09), (0.09, 0.15), (0.15, 0.30)]
    ratios = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        ratios.append(float(np.sum(p_err[mask]) / np.sum(p_true[mask])))
    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False)), ratios
    assert ratios[0] > 1.0  # below the cutoff the reconstruction is worse than useless
    assert ratios[-1] < 0.5  # by 0.15 Hz it mostly tracks


def test_highpass_matches_scipy_butterworth() -> None:
    """Pin the hand-rolled biquad to the reference implementation it replaces.

    It exists only to keep ``dmf.sim`` dependency-light, so it must agree with
    ``scipy.signal`` to machine precision or it is a liability rather than a saving.
    """
    b, a = signal.butter(2, HEAVE_HIGHPASS_HZ / (FS_HZ / 2.0), btype="highpass")
    x = np.random.default_rng(0).normal(size=4096)
    assert np.allclose(highpass_biquad(x, HEAVE_HIGHPASS_HZ, FS_HZ), signal.lfilter(b, a, x))


def test_highpass_is_causal() -> None:
    """A causal filter cannot respond before its input changes -- ``filtfilt`` would."""
    x = np.zeros(512)
    x[256:] = 1.0
    y = highpass_biquad(x, HEAVE_HIGHPASS_HZ, FS_HZ)
    assert np.allclose(y[:256], 0.0)
    assert abs(float(y[256])) > 0.5


def test_heave_reconstruction_recovers_amplitude_in_the_wave_band() -> None:
    """Check the reconstruction gain and lead at the SS5 spectral peak.

    The magnitude is right; the distortion the ``imu`` mode exists to expose is the phase
    lead, measured here at roughly a second.
    """
    t = np.arange(0, 900, 1.0 / FS_HZ)
    period_s = 9.7
    w = 2.0 * np.pi / period_s
    disp = np.cos(w * t)
    rec = heave_from_vertical_acc(-(w**2) * disp, FS_HZ, HEAVE_HIGHPASS_HZ)[2000:]
    assert float(np.std(rec)) == pytest.approx(float(np.std(disp[2000:])), rel=0.05)
    lags = np.arange(-40, 41)
    xcorr = [float(np.corrcoef(np.roll(rec, int(lag)), disp[2000:])[0, 1]) for lag in lags]
    best_lag_s = float(lags[int(np.argmax(xcorr))]) / FS_HZ
    assert 0.5 < best_lag_s < 2.5


def test_bias_random_walk_scales_with_sqrt_time(rng: np.random.Generator) -> None:
    sigma = 0.01
    n = 6000
    walks = np.stack([bias_random_walk(n, sigma, FS_HZ, rng) for _ in range(200)])
    assert np.allclose(walks[:, 0], 0.0)
    for idx in (1000, 3000, 6000 - 1):
        elapsed_s = idx / FS_HZ
        assert float(np.std(walks[:, idx])) == pytest.approx(sigma * np.sqrt(elapsed_s), rel=0.2)


# ---------------------------------------------------------------------------
# Corpus writing
# ---------------------------------------------------------------------------


def _small_config(sim_cfg: SimConfig) -> SimConfig:
    """Return a two-cell, two-seed, 30 s cut-down of the corpus config."""
    return SimConfig(
        **{
            **sim_cfg.__dict__,
            "sea_states": sim_cfg.sea_states[2:3],
            "headings_deg": (90.0, 45.0),
            "speeds_kn": (12.0,),
            "seeds_per_cell": 2,
            "seeds_per_cell_by_vessel": (("s175", 1),),
            "duration_s": 30.0,
            "spinup_s": 10.0,
        }
    )


def test_generate_corpus_is_independent_of_worker_count(sim_cfg: SimConfig, tmp_path: Path) -> None:
    """The property that makes the corpus reproducible rather than merely deterministic.

    Byte equality, not approximate equality: the Parquet payload is a function of the data
    alone, so if scheduling or chunking leaked into the seeds this comparison fails loudly.
    """
    cfg = _small_config(sim_cfg)
    serial = generate_corpus(cfg, tmp_path / "serial", n_workers=1)
    parallel = generate_corpus(cfg, tmp_path / "parallel", n_workers=4)

    files = sorted(p.relative_to(serial) for p in serial.rglob("*.parquet"))
    assert files == sorted(p.relative_to(parallel) for p in parallel.rglob("*.parquet"))
    assert len(files) == len(realization_grid(cfg)) + 1  # realizations plus the manifest
    for rel in files:
        assert (serial / rel).read_bytes() == (parallel / rel).read_bytes(), rel


def test_manifest_indexes_every_realization(sim_cfg: SimConfig, tmp_path: Path) -> None:
    cfg = _small_config(sim_cfg)
    root = generate_corpus(cfg, tmp_path / "corpus", n_workers=2)
    manifest = pd.read_parquet(root / MANIFEST_NAME)
    grid = realization_grid(cfg)

    assert len(manifest) == len(grid)
    assert set(manifest.columns) >= {
        "vessel",
        "ss",
        "heading",
        "speed",
        "seed",
        "n_rows",
        "path",
        "encounter_monotonic",
    }
    assert manifest["n_rows"].unique().tolist() == [int(round(cfg.duration_s * cfg.fs_hz))]
    for _, row in manifest.iterrows():
        target = root / str(row["path"])
        assert target.exists()
        assert len(pd.read_parquet(target)) == int(row["n_rows"])
    # The 45 deg / 12 kn cell is the non-monotonic one and must be labelled as such.
    labels = dict(
        zip(manifest["heading"].tolist(), manifest["encounter_monotonic"].tolist(), strict=False)
    )
    assert labels[90.0] is np.True_ or labels[90.0] is True
    assert labels[45.0] is np.False_ or labels[45.0] is False


def test_realization_path_is_unique_per_spec(sim_cfg: SimConfig) -> None:
    grid = realization_grid(sim_cfg)
    assert len({realization_path(s) for s in grid}) == len(grid)


def _toy_frame(n: int = 6000) -> pd.DataFrame:
    """Return a clean single-frequency motion frame for observation-model tests.

    Units match the corpus schema: seconds, degrees, degrees per second, metres, metres per
    second, metres per second squared.
    """
    t = np.arange(n, dtype=np.float64) / FS_HZ
    w = 2.0 * np.pi / 9.7
    return pd.DataFrame(
        {
            "t": t,
            "roll": 3.0 * np.cos(w * t),
            "pitch": 0.8 * np.cos(w * t + 0.4),
            "heave": 0.7 * np.cos(w * t + 1.1),
            "roll_rate": -3.0 * w * np.sin(w * t),
            "pitch_rate": -0.8 * w * np.sin(w * t + 0.4),
            "heave_rate": -0.7 * w * np.sin(w * t + 1.1),
            "heave_acc": -0.7 * w**2 * np.cos(w * t + 1.1),
        }
    )
