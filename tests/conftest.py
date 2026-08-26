"""Shared pytest fixtures.

Empty in Phase 0. Phase 1 adds the seeded ``rng`` fixture and the small synthetic corpus
fixture that the physics and split tests build on, so that no test constructs its own
generator and reproducibility is a property of the fixture rather than of each test.

Every random draw in the test suite comes from :func:`rng` or :func:`make_rng`. A test that
calls ``np.random.default_rng`` directly is a reproducibility hole and should be rejected in
review.

Unit conventions in the fixtures below follow the package convention: seconds, metres,
radians per second for frequencies, degrees for headings, knots only where the name says so.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dmf.config import DataConfig, SeaState, SimConfig, load_data, load_sim
from dmf.data.splits import load_manifest
from dmf.sim.generate import generate_corpus
from dmf.sim.vessel import DofParams, Vessel

#: Base seed for the whole test suite. Individual streams are ``BASE_SEED + offset`` so that
#: two fixtures never silently share a stream.
BASE_SEED = 20260826

#: The four corpus sea states (DNV/Bretschneider-style representative values).
SEA_STATES: tuple[SeaState, ...] = (
    SeaState(name="SS3", hs_m=1.0, tp_s=7.5, gamma=3.3),
    SeaState(name="SS4", hs_m=1.9, tp_s=8.8, gamma=3.3),
    SeaState(name="SS5", hs_m=3.3, tp_s=9.7, gamma=3.3),
    SeaState(name="SS6", hs_m=5.0, tp_s=12.4, gamma=3.3),
)

#: Corpus heading grid, degrees. 180 head seas, 90 beam seas, 0 following seas.
HEADINGS_DEG: tuple[float, ...] = (180.0, 135.0, 90.0, 45.0)

#: Corpus speed grid, knots.
SPEEDS_KN: tuple[float, ...] = (0.0, 6.0, 12.0)

#: Synthesis band, radians per second.
W_MIN_RAD_S = 0.2
W_MAX_RAD_S = 2.5

#: Stored sampling rate, hertz. The only Hz quantity in the simulator.
FS_HZ = 10.0


@pytest.fixture
def rng() -> np.random.Generator:
    """Return the suite's default seeded generator.

    Returns:
        A ``numpy.random.Generator`` seeded from :data:`BASE_SEED`. Function-scoped, so
        each test gets a fresh stream at the same starting state and test ordering cannot
        change any result.
    """
    return np.random.default_rng(BASE_SEED)


@pytest.fixture
def make_rng() -> Callable[[int], np.random.Generator]:
    """Return a factory for independent seeded generators.

    Used by tests that need several independent realizations (multi-seed PSD averaging,
    the three-seed anti-periodicity check). Offsets are hashed into distinct streams via
    ``SeedSequence`` so that neighbouring offsets are not correlated.

    Returns:
        A callable mapping an integer offset to a ``numpy.random.Generator``.
    """

    def _make(offset: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([BASE_SEED, offset]))

    return _make


@pytest.fixture
def frigate() -> Vessel:
    """Return the primary hull used throughout the physics tests.

    Representative of a 120-130 m frigate: a long, lightly damped roll mode near 12 s and
    heavily damped pitch and heave near 7 and 8.5 s. Constructed directly rather than via
    :func:`dmf.sim.vessel.load_vessel` so that the physics tests fail on physics, not on
    config plumbing. ``configs/sim/vessels/frigate.yaml`` mirrors these values exactly.

    ``roll.zeta`` is 0.06, not the 0.08 first tried. At 0.08 the SS5 beam-seas roll
    response spectrum is bimodal: the resonance peak at 0.541 rad/s stands only 9 percent
    above a second, wave-driven peak at 0.622 rad/s (SS5 has ``wp = 0.648 rad/s``, well
    above ``wn_roll = 0.524``), so the Welch argmax flips between the two under estimator
    noise and Gate 1 criterion 6 measures whichever one won. That criterion is a statement
    that beam-seas roll is resonance-dominated, and at 0.08 this hull is not. At 0.06 the
    resonance peak stands 69 percent above the secondary one and the measured peak is
    stable at +0.6 to +2.4 percent of ``wn_roll``. 0.06 is mid-band for an unstabilised
    frigate (the plan's range is 0.05-0.12) and moves SS5 beam-seas roll RMS from 3.12 to
    3.50 degrees, still inside the published 2-4 degree figure.

    The ``gain`` values are the calibration anchor for absolute motion amplitude and are
    what Gate 1 criterion 7 (SS5 beam-seas roll RMS in single-digit degrees) actually
    tests. Units: periods in seconds, lengths in metres, gains and damping ratios
    dimensionless.

    Returns:
        The frigate parameter set.
    """
    return Vessel(
        name="frigate",
        length_m=124.0,
        draft_m=4.6,
        beam_m=14.0,
        roll=DofParams(tn_s=12.0, zeta=0.06, gain=1.0),
        pitch=DofParams(tn_s=7.0, zeta=0.40, gain=1.0),
        heave=DofParams(tn_s=8.5, zeta=0.35, gain=1.0),
        roll_residual=0.05,
        pitch_residual=0.05,
    )


#: Cut-down corpus used by the Phase 2 split and windowing tests. Two sea states (one of
#: them the held-out SS6), three headings (one of them the held-out 90 deg beam), one
#: speed, and both hulls, so that all four evaluation regimes are constructible and every
#: partition is non-empty. 8 seeds per cell for the frigate, 2 for the held-out s175:
#: 2*3*1*8 = 48 frigate + 2*3*1*2 = 12 s175 = 60 realizations of 600 rows.
SMALL_SEA_STATES: tuple[str, ...] = ("SS5", "SS6")
SMALL_HEADINGS_DEG: tuple[float, ...] = (135.0, 90.0, 45.0)
SMALL_SPEEDS_KN: tuple[float, ...] = (12.0,)
SMALL_SEEDS_FRIGATE = 8
SMALL_SEEDS_S175 = 2
SMALL_DURATION_S = 60.0
SMALL_SPINUP_S = 10.0
SMALL_N_REALIZATIONS = 60
SMALL_N_SAMPLES = 600

#: Window geometry for the small corpus: 5 s lookback, horizons 0.5/1/2 s, 2.5 s stride.
#: Chosen so that a 600-sample realization yields several windows while keeping the same
#: arithmetic the production geometry uses.
SMALL_LOOKBACK = 50
SMALL_HORIZONS: tuple[int, ...] = (5, 10, 20)
SMALL_STRIDE = 25

#: The real corpus, when it has been generated. Absent on a clean checkout.
REAL_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "corpus"
CORPUS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "sim" / "corpus.yaml"
DATA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "default.yaml"


def small_sim_config() -> SimConfig:
    """Return the cut-down corpus configuration used by the Phase 2 fixtures.

    Follows the ``_small_config`` pattern of ``tests/test_generate.py`` but widens the grid
    so that every evaluation regime has a non-empty test set: SS6 must be present for
    ``unseen_seastate``, 90 deg for ``unseen_heading``, and both hulls for
    ``unseen_vessel``.

    Returns:
        A :class:`dmf.config.SimConfig` producing 60 realizations of 600 samples.
    """
    base = load_sim(CORPUS_CONFIG_PATH)
    keep = {s.name for s in base.sea_states} & set(SMALL_SEA_STATES)
    return SimConfig(
        **{
            **base.__dict__,
            "sea_states": tuple(s for s in base.sea_states if s.name in keep),
            "headings_deg": SMALL_HEADINGS_DEG,
            "speeds_kn": SMALL_SPEEDS_KN,
            "seeds_per_cell": SMALL_SEEDS_FRIGATE,
            "seeds_per_cell_by_vessel": (("s175", SMALL_SEEDS_S175),),
            "duration_s": SMALL_DURATION_S,
            "spinup_s": SMALL_SPINUP_S,
        }
    )


@pytest.fixture(scope="session")
def small_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small Parquet corpus once per test session.

    Built in a temporary directory rather than read from ``artifacts/`` so that Gate 2 is
    verifiable on a clean checkout where no corpus has been generated.

    Returns:
        Path to the corpus root, containing ``manifest.parquet`` and 60 realization files.
    """
    root = tmp_path_factory.mktemp("small_corpus")
    return generate_corpus(small_sim_config(), root / "corpus", n_workers=2)


@pytest.fixture(scope="session")
def small_manifest(small_corpus: Path) -> pd.DataFrame:
    """Return the manifest of the small corpus.

    Returns:
        One row per realization, 60 rows.
    """
    return load_manifest(small_corpus)


@pytest.fixture
def small_data_cfg() -> DataConfig:
    """Return a task configuration sized for the small corpus.

    The channel selection and observation mode match ``configs/data/default.yaml``; only
    the window geometry is shrunk, so that the split and normalisation behaviour under test
    is the production behaviour.

    Returns:
        The task configuration.
    """
    base = load_data(DATA_CONFIG_PATH)
    return replace(base, lookback=SMALL_LOOKBACK, horizons=SMALL_HORIZONS, stride=SMALL_STRIDE)


@pytest.fixture(scope="session")
def real_corpus() -> Path:
    """Return the real corpus root, skipping the test if it has not been generated.

    Returns:
        Path to ``artifacts/corpus``.
    """
    if not (REAL_CORPUS_ROOT / "manifest.parquet").exists():
        pytest.skip(f"no generated corpus at {REAL_CORPUS_ROOT}; run `make data`")
    return REAL_CORPUS_ROOT


@pytest.fixture(scope="session")
def real_manifest(real_corpus: Path) -> pd.DataFrame:
    """Return the manifest of the real corpus.

    Returns:
        One row per realization, 2304 rows.
    """
    return load_manifest(real_corpus)
