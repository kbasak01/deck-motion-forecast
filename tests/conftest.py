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

import numpy as np
import pytest

from dmf.config import SeaState
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
