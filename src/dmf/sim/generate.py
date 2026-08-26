"""Corpus generation driver.

Produces the Parquet corpus that every later phase consumes. Generation is embarrassingly
parallel over realizations and is driven through ``multiprocessing``.

Corpus schema, one row per sample::

    t, roll, pitch, heave, roll_rate, pitch_rate, heave_rate, heave_acc,
    seed, ss, heading, speed, vessel

Units: ``t`` seconds; ``roll``/``pitch`` **degrees**; ``heave`` metres; ``roll_rate``/
``pitch_rate`` degrees per second; ``heave_rate`` metres per second; ``heave_acc`` metres
per second squared; ``heading`` degrees; ``speed`` knots. Motion columns are stored as
float32; metadata columns carry the realization's grid coordinates so that
:mod:`dmf.data.splits` can build seed-disjoint splits without re-deriving them.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dmf.config import SimConfig

__all__ = ["RealizationSpec", "generate_corpus", "realization_grid", "simulate_realization"]


@dataclass(frozen=True)
class RealizationSpec:
    """Coordinates of one realization in the corpus grid.

    The seed is the unit of train/test separation for the whole project: splits are by
    realization seed, never by time window. Two windows drawn from the same realization
    overlap in the wave field that generated them, so putting one in train and one in test
    inflates every metric.

    Attributes:
        seed: Realization seed, unique within its grid cell.
        sea_state: Sea state label, e.g. ``"SS5"``.
        heading_deg: Encounter angle, degrees (180 = head seas).
        speed_kn: Forward speed, knots.
        vessel: Vessel config stem, e.g. ``"frigate"``.
    """

    seed: int
    sea_state: str
    heading_deg: float
    speed_kn: float
    vessel: str


def realization_grid(cfg: SimConfig) -> list[RealizationSpec]:
    """Enumerate every realization in the corpus grid.

    Rejects any (speed, heading) combination for which the encounter-frequency map is not
    monotonic across the synthesis band -- see
    :func:`dmf.sim.encounter.encounter_frequency_is_monotonic`. Rejected cells are
    returned nowhere and must be recorded in ``docs/corpus_card.md``, so that the corpus
    documents which regimes it does not cover rather than silently omitting them.

    Args:
        cfg: Corpus-generation settings.

    Returns:
        One :class:`RealizationSpec` per (sea state, heading, speed, vessel, seed) cell
        that survives the encounter-frequency check.
    """
    raise NotImplementedError


def simulate_realization(spec: RealizationSpec, cfg: SimConfig) -> pd.DataFrame:
    """Simulate one realization end to end.

    Draws a jittered frequency grid and a phase set from ``spec.seed``, synthesises the
    elevation record and the three DOF responses from that shared phase set, differentiates
    to obtain the rate and acceleration channels, and discards the leading ``cfg.spinup_s``
    seconds of transient.

    Args:
        spec: Grid coordinates and seed for this realization.
        cfg: Corpus-generation settings.

    Returns:
        A DataFrame with the corpus schema and units given in the module docstring, of
        length ``round(cfg.duration_s * cfg.fs_hz)``.

    Raises:
        ValueError: If ``spec.vessel`` has no config file or ``spec.sea_state`` is not in
            ``cfg.sea_states``.
    """
    raise NotImplementedError


def generate_corpus(cfg: SimConfig, out_dir: Path, n_workers: int) -> Path:
    """Generate the full corpus in parallel and write it to Parquet.

    Args:
        cfg: Corpus-generation settings.
        out_dir: Output directory, created if absent. Lives under ``artifacts/`` and is
            gitignored; the corpus is reproducible from ``cfg`` and the seeds rather than
            committed.
        n_workers: Number of worker processes. Each worker seeds its own generator from
            the realization seed, so output is independent of worker count -- a property
            worth asserting once, since it is the usual way a "reproducible" parallel
            generator turns out not to be.

    Returns:
        Path to the written dataset root.

    Raises:
        ValueError: If ``n_workers`` is not positive.
    """
    raise NotImplementedError
