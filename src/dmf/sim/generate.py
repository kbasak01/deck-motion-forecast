"""Corpus generation driver.

Produces the Parquet corpus that every later phase consumes. Generation is embarrassingly
parallel over realizations and is driven through ``multiprocessing``.

Corpus schema, one row per sample::

    t, roll, pitch, heave, roll_rate, pitch_rate, heave_rate, heave_acc,
    roll_imu, pitch_imu, heave_imu, roll_rate_imu, pitch_rate_imu, heave_rate_imu,
    seed, ss, heading, speed, vessel

Units: ``t`` seconds; ``roll``/``pitch`` **degrees**; ``heave`` metres; ``roll_rate``/
``pitch_rate`` degrees per second; ``heave_rate`` metres per second; ``heave_acc`` metres
per second squared; ``heading`` degrees; ``speed`` knots. Motion columns are stored as
float32; metadata columns carry the realization's grid coordinates so that
:mod:`dmf.data.splits` can build seed-disjoint splits without re-deriving them.

The six ``*_imu`` columns are the same six channels seen through
:func:`dmf.sim.imu.apply_observation_model`, in the same units as their clean
counterparts. They are written alongside the clean channels rather than into a second
corpus so that the Phase 6 ``ideal`` vs ``imu`` ablation is a column selection, not a
regeneration. ``heave_acc`` is shared: it is the raw accelerometer channel, so both
observation modes see the same values.

``t`` is **absolute simulation time** and therefore starts at ``cfg.spinup_s``, not at
zero: the spin-up transient is discarded by slicing, and keeping the original time origin
makes that discard visible in the stored data rather than a claim in a docstring. ``t`` is
the one float64 column; the 24 kB per file it costs buys exact sample timestamps.

Storage layout under ``out_dir``::

    manifest.parquet                       one row per realization
    <vessel>/<ss>_h<heading>_u<speed>_s<seed>.parquet

One file per realization, because the Phase 2 realization-level split is then a partition
of a file list. A split expressed that way is structurally incapable of putting two windows
from the same realization on opposite sides of a train/test boundary.
"""

import hashlib
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dmf.config import SimConfig
from dmf.sim.encounter import is_encounter_monotonic, knots_to_m_s
from dmf.sim.imu import IDEAL_COLUMNS, IMU_COLUMNS, apply_observation_model
from dmf.sim.response import synthesize_motion
from dmf.sim.spectra import sample_components
from dmf.sim.vessel import Vessel, load_vessel

__all__ = [
    "CORPUS_SEED_NAMESPACE",
    "MANIFEST_NAME",
    "RealizationSpec",
    "generate_corpus",
    "realization_grid",
    "realization_path",
    "realization_seed_sequence",
    "simulate_realization",
]

#: Namespace constant mixed into every realization's seed. Bump it to re-key the whole
#: corpus deliberately; nothing else in the pipeline should ever change a seed.
CORPUS_SEED_NAMESPACE: int = 0x646D66_31  # "dmf" + phase 1

#: File name of the corpus index written at the dataset root.
MANIFEST_NAME: str = "manifest.parquet"

#: Directory holding the vessel YAML files, resolved relative to this source file so that
#: generation does not depend on the process working directory.
VESSEL_CONFIG_DIR: Path = Path(__file__).resolve().parents[3] / "configs" / "sim" / "vessels"

#: Stored dtype of every corpus column. ``t`` is float64 for exact timestamps, the motion
#: channels are float32 (about 0.1 percent of their own RMS, far below any physics claim
#: made from them), and the grid coordinates are stored per row so that a single file is
#: self-describing.
CORPUS_DTYPES: dict[str, str] = {
    "t": "float64",
    **dict.fromkeys(IDEAL_COLUMNS, "float32"),
    **dict.fromkeys(IMU_COLUMNS, "float32"),
    "seed": "int32",
    "ss": "str",
    "heading": "float32",
    "speed": "float32",
    "vessel": "str",
}

#: Column order of a corpus file.
CORPUS_COLUMNS: tuple[str, ...] = tuple(CORPUS_DTYPES)


@dataclass(frozen=True)
class RealizationSpec:
    """Coordinates of one realization in the corpus grid.

    The seed is the unit of train/test separation for the whole project: splits are by
    realization seed, never by time window. Two windows drawn from the same realization
    overlap in the wave field that generated them, so putting one in train and one in test
    inflates every metric.

    Attributes:
        seed: Realization seed, unique within its grid cell. A small ordinal
            (``0 .. seeds_per_cell-1``), not the generator entropy:
            :func:`realization_seed_sequence` hashes the whole coordinate tuple into the
            entropy actually used, and :mod:`dmf.data.splits` slices on this ordinal.
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


def realization_seed_sequence(spec: RealizationSpec) -> np.random.SeedSequence:
    """Derive a realization's seed sequence from its grid coordinates alone.

    The entropy is ``blake2b`` of the canonical coordinate string
    ``"<vessel>|<sea_state>|<heading>|<speed>|<seed>"``, mixed with
    :data:`CORPUS_SEED_NAMESPACE`. Three properties follow, and all three are asserted in
    ``tests/test_generate.py``:

    - **Isolated reproducibility.** Any one realization can be regenerated without
      generating, or even enumerating, the rest of the corpus.
    - **Worker-count invariance.** No generator state is shared or consumed in grid order,
      so the corpus does not depend on ``n_workers``, on task scheduling, or on how a pool
      chunks its work. Deriving seeds by counting off a parent generator is the usual way
      a "reproducible" parallel corpus turns out not to be.
    - **Order independence.** Adding a sea state or a heading to the grid does not disturb
      the seeds of the realizations already in it, because nothing is keyed by list index.

    Args:
        spec: Grid coordinates and seed ordinal.

    Returns:
        A ``numpy.random.SeedSequence``. Callers spawn children from it rather than using
        it twice, so that the wave synthesis stream and the IMU stream are independent.
    """
    key = f"{spec.vessel}|{spec.sea_state}|{spec.heading_deg:.6f}|{spec.speed_kn:.6f}|{spec.seed:d}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    words = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)]
    return np.random.SeedSequence([CORPUS_SEED_NAMESPACE, *words])


def realization_path(spec: RealizationSpec) -> Path:
    """Return the corpus-relative path of one realization's Parquet file.

    Args:
        spec: Grid coordinates and seed ordinal.

    Returns:
        A relative path of the form
        ``<vessel>/<ss>_h<heading>_u<speed>_s<seed>.parquet``, with the heading in degrees
        and the speed in knots, both zero-padded to one decimal so the names sort in grid
        order.
    """
    name = (
        f"{spec.sea_state}_h{spec.heading_deg:05.1f}_"
        f"u{spec.speed_kn:04.1f}_s{spec.seed:03d}.parquet"
    )
    return Path(spec.vessel) / name


def realization_grid(cfg: SimConfig) -> list[RealizationSpec]:
    """Enumerate every realization in the corpus grid.

    The **full** grid is returned: no (speed, heading) cell is rejected. Cells for which
    the encounter-frequency map ``w -> w_e`` is not monotonic across the synthesis band are
    kept and handled explicitly -- the response transfer function is evaluated at
    ``abs(w_e)`` while the signed ``w_e`` stays in the cosine argument of the time-domain
    superposition, which keeps superposition well defined where two wave frequencies share
    an encounter frequency. See ``docs/protocol.md`` P1-D1 for the decision and its
    reasoning.

    :func:`dmf.sim.encounter.is_encounter_monotonic` **reports** that condition rather than
    filtering on it. Its verdict is written into the ``encounter_monotonic`` column of the
    manifest and summarised in ``docs/corpus_card.md``, so the corpus documents which of
    its cells sit in the awkward regime instead of silently omitting or silently including
    them. On the corpus grid exactly two cells are affected: (45 deg, 6 kn) and
    (45 deg, 12 kn).

    Args:
        cfg: Corpus-generation settings.

    Returns:
        One :class:`RealizationSpec` per (vessel, sea state, heading, speed, seed) cell,
        in that nesting order. Seed counts are per vessel via
        :meth:`dmf.config.SimConfig.seeds_for`, so the held-out ``s175`` hull can carry
        fewer realizations than the primary hull it is never trained alongside.
    """
    specs: list[RealizationSpec] = []
    for vessel in cfg.vessels:
        for sea in cfg.sea_states:
            for heading_deg in cfg.headings_deg:
                for speed_kn in cfg.speeds_kn:
                    specs.extend(
                        RealizationSpec(
                            seed=seed,
                            sea_state=sea.name,
                            heading_deg=float(heading_deg),
                            speed_kn=float(speed_kn),
                            vessel=vessel,
                        )
                        for seed in range(cfg.seeds_for(vessel))
                    )
    return specs


@cache
def _vessel(name: str) -> Vessel:
    """Load and cache one vessel definition.

    Args:
        name: Vessel config stem, e.g. ``"frigate"``.

    Returns:
        The parsed vessel. Cached because a worker process simulates hundreds of
        realizations of the same hull.

    Raises:
        ValueError: If no config file exists for ``name``.
    """
    path = VESSEL_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown vessel {name!r}: no config at {path}")
    return load_vessel(path)


def simulate_realization(spec: RealizationSpec, cfg: SimConfig) -> pd.DataFrame:
    """Simulate one realization end to end.

    Draws a jittered frequency grid and a phase set from ``spec.seed``, synthesises the
    elevation record and the three DOF responses from that shared phase set, differentiates
    to obtain the rate and acceleration channels, and discards the leading ``cfg.spinup_s``
    seconds of transient.

    The observation model is applied to the **full** ``spinup_s + duration_s`` record
    before the spin-up is sliced away, so that the causal high-pass filters in the IMU
    heave reconstruction and the attitude bias random walk have both settled and
    accumulated realistically by the time the retained window begins.

    Args:
        spec: Grid coordinates and seed for this realization.
        cfg: Corpus-generation settings.

    Returns:
        A DataFrame with the corpus schema and units given in the module docstring, of
        length ``round(cfg.duration_s * cfg.fs_hz)``. ``t`` starts at ``cfg.spinup_s``.

    Raises:
        ValueError: If ``spec.vessel`` has no config file or ``spec.sea_state`` is not in
            ``cfg.sea_states``.
    """
    sea = next((s for s in cfg.sea_states if s.name == spec.sea_state), None)
    if sea is None:
        known = [s.name for s in cfg.sea_states]
        raise ValueError(f"unknown sea state {spec.sea_state!r}, expected one of {known}")
    vessel = _vessel(spec.vessel)

    wave_seq, imu_seq = realization_seed_sequence(spec).spawn(2)
    n_keep = int(round(cfg.duration_s * cfg.fs_hz))
    n_total = n_keep + int(round(cfg.spinup_s * cfg.fs_hz))
    t_s = np.arange(n_total, dtype=np.float64) / cfg.fs_hz

    components = sample_components(
        hs_m=sea.hs_m,
        tp_s=sea.tp_s,
        gamma=sea.gamma,
        n_components=cfg.n_components,
        w_min_rad_s=cfg.w_min_rad_s,
        w_max_rad_s=cfg.w_max_rad_s,
        rng=np.random.default_rng(wave_seq),
        jitter=cfg.jitter_frequencies,
    )
    motion = synthesize_motion(
        components,
        vessel,
        spec.heading_deg,
        knots_to_m_s(spec.speed_kn),
        t_s,
    )

    clean = pd.DataFrame(
        {
            "t": motion.t_s,
            "roll": motion.roll_deg,
            "pitch": motion.pitch_deg,
            "heave": motion.heave_m,
            "roll_rate": motion.roll_rate_dps,
            "pitch_rate": motion.pitch_rate_dps,
            "heave_rate": motion.heave_rate_m_s,
            "heave_acc": motion.heave_acc_m_s2,
        }
    )
    observed = apply_observation_model(clean, "imu", cfg.fs_hz, np.random.default_rng(imu_seq))
    for clean_name, imu_name in zip(IDEAL_COLUMNS, IMU_COLUMNS, strict=False):
        clean[imu_name] = observed[clean_name]

    out = clean.iloc[n_total - n_keep :].reset_index(drop=True)
    out["seed"] = spec.seed
    out["ss"] = spec.sea_state
    out["heading"] = spec.heading_deg
    out["speed"] = spec.speed_kn
    out["vessel"] = spec.vessel
    return out[list(CORPUS_COLUMNS)].astype(CORPUS_DTYPES)


def _manifest_row(spec: RealizationSpec, cfg: SimConfig, n_rows: int) -> dict[str, Any]:
    """Build one manifest entry.

    Args:
        spec: Grid coordinates and seed ordinal.
        cfg: Corpus-generation settings.
        n_rows: Rows actually written for this realization.

    Returns:
        A mapping of manifest column name to value. ``heading`` is degrees, ``speed``
        knots, ``duration_s`` and ``spinup_s`` seconds, ``fs_hz`` hertz.
    """
    w_band = np.linspace(cfg.w_min_rad_s, cfg.w_max_rad_s, 256)
    return {
        "vessel": spec.vessel,
        "ss": spec.sea_state,
        "heading": spec.heading_deg,
        "speed": spec.speed_kn,
        "seed": spec.seed,
        "n_rows": n_rows,
        "fs_hz": cfg.fs_hz,
        "duration_s": cfg.duration_s,
        "spinup_s": cfg.spinup_s,
        "n_components": cfg.n_components,
        "jitter": cfg.jitter_frequencies,
        "encounter_monotonic": is_encounter_monotonic(
            w_band, knots_to_m_s(spec.speed_kn), spec.heading_deg
        ),
        "entropy": f"{realization_seed_sequence(spec).entropy}",
        "path": realization_path(spec).as_posix(),
    }


def _write_one(task: tuple[RealizationSpec, SimConfig, Path]) -> dict[str, Any]:
    """Simulate one realization and write its Parquet file.

    Module-level so that it is picklable by ``multiprocessing``. It touches no shared
    state: everything it needs comes from ``task``, and its seed comes from the spec, so
    two workers can never interfere.

    Args:
        task: A ``(spec, cfg, out_dir)`` tuple.

    Returns:
        The manifest row for the realization written.
    """
    spec, cfg, out_dir = task
    frame = simulate_realization(spec, cfg)
    path = out_dir / realization_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return _manifest_row(spec, cfg, len(frame))


def _run_tasks(
    tasks: list[tuple[RealizationSpec, SimConfig, Path]], n_workers: int
) -> Iterator[dict[str, Any]]:
    """Execute realization tasks serially or across a process pool.

    Args:
        tasks: One entry per realization.
        n_workers: Worker processes. ``1`` runs in the calling process, which keeps the
            single-worker path free of any pool behaviour that a multi-worker run could
            differ from.

    Yields:
        Manifest rows, in completion order. Callers sort before writing the manifest, so
        completion order never reaches disk.
    """
    if n_workers == 1:
        yield from (_write_one(task) for task in tasks)
        return
    # "spawn", not "fork": forking a process that has already imported a threaded BLAS is
    # the classic source of silent hangs, and a corpus generator that deadlocks once every
    # few hundred runs is worse than one that starts a second slower.
    #
    # Each realization is already a dense (n_samples, n_components) trig evaluation, so a
    # worker gains nothing from a multi-threaded BLAS and loses to oversubscription when
    # n_workers is near the core count. Spawned children inherit this environment at
    # import time, which is the only point where these variables are read.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        yield from pool.imap_unordered(_write_one, tasks, chunksize=4)


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
    if n_workers < 1:
        raise ValueError(f"n_workers must be positive, got {n_workers}")
    specs = realization_grid(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(spec, cfg, out_dir) for spec in specs]

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    report_every = max(1, len(tasks) // 20)
    for done, row in enumerate(_run_tasks(tasks, n_workers), start=1):
        rows.append(row)
        if done % report_every == 0 or done == len(tasks):
            elapsed = time.perf_counter() - started
            rate = done / elapsed
            print(
                f"[corpus] {done}/{len(tasks)} realizations "
                f"({elapsed:6.1f} s elapsed, {rate:5.1f}/s, "
                f"eta {(len(tasks) - done) / rate:6.1f} s)",
                flush=True,
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["vessel", "ss", "heading", "speed", "seed"], ignore_index=True
    )
    manifest.to_parquet(
        out_dir / MANIFEST_NAME, engine="pyarrow", compression="snappy", index=False
    )
    return out_dir
