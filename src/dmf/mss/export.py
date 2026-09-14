"""Generate and export the MSS S175 cross-validation records.

One CSV per realization, plus a manifest carrying the realized spectrum statistics that
Gate 8 predicate 1 is read from. Nothing here decides anything; it writes artifacts that
later stages score.

Everything here is simulated. These are MSS trajectories, which are another simulation, not
measurements of a real ship.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from dmf.mss.convert import assert_corpus_units, mss_motion_to_frame
from dmf.mss.synth import WaveGrid, sample_corpus_grid, sample_mss_grid, synthesize_mss_motion
from dmf.mss.vessel import MSSVessel, load_mss_vessel
from dmf.typedefs import FloatArray

#: Namespace mixed into every seed so MSS realizations can never collide with corpus ones.
MSS_SEED_NAMESPACE: Final[bytes] = b"dmf-phase8-mss"

#: Grid conventions exported. ``mss`` is primary; ``corpus`` is the attribution control.
GRID_KINDS: Final[tuple[str, ...]] = ("mss", "corpus")


@dataclass(frozen=True)
class MSSRealizationSpec:
    """Coordinates of one MSS realization.

    Attributes:
        grid_kind: ``"mss"`` or ``"corpus"``.
        heading_deg: Encounter angle, degrees, corpus convention.
        speed_kn: Forward speed, knots.
        seed: Seed ordinal.
    """

    grid_kind: str
    heading_deg: float
    speed_kn: float
    seed: int

    @property
    def stem(self) -> str:
        """Filename stem, mirroring the corpus naming scheme."""
        return f"S175_h{self.heading_deg:05.1f}_u{self.speed_kn:04.1f}_s{self.seed:03d}"


def realization_seed(spec: MSSRealizationSpec) -> np.random.Generator:
    """Derive a reproducible generator from the realization coordinates.

    Uses blake2b over the coordinate string, mirroring
    :func:`dmf.sim.generate.realization_seed_sequence`, so the result is independent of
    iteration order and of how many workers run.

    Args:
        spec: Realization coordinates.

    Returns:
        A seeded generator.
    """
    key = f"{spec.grid_kind}|{spec.heading_deg}|{spec.speed_kn}|{spec.seed}".encode()
    digest = hashlib.blake2b(MSS_SEED_NAMESPACE + key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _time_axis(cfg: dict[str, Any]) -> FloatArray:
    rec = cfg["record"]
    n = int(round(float(rec["duration_s"]) * float(rec["fs_hz"])))
    return float(rec["t_start_s"]) + np.arange(n, dtype=np.float64) / float(rec["fs_hz"])


def _build_grid(
    spec: MSSRealizationSpec, cfg: dict[str, Any], rng: np.random.Generator
) -> WaveGrid:
    sea = cfg["sea_state"]
    grids = cfg["grids"]
    if spec.grid_kind == "mss":
        return sample_mss_grid(
            hs_m=float(sea["hs_m"]),
            tp_s=float(sea["tp_s"]),
            gamma=float(sea["gamma"]),
            n_components=int(grids["n_components"]),
            omega_max_rad_s=float(grids["mss_omega_max_rad_s"]),
            rng=rng,
        )
    if spec.grid_kind == "corpus":
        lo, hi = grids["corpus_band_rad_s"]
        return sample_corpus_grid(
            hs_m=float(sea["hs_m"]),
            tp_s=float(sea["tp_s"]),
            gamma=float(sea["gamma"]),
            n_components=int(grids["n_components"]),
            w_min_rad_s=float(lo),
            w_max_rad_s=float(hi),
            rng=rng,
        )
    raise ValueError(f"unknown grid_kind {spec.grid_kind!r}, expected one of {GRID_KINDS}")


def zero_crossing_period(signal: FloatArray, fs_hz: float) -> float:
    """Mean period between successive upward zero crossings, seconds.

    Args:
        signal: A zero-mean time series.
        fs_hz: Sample rate, Hz.

    Returns:
        Mean up-crossing period in seconds, or ``nan`` if fewer than two crossings.
    """
    centred = np.asarray(signal, dtype=np.float64)
    centred = centred - centred.mean()
    negative = np.signbit(centred)
    crossings = np.where((~negative[1:]) & negative[:-1])[0]
    if crossings.size < 2:
        return float("nan")
    return float(np.mean(np.diff(crossings)) / fs_hz)


def generate_realization(
    spec: MSSRealizationSpec, vessel: MSSVessel, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthesise and convert one realization.

    Args:
        spec: Realization coordinates.
        vessel: Loaded MSS vessel.
        cfg: Parsed ``configs/mss/s175_ss5.yaml``.

    Returns:
        The corpus-schema frame and a manifest row including realized spectrum statistics.
    """
    rng = realization_seed(spec)
    grid = _build_grid(spec, cfg, rng)
    t = _time_axis(cfg)
    speed_m_s = float(spec.speed_kn) * 0.514444
    motion = synthesize_mss_motion(
        vessel,
        grid,
        heading_deg=spec.heading_deg,
        speed_m_s=speed_m_s,
        t_s=t,
        speed_index=int(cfg["vessel"]["speed_index"]),
    )
    frame = mss_motion_to_frame(motion)
    assert_corpus_units(frame)

    fs = float(cfg["record"]["fs_hz"])
    sea = cfg["sea_state"]
    tz_target = float(cfg["match_tolerance"]["tz_over_tp"]) * float(sea["tp_s"])
    # The spectrum-match predicate is a statement about the sea state, so it is read from
    # the EARTH-frame elevation. The encounter-frame record that MSS itself returns is
    # Doppler-compressed by forward speed -- measured here as Tz falling from 7.93 s at
    # 0 kn to 4.86 s at 12 kn in head seas -- which is correct physics, not a mismatch.
    hs_realized = float(4.0 * np.std(motion.elevation_earth_m))
    tz_realized = zero_crossing_period(motion.elevation_earth_m, fs)
    tz_encounter = zero_crossing_period(motion.elevation_m, fs)

    row: dict[str, Any] = {
        **asdict(spec),
        "stem": spec.stem,
        "n_rows": int(frame.shape[0]),
        "fs_hz": fs,
        "duration_s": float(cfg["record"]["duration_s"]),
        "n_components": int(cfg["grids"]["n_components"]),
        "hs_target_m": float(sea["hs_m"]),
        "hs_realized_m": hs_realized,
        "hs_rel_err": hs_realized / float(sea["hs_m"]) - 1.0,
        "tz_target_s": tz_target,
        "tz_realized_s": tz_realized,
        "tz_rel_err": tz_realized / tz_target - 1.0,
        "tz_encounter_s": tz_encounter,
    }
    for dof, unit_col in (("roll", "roll"), ("pitch", "pitch"), ("heave", "heave")):
        row[f"rms_{dof}"] = float(np.sqrt(np.mean(frame[unit_col].to_numpy() ** 2)))
        row[f"tz_{dof}_s"] = zero_crossing_period(frame[unit_col].to_numpy(), fs)
    return frame, row


def generate_all(cfg: dict[str, Any], out_dir: Path) -> Path:
    """Generate every realization in the config and write CSVs plus a manifest.

    Args:
        cfg: Parsed ``configs/mss/s175_ss5.yaml``.
        out_dir: Destination root; one subdirectory per grid convention.

    Returns:
        Path to the written manifest CSV.
    """
    vessel = load_mss_vessel(Path(cfg["vessel"]["mat_path"]))
    rows: list[dict[str, Any]] = []
    for grid_kind in GRID_KINDS:
        target = out_dir / grid_kind
        target.mkdir(parents=True, exist_ok=True)
        for heading in cfg["headings_deg"]:
            for speed in cfg["speeds_kn"]:
                for seed in cfg["seeds"]:
                    spec = MSSRealizationSpec(
                        grid_kind=grid_kind,
                        heading_deg=float(heading),
                        speed_kn=float(speed),
                        seed=int(seed),
                    )
                    frame, row = generate_realization(spec, vessel, cfg)
                    path = target / f"{spec.stem}.csv"
                    frame.to_csv(path, index=False, float_format="%.6g")
                    row["path"] = str(path)
                    rows.append(row)
    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path
