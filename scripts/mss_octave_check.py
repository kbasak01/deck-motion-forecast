#!/usr/bin/env python3
"""Check the NumPy MSS bridge against MSS's own waveMotionRAO.m running in GNU Octave.

Phase 8. `src/dmf/mss/synth.py` is a NumPy port of an m-file; this is what makes that port
a claim rather than an assertion. Two things are verified, in order:

1. **The local patch is behaviour-preserving.** `mss/waveMotionRAO_seeded.m` differs from
   upstream only in where the random phases come from (upstream seeds `rng(12345)` into a
   `persistent`, so every process yields the same realization). Stock and patched are run
   against the phases stock itself would draw, and must agree exactly.
2. **The NumPy port reproduces the m-file.** Both backends are driven from an identical
   wave grid and phase set, across headings and speeds, and compared pointwise.

Driving both from the same phases matters: without it, a parity check compares two
different random draws and can only ever make a distributional statement.

Octave is optional. Without it this script reports SKIPPED and exits 0; `scripts/gate8.py`
decides what an unverified bridge means for the gate.

These are simulated trajectories, not measurements of a real ship.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

from dmf.mss.export import MSSRealizationSpec, _build_grid, realization_seed
from dmf.mss.synth import synthesize_mss_motion
from dmf.mss.vessel import DOF_NAMES, load_mss_vessel

KNOT_M_S = 0.514444

#: Cells checked. Chosen to span both headings and the whole speed axis, because the
#: encounter-frequency term is the only place speed enters and it is heading-weighted.
PARITY_CELLS: tuple[tuple[float, float], ...] = (
    (180.0, 0.0),
    (180.0, 12.0),
    (135.0, 6.0),
    (135.0, 12.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/mss/s175_ss5.yaml"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/mss"),
        help="Where to write octave_parity.csv.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=300,
        help="Timesteps per cell. The m-file is called once per step, so this is the cost "
        "driver: ~80 s for a full 6000-step record, which is why Octave verifies the "
        "bridge and NumPy generates the corpus rather than the other way round.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed relative pointwise deviation.",
    )
    return parser


def _run_octave(mss_dir: Path, func: str, case: Path, out: Path) -> None:
    cmd = [
        "octave",
        "--no-gui",
        "--quiet",
        "--eval",
        f"cd('{mss_dir}'); {func}('{case}','{out}')",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"octave {func} failed:\n{proc.stdout}\n{proc.stderr}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if shutil.which("octave") is None:
        print("[parity] octave not found on PATH -> SKIPPED", file=sys.stderr)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "skipped", "reason": "octave not installed"}]).to_csv(
            args.out_dir / "octave_parity.csv", index=False
        )
        return 0

    cfg = yaml.safe_load(args.config.read_text())
    mss_dir = Path("mss").resolve()
    vessel = load_mss_vessel(Path(cfg["vessel"]["mat_path"]))
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        first = True
        for heading, speed_kn in PARITY_CELLS:
            spec = MSSRealizationSpec("mss", heading, speed_kn, 0)
            grid = _build_grid(spec, cfg, realization_seed(spec))
            t = float(cfg["record"]["t_start_s"]) + np.arange(args.n_steps) / float(
                cfg["record"]["fs_hz"]
            )
            case = tmpdir / f"case_{heading:.0f}_{speed_kn:.0f}.mat"
            sio.savemat(
                str(case),
                {
                    "Omega": grid.w_rad_s,
                    "Amp": grid.amplitude_m,
                    "phases": grid.phase_rad,
                    "t": t,
                    "U": speed_kn * KNOT_M_S,
                    "beta_wave": np.deg2rad(heading),
                },
            )

            if first:
                # Step 1: is the patch behaviour-preserving?
                verdict = tmpdir / "patch_check.csv"
                _run_octave(mss_dir, "check_patch", case, verdict)
                pc = pd.read_csv(verdict)
                rows.append(
                    {
                        "check": "patch_equivalence",
                        "heading_deg": heading,
                        "speed_kn": speed_kn,
                        "dof": "all",
                        "rel_deviation": float(pc["rel_diff"].iloc[0]),
                        "rms_ratio": float("nan"),
                        "passed": bool(pc["max_abs_diff"].iloc[0] == 0.0),
                    }
                )
                print(
                    f"[parity] patch equivalence: max_abs_diff="
                    f"{float(pc['max_abs_diff'].iloc[0]):.3e}",
                    file=sys.stderr,
                )
                first = False

            # Step 2: does the NumPy port reproduce the m-file?
            out = tmpdir / f"out_{heading:.0f}_{speed_kn:.0f}.csv"
            _run_octave(mss_dir, "run_case", case, out)
            oct_df = pd.read_csv(out)
            motion = synthesize_mss_motion(
                vessel, grid, heading, speed_kn * KNOT_M_S, t,
                speed_index=int(cfg["vessel"]["speed_index"]),
            )
            for i, dof in enumerate(DOF_NAMES):
                for label, ours, theirs in (
                    ("eta", motion.eta[i], oct_df[f"eta{i + 1}"].to_numpy()),
                    ("eta_dot", motion.eta_dot[i], oct_df[f"etadot{i + 1}"].to_numpy()),
                ):
                    scale = max(float(np.max(np.abs(theirs))), 1e-30)
                    rel = float(np.max(np.abs(ours - theirs))) / scale
                    ratio = float(
                        np.sqrt(np.mean(ours**2)) / max(np.sqrt(np.mean(theirs**2)), 1e-30)
                    )
                    rows.append(
                        {
                            "check": f"numpy_vs_octave_{label}",
                            "heading_deg": heading,
                            "speed_kn": speed_kn,
                            "dof": dof,
                            "rel_deviation": rel,
                            "rms_ratio": ratio,
                            "passed": rel <= args.tolerance,
                        }
                    )
            worst = max(
                r["rel_deviation"] for r in rows if r["heading_deg"] == heading
                and r["speed_kn"] == speed_kn and r["check"] != "patch_equivalence"
            )
            print(
                f"[parity] heading {heading:5.1f} deg, {speed_kn:4.1f} kn: "
                f"worst relative deviation {worst:.3e}",
                file=sys.stderr,
            )

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "octave_parity.csv"
    table.to_csv(path, index=False)

    worst_all = float(table["rel_deviation"].max())
    n_fail = int((~table["passed"].astype(bool)).sum())
    print(f"[parity] wrote {path} ({len(table)} rows)", file=sys.stderr)
    print(
        f"[parity] worst relative deviation overall {worst_all:.3e} "
        f"(tolerance {args.tolerance}); {n_fail} row(s) failed",
        file=sys.stderr,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
