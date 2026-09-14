"""Convert MSS 6-DOF motion into the corpus storage schema.

This module is the unit and sign boundary, and it is deliberately loud. `CLAUDE.md` names
mixed radians and degrees "the most common bug class in this domain", and Phase 8's
carry-forward delta 2 adds the sharper point: **a units error cannot be caught by the
headline metric.** Skill is ``1 - RMSE_model/RMSE_persistence``, a ratio over the same data,
so a uniform factor of 57.3 cancels exactly and leaves every skill score unchanged. What it
moves is raw RMSE and every absolute threshold, including the quiescence bands
(3.0 deg / 2.0 deg / 0.8 m/s permissive, 1.5 / 1.0 / 0.4 strict). So the unit is asserted
here, at the boundary, and not inferred downstream.

Signs are the trap delta 2 does *not* mention, and they are worse, because they do not
cancel: a linear forecaster is sign-equivariant but a TCN or an LSTM is not, so feeding a
sign-flipped channel silently degrades exactly the models this phase is testing.

**The conventions below were established by measurement, not from MSS's source comments.**
Two comments in ``HYDRO/utils/readdata/read_veres_TF.m:141-147`` are stale and contradicted
by the code beneath them:

* it says "0 deg beam seas", but the loop at ``:178-181`` reverses the heading index
  precisely because "in Veres the headings are defined relative to the bow while the MSS
  standard is relative to the stern", so Veres 0 deg (head) maps to MSS 180 deg. The RAO
  data agrees: roll is ~0 at 0 and 180 deg and maximal at 135 deg, so 0/180 are the
  symmetric directions and 90 deg is beam. ``waveForceRAO.m:21`` and ``encounter.m:7``
  both independently state 0 following / pi head. **MSS 180 deg is head seas, matching the
  corpus convention**, so headings pass through unchanged.
* it says "z-upwards", which would make x-forward/y-starboard/z-up a left-handed frame.
  Measured instead: the MSS heave RAO phase tends to pi as ``w -> 0`` while its amplitude
  tends to 1.0 m/m. A ship contouring a long wave moves *with* the surface, so a phase of
  pi means MSS heave is positive **down** -- SNAME z-down, as expected of Fossen's
  convention and contrary to the comment.

Given SNAME axes (x forward, y starboard, z down), the rotation conventions follow:
positive roll takes starboard down and positive pitch takes the bow up, which are both the
corpus conventions. So **heave is the only channel whose sign changes.** That conclusion is
carried by :data:`MSS_TO_CORPUS_SIGN` and is additionally covered by a sign-flip ablation in
the evaluation, so that the reported conclusion can be shown to be robust to it being wrong.

Everything here is simulated.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from dmf.mss.synth import MSSMotion
from dmf.mss.vessel import DOF_INDEX

#: Sign applied to each MSS DOF to reach the corpus convention.
#:
#: MSS uses SNAME axes (x forward, y starboard, z **down**); the corpus stores heave
#: positive **up**, roll positive to starboard and pitch positive bow-up. Under SNAME,
#: positive roll is already starboard-down and positive pitch is already bow-up, so only
#: heave inverts. Derived in this module's docstring, not copied from a comment.
MSS_TO_CORPUS_SIGN: Final[dict[str, float]] = {
    "roll": +1.0,
    "pitch": +1.0,
    "heave": -1.0,
}

#: Columns emitted, in corpus storage order. A subset of `dmf.sim.generate.CORPUS_COLUMNS`.
MSS_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "t",
    "roll",
    "pitch",
    "heave",
    "roll_rate",
    "pitch_rate",
    "heave_rate",
)

#: Physically implausible magnitudes that almost certainly mean a unit or sign error.
#: A radians-as-degrees slip inflates an angle by 57.3x, which these bracket.
_MAX_PLAUSIBLE_DEG: Final[float] = 60.0
_MAX_PLAUSIBLE_HEAVE_M: Final[float] = 30.0


def mss_motion_to_frame(
    motion: MSSMotion,
    *,
    sign_convention: dict[str, float] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Convert an :class:`MSSMotion` to a corpus-schema DataFrame.

    Angles are converted radians -> degrees and rates radians/s -> degrees/s here and
    nowhere else. Heave and heave rate are metres and metres per second in both
    conventions, and only change sign.

    Args:
        motion: Synthesised motion in MSS units and signs.
        sign_convention: Override for :data:`MSS_TO_CORPUS_SIGN`, used by the sign-flip
            ablation. Keys must be ``roll``, ``pitch`` and ``heave``.
        validate: If True, assert the output units are physically plausible.

    Returns:
        DataFrame with :data:`MSS_FRAME_COLUMNS`; angles in degrees, angular rates in
        degrees per second, heave in metres, heave rate in metres per second.

    Raises:
        ValueError: If ``sign_convention`` has unexpected keys.
        AssertionError: If ``validate`` and a channel is implausibly large, which is what a
            radians/degrees confusion or a missing conversion looks like.
    """
    signs = dict(MSS_TO_CORPUS_SIGN) if sign_convention is None else dict(sign_convention)
    if set(signs) != set(MSS_TO_CORPUS_SIGN):
        raise ValueError(
            f"sign_convention must have keys {sorted(MSS_TO_CORPUS_SIGN)}, got {sorted(signs)}"
        )

    i_roll = DOF_INDEX["roll"]
    i_pitch = DOF_INDEX["pitch"]
    i_heave = DOF_INDEX["heave"]

    frame = pd.DataFrame(
        {
            "t": motion.t_s,
            # rad -> deg
            "roll": signs["roll"] * np.rad2deg(motion.eta[i_roll]),
            "pitch": signs["pitch"] * np.rad2deg(motion.eta[i_pitch]),
            # metres in both conventions; sign only
            "heave": signs["heave"] * motion.eta[i_heave],
            # rad/s -> deg/s
            "roll_rate": signs["roll"] * np.rad2deg(motion.eta_dot[i_roll]),
            "pitch_rate": signs["pitch"] * np.rad2deg(motion.eta_dot[i_pitch]),
            "heave_rate": signs["heave"] * motion.eta_dot[i_heave],
        }
    )
    if validate:
        assert_corpus_units(frame)
    return frame[list(MSS_FRAME_COLUMNS)]


def assert_corpus_units(frame: pd.DataFrame) -> None:
    """Assert a converted frame is in corpus units, at the CSV boundary.

    This is Gate 8 predicate 2. It cannot prove the conversion correct, but it does catch
    the failure that the skill score structurally cannot: angles left in radians (which
    makes every angle ~57x too small) or degrees applied twice (~57x too large).

    Args:
        frame: A frame with :data:`MSS_FRAME_COLUMNS`.

    Raises:
        AssertionError: If a column is missing, non-finite, or implausibly large.
    """
    missing = [c for c in MSS_FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise AssertionError(f"converted frame is missing columns: {missing}")

    for col in MSS_FRAME_COLUMNS:
        values = frame[col].to_numpy()
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"column {col!r} contains non-finite values")

    for col, limit, unit in (
        ("roll", _MAX_PLAUSIBLE_DEG, "deg"),
        ("pitch", _MAX_PLAUSIBLE_DEG, "deg"),
        ("roll_rate", _MAX_PLAUSIBLE_DEG, "deg/s"),
        ("pitch_rate", _MAX_PLAUSIBLE_DEG, "deg/s"),
        ("heave", _MAX_PLAUSIBLE_HEAVE_M, "m"),
        ("heave_rate", _MAX_PLAUSIBLE_HEAVE_M, "m/s"),
    ):
        peak = float(np.max(np.abs(frame[col].to_numpy())))
        if peak > limit:
            raise AssertionError(
                f"column {col!r} peaks at {peak:.3f} {unit}, above the plausible limit "
                f"{limit} {unit}. A factor near 57 means radians were converted twice."
            )
