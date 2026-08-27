"""Corpus channel names and the ``ideal`` <-> ``imu`` correspondence between them.

The same physical quantity has two corpus spellings: the clean channel (``roll``, degrees)
and the noisy twin the ``imu`` observation mode substitutes for it (``roll_imu``, degrees).
Which spelling reaches a results table is decided by ``observation_mode`` at window
construction, so any downstream code that has to recognise a *logical* channel -- the Gate 3
report looking for "roll", for instance -- has to be able to see through the spelling.

This module exists so that correspondence is written down exactly once. It derives from
:data:`dmf.sim.imu.IDEAL_COLUMNS` and :data:`dmf.sim.imu.IMU_COLUMNS`, which remain the
source of truth for the ``_imu`` suffix, and it deliberately carries no torch dependency:
:mod:`dmf.eval.report` is a pandas-only rendering layer and should not acquire one just to
learn that ``roll_imu`` is roll. A suffix rule (``name.endswith("_imu")``) would be a second,
drifting definition of the same fact and is not used anywhere.
"""

from dmf.sim.imu import IDEAL_COLUMNS, IMU_COLUMNS

__all__ = [
    "IDEAL_BY_IMU",
    "IMU_BY_IDEAL",
    "channel_aliases",
    "logical_channel",
]

#: Logical channel name -> ``imu`` corpus column, for the six channels that have a noisy
#: twin. ``heave_acc`` is deliberately absent: it is the raw accelerometer channel and is
#: shared by both observation modes (:mod:`dmf.sim.imu`).
IMU_BY_IDEAL: dict[str, str] = dict(zip(IDEAL_COLUMNS[:6], IMU_COLUMNS, strict=True))

#: The inverse of :data:`IMU_BY_IDEAL`: ``imu`` corpus column -> logical channel name.
IDEAL_BY_IMU: dict[str, str] = {imu: ideal for ideal, imu in IMU_BY_IDEAL.items()}


def logical_channel(name: str) -> str:
    """Return the logical channel a corpus column name denotes.

    Args:
        name: Either a logical channel name (``"roll"``) or its ``imu`` twin
            (``"roll_imu"``). Units are unchanged by the mapping: degrees for angles,
            degrees per second for angular rates, metres for heave, metres per second for
            heave rate.

    Returns:
        The logical name, i.e. the :data:`dmf.sim.imu.IDEAL_COLUMNS` spelling.

    Raises:
        ValueError: If ``name`` is neither spelling of a corpus motion channel.
    """
    if name in IDEAL_COLUMNS:
        return name
    if name in IDEAL_BY_IMU:
        return IDEAL_BY_IMU[name]
    raise ValueError(
        f"unknown channel {name!r}; corpus channels are {list(IDEAL_COLUMNS)} "
        f"and their imu twins {list(IMU_COLUMNS)}"
    )


def channel_aliases(name: str) -> tuple[str, ...]:
    """Return every corpus spelling of the logical channel ``name`` denotes.

    Used by code that must find a channel in a table without knowing which observation
    mode produced it. The order is (logical, ``imu`` twin), so a caller that simply takes
    the first match present prefers the clean spelling.

    Args:
        name: Either spelling of a corpus motion channel, e.g. ``"roll"`` or ``"roll_imu"``.

    Returns:
        One name for ``heave_acc``, which has no noisy twin; two for every other channel.

    Raises:
        ValueError: If ``name`` is neither spelling of a corpus motion channel.
    """
    ideal = logical_channel(name)
    twin = IMU_BY_IDEAL.get(ideal)
    return (ideal,) if twin is None else (ideal, twin)
