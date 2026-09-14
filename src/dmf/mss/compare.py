"""Statistical comparison of MSS trajectories against the corpus.

Phase 8 asks for a comparison that is *statistical*, not point-by-point: the two generators
draw independent phase sets, so their records can never align sample-by-sample and a
pointwise error would be meaningless.

Carry-forward delta 6 sets the standard: "the response spectra overlay closely" is
unfalsifiable without knowing how much two realizations of *this* simulator differ from each
other. So every statistic here is reported as a corpus across-seed mean and standard
deviation, with the MSS value expressed in units of that spread (a z-score). A discrepancy
of 0.4 corpus standard deviations is agreement; one of 12 is not, and the reader should not
have to take the word "closely" on trust.

Everything compared here is simulation against simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.signal import welch

from dmf.typedefs import FloatArray

#: DOFs compared. The corpus forecasts rates too, but RMS and zero-crossing period on the
#: displacement channels are what the plan asks for.
COMPARE_DOFS: Final[tuple[str, ...]] = ("roll", "pitch", "heave")

#: Welch parameters, copied from `tests/test_spectra.py::test_welch_psd_matches_analytic`
#: so the PSD estimator is the one already validated against the analytic JONSWAP.
WELCH_NPERSEG: Final[int] = 2048
WELCH_WINDOW: Final[str] = "hann"

#: Channels that are structurally degenerate on the MSS side and carry no scorable signal.
#:
#: Head-seas roll vanishes by port/starboard symmetry, so MSS gives exactly 0.0000 deg RMS
#: and the corpus gives only the P1-D2 residual floor. A skill score against a zero-variance
#: target is undefined and its persistence denominator is ~0, so these cells produce
#: unbounded values that swamp any aggregate they are included in.
#:
#: P8-D1 pre-registered the exclusion for ``roll`` and missed ``roll_rate``, which is the
#: same artifact -- the derivative of a channel that is identically zero is identically
#: zero. The omission was caught by a sign-ablation summary that came back as 1.1e12 (see
#: P8-D11). Both are excluded here, in one place, rather than filtered ad hoc per analysis.
DEGENERATE_CELLS: Final[frozenset[tuple[str, float]]] = frozenset(
    {("roll", 180.0), ("roll_rate", 180.0)}
)


def drop_degenerate_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Remove the structurally degenerate (DOF, heading) rows from a results table.

    Args:
        table: Any frame with ``dof`` and ``heading_deg`` columns.

    Returns:
        The frame without the :data:`DEGENERATE_CELLS` rows.

    Raises:
        KeyError: If the expected columns are absent.
    """
    for column in ("dof", "heading_deg"):
        if column not in table.columns:
            raise KeyError(f"table has no {column!r} column")
    mask = pd.Series(True, index=table.index)
    for dof, heading in DEGENERATE_CELLS:
        mask &= ~((table["dof"] == dof) & (table["heading_deg"] == heading))
    return table[mask]


#: Band over which band-averaged PSD agreement is summarised, rad/s. Matches
#: `dmf.viz.spectra_plots.GATE1_PSD_BAND_RAD_S`.
PSD_BAND_RAD_S: Final[tuple[float, float]] = (0.4, 1.5)


@dataclass(frozen=True)
class ChannelStats:
    """Summary statistics for one channel of one record.

    Attributes:
        rms: Root mean square, in the channel's storage units (deg or m).
        tz_s: Mean upward zero-crossing period, seconds.
        sig_amp: Significant single amplitude, ``2*rms``, storage units.
    """

    rms: float
    tz_s: float
    sig_amp: float


def zero_crossing_period(signal: FloatArray, fs_hz: float) -> float:
    """Mean period between successive upward zero crossings, seconds.

    Args:
        signal: Time series; its mean is removed first.
        fs_hz: Sample rate, Hz.

    Returns:
        Mean up-crossing period, or ``nan`` if there are fewer than two crossings.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    neg = np.signbit(x)
    idx = np.where((~neg[1:]) & neg[:-1])[0]
    if idx.size < 2:
        return float("nan")
    return float(np.mean(np.diff(idx)) / fs_hz)


def channel_stats(signal: FloatArray, fs_hz: float) -> ChannelStats:
    """Compute RMS, significant single amplitude and zero-crossing period.

    Args:
        signal: Time series in storage units.
        fs_hz: Sample rate, Hz.

    Returns:
        The :class:`ChannelStats`.
    """
    x = np.asarray(signal, dtype=np.float64)
    rms = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
    return ChannelStats(rms=rms, tz_s=zero_crossing_period(x, fs_hz), sig_amp=2.0 * rms)


def response_psd(signal: FloatArray, fs_hz: float) -> tuple[FloatArray, FloatArray]:
    """One-sided PSD against angular frequency.

    `scipy.signal.welch` returns a density per hertz against frequency in hertz. The
    project works in rad/s throughout, so both axes are converted:
    ``w = 2*pi*f`` and ``S(w) = S_f(f) / (2*pi)``. Getting only one of the two right is a
    factor-of-39 error that still looks like a plausible spectrum.

    Args:
        signal: Time series in storage units.
        fs_hz: Sample rate, Hz.

    Returns:
        ``(w_rad_s, psd)`` with psd in units^2 per rad/s.
    """
    x = np.asarray(signal, dtype=np.float64)
    f, pxx = welch(
        x,
        fs=fs_hz,
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NPERSEG // 2,
        window=WELCH_WINDOW,
        detrend="constant",
    )
    return 2.0 * np.pi * f, pxx / (2.0 * np.pi)


def summarize_records(frames: list[pd.DataFrame], fs_hz: float, source: str) -> pd.DataFrame:
    """Summarise a set of records, one row per (record, dof).

    Args:
        frames: Records sharing a schema, each one realization.
        fs_hz: Sample rate, Hz.
        source: Label recorded in the ``source`` column.

    Returns:
        Long-form table with columns ``source, record, dof, rms, sig_amp, tz_s``.
    """
    rows: list[dict[str, object]] = []
    for i, frame in enumerate(frames):
        for dof in COMPARE_DOFS:
            st = channel_stats(frame[dof].to_numpy(), fs_hz)
            rows.append(
                {
                    "source": source,
                    "record": i,
                    "dof": dof,
                    "rms": st.rms,
                    "sig_amp": st.sig_amp,
                    "tz_s": st.tz_s,
                }
            )
    return pd.DataFrame(rows)


def marginalize_over_speed(stats: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-(heading, speed) rows to one row per (heading, DOF), correctly.

    **Ratios and z-scores must not be averaged.** A ratio of means is not the mean of
    ratios, and a z-score is a standardised difference whose denominator differs per cell.
    Averaging them produces a number that describes nothing: for roll at 135 degrees the
    per-speed ratios are {0.648, 1.117, 1.627} and the per-speed z's are {-8.11, +2.51,
    +14.89}, so the mean-of-ratios reads 1.13 and the mean-of-z reads +3.10, while the
    honest summary is a ratio of means of 0.986 at z = -0.32. This function recomputes from
    the means instead, and reports the per-speed spread so a large speed dependence cannot
    hide inside the marginal.

    Args:
        stats: Output of :func:`compare_against_corpus_spread`, with ``heading_deg`` and
            ``speed_kn`` columns.

    Returns:
        One row per (heading, metric, DOF), with ``ratio_mss_over_corpus`` computed as a
        ratio of means, ``z_in_corpus_sd`` from those means, and ``ratio_spread_over_speed``
        naming the range of the per-speed ratios.
    """
    rows: list[dict[str, object]] = []
    for (heading, metric, dof), group in stats.groupby(["heading_deg", "metric", "dof"]):
        corpus_mean = float(group["corpus_mean"].mean())
        corpus_sd = float(group["corpus_sd"].mean())
        mss_mean = float(group["mss_mean"].mean())
        per_speed = group["ratio_mss_over_corpus"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "heading_deg": heading,
                "metric": metric,
                "dof": dof,
                "corpus_mean": corpus_mean,
                "corpus_sd": corpus_sd,
                "mss_mean": mss_mean,
                "ratio_mss_over_corpus": mss_mean / corpus_mean if corpus_mean else float("nan"),
                "z_in_corpus_sd": (
                    (mss_mean - corpus_mean) / corpus_sd if corpus_sd else float("nan")
                ),
                "ratio_min_over_speed": float(np.nanmin(per_speed)),
                "ratio_max_over_speed": float(np.nanmax(per_speed)),
                "ratio_spread_over_speed": float(np.nanmax(per_speed) - np.nanmin(per_speed)),
                "n_speeds": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def compare_against_corpus_spread(
    corpus: pd.DataFrame, mss: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """Express the MSS value in units of the corpus's own across-seed spread.

    This is delta 6 made arithmetic. ``z`` is the MSS mean minus the corpus mean, divided by
    the corpus across-seed standard deviation -- the natural yardstick, because it is how
    far apart two realizations of the corpus generator routinely land.

    Args:
        corpus: Long-form summary of corpus records.
        mss: Long-form summary of MSS records.
        metric: Column to compare, e.g. ``"rms"``.

    Returns:
        One row per DOF with both means, the corpus spread, the ratio and the z-score.
    """
    rows: list[dict[str, object]] = []
    for dof in COMPARE_DOFS:
        c = corpus.loc[corpus["dof"] == dof, metric].to_numpy(dtype=np.float64)
        m = mss.loc[mss["dof"] == dof, metric].to_numpy(dtype=np.float64)
        c_mean, c_sd = float(np.mean(c)), float(np.std(c, ddof=1))
        m_mean, m_sd = float(np.mean(m)), float(np.std(m, ddof=1))
        rows.append(
            {
                "dof": dof,
                "metric": metric,
                "corpus_mean": c_mean,
                "corpus_sd": c_sd,
                "corpus_n": int(c.size),
                "mss_mean": m_mean,
                "mss_sd": m_sd,
                "mss_n": int(m.size),
                "ratio_mss_over_corpus": m_mean / c_mean if c_mean else float("nan"),
                "z_in_corpus_sd": (m_mean - c_mean) / c_sd if c_sd else float("nan"),
            }
        )
    return pd.DataFrame(rows)
