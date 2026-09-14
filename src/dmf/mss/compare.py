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
