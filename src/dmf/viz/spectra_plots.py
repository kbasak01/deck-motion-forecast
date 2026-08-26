"""Wave-spectrum and physics-validation figures.

These figures back ``results/physics_validation.md``, the Gate 1 artifact.
"""

import numpy as np
from matplotlib.figure import Figure

from dmf.typedefs import FloatArray

#: Shared band over which Gate 1 criterion 3 requires 15 percent agreement, radians per
#: second. Drawn on the overlay figure so the tolerance region is visible, not implied.
GATE1_PSD_BAND_RAD_S: tuple[float, float] = (0.4, 1.5)

__all__ = ["plot_autocorrelation", "plot_response_spectra", "plot_spectrum_overlay"]


def plot_spectrum_overlay(
    w_rad_s: FloatArray,
    s_analytic: FloatArray,
    w_welch_rad_s: FloatArray,
    s_welch: FloatArray,
    hs_m: float,
    tp_s: float,
) -> Figure:
    """Overlay the analytic JONSWAP spectrum with the Welch PSD of a synthesised record.

    Gate 1 requires agreement within 15 percent across [0.4, 1.5] rad/s.

    Args:
        w_rad_s: Frequencies of the analytic spectrum, radians per second.
        s_analytic: Analytic spectral density, m^2 s/rad.
        w_welch_rad_s: Frequencies of the Welch estimate, radians per second.
        s_welch: Welch spectral density estimate, m^2 s/rad.
        hs_m: Requested significant wave height, metres, for the title.
        tp_s: Requested peak period, seconds, for the title.

    Returns:
        The figure.
    """
    fig = Figure(figsize=(7.0, 4.2), dpi=140)
    ax = fig.subplots()
    lo, hi = GATE1_PSD_BAND_RAD_S
    ax.axvspan(lo, hi, color="0.90", zorder=0, label=f"Gate 1 band [{lo}, {hi}] rad/s")
    ax.plot(
        w_welch_rad_s, s_welch, color="#c1442a", lw=1.0, alpha=0.85, label="Welch PSD of record"
    )
    ax.plot(w_rad_s, s_analytic, color="#14213d", lw=2.0, label="Analytic JONSWAP")
    ax.set_xlim(0.0, 2.5)
    ax.set_xlabel("angular frequency $\\omega$ (rad/s)")
    ax.set_ylabel("$S(\\omega)$ (m$^2$ s/rad)")
    ax.set_title(f"Wave spectrum recovery — $H_s$ = {hs_m:.1f} m, $T_p$ = {tp_s:.1f} s")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    return fig


def plot_autocorrelation(
    lag_s: FloatArray,
    acf: FloatArray,
    synthesis_period_s: float,
    acf_unjittered: FloatArray | None = None,
) -> Figure:
    """Plot the autocorrelation of an elevation record, marking the synthesis period.

    This is the anti-periodicity figure. A spike at ``2*pi/dw`` means the frequency grid
    was not jittered, the record repeats, and the forecasting task has been reduced to
    memorising a loop. The marker is drawn whether or not a spike is present, so that its
    absence is visible evidence rather than an unstated assumption.

    Args:
        lag_s: Lag axis, seconds.
        acf: Autocorrelation at each lag, dimensionless.
        synthesis_period_s: The period ``2*pi/dw`` at which a uniform grid would repeat,
            seconds.
        acf_unjittered: Optional autocorrelation of the same synthesis with the frequency
            jitter disabled, dimensionless. When given it is overlaid as the control, and
            a second panel zooms on the synthesis period. Plotting the control is what
            turns this figure from an assertion that no spike is present into evidence
            that a spike would have been visible had one existed.

    Returns:
        The figure.
    """
    k = int(round(synthesis_period_s * (len(lag_s) - 1) / float(np.max(lag_s))))
    if acf_unjittered is None:
        panels = [(0.0, float(np.max(lag_s)))]
    else:
        panels = [
            (0.0, float(np.max(lag_s))),
            (synthesis_period_s - 40.0, synthesis_period_s + 40.0),
        ]
    fig = Figure(figsize=(7.4, 3.2 * len(panels)), dpi=140)
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, (lo, hi) in zip(axes, panels, strict=True):
        m = (lag_s >= lo) & (lag_s <= hi)
        ax.axhline(0.0, color="0.6", lw=0.6)
        if acf_unjittered is not None:
            ax.plot(
                lag_s[m],
                acf_unjittered[m],
                color="#e08a1e",
                lw=1.0,
                label="uniform grid (no jitter)",
            )
        ax.plot(lag_s[m], acf[m], color="#14213d", lw=1.0, label="jittered grid (corpus setting)")
        ax.axvline(
            synthesis_period_s,
            color="#c1442a",
            ls="--",
            lw=1.4,
            label=f"synthesis period $2\\pi/d\\omega$ = {synthesis_period_s:.1f} s",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("normalised ACF")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("lag (s)")
    if acf_unjittered is None:
        axes[0].set_title("Anti-periodicity check")
    else:
        axes[0].set_title(
            "Anti-periodicity: the synthesis-period spike exists, and jitter removes it\n"
            f"r(T) = {acf[k]:+.4f} jittered  vs  {acf_unjittered[k]:+.4f} un-jittered"
        )
        axes[1].set_title("zoom on the synthesis period", fontsize=9)
    fig.tight_layout()
    return fig


def plot_response_spectra(
    w_e_rad_s: FloatArray,
    spectra: dict[str, FloatArray],
    wn_rad_s: dict[str, float],
) -> Figure:
    """Plot per-DOF response spectra with their natural frequencies marked.

    Args:
        w_e_rad_s: Encounter frequency axis, radians per second.
        spectra: Response spectral density per DOF name. Units are deg^2 s/rad for roll
            and pitch, m^2 s/rad for heave.
        wn_rad_s: Undamped natural angular frequency per DOF name, radians per second.
            Marked as vertical lines: Gate 1 requires the beam-seas roll spectrum to peak
            within 5 percent of ``wn_roll``.

    Returns:
        The figure.

    Raises:
        ValueError: If ``spectra`` and ``wn_rad_s`` have different key sets.
    """
    if set(spectra) != set(wn_rad_s):
        raise ValueError(f"spectra keys {sorted(spectra)} != wn keys {sorted(wn_rad_s)}")
    names = list(spectra)
    fig = Figure(figsize=(7.0, 2.6 * len(names)), dpi=140)
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    for ax, name in zip(axes, names, strict=True):
        ax.plot(w_e_rad_s, spectra[name], color="#14213d", lw=1.4)
        ax.axvline(
            wn_rad_s[name],
            color="#c1442a",
            ls="--",
            lw=1.2,
            label=f"$\\omega_n$ = {wn_rad_s[name]:.4f} rad/s",
        )
        ax.set_ylabel(name)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("encounter frequency $\\omega_e$ (rad/s)")
    axes[0].set_title("Response spectra with natural frequencies marked")
    fig.tight_layout()
    return fig
