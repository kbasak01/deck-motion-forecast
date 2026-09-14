"""Figures for the Phase 8 MSS cross-validation.

These back `docs/mss_crossvalidation.md`. As everywhere in :mod:`dmf.viz`, no analysis
happens here: every number plotted was computed in `dmf.eval` or `dmf.mss.compare` and
written to a CSV first, so each figure is traceable to a committed table.

Both series in every figure are simulations. "Corpus" is this project's reduced-order
generator; "MSS" is ShipX strip theory. Neither is a measurement of a real ship.
"""

from collections.abc import Mapping

from matplotlib.figure import Figure

from dmf.typedefs import FloatArray

#: House palette, as used across `dmf.viz`: navy is the reference series, rust the
#: comparison series, green a third.
CORPUS_COLOR: str = "#14213d"
MSS_COLOR: str = "#c1442a"
ACCENT_COLOR: str = "#2a7f62"

__all__ = ["plot_response_spectra_overlay", "plot_skill_vs_horizon"]


def plot_response_spectra_overlay(
    w_rad_s: FloatArray,
    corpus_psd: dict[str, FloatArray],
    mss_psd: dict[str, FloatArray],
    dofs: tuple[str, ...],
    units: dict[str, str],
    title: str,
    corpus_band: dict[str, tuple[FloatArray, FloatArray]] | None = None,
) -> Figure:
    """Overlay corpus and MSS response spectra, one panel per DOF.

    The corpus across-seed envelope is drawn as a shaded band where supplied, because a
    two-line overlay without it cannot distinguish "these agree" from "these differ by less
    than this generator's own seed-to-seed scatter" -- the point carry-forward delta 6
    makes.

    Args:
        w_rad_s: Frequency axis shared by every spectrum, radians per second.
        corpus_psd: Mean corpus PSD per DOF, in (storage unit)^2 per rad/s.
        mss_psd: Mean MSS PSD per DOF, same units.
        dofs: DOF names, one panel each, in order.
        units: Storage unit per DOF, for the axis label (e.g. ``{"roll": "deg"}``).
        title: Figure title.
        corpus_band: Optional per-DOF ``(lower, upper)`` across-seed envelope.

    Returns:
        The figure.
    """
    fig = Figure(figsize=(7.4, 3.2 * len(dofs)), dpi=140)
    axes = fig.subplots(len(dofs), 1, squeeze=False)[:, 0]
    for ax, dof in zip(axes, dofs, strict=True):
        if corpus_band is not None and dof in corpus_band:
            lo, hi = corpus_band[dof]
            ax.fill_between(
                w_rad_s,
                lo,
                hi,
                color=CORPUS_COLOR,
                alpha=0.18,
                lw=0,
                label="corpus across-seed range",
            )
        ax.plot(w_rad_s, corpus_psd[dof], color=CORPUS_COLOR, lw=1.8, label="corpus (this project)")
        ax.plot(w_rad_s, mss_psd[dof], color=MSS_COLOR, lw=1.4, label="MSS (ShipX strip theory)")
        ax.set_xlim(0.0, 2.0)
        ax.set_yscale("log")
        ax.set_ylabel(f"$S(\\omega)$ ({units[dof]}$^2$ s/rad)")
        ax.set_title(dof, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("encounter angular frequency $\\omega_e$ (rad/s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_skill_vs_horizon(
    horizons_s: FloatArray,
    series: Mapping[str, Mapping[str, Mapping[str, tuple[FloatArray, FloatArray]]]],
    dofs: tuple[str, ...],
    title: str,
    model_colors: dict[str, str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Skill versus forecast horizon, corpus against MSS, one panel per DOF.

    This is the figure the phase exists to produce. Each model contributes two lines: solid
    for the committed corpus `unseen_vessel` rows and dashed for the MSS trajectories. The
    vertical gap between a model's two lines *is* the answer -- it is how much of that
    model's skill came from wave-response structure that survives an independent
    hydrodynamic computation, and how much came from our own generator.

    Persistence is at zero by construction on both sides, since it is the skill denominator
    and is recomputed on each dataset separately (delta 4).

    Args:
        horizons_s: Forecast horizons, seconds.
        series: ``{model: {"corpus": (mean, sd), "mss": (mean, sd)}}``, each array over
            ``horizons_s``. A model may supply either key or both.
        dofs: DOF names, one panel each.
        title: Figure title.
        model_colors: Optional colour per model.
        ylim: Optional y-axis clip. Skill is unbounded below, so one collapsing series can
            flatten every other curve; series leaving the axis are annotated with the value
            they reach rather than silently cropped.

    Returns:
        The figure.

    Raises:
        KeyError: If ``series`` lacks an entry for a DOF being plotted.
    """
    default_cycle = [CORPUS_COLOR, MSS_COLOR, ACCENT_COLOR, "#8a6d3b", "#5b2a86", "#777777"]
    fig = Figure(figsize=(7.4, 3.2 * len(dofs)), dpi=140)
    axes = fig.subplots(len(dofs), 1, squeeze=False)[:, 0]

    for ax, dof in zip(axes, dofs, strict=True):
        panel = series[dof]
        colors = model_colors or {
            m: default_cycle[i % len(default_cycle)] for i, m in enumerate(sorted(panel))
        }
        for model in sorted(panel):
            for source, style in (("corpus", "-"), ("mss", "--")):
                entry = panel[model].get(source)
                if entry is None:
                    continue
                mean, sd = entry
                color = colors[model]
                ax.plot(
                    horizons_s,
                    mean,
                    style,
                    color=color,
                    lw=1.6,
                    marker="o" if source == "corpus" else "s",
                    ms=3.0,
                    label=f"{model} ({'corpus' if source == 'corpus' else 'MSS'})",
                )
                ax.fill_between(horizons_s, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        # Clip the axis to the region that carries information. Skill is unbounded below --
        # `ar40` reaches -19.9 in heave at 10 s -- and letting one series set the scale
        # flattens every other curve into the zero line. Series that leave the axis are
        # annotated with where they actually go, so nothing is hidden by the clip.
        if ylim is not None:
            ax.set_ylim(*ylim)
            for model in sorted(panel):
                for source in ("corpus", "mss"):
                    entry = panel[model].get(source)
                    if entry is None:
                        continue
                    mean = entry[0]
                    below = mean < ylim[0]
                    if below.any():
                        worst = float(mean[below].min())
                        x = float(horizons_s[int(mean.argmin())])
                        ax.annotate(
                            f"{model} ({source}) -> {worst:.1f}",
                            xy=(x, ylim[0]),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            fontsize=6.5,
                            color=(model_colors or {}).get(model, "0.3"),
                        )
        ax.set_ylabel("skill vs persistence")
        ax.set_title(dof, fontsize=9, loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1].set_xlabel("forecast horizon (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
