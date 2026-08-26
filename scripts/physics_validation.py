#!/usr/bin/env python3
"""Regenerate the Gate 1 physics-validation figures under ``results/``.

Thin CLI wrapper: every figure is drawn by :mod:`dmf.viz.spectra_plots`, and every number
it plots comes from :mod:`dmf.sim`. Backs ``results/physics_validation.md``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy import signal, stats

from dmf.sim.encounter import encounter_frequency, knots_to_m_s
from dmf.sim.response import dof_transfer
from dmf.sim.spectra import jonswap, sample_components, synthesis_period_s, synthesize_elevation
from dmf.sim.vessel import load_vessel
from dmf.viz.spectra_plots import plot_autocorrelation, plot_response_spectra, plot_spectrum_overlay

FS_HZ = 10.0
RECORD_S = 3600.0
N_COMPONENTS = 299
W_MIN, W_MAX = 0.2, 2.5
SS5 = (3.3, 9.7, 3.3)


def _t() -> np.ndarray:
    return np.arange(0.0, RECORD_S, 1.0 / FS_HZ)


def _components(seed: int, *, jitter: bool = True):
    return sample_components(
        *SS5, N_COMPONENTS, W_MIN, W_MAX, np.random.default_rng(seed), jitter=jitter
    )


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    y = x - x.mean()
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    r = np.fft.irfft(f * np.conj(f), 2 * n)[: max_lag + 1]
    return r / (np.arange(n, n - max_lag - 1, -1) * y.var())


def fig_spectrum(out: Path) -> None:
    eta = synthesize_elevation(_components(0), _t())
    f, pf = signal.welch(eta, fs=FS_HZ, nperseg=2048)
    w = 2 * np.pi * f
    wa = np.linspace(0.05, 2.5, 4000)
    plot_spectrum_overlay(wa, jonswap(wa, *SS5), w, pf / (2 * np.pi), SS5[0], SS5[1]).savefig(
        out / "gate1_spectrum_overlay.png"
    )


def fig_acf(out: Path) -> None:
    """The anti-periodicity figure, with the un-jittered control overlaid."""
    t, max_lag = _t(), int(1500 * FS_HZ)
    period = synthesis_period_s(_components(0))
    lag = np.arange(max_lag + 1) / FS_HZ
    rj = _acf(synthesize_elevation(_components(0), t), max_lag)
    ru = _acf(synthesize_elevation(_components(0, jitter=False), t), max_lag)

    fig = plot_autocorrelation(lag, rj, period, acf_unjittered=ru)
    k = int(round(period * FS_HZ))
    print(f"    [fig-acf] T={period:.2f}s  r(T) jittered={rj[k]:+.4f}  un-jittered={ru[k]:+.4f}")
    fig.savefig(out / "gate1_autocorrelation.png")


def fig_response(out: Path) -> None:
    v = load_vessel(Path("configs/sim/vessels/frigate.yaml"))
    w = np.linspace(W_MIN, W_MAX, 6000)
    s = jonswap(w, *SS5)
    we = encounter_frequency(w, 0.0, 90.0)
    spec, wn = {}, {}
    for dof in ("roll", "pitch", "heave"):
        h = dof_transfer(dof, w, we, v, 90.0)
        spec[f"{dof} (beam)"] = np.abs(h) ** 2 * s
        wn[f"{dof} (beam)"] = 2 * np.pi / getattr(v, dof).tn_s
    hh = dof_transfer("roll", w, encounter_frequency(w, 0.0, 180.0), v, 180.0)
    spec["roll (head)"] = np.abs(hh) ** 2 * s
    wn["roll (head)"] = 2 * np.pi / v.roll.tn_s
    plot_response_spectra(we, spec, wn).savefig(out / "gate1_response_spectra.png")


def fig_speed_shift(out: Path) -> None:
    v = load_vessel(Path("configs/sim/vessels/frigate.yaml"))
    w = np.linspace(W_MIN, W_MAX, 6000)
    s = jonswap(w, *SS5)
    fig = Figure(figsize=(7.0, 4.2), dpi=140)
    ax = fig.subplots()
    for kn, c in zip((0.0, 6.0, 12.0), ("#14213d", "#2a7f62", "#c1442a"), strict=True):
        we = encounter_frequency(w, knots_to_m_s(kn), 180.0)
        p = np.abs(dof_transfer("pitch", w, we, v, 180.0)) ** 2 * s
        pk = we[int(np.argmax(p))]
        ax.plot(we, p, color=c, lw=1.5, label=f"{kn:.0f} kn — peak {pk:.4f} rad/s")
        ax.axvline(pk, color=c, ls=":", lw=1.0)
    ax.set_xlim(0.2, 2.0)
    ax.set_xlabel("encounter frequency $\\omega_e$ (rad/s)")
    ax.set_ylabel("pitch response density (deg$^2$ s/rad)")
    ax.set_title("Head seas: forward speed shifts the pitch response peak up")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "gate1_speed_shift.png")


def fig_rayleigh(out: Path) -> None:
    eta = synthesize_elevation(_components(0), _t())
    up = np.where((eta[:-1] <= 0) & (eta[1:] > 0))[0]
    crests = np.array([eta[a:b].max() for a, b in zip(up[:-1], up[1:], strict=True)])
    sc = float(np.sqrt(eta.var()))
    d, p = stats.kstest(crests, "rayleigh", args=(0.0, sc))
    q = stats.rayleigh.ppf((np.arange(len(crests)) + 0.5) / len(crests), scale=sc)
    fig = Figure(figsize=(4.8, 4.8), dpi=140)
    ax = fig.subplots()
    lim = max(q.max(), crests.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#c1442a", lw=1.2, ls="--", label="Rayleigh")
    ax.plot(q, np.sort(crests), ".", ms=3.5, color="#14213d", label=f"{len(crests)} crests")
    ax.set_xlim(0, lim), ax.set_ylim(0, lim)
    ax.set_xlabel("Rayleigh quantile (m)"), ax.set_ylabel("observed crest height (m)")
    ax.set_title(f"Crest heights vs Rayleigh\nKS D = {d:.4f}, p = {p:.4f}")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "gate1_rayleigh_qq.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate Gate 1 physics-validation figures.")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_spectrum, fig_acf, fig_response, fig_speed_shift, fig_rayleigh):
        fn(args.out)
        print(f"  {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
