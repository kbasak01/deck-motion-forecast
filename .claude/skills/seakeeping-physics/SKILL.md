---
name: seakeeping-physics
description: Reference formulas and validation invariants for ocean-wave and vessel-motion simulation - JONSWAP spectrum, random-phase wave synthesis, spectral moments, encounter frequency, and second-order roll/pitch/heave response. Use whenever writing or reviewing code under src/dmf/sim/, or whenever a task mentions sea state, Hs, Tp, wave spectrum, RAO, encounter frequency, or the physical plausibility of simulated deck motion.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Seakeeping simulation reference

Angles are stored in **degrees** in Parquet; convert to radians inside physics functions only.
Frequencies are **rad/s** throughout, never Hz. `g = 9.80665`.

## JONSWAP spectrum (Hs-scaled, DNV-RP-C205 form)

```
wp    = 2*pi/Tp
sigma = 0.07 if w <= wp else 0.09
r     = exp(-(w - wp)**2 / (2 * sigma**2 * wp**2))
S(w)  = (5/16) * Hs**2 * wp**4 * w**-5 * exp(-1.25*(wp/w)**4) * (1 - 0.287*log(gamma)) * gamma**r
```

`gamma = 3.3` default. The `(1 - 0.287*ln gamma)` factor normalizes so `integral S dw = Hs**2/16`.

Spectral moments: `m_n = integral(w**n * S(w) dw)`.
Recovery checks: `Hs = 4*sqrt(m0)`, `Tz = 2*pi*sqrt(m0/m2)`, `Tp/Tz ~ 1.29-1.41` at gamma=3.3.

## Wave elevation synthesis

```
eta(t) = sum_i sqrt(2 * S(w_i) * dw_i) * cos(w_i*t + phi_i),   phi_i ~ U(0, 2*pi)
```

Use 200-400 components over `w in [0.2, 2.5] rad/s`.

**Jitter each frequency within its bin**: `w_i = w_lo_i + u_i*dw_i`, `u_i ~ U(0,1)`. A uniform
frequency grid makes the record repeat with period `2*pi/dw`. If that period is shorter than the
record, the signal is periodic, a forecaster memorizes it, and every downstream result is invalid.
This is the single most dangerous silent bug in the project.

Discard a spin-up transient (>= 120 s) before storing.

## Encounter frequency

```
w_e = w - (w**2 * U / g) * cos(beta)
```

`beta`: 180 deg head seas, 90 deg beam, 0 deg following. `U` in m/s.
`dw_e/dw` changes sign in following seas; either exclude that regime from the corpus or handle it
explicitly and document it. Never let it pass silently.

## DOF response

Second-order transfer function per DOF, evaluated at encounter frequency:

```
H(w_e) = K * F_exc(w) / (1 - (w_e/wn)**2 + 2j*zeta*(w_e/wn))
```

| DOF | Excitation | Heading factor | Tn (s) | zeta |
|---|---|---|---|---|
| Heave | elevation `a`, Smith factor `exp(-k*T_draft)` | weak | 7-10 | 0.25-0.45 |
| Pitch | wave slope `k*a` | `abs(cos(beta))` | 6-9 | 0.30-0.50 |
| Roll | wave slope `k*a` | `abs(sin(beta))` | 10-16 | **0.05-0.12** |

`k = w**2/g` (deep water). Apply a wavelength-vs-length rolloff so short waves do not drive full
response; document the form used.

Synthesize all DOFs from the **same phase set** `phi_i` as the elevation. This preserves the physical
phase relationships between roll, pitch and heave, which is what a multivariate forecaster exploits.

Lightly damped roll is narrowband and resonant; heave is broadband. Per-DOF forecast skill will
differ sharply and that difference is a reportable result, not a bug.

## Validation invariants (Gate 1)

1. `4*sqrt(m0)` recovers Hs within 2%.
2. `Tz` in the expected band for the requested Tp.
3. Welch PSD of a 3600 s record matches analytic `S(w)` within 15% over `[0.4, 1.5] rad/s`.
4. Elevation peak amplitudes pass a Rayleigh KS test, `p > 0.01`.
5. No autocorrelation spike at `2*pi/dw`.
6. Beam seas: roll spectrum peaks within 5% of `wn_roll`. Head seas: roll RMS drops >= 10x.
7. SS5 beam-seas roll RMS in single-digit degrees.
8. Higher forward speed in head seas shifts the response peak to higher `w_e`.

## Sea states

| SS | Hs (m) | Tp (s) |
|---|---|---|
| 3 | 1.0 | 7.5 |
| 4 | 1.9 | 8.8 |
| 5 | 3.3 | 9.7 |
| 6 | 5.0 | 12.4 |

## Model limitations to carry into the README

Linear seakeeping; no nonlinear roll damping, parametric resonance, or green water. Unidirectional
seas; no spreading, no swell/wind-sea bimodality. Constant heading and speed within a realization.
