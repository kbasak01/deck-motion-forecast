# Protocol — decisions, deviations, and threshold changes

This file is the audit trail required by `CLAUDE.md` §Gates: any change to a gate threshold,
and any physics or evaluation parameter chosen in a way that affects whether a gate passes,
is recorded here with its reason. Split and metric definitions are added in Phase 2.

All results in this project are from **simulated** vessel motion. No real deck data is used.

---

## Phase 1 — wave and vessel-response simulation

### Gate 1 thresholds

**No Gate 1 threshold was changed.** All eight criteria are asserted at the values stated in
`docs/IMPLEMENTATION_PLAN.md` §Phase 1.

### P1-D1 — Following-seas encounter frequency: kept, not excluded

`dw_e/dw` changes sign at `w_crit = g / (2*U*cos(beta))`. Inside the `[0.2, 2.5] rad/s`
synthesis band this occurs for **two** corpus cells:

| Heading | Speed | `w_crit` |
|---|---|---|
| 45 deg | 6 kn | 2.247 rad/s |
| 45 deg | 12 kn | 1.124 rad/s |

The plan permits either excluding these or handling them explicitly. **Decision: keep them.**
Time-domain superposition remains well-defined where `w_e` is non-monotonic or negative —
each wave component simply carries its own `w_e`. The second-order transfer function is
evaluated at `|w_e|` so damping stays dissipative, while the **signed** `w_e` is retained in
the cosine argument.

Excluding them would have made the heading axis unbalanced across speeds and complicated the
`unseen_heading` regime. `dmf.sim.encounter.is_encounter_monotonic` reports the condition, and
`tests/test_response.py::test_encounter_monotonicity_over_corpus_grid` asserts the pattern
across the full grid so it can never regress silently.

Note: `is_encounter_monotonic` returns False when the derivative touches exactly zero at a grid
point. That is still weakly monotonic; the conservative answer was chosen deliberately.

### P1-D2 — Roll and pitch heading-factor residual floors

| DOF | Factor | Zero at |
|---|---|---|
| Roll | `sqrt(sin^2(beta) + 0.05^2)` | would be 180 deg (head) |
| Pitch | `sqrt(cos^2(beta) + 0.05^2)` | would be 90 deg (beam) |

Pure `|sin(beta)|` / `|cos(beta)|` make roll identically zero in head seas and pitch
identically zero in beam seas. Each accounts for 25% of the corpus, and a constant-zero target
makes skill-score-vs-persistence `0/0` rather than merely poor. The pitch case is worse: it
lands exactly on the `unseen_heading` regime, whose entire test set **is** beam seas.

`eps = 0.05` puts the floored DOF ~26 dB below its maximum. This also makes Gate 1 criterion 6
("head-seas roll RMS drops by at least an order of magnitude") a real test with a measured
20.02x ratio, rather than one satisfied by a divide-by-zero.

**These floors are an engineering stand-in for hull asymmetry and short-crested residual
excitation, not a derived quantity.** They belong in the README limitations section.

### P1-D3 — Frigate roll damping changed from zeta = 0.08 to 0.06

**This is a physics-parameter change made in the knowledge that it affects whether Gate 1
criterion 6a passes. It is recorded here for that reason.**

Criterion 6a requires the beam-seas roll response spectrum to peak within 5% of `wn_roll`
(0.5236 rad/s at `Tn = 12 s`). At SS5 the frigate's roll response spectrum is **bimodal**: a
resonance peak near `wn_roll` and a wave-driven peak near the spectral peak `wp = 0.648 rad/s`.
Analytic peak separation, verified independently of the implementing agent:

| roll zeta | resonance peak | 2nd peak | dominance ratio | SS5 beam roll RMS |
|---|---|---|---|---|
| 0.05 | 0.5297 (+1.17%) | — | unimodal | 3.76 deg |
| **0.06** | **0.5325 (+1.70%)** | 0.6148 | **1.69x** | **3.50 deg** |
| 0.07 | 0.5362 (+2.40%) | 0.6189 | 1.33x | 3.29 deg |
| 0.08 | 0.5408 (+3.28%) | 0.6216 | 1.09x | 3.12 deg |
| 0.09 | 0.6237 (**+19.11%**) | 0.5474 | wave peak wins | 2.97 deg |

At `zeta = 0.08` the two peaks differ by only 9%, so the Welch argmax flips between them from
seed group to seed group (measured spread over eight 4-seed groups: +2.5% to +17.5%). That is
peak degeneracy in the physics, not estimator variance — the Hann-window smoothing bias was
measured at +0.05% and is negligible.

`zeta = 0.06` is mid-band for an unstabilised frigate (the plan's stated range is 0.05-0.12)
and leaves the resonance peak 69% above the secondary one. The threshold was not touched.

**Consequence to state in the README:** the corpus is, by construction, one in which beam-seas
roll is resonance-dominated at SS5. This hull sits close to a bimodal regime and would become
wave-peak-dominated at `zeta >= 0.09`.

### P1-D4 — Test-design choices that make statistical invariants meaningful

None of these change a threshold; each makes an existing threshold a real test rather than a
coin flip.

- **Criterion 2 (`Tz/Tp` in 0.71-0.78).** `m2` converges slowly (`w^2 S ~ w^-3`), so the
  integration limit decides the outcome: `w_max = 2.5` gives 0.803 (a false fail), `w_max = 6`
  gives 0.782, `w_max = 30` gives 0.7776. Moments are integrated to **30 rad/s**. The measured
  margin against the 0.78 ceiling is only ~0.3%. **If this test ever fails, the fix is the
  integration band, not the threshold.**
- **Criterion 3 (Welch PSD within 15%).** A single 3600 s record at `nperseg = 2048` has ~17%
  per-bin standard deviation, so a per-bin 15% assertion would flake roughly half the time.
  The test averages PSD over **8 independent seeds** (~280 segments) and band-averages into
  **0.1 rad/s** bands. Narrower 0.05 rad/s bands were tried first and rejected: they are finer
  than the Welch resolution, so bands would hold 0-1 points. `nperseg = 1024` was also rejected
  — it fails for *bias*, not variance, smoothing the low-frequency flank and skewing the
  [0.40, 0.50] rad/s band by +22% at SS5.
- **Criterion 4 (Rayleigh KS, `p > 0.01`).** Uses **zero-crossing crest heights**, not all local
  maxima — all-local-maxima follows a Rice distribution for a broadband process and would be
  wrong by construction. The Rayleigh scale is **fixed** at `sqrt(m0)` from the record variance,
  not fitted, so the p-value stays honest. This is the thinnest-margin invariant: over 20 extra
  seeds the minimum p was 0.0114 against the 0.01 threshold.
- **Criterion 5 (anti-periodicity).** Uses `n_components = 299`, not 300, so that
  `w_min/dw = 26.000` exactly and the un-jittered frequency comb is exact. With 300 components
  the un-jittered record is not exactly periodic (it repeats as a rigid common phase rotation,
  measuring `r(T) = -0.68`) and the converse assertion would fail for the wrong reason.
  Consequently **`jitter=False` is defined as placing `w_i` at the bin lower edge**, not the bin
  centre. The jittered ACF threshold is 0.20 rather than 0.10: at 3600 s the large-lag ACF has
  sampling sigma ~0.06, so 0.10 is 1.7 sigma and flakes ~1 seed in 10. The discriminating
  assertion is the peak-to-background ratio (< 1.5 jittered; measured 0.42-0.83, against 5.43
  un-jittered). The ACF uses the unbiased `1/(N-lag)` normalization; the biased estimator scores
  a perfectly periodic signal at 0.77 and would gut the converse test.
- **Criterion 6a is asserted at SS5 and SS6 only.** The roll response spectrum peaks at
  `wn_roll` only when the wave spectrum carries energy there. Analytic peak error by sea state:
  SS6 -0.85%, SS5 +3.28%, SS4 +4.72%, **SS3 +55.9%** (at `zeta = 0.08`). The SS3 figure is
  correct physics — the response peak is dragged to the wave peak at 0.816 rad/s — so criterion
  6 is only meaningful for the higher sea states, and the SS3 behaviour is reported rather than
  suppressed.
- **Criterion 8 uses pitch, not roll.** Lightly damped roll stays resonance-locked at `wn`
  regardless of forward speed, so roll cannot detect an encounter-frequency shift.

### P1-D5 — Phase 0 simulator API replaced

The Phase 0 skeleton stubbed `src/dmf/sim/` with a flat API (`Vessel(tn_roll_s, k_roll, ...)`,
`jittered_frequency_grid`, `component_amplitudes`, `differentiate_series`). Phase 1 replaced it
with `WaveComponents` and `MotionRecord` dataclasses and a per-DOF `DofParams` grouping.

Reason: the bundled dataclasses remove four-parallel-array call signatures, and the old
`differentiate_series` finite-difference helper contradicts the decision to differentiate
analytically in the frequency domain. Rates and acceleration are exact frequency-domain
derivatives, cross-checked against a central difference at 0.07-0.08% relative error.

### P1-D6 — IMU observation model: parameters not fixed by the plan, and the phase lead

The plan (§1.6) specifies "white noise (0.02 deg) + slow bias random walk" on attitude and heave
"reconstructed by double-integrating vertical acceleration through a 2nd-order high-pass at
0.03 Hz". Three things it does not specify had to be chosen, and one consequence has to be
recorded because it can invalidate an evaluation if ignored.

**Chosen parameters** (module constants in `dmf/sim/imu.py`, not YAML, because the corpus writes
both observation modes into every file and there is no per-experiment variation to configure):

| Quantity | Value | Basis |
|---|---|---|
| Attitude bias random walk | 0.002 deg/sqrt(s) | ~0.05 deg of wander over a 600 s record: same order as the white noise, but concentrated far below the wave band, so it is a distinct error source rather than more white noise. |
| Gyro rate white noise | 0.02 deg/s | Not given in the plan. Set equal in magnitude to the attitude noise figure; a real MEMS gyro's angle random walk is smaller, so the rate channels are treated slightly pessimistically. |
| Accelerometer noise | none | The plan puts noise on attitude only. After two integrations an accelerometer noise floor shows up as exactly the low-frequency drift the high-pass already exists to remove, so adding one would double-count. |

**Two high-pass stages, not one.** The `heave_from_vertical_acc` docstring specifies "a high-pass
applied at each integration stage", and `highpass_biquad` is second-order, so the reconstruction is
`HP(integral(HP(integral(acc))))` — two second-order stages. The alternative reading of the plan
(one second-order high-pass across the whole chain) would halve the phase distortion below. The
stricter reading was implemented, per the instruction not to tune the distortion away.

**Consequence: `heave_imu` leads `heave`.** A causal high-pass has phase *lead*, and two stages
double it:

| f (Hz) | combined magnitude | combined phase | equivalent lead |
|---|---|---|---|
| 0.050 | 0.885 | +106 deg | +5.9 s |
| 0.103 (SS5 peak) | 0.993 | +48 deg | +1.3 s |
| 0.200 | 1.000 | +24 deg | +0.3 s |

Measured on a stored SS5 beam-seas realization, the error power `var(heave_imu - heave)/var(heave)`
is 0.70, and it is almost entirely phase: the two RMS values agree to 1%. The relative error per
band falls monotonically with frequency (1.68 below 0.06 Hz, 0.65 at the spectral peak, 0.29 above
0.15 Hz), which is the invariant `tests/test_generate.py` asserts.

**The trap this creates — and what it is not.** This is *not* information leakage. The filter is
causal, and that was verified directly rather than argued: zeroing every acceleration sample after
index 4000 leaves the reconstructed heave before index 4000 bitwise unchanged
(`max|diff| = 0.000e+00`). `heave_imu(t)` is a function of `heave_acc(<= t)` alone, so no future
information is present in the channel. The apparent 1.3 s "lead" is the positive phase response of
the high-pass acting on a narrowband signal — a legitimate prediction computed from past data, of
exactly the kind the project is trying to learn.

The real problem is **baseline comparability**. Because the channel is already phase-advanced,
persistence evaluated on an `imu` input against an `ideal` target is far stronger than persistence
on clean data, measured on a stored SS5 beam-seas realization:

| Horizon | persistence(`heave_imu` at t -> true heave at t+h) | persistence(true at t -> true at t+h) |
|---|---|---|
| 1 s | **0.218 m** | 0.437 m |
| 2 s | **0.324 m** | 0.827 m |
| 3 s | 0.701 m | 1.130 m |
| 5 s | 1.232 m | 1.369 m |

Since every accuracy result in this project is reported as skill score *against persistence*, a
mixed `imu`-input / `ideal`-target task silently changes the denominator, and `ideal` vs `imu` skill
scores stop being comparable — which is the entire point of the Phase 6 ablation.

**Rule for Phase 6: forecast the `imu` channels against `imu` targets, as a shipboard system would.
Never mix an `imu` input with an `ideal` target.** Recorded here because it is invisible in a loss
curve.

### P1-D7 — Corpus mechanics: seeding, layout, and two schema decisions

- **Seeds are derived from grid coordinates, not counted off a parent generator.** Entropy is
  `blake2b("<vessel>|<ss>|<heading>|<speed>|<seed>")` mixed with a namespace constant. This is what
  makes the corpus independent of `n_workers` and of task scheduling, and it lets any single
  realization be regenerated in isolation. `tests/test_generate.py` asserts byte-identical Parquet
  between a 1-worker and a 4-worker run.
- **`RealizationSpec.seed` is a small ordinal (0-39, or 0-7 for S175), not the generator entropy**,
  because `dmf.data.splits` slices on it (the `id` regime is seeds 0-31 vs 32-39).
- **One Parquet file per realization**, so the realization-level split is a partition of a file
  list and is structurally incapable of splitting windows from one realization across a boundary.
- **`t` is absolute simulation time and starts at 120.0 s, not 0.** Keeping the original time
  origin makes the spin-up discard visible in the stored data instead of a claim in a docstring.
  It is the one float64 column; float32 would quantise a 720 s timestamp to ~4e-5 s.
- **Both observation modes are written into every file** (six extra `*_imu` columns), so the
  Phase 6 ablation is a column selection rather than a corpus regeneration. On-disk cost is
  1114 MB against the plan's ~500 MB estimate for `ideal` alone.
- **Per-vessel seed counts** required one additive field on `SimConfig`,
  `seeds_per_cell_by_vessel: tuple[tuple[str, int], ...]`, defaulting to empty. The frigate carries
  40 seeds per cell and the held-out S175 8.
- **The worker pool uses the `spawn` start method**, not `fork`: forking a process that has already
  imported a threaded BLAS is a known source of intermittent hangs. `spawn` requires an importable
  `__main__`, so `generate_corpus` must be called from a script or a test, never from `python -`.
