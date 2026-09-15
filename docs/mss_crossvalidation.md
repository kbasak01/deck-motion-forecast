# MSS cross-validation — do the forecasters transfer to an independent hydrodynamic model?

**These are simulated results.** Both sides of every comparison below are simulations: the corpus
is this project's reduced-order generator, and MSS is ShipX strip theory. Neither is a measurement
of a real ship, and nothing here is real-world validation.

Regenerate with:

```
make mss        # export -> compare -> evaluate -> figures -> gate8
```

Phase 8, timeboxed to 5 hours. Protocol entries P8-D1 through P8-D9 in `docs/protocol.md`.

---

## Why this was worth doing

Every accuracy result in this project comes from one simulator, and two protocol entries already
said that prejudges the headline. P4-D16: "the corpus makes a linear forecaster Bayes-optimal by
construction." And `configs/sim/vessels/s175.yaml:42-45`, on the held-out hull:

> a reduced-order stand-in for a different ship, not a strip-theory computation of the S-175's
> actual RAOs, so `unseen_vessel` measures transfer across a parameter shift rather than across a
> genuinely different hull form.

The MSS toolbox ships exactly what that sentence says we lack: ShipX strip-theory **motion RAOs**
for the ITTC S-175, on a 36-frequency by 36-heading grid. Because our corpus already holds the S175
out of training as the `unseen_vessel` regime, evaluating the same checkpoints on MSS S175
trajectories is a tightly controlled experiment — same models, same normalisation statistics, same
task, same nominal hull. **Only the generator changes.**

## What was done

| | |
|---|---|
| Upstream | `github.com/cybergalactic/MSS` pinned at `98970f7` (2026-09-07) |
| Hull | ITTC S-175, `HYDRO/vessels_shipx/s175/s175.mat`, ShipX strip theory |
| Sea state | SS5: Hs 3.3 m, Tp 9.7 s, gamma 3.3 — identical to `configs/sim/sea_states.yaml` |
| Cells | headings 180 and 135 deg x speeds 0, 6, 12 kn x 3 seeds = 18 realizations |
| Record | 600 s at 10 Hz, 299 jittered components — matched to the corpus exactly |
| Models | `unseen_vessel` checkpoints, default arm, lookback 200, no ablations |
| Controls | sign flip, wave-grid attribution, amplitude rescaling, pipeline parity |
| Baselines | `persistence`, `window_mean`, `damped_persistence`, `ar10/20/40`, `dlinear_ols` **re-scored on the MSS records** (fitted on the corpus training split, never refitted on MSS) |

Octave was installed and MSS's own `waveMotionRAO.m` was run against our NumPy port. They agree to
**4.42e-12** relative, with RMS ratios of 1.00000000 — so the port is not merely statistically
similar to MSS's implementation, it reproduces it. The parity check earned its place twice, catching
two bugs that no self-consistency test could have reached: MSS's gravity constant (9.8100004 against
our 9.80665, which drifted the record 1.3% pointwise while RMS still agreed to 0.1%) and a
nearest-node heading snap that was exact at 180 deg and 38% wrong at 135 deg — the only heading where
roll is live. Details in P8-D4.

## Result 1 — the generators agree on *when*, not on *how much*

Per-DOF statistics at SS5, each MSS value quoted against the corpus's **own across-seed spread**,
because "the spectra overlay closely" is unfalsifiable without that yardstick.

| heading | DOF | corpus | MSS | ratio | z (corpus sd) | per-speed ratio |
|---|---|---|---|---|---|---|
| 135 deg | heave | 0.480 m | 0.363 m | 0.757 | −4.25 | 0.665 – 0.889 |
| 135 deg | pitch | 1.113 deg | 0.871 deg | 0.782 | −3.87 | 0.696 – 0.900 |
| 135 deg | roll | 0.584 deg | 0.576 deg | 0.986 | −0.32 | **0.648 – 1.627** |
| 180 deg | heave | 0.459 m | 0.202 m | 0.441 | −8.76 | 0.353 – 0.584 |
| 180 deg | pitch | 1.501 deg | 0.642 deg | 0.428 | −7.44 | 0.342 – 0.561 |
| 180 deg | roll | 0.038 deg | 0.000 deg | 0.000 | −22.31 | — |

Ratios are ratios **of means**, and z is computed from those means. An earlier version of this
table averaged the per-speed ratios and z-scores instead, which is not a valid summary of either
quantity and printed 1.13 / +3.1 for the roll row where the honest values are 0.986 / −0.32
(P8-D11). `dmf.mss.compare.marginalize_over_speed` now computes the marginal correctly and prints a
warning whenever the per-speed spread exceeds 0.5, which is how the next row was found.

Zero-crossing periods agree much better: ratios 0.856 to 1.045, i.e. within 0.7 to 4.8 corpus
standard deviations.

**The roll row hides a first-order disagreement.** Corpus roll at 135 deg falls 2.5x with forward
speed (0.873 → 0.526 → 0.353 deg RMS at 0 / 6 / 12 kn) while MSS's is essentially flat
(0.566 → 0.588 → 0.574). The per-speed ratio therefore runs 0.648 → 1.117 → **1.627**, at z = +14.9
in the 12 kn cell. The two generators disagree about the *speed dependence of roll response*, not
merely its level, and the marginal ratio of 0.986 conceals that completely. The amplitude gap is
speed-dependent in the other channels too — heave at 180 deg runs 0.353 to 0.584 — so
"our generator is 1.6x hot" is a summary that should not be quoted without its spread.

![Response spectra](../results/mss/mss_response_spectra.png)

## Result 2 — only the DLinear family transfers

![Skill versus horizon](../results/mss/mss_skill_vs_horizon.png)

At the pre-registered cell, pitch at 10 s, against the committed corpus `unseen_vessel` rows. Two
aggregators are shown because they disagree in sign for two models: `mean` is the mean of per-cell
skill, `pooled` forms one skill from summed SSE over all cells. NRMSE is scale-free and does not
depend on the persistence denominator at all.

| model | corpus | MSS (mean) | MSS (pooled) | MSS NRMSE |
|---|---|---|---|---|
| `dlinear` | 0.4144 ± 0.0015 | **0.3935 ± 0.0017** | 0.5153 | 0.858 |
| `dlinear_ols` | 0.4904 | **0.3835** | 0.5065 | 0.880 |
| `ar20` | 0.5258 | 0.1560 | 0.3186 | 1.019 |
| `ar10` | 0.5195 | −0.0004 | 0.1215 | 1.143 |
| `damped_persistence` | 0.0734 | −0.0585 | 0.3326 | 1.026 |
| `window_mean` | 0.0733 | −0.0586 | 0.3326 | 1.026 |
| `transformer` | 0.7654 ± 0.0323 | −0.1773 ± 0.0295 | 0.2198 | 1.108 |
| `ar40` | 0.5412 | −0.7243 | −1.0092 | 1.584 |
| `tcn` | 0.8604 ± 0.0040 | −0.9033 ± 0.1384 | −0.3000 | 1.458 |
| `lstm` | 0.8032 ± 0.0169 | −1.5564 ± 0.5995 | −0.9082 | 1.748 |
| `persistence` | 0.0000 | 0.0000 | 0.0000 | 1.188 |

Closed-form rows are deterministic and carry no seed spread by construction. Every model scored on
both generators appears above; `ar_attitude_only` was never run on the MSS records and so has no
row rather than a withheld one. `persistence` NRMSE is **1.188**, not 1.0 — NRMSE is
`RMSE / signal_std`, and repeating the last sample at a 10 s lead is worse than the record's own
mean. An earlier version of this table printed 1.000 there, which was assumed rather than computed.

**The dividing line is not linear versus deep.** `ar40` is a linear model. It *beats* `dlinear_ols`
on the corpus (0.5412 against 0.4904) and collapses to −0.7243 on MSS, worse than `transformer`.
`ar20` loses 0.37 and is the only model besides the DLinear family to keep positive skill (0.156);
`ar10` loses 0.52 and lands on persistence (−0.0004). What transfers with its skill largely intact
is the DLinear family specifically — `dlinear` loses 0.02 and `dlinear_ols` 0.11 — and nothing else
does.

P3-D1 explains why, and it did so two phases before this run: the corpus has no process noise, so a
sum of sinusoids satisfies an exact linear recursion and **AR identifies the system**. A model that
identifies a generator's dynamics transfers to that generator and to nothing else, and identifying
harder makes it worse — `ar40` loses 1.27 where `ar20` loses 0.37. The deep models are doing the
same thing implicitly. DLinear survives because its trend-plus-seasonal decomposition followed by a
direct per-channel linear map is too constrained to identify the recursion in the first place.

**The ladder is not monotone at the short end, and that qualifies the sentence above.** `ar10`
identifies less than `ar20` and transfers *worse* — 0.52 lost against 0.37, MSS skill −0.0004
against 0.156 — so the AR family peaks at order 20 rather than at its floor. "Identifying harder is
worse" describes the 20 → 40 step, which is the step the argument rests on; it does not describe the
family as a whole, and `ar10` was left out of an earlier version of this table, where the
non-monotonicity would not have been visible.

**Skill in the 1–5 s operational band, per DOF.** An earlier version of this document claimed
"every model keeps positive skill through 5 s." That is false, and it was false in the committed
CSV at the time it was written (P8-D11). It holds only for the DLinear family:

| model | pitch 5 s | heave 5 s | roll 5 s (135 deg) |
|---|---|---|---|
| `dlinear` | 0.956 | 0.933 | 0.948 |
| `dlinear_ols` | 0.936 | 0.923 | 0.952 |
| `tcn` | 0.700 | **0.011** | 0.457 |
| `transformer` | 0.819 | **−0.486** | 0.195 |
| `lstm` | 0.481 | **−0.228** | **−0.191** |
| `ar20` | 0.468 | **−2.354** | **−1.301** |
| `ar40` | **−0.639** | **−4.275** | **−2.370** |

Heave is the channel the project exists for — timing a touchdown on a *heaving* deck — and at 5 s it
is where everything except DLinear is at or below persistence. `lstm` is also negative in roll at 1,
3 and 5 s.

**A caveat on the pre-registered cell.** Pitch at 10 s is the most denominator-fragile cell in the
grid: 10 s is close to Tp = 9.7 s, so persistence is quasi-periodically lucky and its RMSE *falls*
from 1.377 at 5 s to 0.903 at 10 s. Skill is a ratio against that denominator, so the drop is
amplified there. NRMSE, which does not use it, tells a milder version of the same story —
`tcn` 1.458 against `dlinear_ols` 0.880 — and the ordering is unchanged. The cell was fixed in
P3-D12, long before this phase, so it is not a Phase 8 choice; but the headline should not be quoted
without it.

## Controls

| control | result | reading |
|---|---|---|
| Sign-flip ablation | model means move ≤ 0.0061 at pitch/10 s, ≤ 0.0795 over the whole grid | the conclusion does not rest on the P8-D3 sign derivation |
| `persistence` self-skill | exactly 0.0 on both sides | the denominator was re-scored on MSS data, not carried over (delta 4) |
| Normalisation provenance | `unseen_vessel/train` | statistics come from the corpus training split (delta 3) |
| Pipeline parity | agrees to 1e-10 with `evaluate_models` | the external path is the corpus path |
| Wave-grid attribution | −0.04 to −0.19 skill | the wave field is not the cause; the hull response is |
| Amplitude rescaling | no single decomposition; reverses on roll | the normalisation gap is not the explanation |

**Sign-flip ablation.** Individual (cell, seed, record) rows move by up to 1.81, so the small number
above is a statement about the aggregate the conclusion is drawn from, not about every row. Stated
as "skill moves by ≤ 0.006" without that qualifier it was wrong (P8-D11).

**Wave-grid attribution — the control the config declared and the first run skipped.**
`configs/mss/s175_ss5.yaml` defines two wave-grid conventions: MSS's own, and one that reuses
`dmf.sim.spectra.sample_components` so the wave field is bit-for-bit the corpus's. Re-running with
the corpus grid changes pitch/10 s skill by −0.047 (`dlinear_ols`), −0.043 (`ar40`), −0.092 (`tcn`),
−0.100 (`transformer`) and −0.188 (`lstm`) — small, and in the same direction for every model. So
the difference in wave-field discretisation is not what breaks transfer. **What changes is the hull
response**, which is the claim the phase wanted to make and could not make from the first run alone.

**Amplitude rescaling.** Our generator runs hot (Result 1), so after normalisation the MSS records
present at roughly half the amplitude of anything in training; that alone could depress a deep model
with no structural story being true. Rescaling each MSS channel to the corpus mean RMS for its cell
changes amplitude and leaves phase, period and cross-channel relationships intact. At 10 s:

| DOF | `dlinear_ols` | `ar40` | `tcn` | `lstm` | `transformer` |
|---|---|---|---|---|---|
| pitch | 0.000 | +0.051 | +0.324 | +0.450 | +0.004 |
| heave | 0.000 | +2.279 | +0.142 | +1.578 | +0.756 |
| roll | 0.000 | **−13.382** | **−1.629** | **−1.157** | **−2.095** |

The `dlinear_ols` row is exactly zero throughout, which is the control behaving as it must: skill is
scale-invariant and a per-channel constant cannot reach it. Everything else moves, in both
directions, by amounts spanning three orders of magnitude.

**An earlier version of this document reported "recovers ~18%, so ~82% is structural."** That was
`tcn`, at pitch, at 10 s, quoted as though it were a decomposition. It is not one. On roll the
control makes every non-DLinear model *worse* — because the premise fails there: the corpus is
*cold* in roll at speed (MSS/corpus 1.63 at 12 kn), so the rescale shrinks roll in half the cells.
No fraction of the collapse can be attributed to normalisation range from this control; the honest
statement is that scale is not the explanation, and the control does not tell us what is (P8-D11).

## Result 3 — the operational metric cannot be compared across generators

Carry-forward delta 7 required the quiescence detector to be run on the MSS records with the base
rate beside every F1. It is threshold-based on **absolute** limits (3.0 deg / 2.0 deg / 0.8 m·s⁻¹
permissive; 1.5 / 1.0 / 0.4 strict), and delta 2 predicted it would be the part of this phase most
exposed to a scale difference. It was — to the point that two cells have nothing in them to measure.

**Truth side first, per cell, because that is the story.** Pooling these into one number is the
error P6-D19 retracted a whole table for, and an earlier version of this section did exactly that:

| thresholds | heading | kn | MSS base rate | MSS true onsets | corpus base rate | corpus true onsets |
|---|---|---|---|---|---|---|
| permissive | 135 | 0 | 0.9855 | 336 | 0.877 | 212 |
| permissive | 135 | 6 | 0.9705 | 504 | 0.890 | 174 |
| permissive | 135 | 12 | 0.9681 | 630 | 0.944 | 100 |
| permissive | 180 | 0 | **1.0000** | **0 — NOT SCORABLE** | 0.655 | 377 |
| permissive | 180 | 6 | **0.9998** | **0 — NOT SCORABLE** | 0.755 | 334 |
| permissive | 180 | 12 | 0.9988 | 21 (1 of 3 records) | 0.869 | 234 |
| strict | 135 | 0 | 0.5505 | 1197 | 0.318 | 104 |
| strict | 180 | 0 | **0.8357** | 861 | **0.173** | 86 |

Our generator runs ~2x hot (Result 1), so the same absolute limits classify far more of the MSS
record as landable. At permissive thresholds in head seas the MSS deck **never leaves limits**: base
rate 1.0000, zero onsets, nothing to detect. At strict/180/0 kn the base rates differ by a factor of
4.8 (0.836 against 0.173). The two detectors are not facing the same problem, so an MSS F1 must not
be placed beside a corpus F1 and read as a model comparison.

**What is comparable is the ordering within a cell, and it inverts.** On the corpus at strict
thresholds `tcn` and `lstm` beat `dlinear` by 2–4x. On MSS at strict, every cell is topped by
`dlinear` / `dlinear_ols` / `ar40` / `damped_persistence`, with `tcn`, `lstm` and `transformer`
below them. Strict thresholds, F1, mean over 3 training seeds (deterministic rows carry one seed):

| model | 135/0 | 135/6 | 135/12 | 180/0 | 180/6 | 180/12 |
|---|---|---|---|---|---|---|
| base rate | 0.551 | 0.522 | 0.523 | 0.836 | 0.792 | 0.685 |
| `dlinear` | **0.098** | **0.081** | 0.069 | **0.149** | **0.148** | 0.083 |
| `dlinear_ols` | 0.098 | 0.056 | 0.051 | 0.152 | 0.054 | 0.080 |
| `ar40` | 0.077 | 0.073 | **0.173** | 0.125 | 0.067 | **0.114** |
| `tcn` | 0.078 | 0.067 | 0.074 | 0.081 | 0.079 | 0.077 |
| `lstm` | 0.061 | 0.048 | 0.061 | 0.057 | 0.078 | 0.086 |
| `transformer` | 0.039 | 0.036 | 0.086 | 0.059 | 0.051 | 0.068 |
| `rate_matched` (chance timing) | **0.071** | **0.061** | 0.000 | 0.050 | **0.069** | 0.000 |
| `always_quiescent` (onset) | 0.033 | 0.029 | 0.031 | 0.024 | 0.027 | 0.032 |
| `persistence`, `window_mean` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

That ordering is the same direction as the accuracy result, which is worth something: the DLinear
family is on top under the operational metric too.

**Three things this table must not be read as saying.** First, `rate_matched` — a detector that
fires at the right *rate* with chance timing — **beats `transformer` at four of six cells and
`lstm` at three**. Those rows have no measurable timing skill on MSS and should not be described as
detecting anything. Second, onset counts are thin (per-record counts are single digit), no bootstrap
interval is computed because a single record is one resampling unit, and differences below roughly
0.03 F1 at a cell should not be read as real. Third, `persistence` and `window_mean` score exactly
zero because a constant forecast cannot express a transition — that is a property of the model
class, not a measured failure.

The always-yes null scores 0.024–0.033 on the onset formulation while its *per-sample* F1 reaches
0.99 at these base rates, reproducing P6-D2's finding that the onset metric is not naively
base-rate-exploitable — a stronger test of that claim than the corpus can provide, since the corpus
base rate never gets this high.

**Making an operational comparison across generators would need thresholds expressed relative to
each generator's own motion scale.** That is a change to the metric definition and is not made here.

## What this does and does not establish

**It establishes that one model family transfers.** `dlinear` and `dlinear_ols` hold 0.92–0.96 skill
at 5 s in all three DOFs on an independent hydrodynamic computation, and lose only 0.02–0.11 at the
10 s gate cell. Short-horizon deck-motion forecasting with a constrained linear map is learning
something about wave response that is not an artifact of our generator.

**It establishes that nothing else does.** All three deep models and the AR family go to or below
persistence, and the failure is not confined to long lead: at 5 s in heave — the channel the project
exists for — `tcn` is at 0.011, `transformer` at −0.486, `ar40` at −4.275.

**It refutes the framing this phase started with.** The pre-registered prediction (P8-D1) was that
the `unseen_vessel` ordering would hold with `tcn` above 0.5; it did not. But the counter-hypothesis
as stated — deep models lose because they exploit cross-channel phase, a linear map cannot — does
not survive either, because `ar40` is linear, per-channel, and loses *more* than `transformer`. The
distinction that actually separates the two groups is **how completely a model identifies the
generator's dynamics**. P3-D1 recorded in Phase 3 that this corpus has no process noise, so its
motion satisfies an exact linear recursion and AR *identifies the system*; identifying harder makes
transfer worse (`ar40` −1.27 against `ar20` −0.37). The deep models do the same implicitly. DLinear
is too constrained to do it at all, and that is why it survives.

**It establishes that the deep advantage was measured where it does not transfer.** Gates 3, 4 and 5
were all read at pitch / 10 s (P3-D12, P4-D1, P5-D2). There the deep models beat `dlinear_ols` by
0.37 on the corpus and lose to it by 1.29 on MSS. P4-D14 restriction 2 had already found `tcn`
merely ties `dlinear_ols` in the 1–5 s operational band.

**It does not establish that the deep architectures are unsuited to deck-motion forecasting.** It
establishes that *these* checkpoints, trained on *this* corpus, learned structure specific to this
generator. Whether a corpus without the P8-D6 defect supports a deep model that transfers is a
Phase 9 question.

**It does not establish a single cause.** The wave-grid control rules out the wave-field
discretisation and the amplitude control rules out normalisation range, which together point at the
hull response. Within the hull response, P8-D6's cross-DOF phase defect is one candidate among
several — the corpus also applies an `exp(-(kL/4pi)^2)` rolloff and an `exp(-k*draft)` Smith factor
where strip theory solves the radiation-diffraction problem. And the AR result shows the mechanism
cannot be *only* about cross-channel phase.

**It does not establish anything about real ships.** MSS is another simulator. Strip theory is
linear potential flow: no viscous roll damping beyond an empirical term, no parametric resonance, no
green water. Two simulators agreeing or disagreeing bounds generator-specific overfitting; it says
nothing about either one's fidelity to a real deck.

## The limitation this leaves

`src/dmf/sim/response.py:229` applies a **real** wave-slope excitation to roll and pitch, so both
come out in phase with heave where strip theory puts them in quadrature (P8-D6, measured 0.1 deg
against 87.8 deg at w = 0.300 rad/s). Amplitudes are right; only the cross-DOF phase is wrong.
Per-DOF marginals are untouched, so every Gate 1 invariant still holds and
`results/physics_validation.md` is unaffected.

It was not fixed. Fixing it invalidates the corpus and every result in Phases 2 through 7, which is
far outside a 5-hour timebox. It is the largest open item this phase produced and is carried to
Phase 9.
