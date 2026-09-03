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

---

## Phase 2 — windowing, splits, normalization

### Gate 2 thresholds

**No Gate 2 threshold was changed.** All five criteria are asserted at the values stated in
`docs/IMPLEMENTATION_PLAN.md` §Phase 2, in `tests/test_splits.py` and `tests/test_windows.py`,
against both a generated small-corpus fixture and (marked `slow`) the real 2304-realization
corpus.

### The four regimes, with measured realization counts

Built by `dmf.data.splits.build_split` from `artifacts/corpus/manifest.parquet`
(2304 rows: frigate 1920, s175 384), at the default `val_frac = 0.15`:

| Regime | Train | Val | Test | Total | Held-out axis |
|---|---|---|---|---|---|
| `id` | 1296 (seeds 0-26) | 240 (seeds 27-31) | 384 (seeds 32-39) | 1920 | seed ordinal |
| `unseen_seastate` | 1224 (seeds 0-33) | 216 (seeds 34-39) | 480 (all seeds) | 1920 | `ss == SS6` |
| `unseen_heading` | 1224 (seeds 0-33) | 216 (seeds 34-39) | 480 (all seeds) | 1920 | `heading == 90 deg` |
| `unseen_vessel` | 1632 (seeds 0-33) | 288 (seeds 34-39) | 384 (all 8 seeds) | 2304 | `vessel == s175` |

Every regime except `unseen_vessel` is confined to the frigate; `s175` is never trained on in
any regime, which `tests/test_splits.py::test_no_regime_ever_trains_on_the_heldout_vessel`
asserts across all four. The three generalization regimes train on **all** seeds of the cells
they retain, minus validation: holding out a whole axis is what makes the regime hard, so the
seed ordinal is irrelevant there.

The counts above are asserted as literals in
`tests/test_splits.py::test_real_corpus_regime_counts_and_disjointness` (marked `slow`), so
the table cannot rot silently.

### P2-D1 — The `id` test cut is a fraction of the seed ordinals, not the literal seed 32

`docs/IMPLEMENTATION_PLAN.md` §2.2 states the `id` cut as "seeds 0-31 train, 32-39 test".
It is implemented as `TEST_SEED_FRAC = 0.2` of the **distinct seed ordinals present**, taken
from the top of the range.

On the production corpus the two are identical: `ceil(0.2 * 40) = 8` gives seeds 32-39, and
`tests/test_splits.py::test_real_corpus_id_cut_is_seeds_32_to_39` asserts exactly that,
together with val = 27-31 and train = 0-26.

The reason for the fraction is that a hard-coded 32 makes `build_split` raise "empty test set"
on any corpus smaller than the production one. The Gate 2 test fixture is a generated 60-file,
8-seeds-per-cell corpus built in `tmp_path` (so Gate 2 is verifiable on a clean checkout with
no `artifacts/`), where the same rule yields `ceil(0.2 * 8) = 2` → seeds 6-7. Hard-coding the
literal would have forced Gate 2 onto the 1.07 GB corpus, i.e. made the leakage guard
unrunnable in CI.

### P2-D2 — Validation is carved by seed ordinal, globally, with no RNG

Validation takes the top `ceil(val_frac * n_distinct_ordinals)` seed ordinals of the
development pool, rather than a random draw of realizations.

Two consequences, both wanted:

- **The bare seed-ordinal sets become disjoint**, not merely the full realization keys. The
  tests can therefore assert disjointness at the weaker, coarser level too
  (`test_validation_seed_ordinals_are_disjoint_from_train_and_test`), which catches a class of
  bug that key-level disjointness would pass.
- **`build_split` contains no RNG at all.** A split is a pure function of `(meta, regime,
  val_frac)`; nothing has to be recorded to reproduce one.
  `test_build_split_is_deterministic` shuffles the manifest rows and asserts an identical
  `Split`.

`ceil` rather than `round` guarantees a non-empty validation partition on a fixture-scale
corpus. `build_split` raises `ValueError` if any of the three partitions ends up empty:
`unseen_vessel` on a single-vessel corpus is an error, never a silently returned empty
frozenset.

### P2-D3 — `load_data` and `configs/data/default.yaml` added; the YAML key is `fs_hz`, not `fs`

`DataConfig` existed in Phase 0 but had no loader and no config file. `dmf.config.load_data`
follows the `load_sim` pattern: explicit missing-key check, hand-built frozen dataclass, tuples
rather than lists. `load_experiment` is deliberately left unimplemented — it needs `ModelConfig`
and `TrainConfig` wiring, which is Phase 3/4 work.

`docs/IMPLEMENTATION_PLAN.md` §2.1 spells the sampling-rate key `fs`. The config file uses
**`fs_hz`**, matching `SimConfig.fs_hz` and `configs/sim/corpus.yaml`. One spelling for one
quantity; the unit is in the name, which is the convention the rest of the project follows.

`load_data` rejects: non-positive `fs_hz`/`lookback`/`stride`; empty, non-positive, or
non-ascending `horizons`; an `observation_mode` outside `{ideal, imu}`; and `target_dofs` that
is not a **prefix** of `input_channels` (P2-D4).

### P2-D4 — `target_dofs` must be a prefix of `input_channels`, checked at config load

`dmf.models.persistence.Persistence.forward` is documented to return
`x[:, -1, :C_out]` — it forecasts by slicing the *first* `C_out` input channels. That is only
correct if the target channels are the leading channels of the input, in the same order. Any
other ordering makes every baseline silently forecast the wrong DOF, and the resulting skill
scores would be wrong without being obviously wrong.

It is validated in `load_data` rather than discovered later.
`tests/test_windows.py::test_pipeline_sanity_fails_on_a_channel_order_swap` is the converse
check: a deliberately swapped target column order must make the Gate 2 criterion-5 control
raise.

### P2-D5 — `window_spec_from_config` added

`WindowSpec` is consumed by the dataset, by the split integrity checks, and by
`dmf.train.registry.build_model`. Constructing it in three places is how a lookback and a
receptive field drift apart. One constructor, `dmf.data.windows.window_spec_from_config(cfg)`.

Arithmetic now asserted rather than commented
(`tests/test_windows.py::test_window_count_arithmetic_on_a_fixed_geometry`, renamed from
`test_production_window_count_arithmetic` when the P3 task revision made this geometry a
deliberately pinned arithmetic case rather than the production one):
`n_windows(6000, WindowSpec(200, (10,20,30,50), 5)) == 1151`, first start 0, last start 5750,
and `5750 + 250 == 6000` exactly.

### P2-D6 — `NormStats.subset`, `build_norm_stats`, `is_train_partition` added

- **`NormStats.subset(channels)`** returns statistics over a channel subset, carrying
  `fitted_on` and `n_realizations` through unchanged. The dataset scales its **inputs** with the
  full `C_in` statistics but inverts its **targets** with the `C_out` subset, so `invert_norm`
  can keep a strict shape check instead of silently slicing whatever it is handed. A silent
  slice is exactly how a channel-order bug survives.
- **`build_norm_stats`** applies the same provenance and zero-variance validation to a
  pre-computed scale. `DeckMotionDataset` accumulates first and second moments per realization
  in float64 rather than materialising a float64 copy of the training split (`id/train` is
  1296 × 6000 × 6, i.e. 373 MB in float64), and it must not bypass the guard to do so.
- **`is_train_partition(label)`** is the single definition of "was this fitted on train": the
  label must be exactly `"train"` or end in `"/train"`. It is checked at fit time *and* again in
  `apply_norm`, so a `NormStats` smuggled in from a checkpoint is refused even though it was
  never fitted in this process. `test_apply_norm_refuses_statistics_not_fitted_on_train`
  constructs exactly that case with `dataclasses.replace`.

**Positive control for the leak this guards against.** A guard is only worth having if the leak
would have moved a number. `test_train_only_statistics_differ_measurably_from_whole_corpus_statistics`
fits the scale on `unseen_seastate/train` and again over train+test, and asserts the per-channel
ratio departs from 1 by more than 10%. It does: pulling SS6 into the scale is a large,
directional change, and `unseen_seastate` is precisely the regime that holds SS6 out.

Measured train-only scales on `id/train` (1296 realizations, corpus units):
`roll 3.64641 deg`, `pitch 1.24605 deg`, `heave 0.75672 m`, `roll_rate 1.95083 deg/s`,
`pitch_rate 0.92899 deg/s`, `heave_rate 0.44896 m/s`.

### P2-D7 — `DeckMotionDataset.describe_window` and five read-only properties added

`describe_window(index) -> (RealizationKey, start_sample)` exposes a window's provenance. It is
what makes two Gate 2 criteria checkable rather than assertable:

- criterion 3 opens *that* realization's Parquet file and compares the slice;
- "windows never span a realization boundary" becomes an assertion over concrete
  `(key, start)` pairs, not a claim in a docstring.

It is also the hook any later per-cell breakdown of results will use.

The properties `realization_keys`, `window_spec`, `target_columns`, `input_columns`,
`corpus_root`, `windows_per_realization` and `training_series` are added so that
`persistence_pipeline_sanity` can rebuild the raw-array path **without** reaching into the
dataset's internals, and so the raw path can recompute start indices from
`window_start_indices` rather than trusting the dataset's own index arithmetic. If it trusted
it, an off-by-one would cancel and the control would pass.
`test_pipeline_sanity_fails_on_an_off_by_one` shifts the dataset's starts by one sample and
asserts the control raises.

### P2-D8 — `resolve_columns` added; observation mode remaps inputs and targets together

`resolve_columns(names, mode)` maps logical channel names to corpus columns, appending `_imu`
in `imu` mode. `heave_acc` has no `_imu` twin and passes through — it *is* the accelerometer
channel.

The dataset calls it once for `input_channels` and once for `target_dofs`, from the same call
site with the same `mode`, so **an `imu`-input / `ideal`-target task is not constructible**.
Per P1-D6 this is not a tidiness point: `heave_imu` leads the truth by about 1.3 s at the SS5
spectral peak, so persistence on an `imu` input scored against an `ideal` target is roughly
twice as strong at a 1 s horizon (0.218 m vs 0.437 m on the stored realization measured in
P1-D6). That silently changes the denominator of every skill score and is invisible in a loss
curve. Making the mix structurally impossible is cheaper than remembering not to do it.

### P2-D9 — `src/dmf/eval/controls.py` added, and why the control does not import `Persistence`

New module holding the integrity controls. Phase 2 implements one; the shuffle control and the
untrained-model control arrive in Phase 4 and get the same home.

`persistence_pipeline_sanity(dataset, corpus_root, rtol=1e-6)` runs two paths that share
nothing but the realization key list and the window geometry:

- **through the pipeline** — iterate `make_dataloader(..., shuffle=False)`, forecast
  `x[:, -1:, :C_out].expand(-1, H, -1)` in normalized space, map back with
  `invert_norm(pred, stats.subset(target_dofs), window_mean)`;
- **on raw arrays** — re-read each Parquet file with pandas, recompute starts, take
  `series[start + L - 1]` as the forecast and `series[start+L : start+L+H]` as the target. No
  dataset, no normalization, no torch.

The forecast is computed **inline from tensor ops rather than by importing
`dmf.models.persistence.Persistence`**, because `Persistence.forward` and
`BaseForecaster.__init__` are unimplemented Phase 3 work and `src/dmf/models/` is read-only to
the agent that owns evaluation. Phase 3 must re-run this control against the real model once it
exists; until then the control tests the *pipeline*, not the model.

`rtol = 1e-6` is stated rather than assumed. The two paths are not bitwise identical by
construction: the dataset stores its de-meaned, scaled input as float32, so the recovered
forecast carries a relative rounding error of order `2**-24 ≈ 6e-8`.

**Measured on the full `id/test` partition of the real corpus** (384 realizations,
441 984 windows, production geometry `L = 200`, `H = 50`, `stride = 5`):
`max_rel_diff = 5.006e-08`, i.e. the float32 storage floor and nothing else. Persistence RMSE,
identical to six decimal places on both paths:

| Horizon | roll (deg) | pitch (deg) | heave (m) |
|---|---|---|---|
| 10 samples (1 s) | 1.946806 | 0.896462 | 0.441620 |
| 20 samples (2 s) | 3.748621 | 1.639730 | 0.837991 |
| 30 samples (3 s) | 5.273611 | 2.122079 | 1.151041 |
| 50 samples (5 s) | 7.098060 | 2.233934 | 1.442000 |

These are the denominators of every skill score this project will report on `id`.

### P2-D10 — `assert_no_shared_time_index` is vacuous, and is tested with a corrupted split

Under realization-level splitting the intersection of any two partitions' key sets is empty, so
the guard-band loop never executes. That is the point — it is the check that would catch a
future refactor introducing within-realization splitting — but it also means a passing run
proves nothing on its own.

`test_a_corrupted_split_makes_the_time_index_check_raise` therefore forces one test realization
into the training set and asserts both `assert_no_shared_time_index` and `assert_seed_disjoint`
raise. `test_seed_disjointness_catches_a_heldout_axis_leak` covers the subtler case: a split
that is perfectly key-disjoint but carries an `SS6` realization in training must still fail,
because key disjointness alone does not mean the held-out axis was held out.

### P2-D11 — One canonical realization-key constructor, because the dtypes differ

`manifest.parquet` stores `heading`/`speed` as **float64** and `seed` as **int64**; the
per-realization Parquet files store **float32** and **int32**. Keys built from the two sources
would never compare equal, and every set intersection in the leakage guards would be silently
empty — i.e. every disjointness assertion would pass unconditionally.

`dmf.data.splits.realization_key` is the single cast site (`str, float, float, str, int`), and
`test_realization_key_is_dtype_canonical` asserts a key built from a manifest row equals the key
built from the corresponding file's first row, after checking that the file's dtypes really are
float32/int32.

### Gate 2 evidence

| Criterion | Where | Evidence |
|---|---|---|
| 1. Zero seed overlap, all four regimes | `test_split_partitions_are_seed_disjoint`, `test_heldout_axis_never_appears_in_train_or_val` | Pairwise-empty key intersections plus held-out-axis absence, parametrized over all four regimes; repeated on the real corpus with the exact counts above |
| 2. No shared time index | `test_no_time_index_is_shared_between_partitions` + `test_a_corrupted_split_makes_the_time_index_check_raise` | Passes on real splits, raises on a split with one realization forced into both train and test |
| 3. Reconstructed window matches the raw Parquet slice | `test_dataset_window_matches_the_raw_parquet_slice` | `y` and `window_mean` exact (`np.array_equal` on stored float32); input reconstructed to `atol = 1e-6` in degrees/metres through the divide-then-multiply round trip |
| 4. Train-only normalization statistics | `test_fit_norm_stats_refuses_a_non_training_partition`, `test_test_partition_without_train_stats_raises`, `test_train_only_statistics_differ_measurably_from_whole_corpus_statistics` | Refused at fit and at use; `val`/`test` without stats raises; positive control shows the leak would move the scale by > 10% |
| 5. Persistence through the pipeline == persistence on raw arrays | `test_persistence_pipeline_sanity` (`id` and `unseen_seastate`), plus the real-corpus run above | `max_rel_diff = 5.006e-08` over 441 984 windows |

---

## Phase 3 — baselines

### Gate 3 outcome

**Gate 3 as written does not pass, and cannot.** The criterion (`docs/IMPLEMENTATION_PLAN.md`
§Phase 3) is: if AR(p) reaches 0.8 skill vs persistence at 3 s on roll in `id`, the task is too
easy. Measured on the real corpus, `id`/test, closed-form fit on `id`/train:

| model | skill @ 3 s, roll, `id`, `ideal` | same, `imu` |
|---|---|---|
| persistence | 0.000000 | 0.000000 |
| damped persistence | 0.385034 | 0.385033 |
| AR(10) | 0.998261 | 0.992559 |
| **AR(20)** | **0.998692** | **0.995756** |
| AR(40) | 0.999107 | 0.996176 |

AR(20) forecasts roll 3 s ahead to 0.19 deg RMSE (`ideal`) against a persistence denominator of
5.27 deg. The threshold is 0.8; the measurement is 0.9987.

**No threshold was relaxed.** The gate is recorded as failed. Two remedies the plan offers were
tried and are documented below. The gate is **restated** in P3-D12 — same 0.8 threshold, read at
the decision horizon on the binding DOF — and that restatement is applied.

### P3-D1 — Why the task is easy, and why it is not leakage

The shuffle control settles the leakage question. AR(20) refitted on time-shuffled training
targets scores 0.3705 on `id`/test roll at 3 s against a window-mean null of 0.3699 —
`excess = 0.001` against a 2% tolerance. All 36 control rows pass in both observation modes
(`|excess| <= 0.0043`). Destroying the temporal relationship destroys the model's advantage
entirely, which is what a clean pipeline looks like.

The cause is structural. The vessel RAO is a narrowband filter, so deck motion is close to a
modulated sinusoid with a ~12 s roll period, and there is no process noise anywhere in the
generator. A sum of sinusoids satisfies an exact linear recursion, so least-squares AR over 200
lags x 6 channels is not approximating the system, it is identifying it. A 3 s horizon is a
quarter of the roll period. **The gate's chosen cell — roll, 3 s, in-distribution — is the single
easiest cell in the corpus.**

Consequence for every result this project will report: with no process noise the achievable-skill
ceiling is unrealistically high, so absolute numbers flatter every model. Only *relative* model
comparisons and *OOD degradation* should be read as findings. This belongs in the README.

### P3-D2 — `imu` observation mode does not rescue the gate

Tried as the plan's second remedy. AR(20) skill at 3 s on roll moved only from 0.9987 to 0.9958.

The reason is already in P1-D6 and was missed when the remedy was proposed: the `heave_imu`
discrepancy "is almost entirely phase" — the two RMS values agree to 1%. Phase distortion from a
causal high-pass is deterministic and linear, so a linear model absorbs it for free; `heave` skill
is unchanged at 0.997 despite an error/signal power ratio of 0.67. The only genuinely
unpredictable component the IMU model adds is white noise (0.02 deg attitude, 0.02 deg/s gyro),
which against a 3.5 deg roll signal is ~45 dB SNR — below the error AR was already making.

Measured observation-error/signal power, median over sea states and headings: roll 0.001, pitch
0.0008, heave 0.666. The IMU model meaningfully corrupts **heave only**; the high ratios for roll
and pitch occur exactly where that DOF already sits on its P1-D2 residual floor.

`configs/data/imu.yaml` is kept as the `ideal` config's twin, differing in `observation_mode` and
nothing else, so the Phase 6.3 observability ablation is a controlled comparison.

### P3-D3 — Lengthening the horizon alone does not rescue it either

Measured out to 30 s, `imu`, `id`, AR(20) roll: 0.996 @ 3 s, 0.989 @ 5 s, 0.894 @ 10 s,
0.872 @ 15 s, 0.849 @ 20 s, 0.794 @ 30 s. Roll does not cross 0.8 until **30 s**, which is 2.5
roll periods and far outside any touchdown-decision timeline.

But the horizon sweep showed the difficulty is strongly DOF-dependent, and **roll is the most
predictable DOF, not the least**. Lightly damped roll resonance (`Tn = 12 s`, `zeta = 0.06`) is the
most narrowband channel in the corpus; pitch is stiffer and more damped, therefore broader-band,
therefore genuinely hard. Normalised RMSE (RMSE / signal std; 1.0 = no better than the mean),
AR(20), `imu`, `id`: pitch 0.34 @ 3 s and 0.50 @ 5 s where roll is 0.09 and 0.20.

**Pitch is the binding DOF.** The gate measured roll.

### P3-D4 — Task definition revised: six targets, horizons to 15 s

`configs/data/default.yaml` and `configs/data/imu.yaml` changed together:

    target_dofs: [roll, pitch, heave]            ->  [roll, pitch, heave, roll_rate, pitch_rate, heave_rate]
    horizons:    [10, 20, 30, 50]                ->  [10, 20, 30, 50, 100, 150]

**Six targets, for correctness — and it fixes an inconsistency the three-target task carried.**
`dmf.eval.quiescence.detect_quiescent_mask` thresholds `|roll|`, `|pitch|` **and `|heave_rate|`**,
but `heave_rate` was an input channel and never a forecast target. The operational metric could
not supply its own decision variable; it would have had to differentiate a forecast `heave`,
amplifying exactly the high-frequency error the forecast is worst at. The wider framing requires
it independently: a full-scale manned or unmanned helicopter deck landing requires attitudes *and*
their rates to stay bounded through the window, not attitudes alone.

**This does not make the task harder, and it was not adopted for that.** Measured, `id`/`imu`,
AR(20): rate channels sit within 0.03 of their position channels in normalised RMSE at every
horizon. Relative spectral width barely moves (0.11-0.17 across all six), because differentiation
multiplies by omega on a signal that is already narrowband.

**Horizons to 15 s, for difficulty and for operational realism.** 10 s and 15 s are where the
problem stops being trivial, and a full-scale rotorcraft's commit-to-land lead time is longer than
a quadrotor's. The 1-5 s horizons are retained as the operational set. Post-revision, AR(20) skill
on `id` (`ideal`) falls below 0.8 in **four of six channels** at both 10 s and 15 s:

| | 1 s | 2 s | 3 s | 5 s | 10 s | 15 s |
|---|---|---|---|---|---|---|
| roll | 1.000 | 1.000 | 0.999 | 0.992 | 0.926 | 0.909 |
| pitch | 1.000 | 0.998 | 0.984 | 0.928 | **0.545** | **0.530** |
| heave | 1.000 | 1.000 | 0.997 | 0.976 | **0.676** | **0.675** |
| roll_rate | 1.000 | 0.998 | 0.996 | 0.995 | 0.883 | 0.879 |
| pitch_rate | 0.999 | 0.990 | 0.978 | 0.924 | **0.585** | **0.517** |
| heave_rate | 1.000 | 0.998 | 0.993 | 0.983 | **0.636** | **0.651** |

At 15 s pitch and pitch_rate are at 0.985/0.984 normalised RMSE, i.e. effectively unpredictable,
while roll still has headroom at 0.43. The useful design band is **10-15 s and differs per DOF**.

### P3-D5 — Skill vs persistence is not comparable across horizons on this signal

The denominator oscillates with the signal's own period, because persistence error tracks the
autocorrelation. Persistence RMSE for roll on `id`/test does not increase monotonically with lead
time — it *falls* from 7.088 deg at 50 samples to 3.794 deg at 100 samples, because 100 samples
(10 s) is close to one roll period and the signal has come back around. Measured autocorrelation
of roll: -0.89 at 5 s, +0.62 at 10 s.

A skill-vs-horizon curve on this data therefore shows dips that are properties of the reference,
not of the model. **Report normalised RMSE (RMSE / signal std) alongside skill whenever the
horizon varies.** This is not a presentational preference; P3-D7 records a case where reading skill
alone would produce a false conclusion about a model.

**Status: recorded, not implemented.** No `nrmse` or `signal_std` column exists in any committed
artifact or anywhere in `src/dmf/`. Every normalised-RMSE figure quoted in this protocol — including
the value used to reject 15 s as the gate cell in P3-D12 — comes from an ad-hoc measurement outside
the pipeline and is not reproducible from `results/`. The per-cell target standard deviation is
available at scoring time, so this is a column, not a re-run. **It is a Phase 4 prerequisite**: the
mandate is stated twice and honoured nowhere.

### P3-D6 — P2-D9 denominators superseded (the Phase 2 record stands as history)

Raising `max_horizon` from 50 to 150 samples drops the last 20 window starts of every realization:
**1131 windows per realization, not 1151**, and 434 304 windows on `id`/test rather than 441 984.
Every persistence denominator moves in the fifth significant figure. The P2-D9 table remains
accurate *as a Phase 2 record* and is not edited; it is superseded here.

Persistence RMSE, `id`/test, `ideal`, 384 realizations / 434 304 windows. Attitudes in **degrees**,
heave in **metres**, rates in **deg/s** and **m/s**:

| horizon (samples) | roll | pitch | heave | roll_rate | pitch_rate | heave_rate |
|---|---|---|---|---|---|---|
| 10 | 1.944663 | 0.896319 | 0.441743 | 1.067514 | 0.752844 | 0.285223 |
| 20 | 3.744455 | 1.639687 | 0.838214 | 2.048398 | 1.341412 | 0.535198 |
| 30 | 5.267361 | 2.122358 | 1.151318 | 2.865996 | 1.664056 | 0.721625 |
| 50 | 7.087561 | 2.234325 | 1.442156 | 3.798346 | 1.549273 | 0.852486 |
| 100 | 3.793711 | 1.559155 | 0.833518 | 1.958739 | 1.273090 | 0.478612 |
| 150 | 5.251935 | 1.785132 | 1.106453 | 2.898094 | 1.307998 | 0.681423 |

(Roll before the revision, for the record: 1.946806 / 3.748621 / 5.273611 / 7.098060.)

Verified two independent ways: a standalone pandas/numpy pass over the raw Parquet sharing no code
with the dataset pipeline (max abs difference 4.98e-07, the rounding of the pinned literals), and
through `persistence_pipeline_sanity` with the real `Persistence` model to `abs = 5e-6`. All six
DOFs are pinned in `tests/test_models.py::PERSISTENCE_RMSE_ID_TEST`, together with the window count
and an explicit assertion that roll at 100 samples is *below* roll at 50, so the non-monotonicity
of P3-D5 is recorded as intended behaviour rather than rediscovered as a bug.

**`imu` denominators**, measured on the completed sweep, same partition and window count
(`id`/test, 384 realizations, 434 304 windows). Not interchangeable with the `ideal` table above:
skill is always measured against persistence in the same observation mode (P1-D6), so the two are
each internally valid and are not a common yardstick.

| horizon (samples) | roll_imu | pitch_imu | heave_imu | roll_rate_imu | pitch_rate_imu | heave_rate_imu |
|---|---|---|---|---|---|---|
| 10 | 1.944871 | 0.896747 | 0.436086 | 1.067900 | 0.753420 | 0.283867 |
| 20 | 3.744563 | 1.639890 | 0.827238 | 2.048611 | 1.341784 | 0.532558 |
| 30 | 5.267439 | 2.122484 | 1.135577 | 2.866156 | 1.664371 | 0.717851 |
| 50 | 7.087624 | 2.234428 | 1.419148 | 3.798464 | 1.549576 | 0.847150 |
| 100 | 3.793831 | 1.559344 | 0.812388 | 1.958924 | 1.273527 | 0.474706 |
| 150 | 5.252024 | 1.785355 | 1.093555 | 2.898274 | 1.308342 | 0.678040 |

The attitude channels barely move from `ideal` (roll at 30 samples: 5.267439 vs 5.267361) because
the IMU model's attitude noise is ~45 dB below the signal. `heave_imu` moves visibly (1.135577 vs
1.151318) because its reconstruction is a two-stage causal high-pass, i.e. a deterministic phase
and gain change rather than added noise (P1-D6, P3-D2).

### P3-D7 — Where AR(20) loses to persistence, recorded rather than dropped

144 cells (4 regimes x 6 DOFs x 6 horizons). 12 are losses, and
`tests/test_models.py::AR20_NEGATIVE_SKILL_CELLS` asserts the measured loss set **equals** the
recorded set, so a new loss fails the suite and a silent repair fails it too (CLAUDE.md
non-negotiable 6).

`id` and `unseen_seastate` are clean: all 72 cells positive, minima 0.517 (`pitch_rate` @ 150) and
0.158 (`pitch` @ 100).

**`unseen_heading` / pitch and pitch_rate, 20-150 samples.** The P1-D2 residual floor: the
`unseen_heading` test set *is* beam seas, where the pitch heading factor is clamped at
`eps = 0.05`, ~26 dB down. The test-set pitch signal is the engineering stand-in for hull
asymmetry, not the pitch physics the model trained on. `pitch_rate` is new at this task definition
and is worse than `pitch` — it is the derivative of the clamped signal, inheriting the floor with
the noise differentiated up: -6.9 at 2 s where `pitch` is only -0.52, falling to -57 at 15 s. Only
the 1 s horizon survives for either.

**`unseen_vessel` / roll and roll_rate at 150 samples, and this one is not AR degrading.** AR's own
roll RMSE grows smoothly with lead time (0.007, 0.054, 0.19, 0.76, 1.85, 2.03 deg). The
*denominator* collapses: 150 samples is 15 s and the held-out S-175 has `tn_s = 14.5`, so at that
lead the hull has returned almost exactly one roll period and persistence is nearly free — its roll
RMSE *falls* from 3.91 deg at 5 s to 1.68 deg at 15 s. AR's coefficients encode the frigate's 12 s
roll mode and extrapolate at the wrong period. Realization bootstrap CI [-0.639, -0.265] for roll
and [-0.782, -0.329] for roll_rate: transfer failure, not resampling noise.

This is P3-D5 biting in the regime the project most cares about. **Report normalised RMSE for
`unseen_vessel` specifically**, not merely across horizons — skill alone would attribute a
denominator artefact to the model.

### P3-D8 — The shuffle control's null is the window mean, not zero skill

`docs/IMPLEMENTATION_PLAN.md` §5.2 states the null as "skill collapses to ~0". That is wrong for
this task. A least-squares fit on time-shuffled targets degenerates to the conditional mean of the
target, which after `invert_norm` is the **window-mean forecast** — and on a narrowband signal the
window mean *beats* persistence beyond about 2 s (measured on `id`/test: +0.37 skill at 3 s, +0.71
at 5 s). A "skill must be ~0" test would report leakage on a clean pipeline.

`dmf.eval.controls.shuffle_control` therefore scores the shuffled model against
`DampedPersistence(tau -> 0+)` and reports `excess = 1 - MSE_subject/MSE_null`, a variance ratio
rather than a skill difference. Tolerance 2%; measured worst case 0.53% at the pre-revision
geometry, 0.43% after. `strict=True` is retained — a failed integrity control invalidates the
headline table, so it stops the run loudly.

### P3-D9 — The untrained control's stated criterion is wrong, and is reported rather than enforced

The plan's criterion is that a random-init model "must score worse than persistence"
(`skill < 0`). On the real corpus it does not, for the same reason as P3-D8: a small-weight random
projection of a de-meaned window emits something close to zero, which after `invert_norm` is the
window-mean forecast, and the window mean beats persistence past ~2 s. Measured untrained DLinear
on `id`/test: +0.32 skill at 2 s, +0.70 at 5 s.

`untrained_control` is therefore called with `strict=False` and the outcome recorded in
`results/baselines_controls.csv`. A window-mean null does not rescue it either — the same untrained
DLinear still removes 17.5% of that null's error at 2 s on roll — because a random linear map of
the lookback is a bad *filter of genuine past data*, not a null model. The only initialisation
guaranteed to pass is one that emits the window mean exactly, which passes by construction. The
literal criterion is kept and its failures reported, rather than tuning the null or the tolerance
until the control agrees.

### P3-D10 — Deterministic models are exempt from the three-seed rule

CLAUDE.md non-negotiable 5 requires >= 3 seeds for any model-vs-model comparison. Persistence,
damped persistence and AR(p) have no stochastic component: `tests/test_models.py` asserts AR
coefficient recovery is **bitwise** identical across seeds. They emit one row each with
`n_seeds = 1` and `skill_std = NaN`, rendered `n/a`. `dmf.eval.report.build_baselines_table` routes
deterministic and stochastic rows separately and refuses a stochastic group carrying fewer than
three seeds. The exemption is for models proven deterministic, not for single-seed comparisons.

### P3-D11 — Normalisation provenance is now asserted at scoring time

`evaluate_models` documented a provenance guarantee it never checked. `dmf.eval.runner` now calls
`_check_norm_provenance` before reading any window: the statistics must be labelled
`<regime>/train`, and that regime must equal the dataset's own.

The sharp case is **not** the one initially proposed. `dmf.data.splits` confines every regime's
development pool to `PRIMARY_VESSEL`, so `id/train` contains no S-175 and pairing `id` statistics
with `unseen_vessel/test` is a scale *mismatch*, not a leak. The genuine leak is
`id/train` applied to `unseen_seastate/test` or `unseen_heading/test`: `id/train` spans all four
sea states and all four headings, **including the SS6 and 90 deg beam realizations those two
regimes exist to withhold**, so a cross-regime scale is fitted over the very variance the regime
is testing generalisation to.

The check fails closed: a label that does not parse as `<regime>/train` is refused rather than
skipped, because a guard that a relabelling can switch off is not a guard. Note what is still *not*
verified — that `stats.scale` was actually computed from the realizations `fitted_on` names. The
label is an assertion by the fitter, not a checksum.

### P3-D12 — Gate 3 restated: APPLIED 2026-08-27

The gate's cell is saturated by construction (P3-D1), and no change to the task definition makes
roll at 3 s hard: roll is the *most* predictable DOF in the corpus, not a representative one. The
gate is therefore restated, not relaxed — the threshold value 0.8 is unchanged; the cell it is read
at moves to the horizon the decision is actually taken at and the DOF that actually binds.

**Gate 3, as of this entry:**

> AR(p) must not exceed 0.8 skill vs persistence **at the decision horizon (10 s) on the binding
> DOF (pitch)** in the `id` regime.

Measured: **0.545 (`ideal`) / 0.513 (`imu`) — passes.**

Justification for each of the two moves, both measured in P3-D3 and P3-D4:

- **3 s -> 10 s.** 3 s is a quarter of the 12 s roll period; a linear predictor extrapolating a
  narrowband oscillation a quarter period ahead is not being tested. 10 s is where four of six
  channels fall below 0.8 skill, and it is the lead time a full-scale rotorcraft's commit-to-land
  decision needs. 15 s was rejected as the gate cell because pitch is saturated there (normalised
  RMSE 0.985), so it measures an impossible task rather than a hard one.
- **roll -> pitch.** Roll is lightly damped resonance (`Tn = 12 s`, `zeta = 0.06`), the most
  narrowband channel in the corpus. Pitch is stiffer and more damped, hence broader-band: normalised
  RMSE 0.50 at 5 s where roll is 0.20. Gating on the easiest channel measures the corpus, not the
  model.

The original criterion is preserved verbatim in the *Gate 3 outcome* section above and is recorded
there as **failed**; this entry does not retroactively make it pass. Recorded per CLAUDE.md §Gates,
which requires a gate change to be explicit rather than absorbed.

**Phase 4 is unblocked by this entry.**

### P3-D13 — `dlinear_mc` removed; `ar_attitude_only` replaced it, and its first form was itself confounded

`src/dmf/models/dlinear_mc.py` was added outside the plan's four baseline families to separate
"DLinear loses to AR because it sees fewer channels" from "DLinear loses because it is a worse
architecture". At the P3-D4 geometry it needed **2 161 800** parameters against DLinear's **60 300**
— a 36x capacity gap that confounded information set with capacity, which is the confound it
existed to remove. Removed.

**Its replacement repeated the same mistake at 2x, and that is worth recording.**
`ar_attitude_only` was first specified as AR(**20**) on the three attitude channels: 20 lags x 3
channels = 60 features = **54 900** parameters, against `ar20`'s 20 x 6 = 120 features =
**108 900**. The pair introduced to eliminate a capacity confound carried one, and the config
asserted "no capacity confound" while the parameter counts differed by exactly 2x. Caught by the
Gate 3 adversarial audit, not by review.

Corrected to **AR(40) on three channels**: 40 x 3 = 120 features = 108 900 parameters, identical to
`ar20`, same ridge, same solver, sliced from the same cached `p_max = 40` moments at no extra pass.

**The correction changes the conclusion; a second correction then narrows it further.** Median
skill differences over all 144 cells of each mode:

| contrast | what it isolates | ideal | imu |
|---|---|---|---|
| `ar20` - `ar_attitude_only` (both 108 900) | rate channels | +0.0046 | +0.0091 |
| `ar10` -> `ar20` (54 900 -> 108 900) | capacity | +0.0063 | +0.0053 |
| `ar20` -> `ar40` (108 900 -> 216 900) | capacity | +0.0017 | +0.0011 |

An earlier draft compared the first two rows directly and concluded the ordering flips by observation
mode — rate channels worth less than lags under `ideal`, more under `imu`. **That was wrong**, and it
is the exact error P3-D22 warns about: those are medians of two *separate* marginal distributions,
not a paired contrast. Per cell:

| | median(rate effect - lag effect) | cells where rate > lag |
|---|---|---|
| ideal | **+0.00060** | 85/144 |
| imu | +0.00252 | 95/144 |

**Positive in both modes.** The honest reading: under `ideal` the two effects are indistinguishable;
under `imu` the rate channels are worth somewhat more. There is no reversal.

**A confound remains, and it biases the channel effect downward.** Capacity and lag depth cannot both
be held on a lag-feature design. Matching parameters forced `ar_attitude_only` to AR(**40**) on 3
channels — a **4.0 s** lag window against `ar20`'s 6 channels over **2.0 s**. On a 12 s-period
narrowband signal that is not a nuisance. The direction is knowable: `ar_attitude_only` gets the
*longer* window, so `ar20 - ar_attitude_only` **understates** the pure channel effect, by roughly the
`ar20 -> ar40` step (+0.0017 ideal). Corrected, the channel effect is ~+0.0063 — about equal to the
lag effect, not below it. The pair is one information set apart *and* one lag depth apart; only the
parameter count is held.

**The earlier universal claim is withdrawn**, and so is its first replacement. What survives: at
matched parameter count the rate channels are worth a small positive amount in both modes (paired
median +0.0006 ideal, +0.0025 imu), of the same order as doubling the lag budget, and the measurement
still carries a downward bias from the unheld lag depth. This entry has now been wrong twice — first
from an `id`-only slice, then from unpaired medians — which is the argument for wiring
`paired_skill_difference_ci` (P3-D22) before Phase 4 makes claims of this size.

The earlier claim that `ar_attitude_only` "loses to `ar10` on every DOF and horizon" is also
withdrawn — it was false even at the old parametrisation (122/144 across regimes, not 144/144;
`heave_rate` @ 150 on ideal `id` had 0.63532 vs 0.63518), and at AR(40) it now beats `ar10` in most
cells, as a doubled feature count should.

Training MSE on `id`, for the record: ar10 0.30037, ar_attitude_only 0.29307, ar20 0.27801,
ar40 0.25835.

### P3-D14 — `src/dmf/data/channels.py` added

The `ideal` <-> `imu` channel correspondence was private to `dmf/data/dataset.py`, which imports
torch. `dmf/eval/report.py` needs it to resolve the gate cell's DOF across observation modes and is
a pandas-only rendering layer. Re-deriving the mapping there would have put the same fact in two
places — the failure mode that broke the test suite at this revision (below). `channels.py` is
torch-free, derives from `dmf.sim.imu.IDEAL_COLUMNS`/`IMU_COLUMNS`, and is now the single
definition; `dataset.py` imports it and its private copy is deleted.

The gate DOF is resolved by explicit alias pairing, never by an `endswith("_imu")` rule:
`roll_rate_imu` is the near-miss that a substring rule would mis-resolve to roll, and a test
asserts it does not.

### P3-D15 — Test suite no longer mirrors the production geometry

`tests/test_models.py` and `tests/test_windows.py` hand-mirrored the task geometry as module
constants — one comment read "mirrored from `configs/data/default.yaml`". The P3-D4 revision
silently broke 14 tests, which is the duplication failing exactly as designed to.

Both modules now derive from the config (`PRODUCTION_CFG = load_data(...)`,
`PRODUCTION_SPEC = window_spec_from_config(...)`, `N_IN`/`N_OUT` from its channel lists), and
`PRODUCTION_SAMPLES` derives from `configs/sim/corpus.yaml` rather than the literal 6000. Tests
whose subject is a *fixed* arithmetic invariant declare their own local spec (`ARITHMETIC_SPEC`,
`SYNTH_C_IN`/`SYNTH_C_OUT`) so they stay pinned and are immune to future task changes.

One consequence worth recording: `C_out < C_in` was load-bearing in the DLinear
channel-independence tests, which perturb `x[:, :, N_OUT:]`. Since P3-D4 makes
`target_dofs == input_channels`, that became an empty slice — one test silently vacuous, the other
silently failing. Both now run against an explicit proper-prefix geometry.

### P3-D17 — Training batch raised to 1024 and lr scaled; the bottleneck was never the data path

`docs/IMPLEMENTATION_PLAN.md` §Phase 4 gives training defaults of batch 256 and lr 1e-3.
`e01_baselines` now uses **batch 1024, lr 2e-3, num_workers 16**. Recorded because it is a
deviation from a stated default and because it changes the optimisation trajectory.

**Why: the first sweep attempt was 6x slower than budgeted.** One DLinear seed on `id` took 68
minutes at batch 256; the four-regime sweep extrapolated to ~15 h per observation mode, ~30 h for
both. The main process sat at 87% of a single core with the GPU oscillating 10-33%.

**A wrong diagnosis, recorded because the measurement is the useful part.** That profile was read
as per-window Python cost in `DeckMotionDataset.__getitem__`, and a vectorised `__getitems__`
batch-gather was written to remove it. It was verified bitwise-identical to the per-index path on
random batches, single-element batches, realization boundaries and the final partial batch — and
it bought **1.02x** (0.99-1.20x across batch sizes 256-4096). It was reverted. The data path was
never the constraint: the loader delivers 47 k windows/s at 8 workers and 185 k at 16, on ~2 of 36
cores.

**The actual constraint is fixed per-step overhead** — Python, kernel launches, the optimizer —
which for a 60 300-parameter linear map dwarfs the arithmetic. Measured on the production geometry
(A4000, bf16 autocast, H2D + forward + backward + clip + step):

| batch | ms/step | steps/epoch (`id`) | s/epoch |
|---|---|---|---|
| 256 | 6.85 | 5 726 | 39.2 |
| 1 024 | 7.55 | 1 431 | 10.8 |
| 4 096 | 8.03 | 358 | 2.9 |

Step cost is nearly flat in batch size, so 4x the batch is ~3.6x less wall time. With the loader at
7.9 s/epoch on 16 workers, the two overlap at ~11-13 s/epoch against ~68 before: roughly **5x**,
~13 min per seed, ~3 h per observation mode.

**The stated justification for changing lr was wrong, and is corrected here rather than edited
away.** It read: "DLinear under MSE is convex; its optimum is the closed-form least-squares
solution, and `test_dlinear_sgd_reaches_the_closed_form_optimum` asserts SGD reaches it. Batch and
lr therefore govern how fast it converges, not where — the oracle test is the guard."

The oracle test disclaims exactly that use in its own docstring: it runs on a **well-conditioned
synthetic task and deliberately not on the corpus**, because the corpus lag design is numerically
rank-deficient (`cond` past 1e16), so "SGD falls short of the closed-form optimum by orders of
magnitude *while working correctly*". It records that 120 epochs of Adam reach ~5e-2 against an
exact optimum of ~1e-6, still falling. Production runs 60. On a problem this ill-conditioned, under
a fixed epoch budget, *how fast* **is** *where* — so the oracle test is not the guard for this
change, and the convexity argument does not carry.

The batch/lr change itself stands: the measurements above are unaffected, and the wall-time
argument was never in doubt. What was unsupported was the claim that convergence quality was
unaffected. That question is now answered directly by P3-D19 rather than argued from a test that
does not apply. `lr = 2e-3` is square-root scaling for a 4x batch, chosen over
linear scaling's 4e-3 as the conservative option. 4096 was not taken despite being faster still:
at 358 steps/epoch the schedule has too few steps for warmup and cosine decay to mean much.

Applies uniformly to every SGD model in both `e01_baselines` and `e01_baselines_imu`, so the Gate 4
fairness requirement is unaffected. **Phase 4 must re-derive these numbers before adopting them** —
this measurement is for a 60 300-parameter linear map, and a TCN or Transformer will not be
fixed-overhead-bound in the same way.

### P3-D18 — The shuffle control's large negative excess is the P1-D2 floor, not an anomaly

On the full sweep the shuffle control's worst *negative* excess is -0.195 (`ideal`) / -0.215
(`imu`), against ~0.004 on the `id`-only slices measured earlier. Negative excess means the
shuffled model did **worse** than the window-mean null, which is the safe direction and passes a
one-sided criterion — but the size warranted an explanation rather than a shrug.

It is entirely concentrated in `unseen_heading` x {pitch, pitch_rate}. All ten of the largest
negative values are those cells; every other regime lies within +/-0.01, and `id` within +/-0.004,
matching the earlier slices exactly.

Same cause as the negative-skill cells in P3-D7. The `unseen_heading` test set *is* beam seas,
where the pitch heading factor is clamped at `eps = 0.05` (~26 dB down), so test-set pitch is
residual floor rather than pitch physics. The shuffled model was still fitted on 45/135/180 deg
training data where pitch is a real signal, so it carries a fitted amplitude and imposes it on a
signal that has none. The window-mean null has no fitted amplitude to get wrong, so it wins. The
control is measuring a genuine train/test amplitude mismatch, not a pipeline defect.

Worth stating because the earlier `id`-only measurement would have set a misleading expectation for
the tolerance: 2% is comfortable on `id`, and the control legitimately swings twenty times that far
in the safe direction on a held-out-heading cut. The failing direction — a shuffled model doing
*better* than the null, which is the only outcome that indicates leakage — did not occur in any of
the 288 rows across both modes.

### P3-D19 — `dlinear_ols` added: the DLinear optimisation gap is now measured, not assumed

Every `ar*` row is at the exact optimum of its ridge-regularised objective (closed-form Cholesky).
`dlinear` is wherever 60 epochs of AdamW landed. Comparing them conflated an optimisation gap of
unknown size with an architecture gap, and nothing committed recorded which. The wall-clock
signature showed `id` — the gate regime — was the one regime that hit the 60-epoch cap
(1.018 s per training realization against 0.83 for the other three), i.e. it stopped with the
validation loss still falling.

`DLinear(individual=False)` is a linear map, so its optimum is computable. `dlinear_ols` solves it
closed-form by streaming the decomposed design's normal equations on the **same per-regime pass**
the AR moments already use: one 400x400 Cholesky, 0.18 s on top of a 41 s pass. Both rows ship, so
the difference between them **is** the optimisation shortfall.

Measured on `id` (ideal), identical windows and `validate()` call:

| row | n_params | val MSE |
|---|---|---|
| `dlinear` (60 epochs AdamW, 3 seeds) | 60 300 | 0.432852 +/- 0.000007 |
| `dlinear_ols` (ridge 1e-6) | 60 300 | **0.431545** |

**0.30% of the loss — and it moves the conclusion.** The median AR(20)-over-DLinear advantage on
`id` falls from **+0.0194 to +0.0056**, and AR(20) wins **28 of 36 cells instead of 35 of 36**. Most
of the apparent DLinear-vs-AR architecture gap was the epoch budget. Per-cell the OLS row is up to
+0.0758 better (`pitch_rate` @ 100) and, in 5 of 36 test cells, *worse* — it is the exact optimum on
train/val, which carries no guarantee on held-out realizations.

The tiny seed spread (+/- 7e-6) is not evidence against this: three seeds under the same epoch
budget on a convex problem stop in the same place. `skill_std` measures initialisation and shuffle
noise; the optimisation shortfall is systematic and invisible to it.

**The `epochs_run` column P3-D23 added shows the budget bound almost everywhere.** Under `ideal`,
all twelve DLinear runs report `epochs_run = 60` — early stopping never fired. Under `imu` it fired
in **2 of 12** runs (45 on `id`/seed 0, 43 on `unseen_heading`/seed 2); the other ten also hit the
cap. So the budget bound in 22 of 24 runs overall. The consequence is visible in the tables:

| | ideal | imu |
|---|---|---|
| `dlinear_ols` - `dlinear`, median | **+0.0047** | +0.0000 |
| cells where OLS wins | **110/144** | 78/144 |
| `ar20` beats `dlinear` | 107/144 | 102/144 |
| `ar20` beats `dlinear_ols` | **89/144** | 102/144 |

Under `ideal` the closed-form row accounts for 18 of the 107 cells in which AR(20) appeared to beat
DLinear; under `imu` it changes nothing (102 both ways). **Why the gap is mode-dependent is not
explained.** The obvious hypothesis — that `imu` runs converged and `ideal` runs did not — is refuted
by `epochs_run`: the cap bound in 10 of 12 `imu` runs too. A likelier cause is that IMU observation
noise raises the irreducible error and flattens the loss basin, so the remaining distance to the
optimum costs less skill; that is untested. Recorded as an open question, not a finding. Any
DLinear-vs-AR statement must name the mode.

The per-cell spread is wide in both directions (`ideal` min -0.3496, max +0.1270): `dlinear_ols` is
the exact optimum of the regularised *training* objective, which carries no guarantee on held-out
realizations, and in the residual-floor cells it overfits a signal that is mostly floor.

**A second finding, unexpected.** The decomposed design is rank-deficient *by construction* —
`trend = Ax` and `remainder = (I - A)x`, so `[trend, remainder]` spans rank `L`, not `2L`; measured
`cond(R) = 6.0e19`. At `ridge = 0` the unregularised minimiser is reachable and useless: train MSE
0.3182, **validation MSE 2539.8**, the lstsq fallback retaining directions at the rcond cutoff and
amplifying them by ~1e10. At `ridge = 1e-6`: train 0.3761, val 0.4315. So `dlinear_ols` reports the
exact optimum of the *regularised* objective, the same treatment every `ar*` row receives. Stated
here rather than glossed, because "closed-form optimum" without the qualifier would be false.

### P3-D20 — `window_mean` promoted to a baseline row; it beats persistence in 107 of 144 cells

`DampedPersistence(tau -> 0+)` was instantiated in every run as the shuffle control's null (P3-D8)
and its skill recorded only in `baselines_controls.csv`. Measured there on the completed 7-model
sweep, it beats:

| beaten by `window_mean` | ideal | imu |
|---|---|---|
| `persistence` (the skill denominator) | **107/144** | 107/144 |
| `damped_persistence` (fitted, `n_params = 6`) | 54/144 | 55/144 |
| `ar20` | 16/144 | 19/144 |
| `dlinear` | 6/144 | 17/144 |

A zeroth-order baseline beating the reference in 74% of cells, and beating a fitted model in 37%,
while living only in a controls file is CLAUDE.md non-negotiable 6 inverted — not an underperforming
model dropped, but an over-performing trivial one kept out of the table. It is now `window_mean`, a
ninth row in `baselines.csv`.

This also gives P3-D5 its practical remedy. Persistence is a treacherous reference on a narrowband
signal because its error tracks the autocorrelation and is non-monotone in horizon; the window mean
is monotone and is the honest "did the model learn anything beyond the window's level" bar at the
horizons where persistence collapses.

`damped_persistence` is left at one `tau` per channel for all 150 horizon steps (`n_params = 6`) in
this phase. Its fitted `tau` on `id` is 9.3-15.5 samples (0.93-1.55 s), and beyond 100 samples it
sits within 1e-4 skill of its own `tau -> 0` limit, which is where the 54 near-ties come from. A
per-horizon `tau` would be strictly better and is still trivially cheap; not changed here so the
comparison against the completed sweeps stays like-for-like.

### P3-D21 — the gate cell in the generated report, and the report's own provenance

Two defects in `baselines.md`, both found by the Gate 3 audit, both of which made the committed
artifact contradict the repository around it.

**The report declared the superseded gate.** It led with `Gate cell: roll at 3 s (30 samples)` and
showed AR(20) at 0.9987 — the *original* criterion, failing by 0.20 — while `README.md` said the
gate was restated and passing. `build_baselines_markdown` defaulted to roll/30 and `run_experiment`
never passed anything else. The gate cell is now `GATE_DOF = "pitch"` / `GATE_HORIZON_SAMPLES = 100`
as named constants in `dmf.train.experiment`, citing P3-D12, and `_gate_horizon` resolves to 100 on
the production horizon list.

**The `imu` report cited the `ideal` run's files.** Its provenance line named
`results/baselines.csv` and siblings, which are the other mode's artifacts — pointing a reader at
exactly the files the same document's cross-mode caveat warns against mixing. The line is now
generated from the directory the run actually wrote to. Its default when the directory is unknown
is a bare filename located "beside this document", deliberately not the old `results/` literal: a
forgotten argument now degrades to a true-but-vaguer statement rather than a false one.

Same class, also fixed: `BASELINES_CAVEATS` rendered "441 984 windows" into every artifact, a count
superseded by P3-D6 (434 304). The window count is now derived from the table being rendered, and
the remaining geometry literals in `report.py` and `runner.py` docstrings — `384 x 50 x 3`,
"28 evaluation passes (7 models x 4 regimes)" — are stated structurally. A source-level test now
fails if a live line asserts a window count.

### P3-D22 — `skill_ci_lo`/`skill_ci_hi` on multi-seed rows was the mean of intervals; now the envelope

`build_baselines_table` aggregated the per-seed bootstrap intervals with `mean`. The mean of three
intervals is not an interval for anything: it excludes seed variance, and — the reason it matters —
it is **invariant to seed disagreement**, which is the one thing a reader would consult it to
detect.

Now aggregated as `min`/`max`, the envelope of the per-run intervals. For `n_seeds = 1` that is
bitwise the per-run interval, so every deterministic row keeps a genuine realization bootstrap CI,
including the gate row. For multi-seed rows it is conservative and monotone in seed disagreement,
and a caveat renders beside the table stating that it is an envelope, not a calibrated interval for
the seed mean, pointing at `baselines_by_seed.csv` for the per-run intervals.

Measured on the 7-model sweep's 144 `dlinear` cells: the envelope is a median 7% wider than the
mean-of-intervals and up to **16.5x** wider in the `unseen_heading` @ 10-sample cells where the
seeds genuinely disagree. That widening is the finding the mean was suppressing.

A properly calibrated seed-mean interval would have to pool the bootstrap *resamples* across seeds,
which needs the per-realization SSE tensors at scoring time — a re-run, not an aggregation, so it
cannot be recovered in `report.py` from finished intervals.

**Related, implemented but not wired.** `dmf.eval.runner.paired_skill_difference_ci` computes a
paired bootstrap of `skill(a) - skill(b)` over common realization resamples. Model-vs-model claims
in this protocol — P3-D13's rate-channel effect above all — are currently judged by eye against two
*unpaired* marginal intervals, which is far too loose for effects of this size on models scored over
identical realizations with strongly correlated errors. Choosing which pairs to report is a
`run_experiment` decision and was not taken in this pass.

### P3-D23 — DLinear's optimisation state is now recorded

`FitResult` carried `best_epoch`, `epochs_run` and `best_val_loss`; `run_experiment` read only
`wall_time_s` and discarded them, and the sweep logs are empty. That is why P3-D19's finding had to
be inferred from wall-clock ratios rather than read off a column. All three now ship as columns on
`baselines_by_seed.csv` (NaN for closed-form rows), which is the traceability both experiment configs
already claimed to provide when they described "one early-stopping rule" as a fairness guarantee.

### Gate 3 evidence

Final sweep, nine models: `ideal` 2026-08-28 09:11 -> 12:50 (exit 0), `imu` 12:50 -> 16:20 (exit 0).
Artifacts in `results/` and `results/imu/`, five files each. Supersedes the seven-model sweep of
2026-08-27, which was regenerated after the audit findings recorded in P3-D19 .. P3-D23.

| Criterion | Evidence |
|---|---|
| `baselines.csv` covers every baseline / horizon / DOF / regime | **1296 rows per mode** = 4 regimes x 9 models x 6 DOFs x 6 horizons; schema equals `BASELINES_COLUMNS` |
| Gate cell read and acted on | Original gate (roll @ 3 s, `id`) **failed** at 0.9987 / 0.9958 against 0.8; task revised (P3-D4); gate restated to pitch @ 10 s and applied (P3-D12). Final: **AR(20) 0.5446 (`ideal`) / 0.5132 (`imu`)** — passes |
| Reference is exact | Persistence **bitwise 0.0** in all 144 rows of both modes; `rmse_persistence` identical across all nine models in every cell |
| Not attributable to leakage | Shuffle control **0 rows in the failing direction of 288**; worst positive excess 0.0095 (`ideal`) / 0.0064 (`imu`) against a 2% tolerance. Large negative excursions explained in P3-D18 |
| Denominators correct | `ideal` reproduces the P3-D6 table to 4.7e-07 and is pinned in `tests/test_models.py`; the `imu` table is recorded in P3-D6 and reproduces the sweep to 5.0e-07 but is **not** pinned by a test |
| Underperforming cells reported | AR(20) negative in **12/144** (`ideal`), **13/144** (`imu`), matching the sets pinned in `tests/test_models.py` |
| Per-cell table decomposes the headline | Joins to `baselines.csv` with **0 unmatched of 47 520**, both modes |
| Optimisation state traceable | `epochs_run` / `best_epoch` / `best_val_loss` on `baselines_by_seed.csv` (P3-D23); this is what localised P3-D19 |
| Normalisation provenance | Asserted at scoring time, fails closed (P3-D11) |

### What the gate does *not* say

Gate 3 passing is one cell of 36, and the honest reading needs the surrounding count. On the final
tables, AR(20) on `id`:

- exceeds 0.8 skill in **28 of 36 cells**, in both modes;
- at the gate's own 10 s horizon exceeds 0.8 on **roll (0.926) and roll_rate (0.883)** — the gate
  passes only because the restatement names pitch, which is the argmin at that horizon;
- across the **1-5 s operational band** (`CLAUDE.md` §1, P3-D4) exceeds 0.8 in **89 of 96 cells**
  (`ideal`) and **87 of 96** (`imu`) over all four regimes. The exceptions are entirely the
  `unseen_heading` pitch/pitch_rate residual-floor artifact (P1-D2), not difficulty.

**The task is easy by construction (P3-D1). The gate documents where it stops being easy, not that
it is hard.** Any sentence claiming Gate 3 passes must carry that, or it misrepresents the corpus.

The trivial-baseline result belongs here too: `window_mean`, with **zero parameters**, beats
persistence — the denominator of every skill score in the project — in **107 of 144 cells** in both
modes (P3-D20). Skill vs persistence is the metric the plan mandates, and on this signal the
reference is weak in three quarters of the table.

### Phase 3 findings worth carrying into Phase 4

1. **A converged linear model is competitive with AR and beats it on parameters.** At the gate cell,
   `dlinear_ols` scores 0.5683 with 60 300 parameters against `ar20`'s 0.5446 with 108 900 and
   `ar40`'s 0.5765 with 216 900. This is the plan's §0.3 prediction — "include a linear model you
   might lose to" — landing, and it was invisible until P3-D19 removed the optimisation gap. It does
   **not** hold under `imu`, where `dlinear_ols` scores 0.4762 against `ar20`'s 0.5132.
2. **Deep models must be selected on the operational metric, not `id` RMSE.** 28 of 36 `id` cells are
   already above 0.8 skill; the discrimination lives in the OOD regimes and in quiescence detection,
   which is Phase 6 work that should move ahead of Phase 4.
3. **The training budget is not adequate and must be re-derived.** All twelve `ideal` DLinear runs hit
   the 60-epoch cap. A TCN or Transformer on a `cond`-1e19 design will not be fixed-overhead-bound the
   way a 60 300-parameter linear map is (P3-D17), and `epochs_run` must be checked rather than assumed.

---

## Phase 4 — deep models

Entries P4-D1 .. P4-D7 were written **before** the Gate 4 sweep completed, so that the gate
restatement and the training budget are on record as decisions rather than as
rationalisations of a result. The Gate 4 evidence section is added when the sweep lands.

### P4-D1 — Gate 4 restated: threshold unchanged, cell moved. RECORDED 2026-08-28

`docs/IMPLEMENTATION_PLAN.md` §Phase 4 states:

> **Gate 4:** all deep models beat damped persistence at 3 s horizon on the `id` regime by a
> margin exceeding the seed-to-seed standard deviation.

**The cell is saturated and the reference is weak**, which are two independent problems and
both were measured in Phase 3:

- *Saturated.* AR(20) scores 0.9987 skill at 3 s on roll in `id` (Gate 3 outcome), and across
  the whole `id` regime exceeds 0.8 skill in 28 of 36 cells. P3-D1 records why: the response
  is narrowband with no process noise, and 3 s is a quarter of the 12 s roll period. A model
  can pass this cell having learned nothing that discriminates it from a linear filter.
- *Weak reference.* `damped_persistence` is beaten by the **zero-parameter** `window_mean` in
  54 of 144 cells (P3-D20). "Beats damped persistence" is therefore not a floor that means
  what it sounds like — in 37% of cells it is a lower bar than beating a constant.

**Gate 4, as of this entry:**

> All deep models beat the **strongest trivial baseline in the cell** — the better of
> `damped_persistence` and `window_mean`, resolved per cell rather than assumed — at the
> **decision horizon (10 s / 100 samples) on the binding DOF (pitch)** in the `id` regime, by
> a margin exceeding the seed-to-seed standard deviation.

The margin criterion is unchanged. What moves is the cell, by the same two arguments P3-D12
used for Gate 3 and for the same reasons: 10 s is where four of six channels fall below 0.8
skill and is the lead time a full-scale rotorcraft's commit-to-land decision needs; pitch is
stiffer and more damped than the lightly-damped roll resonance, hence broader-band, hence the
argmin at that horizon. 15 s was rejected as the gate cell in P3-D12 because pitch is
saturated there (normalised RMSE 0.985), and that rejection stands.

**The original cell is measured and reported anyway**, per the user instruction that a model
failing to beat damped persistence by more than the seed std is itself the result. Both
readings ship. This entry does not retroactively make the original criterion pass or fail;
it records which one the gate is read at.

**What `skill_std` actually measures. THIS ENTRY ORIGINALLY GOT THIS WRONG; see P4-D13.**
`skill_std` is a ddof=1 standard deviation over exactly three seeds, and it includes both weight
initialisation and data-order variance. An earlier version of this entry asserted that batch
order was identical across seeds and that the spread was initialisation variance alone. That was
false, and the correction is recorded in P4-D13 rather than silently edited away, because the
claim was also rendered into `results/e02/gate4.md` and read by a reviewer.

### P4-D2 — The training budget was re-derived; P3-D17's conclusion does not transfer

Required by the Phase 4 carry-forward: "do not inherit the budget". P3-D17 measured DLinear at
6.85 / 7.55 / 8.03 ms/step for batch 256 / 1024 / 4096 and concluded step cost is nearly flat in
batch size, so 4x the batch buys ~3.6x less wall time. **That is a property of a 60 300-parameter
linear map that is fixed-overhead-bound, and it is false for all three deep models.**

Measured on the production geometry (A4000, bf16 autocast, H2D + forward + backward + clip +
step, `torch.cuda.synchronize()` on both sides of the timed region):

| model | n_params | 512 | 1024 | 2048 | s/epoch @1024 (step only) |
|---|---:|---:|---:|---:|---:|
| `dlinear` | 60 300 | 2.29 | 2.44 | 2.87 | 3.5 |
| `tcn` | 196 804 | 20.34 | 38.60 | 75.24 | 55.2 |
| `transformer` | 2 712 708 | 12.27 | 18.07 | 34.17 | 25.9 |
| `lstm` | 317 828 | 10.16 | 17.93 | 35.26 | 25.7 |

(ms/step.) For the deep models ms/step scales almost exactly **linearly** with batch, so
s/epoch barely moves — `tcn` goes 58.2 -> 55.2 -> 53.8 s/epoch across a 4x batch range. Batch
size is close to free for wall time here, so it is chosen for continuity with the Phase 3 sweep
rather than for throughput: **batch 1024, lr 2e-3**, unchanged from `e01_baselines`.

**Step timing understates epoch cost by ~40%.** An 18-epoch pilot on `id` (seed 0, one run per
architecture, batch 1024, lr 2e-3) measured wall time including validation and the loader:

| model | s/epoch measured | vs step-only | best_epoch | best val MSE | curve |
|---|---:|---:|---:|---:|---|
| `tcn` | 78.8 | +43% | **17 / 18** | 0.132155 | falling monotonically |
| `transformer` | 46.2 | +78% | 16 / 18 | 0.130390 | near plateau, val bouncing |
| `lstm` | 57.6 | +124% | **17 / 18** | **0.094530** | falling monotonically |

**Epoch cap 60, and early stopping is expected not to fire.** Two of three models have their
best epoch at the *last* epoch of the pilot and are still improving; `patience: 15` requires 16
consecutive non-improving epochs and nothing is plateauing. `epochs_run` is recorded per run
(P3-D23) and the outcome is reported rather than presented as convergence.

The cap is nonetheless a real hyperparameter and not merely a stopping point: `dmf.train.loop._lr_at`
sizes the warmup+cosine schedule by `epochs * steps_per_epoch`, so a run at cap 60 anneals to ~0
by epoch 60, whereas the same 60 epochs drawn from a cap-120 schedule would stop mid-decay.
Raising the cap therefore changes the trajectory of runs that stop early, not only of runs that
hit it. 60 matches the Phase 3 nominal cap, which keeps the re-fitted `dlinear` row directly
comparable to the P3-D19 measurement of the same model at the same cap.

**Direction of the residual bias, since it cannot be removed within the budget.** The deep
models are still improving when the cap bites, so their reported numbers **understate** them.
For the question Gate 4 asks — do the deep models beat the trivial and linear baselines — that
is the conservative direction. `dlinear_ols` is closed-form and therefore budget-independent, so
the strongest linear baseline in the table cannot be handicapped by this choice at all.

**Weight decay is applied to LayerNorm and bias parameters**, because `fit` gives AdamW one flat
parameter group with no exclusions. This is conventionally suboptimal for the Transformer.
It is left alone deliberately: excluding them for the Transformer and not for the others is
per-model tuning, which the Gate 4 fairness requirement forbids and which the task explicitly
ruled out.

### P4-D3 — `nrmse` and `signal_std` implemented; P3-D5 closed in the source, not in the artifacts

P3-D5 mandated normalised RMSE twice and recorded its own status as "recorded, not implemented".
It is now implemented.

`signal_std[h, c] = sqrt(syy/n - (sy/n)**2)` — the **population** standard deviation of the
target at lead time exactly `h`, per channel, over the same `n` windows the row's RMSE is
averaged over, in corpus units (deg, m, deg/s, m/s). `nrmse = rmse / signal_std`, dimensionless,
so `nrmse == 1.0` is "no better than the partition mean" and lower is better. A zero or
non-finite `signal_std` raises, following `skill_score`'s zero-denominator precedent, and the
check runs only over the *reported* cells so a constant channel at an unrequested horizon does
not fail a table that never quotes it.

Three implementation facts worth recording:

- **Accumulated once per batch, not once per model, and it is checked by identity.**
  `evaluate_models` builds one `(n_keys, H, C_out)` float64 pair before the model loop and hands
  the same two tensor objects to every model's accumulator, then raises if any accumulator does
  not hold those objects by identity afterwards. Twelve models never sum the same float64 values
  in twelve reduction orders and disagree in the last bits.
- **The by-cell table stores raw `sy`/`syy`, not just derived columns.** `marginalize_cells`
  re-derives `signal_std` from summed moments, because the standard deviation of a union of grid
  cells is not the mean of their standard deviations — on a grid spanning four sea states it is
  much larger.
- **The streaming form loses accuracy in proportion to `(mean/std)**2`.** It agrees with a
  two-pass NumPy `std` to 1e-12 relative on a zero-mean channel and to ~4e-9 on a channel offset
  to 2000 standard deviations; both tolerances are pinned separately rather than hidden under one
  loose bound. Harmless for this corpus — all six channels oscillate about zero — but a future
  target with a large DC offset (an absolute position, a heading in degrees) needs a two-pass or
  Welford form.

**What is NOT fixed, and the wording of P3-D5 needs this correction.** P3-D5 says "this is a
column, not a re-run". That is true prospectively only. Every committed Phase 3 artifact
(`results/*.csv`, `results/imu/*.csv`) predates the column and still carries the pre-nrmse
19-column header. `run_experiment` must re-fit every model to produce a scored table, so
populating those columns for the completed nine-model sweep **is** a re-run of that sweep. The
consequence: the normalised-RMSE figures quoted in P3-D3, P3-D4 and P3-D13 — and the 15 s
rejection in P3-D12 that rests on one of them — remain unreproducible from `results/`. The
Phase 4 sweep re-scores all nine baselines beside the three deep models and therefore carries
`nrmse` for all twelve rows; the Phase 3 directories are left as the Gate 3 record and are not
regenerated.

**Which signal std, since P3-D5 does not say.** The denominator is the standard deviation of the
partition actually being scored, per (lead time, channel) — not a training-split scale and not a
corpus-wide constant. Within a regime this makes `nrmse` comparable across horizons and models,
which is what P3-D5 asked for. Across regimes it compares two different denominators. That is
the *desirable* reading for the P3-D7 `unseen_vessel` case — each hull's error as a fraction of
that hull's own variability is exactly what removes the collapsing-persistence artefact — but it
is not a like-for-like "same signal" comparison, and it is the same class of caveat as `ideal`
vs `imu`.

### P4-D4 — `paired_skill_difference_ci` wired; P3-D22's open item closed

P3-D22 recorded the function as "implemented but not wired", and P3-D13 records a conclusion
published wrong **twice**, the second time from reading two unpaired marginal medians as a
paired contrast. `run_experiment` now computes 13 ordered contrasts per regime and writes
`paired_contrasts.csv`:

| pairs | claim supported |
|---|---|
| {`tcn`,`transformer`,`lstm`} x `ar20` | against the strongest fitted baseline / the Gate 3 subject |
| {`tcn`,`transformer`,`lstm`} x `dlinear_ols` | against the *converged* linear map, so no deep model is credited with the P3-D19 optimisation gap |
| {`tcn`,`transformer`,`lstm`} x `damped_persistence` | the reference Gate 4 names |
| {`tcn`,`transformer`,`lstm`} x `window_mean` | against the zero-parameter forecast that beats persistence in 107/144 cells (P3-D20) |
| `dlinear` vs `dlinear_ols` | the DLinear optimisation gap re-read at this epoch budget |

`skill_diff = skill(a) - skill(b)`, positive meaning `model_a` is better, stated in the schema
rather than left to be inferred. Pairing is seed-by-seed; where one side is deterministic, every
seed of the stochastic side is paired against its single `@0` accumulator. **No aggregated view
is emitted at all**, so there is no mean-of-intervals to misread — the defect P3-D22 corrected
in `build_baselines_table` is not reintroduced here.

The exactness is structural: `BOOTSTRAP_N_BOOT`, `BOOTSTRAP_CI_LEVEL` and `BOOTSTRAP_SEED` are
passed explicitly to *both* `evaluate_models` and `paired_skill_difference_ci`, and a test
asserts they still equal the runner's defaults, because `_bootstrap_counts` is a pure function
of `(n_realizations, n_boot, seed)` and the pairing holds only while both sides draw identical
multinomial counts. No second scoring pass and no re-drawn bootstrap — one matrix product per
(pair, seed).

Measured on the test fixture, the property the wiring exists for: paired interval width is a
median 0.375 of the unpaired `width(a) + width(b)`, the marginals overlap in 15 of 18 cells
while every paired interval excludes zero.

**One correction to P3-D22.** It states that a calibrated seed-mean interval "needs the
per-realization SSE tensors at scoring time — a re-run, not an aggregation, so it cannot be
recovered in `report.py`". True of `report.py`; **not** true of `run_experiment`, where at
contrast time those tensors are in hand and a seed-pooled interval is one more matrix product
away. Not built — it is a different quantity from the paired contrast and was not in scope — but
the "needs a re-run" framing holds only for the reporting layer.

### P4-D5 — The Phase 4 table is not budget-matched, and the Transformer head is why

Parameter counts at the production geometry (L=200, H=150, C_in=C_out=6), from
`n_fitted_parameters`:

| model | n_params | of which head | head share |
|---|---:|---:|---:|
| `dlinear` | 60 300 | — | — |
| `ar20` | 108 900 | — | — |
| `tcn` | 196 804 | 58 500 | 30% |
| `lstm` | 317 828 | 116 100 | 37% |
| `transformer` | **2 712 708** | **2 304 900** | **85%** |

The Transformer is 14x the TCN and 45x DLinear, and essentially all of that is the
flatten-and-project head the plan specifies (§Phase 4): `Linear(20*128 -> 150*6)`. It ships as
specified. A pooled or last-token head would cut it to ~0.4 M and make the column look tidy,
but it would change the architecture being compared, and shrinking the one model most likely to
lose is indistinguishable from tuning it. The spread is reported beside the table instead.

This project has now confounded capacity with its intended variable twice — the removed
`dlinear_mc` at 36x, and `ar_attitude_only`'s first parametrisation at 2x (P3-D13) — so the
spread is stated explicitly rather than left for a reader to compute.

**The TCN head reads the last encoded timestep only**, giving 58 500 head parameters instead of
the 11.5 M a flatten of `64 x 200` would cost. That is not a saving trick: because the receptive
field (253) already covers the lookback (200), the final step is a function of every input
sample, so flattening all 200 steps multiplies the head by 200 without adding information. The
receptive-field precondition is what makes the cheap head the correct one.

### P4-D6 — Two of the three deep models are not bitwise reproducible, and it is not a defect

`set_seed` calls `torch.use_deterministic_algorithms(True, warn_only=False)` nowhere; it uses
`warn_only=True`. Two consequences observed during the pilot:

- the **cuDNN LSTM backward** is nondeterministic and warns rather than raises;
- the **Flash Attention backward** is nondeterministic and warns rather than raises
  (`aten/src/ATen/native/transformers/cuda/attention_backward.cu`), which affects the Transformer.

So `lstm` and `transformer` carry a little run-to-run variance that `tcn` and `dlinear` do not.
This is not a correctness problem — three seeds are reported, and CLAUDE.md non-negotiable 5 is
satisfied — but the three-seed spreads are not measuring quite the same quantity across rows.
Every SGD row's `skill_std` covers initialisation and data-order variance (P4-D13); those two
models add kernel nondeterminism on top. Since the criterion compares each model's margin against
*its own* spread, the noisier model faces the harder bar.
Forcing `warn_only=False` would make the LSTM raise rather than run, so it is not an option
without changing the architecture set.

### P4-D7 — Stale window-count literals, and a guard that could not see them

Three live sentences in `src/dmf/eval/` asserted a window count that P3-D6 superseded — the
pre-revision 441 984, rounded to "~442 000" — together with a "half a gigabyte" array size that
is wrong by 6x at `max_horizon = 150` (the array is ~3.1 GB). Found in `dmf/eval/__init__.py`
and two passages of `dmf/eval/metrics.py`. All three are now structural statements with no
literal.

The interesting part is why they survived. `test_no_hardcoded_window_count_survives_in_the_reporting_modules`
exists precisely to catch this — it was added by P3-D21 — but it iterated a **hardcoded
two-module list**, `(report.py, runner.py)`, and so never looked at `__init__.py` or at
`metrics.py`'s docstrings. The guard reproduced in miniature the duplication defect it exists to
catch, which is the same failure P2-D5 and P3-D15 record in other places. It now globs
`src/dmf/eval/*.py`, asserts the glob found the subpackage, and forbids **today's** count
(434 304) as firmly as yesterday's, because the next horizon change moves it too. Lines citing
a decision record stay exempt: naming a superseded number *as history* is the opposite of the
defect.

### P4-D8 — The Gate 4 read-out is repo code, and it reports two questions rather than one

`src/dmf/eval/gate.py` + `scripts/gate4.py` (`make gate4`) compute the gate from the committed
artifacts, so the gate number is reproducible rather than the output of an analysis script that
was never committed. It writes `gate4.csv` and `gate4.md` beside the sweep's other tables. Both
readings of P4-D1 are always computed and written; the process exit status is taken from
Reading B, which is the reading the gate is read at.

**`verdict` and `paired_verdict` are separate columns and are never merged.** The margin test
("does the model beat the reference by more than its own seed spread?") and the paired bootstrap
("does the realization-resampled skill difference exclude zero?") are different questions with
different failure modes, and a margin can exceed the seed std while the paired interval still
spans zero. Collapsing them into one verdict would hide exactly the disagreement that is worth
seeing. `reference_pool` renders every candidate's skill in the cell, so P4-D1's
"stronger of `damped_persistence` and `window_mean`" resolution is auditable from the row rather
than trusted.

**A NaN `skill_std` is `UNVERIFIED`, never `PASS`.** Deterministic rows carry `skill_std = NaN`
by design (P3-D10); filling that with 0.0 would make every positive margin "exceed" it and turn
the criterion into "is the margin positive". A deep model arriving with a NaN std, or with fewer
than three seeds, is an error condition surfaced as `UNVERIFIED`: the row stays in the table body
(non-negotiable 6), the reading does not pass, and the CLI exits non-zero. `margin == std` is
`FAIL` — the criterion says *exceeding*, which is strict.

**Four caveats the read-out prints in its own Notes section**, because three are properties of
the artifacts rather than of the gate code and would otherwise be invisible at the point of use:

1. `skill_std` covers initialisation **and** data-order variance (P4-D13), so the margin test
   is against the full three-seed spread rather than a narrower one.
2. `skill_std` is not the same quantity in every row (P4-D6): `lstm` and `transformer` carry
   kernel nondeterminism that `tcn` and `dlinear` do not. Since the criterion compares each
   model's margin against *its own* spread, the noisier model faces the harder bar.
3. **The three per-seed paired intervals against a deterministic reference are correlated, not
   independent.** `_pair_seeds` pairs every stochastic seed against the single `@0` deterministic
   accumulator and `BOOTSTRAP_SEED` is fixed, so all three seeds draw identical multinomial
   weights against an identical reference. "Every seed excludes zero" is three correlated
   statements. The envelope reduction (min `ci_lo`, max `ci_hi`, mean `skill_diff`) is the
   conservative choice and follows P3-D22's precedent, but it is not three-fold evidence and must
   not be read as one.
4. `nrmse_mean` is absent from any artifact written before P4-D3. The read-out degrades to a NaN
   column and says so, rather than printing an empty column or failing.

### P4-D9 — P4-D2's prediction was wrong: early stopping fired, and the larger cap made two models worse

P4-D2 stated, from the 18-epoch pilot, that "early stopping is expected not to fire" because two
of three models had their best epoch at the last pilot epoch and were still improving. **On the
sweep it fired for two of the three deep models.** Recorded as a correction rather than edited
into P4-D2, because the reasoning that produced the wrong prediction is the useful part.

Measured on `id`, cap 60, batch 1024, lr 2e-3, patience 15:

| model | `epochs_run` | `best_epoch` | outcome |
|---|---|---|---|
| `dlinear` | 60 / 60 / 60 | 59 / 59 / 58 | cap bound |
| `tcn` | 60 / 60 / 60 | 59 / 59 / 58 | cap bound, still improving |
| `transformer` | **33 / 29 / 41** | 16 / 12 / 24 | early stopped |
| `lstm` | **33 / 32 / 34** | 16 / 15 / 17 | early stopped |

**Why the pilot mispredicted it, and it is the cap-as-hyperparameter mechanism P4-D2 itself
described.** `_lr_at` sizes the warmup+cosine schedule by `epochs * steps_per_epoch`. At the
pilot's cap of 18 the learning rate annealed to ~0 by epoch 18, so validation loss fell
monotonically to the last epoch and nothing plateaued — which is exactly what "still improving,
so a larger cap will not early-stop" was read off. At cap 60 the same epochs are drawn from a
schedule three times longer, so at epoch 16 the learning rate is still near its peak, validation
loss plateaus and bounces, and patience 15 fires. **The pilot could not have predicted this,
because the quantity it measured is not invariant to the cap it was measured at.** A budget pilot
must be run at the cap it is being used to justify, or it measures a different optimisation
problem.

**The consequence is that the larger cap made two of the three models worse.** Best validation
loss on `id`, seed 0 against seed 0, so the comparison is exact:

| model | pilot, cap 18 | sweep, cap 60 | change |
|---|---:|---:|---|
| `tcn` | 0.132155 | **0.120901** | 8.5% better — it used all 60 epochs |
| `transformer` | 0.130390 | 0.134548 | **3.2% worse** |
| `lstm` | **0.094530** | 0.101515 | **7.4% worse** |

The two models that early-stopped did so mid-anneal, checkpointing a best epoch reached while the
learning rate was still high, and never got the low-lr refinement the cap-18 pilot gave them. So
the shipped configuration is demonstrably *not* the best available for `transformer` and `lstm`,
and this project knows it.

**The cap was not changed, and that is deliberate.** Lowering it now — after measuring that a
lower cap helps two specific architectures and hurts a third — would be selecting a
hyperparameter on the results it produces, which is the failure the Gate 4 fairness requirement
and CLAUDE.md non-negotiable 6 both exist to prevent. It would also be per-architecture tuning in
all but name, since one cap cannot be simultaneously raised for `tcn` and lowered for the other
two. The rule as written — one cap, one schedule policy, one stopping rule, chosen in advance and
recorded before the run — is honoured, and its cost is reported here.

**Direction of the resulting bias, which is not uniform across the table.** P4-D2 stated that the
budget understates every deep model equally. That is now known to be wrong: `tcn` is understated
(cap bound, still improving), while `transformer` and `lstm` are understated *differently* — not
by a truncated budget but by an unfavourable interaction between a long cosine schedule and an
early stop. Any statement of the form "architecture X beats architecture Y" in this phase carries
that asymmetry. It does not threaten the Gate 4 verdict, which all three models clear by two
orders of magnitude more than their seed spread, but it does bound how finely the three deep
models can be ranked against *each other*.

**What a Phase 5 that wanted a fair architecture ranking would have to do** — stated because
Phase 5 selects "the best point model from Phase 4": decouple the schedule length from the
stopping criterion, either by sizing the cosine over a fixed horizon independent of the cap, or
by disabling early stopping and reporting the whole curve. That is a change to `dmf.train.loop`
and to every committed comparison, so it is not made mid-phase.

### P4-D10 — The re-scored Phase 3 rows reproduce Phase 3 exactly

The twelve-model design (P4-D2, `e02_deep.yaml`) re-fits and re-scores the nine baselines rather
than joining to the committed Gate 3 tables. That makes the Phase 3 numbers a **positive control
on the whole pipeline**, and they land:

| quantity | Phase 3 record | Phase 4 re-run (`id`) |
|---|---|---|
| `dlinear` best val loss, 3 seeds | 0.432852 +/- 0.000007 (P3-D19) | 0.432844 / 0.432857 / 0.432855, mean **0.432852** |
| `ar20` skill, pitch @ 100 samples | 0.5446 (P3-D12, the Gate 3 cell) | **0.5446** |
| `dlinear_ols` skill, same cell | 0.5683 (P3 carry-forward 1) | **0.5683** |

Nothing about the corpus, the splits, the normalisation or the closed-form solvers moved under
the `nrmse` and paired-contrast changes. `dlinear` also reproduces P3-D19's finding that the cap
binds: `epochs_run = 60` in all three seeds, `best_epoch` 58-59.

**`nrmse` passes its definitional check on real data.** At `id`, pitch, 100 samples:
`window_mean` scores `nrmse = 1.0005` and `damped_persistence` 1.0005 — both are essentially the
partition-mean forecast at that lead, and `nrmse == 1.0` is defined as "no better than the
partition mean", so this is the column validating itself against a known answer rather than
against a fixture. `persistence` scores **1.2562**: at a 10 s lead on this signal the skill-score
denominator is *worse than predicting the mean*, which is P3-D5's and P3-D20's argument made
directly visible in a column for the first time.

### Gate 4 evidence

Sweep: `configs/experiment/e02_deep.yaml`, `ideal`, 2026-08-28 20:38 -> 2026-08-30 04:15
(31 h 37 m, exit 0). Artifacts in `results/e02/`: eight files (six from the sweep, plus `gate4.csv` and `gate4.md`). `results/` and `results/imu/` are
the Gate 3 record and are untouched.

**Gate 4 PASSES on both readings**, read by `scripts/gate4.py` from the committed CSVs
(`results/e02/gate4.{csv,md}`):

| Reading | Cell | Result |
|---|---|---|
| A — original criterion, verbatim | `id`, 3 s, vs `damped_persistence` | **18 of 18 PASS** (3 models x 6 DOFs) |
| B — restated (P4-D1), the gate | `id`, 10 s, pitch, vs stronger trivial | **3 of 3 PASS** |

Every row's paired bootstrap interval also excludes zero in the model's favour, so the margin
test and the resampling test agree here. Reading B margins: `lstm` +0.5023, `tcn` +0.4690,
`transformer` +0.4423, against seed spreads of 0.0016, 0.0006 and 0.0032 — the margins exceed
the spread by two to three orders of magnitude, which is why the P4-D1 caveat about `skill_std` being
initialisation-only variance — since **retracted as false, see P4-D13** — does not change the
verdict either way.

| Criterion | Evidence |
|---|---|
| Table covers every model / horizon / DOF / regime | **1728 rows** = 12 models x 4 regimes x 6 DOFs x 6 horizons; schema equals `BASELINES_COLUMNS` |
| Reference is exact | `persistence` skill **bitwise 0.0** in all 144 rows; `rmse_persistence` identical across all twelve models in every cell |
| Not attributable to leakage | Shuffle control **0 failing rows of 144**; worst excess +0.0095 against a 2% tolerance |
| Three seeds on every stochastic row | All four SGD models carry `n_seeds = 3`; deterministic rows `n/a` per P3-D10 |
| Per-cell table decomposes the headline | Joins with **0 unmatched of 1728** |
| No run diverged | `best_val_loss` finite in all 36 deep runs |
| Optimisation state traceable | `epochs_run` per run: `tcn` 60/60/60 (cap bound), `transformer` 29-60, `lstm` 30-44, `dlinear` 60 in 12 of 12 |
| Phase 3 reproduced as a positive control | P4-D10: `dlinear` val loss, `ar20` and `dlinear_ols` gate skill all reproduce to four decimals |

### P4-D11 — What Gate 4 does *not* say: the deep models lose to a linear model on half the corpus

**The counts in this entry are exact; the framing in its heading does not survive. See P4-D14**,
which shows the result reverses for `tcn` and ties for `lstm` once `unseen_heading` — which this
very entry argues is a corpus artifact — is excluded.

Gate 4 is read on `id`, and P3's carry-forward said in advance that `id` cannot separate
architectures. It cannot. The gate passes and the honest headline is close to its opposite.

**Skill at the gate cell (pitch, 10 s) across all four regimes:**

| model | n_params | `id` | `unseen_seastate` | `unseen_heading` | `unseen_vessel` |
|---|---:|---:|---:|---:|---:|
| `damped_persistence` | 6 | 0.3657 | 0.3082 | 0.2247 | 0.1724 |
| `ar20` | 108 900 | 0.5446 | 0.1584 | **-49.43** | 0.4786 |
| `dlinear` | 60 300 | 0.5147 | **0.4775** | 0.5191 | 0.5019 |
| `dlinear_ols` | 60 300 | 0.5683 | 0.4499 | **0.5775** | 0.5344 |
| `tcn` | 196 804 | 0.8346 | 0.2772 | **-81.09** | **0.8298** |
| `transformer` | 2 712 708 | 0.8080 | 0.1953 | **-79.22** | 0.7250 |
| `lstm` | 317 828 | **0.8680** | 0.3437 | **-279.44** | 0.8007 |

**Paired contrasts against `dlinear_ols`** (envelope over seeds; 36 cells per regime, counted as
the paired interval excluding zero in favour of the deep model / of `dlinear_ols` / spanning zero):

| regime | `tcn` | `transformer` | `lstm` |
|---|---|---|---|
| `id` | 27 / 7 / 2 | 25 / 10 / 1 | 27 / 9 / 0 |
| `unseen_seastate` | 17 / 15 / 4 | 9 / 21 / 6 | 10 / 20 / 6 |
| `unseen_heading` | **0 / 32 / 4** | **0 / 36 / 0** | **0 / 35 / 1** |
| `unseen_vessel` | 16 / 20 / 0 | 10 / 24 / 2 | 12 / 20 / 4 |
| **all 144** | **60 / 74** | **44 / 91** | **49 / 84** |

**All three deep models lose more cells to `dlinear_ols` than they win, across the full table.**
A 60 300-parameter closed-form linear map beats a 2.7 M-parameter transformer in 91 of 144 cells.
This is `docs/IMPLEMENTATION_PLAN.md` §0.3 — "include a linear model you might lose to" — landing,
and it is reported in the README body per CLAUDE.md non-negotiable 6.

**Robustness, counting cells where `nrmse > 1.0`, i.e. the model is worse than simply predicting
the scored partition's mean** (of 144):

| `dlinear_ols` | `dlinear` | `ar40` | `ar20` | `ar10` | `tcn` | `transformer` | `lstm` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | 10 | 16 | 17 | 18 | 25 | 26 | **27** |

By this measure the closed-form linear model is the **most robust model in the table** and the
three deep models are the **least robust of the fitted ones**. The column that makes this legible
did not exist before P4-D3.

**Where the deep models do win, they win large**: `id` (all three, decisively) and
`unseen_vessel` at the gate cell (`tcn` +0.2955 over `dlinear_ols`, interval [+0.2647, +0.3256]).
The `unseen_vessel` result is the interesting one — a held-out hull, and the TCN transfers better
than the linear model at the decision horizon on pitch — but it does **not** hold across that
regime's other cells, where `tcn` still loses 20 of 36. The gate cell is not representative of its
own regime, which is the same lesson P3-D12 recorded for Gate 3.

**`unseen_heading` is a rout, and the mechanism is P1-D2, not a training failure.** That regime's
test set *is* beam seas, where the pitch heading factor is clamped at `eps = 0.05` (~26 dB down),
so test-set pitch is a residual floor rather than pitch physics. A model fitted where pitch is a
real signal imposes a fitted amplitude on a channel that has none, and the damage scales with how
much cross-channel structure the model learned: the two channel-independent DLinear rows — which
forecast pitch from pitch history alone and structurally cannot import amplitude from roll — are
the only fitted models that survive, and they are the *best* models in the cell. AR(20) at -49
was already recorded in P3-D7; the deep models reach -79, -81 and -279. This is a property of the
corpus construction, and P1-D2 already flagged those floors as an engineering stand-in belonging
in the README limitations section. It is not evidence that attention or recurrence is unsound.

### P4-D12 — The DLinear optimisation gap did not close at cap 60

`dlinear` hit `epochs_run = 60` in **12 of 12** runs, exactly as P3-D19/P3-D23 recorded at the
same cap. The paired contrast against its own closed-form optimum: `dlinear_ols` is better in
**108 of 144 cells**, `dlinear` in 27, 9 span zero. So the shortfall P3-D19 measured at 0.30% of
the validation loss is still present and still moves per-cell skill, and the two rows must keep
shipping together — a `deep-vs-dlinear` statement made against the SGD row alone would still be
absorbing an optimisation gap into an architecture claim.

### P4-D13 — Correction: `skill_std` is not initialisation variance only

P4-D1 and P4-D8 originally asserted that `dmf.train.experiment._fit_one` shares one seeded
loader across all three seeds, so "batch order is identical across seeds and only weight
initialisation varies", making `skill_std` a narrower spread than the Gate 4 wording intends.
**That is false**, and it was rendered into `results/e02/gate4.md` as well as into this file. It
was caught by the Gate 4 adversarial review, not by the person who wrote it three times.

`dmf.data.dataset.make_dataloader` constructs one `torch.Generator` seeded at `seed`, which the
`RandomSampler` consumes. **The generator's state advances every epoch and is never reset**, and
`set_seed` does not touch it — it is a private object on the loader, not a global RNG. Verified
directly: four successive epochs drawn from one seeded loader give four different permutations.
Since `_fit_one` builds the loader *before* the seed loop, seed 1 begins from the state seed 0's
run left behind. Batch order therefore differs across seeds and `skill_std` includes data-order
variance as the criterion intends.

The error was in the conservative direction — the margin test is against the full spread, not a
narrower one — so no verdict changes. Gate 4's Reading B margins (0.4423 to 0.5023) exceed the
three-seed spreads (0.0006 to 0.0032) by two to three orders of magnitude either way.

**What is true, and is the more interesting property:** the three seeds are *consecutive segments
of one generator stream*, not independent draws. Seed k's batch order depends on how many epochs
seeds 0..k-1 ran. Because `transformer` and `lstm` early-stop at different epochs per seed
(P4-D9), the data order seed 2 receives is a function of seed 0's and seed 1's stopping points. A
run is still exactly reproducible from its config, but the seeds are not exchangeable, and
changing the epoch cap changes the data order of every seed after the first. Recorded because it
bounds what a three-seed spread means here, and because no test asserts it.

### P4-D14 — What P4-D11's headline does not survive, found by the Gate 4 audit

P4-D11 states that all three deep models lose more cells to `dlinear_ols` than they win across
144 cells. The counts are exact. **The interpretation does not survive two restrictions this
protocol itself endorses**, and the audit was right to call the framing overstated.

**Restriction 1 — drop `unseen_heading`,** the regime P4-D11 argues is a corpus artifact:

| | all 144 | excluding `unseen_heading` (108) |
|---|---|---|
| `tcn` | 60 - 74 | **60 - 42** (reverses) |
| `lstm` | 49 - 84 | **49 - 49** (ties) |
| `transformer` | 44 - 91 | 44 - 55 (still loses) |

32 of `tcn`'s 74 losses, 35 of `lstm`'s 84 and 36 of `transformer`'s 91 are in that one regime.
Asserting the aggregate and then arguing three paragraphs later that the regime is an artifact
is having it both ways. **Corrected reading: `tcn` wins outside `unseen_heading`, `lstm` ties,
`transformer` loses.**

**Restriction 2 — the gate is read in the horizon band that most favours the deep models.**
Deep vs `dlinear_ols`, excluding `unseen_heading`, split by lead time:

| band | `tcn` | `transformer` | `lstm` |
|---|---|---|---|
| **1-5 s** (72 cells) — the operational band CLAUDE.md §1 names | 34 - 34 | 19 - 47 | 23 - 42 |
| **10-15 s** (36 cells) — where Gate 4 Reading B is read | **26 - 8** | **25 - 8** | **26 - 7** |

The deep models' advantage is concentrated at long lead times. In the 1-5 s band that the
project exists to serve, `tcn` ties the linear model and the other two lose to it. Reading B's
10 s cell was pre-registered in P4-D1 before the sweep landed and on Gate-3-era reasoning, so
this is not post-hoc cell selection — but the consequence was unstated and is stated now: **the
gate is read where the deep models look best.**

**Restriction 3 — the robustness gap is also carried by `unseen_heading`.** P4-D11's
`nrmse > 1.0` row (8 for `dlinear_ols`, 27 for `lstm`) becomes, over the other three regimes and
listing **every** fitted model rather than a subset: `ar40` 6, `ar_attitude_only` 6,
`dlinear_ols` 7, `ar20` 7, `ar10` 8, `dlinear` 9, `lstm` 9, `tcn` 10, `transformer` 12.

Two corrections to how this was first written, both caught by the Gate 4 re-review. The worst-case
ratio falls from 3.4x (`lstm` 27 / `dlinear_ols` 8) to **1.7x** (`transformer` 12 / 7), not the
1.3x first stated — that paired a worst case before with a non-worst case after. And
**`dlinear_ols` is not the most robust model once `unseen_heading` is dropped**: `ar40` and
`ar_attitude_only` are marginally better at 6. The first version of this entry asserted the
superlative while omitting the three `ar*` rows that falsify it, which is the omission
CLAUDE.md non-negotiable 6 forbids, committed inside the entry written to correct an
overstatement. The all-144 claim in P4-D11 — `dlinear_ols` lowest of all twelve at 8 — is true
as stated; the excluding-`unseen_heading` version is not.

**Restriction 4 — the two models carrying most of the loss column are shipped in a configuration
this project measured as worse than one it already ran.** P4-D9 records `transformer` and `lstm`
at cap 60 as 3.2% and 7.4% worse than the cap-18 pilot, and scoped its bias caveat to deep-vs-deep
ranking. That scope was too narrow: "`dlinear_ols` beats `transformer` in 91 of 144 cells" is a
statement of exactly that form and runs in the direction of the known handicap. The caveat applies
to every deep-vs-baseline count in P4-D11, not only to the ranking among the three deep models.

### P4-D15 — Two omissions from the Gate 4 evidence table

**The untrained control's outcome was omitted.** The evidence table cited the shuffle control
(0 failing of 144) and not the untrained control, which **fails 76 of 144 rows** (worst excess
+0.787). The failure is expected and was predicted: P3-D9 records that a small-init random linear
map emits approximately the window mean, which beats persistence past ~2 s, so the plan's literal
criterion ("a random-init model must score worse than persistence") is wrong for this task, and
the criterion is kept with its failures reported rather than the tolerance tuned. Quoting only the
control that passed is the defect, not the failing control.

**There is no untrained control for any deep model.** `_run_controls` takes the subject as
`sgd_cfgs[0]`, which `e02_deep.yaml` deliberately keeps as `dlinear` (for continuity with the
P3-D9 record). So the untrained control in this phase is run on the linear baseline, and none of
the three models Gate 4 is about has one. Recorded rather than fixed mid-phase.

### P4-D16 — The corpus makes a linear forecaster optimal by construction

The most important limitation on every model-vs-model claim in this phase, and it is structural.

`dmf.sim.response` applies a linear second-order RAO to a 299-component random-phase sinusoid
superposition. Every realization is exactly `sum_i A_i cos(w_e,i t + psi_i)` with **no process
noise anywhere in the generator** (P3-D1). A finite sum of sinusoids satisfies an exact linear
recursion, so the Bayes-optimal forecaster for this corpus **is linear**, and enough lags of a
linear model can identify the system rather than approximate it — which is exactly what P3-D1
measured when AR(20) reached 0.9987 skill.

The README's existing caveat says the achievable-skill ceiling is inflated and that "only
relative comparisons and out-of-distribution degradation should be read as findings". For a
*linear-versus-nonlinear* comparison that is backwards: the relative comparison is the one the
generator prejudges. "A closed-form linear model is competitive with, or beats, three deep
architectures" is a substantially weaker claim on a noiseless linear system than it would be on a
stochastic process, and it is not evidence about how these architectures would rank on real deck
motion, which has process noise, nonlinear roll damping and short-crested excitation.

Anti-periodicity is handled and is not the explanation (`jitter_frequencies: true`,
`n_components: 299`, asserted in `tests/test_spectra.py`), and the 1 s skills of 0.999 are the
signature of the deterministic-superposition structure rather than of leakage — the shuffle
control passes 144 of 144.

### P4-D17 — 144 cells are not 144 independent tests

The win/loss counts throughout P4-D11 and P4-D14 are 6 DOFs x 6 horizons x 4 regimes over the
same realizations, with the rate channels being exact frequency-domain derivatives of the
position channels (P1-D5), all bootstrapped at 95% with no multiplicity control. They are a
**description of the table**, not 144 hypothesis tests, and a count of "cells where the interval
excludes zero" should not be read as a family-wise error-controlled result. P4-D8 caveat 3
records the smaller version of this for the three-seed envelope; this is the larger one.

---

## Phase 5 — probabilistic heads

Entries P5-D1 .. P5-D6 are written **before** any Phase 5 training run, following the P4-D1
precedent, so that the Gate 5 reading, the head design and the predicted `unseen_heading`
artifact are on record as decisions rather than as rationalisations of a result. The Gate 5
evidence section is added when the sweep lands.

### P5-D1 — `heads.py` is the calibration seam, not a second projection mechanism

`src/dmf/models/heads.py` shipped from Phase 0 as four stubs: `sort_quantiles`, and three
`nn.Module`s (`PointHead`, `QuantileHead`, `GaussianHead`) whose declared contract is
`forward(z: Tensor[B, d_in]) -> ...`, i.e. a projection of a flat encoder representation.
**The three modules are removed. `sort_quantiles` is kept unchanged.** This is a deviation
from a committed API and is recorded for that reason.

**Why the declared contract cannot be adopted.** Phase 5 must attach both heads to DLinear
(`docs/IMPLEMENTATION_PLAN.md` §Phase 5, and the task as given). DLinear is channel-independent:
its head is a pair of `Linear(lookback, ·)` maps applied per channel to the trend and remainder
of the *decomposed input series*, not a projection of a flat encoder vector. There is no `z`.
Adopting the stub's contract would give the project two parallel head mechanisms and leave one
of the two models the phase is required to serve outside the abstraction. An abstraction the
required model cannot use is not the seam.

**And the obvious workaround is a correctness trap.** The four SGD models already emit
`(B, H, C, K)` for `K = max(n_quantiles, 1)`, so a Gaussian head could be had for free by
building with `n_quantiles = 2` and reading channel 0 as the mean and channel 1 as the
log-variance. That places `(mean, log_var)` on exactly the axis `sort_quantiles` is defined to
sort ascending, and sorting them is silent, shape-preserving and catastrophic: it would swap the
mean and the log-variance on every element where `log_var < mean`, producing intervals that are
wrong without being malformed. `head` is therefore a first-class kind, not an arity.

**What replaces them** (all in `dmf.models.heads`):

- `HeadKind = Literal["point", "quantile", "gaussian"]` and `n_output_params(head, quantiles)`
  returning 1 / `Q` / 2 — the single definition of the fan width, imported by `base.py`, by
  every model that carries a head, and by `dmf.train.registry.build_model`. One place decides
  `K`.
- `QUANTILE_FAN_9` — the nine levels the removed `QuantileHead` docstring already named:
  0.05 to 0.95 in steps of 0.1125, i.e.
  `(0.05, 0.1625, 0.275, 0.3875, 0.5, 0.6125, 0.725, 0.8375, 0.95)`. The median is exactly 0.5
  and the outermost pair is exactly the 90 percent interval PICP@90 scores, so neither has to
  be interpolated.
- `PredictiveDistribution` — a frozen value object over a raw model output carrying its
  `HeadKind` and levels, exposing `.point()`, `.quantiles(levels)` and `.interval(alpha)`. It is
  unit-agnostic, so it behaves identically in normalised and in corpus units. Ascending sorting
  is applied on construction for the `quantile` kind; for the `gaussian` kind the quantile axis
  does not exist and the trap above is **structurally unreachable** rather than merely avoided.
- `IntervalPredictor` — the protocol a `ConformalWrapper` implements: `predict(x) ->
  PredictiveDistribution`. A wrapper delegates to the model it holds and offsets the interval
  endpoints from held-out residuals. No model file is touched, which is the requirement
  `docs/IMPLEMENTATION_PLAN.md` §Phase 5 states and Project 6 depends on.

The seam is asserted nowhere and exercised in `tests/test_probabilistic.py`: a `_ShiftCalibrator`
test double implements `IntervalPredictor`, wraps a stub distribution and must move PICP by the
amount it shifted, importing no model. A seam that is only claimed in a docstring is not a seam.

### P5-D2 — Gate 5 reading, pre-registered. RECORDED 2026-08-31

`docs/IMPLEMENTATION_PLAN.md` §Phase 5 states:

> **Gate 5:** PICP@90 within `[0.85, 0.95]` on the `id` regime.

The threshold and the band are **unchanged**. What the plan does not say is *which cell* the
coverage is read at, and coverage varies strongly with lead time, so leaving that open until
after the sweep would be cell selection. Registered now:

- **Reading A — the gate.** Per (model, head), PICP@90 at the decision cell **pitch, 10 s
  (100 samples)**, `id` regime. This is the same cell Gate 3 (P3-D12) and Gate 4 (P4-D1) are
  read at, so all three gates read the same place and the phases stay comparable.
- **Reading B — the surround.** The count of `id` cells whose PICP@90 falls inside `[0.85, 0.95]`,
  of **216** (3 backbones x 2 heads x 6 DOFs x 6 horizons), reported in the same document. Gate 3
  and Gate 4 both required a "what the gate does not say" section after the fact; here it is part
  of the read-out from the start.

Both readings ship. A head that misses the band is reported as missing it: nothing is widened,
recalibrated, or dropped to make the gate pass (CLAUDE.md non-negotiable 6). Coverage is never
reported without mean interval width beside it — coverage alone is trivially achievable by
widening an interval until it is useless.

### P5-D3 — Training budget: cap 60 kept, and what that costs Phase 5

P4-D9 records that `dmf.train.loop._lr_at` sizes the warmup+cosine schedule by
`epochs * steps_per_epoch`, so at cap 60 the `transformer` and `lstm` runs early-stopped
mid-anneal and shipped **3.2% and 7.4% worse** than an 18-epoch pilot of the same models. It
also states what a Phase 5 selecting "the best point model from Phase 4" would have to do about
it: decouple the schedule length from the stopping criterion.

**The budget is unchanged: cap 60, batch 1024, lr 2e-3, patience 15, identical to
`e02_deep.yaml`.** Decided with the user before the sweep. Two reasons:

- Changing `dmf.train.loop` now would make every Phase 5 point row non-comparable to the
  committed `results/e02/` table, and the point rows are what a probabilistic row is judged
  against.
- Choosing a cap after measuring which cap favours which architecture is selecting a
  hyperparameter on the results it produces — the failure P4-D9 declined to commit and Gate 4's
  fairness rule exists to prevent.

**The cost, stated rather than discovered later.** Any Phase 5 sentence comparing `lstm` to
`tcn` — including a comparison of their *calibration* — inherits the P4-D9 asymmetry: `tcn` is
cap-bound and still improving, `lstm` is early-stopped mid-anneal. The bound is on deep-vs-deep
ranking, and P4-D14 restriction 4 records that it extends to deep-vs-baseline counts too.

**The backbones are `tcn`, `lstm` and `dlinear`**, decided with the user rather than inherited.
The plan guesses TCN; `lstm` is the best point model at the gate cell on `id` (0.8680 against
`tcn` 0.8346), which is the regime Gate 5 is read on; `tcn` is the only deep model that wins
outside `unseen_heading` (60-42 against `dlinear_ols`, P4-D14) and is the one not carrying the
P4-D9 handicap. Both ship, so the choice does not have to be made on a contested ranking.
`transformer` is not carried into Phase 5: it loses to `dlinear_ols` on every restriction in
P4-D14 and adds a third of the sweep's wall time.

### P5-D4 — The early-stopping criterion is now head-specific, and `best_val_loss` stops being comparable

`dmf.train.loop.fit`'s docstring states:

> The early-stopping criterion -- validation MSE in normalised space, patience from ``cfg`` --
> is deliberately identical for every model in the project. Gate 4 compares architectures, and a
> comparison in which one model was stopped on a different rule is not a comparison of
> architectures.

**That invariant cannot survive Phase 5 and is amended rather than quietly broken.** A quantile
head has no MSE to stop on: its objective is pinball loss, a Gaussian head's is NLL, and
stopping either on the MSE of a derived point forecast would select the epoch that is best for a
statistic the model is not fitting. Each model is stopped on its own training objective.

Two consequences:

- `best_val_loss` is **not comparable across heads**. A pinball loss and an MSE are not the same
  quantity and their ratio means nothing. A `val_loss_name` column (`mse` / `pinball` /
  `gaussian_nll`) ships on `probabilistic_by_seed.csv` so the comparison cannot be made by
  accident.
- What *is* still identical for every model in the phase: the cap, the patience, the schedule
  policy, the batch size, the learning rate, the data pipeline and the split. The fairness
  property Gate 4 needed is preserved everywhere it can be.

### P5-D5 — Every probabilistic row carries a point forecast, and which one it is

CLAUDE.md non-negotiable 4 requires every accuracy result to be reported as skill against
persistence. A probabilistic model must therefore also emit a point forecast, and the choice is
recorded rather than left implicit:

| head | point forecast |
|---|---|
| `quantile` | the **0.5 quantile** — present exactly in `QUANTILE_FAN_9`, so it is read off, not interpolated |
| `gaussian` | the **mean** |

RMSE, MAE, skill and `nrmse` ship for every probabilistic row against the same persistence
denominator as every other row in the project. This is also what answers the question the phase
would otherwise leave open — whether fitting a distribution costs point accuracy against the same
architecture's point row in `results/e02/`.

**The heads are not parameter-matched to their point rows, and the gap is large.** The final
projection widens by `K`, and `max_horizon` is 150:

| model | point | gaussian (`K=2`) | quantile (`K=9`) |
|---|---:|---:|---:|
| `dlinear` | 60 300 | 120 600 | 542 700 |
| `tcn` | 196 804 | 255 304 | 664 804 |
| `lstm` | 317 828 | 433 928 | 1 246 628 |

So a quantile row against a point row, or a quantile row against a Gaussian row, is **not** a
parameter-matched comparison. P3-D13 records this project publishing a wrong conclusion twice
from a comparison whose confound was not held; the count is stated here in advance and belongs
in the table caveats, not only in this file.

### P5-D6 — Predicted before the sweep: `unseen_heading` coverage will be uninterpretable

Recorded now so that it cannot be produced afterwards as an explanation of a number.

`unseen_heading`'s test set *is* beam seas, where the P1-D2 pitch heading factor is clamped at
`eps = 0.05` (~26 dB down), so test-set pitch and pitch_rate are an engineering residual floor
rather than pitch physics. P3-D7 measured AR(20) at -49 skill there and P4-D11 measured the deep
models at -79, -81 and -279. The prediction for coverage: a head fitted where pitch has amplitude
will emit intervals scaled to that amplitude on a channel that has almost none, so **PICP on
`unseen_heading` pitch and pitch_rate will sit near 1.0 with a mean interval width far wider than
the signal** — coverage that looks excellent and means nothing. That is the exact failure mode the
protocol's "report coverage and sharpness together" rule exists to catch, and it will be the
clearest demonstration of it in the project.

The regime is still run, and the numbers still ship. What is registered here is that a PICP near
1.0 in those cells is not evidence of good calibration.

### P5-D7 — `configs/experiment/e03_quantile.yaml` is named `e03_probabilistic.yaml`

`docs/IMPLEMENTATION_PLAN.md` §0.1 lists the Phase 5 experiment as `e03_quantile.yaml`. The
experiment carries the Gaussian rows too, and a file named for one of the two heads it runs would
misdescribe half its own table. Renamed to `e03_probabilistic.yaml`; recorded because the plan
names a path.

### P5-D8 — Probabilistic scoring: three things the stubs did not pin down

`src/dmf/eval/probabilistic.py` shipped from Phase 0 with five docstring contracts. Three of
them were under-specified in ways that change a published number, so the resolutions are
recorded rather than left in the implementation.

**1. The CRPS quadrature is the rectangle rule, and the number is biased low.**
`crps_from_quantiles` documents "the quantile-weighted integral of the pinball loss" without
pinning the quadrature. Two readings are defensible: `2 * mean_q(PL_q)` (rectangle) and a
trapezoid over the levels. They differ materially — on `QUANTILE_FAN_9` the trapezoid's weights
total `q_max - q_min = 0.90`, so it returns `0.9 x MAE` where the definitional anchor (CRPS of a
degenerate fan is the MAE) requires exactly `MAE`. **The rectangle rule is implemented.** Its
cost, stated because it is a real bias and not a rounding detail: the tails beyond 0.05 and 0.95
are represented only by their nearest level, so the reported figure is biased **low** against
exact CRPS. Every docstring, the module header and the `n_quantiles` column label it an
approximation, as the original stub already required.

**2. Pinball averages over the level axis; it does not sum.** "Mean pinball loss" does not say.
Averaging is forced by the same anchor family — pinball at the median must be exactly half the
MAE — and summing would inflate the column ninefold at `Q = 9` while looking plausible. Pinned by
a median/MAE test and a level-count-invariance test, so a future change to a sum fails the suite
rather than silently rescaling a published column.

**3. The crossed-interval refusal moved from the element to the table.** `picp` raises when any
`lower > upper`; `mean_interval_width` on the same input returns a **negative width silently**.
That asymmetry does not matter on the array path, but production streams *sums*, so `picp`'s
elementwise guard is never reached and an unsorted fan would ship as a published number. The
declared per-function contracts are unchanged; `probabilistic_table_from_sums` additionally
refuses any reported cell whose mean width is negative, naming `sort_quantiles` in the message.
Alongside it: `picp` and `crossing_rate` outside `[0, 1]` (which means the counts and the window
total came from different passes) and non-finite or negative `winkler` / `crps` / `pinball` are
refused. All are validated over the **reported** cells only, following the precedent
`metrics_table_from_sums` set — a bad cell at a horizon the table never quotes must not fail a
table that never quotes it.

**Smaller resolutions, recorded because each is a place two reasonable implementations differ:**

- **The interval is closed.** A target lying exactly on an endpoint counts as covered.
- **Crossing is measured on the raw fan with strict inequality.** Equal adjacent levels are not an
  inversion. `crossing_rate` must be accumulated *before* sorting: measuring it after yields
  exactly zero and measures nothing. Both directions are asserted.
- **Levels are refused, never repaired.** A `quantiles` tuple that is not strictly ascending in
  (0, 1) raises rather than being sorted, because silently reordering the labels attaches the
  wrong level to every column of the fan.
- **`n_quantiles` and `alpha` ship as columns.** CRPS is a `Q`-level quadrature and PICP and
  Winkler are `alpha`-dependent, so a row without them is not self-describing: a Gaussian head
  scored at `Q = 2` and a quantile head at `Q = 9` would otherwise show two visually identical
  CRPS columns holding different quantities. `alpha` labels the column only — it does not rescale
  `winkler_sum`, which must be accumulated at the alpha it is reported at.
- **`crps_sum` is a free input, not derived from `pinball_sum`.** For a quantile fan the two are
  related by construction (`crps = 2 * mean_q pinball`), but the Gaussian head has a closed-form
  CRPS and must be able to stream it through the same table. The consequence is that the table
  cannot detect a caller that accumulates the two inconsistently; that guard belongs in
  `prob_runner.py`, where the head kind is known, and is placed there.

### P5-D9 — Four things the implementation forced, none of them in P5-D1 .. P5-D8

**1. The attribute is `head_kind`, not `head`, and the reason is a `torch` trap.**
`tcn`, `lstm` and `transformer` already assign `self.head = nn.Linear(...)`.
`nn.Module.__setattr__` moves a Module assignment into `_modules` and *removes* the same name
from `__dict__`, so a base-class `self.head = "gaussian"` would be silently clobbered and
`model.head` would return the Linear layer — a head kind that reads as a tensor op, with no
error anywhere. The alternative was renaming those layers, which changes the
`head.weight`/`head.bias` keys of every checkpoint already written under
`artifacts/checkpoints/`. The attribute was renamed instead of the layers.

**2. `BaseForecaster.SUPPORTED_HEADS`, because passing `head` through `build_model` does not
close the hole it was supposed to close.** P5-D1 assumed the existing accepted-keyword check
in `build_model` would refuse `head: gaussian` on a model that cannot carry one. It refuses
only classes that define their own `__init__` — `window_mean`, `damped_persistence`, `ar`,
`dlinear_ols` — and `Persistence` defines none. It inherits `BaseForecaster.__init__`, so
`head` is an accepted keyword, and `head: gaussian` on `persistence` would have constructed a
model that *records itself as Gaussian* and emits a rank-3 point forecast. `SUPPORTED_HEADS`
is a class variable defaulting to `("point",)` and catches exactly the classes the signature
check cannot see. Both refusals are tested, because half a guard is worse than none: it
creates the belief that the case is covered.

**3. `quantile_fan(width)` — a model is constructed with a fan *width*, never with levels.**
`build_model` passes `n_quantiles = len(cfg.quantiles)`; the levels themselves never reach the
model. So the levels have to be recoverable from the width, or a model cannot say what its own
columns mean. `quantile_fan(9) is QUANTILE_FAN_9`, and other widths are spaced 0.05 to 0.95 by
the same rule. `build_model` now **raises** if a config's levels are not `quantile_fan` of
their own length. Without that check a config could be *trained* on its own fan — `resolve_loss`
reads `cfg.quantiles` — and *scored* on the project fan, so column `k` would carry one level in
the loss and a different one in the table, and every number would be plausible.

**4. Both heads are scored on the same nine levels, and the Gaussian CRPS is not closed form.**
P5-D8 left `crps_sum` a free input specifically so a Gaussian head could stream an exact CRPS.
It does not. A Gaussian head is evaluated at the same nine levels via `mean + z(q) * sigma` and
put through the same rectangle-rule quadrature, so the `pinball` and `crps` columns are one
estimator applied to two predictive distributions. An exact Gaussian CRPS beside a nine-level
approximation of the quantile head's would have made the Gaussian row look better **by the
quadrature bias alone** — the bias P5-D8 records as low — which is a comparison artifact, not a
result. The consistency `crps == 2 * mean_q(pinball)` therefore holds by construction here and
`dmf.eval.prob_runner._check_crps_pinball_consistency` asserts it per model, which is the guard
P5-D8 said belongs where the head kind is known.

### P5-D10 — The shuffle control was about to be silently absent, and what it does and does not certify

`dmf.train.experiment._run_controls` selects the shuffle-control subject as
`[m for m in experiment.models if "order" in m.params]` — i.e. the control runs **only if the
experiment carries an AR config**. The first `e03_probabilistic.yaml` carried none. The sweep
would have run, written `baselines_controls.csv`, and shipped a coverage table with **no
leakage control at all**, with nothing in the artifact saying so.

That is the P4-D15 defect exactly — quoting the control that passed while omitting the one that
did not run — and it is worse here, because Gates 3 and 4 both cite the shuffle control as the
evidence that their results are not leakage. `ar_p20.yaml` is added to the experiment. It is
closed-form and slices from the moments the pass already accumulates, so it costs one Cholesky.
`tests/test_models.py::test_the_probabilistic_experiment_config_is_one_run_at_the_phase_4_budget`
now asserts the presence of a config carrying an `order` parameter, keyed on the mechanism
`_run_controls` actually dispatches on rather than on the label, so renaming the row is safe and
deleting its `order` parameter is not.

**What the control does and does not certify, stated because the distinction matters here more
than it did in Phase 4.** The shuffle control refits AR(20) — a **point** model — on
time-shuffled targets. It certifies the point pipeline the probabilistic rows are built on. It
is **not** a leakage control on the heads: no shuffled-target quantile or Gaussian head is
fitted anywhere in this sweep, because refitting a deep head on shuffled targets would cost
roughly what the sweep costs. So a coverage number here carries less certification than a skill
number does, and no sentence about Gate 5 should imply otherwise.

The untrained control has the same shape of gap. Its subject is `sgd_cfgs[0]`, which in this
experiment is `dlinear_quantile`, and it is scored through its **point projection** (the 0.5
quantile) because the null and the published Phase 3/4 comparators are point forecasts. So it
controls an untrained median, not an untrained interval. **There is no control on an untrained
interval in this phase.**

### P5-D11 — `width_ratio`: a mean interval width is not readable on its own

A width of 2.3 deg is sharp for roll at 15 s and uselessly wide for pitch at 1 s, so the width
column the plan asks for cannot be read without a reference. `dmf.eval.report.width_ratio`
divides it by the width an **unconditional** interval would need for the same nominal level
using only the scored partition's own spread, `2 * z(1 - alpha/2) * signal_std`:

> `width_ratio = 1.0` means the interval is no sharper than knowing nothing but the variance.

This is deliberately the same device `nrmse` provides for RMSE (P4-D3), where 1.0 means "no
better than predicting the partition mean", and it is adopted for the same reason: P3-D1 and
P4-D16 establish that absolute magnitudes on this corpus flatter every model, so only
dimensionless, referenced quantities are readable across DOFs, horizons and regimes.

`signal_std` is **carried over from the point pass rather than recomputed.** It is a property of
the targets alone, `evaluate_models` already accumulates it once per batch and shares it across
every model by object identity (P4-D3), and a second float64 reduction of the same values in a
different order would disagree in the last bits — producing two `signal_std` columns in one
results directory that are almost, but not exactly, the same number.

It is a **sharpness** measure and says nothing about calibration alone. A ratio below 1.0 with
PICP at nominal is an informative interval; a ratio below 1.0 with PICP well under nominal is
just an interval that is too narrow. The two columns are rendered adjacent for that reason.

### P5-D12 — Coverage degradation is reported unpaired, and could not honestly be otherwise

`dmf.eval.gate.coverage_degradation` reports `picp_delta` and `width_delta` from `id` to each
held-out regime **with no confidence interval on the delta**, and that is deliberate.

`id` and `unseen_seastate` score **different realizations** — different seed ordinals in the
first case, an entirely held-out sea state in the second. The paired bootstrap P4-D4 wired
(`paired_skill_difference_ci`) works by drawing one multinomial resample of realizations and
scoring both subjects on it, which is exactly what makes it exact for two models on one
partition. Two *partitions* share no realizations, so there is no common resample to draw and
the paired machinery does not apply. Each side ships its own realization bootstrap
(`picp_ci_lo`/`picp_ci_hi`) and the difference ships none.

P3-D13 records this project publishing a wrong conclusion **twice** from reading unpaired
marginal intervals as if they were a paired contrast. A delta interval here would be the third
occurrence, dressed better. The absence is stated in `gate5.md`'s notes rather than left for a
reader to notice.

### P5-D13 — Correction: P5-D6's prediction was half right, and the half it got wrong is the finding

P5-D6, registered before the sweep, predicted that on `unseen_heading` **PICP on pitch and
pitch_rate would sit near 1.0 with a mean interval width far wider than the signal** — coverage
that looks excellent and means nothing — for the P1-D2 residual-floor reason. Measured, median
over the six horizons:

| model | pitch PICP | pitch `width_ratio` | pitch_rate PICP | pitch_rate `width_ratio` |
|---|---:|---:|---:|---:|
| `dlinear_quantile` | **1.000** | **5.19** | **1.000** | **5.49** |
| `dlinear_gaussian` | **1.000** | **5.73** | **1.000** | **6.06** |
| `tcn_quantile` | 0.387 | 1.43 | 0.396 | 1.58 |
| `tcn_gaussian` | 0.454 | 2.23 | 0.538 | 2.65 |
| `lstm_quantile` | 0.254 | 1.75 | 0.286 | 2.57 |
| `lstm_gaussian` | 0.242 | 2.08 | 0.258 | 3.04 |

**The prediction is exactly right for the two DLinear rows and exactly wrong for the four deep
ones.** DLinear covers 100% of targets with intervals five to six times wider than an
unconditional interval — the predicted "perfect coverage, meaningless width". The deep models
instead **under-cover catastrophically**, at 0.24 to 0.54 against a nominal 0.90.

**Why the prediction failed, which is the useful part.** P5-D6 reasoned only about interval
*width*: a head fitted where pitch has amplitude emits intervals scaled to an amplitude the test
set does not contain, therefore over-covers. That reasoning silently assumed the interval stays
*centred on the target*. It does not. P4-D11 already recorded the mechanism and P5-D6 did not
carry it across: the two channel-independent DLinear rows forecast pitch from pitch history alone
and structurally cannot import amplitude from roll, so their point forecast tracks the floored
signal and their oversized interval swallows it. The deep models do import cross-channel
structure, so they impose a roll-driven amplitude on a channel that has none — point skill at
pitch / 10 s measured here at **-133 (`tcn_quantile`) and -337 (`lstm_quantile`)** against
`dlinear_ols`'s +0.577. **A wide interval centred in the wrong place still misses.** Coverage
depends on location and width jointly, and P5-D6 modelled one of them.

**The larger correction: the deep models' interval failure is NOT confined to the floored
channels.** On the four channels the P1-D2 floor does not touch — roll, roll_rate, heave,
heave_rate — median PICP over that regime is:

| `dlinear_gaussian` | `dlinear_quantile` | `tcn_gaussian` | `lstm_quantile` | `lstm_gaussian` | `tcn_quantile` |
|---:|---:|---:|---:|---:|---:|
| 0.928 | 0.914 | 0.294 | 0.176 | 0.166 | 0.185 |

The deep heads cover **17-29%** of targets on channels where their *point* forecasts are
respectable (roll at 10 s: `tcn_gaussian` 0.796, `lstm_quantile` 0.693 skill). So this is not the
corpus artifact and it is not bad point accuracy — it is an interval that is far too narrow for a
heading the model never saw, on channels where the model still forecasts well. This is the same
shape as the README's existing Phase 4 statement that "the floor does not explain the direction on
the other channels", and it is much starker for intervals than it was for point error.

Recorded as a correction rather than edited into P5-D6, following P4-D9: the reasoning that
produced the wrong prediction is what a reader needs, and the entry was written in advance
precisely so that being wrong would be visible.

### P5-D14 — P5-D3's carried-forward handicap largely does not apply, and the reason is the objective

P5-D3 carried P4-D9 forward as a stated bound on every Phase 5 claim: at cap 60 the `lstm` runs
early-stop mid-anneal, checkpointing a best epoch reached while the learning rate is still high,
and ship measurably worse than a shorter-cap pilot would. That was recorded in advance as the
cost of keeping the budget unchanged.

**Measured on the sweep, it mostly does not happen.** `epochs_run` over all 72 SGD runs, cap 60,
patience 15:

| model | objective | runs at the cap | early-stopped | range |
|---|---|---:|---:|---|
| `dlinear_quantile` | pinball | 12 / 12 | 0 | 60 |
| `dlinear_gaussian` | gaussian_nll | 12 / 12 | 0 | 60 |
| `tcn_quantile` | pinball | 12 / 12 | 0 | 60 |
| `tcn_gaussian` | gaussian_nll | 12 / 12 | 0 | 60 |
| `lstm_quantile` | pinball | **10 / 12** | 2 | 34-60 |
| `lstm_gaussian` | gaussian_nll | **7 / 12** | 5 | 28-60 |

In Phase 4 the *point* `lstm` early-stopped in **every** run (33 / 32 / 34 epochs on `id`, P4-D9).
Under pinball it runs to the cap in 10 of 12, and under NLL in 7 of 12. **The head's objective
changes the optimisation trajectory enough to change whether early stopping fires at all.** The
plausible reading — untested, and recorded as such — is that a pinball or NLL validation curve is
noisier and less prone to a 16-epoch plateau than an MSE one, so patience 15 is harder to trip.

**Consequence for Phase 5's claims, and it runs in the unfavourable direction for the deep
models.** `lstm` is no longer the *under*-trained row P5-D3 warned about; it is cap-bound like the
others, so a Phase 5 statement that `lstm_quantile` calibrates worse out of distribution than
`dlinear_quantile` cannot be explained away by a truncated budget. What remains true, and is now
the residual asymmetry, is that `lstm_gaussian` still early-stops in 5 of 12 runs — notably all
three `unseen_vessel` seeds (28 / 36 / 43) — so **that one row is under-trained on the regime it
is scored worst on**, and a `lstm_gaussian`-on-`unseen_vessel` claim carries the P4-D9 caveat while
the other five rows no longer do.

Within a model, seed 0 early-stops where seeds 1 and 2 run to the cap (`lstm_quantile` on `id`:
37 / 60 / 60). That is P4-D13: `make_dataloader`'s generator is never reset between seeds, so the
three seeds are consecutive segments of one stream and receive different data orders.

Recorded as a correction to P5-D3 rather than an edit to it, per P4-D9's own precedent.

### Gate 5 evidence

Sweep: `configs/experiment/e03_probabilistic.yaml`, `ideal`, 2026-08-31 14:21 -> 2026-09-03 08:25
(**66 h 04 m**, exit 0), one A4000. 60.6 h of that is the 72 SGD fits; the balance is the
closed-form rows, four scoring passes and the controls. Artifacts in `results/e03/`: nine files.
`results/` and `results/imu/` (Gate 3) and `results/e02/` (Gate 4) are untouched.

**Gate 5 is read on Reading A and PASSES; Reading B is reported beside it and does not.**

| Reading | Cell | Result |
|---|---|---|
| A — the gate (P5-D2, registered before the sweep) | `id`, pitch, 100 samples | **6 of 6 PASS** |
| B — the surround, every `id` cell | `id`, all 6 DOFs x 6 horizons x 6 rows | **102 of 216 in band** |

| Criterion | Evidence |
|---|---|
| Table covers every model / regime / DOF / horizon | `probabilistic.csv` **864 rows** = 6 models x 4 regimes x 6 DOFs x 6 horizons, exactly; `probabilistic_by_seed.csv` 2592 = 864 x 3 seeds |
| Three seeds on every stochastic row | `n_seeds == 3` on all 864 rows; there are no deterministic probabilistic rows, so the P3-D10 exemption never applies here |
| One nominal level throughout | `alpha == 0.1` on all 864 rows; `n_quantiles == 9` for **both** heads, because a Gaussian head is scored at the same nine levels rather than in closed form (P5-D9 item 4) |
| Every probabilistic row carries its point accuracy | **864 of 864** join to a row of `baselines.csv` on (model, regime, DOF, horizon) — the P5-D5 requirement, and what makes CLAUDE.md non-negotiable 4 hold for a coverage table |
| Reference is exact | `persistence` skill **bitwise 0.0** in all 144 cells of `baselines.csv` |
| Not attributable to leakage | Shuffle control **0 failing rows of 144**, worst excess **+0.0095** against a 2% tolerance — identical to the Gate 4 figure. Note what it does *not* certify: it refits AR(20), a point model, so it is a control on the point pipeline, **not** on the heads (P5-D10) |
| Untrained control reported, not suppressed | Fails **107 of 144** rows, worst excess +0.8146. Expected and predicted: P3-D9 records that the plan's literal criterion is wrong on this task, and it is kept with its failures reported rather than tuned. Its subject is `dlinear_quantile` scored through its point projection, so it controls an untrained *median*, not an untrained interval — **no control on an untrained interval exists in this phase** (P5-D10) |
| Phase 4 reproduced as a positive control | `persistence`, `window_mean`, `damped_persistence` and `dlinear_ols` reproduce `results/e02/baselines.csv` **bitwise (max abs diff 0.0)** across all 144 cells each; `ar20` to **4.8e-06 relative** (its largest absolute departures sit on the `unseen_heading` cells where skill is order -50, i.e. BLAS reduction order on a Cholesky, not a pipeline change). `dlinear_ols` at the gate cell: **0.568282 in both phases**. Nothing in the corpus, splits, normalisation or closed-form solvers moved between Phase 4 and Phase 5 |
| Optimisation state traceable, and checked rather than assumed | `epochs_run` on every run: cap-bound in 12/12 for `dlinear_*` and `tcn_*`, 10/12 for `lstm_quantile`, 7/12 for `lstm_gaussian` (P5-D14) |
| No verdict rests on an unmeasured row | `gate5.csv` carries **0 UNVERIFIED** rows of 222 |
| Normalisation provenance | Asserted at scoring time in both passes, fails closed (P3-D11); the probabilistic pass additionally asserts its realization keys equal the point pass's, so a row's `picp` and its `rmse` cannot describe different windows |

### P5-D15 — What Gate 5 does *not* say: the heads are miscalibrated in the operational band

Gate 5 Reading A passes on six rows of 216. The surrounding counts change the reading, and the
pattern is the same one P4-D14 restriction 2 found for Gate 4 — **the gate is read in the horizon
band where the deep models look best.**

Cells inside `[0.85, 0.95]` on `id`, split by lead time:

| band | `dlinear_q` | `dlinear_g` | `tcn_q` | `tcn_g` | `lstm_q` | `lstm_g` |
|---|---:|---:|---:|---:|---:|---:|
| **1-5 s** (24 cells) — the operational band CLAUDE.md §1 names | **12** | 10 | 1 | 3 | 4 | 6 |
| **10-15 s** (12 cells) — where Gate 5 Reading A is read | 9 | 9 | **12** | **12** | **12** | **12** |

**The four deep rows are perfectly calibrated at 10-15 s — 12 of 12 each, median PICP 0.914-0.916 —
and systematically over-cover at 1-5 s**, where 18 to 23 of their 24 cells sit above 0.95 (median
PICP 0.972-0.979) and not one sits below 0.85. DLinear is the reverse: the best-calibrated family
in the operational band and the worse one at long lead.

The mechanism is not mysterious. P3-D1 records that this corpus is saturated at short lead — AR(20)
reaches 0.999 skill at 1 s — so a deep model's residual at 1-5 s is very small, and its learned
interval, while extremely sharp in absolute terms (median `width_ratio` **0.019-0.035**, i.e. about
3% the width of an unconditional interval), is still wider than the residual warrants. The heads
are conservative exactly where the forecast is easiest.

**Reading A's cell was registered in P5-D2 before the sweep ran, on the Gate 3 and Gate 4 reasoning,
so this is not post-hoc cell selection.** But the consequence was unstated and is stated now: a
sentence of the form "the probabilistic heads are well calibrated in-distribution" is true at the
gate cell and at 10-15 s, and false across the band this project exists to serve. Any Gate 5 claim
must carry the band it is read in.

### P5-D16 — Post-hoc quantile sorting is not cosmetic, and the gate cell hides that

CLAUDE.md §Known traps requires quantile outputs be sorted post-hoc. Measured crossing rate — the
fraction of `(window, horizon, channel)` elements whose **raw** fan had at least one adjacent
inversion, over all 144 cells per model:

| model | median | max |
|---|---:|---:|
| `tcn_quantile` | 0.0006 | 0.266 |
| `lstm_quantile` | 0.0205 | 0.155 |
| `dlinear_quantile` | **0.0626** | **0.966** |

**At the gate cell the rates are 0.000005 to 0.0038, which reads as "sorting is a no-op".** Over the
full table it is not: `dlinear_quantile` crosses on a median 6% of elements and, in its worst cell,
on **96.6%** — a fan that is very nearly fully inverted. Every published PICP and width for that
model depends on the sort having been applied. Recorded because an earlier draft of this phase's
summary described crossing as "negligible" on the strength of the gate cell alone, which is the
single-cell generalisation P3-D12 and P4-D11 both warn about, committed again.

**It is a short-horizon phenomenon, and it is worst exactly where P5-D15 says the intervals are
already suspect.** Median crossing rate on `id`, by lead time:

| model | 1 s | 2 s | 3 s | 5 s | 10 s | 15 s |
|---|---:|---:|---:|---:|---:|---:|
| `dlinear_quantile` | **0.563** | 0.314 | 0.107 | 0.073 | 0.002 | 0.000 |
| `lstm_quantile` | 0.045 | 0.057 | 0.077 | 0.073 | 0.016 | 0.015 |
| `tcn_quantile` | 0.011 | 0.007 | 0.005 | 0.002 | 0.000 | 0.000 |

`dlinear_quantile`'s raw fan is inverted on **56% of elements at a 1 s lead** and essentially never
at 15 s. The mechanism is the same one behind P5-D15: at short lead this corpus is nearly
deterministic (P3-D1), so the predictive spread is tiny — median `width_ratio` **0.019 at 1 s**
against 0.911 at 15 s — and nine levels squeezed into that width are numerically
indistinguishable, so fitting noise reorders them. The sort is doing real work precisely in the
1-5 s operational band and nowhere else.

The column exists because `dmf.eval.prob_runner` reads the raw fan before
`PredictiveDistribution` sorts it (P5-D8). Had it measured after, it would report exactly zero and
this would be invisible.

### P5-D17 — What the Gate 5 adversarial audit falsified, before anything was published

Run against a draft of the README's Phase 5 section. Three claims in that draft were wrong and
are recorded here rather than quietly fixed, because two of them are errors this protocol had
already warned against in writing.

**1. "The ordering reverses" is false, and it is a single-cell generalisation — the exact defect
P5-D16 records me committing earlier the same phase.** The draft said the models best calibrated
in-distribution degrade worst. Cells inside `[0.85, 0.95]` on `id`, of 36, with the median
absolute departure from nominal beside it:

| model | in band on `id` | median \|PICP - 0.90\| |
|---|---:|---:|
| `dlinear_quantile` | **21** | **0.0411** |
| `dlinear_gaussian` | 19 | 0.0427 |
| `lstm_gaussian` | 18 | 0.0511 |
| `lstm_quantile` | 16 | 0.0584 |
| `tcn_gaussian` | 15 | 0.0580 |
| `tcn_quantile` | 13 | 0.0695 |

**DLinear is the best-calibrated family in-distribution as well**, so nothing reverses; the
ordering is monotone. The "reversal" exists only at the single registered gate cell, where the
deep rows sit at 0.912-0.918 and DLinear at 0.858-0.865 — one cell of 36, and the only one where
the sign flips. Withdrawn. The supported statement is "DLinear is the best-calibrated family in
every regime, and the gap widens out of distribution".

**2. The `unseen_heading` PICP range 0.24-0.61 is not reproducible from any cell.** The draft
paired `width_ratio` 4.3-8.5 — which is the pitch / 100-sample cell — with a PICP range taken
from P5-D13's *median over six horizons*. At the cell the widths come from, deep coverage is
**0.500, 0.570, 0.612, 0.832**. The draft's range both mixed two aggregations and excluded
`tcn_gaussian` at 0.832, which is the row that most weakens the location-not-width argument.
Withdrawn; the cell is quoted with all four rows and their seed spreads, which are large
(`lstm_gaussian` 0.5696 +/- **0.3047**).

**3. "The quantile heads beat their point rows" attributes to the head an effect three confounds
account for.** The draft compared `results/e03` rows to `results/e02` rows:

- **Training length.** `lstm` in e02 early-stopped at **32/33/34** epochs on `id`; `lstm_quantile`
  here ran **37/60/60**. The quantile row received roughly twice the optimisation, and P4-D9
  measured that truncation as costing `lstm` 7.4% — an order of magnitude more than the +0.0029
  median gain claimed.
- **Parameter count.** 1 246 628 against 317 828 (P5-D5).
- **The point projection is an order statistic.** `PredictiveDistribution` sorts on construction,
  so `.point()` returns the 5th order statistic of the **sorted** fan, not the trained q=0.5
  output. With `lstm_quantile` crossing on a median 4.3% of `id` elements (up to 15% in the 1-5 s
  band), the published point forecast is in part a median-of-fan smoother, and order-statistic
  smoothing lowers RMSE by itself.

The effect does clear the seed spread (median |delta| / pooled seed sd = 3.3 for `lstm_quantile`,
7.4 for `tcn_quantile`; sign consistent in 36/36 and 34/36 cells), so CLAUDE.md non-negotiable 5
is satisfied — the failure is attribution, not significance. **Reduced to "no measurable point-
accuracy cost".** Also required by non-negotiable 6 and omitted from the draft:
`dlinear_gaussian` **loses to `dlinear` in 28 of 36 `id` cells**.

**4. The audit's most useful finding: two proper scoring rules in the same file contradict the
PICP story, and the draft used neither.** `probabilistic.csv` carries `winkler_mean` and
`crps_mean`, which score location and sharpness jointly. Mean rank of 6, per regime:

| regime | best -> worst by Winkler |
|---|---|
| `id` | `lstm_g` 1.75, `lstm_q` 1.94, `tcn_g` 2.97, `tcn_q` 3.36, **`dlinear_q` 5.36, `dlinear_g` 5.61** |
| `unseen_seastate` | `tcn_g` 1.61, `tcn_q` 1.64, `lstm_g` 3.78, `dlinear_g` 4.11, `dlinear_q` 4.58, `lstm_q` 5.28 |
| `unseen_vessel` | `tcn_q` 2.06, `tcn_g` 3.00, `dlinear_q` 3.03, `lstm_g` 3.89, `dlinear_g` 3.94, `lstm_q` 5.08 |
| `unseen_heading` | `dlinear_q` 1.22, `dlinear_g` 1.83, `tcn_g` 3.19, `tcn_q` 4.22, `lstm_q` 5.00, `lstm_g` 5.53 |

CRPS gives the same ordering, except on `unseen_vessel` where `dlinear_q` (2.92) and `tcn_g`
(3.22) swap 2nd and 3rd. **DLinear is last on `id` and mid-table out of distribution; it wins
only on `unseen_heading`, the floored regime.** So "DLinear is the robustly calibrated family" is a
PICP-in-band result that both proper scores contradict everywhere except the artifact regime.
Selecting the metric that suits the narrative after both were computed is cherry-picking, and the
band counts now ship with the Winkler and CRPS ranks beside them.

**5. "Coverage collapses under sea-state shift" describes two models, not one model shifted.**
`dmf.data.splits.build_split` gives the `id` model seeds 0-31 of **every** cell, SS6 included,
while the `unseen_seastate` model trains on SS3-SS5 only. `0.858 -> 0.648` is therefore a contrast
between two separately fitted models on two disjoint test sets: the training corpus changed as
well as the test distribution. P5-D12 recorded the deltas as unpaired but justified it only by the
realizations differing, not by the rows being different fits. The numbers stand; the mechanism
sentence is corrected to "a model trained without SS6 covers 0.65 on SS6, against 0.86 for a model
trained with it".

**6. Smaller corrections carried into the README.** Two of the six passing Reading A rows have a
realization bootstrap reaching **below** the band floor (`dlinear_gaussian` `picp_ci_lo` 0.8489,
`dlinear_quantile` 0.8410); the verdict is on the seed mean, which is the registered rule, but the
interval belongs beside it. The `unseen_heading` "covers 1.000 at `width_ratio` 11.8" figure is
**pitch specifically** — at the same cell DLinear covers 0.890 on roll and 0.919 on heave — and
stating it as a property of the regime is the P4-D14 pattern again. The -133 and -337 skills are
`-132.30 +/- 41.41` and `-336.99 +/- 121.49`, and the `signal_std` there is **0.0933 deg**, i.e.
the P1-D2 floor.

**7. There is no probabilistic baseline anywhere in this phase.** All five non-head rows in `e03`
are point models, so `probabilistic.csv` contains six learned heads and nothing else. CLAUDE.md
non-negotiable 4 is satisfied for the point column and **has no analogue for the coverage column**:
a reader cannot tell whether 102 of 216 is good, because nothing trivial was measured on that axis.
An empirical-residual interval around `persistence` or `dlinear_ols`, fitted on the validation
split, is closed-form and nearly free, and would be near-perfectly calibrated on `id` by
construction. Its absence means "the heads are calibrated in-distribution" clears no floor. This is
the largest gap in the phase and it is recorded, not fixed — Phase 6 should add it before any
further interval claim.

**8. Sweep wall time was stated as 65 h 04 m and is 66 h 04 m.** Corrected in the evidence section.
`results/e03/sweep.log` is 0 bytes (Python block-buffers stdout to a file), so the wall clock and
exit status are not traceable to a committed artifact — only the per-run `fit_time_s` column is,
and it sums to 60.65 h of SGD.

No leakage was found. The audit checked realization-level seed disjointness with the held-out-axis
assertion, the normalisation provenance guard, that `realization_seed_sequence` hashes the full
coordinate tuple so SS3-seed-5 and SS6-seed-5 share no phases, that windows never span a
realization boundary, and that `crossing_rate` is accumulated on the raw fan before sorting.

### P5-D18 — One audit finding checked and rejected, and one routing defect it exposed

**The audit's note N4 is wrong, and the check that refutes it is worth recording.** It stated that
`picp_ci_lo`/`picp_ci_hi` on the aggregated rows are the **mean** of the per-seed realization
bootstraps, and so carry no seed-to-seed component. `dmf.eval.report.build_probabilistic_table`
aggregates them as `min`/`max` — the P3-D22 envelope — and it does so on **864 of 864 rows**;
only 24 rows also happen to match the mean. The finding came from checking one row
(`dlinear_gaussian` at the gate cell, per-seed 0.8489 / 0.8490 / 0.8490) where the three seeds
agree to four decimals, so `min` and `mean` are indistinguishable there. On the rows where the
seeds genuinely disagree the two differ by up to **0.61** (`lstm_gaussian`, `unseen_heading`,
pitch @ 100: envelope 0.2411 against a mean of 0.5596) and the published value is unambiguously
the envelope.

Recorded rather than silently ignored, because the reviewer's method — verify on one row — is the
same single-cell generalisation P5-D16 and P5-D17 item 1 record *me* committing twice this phase.
It is a cheap error to make in either direction, and the defence is the same: check a row where
the quantity being distinguished actually varies.

**The finding it did expose is a routing defect, and that one is real.** P5-D17 item 7 calls the
absent probabilistic baseline "the largest gap in the phase", and the Gate 5 carry-forward block
written into `docs/IMPLEMENTATION_PLAN.md` §Phase 6 did not mention it. A gap identified as the
largest in a phase and then not routed to the phase that should close it is exactly what the
carry-forward mechanism exists to prevent — the Gate 3 carry-forward's item 1 (quiescence
detection) was ignored for two phases for want of the same discipline. It is now item 2 of that
block, with the implementation note that it needs a new closed-form branch in
`dmf.train.experiment._fit_one`.

### P5-D19 — The deep point path is unchanged by Phase 5, measured rather than argued

The Gate 5 positive control (evidence table) covers `persistence`, `window_mean`,
`damped_persistence`, `ar20` and `dlinear_ols` — every one of them closed-form or
parameter-free. **None of them touches `dmf.train.loop`, `dmf.train.losses` or the SGD path**,
all of which changed this phase along with `base.py` and the four model files. So nothing in
`results/e03/` showed that the deep *point* path still behaves as it did in Phase 4, and the
Phase 5 point-accuracy comparison quotes `results/e02` rows produced by the earlier code. The
Gate 5 audit raised this as its S8 finding; it was a genuine hole.

Closed by measurement. `tcn` was re-fitted at `head="point"` on `id`, three seeds, under current
code and the committed `e02_deep.yaml` budget:

| | seed 0 | seed 1 | seed 2 | `epochs_run` |
|---|---|---|---|---|
| Phase 5 code | 0.120901 | 0.121961 | 0.120818 | 60 / 60 / 60 |
| Phase 4 (`results/e02`) | 0.120901 | 0.121961 | 0.120818 | 60 / 60 / 60 |

**Bitwise across the whole table**: 108 of 108 per-seed per-cell skill values identical, max
absolute difference **0.0**. The head plumbing — `head: HeadKind | None = None` resolving to
`"point"`, `n_output_params` returning 1, `resolve_loss` returning `mse_loss`, the `loss_fn`
keyword defaulting on all three loop entry points — is behaviour-preserving for point models, and
that is now a measurement rather than a reading of the diff. Incidentally it also shows `tcn` *is*
bitwise reproducible run-to-run, which P4-D6 left open for two of the three deep architectures.

**One usability wrinkle found on the way, recorded not fixed.** The control run raised at the very
end, after all three fits had completed: `dmf.train.experiment._contrast_frame` refuses a run in
which a deep model is scored but no `PAIRED_CONTRASTS` pair matches the label list, and a cut-down
two-model config (`persistence` + `tcn`) matches none. The guard is deliberate — it exists so a
mistyped contrast label fails loudly instead of writing an empty file — but it makes an ad-hoc
single-model control run raise on a config that is otherwise valid. **No work was lost**, because
`run_experiment` writes `baselines_by_seed.csv` and `baselines_by_cell.csv` before the contrast
step precisely so a late failure cannot destroy finished fits, and the numbers above were read
from those files. That recoverability design earned its keep here.

### P5-D20 — The correction to P5-D17 repeated P5-D17's own defect, twice

The Gate 5 checkpoint review, run against the *corrected* text, found two failures in the two
paragraphs P5-D17 had just rewritten. Both are the same defect class P5-D17 exists to record: **a
ranking stated as universal when one regime contradicts it.**

**1. "DLinear is the best-calibrated family in every regime" is false on `unseen_seastate`.**
P5-D17 item 1 withdrew "the ordering reverses" and endorsed that sentence as "the supported
statement". It is not supported. Cells in band of 36 on `unseen_seastate`: `dlinear_quantile`
**0**, `dlinear_gaussian` **0**, `tcn_gaussian` **2**, `tcn_quantile` **3**. By median departure
from nominal, `tcn_quantile` is best at 0.1945 and `dlinear_quantile` is **fourth of six** at
0.2796. Worse, the README line supporting it compared `dlinear_quantile` 21/0/19/19 against
`lstm_quantile` 16/0/0/0 alone — the single deep row that scores zero in every OOD regime, and
therefore the comparator that most flatters the claim. `tcn_quantile`, the row that falsifies it,
was not shown. **Withdrawing a single-cell generalisation and replacing it with a
three-of-four-regimes generalisation supported by a hand-picked comparator is not a correction.**
All six rows are now printed.

**2. "Roughly ten times wider relative to signal spread in every regime" is false on the same
regime.** Median `width_ratio` for `dlinear_quantile` against the sharpest deep row: `id` 9.4x,
`unseen_vessel` 13.6x, `unseen_heading` 3.4x, **`unseen_seastate` 2.5x** — and only **1.09x**
against `lstm_gaussian` there. That matters specifically because `unseen_seastate` is the regime
the "DLinear's coverage is bought with width" argument leans on hardest; there the width advantage
is essentially absent.

**3. Also corrected: "CRPS gives the same ordering" as Winkler.** True on three regimes; on
`unseen_vessel` `dlinear_quantile` (2.92) and `tcn_gaussian` (3.22) swap 2nd and 3rd. The swap
mildly *helps* DLinear, so this one was not narrative-serving — it was simply unchecked.

**4. The sweep wall time was stated as fact and is not traceable.** `results/e03/sweep.log` is
0 bytes because Python block-buffers stdout to a file, so 66 h comes from file timestamps, not from
a committed artifact. The README now says so; the traceable figure is the 60.65 h that
`fit_time_s` sums to. P5-D17 item 8 recorded this and the README had not carried it.

**The pattern is now three-for-three in this phase** — P5-D16 (crossing "negligible", from the gate
cell), P5-D17 item 1 (the ordering "reverses", from the gate cell), and this entry (two rankings
"in every regime", from three regimes). Every one was a true statement about a subset published as
a statement about the whole, and every one was caught by an adversarial reader rather than by the
person writing it. The cheap defence, adopted here: **when a claim ranks models, print every model
and every regime the claim quantifies over, and let the table carry the exception.** A sentence
that needs a subset to be true should quote the subset in the sentence.
