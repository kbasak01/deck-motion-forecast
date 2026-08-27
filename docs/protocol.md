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

**No `imu` denominator table is pinned anywhere yet.** Worth adding when someone measures it.

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

### P3-D13 — `dlinear_mc` added, then removed; replaced by `ar_attitude_only`

`src/dmf/models/dlinear_mc.py` was added outside the plan's four baseline families to separate
"DLinear loses to AR because it sees fewer channels" from "DLinear loses because it is a worse
architecture": canonical DLinear is channel-independent and sees only the target channels, while
AR(p) sees all six.

The intent was sound; the instrument was not. At the revised geometry (P3-D4) its output layer
scales with both `H` (50 -> 150) and `C_out` (3 -> 6), giving DLinear **60 300**, AR(20)
**108 900**, AR(40) **216 900**, DLinear-MC **2 161 800**. A 36x capacity gap does not isolate the
information set, it confounds it with capacity — which is precisely the confound the control was
built to remove.

**Removed.** The same question is answered instead by `configs/model/ar_attitude_only.yaml`: AR(20)
fitted on the three attitude channels only, against the existing six-channel `ar20`. Same
architecture, same solver, same fit procedure — the only difference is the input information set,
so the comparison is clean. It costs no extra pass over the training split: `lag_features` is
lag-major with channels within each lag, so the three-channel design is a column subset of the
six-channel one and its moments slice out of the cached six-channel accumulation, the same trick
that already lets one pass at `p = 40` fit every order.

Side effect worth recording: DLinear is now the only SGD-fitted model in `e01_baselines`, which
halves the full sweep — 12 SGD runs (1 model x 3 seeds x 4 regimes) instead of 24.

**Measured on `id` (closed-form slice, 1 465 776 training windows).** Dropping the three rate
channels costs 0.002-0.09 skill: largest on `heave` at 10 s (0.086) and `pitch_rate` at 5 s
(0.049), smallest on `roll` at every horizon (<= 0.031). So the rate channels carry real
information, concentrated in heave and in the rate targets themselves, and almost none in roll —
consistent with roll being the most narrowband channel and therefore the most self-predictable.

A free second reading fell out of it. `ar_attitude_only` (20 lags x 3 channels) lands on exactly
`ar10`'s parameter count (10 lags x 6 channels) — 54 900 either way — giving a budget-matched pair
that was not designed for. `ar_attitude_only` loses to `ar10` on every DOF and horizon: **at fixed
capacity the rate channels buy more than the extra lags do.**

### P3-D16 — AR normal equations are severely ill-conditioned; forecasts are usable, coefficients are not

Measured on `id`/train: `cond(R)` for `ar20` is **1.45e19**, and 2.45e18 for `ar_attitude_only`.
Both are past float64 resolution for an unregularised solve — `ridge = 1e-6` on the whitened
correlation matrix is what makes the Cholesky succeed at all, and the coefficients it returns are
substantially determined by that regularisation rather than by the data alone.

This is expected for a 240-feature design built from 200 samples of a narrowband, heavily
oversampled signal: adjacent lags are nearly collinear. It does not invalidate the forecasts —
those are measured directly on held-out realizations and are what every table in this project
reports — but it does mean **AR coefficient values must not be read structurally**. No claim of the
form "the model learned the roll period" or "lag k dominates" is supportable from these
coefficients. Recorded because the temptation to interpret a linear model's weights is exactly
where this would go wrong.

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

**Why changing lr is safe here and would not be for a deep model.** DLinear under MSE is convex;
its optimum is the closed-form least-squares solution, and
`tests/test_models.py::test_dlinear_sgd_reaches_the_closed_form_optimum` asserts SGD reaches it.
Batch and lr therefore govern how fast it converges, not where — the oracle test is the guard, and
it passes at the new settings. `lr = 2e-3` is square-root scaling for a 4x batch, chosen over
linear scaling's 4e-3 as the conservative option. 4096 was not taken despite being faster still:
at 358 steps/epoch the schedule has too few steps for warmup and cosine decay to mean much.

Applies uniformly to every SGD model in both `e01_baselines` and `e01_baselines_imu`, so the Gate 4
fairness requirement is unaffected. **Phase 4 must re-derive these numbers before adopting them** —
this measurement is for a 60 300-parameter linear map, and a TCN or Transformer will not be
fixed-overhead-bound in the same way.

### Gate 3 evidence

| Criterion | Where | Evidence |
|---|---|---|
| `results/baselines.csv` exists with skill vs persistence for every baseline / horizon / DOF / regime | `dmf.train.experiment.run_experiment` | Closed-form subset measured on `id` in both observation modes; **the full 7-model x 4-regime sweep has not been run** |
| Gate cell read and acted on | P3-D1 .. P3-D4 | AR(20) 0.9987 (`ideal`) / 0.9958 (`imu`) at 3 s roll `id`, against a 0.8 threshold; task revised (P3-D4); gate restated and applied in P3-D12, measured 0.545 / 0.513 against 0.8 |
| Result not attributable to leakage | `baselines_controls.csv`, P3-D1 | Shuffle control: 36/36 rows pass in both modes, `|excess| <= 0.0043` against a 2% tolerance |
| Persistence denominator correct | P3-D6 | Re-measured on 434 304 windows, verified two independent ways, all six DOFs pinned in tests |
| Underperforming cells reported, not dropped | P3-D7 | 12 of 144 cells negative, pinned as an exact set that fails on both a new loss and a silent repair |
| Normalisation provenance | P3-D11 | Asserted at scoring time, fails closed |
