# deck-motion-forecast

Short-horizon (1-15 s) forecasting of ship deck motion — roll, pitch, heave and their rates —
from JONSWAP-driven vessel simulation, for deciding when to commit to a rotorcraft touchdown on a
heaving deck. The decision the forecast serves is a *quiescent window*: an interval in which all
six channels stay inside landing limits for long enough to get the aircraft down, which is what a
full-scale manned or unmanned helicopter needs, not attitudes alone.

**All results in this repository are from simulated vessel motion. No real deck data is used,
and no sim-to-real claim is made.**

Status: **Phase 5 complete. Gate 5 passes at its registered cell. The probabilistic heads are calibrated in-distribution at 10-15 s lead and nowhere else: they over-cover across the 1-5 s operational band, and coverage falls far below nominal under every distribution shift tested. The degradation is the finding, and it is reported rather than fixed. No probabilistic baseline was run, so the coverage column has no floor to clear -- the largest gap in the phase.**

The simulator (`src/dmf/sim/`) and corpus are complete and Gate 1 passes; the realization-level
split, windowing and train-only normalization are complete and Gate 2 passes (`src/dmf/data/`).
The four baseline families — persistence, damped persistence, AR(p), DLinear — are implemented and
tested.

**Gate 3 as originally written did not pass.** AR(20) forecasts roll 3 s ahead in-distribution at
0.9987 skill vs persistence (0.9958 under the `imu` observation model), against a 0.8 "task is too
easy" threshold. This is not leakage — the shuffle control passes 36/36 — it is structural: the
vessel response is narrowband with no process noise, so a 3 s horizon is a quarter of the roll
period and a linear model identifies the system rather than approximating it. Roll is also the
*most* predictable channel in the corpus, so the original gate measured the easiest available cell.

Two things followed. The task now forecasts all six channels — attitudes **and** their rates, which
the operational metric needs and a full-scale rotorcraft landing requires — out to 15 s, where four
of six channels fall below 0.8 skill. And the gate was **restated, not relaxed**: the same 0.8
threshold, read at the decision horizon (10 s) on the binding DOF (pitch), where AR(20) scores
0.545 (`ideal`) / 0.513 (`imu`). Both the original failure and the restatement are recorded in
`docs/protocol.md` §Phase 3, which is the full decision log for this phase.

`results/baselines.csv` (`ideal`) and `results/imu/baselines.csv` (`imu`) carry the Gate 3 sweep:
1296 rows each, nine baselines x four regimes x six DOFs x six horizons, three seeds for the one
SGD-fitted model and deterministic single rows for the eight closed-form ones. Phase 4's
twelve-model sweep is in `results/e02/`.

**Read the gate number with its context.** Passing is one cell of thirty-six. On `id`, AR(20) still
exceeds 0.8 skill in 28 of 36 cells, and across the 1–5 s operational band it exceeds 0.8 in 89 of
96 cells over all four regimes. The task is easy by construction; the gate marks where it stops
being easy, not that it is hard. A zero-parameter `window_mean` baseline beats persistence — the
denominator of every skill score here — in 107 of 144 cells.

Two results worth stating plainly, both of which reversed an earlier claim of ours:

- **A converged linear model is competitive at the gate cell.** `dlinear_ols` (60 300 parameters,
  solved closed-form) scores 0.568 there against AR(20)'s 0.545 with 108 900. Read across the whole
  `id` regime the ranking flips back — AR(20) wins 28 of 36 cells, median +0.0056 — so this is a
  cell-level result, not a general one. What is general: DLinear was previously trained by SGD to a
  60-epoch cap that early stopping never reached, and that shortfall was being read as an
  architecture gap. Removing it drops AR(20)'s wins over DLinear from 107/144 to 89/144 under
  `ideal`.
- **The rate channels are worth less than they first appeared.** At matched parameter count the
  paired per-cell effect is +0.0006 median under `ideal` and +0.0025 under `imu` — small and positive
  in both, and of the same order as simply doubling the lag budget. An earlier version of this line
  claimed the ordering reversed between modes; that came from comparing two unpaired medians and does
  not survive a paired contrast. The measurement also still carries a downward bias, because matching
  parameter counts left the two models with different lag depths.

`docs/protocol.md` §Phase 3 is the full decision log, including the defects an adversarial audit
found in the first sweep and what changed as a result.

A caveat that travels with every number here: the generator has no process noise, so the
achievable-skill ceiling is unrealistically high and absolute values flatter every model. Only
relative comparisons and out-of-distribution degradation should be read as findings.

## Phase 4 — deep models

Three architectures were added: a TCN (dilated causal convolutions, receptive field 253 >= the
200-sample lookback, asserted arithmetically in `tests/test_windows.py`), an encoder-only
Transformer over 1 s patches, and a 2-layer LSTM. All twelve models — the nine Phase 3 baselines
re-fitted and re-scored beside the three new ones — were trained in **one** run under one data
pipeline, one normalization, one horizon list and one early-stopping rule
(`configs/experiment/e02_deep.yaml`, 31 h on one A4000, **`ideal` observation mode only**; the
`imu` ablation is Phase 6.3). Full artifacts in `results/e02/`; the Gate 3 artifacts in
`results/` are untouched.

**Gate 4 passes.** Both the original criterion (3 s on `id` vs damped persistence, 18 of 18
model-DOF rows) and the restated one (10 s on pitch vs the stronger trivial baseline, 3 of 3),
with every paired bootstrap interval excluding zero and margins exceeding the three-seed spread
by two to four orders of magnitude (Reading B 139x-839x, Reading A 709x-12 033x). `make gate4` regenerates the read-out from the CSVs.

Skill at the gate cell (pitch, 10 s lead):

| model | params | `id` | `unseen_seastate` | `unseen_heading` | `unseen_vessel` |
|---|---:|---:|---:|---:|---:|
| `damped_persistence` | 6 | 0.366 | 0.308 | 0.225 | 0.172 |
| `dlinear` | 60 300 | 0.515 | **0.478** | 0.519 | 0.502 |
| `dlinear_ols` | 60 300 | 0.568 | 0.450 | **0.578** | 0.534 |
| `ar20` | 108 900 | 0.545 | 0.158 | -49.4 | 0.479 |
| `ar40` | 216 900 | 0.577 | 0.086 | -44.9 | 0.526 |
| `tcn` | 196 804 | 0.835 | 0.277 | -81.1 | **0.830** |
| `transformer` | 2 712 708 | 0.808 | 0.195 | -79.2 | 0.725 |
| `lstm` | 317 828 | **0.868** | 0.344 | **-279.4** | 0.801 |

**Where the deep models win, and where they do not.** Counted by paired bootstrap against
`dlinear_ols` (36 cells per regime, interval excluding zero in favour of the deep model / of
`dlinear_ols`):

| | `tcn` | `transformer` | `lstm` |
|---|---|---|---|
| `id` | 27 - 7 | 25 - 10 | 27 - 9 |
| `unseen_seastate` | 17 - 15 | 9 - 21 | 10 - 20 |
| `unseen_heading` | 0 - 32 | 0 - 36 | 0 - 35 |
| `unseen_vessel` | 16 - 20 | 10 - 24 | 12 - 20 |
| **all 144** | 60 - 74 | 44 - 91 | 49 - 84 |
| *excluding `unseen_heading`* | *60 - 42* | *44 - 55* | *49 - 49* |

Read those two bottom rows together. Over all 144 cells every deep model loses more than it
wins — but roughly 40% of the losses come from `unseen_heading` alone, which is a corpus
artifact (below), and dropping it reverses the result for `tcn` and ties it for `lstm`. The
honest summary is **`tcn` wins outside `unseen_heading`, `lstm` ties, `transformer` loses.**

The split by lead time is sharper, and cuts against the deep models where it matters most
(excluding `unseen_heading`):

| band | `tcn` | `transformer` | `lstm` |
|---|---|---|---|
| **1-5 s** — the operational band this project exists to serve | 34 - 34 | 19 - 47 | 23 - 42 |
| **10-15 s** — where Gate 4 is read | 26 - 8 | 25 - 8 | 26 - 7 |

The deep models' advantage is concentrated at long lead times. In the 1-5 s band, `tcn` ties the
linear model and the other two lose to it. Gate 4's 10 s cell was fixed in advance of the sweep
and on Gate-3-era reasoning, so this is not cell-picking — but the gate is read where the deep
models look best, and that belongs next to the word "passes".

**Robustness.** Counting cells where normalised RMSE exceeds 1.0 — the model does worse than
predicting the scored partition's mean — `dlinear_ols` is the most robust model in the table at
8 of 144, against `tcn` 25, `transformer` 26, `lstm` 27. That gap also narrows once
`unseen_heading` is removed, where the worst-case gap is `transformer` 12 against `dlinear_ols` 7
— 1.7x rather than 3.4x, and `ar40` and `ar_attitude_only` are then marginally better than
`dlinear_ols` at 6 each. The `nrmse` column is new this phase and the finding was invisible
without it: `tcn` on `unseen_vessel`/roll at 10 s reads as **+0.388 skill** while sitting at
`nrmse` **1.183 +/- 0.019** — worse than predicting that partition's mean — because persistence
is worse still there (`nrmse` 1.512). Skill against a bad reference is not evidence of a good
forecast.

**`unseen_heading` needs care in both directions.** The magnitude is a corpus artifact: that
regime's test set is beam seas, where the pitch heading factor sits on the `eps = 0.05` residual
floor (see limitations), so a model fitted where pitch is a real signal imposes an amplitude on a
channel that has almost none — hence -49 to -279. The two channel-independent DLinear rows
structurally cannot import that amplitude from roll, and survive. **But the floor does not
explain the direction on the other channels**: on roll and heave, where it does not apply, the
deep models still lose 4-6 of 6 cells each, and `lstm` exceeds `nrmse` 1.0 on heave and
heave_rate too. This is a real heading-generalisation failure with an artifact on top of it, not
an artifact alone.

Two methodological notes, both in `docs/protocol.md` §Phase 4:

- **A budget pilot must run at the cap it justifies.** An 18-epoch pilot predicted early stopping
  would not fire; at cap 60 it fired for the Transformer and LSTM, because the cosine schedule is
  sized by the cap and epoch 16 sits near peak learning rate rather than at the end of an anneal.
  Both are *worse* at cap 60 than at cap 18 (3.2% and 7.4%). The cap was not changed after seeing
  this — that would be selecting a hyperparameter on the results it produced (P4-D9). The two
  models carrying most of the loss column above are therefore shipped in a configuration this
  project has measured as worse than one it already ran (P4-D14).
- **Two prerequisites were built first**: normalised RMSE, which `docs/protocol.md` had mandated
  twice and implemented nowhere, and a paired bootstrap for model-vs-model differences, without
  which an earlier claim in this project was published wrong twice (P4-D3, P4-D4).

The re-scored Phase 3 rows reproduce Phase 3 **bitwise** — all 1296 shared rows agree exactly in
`rmse_mean`, `skill_mean` and the bootstrap bounds, while `fit_time_s` differs, so it is a real
re-fit and not a join. That is the positive control on the whole pipeline (P4-D10).

**The limitation that bounds every comparison above.** The simulator applies a linear RAO to a
finite sum of sinusoids with no process noise, so a linear forecaster is **optimal for this
corpus by construction** and enough lags identify the system rather than approximate it. "A
linear model matches three deep architectures" is therefore a much weaker statement here than it
would be on a stochastic process, and says nothing about how these architectures would rank on
real deck motion with nonlinear roll damping and short-crested excitation (P4-D16). The cell
counts are also a description of one table, not 144 independent hypothesis tests (P4-D17).

## Phase 5 -- probabilistic heads

Two predictive-distribution heads were added behind a config flag -- a **quantile** head (nine
levels, 0.05 to 0.95, pinball loss, post-hoc sorting) and a **Gaussian** head (mean and
log-variance, NLL) -- and attached to three backbones: `dlinear`, `tcn` and `lstm`. Six
probabilistic rows, four regimes, three seeds, one run
(`configs/experiment/e03_probabilistic.yaml`, ~66 h on one A4000, `ideal` mode only -- wall clock
from file timestamps, since `sweep.log` captured nothing; the traceable figure is the 60.65 h of
SGD that `fit_time_s` sums to). Full
artifacts in `results/e03/`; `results/` and `results/e02/` are the Gate 3 and Gate 4 records and
are untouched.

`src/dmf/models/heads.py` is structured as a **calibration seam**: a `PredictiveDistribution`
value object and an `IntervalPredictor` protocol, so a `ConformalWrapper` can be attached later
without touching any model. That claim is exercised by a test, not asserted in a docstring.

**Gate 5 passes at its registered cell.** PICP@90 within [0.85, 0.95] on `id`, read at pitch /
10 s -- the cell Gates 3 and 4 are read at, registered in `docs/protocol.md` P5-D2 **before the
sweep ran** -- is 6 of 6. Across every `id` cell it is **102 of 216**.

Two of those six pass on a seed mean whose own realization bootstrap reaches below the band floor
(`dlinear_gaussian` `picp_ci_lo` 0.8489, `dlinear_quantile` 0.8410). The verdict is taken on the
mean, which is the registered rule, but the interval belongs beside it. Note also that the DLinear
rows' seed std is 1e-4 to 2e-4 against a realization CI half-width of 0.016 -- the pinball
objective on a linear model is convex, so three seeds land in the same place and "exceeds the seed
std" is not a meaningful test for those rows.

### Coverage under distribution shift -- the finding

Median PICP@90 over each regime's 36 cells. Nominal is 0.90:

| model | `id` | `unseen_seastate` | `unseen_heading` | `unseen_vessel` |
|---|---:|---:|---:|---:|
| `dlinear_quantile` | 0.903 | 0.620 | 0.939 | 0.943 |
| `dlinear_gaussian` | 0.925 | 0.660 | 0.960 | 0.963 |
| `tcn_quantile` | 0.969 | 0.706 | 0.244 | 0.663 |
| `tcn_gaussian` | 0.958 | 0.689 | 0.405 | 0.548 |
| `lstm_quantile` | 0.958 | 0.390 | 0.206 | 0.325 |
| `lstm_gaussian` | 0.951 | 0.534 | 0.189 | 0.433 |

**Withholding a sea state breaks every model.** Only **5 of 216** `unseen_seastate` cells stay in
band. At the gate cell `lstm_quantile` reads 0.912 against **0.399** -- an interval sold as 90%
covering 40% of the time.

Read that as a contrast between **two separately trained models**, not one model evaluated off
its distribution: `build_split` gives the `id` model seeds 0-31 of every cell *including* SS6,
while the `unseen_seastate` model trains on SS3-SS5 only. The training corpus changed as well as
the test set. Precisely: a model trained without SS6 covers 0.40-0.66 on SS6, against 0.86-0.92
for a model trained with it.

The mechanism is visible only in the sharpness column. Absolute widths mostly *grow* under shift,
which looks like adaptation. Measured against the scored partition's own spread (`width_ratio`,
where 1.0 is an unconditional interval matched to that spread), they do not: `lstm_quantile`
0.034 -> 0.044, and `dlinear_quantile` actually *falls*, 0.319 -> 0.110. SS6 motion is far larger
than the SS3-SS5 the heads were fitted on and the learned width does not scale with it. **The head
memorised an amplitude rather than learning a conditional one.** This is exactly the
coverage-under-domain-shift story that motivates split conformal calibration, which is what the
seam in `heads.py` exists for. It is reported, not corrected.

**DLinear is the best-calibrated family on three regimes of four.** Cells inside the band, of 36:

| model | `id` | `unseen_seastate` | `unseen_heading` | `unseen_vessel` |
|---|---:|---:|---:|---:|
| `dlinear_quantile` | **21** | 0 | **19** | **19** |
| `dlinear_gaussian` | 19 | 0 | 15 | 11 |
| `lstm_gaussian` | 18 | 0 | 0 | 2 |
| `lstm_quantile` | 16 | 0 | 0 | 0 |
| `tcn_gaussian` | 15 | **2** | 3 | 4 |
| `tcn_quantile` | 13 | **3** | 0 | 5 |

**`unseen_seastate` is the exception and it is not a small one**: every family scores zero cells in
band, and by median departure from nominal TCN is least bad (0.195 and 0.211) while
`dlinear_quantile` is *fourth* of six at 0.280. Nothing reverses -- an earlier draft claimed the
ordering flips out of distribution, generalising from the single gate cell -- but the corrected
sentence was itself first written as "best in every regime" and supported by a comparison against
`lstm_quantile` alone, the one row that is 0 everywhere out of distribution. Both are withdrawn
(P5-D17, P5-D20), which is why all six rows are printed above rather than two.

**Two things must travel with that count, and the second undoes most of it.**

*DLinear buys coverage with width, except where it matters most.* Its intervals are far wider
relative to signal spread than the sharpest deep model's on `id` (median `width_ratio` 0.319
against 0.034, **9.4x**) and on `unseen_vessel` (**13.6x**), less so on `unseen_heading`
(**3.4x**) -- and on **`unseen_seastate` the advantage is nearly gone at 2.5x**, falling to 1.09x
against `lstm_gaussian`. So "best calibrated" means "least badly calibrated at a sharpness the deep
models beat by an order of magnitude" on two regimes, and much less than that on the regime where
every family fails.

*The proper scoring rules rank it last.* `probabilistic.csv` also carries **Winkler** and **CRPS**,
which score location and sharpness jointly rather than leaving the trade-off to the reader. Mean
rank of six, by Winkler (CRPS gives the same ordering except on `unseen_vessel`, where
`dlinear_quantile` at 2.92 and `tcn_gaussian` at 3.22 swap 2nd and 3rd):

| regime | best -> worst |
|---|---|
| `id` | `lstm_g` 1.75, `lstm_q` 1.94, `tcn_g` 2.97, `tcn_q` 3.36, **`dlinear_q` 5.36, `dlinear_g` 5.61** |
| `unseen_seastate` | `tcn_g` 1.61, `tcn_q` 1.64, `lstm_g` 3.78, `dlinear_g` 4.11, `dlinear_q` 4.58, `lstm_q` 5.28 |
| `unseen_vessel` | `tcn_q` 2.06, `tcn_g` 3.00, `dlinear_q` 3.03, `lstm_g` 3.89, `dlinear_g` 3.94, `lstm_q` 5.08 |
| `unseen_heading` | `dlinear_q` 1.22, `dlinear_g` 1.83, `tcn_g` 3.19, `tcn_q` 4.22, `lstm_q` 5.00, `lstm_g` 5.53 |

**DLinear is last on `id` and mid-table out of distribution; it wins only on `unseen_heading`, the
floored regime.** So the coverage-in-band ranking and the two proper scores disagree almost
everywhere, and reporting only the first would have been choosing the metric that suited the
story.

### What Gate 5 does not say

**The gate is read in the band where the heads look best.** Cells in band on `id`, split by lead
time:

| band | `dlinear_q` | `dlinear_g` | `tcn_q` | `tcn_g` | `lstm_q` | `lstm_g` |
|---|---:|---:|---:|---:|---:|---:|
| **1-5 s** (24 cells) -- the operational band this project exists to serve | **12** | 10 | 1 | 3 | 4 | 6 |
| **10-15 s** (12 cells) -- where Gate 5 is read | 9 | 9 | **12** | **12** | **12** | **12** |

The four deep rows are **perfectly calibrated at 10-15 s and systematically over-cover at 1-5 s**,
where 18 to 23 of their 24 cells sit above 0.95 -- while being 5.9x (`tcn_quantile`) to 8.9x
(`lstm_quantile`) sharper than `dlinear_quantile` in that band. None sits below 0.85, so the failure is
conservative -- but it is a failure, and it is in the band the landing decision is taken in. The
cause is P3-D1: this corpus is nearly deterministic at short lead, so the residual is tiny and the
learned interval, though only ~3% the width of an unconditional one, is still wider than warranted.
Reading A's cell was fixed in advance, so this is not cell-picking -- but "the heads are well
calibrated in-distribution" is true at the gate cell and false across the operational band, and the
band belongs next to the claim.

**An advance prediction was wrong, and the correction is the more interesting result.** P5-D6
predicted, before the sweep, that `unseen_heading` pitch would show near-perfect coverage at
meaningless width, for the P1-D2 residual-floor reason. At that cell (pitch, 10 s) it holds exactly
for DLinear -- **1.000** coverage at `width_ratio` 11.8 -- and is wrong for the deep heads, which
**under**-cover at **0.500, 0.570, 0.612 and 0.832** while themselves sitting 4.3-8.5x wider than
unconditional. Those deep figures carry very large seed spreads (`lstm_gaussian` 0.570 +/- 0.305),
and the qualifier is pitch specifically: at the same cell DLinear covers 0.890 on roll and 0.919 on
heave. The prediction reasoned about interval width and silently assumed the interval stays
centred on the target. It does not: the deep models import cross-channel structure and impose a
roll-driven amplitude on a channel that has none (point skill there is **-132.3 +/- 41.4** and
**-337.0 +/- 121.5** against `dlinear_ols`'s +0.577, on a channel whose `signal_std` is 0.093 deg,
i.e. the P1-D2 floor itself), and a wide interval centred in the wrong place still misses.
Coverage depends on location and width jointly. Recorded in full as P5-D13, and the entry was
written in advance precisely so that being wrong would be visible.

That correction generalises: on the four channels the residual floor does *not* touch, the deep
heads still cover only **17-29%** of targets on `unseen_heading` while forecasting those channels
well (roll at 10 s: `tcn_gaussian` 0.796 skill). This is a real heading-generalisation failure of
the intervals, not the corpus artifact.

### Two smaller results

**The heads carry no measurable point-accuracy cost.** Every probabilistic row also ships RMSE and
skill against the same persistence denominator as every other row in the project. Against the
Phase 4 point rows on `id` the quantile heads come out marginally ahead -- `lstm_quantile` in 36
of 36 cells, `tcn_quantile` in 34 of 36, median +0.0029 and +0.0011 -- and the difference does
exceed the combined seed spread in 33-36 of 36 cells.

**That is stated as "no cost", not as "better", because three confounds all push the same way** and
together are larger than the effect. `lstm` in Phase 4 early-stopped at 32/33/34 epochs while
`lstm_quantile` here ran 37/60/60, so the quantile row got roughly twice the optimisation -- and
P4-D9 measured that truncation as costing `lstm` 7.4%, an order of magnitude more than the gain
claimed. The heads are not parameter-matched (1 246 628 against 317 828). And the point projection
is the median of the **sorted** fan, so where the raw fan crosses it is partly an order-statistic
smoother, which lowers RMSE by itself. The comparison is also across two separate sweeps with no
paired resample. It is *not* across a behaviour change in the point path: `tcn` re-fitted at
`head=point` under Phase 5 code reproduces its Phase 4 row **bitwise**, 108 of 108 cells
(P5-D19).

Not all six rows gain: **`dlinear_gaussian` loses to `dlinear` in 28 of 36 `id` cells.**

**Post-hoc quantile sorting is not cosmetic.** At the gate cell the crossing rate is 5e-6 to
0.0038, which reads as a no-op. Over the full table `dlinear_quantile`'s raw fan is inverted on a
median 6% of elements and, in its worst cell, **96.6%**. It is a short-horizon phenomenon --
0.563 at a 1 s lead against 0.000 at 15 s -- because the predictive spread at short lead is so
small that nine levels squeezed into it are numerically indistinguishable. The sort does real work
in the 1-5 s band and nowhere else.

### Limitations specific to this phase

- **There is no probabilistic baseline anywhere in this phase, and it is the largest gap.** All
  five non-head rows in `e03` are point models, so `probabilistic.csv` holds six learned heads and
  nothing else. Non-negotiable 4 -- "a result without its persistence baseline is not a result" --
  is satisfied for the point column and **has no analogue for the coverage column**: a reader
  cannot tell whether 102 of 216 is good, because nothing trivial was measured on that axis. An
  empirical-residual interval around `persistence` or `dlinear_ols`, fitted on the validation
  split, is closed-form and nearly free, and would be near-perfectly calibrated on `id` by
  construction. Recorded, not fixed (P5-D17).
- **Nothing here is calibrated.** `heads.py` provides the seam a `ConformalWrapper` attaches to and
  a test proves the seam composes, but no conformal calibration is run -- that is Project 6. These
  are **uncalibrated heads**; the project deliverable "calibrated prediction intervals" is not yet
  met.
- **The coverage numbers carry less certification than the skill numbers.** The shuffle control
  refits AR(20), a *point* model, so it certifies the point pipeline the heads are built on. No
  shuffled-target head is fitted anywhere, and the untrained control runs on a quantile model's
  *median*. **There is no leakage control and no untrained control on an interval** (P5-D10).
- **The degradation deltas are unpaired and carry no interval.** Two regimes score different
  realizations, so no common bootstrap resample exists. Each side carries its own CI; the delta
  carries none, deliberately (P5-D12).
- **The heads are not parameter-matched.** The final projection widens with the head, so a quantile
  row carries ~9x the head parameters of its point twin (`lstm` 317 828 -> 1 246 628). A
  head-vs-head or head-vs-point difference is not an architecture result (P5-D5).
- `best_val_loss` is not comparable across heads -- each model is early-stopped on its own
  objective, named in `val_loss_name` (P5-D4).
- **Short-horizon calibration here is not measuring wave uncertainty.** The generator has no
  process noise, so at 1-5 s the correct predictive distribution is close to a point mass -- AR(20)
  reaches 0.99998 skill on `id`/heave at 1 s. PICP@90 in that band scores a model's ability to
  calibrate its own optimisation residual, not aleatoric uncertainty, and will not transfer.
- The Phase 4 caveat that a linear forecaster is **optimal for this corpus by construction**
  (P4-D16) applies here unchanged, and the no-process-noise ceiling makes every interval sharper
  than one fitted to real deck motion would be.

`docs/protocol.md` §Phase 5 is the full decision log: P5-D1 to P5-D8 were written *before* the
sweep, P5-D13 to P5-D16 record what the sweep falsified.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint && make test
```
