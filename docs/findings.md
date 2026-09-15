# Findings — the phase-by-phase record

**These are simulated results.** The corpus is JONSWAP-driven vessel simulation; no real deck
data enters this project at any point, and no number here is evidence about a real ship.

This is the narrative layer of the project, between the [README](../README.md) — which carries
the headline tables — and [`docs/protocol.md`](protocol.md), which is the complete decision log
at roughly ten times this length. It is written phase by phase, in the order the work happened,
and it keeps the claims this project **withdrew** next to the ones that replaced them. That is
deliberate: several headline readings here reversed at least once, and a document that shows
only the surviving version of each hides how much of the work was finding out that the first
version was wrong.

Where this file and `docs/IMPLEMENTATION_PLAN.md` disagree, the plan is the *original* plan and
this is what happened.

Status: **Phases 1–8 complete.** Gates 1, 2, 4, 6, 7 and 8 pass as written; Gate 3 failed as
written and was restated at the same threshold; Gate 5 passes at its registered cell and does
not pass across the surrounding table. Every one of those is unpacked below.

---

## Phase 3 — baselines, and the gate that failed

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

All twelve models, with `nrmse` in distribution. Closed-form models are deterministic and carry one
seed; SGD models are mean ± std over three.

| model | params | `id` | `unseen_seastate` | `unseen_heading` | `unseen_vessel` | `nrmse` (`id`) |
|---|---:|---:|---:|---:|---:|---:|
| `persistence` | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.256 |
| `window_mean` | 0 | 0.3656 | 0.3082 | 0.2246 | 0.1724 | 1.001 |
| `damped_persistence` | 6 | 0.3657 | 0.3082 | 0.2247 | 0.1724 | 1.000 |
| `ar10` | 54 900 | 0.5348 | 0.1384 | **−50.11** | 0.4644 | 0.857 |
| `ar20` | 108 900 | 0.5446 | 0.1584 | **−49.43** | 0.4786 | 0.848 |
| `ar40` | 216 900 | 0.5765 | 0.0856 | **−44.94** | 0.5263 | 0.817 |
| `ar_attitude_only` | 108 900 | 0.5419 | 0.2112 | **−18.27** | 0.4778 | 0.850 |
| `dlinear` | 60 300 | 0.5147 ± 0.0001 | **0.4775** ± 0.0003 | 0.5191 ± 0.0013 | 0.5019 ± 0.0009 | 0.875 |
| `dlinear_ols` | 60 300 | 0.5683 | 0.4499 | **0.5775** | 0.5344 | 0.825 |
| `tcn` | 196 804 | 0.8346 ± 0.0006 | 0.2772 ± 0.0322 | −81.09 ± 6.65 | **0.8298** ± 0.0034 | 0.511 |
| `transformer` | 2 712 708 | 0.8080 ± 0.0032 | 0.1953 ± 0.0270 | −79.22 ± 25.17 | 0.7250 ± 0.0141 | 0.550 |
| `lstm` | 317 828 | **0.8680** ± 0.0016 | 0.3437 ± 0.0241 | **−279.44** ± 59.98 | 0.8007 ± 0.0044 | 0.456 |

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
(Phase 8 qualifies this: on an independent hydrodynamic model none of the three
transfers, and `tcn` falls to -0.9033 skill at the same cell where it reads 0.8604 here.)

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


## Phase 6 — evaluation, the operational metric, and five ablations

Phase 6 is where the project stops reporting RMSE and starts reporting the decision. It produced
`results/results.md` — 20 tables, every one of them carrying the CSV it was read from and that
file's row count — and it is regenerated end to end by `make eval`.

**Gate 6 passes, 7 of 7 pre-registered predicates.** It is a *process* gate with no threshold and
no cell: re-rendering the committed CSVs reproduces `results.md` byte for byte, every table names
a source that exists, every stated row count matches, every F1 carries its base rate, and no
coverage row pools the 1–5 s and 10–15 s bands. What a Gate 6 pass does **not** certify is also
recorded (P6-D14): it covers the traceability of the nine planned tables and says nothing about
whether the interval rule or the interval controls are any good.

### The operational metric: quiescent-window detection

A quiescent window is an interval in which roll, pitch and heave rate all stay inside landing
limits for long enough to get an aircraft down. Two threshold sets (`permissive` and `strict`)
and two decision rules: the **point** rule flags a window when the point forecast is inside
limits, the **interval** rule flags it when the *whole* 90 % predictive interval is.

Mean F1 over the scorable cells of each regime, with the base rate and the two forecast-free
nulls beside it. `always_quiescent` says yes always; `rate_matched` fires at the true onset rate
with chance timing. Both are forecast-free, so they apply to either rule. Spreads are across the
three training seeds; closed-form models are deterministic and marked `det.`:

| regime | thresholds | base rate | best **interval** | best **point** | `always_quiescent` | `rate_matched` |
|---|---|---:|---|---|---:|---:|
| `id` | permissive | 0.5655 | `dlinear_gaussian` **0.4405** ± 0.0012 | `lstm_quantile` 0.2724 ± 0.0230 | 0.0406 | 0.0463 |
| `id` | strict | 0.3494 | `lstm_gaussian` **0.4211** ± 0.0061 | `lstm_quantile` 0.1939 ± 0.0255 | 0.0143 | 0.0144 |
| `unseen_seastate` | permissive | 0.2068 | `tcn_gaussian` **0.3749** ± 0.0106 | `lstm_gaussian` 0.1666 ± 0.0037 | 0.0352 | 0.0419 |
| `unseen_seastate` | strict | 0.0245 | `lstm_gaussian` **0.3863** ± 0.0107 | `lstm_gaussian` 0.1443 ± 0.0040 | 0.0042 | 0.0054 |
| `unseen_heading` | permissive | 0.5227 | `dlinear_gaussian` **0.4292** ± 0.0010 | `damped_persistence` 0.1998 (det.) | 0.0382 | 0.0455 |
| `unseen_heading` | strict | 0.3511 | `tcn_gaussian` **0.2031** ± 0.0238 | `dlinear_ols` 0.0798 (det.) | 0.0132 | 0.0136 |
| `unseen_vessel` | permissive | 0.6686 | `dlinear_gaussian` **0.4149** ± 0.0038 | `damped_persistence` 0.2119 (det.) | 0.0404 | 0.0478 |
| `unseen_vessel` | strict | 0.4296 | `tcn_gaussian` **0.2748** ± 0.0027 | `tcn_quantile` 0.0980 ± 0.0008 | 0.0167 | 0.0224 |

Four things in that table, and three of them are unflattering.

1. **The interval rule roughly doubles the point rule, everywhere.** Requiring the whole band to
   clear the limit is a materially better detector than requiring the median to, which is the
   clearest operational argument in the project for carrying a predictive distribution at all.
2. **`persistence` and `window_mean` score F1 = 0.0000 in every cell.** They never fire. That is
   not a rounding artifact — a forecast that repeats the last sample never produces a sustained
   in-limits run at these thresholds. The skill-score denominator of the entire project is
   useless at the task the project exists to serve.
3. **The best F1 in the project is 0.44.** Against base rates of 0.02–0.67 the nulls are far
   below that, so the detectors are doing real work — but an F1 of 0.44 on the permissive
   thresholds, and 0.20 on `unseen_heading` strict, is not a solved problem, and the point rule
   is worse than that everywhere.
4. **The false-alarm rate is the usability problem the F1 hides.** For the best-F1 model in each
   row it runs 2.6–41.7 per minute on the point rule and 0.3–5.6 on the interval rule; across all
   models and scorable cells the medians are 16.7 and 2.8, with maxima of 117.0 and 60.0. A
   detector firing 40 times a minute is not something a pilot or an autoland controller can be
   asked to act on.

**Scorability is a property of the cell, not of the sea state.** On `id` at `permissive`
thresholds, 54 of 288 cell-rows have nothing in them to detect — the deck never leaves limits, so
the base rate is 1.0 and there is no onset. All 54 are SS3. Those render `not scorable` rather
than F1 = 0 (which reads as model failure) or F1 = 1 (which reads as a perfect detector). An
earlier version of this analysis pooled them into a sea-state roll-up and had to be retracted
whole (P6-D19), and the same error was caught again in Phase 8 before publication (P8-D14). F1 is
never pooled across sea states anywhere in this project for that reason.

Lead time — how far ahead of a true onset the model flagged it — is the quantity that decides
whether a detection is actionable at all. At SS5 / beam seas / 12 kn under `strict` limits on the
interval rule, `lstm_gaussian` has a median lead of 2.80 s over 146 matched onsets at a base rate
of 0.0827; the distribution is in `results/quiescence_lead_time_hist.png`. For the best-F1 model in
each row of the table above, the per-cell median lead averaged over that row's scorable cells — a
mean of per-cell medians, not the median of a pooled distribution — is 0.49–3.02 s on the interval
rule and 0.53–7.70 s on the point rule — the point rule buys its longer leads with the precision and false-alarm rates
above.

### The probabilistic floor Phase 5 did not have

Phase 5 shipped six learned interval heads and **no probabilistic baseline at all**, which meant
its coverage column cleared no floor — recorded at the time as that phase's largest gap. Phase 6
built the floor: `residual_interval`, an empirical-residual band fitted on the validation split
around a closed-form point forecast. It has no learned width.

Cells inside the Gate 5 band [0.85, 0.95], floor against the six learned heads:

| regime | `residual_interval` floor | best learned head | all six heads |
|---|---:|---:|---:|
| `id` | **36 of 36** | `dlinear_quantile` 21 of 36 | 102 of 216 |
| `unseen_vessel` | **26 of 36** | `dlinear_quantile` 19 of 36 | 41 of 216 |
| `unseen_heading` | 12 of 36 | `dlinear_quantile` 19 of 36 | 37 of 216 |
| `unseen_seastate` | **0 of 36** | — | 5 of 216 |

**In distribution the trivial baseline is perfectly calibrated and every learned head is not.**
The floor is 36 of 36 on `id`; the best head manages 21. That is the comparison Phase 5 could not
make, and it goes the way that is least flattering to the six models this project spent 66 hours
fitting.

**And it does not settle the question, because the proper score disagrees.** Winkler scores
location and sharpness jointly rather than leaving the trade-off to the reader. Median Winkler on
`id` (lower is better): floor **1.5897**, `lstm_gaussian` **0.1714**, `lstm_quantile` 0.2238,
`tcn_gaussian` 0.2877, `tcn_quantile` 0.2848 — the deep heads beat the floor by roughly 9x — while
`dlinear_gaussian` 2.1353 and `dlinear_quantile` 2.0937 **lose** to it. So the floor wins on
calibration and loses badly on the joint score, because it buys its coverage with width. Reporting
either column alone would support a confident and opposite conclusion; both are printed.

**On `unseen_seastate` the floor fails too — 0 of 36.** That matters for how Phase 5's headline is
read. Coverage collapse under sea-state shift is not a defect of the learned heads specifically;
the residual-interval baseline, which is calibrated by construction in-distribution, collapses to
PICP 0.2202 on `unseen_seastate` heave at 5 s. A band fitted on SS3–SS5 residuals is simply the
wrong width for SS6, however it was obtained.

### Five ablations

Counted as cells whose paired bootstrap interval excludes zero, per regime. The medians are near
zero because most cells sit near 1.0 skill at short lead, so the counts are the readable statistic:

| ablation | `id` (worse–better) | `unseen_seastate` | `unseen_heading` | `unseen_vessel` |
|---|---|---|---|---|
| `attitude_only` (drop the rate channels) | 66 – 10 | 59 – 10 | 18 – 14 | 18 – 15 |
| `lookback_10s` (L = 100 vs 200) | 152 – 37 | 71 – 53 | 82 – 40 | 95 – 10 |
| `lookback_40s` (L = 400 vs 200) | 19 – **188** | **100** – 72 | 16 – **105** | 19 – **95** |
| `revin` | 93 – **0** | 24 – **78** | 14 – 10 | 10 – 6 |
| `ss_conditioned` | 5 – **79** | **91** – 1 | 12 – 15 | 10 – 2 |

- **The rate channels earn their place, modestly.** Dropping them is worse in 66 of 144 `id` cells
  and better in 10. They were added for correctness — the quiescence detector thresholds heave
  rate, so the three-attitude task could not supply its own decision variable — and they also
  turn out to help.
- **More lookback helps in distribution and hurts under sea-state shift.** L = 400 wins 188 of 288
  `id` cells and loses 19; on `unseen_seastate` it loses 100 and wins 72. It is also not
  parameter-matched (+12.6 %), and its OOD intervals bootstrap over 12 clusters.
- **RevIN is the mirror image and the cleanest result in the set.** It is worse or neutral in
  distribution — 93 cells worse on `id` and **not one better** — and better under sea-state shift,
  78 cells better against 24 worse. Removing the per-window level is exactly the thing that should
  help when the amplitude distribution moves and cost you when it does not. It has no closed-form
  vehicle (P6-D13), so it is read from `tcn` alone.
- **Sea-state conditioning is not an upper bound, and it is not deployable either.** Conditioning
  on the true sea state buys a small but resolvable gain in distribution (79 cells better, 5
  worse) and costs a great deal where the sea state is out of distribution (91 cells worse, 1
  better, mean −0.0445). An earlier version of this entry said the in-distribution effect was "not
  measurable"; that compared a paired difference against a marginal interval 21.5x too wide and
  was retracted (P6-D20). The arm consumes privileged information — at deployment the sea state is
  estimated online from the same motion record the forecaster reads — so no row of it is a
  deployable result. But *upper bound* does not survive either: a bound that lies below the
  unconditioned baseline on the regime that matters bounds nothing.
- **The `imu` observability cost is real and small.** The observation model costs skill in 177 of
  288 `id` cells. The published claim that the `imu` persistence denominator is "roughly halved"
  was wrong by about 40x in the ratio and is retracted (P6-D23): the two denominators agree to
  about 1 % (mean ratio 0.9963). The correction *strengthens* the arm — with the denominators
  matched, the measured loss is a genuine observability cost rather than a scale artifact.

### The integrity controls, and where one of them is vacuous

- **Shuffle control** (refit on time-shuffled targets): passes. Its null is the window mean, not
  zero skill — the plan's §5.2 wording is wrong for this task (P3-D8). It asserts only on cells
  whose test signal is **not** on the P1-D2 residual floor; floored cells are computed and
  reported, never asserted on. That exemption was introduced after the control failed at 5.52 %
  against a 2 % tolerance on one arm, and the alternative — raising the tolerance — was rejected
  as relaxing a threshold to make a control pass.
- **Untrained control**: a randomly initialised model is supposed to score worse than persistence.
  It does not, in 76 of 144 rows. This is **reported, not enforced** (`strict=False`), because the
  criterion is wrong for a task where a zero-parameter window mean already beats persistence in
  107 of 144 cells. The failure ships rather than the tolerance being tuned.
- **Pipeline sanity**: persistence through the full dataset pipeline matches persistence computed
  directly on the raw Parquet, per DOF and horizon.
- **Interval controls, and the honest part**: the shuffle tolerance for the interval controls was
  changed **after seeing the data**, 0.02 → 0.10, and is labelled as such (P6-D21). The
  consequence is worse than the change: **210 of 432 shuffle rows (48.6 %) are now unasserted**,
  and on `unseen_seastate` **zero cells are asserted**. The interval shuffle control is
  *vacuous* on that regime. "Passed" there means nothing was judged, and a regime with no asserted
  cell must never be read as a regime that passed.

`docs/protocol.md` §Phase 6 is the full log. Six self-corrections and four retractions landed in
this phase, three of them the same shape: a subset — one seed, one row, one speed — published as
if it were the whole.

## Phase 7 — ONNX export and the CPU-versus-GPU question

**Not duplicated here.** Phase 7's write-up is the README's
[Inference latency](../README.md#inference-latency----onnx-export-and-the-cpu-versus-gpu-question)
section, with the full tables in [`results/latency.md`](../results/latency.md).

That is not laziness, it is P7-D14: a re-review of this phase found five false claims that had
drifted out of agreement with the artifacts they described, one of them published in both
`results/latency.md` and the README. The lesson recorded there is that *in a document generated
from measurements, a sentence a human wrote and a number a function computed will drift apart, and
the sentence is the one that is wrong.* `make gate7` now checks the README's latency section
mechanically — every millisecond figure against the committed tables, every `Nx` multiplier
against a ratio of two measured p50 values — and a hand-maintained third copy here would be the
one copy nothing checks.

## Phase 8 -- cross-validation against an independent hydrodynamic model

Every number above comes from one simulator, and two protocol entries already said that prejudges
the headline: P4-D16 ("the corpus makes a linear forecaster Bayes-optimal by construction") and
`configs/sim/vessels/s175.yaml`, which admits the held-out hull is "a reduced-order stand-in ...
not a strip-theory computation of the S-175's actual RAOs". Phase 8 tests the models against the
computation that file says it is not: the MSS toolbox's ShipX strip-theory motion RAOs for the same
ITTC S-175, under a JONSWAP matched to SS5. Because the corpus already holds the S175 out as
`unseen_vessel`, the same checkpoints, normalisation statistics and task apply to both -- only the
generator changes.

**Both sides are simulations.** MSS is another simulator, not a measurement of a real deck. This
bounds generator-specific overfitting; it says nothing about fidelity to a real ship.

### The result: one model family transfers, and it is not the one the gates favour

Pitch at 10 s, the cell Gates 3-5 are read at, against the committed `unseen_vessel` rows:

| model | corpus | MSS | change |
|---|---|---|---|
| `dlinear` | 0.4144 | **0.3935** | -0.021 |
| `dlinear_ols` | 0.4904 | **0.3835** | -0.107 |
| `damped_persistence` | 0.0734 | -0.0585 | -0.132 |
| `window_mean` | 0.0733 | -0.0586 | -0.132 |
| `ar20` | 0.5258 | 0.1560 | -0.370 |
| `ar10` | 0.5195 | -0.0004 | -0.520 |
| `transformer` | 0.7654 | -0.1773 | -0.943 |
| `ar40` | 0.5412 | **-0.7243** | -1.266 |
| `tcn` | 0.8604 | **-0.9033** | -1.764 |
| `lstm` | 0.8032 | **-1.5564** | -2.360 |
| `persistence` | 0.0000 | 0.0000 | 0.000 |

Every model scored on both generators is listed; `ar_attitude_only` was never run on the MSS
records and so has no transfer number. The DLinear family loses 0.02-0.11. `ar20` is the only other
model left with positive skill (0.156), `ar10` lands on persistence (-0.0004), and the four
heaviest models go below it. In the 1-5 s
operational band the split is the same: at 5 s in **heave**, the channel this project exists for,
`dlinear_ols` holds 0.923 while `tcn` is at 0.011, `transformer` at -0.486 and `ar40` at -4.275.

**The dividing line is not linear versus deep.** `ar40` is linear and per-channel, it *beats*
`dlinear_ols` on the corpus, and it transfers worse than `transformer`. P3-D1 explains it, two
phases early: this corpus has no process noise, so its motion satisfies an exact linear recursion
and AR *identifies the system*. A system identifier transfers to that system and nothing else, and
identifying harder is worse -- `ar40` loses 1.27 where `ar20` loses 0.37. The deep models do the
same implicitly; DLinear is too constrained to do it at all. **The ladder does not extend downward**:
`ar10` identifies less than `ar20` and transfers worse (-0.520 against -0.370), so the AR family
peaks at order 20 and the claim holds for the 20 -> 40 step rather than for the family.

**This lands on the gate cell.** Gates 3, 4 and 5 are all read at pitch / 10 s. There the deep
models beat `dlinear_ols` by 0.37 on the corpus and lose to it by 1.29 on MSS. Read together with
the 1-5 s table above, the deep models' advantage is concentrated exactly where it does not
transfer.

### What the controls rule out

The wave-field discretisation is not the cause: re-running with the corpus's own wave grid and MSS's
RAOs moves skill by -0.04 to -0.19 for every model, same direction. Normalisation range is not the
cause either: rescaling MSS channels to corpus RMS leaves `dlinear_ols` exactly unchanged (skill is
scale-invariant) and moves everything else in both directions, by -13.4 to +2.3 depending on DOF.
What remains is the hull response itself.

One concrete defect was found in our simulator while building the bridge, before any model was run:
`src/dmf/sim/response.py` applies a **real** wave-slope excitation, so roll and pitch come out in
phase with heave where strip theory puts them in quadrature (measured 0.1 deg against 87.8 deg).
Amplitudes are correct and every Gate 1 invariant still holds, because a common phase rotation
within one channel does not change its spectrum. It is unfixed -- fixing it invalidates the corpus
and Phases 2-7 -- and it is not sufficient on its own to explain the AR result.

### The operational metric could not be compared at all

Quiescence detection is threshold-based on absolute limits (3.0 deg / 2.0 deg / 0.8 m/s). Our
generator runs ~2x hot, so those same limits classify far more of the MSS record as landable: at
permissive thresholds in head seas the MSS deck **never leaves limits** -- base rate 1.0000, zero
onsets, nothing to detect -- against a corpus base rate of 0.655 in the same cell. Two of six
permissive cells are unscorable outright.

This is the sharpest result in the phase and it is a methodological one. Skill is a ratio and was
completely unaffected by the amplitude gap; the operational metric, the one this project exists to
serve, was dominated by it. Where the metric *is* scorable the ordering matches the accuracy
finding -- `dlinear`/`dlinear_ols`/`ar40` on top, the deep models below -- but chance-timing
(`rate_matched`) beats `transformer` at four of six strict cells, so those rows are not detecting
anything measurable.

Full write-up including the statistical comparison of the two generators, the Octave parity check
(the NumPy bridge reproduces MSS's own `waveMotionRAO.m` to 4.4e-12), and the corrections made after
the Gate 8 review: `docs/mss_crossvalidation.md`, protocol entries P8-D1 to P8-D15.


---

## Where to go next

- [`docs/protocol.md`](protocol.md) — the complete decision log, ~5200 lines, P1-D1 onward. Every
  retraction quoted above is there in full, with what was measured and what changed as a result.
- [`docs/mss_crossvalidation.md`](mss_crossvalidation.md) — Phase 8 in full, including the Octave
  parity check and the statistical comparison of the two generators.
- [`docs/corpus_card.md`](corpus_card.md) — what the corpus contains and how to regenerate it.
- [`results/results.md`](../results/results.md) — the machine-generated Phase 6 report. **6 MB**;
  it is the source of the tables above, not a document to read front to back.
