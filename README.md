# deck-motion-forecast

Short-horizon (1-15 s) forecasting of ship deck motion — roll, pitch, heave and their rates —
from JONSWAP-driven vessel simulation, for deciding when to commit to a rotorcraft touchdown on a
heaving deck. The decision the forecast serves is a *quiescent window*: an interval in which all
six channels stay inside landing limits for long enough to get the aircraft down, which is what a
full-scale manned or unmanned helicopter needs, not attitudes alone.

**All results in this repository are from simulated vessel motion. No real deck data is used,
and no sim-to-real claim is made.**

Status: **Phase 4 complete. Gate 4 passes. The deep models win at 10-15 s lead and in-distribution; a 60 300-parameter closed-form linear model is at least their equal across the 1-5 s operational band and beats all three of them on two of the three out-of-distribution regimes.**

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

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint && make test
```
