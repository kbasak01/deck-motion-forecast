# Gate 4 read-out

Simulated results only. Gate 4: **all deep models beat the reference by a margin exceeding the seed-to-seed standard deviation**. Two readings of the cell, both computed here; the gate is read at Reading B (`docs/protocol.md` P4-D1) and Reading A is reported whether or not it passes.

Every number here traces to the CSVs it was read from: `results/e02/baselines.csv` and `results/e02/paired_contrasts.csv`.

## Outcome

- **Reading A -- PASS**: 18 of 18 rows pass.
- **Reading B -- PASS**: 3 of 3 rows pass.

## Reading A -- original criterion (docs/IMPLEMENTATION_PLAN.md Phase 4): 3 s vs damped_persistence

Regime `id`, horizon 3 s (30 samples). `margin = skill_mean - reference_skill_mean`; PASS iff `margin > skill_std`.

**Reading A -- PASS**: 18 of 18 rows pass.

| model | dof | n_seeds | skill_mean | nrmse_mean | skill_std | reference | reference_skill_mean | margin | verdict | paired_skill_diff | paired_ci_lo | paired_ci_hi | paired_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | heave | 3 | 0.9987 | 0.0547 | 0.0001 | damped_persistence | 0.4663 | 0.5324 | PASS | 0.5324 | 0.5171 | 0.5472 | excludes 0 (model better) |
| transformer | heave | 3 | 0.9972 | 0.0799 | 0.0005 | damped_persistence | 0.4663 | 0.5309 | PASS | 0.5309 | 0.5153 | 0.5460 | excludes 0 (model better) |
| lstm | heave | 3 | 0.9987 | 0.0545 | 0.0001 | damped_persistence | 0.4663 | 0.5324 | PASS | 0.5324 | 0.5171 | 0.5472 | excludes 0 (model better) |
| tcn | heave_rate | 3 | 0.9980 | 0.0710 | 0.0001 | damped_persistence | 0.5292 | 0.4689 | PASS | 0.4689 | 0.4571 | 0.4805 | excludes 0 (model better) |
| transformer | heave_rate | 3 | 0.9966 | 0.0936 | 0.0005 | damped_persistence | 0.5292 | 0.4674 | PASS | 0.4674 | 0.4552 | 0.4793 | excludes 0 (model better) |
| lstm | heave_rate | 3 | 0.9983 | 0.0660 | 0.0004 | damped_persistence | 0.5292 | 0.4691 | PASS | 0.4691 | 0.4570 | 0.4810 | excludes 0 (model better) |
| tcn | pitch | 3 | 0.9973 | 0.0884 | 0.0001 | damped_persistence | 0.5925 | 0.4049 | PASS | 0.4049 | 0.3935 | 0.4164 | excludes 0 (model better) |
| transformer | pitch | 3 | 0.9957 | 0.1115 | 0.0003 | damped_persistence | 0.5925 | 0.4033 | PASS | 0.4033 | 0.3918 | 0.4153 | excludes 0 (model better) |
| lstm | pitch | 3 | 0.9981 | 0.0744 | 0.0003 | damped_persistence | 0.5925 | 0.4056 | PASS | 0.4056 | 0.3941 | 0.4174 | excludes 0 (model better) |
| tcn | pitch_rate | 3 | 0.9960 | 0.1136 | 0.0001 | damped_persistence | 0.6473 | 0.3487 | PASS | 0.3487 | 0.3423 | 0.3555 | excludes 0 (model better) |
| transformer | pitch_rate | 3 | 0.9943 | 0.1360 | 0.0005 | damped_persistence | 0.6473 | 0.3470 | PASS | 0.3470 | 0.3401 | 0.3543 | excludes 0 (model better) |
| lstm | pitch_rate | 3 | 0.9976 | 0.0873 | 0.0002 | damped_persistence | 0.6473 | 0.3503 | PASS | 0.3503 | 0.3439 | 0.3571 | excludes 0 (model better) |
| tcn | roll | 3 | 0.9991 | 0.0439 | 0.0001 | damped_persistence | 0.3850 | 0.6140 | PASS | 0.6140 | 0.6063 | 0.6211 | excludes 0 (model better) |
| transformer | roll | 3 | 0.9979 | 0.0654 | 0.0004 | damped_persistence | 0.3850 | 0.6129 | PASS | 0.6129 | 0.6050 | 0.6204 | excludes 0 (model better) |
| lstm | roll | 3 | 0.9990 | 0.0452 | 0.0001 | damped_persistence | 0.3850 | 0.6140 | PASS | 0.6140 | 0.6063 | 0.6211 | excludes 0 (model better) |
| tcn | roll_rate | 3 | 0.9987 | 0.0528 | 0.0001 | damped_persistence | 0.4095 | 0.5892 | PASS | 0.5892 | 0.5805 | 0.5972 | excludes 0 (model better) |
| transformer | roll_rate | 3 | 0.9976 | 0.0707 | 0.0006 | damped_persistence | 0.4095 | 0.5881 | PASS | 0.5881 | 0.5790 | 0.5966 | excludes 0 (model better) |
| lstm | roll_rate | 3 | 0.9989 | 0.0483 | 0.0000 | damped_persistence | 0.4095 | 0.5894 | PASS | 0.5894 | 0.5808 | 0.5974 | excludes 0 (model better) |

## Reading B -- restated criterion (docs/protocol.md P4-D1): 10 s on pitch vs the stronger of damped_persistence and window_mean, resolved per cell

Regime `id`, horizon 10 s (100 samples). `margin = skill_mean - reference_skill_mean`; PASS iff `margin > skill_std`.

**Reading B -- PASS**: 3 of 3 rows pass.

| model | dof | n_seeds | skill_mean | nrmse_mean | skill_std | reference | reference_skill_mean | margin | verdict | paired_skill_diff | paired_ci_lo | paired_ci_hi | paired_verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | pitch | 3 | 0.8346 | 0.5109 | 0.0006 | damped_persistence | 0.3657 | 0.4690 | PASS | 0.4690 | 0.4389 | 0.4991 | excludes 0 (model better) |
| transformer | pitch | 3 | 0.8080 | 0.5504 | 0.0032 | damped_persistence | 0.3657 | 0.4423 | PASS | 0.4423 | 0.4120 | 0.4742 | excludes 0 (model better) |
| lstm | pitch | 3 | 0.8680 | 0.4564 | 0.0016 | damped_persistence | 0.3657 | 0.5023 | PASS | 0.5023 | 0.4706 | 0.5354 | excludes 0 (model better) |

Reference resolved per cell as the strongest of `damped_persistence`, `window_mean`; both candidates' skill in the cell is shown so the resolution can be checked, since which one wins varies across the grid (P3-D20).

| dof | reference | reference_pool |
|---|---|---|
| pitch | damped_persistence | damped_persistence=0.3657, window_mean=0.3656 |

## Notes

- **The criterion is strict.** `verdict` is PASS iff `margin > skill_std`; `margin == skill_std` is FAIL, because the gate says the margin must *exceed* the seed-to-seed standard deviation.
- **`skill_std` covers initialisation and data-order variance (P4-D13).** `_fit_one` builds the loaders once per model config, but the shuffle generator's state advances every epoch and is never reset, so each seed trains on a different batch order and the three-seed spread is the full one the criterion intends. Note what this costs in exchange: seed k's batch order depends on how many epochs seeds 0..k-1 ran, so the seeds are consecutive segments of one stream rather than independent draws, and a change to the epoch cap changes the data order of every seed after the first.
- **The spread is not the same quantity in every row (P4-D6).** Every SGD row's `skill_std` covers initialisation and data-order variance; `lstm` and `transformer` additionally carry cuDNN/Flash-Attention backward nondeterminism that `tcn` and `dlinear` do not. Since the criterion compares each model's margin against its own spread, the noisier model faces the harder bar.
- **`margin` and `paired_skill_diff` answer different questions and are not collapsed into one verdict.** The margin test asks whether the advantage exceeds the spread of the fitting procedure across seeds; the paired bootstrap asks whether it survives resampling held-out realizations with both models scored on the same resample. A margin can exceed the seed std while the paired interval still spans zero, and the reverse.
- **`paired_ci_lo`/`paired_ci_hi` are the envelope over seeds, not a mean of intervals.** The contrasts file emits one interval per seed and no aggregated view (P4-D4); the envelope contains every seed's interval, so `excludes 0` here means every seed excluded zero. It is not a calibrated interval for the seed mean.
- **`nrmse_mean` is beside `skill_mean` because skill is not comparable across horizons on this signal (P3-D5).** Persistence error tracks the autocorrelation and is non-monotone in lead time, so a skill-vs-horizon reading shows dips that belong to the reference. `nrmse = rmse / signal_std` of the scored partition at that exact lead time; 1.0 is 'no better than the partition mean' and lower is better.
- **Simulated results only.** Every artifact read here comes from the JONSWAP-driven vessel simulation; no real deck data is involved.
