---
name: forecast-protocol
description: The evaluation protocol for this project - split policy, metric definitions, skill score, quiescent-window detection, and probabilistic scoring. Use whenever defining or computing a metric, building a train/test split, writing a results table, or reviewing whether a reported improvement is real. Use proactively before any number is written into results/ or the README.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Forecasting evaluation protocol

## Task

Input `(B, L=200, C_in)` at `fs=10 Hz` (20 s lookback). Output `(B, H, C_out)` for
`H in {10, 20, 30, 50}` (1, 2, 3, 5 s). Targets: roll, pitch, heave. **Direct multi-horizon**, never
autoregressive rollout — direct avoids error compounding and gives one fixed-shape ONNX graph.

## Split policy

Splits are **by realization seed**, never by time window. Overlapping windows from one realization
share information; splitting between them inflates every metric.

| Regime | Train | Test |
|---|---|---|
| `id` | seeds 0-31 of every cell | seeds 32-39 of every cell |
| `unseen_seastate` | SS3, SS4, SS5 | SS6 |
| `unseen_heading` | 180, 135, 45 deg | 90 deg (beam) |
| `unseen_vessel` | frigate | S175 |

All four are reported for every model. Normalization statistics are computed on the training split
only. Per-window de-meaning plus a global per-channel scale from train.

## Metrics

**Point.** Per (model, regime, DOF, horizon): RMSE, MAE, and

```
skill = 1 - MSE_model / MSE_persistence
```

Skill score is the headline; raw RMSE alone hides that persistence is already strong on a narrowband
signal. Also report phase lag at maximum cross-correlation (peak-timing error).

**Quiescent-window detection.** The operational metric. A landing window is a sustained interval
where all thresholds hold:

| Set | abs(roll) | abs(pitch) | abs(heave rate) | Sustained |
|---|---|---|---|---|
| `permissive` | <= 3.0 deg | <= 2.0 deg | <= 0.8 m/s | >= 2.0 s |
| `strict` | <= 1.5 deg | <= 1.0 deg | <= 0.4 m/s | >= 3.0 s |

Ground truth from the true future; prediction from the forecast. Report precision, recall, F1 on
window onsets (+-0.5 s tolerance), the lead-time distribution, false alarms per minute, **and the
base rate of quiescent windows per sea state**. F1 without its base rate is not interpretable.

**Probabilistic.** Pinball loss, CRPS, PICP@90, mean interval width, Winkler score. Report coverage
and sharpness together — a maximally wide interval has perfect coverage. Report coverage degradation
under `unseen_seastate` rather than fixing it; the degradation is the finding.

## Reporting rules

- Every table is mean +- std over >= 3 seeds. Differences smaller than the seed std are not results.
- Parameter counts and wall-clock train time appear in every model comparison table.
- No model is dropped for underperforming. If DLinear beats the Transformer, that goes in the README
  body.
- Every number in the README traces to a committed CSV in `results/`.
- The simulation-only caveat appears in the README's first paragraph.

## Integrity controls

Run these before believing any result:

1. **Shuffle control** - retrain on time-shuffled targets; skill must collapse to ~0.
2. **Untrained control** - a random-init model must score worse than persistence.
3. **Pipeline sanity** - persistence through the dataset pipeline must match persistence computed
   directly on raw arrays.
