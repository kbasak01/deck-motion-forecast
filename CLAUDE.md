# CLAUDE.md — deck-motion-forecast

## What this project is

Short-horizon (1–5 s) forecasting of 6-DOF ship deck motion (roll, pitch, heave) from JONSWAP-driven
vessel simulation, for timing a quadrotor touchdown on a heaving deck. Deliverables: a reproducible
simulator, six forecasting models, an operational evaluation, split-conformal calibrated prediction
intervals (exact in distribution; characterised where the calibration stops working under shift),
and an ONNX latency study.

Full spec: `docs/IMPLEMENTATION_PLAN.md`. Read the phase you are working in before writing code.

## Non-negotiables

1. **Simulation only.** No real deck data. Every result statement in code comments, docstrings, and
   the README must be compatible with "these are simulated results." Never write a claim that implies
   real-world validation.
2. **Realization-level splits.** Train/test splits are by simulation seed, never by time window.
   Any code that shuffles windows before splitting is a bug. If you are about to write
   `train_test_split(windows, ...)`, stop.
3. **Normalization statistics come from the training split only.** Never compute mean/std over the
   full corpus.
4. **Persistence is the reference.** Every accuracy result is reported as skill score vs. persistence
   alongside raw RMSE. A result without its persistence baseline is not a result.
5. **Three seeds minimum.** These models train in minutes. Any model-vs-model comparison reports
   mean ± std over ≥ 3 seeds. Single-seed comparisons are noise.
6. **Report what you find.** If a linear model beats the Transformer, or the CPU beats the GPU, that
   goes in the README body. Never drop an underperforming model from the results table.

## Architecture rules

- Everything importable and testable lives in `src/dmf/`. `scripts/` holds argparse wrappers only.
- All forecasting models implement the `ForecastModel` protocol in `src/dmf/models/base.py`:
  `forward(x: Tensor[B, L, C_in]) -> Tensor[B, H, C_out]` for point models,
  `-> Tensor[B, H, C_out, Q]` for quantile models. Direct multi-horizon output, never
  autoregressive rollout.
- Configuration is YAML in `configs/`, loaded into dataclasses. No magic numbers in model code.
- New model = new file in `src/dmf/models/` + config in `configs/model/` + registry entry.
  Nothing else changes.
- Simulation code (`src/dmf/sim/`) is pure NumPy and has no torch dependency.

## Commands

```
make data      # generate the corpus (parallel, ~15 min)
make train     # train one config: make train CFG=configs/experiment/e02_deep.yaml
make eval      # regenerate every table in results/
make bench     # ONNX export, parity check, latency benchmark
make test      # pytest
make lint      # ruff + mypy
make all       # data -> train -> eval -> bench
```

## Gates

Each phase ends with a gate defined in the implementation plan. Do not start phase N+1 until
gate N passes. When a gate fails, fix it — do not work around it and do not relax the threshold
without saying so explicitly and recording it in `docs/protocol.md`.

**Read `docs/protocol.md` for the phases already completed before starting a new one.** It is not
only an audit trail: earlier phases measure things that invalidate later ones, and
`IMPLEMENTATION_PLAN.md` is the *original* plan and is not rewritten as they do. Where the two
disagree, the protocol is what actually happened. Completed phases leave a "Before you start" block
at the head of the next phase in the plan; if you are starting a phase that has one, read it.

## Specialists

Delegate to these agents rather than doing their work in the main thread:

| Agent | Owns |
|---|---|
| `seakeeping-sim` | `src/dmf/sim/`, physics invariants, Gate 1 |
| `forecast-modeler` | `src/dmf/models/`, `src/dmf/train/`, Gates 3–5 |
| `eval-auditor` | `src/dmf/eval/`, split integrity, Gate 6 — **read-only over `src/dmf/models/`** |
| `deploy-benchmarker` | `src/dmf/deploy/`, Gate 7 |
| `results-skeptic` | Adversarial review; finds leakage, overclaims, missing baselines |

## Known traps in this project

- **Wave synthesis periodicity.** Uniform frequency grids make the wave record repeat with period
  `2*pi/dw`, which the forecaster memorizes. Frequencies must be jittered within their bins, and
  `tests/test_spectra.py` must assert no autocorrelation spike at the synthesis period.
- **TCN receptive field.** Must be ≥ `lookback`. Assert the arithmetic in a test, not a comment.
- **Un-synchronized CUDA timing.** Always `torch.cuda.synchronize()` around timed regions. Numbers
  that look impossibly fast are wrong, not impressive.
- **Following-seas encounter frequency.** `w_e = w - w^2*U*cos(beta)/g` is non-monotonic for
  following seas. Either exclude that regime from the corpus or handle it explicitly.
- **Quantile crossing.** Sort quantile outputs post-hoc.
- **Base rates in quiescence detection.** Always report the base rate next to the F1.

## Style

- Type hints everywhere; `mypy` clean.
- Docstrings state units (radians vs degrees, m vs m/s) — mixing these has already been the most
  common bug class in this domain.
- Angles are stored in **degrees** in Parquet and converted to radians only inside physics code.
- No notebook-driven development; notebooks may read from `results/` but never define logic.
