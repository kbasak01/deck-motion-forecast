---
name: new-model
description: Scaffold a new forecasting model that conforms to the project's shared interface, config pattern, and registry, so it can be compared fairly against the existing models.
argument-hint: <model-name> [one-line description]
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

Scaffold a new forecasting model named `$1`. Delegate the implementation to the `forecast-modeler`
agent.

Create exactly these, and nothing else:

1. `src/dmf/models/$1.py` — implements the `ForecastModel` protocol from `src/dmf/models/base.py`:
   `forward(x: Tensor[B, L, C_in]) -> Tensor[B, H, C_out]`, or `[B, H, C_out, Q]` if it has a
   quantile head. Direct multi-horizon output; no autoregressive rollout.
2. `configs/model/$1.yaml` — every hyperparameter, no magic numbers in the model file.
3. A registry entry in `src/dmf/train/registry.py`.
4. `tests/test_models.py::test_$1_shapes` — asserts output shape for both point and quantile modes,
   and asserts the effective context or receptive field covers the full lookback.

Do not modify any existing model file. Do not modify the data pipeline, the normalization, or the
evaluation code — if the new model needs any of those changed, stop and explain why instead.

Then train it for 3 seeds on `configs/experiment/e02_deep.yaml` and report skill score vs persistence
at each horizon on the `id` regime, as mean ± std, alongside the existing models' numbers and its
parameter count.
