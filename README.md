# deck-motion-forecast

Short-horizon (1-15 s) forecasting of ship deck motion — roll, pitch, heave and their rates —
from JONSWAP-driven vessel simulation, for deciding when to commit to a rotorcraft touchdown on a
heaving deck. The decision the forecast serves is a *quiescent window*: an interval in which all
six channels stay inside landing limits for long enough to get the aircraft down, which is what a
full-scale manned or unmanned helicopter needs, not attitudes alone.

**All results in this repository are from simulated vessel motion. No real deck data is used,
and no sim-to-real claim is made.**

Status: **Phase 3 complete. Gate 3 as written failed; the task was revised and the criterion restated at the binding DOF.**

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

`results/baselines.csv` (`ideal`) and `results/imu/baselines.csv` (`imu`) now carry the full sweep:
1296 rows each, nine baselines x four regimes x six DOFs x six horizons, three seeds for the one
SGD-fitted model and deterministic single rows for the eight closed-form ones. Regenerate with
`make train`. No deep model is trained yet — that is Phase 4.

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

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint && make test
```
