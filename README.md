# deck-motion-forecast

Short-horizon (1-15 s) forecasting of ship deck motion — roll, pitch, heave and their rates —
from JONSWAP-driven vessel simulation, for deciding when to commit to a rotorcraft touchdown on a
heaving deck. The decision the forecast serves is a *quiescent window*: an interval in which all
six channels stay inside landing limits for long enough to get the aircraft down, which is what a
full-scale manned or unmanned helicopter needs, not attitudes alone.

**All results in this repository are from simulated vessel motion. No real deck data is used,
and no sim-to-real claim is made.**

Status: **Phase 3 — Gate 3 read, task revised, gate restated and passing.**

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

The full 7-model x 4-regime baseline sweep has not been run, so `results/baselines.csv` does not
yet exist. No deep model is trained yet — that is Phase 4. See `docs/IMPLEMENTATION_PLAN.md` for
the build plan and `CLAUDE.md` for the working rules. This README is replaced by the full write-up
in Phase 9.

A caveat that travels with every number here: the generator has no process noise, so the
achievable-skill ceiling is unrealistically high and absolute values flatter every model. Only
relative comparisons and out-of-distribution degradation should be read as findings.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint && make test
```
