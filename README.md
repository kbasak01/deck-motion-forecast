# deck-motion-forecast

Short-horizon (1-5 s) forecasting of 6-DOF ship deck motion (roll, pitch, heave) from
JONSWAP-driven vessel simulation, for timing a quadrotor touchdown on a heaving deck.

**All results in this repository are from simulated vessel motion. No real deck data is used,
and no sim-to-real claim is made.**

Status: **Phase 2 complete (windowing, splits, normalization)**. The simulator is implemented
and Gate 1 passes (`src/dmf/sim/`, `docs/corpus_card.md`); the corpus is generated; the
realization-level split, windowing, and train-only normalization are implemented and Gate 2
passes (`src/dmf/data/`, `docs/protocol.md` §Phase 2). No forecasting model is trained yet —
that is Phase 3. See `docs/IMPLEMENTATION_PLAN.md` for the full build plan and `CLAUDE.md` for
the working rules. This README is replaced by the full write-up in Phase 9.

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make lint && make test
```
