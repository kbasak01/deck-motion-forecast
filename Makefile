# deck-motion-forecast -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
# Interpreter. The project's dependencies live in .venv, and a bare `python` is not on PATH
# on a clean shell -- `make eval` exited 127 for that reason, which would have failed Gate 6
# predicate 1 for an environment cause unrelated to the phase. Overridable: `make PY=python`.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
# Same reasoning for the tooling: `make lint` exited 127 on a clean shell because `ruff` and
# `mypy` are only on PATH with the venv activated. Overridable: `make RUFF=ruff MYPY=mypy`.
RUFF ?= $(shell [ -x .venv/bin/ruff ] && echo .venv/bin/ruff || echo ruff)
MYPY ?= $(shell [ -x .venv/bin/mypy ] && echo .venv/bin/mypy || echo mypy)
PYTEST ?= $(shell [ -x .venv/bin/pytest ] && echo .venv/bin/pytest || echo pytest)

CFG ?= configs/experiment/e01_baselines.yaml
SIMCFG ?= configs/sim/corpus.yaml
WORKERS ?= 8
# Each gate reads its own phase's artifact directory; RESULTS overrides both.
RESULTS ?=
GATE4_DIR ?= $(or $(RESULTS),results/e02)
GATE5_DIR ?= $(or $(RESULTS),results/e03)
# Gate 6 is read over the assembled report, which lives at the top of results/.
GATE6_DIR ?= $(or $(RESULTS),results)

.PHONY: data train eval rescore gate4 gate5 gate6 gate6-full bench test lint format all

data:   ; $(PY) scripts/generate_corpus.py --config $(SIMCFG) --out artifacts/corpus --workers $(WORKERS)
train:  ; $(PY) scripts/train.py --config $(CFG)
eval:   ; $(PY) scripts/evaluate.py --all-regimes
# The lookback arms scored on the origins they share (docs/protocol.md P6-D4 item 1). Run it
# after the sweep and BEFORE `make eval`: until it has, `ablations.csv` carries no lookback
# rows and Gate 6 predicate 5 cannot pass. It re-scores from committed checkpoints and
# retrains nothing.
rescore: ; $(PY) scripts/rescore_matched.py --all-arms
gate4:  ; $(PY) scripts/gate4.py --results-dir $(GATE4_DIR) --out-dir $(GATE4_DIR)
gate5:  ; $(PY) scripts/gate5.py --results-dir $(GATE5_DIR) --out-dir $(GATE5_DIR)
gate6:  ; $(PY) scripts/gate6.py --results-dir $(GATE6_DIR) --out-dir $(GATE6_DIR)

# Gate 6 predicate 1 is "`make eval` exits 0", which no artifact can testify to. This target
# is the evidence: make stops on a non-zero exit, so the gate step runs only if eval passed,
# and --eval-exit-code 0 records that rather than assuming it. `make gate6` alone reports
# predicate 1 as UNVERIFIED, which is not a pass.
gate6-full: eval
	python scripts/gate6.py --results-dir $(GATE6_DIR) --out-dir $(GATE6_DIR) --eval-exit-code 0

bench:  ; $(PY) scripts/benchmark.py --export --parity --latency
test:   ; $(PYTEST)
lint:   ; $(RUFF) check src tests && $(RUFF) format --check src tests && $(MYPY) src
format: ; $(RUFF) format src tests && $(RUFF) check --fix src tests

all: data train eval bench
