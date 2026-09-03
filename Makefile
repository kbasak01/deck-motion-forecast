# deck-motion-forecast -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
CFG ?= configs/experiment/e01_baselines.yaml
SIMCFG ?= configs/sim/corpus.yaml
WORKERS ?= 8
# Each gate reads its own phase's artifact directory; RESULTS overrides both.
RESULTS ?=
GATE4_DIR ?= $(or $(RESULTS),results/e02)
GATE5_DIR ?= $(or $(RESULTS),results/e03)

.PHONY: data train eval gate4 gate5 bench test lint format all

data:   ; python scripts/generate_corpus.py --config $(SIMCFG) --out artifacts/corpus --workers $(WORKERS)
train:  ; python scripts/train.py --config $(CFG)
eval:   ; python scripts/evaluate.py --all-regimes
gate4:  ; python scripts/gate4.py --results-dir $(GATE4_DIR) --out-dir $(GATE4_DIR)
gate5:  ; python scripts/gate5.py --results-dir $(GATE5_DIR) --out-dir $(GATE5_DIR)
bench:  ; python scripts/benchmark.py --export --parity --latency
test:   ; pytest
lint:   ; ruff check src tests && ruff format --check src tests && mypy src
format: ; ruff format src tests && ruff check --fix src tests

all: data train eval bench
