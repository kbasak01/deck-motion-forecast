# deck-motion-forecast -- see docs/IMPLEMENTATION_PLAN.md for what each phase produces.
CFG ?= configs/experiment/e01_baselines.yaml
SIMCFG ?= configs/sim/corpus.yaml
WORKERS ?= 8
RESULTS ?= results/e02

.PHONY: data train eval gate4 bench test lint format all

data:   ; python scripts/generate_corpus.py --config $(SIMCFG) --out artifacts/corpus --workers $(WORKERS)
train:  ; python scripts/train.py --config $(CFG)
eval:   ; python scripts/evaluate.py --all-regimes
gate4:  ; python scripts/gate4.py --results-dir $(RESULTS) --out-dir $(RESULTS)
bench:  ; python scripts/benchmark.py --export --parity --latency
test:   ; pytest
lint:   ; ruff check src tests && ruff format --check src tests && mypy src
format: ; ruff format src tests && ruff check --fix src tests

all: data train eval bench
