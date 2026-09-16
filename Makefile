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
CONFORMAL_DIR ?= $(or $(RESULTS),results/e05)
# Gate 6 is read over the assembled report, which lives at the top of results/.
GATE6_DIR ?= $(or $(RESULTS),results)
# Gate 7 reads the deploy artifacts, which also live at the top of results/.
GATE7_DIR ?= $(or $(RESULTS),results)

.PHONY: data train eval rescore gate4 gate5 gate6 gate6-full gate7 bench report \
        figures figures-extract test lint format all sweeps mss conformal gate10

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
gate7:  ; $(PY) scripts/gate7.py --results-dir $(GATE7_DIR) --out-dir $(GATE7_DIR)

# Gate 6 predicate 1 is "`make eval` exits 0", which no artifact can testify to. This target
# is the evidence: make stops on a non-zero exit, so the gate step runs only if eval passed,
# and --eval-exit-code 0 records that rather than assuming it. `make gate6` alone reports
# predicate 1 as UNVERIFIED, which is not a pass.
gate6-full: eval
	$(PY) scripts/gate6.py --results-dir $(GATE6_DIR) --out-dir $(GATE6_DIR) --eval-exit-code 0

# The whole phase, in the order the methodology requires: export, parity on every provider,
# the sweep twice, the PyTorch-CUDA rows, the thread sweep, then render and gate. `make
# bench` alone cannot produce the committed artifacts -- it has one --torch-device and no
# thread axis -- so the script is the reproducing command, not this line.
bench:  ; ./scripts/run_phase7.sh $(GATE7_DIR)
# Re-render results/latency.md and the Pareto figure from the committed CSVs. No GPU, no
# checkpoints, no corpus: it cannot start a measurement, which is what makes "is the
# document a function of the CSVs?" answerable in a second.
report: ; $(PY) scripts/benchmark.py --report
test:   ; $(PYTEST)
lint:   ; $(RUFF) check src tests && $(RUFF) format --check src tests && $(MYPY) src
format: ; $(RUFF) format src tests && $(RUFF) check --fix src tests

# Re-render the two Phase 9 figures from committed artifacts alone: the 12 kB
# results/headline_trace.npz and the e04 quiescence CSVs. No corpus, no checkpoints, no GPU,
# which is what makes "is the headline figure a function of committed artifacts?" answerable
# in a second -- the same property `report` gives the Pareto figure.
figures: ; $(PY) scripts/make_figures.py --render-only
# Re-extract the trace from the corpus and the committed checkpoints, then render. This is
# the one `all` runs, because a fresh reproduction must not re-render a committed trace it
# did not itself produce.
figures-extract: ; $(PY) scripts/make_figures.py

# Every training run the committed tables are read from, in dependency order. Separated from
# `all` so the ~155 h of fitting can be started on its own and resumed per stage.
#
# The e04 lookback arms MUST be followed by `rescore` before `eval`: until it has run,
# ablations.csv carries no lookback rows and Gate 6 predicate 5 cannot pass (P6-D15/D16).
sweeps:
	$(PY) scripts/train.py --config configs/experiment/e01_baselines.yaml --results-dir results
	$(PY) scripts/train.py --config configs/experiment/e01_baselines_imu.yaml --results-dir results/imu
	$(PY) scripts/train.py --config configs/experiment/e02_deep.yaml --results-dir results/e02
	$(PY) scripts/train.py --config configs/experiment/e03_probabilistic.yaml --results-dir results/e03
	./scripts/run_e04.sh cheap
	./scripts/run_e04.sh deep
	./scripts/post_sweep.sh

# The whole project, in the order the methodology requires. `make all` is what
# docs/IMPLEMENTATION_PLAN.md 5.5 ("fresh clone + make all reproduces every committed
# number") is read against, so it runs the sweeps and every gate rather than one config and
# one gate -- the previous recipe was `data train eval bench report gate7`, which trained
# only e01 and left gates 4, 5, 6 and 8 unrun. Expected wall clock is about 160 h on one
# RTX A4000; the per-stage breakdown is in the README.
all: data sweeps gate4 gate5 gate6-full conformal gate10 bench report gate7 mss figures-extract

# Phase 10: split-conformal calibration. Retrains nothing -- it loads the committed Phase 5
# checkpoints, fits one conformal scale per (horizon, channel) on each regime's VALIDATION
# split, and re-scores on test. Writes only under results/e05/; results/e03/ is read and never
# regenerated, which is how Gate 5's "report the degradation, do not fix it" survives the arm.
conformal:
	$(PY) scripts/conformal.py --all-regimes --out-root $(CONFORMAL_DIR)

gate10: ; $(PY) scripts/gate10.py --results-dir $(CONFORMAL_DIR) --out-dir $(CONFORMAL_DIR)

# Phase 8: MSS cross-validation. Needs the corpus and the deep checkpoints; the Octave
# parity step skips cleanly when Octave is absent. Every step is idempotent.
mss:
	$(PY) scripts/mss_export.py
	$(PY) scripts/mss_octave_check.py
	$(PY) scripts/mss_compare.py
	$(PY) scripts/mss_evaluate.py --grid-kind mss --sign-ablation
	$(PY) scripts/mss_evaluate.py --grid-kind corpus
	$(PY) scripts/mss_evaluate.py --grid-kind mss --rescale-to-corpus
	$(PY) scripts/mss_quiescence.py
	$(PY) scripts/mss_figures.py
	$(PY) scripts/gate8.py
