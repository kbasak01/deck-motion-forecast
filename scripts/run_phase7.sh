#!/usr/bin/env bash
# Reproduce every Phase 7 artifact, end to end, in the order the methodology requires.
#
# This script exists because no single committed command produced the committed artifacts:
# `make bench` cannot emit the PyTorch-on-CUDA rows (its `--torch-device` defaults to cpu)
# and cannot emit the thread sweep, so `results/latency.csv` was assembled from several
# hand-typed invocations that nothing recorded. A benchmark nobody can re-run is not
# reproducible, whatever its environment stamp says.
#
# Order is load-bearing:
#   1. export + parity, on EVERY execution provider that will be benchmarked. TF32 is off
#      throughout (docs/protocol.md P7-D9); with it on, all three GPU providers fail the
#      parity bar the CPU provider passes.
#   2. the main sweep, PyTorch on the CPU, both repeats.
#   3. the same sweep's PyTorch rows on CUDA, appended -- a separate invocation only
#      because `--torch-device` is one flag for both PyTorch backends.
#   4. the ORT CPU thread sensitivity sweep, written under its own stem.
#   5. render the document and the figure, then read the gate.
#
# A throwaway pass runs first: the machine's clocks and page cache should be warm before
# the first timed iteration, and it is cheaper to say so here than to explain a cold first
# configuration later.
#
# Usage:  scripts/run_phase7.sh [RESULTS_DIR]
# Takes roughly two hours on the reference machine. Every step is idempotent.
set -euo pipefail

RESULTS_DIR="${1:-results}"
PY="${PY:-.venv/bin/python}"
BENCH="${PY} scripts/benchmark.py"
WARMUP="${WARMUP:-200}"
TIMED="${TIMED:-2000}"

echo "== 0/5 throwaway pass: warm clocks and page cache (not recorded) =="
${BENCH} --latency --batch-size 1 32 --warmup 20 --timed 50 \
  --results-dir "$(mktemp -d)" --latency-stem throwaway >/dev/null

echo "== 1/5 export and parity, every provider, TF32 off =="
${BENCH} --export --parity --results-dir "${RESULTS_DIR}"

echo "== 2/5 main sweep, PyTorch on CPU, two repeats =="
${BENCH} --latency --batch-size 1 32 --threads 1 \
  --warmup "${WARMUP}" --timed "${TIMED}" --repeats 2 \
  --torch-device cpu --results-dir "${RESULTS_DIR}"

echo "== 3/5 PyTorch backends on CUDA, two repeats, appended =="
${BENCH} --latency --backend torch-eager torch-compile --torch-device cuda \
  --batch-size 1 32 --threads 1 \
  --warmup "${WARMUP}" --timed "${TIMED}" --repeats 2 --append \
  --results-dir "${RESULTS_DIR}"

echo "== 4/5 ORT CPU thread sensitivity, batch 1 =="
${BENCH} --latency --backend ort-cpu --batch-size 1 --threads 1 2 4 8 18 \
  --warmup "${WARMUP}" --timed "${TIMED}" \
  --results-dir "${RESULTS_DIR}" --latency-stem latency_threads

echo "== 5/5 render and read the gate =="
${BENCH} --report --results-dir "${RESULTS_DIR}"
${PY} scripts/gate7.py --results-dir "${RESULTS_DIR}" --out-dir "${RESULTS_DIR}"
