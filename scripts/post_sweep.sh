#!/usr/bin/env bash
# Phase 6 post-sweep sequence. Order matters: each step writes inputs the next one reads.
set -u
cd "$(dirname "$0")/.."
S=artifacts/logs/e04/status_post.txt
echo "start $(date -Is)" >> "$S"

echo "RUN rescore_matched $(date -Is)" >> "$S"
.venv/bin/python -u scripts/rescore_matched.py --all-arms \
    > artifacts/logs/e04/rescore_matched.log 2>&1
rc=$?; echo "DONE rescore_matched exit=$rc $(date -Is)" >> "$S"
[ $rc -ne 0 ] && { echo "ABORT at rescore_matched" >> "$S"; exit $rc; }

echo "RUN e04g_residual_floor $(date -Is)" >> "$S"
mkdir -p results/e04/e04g_residual_floor
.venv/bin/python -u scripts/train.py --config configs/experiment/e04g_residual_floor.yaml \
    --results-dir results/e04/e04g_residual_floor \
    > artifacts/logs/e04/e04g_residual_floor.log 2>&1
rc=$?; echo "DONE e04g_residual_floor exit=$rc $(date -Is)" >> "$S"
[ $rc -ne 0 ] && { echo "ABORT at e04g" >> "$S"; exit $rc; }

echo "ALLDONE post $(date -Is)" >> "$S"
