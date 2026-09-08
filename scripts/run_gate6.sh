#!/usr/bin/env bash
# Phase 6 Gate 6. `--controls-only` reuses the committed scoring CSVs (step 1a cost 5h37m and
# is deterministic given the checkpoints); controls, assembly and render are redone, then the
# Gate 6 read-out runs with --eval-exit-code so predicate 1 is VERIFIED, not UNVERIFIED.
set -u
cd "$(dirname "$0")/.."
S=artifacts/logs/e04/status_gate6.txt
echo "start $(date -Is)" >> "$S"
PYTHONUNBUFFERED=1 .venv/bin/python scripts/evaluate.py --all-regimes --controls-only \
    > artifacts/logs/e04/gate6_full.log 2>&1
rc=$?
echo "DONE eval exit=$rc $(date -Is)" >> "$S"
if [ $rc -ne 0 ]; then echo "ABORT at eval" >> "$S"; exit $rc; fi
PYTHONUNBUFFERED=1 .venv/bin/python scripts/gate6.py --results-dir results --out-dir results \
    --eval-exit-code 0 --print >> artifacts/logs/e04/gate6_full.log 2>&1
rc=$?
echo "DONE gate6 exit=$rc $(date -Is)" >> "$S"
echo "ALLDONE $(date -Is)" >> "$S"
exit $rc
