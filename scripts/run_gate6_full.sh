#!/usr/bin/env bash
# Gate 6 predicate 1 demands `results/results.md` be regenerated END-TO-END by `make eval`.
# The committed document was rendered with --render-only and its predicate-1 PASS came from an
# operator-supplied --eval-exit-code, which is an assertion rather than an artifact. This runs
# the real thing and tees the log, so the predicate has evidence behind it.
set -u
cd "$(dirname "$0")/.."
S=artifacts/logs/e04/status_gate6_full.txt
echo "start $(date -Is)" >> "$S"
PYTHONUNBUFFERED=1 make gate6-full > artifacts/logs/e04/gate6_full_endtoend.log 2>&1
rc=$?
echo "DONE gate6-full exit=$rc $(date -Is)" >> "$S"
echo "ALLDONE $(date -Is)" >> "$S"
exit $rc
