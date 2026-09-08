#!/usr/bin/env bash
# Phase 6.3 ablation sweep driver. Each arm writes its own results subdirectory; the frozen
# Gate 3-5 records under results/, results/imu/, results/e02/ and results/e03/ are never touched.
# Usage: scripts/run_e04.sh cheap | deep
set -u
cd "$(dirname "$0")/.."
STAGE="${1:?usage: run_e04.sh cheap|deep}"

CHEAP=(e04a_obs_mode_ood e04b_channels_ood e04c_ss_conditioned_ood e04c_ss_conditioned_ar_id
       e04d_lookback_10s_ood e04e_lookback_40s_ood e04f_revin_ood)
DEEP=(e04a_obs_mode e04b_channels e04c_ss_conditioned e04d_lookback_10s e04e_lookback_40s e04f_revin)

case "$STAGE" in
  cheap) RUNS=("${CHEAP[@]}") ;;
  deep)  RUNS=("${DEEP[@]}") ;;
  # Any other argument list is taken as explicit config stems, so an aborted stage can be
  # resumed from the arm that failed without re-running the arms already certified.
  *) RUNS=("$@") ; STAGE="explicit" ;;
esac

STATUS=artifacts/logs/e04/status_${STAGE}.txt
echo "start $(date -Is) stage=$STAGE n=${#RUNS[@]}" >> "$STATUS"
for name in "${RUNS[@]}"; do
  out="results/e04/${name}"
  mkdir -p "$out"
  echo "RUN  $name $(date -Is)" >> "$STATUS"
  # -u: unbuffered, so the log is readable while it runs. results/e03/sweep.log being 0 bytes
  # is exactly this mistake (docs/protocol.md P5-D20 item 4) and it cost that phase its
  # traceable wall-clock figure.
  .venv/bin/python -u scripts/train.py \
      --config "configs/experiment/${name}.yaml" \
      --results-dir "$out" \
      > "artifacts/logs/e04/${name}.log" 2>&1
  rc=$?
  echo "DONE $name exit=$rc $(date -Is)" >> "$STATUS"
  if [ $rc -ne 0 ]; then
    echo "ABORT: $name failed with exit $rc; stopping stage $STAGE" >> "$STATUS"
    exit $rc
  fi
done
echo "ALLDONE $STAGE $(date -Is)" >> "$STATUS"
