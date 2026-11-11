#!/bin/bash
# Round 18: stages in sequence (one process at a time); results and logs committed and pushed after each stage.
cd "$(dirname "$0")"; mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 R13_INPUT=U
for stg in ${@:-A B C}; do
  echo "=== $(date '+%F %T') stage $stg"
  python3 run_round18.py $stg > "logs/${stg}.stdout" 2>&1; rc=$?
  line=$(grep -E "PASS|FAIL|STOP|done in" "logs/${stg}.log" | tail -1 | cut -c1-200)
  git add results logs/${stg}.log 2>/dev/null; [ $rc -ne 0 ] && git add -f "logs/${stg}.stdout"
  for i in 1 2 3 4; do git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "round 18: stage $stg (exit $rc) - $line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014RLNiM3cHJbMMhSG8TA2UB" && break; sleep $((2**i)); done
  for i in 1 2 3 4; do git push -q origin round18-accumulated-information && break; sleep $((2**i)); done
  echo "=== $(date '+%F %T') committed $stg (exit $rc): $(git log --oneline -1 | cut -c1-160)"
  [ $rc -ne 0 ] && { echo "=== stage $stg failed, sequence stopped"; exit $rc; }
done
echo "=== $(date '+%F %T') sequence done"
