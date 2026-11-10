#!/bin/bash
# Round 14: stages in sequence, one process at a time: part1 (U), budgets (U), input SCH, P5514, LO, HI, N. After each stage its results and
# log are committed and pushed (branch round14-one-parameter-rule). A failed part1 (reproduction) stops the sequence (prompt §2 1b); a failed
# later stage is committed and the sequence goes on.
cd "$(dirname "$0")"; mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=${@:-"part1:U budgets:U input:SCH input:P5514 input:LO input:HI input:N"}
for st in $STEPS; do
  stg=${st%%:*}; n=${st##*:}
  echo "=== $(date '+%F %T') stage $stg input $n"
  R13_INPUT=$n python3 run_round14.py $stg > "logs/${stg}_${n}.stdout" 2>&1; rc=$?
  line=$(grep -E "DONE|STOP|done in" "logs/${stg}_${n}.log" | tail -1 | cut -c1-300)
  git add results logs/${stg}_${n}.log 2>/dev/null
  [ $rc -ne 0 ] && git add -f "logs/${stg}_${n}.stdout"
  for i in 1 2 3 4; do git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "round 14: stage $stg input $n (exit $rc) - $line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014RLNiMxPH94ph7Ux49wDea" && break; sleep $((2**i)); done
  for i in 1 2 3 4; do git push -q origin round14-one-parameter-rule && break; sleep $((2**i)); done
  echo "=== $(date '+%F %T') committed $stg $n (exit $rc): $(git log --oneline -1 | cut -c1-200)"
  if [ $rc -ne 0 ] && [ "$stg" = "part1" ]; then echo "=== part1 failed: sequence stopped"; break; fi
done
echo "=== $(date '+%F %T') sequence done"
