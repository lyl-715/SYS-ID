#!/bin/bash
# Round 13 Part A: the six inputs in sequence, one process at a time (DECISION 13-17), in the order U, SCH, P5514, LO, HI, N.
# After each input its json, J matrices, packed table and log are committed and pushed (branch round13-input-trigger), so a time
# limit does not lose finished inputs. A stopped input is committed as well (its log and partial json) and the sequence goes on.
# Usage (from this directory): ROUND13_WORK=<dir> bash run_partA_sequence.sh [inputs...]
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INPUTS=${@:-U SCH P5514 LO HI N}
for n in $INPUTS; do
  echo "=== $(date '+%F %T') Part A input $n"
  R13_INPUT=$n python3 run13.py partA > "logs/partA_${n}.stdout" 2>&1; rc=$?
  line=$(grep "PARTA $n DONE" "logs/partA_${n}.log" | tail -1 | cut -c1-400)
  [ -z "$line" ] && line="STOPPED (exit $rc): $(grep -E 'STOP' "logs/partA_${n}.log" | tail -1 | cut -c1-300)"
  echo "$line"
  for f in "results/partA_${n}.json" "results/partA_${n}" "results/table_A_${n}.npz" "logs/partA_${n}.log"; do [ -e "$f" ] && git add "$f"; done
  if [ $rc -ne 0 ]; then git add -f "logs/partA_${n}.stdout"; fi   # *.stdout is git-ignored; kept only for a stopped input (traceback)
  for i in 1 2 3 4; do
    git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "round 13 Part A: input $n - $line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014RLNiMxPH94ph7Ux49wDea" && break; sleep $((2**i)); done
  for i in 1 2 3 4; do git push -q origin round13-input-trigger && break; sleep $((2**i)); done
  echo "=== $(date '+%F %T') committed and pushed $n: $(git log --oneline -1)"
done
echo "=== $(date '+%F %T') sequence done"
