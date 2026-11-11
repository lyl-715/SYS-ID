#!/bin/bash
# Round 17 Part B in the background; results and log committed and pushed when it finishes.
cd "$(dirname "$0")"; mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 R13_INPUT=U
python3 run_round17.py partB > logs/partB.stdout 2>&1; rc=$?
line=$(grep -E "DONE|STOP|done in" logs/partB.log | tail -1 | cut -c1-200)
git add results logs/partB.log; [ $rc -ne 0 ] && git add -f logs/partB.stdout
git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "round 17 Part B (exit $rc) - $line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014RLNiMxPH94ph7Ux49wDea"
for i in 1 2 3 4; do git push -q origin round17-absolute-diagnostics && break; sleep $((2**i)); done
echo "=== $(date '+%F %T') Part B done (exit $rc): $(git log --oneline -1 | cut -c1-150)"
