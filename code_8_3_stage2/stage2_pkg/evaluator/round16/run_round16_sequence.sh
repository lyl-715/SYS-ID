#!/bin/bash
# Round 16: run the stages one after another; commit and push each stage's log and result when it finishes.
cd "$(dirname "$0")"
mkdir -p logs results
for st in "$@"; do
  python3 run_round16.py "$st" > "logs/${st}.stdout" 2>&1; code=$?
  last=$(grep "=== ROUND 16 stage" "logs/${st}.log" | tail -1)
  git add "logs/${st}.log" "results/${st}.json" 2>/dev/null
  git commit -q -m "round 16: stage ${st} (exit ${code}) - ${last}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014RLNiMuryYPXc3Npj631ed" -- "logs/${st}.log" "results/${st}.json"
  for i in 2 4 8 16; do git push -q -u origin round16-random-inputs && break; sleep $i; done
  if [ $code -ne 0 ]; then echo "stage ${st} failed with exit ${code}"; exit $code; fi
done
