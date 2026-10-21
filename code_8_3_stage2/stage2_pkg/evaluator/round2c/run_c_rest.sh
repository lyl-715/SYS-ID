#!/bin/sh
# Round 2c Part C after the 2-D A_1 loop: 2-D A_2 (4 threads; cached lambdas replay from logs/gold_2d_A2.jsonl), then the
# two 1-D loops concurrently (1 thread each), then C2 and the C3 table. Usage (from this directory): setsid nohup sh run_c_rest.sh &
cd "$(dirname "$0")"
python3 logdet_dual.py outer --A 2 --model 2d --threads 4 >> logs/c1_2d_A2.stdout 2>&1 || exit 1
python3 logdet_dual.py outer --A 1 --model 1d --threads 1 > logs/c1_1d_A1.stdout 2>&1 &
p1=$!
python3 logdet_dual.py outer --A 2 --model 1d --threads 1 > logs/c1_1d_A2.stdout 2>&1 &
p2=$!
wait $p1 || exit 1
wait $p2 || exit 1
python3 logdet_dual.py sroute > logs/c2.stdout 2>&1 || exit 1
python3 logdet_dual.py table > logs/c3.stdout 2>&1 || exit 1
echo ALLDONE > logs/c_rest.stdout
