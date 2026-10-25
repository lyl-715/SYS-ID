本轮预测在已看到第 8 轮补充分析的情况下提交，不是盲预测；判定仍按冻结规则。

# Predictions — round 8b (confirmation run: Newton-decrement convergence acceptance, fresh seeds)

Spec: `reports_CN/plan_round8b_CN.md` v2 (frozen; committed in a9ea099 together with the round-8 Part 0 corrections,
sha256 083ca455cfb2d0f3fc7825721576811dacf2f692c5a8f87ce0ca536920ab4e40), the amendment to round 8 v3.1
(`reports_CN/plan_round8_CN.md`, sha256 edd4a2db47c342ecc60859a23ab917ca5a04bb23014c0e065ed53ec07858a6bf).

This file, the engineering-test seed list below and the scoring readings (`decisions_round8b.md`, section S) are committed
before any numerical run of this round, the engineering tests included. Before this commit nothing was computed: the code,
logs and report of round 8 were read, the Part 0 text corrections were written (numbers as given by the spec, not re-run),
and numba 0.68.0 was installed (missing in the container; the version of rounds 6-8). The report `report_round8b_CN.md`
cites this file's commit hash and marks every prediction right / wrong / partly right, wrong ones included.

## Predictions

Q1-Q6 are copied verbatim from the frozen round-8 spec `reports_CN/plan_round8_CN.md` v3.1 §4 (lines 77-84, extracted
with `sed -n '77,84p'`, not retyped). Q7 is replaced as the round-8b spec states ("Q7 改为…"); "两方案" in Q5/Q6 are the
sequential scheme and the SoD+SoD control, as in round 8.

| 编号 | 预测 |
|---|---|
| Q1 | $\varepsilon_1$ 中位 $\in[0.2,0.4]$，90% 分位 $\le0.6$，超过 0.5 的比例 $\le15\%$；第一段估计失败率 $\le2\%$ |
| Q2 | $|{-\log\det\widehat M_1}-\log\det J_1|\le0.5$ nat（诊断）；Hessian–FIM 白化残差 $\le0.05$ |
| Q3 | $\Pr\{\mathbb E[N_2\mid T,\theta_0]>B\}\le10\%$；$\Pr\{N_2>B\}\le25\%$；总期望发送 $\le2B$ |
| Q4 | 全体 $R$ 次：$\Delta D\in[+2.5,+4]$ nat，bootstrap 95% 区间下端 $\ge+1.5$ |
| Q5 | 两方案 $|{-\log\det\widehat M}-\log\det J|\le0.5$ nat（诊断） |
| Q6 | 设计失败率 $\le2\%$、估计失败率 $\le2\%$；两方案各参数 $|$偏差$|/{\rm RMSE}\le0.2$ |
| Q7 | 每次重复主工作中位 $\le110$ s，总 $\le3$ h（单跑主分析） |

Scoring: on the primary analysis (the only analysis of this round), sub-claims as in `decisions_round8b.md` S-0 to S-7,
committed with this file.

## Seeds

**Official seeds of this round** (never used for debugging; used only by the official pipeline after every A3 check has
passed, and by its Part C):

| use | seed |
|---|---|
| stage-1 noise e^(1), replication r = 1..300 | 50261028 + r |
| stage-2 noise e^(2) (shared by both schemes) | 50271028 + r |
| stage-1 restart (estimator, delta = 0.01 s * d) | 60261028 + r |
| joint restart (both schemes) | 70261028 + r |
| paired bootstrap (B_boot = 2000) | 50281028 |

**Engineering-test seeds (round-8 seeds only).** Every numerical run before the official pipeline uses only the seeds below;
all of them were already used in round 8 (its pipeline, its Part A3 checks or its probe smoke test):

| engineering test | seeds |
|---|---|
| E1. Part A3 re-checks under the new acceptance rule (`checks_r8b.py`): the synthetic records, random windows and random theta points of round 8's `checks_r8.py`, unchanged | 92000000 + {0..99, 500, 1000, 2000, 2001, 2002, 2100, 2101, 2102, 2200, 2201, 2202} |
| E1. the archived §8.4 ML record (one constant-SoD record) | 20261014 (row 0) |
| E1. restart seed of the estimator calls inside the A3 checks (r = 0, as in round 8's A3) | 30261028 |
| E2. reproduction of round 8's supplementary analysis S by the round-8b pipeline, bit for bit against `round8/results/archive_round8.jsonl` (analysis S), replications r in {1, 2, 98, 134} (98 and 134 exercise the joint-estimation fallback) | e^(1) 20261028 + r, e^(2) 20271028 + r, restarts 30261028 + r / 40261028 + r |
| E3. Part C code on round 8's archive (analysis S), reproducing round 8's S statistics, and on the E2 output | bootstrap 20281028 |
| E4. reserve: a further pipeline smoke test, only if needed (round 8's probe seeds) | 93000000 + r / 94000000 + r, restarts 30261028 + r / 40261028 + r, r = 10001..10004 |
