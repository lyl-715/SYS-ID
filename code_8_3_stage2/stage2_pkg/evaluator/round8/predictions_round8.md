# Predictions — round 8 (full two-stage sequential verification with unknown true parameters)

Committed before any run of this round's code: before the Part A tests (estimator checks, design-chain regression, isolation
checks), before any DP, evaluator call, simulation or estimation of the pipeline. The report `report_round8_CN.md` cites this
file's commit hash and marks every prediction right / wrong / partly right, wrong ones included.

Disclosure (also DECISION P-1 in `decisions_round8.md`): before this commit a scratch prototype, not committed, ran scipy
L-BFGS-B on eight stage-1 SoD records generated with seeds 20261029–20261036 (these equal the stage-1 noise seeds
20261028 + r of r = 1..8) to check whether the frozen gradient threshold of spec §1.2 is attainable in double precision. It
looked only at the optimiser's termination (message, gradient infinity norm, iteration counts) and at the difference between
two arithmetic forms of the same objective; no ε_1, estimation error, transmission count or design quantity was computed.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round8_CN.md` v3.1 (commit f06a994,
sha256 edd4a2db47c342ecc60859a23ab917ca5a04bb23014c0e065ed53ec07858a6bf): §4 = spec lines 75–85, extracted with
`sed -n '75,85p'`, not retyped.

---

# 4. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| Q1 | $\varepsilon_1$ 中位 $\in[0.2,0.4]$，90% 分位 $\le0.6$，超过 0.5 的比例 $\le15\%$；第一段估计失败率 $\le2\%$ |
| Q2 | $|{-\log\det\widehat M_1}-\log\det J_1|\le0.5$ nat（诊断）；Hessian–FIM 白化残差 $\le0.05$ |
| Q3 | $\Pr\{\mathbb E[N_2\mid T,\theta_0]>B\}\le10\%$；$\Pr\{N_2>B\}\le25\%$；总期望发送 $\le2B$ |
| Q4 | 全体 $R$ 次：$\Delta D\in[+2.5,+4]$ nat，bootstrap 95% 区间下端 $\ge+1.5$ |
| Q5 | 两方案 $|{-\log\det\widehat M}-\log\det J|\le0.5$ nat（诊断） |
| Q6 | 设计失败率 $\le2\%$、估计失败率 $\le2\%$；两方案各参数 $|$偏差$|/{\rm RMSE}\le0.2$ |
| Q7 | 每次重复 $\le60$ s，总 $\le6$ h（含 bootstrap） |
