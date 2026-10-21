# Predictions — round 2c (direct log-det dual prototype, two fixed A matrices)

Committed before any run of this round (before the construction of Part A, before any DP, any evaluator call and any
route-1 solve). The report `report_round2c_CN.md` cites this file's commit hash and marks every prediction right / wrong /
partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round2c_CN.md` (commit a716952,
sha256 4dccc4494dc211b49fc9463535a12e84b2b8439546bc49e6a882ba1f8e867278): §4 = spec lines 77–87, extracted with `sed -n '77,87p'`, not retyped.

---

# 4. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| L1 | 检查 1–4 全过 |
| L2 | $A_1$、二维加速度模型：$\min_\lambda\mathcal D(A_1,\lambda)-\log\det J_C\in[1.0,\ 3.0]$ nat |
| L3 | $S$ 路线对应值 $U(619.129)-\log\det J_C\in[3.5,\ 4.0]$ nat；直接路线比它低至少 0.5 nat |
| L4 | $A_2$ 的结果比 $A_1$ 高 0.2–1.0 nat（对均匀权重取等号，对最优候选附近松） |
| L5 | 一维无约束模型下，$A_1$ 的结果比二维高 2–4 nat |
| L6 | $\lambda^\star/\bar\gamma\in[0.3,\ 3]$（与 2b 轮 $\lambda^\star\approx0.94$ 同量级） |
| L7 | 二维外层总耗时 $\le0.5$ h |
