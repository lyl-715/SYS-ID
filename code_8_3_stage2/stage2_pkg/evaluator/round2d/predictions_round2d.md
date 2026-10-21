# Predictions — round 2d (outer minimisation over A of the direct log-det dual)

Committed before any run of this round (before any DP, any oracle call, any master-problem solve and before the code of
this round is written). The report `report_round2d_CN.md` cites this file's commit hash and marks every prediction
right / wrong / partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round2d_CN.md` (commit ac058c4,
sha256 782040fbec2d7d86062ec4c2aa6a921a1876fc41a9a0a1bd309e4914c796c4b0): §4 = spec lines 71–81, extracted with `sed -n '71,81p'`, not retyped.

---

# 4. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| M1 | 检查 1–4 全过 |
| M2 | $\min\mathcal D-197.197\in[2.5,\ 3.5]$ nat |
| M3 | 比 $S$ 路线的 3.776 低至少 0.3 nat |
| M4 | 终止时 gap $\le0.02$，迭代数 $\le200$ |
| M5 | 最优 $\tilde A$ 不是单位阵：$\|\tilde A^\star-I\|_F>0.2$；最优 $\lambda^\star$ 与 2c 的 $\lambda^\star_{2c}$ 同量级（比值在 $[0.5,2]$） |
| M6 | 最优点处策略前向的 $n$ 落在 $B-1\pm5$ 内（强对偶下互补松弛） |
| M7 | 总耗时 $\le4$ h |
