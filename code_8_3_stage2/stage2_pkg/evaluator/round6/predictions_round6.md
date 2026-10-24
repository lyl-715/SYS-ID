# Predictions — round 6 (budget-margin threshold tables: budget retention under mismatch, grid check, equal-send comparison)

Committed before any run of this round (before any code of this round is written, before the Part 0 tests, before any
evaluator call, forward pass or λ scan). The report `report_round6_CN.md` cites this file's commit hash and marks every
prediction right / wrong / partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round6_CN.md` v3.2 (commit a973aa6,
sha256 6d93e162d52914b6e88fc1d21c0d3927ca47f08ffeec33e6d7a014fc4eb9565c): §5 = spec lines 53–63, extracted with
`sed -n '53,63p'`, not retyped.

---

# 5. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| M1 | （预算利用率检查，不作科学发现）三张新表名义点 $B_m-\mathbb E[N]\le0.5$ |
| M2 | $m^\star=0.10$：$m=0.05$ 在 26 点中至少 2 点越线（P4 $\varepsilon=0.5$ 或 P1 $+0.5$）；$m=0.10$ 26 点全守；$m=0.20$ 26 点全守，且 P4 $\varepsilon=1$ 的 10 个方向和 P1 $\pm1$ 也守 |
| M3 | 代价：$\Delta m_D(m;\theta_0)\ge\Delta m_D(0;\theta_0)-6m$，即 $m=0.05,0.10,0.20$ 分别 $\ge7.5304,\ 7.2304,\ 6.6304$（基准 $7.8304$） |
| M4 | 四张表在 26 点上 $\rho_{26}\ge0.8$（信息保持）；P4 中位数曲线 $g_{\rm med}(\varepsilon)$（10 个方向的中位数，$\varepsilon\in\{0.2,0.5,1,2\}$）对每个 $m$ 单调不降 |
| M5 | 网格复核：选中表三个点的 $|\Delta\mathbb E[N]|\le0.5$ 次，预算判定不翻转（名义点离预算线 $\approx mB$，远） |
| M6 | 等发送数：21 点全部匹配成功（残差 $\le1$）；20 个 P4 点上 $\Delta m_D^{\rm eq}$ 的中位数 $\ge+6.0$；$\Delta m_D-\Delta m_D^{\rm eq}$ 在 P4 点的中位数 $\le1.5$ nat |
| M7 | 总耗时 $\le2.5$ h |
