# Predictions — round 5 (model-mismatch sensitivity of the frozen threshold tables; fixed-design two-stage comparison)

Committed before any run of this round (before any code of this round is written, before the Part 0 tests, before any
evaluator call, forward pass or λ scan). The report `report_round5_CN.md` cites this file's commit hash and marks every
prediction right / wrong / partly right, wrong ones included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round5_CN.md` v3.2 (commit c669511,
sha256 cc938d1344b1bfa46cbe7970baea74dd1b0442733deae906157a0913e763a8ae): §3 = spec lines 69–78, extracted with
`sed -n '69,78p'`, not retyped.

---

# 3. 预测（跑前提交）

| 编号 | 预测 |
|---|---|
| R1 | U/加权表：十个方向增益的中位数曲线 $g_{\rm med}(\varepsilon)=\mathrm{median}_j\,\Delta m_D(\theta_j(\varepsilon))$ 在 $\varepsilon=0.5$ 时低于名义值的一半；$\varepsilon=2$ 时 $\le0$ |
| R2 | P1 的一个符号在 $\varepsilon\le1$ 时 $\mathbb E[N]>B$，另一符号 $\mathbb E[N]<200$ |
| R3 | P4 的 $\varepsilon=0.2$：10 个方向中至少 8 个增益保持名义值的 $\ge70\%$ 且 $\mathbb E[N]\le B$ |
| R4 | 名义点：B 比 A 高 $\ge3$；D 比 B 高 $\le1$（B 比 A′ 高 4.852 是已知基准，只作复算检查，不计命中）。P4 $\varepsilon=0.2$ 的 10 个方向上，B 相对 A′ 的增益中位数 $\ge3$，且总发送数 $\le2B$ 的方向 $\ge8$ 个 |
| R5 | 同一 $\varepsilon$ 下 P5514 表的增益保持率低于 U 表 |
| R6 | 全部（约 400 次扰动前向 + 241 次方案 A 前向 + 1 次 $\lambda$ 扫描）$\le1.5$ h |
