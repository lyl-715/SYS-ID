# Predictions — round 2b (acceleration-constrained 2-D closed-loop estimate + in-family max-S search)

Committed before any run of this round (before the constants of A0, before any DP, before any evaluator call). The report
`report_round2b_CN.md` cites this file's commit hash and marks every prediction right / wrong / partly right, wrong ones
included.

The block below is copied verbatim from the frozen spec `reports_CN/plan_round2b_CN.md` v3.2 (commit 20b14bb,
sha256 73a5c2e8c87a0b37565ea4a40101231dacc88787171fedba8afd2856ef0b2988): §6 = spec lines 114–125, extracted with `sed -n`,
not retyped.

---

# 6. 预测（跑前提交）

| 编号 | 预测 | 判定方式 |
|---|---|---|
| T1 | 同模型回归（3.1）相对差 $\le10^{-10}$；跨网格诊断（3.2）差异 $<1\%$ | 日志 |
| T2 | 带加速度约束的对偶估算 $\widetilde S_{\rm acc}\in[600,800]$ | 对偶值 |
| T3 | 两个从前向分布 $\mu_k(\delta,v_{\rm prev})$ 直接可算的时间平均量（$k=2..K$ 平均）：$\overline{\Pr}\{18\le\lvert\delta\rvert/\sigma\le19\}<40\%$；低速占比 $\frac1{K-1}\sum_{k=2}^{K}\Pr\{\lvert v_{k-1}\rvert\le0.5\sigma\}<20\%$（统计的是状态分量 $v_{\rm prev}$，即实际增量 $v_1,\dots,v_{K-1}$，不含终步无作用的 $v_K$；$\lvert v\rvert\le0.5\sigma$ 不等同于 $v=0$ 停驻）。一维无约束模型按同一时间平均口径重算作对照（2a 轮的 75% 是特定时刻的占据分布，不能直接当基线） | 前向日志 |
| T4 | $S_{\rm found}\in[500,560]$；其功率 $p_1>0.4$（$\omega_1=3\sqrt2$ 为最低频）；其 $\Delta m_D$ 低于 P5514 的 0.701 | 存档 |
| T5 | $\widetilde S_{\rm acc}-S_{\rm found}\in[100,250]$ | 两数之差 |
| T6 | 单个 $\lambda$ 的二维 DP $\le20$ min；每个模型外层约 18 次评价，二维 3–6 h | 耗时日志 |

加速度约束不禁止 $v=0$ 停驻，只限制到达它的速度；T3"不再停住"是预测，不是理论必然。
