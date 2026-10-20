# Predictions — round 2a (recheck of the seven candidates, analytic S bound, discrete MDP estimate)

Committed before any run of this round. The report cites this file's commit hash and marks every prediction right / wrong /
partly right, wrong ones included; B2 is reported as "本轮未检验" (not scored).

Both blocks are copied verbatim from the frozen spec `reports_CN/plan_recheck_bounds_CN.md` v4.1 (commit aab99e6):
§A5 = spec lines 32–41 (R1–R6), §B6 = spec lines 144–153 (B1–B6), extracted with `sed -n`, not retyped.

---

## A5. 预测

| 编号 | 预测 |
|---|---|
| R1 | 七个候选的 $\Delta m_D$ 都在 MC 值的 2 se 内 |
| R2 | $\tau_N$ 在 0.005–0.02 之间 |
| R3 | P5514 的可行性为"边界未定"（它由搜索推到预算边界上） |
| R4 | 阶段 3 以 0.07%–0.09% 余量合格的两个候选也为"边界未定" |
| R5 | 无预算候选 U-L1657 不可行，$\mathbb E[N]\approx423$ |
| R6 | 七个候选的 $S$ 都在 $[460,\,500]$，U-L1657 除外（更大） |

## B6. 预测

| 编号 | 预测 |
|---|---|
| B1 | 候选系数 $(a,b)=(2.76253,0.094651)$ 通过三段全域验证；$\overline S=1344.357299<1345$ |
| B2 | （本轮未检验：不搜索新系数，此条不计对错）在网格上搜不到使 $1+315a+4999b$ 低于 1344.36 超过 5 的新系数 |
| B3 | 离散 MDP 对偶估算值 $\widetilde S_{\rm dual}\in[700,1100]$；对应策略前向的 $\mathbb E[N]$ 与 316 之差 $<5\%$ |
| B4 | 一致性检查通过（前向 = 后向到 $10^{-10}$）；与评价器均匀输入的诊断差值 $<3\%$ |
| B5 | 最优策略形状：发送后把 $\delta$ 推到边缘内侧 $1$–$2\sigma$ 处停住（$v=0$），$\lambda$ 取到 $[0.5,3]$ |
| B6 | 绝对严格上界 $U^{\rm upper}(1344.36)$ 比 $U(466)=198.42$ 高约 $9\log(1344/466)\approx9.5$ nat（$U$ 随 $S$ 近似按 $9\log S$ 增长） |
