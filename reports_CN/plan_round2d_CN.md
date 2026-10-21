---
title: "第 2d 轮计划书：对 A 做外层最小化——闭环、加速度、真实放置下的 log det 估算"
date: "2026-10-22"
---

**要回答的问题。** 2c 轮固定 $A$ 的直接路线（+4.39 nat）没有比 $S$ 路线（+3.78）紧，但分不清是"保留方向没用"还是"被固定 $A$ 的损失抵消"。本轮把 $A$ 也优化掉。理论上（§1.3）$\min_{A,\lambda}\mathcal D$ 等于"闭环 + 加速度约束 + 由动态决定的放置"下 $\log\det$ 的最大值，它必然 $\le$ $S$ 路线的 $U(\widetilde S_{\rm acc})$（后者放置自由）；两者之差就是"动态限制放置"值多少。全部产出仍是离散模型上的估算。

# 1. 理论

## 1.1 目标函数（白化坐标）

记 $M_0=M(w(\hat u))$（P5514 的权重，2c 轮的 $A_1=M_0^{-1}$），$g_k:=\sigma^{-1}M_0^{-1/2}h_{U,k}$。参数化 $A=M_0^{-1/2}\tilde AM_0^{-1/2}$，$\tilde A\succ0$。则
$$
\gamma_k(\tilde A)=g_k^{\top}\tilde Ag_k,\qquad -\log\det A=-\log\det\tilde A+\log\det M_0 ,
$$
$$
\mathcal D(\tilde A,\lambda)=\log\det M_0-\log\det\tilde A-9+\gamma_1(\tilde A)+\lambda(B-1)+\widetilde V_{\tilde A,\lambda},\qquad
\widetilde V_{\tilde A,\lambda}=\max_{\text{策略}}\sum_{k\ge2}\mathbb E\big[\gamma_k(\tilde A)c(\delta_k)-\lambda q(\delta_k)\big].
$$
$\tilde A=I$ 就是 2c 轮的 $A_1$（此时 $\sum_kw_k(\hat u)\gamma_k=9$）。白化只为数值条件好，不改问题。

## 1.2 凸性与次梯度

$\gamma_k$ 对 $\tilde A$ 线性；$\widetilde V$ 是一族对 $(\tilde A,\lambda)$ 线性的函数（每条策略一个）的最大值，凸；$-\log\det\tilde A$ 凸；其余线性。所以 $\mathcal D$ 对 $(\tilde A,\lambda)$ 联合凸（$\tilde A\succ0$，$\lambda\ge0$）。

在 $(\tilde A_j,\lambda_j)$ 处取到 $\widetilde V$ 最大的策略 $\pi_j$，前向传播得它的权重 $w^{(j)}_k=\mathbb E_{\pi_j}[c(\delta_k)]$（$w^{(j)}_1=1$）和发送数 $n_j=\sum_{k\ge2}\mathbb E_{\pi_j}[q(\delta_k)]$。则
$$
\partial_{\tilde A}\mathcal D\ni-\tilde A_j^{-1}+\widetilde M_j,\quad\widetilde M_j:=\sum_{k\ge1}w^{(j)}_kg_kg_k^{\top};\qquad
\partial_\lambda\mathcal D\ni(B-1)-n_j .
$$
（$\widetilde V$ 的次梯度是最大化策略下线性函数的梯度，这是凸分析的标准结论；$\gamma_1$ 的梯度已并入 $\widetilde M_j$ 的 $k=1$ 项。）

## 1.3 最小值是什么

对固定的权重 $w$，$\min_{\tilde A}[\mathrm{tr}(\tilde A\widetilde M(w))-\log\det\tilde A]=\log\det\widetilde M(w)+9$，在 $\tilde A=\widetilde M(w)^{-1}$ 取到。闭环（含随机化）可达的 $(w,n)$ 集合是凸的（策略混合），$\mathcal D$ 对 $\tilde A$ 凸、对 $w$ 线性，由 Sion 极小极大定理
$$
\min_{\tilde A\succ0}\max_{w\text{ 可达}}\big[\cdots\big]=\max_{w\text{ 可达}}\big[\log\det M(w)-\lambda(n(w)-B+1)\big],
$$
再对 $\lambda\ge0$ 取最小，$\log\det M$ 对 $w$ 凹、可达集凸，强对偶成立：
$$
\boxed{\ \min_{\tilde A,\lambda}\mathcal D=\max\{\log\det M(w):\ w\text{ 闭环（加速度约束）可达},\ n(w)\le B-1\}\ }
$$
即"闭环 + 加速度 + 真实动态放置"下的 $\log\det$ 最大值（连续模型的陈述；本轮算的是离散估算）。它 $\le U(\widetilde S_{\rm acc})$，因为后者在 $S\le\widetilde S_{\rm acc}$ 下允许任意放置。差值 $=$ 动态对放置的限制。

# 2. 算法：带信赖域的切平面法

内层是现成的二维 DP（每次约 25 s，4 线程）加一次前向（约 10 s）。外层 46 维（$\tilde A$ 对称 45 + $\lambda$），用 Kelley 切平面 + 信赖域，主问题含精确的 $-\log\det\tilde A$（它给主问题强凸性，比纯线性切平面稳）。

1. 初始点 $(\tilde A_0,\lambda_0)=(I,\ \lambda^\star_{2c})$；信赖域半径 $r=0.5$（Frobenius）。
2. 迭代 $j$：在 $(\tilde A_j,\lambda_j)$ 跑 DP 得 $\widetilde V_j$、策略 $\pi_j$；前向得 $w^{(j)}$、$n_j$；记 $\mathcal D_j$ 和次梯度；更新 $\mathcal D_{\rm best}$。
3. 主问题（cvxpy，变量 $\tilde A\succeq0$、$\lambda\in[0,0.5]$、$t$）：
$$
\min\ \log\det M_0-\log\det\tilde A-9+g_1^{\top}\tilde Ag_1+\lambda(B-1)+t
$$
$$
\text{s.t.}\ t\ \ge\ \widetilde V_i+\mathrm{tr}\big((\tilde A-\tilde A_i)\widetilde M_i^{(2+)}\big)-(\lambda-\lambda_i)n_i\quad\forall i\le j,\qquad\|\tilde A-\tilde A_{\rm c}\|_F\le r,
$$
$\widetilde M_i^{(2+)}=\sum_{k\ge2}w^{(i)}_kg_kg_k^{\top}$（$\gamma_1$ 已在目标里精确出现，切平面只模拟 $\widetilde V$）；$\tilde A_{\rm c}$ 是当前中心（上一次成功步的点）。解得 $(\tilde A_{j+1},\lambda_{j+1})$ 和模型值 $m_{j+1}$。
4. 信赖域更新：预测下降 $\mathcal D_{\rm c}-m_{j+1}$，实际下降 $\mathcal D_{\rm c}-\mathcal D_{j+1}$；比值 $\ge0.25$ 为成功步（中心移到新点，$r\leftarrow1.5r$，上限 2），否则失败（中心不动，$r\leftarrow r/2$，下限 $10^{-3}$）。切平面无论成败都加入。
5. 停止：$\mathcal D_{\rm best}-m_{\rm global}\le0.02$ nat（$m_{\rm global}$ 是去掉信赖域约束的主问题最小值，是 $\min\mathcal D$ 的下界）；或 300 次迭代。两者都报。

每次迭代约 35 s，300 次约 3 h。

# 3. 检查（必须过，否则不报估算值）

1. **回归**：$(\tilde A,\lambda)=(I,\lambda^\star_{2c})$ 处 $\mathcal D$ 逐位复现 2c 轮的 2-D $A_1$ 值（$-197.197$ 后为 4.385…，用存档全精度）。
2. **次梯度有效性**（支撑超平面）：在 3 个随机方向 $\Delta$（对称、$\|\Delta\|_F=1$）和 $\epsilon\in\{10^{-2},10^{-3}\}$ 上验证 $\mathcal D(\tilde A+\epsilon\Delta,\lambda)\ge\mathcal D(\tilde A,\lambda)+\epsilon\,\mathrm{tr}(G\Delta)-10^{-9}$，$G$ 为所报次梯度。这是凸函数次梯度的定义，比有限差分可靠（DP 是分段线性，有限差分会撞折点）。
3. **切平面有效性**：每次迭代后，所有已评价点上 $\mathcal D_i\ge$ 当时模型在该点的值（按构造应成立），报最大违反量。
4. **下界性质**：$m_{\rm global}$ 单调不减（切平面只增）；报序列。

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

# 5. 怎么读

把四个数放一行：真实已评估 +0.70；本轮 $\min\mathcal D$；$S$ 路线 +3.78；2c 固定 $A$ +4.39。

- M3 成立：动态确实限制放置，直接路线有用；剩余（$\min\mathcal D-0.70$）是"闭环能反应、开环不能"加离散化，这是现有工具能到的最紧估算。
- $\min\mathcal D$ 接近 3.78：放置限制不值钱，闭环的优势全在反应本身。

两种情况都不是界；下一轮若要严格化，势函数证书的目标值就是 $\min\mathcal D$。

# 6. 产出

`round2d/`：`logdet_outer.py`（主问题、信赖域、日志）、`checks.py`、每次迭代一行日志（$\mathcal D_j$、$m_j$、gap、$r$、成败）、最优 $\tilde A^\star$、$\lambda^\star$、对应策略的 $w^\star$ 与 $n^\star$、`report_round2d_CN.md`：四项检查；收敛曲线；四个数的对照行；预测对照。禁用词：上界、全局。
