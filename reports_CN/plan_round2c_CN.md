---
title: "第 2c 轮计划书：直接对 log det 做对偶的小原型"
date: "2026-10-22"
---

**要回答的问题。** 现有上界路线先界总权重 $S$，再经"权重自由放置"的 $U(S)$ 折算成 $\log\det$；折算一步丢掉了各时刻的信息方向和动态限制。本轮试另一条路：用矩阵不等式把 $\log\det$ 直接写成逐步相加的量，套用已有的二维（加速度约束）闭环 DP，得到 $\log\det$ 上界的**估算**，和 $S$ 路线的估算比，看有没有改善。只做小原型：固定两个 $A$，不优化 $A$。全部产出是离散模型上的估算，不是界。

# 1. 理论

## 1.1 矩阵不等式

对任意 $9\times9$ 正定矩阵 $X$，$\log\det X\le\mathrm{tr}X-9$（对每个特征值用 $\log x\le x-1$）。取 $X=A^{1/2}MA^{1/2}$，$A\succ0$：
$$
\log\det M\ \le\ \mathrm{tr}(AM)-\log\det A-9,\qquad\text{等号当且仅当 }A=M^{-1}.
$$

## 1.2 把输入依赖分离出去

回归向量 $h_k(u)=T(u)\,h_{U,k}$，$T(u)=\mathrm{blockdiag}\big(1,\ 2a_1R(\chi_1),\dots,2a_4R(\chi_4)\big)$ 与 $k$ 无关，$R$ 为 $2\times2$ 旋转，$h_{U,k}$ 为均匀输入的回归向量。于是
$$
J(u)=T\,M(w(u))\,T^{\top},\qquad M(w):=\sigma^{-2}\sum_kw_k\,h_{U,k}h_{U,k}^{\top},
$$
$$
\log\det J(u)=\log\det M(w(u))+2\sum_m\log(4p_m)\ \le\ \log\det M(w(u))
$$
（$\sum p_m=1$ 时功率项 $\le0$）。这是项目已有的分解（$H(u)=H_UT(u)$）。以后只需界 $\log\det M(w)$，其中 $w=w(u)$ 是输入通过触发动态产生的权重序列。

## 1.3 逐步相加的形式

固定 $A\succ0$，记
$$
\gamma_k:=\sigma^{-2}\,h_{U,k}^{\top}A\,h_{U,k}\ \ge0 ,
$$
则 $\mathrm{tr}(AM(w))=\sum_kw_k\gamma_k=\gamma_1+\sum_{k\ge2}\gamma_k\,\mathbb E[c(\delta_k)]$（首步 $w_1=1$）。所以
$$
\log\det M(w(u))\ \le\ -\log\det A-9+\gamma_1+\sum_{k\ge2}\gamma_k\,\mathbb E[c(\delta_k)] .
$$
右边是逐步相加的，与 $S$ 唯一的区别是每步多乘一个随时间变的系数 $\gamma_k$。

## 1.4 闭环放松与对偶

同 2b 轮：允许增量 $v_k$ 依赖 $(\delta_k,v_{k-1})$，$|v|\le v_{\max}$、$|\Delta v|\le a_{\max}$；真实输入是特例。预算 $\sum_{k\ge2}\mathbb E[q]\le B-1$ 定价 $\lambda\ge0$：
$$
\sum_{k\ge2}\gamma_k\mathbb E[c(\delta_k)]\ \le\ \underbrace{\max_{\text{策略}}\sum_{k\ge2}\mathbb E\big[\gamma_kc(\delta_k)-\lambda q(\delta_k)\big]}_{\widetilde V_{A,\lambda}}+\lambda(B-1).
$$
合起来，对任意 $A\succ0$、$\lambda\ge0$，
$$
\boxed{\ \mathcal D(A,\lambda)=-\log\det A-9+\gamma_1+\lambda(B-1)+\widetilde V_{A,\lambda}\ }
$$
是族内所有可行输入 $\log\det J(u)$ 的上界（连续模型）；本轮算的是其离散估算。对 $\lambda$ 取最小；$A$ 固定。减去 $\log\det J_C=197.197$ 得 $\Delta m_D$ 的上界估算。

DP 的每步得分是 $\gamma_kc(\delta)-\lambda q(\delta)$，随 $k$ 变；状态、转移、动作集与 2b 轮的二维模型完全相同，不需要把矩阵放进状态。

## 1.5 和 $S$ 路线比什么

$S$ 路线在同一离散模型上的对应数是 $U(\widetilde S_{\rm acc};u_{\rm unif})-\log\det J_C$，$\widetilde S_{\rm acc}=619.129$，$U$ 用路 1 求解器算（cap 1）。粗算 $U(619)\approx U(466)+9\log(619/466)=198.42+2.56=200.98$，即约 $+3.8$ nat。直接路线若明显低于它，说明"保留方向"确实收紧。

# 2. 设置

**两个 $A$（都从评价器的权重算，不含任何未验收的量）：**
- $A_1=M(w(\hat u))^{-1}$，$\hat u=$ P5514（预算内已评估最优），$w(\hat u)$ 由评价器 B/$n=24$ 给出。不等式在 $w=w(\hat u)$ 处取等号，所以对 $\hat u$ 附近的输入最紧。
- $A_2=M(w_U)^{-1}$，均匀输入的权重。作对照。

**$\gamma_k$ 的量级**：$\mathrm{tr}(A_1M(w(\hat u)))=9$，所以 $\sum_kw_k\gamma_k=9$，$\gamma_k$ 的量级约 $9/S\approx0.02$。$\lambda$ 的量级随之缩小：取 $\lambda\in[0,\ 20\bar\gamma]$，$\bar\gamma=\frac1K\sum_k\gamma_k$。

**外层精度**：$\mathcal D$ 对 $\lambda$ 的斜率绝对值 $\le4684$（同前），要让估算误差 $\le0.02$ nat，取 $\epsilon_\lambda=4\times10^{-6}$；黄金分割约 25 次评价，每次二维 DP 约 1 min（2b 轮实测）。不设细化规则（精度已直接定到需要的水平）。

**模型**：二维加速度约束 DP（2b 轮的 `mdp2d_estimate.py`，网格 $\sigma/30$，$L=137$，$\Delta j\in\{-1,0,1\}$），得分改为 $\gamma_kc-\lambda q$；一维无约束 DP 同样跑一遍作对照。

# 3. 检查（必须过，否则不报估算值）

1. **迹恒等式**：用评价器算出的 $w(\hat u)$ 和 $A_1$，$\sum_kw_k\gamma_k=\mathrm{tr}(A_1M(w(\hat u)))=9$，相对差 $\le10^{-10}$；$A_2$ 与 $w_U$ 同。
2. **不等式在已知输入上成立**：对七个旧候选和 U，用评价器的 $J(u)$、$w(u)$ 数值验证 $\log\det J(u)\le\log\det M(w(u))\le\mathrm{tr}(A_iM(w(u)))-\log\det A_i-9$（$i=1,2$），并报每个输入两个不等号的松紧（差值）。P5514 在 $A_1$ 下第二个不等号应取等号（差 $\le10^{-8}$）。
3. **回归到 $S$ 路线**：令 $\gamma_k\equiv1$（$A$ 任意，只改得分），二维 DP 在 2b 轮存档的 $\lambda^\star=0.9381038031641779$ 处应逐位复现 $\widetilde V=\widetilde S_{\rm acc}-1-\lambda^\star(B-1)$ 对应的值（即 2b 轮的 619.1292 减去常数项），相对差 $\le10^{-10}$。
4. **前向—后向一致性**：固定均匀输入取整后的开环策略，带 $\gamma_k$ 的得分，前向与后向的 $\sum\gamma_k\mathbb E[c]$ 相对差 $\le10^{-10}$。

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

# 5. 怎么读

- L3 成立：直接路线确实收紧，值得下一轮做 $A$ 的外层优化（$\mathcal D$ 对 $A$ 凸但可能不光滑）和严格化。
- 直接路线不比 $S$ 路线低：说明松弛的大头不在"方向"，在闭环本身；记录，不再沿此路投入。
- 两种情况都不改变"估算"性质；离散化误差方向仍未定。

# 6. 产出

`round2c/`：`logdet_dual.py`（$A$ 的构造、$\gamma_k$、带权 DP、外层）、`checks.py` + 日志（四项检查）、`report_round2c_CN.md`，只回答：四项检查；两个 $A$ × 两个模型的 $\min_\lambda\mathcal D-\log\det J_C$；$S$ 路线对应值；预测对照。禁用词：上界（本轮无严格结果）、全局。

# 7. 本轮未处理、下一轮再定的

$A$ 的外层优化；把离散 DP 换成有方向保证的证书（势函数）；功率项 $2\sum\log(4p_m)\le0$ 的松弛（它把功率与触发动态的耦合丢了）。
