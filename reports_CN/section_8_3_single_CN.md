---
title: "§8.3 绝对尺度、阈值、噪声水平与发送预算（单场景版，工作稿）"
date: "2026-10-09"
---

**本节的任务与范围。** §8.1–8.2 定义了估计对象 $\vartheta\in\mathbb R^9$、信息矩阵 (8.0)、准则 $m_D,m_E$ 和参考 (B)(C)，全部只用比值 $d=\Delta/\sigma$ 陈述。幅值/相位扫描要在一组固定的 $(\Delta,\sigma,\Delta_t)$ 下进行，所以本节要把这组数定下来，并给出该组参数下的发送预算 $B$ 与周期参考的样本数 $N$。按 Enrico 的意见，本文只取一个场景：$\Delta=\Delta_*$，$\sigma=\Delta_*/20$，$\Delta_t=10^{-3}$；其他噪声水平不在本节的结论范围内（8.3.5 附注）。

逻辑顺序：8.3.1 证明事件过程分别依赖 $\Delta$ 与 $\sigma$，所以除 $d$ 外必须再定一个绝对量；8.3.2 用均匀输入的无噪发送数目标定 $\Delta_*$，并证明这个定义良定、可精确计算；8.3.3 说明所得计数由信号的哪些量决定；8.3.4 说明 $\Delta_*$ 相对时钟是否足够粗；8.3.5 定 $\sigma$；8.3.6 用 MC 得到预算 $B$、样本数 $N$，并给出候选的合格判定规则。

# 8.3.0 记号

以下为 §8.3 全部符号。与 §8.1 相同的符号注明原编号。

## (a) 系统与频率

- $s$：Laplace 变量；$j=\sqrt{-1}$；$\omega$：角频率。
- 系统 $G(s)=\dfrac{2}{(1+s)^2}$。直流增益 $G(0)=2$。频响 $G(j\omega)=\dfrac{2}{(1+j\omega)^2}$。由 $(1+j\omega)^2=(1-\omega^2)+2j\omega$，
  $$G(j\omega)=\frac{2\big[(1-\omega^2)-2j\omega\big]}{(1-\omega^2)^2+4\omega^2}=\frac{2(1-\omega^2)}{(1+\omega^2)^2}-j\,\frac{4\omega}{(1+\omega^2)^2},$$
  其中分母用了 $(1-\omega^2)^2+4\omega^2=(1+\omega^2)^2$。于是 $|G(j\omega)|=\dfrac{2}{1+\omega^2}$，$\angle G(j\omega)=-2\arctan\omega$。
- 频率集 $\{\omega_1,\omega_2,\omega_3,\omega_4\}=\{3\sqrt2,\ 3\pi,\ 12,\ 12.3\}$，固定（§8.1.1）。数值：

| $m$ | $\omega_m$ | $\mathrm{Re}\,G(j\omega_m)$ | $\mathrm{Im}\,G(j\omega_m)$ | $|G(j\omega_m)|$ | $\angle G(j\omega_m)$ (rad) |
|---|---|---|---|---|---|
| 1 | 4.2426 | $-0.09418$ | $-0.04701$ | 0.10526 | $-2.6786$ |
| 2 | 9.4248 | $-0.02177$ | $-0.00467$ | 0.02227 | $-2.9302$ |
| 3 | 12 | $-0.01360$ | $-0.00228$ | 0.01379 | $-2.9753$ |
| 4 | 12.3 | $-0.01296$ | $-0.00212$ | 0.01313 | $-2.9793$ |

## (b) 输入

- 输入 $u(t)=a_0+\sum_{m=1}^4a_m\cos(\omega_mt+\chi_m)$。$a_0$：直流幅值；$a_m>0$：第 $m$ 音幅值；$\chi_m$：第 $m$ 音相位。
- 均匀输入 $u_{\rm unif}$：$a_0=1$，$a_m=\tfrac12$（$m=1,\dots,4$），$\chi_m=0$。满足功率归一 $\sum_{m=1}^4a_m^2=4\cdot\tfrac14=1$（§8.1.6）。本节只用 $u_{\rm unif}$。

## (c) 时钟与窗口

- 时钟步长 $\Delta_t=10^{-3}$；时钟点数 $K=5000$；时钟时刻 $t_k=k\Delta_t$，$k=1,\dots,K$；窗口长度 $T=K\Delta_t=5$。
- 相位 $\psi_{m,k}:=\omega_mt_k+\chi_m$。

## (d) 估计对象与回归向量

- 估计对象 (1.5)：$\vartheta=\big[G(0),\ \mathrm{Re}\,G(j\omega_1),\ \mathrm{Im}\,G(j\omega_1),\ \dots,\ \mathrm{Re}\,G(j\omega_4),\ \mathrm{Im}\,G(j\omega_4)\big]^\top\in\mathbb R^9$。真值记 $\vartheta_0$。
- 回归向量 (1.6)：
  $$h_k=\big[a_0,\ a_1\cos\psi_{1,k},\ -a_1\sin\psi_{1,k},\ \dots,\ a_4\cos\psi_{4,k},\ -a_4\sin\psi_{4,k}\big]^\top\in\mathbb R^9.$$
- 无噪稳态输出 $x_k:=h_k^\top\vartheta_0$。展开：
  $$x_k=a_0G(0)+\sum_{m=1}^4a_m\big[\mathrm{Re}\,G(j\omega_m)\cos\psi_{m,k}-\mathrm{Im}\,G(j\omega_m)\sin\psi_{m,k}\big].$$
  用 $\mathrm{Re}\,G=|G|\cos\angle G$、$\mathrm{Im}\,G=|G|\sin\angle G$ 和 $\cos A\cos B-\sin A\sin B=\cos(A+B)$，
  $$x_k=a_0G(0)+\sum_{m=1}^4a_m|G(j\omega_m)|\cos\big(\psi_{m,k}+\angle G(j\omega_m)\big),$$
  即线性系统对正弦输入的稳态响应：每个音被 $|G|$ 缩放、被 $\angle G$ 移相。
- 各音输出幅值 $C_m:=a_m|G(j\omega_m)|$：$C_1=0.05263$，$C_2=0.01113$，$C_3=0.00690$，$C_4=0.00657$；$C_{\min}:=\min_mC_m=C_4$。$x_k$ 在时钟上的范围 $[1.9289,\,2.0653]$。

## (e) 噪声与观测

- $v_k\sim N(0,\sigma^2)$，$k=1,\dots,K$，独立同分布；$\sigma>0$ 为噪声标准差。
- 观测 $z_k:=x_k+v_k$。
- $\phi(\xi)=\dfrac{1}{\sqrt{2\pi}}e^{-\xi^2/2}$：标准正态密度；$\Phi(\xi)=\int_{-\infty}^\xi\phi$：其分布函数。

## (f) 触发（send-on-delta）

- 阈值 $\Delta>0$。比值 $d:=\Delta/\sigma$。
- 参考值 $r_k$：时刻 $k$ 做判定时所用的、最近一次已发送的观测。
- 事件指示 $\epsilon_k\in\{0,1\}$。规则 (1.2)：
  $$\epsilon_1:=1,\quad r_2:=z_1;\qquad k\ge2:\ \ \epsilon_k=\mathbf 1\{|z_k-r_k|\ge\Delta\},\quad r_{k+1}=\begin{cases}z_k,&\epsilon_k=1,\\ r_k,&\epsilon_k=0.\end{cases}$$
  $\mathbf 1\{\cdot\}$ 为指示函数。首样本强制发送。
- 当前参考的时刻（$k\ge2$）$j(k):=\max\{i<k:\ \epsilon_i=1\}$，于是 $r_k=z_{j(k)}$；$j(2)=1$ 恒成立。
- 事件数 $N_{\rm ev}:=\sum_{k=1}^K\epsilon_k$（含首次）。事件时刻按序记 $j_0=1<j_1<\dots<j_{N_{\rm ev}-1}$。
- 无噪事件数 $N^{\rm nf}(\Delta)$：在规则中把 $z_k$ 换成 $x_k$（等价于 $\sigma=0$）所得的 $N_{\rm ev}$。它只依赖 $\Delta$ 与 $\{x_k\}$。

## (g) 信息系数与信息矩阵

- 无事件区间端点 (4.2)，$k\ge2$：$\alpha_k:=\dfrac{r_k-\Delta-x_k}{\sigma}$，$\beta_k:=\dfrac{r_k+\Delta-x_k}{\sigma}$；$P_k:=\Phi(\beta_k)-\Phi(\alpha_k)$。
- 单时刻信息系数 (5.9)(5.10)，$k\ge2$：
  $$c_k=c(\alpha_k,\beta_k)=\underbrace{1-P_k-\alpha_k\phi(\alpha_k)+\beta_k\phi(\beta_k)}_{c_{{\rm ev},k}}+\underbrace{\frac{\big(\phi(\alpha_k)-\phi(\beta_k)\big)^2}{P_k}}_{c_{{\rm no},k}}.$$
  首样本另列：$c_1:=1$，相应 $w_1=1$（完整高斯观测，不走删失公式）。
- 权重 $w_k:=\mathbb E[c_k]$（对噪声实现取期望）；权重和 $S:=\sum_{k=1}^Kw_k$。
- 信息矩阵 (8.0)：$J(\vartheta_0;u)=\sigma^{-2}\sum_{k=1}^Kw_kh_kh_k^\top$。

## (h) 预算与参考

- 预算 $B_d:=\mathbb E\big[N_{\rm ev}(u_{\rm unif};\Delta,\sigma)\big]$，下标 $d$ 标记所用的 $d=\Delta/\sigma$。
- 参考 (B)：$J_{\rm per}(N)=\sigma^{-2}\sum_{i=1}^Nh^{\rm unif}(t_i)h^{\rm unif}(t_i)^\top$，$t_i=iT/N$，$h^{\rm unif}$ 为 $u_{\rm unif}$ 的回归向量。
- 参考 (C)：$J_{\rm sod}(u_{\rm unif})=J(\vartheta_0;u_{\rm unif})$。

## (i) 其他

- $\mathrm{sign}$：符号函数；$\lfloor\cdot\rfloor$：向下取整；$|\cdot|$：绝对值。
- MC：$R$ 次独立噪声实现，第 $r$ 次的量加上标 $(r)$；$\widehat{(\cdot)}$ 表示 $R$ 次样本均值；标准误 $\mathrm{se}:=s/\sqrt R$，$s$ 为样本标准差。

# 8.3.1 为什么必须定绝对尺度

## (a) 问题

§8.1–8.2 保留了对 $\Delta,\sigma$ 的符号依赖，但尚未指定其绝对数值。§8.3 要用带噪触发的 MC 估计 $B_d$ 和 $J_{\rm sod}(u_{\rm unif})$。要算它们，必须先回答：带噪触发是不是只由 $d$ 决定？如果是，可以任取一个 $\sigma$；如果不是，就必须另外定一个绝对量。下面证明不是。

## (b) 触发条件的分解

取任一时刻 $k\ge2$，记当前参考的时刻 $j=j(k)$（依赖此前的事件序列），则 $r_k=z_j=x_j+v_j$。代入观测 $z_k=x_k+v_k$：
$$z_k-r_k=(x_k+v_k)-(x_j+v_j)=(x_k-x_j)+(v_k-v_j).\tag{8.20}$$
触发条件 $|z_k-r_k|\ge\Delta$ 两边除以 $\sigma>0$（不改变不等号），并用 $d=\Delta/\sigma$：
$$\Big|\frac{x_k-x_j}{\sigma}+\frac{v_k-v_j}{\sigma}\Big|\ \ge\ d.\tag{8.21}$$

## (c) 两项各依赖什么

**第二项** $\dfrac{v_k-v_j}{\sigma}=\dfrac{v_k}{\sigma}-\dfrac{v_j}{\sigma}$。$v_k/\sigma\sim N(0,1)$，与 $\sigma$ 无关。$v_j/\sigma$：$j$ 是上一次触发的时刻，由此前的 $z$ 序列决定，所以 $v_j$ 是"被选中"的噪声，一般不能按未选择的高斯噪声处理（例外是 $j=1$：首次强制发送不经选择，$v_1\sim N(0,\sigma^2)$）。其条件分布依赖此前的信号路径（在 $\sigma$ 单位下依赖 $(x_i-x_{j'})/\sigma$ 之类的量），因此第二项也通过选择效应依赖第一项那类量，不能单独说"与 $\sigma$ 无关"。

**第一项** $\dfrac{x_k-x_j}{\sigma}$。由 8.3.0(d)，
$$x_k-x_j=\sum_{m=1}^4C_m\Big[\cos\big(\psi_{m,k}+\angle G(j\omega_m)\big)-\cos\big(\psi_{m,j}+\angle G(j\omega_m)\big)\Big],$$
直流项 $a_0G(0)$ 相减消去。$C_m=a_m|G(j\omega_m)|$ 由输入幅值和系统决定，是固定的绝对量，不含 $\sigma$。所以第一项与 $\sigma$ 成反比：固定 $d$、把 $\sigma$ 换成 $\sigma/10$，第一项变成原来的 10 倍。

**联合缩放不变性。** 若把 $x$、$\Delta$、$\sigma$ 三者同时乘以 $\lambda>0$（噪声在各自 $\sigma$ 单位下取同一组抽样），(8.21) 的两项都不变，$\alpha_k,\beta_k$ 也不变，事件过程与 $c_k$ 完全不变。所以无量纲量有两个：$\Delta/\sigma$ 和 $x/\sigma$。本文 $x$ 由 $a_0=1$、$\sum a_m^2=1$ 和 $G$ 钉死，不能缩放；于是 $(\Delta,\sigma)$ 两个绝对量中只有一个能被 $d$ 吸收，另一个必须单独指定。以下 (d)(e) 的"$(\Delta,\sigma)$ 的函数"都在 $u,G,\Delta_t$ 固定的前提下陈述。

## (d) 结论：事件过程不是 $d$ 的函数

两套参数 $(\Delta,\sigma)$ 与 $(\Delta',\sigma')=(\lambda\Delta,\lambda\sigma)$，$\lambda>0$，$\lambda\ne1$，$d$ 相同。不能对一般的 $k$ 用同一个 $j(k)$ 比较两套参数：尺度改变后此前的事件可能不同，$j(k)$ 随之不同。所以取 $k=2$，此时两套参数都有 $j(2)=1$，参考 $r_2=z_1$，不含选择效应。

记 $\delta x:=x_2-x_1$，$\mu:=\delta x/\sigma$，$\xi_k:=v_k/\sigma\sim N(0,1)$ 独立。由 (8.21)，
$$\epsilon_2=1\iff|\mu+\xi_2-\xi_1|\ge d.$$
$\xi_2-\xi_1$ 是两个独立标准正态之差，服从 $N(0,2)$。于是
$$\Pr(\epsilon_2=1)=\Pr(\xi_2-\xi_1\le-d-\mu)+\Pr(\xi_2-\xi_1\ge d-\mu)=\Phi\Big(\frac{-d-\mu}{\sqrt2}\Big)+1-\Phi\Big(\frac{d-\mu}{\sqrt2}\Big).\tag{8.22}$$
(8.22) 是 $(d,\mu)$ 的函数，而 $\mu=\delta x/\sigma$ 随 $\sigma$ 的绝对值变。本信号 $\delta x=1.5414\times10^{-4}\ne0$。固定 $d$：
$$\sigma\to0:\ \mu\to+\infty,\ \Pr(\epsilon_2=1)\to1;\qquad\sigma\to\infty:\ \mu\to0,\ \Pr(\epsilon_2=1)\to2\Phi(-d/\sqrt2).$$
$d=20$ 时后者为 $2.1\times10^{-45}$。两个极限不同，所以 $\Pr(\epsilon_2=1)$ 不是 $d$ 的函数，事件过程不能仅由 $d$ 决定。这是严格证明，不依赖后续索引。

事件序列不同不必然意味着总事件数不同，但一般也会不同。数值（$u_{\rm unif}$，$d=20$，$\lambda=10$；MC 用 `numpy.random.default_rng(1)`，$R=200$，$\pm$ 为标准误）：

| | $\Delta$（仿真用未舍入值） | $\sigma$ | $d$ | $N^{\rm nf}$ | 带噪 $\widehat N_{\rm ev}$ |
|---|---|---|---|---|---|
| 原尺度 | $0.002384694482635785$ | $\Delta/20$ | 20 | 321 | $317.3\pm0.2$ |
| 放大 10 倍 | $0.02384694482635785$ | $\Delta/20$ | 20 | 30 | $29.4\pm0.1$ |

同一 $d$，事件数相差约一个数量级。注意无噪计数对阈值舍入敏感：$\Delta=0.002385$ 给 $N^{\rm nf}=319$，故表中给出仿真实际使用的全精度值（该值是 8.3.2 定义的 $\Delta_*$）。

## (e) $w_k$ 与 $J$ 同样不是 $d$ 的函数

$k\ge2$ 时 $\alpha_k,\beta_k$ 中出现 $r_k-x_k=z_j-x_k=(x_j-x_k)+v_j$，所以
$$\alpha_k=\frac{x_j-x_k}{\sigma}+\frac{v_j}{\sigma}-d,\qquad\beta_k=\frac{x_j-x_k}{\sigma}+\frac{v_j}{\sigma}+d.$$
第一项与 (c) 中同一类，随 $\sigma$ 的绝对值变；且 $j=j(k)$ 本身由 (d) 中证明不是 $d$ 的函数的事件过程决定。

同样用 $k=2$ 给出严格论证。$j(2)=1$，$v_1=\sigma\xi_1$ 未经选择，
$$\alpha_2=\xi_1-\mu-d,\qquad\beta_2=\xi_1-\mu+d,\qquad w_2=\mathbb E_{\xi_1}\big[c(\xi_1-\mu-d,\ \xi_1-\mu+d)\big].$$
固定 $d$：$\sigma\to0$ 时 $\mu\to+\infty$，对每个固定的 $\xi_1$ 有 $\alpha_2,\beta_2\to-\infty$，于是 $P_2\to0$，$\alpha_2\phi(\alpha_2),\beta_2\phi(\beta_2)\to0$，$c_{{\rm ev},2}\to1$，$c_{{\rm no},2}=(\phi(\alpha_2)-\phi(\beta_2))^2/P_2\to0$（用 $P_2\ge$ 区间内密度的下界 $\times2d$ 与 $\phi$ 的指数衰减），故 $w_2\to1$。$\sigma\to\infty$ 时 $\mu\to0$，$w_2\to\mathbb E[c(\xi_1-d,\xi_1+d)]$；由 (5.11) $c=1-P\,\mathrm{Var}(\xi\mid\alpha<\xi<\beta)$，有限区间上 $P>0$、条件方差 $>0$，故 $c<1$ 对每个 $\xi_1$ 成立，$w_2<1$。两个极限不同，$w_2$ 不是 $d$ 的函数。

于是在 $u,G,\Delta_t$ 固定下，$c_k$、$w_k=\mathbb E[c_k]$、$S$、以及 $J=\sigma^{-2}\sum w_kh_kh_k^\top$ 都是 $(\Delta,\sigma)$ 的函数，不是 $d$ 的函数（$w_k$ 当然还依赖 $\{x_k\}$，即 $u$ 与 $G$；这里说的是传感器参数的依赖形式）。这是 §8.2.6 关于 $\sigma$ 的结论（$m_D$ 对 $\sigma$ 的变化不具有不变性）的具体来源。

## (f) 关于 CRLB 的尺度

由 (8.0)，
$$J^{-1}=\sigma^2\Big[\sum_{k=1}^Kw_k(\Delta,\sigma)\,h_kh_k^\top\Big]^{-1}.$$
只有括号外的 $\sigma^2$ 是显式缩放；括号内的 $w_k$ 随 $(\Delta,\sigma)$ 变。所以"CRLB 按 $\sigma^2$ 走"只对显式因子成立，对整个 $J^{-1}$ 不成立。候选之间的比较（$\Delta m_D$、$m_E$）中 $\sigma^{-2}$ 约掉，但 $w_k$ 的差异不约掉，因此比较结果也依赖绝对尺度。

## (g) 本小节的结论

1. 在 $u,G,\Delta_t$ 固定下，事件过程 $\{\epsilon_k\}$、$w_k$、$S$、$J$ 都是 $(\Delta,\sigma)$ 的函数而非 $d$ 的函数（(d)(e) 用 $k=2$ 严格证明；$N_{\rm ev}$ 一般也随之不同，数值见 (d) 的表）。
2. 因此 §8.3 在 $\sigma=\Delta/d$ 之外必须再定一个绝对量。两个自由度减去一个约束，剩一个要定：定 $\Delta$ 或定 $\sigma$。定 $\Delta$ 的理由在 8.3.2，$\sigma$ 的取法在 8.3.5。
3. 所有后续结论都要附 $(\Delta,\sigma,\Delta_t)$ 三个绝对量，不能只附 $d$。

# 8.3.2 阈值 $\Delta_*$ 的定义

## (a) 定哪一个、怎么定

8.3.1(g) 留下一个待定的绝对量：在 $\sigma=\Delta/d$ 之外，定 $\Delta$ 或定 $\sigma$。本文定 $\Delta$，定法是：给定目标事件数 $N_0^{\rm nf}$，取使均匀输入 $u_{\rm unif}$ 的**无噪**事件数达到 $N_0^{\rm nf}$ 的最大阈值。理由：

1. $\Delta$ 是传感器参数（brief：固定、不是设计变量）。"一个窗口内大约发多少次"是它最直接的物理含义，与 brief 的通信视角一致。
2. 无噪事件数 $N^{\rm nf}(\Delta)$ 是 $\Delta$ 的确定性函数（8.3.0(f)），无 MC 误差；而 $B_d$ 是含噪期望，要 MC 估计。用 $N^{\rm nf}$ 定 $\Delta$ 不与 $B_d$ 循环。
3. 与旧实验的做法一致。

工作约定：$N_0^{\rm nf}=320$（旧实验含噪事件数 $\approx320$ 的量级；没有更强的理论依据，8.3.4 给出它的时钟裕量）。

无噪输出取稳态 $x_k=h_k^\top\vartheta_0$ 而非零状态启动。理由：回归模型 $y_k=h_k^\top\vartheta+e_k$ 的 $h_k$ 只含常数与 $\cos,\sin$，没有暂态项，估计对象只描述稳态；零状态启动的输出从 0 上升到 $a_0G(0)=2$，其时钟总变差为 $2.068$，而稳态输出的总变差只有 $0.839$（总变差的定义见 (b)），阶跃段会主导事件数，而它不在模型里。

## (b) 本小节用到的量

- 无噪触发：在 8.3.0(f) 的规则中令 $\sigma=0$，即 $z_k=x_k$。参考 $r_k=x_{j(k)}$。记事件指示为 $\epsilon_k(\Delta)$，事件数 $N^{\rm nf}(\Delta)=\sum_k\epsilon_k(\Delta)$。
- 临界集 $\mathcal C:=\{|x_k-x_j|:\ 1\le j<k\le K\}\setminus\{0\}$，有限集。把它的元素升序记为 $0<c_1<c_2<\dots<c_L$，并令 $c_0:=0$。
- 可行集 $\mathcal F:=\{\Delta>0:\ N^{\rm nf}(\Delta)\ge N_0^{\rm nf}\}$。
- 时钟上的总变差 $V_T:=\sum_{k=1}^{K-1}|x_{k+1}-x_k|$。

## (c) $N^{\rm nf}$ 是分段常数

**引理 4.** 若 $\Delta,\Delta'$ 落在同一个区间 $(c_i,c_{i+1}]$（$0\le i<L$）或同一个 $(c_L,\infty)$ 内，则 $\epsilon_k(\Delta)=\epsilon_k(\Delta')$ 对所有 $k$ 成立，特别地 $N^{\rm nf}(\Delta)=N^{\rm nf}(\Delta')$。

*证明.* 对 $k$ 归纳证明两套阈值下的事件指示和参考时刻相同。$k=1$：$\epsilon_1=1$ 与阈值无关。设 $k\ge2$ 且对所有 $i<k$ 两套的 $\epsilon_i$ 相同，则 $j(k)=\max\{i<k:\epsilon_i=1\}$ 相同，记为 $j$，两套的参考都是 $x_j$。时刻 $k$ 的判定是比较 $|x_k-x_j|$ 与阈值。$|x_k-x_j|\in\mathcal C\cup\{0\}$，所以它要么 $\le c_i$，要么 $\ge c_{i+1}$，不可能严格落在 $(c_i,c_{i+1})$ 内（$\mathcal C$ 中没有这样的元素）。若 $|x_k-x_j|\le c_i$：因 $\Delta,\Delta'>c_i$，两套都判"不触发"。若 $|x_k-x_j|\ge c_{i+1}$：因 $\Delta,\Delta'\le c_{i+1}$，两套都判"触发"（规则是 $\ge$）。对 $(c_L,\infty)$：所有 $|x_k-x_j|\le c_L<\Delta,\Delta'$，两套都不触发。故 $\epsilon_k(\Delta)=\epsilon_k(\Delta')$，归纳成立。$\square$

推论：$N^{\rm nf}$ 在每个 $(c_i,c_{i+1}]$ 上为常数，在 $(c_L,\infty)$ 上恒为 1（只有首次强制发送）。它只在 $\mathcal C$ 的点处可能跳变，且每段**左开右闭**——这来自规则用 $\ge$：在 $\Delta=c_{i+1}$ 处差为 $c_{i+1}$ 的比较仍触发，$\Delta$ 略大就不触发。

## (d) $\mathcal F$ 的最大值存在、取到、且是临界值

**引理 5.** 若 $N_0^{\rm nf}\ge2$ 且 $\mathcal F\ne\varnothing$，则 $\max\mathcal F$ 存在，并且 $\max\mathcal F\in\mathcal C$。

*证明.* 由引理 4，$\mathcal F$ 是若干个区间 $(c_i,c_{i+1}]$ 之并（区间 $(c_L,\infty)$ 上 $N^{\rm nf}=1<N_0^{\rm nf}$，不在 $\mathcal F$ 中）。有限个左开右闭区间之并，取其中右端点最大者 $c_{i^*+1}$，它属于 $\mathcal F$（右端点包含在区间内），且 $\mathcal F$ 中没有更大的点。故 $\max\mathcal F=c_{i^*+1}\in\mathcal C$。$\square$

定义
$$\Delta_*:=\max\mathcal F=\max\{\Delta>0:\ N^{\rm nf}(\Delta)\ge N_0^{\rm nf}\}.\tag{8.23}$$
由引理 5，$\Delta_*$ 等于某一对时钟点的输出差 $|x_k-x_j|$。实际计数 $N^{\rm nf}(\Delta_*)$ 可以大于 $N_0^{\rm nf}$（分段常数函数不保证恰好取到 320），报实际值。

## (e) $N^{\rm nf}$ 不是单调的

直觉"阈值变大，每个事件要走更远，事件只会变少"不成立。本信号的反例：
$$N^{\rm nf}(0.00231050)=326,\qquad N^{\rm nf}(0.00231075)=330.$$
机制：两套阈值的事件序列在 $k=1737$ 首次分叉——$\Delta=0.00231050$ 时该点触发，$\Delta=0.00231075$ 时不触发。计数分解：

| 计数位置 | $\Delta=0.00231050$ | $\Delta=0.00231075$ |
|---|---|---|
| $k<1737$ | 126 | 126 |
| $k=1737$ | 1 | 0 |
| $k>1737$ | 199 | 204 |
| 合计 | 326 | 330 |

较大阈值在分叉点少发送一次，但其参考值停在更早的时刻，与随后信号路径的相对位置改变，此后多发送五次，总事件数多四次。参考更新使两条历史分叉，简单的逐点比较不能推出总事件数单调。

后果：$\mathcal F$ 一般不是区间，不能用二分法求 $\max\mathcal F$（二分法假定可行集是区间）。

## (f) 上界：把搜索范围封住

**引理 6.** $N^{\rm nf}(\Delta)-1\le V_T/\Delta$。

*证明.* 事件 $j_0<\dots<j_{N-1}$，$N=N^{\rm nf}(\Delta)$。事件 $j_{\ell+1}$ 的定义是 $|x_{j_{\ell+1}}-x_{j_\ell}|\ge\Delta$（参考 $r=x_{j_\ell}$）。三角不等式：$|x_{j_{\ell+1}}-x_{j_\ell}|\le\sum_{k=j_\ell}^{j_{\ell+1}-1}|x_{k+1}-x_k|$。对 $\ell=0,\dots,N-2$ 求和，右边各段的求和范围互不重叠、都含于 $\{1,\dots,K-1\}$，故 $(N-1)\Delta\le\sum_\ell|x_{j_{\ell+1}}-x_{j_\ell}|\le V_T$。$\square$

（8.3.3 会把这个界收紧。）于是 $N^{\rm nf}(\Delta)\ge320$ 要求 $\Delta\le U:=V_T/319$。本信号 $V_T=0.839229755797285$，$U=0.002630814281496191\approx0.00263081$。所以 $\mathcal F\subset(0,U]$。

## (g) 计算 $\Delta_*$：对临界值穷举

由引理 5、6，$\Delta_*$ 是 $\mathcal C\cap(0,U]$ 中满足 $N^{\rm nf}\ge320$ 的最大元素。算法：

1. 生成 $\mathcal C$（$K(K-1)/2=12{,}497{,}500$ 个差值），取落在 $[0.0023,\ U]$ 内的不同值（$U$ 用全精度），降序排列。
2. 从大到小逐个计算 $N^{\rm nf}$，遇到第一个 $\ge320$ 的停止。

为什么降序遇到的第一个可行值就是 $\max\mathcal F$：$U$ 以上无可行值（引理 6），而比它大的临界值已全部验证为不可行。下限 $0.0023$ 只限制搜索量（窗口越宽临界值越多），不参与论证：由于在窗口内找到了可行值，下限未起作用；若窗口内无可行值，须下移下限重搜。

窗口内有 82,631 个临界值，计算到第 61,361 个时停止。结果：
$$\Delta_*=0.002384694482635785=|x_{1352}-x_{1330}|,\qquad N^{\rm nf}(\Delta_*)=321.$$
全局最大性来自完整扫描：所有更大的临界值均已逐个验证为不可行。紧邻其上的临界值 $c=0.0023846964965797$ 给 $N^{\rm nf}=319$，这只进一步验证了该处的局部跳变，不是最大性的依据。

舍入敏感性：$N^{\rm nf}(0.002385)=319$，$N^{\rm nf}(0.00238)=321$。所以 $\Delta_*$ 在正文中可写作 $2.3847\times10^{-3}$，但所有仿真使用上面的全精度值；任何复现都必须用全精度值。

## (h) 本小节的结论

1. $\Delta$ 由 $u_{\rm unif}$ 的无噪事件数反定，目标 $N_0^{\rm nf}=320$（工作约定），用稳态输出。
2. $N^{\rm nf}(\Delta)$ 分段常数、左开右闭（引理 4），不单调（(e)），$\max\mathcal F$ 存在且是临界值（引理 5），搜索范围由引理 6 封住。
3. $\Delta_*=0.002384694482635785$，$N^{\rm nf}(\Delta_*)=321$。下一小节（8.3.3）给出 $N^{\rm nf}$ 的两侧界与超调，说明 $N_0^{\rm nf}\mapsto\Delta_*$ 的形状。

# 8.3.3 无噪事件数的结构：两侧界与超调

## (a) 目的

8.3.2 把 $\Delta_*$ 定义成一个只能靠跑链才能得到的数。本小节回答：$N^{\rm nf}(\Delta)$ 的大小由信号的哪些量决定，决定到什么程度。结果是一对精确的不等式 (8.27)，把 $N^{\rm nf}$ 夹在一个由总变差、超调和极值个数决定的窗口里；窗口宽约 30。用它做启发式反算，本例中给出 $\Delta_*$ 的一个相对半宽约 5% 的预测窗口；精确值仍靠 8.3.2 的穷举。副产品是看清两种"路程损失"的来源：时钟离散造成的超调，和转向处的折返。

## (b) 本小节用到的量

- 事件时刻 $j_0=1<j_1<\dots<j_{N-1}$，$N=N^{\rm nf}(\Delta)$；参考在 $[j_\ell,j_{\ell+1})$ 内恒为 $r=x_{j_\ell}$。
- 超调 $o_\ell:=|x_{j_{\ell+1}}-x_{j_\ell}|-\Delta$，$\ell=0,\dots,N-2$；平均超调 $\bar o:=\frac1{N-1}\sum_\ell o_\ell$（需 $N\ge2$）。由事件定义 $|x_{j_{\ell+1}}-x_{j_\ell}|\ge\Delta$，故 $o_\ell\ge0$。
- 时钟总变差 $V_T=\sum_{k=1}^{K-1}|x_{k+1}-x_k|$；最大单步变化 $\delta_{\max}:=\max_k|x_{k+1}-x_k|$。
- 局部极值：先删去增量序列 $\{x_{k+1}-x_k\}$ 中的零增量，再在剩下的序列里统计符号反转的次数，记为 $n_{\rm ext}$；每次反转对应一个局部极值（平台转向也被计入）。本信号无零增量，$n_{\rm ext}=15$。
- 单调片段：把一段连续时钟点序列在每个局部极值处切开，得到的每一段内 $x$ 单调。片段的带符号位移 $m:=x_{\rm end}-x_{\rm start}$。

## (c) 上界（精确）

**引理 7.** $$(N-1)(\Delta+\bar o)\le V_T,\qquad\text{即}\qquad N-1\le\frac{V_T}{\Delta+\bar o}.\tag{8.24}$$

*证明.* 与引理 6 相同，只是不丢掉超调。$|x_{j_{\ell+1}}-x_{j_\ell}|=\Delta+o_\ell$（超调的定义）。三角不等式 $|x_{j_{\ell+1}}-x_{j_\ell}|\le\sum_{k=j_\ell}^{j_{\ell+1}-1}|x_{k+1}-x_k|$。对 $\ell=0,\dots,N-2$ 求和：左边 $=\sum_\ell(\Delta+o_\ell)=(N-1)(\Delta+\bar o)$；右边各段时钟区间 $[j_\ell,j_{\ell+1})$ 互不重叠、含于 $[1,K)$，故 $\le V_T$。$\square$

引理 6 是 (8.24) 丢掉 $\bar o\ge0$ 后的弱形式。

## (d) 下界（精确）

(8.24) 的松弛量 $V_T-(N-1)(\Delta+\bar o)$ 是"走过但没换成事件"的路程。它只来自两处：相邻事件之间的折返，和最后一个事件之后的尾段。

**引理 8.** $$V_T-(N-1)(\Delta+\bar o)<2\Delta\,(n_{\rm ext}+1).\tag{8.25}$$

*证明.* 分两部分。

*相邻事件之间。* 取 $\ell$，参考 $r=x_{j_\ell}$，净位移 $D_\ell:=x_{j_{\ell+1}}-x_{j_\ell}$，$|D_\ell|=\Delta+o_\ell$。对 $j_\ell<k<j_{\ell+1}$ 无事件，故
$$|x_k-r|<\Delta,\qquad\text{即}\qquad x_k\in(r-\Delta,\ r+\Delta).\tag{8.26}$$
把 $x_{j_\ell},\dots,x_{j_{\ell+1}}$ 切成单调片段 $m_0,\dots,m_{n_\ell}$（$n_\ell$ 为严格位于 $(j_\ell,j_{\ell+1})$ 内的局部极值个数）。则
$$\sum_i|m_i|=\sum_{k=j_\ell}^{j_{\ell+1}-1}|x_{k+1}-x_k|\quad(\text{该段路程}),\qquad\sum_im_i=D_\ell\quad(\text{位移之和等于净位移}).$$
把片段按符号与 $D_\ell$ 相同（记 $+$）或相反（记 $-$）分组，令 $\Sigma_+:=\sum_{+}|m_i|$，$\Sigma_-:=\sum_{-}|m_i|$。则 $\sum_i|m_i|=\Sigma_++\Sigma_-$，而 $|D_\ell|=|\sum_im_i|=\Sigma_+-\Sigma_-$（同号片段全部加、反号片段全部减）。相减得该段的多余路程
$$\sum_i|m_i|-|D_\ell|=2\Sigma_-.$$
现在估计 $\Sigma_-$。(i) 最后一个片段是 $+$ 的：设 $D_\ell>0$（反之对称），则 $x_{j_{\ell+1}}\ge r+\Delta$ 而由 (8.26) $x_{j_{\ell+1}-1}<r+\Delta$，所以最后一步向上，最后一片段向上，与 $D_\ell$ 同号。(ii) 每个 $-$ 片段的两个端点都是 $x_{j_\ell}=r$ 或 (8.26) 中的内部点，全在 $[r-\Delta,r+\Delta)$ 与 $(r-\Delta,r+\Delta]$ 的交内（端点 $r$ 本身在带内），故 $|m_i|<2\Delta$。(iii) 计数。单调片段符号交替，共 $n_\ell+1$ 个，最后一个是 $+$。分奇偶：

- $n_\ell=0$：只有一个片段，是 $+$，$\Sigma_-=0$。
- $n_\ell=2q\ge2$：$2q+1$ 个片段，最后一个 $+$，交替得恰有 $q$ 个 $-$ 片段，每个 $<2\Delta$：$\Sigma_-<2q\Delta=n_\ell\Delta$。
- $n_\ell=2q+1$：$2q+2$ 个片段，最后一个 $+$，交替得第一个是 $-$，共 $q+1$ 个 $-$ 片段。第一个 $-$ 片段从 $x_{j_\ell}=r$ 出发、终点是带内的点，故长度 $<\Delta$；其余 $q$ 个 $<2\Delta$：$\Sigma_-<\Delta+2q\Delta=n_\ell\Delta$。

三种情况合起来：多余路程 $=2\Sigma_-<2\Delta\,n_\ell$（$n_\ell\ge1$），或 $=0$（$n_\ell=0$）。注意只用 $|m_i|<2\Delta$ 和"$-$ 片段数 $\le n_\ell$"相乘只能得到 $4\Delta n_\ell$，奇偶计数是必要的。

*尾段.* $k>j_{N-1}$ 无事件，所有点满足 (8.26)（参考 $r=x_{j_{N-1}}$）。尾段切成 $n'+1$ 个单调片段（$n'$ 为尾段内部极值个数）。第一个片段从 $r$ 出发、终点在带内，长度 $<\Delta$；其余 $n'$ 个端点都在带内，$|m_i|<2\Delta$。尾段路程 $<\Delta+2\Delta n'<2\Delta(n'+1)$。

*相加.* 各事件区间内部极值与尾段内部极值互不重复，且都是全窗的局部极值，故 $\sum_\ell n_\ell+n'\le n_{\rm ext}$。于是
$$\begin{aligned}V_T&=\sum_{\ell=0}^{N-2}\Big[(\Delta+o_\ell)+2\Sigma_-^{(\ell)}\Big]+(\text{尾段路程})\\&<(N-1)(\Delta+\bar o)+2\Delta\sum_\ell n_\ell+2\Delta(n'+1)\ \le\ (N-1)(\Delta+\bar o)+2\Delta(n_{\rm ext}+1).\ \square\end{aligned}$$

合并 (8.24)(8.25)：
$$\boxed{\ \frac{V_T}{\Delta+\bar o}-\frac{2\Delta\,(n_{\rm ext}+1)}{\Delta+\bar o}\ <\ N^{\rm nf}(\Delta)-1\ \le\ \frac{V_T}{\Delta+\bar o}\ }\tag{8.27}$$

## (e) 超调的近似

(8.27) 含 $\bar o$，它本身要从事件序列读出。要在跑链之前预测，需要 $\bar o$ 的先验估计。

事件在时钟点 $j_{\ell+1}$ 触发，意味着输出在 $(t_{j_{\ell+1}-1},t_{j_{\ell+1}}]$ 内越过了 $r\pm\Delta$；超调 $o_\ell$ 是越界后在这一步剩余部分又走的路程。若越界时刻在该步内均匀分布，则超调约为 $U\,|\dot x|\Delta_t$，$U\sim\mathrm{Unif}(0,1)$，均值 $\tfrac12|\dot x|\Delta_t$。事件在时间上的密度约为 $|\dot x|/\Delta$（单位时间走过的路程除以每事件消耗的路程）。按事件密度加权：
$$\bar o\approx\frac{\int_0^T\tfrac12|\dot x(t)|\Delta_t\cdot\frac{|\dot x(t)|}{\Delta}\,dt}{\int_0^T\frac{|\dot x(t)|}{\Delta}\,dt}=\frac{\Delta_t}{2}\cdot\frac{\int_0^T\dot x(t)^2\,dt}{\int_0^T|\dot x(t)|\,dt}.\tag{8.28}$$
$\Delta$ 在分子分母中约掉，所以这个估计不依赖 $\Delta$，可以在定 $\Delta$ 之前算。它是近似，不是引理：忽略了转向附近事件密度的改变、以及越界位置在离散时钟上并非均匀。注意不能用全窗平均 $V_T/(2K)$ 代替 (8.28)：事件集中在斜率大的地方，全窗平均会低估。

## (f) 核数（$u_{\rm unif}$，稳态，$\Delta=\Delta_*$）

信号量：$V_T=0.839230$，$n_{\rm ext}=15$，$\delta_{\max}=4.755\times10^{-4}$。事件序列读出：$N=321$，$\bar o=1.3719\times10^{-4}$（$\bar o/\Delta_*=5.75\%$），$o_\ell\in[0,\ 4.677\times10^{-4}]$（最小值 0 出现在定义 $\Delta_*$ 的那对点上，最大值不超过 $\delta_{\max}$）。

| 量 | 值 |
|---|---|
| (8.28) 的 $\bar o$ | $1.3415\times10^{-4}$（与实测差 $-2.2\%$） |
| 全窗平均 $V_T/(2K)$ | $0.839\times10^{-4}$（低估 39%） |
| 上界 $V_T/(\Delta_*+\bar o)$ | 332.78 |
| 下界 $V_T/(\Delta_*+\bar o)-2\Delta_*(n_{\rm ext}+1)/(\Delta_*+\bar o)$ | 302.52 |
| $N^{\rm nf}-1$ | 320 |

(8.27) 成立。松弛量的实际分解：相邻事件之间的折返路程合计 $0.02913$，尾段 $0.00310$，和为 $0.03223=V_T-(N-1)(\Delta_*+\bar o)$（恒等，核对无误）；折合 $(\Delta_*+\bar o)$ 单位 12.78 个事件当量，引理 8 的上界 30.26。折返的实际损失约为上界的 40%：上界假设每个 $-$ 片段都接近 $2\Delta$，实际小波动远小于带宽。

## (g) 启发式反算：$\Delta_*$ 的预测窗口

(8.27) 是精确的，但其中 $\bar o$ 是事件序列的实际平均超调，扫描前并不知道；用 (8.28) 的先验值 $1.3415\times10^{-4}$ 代替后，不等式不再是严格保证。扫描前知道的事件数也只有目标 $N_0^{\rm nf}=320$（含首次强制发送），不是扫描结果 321；用目标做粗估应取 $N-1\approx N_0^{\rm nf}-1=319$。按此把 (8.27) 两边解出 $\Delta$：上界 $319(\Delta+\bar o)\le V_T$ 给 $\Delta\le V_T/319-\bar o=2.497\times10^{-3}$；下界 $319(\Delta+\bar o)+2\Delta(n_{\rm ext}+1)>V_T$ 给 $\Delta>(V_T-319\bar o)/351=2.269\times10^{-3}$。预测窗口
$$\Delta_*\in(2.269\times10^{-3},\ 2.497\times10^{-3}),$$
相对半宽 4.8%。实际 $\Delta_*=2.385\times10^{-3}$ 落在其中。这只是本例对启发式预测的验证，不构成可行性或最大性的严格保证；精确值由 8.3.2 的穷举确定。它说明的是：$N_0^{\rm nf}\mapsto\Delta_*$ 的映射由 $(V_T,\bar o,n_{\rm ext})$ 三个量决定到百分之几，个位精度要跑链。

## (h) 本小节的结论

1. 精确界 (8.27)：$N^{\rm nf}-1$ 被 $V_T/(\Delta+\bar o)$ 从上夹住，松弛量小于 $2\Delta(n_{\rm ext}+1)$（引理 7、8）。
2. 两种损失：时钟超调 $\bar o\approx5.8\%\Delta$（每事件多走的路程），转向折返约 13 个事件当量（走了但没换成事件）。
3. (8.28) 给出不依赖 $\Delta$ 的超调先验，本例误差 2%；据此做启发式反算，本例给出相对半宽 4.8% 的预测窗口，$\Delta_*$ 落在其中。
4. 这些都是无噪的性质。带噪时相邻发送值的跳变量为 $\Delta+o^z_\ell$，$o^z_\ell$ 含噪声的超出部分，它不再等于无噪信号每次事件消耗的路程，见 8.3.6。

# 8.3.4 时钟分辨与 $N_0^{\rm nf}$ 的范围

## (a) 目的

触发只在时钟点判定。阈值相对每步位移越小，时钟对事件序列的影响越大；极端时事件数由时钟而非信号决定。本小节定义一个分辨裕量 $\rho$，证明它控制事件之间的最小间隔和超调的上界（引理 9），然后用它说明 $N_0^{\rm nf}$ 不能取得太大；再结合参考 (B) 的条件数说明不能取得太小。8.3.2 的工作约定 320 落在这个范围里。

## (b) 本小节用到的量

- 最大单步位移 $\delta_{\max}:=\max_{1\le k\le K-1}|x_{k+1}-x_k|$；本信号 $4.7549\times10^{-4}$。
- 分辨裕量 $\rho:=\Delta/\delta_{\max}$：阈值是最大单步位移的多少倍。
- 事件间隔 $g_\ell:=j_{\ell+1}-j_\ell$（时钟步数），$\ell=0,\dots,N-2$；平均间隔 $\bar g:=\frac1{N-1}\sum_\ell g_\ell=\frac{j_{N-1}-1}{N-1}$。注意 $K/N$（窗口步数除以发送数）包含最后一次发送之后的尾段，不是平均间隔。
- $\lceil\cdot\rceil$：向上取整。

## (c) 引理 9：裕量控制最小间隔与超调

**引理 9.** 无噪触发下，对每个 $\ell$：
$$\text{(i)}\ \ g_\ell\ge\lceil\rho\rceil;\qquad\text{(ii)}\ \ o_\ell<\delta_{\max}=\Delta/\rho;\qquad\text{(iii)}\ \ N-1\le\frac{K-1}{\lceil\rho\rceil}.\tag{8.29}$$

*证明.* (i) 事件定义给 $|x_{j_{\ell+1}}-x_{j_\ell}|\ge\Delta$。三角不等式：$|x_{j_{\ell+1}}-x_{j_\ell}|\le\sum_{k=j_\ell}^{j_{\ell+1}-1}|x_{k+1}-x_k|\le g_\ell\,\delta_{\max}$（$g_\ell$ 步，每步 $\le\delta_{\max}$）。合起来 $g_\ell\delta_{\max}\ge\Delta$，即 $g_\ell\ge\Delta/\delta_{\max}=\rho$；$g_\ell$ 是整数，故 $g_\ell\ge\lceil\rho\rceil$。

(ii) 记 $r=x_{j_\ell}$。先证 $|x_{j_{\ell+1}-1}-r|<\Delta$：若 $g_\ell>1$，事件前一点 $x_{j_{\ell+1}-1}$ 是段内中间点、没触发，故与参考的距离 $<\Delta$；若 $g_\ell=1$，前一点就是参考点 $x_{j_\ell}=r$，距离为 0。两种情况都成立。再走一步：$|x_{j_{\ell+1}}-r|\le|x_{j_{\ell+1}-1}-r|+|x_{j_{\ell+1}}-x_{j_{\ell+1}-1}|<\Delta+\delta_{\max}$。超调 $o_\ell=|x_{j_{\ell+1}}-r|-\Delta<\delta_{\max}$。

(iii) 间隔之和 $\sum_{\ell=0}^{N-2}g_\ell=j_{N-1}-j_0\le K-1$，每项 $\ge\lceil\rho\rceil$，故 $(N-1)\lceil\rho\rceil\le K-1$。$\square$

推论：平均超调 $\bar o/\Delta<1/\rho$。(iii) 是与引理 6 互补的上界：引理 6 由路程限制事件数，(iii) 由时钟步数限制。

## (d) 两个极端

- $\rho\gg1$：事件之间至少 $\lceil\rho\rceil$ 步，超调 $<\Delta/\rho$，相对阈值较小。引理严格支持的就是这两条。
- $\rho\le1$：$\lceil\rho\rceil=1$，(i)(iii) 不再约束，相邻时钟点可以连续触发；超调上界 $\delta_{\max}\ge\Delta$，不再小于阈值，时钟离散的影响可能较大。但 $\rho$ 只用了**最大**单步位移，$\rho\le1$ 本身推不出整个窗口被时钟主导（例：信号长期不变、最后一步跳 $2\Delta$，则 $\rho=1/2$ 而总共只发送两次）；是否时钟主导还取决于全窗单步位移的分布。极端情形 $\Delta\le\min_k|x_{k+1}-x_k|$ 时每一步都触发、$N=K$（归纳：参考总是前一点，每步差 $\ge\Delta$）；本信号最小单步位移 $8.3\times10^{-9}$，这个极端不会在实际 $\Delta$ 下出现。本例中能说的是：随着 $\rho$ 降低，(e) 表中的平均相对超调 $\bar o/\Delta$ 增大，时钟离散带来的超调影响增强。

## (e) 核数：$\rho$ 随 $N_0^{\rm nf}$ 的变化

对若干目标 $N_0^{\rm nf}$，按 8.3.2 的方法穷举得到 $\Delta_*$，并从事件序列读出最小间隔、平均间隔、平均超调：

| $N_0^{\rm nf}$ | $\Delta_*$ | $N^{\rm nf}$ | $\rho$ | $\lceil\rho\rceil$ | 实测 $\min g_\ell$ | $\bar g$ | $K/N$ | $\bar o/\Delta$ | 界 $1/\rho$ |
|---|---|---|---|---|---|---|---|---|---|
| 200 | $3.8462\times10^{-3}$ | 201 | 8.09 | 9 | 9 | 24.6 | 24.9 | 3.7% | 12% |
| **320** | $2.3847\times10^{-3}$ | 321 | 5.02 | 6 | 6 | 15.4 | 15.6 | 5.8% | 20% |
| 500 | $1.5021\times10^{-3}$ | 500 | 3.16 | 4 | 4 | 9.9 | 10.0 | 9.6% | 32% |
| 800 | $9.0450\times10^{-4}$ | 800 | 1.90 | 2 | 2 | 6.2 | 6.2 | 14.7% | 53% |
| 1200 | $5.6975\times10^{-4}$ | 1201 | 1.20 | 2 | 2 | 4.1 | 4.2 | 21.6% | 83% |

表中 $\Delta_*$ 为舍入显示，仿真使用穷举得到的全精度值（舍入值可改变计数，见 8.3.2(g)）。

引理 9(i) 在五个情形下都取等（实测最小间隔恰为 $\lceil\rho\rceil$）；(ii) 的界 $1/\rho$ 都成立，实际超调约为界的 $1/3$：越界时刻平均在半步附近而界按整步算（解释约 $1/2$），再加上事件触发步的位移通常小于全窗最大单步位移 $\delta_{\max}$。$N_0^{\rm nf}=320$ 时 $\Delta_*$ 是最大单步位移的 5 倍，事件之间至少 6 步、平均 $\bar g=15.4$ 步。

量级估计：忽略超调与折返，$\Delta_*\approx V_T/N_0^{\rm nf}$（引理 6 只给上界 $N-1\le V_T/\Delta$，这里把它当作近似等式），故 $\rho\approx V_T/(N_0^{\rm nf}\delta_{\max})=1765/N_0^{\rm nf}$；表中数值与此相符（$N_0=320$：估计 5.5，实际 5.0，差来自超调与折返）。

## (f) $N_0^{\rm nf}$ 的范围

**上限（时钟）。** 要求时钟不主导事件序列。按 (d)，$\rho$ 接近 1 时超调上界与阈值同量级；$\rho\ge3$ 时超调上界 $\le\Delta/3$、实际 $\lesssim10\%$。取 $\rho\ge3$ 为工作标准，对应 $N_0^{\rm nf}\lesssim500$。这是约定的量级界线，不是定理；它的依据是上表。

**下限（周期参考的条件数）。** 条件数直接对应的是参考 (B) 的实际样本数 $N=\lfloor B\rfloor$，它与 $N^{\rm nf}$ 同量级（8.3.6 将给出 $B\approx317$）。§8.2.5 已算：$J_{\rm per}(N)$ 在 $N=20$ 时条件数 $7.9\times10^3$，$N\ge40$ 后约 $10^2$。取 $N_0^{\rm nf}\gtrsim100$ 作为保守的工作下限，离病态区有余量；它不是条件数推出的必要下限。

**本文的 320** 在 $[100,500]$ 内：$\rho=5$，$N\approx300$。它仍是约定（8.3.2(a)），本小节只说明约定没有落在两端的坏区。

**与旧约定 $\Delta\le C_{\min}/100$ 的关系。** 本文 $\Delta_*/C_{\min}=0.363$，远大于 0.01。旧约定针对的是无噪事件估计器的有限 $\Delta$ 偏差；本文的信息来自全部时钟点的权重 $w_k$（§8.1.4 已证 $w_k>0$），$\Delta$ 由通信目标定义，不受该约定限制。第四音（$C_4=0.006566$）的峰峰值 $2C_4=5.5\Delta_*$，单独可以触发事件（只保留第四音、同一稳态相位与触发规则，得 99 次发送，含首次强制发送）。在多音叠加时，事件次数不能直接归属于某个频率；各频率参数的估计精度由信息矩阵 $J$ 及其 CRLB 评估。

## (g) 本小节的结论

1. 分辨裕量 $\rho=\Delta/\delta_{\max}$ 控制最小事件间隔 $\lceil\rho\rceil$ 和超调上界 $\Delta/\rho$（引理 9），本信号五个目标下 (i) 均取等。
2. 量级估计 $\rho\approx1765/N_0^{\rm nf}$；$\rho\ge3$ 的工作标准给 $N_0^{\rm nf}\lesssim500$；参考 (B) 的条件数给保守工作下限 $N_0^{\rm nf}\gtrsim100$。$[100,500]$ 是工作范围，不是定理。
3. $N_0^{\rm nf}=320$：$\rho=5.0$，最小间隔 6 步，平均间隔 $\bar g=15.4$ 步，$\bar o/\Delta=5.8\%$。


# 8.3.5 噪声水平 $\sigma$

## (a) 目的

8.3.1(g) 留下一个待定的绝对量，8.3.2 定了 $\Delta_*$，还剩 $\sigma$。本小节定 $\sigma$ 并说明理由。

## (b) 定法

$$\sigma:=\Delta_*/d_0,\qquad d_0=20,\qquad\sigma=1.1923\times10^{-4}.\tag{8.29}$$

理由：

1. $d_0=20$ 是 §8.1 原计划的三个比值 $\{5,20,50\}$ 的中间值；在它之下（$d=5$）噪声对事件序列的扰动已明显（8.3.6 附注：平均超调从 5.8% 升到 15.7%），在它之上（$d=50$）噪声接近可忽略，设计结论对噪声不敏感的可能性变大。取中间值，结论既不被噪声主导也不退化为无噪情形。
2. 噪声相对信号幅值：$C_{\min}/\sigma=0.006566/1.1923\times10^{-4}=55$，最弱音幅值是噪声标准差的 55 倍；$C_1/\sigma=441$。信号在所有音上都清楚高于噪声。
3. 时钟裕量 $\rho=\Delta_*/\delta_{\max}=5.0$ 与 $\sigma$ 无关（8.3.4），不受此选择影响。

## (c) 结论的范围

所有后续结论（预算、参考矩阵、候选排名）限于 $(\Delta_*,\sigma,\Delta_t)=(2.3847\times10^{-3},\ 1.1923\times10^{-4},\ 10^{-3})$。由 8.3.1，换 $\sigma$ 会改变 $w_k$ 和 $J$，候选排名可能随之变；这一点在结论中明确标注，不在本节内检验。

*附注（其他噪声水平）.* 8.3.6 的仿真同时跑了 $d\in\{5,50\}$（$\sigma=\Delta_*/d$），预算分别为 $304.29\pm0.07$ 和 $319.14\pm0.04$，作为将来稳健性检查的参考值保留，不进入本节结论。固定 $\Delta$、改变 $\sigma$ 是"同一传感器、不同噪声水平"的扫描；固定 $\sigma$、改变 $\Delta$ 是"同一噪声、不同阈值"的扫描；两者都可行，本节都不做。

# 8.3.6 发送预算 $B$ 与样本数 $N$

## (a) 目的

在 8.3.5 的参数下，均匀输入的期望发送数是预算 $B$：候选输入的发送数不得超过它，周期参考 (B) 的样本数由它取整。本小节定义 $B$ 的 MC 估计量与取整规则，说明噪声通过什么机制改变发送数，给出 10000 次 MC 的结果与一个对照，最后写出候选的合格判定规则。

## (b) 本小节用到的量

- $N_{\rm ev}^{(r)}$：第 $r$ 次噪声实现下 $u_{\rm unif}$ 的发送数（含首次）；$B:=\mathbb E[N_{\rm ev}(u_{\rm unif};\Delta_*,\sigma)]$。
- 估计量 $\widehat B=\frac1R\sum_rN_{\rm ev}^{(r)}$；样本标准差 $s$；标准误 $\mathrm{se}=s/\sqrt R$。
- 周期参考的样本数 $N:=\lfloor\widehat B\rfloor$。
- 带噪事件时刻 $j_0=1<j_1<\dots$；带噪超调 $o^z_\ell:=|z_{j_{\ell+1}}-z_{j_\ell}|-\Delta_*\ge0$（参考是带噪的发送值 $z_{j_\ell}$）；平均 $\bar o^z$。

## (c) 为什么必须用 MC，以及估计误差

$B$ 是对噪声的期望。$N_{\rm ev}$ 由触发链决定，链是带状态的非线性递推（每次发送的带噪值成为下一段的参考），没有闭式，所以 $B$ 只能用样本均值 $\widehat B$ 逼近。大数定律保证 $R\to\infty$ 时 $\widehat B\to B$；中心极限定理给 $\widehat B$ 的误差尺度 $\mathrm{se}=s/\sqrt R$。

取整 $N=\lfloor\widehat B\rfloor$ 是对 $\lfloor B\rfloor$ 的估计，当 $\widehat B$ 离整数不到几个 $\mathrm{se}$ 时可能取错。处理：报 $\widehat B\pm2\,\mathrm{se}$，要求该区间不含整数；含了就加大 $R$。

## (d) 仿真方法

1. 无噪输出 $x_k$（8.3.0(d)）与 $\Delta_*$、$\sigma$ 固定。
2. 用 `numpy.random.default_rng(20261006)` 一次抽出 $R\times K=10000\times5000$ 个标准正态数 $\xi_k^{(r)}$。
3. 对每个 $r$：$z_k=x_k+\sigma\xi_k^{(r)}$；在时钟上施行触发规则 (1.2)：$k=1$ 强制发送、$r_2=z_1$；$k\ge2$ 逐点比较 $|z_k-r_k|\ge\Delta_*$，触发则参考换为 $z_k$，否则不变。记录发送数 $N_{\rm ev}^{(r)}$ 和该次所有事件的超调平均 $\bar o^{z(r)}$。同时用更新前的参考 $r_k$ 算 $\alpha_k,\beta_k$，代 (5.10) 得 $c_k$，累加得 $\widehat w_k=\frac1R\sum_rc_k^{(r)}$（供后续参考矩阵用）。
4. 汇总：$\widehat B$、$s$、$\mathrm{se}$、$\widehat B\pm2\,\mathrm{se}$、$N$、单次最小最大值、$\bar o^z$。

数值成分只有四种：伪随机正态抽样；逐点触发递推（纯比较与赋值）；样本均值与标准误；$\Phi,\phi$ 的数值计算（`scipy.special.ndtr`，$P_k$ 在双精度下为 0 时令 $c_{{\rm no},k}=0$ 以免 $0/0$）。没有求解器、优化或积分近似。

## (e) 噪声改变发送数的机制

**带噪套叠恒等式。** 带噪事件 $j_{\ell+1}$ 的定义是 $|z_{j_{\ell+1}}-z_{j_\ell}|\ge\Delta_*$，即相邻发送值的跳变量为 $\Delta_*+o^z_\ell$。在一串同向上行事件 $j_0<\dots<j_n$ 中逐项相加：
$$z_{j_n}-z_{j_0}=\sum_{\ell=0}^{n-1}(z_{j_{\ell+1}}-z_{j_\ell})=n\Delta_*+\sum_{\ell=0}^{n-1}o^z_\ell.$$
左边 $=(x_{j_n}-x_{j_0})+(v_{j_n}-v_{j_0})$，移项：
$$x_{j_n}-x_{j_0}=n\Delta_*+\sum_{\ell=0}^{n-1}o^z_\ell-(v_{j_n}-v_{j_0}).\tag{8.30}$$
中间各次发送的噪声 $v_{j_1},\dots,v_{j_{n-1}}$ 在套叠中各加一次减一次，相消。记 $L:=x_{j_n}-x_{j_0}$（这段无噪上升，由信号定）和 $\bar o^z:=\frac1n\sum_\ell o^z_\ell$，则 $L=n(\Delta_*+\bar o^z)-(v_{j_n}-v_{j_0})$，
$$n=\frac{L+(v_{j_n}-v_{j_0})}{\Delta_*+\bar o^z}.$$
$L$、$\Delta_*$ 不随噪声变；$v_{j_n}-v_{j_0}$ 是两个噪声值，量级 $\sigma=\Delta_*/20$，不随 $n$ 增长。所以在一段固定的上升里，噪声改变 $n$ 的唯一通道是分母的 $\bar o^z$。特别地，不存在"每次提前触发省一点、累加起来"的效应——中间噪声已经相消。

**为什么 $\bar o^z$ 大于无噪的 $\bar o$。** 超调只在触发的那一步记录，且由定义 $\ge0$。设信号向上逼近阈值。噪声往上推：这一步提前触发，超调变小，但最低到 0——无噪超调平均只有 $0.058\Delta_*$（8.3.3），能降的空间就这么多。噪声往下拉：若拉到阈值以下，这一步**不触发**，不记录负超调，而是推迟到下一步；下一步信号又多走了一个时钟步（最大 $\delta_{\max}=\Delta_*/5$），超调变大。降有界、升无界，期望上升；噪声越大这两类情形越频繁，$\bar o^z$ 越大。一个可手算的例子：每步走 0.1、$\Delta=0.25$、噪声 $\pm0.06$ 等概率，无噪超调 0.05；带噪时在 0.2 处以概率 $1/2$ 提前触发（超调 0.01），否则在 0.3 处以概率 $1/2$ 触发（0.11），否则在 0.4 处触发（平均 0.15），期望超调 $0.5\times0.01+0.25\times0.11+0.25\times0.15=0.07>0.05$。

**预测。** $\bar o^z>\bar o$，故每事件消耗增大，$B<N^{\rm nf}=321$。这是机制预测，不是定理：(8.30) 只覆盖同向事件串；端点噪声 $v_{j_0},v_{j_n}$ 由触发过程选择，期望不为零；转向处的折返（引理 8）在带噪下也会变。结论只在 (f) 所测范围内陈述。

## (f) 结果

| 量 | 值 |
|---|---|
| $\widehat B$ | 316.87 |
| $\mathrm{se}$ | 0.022 |
| $s$ | 2.25 |
| $\widehat B\pm2\,\mathrm{se}$ | $[316.83,\ 316.92]$ |
| $N=\lfloor\widehat B\rfloor$ | 316 |
| 单次发送数范围 | 309–324 |
| $\bar o^z/\Delta_*$ | 7.65%（无噪 5.75%） |
| $\widehat B/N^{\rm nf}$ | 0.987 |

区间 $[316.83,316.92]$ 不含整数，取整 $N=316$ 在当前 MC 精度下稳定（$R=2000$ 的早期结果 $316.85\pm0.10$ 与此一致）。预测成立：$\bar o^z$ 高于无噪值，$\widehat B$ 低于 321 约 190 个标准误。单次发送数可以高于 321（最多 324），"低于 321"是平均意义的陈述。

跳变量比 $\eta:=(\Delta_*+\bar o)/(\Delta_*+\bar o^z)=0.982$ 与 $\widehat B/N^{\rm nf}=0.987$ 同向但不相等；超调、端点噪声、转向各自的贡献未分解，$\eta$ 不作预测公式。

*附注（其他噪声水平，同一批抽样按 $\sigma$ 缩放）.* $d=50$：$319.14\pm0.04$，$\bar o^z/\Delta_*=6.22\%$；$d=5$：$304.29\pm0.07$，$15.7\%$。噪声越大，超调越大、发送越少，与 (e) 一致。

**对照（纯噪声背景）。** 把输出换成常数 $x_k\equiv2$，同一 $(\Delta_*,\sigma)$、同一批抽样（前 2000 行）：平均发送数 $1.00$，即只有强制的首次，噪声单独不产生任何事件（$d=5$ 时也只有 $2.86\pm0.12$）。所以 321 → 317 的减少不是虚假事件，是跳变量变大。对照只衡量背景水平，不用于分解主实验的事件来源。

## (g) 候选输入的合格判定规则

对任一候选 $u$（$a_0=1$，$\sum a_m^2=1$），在 $(\Delta_*,\sigma)$ 下：

1. **约束**：$\mathbb E[N_{\rm ev}(u)]\le B$。这是共同上界，不是等发送数；每个候选报其实际 $\widehat{\mathbb E}[N_{\rm ev}(u)]$ 与标准误。
2. **判定统计量**：候选与均匀输入用同一批噪声，逐次相减
$$D^{(r)}:=N_{\rm ev}^{(r)}(u)-N_{\rm ev}^{(r)}(u_{\rm unif}),\qquad\widehat D=\frac1R\sum_rD^{(r)}=\widehat B_u-\widehat B,\qquad\mathrm{se}(\widehat D)=\frac{s_D}{\sqrt R}.$$
配对不改变点估计（$\widehat D$ 恒等于两均值之差），它的作用是正确计算差值的标准误：$\mathrm{Var}(N_u-N_{\rm unif})=\mathrm{Var}N_u+\mathrm{Var}N_{\rm unif}-2\,\mathrm{Cov}(N_u,N_{\rm unif})$，共同噪声下两者通常正相关，协方差项使差值方差小于各自方差之和；协方差符号不由共同随机数保证，故 $s_D$ 必须从 $D^{(r)}$ 实测。
3. **判定**：
$$\widehat D+2\,\mathrm{se}(\widehat D)<0\ \Rightarrow\ \text{合格};\qquad\widehat D-2\,\mathrm{se}(\widehat D)>0\ \Rightarrow\ \text{超预算};\qquad\text{其余}\ \Rightarrow\ \text{未定}.$$
未定者加大 $R$ 后再判，仍未定则保留并单独标注。$u_{\rm unif}$ 按定义合格（否则它恒有 $\widehat D=\mathrm{se}=0$，会被标为未定）。
4. **超预算候选**：从受预算约束的排序中剔除；在完整扫描结果中保留并标注实际发送数与超出量，用于回答"更多通信是否换来更多信息"。

# 8.3.7 权重和 $S$ 与两个参考矩阵

## (a) 目的

8.3.6 的仿真顺带得到了每个时钟点的权重估计 $\widehat w_k$。本小节用它构造参考 (C) $J_{\rm sod}(u_{\rm unif})$，用 $N=316$ 构造参考 (B) $J_{\rm per}(316)$，核验两者的秩与条件数，并把两者比一次——这是 (8.19) 中"基准输入下采样方案变化"那一项，也是 brief 第三问（事件采样对周期采样）在基准输入上的答案。权重和 $S$ 作为尺度量一并报告，并证明它在 $\sigma\to0$ 时的极限，以说明它不是事件数。

## (b) 本小节用到的量

- 权重和 $S:=\sum_{k=1}^Kw_k$；估计 $\widehat S=\frac1R\sum_r\sum_kc_k^{(r)}$。
- (C) $J_{\rm sod}(u_{\rm unif}):=\sigma^{-2}\sum_k\widehat w_kh_kh_k^\top$，$h_k$ 为 $u_{\rm unif}$ 的回归向量 (1.6)。
- (B) $J_{\rm per}(N):=\sigma^{-2}\sum_{i=1}^Nh^{\rm unif}(t_i)h^{\rm unif}(t_i)^\top$，$t_i=iT/N$（理想等间隔，允许离开触发时钟网格），$N=316$。
- 广义特征值 $\lambda_1\le\dots\le\lambda_9$：$J_{\rm sod}v=\lambda J_{\rm per}v$（§8.2.4）。$\Delta m_D=\log\det J_{\rm sod}-\log\det J_{\rm per}=\sum_i\log\lambda_i$；$m_E=\lambda_1$。
- 批标准误：把 10000 次分成 10 批各 1000 次，每批单独算 $\widehat S$、$\log\det J_{\rm sod}$、$\lambda_i$，报 10 个批值的标准误。

## (c) $S$ 不是事件数：$\sigma\to0$ 的极限

$S=\sum_k\mathbb E[c_{{\rm ev},k}]+\sum_k\mathbb E[c_{{\rm no},k}]$ 是按 (5.10) 的两项分解，不是按事件点/非事件点分解（$c_{\rm ev}$ 在非事件点也不为零）。

**引理 10.** 固定 $\Delta$，若无噪轨迹中没有 $|x_k-x_j|$ 恰等于 $\Delta$ 的比较，则 $\sigma\to0$ 时 $S\to N^{\rm nf}(\Delta)$。

*证明.* $\sigma\to0$ 时带噪事件序列趋于无噪序列，参考 $r_k\to x_j$（$j=j(k)$ 的无噪值）。(i) 无噪事件点：$|x_j-x_k|>\Delta$，则 $\alpha_k,\beta_k$ 同号且 $\to\pm\infty$。设都 $\to+\infty$：$P_k=\Phi(\beta_k)-\Phi(\alpha_k)\to0$，$\alpha_k\phi(\alpha_k)\to0$，$\beta_k\phi(\beta_k)\to0$，故 $c_{{\rm ev},k}\to1$；$c_{{\rm no},k}=(\phi(\alpha_k)-\phi(\beta_k))^2/P_k$，由 Mills 比 $P_k\ge\phi(\alpha_k)\big(\tfrac1{\alpha_k}-\tfrac1{\alpha_k^3}\big)\cdot c'$（$c'>0$ 常数，对足够大的 $\alpha_k$），得 $c_{{\rm no},k}\lesssim\alpha_k\phi(\alpha_k)\to0$。故 $c_k\to1$。(ii) 非事件点：$|x_j-x_k|<\Delta$，$\alpha_k\to-\infty$，$\beta_k\to+\infty$，$P_k\to1$，四个尾项都 $\to0$，$c_k\to0$。(iii) $c_1=1$ 对应 $j_0$。求和 $S\to N^{\rm nf}$。$\square$

两点说明。(1) 本文的 $\Delta_*$ 恰是一个等号比较：无噪链上 1330 是事件时刻、下一个事件是 1352，$|x_{1352}-x_{1330}|=\Delta_*$。引理的假设在 $\Delta_*$ 处不成立——该比较在噪声趋零时仍由噪声差的符号决定——故本文不据引理断言 $S\to321$。引理说明的是一般阈值下 $S$ 的零噪声极限是事件数而非别的量；有限噪声下 $S\ne B$ 由定义与 (d) 的数值直接说明。(2) 证明末尾由 $0\le c_k\le1$，用有界收敛定理交换极限与期望。

$\sigma>0$ 时 $S$ 与 $B$ 的差来自两处：$c_{{\rm no},k}$ 项（非事件观测的信息；在一侧带边 $(\alpha,\beta)=(-2d,0)$ 处 $c_{\rm no}=\phi(0)^2/\tfrac12=1/\pi\approx0.318$，总系数 $c\approx0.818$），以及 $c_{{\rm ev},k}$ 与事件概率之差（总和为正，逐点不保证符号）——$c_{{\rm ev},k}=\mathbb E[\xi_k^2\mathbf 1\{\epsilon_k=1\}\mid\mathcal F_{k-1}]$ 不等于 $q_k=\Pr(\epsilon_k=1\mid\mathcal F_{k-1})$，所以 $\sum_k\mathbb E[c_{{\rm ev},k}]\ne B$。$S$ 本身只是尺度，不决定九维精度——精度要看权重落在哪些方向，即 (e) 的 $\lambda_i$。

## (d) 结果：$S$ 与两个参考矩阵

| 量 | 值 | 误差 |
|---|---|---|
| $\widehat S$ | 465.96 | 批 se 0.12（CLT se 0.12 一致） |
| $\widehat S-\widehat B$ | 149.1 | 总信息权重与期望发送数之差 |
| $\widehat S_{\rm ev}=1+\frac1R\sum_r\sum_{k=2}^Kc^{(r)}_{{\rm ev},k}$，$\widehat S_{\rm no}=\frac1R\sum_r\sum_{k=2}^Kc^{(r)}_{{\rm no},k}$ | 353.0，112.9 | $\widehat S-\widehat B=(\widehat S_{\rm ev}-\widehat B)+\widehat S_{\rm no}=36.2+112.9$ |
| $\widehat S/K$ | 0.093 | 平均权重 |
| $J_{\rm per}(316)$：秩 / 条件数 / $\log\det$ | 9 / 98.7 / 194.155 | 确定性，无误差 |
| $J_{\rm sod}(u_{\rm unif})$：秩 / 条件数 / $\log\det$ | 9 / 94.9 / 197.197 | 批 se 0.003 |

秩核验：$J_{\rm per}(316)$ 满秩——$N=316\ge9$ 且 $e^{j\nu T/N}$ 互异（$\omega_{\max}T/316=0.195<\pi$），不在 8.3.5 附注提到的小 $N$ 病态点上；$J_{\rm sod}$ 满秩——$\mathrm{rank}\,H=9$ (8.1)，且全部 5000 个 $\widehat w_k$ 均为正（最小 $6.4\times10^{-50}$；93.28% 大于 $10^{-6}$，其余很小但为正），结合 $\mathrm{rank}\,H=9$ 得估计矩阵正定。

数值方法：$\log\det$ 与条件数由 `numpy.linalg.eigvalsh` 的特征值算（$\log\det=\sum\log\lambda$，避免直接行列式溢出）；广义特征值由 `scipy.linalg.eigh(J_sod, J_per)`。

## (e) (C) 对 (B)：九个广义特征值

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| $\lambda_i$ | 0.763 | 0.873 | 1.290 | 1.357 | 1.464 | 1.549 | 1.725 | 1.753 | 2.618 |
| 批 se | 0.0007 | 0.0007 | 0.0005 | 0.0006 | 0.0006 | 0.0007 | 0.0006 | 0.0005 | 0.0006 |

$$\Delta m_D=\sum_i\log\lambda_i=+3.041,\qquad m_E=\lambda_1=0.763.$$

核对：用未舍入值，$\log\det J_{\rm sod}-\log\det J_{\rm per}=197.196558-194.155294=3.041264$，与 $\sum\log\lambda_i$ 一致到 $10^{-12}$。

**解读（按 §8.2 的含义，不超出）。**
- $\Delta m_D=+3.04$：同一均匀输入、同样 316 个样本的量级，send-on-delta 的 $\log\det J$ 比理想周期采样高 3.04。$e^{-3.04/18}=0.845$：CRLB 椭球主轴的几何平均比周期采样小 15.5%（实际估计器是否达到由 §8.4 核验）。
- $m_E=0.763<1$：存在一个参数的线性组合，其 CRLB 方差在 send-on-delta 下是周期采样的 $1/0.763=1.31$ 倍。七个方向优于周期（$\lambda_i>1$），两个方向劣于（$\lambda_1,\lambda_2<1$）。达到该比值的线性组合系数是 $a=J_{\rm per}v_1$（$v_1$ 为 $\lambda_1$ 的广义特征向量；由 $J_{\rm sod}v_1=\lambda_1J_{\rm per}v_1$ 得 $a^\top J_{\rm sod}^{-1}a/a^\top J_{\rm per}^{-1}a=1/\lambda_1$；用 $v_1$ 本身作系数只得 0.853）。欧氏归一化后按参数顺序 $a=(-0.561,-0.008,0.487,0.092,-0.196,0.097,-0.405,-0.246,-0.409)$：主要混合 $G(0)$、$\mathrm{Im}\,G(j\omega_1)$、$\mathrm{Im}\,G(j\omega_3)$、$\mathrm{Im}\,G(j\omega_4)$，不对应单个参数。
- 两个准则方向相反，正是 §8.2.2 说的"$m_D$ 控制乘积、不保证最弱方向"的实例。
- 逐参数的 CRLB 比（$J^{-1}$ 对角元之比，send-on-delta / 周期）：$G(0)$ 0.70；$\omega_1$ 的 Re、Im 0.97、0.86；$\omega_2$ 0.70、0.70；$\omega_3$ 0.61、0.66；$\omega_4$ 0.64、0.69。九个参数单独看都优于周期采样，但最差线性组合劣于——$\lambda_{\min}$ 回答的是组合不是参数（§8.2.4）。

**范围。** 以上是基准输入 $u_{\rm unif}$ 在 $(\Delta_*,\sigma,\Delta_t)$ 下的结果；$J_{\rm sod}$ 是 MC 估计（批 se 已报），在 $\vartheta_0$ 处局部成立。它回答 brief 第三问在基准输入上的部分；候选输入的 $J$ 与 (B)(C) 的比较在第 3 步做。(8.19) 的第二项由此确定：$\log\det J_{\rm sod}(u_{\rm unif})-\log\det J_{\rm per}(316)=3.041$，候选相对 (B) 的 $\Delta m_D$ 等于相对 (C) 的 $\Delta m_D$ 加这个常数。

# 8.3.8 小结

| 量 | 值 | 来源 |
|---|---|---|
| $\Delta_t$，$K$，$T$ | $10^{-3}$，5000，5 | §8.1 |
| $\Delta_*$ | $0.002384694482635785$（$N^{\rm nf}=321$） | 8.3.2，穷举 |
| $\rho=\Delta_*/\delta_{\max}$ | 5.0 | 8.3.4 |
| $\sigma$ | $\Delta_*/20=1.1923\times10^{-4}$ | 8.3.5，约定 |
| $\widehat B\pm2\,\mathrm{se}$ | $316.87\pm0.04$（$R=10000$） | 8.3.6，MC |
| $N=\lfloor\widehat B\rfloor$ | 316 | 8.3.6 |
| $\widehat S$ | $465.96\pm0.25$ | 8.3.7 |
| $J_{\rm per}(316)$：秩、条件数、$\log\det$ | 9，98.7，194.155 | 8.3.7 |
| $J_{\rm sod}(u_{\rm unif})$：秩、条件数、$\log\det$ | 9，94.9，$197.197\pm0.006$ | 8.3.7 |
| $\Delta m_D^{\rm (sod\,vs\,per)}$，$m_E$ | $+3.04$，$0.763$ | 8.3.7 |

已证：事件过程与 $J$ 分别依赖 $(\Delta,\sigma)$（8.3.1）；$N^{\rm nf}$ 分段常数、最大可行阈值存在且为临界值、不单调（引理 4–6）；两侧界 (8.27)（引理 7、8）；时钟裕量控制最小间隔与超调（引理 9）。约定：$N_0^{\rm nf}=320$，$d_0=20$。所有后续结论限于这组 $(\Delta_*,\sigma,\Delta_t)$。下一步：§8.4 的 CRLB MC 核验（估计器是否达到 $J^{-1}$），然后第 3 步幅值/相位扫描。
