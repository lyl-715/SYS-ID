# Predictions and acceptance criteria — evaluator acceptance (§8.5, single scenario d = 20)

Committed before any run of the evaluator (working rule 1). The report cites this file's commit hash and marks
every prediction right / wrong / partly right after the run, wrong ones included.

Both blocks below are copied verbatim from the frozen spec `reports_CN/plan_evaluator_CN.md` v4.1
(bundle commit 4e84ebc): §8 = spec lines 174–193, §9 = spec lines 195–215 (extracted with `sed -n`, not retyped).
Configuration: `config.json` (step-0 commit 8e4d0cd).

---

# 8. 跑前预测

按工作规则 1，下面的预测写进 `predictions.md` 并在跑前提交，跑后逐条对照，错的也报。

| 编号 | 预测 | 判定方式 |
|---|---|---|
| P1 | v0 质量漂移约 0.2% @ $\sigma/12$，随 $h$ 线性减小；A、B 的漂移 $\le10^{-12}$（5000 步舍入量级） | 每步质量日志 |
| P2 | 收敛阶：v0 $\approx1$，A $\approx1$，B $\approx2$（绝对 $\log\det$ 与 $\mathbb E[N]$ 上） | $p_{\rm obs}$ |
| P3 | A 的 $\mathbb E[N]$ 偏高于 MC，偏差随 $h$ 减半；$\sigma/12$ 时与 v0 的 $+0.08$ 同量级 | $\mathbb E[N]$ 表 |
| P4 | B 在 $n=24$ 上，U 的 $\mathbb E[N]$、$\log\det$ 落在 MC 2 se 内；199 项中 $|z|>4.496$ 的为 0 | §7 判定表 |
| P5 | 用精确 $P$ 与用 $\hat P$ 算的 $\mathbb E[N]$ 之差：A 为 $O(h)$，B 为 $O(h^2)$，与 P2 一致 | 零成本诊断 |
| P6 | 尾部裕量 $8\Delta\to16\Delta$，所有量变化 $<10^{-8}$ | 尾部表 |
| P7 | 稳态耗时 $n=24\approx2\times$ $n=12$；numpy 实现 $n=12$ 约 7 s，numba 可再快 3–5 倍 | 耗时表 |
| P8 | 输入 N（无预算）的 $\mathbb E[N]$ 在 MC 的 2 se 内，与预算内三个输入无系统差别 | $\mathbb E[N]$ 表 |
| P9 | $\Delta m_D$ 的网格差 $d_2$ 小于绝对 $\log\det$ 的 $d_2$（误差相消），但两者的 $p_{\rm obs}$ 都接近 2 | 收敛表 |
| P10 | 四个输入的 $\mathrm{tr}(\bar D)/9$ 的 $|z|<2$，且四个不全同号；45 个上三角元素没有系统性的同号块 | §7 判定表 |
| P11 | 非线性偏差：$D>0$，且 $-D/19$ 的绝对值小于 $\log\det$ 的 se（0.003 量级） | 诊断表 |
| P12 | 三条数学不变量在 36 次评价中全部成立，$c_k$ 在尾部（$|\alpha|,|\beta|>30$）无 NaN、无负值 | 不变量日志 |

预测只写方向和量级，不写精确数字；跑完后在报告里逐条标"对/错/部分对"。

# 9. 验收标准（预注册）

三类标准加耗时线彼此独立，各自通过或不通过，不合并成一个总分。三类数值标准都通过即"数值验收通过"；耗时单独报告；四项都通过才写"评价器验收通过"，之后才用它复核候选。信息系数 `cfun` 已在 §8.4 的测试设置下获得 ML 仿真交叉验证（4000 次，标准差比 0.986–1.012）；本轮另检查评价器中的数值不变量和尾部计算，两者互补，都不单独证明 `cfun` 在所有 $(\alpha,\beta)$ 上正确。

| 类别 | 标准 | 通过条件 |
|---|---|---|
| 离散实现 | 非负性 | A、B 所有步 $\min m\ge0$；v0 只记录，不作通过项 |
| 离散实现 | 质量守恒 | A、B 每步 $|\sum m-1|\le10^{-10}$，并报实际最大漂移是否在舍入量级 |
| 离散实现 | 核截断 | 归一化前核质量与 1 的差 $\le10^{-12}$，因子记录 |
| 离散实现 | 触发约定 | 与 `mc83.sod` 逐条一致（$\ge\Delta$、首步强制、参考值取 $z$），写进 README |
| 离散实现 | 数学不变量 | $0\le c_k\le1$、$1\le S\le K$、$0\preceq J\preceq\sigma^{-2}\sum h_kh_k^{\top}$，A、B 的 24 次评价全部成立；v0 的 12 次只记录 |
| 数值收敛 | 最细两档 | 版本 B：$|\Delta m_D(12)-\Delta m_D(24)|\le0.002$，$|\log\det(12)-\log\det(24)|\le0.005$，$|\mathbb E[N](12)-\mathbb E[N](24)|\le0.02$，$|m_E(12)-m_E(24)|\le0.002$，$\|J_{24}^{-1/2}J_{12}J_{24}^{-1/2}-I\|_2\le0.005$，四个输入都满足 |
| 数值收敛 | 尾部 | $8\Delta\to16\Delta$ 所有量变化 $\le10^{-6}$ |
| 数值收敛 | 收敛阶 | 报 $p_{\rm obs}$，不设通过线；与 P2 对照 |
| MC 一致性 | 冻结清单 | 199 项（§7；$m_E$ 只展示），版本 B、$n=24$ 对 MC（$R=10000$，新种子，20 批配对）；点估计来自汇总矩阵 |
| MC 一致性 | 阈值 | 199 项全部 $|z|\le t_{19,\,1-0.05/404}=4.496$（按 202 项校正的近似 $t_{19}$ Bonferroni） |
| MC 一致性 | 符号诊断 | 报 $\mathrm{tr}(\bar D)/9$ 的符号与各输入对角元同号数；四个输入 $\mathrm{tr}(\bar D)/9$ 全同号且 $|z|>2$ 时标"可疑尺度偏差"，需解释后才算通过 |
| MC 一致性 | 非线性诊断 | 报 $D$ 与 $-D/19$；不设通过线 |
| 耗时 | 通过线 | 版本 B、受验网格，稳态单次 $\le60$ s（numpy）或 $\le15$ s（numba），硬件与实现跑前冻结；不通过时单独报告，不影响数值验收结论 |

不通过的处理见 §7 末段。通过后的措辞固定为："评价器在四个测试输入上与 MC 在其分辨率下未发现显著差异（199 项正式比较，阈值按 202 项校正的近似 $t_{19}$ Bonferroni）；网格 $\sigma/12$ 与 $\sigma/24$ 之间 $\Delta m_D$ 的变化不超过 0.002 nat、$m_E$ 不超过 0.002。"若某输入按 §6 切换到 $n=48$，措辞中的网格改为实际受验的 24/48，并注明是哪个输入。
