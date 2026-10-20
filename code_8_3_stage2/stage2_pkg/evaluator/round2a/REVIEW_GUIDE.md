# 复核指南 —— 第 2a 轮：七个候选复核、$S$ 的解析认证上界、离散 MDP 对偶估算值

方案：`reports_CN/plan_recheck_bounds_CN.md` v4.2（冻结，`6863d7c`）。报告：`evaluator/round2a/report_recheck_bounds_CN.md`。
本包与 GitHub `lyl-715/SYS-ID` 分支 `claude/optimistic-newton-xv5jvl` 的 `code_8_3_stage2/stage2_pkg/` 目录结构相同，**供阅读**。**复现需要完整的仓库检出，只有本包不够**；下面的命令都在完整检出中运行。

## 包内容（文件图）

| 路径 | 内容 |
|---|---|
| `REVIEW_GUIDE.md` | 本文件 |
| `reports_CN/plan_recheck_bounds_CN.md` | 方案 v4.2（预算相对均匀输入，$B:=\mathbb E[N](u_{\rm unif})$） |
| `reports_CN/global_bound_notes_CN.md` | 方案引用的《思考》（事实 A、B 与三层松弛），仅作背景 |
| **`evaluator/round2a/`（全部）** | 本轮的全部代码、配置、预测、日志、结果、图、报告 |
| `  predictions_recheck_bounds.md` | R1–R6、B1–B6 逐字摘自方案；含 v4.2 跑前更正说明（`2cfb3aa` → `809fb0d`） |
| `  make_config_recheck.py` → `config_recheck.json` | A0：七个候选（完整 $a_m,\chi_m$ 与来源记录号）、MC 值与配对 se、$\tau_N$、$\tau_E$、$B$ 及其来源行 |
| `  recheck7.py` → `recheck7_table.md`、`logs/recheck7.log`、`results/J_*.npy`、`results/recheck7.json` | A1–A2：七个候选 + U，用 `evaluator_fast.py`（B，$n=24$） |
| `  envelope_proof.py` → `logs/envelope_proof.log`（每段一行）、`results/envelope_segments.json` | B2：$c\le aq+b$ 的三段全域验证（中段 Arb 球算术） |
| `  mdp_estimate.py` → `logs/mdp_estimate.log`、`results/mdp_policy.npz` | B3/B4：一致性检查、黄金分割、策略前向（估算，不是界） |
| `  mdp_policy_plots.py` → `logs/mdp_policy.log`、`fig_mdp_policy.png`、`fig_mdp_occupancy.png` | 策略形状与占据分布 |
| `  b5_conversion.py` → `logs/b5_conversion.log` | B5：路 1 证书的浮点处理核实、$U^{\rm upper}$、情况 (2)、相对增益估算 |
| `  report_recheck_bounds_CN.md` | 报告 |
| `  build_review_round2a.py`、`REVIEW_GUIDE.md` | 本包的构建脚本与本文件 |
| `evaluator/log_excerpts/analysis_log_lines_197-213.txt` | `evaluator/logs/analysis.log` 第 197–213 行（$\tau_N$、$\tau_E$ 的来源，原样、保留行号） |
| `evaluator/log_excerpts/run_all_log_line_1021.txt` | `evaluator/logs/run_all.log` 第 1021 行（$B=316.8921168493721$） |
| `evaluator/log_excerpts/speedup_regression_log_line_65.txt` | `evaluator/logs/speedup_regression.log` 第 65 行（快速评价器逐位复现 $B$） |
| 未改的依赖（附上供阅读；重跑用完整仓库） | `mc83.py`（`cfun`）、`mc85_scan.py`、`stage2_common.py`、`stage2_local.py`、`wk_d20_10000.npy`；`evaluator/density_recursion_v1.py`、`evaluator_fast.py`、`config.json`、`config_fast.json`、`results/rec_B_U_n24_m8.json`；路 1 的 `bound/bound_common.py`、`bound_report.md`、`bound_inner.log` |
| `config_recheck.json` 的来源 | `stage2/final_batch1.json`、`stage2/stage2_final.log`（第 16–22 行）、`stage2/stage2_report.md`（第 228 行，R4 的余量）、`stage3/unconstrained.json`、`stage3/stage3_unconstrained.log`（第 31–33 行） |

## 环境

Python 3.13.16，numpy 2.5.3，scipy 1.18.1，numba 0.68.0，**python-flint 0.9.0（FLINT 3.6.0）**，matplotlib（只用于图）。
安装：`pip install numpy scipy numba python-flint matplotlib`。路 1 求解器只用到 numpy/numba（cvxpy 不需要）。

## 一条命令复现包络证明

在完整仓库检出的 `code_8_3_stage2/stage2_pkg/evaluator/round2a/` 下：

    python3 envelope_proof.py | grep -E "^ENV \(ii\) segments|VERDICT"

应输出（不到 1 秒）：

    ENV (ii) segments 1000; total bisections 4126; uncertified pieces 0; worst (largest) leaf upper endpoint -1.05263937285e-10 in segment [18.46, 18.47]; deepest bisection level 13 (limit 13); narrowest piece 1/819200 = 1.2207e-06 (>= 1e-6: True); time ... s
    ENV VERDICT: PASS: c <= a q + b certified for all real delta; hence for the true budget B, S <= 1 + a (B - 1) + 4999 b (解析认证的上界); under the additional, uncertified premise B <= 317 this gives S <= 1347.119829 < 1348

它会改写 `logs/envelope_proof.log`；除耗时外应与仓库中的原日志逐行相同（见下节的对比命令）。

## 每个数怎么重算

先克隆仓库，保留原日志作参照：

    git clone -b claude/optimistic-newton-xv5jvl https://github.com/lyl-715/SYS-ID.git
    cd SYS-ID/code_8_3_stage2/stage2_pkg/evaluator/round2a && cp -r logs logs.ref

| 数（报告位置） | 日志行 | 命令（在 `evaluator/round2a/` 下） | 耗时 |
|---|---|---|---|
| $\tau_N=0.04$、$\tau_E=0.0003$、$B=316.8921$（§1） | `../log_excerpts/*`；`config_recheck.json` | `python3 make_config_recheck.py`，或下面的只读核对命令 | 即时 |
| 七行表：$\Delta m_D$、$z$、$\mathbb E[N]$、可行性、$m_E$、$\lambda_0$ 判定、$S$（§2） | `logs/recheck7.log` 的 `A1`、`A2` 行 | `python3 recheck7.py` | 约 20 s |
| 包络的三段验证、4126 次二分、最差上端 $-1.05\times10^{-10}$；$\overline S=1347.119829$ 是 $1+316a+4999b$ 的算术，以未认证前提 $B\le317$ 为条件（§3） | `logs/envelope_proof.log` 的 `ENV`、`SEG`、`XCHECK` 行 | `python3 envelope_proof.py` | < 1 s |
| B4 一致性（$3.7\times10^{-15}$、$2.3\times10^{-15}$）与诊断差（§4） | `logs/mdp_estimate.log` 的 `B4` 行 | `python3 mdp_estimate.py` | 约 10 min |
| $\widetilde S_{\rm dual}=1210.956$、$\lambda^\*=1.8335$、策略前向 $S=1210.798$、$\mathbb E[N]=316.806$（§4） | 同上 `GOLD`、`MDP`、`POLICY` 行 | 同上 | — |
| 策略形状（$\pm18.625\sigma$、74.7%）与两张图（§4） | `logs/mdp_policy.log` 的 `SHAPE` 行 | `python3 mdp_policy_plots.py`（在 `mdp_estimate.py` 之后） | 约 1 min |
| 浮点核实、情况 (2)、数值证书值 $U_{\rm cp}\approx207.928576$（$W=1347.119829$ 带前提 $B\le317$）、估算 10.732 与 9.781 nat（§5） | `logs/b5_conversion.log` 的 `B5` 行 | `python3 b5_conversion.py`（在 `mdp_estimate.py` 之后） | 约 15 s |

$\tau$ 与 $B$ 的只读核对命令（在完整检出的 `code_8_3_stage2/stage2_pkg/` 下；包内的 `log_excerpts/` 是同几行的摘录，供阅读）：

    python3 - <<'EOF'
    import json
    from decimal import Decimal, ROUND_CEILING
    def up1(x):
        d = Decimal(repr(x)); e = d.adjusted(); return float(d.scaleb(-e).to_integral_value(rounding=ROUND_CEILING).scaleb(e))
    L = [l.split() for l in open('evaluator/logs/analysis.log').read().splitlines()[196:213] if l.startswith('STAB')]
    EN = [float(x[4]) for x in L if x[3] == 'EN']; ME = [float(x[4]) for x in L if x[3] == 'mE']
    B = float(open('evaluator/logs/run_all.log').read().splitlines()[1020].split()[-1])
    c = json.load(open('evaluator/round2a/config_recheck.json'))
    print('tau_N', up1(10*max(EN)), c['tau_N']['value'], '| tau_E', up1(10*max(ME)), c['tau_E']['value'], '| B', B, c['budget_B']['value'])
    EOF

应输出 `tau_N 0.04 0.04 | tau_E 0.0003 0.0003 | B 316.8921168493721 316.8921168493721`。

日志对比（耗时字段屏蔽；所有数都是确定性的，应逐行相同）：

    for f in recheck7 envelope_proof mdp_estimate mdp_policy b5_conversion; do
      diff <(sed -E 's/[0-9.]+ s\b/X s/g' logs.ref/$f.log) <(sed -E 's/[0-9.]+ s\b/X s/g' logs/$f.log) >/dev/null && echo "$f identical" || echo "$f DIFFERS"; done

## 数的追溯

报告每个数取自 `evaluator/round2a/logs/` 的一行：`A1`/`A2`（recheck7.log），`ENV`/`SEG`/`XCHECK`（envelope_proof.log），`B4`/`GOLD`/`MDP`/`POLICY`（mdp_estimate.log），`SHAPE`（mdp_policy.log），`B5`（b5_conversion.log）；$\tau$、$B$ 的来源行在 `log_excerpts/`。表 A 由 `recheck7.py` 生成后原样插入报告。例外：§4 中 $k=1250$ 时 284 个中间速度节点（$0<|v^\*|<110h$）出自审核，不在日志中，可由 `results/mdp_policy.npz` 的 `pols[1250]` 数出。

## 提交与时间线（核"跑前提交"）

| 提交 | 内容 |
|---|---|
| `aab99e6` | 方案 v4.1 与《思考》 |
| `2cfb3aa` | 预测（v4.1 文本），任何运行之前 |
| `cc214d6` | A0：冻结 config（当时为绝对预算 316） |
| `6863d7c` | 方案 v4.2：用户在任何评价运行之前更正预算定义（相对均匀输入） |
| `809fb0d` | 预测文件与 config 的跑前修订（v4.2 文本，注明是方案更正而非预测结果） |
| `fe45de9` | A1–A2 运行与表（第一次评价运行） |
| `73d0428`、`db9e741`、`579fe44` | B2、B3/B4、B5 |
| `92b2d18` | 报告 |
| 本包所在的提交 | 审核后修订，无新运行、数值未变：$S$ 结论改为条件形式（$B\le317$ 为未认证前提）、B1 分两部分判定、B5 第 (2) 种情况措辞、策略形状措辞、复现须用完整仓库；`envelope_proof.log` 的 VERDICT 行与 `b5_conversion.log` 的说明文字随脚本输出文字同步修改 |

## 建议重点核的地方

1. **预算更正的时序。** v4.1 的预测（`2cfb3aa`）与 A0（`cc214d6`）在前；v4.2（`6863d7c`）与修订（`809fb0d`）都在第一次评价运行（`fe45de9`）之前。预测文件开头的说明是否把这一更正写清楚，没有把它当成预测结果。
2. **七个候选的选取与 U-L1657 的 $z$ 列。** 方案原文"阶段 4 冻结池中的七个最终候选"与"至少包含 U-L1657"不能同时成立；由用户决定为阶段 2 的最终候选去掉 UNIF、加 U-L1657。U-L1657 没有 R = 4000 配对 se，$z$ 列"未定"。
3. **包络证明（`envelope_proof.py`）。**
   - 区间库：`mpmath.iv` 在本环境没有可用的 `erf/erfc/ncdf`，改用 python-flint 的 Arb 球算术（方案允许"或等价"）；端点为精确有理数，片段是两端点球的并。
   - 中段被求值的式子：$g=(1-a)q-\alpha\varphi(\alpha)+\beta\varphi(\beta)+(\varphi(\alpha)-\varphi(\beta))^2/P-b$，即 $c_{\rm ev}+c_{\rm no}-aq-b$ 只把两处 $q$ 合并；平方写成乘积（Arb 的 `**2` 对含 0 的球给 NaN）。没有用中心形式或导数。
   - 余量很小：最差片段上端 $-1.05\times10^{-10}$，在 $t=\delta/\sigma\in[18.46,18.47]$，二分到最细一层（宽 $1/819200\ge10^{-6}$）。
   - 第 (i)、(iii) 段的解析论证，特别是 $c\le1$ 的 Cauchy–Schwarz 证明（脚本开头），以及偶对称与 $\Delta/\sigma=20$ 的单位换算。
   - 认证范围：认证的是 $c\le aq+b$（全体实数 $\delta$）；由此对真实预算 $B$，$S\le1+a(B-1)+4999b$。$\overline S=1+316a+4999b=1347.119829$ 另需前提 $B\le317$，该前提未认证（$B=316.892$ 是评价器的数值）；报告把 B1 分两部分判定。
4. **离散 MDP（`mdp_estimate.py`）。** 发送分支的屏蔽和用左前缀和加右后缀和（与"总和减窗口"同一个量；后者在 $\delta\approx0$ 处相消为 0）；$P_i:=1-q_i$；平局取 $|v|$ 最小；措辞只用"离散 MDP 对偶估算值"；B4 的相对 $10^{-10}$ 检查先于估算。策略前向的 $\mathbb E[N]=316.806$ 满足 $B$，但比 316 多 0.806（方案 B3 预测的比较对象仍是 316）。
5. **B5 的浮点核实。** `bound/bound_common.py` 第 77、84、96 行的计算是否确实没有任何浮点保守处理，从而落在第 (2) 种情况；207.928576 的措辞是否始终是"数值证书，未作浮点保守处理"，以及它所用的 $W=1347.119829$ 带未认证前提 $B\le317$ 是否处处写明。
6. **预测判定。** B1（分两部分：包络认证通过；预算前提未认证）、R1（六个对、一个未定）、R3/R4（错；阶段 2 报告的"余量"是配对差减两倍 se，不是评价器上与 $B$ 的距离；R4 写"阶段 3"，但 0.07%–0.09% 的余量出自阶段 2 报告第 228 行）、B3/B6（部分对）。

## 已知的方案外选择（报告 §7 已列）

Arb 代替 `mpmath.iv`；中段式子合并两处 $q$、平方写成乘积；MDP 的前缀加后缀和、$P_i=1-q_i$、平局规则；197.197 按方案原文；七个候选与 U-L1657 $z$ 列按用户决定。
