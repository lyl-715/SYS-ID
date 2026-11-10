---
title: "第 13 轮 Part 0 报告：其他输入的基础设施与回归闸门"
date: "2026-11-10"
---

**Prompt。** `reports_CN/prompt_round13_part0_infrastructure_EN.md`（配合 `reports_CN/README_round13.md`）。分支 `round13-input-trigger` 自 `round11-edge-actions` 的 `1b88f01` 开出；`claude/optimistic-newton-xv5jvl`（`3314e7d`）、`round10-upper-bound`（`2b1af1b`）、`round11-edge-actions`（`1b88f01`）均未改动。
- `c65c7cb`：本包四个 `.md` 入 `reports_CN/`（本轮第一个提交）；
- `ff3debf`：代码、闸门结果与日志（`round13/`）；
- 本报告：其后的提交。

**口径。** 只做 Part 0；Part A 未开始。所有 I 为名义真值 θ₀ 处的 `log det J`（nat），未注明 σ/120 者均在 σ/24 设计网格上。所有输入的预算均为 B = 316.8921168493721。

**来源。** 路径相对 `code_8_3_stage2/stage2_pkg/evaluator/`。
- `[L X:n]`：`round13/logs/gate_X.log` 第 n 行；`[L sum:n]`：`round13/logs/gate_summary.log` 第 n 行。
- `[J a.b]`：`round13/results/gate13.json` 的字段 `a.b`。
- 代码：`round13/eval13.py`、`round13/dp13.py`（`round11/eval11.py`、`round11/dp11.py` 的副本，完整差异见 `round13/results/diff_eval11_eval13.txt`、`diff_dp11_dp13.txt`）；`round13/common13.py`（辅助函数）；`round13/run13.py`（驱动）；`round13/inputs13.json`。
- U 设计的扫描记录：`round13/results/gate_U_design/scan_edge_j0.json`、`scan_edge_j1.json`；设计汇总：`design_edge.json`；J 矩阵：`J_*.npy`。
- 环境：Python 3.13.18、numpy 2.5.3、scipy 1.18.1、numba 0.68.0（与第 11 轮相同，见 `round11/results/state_inputs.json`）；4 vCPU、16 GB 内存。

## 1. 结论

**闸门 G0–G4 全部通过，Part A 可以开始。** 对输入 U，第 13 轮副本逐位复现第 11 轮：模块网格、两次扫描的全部 24 次 DP、T₀ 与 T₁ 的编码、二者在 σ/24 与 σ/120 上的 I 与 E[N]，以及证书 UB = 207.56048558735236。六个输入登记后的 G 与 I_full 与事先给出的识别值一致。
来源：[L sum:2–7]；[J gates.\*, all_pass]。

## 2. 闸门（每项一行）

| 闸门 | 比较内容 | 最大偏差 | 结果 |
|---|---|---|---|
| G0(a) | 同一进程中 `eval13` 对 `eval11`：XK、HK、Z、KA、KB、FL、CL（dtype、形状、字节），以及 DZ、Z0、G、X_PAD、NAME、B、s | 无（全部逐位相等） | 通过 [L U:4] |
| G0(b) | `design(edge_set(), gamma_from_J, logdet9, 9, 'edge')` 对 `round11/results/scan_edge_j0.json`、`scan_edge_j1.json` 逐次比较：lam、EN、S、F、dsum、dual、logdet、feasible、phase；另比较 gbar、gamma1、final_run、min_dual、lam_bracket、次数 | 无：12/12 与 12/12 次逐位相等；final_run 10 与 11；gbar/gamma1 为 0.021587304272182303/0.02987366782571452（j0）、0.006849709430879255/0.009928291424921748（j1） | 通过 [L U:8, 21, 23, 36, 40–41] |
| G0(c) | T₀、T₁ 编码对 `round11/results/table_edge_A0.npz`、`table_edge_iter.npz`（4999 × 35146，uint16）；I、EN24、I120、EN120 对 `round11.json → tables.edge_j0 / edge_j1`（容差相对 1e-12） | 编码逐位相等；8 个数值相对偏差均为 0（T₀：207.10197960571716 / 316.61009947208913 / 207.0999194594685 / 316.413413652473；T₁：207.35521730257756 / 316.7261244018059 / 207.35308449872582 / 316.5159931346839） | 通过 [L U:42–43] |
| G0(d) | 重算的 UB = I(T₀) − 9 + min dual(j1) 对由存档字段算出的值 | 207.56048558735236 对 207.56048558735236，差 0 | 通过 [L U:44] |
| G1 | R13_INPUT = P5514 时 `eval13.XK, HK` 对 `dr.input_arrays('P5514')`（由 `evaluator_window` 登记）与 `dr.input_arrays('D')`（config.json）；R13_INPUT = N 时对 `dr.input_arrays('N')` | 无（逐位） | 通过 [L P5514:4]、[L N:4] |
| G2 σ/24 | `eval13.evaluate(ConstRows(480))` 对已接受算子（`des.sod_reference()` → `evaluator_fast`）及 prompt 给出的数值（容差相对 1e-9） | E[N] 316.8921168493916 对 316.8921168493721（相对 6.2e-14）；logdet 197.20097668224133 对 197.20097668224076（相对 2.9e-15） | 通过 [L U:6] |
| G2 σ/120 | `eval13.evaluate_refined(ConstRows(480), 5)` 对 `round7/results/archive_match.jsonl` 中的条目（θ₀, M = 480, f = 5）；该条目存在（容差相对 1e-9） | E[N] 316.8911511746132 对 316.8911511745014（相对 3.5e-13）；logdet 197.20115853002983 对 197.2011585300231（相对 3.4e-14） | 通过 [L U:7] |
| G3 | LO：`fine_grid(5)` 不触发 BoundaryNode；`zf[0] == Z0`；每个细节点的粗格单元在 [0, G−1] 内；ConstRows(480) 在 σ/24、σ/120 上质量守恒 ≤ 1e-12 / ≤ 1e-10 | zf[0] = Z0 = 1.8994231576096559（若用 U 的默认原点则为 1.9098150805683891）；单元 [0, 40014] = [0, G−1]；离单元边界最近 0.1 个单元宽；质量偏差 8.0e-14（σ/24）、2.5e-13（σ/120） | 通过 [L LO:4–5] |
| G4 | 六个输入：Σa²、输出范围、单步最大位移、G、I_full；I_full 容差 1e-4，G 须完全相等，单步 ≤ 4.6σ，abs(Σa² − 1) ≤ 1e-12 | abs(I_full − 参考) 最大 2.2e-5（P5514）；六个 G 全等；最大单步 4.28σ（LO） | 通过 [L X:3] |

G3 中 LO 在 ConstRows(480) 下的数值：σ/24 E[N] = 367.814772611902，logdet = 197.35269715259415；σ/120 E[N] = 367.8138573050662，logdet = 197.3528664358578。诊断：LO 进程还在 Δ\* 处跑了已接受算子，得 E[N] 367.814772611879、logdet 197.35269715259363，与 eval13 的相对差为 6.3e-14 / 2.6e-15 [L LO:7]。

G4 表 [J g4_table.\*]（I_full 单位 nat；范围与单步单位 σ；参考值为 README 中的识别值）：

| 输入 | Σa² − 1 | 输出范围 | 最大单步 | G | I_full | 参考 I_full | abs(差) | G = 参考 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| U | 0 | 1144.356 | 3.988 | 35146 | 219.008506 | 219.0085 | 5.9e-6 | 是 |
| P5514 | −1.1e-16 | 1136.209 | 3.176 | 34951 | 219.007978 | 219.0080 | 2.2e-5 | 是 |
| SCH | 0 | 1121.426 | 3.028 | 34596 | 219.008506 | 219.0085 | 5.9e-6 | 是 |
| LO | −1.1e-16 | 1347.243 | 4.282 | 40015 | 218.034288 | 218.0343 | 1.2e-5 | 是 |
| HI | −1.1e-16 | 869.106 | 3.436 | 28540 | 218.034288 | 218.0343 | 1.2e-5 | 是 |
| N | 0 | 1482.514 | 4.039 | 43262 | 218.416907 | 218.4169 | 6.7e-6 | 是 |

六个输入的范围与单步在 README 所印位数（1 位、2 位小数）上也一致。SCH 相位：(0, 4.71238898038469, 1.5707963267948966, 3.141592653589793) = (0, 3π/2, π/2, π)。

U 的设计即 G0(b)–(d) 那次运行，与第 11 轮完全相同，也就是 Part A 中 U 的条目。T\* = T₁（edge_j1）；I = 207.35521730257756（σ/24）、207.35308449872582（σ/120）；σ/120 上 E[N] = 316.5159931346839 ≤ B。UB′（B + 0.25）= 207.56525182437946，故缺口 UB′ − I(T\*) = **0.2100 nat**（≤ 0.5，不跑第三次扫描）。前向/后向检查为 5.2e-13（T₀）与 4.3e-13（T₁）；两张表均满足 I(T) ≤ UB′。
来源：[L U:22, 37–39]。

## 3. 运行时间与内存

所有运行均为 4 个 numba 线程、1 个 BLAS/OMP 线程。U 与 LO 各自单独运行：四个短闸门（SCH、P5514、N、HI）已先结束；LO 于 12:18:20 结束，U 于 12:18:28 开始。

| | U（G = 35146） | LO（G = 40015） |
|---|---:|---|
| 一次 DP（边动作集，100 个动作） | 24 次中位数 41.2 s（范围 37.6–49.5 s） | 42.9 s（一次，λ = 5γ̄，切点为 LO 的 SoD） |
| 一次 σ/24 前向，DP 表 | 扫描中 24 次中位数 19.2 s；带形状统计时 T₀ / T₁ 为 28.4 / 29.3 s | 22.5 s |
| 一次 σ/24 前向，常数 SoD（ConstRows(480)） | 4.6 s | 5.1 s |
| 一次 σ/120 前向，DP 表 | 105.8 s（T₀）、110.2 s（T₁）；G_f = 175724 | 未跑 |
| 一次 σ/120 前向，常数 SoD | 24.9 s | 26.8 s（G_f = 200071） |
| 固定策略后向，σ/24 | 15.7 / 17.4 s（T₀ / T₁） | 未跑 |
| 一次扫描（12 次 DP + 12 次前向） | 710 s（j0）、765 s（j1） | — |
| 整个设计（两次扫描及 T₀/T₁ 的评价） | 1820 s；闸门进程共 1856 s | — |

来源：[J per_input.U.timing, per_input.LO.timing, per_input.LO.G3]；[L U:22, 37]；[L LO:5, 8]。

**U 闸门进程的内存峰值：9.1 GB**（VmHWM 9 095 468 kB，从进程外每 5 s 读取一次；`round13/logs/peak_rss_U_gate.txt`）。prompt 预计每个进程约 1.5 GB（见 §4 第 1 条）。另用探针在 T₁ 上测了三个峰值（`round13/mem_probe.py`，输出 `round13/logs/mem_probe.log`）：
- 一次 σ/24 前向加一次 σ/120 前向，不带 `Stats`：0.55 GB；
- 一次 σ/24 前向，带 `eval13.Stats`：2.7 GB（收集 1.34 × 10⁸ 个节点值）；
- `Stats.result()` 之后：7.9 GB。

## 4. prompt 与 1b88f01 代码的不一致（只报告，不绕开）

1. **每进程内存。** README 与 Part 0 §2.4 写每个进程约 1.5 GB，因此可四个并行。U 闸门进程实测峰值 9.1 GB（§3）。§3 的探针表明峰值出现在 `eval13.Stats`（原样复制自 `round11/eval11.py`）：不带它 0.55 GB，填充它的前向后 2.7 GB，`Stats.result()` 后 7.9 GB。`Stats` 保存全部 4999 步每个节点的迎面边偏移与质量（U 的 T₁ 共 1.34 × 10⁸ 个值，每个 float64 数组约 1.1 GB），`result()` 再对它们做拼接、排序与累加。`design()` 对每张扫描表各跑一次 `Stats`，与第 11 轮 `post11.analyse` 相同。在这台 16 GB 机器上，四个这样的进程无法并行。Part 0 对此不作改动。Part A 需要你先定：(a) `Stats` 运行时最多一个设计进程（或各输入依次运行）；(b) 只对 T\* 算 `Stats`；(c) 改用更省内存的统计实现，但那样就不再是 `eval11.Stats` 的副本。
2. **G1 按构造成立。** `eval13.XK, HK` 由 `ew.design_grid(NAME)` → `dr.input_arrays(NAME)` 生成，再与 `dr.input_arrays(NAME)` 比较等于函数与自身比较。G1 中独立的部分是 P5514 对 `dr.input_arrays('D')`：后者用的是 config.json 里同一组振幅与相位的副本，结果逐位相等。
3. **G2 σ/120 比较的是两条代码路径。** 存档条目（θ₀, M = 480, f = 5）由第 7 轮的 *reuse* 路径算出，即已接受算子 `evaluator_fast` 在 n = 120、68σ 支撑窗下的结果（`"route": "reuse"`）。`eval13.evaluate_refined` 则是第 11 轮的全网格一般路径，核窗 88σ。两者相对差 3.5e-13。
4. **`inputs13.json` 原文。** `PAR` 的 `"p"` 是字符串（"filled by Part B …"）。按 §2.2 第 1 条"凡有 `p` 或 `a` 的条目都登记"会尝试登记它，故 Part B 之前跳过（DECISION 13-2）。`SCH` 只有 `chi_rule`、没有 `chi`，相位按规则计算（DECISION 13-3）。
5. 运行时间指南（不算矛盾）：DP 实测 41–43 s（指南 45–60 s）；DP 表的 σ/24 前向 19–22 s（指南 20–25 s）；常数 SoD 前向只要约 5 s；DP 表的 σ/120 约 110 s。
6. "import round4 可能生成空文件夹"：本次会话中没有生成。每个进程结束后，round4–round11、config.json、density_recursion_v1.py、evaluator_fast.py 的 `git status` 均干净 [L X:末行]。

prompt 中关于代码的其余陈述均已对照代码核实无误。核实范围包括 eval11 第 43 / 56 / 329 行、evaluator_window 第 56–58 行、`dual = g1 + dsum + lam (B − 1)`、X_PAD = 960、`_step_full` 遇任一越界边即停，以及全部存档参考值。

## 5. DECISION 列表

- **DECISION 13-1.** 除 §2.2 的改动 (1)–(4) 外，`eval13.py` 加了 `import json`（改动 1 需要）和 docstring 开头的出处说明；`dp13.py` 只加了 docstring 出处说明。差异存档：`round13/results/diff_eval11_eval13.txt`、`diff_dp11_dp13.txt`。
- **DECISION 13-2.** 登记规则：已在 `dr.CFG['inputs']` 中的条目不重新登记（U、N 来自 config.json，P5514 来自 evaluator_window）。若这类条目列出数值 `a`/`chi`（U），则断言其与已登记值相等。只有 `a` 或 `p` 为数值 4 维向量的条目才登记，故 Part 0 中跳过 PAR。振幅 a = sqrt(p)，float64。
- **DECISION 13-3.** SCH 相位按 `chi_rule` 计算：`np.mod(-π m (m−1)/4, 2π)`，m = 1..4，float64；断言其与 (0, 3π/2, π/2, π) 的绝对差 ≤ 1e-15。
- **DECISION 13-4.** abs(Σa² − 1) ≤ 1e-12 的检查用于 `dr.CFG['inputs']` 中全部输入（U、D、E、N、P5514、SCH、LO、HI），不只用于新登记的输入。
- **DECISION 13-5.** §2.2 的 KA/KB 范围断言放在 `eval13` 模块层（网格之后），`Design11.__init__` 中的同一断言保留。
- **DECISION 13-6.** `matched_sod` 并列规则：σ/24 上两个夹住 B 的 M 若 abs(E[N] − B) 相同，取较大的 M（沿用 prompt 给 σ/120 的规则）。M\* 若未在 σ/24 上算过则补算。二分按 E[N] − B 的符号保持夹区间。
- **DECISION 13-7.** `design()` 对每次扫描的最终表（T₀、T₁，以及跑了的话 T₂）都在 σ/24 与 σ/120 上评价，并各做一次固定策略后向检查。形状统计 `Stats` 用扫描 j0 的 γ（切点为 SoD，沿用 `round11/post11.analyse` 的做法）。准则并列时 T\* 取较早的表。写完 json 之后，以下任一情况即停止运行：T\* 的 σ/120 E[N] > B、某表准则值 > UB′、前向/后向偏差 > 1e-12。
- **DECISION 13-8.** 扫描断点续跑：只有当状态 json 有最终表、其切点（gbar, γ₁）与重算值逐位相等、且最终表的 `.npy` 仍在时，才跳过该扫描；否则重跑。
- **DECISION 13-9.** DP 表放在 `$ROUND13_WORK/<输入>/<标签>/`。不同输入的标签相同，故按输入分目录。本次会话 `ROUND13_WORK` 设为会话 scratchpad（不入库），因此扫描 json 中的 `table` 路径指向那里。
- **DECISION 13-10.** `gate` 阶段按输入各占一个进程（U：G0、G2、G4；P5514、N：G1、G4；LO：G3、DP 计时、G4；SCH、HI：G4）。另设 `gate_summary` 阶段把六个文件汇总为 `gate13.json`。`partA`、`partB` 阶段在本提交中拒绝运行。
- **DECISION 13-11.** 线程：每进程 4 个 numba 线程（`R13_NUMBA_THREADS`，与第 11 轮驱动相同）；OMP/BLAS 为 1。
- **DECISION 13-12.** G4 通过规则：abs(I_full − 参考) ≤ 1e-4，G 相等，最大单步 ≤ 4.6σ，abs(Σa² − 1) ≤ 1e-12。范围与单步另按所印位数（1 位、2 位小数）比较，仅作信息。
- **DECISION 13-13.** G2 σ/24 同时与 `des.sod_reference()` 及 prompt 给出的数值（316.8921168493721、197.20097668224076）比较，两者均通过。
- **DECISION 13-14.** LO 的 DP 计时：边动作集，切点为 LO 的 SoD，λ = 5γ̄（即一次扫描的第一次运行）。该表另在 σ/24 上评价作为诊断：E[N] 82.79，I 197.30，DP 值对策略值 2.9e-13 [L LO:8]。
- **DECISION 13-15.** `ConstRows.rows_full` 每步返回同一对数组（不复制）。被截断节点的质量检查只在 M > 958 时执行，用 §2.4 的严格不等式。
- **DECISION 13-16.** G1 的附加检查（已报告，计入通过标志）：P5514 的 a、χ 与 `round2a/config_recheck.json` candidates[0] 相等；第 13 轮不重新登记 P5514 与 N。

## 6. 文件

`round13/`：`inputs13.json`、`eval13.py`、`dp13.py`、`common13.py`、`run13.py`、`.gitignore`（`work/`、`__pycache__/`）；`results/gate13.json`、`results/gate_<输入>.json`（六个）、`results/gate_U_design/`（两个扫描 json、`design_edge.json`、五个 `J_*.npy`）、`results/diff_*.txt`；`logs/gate_<输入>.log`（六个）、`logs/gate_summary.log`、`logs/peak_rss_U_gate.txt`；`mem_probe.py` 与 `logs/mem_probe.log`（§3 的内存探针）。不提交 DP 表：U 的表与第 11 轮打包表逐位相同。

命令（在 `round13/` 下，`OMP_NUM_THREADS=1`）：
```bash
R13_INPUT=U ROUND13_WORK=<dir> python3 run13.py gate
```
对 SCH、P5514、N、HI、LO 运行同一命令，然后运行 `python3 run13.py gate_summary`。

## 7. 未做

- Part A、Part B（按指示）。未写预测文件：Part 0 没有预测文件。
- `matched_sod` 已写好但没有运行：Part 0 没有闸门需要它，它将在 Part A 中首次运行。M > 958（截断）的 `ConstRows` 也未经实际运行检验。
- LO 的 DP 表未做 σ/120 评价；LO 的内存未测。
- PAR 未登记，由 Part B 填入。
- §4 第 1 条的内存问题只报告、未改动；Part A 并行运行各输入之前需要先定。
