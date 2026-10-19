# 复核指南 —— §8.5 评价器数值验收（单场景 d = 20）

## 包内容
- 方案：`reports_CN/plan_evaluator_CN.md`（v4.1，冻结；本轮不改判据、清单、阈值）。
- 代码（`evaluator/`，与 GitHub `lyl-715/SYS-ID` 分支 `claude/optimistic-newton-xv5jvl` 的 `code_8_3_stage2/stage2_pkg/evaluator/` 一致）：
  - `density_recursion_v1.py`：三个版本 v0 / A / B 一个开关；单元测试 `--test`
  - `make_config.py` → `config.json`（步骤 0）；`predictions.md`（步骤 1，逐字摘自方案 §8、§9，跑前提交）
  - `run_all.py`（36 次评价 + 尾部 + 不变量 + §6 触发检查）→ `mc_reference.py`（MC）→ `analysis.py` → `report_tables.py`；`profile_runtime.py`（P7 的耗时剖析）
- 只读依赖（包根目录，未改）：`mc83.py`、`mc85_scan.py`、`mc85_post.py`、`stage2_common.py`、`density_recursion.py`（v0 原型）、`wk_d20_10000.npy`
- 候选与旧值的来源：`stage2/final_batch1.json`、`stage2/stage2_final.log`、`stage3/dense_R2000.json`、`stage3/stage3_dense.log`、`stage3/unconstrained.json`、`stage3/stage3_unconstrained.log`、`amp_nm_P5514.json`、`README.md`（第 21 行）、`reports_CN/section_8_3_single_CN.md`（第 522、527、562 行）
- 输出（`evaluator/`）：`report_evaluator_CN.md`（报告）、`logs/*.log`、`results/`（每次评价的记录 JSON、$J$、$w_k$、逐步质量；`results/mc/` 的批矩阵与汇总；`analysis.json`）

## 复现
在 `evaluator/` 下（需 numpy、scipy、numba）：

    python3 density_recursion_v1.py --test && python3 run_all.py && python3 mc_reference.py && python3 analysis.py && python3 report_tables.py > logs/report_tables.md

约 7 分钟（`run_all.py` 单进程计时，约 381 s；MC 4 进程约 20 s）。评价器的数是确定性的，应与 `logs/*.log` 逐数一致；耗时随机器负载变化。

## 数的追溯
报告的表由 `report_tables.py` 从存盘结果生成（`logs/report_tables.md`），正文的数取自以下日志行：
- `logs/run_all.log`：`EV <版本>/<输入>/n=<n>/<裕量> <量> <值>`，每个数一行；`TRIGGER-CHECK` / `TRIGGER-SUMMARY`
- `logs/mc_reference.log`：`MC <输入> <量> <值>`；`OLD-U`（与旧种子 20261006 的一致性）
- `logs/analysis.log`：`CONV`、`STAB`、`TAIL`、`ITEM`（199 项）、`WHITENED`、`SIGN`、`NONLIN`、`DISPLAY`、`APPX`、`RUNTIME`、`SEARCH-ROUND`、`CRIT`、`P1`–`P9`
- `logs/unit_tests.log`（第二次）与 `logs/unit_tests_run1_arange_grid.log`（第一次，(b) 未过）；`logs/profile_runtime.log`

请按 日志 → 表（`report_tables.md`）→ 报告正文 三层核。

## 建议重点核的地方
1. **v0 是否就是原型。** 单元测试 (a)：v1 的 v0 与原型 `density_recursion.py` 的 `run(12)` 在 $\mathbb E[N]$、$w_k$、$J$、$G$、每 1000 步质量上逐位相同（`(a) bitwise vs prototype` 行）。
2. **A/B 的一步是否就是方案 §4。** `_window` 返回 $(\sum_j v_j\omega_{ij},\ \sum_j v_j(1-\omega_{ij}))$，$\omega=1\{|i-j|<M\}+w_b1\{|i-j|=M\}$，A 取 $w_b=0$、B 取 $1/2$；`_ab_step` 中 $m'_j=m_j\hat P_j+\hat g_j\sum_i m_i(1-\omega_{ij})$。$\omega$ 对称，守恒恒等式成立；$1-\hat P$ 与外侧和直接由单调前缀数组的差得到，非负逐位成立。核对单元测试 (b)（A = 0.483377404983273，B = 0.5）与 (c)（守恒 $\le1.5\times10^{-13}$）。
3. **节点构造的改动。** 第一次单元测试用原型的 `np.arange` 构造 A/B 与 (b) 的节点，(b) 差 $-1.8\times10^{-12}$；改为 $z_0+ih$ 后通过。请判断这一改动是否只是按方案 §4 实现节点（`config.json` 的 `grids.nodes` 同时写了两种形式），是否影响任何结论。
4. **$c_k$ 只在 $m_k\ne0$ 的区间上求值。** 区间外 $m_k$ 恰为 0，结果应精确不变；不变量检查另在全网格上做（`c_min`、`c_max`、`tail_*` 行）。
5. **MC 与比较清单。** `mc_reference.py` 自己的循环与 `stage2_common.evaluate_crn` 逐位相同（`bitwise_equal_to_evaluate_crn` 行）；点估计来自汇总矩阵；$W=J_{B,24}^{-1/2}$；199 项的构成（每输入 49 项 + 3 个配对 $\Delta m_D$）；阈值 $t_{19,1-0.05/404}=4.4958$。
6. **$\mathrm{tr}(\bar D)/9$ 与 $\log\det$ 两项的 $z$ 在每个输入上相同到两位小数**（一阶上 $\log\det(I+\bar D)\approx\mathrm{tr}\bar D$）。冻结清单不变，但请判断报告对这一点的说明是否足够。
7. **耗时线的通过很勉强。** B/N/$n=24$ 为 14.88 s（线 15 s），只测一次；实现是 numba 递推 + numpy `cfun` 的混合（报告 §6 第 2 条），剖析显示时间主要在全网格的 numba 步。请判断"通过"的写法是否恰当。
8. **预测对照（报告表 3）。** 特别是 P1（1000 步检查点 0.19% 与逐步最大 1% 的区分）、P3、P7 的"错"，P10 第三条的"未定"，P12 的 v0 舍入超界。
9. **旧 MC 的一致性。** $S$ 差 0.289，为新 se 的 3.64 倍；旧 se 在 §8.3 报告中有两个不同的值（0.12 与 0.25），因此只报原始差。请判断是否需要另行核查旧 $S$。

## 已知的方案外选择（报告 §6 已列）
A/B 节点用 $z_0+ih$（v0 保留 `arange`）；$c_k$ 用原样 `cfun` 在 $m_k\ne0$ 区间上求值；$1-\hat P$ 用外侧前缀和；尾部检查四个输入都做；$c_k$ 不变量在全网格上做且 A/B 共用；附加的 $0\le w_k\le1$ 记录；耗时剖析脚本。
