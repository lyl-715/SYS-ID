# Round 10 执行 prompt（给 Claude Code）

仓库 `lyl-715/SYS-ID`，分支 `claude/optimistic-newton-xv5jvl`，从 head `3314e7d` 开新分支 `round10-upper-bound`。**第一个 commit：把随附的 `plan_round10_logdet_upper_bound_CN.md` 放到 `reports_CN/`**（`3314e7d` 里没有这个文件），然后通读。本 prompt 是它的执行版；两者冲突以方案为准。

## 0. 规矩

- `code_8_3_stage2/stage2_pkg/evaluator/round4` … `round8b` 全部只读。可以 import，不许改一行。新代码全部放 `code_8_3_stage2/stage2_pkg/evaluator/round10/`。
- 第一步先写 `round10/predictions_round10.md`（内容见 §6），commit，然后才许运行任何计算。
- 每个方案里没定的选择，在报告里写一行 `DECISION 10-x: …`，不要静默决定。
- 守恒/断言失败不是退路：停下来报告。
- 报告用中文，文件 `reports_CN/report_round10_CN.md`，结构按 §7。所有数字给来源文件和字段。

## 1. 输入（先复现再使用）

从冻结代码取场景：K = 5000，σ = Δ*/20；回归向量直接用 `round4/evaluator_window.py` 的 `design_grid(name)` 返回的 `Hk`（不要重建）。**所有输入与复现检查直接读存档字段的全精度；下面括号里的取整数字只供识别。** 使用前复现以下存档值，相对误差 ≤ 1e-9 才继续：
- 名义预算 B：`round4/results/mc.json["U/SoD"]["EN_eval"]`（316.8921168493721）。
- 常数 SoD 名义：`mc.json["U/SoD"]["logdet_eval"]`（197.20097668224076）。
- 名义表：`mc.json["U/W"]["logdet_eval"]`（205.03141309913676），即 ΔI = 7.83。
- T4：`round6/results/scan_U_W_m020.json["runs"][1]["logdet"]`（203.29000089955792；该行 κ = 4.000，`EN` = 262.7986258567231，即 B = 262.8 的预算）。
- T4 预算下的常数 SoD 参照：`round7/results/partD_match.json["nominal"]["T4"]["Q"]`（195.4410611333331，M = 581 的等发送数匹配）。**不要用 197.201 − 1.743**：存档里的 1.743 是相对另一条匹配基线 197.18397600616956 的下降量，不是相对原始 SoD。
- 全采样信息 log det(σ⁻² Σ_k h_k h_kᵀ) 没有存档，算出来记录，作检查 4.2 的参照。

## 2. 要实现的函数（`round10/ub.py`，只用 numpy/scipy）

```
a_sym(q)   = norm.isf(q/2)                      # 对称窗半宽（σ 单位）
c_max(q)   = q + 2*a*norm.pdf(a)                # 固定 q 的最大信息权重（对称窗）
dc_max(q)  = a**2                               # 引理 B
a_one(q)   = norm.ppf(q)                        # 单边：alpha = Φ^{-1}(q)
c_one(q)   = q - alpha*norm.pdf(alpha) + norm.pdf(alpha)**2/(1-q)
dc_one(q)  = (alpha - mu)**2,  mu = norm.pdf(alpha)/(1-q)
f(q; H, sigma)      = logdet( sigma^-2 [ H[0] + Σ_{k≥2} c(q_k) H[k] ] ),  H[k] = h_k h_kᵀ
grad_f(q; ...)      = sigma^-2 * (h_kᵀ J^{-1} h_k) * dc(q_k)
project(q, Bm1)     = 投影到 {0≤q≤1, Σq ≤ Bm1}（对 q − t 裁剪到 [0,1]，对 t 二分使 Σ = Bm1；若 Σclip(q) ≤ Bm1 则 t = 0）
u_cert(q_hat)       = f(q_hat) + max_{v∈C} ∇f(q_hat)ᵀ(v − q_hat)
                      内层：按梯度分量降序，依次令 v_k = 1 直到预算用尽，最后一个取分数，其余 0；
                      注意 v 必须满足 0≤v≤1 且 Σ v ≤ Bm1，负梯度分量取 v_k = 0
solve(H, sigma, Bm1, c, dc, tol=0.01)
                      投影梯度上升（步长回溯），每轮算 gap = u_cert − f，gap ≤ tol 停；
                      返回 q_hat, f, u_cert, gap, 乘子 λ̂（取未饱和分量梯度的中位数）
```
数值注意：
- q → 0 时 a → ∞、c_max′ → ∞；用 `norm.isf` / `norm.ppf` 的稳定实现，c_max(0) = 0、c_max(1) = 1、c_one(0) = 0、c_one(1) = 1 显式处理。
- **迭代点必须始终严格在 0 < q_k < 1 内**：初值 q_k ≡ (B−1)/(K−1)；投影得到候选方向后用阻尼与回溯，只接受满足 0 < q_k < 1 的点（例如夹到 [1e-9, 1 − 1e-9] 后再回溯）。认证里的线性最大化 max_{v∈C} ∇f(q̂)ᵀ(v − q̂) 仍在原始可行域 C 上做，**不得通过截断梯度或缩小 C 得到有限证书**。
- J 的逆用 Cholesky。

## 3. 要跑的东西

对两个预算 B ∈ {316.892, 262.8}，各跑：
1. **UB（对称 c_max）**：q̂, f(q̂), U_cert, gap；差距区间 g_L = f(q̂) − I_表、g_U = U_cert − I_表（名义预算对名义表，262.8 对 T4）；D-efficiency 下界 exp(−g_U/9)。
2. **V₁（单边 c_one）**：同样输出，只作参照。
3. **KKT 水填充交叉核对**（DECISION 10-1）：固定 J 迭代求 λ 使 Σ q = B−1、σ⁻²h_kᵀJ⁻¹h_k·dc(q_k) = λ；与投影梯度的 f 相差 ≤ 0.01 则通过。
4. **发送分布图**：q̂_k（上界的最优发送率）与名义表的实际每步发送概率同图；后者直接用 `part_c.table_fields('U/W')` 返回的 `fw['Q'][1:]`（评价器已算好，不要再用 `m_window` 重算）。横轴时间 (s)，纵轴每步发送概率；字号 ≥ 9 pt，题注写明看点。

## 4. 检查（全部通过才算完成）

1. 按预算匹配（全精度）：U_cert(B_nominal) ≥ `U/W.logdet_eval` 且 ≥ `U/SoD.logdet_eval`；U_cert(262.7986258567231) ≥ `runs[1].logdet` 且 ≥ `partD_match…["T4"]["Q"]`。
2. q ≡ 1 时 f 等于全采样 log det。
3. 标量版（H[k] → 1，目标 Σ c(q_k)）：预算 316.8921168493721 下 S_sym = 1636.15048、S_one = 1346.81561（审稿独立算得；允许 ±0.001）。
4. 解析梯度对中心差分，相对误差 ≤ 1e-6，随机 20 个分量。
5. 引理 A 数值复核：q ∈ {0.001, 0.01, 0.063, 0.2, 0.5}，各扫 α 2000 点，**扫描点显式包含理论对称点 α = −Φ⁻¹(1 − q/2)**；检查所有扫描值均 ≤ c_max(q) + 1e-10，且对称点处取得 c_max(q)。不用"最大点落在对称附近"作判据。
6. gap ≤ 0.01 两个预算都达到；达不到就继续迭代，报告实际 gap。

## 5. 输出文件

- `round10/results/round10.json`：每个预算一组 {f, u_cert, gap, g_L, g_U, deff_lower, lambda_hat, n_iter, V1_f, V1_u_cert, kkt_f}，以及检查 1–6 的数值与通过标志。
- `round10/results/q_opt_B316.npy`、`q_opt_B263.npy`、`q_table_nominal.npy`。
- `round10/results/fig_send_profile.png`。
- `reports_CN/report_round10_CN.md`。

## 6. 预测（先写、先 commit，再跑）

```
- ΔI_UB = U_cert(316.892) − 197.201 ∈ [10, 14] nat
- g_* ∈ [2, 6] nat；D-efficiency 下界 ∈ [51%, 80%]
- V₁ − 197.201 ∈ [8.5, 11] nat
- 上界最优发送率 q̂_k 不均匀，与名义表的发送率同向（Spearman > 0.5）
- 两个预算的 gap 在 ≤ 200 次迭代内降到 0.01
```

## 7. 报告结构

1. 一句话结论：当前表距该窗口类名义最优 [g_L, g_U] nat，D-efficiency ≥ X%。
2. 结果表（两预算 × {I_表, f(q̂), U_cert, g_L, g_U, D-eff 下界, V₁}）。
3. 判定：按方案 §6 的表（g_U < 2 / g_L > 4 / 跨阈值 / V₁ 只报位置）。
4. 预测对照：§6 每条写"中/不中"。
5. 检查 1–6 的结果。
6. 发送分布图及一段解读（只描述，不归因）。
7. DECISION 列表。
8. 本轮没做的事（照方案 §9 抄），并注明：本轮以预测中心窗口的相关研究（You–Xie–Song 2013；Han–You–Xie–Wu–Shi 2015）为背景，构造用于本文性能比较的 D 准则基准，不主张其理论新颖性。
