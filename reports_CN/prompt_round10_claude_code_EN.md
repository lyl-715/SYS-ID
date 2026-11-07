# Round 10 execution prompt (for Claude Code)

Repository `lyl-715/SYS-ID`, branch `claude/optimistic-newton-xv5jvl`; create a new branch `round10-upper-bound` from head `3314e7d`. **First commit: put the accompanying `plan_round10_logdet_upper_bound_EN.md` (and the CN original) into `reports_CN/`** — `3314e7d` does not contain it — then read it in full. This prompt is the execution version of that plan; where they conflict, the plan wins.

## 0. Rules

- `code_8_3_stage2/stage2_pkg/evaluator/round4` … `round8b` are read-only. Import from them; do not change a line. All new code goes to `code_8_3_stage2/stage2_pkg/evaluator/round10/`.
- First write `round10/predictions_round10.md` (content in §6) and commit it; only then run any computation.
- Every choice the plan leaves open is recorded in the report as a line `DECISION 10-x: …`; never decide silently.
- Conservation/assertion failures are not fallbacks: stop and report.
- Report in `reports_CN/report_round10_CN.md` (Chinese; an English copy `report_round10_EN.md` is welcome), structure in §7. Every number cites its source file and field.

## 1. Inputs (reproduce before use)

Take the scenario from the frozen code: K = 5000, σ = Δ*/20; use the regressors `Hk` returned by `design_grid(name)` in `round4/evaluator_window.py` (do not rebuild them). **All inputs and reproduction checks read the archived fields at full precision; the rounded numbers in parentheses are for identification only.** Reproduce the following archived values with relative error ≤ 1e-9 before proceeding:
- Nominal budget B: `round4/results/mc.json["U/SoD"]["EN_eval"]` (316.8921168493721).
- Constant SoD, nominal: `mc.json["U/SoD"]["logdet_eval"]` (197.20097668224076).
- Nominal table: `mc.json["U/W"]["logdet_eval"]` (205.03141309913676), i.e. ΔI = 7.83.
- T4: `round6/results/scan_U_W_m020.json["runs"][1]["logdet"]` (203.29000089955792; that run has κ = 4.000 and `EN` = 262.7986258567231, which is the B = 262.8 budget).
- Constant-SoD reference at the T4 budget: `round7/results/partD_match.json["nominal"]["T4"]["Q"]` (195.4410611333331; equal-transmission match with M = 581). **Do not use 197.201 − 1.743**: the archived 1.743 is a drop relative to a different matched baseline (197.18397600616956), not relative to the original SoD.
- Full-data information log det(σ⁻² Σ_k h_k h_kᵀ) is not archived; compute and record it as the reference for check 4.2.

## 2. Functions to implement (`round10/ub.py`, numpy/scipy only)

```
a_sym(q)   = norm.isf(q/2)                      # symmetric half-width (units of σ)
c_max(q)   = q + 2*a*norm.pdf(a)                # max information weight at fixed q (symmetric window)
dc_max(q)  = a**2                               # Lemma B
a_one(q)   = norm.ppf(q)                        # one-sided: alpha = Φ^{-1}(q)
c_one(q)   = q - alpha*norm.pdf(alpha) + norm.pdf(alpha)**2/(1-q)
dc_one(q)  = (alpha - mu)**2,  mu = norm.pdf(alpha)/(1-q)
f(q; H, sigma)      = logdet( sigma^-2 [ H[0] + Σ_{k≥2} c(q_k) H[k] ] ),  H[k] = h_k h_kᵀ
grad_f(q; ...)      = sigma^-2 * (h_kᵀ J^{-1} h_k) * dc(q_k)
project(q, Bm1)     = projection onto {0≤q≤1, Σq ≤ Bm1} (clip q − t to [0,1], bisect t so that Σ = Bm1;
                      t = 0 if Σ clip(q) ≤ Bm1)
u_cert(q_hat)       = f(q_hat) + max_{v∈C} ∇f(q_hat)ᵀ(v − q_hat)
                      inner problem: sort gradient components in descending order, set v_k = 1 in that order
                      until the budget is spent, fractional last component, the rest 0;
                      v must satisfy 0≤v≤1 and Σ v ≤ Bm1; components with negative gradient get v_k = 0
solve(H, sigma, Bm1, c, dc, tol=0.01)
                      projected gradient ascent with backtracking; each iteration compute gap = u_cert − f,
                      stop when gap ≤ tol; return q_hat, f, u_cert, gap, multiplier λ̂
                      (median gradient over unsaturated components)
```
Numerical notes:
- As q → 0, a → ∞ and c_max′ → ∞; use the stable `norm.isf` / `norm.ppf`; handle c_max(0) = 0, c_max(1) = 1, c_one(0) = 0, c_one(1) = 1 explicitly.
- **Iterates must stay strictly inside 0 < q_k < 1**: initial point q_k ≡ (B−1)/(K−1); after projection, use damping and backtracking and accept only points with 0 < q_k < 1 (e.g. clamp to [1e-9, 1 − 1e-9] before backtracking). The linear maximisation in the certificate, max_{v∈C} ∇f(q̂)ᵀ(v − q̂), is still over the original feasible set C; **do not obtain a finite certificate by truncating the gradient or shrinking C**.
- Invert J via Cholesky.

## 3. What to run

For each budget B ∈ {316.8921168493721, 262.7986258567231}:
1. **UB (symmetric c_max)**: q̂, f(q̂), U_cert, gap; gap interval g_L = f(q̂) − I_table, g_U = U_cert − I_table (nominal budget against the nominal table, 262.8 against T4); D-efficiency lower bound exp(−g_U/9).
2. **V₁ (one-sided c_one)**: same outputs, reference only.
3. **KKT water-filling cross-check** (DECISION 10-1): fixed-point iteration on J, solve λ so that Σ q = B−1 and σ⁻²h_kᵀJ⁻¹h_k·dc(q_k) = λ; passes if f differs from the projected-gradient value by ≤ 0.01.
4. **Transmission-profile figure**: q̂_k (the bound's optimal transmission probability) and the nominal table's actual per-step transmission probability on one plot; take the latter directly from `fw['Q'][1:]` returned by `part_c.table_fields('U/W')` (already computed by the evaluator; do not recompute from `m_window`). x-axis time (s), y-axis per-step transmission probability; font ≥ 9 pt; caption states what to look at.

## 4. Checks (all must pass)

1. Per-budget matching (full precision): U_cert(B_nominal) ≥ `U/W.logdet_eval` and ≥ `U/SoD.logdet_eval`; U_cert(262.7986258567231) ≥ `runs[1].logdet` and ≥ `partD_match…["T4"]["Q"]`.
2. With q ≡ 1, f equals the full-data log det.
3. Scalar version (H[k] → 1, objective Σ c(q_k)): at budget 316.8921168493721, S_sym = 1636.15048 and S_one = 1346.81561 (computed independently by the reviewer; tolerance ±0.001).
4. Analytic gradient against central differences, relative error ≤ 1e-6 on 20 random components.
5. Lemma A numerical re-check: q ∈ {0.001, 0.01, 0.063, 0.2, 0.5}, 2000 α points each, **the scan explicitly includes the theoretical symmetric point α = −Φ⁻¹(1 − q/2)**; check that all scanned values are ≤ c_max(q) + 1e-10 and that the symmetric point attains c_max(q). Do not use "the maximum falls near the symmetric point" as the criterion.
6. gap ≤ 0.01 reached for both budgets; if not, keep iterating and report the actual gap.

## 5. Output files

- `round10/results/round10.json`: per budget {f, u_cert, gap, g_L, g_U, deff_lower, lambda_hat, n_iter, V1_f, V1_u_cert, kkt_f}, plus the values and pass flags of checks 1–6.
- `round10/results/q_opt_B316.npy`, `q_opt_B263.npy`, `q_table_nominal.npy`.
- `round10/results/fig_send_profile.png`.
- `reports_CN/report_round10_CN.md` (and the EN copy).

## 6. Predictions (write and commit first, then run)

```
- ΔI_UB = U_cert(316.892…) − 197.20097668224076 ∈ [10, 14] nat
- g_* ∈ [2, 6] nat; D-efficiency lower bound ∈ [51 %, 80 %]
- V₁ − 197.20097668224076 ∈ [8.5, 11] nat
- the bound's optimal q̂_k is non-uniform and co-directional with the nominal table's profile (Spearman > 0.5)
- the gap reaches 0.01 within 200 iterations for both budgets
```

## 7. Report structure

1. One-sentence conclusion: the current table is [g_L, g_U] nat from the nominal optimum of this window class, D-efficiency ≥ X %.
2. Results table (two budgets × {I_table, f(q̂), U_cert, g_L, g_U, D-eff lower bound, V₁}).
3. Verdict by the table in plan §6 (g_U < 2 / g_L > 4 / straddling / V₁ position only).
4. Predictions: each line of §6 marked hit / miss.
5. Results of checks 1–6.
6. The transmission-profile figure with one paragraph of reading (describe only; do not attribute).
7. DECISION list.
8. Not done in this round (copy plan §9), and note: this round uses the prediction-centred-window literature (You–Xie–Song 2013; Han–You–Xie–Wu–Shi 2015) as background to construct a D-criterion benchmark for the comparison in this paper, without claiming theoretical novelty.
