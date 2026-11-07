# Round 10 plan: nominal optimum of log det J at a given budget, and the certified gap of the current table

Status: **frozen (2026-11-08; two external review rounds incorporated)**. Purpose: answer "how far is the 7.83-nat table from the best possible?" The answer is a certified interval [g_L, g_U] of width at most 0.01 nat.

Literature placement: the framework "window centred on the predicted output + censored ML + CRLB" exists — You, Xie, Song (IEEE TSP 2013) give the CRLB under a deterministic innovation-threshold scheduler and an adaptive scheduler that attains it asymptotically; Han, You, Xie, Wu, Shi (IEEE TSP 2015) optimise the scheduling parameters under a rate constraint (spectral-radius criterion, SDP relaxation) and prove that the optimal reference is the predicted output. This round uses that work as background to construct a D-criterion benchmark for the performance comparison in this paper, without claiming theoretical novelty; the paper's contribution is on the SoD-structured side (window around the last transmitted value). The benchmark quantifies the gap between the current SoD-structured table and the nominal optimum of the larger window class, without attributing that gap to structure or design.

## 1. The inequality and its scope

**Window class W**: the interval window at step k is determined by the time and by the **history known to the receiver** (transmitted values and silence events). This covers the paper's window table (the table reads only k and r_k) but **not** rules that use untransmitted measurements (e.g. the z_k state of the Dozzi trigger, which uses all measurements) — if the window depended on untransmitted measurements, silence events would carry information about them and the per-step accumulation of Proposition 2 would no longer cover all the information; the bound would fail. This restriction is part of the theorem statement.

For any rule in W with Σ_{k≥2} E[q_k] ≤ B − 1,

log det J ≤ UB(B) := max_{q̄} f(q̄),  f(q̄) = log det( σ⁻² [ h₁h₁ᵀ + Σ_{k≥2} c_max(q̄_k) h_k h_kᵀ ] ),
subject to Σ_{k≥2} q̄_k ≤ B − 1, 0 ≤ q̄_k ≤ 1,

where c_max(q) = q + 2a·φ(a), a = Φ⁻¹(1 − q/2), is the information weight of the symmetric window at transmission probability q.

Three steps:
- **Lemma A (at fixed q the symmetric window maximises the information).** Along the constraint Φ(β) − Φ(α) = 1 − q, dc/dα = φ(α)(β − α)[2μ_T − (α + β)], μ_T the truncated mean. With midpoint m = (α+β)/2 and half-width d = (β−α)/2, (1−q)(μ_T − m) = ∫₀^d t[φ(m+t) − φ(m−t)] dt; for m > 0, (m+t)² > (m−t)², the integrand is negative, μ_T < m, hence dc/dα < 0; the reverse for m < 0. So c is maximised at α = −β. Analytic proof; no interval arithmetic needed.
- **Lemma B (c_max is concave).** dc_max/dq = a² = [Φ⁻¹(1 − q/2)]², decreasing in q.
- **Third step.** w_k = E_r[c(α_k, β_k)] ≤ E_r[c_max(q_k(r))] ≤ c_max(q̄_k) (Lemma A, then Jensen with Lemma B); J ⪯ σ⁻² Σ c_max(q̄_k) h_k h_kᵀ (Loewner order); log det is monotone.

**UB is attainable, hence it is the nominal optimum of class W, not merely a loose bound.** With the optimal q̄*, the time-varying window [x̄_k − σa(q̄*_k), x̄_k + σa(q̄*_k)] centred on the nominal output (a function of time only, hence in W) yields at the nominal truth exactly q = q̄*_k and c = c_max(q̄*_k) at every step; its information matrix equals the one in UB. So g = UB − log det J_table is "the distance of the current SoD table from the nominal optimum of the class".

f is concave (log det is concave and Loewner-monotone; c_max is concave in q), so the global optimum is computable.

## 2. Computation and certification

- Variables q̄ ∈ ℝ^{K−1}, K = 5000. Analytic gradient ∂f/∂q̄_k = σ⁻² h_kᵀ J⁻¹ h_k · a_k².
- Solver: projected gradient ascent; KKT water-filling (unsaturated components satisfy σ⁻²h_kᵀJ⁻¹h_k·a(q̄_k)² = λ) as a cross-check (DECISION 10-1).
- **Certification.** The optimiser's value f(q̂) is only a lower bound on UB. Use the tangent plane of the concave function: at a feasible interior point q̂,

  U_cert = f(q̂) + max_{v∈C} ∇f(q̂)ᵀ(v − q̂),

  whose inner problem is a linear budget allocation (sort by gradient, fill components to 1 in descending order, fractional last component; exact). Then f(q̂) ≤ UB ≤ U_cert. **Stopping rule: U_cert − f(q̂) ≤ 0.01 nat.** No Lagrangian plug-in, no interval arithmetic.
- **Certified gap interval.** Let I_table = log det J_table, g_L = f(q̂) − I_table, g_U = U_cert − I_table. The true gap g_* = UB − I_table lies in [g_L, g_U], an interval of width ≤ 0.01 nat. The text reports this interval; if a single number is wanted, report "at most g_U nat from the nominal optimum", with the certified D-efficiency lower bound e^{−g_U/9}. U_cert is never called the optimum on its own.
- Two budgets. The nominal budget is B = 316.8921168493721, read from `round4/results/mc.json → U/SoD → EN_eval`; 316.892 is its rounded display value. The T4 budget is B = 262.7986258567231 (`round6/results/scan_U_W_m020.json → runs[1] → EN`; 262.8 rounded).
- Side output: the time profile of q̄*_k against the nominal table's actual per-step transmission probability.

## 3. The one-sided reference value V₁

Repeat with c_max replaced by the one-sided c₁(q); call it **V₁, the ideal one-sided window reference**. Ideal one-sided windows and SoD windows are not nested (the former have no reference-value constraint; the latter can use both edges in some states), so UB − V₁ is **not** the cost of the SoD structure and V₁ − log det J_table is **not** a recoverable design loss. V₁ is reported for position only, with no attribution.

## 4. Checks

1. Per-budget matching: for each budget B, U_cert(B) is not below the information of every comparison policy confirmed to satisfy that budget — at the nominal budget, the current table (7.83 + log det J_SoD) and the constant-SoD baseline; at B = 262.8, T4 and the baseline that satisfies that budget. A bound at a smaller budget is not required to exceed a policy that uses a larger budget. A failure means the derivation or the code is wrong: stop.
2. With q̄_k ≡ 1, f equals the full-data information log det(σ⁻² Σ h_k h_kᵀ).
3. Scalar version (h_k h_kᵀ → 1, objective → Σ c_max(q̄_k)): at budget 316.892 reproduce S_sym = 1636.15 and S_one = 1346.82 (independently computed by the reviewer).
4. Analytic gradient against finite differences.
5. Numerical re-check of Lemma A: on a fine grid of q, scan α; the maximum is at the symmetric point.

## 5. Predictions (submitted after freezing, before running)

- ΔI_UB = U_cert − log det J_SoD ∈ [10, 14] nat (nominal budget).
- g_* ∈ [2, 6] nat (certified interval ≤ 0.01 wide); D-efficiency lower bound e^{−g_U/9} ∈ [51 %, 80 %].
- V₁ − log det J_SoD ∈ [8.5, 11].
- q̄*_k is non-uniform: steps whose h_k points along weak directions of J_SoD receive more budget.

## 6. Verdict

| Certified result | Conclusion |
|---|---|
| g_U < 2 nat | the current table is within 2 nat of the nominal optimum (certified D-efficiency > 80 %) |
| g_L > 4 nat | the current table is more than 4 nat from the nominal optimum (D-efficiency < 64 %); worth exploring improvements (Round 11: edge parametrisation, iteration of A) |
| interval straddles a threshold | tighten the solver gap before deciding |
| V₁ above or below the table | position relative to the ideal one-sided policy only; not used to attribute the gap |

Do not write "the gap is small, hence the remaining gap is due to the SoD structure": a small gap proves closeness to the optimum, not the cause of the remainder.

## 7. Reporting

Main result: the gap interval [g_L, g_U] (nat), or "at most g_U nat". Percentages as the certified D-efficiency lower bound exp(−g_U/9) (2 nat ≈ 80.1 %, 4 nat ≈ 64.1 %). 7.83/ΔI_UB may be reported only as "the fraction of the log-det gain over constant SoD", never as "X % of the bound attained".

## 8. Deliverables

- Lemmas A and B with proofs (new appendix after Appendix F).
- A paragraph in Section IV-E: certified interval of the nominal optimum [f(q̂), U_cert], gap interval [g_L, g_U] of the current table, D-efficiency lower bound; V₁ and the transmission-profile figure as explanatory references; You 2013 and Han 2015 cited for placement.
- In the conclusion, after "7.83": "at most g_U nat from the nominal optimum of this window class (D-efficiency ≥ X %)".
- Code: a stand-alone script with inputs (h_k, σ, B) and outputs U_cert, f(q̂), q̄*, profile figure. Reusable on any system; it is "step 0" of Round 9.
- Effort: half a day of derivation, half a day of code, minutes of runtime.

## 9. Not in this round

- Bounds under model mismatch (the bound concerns the true model; mismatch affects only the table).
- Non-interval silence sets; rules depending on untransmitted measurements (see §1).
- Attribution of the gap to structure or design (no basis, see §3).
