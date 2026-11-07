# Round 10 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round10_claude_code_EN.md` §6 (operational form) and
`reports_CN/plan_round10_logdet_upper_bound_EN.md` §5 (frozen 2026-11-08). Nothing below has been computed yet:
at the time of this commit no round-10 code exists and no round-10 number has been produced.

Notation: B_nom = 316.8921168493721 (`round4/results/mc.json["U/SoD"]["EN_eval"]`),
log det J_SoD = 197.20097668224076 (`mc.json["U/SoD"]["logdet_eval"]`), I_table = 205.03141309913676
(`mc.json["U/W"]["logdet_eval"]`), g_* = UB(B_nom) − I_table, certified interval [g_L, g_U].

```
- ΔI_UB = U_cert(316.8921168493721) − 197.20097668224076 ∈ [10, 14] nat
- g_* ∈ [2, 6] nat; D-efficiency lower bound ∈ [51 %, 80 %]
- V₁ − 197.20097668224076 ∈ [8.5, 11] nat
- the bound's optimal q̂_k is non-uniform and co-directional with the nominal table's profile (Spearman > 0.5)
- the gap reaches 0.01 within 200 iterations for both budgets
```

Scoring rules (fixed now, before running):

1. ΔI_UB: hit iff 10 ≤ U_cert(B_nom) − 197.20097668224076 ≤ 14.
2. g_*: hit iff the certified interval [g_L, g_U] lies inside [2, 6]; miss iff it lies outside; "undecided" if it
   straddles 2 or 6. The D-efficiency part is scored separately: hit iff exp(−g_U/9) ∈ [0.51, 0.80].
3. V₁: hit iff V1_f(B_nom) − 197.20097668224076 ∈ [8.5, 11] (V1_u_cert is reported alongside; if the two straddle a
   boundary the line is "undecided").
4. Profile: hit iff q̂_k (k = 2..K, nominal budget) is non-uniform (max/min > 1.5, a threshold fixed here) and the
   Spearman rank correlation between q̂_k and the nominal table's per-step transmission probability `fw['Q'][1:]`
   exceeds 0.5.
5. Iterations: hit iff the projected-gradient solver reaches U_cert − f(q̂) ≤ 0.01 within 200 iterations for both
   budgets (symmetric c_max; the V₁ runs are reported but not scored).

Plan §5, fourth bullet, in qualitative form: "q̄*_k is non-uniform: steps whose h_k points along weak directions of
J_SoD receive more budget." It is the same prediction as line 4; line 4 is the scored version. As a descriptive
companion (not scored) the report gives the Spearman correlation of q̂_k with σ⁻² h_kᵀ J_SoD⁻¹ h_k, J_SoD the archived
constant-SoD information matrix `round4/results/J_U_SoD.npy`.
