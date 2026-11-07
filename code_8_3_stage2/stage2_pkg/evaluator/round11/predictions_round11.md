# Round 11 predictions (written and committed before any computation)

Source: `reports_CN/prompt_round11_claude_code_EN.md` §6 (operational form) and
`reports_CN/plan_round11_edge_actions_EN.md` §6 (frozen 2026-11-09). Nothing below has been computed yet: at the time of
this commit no round-11 code exists and no round-11 number has been produced.

Notation: I_SoD = 197.20097668224076 (`round4/results/mc.json["U/SoD"]["logdet_eval"]`), I_old = 205.03141309913676
(`mc.json["U/W"]["logdet_eval"]`), U_cert = 209.07220154478713 (`round10/results/round10.json["budgets"]["B316"]["u_cert"]`),
ΔI = I(T) − I_SoD, g′ = U_cert − I(T), D-eff lower bound = exp(−g′/9). Cells: {old, edge} action set × {A₀, iterated A}.
"Best cell" = the cell with the largest I(T) among the four cells whose table passes the σ/120 budget re-evaluation and
all checks of prompt §4.

```
- edge/A0: ΔI ∈ [8.5, 10.0]; body-transmission share (|r − x̄_k| ≥ 40σ bin) < 5 %; facing-edge 10–90 % range within [1.5, 2.3]σ
- iteration of A: +0 to +0.5 nat in each action set; no acceptance after round 2
- old/iterated: ΔI ∈ [7.8, 8.3]
- best cell: g′ ∈ [2.0, 3.4] nat; D-eff ∈ [69 %, 80 %]
- tangent overestimate E_0 under edge > 4 nat
Scoring: each line hit iff the reported value lies in the stated range; "no acceptance after round 2" hit iff round 3 is never accepted.
```

Scoring rules (fixed now, before running). Every item below is scored separately as hit / miss.

1. edge/A₀ ΔI: hit iff 8.5 ≤ I(T_edge,A₀) − I_SoD ≤ 10.0 (I on the σ/24 design grid, the evaluator's log det J).
2. edge/A₀ body-transmission share: hit iff the share of the transmissions E[N] − 1 (k ≥ 2) falling in the
   |r − x̄_k| ≥ 40σ bin is < 5 %. (The forced first send, k = 1, is not in any offset bin; the share is taken over
   Σ_{k≥2} Σ_i m_{k,i} q̂_{k,i}.)
3. edge/A₀ facing-edge range: hit iff both the 10 % and the 90 % mass-weighted quantiles of the facing-edge offset
   (distance from x̄_k to the near edge, `round4/part_c.shape` convention, σ units) lie in [1.5, 2.3].
4. Iteration of A, old actions: hit iff 0 ≤ I(old/iterated) − I(old/A₀) ≤ 0.5.
5. Iteration of A, edge actions: hit iff 0 ≤ I(edge/iterated) − I(edge/A₀) ≤ 0.5.
6. No acceptance after round 2: hit iff round 3 is never accepted in either action set (a round 3 that is never run
   because an earlier round was rejected counts as "not accepted").
7. old/iterated ΔI: hit iff 7.8 ≤ I(old/iterated) − I_SoD ≤ 8.3.
8. Best cell g′: hit iff 2.0 ≤ U_cert − I(best) ≤ 3.4.
9. Best cell D-eff: hit iff 0.69 ≤ exp(−g′/9) ≤ 0.80.
10. Tangent overestimate: hit iff E_0(T_edge,A₀) = F_0(T) + γ_{1,0} − 9 − [I(T) − I_SoD] > 4 nat
    (round-0 tangent point A₀ = J_SoD⁻¹, reference T_ref,0 = constant SoD).

Values reported at σ/24 (the design grid) are the scored values; σ/120 values are reported alongside and are used only
for the budget check (check 7).
