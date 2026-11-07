# Round 11 plan: edge-parametrised action set + iteration of the tangent point A; redesign of the nominal table

Status: **frozen (2026-11-09; three external review corrections incorporated)**. Premise: Round 10 gives, at the nominal budget, a certified gap of the current table (U/W) to the nominal optimum of window class W of g ∈ [4.037, 4.041] nat (D-efficiency ≥ 63.8 %), i.e. the "worth exploring improvements" row. This round changes only two things on the design side and measures how much they recover.

## 1. Question

Same scenario (Section II-D), same nominal budget B = 316.8921168493721 (`round4/results/mc.json → U/SoD → EN_eval`), same SoD structure (window around the reference value r; the state and transition of evaluator and DP unchanged). Two changes:

- **Action set**: from "half-widths relative to r in 2–40σ" to "edge positions relative to the predicted output x̄_k".
- **Tangent point**: from A = J_SoD⁻¹ only to iterated A ← J(T)⁻¹, accepted on the true log det.

Report four cells: {old actions, new actions} × {A₀, iterated A}, each with ΔI and the gap g′ = U_cert(B) − I_table to UB. U_cert is Round 10's 209.07220154478713 (`round10/results/round10.json → budgets.B316.u_cert`).

Unchanged: budget, σ, input, grid step, measurement kernel, transition formula, estimator, sequential procedure, Section V mismatch and margin. **Changed**: the reference-state support and the action boundary index range (§2.1) — the old code stays read-only; the changes live in a stand-alone implementation under `round11/`.

## 2. New action set (DECISION 11-1)

An action is a pair of edge positions (e⁻, e⁺), e ∈ E = {1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, ∞} (units of σ), 100 actions. Given node r_i and predicted output x̄_k:

- lower edge = min(x̄_k − e⁻σ, r_i − 2σ), upper edge = max(x̄_k + e⁺σ, r_i + 2σ); e = ∞ means only r ∓ 2σ is kept on that side.
- Both edges rounded outward to the nearest grid node (δ_g = σ/24); half-widths L = r_i − lower, U = upper − r_i, both ≥ 2σ = 48 grid units.

Rationale: what the DP cares about is the position of the edge relative to x̄_k (Section IV-E: the facing edge about 1.9σ beyond the output). Parametrising by half-width reaches it only indirectly through r: far from the output the 40σ cap binds (18 % of the budget becomes body transmissions), near it the 2σ step limits placement (±1σ spread). Parametrising by edge removes both and reduces the action count from 400 to 100.

"At least 2σ from r" keeps the lower-limit semantics of the old action set so that the two sets are comparable when r is close to x̄_k; alternatives (1σ or none) are the fallback of DECISION 11-1 and are not run this round.

### 2.1 Support (the main implementation change, DECISION 11-3)

Half-widths of the new actions can exceed 40σ (r_i = x̄_k + 50σ, e⁻ = 2 gives L = 52σ), while the frozen code hard-codes: maximum half-width 40σ (boundary index extension X = 960), reference states confined to x̄_k ± 88σ, and zero value beyond the next step's support. Enlarging the index arrays alone is not enough: a new window can contain the current output while keeping a distant old reference value; under continued silence r is not updated, and the old support that moves with x̄_k would cut off reference values that still carry mass.

Requirements:
- Grid step δ_g, measurement kernel (exactly zero beyond x̄_k ± 39σ) and transition formula (10) unchanged.
- **Reference-state support and measurement-kernel support handled separately. DECISION 11-3 (fixed by review): the state support is the whole design grid** (covering the predicted-output range extended by 8Δ* at both ends, as in Section IV-C), used for both the DP and the forward evaluation. DP states are not pruned by the non-zero mass of a candidate policy, since the DP compares policies and such pruning could exclude states that another policy can reach; the full grid also simplifies the regression check. Every old reference value that can persist is kept.
- Action boundary index range taken over the full grid width, no longer capped at 40σ.
- Assertions as before: per-step mass conservation, non-negativity, **zero mass at the support boundary**; any failure stops the run.
- The regression gate (§5.1) still holds: with the old action set, reference mass never leaves the old ±88σ (beyond 79σ a transmission is certain), so bit-for-bit reproduction is a direct test of the new support implementation.

## 3. Iteration of A (DECISION 11-2)

- Round 0: A₀ = J_SoD(θ*)⁻¹, price scan as in Section IV-D, table T₀, true I(T₀) from the evaluator.
- Round j: A_j = J(T_{j−1})⁻¹, recompute γ_k and γ̄, scan, obtain T_j, compute I(T_j).
- Acceptance: continue only if I(T_j) > I(T_{j−1}) + 0.05 nat; otherwise stop and keep the previous table. At most 3 rounds.
- Each scan 10–12 DP runs; with the new action set each DP is expected to take < 10 s.

## 4. Numbers to report (per cell)

- ΔI = I(T) − I_SoD; E[N] (σ/24 design value and σ/120 re-evaluation; the latter must be ≤ B); g′ = U_cert − I(T); D-efficiency lower bound exp(−g′/9).
- Window shape: mass-weighted median and 10–90 % range of the facing edge relative to x̄_k; transmissions and information per transmission binned by offset (same convention as `cap_cost_by_offset.py`), to see how far the share of body transmissions drops from 18 %.
- Tangent overestimate, **defined relative to the current tangent point**: with T_ref,j the reference of round j (constant SoD for round 0, then the last accepted table), E_j(T) = tr(J_ref,j⁻¹ J(T)) − p − [I(T) − I(T_ref,j)]. When using the DP archive's F (which sums k ≥ 2 only), write E_j(T) = F_j(T) + γ_{1,j} − p − [I(T) − I(T_ref,j)]. The change of tangent point must not be mixed into the overestimate.
- I(T_j) for every iteration of A.

Main result, one table: four cells × {ΔI, g′, D-eff, share of body transmissions}.

## 5. Checks

1. Regression: the new code with the old action set reproduces the Round-4 nominal table bit for bit (entries, E[N] = 316.6035659260541, ΔI = 7.83).
2. Forward = backward for every table (total score agrees to 1e-12).
3. Conservation, non-negativity, support assertions.
4. Monte Carlo check of the final table: 10⁴ trajectories; E[N], S and log det J within statistical error.
5. Under the new action set, L, U ≥ 48 grid units at every (k, i) and edges on grid nodes.
6. I(T) ≤ U_cert (Round 10's bound must cover the new table; if not, Round 10 or this round is wrong: stop).

## 6. Predictions (submitted after freezing, before running)

- New actions + A₀: ΔI ∈ [8.5, 10.0] (recovering 0.7–2.2 nat); body-transmission share < 5 %; the 10–90 % range of the facing-edge position narrows to 1.5–2.3σ.
- Iteration of A adds 0–0.5 nat under either action set; no acceptance after round 2.
- Old actions + iterated A: ΔI ∈ [7.8, 8.3].
- Best cell: g′ ∈ [2.0, 3.4], D-efficiency ∈ [69 %, 80 %].
- The tangent overestimate under the new action set is still > 4 nat (the surrogate's overestimate does not disappear with the action set).

## 7. Verdict

Define δI = I_best − I_old (old table I = 205.03141309913676). Three mutually exclusive cases:

| Condition | Action |
|---|---|
| δI < 0.5 nat | report the actual gain, write "this round did not reach the pre-set 0.5 nat improvement threshold", main line unchanged |
| δI ≥ 0.5 and g′ < 2 nat | Round 12: redo the margin design (T4 equivalent) and the two-stage confirmation with the new action set; the paper's main result is replaced |
| δI ≥ 0.5 and g′ ≥ 2 nat | report as a supplementary result in IV-E ("with the new action set and tangent point, ΔI = X, g′ from the nominal optimum"), main line unchanged |

No attribution: in every case the source of the gap remains undetermined (Round 10 §3). A finite action set plus at most three iterations finding no clear gain only shows that this round's search found a limited gain, not that the design side is not the source; a positive gain below 0.5 nat is not written as "did not narrow the gap".

## 8. Deliverables

- `round11/`: a stand-alone implementation of the edge-parametrised DP and forward evaluation (action-to-(L,U) mapping plus the support handling of §2.1; transition and score formulas copied from the old code), scan and iteration scripts, four-cell results json, shape and binned statistics, figures (edge-position distributions of old and new table, same scale).
- Report `report_round11_CN.md`: one-sentence conclusion, four-cell table, verdict, predictions hit/miss, checks, DECISIONs, not done this round.
- Effort: half a day of code (mostly reaching bit-for-bit regression), about one hour of runtime (4 cells × at most 4 rounds × 12 DP runs).

## 9. Not in this round

- Redoing the margin tables and the two-stage experiment (Round 12 depending on the verdict).
- The Dozzi scenario (Round 9; waits for the action set settled here).
- Interpreting the change of g′ as "structure vs design".
