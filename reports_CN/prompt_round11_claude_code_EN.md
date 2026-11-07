# Round 11 execution prompt (for Claude Code)

Repository `lyl-715/SYS-ID`. Start from branch `round10-upper-bound` (head = the Round-10 report commit); create branch `round11-edge-actions`. **First commit: put the accompanying `plan_round11_edge_actions_EN.md` and `plan_round11_edge_actions_CN.md` into `reports_CN/`**, then read the EN plan in full. This prompt is the execution version; where they conflict, the plan wins.

## 0. Rules

- `round4` … `round10` under `code_8_3_stage2/stage2_pkg/evaluator/` are read-only; import from them, never edit. New code only in `round11/`.
- Write `round11/predictions_round11.md` (content in §6, with scoring rules spelled out) and commit it before any computation.
- Every open choice is recorded in the report as `DECISION 11-x: …`. DECISION 11-1 (edge set and the 2σ-from-r rule), 11-2 (acceptance rule for iterated A) and 11-3 (full-grid support) are fixed by the plan and are not to be re-decided.
- Assertion failures (conservation, non-negativity, support boundary) stop the run; they are not fallbacks.
- Report `reports_CN/report_round11_CN.md` (plus an EN copy), structure in §7; every number cites file and field.
- Importing `round4/trigger_dp.py` creates empty folders under `round4/results/`; delete them after each run (as in Round 10) so `git diff` on round4–round10 stays empty.

## 1. Inputs (full precision, reproduce before use)

- B = `round4/results/mc.json["U/SoD"]["EN_eval"]` (316.8921168493721); I_SoD = `…["U/SoD"]["logdet_eval"]` (197.20097668224076); old table I_old = `…["U/W"]["logdet_eval"]` (205.03141309913676), E[N]_old = `…["U/W"]["EN_eval"]` (316.6035659260541).
- U_cert = `round10/results/round10.json["budgets"]["B316"]["u_cert"]` (209.07220154478713).
- Regressors `Hk` from `round4/evaluator_window.design_grid('U')`; σ, Δ*, K from the frozen scenario constants; the old nominal table from `round4/results/table_U_W.npz` (or wherever `part_c.table_fields('U/W')` loads it).
- Reproduce I_SoD, I_old and E[N]_old with the frozen evaluator (relative error ≤ 1e-9) before proceeding.

## 2. Implementation (`round11/`)

### 2.1 Stand-alone DP + forward evaluator
Copy the transition and score formulas of `round4/trigger_dp.py` and the forward recursion of `round4/evaluator_window.py` into `round11/dp11.py` / `round11/eval11.py`. Do not change: grid step δ_g = σ/24, the measurement kernel (exactly zero beyond ±39σ), the two-branch mass update (10), the trapezoidal edge weights, the "two tails, never 1 − P̂" rule, prefix/suffix-sum arrival term, fixed tie order over the action list.

Change (plan §2.1, DECISION 11-3):
- **Reference-state support (DECISION 11-3, fixed by review): the full design grid at every step**, for both the DP and the forward evaluation. Do not prune DP states by the non-zero mass of any candidate policy: the DP compares policies, and pruning by one policy's forward mass could exclude states another policy can reach. Never zero out a node that carries mass.
- **Boundary index range**: taken over the full grid width; remove the 40σ (X = 960) cap.
- Assert at every step: |Σ m − 1| ≤ 1e-12, m ≥ 0, mass at the first and last support node = 0.

### 2.2 Action sets
- `old`: (L, U) ∈ {48, 96, …, 960}² grid units (400 actions), applied exactly as the old code does.
- `edge` (plan §2): (e⁻, e⁺) ∈ E², E = {1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, ∞} (σ units), 100 actions. For node i at step k: lower = min(x̄_k − e⁻σ, r_i − 2σ), upper = max(x̄_k + e⁺σ, r_i + 2σ); e = ∞ keeps only r_i ∓ 2σ; round both edges outward to grid nodes; L = r_i − lower, U = upper − r_i, both ≥ 48 grid units (assert).
- The table format stays "action code per (k, i)" with the action list stored alongside, so the forward evaluator and the sensor look-up are unchanged.

### 2.3 Price scan and iteration of A
- Scan exactly as Section IV-D of the paper (λ₀ = 5γ̄, double until feasible, ×0.8 until infeasible, bisect 8 times, return the feasible table with the largest F = Σ_{k≥2} γ_k w_k). Feasibility during the scan on the σ/24 grid; final E[N] of every reported table re-evaluated on the σ/120 grid with `round7/evaluator_refine.evaluate` (or the equivalent frozen path) and required ≤ B.
- Iteration of A (plan §3): A₀ = J_SoD⁻¹; A_j = J(T_{j−1})⁻¹ with γ_k, γ̄ recomputed; accept T_j only if I(T_j) > I(T_{j−1}) + 0.05 nat; at most 3 rounds; keep the last accepted table.
- Order: regression gate first (§2.4); only after it passes run the four cells old/A₀, old/iterated, edge/A₀, edge/iterated. Final conclusions use only tables that pass the σ/120 budget re-evaluation and all checks in §4.

### 2.4 Regression gate (check 5.1, blocking)
Before any new-action run: `edge`-code path with the `old` action set, A₀, nominal scan → must reproduce the Round-4 table bit for bit (action codes, E[N] = 316.6035659260541 to 1e-12, I = 205.03141309913676 to 1e-9). If not, stop and report; nothing else counts.

## 3. Per-cell outputs

- ΔI, E[N] (σ/24 and σ/120), g′ = U_cert − I(T), D-eff lower bound exp(−g′/9).
- Facing-edge position relative to x̄_k: mass-weighted median and 10–90 % range (same definition as `round4/results/shape_U_W.json`).
- Binned transmissions and information per transmission by |r − x̄_k| (bins [0,2), [2,5), [5,10), [10,20), [20,40), [40,∞)), using `fw['Q']`-style per-step send probability and the evaluator's per-node c; report the share of the budget in the last bin and its info/send.
- Tangent overestimate E_j(T) = F_j(T) + γ_{1,j} − p − [I(T) − I(T_ref,j)] with T_ref,0 = constant SoD and T_ref,j = last accepted table (plan §4).
- For iterated cells: I(T_j) for every round and which rounds were accepted.
- Figures: edge-position distribution (histogram of facing-edge offset, mass-weighted) for the old table and the best edge table, same axes; font ≥ 9 pt; caption says what to look at.

## 4. Checks (all must pass)

1. Regression gate (§2.4).
2. Forward total score = backward recursion value for every table (≤ 1e-12).
3. Conservation / non-negativity / zero boundary mass at every step of every run.
4. Monte Carlo on the best table: 10⁴ trajectories; E[N], S, log det J within statistical error (report z-scores).
5. Under `edge`: L, U ≥ 48 everywhere; edges on grid nodes.
6. I(T) ≤ U_cert for every table (if violated: stop; Round 10 or this round is wrong).
7. σ/120 E[N] ≤ B for every reported table.

## 5. Output files

- `round11/results/round11.json`: four cells × all §3 numbers; per-iteration log; check results.
- `round11/results/table_edge_A0.npz`, `table_edge_iter.npz`, `table_old_iter.npz` (+ action lists).
- `round11/results/fig_edge_positions.png`.
- `reports_CN/report_round11_CN.md` (+ EN copy).

## 6. Predictions (write and commit first)

```
- edge/A0: ΔI ∈ [8.5, 10.0]; body-transmission share (|r − x̄_k| ≥ 40σ bin) < 5 %; facing-edge 10–90 % range within [1.5, 2.3]σ
- iteration of A: +0 to +0.5 nat in each action set; no acceptance after round 2
- old/iterated: ΔI ∈ [7.8, 8.3]
- best cell: g′ ∈ [2.0, 3.4] nat; D-eff ∈ [69 %, 80 %]
- tangent overestimate E_0 under edge > 4 nat
Scoring: each line hit iff the reported value lies in the stated range; "no acceptance after round 2" hit iff round 3 is never accepted.
```

## 7. Report structure

1. One-sentence conclusion: δI = I_best − I_old, the best cell, its g′ and D-eff.
2. Four-cell table (ΔI, g′, D-eff, body-transmission share).
3. Verdict by plan §7 (δI < 0.5 / δI ≥ 0.5 & g′ < 2 / δI ≥ 0.5 & g′ ≥ 2), no attribution sentence.
4. Predictions hit/miss.
5. Checks 1–7.
6. Figure and one descriptive paragraph.
7. DECISION list (including 11-1, 11-2, 11-3 and anything else).
8. Not done (plan §9).
