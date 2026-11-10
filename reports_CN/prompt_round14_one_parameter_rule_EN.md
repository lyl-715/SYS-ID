# Round 14 — a one-parameter send-on-delta rule: reproduction, Monte Carlo check, budgets and inputs

## 0. Why, and rules

A reviewer's first objection to the window table is that it is too large to compute, store and send to a sensor. The table's shape suggests that a rule with one parameter does almost as well: the edge facing the predicted output sits a fixed number of σ beyond it, the other edge hugs the last transmitted value. A computation made **outside the repository** (round 11's `eval11.forward`, input `U`) gave 10.30 nat over constant SoD for such a rule, against 10.44 for the table `T′`. This round (1) reproduces that inside the repository and checks it by Monte Carlo, and (2) tests whether the rule's parameter is set by the transmission rate alone, across budgets and inputs.

- Create branch `round14-one-parameter-rule` from the current head of `round13-input-trigger`; state that commit in the report. Push only to the new branch. All earlier branches and rounds 4–13 are read-only; new code only in `evaluator/round14/`. This round does not depend on Part B of round 13 and must not touch its files.
- First commit: this prompt into `reports_CN/`. Second commit: `round14/predictions_round14.md` (§6), before any computation.
- Use `round13/eval13.py` (one process per input, `R13_INPUT=<name>`) and `round13/common13.py`. No DP is run in this round.
- Assertion failures stop the run. If a σ/24 forward stops on the mass-conservation tolerance with a deviation below 1e-11 and the tolerance decision of round 13 (σ/24 tolerance 1e-11) has not been applied to `eval13` yet, apply it now and record it. Non-negativity and zero mass at the grid ends stay strict.
- Every choice left open here is a `DECISION 14-x` line in the report. Delete any empty folders created under `round4/`.

## 1. The rule

**Continuous form (what a sensor would run).** With `x̄_k` the predicted output, `r` the last transmitted value, one parameter `e > 0`:

```
lower edge = min( x̄_k − e σ ,  r − σ )
upper edge = max( x̄_k + e σ ,  r + σ )
transmit iff  z_k ≤ lower edge  or  z_k ≥ upper edge
```

**Lattice form (what the evaluator computes).** On the σ/24 design grid, with `FL[p] = floor(u_p)`, `CL[p] = ceil(u_p)`, `u_p = (x̄ − Z0)/DZ` (`eval13.FL`, `eval13.CL`), an integer `n` and array index `p = 1 … K−1` (step `k = p + 1`), node `i` has half-widths

```
L_i = max( 24 ,  i − FL[p] + n ),      U_i = max( 24 ,  CL[p] − i + n )        (grid units; 24 = 1 σ)
```

This is exactly what `eval11._rows` produces for an edge action with offset `n` grid steps and minimum distance 24. The lower edge node is `FL[p] − n`, i.e. between `n` and `n + 1` grid steps below `x̄` (outward rounding), so the mean physical offset is `(n + ½)/24` σ.

**Mixed lattice rule `R(n, j)`.** One grid step in `n` changes `E[N]` by about 27 transmissions, so the budget cannot be met with a single integer. `R(n, j)`, `j ∈ {0, …, 20}`, uses offset `n` at the steps with `p mod 20 < j` and `n + 1` at the others. Its lattice parameter is `ē = (n + 1 − j/20)/24` σ and its mean physical offset is `e_eff = ē + 1/48` σ.

Implement the rule as an object with `rows_full(p)` returning the two `int64` arrays over the full grid, so that `eval13.forward(Z, DZ, XK, KA, KB, rule.rows_full, X_PAD, TOL24)` and `eval13.evaluate_refined(rule, 5)` evaluate it on σ/24 and on σ/120 (table semantics: half-widths of the coarse cell, multiplied by 5, window centred at the fine node).

**Fitting procedure `fit_rule(B)`** (deterministic, σ/24):
1. Find the integer `n` with `E[N](R(n, 20)) > B ≥ E[N](R(n, 0))` — `R(n, 20)` is offset `n` at every step, `R(n, 0)` is offset `n + 1` at every step. Start at `n = round(24·Φ⁻¹(1 − q*))`, `q* = (B − 1)/(K − 1)`, and step by ±1.
2. `j0 = floor( 20·(B − E[N](R(n,0))) / (E[N](R(n,20)) − E[N](R(n,0))) )`. Evaluate `R(n, j0)`; decrease `j` while infeasible; then evaluate `j + 1` to confirm it is infeasible. Return the largest `j` with `E[N] ≤ B` on σ/24.
3. Re-evaluate the returned rule on σ/120; `E[N]_{120} ≤ B` is required.

## 2. Part 1 — reproduction for input `U` at `B = 316.8921168493721`

**1a. Pure offsets.** `R(n, 20)` on σ/24, to be matched to 1e-9 relative (values from the outside computation, tolerance 1e-11):

| n | e = n/24 (σ) | E[N] | S | log det J |
|---|---|---|---|---|
| 36 | 1.5000 | 354.48311118855713 | 1500.472396458184 | 208.1697508229414 |
| 37 | 1.5417 | 326.06595206524514 | 1420.267400502641 | 207.67570557568936 |
| 38 | 1.5833 | 299.41680466967017 | 1341.9805960911076 | 207.1658140469431 |
| 40 | 1.6667 | 251.1909482065696 | 1191.7274066151504 | 206.0980178306077 |

**1b. The fitted rule.** `fit_rule(B)` must return `n = 37, j = 13`. Reference values for `R(37, 13)`:

| grid | E[N] | S | log det J | cond(J) |
|---|---|---|---|---|
| σ/24 | 316.7251743087337 | 1392.8421245551658 | 207.5002944958199 | 99.1267 |
| σ/120 | 316.4986983108747 | 1392.4592724251975 | 207.49781936247174 | 99.1269 |

Also report `R(37, 14)` on σ/24 (expected infeasible). If the numbers do not reproduce to 1e-9 relative, stop and report the differences; do not continue.

**1c. Monte Carlo of the lattice rule (runtime look-up).** Copy `eval11._mc_batch` with one change — the half-widths of the looked-up node are `Lm = max(i − FL[p] + n_p, 24)`, `Up = max(CL[p] − i + n_p, 24)` with `n_p` from `R(37, 13)` — everything else unchanged (nearest node with ties to the smaller index, window centred at the continuous `r`, `c` from `dr.cfun`, same random-number order, `R = 10^4` trajectories in 20 batches, seeds `20261109 + b`). Report z-scores of the σ/120 values of `E[N]`, `S`, `log det J` against the Monte Carlo means; pass rule `|z| ≤ 3` and no clipped look-ups (round 11's DECISION 11-7).

**1d. Monte Carlo of the continuous rule (no grid).** Simulate the continuous form of §1 directly: `lo = min(XK[p] − e·s, r − s)`, `hi = max(XK[p] + e·s, r + s)`, `c = dr.cfun((lo − XK[p])/s, (hi − XK[p])/s)`, transmit iff `z ≤ lo` or `z ≥ hi`. Same `R`, batches and seeds for every `e` (common random numbers), for `e ∈ {1.50, 1.54, 1.56, 1.58, 1.60, 1.64}`. Report `E[N]`, `S`, `log det J` with standard errors for each `e`; interpolate linearly in `e` to the value `e_C` where `E[N] = B`, and the corresponding `log det J`. Then run one more Monte Carlo at `e_C` rounded **up** to 0.005 σ and report it as the continuous rule's result (`E[N]` with its standard error against `B`; `ΔI` against constant SoD).
   Compare: `e_C` against the lattice rule's `e_eff = ē + 1/48`; and the continuous rule's `log det J` against the lattice rule's σ/120 value at (nearly) the same `E[N]`.

**1e. Context lines** (from the archives, quoted, not recomputed): constant SoD 197.20097668224076 at `E[N] = B`; `T′` 207.6467 (σ/24) / 207.6445 (σ/120), `E[N]_{120}` = 316.633; certificate at `B + 0.25`: 207.7146; full data `I_full` = 219.00850592919505, `cond(J_full)`, `cond(J_SoD)`. Derived for the rule: `ΔI` over constant SoD, loss against `T′`, gap to the certificate, `exp(−gap/9)`, and the geometric-mean standard-deviation ratio to full data `exp((I_full − I)/18)` for constant SoD, the rule and `T′`.

## 3. Part 2 — budgets and inputs

**2a. Budget sweep, input `U`.** `B ∈ {160, 240, 316.8921168493721, 400, 480}`. For each budget:
- `fit_rule(B)` → `(n, j)`, `ē`, `e_eff`, `E[N]`, `S`, `log det J` on σ/24 and σ/120.
- Matched constant SoD at that budget: `common13.matched_sod(B)` (σ/120 values; signed residual `E[N] − B`).
- The optimum over all window rules with receiver-known windows, and the constant prediction-centred threshold, from round 10's code: with `prob = ub.Problem(...)` built as `round10/run_round10.py` builds it for input `U`, report `solve(prob, B − 1)` → `f`, `u_cert` (class-W optimum) and `prob.fg(q_uniform)[0]` with `q_uniform = (B − 1)/(K − 1)` (same threshold at every step). For `B = 316.892` these must reproduce `round10.json → budgets.B316.f = 209.0685398686887`, `u_cert = 209.07220154478713`, and `history[0].f = 208.9566` (4 decimals).
- `q* = (B − 1)/(K − 1)`, `e_1 = Φ⁻¹(1 − q*)`, `δ = e_eff − e_1`.

One table, one row per budget: `B`, `q*`, `e_1`, `e_eff`, `δ`, `E[N]_{120}`, `I_rule`, `I_SoD` (matched), `ΔI = I_rule − I_SoD`, information per transmission `(S − 1)/(E[N] − 1)` for rule and SoD, `I` of the constant prediction-centred threshold, class-W optimum.

**2b. Inputs at the nominal budget.** For `SCH`, `P5514`, `LO`, `HI` (and `N` only if its Part A result exists in `round13/results/partA_N.json`), one process per input: `fit_rule(B)` → `(n, j)`, `e_eff`, `E[N]`, `S`, `log det J` on σ/24 and σ/120. From `round13/results/partA_<name>.json` read `I_full`, the matched SoD and the designed table `T*` (σ/120). Report per input: `e_eff`, `δ`, `I_rule`, `c_rule = I_rule − I_full`, `I_rule − I_table`, `I_rule − I_SoD`.

## 4. Questions the report must answer (fixed before running)

- **Q1** Does the rule reproduce (1a, 1b) and pass Monte Carlo (1c)?
- **Q2** Does the continuous rule (1d) meet the budget with `log det J` within 0.05 nat of the lattice rule's value?
- **Q3** Is `δ` small and of one sign: `0 < δ ≤ 0.10 σ` for all five budgets and all inputs?
- **Q4** Is `ΔI` over matched SoD at least 8 nat at every budget?
- **Q5** Across the inputs, is the range of `c_rule` at most 0.1 nat, and is the rule within 0.3 nat of the designed table of each input (it may be above: the rule uses a 1 σ minimum distance, the round-13 tables 2 σ)?

Answer each with yes/no and the numbers. No causal statements.

## 5. Checks

1. Every forward: mass conservation within tolerance, no negative mass, zero mass at the grid ends, no window edge outside the padded range.
2. Every fitted rule: `E[N]_{24} ≤ B` and `E[N]_{120} ≤ B`; `j + 1` infeasible on σ/24.
3. Monte Carlo (1c): `|z| ≤ 3`, zero clipped look-ups. Monte Carlo (1d): report standard errors; no pass rule beyond Q2.
4. Round-10 reproduction at the nominal budget (2a).
5. For `U` at the nominal budget the fitted rule of 2a is bitwise the rule of 1b.

## 6. Predictions (commit before running)

```
- 1a, 1b reproduce to 1e-9; fit_rule(B_nom) = (37, 13).
- 1c: |z| <= 3 on all three quantities.
- 1d: e_C in [1.56, 1.60]; continuous-rule log det within 0.05 nat of 207.498.
- 2a: delta in [0.02, 0.08] sigma at every budget; e_eff about 1.88, 1.70, 1.58, 1.45, 1.35 (each +-0.04) for B = 160, 240, 317, 400, 480;
      Delta I over matched SoD in [9, 12] nat at every budget; rule below the class-W optimum by 1.2 to 2.0 nat at every budget.
- 2b: range of c_rule <= 0.1 nat; I_rule - I_table in [-0.1, +0.3] nat for every input; delta in [0.02, 0.08] sigma.
Scoring: a line is a hit iff every stated number falls in its range.
```

## 7. Outputs

- `round14/rule14.py` (rule object, `fit_rule`, the two Monte Carlo routines), `round14/run_round14.py`, `round14/results/round14.json` (every number above), logs.
- `reports_CN/report_round14_CN.md` (+ EN copy): (1) answers to Q1–Q5, one line each; (2) the reproduction tables; (3) the two Monte Carlo tables; (4) the budget table; (5) the input table; (6) predictions hit/miss; (7) checks with quoted log lines; (8) DECISION list; (9) wall times; (10) not done.
- Runtime guide: a σ/24 forward about 20 s, a σ/120 forward about 110 s; `fit_rule` needs 5–8 σ/24 forwards and one σ/120 forward; about 1.5–2 h in total, inputs run one process at a time.

## 8. Not in this round

No DP, no new tables, no certificate at other budgets, no rule with more than one parameter, no model mismatch, no change to rounds 4–13.
