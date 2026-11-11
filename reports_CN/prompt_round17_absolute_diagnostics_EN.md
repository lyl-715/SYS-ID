# Round 17 — how good is the information matrix in absolute terms, and how restrictive is the class?

## 0. Why, and rules

A reviewer's third objection: a certificate of near-optimality *within a class* does not say that the Fisher information matrix is good in absolute terms or well conditioned, and the class may be restrictive. This round (A) reports, for every design already in the repository, how much of the full-data information is kept in each parameter direction and how the matrix is conditioned; (B) measures what one restriction of the certified class — the window must stay at least 1 σ away from the last transmitted value — costs. Part C (a certificate for a class with a smaller minimum distance) is described but **on hold: do not run it unless told**.

- Create branch `round17-absolute-diagnostics` from the head of `round14-one-parameter-rule`; state that commit. Push only to the new branch. Rounds 4–16 are read-only; new code only in `evaluator/round17/`.
- First commit: this prompt into `reports_CN/`. Second commit: `round17/predictions_round17.md` (§6) before any computation.
- Input `U`, budget `B = 316.8921168493721`. Parts A and B run no DP.
- Assertion failures stop the run. Every choice left open is a `DECISION 17-x` line. Delete any empty folders created under `round4/`.

## 1. Part A — diagnostics of the information matrix, per design

**Designs and where their `J` comes from** (σ/24 design grid unless stated; each must reproduce its archived `log det J` to 1e-9 relative before its diagnostics are used):

| design | source | `log det J` to reproduce |
|---|---|---|
| full data | `J_full = HKᵀ HK / s²` | 219.00850592919505 |
| constant SoD at Δ* | `round4/results/J_U_SoD.npy` | 197.20097668224076 |
| round-4 table (half-widths) | re-evaluate `round4/results/table_U_W.npz` as Round 11's `inputs` stage did | 205.03141309913676 |
| `T₁` (edges, 100 actions) | `round11/results/table_edge_iter.npz` | 207.35521730257756 |
| `T′` (edges, 884 actions) | `round11/addendum/results/table_edge_Sprime.npz` | the σ/24 value in `round11b.json` (207.6467…) |
| one-parameter rule `R(37, 13)` | Round 14 | 207.5002944958199 |
| fixed threshold on the prediction residual, no SoD | `W_k = c_max(q)` for `k ≥ 2` with `q = (B − 1)/(K − 1)`, `W_1 = 1`, from `round10/ub.py` | `round10.json → budgets.B316.history[0].f` (208.9566…) |
| optimum over all receiver-known windows (class W) | `W_k = c_max(q̂_k)`, `q̂` from `solve(prob, B − 1)` | 209.0685398686887 |

**Diagnostics** for each design, with `I = log det J`:
1. `ΔI = I − I_SoD`, `I − I_full`, `S = Σ W_k`, `E[N]`.
2. Generalised eigenvalues `μ_1 ≤ … ≤ μ_9` of `J v = μ J_full v` (`scipy.linalg.eigh(J, J_full)`): all nine, with minimum, maximum and their ratio. `μ` is the fraction of the full-data information kept in that direction. Check: `exp(mean(log μ)) = exp((I − I_full)/9)` to 1e-10.
3. `cond(J)` (2-norm), `λ_min(J)`, `λ_max(J)`.
4. Per-parameter Cramér–Rao standard-deviation ratio to full data, `sqrt(diag(J⁻¹)/diag(J_full⁻¹))`: nine values, minimum, maximum; and the geometric mean `exp((I_full − I)/18)`.
5. `tr(J⁻¹)/tr(J_full⁻¹)` and `λ_min(J)/λ_min(J_full)`.

Reference values from a computation outside the repository: `cond(J_full) = 98.67923064906208`; constant SoD: `μ_min = 0.048517708610531125`, `μ_max = 0.16545799228159686`, `cond = 94.92214093489969` (to be matched to 1e-9). For the rule a Monte Carlo of the continuous version gave `μ` between 0.262 and 0.290 and `cond ≈ 99.2` (approximate; not a gate).

One table, one row per design: `ΔI`, `μ_min`, `μ_max`, `cond`, standard-deviation ratio (minimum, geometric mean, maximum). A second table with the nine `μ` of every design.

## 2. Part B — the minimum distance from the last transmitted value

The certified class requires the window to extend at least 1 σ beyond the last transmitted value on each side; the earlier class required 2 σ. Generalise Round 14's rule and `fit_rule` to a minimum half-width `M_min` (grid units):

```
L_i = max( M_min ,  i − FL[p] + n_p ),      U_i = max( M_min ,  CL[p] − i + n_p )
```

For `M_min ∈ {6, 12, 18, 24, 36, 48}` (0.25, 0.5, 0.75, 1, 1.5, 2 σ):
1. `fit_rule(B)` → `(n, j)`; `E[N]`, `S`, `log det J` on σ/24 and σ/120; the Part A diagnostics 2–4 on the σ/24 matrix.
2. Monte Carlo of the continuous rule with `lo = min(x − eσ, r − mσ)`, `hi = max(x + eσ, r + mσ)`, `m = M_min/24`: scan `e` in steps of 0.04 around the lattice rule's `e_eff` until `B` is bracketed, interpolate to `E[N] = B` (`e_fit`, `log det J` at `B`); `R = 10^4`, the same seeds for every `m`.

Check: `M_min = 24` gives Round 14's `R(37, 13)` bit for bit.

Reference values from outside (continuous rule, 4000 trajectories, another seed; not a gate):

| m (σ) | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|
| `e_fit` | 1.607 | 1.598 | 1.588 | 1.578 | 1.554 | 1.528 |
| `log det J` at `B` | 207.584 | 207.553 | 207.522 | 207.490 | 207.416 | 207.298 |

Report one table (lattice σ/120 and continuous Monte Carlo side by side) and the two differences `I(0.25σ) − I(1σ)` and `I(1σ) − I(2σ)`.

## 3. Part C — on hold

Do not run unless told. Description only, so that the cost can be judged: extend the union action set of Round 11b (`sprime.set_union()`, 884 actions) by two blocks with minimum distances 12 and 6 grid units and edge offsets `{1.2, 1.3, …, 2.2, inf}` (144 actions each, 1172 in total); with tangent point `T′` run one price scan, take the certificate `I(T′) − 9 + min_λ [dual(λ) + 0.25 λ]` for the enlarged class and the best table found. Estimated cost: about 9–10 min per DP run, 12 runs.

## 4. Checks

1. Every archived `log det J` of §1 reproduces to 1e-9 relative; the constant-SoD reference values of §1 reproduce to 1e-9.
2. The geometric-mean identity of diagnostic 2 holds for every design.
3. Part B: `M_min = 24` is bitwise Round 14's rule; every fitted rule has `E[N] ≤ B` on σ/24 and σ/120 and `j + 1` infeasible; forwards conserve mass within tolerance.
4. Round 10's `f` and `u_cert` at the nominal budget reproduce exactly.

## 5. Questions the report must answer (fixed before running)

- **Q1** For each design: what fraction of the full-data information is kept in the worst and in the best direction, and what is the condition number against the full-data one?
- **Q2** Is the gain of the designed triggers over constant SoD larger, equal or smaller in the worst direction than on average (ratio of `μ_min` against ratio of geometric means)?
- **Q3** Do the designed triggers leave the condition number within 5 % of the full-data value?
- **Q4** Does the information of the rule increase monotonically as the minimum distance decreases from 2 σ to 0.25 σ, and by how much from 1 σ to 0.25 σ?

Answer with numbers. No causal statements.

## 6. Predictions (commit before running)

```
- All reproductions pass.
- Q1/Q3: rule R(37,13): mu_min in [0.255, 0.270], mu_max in [0.283, 0.297], cond in [97, 101].
         T': mu_min >= 0.26, mu_max <= 0.31, cond in [97, 101].
         fixed residual threshold: all nine mu within 0.001 of 0.327.
         class-W optimum: all mu in [0.30, 0.37].
- Q2: larger in the worst direction: mu_min(rule)/mu_min(SoD) >= 5, against a ratio of geometric means of about 3.1.
- Per-parameter standard-deviation ratios: rule all in [1.80, 2.00]; constant SoD all in [2.4, 4.6].
- Q4: monotone; I(0.25 sigma) - I(1 sigma) in [0.05, 0.15] nat; I(1 sigma) - I(2 sigma) in [0.15, 0.25] nat (both grids and the Monte Carlo).
Scoring: a line is a hit iff every stated number falls in its range.
```

## 7. Outputs

- `round17/diag17.py`, `round17/run_round17.py`, `round17/results/round17.json`, logs.
- `reports_CN/report_round17_CN.md` (+ EN copy): (1) answers to Q1–Q4; (2) the two Part A tables; (3) the Part B table; (4) predictions hit/miss; (5) checks with quoted log lines; (6) DECISION list; (7) wall times; (8) not done (Part C on hold; one input; one budget).
- Runtime guide: Part A needs a handful of σ/24 forwards (about 20 s each; `T′` needs the `sprime` action set); Part B needs about 8 σ/24 forwards and one σ/120 forward (about 110 s) per `M_min`, plus a few minutes of Monte Carlo. About 40 min in total.
