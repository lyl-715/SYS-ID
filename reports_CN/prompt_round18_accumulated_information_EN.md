# Round 18 — accumulated information, the non-transmissions, and the estimator

## 0. Why, and rules

A reviewer's fourth objection: the value of a new sample depends on what has already been collected, so why not put the accumulated information matrix in the state, or use a recursive-least-squares formulation, instead of a surrogate with fixed weights. This round measures three things by Monte Carlo: (A) how much the information accumulated in a single experiment fluctuates, and how much of the information sits in the non-transmissions; (B) what happens when the estimator uses only the transmitted samples (least squares) instead of the censored likelihood; (C) whether a rule that adapts its threshold to the *running* information does better than one that uses the *terminal* weights, or than no adaptation.

- Create branch `round18-accumulated-information` from the head of `round15-rule-estimated-model`; state that commit. Push only to the new branch. Rounds 4–17 are read-only; new code only in `evaluator/round18/`.
- First commit: this prompt into `reports_CN/`. Second commit: `round18/predictions_round18.md` (§6) before any computation.
- Input `U`, true model, `B = 316.8921168493721`, `I_SoD = 197.20097668224076`. No DP, no grid evaluator except where a check says so.
- Assertion failures stop the run. Every choice left open is a `DECISION 18-x` line. Delete any empty folders created under `round4/`.

## 1. Triggers used in this round

All simulated in continuous form (no grid), vectorised over trajectories as `eval11._mc_batch`; first step always transmitted; `x_k = h_kᵀθ*`; `c = dr.cfun((lo − x_k)/σ, (hi − x_k)/σ)` summed over its two returned parts.

| name | window |
|---|---|
| `SOD` | constant SoD at Δ*: `lo = r − Δ*`, `hi = r + Δ*` |
| `RULE` | Round 14's continuous rule: `lo = min(x_k − e σ, r − σ)`, `hi = max(x_k + e σ, r + σ)`, `e = 1.580` |
| `RULE_FB` | `RULE` with Round 15's selected feedback `VF(100, 3)` |
| `PRED` | no SoD structure: `lo = x_k − c σ`, `hi = x_k + c σ` with `c = Φ⁻¹(1 − q/2)`, `q = (B − 1)/(K − 1)` (about 1.858) |

## 2. Part A — realised information and the value of the non-transmissions

For each trigger, `R = 10^4` trajectories (10 batches, seeds `20261115 + b`). Per trajectory accumulate two 9×9 matrices:

```
J_cond = σ⁻² ( h_1 h_1ᵀ + Σ_{k≥2} c_k h_k h_kᵀ )          c_k evaluated at the trajectory's own reference value
J_sent = σ⁻² Σ_{k transmitted} h_k h_kᵀ
```

`J_cond` is the conditional information accumulated along that run (its mean over runs is the Fisher information `J` of the experiment); `J_sent` is what an estimator that treats the transmitted samples as ordinary samples and ignores the non-transmissions would count.

Report per trigger:
1. `log det` of the mean of `J_cond` (= `I`), its `ΔI` over `I_SoD`; mean, standard deviation, 5 % and 95 % of the per-trajectory `log det J_cond`; the difference `I − mean(log det J_cond)`; the correlation of `log det J_cond` with `N` across trajectories.
2. `log det` of the mean of `J_sent`; `I − log det mean(J_sent)` (the value of the non-transmissions, in nat); total weight `S` against `E[N]`.
3. `E[N]` and its standard deviation.

Reference values from a computation outside the repository (2000 trajectories, another seed, `RULE` at `e = 1.5768`; not a gate):

| | `I` | mean per-run `log det` | sd | `I −` mean | `log det mean(J_sent)` |
|---|---|---|---|---|---|
| `RULE` | 207.503 | 207.501 | 0.063 | 0.002 | 194.179 |
| `SOD` | 197.194 | 197.160 | 0.263 | 0.034 | 193.448 |

Check: `I` of `SOD` and of `RULE` agree with the archives (197.20097668224076; Round 14's continuous rule at `e = 1.580`) within Monte Carlo error; `PRED` agrees with Round 10's `history[0].f = 208.9566` within Monte Carlo error.

## 3. Part B — least squares on the transmitted samples against censored maximum likelihood

For `SOD`, `RULE` and `PRED`: `N_rep = 500` complete experiments each (seeds `20261116 + i`, the same noise for the three triggers). For every experiment compute two estimates of `θ`:

- **LS**: ordinary least squares on the transmitted pairs `(h_k, z_k)` only.
- **CML**: the censored maximum likelihood of rounds 8/8b (`round8/estimator_r8.py`: `StageData`, `Objective`, `run_lbfgsb`, acceptance rule of `round8b/estimator_r8b.py`), with the record's windows `(lo_k, hi_k)` at the non-transmitted steps. The windows are those the receiver can reconstruct: from `r` and, for `RULE` and `PRED`, from the *design* output `x_k` (the model known to the receiver, here the true one). They are fixed numbers in the likelihood; they do not move with the `θ` being estimated. Initial value: the LS estimate.

Report per trigger and estimator, with `J` the Fisher information from Part A:
1. Bias: per-parameter mean error divided by its Cramér–Rao standard deviation `sqrt(diag(J⁻¹))`, with its Monte Carlo standard error; the largest absolute value; and the Mahalanobis norm of the mean error, `sqrt(b̄ᵀ J b̄)`.
2. Root-mean-square error per parameter divided by the Cramér–Rao standard deviation: nine values, minimum, maximum.
3. `D̂ = −log det` of the mean-square-error matrix (errors about the true `θ`), against `I`; and `−log det` of the sample covariance (errors about their mean).
4. For CML: number of runs not accepted by the round-8b rule.

Check: for `SOD` the CML numbers are consistent with rounds 8/8b (errors attain the Cramér–Rao bound: `D̂` within 0.3 nat of `I`).

## 4. Part C — a rule driven by the running information

One-sided edge for a direction weight: for `ρ > 0` let `E(ρ) = argmax over e ∈ [0.5, 4.0] (step 0.005) of ρ · c_one(e) − q_one(e)`, with `q_one(e) = Φ(−e)`, `c_one(e) = Φ(−e) + e φ(e) + φ(e)²/Φ(e)` (the information fraction and transmission probability of a window with one edge `e σ` beyond the output and the other far away). Tabulate `E` once; check that it is non-increasing in `ρ`.

Reference information `J_ref` = the mean `J_cond` of `RULE` from Part A; `W_ref` its per-step weights, `S_ref = Σ W_ref`, `p = 9`.

Three variants of the rule (same window formula as `RULE`, only `e_k` differs), each tuned to the budget by bisection on a price `λ` so that the Monte Carlo `E[N] = B` within 0.3 (`R = 2000`, noise seed `20261117` for every run):

- **C0 — no adaptation.** `e_k = e` constant, `e` fitted to the budget (Round 14).
- **C1 — terminal weights.** `γ_k = σ⁻² h_kᵀ J_ref⁻¹ h_k` (fixed numbers, the same for every trajectory); `e_k = E(γ_k / λ)`.
- **C2 — running information.** Per trajectory, with the information accumulated so far `M_k = ε J_ref + σ⁻² ( h_1 h_1ᵀ + Σ_{2≤j<k} c_j h_j h_jᵀ )`, `ε = 0.02`, and `S_k = ε S_ref + 1 + Σ_{2≤j<k} c_j`: running weight `g_k = σ⁻² h_kᵀ M_k⁻¹ h_k`, relative weight `g_k S_k / p` (it averages about 1), and `e_k = E( (g_k S_k / p) · (p / S_ref) / λ )`.

The receiver can reproduce `M_k` (it knows the windows and the transmitted values), so `C2`'s windows are known to it.

Report for each variant: `λ`, `E[N]`, `I = log det` of the mean `J_cond`, `ΔI`, mean and standard deviation of the per-trajectory `log det J_cond`, `S`, and the 5 %, 50 %, 95 % of `e_k` over (trajectory, step). Differences `I(C1) − I(C0)`, `I(C2) − I(C1)` with a paired standard error from ten repetitions with noise seeds `20261117 + b` (λ fixed at the tuned value).

Check: every `I` is below the certificate of the class with minimum distance 1 σ, 207.7146 (Round 11b), up to Monte Carlo error; if one exceeds it by more than three standard errors, stop and report.

## 5. Questions the report must answer (fixed before running)

- **Q1** By how much does the information accumulated in a single experiment fluctuate under the rule and under constant SoD (standard deviation of the per-run `log det`), and how large is `I −` mean per-run `log det`?
- **Q2** How many nat of each trigger's information are in the non-transmissions?
- **Q3** Is least squares on the transmitted samples biased under `RULE`, `PRED`, `SOD` (largest bias in Cramér–Rao standard deviations), and how far is its `D̂` below `I`?
- **Q4** Does the censored maximum likelihood attain the Cramér–Rao bound under `RULE` and `PRED` (`D̂` within 0.3 nat of `I`, largest bias below 0.2 standard deviations)?
- **Q5** Does the rule with running information (`C2`) beat the rule with terminal weights (`C1`)? Does `C1` beat no adaptation (`C0`)? By how much?

Answer with numbers. No causal statements.

## 6. Predictions (commit before running)

```
- Part A: sd of per-run log det: RULE in [0.04, 0.09], SOD in [0.20, 0.33]; I - mean: RULE <= 0.005, SOD in [0.02, 0.05].
          Value of the non-transmissions: RULE in [12.8, 13.8] nat, SOD in [3.3, 4.1] nat, PRED in [14.0, 15.5] nat.
- Part B: LS under RULE: largest bias above 3 Cramer-Rao standard deviations; D-hat more than 10 nat below I.
          LS under SOD: largest bias above 0.5 standard deviations.
          CML under RULE and PRED: largest bias below 0.2 standard deviations; D-hat within 0.3 nat of I; at most 2 % of runs not accepted.
- Part C: I(C1) - I(C0) in [0.00, 0.20] nat; I(C2) - I(C1) in [-0.20, +0.05] nat.
Scoring: a line is a hit iff every stated number falls in its range.
```

## 7. Outputs

- `round18/acc18.py`, `round18/run_round18.py`, `round18/results/round18.json`, logs.
- `reports_CN/report_round18_CN.md` (+ EN copy): (1) answers to Q1–Q5; (2) Part A table; (3) Part B tables (bias, RMSE ratios, `D̂`) for the three triggers and two estimators; (4) Part C table; (5) predictions hit/miss; (6) checks with quoted log lines; (7) DECISION list; (8) wall times; (9) not done (one input, one budget, true model; no recursive estimator was built).
- Runtime guide: Part A a few minutes; Part B 1500 experiments × two estimators, about 10–20 min; Part C needs a batched 9×9 inverse per step for `C2` (about 10 s per run of 2000 trajectories) and about 10 runs per bisection; under one hour in total.
