---
title: "Round 14 report: a one-parameter send-on-delta rule (reproduction, Monte Carlo, budgets, inputs)"
date: "2026-11-10"
---

English copy of `report_round14_CN.md`. Where the two differ, the Chinese version is the report of record.

**Prompt.** `reports_CN/prompt_round14_one_parameter_rule_EN.md`. Branch `round14-one-parameter-rule`, created from the head of `round13-input-trigger` at **`0d366ff`** (the Round 13 Part A report). Commits:
- `dbe5694`: the prompt (first commit);
- `8f992f1`: `round14/predictions_round14.md`, before any computation (second commit);
- `413e945`: code;
- `4736c89`: sequence-script fix (§8, DECISION 14-10);
- `880a7a4`: Part 1;
- `3328d3b`: budgets;
- `ed87ef4`, `2724ecf`, `9380bd5`, `c4b6709`, `1eecc21`: SCH, P5514, LO, HI, N;
- `4f5408f`: summary;
- this report: the commit after that.

**Scope.**
- System, noise, K = 5000 and the nominal budget B = 316.8921168493721 are as in Rounds 4–13.
- The evaluator is `round13/eval13.py`, with the σ/24 conservation tolerance 1e-11 already in place (Round 13 DECISION 13-30; the tolerance clause of prompt §0 therefore did not apply). No DP was run.
- I is log det J at θ₀. Quantities compared with Round 13 or with the matched SoD use σ/120. The σ/24 values are listed alongside (DECISION 14-1).
- No causes are claimed.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/`.
- `[L X:n]` is line n of `round14/logs/X.log`; `[J a.b]` is field `a.b` of `round14/results/round14.json`.
- Code: `round14/rule14.py` (rule object, `fit_rule`, both Monte Carlo routines), `round14/run_round14.py` (stages `part1`, `budgets`, `input`), `round14/summary14.py`.

## 1. Answers to Q1–Q5

Source of the five answers: [L summary:2–6].

- **Q1 — yes.** Every 1a and 1b number reproduces **exactly** (relative deviation 0); `fit_rule(B)` = (37, 13). The Monte Carlo of the lattice rule gives z = +0.08 (E[N]), +1.40 (S), +1.89 (log det J), with 0 clipped look-ups [L part1_U:2–14].
- **Q2 — yes.** The continuous rule at e = 1.580 σ (e_C = 1.5763 rounded up) has E[N] = 314.50 ± 0.18 ≤ B, and log det J = 207.4639. That is −0.034 nat from the lattice rule's σ/120 value, 207.4978 [L part1_U:21].
- **Q3 — yes.** δ = e_eff − e₁ lies in [0.035, 0.057] σ for all five budgets and all six inputs [J questions.Q3].
- **Q4 — yes.** ΔI over the matched constant SoD is 12.41, 11.20, 10.30, 9.47, 8.85 nat at B = 160, 240, 317, 400, 480, so every value is ≥ 8 [J questions.Q4].
- **Q5 — no.** The range of c_rule = I_rule − I_full over the six inputs is **0.117 nat > 0.1** (N −11.614 … HI −11.497). The second part of Q5 holds: the rule is within 0.3 nat of the designed table of every input, at +0.056 … +0.145 nat (above the table everywhere) [J questions.Q5].

## 2. Reproduction (input U, B = 316.8921168493721)

**1a. Pure offsets R(n, 20), σ/24** [L part1_U:2–5]. All relative deviations from the reference are 0.

| n | e = n/24 (σ) | E[N] | S | log det J | mass deviation |
|---|---|---:|---:|---:|---:|
| 36 | 1.5000 | 354.48311118855713 | 1500.472396458184 | 208.1697508229414 | 7.6e-13 |
| 37 | 1.5417 | 326.06595206524514 | 1420.267400502641 | 207.67570557568936 | 8.4e-13 |
| 38 | 1.5833 | 299.41680466967017 | 1341.9805960911076 | 207.1658140469431 | 8.3e-13 |
| 40 | 1.6667 | 251.1909482065696 | 1191.7274066151504 | 206.0980178306077 | 4.1e-13 |

**1b. Fitted rule.** `fit_rule(B)` started at n = round(24 Φ⁻¹(1 − q\*)) = 37 and needed no ±1 step. j₀ = 13, R(37, 13) is feasible, and R(37, 14) is infeasible. The result is **R(37, 13)**: ē = 1.55625 σ, e_eff = 1.57708 σ [L part1_U:6–13].

| grid | E[N] | S | log det J | cond(J) | deviation from reference |
|---|---:|---:|---:|---:|---|
| σ/24 | 316.7251743087337 | 1392.8421245551658 | 207.5002944958199 | 99.126697 | 0 / 0 / 0; cond 2.8e-6 |
| σ/120 | 316.4986983108747 | 1392.4592724251975 | 207.49781936247174 | 99.126927 | 0 / 0 / 0; cond 2.7e-5 |
| R(37, 14), σ/24 | 318.0608520286389 (> B, infeasible) | 1396.7630147487828 | 207.5255779539911 | — | — |

**1e. Context** (archives; cond values computed from archived or rebuilt J, DECISION 14-7) [L part1_U:22–24]:

| | σ/24 | σ/120 |
|---|---:|---:|
| constant SoD at E[N] = B (`round4/results/mc.json`) | 197.20097668224076 | — |
| T′ (`round11/addendum/results/round11b.json` Tprime) | 207.64665 | 207.64453 (E[N] 316.633) |
| certificate at B + 0.25 (round11b.json proposition5.UB) | 207.71459 | — |
| I_full | 219.00851 | — |
| rule R(37, 13) | 207.50029 | 207.49782 |
| ΔI rule over constant SoD | **10.299** | 10.297 |
| loss of the rule against T′ | 0.146 | 0.147 |
| gap of the rule to the certificate | 0.214 | 0.217 |
| exp(−gap/9) | 0.977 | 0.976 |
| geometric-mean SD ratio to full data exp((I_full − I)/18): SoD / rule / T′ | 3.359 / 1.895 / 1.880 | 3.359 / 1.896 / 1.880 |

cond(J_full) = 98.68, cond(J_SoD) = 94.92, cond(J_rule) = 99.13.

## 3. Monte Carlo

**1c. Lattice rule R(37, 13), runtime look-up** (10⁴ trajectories, 20 batches, seeds 20261109 + b) [L part1_U:14]:

| | Monte Carlo mean ± SE | σ/120 evaluator | z |
|---|---:|---:|---:|
| E[N] | 316.4844 ± 0.1822 | 316.4987 | +0.08 |
| S | 1392.3208 ± 0.0988 | 1392.4593 | +1.40 |
| log det J | 207.49682 ± 0.00053 (batch SE) | 207.49782 | +1.89 |

Clipped look-ups: 0. Pass (abs(z) ≤ 3).

**1d. Continuous rule** (no grid; the same seeds for every e) [L part1_U:15–21]:

| e (σ) | E[N] ± SE | S ± SE | log det J ± SE |
|---|---:|---:|---:|
| 1.50 | 369.245 ± 0.196 | 1541.416 ± 0.100 | 208.4116 ± 0.0006 |
| 1.54 | 341.060 ± 0.189 | 1463.486 ± 0.099 | 207.9450 ± 0.0005 |
| 1.56 | 327.600 ± 0.185 | 1425.137 ± 0.098 | 207.7062 ± 0.0006 |
| 1.58 | 314.496 ± 0.182 | 1387.263 ± 0.097 | 207.4639 ± 0.0005 |
| 1.60 | 301.900 ± 0.177 | 1349.833 ± 0.096 | 207.2180 ± 0.0005 |
| 1.64 | 277.726 ± 0.170 | 1276.388 ± 0.094 | 206.7149 ± 0.0007 |

- **e_C = 1.57634 σ.** This is the linear interpolation between 1.56 and 1.58; the interpolated log det J there is 207.5082.
- **Rounded up to 0.005 σ, e = 1.580.** This is a value already in the list. With common random numbers the "one more Monte Carlo" is identical to the e = 1.58 row (DECISION 14-6).
- **Continuous rule's result:** E[N] = 314.496 ± 0.182, which is B − 2.40 (z = −13.2); log det J = 207.4639; ΔI over constant SoD = 10.263 nat.
- **Comparison:** e_C − e_eff(lattice) = −0.0007 σ. The continuous rule's log det J at e = 1.58 minus the lattice σ/120 value is −0.034 nat; the value interpolated to e_C minus the lattice value is +0.010 nat.

## 4. Budget sweep (input U)

All quantities are on σ/120 except E[N]₂₄. I_SoD is the matched constant SoD (`common13.matched_sod`). "Const. threshold" is round 10's `prob.fg(q_uniform)[0]`. "Class W" is round 10's `solve(prob, B − 1)` value f, with u_cert in brackets [L budgets_U:42, 66, 85, 107, 134].

| B | q\* | e₁ | rule | e_eff | δ | E[N]₁₂₀ | I_rule | I_SoD (M\*, residual) | ΔI | info/send rule / SoD | const. threshold | class W (u_cert) | class W − rule |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|
| 160 | 0.03181 | 1.8549 | R(44, 3) | 1.8896 | +0.0347 | 159.538 | 203.2403 | 190.8345 (955, +0.037) | 12.406 | 5.464 / 1.458 | 204.6585 | 204.8048 (204.8112) | 1.565 |
| 240 | 0.04781 | 1.6665 | R(40, 9) | 1.7104 | +0.0439 | 239.051 | 205.7923 | 194.5949 (637, −0.148) | 11.197 | 4.835 / 1.465 | 207.2520 | 207.3782 (207.3855) | 1.586 |
| 316.892 | 0.06319 | 1.5285 | R(37, 13) | 1.5771 | +0.0486 | 316.499 | 207.4978 | 197.2012 (480, −0.001) | 10.297 | 4.410 / 1.473 | 208.9566 | 209.0685 (209.0722) | 1.571 |
| 400 | 0.07982 | 1.4063 | R(34, 9) | 1.4604 | +0.0541 | 398.901 | 208.8570 | 199.3901 (376, +0.195) | 9.467 | 4.068 / 1.481 | 210.3321 | 210.4319 (210.4378) | 1.575 |
| 480 | 0.09582 | 1.3057 | R(32, 16) | 1.3625 | +0.0568 | 479.288 | 209.9034 | 201.0545 (310, −0.548) | 8.849 | 3.802 / 1.483 | 211.3709 | 211.4610 (211.4647) | 1.558 |

- The rule lies 1.56–1.59 nat below the class-W optimum at every budget. It also lies 1.42–1.48 nat below round 10's constant prediction-centred threshold (a value of the class-W model).
- δ grows with B, from +0.035 to +0.057.
- The matched SoD was accepted at every budget (abs(residual) ≤ 1). At B = 160 its threshold, M\* = 955 (39.8σ), is just below the clipping limit 958, so no clipped node was needed.

## 5. Inputs at the nominal budget

I_full, I_table (T\*) and I_SoD are from `round13/results/partA_<name>.json` (σ/120) [L input_X:6–7]; [J input_table].

| input | rule | e_eff | δ | E[N]₁₂₀ | I_rule | c_rule = I_rule − I_full | I_rule − I_table | I_rule − I_SoD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U | R(37, 13) | 1.5771 | +0.0486 | 316.499 | 207.4978 | −11.511 | +0.145 | 10.297 |
| SCH | R(37, 14) | 1.5750 | +0.0465 | 316.606 | 207.4731 | −11.535 | +0.107 | 9.622 |
| P5514 | R(37, 14) | 1.5750 | +0.0465 | 316.079 | 207.4540 | −11.554 | +0.118 | 9.531 |
| LO | R(37, 14) | 1.5750 | +0.0465 | 315.865 | 206.4899 | −11.544 | +0.118 | 10.517 |
| HI | R(37, 12) | 1.5792 | +0.0506 | 316.585 | 206.5372 | −11.497 | +0.140 | 9.746 |
| N | R(37, 19) | 1.5646 | +0.0361 | 316.119 | 206.8028 | −11.614 | +0.056 | 10.141 |

- The range of c_rule is 0.117 nat; over the five inputs without N it is 0.057 nat. For comparison, the range of c_tab of the designed tables in Round 13 was 0.035 nat.
- All six fitted rules use n = 37 and differ only in j (12–19).

## 6. Predictions (`round14/predictions_round14.md`, `8f992f1`)

**3 hits, 2 misses** [L summary:18–23].

| line | values | result |
|---|---|---|
| 1a, 1b reproduce to 1e-9; fit = (37, 13) | all deviations 0; cond within 2.8e-6 / 2.7e-5; (37, 13) | hit |
| 1c abs(z) ≤ 3 | +0.08, +1.40, +1.89 | hit |
| 1d e_C ∈ [1.56, 1.60]; continuous log det within 0.05 of 207.498 | e_C 1.5763; 207.4639 (−0.034) | hit |
| 2a δ ∈ [0.02, 0.08]; e_eff 1.88/1.70/1.58/1.45/1.35 ± 0.04; ΔI ∈ [9, 12]; class W − rule ∈ [1.2, 2.0] | δ 0.035–0.057 ✓; e_eff 1.890/1.710/1.577/1.460/1.363 ✓; **ΔI 12.41 at B = 160 and 8.85 at B = 480 outside [9, 12]**; class W − rule 1.56–1.59 ✓ | **miss** |
| 2b range c_rule ≤ 0.1; I_rule − I_table ∈ [−0.1, +0.3]; δ ∈ [0.02, 0.08] | **range 0.117**; +0.056 … +0.145 ✓; δ 0.036–0.051 ✓ | **miss** |

## 7. Checks (prompt §5)

1. **Every forward** passed the asserted conservation (σ/24 1e-11, σ/120 1e-10), non-negativity, zero mass at the grid ends, and window edges inside the padded range. The largest deviations are 9.6e-13 (σ/24) and 1.52e-12 (σ/120; the B = 160 rule) [J checks.1_forwards].
2. **Every fitted rule** (11 fits: 1b, the five budgets of 2a and the five inputs of 2b; the nominal fit of 2a is the same rule as 1b) has E[N]₂₄ ≤ B and E[N]₁₂₀ ≤ B, and j + 1 is infeasible on σ/24 [J checks.2_fitted_rules].
3. **Monte Carlo 1c:** abs(z) ≤ 3 and 0 clipped look-ups (§3). 1d: standard errors are reported.
4. **Round 10 reproduction at the nominal budget:**
   ```
   [L budgets_U:84] CHECK4 round 10 at the nominal budget: f 209.0685398686887 (archive 209.0685398686887, rel 0.0e+00); u_cert 209.07220154478713 (archive 209.07220154478713, rel 0.0e+00); history[0].f 208.95657763962944 (208.9566 at 4 dp: True) -> PASS
   ```
5. **The nominal fit of 2a is bitwise the rule of 1b:**
   ```
   [L budgets_U:73] CHECK5 nominal budget: fitted rule vs 1b {'n_j': True, 'rec24_bitwise': True, 'rec120_bitwise': True, 'pass_': True}
   ```

Quoted reproduction lines:
```
[L part1_U:2]  1a R(36, 20) e = n/24 = 1.5000: EN 354.48311118855713 S 1500.472396458184 logdet 208.1697508229414; rel 0.0e+00 / 0.0e+00 / 0.0e+00; mass dev 7.64e-13 (20.4 s) -> PASS
[L part1_U:11] 1b R(37, 13) sigma/24: EN 316.7251743087337 S 1392.8421245551658 logdet 207.5002944958199 cond 99.12669719075502; rel 0.0e+00 / 0.0e+00 / 0.0e+00, |cond - ref| 2.8e-06 -> PASS
[L part1_U:12] 1b R(37, 13) sigma/120: EN 316.4986983108747 S 1392.4592724251975 logdet 207.49781936247174 cond 99.12692721625616; rel 0.0e+00 / 0.0e+00 / 0.0e+00, |cond - ref| 2.7e-05 -> PASS
[L part1_U:14] 1c MC lattice R(37, 13): EN 316.4844 (0.1822) S 1392.320798611614 (0.099) logdet 207.4968247522991 (0.0005); z(sigma/120) EN +0.08 S +1.40 logdet +1.89; clipped 0; 3.3 s -> PASS
```
`git status` on rounds 4–13 and on the frozen files was clean after every stage, and no empty folder was created under `round4/` [L X: CLEANUP lines].

## 8. Where the prompt and the code differ

1. **§1: "This is exactly what `eval11._rows` produces for an edge action with offset n and minimum distance 24."** `eval11._rows` (and its copy in `eval13`) hard-codes a minimum distance of 48 grid units (`if Lm < 48: Lm = 48`), so it cannot produce the 24-unit minimum. The rule therefore implements the formula of §1 directly in `rule14.Rule.rows_full`, without `_rows`. The bitwise agreement with the outside computation (1a, 1b) shows it used the same formula.
2. **§2 1d: "one more Monte Carlo at e_C rounded up to 0.005 σ".** Here that value is 1.580, which is already in the e list. With the prescribed common random numbers, that run is identical to the e = 1.58 row.
3. **§0, tolerance clause.** It did not apply: `eval13` already has TOL24 = 1e-11 (Round 13, DECISION 13-30), and no forward came near it.
4. **§2 1e: cond(J_full) and cond(J_SoD) are "quoted, not recomputed".** The archives store no condition numbers. They were computed from J_full = HKᵀHK/σ² and from the archived J_SoD matrix (DECISION 14-7).

## 9. DECISION list

- **DECISION 14-1.** Scoring conventions and grids were fixed in `predictions_round14.md` before running:
  - quantities compared with the matched SoD, with Round 13's tables and with round 10's class-W optimum use σ/120;
  - "class-W optimum" means round 10's f, with u_cert reported alongside;
  - Q2's "meets the budget" means the point estimate is ≤ B;
  - Q3 and the 2b line use all six inputs, U included.
- **DECISION 14-2.** The rule is implemented directly from the §1 formula (`rule14.Rule`), not through `eval11._rows` (§8 item 1). σ/24 values come from `eval13.evaluate` (= `eval13.forward` with TOL24 + `record`), σ/120 values from `eval13.evaluate_refined(rule, 5)`.
- **DECISION 14-3.** cond(J) is the 2-norm condition number (`np.linalg.cond`). It matches the references to 4 decimals.
- **DECISION 14-4.** `fit_rule`:
  - n₀ uses Python `round`;
  - R(n, 0) is evaluated as R(n + 1, 20), since the half-widths are identical;
  - j₀ is clipped to [0, 19];
  - if R(n, j₀) is feasible, j increases while R(n, j + 1) is feasible. This branch was never triggered: in all 11 fits R(n, j₀) was feasible and R(n, j₀ + 1) infeasible.
  - At B = 160 and 480 the starting n needed one ±1 step.
- **DECISION 14-5.** Monte Carlo runs on a Pool of 4 workers, one batch of 500 trajectories per seed, and aggregates as `eval11.monte_carlo` (mean ± SE; log det J of the mean weights; batch SE of log det J). The continuous routine has no look-up and so no clipping.
- **DECISION 14-6.** e_C is the linear interpolation of the Monte Carlo E[N] between the two e values that bracket B; log det J is interpolated the same way. Rounding up is e_up = ⌈e_C / 0.005⌉ · 0.005 = 1.580; the run is the same as the e = 1.58 row and is reported once.
- **DECISION 14-7.** 1e:
  - ΔI, loss against T′, gap and SD ratios are reported on both grids. The certificate is a σ/24 quantity.
  - cond(J_full) is from HK.
  - cond(J_SoD) is from `round13/results/gate_U_design/J_sod_reference_s24.npy`, which is bitwise the accepted operator's J at Δ\* (Round 13 G2).
- **DECISION 14-8.** 2a:
  - class-W optimum: `ub.solve(ub.Problem(HK, σ, c_max, dc_max), B − 1, tol = 0.01)` with a fresh `Problem` per budget, as round 10 builds it;
  - constant prediction-centred threshold: `fg(q_uniform)[0]` with its own `Problem` instance;
  - matched SoD: `common13.matched_sod(B)`, with its J matrices in `round14/results/budgets/`.
- **DECISION 14-9.** 2b reads I_full, I_table and I_SoD from `round13/results/partA_<name>.json → derived` (σ/120). The U row combines Part 1 with `partA_U.json`.
- **DECISION 14-10.** Stages run one process at a time with 4 numba threads, and each stage is committed and pushed when it finishes. The first launch of the sequence failed before any computation, because the shell redirect needed `logs/`, which only the Python driver creates. This was fixed in `4736c89` and the sequence was relaunched.
- **DECISION 14-11.** Rounds 4–13 were not modified. Importing `eval13` / `common13` may write numba compiled-code cache files under `round13/__pycache__`, which is git-ignored.

## 10. Wall times

- σ/24 forward of a rule: 16–25 s (U 20 s, HI 16 s, N 24–25 s).
- σ/120 forward of a rule: 90–144 s (U 115 s).
- Constant-SoD forwards (matched SoD): about 5 s on σ/24 and 25 s on σ/120.
- One Monte Carlo of 10⁴ trajectories (4 workers): 2.5–3.3 s.
- Round 10 `solve`: 4–6 iterations, under 0.1 s.
- Stages: Part 1 298 s; budgets 1681 s; inputs SCH 191 s, P5514 181 s, LO 222 s, HI 155 s, N 218 s. About 50 min in total.

## 11. Not done

- Part B of Round 13 (as instructed); no DP; no new tables; no certificate at other budgets; no rule with more than one parameter; no model mismatch (prompt §8).
- No Monte Carlo of the fitted rules at other budgets or other inputs; the prompt asks for 1c and 1d at the nominal budget and input U only.
