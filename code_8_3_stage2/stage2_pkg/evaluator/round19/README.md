Starting branch: `round18-accumulated-information` at commit `345b520` (the Round 18 report). This is the most recent branch that has both the one-parameter rule (`round14/rule14.py`, `round18/acc18.py` RULE) and the count-feedback code (`round15/mismatch15.py` VF, `round18/acc18.py` RULE_FB). Branch `round19-two-stage-rule`; number 19 was free.

# Round 19: complete two-stage experiment with the one-parameter rule

The prompt is in `reports_CN/prompt_round19_two_stage_rule_EN.md`. The folder `round19/` is `code_8_3_stage2/stage2_pkg/evaluator/round19/`, next to rounds 10–18. All paths below are relative to that folder.

## 0. Conclusion

| configuration | R | log det J | gain over constant SoD (95 % bootstrap) | mean N (sd, max) | fraction N > B |
|---|---:|---:|---|---|---:|
| **primary**, K1 = 2500 | 1000 | 202.805 | **+5.605** [+5.581, +5.631] | 315.20 (3.52, 330) | **0.316** |
| K1 = 1250 | 300 | 203.868 | +6.670 [+6.438, +6.884], **budget not kept** | **555.6 (233.7, 1209)** | **0.930** |
| K1 = 3750 | 300 | 200.909 | +3.711 [+3.678, +3.744] | 313.66 (2.63, 322) | 0.100 |
| control: K1 = 2500, true model in stage 2 | 300 | 203.110 | +5.913 [+5.874, +5.949] | 314.74 (1.71, 319) | 0.147 |
| baseline: constant SoD throughout | 1000 | 197.200 | 0 (reference) | 316.91 (2.34, 324) | 0.560 |

**Primary, K1 = 2500**
- The model is estimated from the first half and the rule runs the second half. The whole experiment then gains **+5.61 nat** of log det J over constant SoD throughout. That is 4.7 nat less than the known-model +10.30.
- The control separates the two costs. Model error costs **0.30 nat** (control minus primary on the same 300 replications: +0.297 [+0.262, +0.332]). The first stage costs the rest, 10.30 − 5.91 = 4.39 nat.
- **Budget:** the mean total count is below B (315.2), but **31.6 % of replications exceed B**.

**K1 = 1250: the budget is not kept (reported, not repaired)**
- A 1.25-s SoD record gives a stage-1 model whose prediction over stage 2 is far off. The median RMS prediction error is 3.5 σ (post hoc: 0.28 σ on the fitted stage-1 steps, 4.0 σ over stage 2).
- The feedback then saturates: e_k reaches 4 in 269 of 300 replications.
- Transmissions run away: mean 556, up to 1209. 93 % of replications exceed B, and 117 have more than twice the stage-2 plan.
- Its +6.67 nat is therefore not a budget-respecting number.

**Comparison with the earlier two-stage experiment**
- Its headline 3.67 nat (round 8b, "old table") used the **empirical measure** ΔD̂, the change in −log det of the empirical MSE matrix. It did not use log det J.
- On that measure this round's primary gives **ΔD̂ = +5.44 [+5.01, +5.84]**. The log det J gain is +5.61.
- The two experiments are not the same design (§5.1).

**Predictions:** prompt author 3 of 7 lines hit; mine 6 of 11 (§6).

## 1. Commits, seeds, run

**Commits.** None of the commits below was force-pushed.

| commit | content |
|---|---|
| `e3ef9ff` | prompt into `reports_CN/` |
| `313b2c3` | **`predictions.md`**: prompt §7 verbatim, scoring conventions, my own predictions (labelled), procedure, seeds. A separate commit, before any round-19 code existed |
| `6f99b62` | code: `two_stage19.py`, `run_round19.py`, `summary19.py`, sequence script |
| `2b17f5c` | stage `impl`: implementation checks |
| `5ec2dfe` | stage `checks`: prompt §5 (a), (b) |
| `3e543d9` | stage `main` |
| `6c358de` | stage `summary` |
| `363bb32` | `posthoc19.py`: descriptive diagnostics written after the summary from the saved arrays; no new simulation, no variant |
| this README | the commit after `363bb32` |

**Seeds** (also fixed in `predictions.md` §4):

| use | seed |
|---|---|
| noise of replication i (same for every configuration and the baseline) | `np.random.default_rng(190000000 + i).standard_normal(K)`; value 0 = noise of z_1, value p = step p + 1 |
| stage-1 CML restart | `191000000 + i` |
| final (joint) CML restart | `192000000 + i` |
| checks (a), (b): trajectory i, i = 0..9999 | `193000000 + i` |
| bootstrap (2000 resamples) | `194000000` |
| implementation checks only | `195000000 + …` |

**Replications.**
- Primary: i = 0..999.
- Secondary, control: i = 0..299, i.e. the primary's first 300 noise sequences (paired).
- Baseline: i = 0..999.
- The primary was **not** reduced: the whole `main` stage took 147 s (`logs/main.log`).

**Environment.**
- Python 3.13.18, numpy 2.5.3, scipy 1.18.1, numba 0.68.1. These are the versions round 18 recorded; numba was installed in this container at that version.
- 4 workers.

**Raw results.**
- `results/rows.csv`: one row per replication and configuration (2900 rows). Columns:
  - counts: N_1, N_2, N, over_B, plan_2;
  - stage-2 threshold: e_k mean, min, max; whether u_k reached 0.5 or 4; the abnormal-count flag;
  - prediction error of θ̃: RMS (all K), RMS (stage 2), max;
  - for the stage-1 and final CML: convergence flag, used call, fallback, first-call iterations, Newton decrement, message;
  - per-run log det J_cond;
  - θ̃ (9 values) and the final estimate (9 values).
- `results/main.npz`: per-run J_cond, mean weights w̄_k, θ̃ and final estimates.
- `results/round19.json`: all statistics.
- Logs `logs/*.log`; the text tables are in `logs/summary.log`.

## 2. What was run

**Stage 1 (steps 1..K1).** Constant SoD with Δ* (`acc18.window` 'SOD').

**θ̃.** `estimator_r8b.estimate_stage1`, unchanged. It is fitted to the stage-1 record (transmitted pairs plus the windows of the silent steps, reconstructed by the receiver):
- LS start;
- one restart;
- the round-8b Newton-decrement acceptance;
- fallback θ_LS.

**Stage 2 (steps K1+1..K).**
- Window: lower = min(x̄_k − e_kσ, r_k − σ), upper = max(x̄_k + e_kσ, r_k + σ), with x̄_k = h_kᵀθ̃ (control: x̄_k = x_k).
- Threshold: e_k = clip(e_0 + (N_k − (N_1 + q_2(k − K1)))/(2 q_2 T), 0.5, 4), with e_0 = 1.577, T = 100, q_2 = (B − 3 − N_1)/(K − K1).
- r_k carries over from stage 1. No fallback.
- This is `acc18.window` 'RULE_FB' / `mismatch15.run_job` with x_k → x̄_k and plan 1 + q_t(k − 1) → N_1 + q_2(k − K1). The two forms are the same floating-point expression when K1 = 1.

**Final estimate.** `estimator_r8b.estimate_joint` on the whole record, unchanged: start θ̃, one restart, the same acceptance, fallback θ̃. The windows are reconstructed by the receiver:
- SoD for k ≤ K1;
- for k > K1, the rule with x̄ from θ̃ and e_k from the counts in the record.

The baseline's final estimate is the same call on its own SoD record, from the same θ̃. Its stage-1 record is identical to the two-stage one.

**Information.**
- J = σ⁻² Σ_k w̄_k h_k h_kᵀ, where w̄_k is the replication mean of c_k = sum(`dr.cfun`((lo_k − x_k)/σ, (hi_k − x_k)/σ)), with the true x_k and the realised window; c_1 = 1.
- Gain = log det J(config) − log det J(baseline) on the same replications.
- Bootstrap: 2000 paired resamples over replications.

**Implementation checks** (`logs/impl.log`; all PASS):
- **I1:** the simulator reproduces `acc18.simulate_B` for SOD, RULE and RULE_FB bit for bit (record and windows).
- **I2:** it reproduces `acc18.batch_A` bit for bit in the eval11 random-number order (w̄, N, S, per-run log det J_cond).
- **I3:** `lik_data` equals `estimator_r8.StageData(…, ('sod',))`.
- **I4:** the receiver's windows equal the sensor's at every step of 40 two-stage records.
- **In the main run as well,** the receiver's windows equal the sensor's at every silent step of all 1600 stage-1 records and all 2900 whole records (`logs/main.log`).

## 3. The two checks (prompt §5), `logs/checks.log`

10 000 trajectories (10 batches of 1000). The SE of log det is the batch SE (round-18 convention); the SE of E[N] is sd/√10 000.

| check | result | target | z | verdict |
|---|---|---|---:|---|
| (a) constant SoD throughout: log det J | 197.19698 (SE 0.00208) | 197.201 | −1.93 | **PASS** |
| (a) constant SoD throughout: E[N] | 316.827 (SE 0.0225) | 316.89 | −2.81 | **PASS** |
| (b) rule throughout, true model, no feedback, e = 1.577: log det J | 207.50123 (SE 0.00061) | 207.501 | +0.37 | **PASS** |

- The (a) count passes, but close to the 3-SE limit. Against B = 316.892 it would be z = −2.90.
- Context for (b): gain +10.300 over the archived SoD value; E[N] 316.76.

## 4. Results

### 4.1 Information

Source: `logs/summary.log`, "INFORMATION" and "EMPIRICAL MEASURE".

| configuration | log det J | baseline, same replications | gain | bootstrap SE | 95 % interval | ΔD̂ (empirical) | 95 % interval |
|---|---:|---:|---:|---:|---|---:|---|
| primary | 202.8052 | 197.2002 | **+5.6050** | 0.0128 | [+5.581, +5.631] | +5.439 | [+5.008, +5.845] |
| K1 = 1250 | 203.8681 | 197.1977 | +6.6704 | 0.1166 | [+6.438, +6.884] | −2.049 | [−3.382, +3.793] |
| K1 = 3750 | 200.9089 | 197.1977 | +3.7112 | 0.0168 | [+3.678, +3.744] | +3.524 | [+2.913, +4.125] |
| control | 203.1104 | 197.1977 | +5.9127 | 0.0190 | [+5.874, +5.949] | +5.823 | [+5.124, +6.498] |

- The cost of model error at K1 = 2500 is control minus primary on replications 0..299: **+0.297 nat** [+0.262, +0.332]. On those replications the primary gains +5.616.
- The prompt's proportional reference 9 ln(K1/K + (1 − K1/K) e^{10.30/9}) is 8.619 / 6.549 / 3.858 for K1 = 1250 / 2500 / 3750.
  - Even with the true model, the control is 0.64 nat below its reference.
  - Post hoc (`logs/posthoc.log` D2): the SoD information is not split evenly in time. log det of its stage parts is 188.37 (steps 1..2500) against 186.38 (steps 2501..5000); equal halves would give 190.96 each. Stage 1 also uses 54.2 % of the SoD transmissions (N_1 = 171.7 of 316.9).
  - The prompt's basis ("roughly proportional") does not hold here. This is a description; no variant was run to apportion the 0.64 nat.

### 4.2 Budget

Source: `logs/summary.log`, "BUDGET". N counts every transmission, the forced first one included. "N > B" means N ≥ 317. plan_2 = B − 3 − N_1.

| configuration | mean N | sd | min | max | fraction N > B | mean N_1 (sd) | mean N_2 (sd) | mean plan_2 | mean N_2 − plan_2 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| primary | 315.197 | 3.522 | 305 | 330 | **0.316** (316 of 1000) | 171.67 (1.51) | 143.52 (3.82) | 142.22 | +1.30 |
| K1 = 1250 | **555.61** | **233.67** | 302 | **1209** | **0.930** (279 of 300) | 87.98 (0.91) | 467.63 (233.67) | 225.91 | **+241.72** |
| K1 = 3750 | 313.657 | 2.632 | 305 | 322 | 0.100 (30 of 300) | 228.76 (2.15) | 84.89 (3.16) | 85.13 | −0.24 |
| control | 314.743 | 1.705 | 309 | 319 | 0.147 (44 of 300) | 171.69 (1.48) | 143.05 (2.20) | 142.20 | +0.85 |
| baseline | 316.908 | 2.342 | 309 | 324 | 0.560 | 171.67 (1.51) | 145.23 (1.68) | – | – |

The primary's fraction above B on replications 0..299 is 0.317. Full count histograms are in `logs/summary.log`.

A reading, not a settled cause: the stage-2 plan rate is below the rule's rate at e_0, so the feedback holds e above e_0 and the count ends above plan.
- Stage 1 uses more than half the budget. So q_2 ≈ 142.2/2500 = 0.0569 per step, below the rate of the rule at e_0 = 1.577 (316.76/5000 ≈ 0.063, check (b)).
- The proportional feedback can only hold e_k above e_0 while N_k stays above plan_k. Mean e_k is 1.644 (control) and 1.670 (primary).
- N_2 ends above plan (+0.85, +1.30) and uses part of the 3-transmission margin.
- With model error the count's sd doubles (3.52 against 1.71 in the control).

### 4.3 Model: prediction error of θ̃

RMS over k of (h_kᵀθ̃ − x_k)/σ, computed with `mismatch15.pred_error`.

| K1 | mean | P10 | median | P90 | max | stage 2 only: median (P90) | max\|·\|: median (P90) |
|---:|---:|---:|---:|---:|---:|---|---|
| 1250 | 4.034 | 1.483 | **3.479** | 7.126 | 11.059 | 4.014 (8.228) | 9.25 (18.73) |
| 2500 | 0.293 | 0.176 | **0.283** | 0.430 | 0.717 | 0.338 (0.549) | 0.71 (1.16) |
| 3750 | 0.196 | 0.130 | **0.192** | 0.254 | 0.511 | 0.223 (0.358) | 0.46 (0.69) |

The control uses the primary's θ̃ only as the start of its final estimate.

Post hoc (`logs/posthoc.log` D1):

| K1 | cond J_1 | RMS prediction error, stage-1 steps (fitted) | RMS prediction error, stage-2 steps |
|---:|---:|---:|---:|
| 1250 | 8.2·10⁴ | 0.28 σ | 4.01 σ |
| 2500 | 445 | 0.20 σ | 0.34 σ |
| 3750 | 176 | 0.17 σ | 0.22 σ |

At K1 = 1250 the stage-1 model fits the data it saw but extrapolates badly. One plausible reading, not tested: the 12 and 12.3 rad/s components drift apart by only 0.375 rad in 1.25 s, so a 1.25-s record barely separates them.

### 4.4 Estimator (final estimate)

Source: `logs/summary.log`, "ESTIMATOR".
- CR sd = √diag J⁻¹, with each configuration's own J.
- I − D̂ = log det(M̂ J), where M̂ is the empirical MSE matrix about θ* (bias included). A positive value means the MSE is above the CR bound.
- "var ratio" = e^{(I−D̂)/9}; "sd ratio" = e^{(I−D̂)/18}.

| configuration | runs | n | largest \|bias\| (CR sd) | I − D̂ | var ratio | sd ratio |
|---|---|---:|---:|---:|---:|---:|
| primary | all | 1000 | **0.068** (θ3) | **+0.533** | 1.061 | 1.030 |
| primary | final CML accepted | 997 | 0.077 | +0.372 | 1.042 | 1.021 |
| K1 = 1250 | all | 300 | 0.649 (θ7) | +8.531 | 2.580 | 1.606 |
| K1 = 1250 | final CML accepted | 298 | 0.246 | +3.032 | 1.401 | 1.184 |
| K1 = 3750 | all | 300 | 0.097 | −0.001 | 1.000 | 1.000 |
| control | all | 300 | 0.147 | −0.099 | 0.989 | 0.995 |
| baseline | all | 1000 | 0.068 | +0.367 | 1.042 | 1.021 |
| baseline | final CML accepted | 992 | 0.057 | +0.108 | 1.012 | 1.006 |

- **The "known shortfall of about 3 %"** shows up as the primary's sd ratio of 1.030 (all runs) and 1.021 (accepted runs). It is reported and not fixed.
- **Modelling note** (post hoc, `logs/posthoc.log` D3): when the per-run information varies, (mean J)⁻¹ understates the attainable MSE (Jensen's inequality). −log det mean(J_cond⁻¹) is a fairer reference. Against it:
  - primary: +0.29 (all runs) and +0.13 (accepted runs);
  - K1 = 1250, accepted runs: −0.13. So most of its +3.03 is the spread of per-run information (I − I_inv = 3.17), not estimator inefficiency.
- At n = 300, D̂ also has an upward finite-sample bias of order +0.1 to +0.2 (round 18 DECISION 18-12: mean +0.091 at n = 500). This may explain the slightly negative values for K1 = 3750 and the control.

### 4.5 Failure cases

| configuration | stage-1 CML not accepted (θ̃ = fallback θ_LS) | final CML not accepted (final = fallback θ̃) | u_k ≤ 0.5 at some step | u_k ≥ 4 at some step | abnormal N_2 (< 10 or > 2·plan_2) |
|---|---|---|---:|---:|---:|
| primary | **12** of 1000: 4, 23, 49, 163, 305, 316, 447, 688, 692, 834, 837, 850 | 3: 829, 886, 975 | 1 (replication 447) | 0 | 0 |
| K1 = 1250 | 7 of 300: 0, 19, 45, 51, 100, 160, 187 | 2: 165, 240 | **110** | **269** | **117** |
| K1 = 3750 | 8 of 300: 22, 59, 93, 117, 164, 175, 178, 234 | 7: 31, 61, 99, 167, 204, 212, 230 | 0 | 0 | 0 |
| control | 4 (the primary's 4, 23, 49, 163) | 2: 157, 218 | 0 | 0 | 0 |
| baseline | (= primary stage 1) | 8: 220, 649, 715, 725, 890, 948, 949, 965 | – | – | – |

- In every non-accepted CML run, the first optimiser call ended with `ABNORMAL` (L-BFGS-B line search). The second call's message was not saved. No run was dropped: all statistics above use all replications, with the accepted-only figures alongside.
- The primary's stage-1 failures have θ̃ = θ_LS. Their prediction errors are 0.37–0.72 σ, against a median of 0.28 σ.
- Replication 447 (a stage-1 failure) is the one where u_k reached 0.5. Its N was 311.
- The e_k range over stage 2:
  - primary 0.500–3.141 (mean 1.670);
  - K1 = 3750 0.580–2.501 (mean 1.556);
  - control 1.008–2.302 (mean 1.644);
  - K1 = 1250 0.500–4.000 (mean 2.947).

## 5. The earlier two-stage experiment

### 5.1 What it measured

The "3 Nov draft" is not in the repository. The 3.67 nat matches **round 8b** (`round8b/report_round8b_CN.md`, `round8b/results/stats_round8b.json`): ΔD = 3.672, bootstrap 95 % interval [3.141, 4.222], R = 300.

That headline used the **empirical D measure**: D = −log det M̂, with M̂ the empirical MSE matrix of the final estimate about θ₀, and ΔD = D_seq − D_SoD+SoD (`reports_CN/plan_round8_CN.md` line 55). The same archive gives the information measure log det J_seq − log det J_SoD+SoD = 207.071 − 203.440 = **3.632**.

The design was different:
- two 5-s stages (10 s in all) with total budget 2B;
- a DP-designed table in stage 2;
- constant SoD + constant SoD as the comparison.

This round uses one 5-s record with budget B, split at K1.

### 5.2 Both measures, this round

| measure | primary (K1 = 2500) | K1 = 3750 | control | earlier round 8b |
|---|---:|---:|---:|---:|
| log det J gain | +5.605 [5.581, 5.631] | +3.711 | +5.913 | 3.632 |
| ΔD̂ (empirical) | +5.439 [5.008, 5.845] | +3.524 | +5.823 | **3.672** (headline) |

## 6. Predictions against results

Scoring conventions are those of `predictions.md` §2–§3, fixed before running. Source: `logs/summary.log`, "PREDICTIONS".

### 6.1 Prompt author (3 of 7)

| # | prediction | result | score |
|---|---|---|---|
| 1 | K1 = 2500: net gain 6.0 (5.3–6.6) | +5.605 | **hit** |
| 2 | K1 = 1250: net gain 8.1 (7.3–8.6) | +6.670, with the budget not kept (mean N 556) | **miss** |
| 3 | K1 = 3750: net gain 3.7 (3.2–4.0) | +3.711 | **hit** |
| 4 | control 6.4 (scored as rounding to 6.4, i.e. in [6.35, 6.45)) | +5.913; 6.4 is outside its 95 % interval [5.874, 5.949] | **miss** |
| 5 | mean N between 312 and 316, and 3–25 % above B | mean 315.20 (in); 31.6 % above B (out) | **miss** |
| 6 | K1 = 1250 has a higher fraction above B than K1 = 2500 | 0.930 vs 0.316 | **hit** |
| 7 | largest bias < 0.3 and MSE at most 5 % above CR. Scored reading: e^{(I−D̂)/9} ≤ 1.05 | bias 0.068 (yes); variance ratio 1.061 (no). The sd reading would give 1.030 (yes) | **miss** |

### 6.2 Mine (6 of 11)

| # | prediction | result | score |
|---|---|---|---|
| C1 | K1 = 2500 gain in [5.8, 6.5] | 5.605 | miss |
| C2 | K1 = 1250 gain in [7.4, 8.3] | 6.670 | miss |
| C3 | K1 = 3750 gain in [3.3, 3.8] | 3.711 | hit |
| C4 | control in [6.2, 6.6] and control − primary in [0.05, 0.40] | 5.913; +0.297 | miss |
| C5 | primary mean N in [314.0, 315.5], sd in [1.6, 2.6], fraction above B in [0.08, 0.25] | 315.20; 3.52; 0.316 | miss |
| C6 | fraction above B ordered K1 = 1250 > 2500 > 3750 | 0.930 > 0.316 > 0.100 | hit |
| C7 | stage-1 non-acceptance in [1, 15]; final ≤ 25 | 12; 3 | hit |
| C8 | median prediction error, K1 = 2500, in [0.12, 0.25]; ordering 1250 > 2500 > 3750 | 0.283; ordering holds | miss |
| C9 | clip reached in ≤ 2 replications; no abnormal N_2 | 1; 0 | hit |
| C10 | bias < 0.3; I − D̂ in [0.3, 1.0] (all) and [0.15, 0.6] (accepted) | 0.068; 0.533; 0.372 | hit |
| C11 | \|ΔD̂ − log det J gain\| ≤ 0.5 | 5.439 vs 5.605 | hit |

What both sets of predictions missed:
- The proportional basis does not hold in time: stage 1 takes 54 % of the transmissions and more than half of the SoD information.
- The feedback ends above plan when the plan rate is below the rule's rate at e_0.
- The K1 = 1250 stage-1 model is unusable for extrapolation, and the budget fails as a result.

## 7. Deviations

**Contradictions between prompt and code.** I found no case where the prompt and the existing code disagreed in a way that forced "the code wins". The places where I had to choose or interpret are below.

1. **Folder.** "round19/" is `evaluator/round19/`, following the code's convention for rounds 10–18. The imports (`EV = parent`) assume it.
2. **Stage-1 non-convergence.**
   - Prompt: "counted and listed separately".
   - Code: the existing `estimate_stage1` returns the fallback θ_LS when neither call is accepted. That θ_LS is used as θ̃ in stage 2. This is the code's behaviour and the code wins.
   - The runs are listed in §4.5, and every statistic is shown with them included and with accepted-only runs.
   - Likewise, the final estimate's own fallback θ̃ is the code's (`estimate_joint`).
3. **Final estimate.** "Censored ML on the whole record" is implemented with the existing two-stage estimator `estimator_r8b.estimate_joint`, started at θ̃ (the round-8 procedure). Round 18 used `estimate_stage1` (LS start) on a whole record instead. Both are the same likelihood and acceptance rule; only the start point and the fallback differ.
4. **e_0.** Round 18's module constant for RULE / RULE_FB is e = 1.58 (`acc18.E_RULE`, asserted). The prompt sets e_0 = 1.577. I used 1.577, a parameter of this round, and check (b) uses it too. The 1.58 appears only in the implementation checks that reproduce round-18 routines.
5. **x̄_k.** Stage 2 uses x̄_k = h_kᵀθ̃ literally (`predict`, one gemv; the receiver calls the same function).
   - For the control, x̄_k = x_k (the stored true output, `XK`), following round 15's convention (DECISION 15-2) that a zero model error is the true model exactly.
   - h_kᵀθ* and x_k differ only by rounding (~10⁻¹⁶ relative).
6. **"e_k hits 0.5 or 4".** Counted as the unclipped u_k ≤ 0.5 or ≥ 4 at any stage-2 step, the counting of `mismatch15.run_job` / `acc18`.
7. **Check (a) target.** E[N] is compared with 316.89 as written. Against B exactly, z = −2.90 instead of −2.81; both pass. Check SEs are batch SEs (10 × 1000) as in round 18.
8. **Control, "everything else identical".** Same noise (replications 0..299), same stage 1, same θ̃ as the start of the final estimate. Only x̄ in stage 2 differs.
9. **Secondary configurations.** They use the primary's first 300 noise sequences, and their gains are paired with the baseline on those replications.
10. **Earlier experiment.** The "3 Nov draft" is not in the repository. I took its 3.67 to be round 8b's ΔD = 3.672 (§5).
11. **Post-hoc diagnostics** (`posthoc19.py`, D1–D4) were added after the summary. They read saved arrays only (no new simulation, estimate or variant) and are labelled descriptive wherever used.

## 8. Not done

- **No repair of K1 = 1250.** No change to the clip, the margin, the feedback or a fallback, and no re-tuning of e_0 or T: the prompt rules out repair. For the same reason no budget-matched gain is given for K1 = 1250; its +6.67 nat comes with a mean of 556 transmissions.
- **No decomposition of the control's 0.64 nat** below the proportional reference into time split, margin and feedback. That would need variants not listed in the prompt (for example the rule with the true model and no feedback in stage 2). Only the post-hoc time split of the saved information is given (§4.1).
- **No supplementary Newton analysis** of the non-accepted CML runs (round 18 had one), and no change of the acceptance rule.
- **No figures** (as instructed), no Chinese copy of this README, and no pull request.
- **The interpretations above are not tested.** These are the frequency-separation reading of the K1 = 1250 extrapolation error and the rate-mismatch reading of the budget overshoot.
