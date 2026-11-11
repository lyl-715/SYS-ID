---
title: "Round 18 report: accumulated information, the non-transmissions, and the estimator"
date: "2026-11-11"
---

English copy of `report_round18_CN.md`. Where the two differ, the Chinese version is the report of record.

**Conclusion.**
- **Part A.** Per-trajectory accumulated information fluctuates little:
  - The sd of the per-run log det J_cond is 0.063 nat under the rule and 0.257 nat under constant SoD.
  - I minus the mean per-run log det is 0.002 and 0.033 nat.
  - The non-transmissions carry 13.35 nat (rule), 3.75 nat (SoD) and 14.77 nat (PRED) of the information.
- **Part B, least squares.** LS on the transmitted samples is biased by up to 35.7 Cramér–Rao standard deviations under the rule and 5.3 under SoD. Its D̂ lies 21.5 nat (rule), 6.1 nat (SoD) and 30.3 nat (PRED) below I.
- **Part B, censored ML (pre-registered estimator, all 500 runs).**
  - Not accepted: 3 / 8 / 3 runs (SoD / rule / PRED). Those runs take the fallback θ_LS.
  - Under these numbers the CML does not attain the Cramér–Rao bound in the sense of Q4, for either the rule or PRED.
  - **The check "SOD, censored ML within 0.3 nat of I" FAILED: D̂ − I = −0.339 nat.**
  - On the accepted runs only, D̂ − I is −0.137 (SoD), −0.559 (rule) and −0.312 nat (PRED).
- **Part B, supplementary Newton analysis** (not the pre-registered estimator). All 14 non-accepted runs already had a Newton decrement ≤ 6.4e-7 at the end of the first L-BFGS-B call, so 0 Newton steps were needed. With those estimates substituted, D̂ − I is −0.129 / −0.580 / −0.300 nat.
- **Part C.**
  - The terminal-weight rule C1 is +0.101 nat above the constant rule C0 (paired SE 0.0002).
  - The running-information rule C2 is −0.709 nat below C1 (paired SE 0.0004).
- **Predictions:** 4 of 6 lines hit; B3 and C miss.

**History of this round.**
- A first session created branch `round18-accumulated-information` from the head of `round15-rule-estimated-model` at **`6217acb`** (the Round 15 report). It committed the prompt, the predictions, the code, stage A and stage B. It **stopped after stage B, at head `e85a3f0`**.
- This session (the second) **continued from `e85a3f0`** on the same branch, without force-pushing (user decision):
  - It reviewed `acc18.py` and `run_round18.py`.
  - It re-ran stages A and B and checked that they reproduce the committed results exactly; the committed results were kept.
  - It added the supplementary Newton stage (user instruction), ran stage C and the summary, and wrote this report.

**Commits:**
- first session:
  - `26ef03c`: the prompt (first commit);
  - `e097fac`: `round18/predictions_round18.md`, before any computation (second commit);
  - `a0cacd7`: code;
  - `5e758eb`: stage A;
  - `e85a3f0`: stage B;
- second session:
  - `eb29841`: re-run check of A and B;
  - `a7e1dde`: supplementary code;
  - `3836713`: stage Bn (supplementary);
  - `17858ed`: stage C;
  - `cda401b`: summary;
  - this report: the commit after that.

**Scope.**
- Input U, true model, B = 316.8921168493721, I_SoD = 197.20097668224076. ΔI = I − I_SoD.
- No DP and no grid evaluator. Every number is a Monte Carlo of the continuous triggers, or comes from the estimators on simulated experiments.
- No causes are claimed.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/round18/`.
- `[L X:n]` is line n of `logs/X.log`; `[J a.b]` is field `a.b` of `results/round18.json`.
- Code:
  - `acc18.py`: triggers, Part A batches, Part B data and statistics, edge table, Part C runs;
  - `run_round18.py`: stages A, B, Bn, C, summary;
  - `supp18.py`: the supplementary Newton steps;
  - `summary18.py`;
  - `rerun_check18.py`: the reproduction check.
- Results: `results/partA.json`, `partB.json`, `partB_estimates.npz`, `partB_newton.json`, `partC.json`, `round18.json`.

## 1. Answers to Q1–Q5

Sources: [L summary:28–32]; [J questions].

- **Q1.** Per-run log det J_cond, over 10⁴ trajectories:
  - Standard deviation: **RULE 0.0626 nat, SOD 0.2573 nat**, RULE_FB 0.3150 nat, PRED 0 (2.8e-14; under PRED, J_cond is the same in every run).
  - I − mean per-run log det: **RULE 0.0019 nat, SOD 0.0334 nat**, RULE_FB 0.0288 nat, PRED 0.
  - 5–95 % range: RULE 207.360–207.567; SOD 196.749–197.588.
  - Correlation with N: SOD +0.221, RULE +0.515, RULE_FB −0.098 [L A:5–12].
- **Q2.** I − log det mean(J_sent) is the information in the non-transmissions, with batch SE ≤ 0.005 nat:
  - SOD 3.751 nat;
  - RULE 13.349 nat;
  - RULE_FB 13.253 nat;
  - PRED 14.771 nat [L A:6, 8, 10, 12].
- **Q3. Least squares on the transmitted samples.** "Largest bias" is the largest absolute mean error over the nine parameters, in Cramér–Rao standard deviations (CR sd); I − D̂ is in nat. The numbers on the accepted runs only (n = 497 / 492 / 497) differ by at most 0.013 in largest bias and 0.013 nat in I − D̂.

  | trigger | largest bias (all 500) | in MC SE | I − D̂ (all 500) |
  |---|---|---:|---:|
  | RULE | 35.71 CR sd (θ3) | 348 | 21.45 |
  | SOD | 5.35 CR sd (θ3) | 107 | 6.06 |
  | PRED | 0.485 CR sd (θ9) | 2.0 | 30.29 |

  Under PRED the nine LS RMSE ratios are 5.24–5.58 and −log det cov equals D̂ to 0.001 nat [§3].
- **Q4. Censored ML under RULE and PRED.** Criterion: D̂ within 0.3 nat of I and largest bias < 0.2 CR sd.

  | run set | RULE: largest bias | RULE: D̂ − I | RULE attains? | PRED: largest bias | PRED: D̂ − I | PRED attains? |
  |---|---:|---:|---|---:|---:|---|
  | pre-registered, all 500 runs (fallback θ_LS in 8 / 3 runs) | 0.421 | −4.561 | **no** | 0.090 | −1.247 | **no** |
  | accepted runs only (492 / 497) | 0.151 | −0.559 | no | 0.080 | −0.312 | no |
  | supplementary, Newton estimates substituted (all 500) | 0.159 | −0.580 | no | 0.081 | −0.2995 | yes, by 0.0005 nat |

  For reference (DECISION 18-12, descriptive): if the errors attained the bound exactly, D̂ − I would have mean +0.091 and sd 0.191 at n = 500 [L summary:21–23].
- **Q5.**
  - **C2 does not beat C1:** I(C2) − I(C1) = **−0.7088 nat** (paired SE 0.0004 over ten seeds; per seed −0.7106 … −0.7063).
  - **C1 beats C0:** I(C1) − I(C0) = **+0.1013 nat** (paired SE 0.0002; per seed +0.1003 … +0.1023).
  - I(C2) − I(C0) = −0.6076 nat (SE 0.0004) [L C:43–45].

## 2. Part A — realised information and the value of the non-transmissions

R = 10⁴ trajectories per trigger (10 batches of 1000, seeds 20261115 + b) [L A:5–12]; [J partA]. I = log det of the mean J_cond (nat). "Per-run" refers to log det J_cond of one trajectory. Value of non-tx = I − log det mean(J_sent).

| trigger | I | ΔI | per-run mean | sd | P5 | P95 | I − mean | corr(log det, N) | log det mean(J_sent) | value of non-tx (batch SE) | S | E[N] (SE) | sd N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOD | 197.2005 | −0.0005 | 197.1671 | 0.2573 | 196.7491 | 197.5884 | 0.03338 | +0.221 | 193.4493 | 3.7512 (0.0028) | 466.12 | 316.875 (0.022) | 2.246 |
| RULE | 207.4643 | +10.2634 | 207.4624 | 0.0626 | 207.3596 | 207.5670 | 0.00193 | +0.515 | 194.1150 | 13.3494 (0.0047) | 1387.30 | 314.391 (0.180) | 18.040 |
| RULE_FB | 207.3711 | +10.1701 | 207.3423 | 0.3150 | 206.8153 | 207.8515 | 0.02878 | −0.098 | 194.1183 | 13.2528 (0.0034) | 1371.83 | 314.159 (0.018) | 1.807 |
| PRED | 208.9566 | +11.7556 | 208.9566 | 0.0000 | 208.9566 | 208.9566 | −0.00000 | — | 194.1853 | 14.7713 (0.0046) | 1636.15 | 316.570 (0.171) | 17.097 |

- Total weight S against E[N]: S / E[N] = 1.47 (SOD), 4.41 (RULE), 4.37 (RULE_FB), 5.17 (PRED).
- Batch SE of I: SOD 0.0029, RULE 0.0008, RULE_FB 0.0035, PRED 0.0000.
- RULE_FB: e_k has mean 1.595 σ and sd 0.144 σ; it never reaches a clip bound [L A:10].
- Outside reference (not a gate, e = 1.5768 there, 1.580 here) [L A:16–17]:
  - RULE: I 207.464 here / 207.503 there; per-run mean 207.462 / 207.501; sd 0.063 / 0.063; I − mean 0.002 / 0.002; log det mean(J_sent) 194.115 / 194.179.
  - SOD: I 197.200 / 197.194; per-run mean 197.167 / 197.160; sd 0.257 / 0.263; I − mean 0.033 / 0.034; log det mean(J_sent) 193.449 / 193.448.

## 3. Part B — least squares against censored maximum likelihood

### 3.1 Setup and the non-accepted runs

There are 500 experiments per trigger (seeds 20261116 + i, the same noise for the three triggers). N per experiment, mean (sd): SOD 316.78 (2.27), RULE 314.85 (18.24), PRED 316.73 (16.80) [L B:3–5].

Under the round-8b acceptance rule, a CML run is not accepted when neither of its two L-BFGS-B calls is accepted. The pre-registered estimator (DECISION 18-6) then uses the **fallback θ_LS**: least squares on the transmitted pairs of the same run [L B:10, 15, 20]; [J partB.<T>].

| trigger | not accepted (of 500) | runs | first call not accepted | restart used | fallback θ_LS | first-call nit median | largest decrement of the first call |
|---|---:|---|---:|---:|---:|---:|---:|
| SOD | **3** | 183, 269, 477 | 7 | 4 | 3 | 41 | 1.1e-6 |
| RULE | **8** | 46, 92, 114, 202, 303, 340, 424, 439 | 15 | 7 | 8 | 37 | 1.3e-6 |
| PRED | **3** | 132, 266, 427 | 8 | 5 | 3 | 36 | 1.6e-6 |

- Each of the 14 non-accepted runs ended **both** calls with `ABNORMAL` (L-BFGS-B success = False) (§3.5).
- LS had rank 9 in every run.

In every table below, each trigger and estimator appears twice: **all 500 runs** (pre-registered; fallback included) and **accepted runs only** (n = 497 / 492 / 497). LS on the accepted runs uses the same subset. The supplementary Newton row is labelled "(supp.)".

### 3.2 Bias (mean error / CR sd)

CR sd = sqrt(diag J⁻¹), with J the Part A mean J_cond of the same trigger. "MC SE" is the range over the nine parameters of the Monte Carlo SE of the bias. [L B:6, 8, 11, 13, 16, 18]; [L summary:6–20].

| trigger | estimator, runs | n | θ1 | θ2 | θ3 | θ4 | θ5 | θ6 | θ7 | θ8 | θ9 | MC SE | largest abs | largest abs/MC SE | Mahalanobis sqrt(b̄ᵀJb̄) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOD | LS, all | 500 | +0.758 | −0.373 | −5.348 | −0.321 | −2.007 | −0.407 | −0.828 | +1.054 | −0.397 | 0.048–0.052 | 5.348 (θ3) | 107.2 (θ3) | 7.995 |
| SOD | LS, accepted | 497 | +0.762 | −0.378 | −5.355 | −0.316 | −2.009 | −0.415 | −0.831 | +1.060 | −0.400 | 0.048–0.052 | 5.355 (θ3) | 107.3 (θ3) | 8.006 |
| SOD | CML, all | 500 | −0.070 | +0.057 | −0.031 | −0.007 | −0.102 | −0.067 | −0.001 | +0.051 | −0.078 | 0.042–0.048 | 0.102 (θ5) | 2.2 (θ5) | 0.222 |
| SOD | CML, accepted | 497 | −0.071 | +0.054 | −0.005 | +0.001 | −0.093 | −0.072 | +0.000 | +0.051 | −0.079 | 0.043–0.047 | 0.093 (θ5) | 2.0 (θ5) | 0.199 |
| SOD | CML, Newton (supp.) | 500 | −0.075 | +0.059 | +0.000 | −0.000 | −0.093 | −0.065 | +0.003 | +0.046 | −0.075 | 0.043–0.046 | 0.093 (θ5) | 2.0 (θ5) | 0.197 |
| RULE | LS, all | 500 | +1.551 | +12.128 | −35.710 | +1.912 | −20.986 | +0.160 | −7.184 | +2.602 | −4.918 | 0.102–0.110 | 35.710 (θ3) | 347.8 (θ3) | 50.904 |
| RULE | LS, accepted | 492 | +1.539 | +12.137 | −35.711 | +1.913 | −21.005 | +0.147 | −7.185 | +2.617 | −4.932 | 0.103–0.110 | 35.711 (θ3) | 344.5 (θ3) | 50.920 |
| RULE | CML, all | 500 | +0.074 | +0.173 | −0.421 | −0.022 | −0.265 | +0.010 | −0.118 | +0.043 | −0.050 | 0.048–0.206 | 0.421 (θ3) | 2.2 (θ5) | 0.654 |
| RULE | CML, accepted | 492 | +0.039 | −0.014 | +0.151 | −0.052 | +0.053 | −0.005 | −0.004 | +0.016 | +0.015 | 0.046–0.049 | 0.151 (θ3) | 3.3 (θ3) | 0.179 |
| RULE | CML, Newton (supp.) | 500 | +0.042 | −0.016 | +0.159 | −0.049 | +0.061 | −0.012 | +0.000 | +0.019 | +0.010 | 0.045–0.048 | 0.159 (θ3) | 3.4 (θ3) | 0.189 |
| PRED | LS, all | 500 | +0.056 | −0.234 | +0.231 | +0.117 | −0.405 | +0.361 | −0.404 | −0.019 | +0.485 | 0.234–0.250 | 0.485 (θ9) | 2.0 (θ9) | 0.733 |
| PRED | LS, accepted | 497 | +0.068 | −0.225 | +0.246 | +0.121 | −0.399 | +0.366 | −0.392 | −0.035 | +0.477 | 0.236–0.251 | 0.477 (θ9) | 2.0 (θ9) | 0.732 |
| PRED | CML, all | 500 | +0.007 | −0.062 | +0.032 | +0.016 | −0.087 | +0.056 | −0.078 | +0.009 | +0.090 | 0.046–0.050 | 0.090 (θ9) | 1.8 (θ5) | 0.143 |
| PRED | CML, accepted | 497 | +0.019 | −0.052 | +0.046 | +0.020 | −0.080 | +0.059 | −0.064 | −0.006 | +0.079 | 0.044–0.047 | 0.080 (θ5) | 1.7 (θ5) | 0.136 |
| PRED | CML, Newton (supp.) | 500 | +0.017 | −0.054 | +0.042 | +0.020 | −0.081 | +0.058 | −0.066 | −0.003 | +0.080 | 0.044–0.047 | 0.081 (θ5) | 1.8 (θ5) | 0.136 |

### 3.3 Root-mean-square error / CR sd

[L B:7, 9, 12, 14, 17, 19]; [J partB.<T>.sets.<set>.rmse_cr].

| trigger | estimator, runs | θ1 | θ2 | θ3 | θ4 | θ5 | θ6 | θ7 | θ8 | θ9 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOD | LS, all | 1.374 | 1.224 | 5.463 | 1.195 | 2.294 | 1.166 | 1.365 | 1.502 | 1.209 | 1.166 | 5.463 |
| SOD | LS, accepted | 1.377 | 1.226 | 5.469 | 1.193 | 2.297 | 1.166 | 1.366 | 1.507 | 1.207 | 1.166 | 5.469 |
| SOD | CML, all | 1.003 | 1.019 | 1.064 | 1.025 | 1.029 | 0.989 | 0.997 | 0.949 | 1.042 | 0.949 | 1.064 |
| SOD | CML, accepted | 1.004 | 1.020 | 1.013 | 1.022 | 1.024 | 0.988 | 0.997 | 0.951 | 1.039 | 0.951 | 1.039 |
| SOD | CML, Newton (supp.) | 1.004 | 1.020 | 1.014 | 1.020 | 1.022 | 0.991 | 0.996 | 0.951 | 1.040 | 0.951 | 1.040 |
| RULE | LS, all | 2.860 | 12.357 | 35.783 | 3.008 | 21.129 | 2.376 | 7.534 | 3.481 | 5.439 | 2.376 | 35.783 |
| RULE | LS, accepted | 2.859 | 12.367 | 35.785 | 3.006 | 21.146 | 2.366 | 7.537 | 3.489 | 5.451 | 2.366 | 35.785 |
| RULE | CML, all | 1.114 | 1.816 | 4.629 | 1.083 | 2.743 | 1.097 | 1.381 | 1.070 | 1.219 | 1.070 | 4.629 |
| RULE | CML, accepted | 1.059 | 1.056 | 1.043 | 1.017 | 1.060 | 1.042 | 1.029 | 1.011 | 1.079 | 1.011 | 1.079 |
| RULE | CML, Newton (supp.) | 1.057 | 1.051 | 1.047 | 1.013 | 1.056 | 1.048 | 1.028 | 1.020 | 1.083 | 1.013 | 1.083 |
| PRED | LS, all | 5.236 | 5.368 | 5.581 | 5.289 | 5.423 | 5.407 | 5.338 | 5.281 | 5.419 | 5.236 | 5.581 |
| PRED | LS, accepted | 5.246 | 5.364 | 5.593 | 5.286 | 5.425 | 5.418 | 5.329 | 5.287 | 5.415 | 5.246 | 5.593 |
| PRED | CML, all | 1.025 | 1.113 | 1.068 | 1.091 | 1.096 | 1.042 | 1.124 | 1.035 | 1.130 | 1.025 | 1.130 |
| PRED | CML, accepted | 1.002 | 1.017 | 1.042 | 0.999 | 1.024 | 1.016 | 1.002 | 0.990 | 1.032 | 0.990 | 1.042 |
| PRED | CML, Newton (supp.) | 1.000 | 1.018 | 1.040 | 0.999 | 1.024 | 1.014 | 1.004 | 0.989 | 1.032 | 0.989 | 1.040 |

### 3.4 D̂ = −log det of the mean-square-error matrix, against I

D̂ uses errors about the true θ; −log det cov uses errors about their mean. The bootstrap SE of D̂ (2000 resamples) is descriptive (DECISION 18-12). [L B:7, 9, …]; [L summary:6–23].

| trigger | I | estimator, runs | n | D̂ | D̂ − I | bootstrap SE of D̂ | −log det cov | −log det cov − I |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| SOD | 197.200 | LS, all | 500 | 191.138 | −6.063 | 0.179 | 195.143 | −2.058 |
| SOD | 197.200 | LS, accepted | 497 | 191.125 | −6.076 | 0.174 | 195.145 | −2.055 |
| SOD | 197.200 | CML, all | 500 | 196.862 | **−0.339** | 0.204 | 196.886 | −0.315 |
| SOD | 197.200 | CML, accepted | 497 | 197.064 | −0.137 | 0.177 | 197.082 | −0.118 |
| SOD | 197.200 | CML, Newton (supp.) | 500 | 197.072 | −0.129 | 0.183 | 197.089 | −0.111 |
| RULE | 207.464 | LS, all | 500 | 186.010 | −21.455 | 0.182 | 192.446 | −15.018 |
| RULE | 207.464 | LS, accepted | 492 | 186.011 | −21.453 | 0.182 | 192.461 | −15.003 |
| RULE | 207.464 | CML, all | 500 | 202.903 | −4.561 | 0.561 | 202.902 | −4.562 |
| RULE | 207.464 | CML, accepted | 492 | 206.905 | −0.559 | 0.189 | 206.915 | −0.549 |
| RULE | 207.464 | CML, Newton (supp.) | 500 | 206.885 | −0.580 | 0.189 | 206.899 | −0.566 |
| PRED | 208.957 | LS, all | 500 | 178.664 | −30.293 | 0.192 | 178.664 | −30.292 |
| PRED | 208.957 | LS, accepted | 497 | 178.653 | −30.303 | 0.190 | 178.654 | −30.303 |
| PRED | 208.957 | CML, all | 500 | 207.710 | −1.247 | 0.471 | 207.711 | −1.246 |
| PRED | 208.957 | CML, accepted | 497 | 208.645 | −0.312 | 0.184 | 208.645 | −0.312 |
| PRED | 208.957 | CML, Newton (supp.) | 500 | 208.657 | −0.300 | 0.190 | 208.657 | −0.299 |

**Reference distribution of D̂ − I for errors that attain the bound exactly** (zero mean, covariance J⁻¹; nM̂ ~ Wishart(J⁻¹, n); descriptive, DECISION 18-12) [L summary:21–23]:

| n | mean | sd | P(\|D̂ − I\| ≤ 0.3) | P(D̂ − I < −0.3) |
|---:|---:|---:|---:|---:|
| 500 | +0.091 | 0.191 | 0.845 | 0.020 |
| 497 | +0.091 | 0.191 | 0.843 | 0.020 |
| 492 | +0.092 | 0.192 | 0.841 | 0.020 |

How far the observed values sit below that reference mean, in reference sd:
- SOD, all runs (−0.339): 2.25 sd.
- RULE, accepted (−0.559): 3.39 sd.
- RULE, Newton substitution (−0.580): 3.51 sd.
- PRED, accepted (−0.312): 2.11 sd.

This is context only. No check or score uses it.

### 3.5 Supplementary: Newton steps for the non-accepted runs (not the pre-registered estimator; DECISION 18-S1 … 18-S4)

Procedure (`supp18.py`, stage Bn):
- For the 14 non-accepted runs only, `estimator_r8b.estimate_stage1` was re-run. It reproduced stage B exactly: not accepted, and fallback θ_LS bitwise equal to the archive [L Bn:3].
- Starting at the end point of the first L-BFGS-B call, full Newton steps with the analytic gradient and Hessian were to be taken until η ≤ 1e-5, or 50 steps.

| trigger | run | N | call 1: success, message, nit, η | call 2: success, η | Newton steps | converged | max abs(θ_call1 − θ̂)/CR sd | max abs(θ_call2 − θ̂)/CR sd | max abs(θ_LS − θ̂)/CR sd | max abs error/CR sd: θ̂ / θ_LS |
|---|---:|---:|---|---|---:|---|---:|---:|---:|---|
| SOD | 183 | 320 | False, ABNORMAL, 39, 8.0e-8 | False, 1.0e-7 | 0 | yes | 0 | 8.9e-8 | 5.52 | 1.96 / 5.38 |
| SOD | 269 | 316 | False, ABNORMAL, 41, 8.0e-8 | False, 3.3e-7 | 0 | yes | 0 | 2.3e-7 | 4.82 | 1.54 / 4.01 |
| SOD | 477 | 315 | False, ABNORMAL, 42, 1.9e-7 | False, 4.0e-7 | 0 | yes | 0 | 2.1e-7 | 5.22 | 1.90 / 3.32 |
| RULE | 46 | 283 | False, ABNORMAL, 35, 6.4e-7 | False, 3.6e-7 | 0 | yes | 0 | 5.2e-7 | 37.27 | 2.17 / 36.05 |
| RULE | 92 | 312 | False, ABNORMAL, 40, 3.3e-7 | False, 3.3e-7 | 0 | yes | 0 | 3.0e-7 | 36.69 | 2.23 / 34.84 |
| RULE | 114 | 288 | False, ABNORMAL, 35, 5.2e-8 | False, 2.8e-7 | 0 | yes | 0 | 1.5e-7 | 33.50 | 1.97 / 31.62 |
| RULE | 202 | 326 | False, ABNORMAL, 30, 1.5e-7 | False, 5.6e-7 | 0 | yes | 0 | 4.0e-7 | 38.60 | 1.96 / 38.74 |
| RULE | 303 | 303 | False, ABNORMAL, 35, 1.2e-7 | False, 1.8e-7 | 0 | yes | 0 | 1.4e-7 | 37.42 | 1.33 / 37.49 |
| RULE | 340 | 299 | False, ABNORMAL, 38, 4.1e-7 | False, 9.7e-7 | 0 | yes | 0 | 5.2e-7 | 35.33 | 2.17 / 34.08 |
| RULE | 424 | 285 | False, ABNORMAL, 32, 2.9e-8 | False, 5.5e-7 | 0 | yes | 0 | 3.5e-7 | 35.76 | 1.89 / 35.13 |
| RULE | 439 | 323 | False, ABNORMAL, 37, 1.4e-7 | False, 1.3e-6 | 0 | yes | 0 | 7.0e-7 | 35.35 | 1.61 / 36.94 |
| PRED | 132 | 272 | False, ABNORMAL, 33, 3.0e-7 | False, 4.3e-8 | 0 | yes | 0 | 1.3e-7 | 8.04 | 1.71 / 9.75 |
| PRED | 266 | 336 | False, ABNORMAL, 39, 3.6e-7 | False, 2.1e-6 | 0 | yes | 0 | 1.0e-6 | 7.59 | 1.89 / 9.48 |
| PRED | 427 | 319 | False, ABNORMAL, 28, 4.1e-7 | False, 3.2e-7 | 0 | yes | 0 | 2.6e-7 | 6.72 | 1.61 / 8.33 |

- **14 of 14 converge** (3 / 8 / 3), each with 0 Newton steps.
  - At the first call's end point the Newton decrement was already 2.9e-8 … 6.4e-7 ≤ 1e-5. So θ̂ (the supplementary estimate) is the first call's end point.
  - The restart call's end point lies within 1.0e-6 CR sd of it.
  - The pre-registered fallback θ_LS lies 4.8–5.5 (SOD), 33.5–38.6 (RULE) and 6.7–8.0 (PRED) CR sd from it in the largest parameter [L Bn:4–23].
- These runs were not accepted under the round-8b rule only because the calls ended with success = False. That rule does not accept such a call however small η is (`estimator_r8b`, docstring).
- With the supplementary estimates substituted, see the "Newton (supp.)" rows of §3.2–3.4:
  - D̂ − I is −0.129 (SOD), −0.580 (RULE) and −0.2995 (PRED) nat.
  - The largest bias is 0.093, 0.159 and 0.081 CR sd.

### 3.6 The check of prompt §3

**The check "for SOD the CML numbers are consistent with rounds 8/8b: D̂ within 0.3 nat of I" FAILED.** D̂ = 196.862 against I = 197.200, so D̂ − I = **−0.339 nat**, and |D̂ − I| = 0.339 > 0.3:

```
[L B:21] CHECK B SOD CML: D-hat 196.862 vs I 197.200: |D-hat - I| 0.339 <= 0.3 -> FAIL (round 8b stage 1, R = 300: -log det M1 196.881 vs log det J1 197.201)
```

- The threshold was not changed. The pre-registered estimator (all 500 runs, 3 fallbacks to θ_LS) is the number of record.
- For information, without re-scoring: on the 497 accepted runs D̂ − I = −0.137, and with the supplementary substitution −0.129.
- The first session's stage B recorded this failure and continued. The driver records the check in `partB.json → checks.B_SOD_CML` and does not stop on it. This session continued on the user's instruction.

## 4. Part C — a rule driven by the running information

**Edge table** [L C:3]. 701 of the 701 grid edges lie on the upper envelope of ρ c_one(e) − q_one(e).
- E = 4.0 for ρ < 0.062574 and E = 0.5 for ρ ≥ 0.979565; E = 1.58 on [0.344635, 0.346245).
- E is non-increasing in ρ, and the table agrees with the direct argmax on 21401 values of ρ.

**Reference and weights** [L C:2, 4].
- J_ref = mean J_cond of RULE from Part A (log det 207.4643); S_ref = 1387.3026; ε = 0.02.
- γ_k ranges from 0.00521 to 0.00979 (median 0.00652). Σ W_ref,k γ_k = 9.000000 = p, and p / S_ref = 0.006487.

**Tuned variants** (R = 2000, noise seed 20261117, E[N] = B within 0.3) [L C:15, 27, 38–42]; [J partC]:

| variant | tuned parameter | tuning runs | E[N] (SE) | sd N | I | ΔI | per-run mean | per-run sd | S | e_k P5 / P50 / P95 (mean), σ | I over 10 seeds: mean (sd) | E[N] over 10 seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| C0 | e = 1.57734 σ | 10 | 316.703 (0.408) | 18.23 | 207.4975 | +10.2965 | 207.4955 | 0.0614 | 1392.45 | 1.577 / 1.577 / 1.577 (1.5773) | 207.4973 (0.0019) | 316.238 |
| C1 | λ = 0.0188293 | 11 | 316.764 (0.405) | 18.13 | 207.5989 | +10.3980 | 207.5971 | 0.0613 | 1369.00 | 1.300 / 1.580 / 1.810 (1.5957) | 207.5985 (0.0017) | 315.962 |
| C2 | λ = 0.0216284 | 10 | 316.779 (0.408) | 18.25 | 206.8893 | +9.6884 | 206.8882 | 0.0705 | 1352.24 | 1.395 / 1.485 / 2.210 (1.6217) | 206.8897 (0.0022) | 316.114 |

**Differences** (λ or e fixed at the tuned value; ten noise seeds 20261117 + b; paired) [L C:43–45]:

| difference | mean over 10 seeds | paired SE | per-seed range | at the tuning seed | E[N] difference, mean |
|---|---:|---:|---|---:|---:|
| I(C1) − I(C0) | **+0.1013** | 0.0002 | +0.1003 … +0.1023 | +0.1015 | −0.276 |
| I(C2) − I(C1) | **−0.7088** | 0.0004 | −0.7106 … −0.7063 | −0.7096 | +0.152 |
| I(C2) − I(C0) | −0.6076 | 0.0004 | −0.6090 … −0.6053 | −0.6081 | −0.125 |

- In C2 the relative weight g_k S_k / p averages **1.140** over (trajectory, step) at the tuned λ. ρ ranges over [0.170, 0.424].
- The running J_cond agrees with the directly accumulated one to 2.1e-14 (relative).
- e_k never reaches 0.5 or 4.0 in C1 or C2 [L C:27, 38].
- The E[N] differences between variants are at most 0.28 transmissions on average. The C0 tuning runs give a slope of about 0.019 nat per transmission (e 1.5813 → 1.5766: E[N] 314.21 → 317.20, I 207.4498 → 207.5070) [L C:11–13]. At that slope, 0.28 transmissions corresponds to ≤ 0.005 nat (descriptive).

## 5. Predictions (`predictions_round18.md`, `e097fac`)

**4 of 6 lines hit.** By bullet, Part A is a hit, Part B a miss and Part C a miss [L summary:33–39]; [J predictions].

| line | values | result |
|---|---|---|
| A1: sd RULE ∈ [0.04, 0.09], SOD ∈ [0.20, 0.33]; I − mean RULE ≤ 0.005, SOD ∈ [0.02, 0.05] | 0.0626, 0.2573; 0.00193, 0.03338 | hit |
| A2: non-tx RULE ∈ [12.8, 13.8], SOD ∈ [3.3, 4.1], PRED ∈ [14.0, 15.5] | 13.349, 3.751, 14.771 | hit |
| B1: LS RULE largest bias > 3; D̂ more than 10 nat below I | 35.710; I − D̂ = 21.455 | hit |
| B2: LS SOD largest bias > 0.5 | 5.348 | hit |
| B3: CML RULE and PRED: largest bias < 0.2; abs(D̂ − I) ≤ 0.3; ≤ 2 % not accepted | RULE **0.421**, **−4.561**, 8 (1.6 %); PRED 0.090, **−1.247**, 3 (0.6 %) | **miss** |
| C: I(C1) − I(C0) ∈ [0.00, 0.20]; I(C2) − I(C1) ∈ [−0.20, +0.05] | +0.1013 ✓; **−0.7088** | **miss** |

The scoring uses the pre-registered statistics (all 500 runs, fallback included; DECISION 18-1 / 18-6). For information only, not a re-score: on the accepted runs B3 would still miss (RULE −0.559, PRED −0.312), and with the supplementary Newton substitution it would still miss (RULE −0.580).

## 6. Checks

1. **Simulation against Rounds 14 and 15** (bitwise, batch 0):
   ```
   [L A:3] CHECK sim RULE batch 0 vs rule14._mc_cont(e = 1.58, 1000, 20261115): Wk, N, S bitwise equal True -> PASS
   [L A:4] CHECK sim RULE_FB batch 0 vs mismatch15.run_job(delta = 0, VF(100, 3), 1000, 20261115): Wk, N, S, e sums bitwise equal True -> PASS
   ```
2. **Part A against the archives** (prompt §2):
   ```
   [L A:13] CHECK A SOD: I 197.20049416119426 vs archive 197.20097668224076: diff -0.00048, batch SE 0.00289, z -0.17 -> PASS
   [L A:14] CHECK A RULE: I 207.46433057628943 vs Round 14 continuous rule e = 1.580 207.4639397749701: diff +0.00039, combined SE 0.00095, z +0.41 -> PASS
   [L A:15] CHECK A PRED: I 208.95657763962117 vs round 10 history[0].f 208.95657763962944: diff -8.27e-12 (J_cond is the same for every run: per-run sd 2.8e-14; agrees at 4 dp: True) -> PASS
   ```
3. **Part B data** [L B:2–5]:
   - The noise order equals that of `eval11._mc_batch` with R = 1.
   - The receiver's windows equal the sensor's at every non-transmitted step, for all 3 × 500 experiments: PASS.
4. **Part B, SOD CML within 0.3 nat of I: FAILED** (§3.6, [L B:21]).
5. **Re-run of stages A and B by this session** [L rerun_AB:1–21]:
   ```
   [L rerun_AB:21] CHECK re-run of stages A and B reproduces the committed results exactly (outside wall times): True -> PASS
   ```
   - partA.json and partB.json: 0 differences outside `wall_s` / `wall_s_sum`.
   - The 12 .npy files are byte-identical.
   - partB_estimates.npz: 24 arrays, all bitwise equal.
   - Logs A and B: every line is equal once wall times are removed.
6. **Supplementary stage reproduces stage B's decisions**:
   ```
   [L Bn:3] CHECK Bn: estimator_r8b.estimate_stage1 re-run on the 14 non-accepted runs reproduces stage B (not accepted, fallback theta_LS bitwise equal to the archive): True -> PASS
   ```
7. **Part C**:
   ```
   [L C:3]  EDGE TABLE: 701 of 701 grid edges are on the upper envelope; ... E non-increasing along the table: True; table vs direct argmax on 21401 rho values ...: equal True; direct argmax non-increasing: True -> PASS
   [L C:39] CHECK C determinism: repetition b = 0 (seed 20261117) equals the final tuning run for C0, C1, C2: True -> PASS
   [L C:46] CHECK C certificate C0: I(tuned) 207.4975, mean over seeds 207.4973 vs certificate 207.7146: (I - UB)/SE = -114.7 (single run, SE 0.0019), -362.9 (mean, SE 0.0006) -> PASS
   [L C:47] CHECK C certificate C1: I(tuned) 207.5989, mean over seeds 207.5985 vs certificate 207.7146: (I - UB)/SE = -69.4 (single run, SE 0.0017), -220.3 (mean, SE 0.0005) -> PASS
   [L C:48] CHECK C certificate C2: I(tuned) 206.8893, mean over seeds 206.8897 vs certificate 207.7146: (I - UB)/SE = -375.8 (single run, SE 0.0022), -1187.8 (mean, SE 0.0007) -> PASS
   ```
8. **Summary consistency**: the archived Part B statistics recompute exactly from the archived estimates [L summary:24].
9. **Frozen files**: `git status` on rounds 4–15 and on the frozen files was clean after every stage, and no empty folder was created under `round4/` [L X: CLEANUP lines].

## 7. Where the prompt and the code differ

1. **§3, "runs not accepted by the round-8b rule".**
   - The round-8b rule (`estimator_r8b.accepted`) rejects every L-BFGS-B call that ends with success = False, whatever its Newton decrement.
   - All 14 non-accepted runs ended both calls with `ABNORMAL`. Their first call's end point had η = 2.9e-8 … 6.4e-7, below the rule's own threshold of 1e-5.
   - Following round 8b's `estimate_stage1`, the pre-registered estimate of such a run is θ_LS. The code wins: the pre-registered numbers use θ_LS. The supplementary §3.5 shows the alternative.
2. **§3, "Initial value: the LS estimate".** Round 8b's `estimate_stage1` also runs a restart call from θ_LS + 0.01 s·d. It was used unchanged, with restart seed 80261116 + i (DECISION 18-6).
3. **§3, check "consistent with rounds 8/8b … D̂ within 0.3 nat of I".**
   - Round 8b's own stage-1 value (R = 300, `round8b/report_round8b_CN.md` line 97) is −log det M̂₁ = 196.881 against log det J₁ = 197.201, a difference of −0.320 nat. It would also lie outside 0.3.
   - The check was applied as written and failed (§3.6).
4. **§2, check "PRED agrees with Round 10's history[0].f within Monte Carlo error".** Under PRED, c_k does not depend on r, so J_cond is identical in every run and the Monte Carlo error of I is 0. The agreement is −8.3e-12.
5. **§4, "each tuned by bisection on a price λ".** C0 has no λ. Its constant e was bisected on [1.50, 1.70] σ (DECISION 18-9). This gives e = 1.57734 σ (E[N] = 316.70 at the tuning seed), not Round 14's e = 1.580 (E[N] = 314.5).
6. **§4, "g_k S_k / p (it averages about 1)".** At the tuned λ it averages 1.140 [L C:38]. This is a number, not a code difference.
7. **§2, outside reference.** It used e = 1.5768; this round uses e = 1.580 (Round 14's rounded-up value, as the prompt's §1 table states). Its I and log det mean(J_sent) are 0.039 and 0.064 nat higher than here. It is not a gate.
8. **§3, runtime guide "10–20 min" for Part B.** Stage B took 55 s (first session) and 41 s (re-run).

## 8. DECISION list

- **DECISION 18-1** (first session, `e097fac`). Scoring conventions:
  - six physical lines A1, A2, B1, B2, B3, C;
  - closed ranges; "above" and "below" strict;
  - "at most 2 %" means ≤ 10 runs;
  - pre-registered Part B statistics over all 500 runs, fallback included;
  - Part C differences are means over ten seeds.
- **DECISION 18-2.** Trigger parameters:
  - e = 1.580 is read from `round14/results/part1.json → d.e_up` (asserted);
  - RULE_FB is `mismatch15.run_job`'s feedback with δ = 0, e₀ = 1.580, VF(100, 3), clipped to [0.5, 4.0];
  - PRED: c = Φ⁻¹(1 − q/2) = 1.857845, q = (B − 1)/(K − 1) = 0.063191;
  - Δ\* = `eval13.D` [L A:2].
- **DECISION 18-3.** Part A:
  - the random-number order is that of `eval11._mc_batch`;
  - I = log det of the mean per-step weight, equal to log det of the mean J_cond to ≤ 5.7e-14;
  - batch SEs are over the 10 batches;
  - the SE of the non-transmission value is from per-batch differences;
  - the correlation for PRED is undefined (per-run sd 2.8e-14 < 1e-9);
  - RULE_FB is included as a fourth trigger.
- **DECISION 18-4.** Part A checks:
  - SOD: abs(z) ≤ 3 against the archive with the batch SE;
  - RULE: abs(z) ≤ 3 against Round 14's continuous rule at e = 1.58 (207.46394, batch SE 0.00055), with the combined SE;
  - PRED: abs(difference) ≤ 5e-5 (agreement to 4 decimals), since its Monte Carlo error is 0.
- **DECISION 18-5.** Part B data:
  - experiment i uses `default_rng(20261116 + i).standard_normal(K)`, value 0 for z₁;
  - the same noise is used for the three triggers;
  - the receiver's windows are rebuilt from the record (`estimator_r8.references` and x_k) and checked equal to the sensor's;
  - the restart seed is 80261116 + i.
- **DECISION 18-6** (first session, `e097fac`). Estimator procedure:
  - LS = `estimator_r8.ls_init`;
  - CML = `estimator_r8b.estimate_stage1` unchanged (first call from θ_LS, restart, round-8b acceptance, fallback θ_LS);
  - statistics over every run (fallback included), plus accepted-only when any run falls back.
- **DECISION 18-7.** Part B statistics:
  - CR sd from the same trigger's Part A J;
  - MC SE of the bias = sd(errors)/√n / CR sd;
  - D̂ about θ\*; −log det cov with ddof = 1;
  - "accepted only" = the runs whose CML was accepted, with LS restricted to the same runs;
  - `round18.json` holds all three sets (all, accepted, supplementary).
- **DECISION 18-8.** The edge table:
  - upper envelope of the 701 lines;
  - ties go to the smaller e;
  - verified against the direct argmax on 21401 values of ρ.
- **DECISION 18-9.** Part C tuning:
  - C0: arithmetic bisection on e ∈ [1.50, 1.70];
  - C1, C2: geometric bisection on λ from λ₀ = (p/S_ref)/ρ_mid(E = 1.58), bracketed by ×1.25^±1 and widened by ×1.25 until it brackets B;
  - stop at the first run with abs(E[N] − B) ≤ 0.3, at most 60 iterations;
  - the three tunings ran in a pool of 3 processes; the repetitions in a pool of 4.
- **DECISION 18-10.** C2 implementation:
  - σ²M_k is accumulated as ε σ² J_ref + h₁h₁ᵀ + Σ c_j h_j h_jᵀ;
  - g_k comes from a batched 9 × 9 solve;
  - e_k quantiles are over (trajectory, step), k = 2 … K, from counts on the 0.005 grid;
  - in C1, e_k is the same for every trajectory.
- **DECISION 18-11.** Part C certificate check: z = (I − 207.71459)/SE, with SE = the sd over the ten seeds (single tuned run) or that sd/√10 (mean). The run stops if z > 3.
- **DECISION 18-12.** Descriptive context for D̂, not used in any check or score:
  - bootstrap SE of D̂: 2000 resamples, seed 20261118;
  - reference distribution of D̂ − I under exact attainment: Wishart, 10⁶ draws, seed 20261119.
- **DECISION 18-13** (second session):
  - continue on top of `e85a3f0` without force-pushing (user decision); the first session's prompt and predictions commits are kept;
  - the container lacked numba, so numba 0.68.1 was installed (the version in the first session's logs; numpy 2.5.3, scipy 1.18.1, Python 3.13.18 as there);
  - stages A and B were re-run in a separate worktree, with the criteria of `rerun_check18.py`; the committed results were kept.
- **DECISION 18-S1** (supplementary, user instruction). The Newton steps start at the end point of the first L-BFGS-B call (the call started from θ_LS).
- **DECISION 18-S2** (supplementary). The Newton iteration:
  - full Newton steps without damping or line search;
  - η is evaluated before every step and the run counts as converged at the first η ≤ 1e-5 (inclusive, as in the round-8b rule);
  - at most 50 steps; it stops at a non-finite value or a failed Cholesky factorisation.
- **DECISION 18-S3** (supplementary). A converged run's estimate replaces θ_LS in a supplementary copy; a non-converged run would keep θ_LS (none occurred). Only the "Newton (supp.)" rows use this copy.
- **DECISION 18-S4** (supplementary). Stage Bn re-runs `estimate_stage1` on those runs and requires it to reproduce stage B (not accepted, θ_LS bitwise) before using the end points.

## 9. Wall times

| stage | wall time | detail |
|---|---|---|
| A | 10.8 s (first session) / 8.5 s (re-run) | |
| B | 55.2 s / 40.7 s | 1500 experiments × two estimators, 4 workers |
| Bn | 0.9 s | |
| C | 137.7 s | tuning: C0 12 s (10 runs of 1.2 s), C1 13 s (11 runs), C2 101 s (10 runs of about 10 s); ten repetitions of three variants: about 35 s |
| summary | a few seconds (not timed) | |

Total computing time in this round: under 5 min.

## 10. Not done

- One input (U), one budget (B = 316.892) and the true model only.
- No recursive estimator was built. LS is batch least squares on the transmitted samples, and CML is the batch censored ML of rounds 8/8b.
- No Part B estimation under RULE_FB.
- Part C uses R = 2000 per run and the prescribed ε = 0.02 only. No other ε, no other reference J_ref, and no per-step breakdown of where C2 differs from C1.
- The supplementary Newton analysis applies only to the 14 non-accepted runs. The accepted runs were not re-optimised.
