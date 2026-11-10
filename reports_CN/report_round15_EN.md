---
title: "Round 15 report: the one-parameter rule when the sensor's model is an estimate (no correction, margin, feedback)"
date: "2026-11-10"
---

English copy of `report_round15_CN.md`. Where the two differ, the Chinese version is the report of record.

**Prompt.** `reports_CN/prompt_round15_rule_with_estimated_model_EN.md`. Branch `round15-rule-estimated-model`, created from the head of `round14-one-parameter-rule` at **`6b79535`** (the Round 14 report). Commits:
- `33f8d0c`: the prompt (first commit);
- `14e7ee3`: `round15/predictions_round15.md`, before any computation (second commit);
- `f27701a`: code;
- `4b8b60d`: check 1 and the sequence script;
- the check 2 and Monte Carlo stage commits;
- `15dd48c`: summary;
- this report: the commit after that.

**Scope.**
- Input U and budget B = 316.8921168493721 only. No DP.
- Except checks 1–2, every number is a Monte Carlo of the continuous rule of Round 14, with the receiver knowing the model θ̃ and, for the feedback, the count. ΔI = log det J − 197.20097668224076 (constant SoD at Δ\*).
- No two-stage experiment was run: the models are drawn from the distribution such an estimate would have (prompt §0).
- No causes are claimed.

**Sources.** Paths are relative to `code_8_3_stage2/stage2_pkg/evaluator/round15/`.
- `[L X:n]` is line n of `logs/X.log`. `summary.log` contains four runs, because the figure layout was adjusted and the summary re-run (DECISION 15-10). The cited lines are from the last run, which starts at line 295.
- `[J a.b]` is field `a.b` of `results/round15.json`. That file holds one record per job, 2716 in all, plus the summaries.
- Code: `mismatch15.py` (models, variants, Monte Carlo job, mismatched lattice rule), `run_round15.py` (stages `check1`, `check2`, `mc`, `summary`), `summary15.py`.

**Models** [L mc:3–7]; [J models_main_by_level]. The median rms prediction error over k is 0.139 / 0.204 / 0.285 σ at c = 1 / 2 / 4, and the median of the largest prediction error is 0.339 / 0.489 / 0.645 σ (95th percentile 0.477 / 0.660 / 1.033 σ). The scaled models have largest errors of exactly 0.5 / 1 / 2 / 4 σ. The gross-error models have a largest error of 346.4 σ (rms 335.6 σ).

## 1. Answers to Q1–Q7

Sources: [L summary:375–381]; [J questions].

- **Q1.** Without correction (V0), 52.5 % / 50.0 % / 67.5 % of the estimated models have E[N] > B at c = 1 / 2 / 4.
  - E[N] − B: median +1.3 / +2.1 / +11.8; 5–95 % range −30.4…+22.8 / −29.9…+38.8 / −31.7…+68.3; maximum +31.0 / +49.7 / +111.7 transmissions.
  - The true model has E[N] − B = −2.35.
- **Q2 — yes at every level.** The median ΔI of V0 is 10.201 / 10.032 / 10.134 nat, against 10.264 for the true model. The differences are −0.063 / −0.232 / −0.129 nat.
- **Q3 — yes.** The complying margins are Δ = 0.10, 0.15, 0.20 and 0.30; VM(0.05) has 15 % of models over budget at c = 2. **VM\* = VM(0.10).**
  - Its median ΔI at c = 2 is 8.755, i.e. 1.508 nat below the true-model V0. Under the true model VM(0.10) itself costs 1.266 nat.
  - At c = 4, 10 % of models are over budget.
- **Q4 — yes.** The complying feedback settings are VF(50, 3), VF(50, 6), VF(100, 3), VF(100, 6) and VF(300, 6). **VF\* = VF(100, 3).**
  - Its median ΔI at c = 2 is 9.985, i.e. 0.278 nat below the true-model V0. Under the true model it costs 0.098 nat.
  - At c = 4, 10 % of models are over budget.
  - True-model cost of VF(T, 0): 0.135 / 0.042 / −0.014 / −0.029 nat for T = 50 / 100 / 300 / 1000.
- **Q5 — yes.** At equal compliance, VF\* is above VM\* in median ΔI by **+1.149 / +1.230 / +0.949 nat** at c = 1 / 2 / 4. Under the true model the difference is +1.169.
- **Q6.**
  - Median sd of N over single experiments: V0 18.2–18.6; VM\* 16.5–17.0; VF(50, ·) 1.3–1.4; VF(100, ·) 1.8–1.9 (VF\* 1.85 / 1.90 / 1.84); VF(300, ·) 3.1; VF(1000, ·) 5.6–5.7. These are nearly the same at every level and under the true model.
  - Pooled fraction of single experiments with N > B: V0 0.51 / 0.55 / 0.66; VF\* 0.12 / 0.18 / 0.25; VM\* 0.002 / 0.015 / 0.126.
- **Q7.**
  - Up to ε = 0.5 σ, VM\* keeps 0 % of the scaled models over budget and VF\* 2.5 %. At ε = 1 they have 27.5 % and 17.5 % over budget. VF\* has 30 % at ε = 2 and 62.5 % at ε = 4. V0 already has 52.5 % at ε = 0.5.
  - At the 2 % gross error the three variants give **identical** results: +2 % gives E[N] = 1.06 and ΔI = −309.6; −2 % gives E[N] = 404.0 and ΔI = −9.77.
  - Under the gross error VF\* sits at the lower clip bound for 95.7 % of the steps (+2 %) and at the upper bound for 99.1 % (−2 %).

## 2. The three level tables (main set, 40 models per level)

"frac E[N] > B" is the fraction of models whose Monte Carlo E[N] exceeds B. "pooled N > B" is the fraction of single experiments (trajectories) with N > B, pooled over the 40 models. The true-model line uses R = 20000 [L summary:297–350]; [J levels, true].

### c = 1
| variant | frac E[N] > B | E[N] − B: 5 % / median / 95 % / max | ΔI: min / 5 % / median / 95 % | median sd N | pooled N > B | true model: E[N] − B / ΔI / sd N / N > B |
|---|---:|---|---|---:|---:|---|
| V0 | 0.525 | −30.4 / +1.3 / +22.8 / +31.0 | 9.025 / 9.525 / 10.201 / 10.607 | 18.16 | 0.507 | −2.35 / 10.264 / 18.32 / 0.451 |
| VM(0.05) | 0.000 | −59.1 / −29.7 / −9.6 / −2.0 | 8.368 / 8.880 / 9.579 / 10.006 | 17.33 | 0.085 | −33.23 / 9.642 / 17.43 / 0.032 |
| VM(0.10) | 0.000 | −85.2 / −58.4 / −39.8 / −32.4 | 7.688 / 8.212 / 8.933 / 9.381 | 16.52 | 0.002 | −61.66 / 8.997 / 16.54 / 0.000 |
| VM(0.15) | 0.000 | −109.1 / −84.8 / −67.2 / −60.6 | 6.982 / 7.522 / 8.264 / 8.734 | 15.64 | 0.000 | −87.88 / 8.330 / 15.68 / 0.000 |
| VM(0.20) | 0.000 | −131.3 / −108.9 / −92.6 / −86.6 | 6.256 / 6.808 / 7.574 / 8.061 | 14.87 | 0.000 | −111.96 / 7.638 / 14.88 / 0.000 |
| VM(0.30) | 0.000 | −169.7 / −151.0 / −137.4 / −131.9 | 4.732 / 5.311 / 6.125 / 6.646 | 13.30 | 0.000 | −153.86 / 6.187 / 13.25 / 0.000 |
| VF(50,0) | 0.900 | −0.1 / +0.4 / +0.8 / +1.4 | 9.957 / 9.989 / 10.060 / 10.102 | 1.33 | 0.712 | +0.34 / 10.128 / 1.31 / 0.715 |
| VF(50,3) | 0.000 | −3.1 / −2.6 / −2.2 / −1.6 | 9.898 / 9.933 / 10.002 / 10.047 | 1.33 | 0.049 | −2.63 / 10.070 / 1.30 / 0.041 |
| VF(50,6) | 0.000 | −6.0 / −5.6 / −5.2 / −4.5 | 9.842 / 9.871 / 9.944 / 9.989 | 1.32 | 0.000 | −5.60 / 10.012 / 1.29 / 0.000 |
| VF(100,0) | 0.675 | −1.1 / +0.3 / +1.4 / +2.9 | 9.991 / 10.036 / 10.137 / 10.194 | 1.86 | 0.621 | +0.23 / 10.221 / 1.83 / 0.633 |
| VF(100,3) | 0.000 | −4.0 / −2.7 / −1.6 / −0.1 | 9.933 / 9.981 / 10.082 / 10.140 | 1.85 | 0.123 | −2.71 / 10.166 / 1.82 / 0.099 |
| VF(100,6) | 0.000 | −7.0 / −5.6 / −4.5 / −3.0 | 9.876 / 9.925 / 10.024 / 10.081 | 1.84 | 0.004 | −5.66 / 10.109 / 1.81 / 0.002 |
| VF(300,0) | 0.475 | −3.5 / −0.1 / +3.1 / +5.0 | 9.944 / 9.984 / 10.181 / 10.280 | 3.13 | 0.536 | −0.08 / 10.277 / 3.10 / 0.540 |
| VF(300,3) | 0.100 | −6.3 / −2.9 / +0.2 / +2.2 | 9.890 / 9.928 / 10.127 / 10.227 | 3.11 | 0.257 | −2.90 / 10.224 / 3.08 / 0.204 |
| VF(300,6) | 0.000 | −9.1 / −5.7 / −2.6 / −0.7 | 9.834 / 9.874 / 10.074 / 10.174 | 3.10 | 0.080 | −5.72 / 10.170 / 3.06 / 0.040 |
| VF(1000,0) | 0.575 | −8.4 / +1.0 / +6.3 / +10.7 | 9.802 / 9.885 / 10.210 / 10.330 | 5.65 | 0.526 | −0.22 / 10.293 / 5.66 / 0.512 |
| VF(1000,3) | 0.300 | −10.8 / −1.4 / +3.9 / +8.2 | 9.758 / 9.840 / 10.164 / 10.285 | 5.63 | 0.397 | −2.63 / 10.248 / 5.63 / 0.346 |
| VF(1000,6) | 0.125 | −13.1 / −3.9 / +1.4 / +5.7 | 9.712 / 9.794 / 10.117 / 10.239 | 5.60 | 0.273 | −5.06 / 10.203 / 5.61 / 0.200 |

### c = 2
| variant | frac E[N] > B | E[N] − B: 5 % / median / 95 % / max | ΔI: min / 5 % / median / 95 % | median sd N | pooled N > B | true model: E[N] − B / ΔI / sd N / N > B |
|---|---:|---|---|---:|---:|---|
| V0 | 0.500 | −29.9 / +2.1 / +38.8 / +49.7 | 9.236 / 9.406 / 10.032 / 10.703 | 18.27 | 0.546 | −2.35 / 10.264 / 18.32 / 0.451 |
| VM(0.05) | 0.150 | −58.0 / −28.8 / +5.9 / +16.1 | 8.592 / 8.770 / 9.402 / 10.111 | 17.53 | 0.176 | −33.23 / 9.642 / 17.43 / 0.032 |
| VM(0.10) | 0.000 | −83.8 / −57.2 / −24.9 / −14.7 | 7.923 / 8.113 / 8.755 / 9.494 | 16.63 | 0.015 | −61.66 / 8.997 / 16.54 / 0.000 |
| VM(0.15) | 0.000 | −107.6 / −83.3 / −53.3 / −43.4 | 7.232 / 7.433 / 8.084 / 8.855 | 15.81 | 0.000 | −87.88 / 8.330 / 15.68 / 0.000 |
| VM(0.20) | 0.000 | −129.3 / −107.4 / −79.7 / −70.1 | 6.517 / 6.732 / 7.389 / 8.192 | 15.18 | 0.000 | −111.96 / 7.638 / 14.88 / 0.000 |
| VM(0.30) | 0.000 | −167.2 / −149.6 / −125.4 / −117.2 | 5.015 / 5.265 / 5.923 / 6.803 | 13.80 | 0.000 | −153.86 / 6.187 / 13.25 / 0.000 |
| VF(50,0) | 0.825 | −0.3 / +0.3 / +1.3 / +2.3 | 9.802 / 9.880 / 9.995 / 10.074 | 1.36 | 0.722 | +0.34 / 10.128 / 1.31 / 0.715 |
| VF(50,3) | 0.000 | −3.2 / −2.6 / −1.7 / −0.7 | 9.742 / 9.820 / 9.935 / 10.013 | 1.36 | 0.067 | −2.63 / 10.070 / 1.30 / 0.041 |
| VF(50,6) | 0.000 | −6.2 / −5.6 / −4.7 / −3.7 | 9.688 / 9.762 / 9.877 / 9.955 | 1.35 | 0.000 | −5.60 / 10.012 / 1.29 / 0.000 |
| VF(100,0) | 0.525 | −1.4 / +0.2 / +2.5 / +5.0 | 9.742 / 9.903 / 10.042 / 10.159 | 1.92 | 0.637 | +0.23 / 10.221 / 1.83 / 0.633 |
| VF(100,3) | 0.025 | −4.3 / −2.7 / −0.5 / +2.0 | 9.687 / 9.846 / 9.985 / 10.102 | 1.90 | 0.175 | −2.71 / 10.166 / 1.82 / 0.099 |
| VF(100,6) | 0.000 | −7.2 / −5.7 / −3.5 / −1.0 | 9.625 / 9.790 / 9.929 / 10.047 | 1.90 | 0.017 | −5.66 / 10.109 / 1.81 / 0.002 |
| VF(300,0) | 0.475 | −2.6 / −0.2 / +5.4 / +9.9 | 9.696 / 9.875 / 10.059 / 10.212 | 3.14 | 0.595 | −0.08 / 10.277 / 3.10 / 0.540 |
| VF(300,3) | 0.225 | −5.4 / −3.0 / +2.5 / +7.1 | 9.639 / 9.820 / 10.004 / 10.158 | 3.12 | 0.332 | −2.90 / 10.224 / 3.08 / 0.204 |
| VF(300,6) | 0.050 | −8.2 / −5.8 / −0.3 / +4.2 | 9.584 / 9.767 / 9.949 / 10.105 | 3.11 | 0.140 | −5.72 / 10.170 / 3.06 / 0.040 |
| VF(1000,0) | 0.600 | −6.8 / +1.0 / +14.7 / +16.3 | 9.761 / 9.776 / 10.053 / 10.323 | 5.70 | 0.595 | −0.22 / 10.293 / 5.66 / 0.512 |
| VF(1000,3) | 0.425 | −9.1 / −1.4 / +12.2 / +13.7 | 9.716 / 9.730 / 10.008 / 10.278 | 5.66 | 0.475 | −2.63 / 10.248 / 5.63 / 0.346 |
| VF(1000,6) | 0.225 | −11.5 / −3.8 / +9.7 / +11.2 | 9.670 / 9.684 / 9.963 / 10.232 | 5.64 | 0.359 | −5.06 / 10.203 / 5.61 / 0.200 |

### c = 4
| variant | frac E[N] > B | E[N] − B: 5 % / median / 95 % / max | ΔI: min / 5 % / median / 95 % | median sd N | pooled N > B | true model: E[N] − B / ΔI / sd N / N > B |
|---|---:|---|---|---:|---:|---|
| V0 | 0.675 | −31.7 / +11.8 / +68.3 / +111.7 | 8.886 / 9.115 / 10.134 / 11.023 | 18.58 | 0.662 | −2.35 / 10.264 / 18.32 / 0.451 |
| VM(0.05) | 0.375 | −59.7 / −19.7 / +33.8 / +73.7 | 8.261 / 8.469 / 9.525 / 10.435 | 17.82 | 0.364 | −33.23 / 9.642 / 17.43 / 0.032 |
| VM(0.10) | 0.100 | −85.4 / −48.6 / +1.6 / +38.3 | 7.615 / 7.802 / 8.896 / 9.826 | 16.98 | 0.126 | −61.66 / 8.997 / 16.54 / 0.000 |
| VM(0.15) | 0.025 | −108.9 / −75.3 / −27.3 / +5.3 | 6.945 / 7.113 / 8.243 / 9.192 | 16.19 | 0.029 | −87.88 / 8.330 / 15.68 / 0.000 |
| VM(0.20) | 0.000 | −131.1 / −99.9 / −54.1 / −25.6 | 6.245 / 6.402 / 7.568 / 8.535 | 15.52 | 0.002 | −111.96 / 7.638 / 14.88 / 0.000 |
| VM(0.30) | 0.000 | −169.5 / −142.4 / −101.9 / −80.2 | 4.743 / 4.919 / 6.144 / 7.162 | 13.97 | 0.000 | −153.86 / 6.187 / 13.25 / 0.000 |
| VF(50,0) | 0.875 | −0.3 / +0.6 / +1.7 / +3.3 | 9.548 / 9.648 / 9.883 / 10.052 | 1.32 | 0.760 | +0.34 / 10.128 / 1.31 / 0.715 |
| VF(50,3) | 0.025 | −3.3 / −2.4 / −1.3 / +0.3 | 9.487 / 9.585 / 9.826 / 9.992 | 1.31 | 0.101 | −2.63 / 10.070 / 1.30 / 0.041 |
| VF(50,6) | 0.000 | −6.3 / −5.4 / −4.3 / −2.7 | 9.425 / 9.528 / 9.764 / 9.934 | 1.30 | 0.001 | −5.60 / 10.012 / 1.29 / 0.000 |
| VF(100,0) | 0.725 | −1.5 / +0.8 / +3.3 / +6.4 | 9.505 / 9.574 / 9.900 / 10.131 | 1.84 | 0.699 | +0.23 / 10.221 / 1.83 / 0.633 |
| VF(100,3) | 0.100 | −4.4 / −2.2 / +0.3 / +3.4 | 9.445 / 9.517 / 9.844 / 10.074 | 1.84 | 0.250 | −2.71 / 10.166 / 1.82 / 0.099 |
| VF(100,6) | 0.025 | −7.4 / −5.1 / −2.6 / +0.4 | 9.387 / 9.455 / 9.786 / 10.019 | 1.83 | 0.034 | −5.66 / 10.109 / 1.81 / 0.002 |
| VF(300,0) | 0.550 | −4.8 / +0.8 / +7.9 / +11.2 | 9.399 / 9.468 / 9.873 / 10.152 | 3.14 | 0.599 | −0.08 / 10.277 / 3.10 / 0.540 |
| VF(300,3) | 0.325 | −7.6 / −2.0 / +5.0 / +8.3 | 9.343 / 9.412 / 9.818 / 10.098 | 3.11 | 0.383 | −2.90 / 10.224 / 3.08 / 0.204 |
| VF(300,6) | 0.150 | −10.4 / −4.8 / +2.2 / +5.4 | 9.287 / 9.354 / 9.763 / 10.045 | 3.10 | 0.202 | −5.72 / 10.170 / 3.06 / 0.040 |
| VF(1000,0) | 0.575 | −11.3 / +2.1 / +18.0 / +19.7 | 9.326 / 9.406 / 9.920 / 10.199 | 5.73 | 0.590 | −0.22 / 10.293 / 5.66 / 0.512 |
| VF(1000,3) | 0.475 | −13.6 / −0.3 / +15.5 / +17.2 | 9.278 / 9.358 / 9.873 / 10.153 | 5.70 | 0.497 | −2.63 / 10.248 / 5.63 / 0.346 |
| VF(1000,6) | 0.350 | −15.9 / −2.7 / +12.9 / +14.6 | 9.230 / 9.311 / 9.825 / 10.106 | 5.68 | 0.405 | −5.06 / 10.203 / 5.61 / 0.200 |

## 3. Selection and head-to-head

**Selection** (prompt §4b; a variant complies if at most 5 % of the models, i.e. 2 of 40, have E[N] > B at c = 1 and at c = 2) [L mc:56–75]; [J selection]:

| variant | frac E[N] > B, c = 1 | c = 2 | complies | median ΔI c = 2 |
|---|---:|---:|---|---:|
| V0 | 0.525 | 0.500 | no | 10.032 |
| VM(0.05) | 0.000 | 0.150 | no | 9.402 |
| VM(0.10) | 0.000 | 0.000 | yes | 8.755 |
| VM(0.15) | 0.000 | 0.000 | yes | 8.084 |
| VM(0.20) | 0.000 | 0.000 | yes | 7.389 |
| VM(0.30) | 0.000 | 0.000 | yes | 5.923 |
| VF(50,0) | 0.900 | 0.825 | no | 9.995 |
| VF(50,3) | 0.000 | 0.000 | yes | 9.935 |
| VF(50,6) | 0.000 | 0.000 | yes | 9.877 |
| VF(100,0) | 0.675 | 0.525 | no | 10.042 |
| VF(100,3) | 0.000 | 0.025 | yes | 9.985 |
| VF(100,6) | 0.000 | 0.000 | yes | 9.929 |
| VF(300,0) | 0.475 | 0.475 | no | 10.059 |
| VF(300,3) | 0.100 | 0.225 | no | 10.004 |
| VF(300,6) | 0.000 | 0.050 | yes | 9.949 |
| VF(1000,0) | 0.575 | 0.600 | no | 10.053 |
| VF(1000,3) | 0.300 | 0.425 | no | 10.008 |
| VF(1000,6) | 0.125 | 0.225 | no | 9.963 |

VM\* = VM(0.10) has the largest median ΔI at c = 2 among the complying margins. VF\* = VF(100, 3) has the largest among the complying feedback settings.

**Head-to-head** (from the tables above; ΔI medians; c = 1 / 2 / 4 / true model):

| | frac E[N] > B | median E[N] − B | median ΔI | 5 % ΔI | median sd N |
|---|---|---|---|---|---|
| V0 | 0.525 / 0.500 / 0.675 / — | +1.3 / +2.1 / +11.8 / −2.35 | 10.201 / 10.032 / 10.134 / 10.264 | 9.525 / 9.406 / 9.115 | 18.16 / 18.27 / 18.58 / 18.32 |
| VM\* = VM(0.10) | 0.000 / 0.000 / 0.100 / — | −58.4 / −57.2 / −48.6 / −61.66 | 8.933 / 8.755 / 8.896 / 8.997 | 8.212 / 8.113 / 7.802 | 16.52 / 16.63 / 16.98 / 16.54 |
| VF\* = VF(100, 3) | 0.000 / 0.025 / 0.100 / — | −2.7 / −2.7 / −2.2 / −2.71 | 10.082 / 9.985 / 9.844 / 10.166 | 9.981 / 9.846 / 9.517 | 1.85 / 1.90 / 1.84 / 1.82 |
| VF\* − VM\* (median ΔI) | | | **+1.149 / +1.230 / +0.949 / +1.169** | | |

For VF\* over the main set, the mean e_k (median over models) is 1.597 / 1.601 / 1.613 σ and its sd 0.160 / 0.178 / 0.202 σ. Clipping was essentially absent: at most 3.0e-6 of the steps at a bound.

## 4. Scaled and gross-error models

Scaled models: the 40 c = 1 directions rescaled so that the largest prediction error is ε σ [L summary:352–369]; [J scaled, gross].

| variant | ε (σ) | frac E[N] > B | median E[N] − B | max E[N] − B | median ΔI |
|---|---:|---:|---:|---:|---:|
| V0 | 0.5 | 0.525 | +4.20 | +50.67 | 10.163 |
| V0 | 1 | 0.750 | +23.17 | +134.37 | 9.922 |
| V0 | 2 | 0.825 | +105.71 | +365.39 | 9.150 |
| V0 | 4 | 0.875 | +288.49 | +758.72 | 8.760 |
| VM(0.10) | 0.5 | 0.000 | −55.73 | −14.84 | 8.897 |
| VM(0.10) | 1 | 0.275 | −35.62 | +61.91 | 8.681 |
| VM(0.10) | 2 | 0.675 | +48.09 | +285.41 | 8.067 |
| VM(0.10) | 4 | 0.875 | +248.31 | +696.39 | 7.949 |
| VF(100, 3) | 0.5 | 0.025 | −2.69 | +0.52 | 9.973 |
| VF(100, 3) | 1 | 0.175 | −2.71 | +4.95 | 9.518 |
| VF(100, 3) | 2 | 0.300 | −2.12 | +15.39 | 8.232 |
| VF(100, 3) | 4 | 0.625 | +2.86 | +57.85 | 5.696 |

Gross error θ̃ = (1 ± 0.02) θ\*, with the prediction off by up to 346 σ:

| model | variant | E[N] | sd N | ΔI | VF\* steps at lower / upper clip |
|---|---|---:|---:|---:|---|
| +2 % | V0, VM(0.10), VF(100, 3) (identical) | 1.06 | 0.24 | −309.64 | 0.957 / 0.000 |
| −2 % | V0, VM(0.10), VF(100, 3) (identical) | 404.00 | 5.98 | −9.77 | 0.000 / 0.991 |

With the prediction this far from the output, only the r ± σ edge of the rule is ever active, so e_k has no effect on the decisions; with common random numbers the three variants are therefore identical.
- **+2 %:** the prediction lies 335 σ above the output, so only the lower edge r − σ can be crossed. U's output has its global minimum at step 1, so after the forced first transmission almost no step crosses it.
- **−2 %:** only the upper edge r + σ can be crossed, and every rise of the output by about 1 σ above the last transmitted value triggers a transmission.

## 5. True-model lines, reference set, check 4

**True model** (δ = 0, R = 20000 per variant) [L mc:8–32]; [J true]:

| variant | E[N] (SE) | sd N | N > B | log det J (batch SE) | ΔI | e_k mean / sd |
|---|---|---:|---:|---|---:|---|
| V0 | 314.543 (0.130) | 18.32 | 0.451 | 207.4646 (0.0004) | 10.264 | 1.580 / 0 |
| VM(0.05) / (0.10) / (0.15) / (0.20) / (0.30) | 283.66 / 255.23 / 229.01 / 204.93 / 163.03 | 17.4 … 13.3 | 0.032 / 0 / 0 / 0 / 0 | — | 9.642 / 8.997 / 8.330 / 7.638 / 6.187 | — |
| VF(50, 0 / 3 / 6) | 317.24 / 314.26 / 311.29 | 1.31 / 1.30 / 1.29 | 0.715 / 0.041 / 0.000 | — | 10.128 / 10.070 / 10.012 | 1.605 … 1.615 / 0.205 |
| VF(100, 0 / 3 / 6) | 317.12 / 314.18 / 311.24 | 1.83 / 1.82 / 1.81 | 0.633 / 0.099 / 0.002 | — | 10.221 / 10.166 / 10.109 | 1.591 … 1.600 / 0.144 |
| VF(300, 0 / 3 / 6) | 316.81 / 313.99 / 311.17 | 3.10 / 3.08 / 3.06 | 0.540 / 0.204 / 0.040 | — | 10.277 / 10.224 / 10.170 | 1.581 … 1.590 / 0.082 |
| VF(1000, 0 / 3 / 6) | 316.68 / 314.26 / 311.83 | 5.66 / 5.63 / 5.61 | 0.512 / 0.346 / 0.200 | — | 10.293 / 10.248 / 10.203 | 1.578 … 1.586 / 0.043 |

Under the true model no VF setting reached a clip bound.

**Check 4** (true model at e₀ = 1.5768; not a gate) [J checks.4_outside_mc]:

| variant | E[N] here (SE) | outside | sd N here / outside | log det J here / outside |
|---|---|---|---|---|
| V0 | 316.615 (0.130) | 316.89 | 18.38 / 18.5 | 207.5037 / 207.505 |
| VF(1000, 0) | 317.073 (0.040) | 317.07 | 5.67 / 5.75 | 207.5023 / 207.498 |
| VF(300, 0) | 316.935 (0.022) | 316.94 | 3.10 / 3.14 | 207.4813 / 207.476 |
| VF(100, 0) | 317.162 (0.013) | 317.16 | 1.83 / 1.83 | 207.4241 / 207.421 |

The differences are −0.27 … +0.003 in E[N] and −0.001 … +0.005 in log det J. The outside run used 4000 trajectories and another seed, so its standard error of E[N] for V0 is about 0.29.

**Reference set** (12 models, E[N] / ΔI; V0 · VF(1000, 0) · VF(300, 0) · VF(100, 0)) [J reference_set]:
- E[N] spread over the 12 models: V0 270.0–385.0; VF(1000, 0) 304.0–333.6; VF(300, 0) 311.3–324.1; VF(100, 0) 315.3–319.8.
- ΔI ranges: V0 8.92–10.91; VF(100, 0) 9.81–10.17.

## 6. Predictions (`predictions_round15.md`, `14e7ee3`)

**6 hits, 2 misses** [L summary:382–390].

| line | values | result |
|---|---|---|
| check 1 to 1e-9; check 2 passes | check 1: all deviations 0; check 2: abs(z) ≤ 2.09 | hit |
| Q1: V0 over budget in [30 %, 65 %] at every level; half-widths in [20, 50] / [30, 70] / [40, 100] | fractions 0.525 / 0.500 / **0.675**; half-widths 26.6 / 34.4 / 50.0 | **miss** (c = 4 fraction) |
| Q2 yes at c = 1 and c = 2 | −0.063, −0.232 | hit |
| Q3 complying margin Δ ≥ 0.10; cost at c = 2 ≥ 0.8 | VM(0.05) does not comply; VM\* = VM(0.10); cost 1.508 | hit |
| Q4 VF(100, 3) complies; some complying VF within 0.4 at c = 2; true-model cost of VF(T, 0) ≤ 0.15 for T ≥ 100 | yes; 0.278–0.387 for all five complying settings; 0.042 / −0.014 / −0.029 | hit |
| Q5 VF\* above VM\* by ≥ 0.3 at c = 1 and c = 2 | +1.149, +1.230 | hit |
| Q6 median sd N < 4 for T ≤ 100; V0 in [16, 21] | 1.30–1.92; 18.16–18.58 | hit |
| Q7 VF\* ≤ 5 % up to ε = 0.5; gross error E[N] > 2B for every variant | 0.025; **E[N] = 1.06 (+2 %), 404.0 (−2 %)** | **miss** (gross error) |

## 7. Checks (prompt §5)

1. **Outside computation, deterministic part.** All 12 reference models reproduce **exactly**: E[N] and log det J have relative deviation 0, and the rms / max prediction errors agree to 6 decimals. FLd − FL ranges from [−4, 6] to [−17, 16] grid steps [L check1:3–15]. For example:
   ```
   [L check1:3]  CHECK1 c=1 d=0: EN 293.130066163623 (rel 0.0e+00) logdet 206.94719294319475 (rel 0.0e+00); rms 0.113856 max 0.243846 (ref 0.113856 / 0.243846); FLd - FL in [-4, 6]; mass dev 8.66e-13 (20.2 s) -> PASS
   [L check1:14] CHECK1 c=4 d=3: EN 386.8182332082853 (rel 0.0e+00) logdet 208.14034340245658 (rel 0.0e+00); rms 0.338782 max 0.706108 (ref 0.338782 / 0.706108); FLd - FL in [-17, 16]; mass dev 8.63e-13 (18.4 s) -> PASS
   [L check1:15] CHECK1 all 12 reference models: PASS
   ```
2. **Evaluator against simulation under mismatch** (σ/120 against the Monte Carlo of the mismatched lattice rule, R = 10⁴):
   ```
   [L check2:2] CHECK2 ref_c1_d1: ... z EN -0.48 S +0.06 logdet +0.26; clipped 0 -> PASS
   [L check2:3] CHECK2 ref_c2_d1: ... z EN -0.33 S -0.40 logdet -0.32; clipped 0 -> PASS
   [L check2:4] CHECK2 ref_c4_d3: ... z EN -0.13 S +2.02 logdet +2.09; clipped 0 -> PASS
   ```
3. **True model against Round 14** (V0, R = 20000, against Round 14's continuous rule at e₀ = 1.580): E[N] 314.543 against 314.496, S 1387.37 against 1387.26, log det J 207.4646 against 207.4639. z = +0.21 / +0.88 / +1.02 with the combined standard errors, so it passes (DECISION 15-6) [L summary:372].
4. **Outside Monte Carlo:** §5, not a gate.
5. **Every job** (2716 records): ΔI finite, 0 ≤ W_k ≤ 1, N ≥ 1; no failure [L mc:81].

`git status` on rounds 4–14 and the frozen files was clean after every stage, and no empty folder was created under `round4/` [L X: CLEANUP lines].

## 8. Figure

`results/fig_round15.png` (one column, 8 pt).

**Caption:** "Number of transmissions above the budget (top) and information gain over constant SoD (bottom) of the one-parameter rule when the sensor's model is an estimate with covariance c J_SoD⁻¹ (c = 1: one constant-SoD experiment), input U, B = 316.89. Median over 40 models (point) and 5–95 % range (whisker) for no correction (V0), the selected margin (VM\*) and the selected count feedback (VF\*). What to look at: how far each variant's whisker reaches above 0 in the top panel, and how far its points sit below the dashed true-model line in the bottom panel."

- **Top panel.** V0's whiskers reach +22.8 / +38.8 / +68.3 above the budget. VM\*'s whiskers lie below 0 except at c = 4 (+1.6). VF\*'s whiskers are narrow, between −4.4 and +0.3.
- **Bottom panel.** VF\*'s points lie 0.18 / 0.28 / 0.42 nat below the true-model line, and VM\*'s 1.33 / 1.51 / 1.37 nat below it. V0's points lie 0.06–0.23 nat below it, with whiskers 1.1–1.9 nat long.

## 9. Where the prompt and the code differ

I found no contradiction between the prompt and the code.
- §1's θ\* = [G(0), Re/Im G(jω_m)] reproduces XK to 8.9e-16 (absolute).
- The lattice rule of check 1 is Round 14's rule with FL and CL taken from the prediction; with the prediction defined as in DECISION 15-2, it reproduces the outside computation exactly.
- The prediction "E[N] > 2B at the 2 % gross error" does not hold, for the reason given in §4. That is a result, not a discrepancy.

## 10. DECISION list

- **DECISION 15-1.** Scoring conventions, fixed in `predictions_round15.md` before running:
  - "E[N] > B" uses the job's point estimate;
  - percentiles over models use linear interpolation;
  - the Q1 half-width is (P95 − P5)/2;
  - "true-model V0" means δ = 0 with R = 20000;
  - Q3 also requires that VM(0.05) does not comply;
  - Q6 uses all three levels and the VF settings with T ∈ {50, 100}.
- **DECISION 15-2.** The prediction is x̄_k = XK_k + h_kᵀ δ (the evaluator's true output plus the prediction error), so δ = 0 is exactly the true model. θ\* = [G(0), Re G(jω₁), Im G(jω₁), …] is used for the gross error (δ = ±0.02 θ\*) and checked against XK (8.9e-16). Check 1 reproduces exactly with this definition.
- **DECISION 15-3.** The rms and max prediction errors are taken over all K steps.
- **DECISION 15-4.** Feedback statistics are over (trajectory, step) for k = 2 … K. A step counts as "at a clip bound" when the unclipped value is ≤ 0.5 or ≥ 4.0; the lower and upper bounds are reported separately.
- **DECISION 15-5.** True-model runs (all 18 variants and the four check-4 runs) use R = 20000 as 10 batches of 2000 with seeds 20261112 + b. The batch SE of log det J is computed from the 10 batches. Batch 0 uses the same seed as every model job (common random numbers).
- **DECISION 15-6.** Check 3 passes iff abs(z) ≤ 3 for E[N], S and log det J, using the combined standard error (Round 14's and this round's; batch SE for log det J).
- **DECISION 15-7.** The check 2 Monte Carlo is Round 14's routine 1c (R = 10⁴, 20 batches of 500, seeds 20261109 + b) with FL and CL of the prediction. The σ/120 evaluation is `eval13.evaluate_refined` of the same lattice rule; the kernel windows and the information use the true XK.
- **DECISION 15-8.** Selection: compliance uses the point estimates of the 40 models. No tie occurred, and no family lacked a complying member.
- **DECISION 15-9.** Jobs run on a pool of 4 worker processes and are checkpointed line by line in `results/jobs.jsonl`. The stages ran in sequence, each committed and pushed when it finished.
- **DECISION 15-10.** Figure: categorical colours 1–3 (validated); horizontal offsets of ±0.18 per variant; the dashed lines are labelled in the panel corners. The summary was re-run three times to move the labels, and its numbers were unchanged.
- **DECISION 15-11.** Rounds 4–14 were not modified. Numba may write compiled-code cache files under `round13/__pycache__`, which is git-ignored.

## 11. Wall times

- Check 1: 228 s, i.e. 12 σ/24 forwards of 18–20 s.
- Check 2: about 5.3 min, i.e. three σ/120 forwards of 100–104 s plus three Monte Carlo runs of about 3 s.
- Monte Carlo stage: 988 s.
  - True model: 220 jobs of 2000 trajectories in 76 s.
  - Main and reference sets: 2208 jobs in about 12.7 min.
  - Scaled and gross-error models: 486 jobs in 155 s.
  - This is about 1.4 s per job per worker process.
- Summary: under 3 s.
- About 25 min in total.

## 12. Not done

- No simulated first stage and no maximum-likelihood estimation: the models are drawn from the Cramér–Rao distribution of such an estimate.
- One input (U) and one budget (B = 316.892).
- The feedback law is not optimised: only the 12 prescribed (T, m) settings were run. Margins other than the five prescribed ones were not run.
- No σ/24 or σ/120 evaluation of the continuous rule: the model jobs are Monte Carlo only, as prescribed.
- Part B of Round 13 is on hold.
